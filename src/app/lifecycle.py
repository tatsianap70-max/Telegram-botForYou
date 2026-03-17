"""Управление жизненным циклом приложения.

Класс ApplicationLifecycle инкапсулирует всю логику startup и shutdown:
- Инициализация бота и диспетчера
- Выбор режима работы (polling/webhook)
- Запуск планировщика
- Корректная остановка всех компонентов

Оптимизация startup:
- Параллельная инициализация независимых компонентов
- Фоновые задачи для некритичных HTTP-операций
- Удаление webhook параллельно с инициализацией (dev mode)
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
from typing import TYPE_CHECKING, Any

from src.bot.loader import create_bot, create_dispatcher, register_bot_commands
from src.db.base import DatabaseSession
from src.db.exceptions import DatabaseError
from src.db.repositories.generation_repo import GenerationRepository

if TYPE_CHECKING:
    from collections.abc import Coroutine

    from aiogram import Bot, Dispatcher
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from fastapi import FastAPI

    from src.config.settings import Settings
    from src.config.yaml_config import YamlConfig

from src.bot.setup import setup_bot
from src.bot.webhook import normalize_domain, remove_webhook, setup_webhook
from src.db.base import get_engine
from src.db.migrations import check_migrations
from src.scheduler import create_scheduler, start_scheduler, stop_scheduler
from src.services.ai_service import AIService, create_ai_service
from src.utils.i18n import init_localization
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = get_logger(__name__)


class ApplicationLifecycle:
    """Управление жизненным циклом приложения.

    Инкапсулирует всю логику startup и shutdown для FastAPI приложения.

    Attributes:
        settings: Настройки приложения из .env
        yaml_config: Конфигурация из config.yaml
        bot: Telegram Bot instance (создаётся при startup)
        dp: aiogram Dispatcher (создаётся при startup)
        scheduler: APScheduler instance для автопродления подписок
        polling_task: asyncio.Task для long polling mode
    """

    def __init__(self, settings: Settings, yaml_config: YamlConfig) -> None:
        """Инициализировать lifecycle manager.

        Args:
            settings: Настройки приложения из .env
            yaml_config: Конфигурация из config.yaml
        """
        self.settings = settings
        self.yaml_config = yaml_config

        # Компоненты, которые создаются при startup
        self.bot: Bot | None = None
        self.dp: Dispatcher | None = None
        self.scheduler: AsyncIOScheduler | None = None
        self.polling_task: asyncio.Task[None] | None = None

        # Фоновые задачи для некритичных операций
        self._background_tasks: list[asyncio.Task[None]] = []

        # AI-сервис (для передачи в setup_bot)
        self._ai_service: AIService | None = None

    async def startup(self, app: FastAPI) -> None:
        """Выполнить startup приложения.

        Создаёт и настраивает все компоненты:
        1. Проверка миграций БД
        2. Бот и диспетчер
        3. AI-сервис и локализация (параллельно с webhook)
        4. Режим работы (polling/webhook)
        5. Планировщик подписок

        Оптимизация: независимые операции выполняются параллельно,
        некритичные HTTP-запросы выносятся в фоновые задачи.

        Args:
            app: FastAPI приложение для сохранения bot и dp в app.state

        Raises:
            SystemExit: При критических ошибках конфигурации webhook
        """
        logger.info("Запуск приложения...")

        # Проверяем статус миграций базы данных
        # Выводим предупреждение если миграции не применены
        await check_migrations(get_engine())

        # Инициализация бота и диспетчера
        await self._init_bot()
        await self._init_dispatcher()

        # Проверяем что инициализация прошла успешно
        assert self.bot is not None, "Bot initialization failed"
        assert self.dp is not None, "Dispatcher initialization failed"

        # Сохраняем bot и dp в app.state для доступа из API endpoints
        app.state.bot = self.bot
        app.state.dp = self.dp

        # === ПАРАЛЛЕЛЬНАЯ ИНИЦИАЛИЗАЦИЯ ===
        # Запускаем независимые операции параллельно для ускорения startup
        if self.settings.app.is_production:
            # Production: webhook + инициализация сервисов параллельно
            await self._startup_production(app)
        else:
            # Development: polling mode с максимальной параллелизацией
            await self._startup_development(app)

        logger.info("✅ Приложение запущено успешно")

    async def _startup_production(self, app: FastAPI) -> None:
        """Startup для production mode (webhook).

        В production важна надёжность, поэтому webhook устанавливается
        синхронно перед запуском остальных компонентов.
        """
        assert self.bot is not None
        assert self.dp is not None
        assert self.settings.app.domain is not None

        # Параллельно инициализируем AI-сервис и локализацию
        self._ai_service, _ = await asyncio.gather(
            asyncio.to_thread(create_ai_service),
            asyncio.to_thread(init_localization),
        )

        # Проверяем что AI-сервис инициализирован
        assert self._ai_service is not None, "AI service initialization failed"

        # Настраиваем бота: middleware, error handlers, роутеры
        setup_bot(
            self.dp,
            self.yaml_config,
            self._ai_service,
            self.bot,
            self.settings.channel,
        )

        # Устанавливаем webhook (критично для production)
        await self._start_webhook_mode()

        # Запускаем планировщик
        await self._start_scheduler(app)

        # Фоновые задачи: регистрация команд, логирование, очистка
        self._start_background_task(
            self._register_commands_background(),
            "register_bot_commands",
        )
        self._start_background_task(
            self._log_startup_urls(),
            "log_startup_urls",
        )
        self._start_background_task(
            self._cleanup_stuck_generations(),
            "cleanup_stuck_generations",
        )

    async def _startup_development(self, app: FastAPI) -> None:
        """Startup для development mode (polling).

        В development важна скорость запуска, поэтому:
        1. Удаление webhook запускается параллельно с инициализацией
        2. Polling стартует сразу после удаления webhook
        3. Некритичные операции выносятся в background
        """
        assert self.bot is not None
        assert self.dp is not None

        logger.info("🔧 Development mode: запускаю long polling")

        # === ФАЗА 1: Параллельная инициализация ===
        # Удаляем webhook параллельно с инициализацией AI и локализации
        webhook_task = asyncio.create_task(
            remove_webhook(self.bot),
            name="remove_webhook",
        )
        ai_task = asyncio.create_task(
            asyncio.to_thread(create_ai_service),
            name="create_ai_service",
        )
        i18n_task = asyncio.create_task(
            asyncio.to_thread(init_localization),
            name="init_localization",
        )

        # Ждём завершения всех задач
        self._ai_service, _, _ = await asyncio.gather(
            ai_task,
            i18n_task,
            webhook_task,
        )

        # Проверяем что AI-сервис инициализирован
        assert self._ai_service is not None, "AI service initialization failed"

        # Настраиваем бота: middleware, error handlers, роутеры
        setup_bot(
            self.dp,
            self.yaml_config,
            self._ai_service,
            self.bot,
            self.settings.channel,
        )

        # === ФАЗА 2: Быстрый старт polling ===
        # Запускаем polling как фоновую задачу СРАЗУ
        self.polling_task = asyncio.create_task(
            self.dp.start_polling(self.bot),
            name="telegram_polling",
        )
        logger.info("✅ Polling mode активирован")

        # === ФАЗА 3: Фоновые задачи ===
        # Запускаем планировщик (быстро, не блокирует)
        await self._start_scheduler(app)

        # Некритичные операции в background
        self._start_background_task(
            self._register_commands_background(),
            "register_bot_commands",
        )
        self._start_background_task(
            self._log_startup_urls(),
            "log_startup_urls",
        )
        self._start_background_task(
            self._cleanup_stuck_generations(),
            "cleanup_stuck_generations",
        )

    def _start_background_task(
        self,
        coro: Coroutine[Any, Any, None],
        name: str,
    ) -> None:
        """Запустить фоновую задачу и сохранить ссылку для cleanup.

        Args:
            coro: Корутина для выполнения в фоне.
            name: Имя задачи для логирования.
        """
        task = asyncio.create_task(coro, name=name)
        self._background_tasks.append(task)

    async def _register_commands_background(self) -> None:
        """Зарегистрировать команды бота в фоне.

        Обёртка для register_bot_commands с обработкой ошибок.
        """
        assert self.bot is not None
        try:
            await register_bot_commands(self.bot)
            logger.debug("✅ Команды бота зарегистрированы")
        except Exception:
            logger.exception("Ошибка регистрации команд бота (некритично)")

    async def shutdown(self) -> None:
        """Выполнить shutdown приложения.

        Останавливает все компоненты в обратном порядке:
        1. Планировщик
        2. Фоновые задачи
        3. Polling task (если был запущен)
        4. Bot session
        """
        logger.info("Остановка приложения...")

        # Останавливаем планировщик
        if self.scheduler is not None:
            stop_scheduler(self.scheduler)
            logger.debug("Планировщик остановлен")

        # Останавливаем фоновые задачи
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        if self._background_tasks:
            logger.debug("Фоновые задачи остановлены: %d", len(self._background_tasks))
        self._background_tasks.clear()

        # Останавливаем polling (если был запущен)
        if self.polling_task is not None:
            self.polling_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.polling_task
            logger.debug("Polling остановлен")

        # Закрываем сессию бота
        if self.bot is not None:
            await self.bot.session.close()
            logger.debug("Bot session закрыта")

        logger.info("✅ Приложение остановлено")

    async def _init_bot(self) -> None:
        """Создать Bot instance."""
        self.bot = create_bot(self.settings.bot.token.get_secret_value())
        logger.debug("Bot создан")

    async def _init_dispatcher(self) -> None:
        """Создать Dispatcher instance."""
        self.dp = create_dispatcher(self.settings.fsm)
        logger.debug("Dispatcher создан")

    async def _start_webhook_mode(self) -> None:
        """Запустить webhook mode для production.

        Нормализует домен и устанавливает webhook на Telegram API.
        При ошибках завершает приложение через sys.exit(1).

        Raises:
            SystemExit: При критических ошибках настройки webhook
        """
        assert self.bot is not None
        assert self.settings.app.domain is not None

        domain = normalize_domain(self.settings.app.domain)
        logger.info("🌐 Production mode: настраиваю webhook для домена %s", domain)

        try:
            webhook_ok = await setup_webhook(self.bot, domain)
            if not webhook_ok:
                # Не удалось установить webhook после всех попыток
                logger.error(
                    "❌ Не удалось установить webhook после 3 попыток. "
                    "Проверьте доступность домена и настройки сети."
                )
                sys.exit(1)

            logger.info("✅ Webhook mode активирован")

        except RuntimeError as e:
            # Ошибка конфигурации (4xx от Telegram API)
            logger.error("❌ Фатальная ошибка конфигурации webhook: %s", e)
            sys.exit(1)

    async def _cleanup_stuck_generations(self) -> None:
        """Очистить зависшие генерации при startup.

        Помечает как FAILED все генерации со статусом PENDING,
        которые были созданы больше максимального таймаута назад.

        Это критически важно для очистки "утечек" при:
        - Крешах сервера во время обработки
        - Таймаутах запросов к AI-провайдерам
        - Неожиданных завершениях процессов
        """
        # Используем максимальный таймаут из всех типов генерации
        max_timeout = max(
            self.yaml_config.generation_timeouts.chat,
            self.yaml_config.generation_timeouts.image,
            self.yaml_config.generation_timeouts.image_edit,
        )

        # Добавляем запас времени (2x) для безопасности
        # Это гарантирует, что мы не пометим как зависшую генерацию,
        # которая ещё может завершиться
        cleanup_timeout = max_timeout * 2

        try:
            async with DatabaseSession() as session:
                repo = GenerationRepository(session)
                cleaned_count = await repo.cleanup_stuck_generations(cleanup_timeout)

                if cleaned_count > 0:
                    logger.warning(
                        "⚠️  Очищено зависших генераций: %d (старше %d сек)",
                        cleaned_count,
                        cleanup_timeout,
                    )
                else:
                    logger.debug("✅ Зависших генераций не найдено")

        except (OSError, RuntimeError, DatabaseError) as e:
            # Не падаем при ошибке cleanup - это не критично для запуска
            # OSError — ошибки файловой системы, RuntimeError — ошибки asyncio
            # DatabaseError — ошибки БД (от SQLAlchemy)
            logger.error("Ошибка при очистке зависших генераций: %s", e)

    async def _start_scheduler(self, app: FastAPI) -> None:
        """Запустить планировщик задач.

        Планировщик выполняет:
        - Автопродление подписок (если есть подписочные тарифы)
        - Обработку рассылок (каждые 30 секунд)

        Args:
            app: FastAPI приложение для сохранения scheduler в app.state
        """
        self.scheduler = create_scheduler(self.yaml_config, self.bot)  # type: ignore[arg-type]
        start_scheduler(self.scheduler)

        # Сохраняем scheduler в app.state для возможного доступа из API
        app.state.scheduler = self.scheduler

        logger.info("✅ Планировщик запущен")

    async def _log_startup_urls(self) -> None:
        """Вывести в лог ссылки на бота и админку после полного запуска."""
        assert self.bot is not None

        # Получаем информацию о боте для формирования ссылки
        bot_info = await self.bot.get_me()
        bot_url = f"https://t.me/{bot_info.username}"
        logger.info("🤖 Бот: %s", bot_url)

        # Формируем URL админки если она включена
        if self.settings.admin.is_enabled:
            if self.settings.app.domain:
                domain = self.settings.app.domain.rstrip("/")
                if not domain.startswith("http"):
                    domain = f"https://{domain}"
                admin_url = f"{domain}/admin"
            else:
                # В dev-режиме показываем localhost с дефолтным портом
                admin_url = "http://localhost:8000/admin"
            logger.info("🔧 Админка: %s", admin_url)
