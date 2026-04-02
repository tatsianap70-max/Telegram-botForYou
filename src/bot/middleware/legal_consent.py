"""Middleware для проверки согласия пользователя с юридическими документами.

Этот модуль блокирует пользователей, которые не приняли условия использования.
Проверка включается через legal.enabled в config.yaml.

Логика работы:
1. Если legal.enabled=false или документы не настроены — пропускаем все запросы
2. Получаем пользователя из БД и проверяем accepted_legal_version
3. Результат кешируется на длительный срок (версия документов меняется редко)
4. Если пользователь не принял условия — показываем запрос на согласие
5. Callback "legal:accept" пропускается для обработки согласия

Кеширование:
- Результат проверки хранится в памяти (in-memory)
- Формат кеша: {user_id: (accepted_version: str | None, expires_at: float)}
- TTL по умолчанию: 24 часа (версия документов меняется редко)
- Кеш очищается при принятии условий

Dependency Injection:
- Middleware принимает legal_config как параметр
- Это позволяет легко тестировать с mock-объектами
"""

import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Protocol

from aiogram import BaseMiddleware
from aiogram.exceptions import TelegramAPIError
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    TelegramObject,
    User,
)
from typing_extensions import override

from src.db.base import DatabaseSession
from src.db.repositories.user_repo import UserRepository
from src.utils.i18n import Localization
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from src.config.yaml_config import LegalConfig

logger = get_logger(__name__)

# Callback для принятия условий — должен проходить без проверки
CALLBACK_LEGAL_ACCEPT = "legal:accept"
START_LANGUAGE_CALLBACK_PREFIX = "start_lang:"
START_COMMAND_PREFIX = "/start"

# TTL кеша по умолчанию: 24 часа (86400 секунд)
# Версия документов меняется редко, поэтому можно кешировать надолго
DEFAULT_CACHE_TTL_SECONDS = 86400


class TimeProvider(Protocol):
    """Протокол для получения текущего времени (для тестирования)."""

    def __call__(self) -> float:
        """Вернуть текущее время в секундах (epoch)."""
        ...


def default_time_provider() -> float:
    """Стандартный провайдер времени — time.time()."""
    return time.time()


class LegalConsentMiddleware(BaseMiddleware):
    """Middleware для проверки согласия с юридическими документами.

    Блокирует пользователей, которые не приняли условия использования.
    Результат проверки кешируется для уменьшения нагрузки на БД.

    Attributes:
        _legal_config: Конфигурация юридических документов из config.yaml.
        _cache_ttl_seconds: Время жизни кеша в секундах.
        _cache: Словарь {user_id: (accepted_version: str | None, expires_at: float)}.
        _time_provider: Функция для получения текущего времени.
    """

    def __init__(
        self,
        legal_config: "LegalConfig",
        cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
        time_provider: TimeProvider | None = None,
    ) -> None:
        """Создать middleware для проверки согласия.

        Args:
            legal_config: Конфигурация юридических документов.
            cache_ttl_seconds: Время жизни кеша результата проверки.
                По умолчанию 86400 секунд (24 часа).
            time_provider: Провайдер времени (для тестирования).
        """
        super().__init__()
        self._legal_config = legal_config
        self._cache_ttl_seconds = cache_ttl_seconds
        self._time_provider = time_provider or default_time_provider

        # Кеш: {user_id: (accepted_version: str | None, expires_at: float)}
        self._cache: dict[int, tuple[str | None, float]] = {}

        logger.info(
            "LegalConsentMiddleware инициализирован: version=%s, cache_ttl=%d сек",
            legal_config.version,
            cache_ttl_seconds,
        )

    @staticmethod
    def _is_start_command(event: TelegramObject) -> bool:
        """Проверить, что событие — команда /start."""
        if not isinstance(event, Message):
            return False
        event_text = getattr(event, "text", None)
        return isinstance(event_text, str) and event_text.startswith(
            START_COMMAND_PREFIX
        )

    @staticmethod
    def _is_start_language_callback(event: TelegramObject) -> bool:
        """Проверить, что callback относится к стартовому выбору языка."""
        if not isinstance(event, CallbackQuery):
            return False
        return isinstance(event.data, str) and event.data.startswith(
            START_LANGUAGE_CALLBACK_PREFIX
        )

    @override
    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        """Проверить согласие перед выполнением handler.

        Логика:
        1. Если это callback "legal:accept" — пропускаем для обработки согласия
        2. Получаем user_id из события
        3. Проверяем кеш — если есть валидный результат, используем его
        4. Если кеша нет — проверяем в БД
        5. Если не принял условия — блокируем и показываем запрос

        Args:
            handler: Обработчик события (следующий в цепочке middleware).
            event: Входящее событие (Message, CallbackQuery и т.д.).
            data: Словарь данных, передаваемых в handler.

        Returns:
            Результат выполнения handler или None (если заблокирован).
        """
        # Получаем пользователя из события
        telegram_user: User | None = data.get("event_from_user")
        if telegram_user is None:
            # Событие без пользователя — пропускаем
            return await handler(event, data)

        user_id = telegram_user.id

        # Callback "legal:accept" должен проходить без проверки
        # Иначе пользователь не сможет принять условия
        if isinstance(event, CallbackQuery) and event.data == CALLBACK_LEGAL_ACCEPT:
            # Очищаем кеш — после принятия нужна актуальная проверка
            self.clear_cache(user_id)
            return await handler(event, data)

        # /start проходит до legal-check, чтобы показать входной language gate.
        if self._is_start_command(event):
            return await handler(event, data)

        # Callback стартового выбора языка должен проходить до legal-check.
        if self._is_start_language_callback(event):
            return await handler(event, data)

        # Проверяем согласие (из кеша или БД)
        # Передаём telegram_user для автоматической регистрации если пользователя нет
        has_consent = await self._check_consent(user_id, telegram_user)

        if has_consent:
            # Пользователь принял условия — пропускаем к handler
            return await handler(event, data)

        # Пользователь НЕ принял условия — блокируем
        logger.debug(
            "Пользователь %d заблокирован: не принял условия использования",
            user_id,
        )

        # Получаем объект локализации для сообщений
        l10n: Localization | None = data.get("l10n")

        # Показываем запрос на согласие
        await self._send_consent_required_message(event, l10n)

        # Возвращаем None — handler не вызывается
        return None

    async def _check_consent(self, user_id: int, telegram_user: User) -> bool:
        """Проверить согласие пользователя с документами.

        Сначала проверяем кеш. Если кеш валиден — сравниваем версии.
        Если кеша нет или он устарел — проверяем в БД.
        Если пользователя нет в БД — создаём его автоматически.

        Args:
            user_id: ID пользователя в Telegram.
            telegram_user: Объект User из aiogram для автоматической регистрации.

        Returns:
            True если принял актуальную версию, False если нет.
        """
        current_time = self._time_provider()
        required_version = self._legal_config.version

        # Проверяем кеш
        if user_id in self._cache:
            accepted_version, expires_at = self._cache[user_id]
            if current_time < expires_at:
                # Кеш валиден — сравниваем версии
                has_consent = accepted_version == required_version
                logger.debug(
                    "Кеш согласия для user_id=%d: accepted=%s, required=%s, ok=%s",
                    user_id,
                    accepted_version,
                    required_version,
                    has_consent,
                )
                return has_consent

            # Кеш устарел — удаляем
            del self._cache[user_id]

        # Кеша нет — проверяем в БД (и создаём пользователя если его нет)
        accepted_version = await self._get_or_create_user_and_check(
            user_id, telegram_user
        )

        # Сохраняем в кеш
        if self._cache_ttl_seconds > 0:
            expires_at = current_time + self._cache_ttl_seconds
            self._cache[user_id] = (accepted_version, expires_at)

        has_consent = accepted_version == required_version
        logger.debug(
            "Проверка согласия в БД: user_id=%d, accepted=%s, required=%s, ok=%s",
            user_id,
            accepted_version,
            required_version,
            has_consent,
        )

        return has_consent

    async def _get_or_create_user_and_check(
        self, user_id: int, telegram_user: User
    ) -> str | None:
        """Получить версию принятых документов из БД.

        Если пользователя нет в БД — создаёт его автоматически.
        Это позволяет новым пользователям сразу видеть запрос на согласие
        и принимать условия без необходимости вызывать /start.

        Args:
            user_id: ID пользователя в Telegram.
            telegram_user: Объект User из aiogram для создания записи.

        Returns:
            Версия принятых документов или None если не принял.
        """
        async with DatabaseSession() as session:
            repo = UserRepository(session)
            user, created = await repo.get_or_create(
                telegram_id=user_id,
                username=telegram_user.username,
                first_name=telegram_user.first_name,
                last_name=telegram_user.last_name,
                language=telegram_user.language_code or "ru",
            )

            if created:
                logger.info(
                    "Автоматически создан пользователь в LegalConsentMiddleware: "
                    "telegram_id=%d, username=%s",
                    user_id,
                    telegram_user.username,
                )

            return user.accepted_legal_version

    def _create_consent_keyboard(
        self, l10n: Localization | None
    ) -> InlineKeyboardMarkup:
        """Создать клавиатуру для запроса согласия.

        Args:
            l10n: Объект локализации для текстов кнопок.

        Returns:
            InlineKeyboardMarkup с кнопками документов и согласия.
        """
        # Тексты кнопок с fallback на русский
        privacy_text = (
            l10n.get("legal_privacy_policy_button")
            if l10n
            else "📄 Политика конфиденциальности"
        )
        terms_text = (
            l10n.get("legal_terms_of_service_button")
            if l10n
            else "📜 Пользовательское соглашение"
        )
        accept_text = l10n.get("legal_accept_button") if l10n else "✅ Принимаю"

        buttons = [
            [
                InlineKeyboardButton(
                    text=privacy_text,
                    url=self._legal_config.privacy_policy_url,
                )
            ],
            [
                InlineKeyboardButton(
                    text=terms_text,
                    url=self._legal_config.terms_of_service_url,
                )
            ],
            [
                InlineKeyboardButton(
                    text=accept_text,
                    callback_data=CALLBACK_LEGAL_ACCEPT,
                )
            ],
        ]

        return InlineKeyboardMarkup(inline_keyboard=buttons)

    async def _send_consent_required_message(
        self,
        event: TelegramObject,
        l10n: Localization | None,
    ) -> None:
        """Отправить сообщение с запросом на согласие.

        Args:
            event: Событие (Message или CallbackQuery).
            l10n: Объект локализации для текстов.
        """
        # Создаём клавиатуру для согласия
        keyboard = self._create_consent_keyboard(l10n)

        # Получаем текст сообщения
        if l10n is not None:
            text = l10n.get("legal_acceptance_request")
        else:
            text = (
                "Для использования бота необходимо принять условия использования.\n\n"
                "Пожалуйста, ознакомьтесь с документами и нажмите «Принимаю»."
            )

        # Отправляем сообщение в зависимости от типа события
        try:
            if isinstance(event, Message):
                await event.answer(text, reply_markup=keyboard)
            elif isinstance(event, CallbackQuery):
                await event.answer(show_alert=False)
                if event.message:
                    await event.message.answer(text, reply_markup=keyboard)
        except TelegramAPIError as e:
            logger.warning(
                "Не удалось отправить запрос на согласие: %s",
                e,
            )

    def clear_cache(self, user_id: int | None = None) -> None:
        """Очистить кеш согласия.

        Используется после принятия условий для актуализации данных.

        Args:
            user_id: ID пользователя для очистки.
                Если None — очищается весь кеш.
        """
        if user_id is None:
            count = len(self._cache)
            self._cache.clear()
            logger.info("Кеш согласия полностью очищен (%d записей)", count)
        elif user_id in self._cache:
            del self._cache[user_id]
            logger.debug("Кеш согласия очищен для user_id=%d", user_id)

    def update_cache(self, user_id: int, accepted_version: str) -> None:
        """Обновить кеш после принятия условий.

        Вызывается из handler после успешного сохранения согласия в БД.

        Args:
            user_id: ID пользователя.
            accepted_version: Версия принятых документов.
        """
        if self._cache_ttl_seconds > 0:
            expires_at = self._time_provider() + self._cache_ttl_seconds
            self._cache[user_id] = (accepted_version, expires_at)
            logger.debug(
                "Кеш согласия обновлён: user_id=%d, version=%s",
                user_id,
                accepted_version,
            )


def create_legal_consent_middleware(
    legal_config: "LegalConfig",
    cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
) -> LegalConsentMiddleware:
    """Создать LegalConsentMiddleware с production зависимостями.

    Factory function для удобного создания middleware.

    Args:
        legal_config: Конфигурация юридических документов.
        cache_ttl_seconds: TTL кеша в секундах (по умолчанию 24 часа).

    Returns:
        Настроенный LegalConsentMiddleware.

    Example:
        if yaml_config.legal.enabled and yaml_config.legal.has_documents():
            middleware = create_legal_consent_middleware(yaml_config.legal)
            dp.message.middleware(middleware)
            dp.callback_query.middleware(middleware)
    """
    return LegalConsentMiddleware(
        legal_config=legal_config,
        cache_ttl_seconds=cache_ttl_seconds,
    )
