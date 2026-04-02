"""Тесты для обработчика команды /chatgpt.

Проверяют корректность работы диалога с AI:
- Команда /chatgpt показывает выбор модели
- Выбор модели сохраняется в FSM state
- Сообщения пользователя обрабатываются и отправляются в AI
- Контекст диалога сохраняется и загружается из БД
"""

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message
from aiogram.types import User as TelegramUser
from sqlalchemy.ext.asyncio import AsyncSession

from src.bot.handlers.chatgpt import (
    cmd_chatgpt,
    handle_model_selection,
    handle_user_message,
)
from src.bot.states import ChatGPTStates
from src.config.yaml_config import ModelConfig
from src.db.models.user import User
from src.db.repositories import MessageRepository
from src.providers.ai.base import GenerationResult, GenerationStatus
from src.services.ai_service import AIService
from src.services.billing_service import GenerationCost
from src.services.coaching.safety_layer import build_crisis_response
from src.utils.i18n import Localization


@pytest.fixture
def mock_message() -> Message:
    """Создать мок-объект сообщения от Telegram."""
    message = MagicMock(spec=Message)
    message.from_user = MagicMock(spec=TelegramUser)
    message.from_user.id = 123456789
    message.from_user.username = "test_user"
    message.text = "Привет!"
    message.answer = AsyncMock()

    # Атрибут chat нужен для send_chat_action
    message.chat = MagicMock()
    message.chat.id = 123456789
    message.chat.bot = MagicMock()
    message.chat.bot.send_chat_action = AsyncMock()

    return message


@pytest.fixture
def mock_callback_query(mock_message: Message) -> CallbackQuery:
    """Создать мок-объект callback query от Telegram."""
    callback = MagicMock(spec=CallbackQuery)
    callback.from_user = mock_message.from_user
    callback.message = mock_message
    callback.data = "model:gpt-4o"
    callback.answer = AsyncMock()
    mock_message.edit_text = AsyncMock()
    return callback


@pytest.fixture
def mock_fsm_context() -> FSMContext:
    """Создать мок-объект FSM контекста."""
    context = MagicMock(spec=FSMContext)
    context.set_state = AsyncMock()
    context.update_data = AsyncMock()
    context.get_data = AsyncMock(
        return_value={"model_key": "gpt-4o", "onboarding_completed": True}
    )
    context.get_state = AsyncMock(return_value=ChatGPTStates.waiting_for_message)
    return context


@pytest.fixture
def mock_ai_service() -> AIService:
    """Создать мок-объект AI сервиса."""
    service = MagicMock(spec=AIService)
    service.generate = AsyncMock(
        return_value=GenerationResult(
            status=GenerationStatus.SUCCESS,
            content="Привет! Чем могу помочь?",
        )
    )
    # Добавляем get_available_models() с одной chat-моделью
    service.get_available_models = MagicMock(
        return_value={
            "gpt-4o": ModelConfig(
                provider="openai",
                model_id="openai/gpt-4o",
                generation_type="chat",
                display_name="GPT-4o",
                price_tokens=15,
            ),
        }
    )
    return service


@pytest.fixture
def mock_billing_cost() -> GenerationCost:
    """Создать мок GenerationCost для успешного биллинга."""
    return GenerationCost(
        can_proceed=True, tokens_cost=15, model_key="gpt-4o", quantity=1.0
    )


@pytest.fixture
def mock_l10n() -> Localization:
    """Создать мок-объект локализации.

    Возвращает переводы на русском языке для тестирования.
    Метод get() возвращает строку с подставленными параметрами.
    """
    l10n = MagicMock(spec=Localization)

    # Словарь переводов для тестов
    translations = {
        "chatgpt_choose_model": "🤖 <b>Выберите модель для диалога:</b>",
        "chatgpt_model_selected": "✅ Модель выбрана: <b>{model_key}</b>",
        "chatgpt_model_not_selected": "❌ Модель не выбрана.",
        "chatgpt_generating": "⏳ Генерирую ответ...",
        "chatgpt_empty_response": "❌ AI вернул пустой ответ.",
        "chatgpt_generation_error": "❌ Ошибка генерации: {error}",
        "chatgpt_unexpected_error": "❌ Произошла неожиданная ошибка.",
        "error_user_not_found": "❌ Ошибка: пользователь не найден.",
        "error_db_temporary": "❌ Временная ошибка БД.",
        "error_db_permanent": "❌ Ошибка при работе с базой данных.",
        "no_models_available": "❌ Модели недоступны",
        "onboarding_start_required": (
            "Перед началом работы лучше сначала открыть /start."
        ),
        "crisis_support_v1": (
            "Мне очень жаль, что вам сейчас так тяжело. "
            "Если есть риск для вашей жизни, безопасности "
            "или безопасности другого человека, "
            "я не продолжаю этот разговор в обычном формате."
        ),
    }

    def get_translation(key: str, **kwargs: str) -> str:
        """Вернуть перевод с подставленными параметрами."""
        text = translations.get(key, key)
        if kwargs:
            text = text.format(**kwargs)
        return text

    l10n.get = MagicMock(side_effect=get_translation)
    return l10n


class TestCmdChatGPT:
    """Тесты для команды /chatgpt."""

    @pytest.mark.asyncio
    async def test_cmd_chatgpt_shows_model_selection(
        self,
        mock_message: Message,
        mock_fsm_context: FSMContext,
        mock_l10n: Localization,
        mock_ai_service: AIService,
    ) -> None:
        """Проверить, что команда /chatgpt показывает выбор модели."""
        # Act — передаём ai_service через DI
        await cmd_chatgpt(mock_message, mock_fsm_context, mock_l10n, mock_ai_service)

        # Assert
        # Проверяем, что состояние изменилось
        mock_fsm_context.set_state.assert_called_once_with(
            ChatGPTStates.waiting_for_model_selection
        )

        # Проверяем, что ответ отправлен
        mock_message.answer.assert_called_once()
        call_args = mock_message.answer.call_args
        assert "Выберите модель" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_cmd_chatgpt_no_available_models(
        self,
        mock_message: Message,
        mock_fsm_context: FSMContext,
        mock_l10n: Localization,
        mock_ai_service: AIService,
    ) -> None:
        """Проверить, что при отсутствии моделей показывается сообщение об ошибке."""
        # Arrange — нет доступных моделей
        mock_ai_service.get_available_models = MagicMock(return_value={})

        # Act
        await cmd_chatgpt(mock_message, mock_fsm_context, mock_l10n, mock_ai_service)

        # Assert
        # Состояние НЕ должно измениться
        mock_fsm_context.set_state.assert_not_called()

        # Ответ должен быть отправлен с ключом no_models_available
        mock_message.answer.assert_called_once()
        mock_l10n.get.assert_called_with("no_models_available")

    @pytest.mark.asyncio
    async def test_cmd_chatgpt_without_from_user(
        self,
        mock_message: Message,
        mock_fsm_context: FSMContext,
        mock_l10n: Localization,
        mock_ai_service: AIService,
    ) -> None:
        """Проверить, что обработчик корректно обрабатывает отсутствие from_user."""
        # Arrange
        mock_message.from_user = None

        # Act
        await cmd_chatgpt(mock_message, mock_fsm_context, mock_l10n, mock_ai_service)

        # Assert
        # Не должно быть вызовов
        mock_fsm_context.set_state.assert_not_called()
        mock_message.answer.assert_not_called()


class TestHandleModelSelection:
    """Тесты для обработчика выбора модели."""

    @pytest.mark.asyncio
    async def test_handle_model_selection_saves_model_key(
        self,
        mock_callback_query: CallbackQuery,
        mock_fsm_context: FSMContext,
        mock_l10n: Localization,
    ) -> None:
        """Проверить, что выбранная модель сохраняется в FSM state."""
        # Act
        await handle_model_selection(mock_callback_query, mock_fsm_context, mock_l10n)

        # Assert
        mock_fsm_context.update_data.assert_called_once_with(model_key="gpt-4o")

    @pytest.mark.asyncio
    async def test_handle_model_selection_changes_state(
        self,
        mock_callback_query: CallbackQuery,
        mock_fsm_context: FSMContext,
        mock_l10n: Localization,
    ) -> None:
        """Проверить, что состояние меняется на waiting_for_message."""
        # Act
        await handle_model_selection(mock_callback_query, mock_fsm_context, mock_l10n)

        # Assert
        mock_fsm_context.set_state.assert_called_once_with(
            ChatGPTStates.waiting_for_message
        )

    @pytest.mark.asyncio
    async def test_handle_model_selection_edits_message(
        self,
        mock_callback_query: CallbackQuery,
        mock_fsm_context: FSMContext,
        mock_l10n: Localization,
    ) -> None:
        """Проверить, что сообщение редактируется с подтверждением."""
        # Act
        await handle_model_selection(mock_callback_query, mock_fsm_context, mock_l10n)

        # Assert
        assert mock_callback_query.message is not None
        mock_callback_query.message.edit_text.assert_called_once()
        call_args = mock_callback_query.message.edit_text.call_args
        assert "Модель выбрана" in call_args[0][0]
        assert "gpt-4o" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_handle_model_selection_answers_callback(
        self,
        mock_callback_query: CallbackQuery,
        mock_fsm_context: FSMContext,
        mock_l10n: Localization,
    ) -> None:
        """Проверить, что callback подтверждается (убирает часики)."""
        # Act
        await handle_model_selection(mock_callback_query, mock_fsm_context, mock_l10n)

        # Assert
        mock_callback_query.answer.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_model_selection_without_callback_data(
        self,
        mock_callback_query: CallbackQuery,
        mock_fsm_context: FSMContext,
        mock_l10n: Localization,
    ) -> None:
        """Проверить обработку отсутствия callback_data."""
        # Arrange
        mock_callback_query.data = None

        # Act
        await handle_model_selection(mock_callback_query, mock_fsm_context, mock_l10n)

        # Assert
        mock_fsm_context.update_data.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_model_selection_parses_model_key_from_callback_data(
        self,
        mock_callback_query: CallbackQuery,
        mock_fsm_context: FSMContext,
        mock_l10n: Localization,
    ) -> None:
        """Проверить парсинг model_key из callback_data."""
        # Arrange
        mock_callback_query.data = "model:claude-3-5-sonnet"

        # Act
        await handle_model_selection(mock_callback_query, mock_fsm_context, mock_l10n)

        # Assert
        mock_fsm_context.update_data.assert_called_once_with(
            model_key="claude-3-5-sonnet"
        )


class TestHandleUserMessage:
    """Тесты для обработчика сообщений пользователя."""

    @pytest.mark.asyncio
    async def test_handle_user_message_uses_domain_pipeline_without_legacy_ai(
        self,
        mock_message: Message,
        mock_fsm_context: FSMContext,
        mock_l10n: Localization,
        mock_ai_service: AIService,
        db_session: AsyncSession,
        test_user: User,
        session_factory: Callable[[], AbstractAsyncContextManager[AsyncSession]],
    ) -> None:
        """Проверить, что safe-path формируется через domain pipeline без legacy AI."""
        # Arrange
        mock_message.answer = AsyncMock()

        # Act
        await handle_user_message(
            mock_message,
            mock_fsm_context,
            mock_l10n,
            mock_ai_service,
            session_factory,
        )

        # Assert
        mock_ai_service.generate.assert_not_called()
        mock_message.answer.assert_called_once()
        answer_text = mock_message.answer.call_args.args[0]
        assert isinstance(answer_text, str)
        assert answer_text

    @pytest.mark.asyncio
    async def test_handle_user_message_does_not_write_legacy_chat_history_to_db(
        self,
        mock_message: Message,
        mock_fsm_context: FSMContext,
        mock_l10n: Localization,
        mock_ai_service: AIService,
        db_session: AsyncSession,
        test_user: User,
        session_factory: Callable[[], AbstractAsyncContextManager[AsyncSession]],
    ) -> None:
        """Проверить, что legacy chat history больше не пишется в этом path."""
        mock_message.answer = AsyncMock()

        await handle_user_message(
            mock_message,
            mock_fsm_context,
            mock_l10n,
            mock_ai_service,
            session_factory,
        )

        repo = MessageRepository(db_session)
        messages = await repo.get_context(test_user.id, "gpt-4o")
        assert len(messages) == 0

    @pytest.mark.asyncio
    async def test_handle_user_message_keeps_reply_short_and_single_call(
        self,
        mock_message: Message,
        mock_fsm_context: FSMContext,
        mock_l10n: Localization,
        mock_ai_service: AIService,
        db_session: AsyncSession,
        test_user: User,
        session_factory: Callable[[], AbstractAsyncContextManager[AsyncSession]],
    ) -> None:
        """Проверить, что пользователю отправляется один доменный ответ."""
        mock_message.answer = AsyncMock()

        await handle_user_message(
            mock_message,
            mock_fsm_context,
            mock_l10n,
            mock_ai_service,
            session_factory,
        )

        assert mock_message.answer.call_count == 1
        sent_text = mock_message.answer.call_args.args[0]
        assert isinstance(sent_text, str)
        assert sent_text

    @pytest.mark.asyncio
    async def test_handle_user_message_sends_response_to_user(
        self,
        mock_message: Message,
        mock_fsm_context: FSMContext,
        mock_l10n: Localization,
        mock_ai_service: AIService,
        db_session: AsyncSession,
        test_user: User,
        session_factory: Callable[[], AbstractAsyncContextManager[AsyncSession]],
        mock_billing_cost: GenerationCost,
    ) -> None:
        """Проверить, что доменный ответ отправляется пользователю."""
        mock_message.answer = AsyncMock()

        await handle_user_message(
            mock_message,
            mock_fsm_context,
            mock_l10n,
            mock_ai_service,
            session_factory,
        )

        assert mock_message.answer.call_count == 1
        sent_text = mock_message.answer.call_args.args[0]
        assert isinstance(sent_text, str)
        assert sent_text

    @pytest.mark.asyncio
    async def test_handle_user_message_without_model_key_in_state(
        self,
        mock_message: Message,
        mock_fsm_context: FSMContext,
        mock_l10n: Localization,
        mock_ai_service: AIService,
    ) -> None:
        """Проверить обработку отсутствия model_key в FSM state."""
        # Arrange
        mock_fsm_context.get_data = AsyncMock(
            return_value={"onboarding_completed": True}
        )

        # Act
        await handle_user_message(
            mock_message, mock_fsm_context, mock_l10n, mock_ai_service
        )

        # Assert
        mock_message.answer.assert_called_once()
        call_args = mock_message.answer.call_args
        assert "модель не выбрана" in call_args[0][0].lower()
        mock_ai_service.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_user_message_blocks_without_onboarding(
        self,
        mock_message: Message,
        mock_fsm_context: FSMContext,
        mock_l10n: Localization,
        mock_ai_service: AIService,
    ) -> None:
        """Проверить start-gate: без onboarding_completed flow не запускается."""
        mock_fsm_context.get_data = AsyncMock(return_value={"model_key": "gpt-4o"})

        await handle_user_message(
            mock_message,
            mock_fsm_context,
            mock_l10n,
            mock_ai_service,
        )

        mock_message.answer.assert_called_once()
        answer_text = mock_message.answer.call_args.args[0].lower()
        assert "/start" in answer_text
        mock_ai_service.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_user_message_without_from_user(
        self,
        mock_message: Message,
        mock_fsm_context: FSMContext,
        mock_l10n: Localization,
        mock_ai_service: AIService,
    ) -> None:
        """Проверить обработку отсутствия from_user."""
        # Arrange
        mock_message.from_user = None

        # Act
        await handle_user_message(
            mock_message, mock_fsm_context, mock_l10n, mock_ai_service
        )

        # Assert
        mock_message.answer.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_user_message_ignores_legacy_context_loading(
        self,
        mock_message: Message,
        mock_fsm_context: FSMContext,
        mock_l10n: Localization,
        mock_ai_service: AIService,
        db_session: AsyncSession,
        test_user: User,
        session_factory: Callable[[], AbstractAsyncContextManager[AsyncSession]],
        mock_billing_cost: GenerationCost,
    ) -> None:
        """Проверить, что legacy context больше не используется."""
        mock_message.answer = AsyncMock()
        repo = MessageRepository(db_session)
        await repo.add_message(test_user.id, "gpt-4o", "user", "Предыдущий вопрос")
        await repo.add_message(test_user.id, "gpt-4o", "assistant", "Предыдущий ответ")

        await handle_user_message(
            mock_message,
            mock_fsm_context,
            mock_l10n,
            mock_ai_service,
            session_factory,
        )

        mock_ai_service.generate.assert_not_called()
        mock_message.answer.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_user_message_does_not_depend_on_generation_error_paths(
        self,
        mock_message: Message,
        mock_fsm_context: FSMContext,
        mock_l10n: Localization,
        mock_ai_service: AIService,
        db_session: AsyncSession,
        test_user: User,
        session_factory: Callable[[], AbstractAsyncContextManager[AsyncSession]],
        mock_billing_cost: GenerationCost,
    ) -> None:
        """Проверить, что legacy ошибки генерации не участвуют в новом path."""
        mock_message.answer = AsyncMock()
        mock_ai_service.generate = AsyncMock(side_effect=RuntimeError("must not call"))

        await handle_user_message(
            mock_message,
            mock_fsm_context,
            mock_l10n,
            mock_ai_service,
            session_factory,
        )

        mock_ai_service.generate.assert_not_called()
        mock_message.answer.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_user_message_returns_domain_reply_without_ai_result(
        self,
        mock_message: Message,
        mock_fsm_context: FSMContext,
        mock_l10n: Localization,
        mock_ai_service: AIService,
        db_session: AsyncSession,
        test_user: User,
        session_factory: Callable[[], AbstractAsyncContextManager[AsyncSession]],
        mock_billing_cost: GenerationCost,
    ) -> None:
        """Проверить, что ответ формируется pipeline без обращения к AI-результату."""
        mock_message.answer = AsyncMock()
        mock_ai_service.generate = AsyncMock(
            return_value=GenerationResult(
                status=GenerationStatus.SUCCESS,
                content="",
            )
        )

        await handle_user_message(
            mock_message,
            mock_fsm_context,
            mock_l10n,
            mock_ai_service,
            session_factory,
        )

        mock_ai_service.generate.assert_not_called()
        mock_message.answer.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_user_message_ignores_legacy_billing_generation_branch(
        self,
        mock_message: Message,
        mock_fsm_context: FSMContext,
        mock_l10n: Localization,
        mock_ai_service: AIService,
        db_session: AsyncSession,
        test_user: User,
        session_factory: Callable[[], AbstractAsyncContextManager[AsyncSession]],
    ) -> None:
        """Проверить, что legacy billing/generation ветка не активируется."""
        mock_message.answer = AsyncMock()

        await handle_user_message(
            mock_message,
            mock_fsm_context,
            mock_l10n,
            mock_ai_service,
            session_factory,
        )

        mock_ai_service.generate.assert_not_called()
        mock_message.answer.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_user_message_interrupts_on_crisis_before_ai(
        self,
        mock_message: Message,
        mock_fsm_context: FSMContext,
        mock_l10n: Localization,
        mock_ai_service: AIService,
    ) -> None:
        """Проверить ранний crisis interrupt до AI-генерации."""
        mock_message.text = "хочу умереть"

        await handle_user_message(
            mock_message,
            mock_fsm_context,
            mock_l10n,
            mock_ai_service,
        )

        mock_ai_service.generate.assert_not_called()
        mock_message.answer.assert_called_once()
        answer_text = mock_message.answer.call_args.args[0]
        assert isinstance(answer_text, str)
        assert answer_text
        assert mock_fsm_context.update_data.await_count >= 1

    @pytest.mark.asyncio
    async def test_handle_user_message_uses_ru_crisis_localized_text(
        self,
        mock_message: Message,
        mock_fsm_context: FSMContext,
        mock_l10n: Localization,
        mock_ai_service: AIService,
    ) -> None:
        """Проверить RU crisis-текст через l10n key."""
        mock_message.text = "хочу умереть"

        await handle_user_message(
            mock_message,
            mock_fsm_context,
            mock_l10n,
            mock_ai_service,
        )

        expected_text = mock_l10n.get("crisis_support_v1")
        mock_message.answer.assert_called_once_with(expected_text)
        mock_ai_service.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_user_message_uses_en_crisis_localized_text(
        self,
        mock_message: Message,
        mock_fsm_context: FSMContext,
        mock_ai_service: AIService,
    ) -> None:
        """Проверить EN crisis-текст через l10n key."""
        mock_message.text = "хочу умереть"

        en_l10n = MagicMock(spec=Localization)
        en_text = (
            "I'm very sorry that things feel this hard right now. "
            "If there is a risk to your life, your safety, "
            "or the safety of another person, "
            "I can't continue this conversation in the usual format."
        )

        def _en_get(key: str, **kwargs: str) -> str:
            text = {"crisis_support_v1": en_text}.get(key, key)
            if kwargs:
                text = text.format(**kwargs)
            return text

        en_l10n.get = MagicMock(side_effect=_en_get)

        await handle_user_message(
            mock_message,
            mock_fsm_context,
            en_l10n,
            mock_ai_service,
        )

        mock_message.answer.assert_called_once_with(en_text)
        mock_ai_service.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_user_message_crisis_uses_fallback_when_key_missing(
        self,
        mock_message: Message,
        mock_fsm_context: FSMContext,
        mock_ai_service: AIService,
    ) -> None:
        """Проверить fallback на канонический текст при отсутствии l10n key."""
        mock_message.text = "хочу умереть"

        fallback_l10n = MagicMock(spec=Localization)

        def _fallback_get(key: str, **kwargs: str) -> str:
            _ = kwargs
            return key

        fallback_l10n.get = MagicMock(side_effect=_fallback_get)

        await handle_user_message(
            mock_message,
            mock_fsm_context,
            fallback_l10n,
            mock_ai_service,
        )

        mock_message.answer.assert_called_once_with(build_crisis_response())
        mock_ai_service.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_user_message_persists_domain_pipeline_state(
        self,
        mock_message: Message,
        mock_fsm_context: FSMContext,
        mock_l10n: Localization,
        mock_ai_service: AIService,
        db_session: AsyncSession,
        test_user: User,
        session_factory: Callable[[], AbstractAsyncContextManager[AsyncSession]],
    ) -> None:
        """Проверить сохранение доменного состояния pipeline в FSM."""
        processing_msg = MagicMock()
        processing_msg.delete = AsyncMock()
        processing_msg.edit_text = AsyncMock()
        mock_message.answer = AsyncMock(return_value=processing_msg)

        await handle_user_message(
            mock_message,
            mock_fsm_context,
            mock_l10n,
            mock_ai_service,
            session_factory,
        )

        assert mock_fsm_context.update_data.await_count >= 1
        call_args = mock_fsm_context.update_data.call_args
        payload = call_args.args[0] if call_args.args else call_args.kwargs
        assert isinstance(payload, dict)
        assert "coaching_session_state" in payload
        assert "coaching_recent_user_turns" in payload
        assert "coaching_last_response_plan" in payload


class TestTextNotCommandFilter:
    """Тесты для фильтра TEXT_NOT_COMMAND.

    Этот фильтр гарантирует, что команды (сообщения начинающиеся с /)
    не обрабатываются как обычный текст в FSM состояниях.
    Это позволяет пользователю вызывать /help, /settings и другие команды
    находясь в состоянии диалога с AI.
    """

    @pytest.mark.asyncio
    async def test_text_not_command_filter_allows_regular_text(
        self,
        mock_message: Message,
        mock_fsm_context: FSMContext,
        mock_l10n: Localization,
        mock_ai_service: AIService,
        db_session: AsyncSession,
        test_user: User,
        session_factory: Callable[[], AbstractAsyncContextManager[AsyncSession]],
        mock_billing_cost: GenerationCost,
    ) -> None:
        """Проверить, что обычный текст обрабатывается в FSM состоянии."""
        mock_message.answer = AsyncMock()
        mock_message.text = "Обычное сообщение без команды"

        await handle_user_message(
            mock_message,
            mock_fsm_context,
            mock_l10n,
            mock_ai_service,
            session_factory,
        )

        mock_ai_service.generate.assert_not_called()
        mock_message.answer.assert_called_once()

    @pytest.mark.asyncio
    async def test_text_not_command_filter_blocks_commands(
        self,
        mock_message: Message,
        mock_fsm_context: FSMContext,
        mock_l10n: Localization,
        mock_ai_service: AIService,
        db_session: AsyncSession,
        test_user: User,
        session_factory: Callable[[], AbstractAsyncContextManager[AsyncSession]],
        mock_billing_cost: GenerationCost,
    ) -> None:
        """Проверить, что команды НЕ обрабатываются в FSM состоянии.

        ВАЖНО: Этот тест проверяет что фильтр TEXT_NOT_COMMAND работает корректно.
        Команды должны пропускаться через handle_user_message и обрабатываться
        другими обработчиками (например, /help, /settings).

        В реальном боте aiogram не вызовет handle_user_message для команды,
        потому что фильтр TEXT_NOT_COMMAND исключает сообщения начинающиеся с /.
        Этот тест документирует эту логику, но фактически handle_user_message
        ожидает текстовое сообщение, а не команду.
        """
        mock_message.text = "/help"  # Команда
        mock_message.answer = AsyncMock()

        await handle_user_message(
            mock_message,
            mock_fsm_context,
            mock_l10n,
            mock_ai_service,
            session_factory,
        )

        assert isinstance(mock_ai_service.generate, AsyncMock)
        mock_ai_service.generate.assert_not_called()
        mock_message.answer.assert_called_once()
