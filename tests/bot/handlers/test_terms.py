"""Тесты для обработчика юридических документов.

Модуль тестирует:
- cmd_terms (команда /terms для просмотра документов)
- callback_accept_terms (обработка нажатия "Принимаю")
- show_terms_acceptance_request (вспомогательная функция для /start)

Тестируемая функциональность:
1. /terms показывает ссылки на документы
2. /terms показывает сообщение если документы отключены
3. /terms показывает сообщение если документы не настроены
4. callback "legal:accept" сохраняет согласие в БД
5. callback начисляет регистрационный бонус при первом согласии
6. callback обрабатывает повторное нажатие (already accepted)
7. callback обрабатывает отсутствие пользователя в БД
8. show_terms_acceptance_request показывает запрос на согласие
9. show_terms_acceptance_request возвращает False если документы не настроены
10. Локализация работает для обоих языков (ru, en)
"""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message, User

from src.bot.handlers.terms import (
    START_DIALOG_CALLBACK,
    callback_accept_terms,
    callback_start_dialog,
    cmd_terms,
    show_terms_acceptance_request,
)
from src.db.models.user import User as DbUser
from src.utils.i18n import Localization

# ==============================================================================
# ФИКСТУРЫ
# ==============================================================================


@pytest.fixture
def mock_message() -> MagicMock:
    """Мок Message с пользователем."""
    message = MagicMock(spec=Message)
    message.from_user = User(
        id=123456789,
        is_bot=False,
        first_name="Test User",
        username="testuser",
    )
    message.answer = AsyncMock()
    message.answer_photo = AsyncMock()
    return message


@pytest.fixture
def mock_callback() -> MagicMock:
    """Мок CallbackQuery с пользователем и сообщением."""
    callback = MagicMock(spec=CallbackQuery)
    callback.from_user = User(
        id=123456789,
        is_bot=False,
        first_name="Test User",
        username="testuser",
    )
    callback.message = MagicMock(spec=Message)
    callback.message.answer = AsyncMock()
    callback.message.answer_photo = AsyncMock()
    callback.message.edit_text = AsyncMock()
    callback.message.edit_reply_markup = AsyncMock()
    callback.answer = AsyncMock()
    callback.data = "legal:accept"
    return callback


@pytest.fixture
def mock_fsm_context() -> FSMContext:
    """Мок FSMContext для проверки установки onboarding-флага."""
    context = MagicMock(spec=FSMContext)
    context.update_data = AsyncMock()
    context.set_state = AsyncMock()
    return context


@pytest.fixture
def mock_l10n_ru() -> MagicMock:
    """Мок Localization для русского языка."""
    l10n = MagicMock(spec=Localization)
    l10n.language = "ru"

    def get_translation(key: str, **kwargs: Any) -> str:
        translations = {
            "legal_documents_message": (
                "📜 <b>Юридические документы</b>\n\nПожалуйста, ознакомьтесь:"
            ),
            "legal_disabled": ("ℹ️ Юридические документы в данный момент недоступны."),
            "legal_not_configured": ("⚠️ Юридические документы ещё не настроены."),
            "legal_acceptance_request": (
                "📜 <b>Добро пожаловать!</b>\n\n"
                "Прежде чем начать, ознакомьтесь с нашими документами:"
            ),
            "legal_accepted_notification": "✅ Спасибо! Условия приняты.",
            "legal_accepted_message": (
                "✅ <b>Условия приняты</b>\n\nТеперь вы можете использовать бота."
            ),
            "legal_already_accepted": "Вы уже приняли эти условия",
            "product_onboarding_message": (
                "🌿 <b>Добро пожаловать.</b>\n\n"
                "Этот бот помогает спокойно разобрать одну ситуацию за раз."
            ),
            "start_dialog_button": "Начать диалог",
            "billing_registration_bonus": "🎁 Вам начислено {amount} токенов!",
            "error_callback_data": "❌ Ошибка обработки запроса",
            "error_user_not_found": "❌ Пользователь не найден",
            "chatgpt_choose_model": "Выберите модель для диалога",
            "no_models_available": "Нет доступных моделей",
        }
        text = translations.get(key, key)
        if kwargs:
            return text.format(**kwargs)
        return text

    l10n.get.side_effect = get_translation
    return l10n


@pytest.fixture
def mock_l10n_en() -> MagicMock:
    """Мок Localization для английского языка."""
    l10n = MagicMock(spec=Localization)
    l10n.language = "en"

    def get_translation(key: str, **kwargs: Any) -> str:
        translations = {
            "legal_documents_message": ("📜 <b>Legal Documents</b>\n\nPlease review:"),
            "legal_disabled": "ℹ️ Legal documents are currently unavailable.",
            "legal_not_configured": ("⚠️ Legal documents are not yet configured."),
            "legal_acceptance_request": (
                "📜 <b>Welcome!</b>\n\nBefore you start, please review our documents:"
            ),
            "legal_accepted_notification": "✅ Thank you! Terms accepted.",
            "legal_accepted_message": (
                "✅ <b>Terms Accepted</b>\n\nYou can now use the bot."
            ),
            "legal_already_accepted": "You have already accepted these terms",
            "product_onboarding_message": (
                "🌿 <b>Welcome.</b>\n\n"
                "This bot helps you calmly work through one situation at a time."
            ),
            "start_dialog_button": "Start conversation",
            "billing_registration_bonus": ("🎁 You've been credited {amount} tokens!"),
            "error_callback_data": "❌ Error processing request",
            "error_user_not_found": "❌ User not found",
            "chatgpt_choose_model": "Choose a model for chat",
            "no_models_available": "No models available",
        }
        text = translations.get(key, key)
        if kwargs:
            return text.format(**kwargs)
        return text

    l10n.get.side_effect = get_translation
    return l10n


# ==============================================================================
# ТЕСТЫ cmd_terms
# ==============================================================================


@pytest.mark.asyncio
async def test_cmd_terms_shows_documents_when_enabled(
    mock_message: MagicMock,
    mock_l10n_ru: MagicMock,
) -> None:
    """Тест: /terms показывает ссылки на документы если они включены и настроены."""
    with (
        patch("src.bot.handlers.terms.yaml_config") as mock_config,
        patch(
            "src.bot.handlers.terms.create_legal_documents_keyboard"
        ) as mock_keyboard,
    ):
        # Настраиваем мок конфигурации
        mock_legal = MagicMock()
        mock_legal.enabled = True
        mock_legal.has_documents.return_value = True
        mock_legal.privacy_policy_url = "https://example.com/privacy"
        mock_legal.terms_of_service_url = "https://example.com/terms"
        mock_config.legal = mock_legal

        # Мок клавиатуры
        mock_keyboard.return_value = MagicMock()

        await cmd_terms(mock_message, mock_l10n_ru)

    # Проверяем что сообщение отправлено
    mock_message.answer.assert_called_once()
    call_args = mock_message.answer.call_args[0]
    assert "Юридические документы" in call_args[0]

    # Проверяем что клавиатура создана с правильными параметрами
    mock_keyboard.assert_called_once_with(
        l10n=mock_l10n_ru,
        privacy_policy_url="https://example.com/privacy",
        terms_of_service_url="https://example.com/terms",
    )


@pytest.mark.asyncio
async def test_cmd_terms_shows_disabled_message_when_disabled(
    mock_message: MagicMock,
    mock_l10n_ru: MagicMock,
) -> None:
    """Тест: /terms показывает сообщение если документы отключены."""
    with patch("src.bot.handlers.terms.yaml_config") as mock_config:
        mock_legal = MagicMock()
        mock_legal.enabled = False
        mock_config.legal = mock_legal

        await cmd_terms(mock_message, mock_l10n_ru)

    # Проверяем что показано сообщение об отключении
    mock_message.answer.assert_called_once()
    call_args = mock_message.answer.call_args[0]
    assert "недоступны" in call_args[0]


@pytest.mark.asyncio
async def test_cmd_terms_shows_not_configured_message(
    mock_message: MagicMock,
    mock_l10n_ru: MagicMock,
) -> None:
    """Тест: /terms показывает сообщение если документы не настроены."""
    with patch("src.bot.handlers.terms.yaml_config") as mock_config:
        mock_legal = MagicMock()
        mock_legal.enabled = True
        mock_legal.has_documents.return_value = False
        mock_config.legal = mock_legal

        await cmd_terms(mock_message, mock_l10n_ru)

    # Проверяем что показано сообщение о ненастроенных документах
    mock_message.answer.assert_called_once()
    call_args = mock_message.answer.call_args[0]
    assert "не настроены" in call_args[0]


@pytest.mark.asyncio
async def test_cmd_terms_localization_en(
    mock_message: MagicMock,
    mock_l10n_en: MagicMock,
) -> None:
    """Тест: /terms использует английскую локализацию."""
    with (
        patch("src.bot.handlers.terms.yaml_config") as mock_config,
        patch(
            "src.bot.handlers.terms.create_legal_documents_keyboard"
        ) as mock_keyboard,
    ):
        mock_legal = MagicMock()
        mock_legal.enabled = True
        mock_legal.has_documents.return_value = True
        mock_legal.privacy_policy_url = "https://example.com/privacy"
        mock_legal.terms_of_service_url = "https://example.com/terms"
        mock_config.legal = mock_legal

        mock_keyboard.return_value = MagicMock()

        await cmd_terms(mock_message, mock_l10n_en)

    # Проверяем что использована английская локализация
    mock_message.answer.assert_called_once()
    call_args = mock_message.answer.call_args[0]
    assert "Legal Documents" in call_args[0]


# ==============================================================================
# ТЕСТЫ callback_accept_terms
# ==============================================================================


@pytest.mark.asyncio
async def test_callback_accept_terms_saves_acceptance(
    mock_callback: MagicMock,
    mock_l10n_ru: MagicMock,
) -> None:
    """Тест: callback сохраняет согласие в БД."""
    with (
        patch("src.bot.handlers.terms.yaml_config") as mock_config,
        patch("src.bot.handlers.terms.DatabaseSession") as mock_session_cls,
        patch("src.bot.handlers.terms.UserRepository") as mock_repo_cls,
        patch("src.bot.handlers.terms.create_billing_service") as mock_billing_cls,
    ):
        # Настраиваем конфигурацию
        mock_legal = MagicMock()
        mock_legal.version = "1.0"
        mock_config.legal = mock_legal

        # Настраиваем DatabaseSession
        mock_session = AsyncMock()
        mock_session_cls.return_value.__aenter__.return_value = mock_session

        # Настраиваем UserRepository
        mock_user = MagicMock(spec=DbUser)
        mock_user.balance = 0
        mock_repo = MagicMock()
        mock_repo.get_by_telegram_id = AsyncMock(return_value=mock_user)
        mock_repo.needs_terms_acceptance.return_value = True
        mock_repo.accept_terms = AsyncMock()
        mock_repo_cls.return_value = mock_repo

        # Мокируем биллинг
        mock_billing = MagicMock()
        mock_billing.grant_registration_bonus = AsyncMock(return_value=0)
        mock_billing_cls.return_value = mock_billing

        await callback_accept_terms(mock_callback, mock_l10n_ru)

    # Проверяем что accept_terms был вызван
    mock_repo.accept_terms.assert_called_once_with(mock_user, "1.0")

    # Проверяем что отправлено уведомление
    mock_callback.answer.assert_called_once()

    # Проверяем что сообщение обновлено
    mock_callback.message.edit_text.assert_called_once()

    requested_keys = [call.args[0] for call in mock_l10n_ru.get.call_args_list]
    assert "product_onboarding_message" in requested_keys
    assert "start_dialog_button" in requested_keys
    assert "start_message" not in requested_keys


@pytest.mark.asyncio
async def test_callback_accept_terms_attaches_start_dialog_button(
    mock_callback: MagicMock,
    mock_l10n_ru: MagicMock,
) -> None:
    """Тест: post-legal onboarding содержит кнопку перехода в диалог."""
    with (
        patch("src.bot.handlers.terms.yaml_config") as mock_config,
        patch("src.bot.handlers.terms.DatabaseSession") as mock_session_cls,
        patch("src.bot.handlers.terms.UserRepository") as mock_repo_cls,
        patch("src.bot.handlers.terms.create_billing_service") as mock_billing_cls,
        patch(
            "src.bot.handlers.terms.POST_LEGAL_ONBOARDING_IMAGE"
        ) as mock_onboarding_image,
    ):
        mock_legal = MagicMock()
        mock_legal.version = "1.0"
        mock_config.legal = mock_legal

        mock_session = AsyncMock()
        mock_session_cls.return_value.__aenter__.return_value = mock_session

        mock_user = MagicMock(spec=DbUser)
        mock_user.registration_bonus_granted = True
        mock_repo = MagicMock()
        mock_repo.get_by_telegram_id = AsyncMock(return_value=mock_user)
        mock_repo.needs_terms_acceptance.return_value = True
        mock_repo.accept_terms = AsyncMock()
        mock_repo_cls.return_value = mock_repo

        mock_billing = MagicMock()
        mock_billing.grant_registration_bonus = AsyncMock(return_value=0)
        mock_billing_cls.return_value = mock_billing

        # Проверяем text-only ветку: проще извлечь reply_markup из answer().
        mock_onboarding_image.exists.return_value = False

        await callback_accept_terms(mock_callback, mock_l10n_ru)

    mock_callback.message.answer.assert_called_once()
    reply_markup = mock_callback.message.answer.call_args.kwargs["reply_markup"]
    assert len(reply_markup.inline_keyboard) == 1
    assert len(reply_markup.inline_keyboard[0]) == 1
    button = reply_markup.inline_keyboard[0][0]
    assert button.callback_data == START_DIALOG_CALLBACK
    assert button.text == "Начать диалог"


@pytest.mark.asyncio
async def test_callback_start_dialog_moves_to_chat_flow(
    mock_callback: MagicMock,
    mock_l10n_ru: MagicMock,
    mock_fsm_context: FSMContext,
) -> None:
    """Тест: кнопка onboarding переводит пользователя в выбор модели chat."""
    ai_service = MagicMock()
    ai_service.get_available_models.return_value = {
        "gpt-4o-mini": SimpleNamespace(
            generation_type="chat",
            display_name="GPT-4o-mini",
            price_tokens=1,
        )
    }

    await callback_start_dialog(
        callback=mock_callback,
        state=mock_fsm_context,
        l10n=mock_l10n_ru,
        ai_service=ai_service,
    )

    mock_fsm_context.update_data.assert_called_once_with({"onboarding_completed": True})
    mock_fsm_context.set_state.assert_called_once()
    mock_callback.answer.assert_called_once()
    mock_callback.message.answer.assert_called_once()
    call_args = mock_callback.message.answer.call_args
    assert "Выберите модель" in call_args.args[0]
    assert call_args.kwargs["reply_markup"] is not None


@pytest.mark.asyncio
async def test_callback_accept_terms_sets_onboarding_completed_flag(
    mock_callback: MagicMock,
    mock_l10n_ru: MagicMock,
    mock_fsm_context: FSMContext,
) -> None:
    """Тест: после принятия условий ставится onboarding_completed=True."""
    with (
        patch("src.bot.handlers.terms.yaml_config") as mock_config,
        patch("src.bot.handlers.terms.DatabaseSession") as mock_session_cls,
        patch("src.bot.handlers.terms.UserRepository") as mock_repo_cls,
        patch("src.bot.handlers.terms.create_billing_service") as mock_billing_cls,
        patch(
            "src.bot.handlers.terms.POST_LEGAL_ONBOARDING_IMAGE"
        ) as mock_onboarding_image,
    ):
        mock_legal = MagicMock()
        mock_legal.version = "1.0"
        mock_config.legal = mock_legal

        mock_session = AsyncMock()
        mock_session_cls.return_value.__aenter__.return_value = mock_session

        mock_user = MagicMock(spec=DbUser)
        mock_user.balance = 0
        mock_user.registration_bonus_granted = True
        mock_repo = MagicMock()
        mock_repo.get_by_telegram_id = AsyncMock(return_value=mock_user)
        mock_repo.needs_terms_acceptance.return_value = True
        mock_repo.accept_terms = AsyncMock()
        mock_repo_cls.return_value = mock_repo

        mock_billing = MagicMock()
        mock_billing.grant_registration_bonus = AsyncMock(return_value=0)
        mock_billing_cls.return_value = mock_billing

        mock_onboarding_image.exists.return_value = False

        await callback_accept_terms(
            mock_callback,
            mock_l10n_ru,
            mock_fsm_context,
        )

    mock_fsm_context.update_data.assert_called_once_with(
        {"onboarding_completed": True}
    )


@pytest.mark.asyncio
async def test_callback_accept_terms_grants_registration_bonus(
    mock_callback: MagicMock,
    mock_l10n_ru: MagicMock,
) -> None:
    """Тест: callback начисляет регистрационный бонус при первом согласии."""
    with (
        patch("src.bot.handlers.terms.yaml_config") as mock_config,
        patch("src.bot.handlers.terms.DatabaseSession") as mock_session_cls,
        patch("src.bot.handlers.terms.UserRepository") as mock_repo_cls,
        patch("src.bot.handlers.terms.create_billing_service") as mock_billing_cls,
        patch(
            "src.bot.handlers.terms.POST_LEGAL_ONBOARDING_IMAGE"
        ) as mock_onboarding_image,
    ):
        mock_legal = MagicMock()
        mock_legal.version = "1.0"
        mock_config.legal = mock_legal
        # Мокаем биллинг конфиг чтобы grant_registration_bonus вызывался
        mock_config.billing = MagicMock()
        mock_config.billing.registration_bonus = 100

        mock_session = AsyncMock()
        mock_session_cls.return_value.__aenter__.return_value = mock_session

        # Пользователь с балансом 0 (бонус ещё не начислялся)
        mock_user = MagicMock(spec=DbUser)
        mock_user.balance = 0
        mock_user.registration_bonus_granted = False  # Бонус ещё не начислен
        mock_repo = MagicMock()
        mock_repo.get_by_telegram_id = AsyncMock(return_value=mock_user)
        mock_repo.needs_terms_acceptance.return_value = True
        mock_repo.accept_terms = AsyncMock()
        mock_repo_cls.return_value = mock_repo

        # Биллинг начисляет бонус 100
        mock_billing = MagicMock()
        mock_billing.grant_registration_bonus = AsyncMock(return_value=100)
        mock_billing_cls.return_value = mock_billing

        # Мокируем welcome image — картинка существует
        mock_onboarding_image.exists.return_value = True

        await callback_accept_terms(mock_callback, mock_l10n_ru)

    # Проверяем что бонус был начислен
    mock_billing.grant_registration_bonus.assert_called_once_with(mock_user)

    # Проверяем что отправлено приветствие с картинкой
    mock_callback.message.answer_photo.assert_called_once()

    # Проверяем что отправлено уведомление о бонусе
    mock_callback.message.answer.assert_called_once()
    bonus_call = mock_callback.message.answer.call_args[0]
    assert "100 токенов" in bonus_call[0]


@pytest.mark.asyncio
async def test_callback_accept_terms_handles_already_accepted(
    mock_callback: MagicMock,
    mock_l10n_ru: MagicMock,
) -> None:
    """Тест: callback обрабатывает повторное нажатие (already accepted)."""
    with (
        patch("src.bot.handlers.terms.yaml_config") as mock_config,
        patch("src.bot.handlers.terms.DatabaseSession") as mock_session_cls,
        patch("src.bot.handlers.terms.UserRepository") as mock_repo_cls,
    ):
        mock_legal = MagicMock()
        mock_legal.version = "1.0"
        mock_config.legal = mock_legal

        mock_session = AsyncMock()
        mock_session_cls.return_value.__aenter__.return_value = mock_session

        mock_user = MagicMock(spec=DbUser)
        mock_repo = MagicMock()
        mock_repo.get_by_telegram_id = AsyncMock(return_value=mock_user)
        # Пользователь уже принял условия
        mock_repo.needs_terms_acceptance.return_value = False
        mock_repo_cls.return_value = mock_repo

        await callback_accept_terms(mock_callback, mock_l10n_ru)

    # Проверяем что показано уведомление
    mock_callback.answer.assert_called_once()
    call_args = mock_callback.answer.call_args[0]
    assert "уже приняли" in call_args[0]

    # Проверяем что клавиатура удалена
    mock_callback.message.edit_reply_markup.assert_called_once_with(reply_markup=None)


@pytest.mark.asyncio
async def test_callback_accept_terms_handles_user_not_found(
    mock_callback: MagicMock,
    mock_l10n_ru: MagicMock,
) -> None:
    """Тест: callback обрабатывает отсутствие пользователя в БД."""
    with (
        patch("src.bot.handlers.terms.yaml_config") as mock_config,
        patch("src.bot.handlers.terms.DatabaseSession") as mock_session_cls,
        patch("src.bot.handlers.terms.UserRepository") as mock_repo_cls,
    ):
        mock_legal = MagicMock()
        mock_legal.version = "1.0"
        mock_config.legal = mock_legal

        mock_session = AsyncMock()
        mock_session_cls.return_value.__aenter__.return_value = mock_session

        # Пользователь не найден
        mock_repo = MagicMock()
        mock_repo.get_by_telegram_id = AsyncMock(return_value=None)
        mock_repo_cls.return_value = mock_repo

        await callback_accept_terms(mock_callback, mock_l10n_ru)

    # Проверяем что показано сообщение об ошибке
    mock_callback.answer.assert_called_once()
    call_args = mock_callback.answer.call_args
    assert "не найден" in call_args[0][0]
    assert call_args[1]["show_alert"] is True


@pytest.mark.asyncio
async def test_callback_accept_terms_handles_missing_message(
    mock_l10n_ru: MagicMock,
) -> None:
    """Тест: callback обрабатывает отсутствие message."""
    callback = MagicMock(spec=CallbackQuery)
    callback.message = None
    callback.from_user = User(id=123456789, is_bot=False, first_name="Test")
    callback.answer = AsyncMock()

    await callback_accept_terms(callback, mock_l10n_ru)

    # Проверяем что показано сообщение об ошибке
    callback.answer.assert_called_once()


@pytest.mark.asyncio
async def test_callback_accept_terms_handles_missing_user(
    mock_l10n_ru: MagicMock,
) -> None:
    """Тест: callback обрабатывает отсутствие from_user."""
    callback = MagicMock(spec=CallbackQuery)
    callback.message = MagicMock(spec=Message)
    callback.from_user = None
    callback.answer = AsyncMock()

    await callback_accept_terms(callback, mock_l10n_ru)

    # Проверяем что показано сообщение об ошибке
    callback.answer.assert_called_once()


@pytest.mark.asyncio
async def test_callback_accept_terms_localization_en(
    mock_callback: MagicMock,
    mock_l10n_en: MagicMock,
) -> None:
    """Тест: callback использует английскую локализацию."""
    with (
        patch("src.bot.handlers.terms.yaml_config") as mock_config,
        patch("src.bot.handlers.terms.DatabaseSession") as mock_session_cls,
        patch("src.bot.handlers.terms.UserRepository") as mock_repo_cls,
        patch("src.bot.handlers.terms.create_billing_service") as mock_billing_cls,
    ):
        mock_legal = MagicMock()
        mock_legal.version = "1.0"
        mock_config.legal = mock_legal

        mock_session = AsyncMock()
        mock_session_cls.return_value.__aenter__.return_value = mock_session

        mock_user = MagicMock(spec=DbUser)
        mock_user.balance = 0
        mock_repo = MagicMock()
        mock_repo.get_by_telegram_id = AsyncMock(return_value=mock_user)
        mock_repo.needs_terms_acceptance.return_value = True
        mock_repo.accept_terms = AsyncMock()
        mock_repo_cls.return_value = mock_repo

        mock_billing = MagicMock()
        mock_billing.grant_registration_bonus = AsyncMock(return_value=0)
        mock_billing_cls.return_value = mock_billing

        await callback_accept_terms(mock_callback, mock_l10n_en)

    # Проверяем что использована английская локализация
    call_args = mock_callback.message.edit_text.call_args[0]
    assert "Terms Accepted" in call_args[0]


@pytest.mark.asyncio
async def test_callback_accept_terms_no_bonus_if_balance_not_zero(
    mock_callback: MagicMock,
    mock_l10n_ru: MagicMock,
) -> None:
    """Тест: callback не начисляет бонус если баланс не равен 0."""
    with (
        patch("src.bot.handlers.terms.yaml_config") as mock_config,
        patch("src.bot.handlers.terms.DatabaseSession") as mock_session_cls,
        patch("src.bot.handlers.terms.UserRepository") as mock_repo_cls,
        patch("src.bot.handlers.terms.create_billing_service") as mock_billing_cls,
        patch(
            "src.bot.handlers.terms.POST_LEGAL_ONBOARDING_IMAGE"
        ) as mock_onboarding_image,
    ):
        mock_legal = MagicMock()
        mock_legal.version = "1.0"
        mock_config.legal = mock_legal

        mock_session = AsyncMock()
        mock_session_cls.return_value.__aenter__.return_value = mock_session

        # Пользователь с балансом > 0 (бонус уже начислялся)
        mock_user = MagicMock(spec=DbUser)
        mock_user.balance = 100
        mock_repo = MagicMock()
        mock_repo.get_by_telegram_id = AsyncMock(return_value=mock_user)
        mock_repo.needs_terms_acceptance.return_value = True
        mock_repo.accept_terms = AsyncMock()
        mock_repo_cls.return_value = mock_repo

        mock_billing = MagicMock()
        mock_billing.grant_registration_bonus = AsyncMock(return_value=0)
        mock_billing_cls.return_value = mock_billing

        # Мокируем welcome image — картинка существует
        mock_onboarding_image.exists.return_value = True

        await callback_accept_terms(mock_callback, mock_l10n_ru)

    # Проверяем что биллинг НЕ был вызван
    mock_billing_cls.assert_not_called()

    # Проверяем что приветствие с картинкой было отправлено
    mock_callback.message.answer_photo.assert_called_once()

    # Проверяем что answer не вызывался (бонус не начислен)
    mock_callback.message.answer.assert_not_called()


# ==============================================================================
# ТЕСТЫ show_terms_acceptance_request
# ==============================================================================


@pytest.mark.asyncio
async def test_show_terms_acceptance_request_shows_request(
    mock_message: MagicMock,
    mock_l10n_ru: MagicMock,
) -> None:
    """Тест: show_terms_acceptance_request показывает запрос на согласие."""
    with (
        patch("src.bot.handlers.terms.yaml_config") as mock_config,
        patch(
            "src.bot.handlers.terms.create_terms_acceptance_keyboard"
        ) as mock_keyboard,
    ):
        mock_legal = MagicMock()
        mock_legal.has_documents.return_value = True
        mock_legal.privacy_policy_url = "https://example.com/privacy"
        mock_legal.terms_of_service_url = "https://example.com/terms"
        mock_config.legal = mock_legal

        mock_keyboard.return_value = MagicMock()

        result = await show_terms_acceptance_request(mock_message, mock_l10n_ru)

    # Проверяем что вернулось True
    assert result is True

    # Проверяем что сообщение отправлено
    mock_message.answer.assert_called_once()
    call_args = mock_message.answer.call_args[0]
    assert "Добро пожаловать" in call_args[0]

    # Проверяем что клавиатура создана с правильными параметрами
    mock_keyboard.assert_called_once_with(
        l10n=mock_l10n_ru,
        privacy_policy_url="https://example.com/privacy",
        terms_of_service_url="https://example.com/terms",
    )


@pytest.mark.asyncio
async def test_show_terms_acceptance_request_returns_false_if_not_configured(
    mock_message: MagicMock,
    mock_l10n_ru: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Тест: show_terms_acceptance_request возвращает False.

    Когда документы не настроены.
    """
    import logging

    caplog.set_level(logging.WARNING)

    with patch("src.bot.handlers.terms.yaml_config") as mock_config:
        mock_legal = MagicMock()
        mock_legal.has_documents.return_value = False
        mock_config.legal = mock_legal

        result = await show_terms_acceptance_request(mock_message, mock_l10n_ru)

    # Проверяем что вернулось False
    assert result is False

    # Проверяем что сообщение НЕ отправлено
    mock_message.answer.assert_not_called()

    # Проверяем что залогировано предупреждение
    assert any("не настроены" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_show_terms_acceptance_request_localization_en(
    mock_message: MagicMock,
    mock_l10n_en: MagicMock,
) -> None:
    """Тест: show_terms_acceptance_request использует английскую локализацию."""
    with (
        patch("src.bot.handlers.terms.yaml_config") as mock_config,
        patch(
            "src.bot.handlers.terms.create_terms_acceptance_keyboard"
        ) as mock_keyboard,
    ):
        mock_legal = MagicMock()
        mock_legal.has_documents.return_value = True
        mock_legal.privacy_policy_url = "https://example.com/privacy"
        mock_legal.terms_of_service_url = "https://example.com/terms"
        mock_config.legal = mock_legal

        mock_keyboard.return_value = MagicMock()

        result = await show_terms_acceptance_request(mock_message, mock_l10n_en)

    # Проверяем что использована английская локализация
    assert result is True
    call_args = mock_message.answer.call_args[0]
    assert "Welcome" in call_args[0]
