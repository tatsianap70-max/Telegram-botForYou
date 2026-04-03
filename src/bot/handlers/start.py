"""Обработчик команды /start.

При первом /start создаёт пользователя в базе данных.
При повторном — обновляет данные профиля (username, имя).

Автоопределение языка:
- При первом /start определяет язык из Telegram (User.language_code)
- Если язык доступен в available_languages — использует его
- Если нет — использует default_language из конфига

Согласие с документами:
- При первом /start показывает запрос на согласие с юр. документами
- После согласия показывает приветствие и начисляет бонус
- При обновлении версии документов запрашивает повторное согласие
"""

from aiogram import F, Router
from aiogram.filters import CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.types import (
    CallbackQuery,
    FSInputFile,
    InaccessibleMessage,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from src.bot.handlers.terms import (
    POST_LEGAL_ONBOARDING_IMAGE,
    POST_LEGAL_ONBOARDING_TEXT_KEY,
    START_DIALOG_CALLBACK,
    show_terms_acceptance_request,
)
from src.config.yaml_config import yaml_config
from src.db.base import DatabaseSession
from src.db.repositories.user_repo import UserRepository
from src.services.billing_service import create_billing_service
from src.services.referral_service import create_referral_service
from src.utils.i18n import Localization, create_localization
from src.utils.logging import get_logger

router = Router(name="start")
logger = get_logger(__name__)

# =============================================================================
# КОНСТАНТЫ
# =============================================================================

# Длина префикса команды "/start " для извлечения параметра
# Telegram deep links: t.me/bot?start=promo → /start promo
COMMAND_START_PREFIX_LENGTH = 7  # len("/start ") = 7
ONBOARDING_COMPLETED_KEY = "onboarding_completed"
START_LANGUAGE_CALLBACK_PREFIX = "start_lang:"
ENTRY_LANGUAGES: tuple[str, str] = ("ru", "en")
START_LANGUAGE_CHOICE_TEXT_KEY = "language_command"
# Оставляем имя WELCOME_IMAGE для обратной совместимости тестов / патчей.
WELCOME_IMAGE = POST_LEGAL_ONBOARDING_IMAGE
PRODUCT_ONBOARDING_TEXT_KEY = POST_LEGAL_ONBOARDING_TEXT_KEY


def _create_start_dialog_keyboard(l10n: Localization) -> InlineKeyboardMarkup:
    """Создать inline-кнопку перехода к основному диалогу."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=l10n.get("start_dialog_button"),
                    callback_data=START_DIALOG_CALLBACK,
                )
            ]
        ]
    )


async def _mark_onboarding_completed(state: FSMContext | None) -> None:
    """Отметить, что пользователь прошёл входной слой."""
    if state is None:
        return
    await state.update_data({ONBOARDING_COMPLETED_KEY: True})


def _create_start_language_keyboard(l10n: Localization) -> InlineKeyboardMarkup:
    """Создать клавиатуру выбора языка для entry-path."""
    buttons: list[list[InlineKeyboardButton]] = [
        [
            InlineKeyboardButton(
                text=l10n.get(f"language_name_{language_code}"),
                callback_data=(
                    f"{START_LANGUAGE_CALLBACK_PREFIX}{language_code}"
                ),
            )
        ]
        for language_code in ENTRY_LANGUAGES
    ]
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def _should_show_start_language_gate(has_saved_language: bool) -> bool:
    """Проверить, нужен ли явный language gate в /start."""
    if has_saved_language or not Localization.is_enabled():
        return False
    available_languages = set(Localization.get_available_languages())
    return set(ENTRY_LANGUAGES).issubset(available_languages)


def _extract_start_language(data: str | None) -> str | None:
    """Извлечь язык из callback_data стартового language gate."""
    if not isinstance(data, str) or not data.startswith(
        START_LANGUAGE_CALLBACK_PREFIX
    ):
        return None
    language_code = data[len(START_LANGUAGE_CALLBACK_PREFIX) :]
    if language_code in ENTRY_LANGUAGES:
        return language_code
    return None


def _extract_start_param(message: Message) -> str | None:
    """Извлечь start-параметр из команды /start.

    Telegram позволяет передавать параметры через deep link:
    t.me/bot?start=promo_winter → /start promo_winter

    Args:
        message: Сообщение с командой /start.

    Returns:
        Параметр после /start или None если его нет.
    """
    if message.text and len(message.text) > COMMAND_START_PREFIX_LENGTH:
        return message.text[COMMAND_START_PREFIX_LENGTH:].strip() or None
    return None


def _detect_user_language(telegram_language_code: str | None) -> str:
    """Определить язык пользователя на основе language_code из Telegram.

    Логика:
    1. Если language_code пустой или None → язык по умолчанию
    2. Если язык есть в available_languages → используем его
    3. Иначе → язык по умолчанию

    Args:
        telegram_language_code: Код языка из Telegram (ru, en, zh и т.д.).
            Может быть полным locale: "ru-RU", "en-US".

    Returns:
        Код языка для использования в боте (ISO 639-1: ru, en).

    Example:
        >>> _detect_user_language("ru")  # ru в available_languages
        "ru"
        >>> _detect_user_language("ru-RU")  # Полный locale → извлекаем "ru"
        "ru"
        >>> _detect_user_language("fr")  # fr НЕ в available_languages
        "ru"  # default_language
        >>> _detect_user_language(None)
        "ru"  # default_language
    """
    if not telegram_language_code:
        return Localization.get_default_language()

    # Telegram может передать полный locale вида "ru-RU" или "en-US"
    # Берём только код языка до дефиса (ISO 639-1)
    language_code = telegram_language_code.lower().split("-")[0]

    # Проверяем, доступен ли язык
    available_languages = Localization.get_available_languages()
    if language_code in available_languages:
        return language_code

    # Язык недоступен — используем язык по умолчанию
    return Localization.get_default_language()


@router.message(CommandStart())
async def cmd_start(
    message: Message,
    l10n: Localization,
    state: FSMContext | None = None,
) -> None:
    """Обработать команду /start.

    При первом запуске:
    - Создаёт пользователя в БД
    - Определяет язык на основе Telegram language_code
    - Сохраняет start-параметр (откуда пришёл)
    - Показывает запрос на согласие с юр. документами (если настроено)

    При повторном запуске:
    - Обновляет данные профиля (username мог измениться)
    - Проверяет, нужно ли повторное согласие (если версия изменилась)

    Args:
        message: Входящее сообщение с командой /start.
        l10n: Объект локализации (внедряется через LanguageMiddleware).
    """
    tg_user = message.from_user
    if not tg_user:
        # Теоретически невозможно для личных сообщений
        await message.answer(l10n.get(PRODUCT_ONBOARDING_TEXT_KEY))
        return

    # Извлекаем start-параметр (для аналитики и рефералов)
    source = _extract_start_param(message)

    # Определяем язык на основе Telegram language_code
    # Если язык доступен — используем его, иначе — язык по умолчанию
    detected_language = _detect_user_language(tg_user.language_code)

    # Получаем конфигурацию юридических документов
    legal_config = yaml_config.legal

    # Флаг: нужно ли показать запрос на согласие
    needs_terms_acceptance = False
    # Флаг: есть ли у пользователя уже сохранённый язык до текущего /start.
    has_saved_language = False

    # Сохраняем/обновляем пользователя в БД
    # Переменная для хранения бонуса (начисляется только после согласия)
    registration_bonus = 0
    # Бонус за переход по реферальной ссылке
    referral_bonus = 0

    async with DatabaseSession() as session:
        repo = UserRepository(session)
        user, created = await repo.get_or_create(
            telegram_id=tg_user.id,
            username=tg_user.username,
            first_name=tg_user.first_name,
            last_name=tg_user.last_name,
            language=detected_language,
            source=source,
        )
        user_language = getattr(user, "language", None)
        has_saved_language = (
            not created
            and isinstance(user_language, str)
            and bool(user_language.strip())
        )

        if created:
            logger.info(
                "Новый пользователь: %s (id=%d), language=%s, source=%s",
                tg_user.full_name,
                tg_user.id,
                detected_language,
                source,
            )

            # Обрабатываем реферальную ссылку (если есть)
            # Формат: ref_TELEGRAM_ID → начисление бонусов обоим
            referral_service = create_referral_service(session)
            referral_result = await referral_service.process_referral(
                invitee=user,
                start_param=source,
            )
            if referral_result.success:
                referral_bonus = referral_result.invitee_bonus
                logger.info(
                    "Реферал обработан для user_id=%d: invitee_bonus=%d",
                    user.id,
                    referral_bonus,
                )

            # Проверяем, нужно ли запросить согласие с документами
            if legal_config.enabled and legal_config.has_documents():
                needs_terms_acceptance = True
            else:
                # Если документы не настроены — начисляем бонус сразу
                billing = create_billing_service(session)
                registration_bonus = await billing.grant_registration_bonus(user)
        else:
            # Обновляем данные профиля (могли измениться)
            await repo.update_profile(
                user=user,
                username=tg_user.username,
                first_name=tg_user.first_name,
                last_name=tg_user.last_name,
            )
            logger.info(
                "Пользователь вернулся: %s (id=%d)",
                tg_user.full_name,
                tg_user.id,
            )

            # Проверяем, нужно ли повторное согласие (версия изменилась)
            if (
                legal_config.enabled
                and legal_config.has_documents()
                and repo.needs_terms_acceptance(user, legal_config.version)
            ):
                needs_terms_acceptance = True
                logger.info(
                    "Требуется повторное согласие: id=%d, "
                    "accepted_version=%s, current_version=%s",
                    tg_user.id,
                    user.accepted_legal_version,
                    legal_config.version,
                )

    if _should_show_start_language_gate(has_saved_language):
        await message.answer(
            l10n.get(START_LANGUAGE_CHOICE_TEXT_KEY),
            reply_markup=_create_start_language_keyboard(l10n),
        )
        return

    # Если нужно согласие — показываем запрос
    if needs_terms_acceptance:
        shown = await show_terms_acceptance_request(message, l10n)
        if shown:
            # Запрос показан — не показываем приветствие,
            # оно будет показано после согласия
            return

    # Отправляем приветственное сообщение с картинкой на языке пользователя
    # Если файл welcome.jpg существует — отправляем фото с подписью,
    # иначе — только текстовое сообщение.
    #
    onboarding_keyboard = _create_start_dialog_keyboard(l10n)
    if WELCOME_IMAGE.exists():
        await message.answer_photo(
            photo=FSInputFile(WELCOME_IMAGE),
            caption=l10n.get(PRODUCT_ONBOARDING_TEXT_KEY),
            reply_markup=onboarding_keyboard,
        )
    else:
        await message.answer(
            l10n.get(PRODUCT_ONBOARDING_TEXT_KEY),
            reply_markup=onboarding_keyboard,
        )

    # Если начислен бонус — уведомляем пользователя
    if registration_bonus > 0:
        await message.answer(
            l10n.get("billing_registration_bonus", amount=registration_bonus)
        )

    # Если начислен реферальный бонус — уведомляем пользователя
    if referral_bonus > 0:
        await message.answer(l10n.get("referral_invitee_bonus", amount=referral_bonus))

    await _mark_onboarding_completed(state)


@router.callback_query(F.data.startswith(START_LANGUAGE_CALLBACK_PREFIX))
async def callback_start_language_selection(
    callback: CallbackQuery,
    state: FSMContext | None = None,
) -> None:
    """Обработать выбор языка в стартовом entry-path."""
    if (
        callback.message is None
        or isinstance(callback.message, InaccessibleMessage)
        or callback.from_user is None
    ):
        return

    selected_language = _extract_start_language(callback.data)
    if selected_language is None:
        fallback_l10n = create_localization(Localization.get_default_language())
        await callback.answer(
            fallback_l10n.get("error_language_not_supported"),
            show_alert=True,
        )
        return

    selected_l10n = create_localization(selected_language)
    legal_config = yaml_config.legal
    needs_terms_acceptance = False

    async with DatabaseSession() as session:
        repo = UserRepository(session)
        user = await repo.get_by_telegram_id(callback.from_user.id)
        if user is None:
            user, _ = await repo.get_or_create(
                telegram_id=callback.from_user.id,
                username=callback.from_user.username,
                first_name=callback.from_user.first_name,
                last_name=callback.from_user.last_name,
                language=selected_language,
                source=None,
            )
        else:
            await repo.update_language(user, selected_language)

        if (
            legal_config.enabled
            and legal_config.has_documents()
            and repo.needs_terms_acceptance(user, legal_config.version)
        ):
            needs_terms_acceptance = True

    await callback.message.edit_reply_markup(reply_markup=None)
    await callback.answer()

    if needs_terms_acceptance:
        shown = await show_terms_acceptance_request(
            callback.message,
            selected_l10n,
        )
        if shown:
            return

    onboarding_keyboard = _create_start_dialog_keyboard(selected_l10n)
    if WELCOME_IMAGE.exists():
        await callback.message.answer_photo(
            photo=FSInputFile(WELCOME_IMAGE),
            caption=selected_l10n.get(PRODUCT_ONBOARDING_TEXT_KEY),
            reply_markup=onboarding_keyboard,
        )
    else:
        await callback.message.answer(
            selected_l10n.get(PRODUCT_ONBOARDING_TEXT_KEY),
            reply_markup=onboarding_keyboard,
        )

    await _mark_onboarding_completed(state)


@router.message(F.text.casefold() == "старт")
@router.message(F.text.casefold() == "start")
async def cmd_start_alias(
    message: Message,
    l10n: Localization,
    state: FSMContext | None = None,
) -> None:
    """Обработать текстовый алиас команды /start.

    Пользователи часто пишут "Старт" вручную вместо слеш-команды.
    Направляем такой ввод в основной обработчик /start.
    """
    await cmd_start(message, l10n, state)
