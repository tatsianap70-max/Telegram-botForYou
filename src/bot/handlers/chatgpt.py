"""Обработчик команды /chatgpt — диалог с AI-моделями."""

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from uuid import uuid4

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import BotCommand, CallbackQuery, InaccessibleMessage, Message
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from src.bot.keyboards import create_model_selection_keyboard
from src.bot.states import ChatGPTStates
from src.config.yaml_config import yaml_config
from src.db.base import DatabaseSession
from src.db.repositories import MessageRepository
from src.providers.ai.base import GenerationType
from src.services.ai_service import AIService, create_ai_service
from src.services.coaching.adaptive_session_engine import AdaptiveSessionEngine
from src.services.coaching.anti_loop import evaluate_anti_loop
from src.services.coaching.contracts.enums import SessionMode
from src.services.coaching.contracts.session import SessionState
from src.services.coaching.premium_trigger import (
    evaluate_premium_trigger,
    mark_trigger_shown,
)
from src.services.coaching.response_planner import (
    ResponsePlan as DomainResponsePlan,
)
from src.services.coaching.response_planner import (
    plan_response,
)
from src.services.coaching.safety_layer import (
    build_crisis_response,
    evaluate_safety,
    should_interrupt_session,
)
from src.services.coaching.session_manager import SessionManager
from src.services.generation import ChatGenerationService
from src.utils import send_chat_action, send_long_message
from src.utils.i18n import Localization
from src.utils.logging import get_logger

COMMAND = BotCommand(command="chatgpt", description="💬 Диалог с AI")
router = Router(name="chatgpt")
fsm_router = Router(name="chatgpt_fsm")
logger = get_logger(__name__)

GENERATION_TYPE_CHAT = "chat"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
COACHING_STATE_KEY = "coaching_session_state"
COACHING_RECENT_TURNS_KEY = "coaching_recent_user_turns"
COACHING_LAST_PLAN_KEY = "coaching_last_response_plan"


async def _send_ai_response(message: Message, content: str) -> None:
    """Отправить ответ AI пользователю с typing indicator и разбиением на части."""
    await send_chat_action(message, GenerationType.CHAT)
    await send_long_message(message, content)


def _create_default_coaching_state(user_id: int) -> SessionState:
    """Создать дефолтное состояние доменной сессии для пользователя."""
    return SessionState(
        session_id=f"session-{uuid4().hex[:12]}",
        cycle_id=f"cycle-{uuid4().hex[:12]}",
        user_id=user_id,
        mode=SessionMode.FREE,
    )


def _load_coaching_state(raw_state: object, user_id: int) -> SessionState:
    """Загрузить SessionState из FSM или создать новый."""
    if isinstance(raw_state, SessionState):
        if raw_state.user_id == user_id:
            return raw_state
        return _create_default_coaching_state(user_id)

    if isinstance(raw_state, dict):
        try:
            restored = SessionState.model_validate(raw_state)
            if restored.user_id == user_id:
                return restored
        except ValidationError:
            pass

    return _create_default_coaching_state(user_id)


def _serialize_domain_plan(plan: DomainResponsePlan | None) -> dict[str, object] | None:
    """Преобразовать ResponsePlan к сериализуемому виду."""
    if plan is None:
        return None
    return {
        "reply_type": plan.reply_type,
        "should_interrupt": plan.should_interrupt,
        "use_reflection": plan.use_reflection,
        "reflection_depth": plan.reflection_depth,
        "next_step_type": plan.next_step_type,
        "show_premium": plan.show_premium,
        "premium_reason": plan.premium_reason,
        "should_complete": plan.should_complete,
        "completion_stage": plan.completion_stage,
    }


async def _persist_coaching_pipeline_state(
    state: FSMContext,
    coaching_state: SessionState,
    recent_user_turns: list[str],
    response_plan: DomainResponsePlan | None,
) -> None:
    """Сохранить обновленное доменное состояние в FSM."""
    payload: dict[str, object] = {
        COACHING_STATE_KEY: coaching_state.model_dump(mode="json"),
        COACHING_RECENT_TURNS_KEY: recent_user_turns[-3:],
    }
    serialized_plan = _serialize_domain_plan(response_plan)
    if serialized_plan is not None:
        payload[COACHING_LAST_PLAN_KEY] = serialized_plan
    await state.update_data(payload)


@router.message(Command(COMMAND))
async def cmd_chatgpt(
    message: Message,
    state: FSMContext,
    l10n: Localization,
    ai_service: AIService | None = None,
) -> None:
    """Обработать команду /chatgpt и показать выбор модели."""
    if not message.from_user:
        return

    if ai_service is None:
        ai_service = create_ai_service()

    available_models = ai_service.get_available_models()
    keyboard = create_model_selection_keyboard(available_models, GENERATION_TYPE_CHAT)

    if not keyboard.inline_keyboard:
        await message.answer(l10n.get("no_models_available"))
        logger.warning(
            "Нет доступных chat-моделей для пользователя %d (проверьте API-ключи)",
            message.from_user.id,
        )
        return

    await state.set_state(ChatGPTStates.waiting_for_model_selection)
    await message.answer(
        l10n.get("chatgpt_choose_model"),
        reply_markup=keyboard,
    )
    logger.info("Пользователь %d начал диалог /chatgpt", message.from_user.id)


@fsm_router.callback_query(
    ChatGPTStates.waiting_for_model_selection,
    F.data.startswith("model:"),
)
async def handle_model_selection(
    callback: CallbackQuery,
    state: FSMContext,
    l10n: Localization,
) -> None:
    """Обработать выбор модели пользователем."""
    if (
        not callback.data
        or not callback.message
        or isinstance(callback.message, InaccessibleMessage)
    ):
        return

    model_key = callback.data.split(":", 1)[1]
    await state.update_data(model_key=model_key)
    await state.set_state(ChatGPTStates.waiting_for_message)
    await callback.message.edit_text(
        l10n.get("chatgpt_model_selected", model_key=model_key),
    )
    await callback.answer()

    logger.info(
        "Пользователь %d выбрал модель: %s",
        callback.from_user.id,
        model_key,
    )


@fsm_router.message(ChatGPTStates.waiting_for_message, F.text, ~F.text.startswith("/"))
async def handle_user_message(
    message: Message,
    state: FSMContext,
    l10n: Localization,
    ai_service: AIService | None = None,
    session_factory: Callable[
        [], AbstractAsyncContextManager[AsyncSession]
    ] = DatabaseSession,
) -> None:
    """Обработать сообщение пользователя и сгенерировать ответ AI."""
    if not message.from_user or not message.text:
        return

    state_data = await state.get_data()
    model_key = state_data.get("model_key")
    if not model_key:
        await message.answer(l10n.get("chatgpt_model_not_selected"))
        return

    session_manager = SessionManager()
    adaptive_engine = AdaptiveSessionEngine(session_manager=session_manager)

    coaching_state = _load_coaching_state(
        state_data.get(COACHING_STATE_KEY),
        message.from_user.id,
    )
    session_manager.enforce_scenario_policy(coaching_state)

    raw_recent_turns = state_data.get(COACHING_RECENT_TURNS_KEY, [])
    recent_user_turns = (
        [turn for turn in raw_recent_turns if isinstance(turn, str)]
        if isinstance(raw_recent_turns, list)
        else []
    )
    recent_user_turns.append(message.text)
    recent_user_turns = recent_user_turns[-3:]

    safety_decision = evaluate_safety(message.text, state=coaching_state)
    coaching_state.safety_decision = safety_decision

    if should_interrupt_session(safety_decision):
        response_plan = plan_response(
            state=coaching_state,
            safety_decision=safety_decision,
            adaptive_engine_result=None,
            anti_loop_decision=None,
            premium_decision=None,
        )
        await message.answer(build_crisis_response())
        await _persist_coaching_pipeline_state(
            state,
            coaching_state,
            recent_user_turns,
            response_plan,
        )
        return

    adaptive_result = adaptive_engine.process_user_message(
        coaching_state,
        message.text,
    )
    anti_loop_decision = evaluate_anti_loop(
        coaching_state,
        recent_user_turns,
    )
    premium_decision = evaluate_premium_trigger(coaching_state)
    response_plan = plan_response(
        state=coaching_state,
        safety_decision=safety_decision,
        adaptive_engine_result=adaptive_result,
        anti_loop_decision=anti_loop_decision,
        premium_decision=premium_decision,
    )
    if response_plan.show_premium:
        mark_trigger_shown(coaching_state)

    try:
        processing_msg = await message.answer(l10n.get("chatgpt_generating"))

        async with session_factory() as session:
            message_repo = MessageRepository(session)

            from src.db.repositories import UserRepository

            user_repo = UserRepository(session)
            user = await user_repo.get_by_telegram_id(message.from_user.id)
            if not user:
                await processing_msg.edit_text(l10n.get("error_user_not_found"))
                return

            context_messages = await message_repo.get_context(
                user_id=user.id,
                model_key=model_key,
                max_messages=yaml_config.limits.max_context_messages,
            )

            await message_repo.add_message(
                user_id=user.id,
                model_key=model_key,
                role=ROLE_USER,
                content=message.text,
            )

            messages_for_ai = []
            model_config = yaml_config.get_model(model_key)
            if model_config and model_config.system_prompt:
                messages_for_ai.append(
                    {"role": "system", "content": model_config.system_prompt}
                )

            messages_for_ai.extend(
                [{"role": msg.role, "content": msg.content} for msg in context_messages]
            )
            messages_for_ai.append({"role": ROLE_USER, "content": message.text})

            generation_service = ChatGenerationService(session, ai_service=ai_service)
            result = await generation_service.execute(
                telegram_user_id=message.from_user.id,
                model_key=model_key,
                processing_msg=processing_msg,
                l10n=l10n,
                messages=messages_for_ai,
                user_id=user.id,
            )

            if not result.success:
                return

            await processing_msg.delete()
            await _send_ai_response(message, result.content)
    finally:
        await _persist_coaching_pipeline_state(
            state,
            coaching_state,
            recent_user_turns,
            response_plan,
        )
