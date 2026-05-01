"""Точечные тесты роутинга fallback-сборки ответа по step_type."""

import pytest

from src.bot.handlers.chatgpt import _compose_domain_response
from src.services.coaching.contracts.enums import SessionMode
from src.services.coaching.contracts.session import SessionState
from src.services.coaching.response_planner import ResponsePlan as DomainResponsePlan


def _build_state() -> SessionState:
    return SessionState(
        session_id="sess-compose",
        cycle_id="cycle-compose",
        user_id=1,
        mode=SessionMode.FREE,
        question_count=1,
    )


def _build_plan(step_type: str) -> DomainResponsePlan:
    return DomainResponsePlan(
        reply_type="coach_message",
        should_interrupt=False,
        use_reflection=True,
        reflection_depth=None,
        next_step_type=step_type,
        show_premium=False,
        premium_reason=None,
        should_complete=False,
        completion_stage=None,
    )


def _build_plan_with_reply(step_type: str, reply_type: str) -> DomainResponsePlan:
    return DomainResponsePlan(
        reply_type=reply_type,
        should_interrupt=False,
        use_reflection=True,
        reflection_depth=None,
        next_step_type=step_type,
        show_premium=False,
        premium_reason=None,
        should_complete=False,
        completion_stage=None,
    )


def test_coach_message_emotion_contact_does_not_use_stage_question_prompt() -> None:
    """При emotion_contact берется вопрос от step_type, а не stage-шаблон."""
    state = _build_state()
    result = _compose_domain_response(
        coaching_state=state,
        user_text="Мне тревожно, не знаю, как успокоиться.",
        response_plan=_build_plan("emotion_contact"),
    )

    assert "Что в этом переживается сильнее всего прямо сейчас?" in result
    assert "Что здесь для вас сейчас самое важное?" not in result


def test_coach_message_regular_step_uses_step_type_question_not_stage() -> None:
    """Обычный coach-message кейс строится от step_type, а не от stage."""
    state = _build_state()
    result = _compose_domain_response(
        coaching_state=state,
        user_text="Хочу разобрать рабочую ситуацию.",
        response_plan=_build_plan("focus_soft"),
    )

    assert "На чем в этой ситуации лучше сфокусироваться сначала?" in result
    assert "Что здесь для вас сейчас самое важное?" not in result


@pytest.mark.parametrize(
    "step_type",
    ["structured_progress", "state_clarification", "clarity_structuring"],
)
def test_coach_message_target_steps_no_mirroring_reflection(
    step_type: str,
) -> None:
    state = _build_state()
    result = _compose_domain_response(
        coaching_state=state,
        user_text="Мне тревожно и неясно, как двигаться дальше.",
        response_plan=_build_plan(step_type),
    )

    assert "Похоже, здесь много про" not in result


@pytest.mark.parametrize(
    "step_type",
    ["structured_progress", "state_clarification", "clarity_structuring"],
)
def test_coach_message_target_steps_do_not_use_stage_questions(step_type: str) -> None:
    state = _build_state()
    result = _compose_domain_response(
        coaching_state=state,
        user_text="Я запуталась и не понимаю, как успокоиться.",
        response_plan=_build_plan(step_type),
    )

    assert "Что в этом факт, а что ваша интерпретация?" not in result
    assert "Что запускается первым: мысль, эмоция или реакция?" not in result


def test_special_branches_stay_unchanged() -> None:
    state = _build_state()

    anti_loop_result = _compose_domain_response(
        coaching_state=state,
        user_text="Я уже отвечала на это.",
        response_plan=_build_plan_with_reply("simplify", "anti_loop"),
    )
    decision_result = _compose_domain_response(
        coaching_state=state,
        user_text="Хочу понять, как лучше продолжить.",
        response_plan=_build_plan_with_reply("structured_progress", "decision_prompt"),
    )

    assert "Похоже, лучше упростить до одной точки." in anti_loop_result
    assert "Что сейчас здесь самое важное?" in anti_loop_result
    assert "Что здесь для вас сейчас самое важное?" in decision_result
