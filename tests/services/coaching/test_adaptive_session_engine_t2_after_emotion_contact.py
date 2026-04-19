"""Точечные тесты T2-перехода после T1 emotion_contact."""

from src.services.coaching.adaptive_session_engine import AdaptiveSessionEngine
from src.services.coaching.contracts.enums import SessionMode, SessionStage
from src.services.coaching.contracts.session import SessionState


def _build_state() -> SessionState:
    return SessionState(
        session_id="sess-t2-guard",
        cycle_id="cycle-t2-guard",
        user_id=1,
        mode=SessionMode.FREE,
        stage=SessionStage.TOPIC_DEFINITION,
        topic_id="topic-1",
    )


def test_t2_after_emotion_contact_does_not_default_to_state_clarification() -> None:
    """После T1 emotion_contact обычный ответ на T2 не уходит в clarification."""
    engine = AdaptiveSessionEngine()
    state = _build_state()

    first = engine.process_turn(state, "Мне тревожно перед разговором.")
    second = engine.process_turn(state, "Я хочу начать с самого сложного момента.")

    assert first.response_plan.step_type == "emotion_contact"
    assert second.response_plan.step_type != "state_clarification"


def test_t2_after_emotion_contact_with_confusion_uses_state_clarification() -> None:
    """После T1 emotion_contact явное непонимание на T2 ведет в clarification."""
    engine = AdaptiveSessionEngine()
    state = _build_state()

    first = engine.process_turn(state, "Мне тревожно перед разговором.")
    second = engine.process_turn(state, "Не понимаю, что ты имеешь в виду.")

    assert first.response_plan.step_type == "emotion_contact"
    assert second.response_plan.step_type == "state_clarification"
