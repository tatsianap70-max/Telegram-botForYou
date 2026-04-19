"""Точечный регрессионный тест для тревожного T1-входа."""

from src.services.coaching.adaptive_session_engine import AdaptiveSessionEngine
from src.services.coaching.contracts.enums import SessionMode, SessionStage
from src.services.coaching.contracts.session import SessionState


def _build_state() -> SessionState:
    return SessionState(
        session_id="sess-trevozhno",
        cycle_id="cycle-trevozhno",
        user_id=1,
        mode=SessionMode.FREE,
        stage=SessionStage.TOPIC_DEFINITION,
        topic_id="topic-1",
    )


def test_first_turn_trevozhno_with_uncertainty_stays_emotional_soft_entry() -> None:
    """Форма «тревожно» на T1 должна приоритизировать emotional soft-entry."""
    engine = AdaptiveSessionEngine()
    state = _build_state()

    result = engine.process_turn(state, "Мне тревожно, не знаю, как мне успокоиться.")

    assert result.state.stage == SessionStage.TOPIC_DEFINITION
    assert result.response_plan.step_type == "emotion_contact"
