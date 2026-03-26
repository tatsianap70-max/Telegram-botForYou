"""Тесты для SessionManager."""

from src.services.coaching.contracts.enums import (
    SafetyLevel,
    ScenarioType,
    SessionMode,
    SessionStage,
)
from src.services.coaching.contracts.session import SessionState
from src.services.coaching.session_manager import (
    DeepSessionLimitError,
    InvalidStageTransitionError,
    SessionCompletionError,
    SessionManager,
    TopicSwitchPolicyError,
)


def _build_free_state() -> SessionState:
    return SessionState(
        session_id="session-1",
        cycle_id="cycle-1",
        user_id=101,
        mode=SessionMode.FREE,
        topic_id="topic-a",
        timezone="Europe/Oslo",
    )


def _build_premium_state() -> SessionState:
    return SessionState(
        session_id="session-2",
        cycle_id="cycle-2",
        user_id=202,
        mode=SessionMode.PREMIUM,
        topic_id="topic-p",
        timezone="Europe/Oslo",
    )


def test_transition_stage_accepts_valid_transition() -> None:
    """Менеджер принимает валидный переход стадии."""
    manager = SessionManager()
    state = _build_free_state()

    state = manager.transition_stage(state, SessionStage.TENSION_REDUCTION)

    assert state.stage == SessionStage.TENSION_REDUCTION


def test_transition_stage_rejects_invalid_transition() -> None:
    """Менеджер отклоняет невалидный переход стадии."""
    manager = SessionManager()
    state = _build_free_state()

    try:
        manager.transition_stage(state, SessionStage.COMPLETION)
    except InvalidStageTransitionError:
        pass
    else:
        raise AssertionError("Ожидалось InvalidStageTransitionError")


def test_set_topic_rejects_direct_topic_change_in_same_cycle() -> None:
    """Смена темы без decision node запрещена."""
    manager = SessionManager()
    state = _build_free_state()

    try:
        manager.set_topic(state, "topic-b")
    except TopicSwitchPolicyError:
        pass
    else:
        raise AssertionError("Ожидалось TopicSwitchPolicyError")


def test_request_topic_switch_moves_state_to_decision_node() -> None:
    """Запрос смены темы переводит сессию в decision node."""
    manager = SessionManager()
    state = _build_free_state()

    state = manager.request_topic_switch(state, "topic-b")

    assert state.pending_topic_id == "topic-b"
    assert state.stage == SessionStage.DECISION_NODE


def test_apply_topic_switch_decision_resets_cycle_state() -> None:
    """Подтверждённая смена темы создаёт новый цикл и сбрасывает метрики."""
    manager = SessionManager()
    state = _build_free_state()
    state.question_count = 7
    state.contentful_message_count = 6
    state.insight_count = 2
    state.premium_trigger_shown = True
    state.stage = SessionStage.DECISION_NODE
    state.pending_topic_id = "topic-new"
    state.scenario_type = ScenarioType.CLARITY_FALLBACK

    state = manager.apply_topic_switch_decision(
        state,
        approve_switch=True,
        new_cycle_id="cycle-3",
    )

    assert state.cycle_id == "cycle-3"
    assert state.topic_id == "topic-new"
    assert state.pending_topic_id is None
    assert state.stage == SessionStage.TOPIC_DEFINITION
    assert state.question_count == 0
    assert state.contentful_message_count == 0
    assert state.insight_count == 0
    assert not state.premium_trigger_shown
    assert state.scenario_type == ScenarioType.A_STRUCTURED


def test_can_show_premium_trigger_checks_all_conditions() -> None:
    """Premium trigger доступен только при выполнении формальных условий."""
    manager = SessionManager()
    state = _build_free_state()
    state.insight_count = 1
    state.contentful_message_count = 5

    assert manager.can_show_premium_trigger(state)

    state.premium_trigger_shown = True
    assert not manager.can_show_premium_trigger(state)


def test_can_show_premium_trigger_blocks_on_crisis_or_resistance() -> None:
    """В кризисе и при сопротивлении trigger показывать нельзя."""
    manager = SessionManager()
    state = _build_free_state()
    state.insight_count = 1
    state.contentful_message_count = 5

    state.safety_decision.level = SafetyLevel.CRISIS
    assert not manager.can_show_premium_trigger(state)

    state.safety_decision.level = SafetyLevel.SAFE
    assert not manager.can_show_premium_trigger(state, has_user_resistance=True)


def test_deep_premium_limit_allows_only_one_session_per_day() -> None:
    """После резервирования deep-сессии повторный запуск блокируется."""
    manager = SessionManager()
    state = _build_premium_state()

    assert manager.can_start_deep_premium_session(state)
    manager.reserve_deep_premium_session(state)
    assert not state.deep_session_allowed_today

    try:
        manager.reserve_deep_premium_session(state)
    except DeepSessionLimitError:
        pass
    else:
        raise AssertionError("Ожидалось DeepSessionLimitError")


def test_finalize_if_ready_moves_to_completion() -> None:
    """Сессия завершается только при выполнении критериев."""
    manager = SessionManager()
    state = _build_free_state()
    state.stage = SessionStage.REFLECTION_SUMMARY
    state.completion_criteria.mechanism_formulated = True
    state.completion_criteria.influence_zone_defined = True
    state.completion_criteria.state_shift_confirmed = True
    state.completion_criteria.integration_summary_given = True

    state = manager.finalize_if_ready(state)

    assert state.stage == SessionStage.COMPLETION
    assert state.completion_criteria.is_completed


def test_finalize_if_ready_raises_when_criteria_not_met() -> None:
    """Если критерии не выполнены, завершение сессии запрещено."""
    manager = SessionManager()
    state = _build_free_state()
    state.stage = SessionStage.REFLECTION_SUMMARY
    state.completion_criteria.mechanism_formulated = True

    try:
        manager.finalize_if_ready(state)
    except SessionCompletionError:
        pass
    else:
        raise AssertionError("Ожидалось SessionCompletionError")
