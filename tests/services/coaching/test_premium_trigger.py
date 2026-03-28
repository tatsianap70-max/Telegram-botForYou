"""Тесты для доменного модуля premium_trigger (v2)."""

from src.services.coaching.contracts.emotion import EmotionDetectionResult
from src.services.coaching.contracts.enums import (
    EmotionState,
    ScenarioType,
    SessionMode,
    SessionStage,
)
from src.services.coaching.contracts.safety import SafetyDecision
from src.services.coaching.contracts.session import SessionState
from src.services.coaching.premium_trigger import (
    InsightQuality,
    PremiumReason,
    build_premium_message,
    build_premium_reflection,
    build_transition_bridge,
    determine_premium_reason,
    evaluate_insight_quality,
    evaluate_premium_trigger,
    is_bad_timing,
    is_psychological_moment_open,
    mark_trigger_shown,
    pre_premium_reflection,
    premium_readiness_check,
    should_offer_premium,
)


def _build_state(
    *,
    mode: SessionMode = SessionMode.FREE,
    stage: SessionStage = SessionStage.REFLECTION_SUMMARY,
    scenario_type: ScenarioType = ScenarioType.A_STRUCTURED,
) -> SessionState:
    state = SessionState(
        session_id="session-1",
        cycle_id="cycle-1",
        user_id=11,
        topic_id="topic-1",
        mode=mode,
        stage=stage,
        scenario_type=scenario_type,
        question_count=6,
        insight_count=1,
        contentful_message_count=5,
        premium_trigger_shown=False,
        load_limiter_active=False,
        low_engagement_turns=0,
        safety_decision=SafetyDecision(
            must_block_premium_trigger=False,
        ),
        emotion_result=EmotionDetectionResult(
            state=EmotionState.ANXIETY,
            confidence=0.8,
            is_confident=True,
            fallback_to_clarity=False,
        ),
    )
    state.completion_criteria.mechanism_formulated = True
    return state


def test_evaluate_insight_quality_low_for_formal_only_insight() -> None:
    """Формальный одиночный инсайт без критериев должен быть LOW."""
    state = _build_state()
    state.insight_count = 1
    state.completion_criteria.mechanism_formulated = False
    state.completion_criteria.state_shift_confirmed = False
    state.completion_criteria.integration_summary_given = False

    quality = evaluate_insight_quality(state)

    assert quality is InsightQuality.LOW


def test_evaluate_insight_quality_medium_and_high_by_completion_signals() -> None:
    """Качество должно расти по доступным completion-сигналам."""
    medium_state = _build_state()
    medium_state.insight_count = 1
    medium_state.completion_criteria.mechanism_formulated = True

    high_state = _build_state()
    high_state.insight_count = 2
    high_state.completion_criteria.mechanism_formulated = True
    high_state.completion_criteria.state_shift_confirmed = True

    assert evaluate_insight_quality(medium_state) is InsightQuality.MEDIUM
    assert evaluate_insight_quality(high_state) is InsightQuality.HIGH


def test_premium_readiness_check_requires_safe_and_quality() -> None:
    """Readiness должен блокироваться при небезопасном уровне и LOW качестве."""
    state = _build_state()

    assert premium_readiness_check(state)

    state.safety_decision.level = state.safety_decision.level.CRISIS
    assert not premium_readiness_check(state)

    state.safety_decision.level = state.safety_decision.level.SAFE
    state.completion_criteria.mechanism_formulated = False
    state.completion_criteria.state_shift_confirmed = False
    state.completion_criteria.integration_summary_given = False
    assert not premium_readiness_check(state)


def test_is_bad_timing_true_for_overload_short_answers_or_resistance() -> None:
    """Плохой тайминг должен ловить перегруз, короткие ответы и сопротивление."""
    overload_state = _build_state(scenario_type=ScenarioType.CLARITY_FALLBACK)
    overload_state.emotion_result.state = EmotionState.OVERLOAD

    short_state = _build_state()
    short_state.low_engagement_turns = 2

    resistance_state = _build_state()
    resistance_state.__dict__["resistance_detected"] = True

    assert is_bad_timing(overload_state)
    assert is_bad_timing(short_state)
    assert is_bad_timing(resistance_state)


def test_is_psychological_moment_open_distinct_from_bad_timing() -> None:
    """Психологическая открытость оценивается отдельно от bad timing."""
    state = _build_state()

    assert is_psychological_moment_open(state)

    state.completion_criteria.mechanism_formulated = False
    state.completion_criteria.state_shift_confirmed = False
    state.completion_criteria.integration_summary_given = False
    assert not is_psychological_moment_open(state)


def test_determine_premium_reason_supports_all_priority_paths() -> None:
    """Причина должна выбираться по приоритету user->memory->pattern->depth."""
    wants_deeper = _build_state()
    wants_deeper.__dict__["user_wants_deeper"] = True

    memory_value = _build_state()
    memory_value.insight_count = 2
    memory_value.completion_criteria.mechanism_formulated = True

    pattern = _build_state(stage=SessionStage.PATTERN_HIGHLIGHT)

    depth = _build_state(stage=SessionStage.INSIGHT)
    depth.completion_criteria.mechanism_formulated = True

    assert determine_premium_reason(wants_deeper) is PremiumReason.USER_WANTS_DEEPER
    assert determine_premium_reason(memory_value) is PremiumReason.PATTERN_MEMORY_VALUE
    assert determine_premium_reason(pattern) is PremiumReason.REPEATING_PATTERN
    assert determine_premium_reason(depth) is PremiumReason.DEPTH_AVAILABLE


def test_build_transition_bridge_returns_named_structure() -> None:
    """Transition bridge должен возвращать именованную структуру с 2 полями."""
    state = _build_state()

    bridge = build_transition_bridge(state, PremiumReason.PATTERN_MEMORY_VALUE)

    assert bridge.recognition
    assert bridge.depth_bridge


def test_evaluate_premium_trigger_offer_flow_and_reason() -> None:
    """При readiness+timing+open решение должно вести к offer с reason."""
    state = _build_state()

    decision = evaluate_premium_trigger(state)

    assert decision.should_offer
    assert not decision.should_postpone
    assert decision.reason is not None
    assert decision.timing_ok
    assert decision.psychological_open


def test_evaluate_premium_trigger_postpone_only_once_per_cycle() -> None:
    """Postpone допустим один раз на цикл, затем только no-offer/offer."""
    state = _build_state()
    state.low_engagement_turns = 2

    first = evaluate_premium_trigger(state)
    second = evaluate_premium_trigger(state)

    assert not first.should_offer
    assert first.should_postpone
    assert not second.should_offer
    assert not second.should_postpone


def test_build_premium_message_and_reflections_are_soft() -> None:
    """Тексты перехода должны быть мягкими и без давления."""
    state = _build_state()
    reason = PremiumReason.PATTERN_MEMORY_VALUE

    message = build_premium_message(state, reason)
    pre = pre_premium_reflection(
        state,
        last_insight="Я заметила повторяющийся автоматический ответ.",
    )
    full = build_premium_reflection(
        state,
        last_insight="Я заметила повторяющийся автоматический ответ.",
    )

    assert "запоминает паттерны" in message
    assert "устойчивость" in message
    assert "куп" not in message.lower()
    assert pre is not None
    assert full is not None


def test_mark_trigger_shown_and_should_offer_compatibility_wrapper() -> None:
    """mark_trigger_shown и should_offer_premium должны быть согласованы."""
    state = _build_state()
    assert should_offer_premium(state)

    updated = mark_trigger_shown(state)

    assert updated is state
    assert updated.premium_trigger_shown
    assert not should_offer_premium(state)
