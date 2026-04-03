"""Тесты для доменного policy-модуля response_planner."""

from dataclasses import dataclass

from src.services.coaching.anti_loop_types import (
    AntiLoopDecision,
    RecoveryStrategy,
    SignalStrength,
)
from src.services.coaching.contracts.enums import (
    SafetyLevel,
    SessionMode,
    SessionStage,
)
from src.services.coaching.contracts.safety import SafetyDecision
from src.services.coaching.contracts.session import SessionState
from src.services.coaching.premium_trigger import (
    InsightQuality,
    PremiumDecision,
    PremiumReason,
)
from src.services.coaching.response_planner import plan_response


@dataclass(slots=True)
class _AdaptiveResponsePlan:
    step_type: str | None = None
    stage: SessionStage | str | None = None


@dataclass(slots=True)
class _AdaptiveState:
    stage: SessionStage | str


@dataclass(slots=True)
class _AdaptiveResult:
    reply_type: str = "coach_message"
    response_plan: _AdaptiveResponsePlan | None = None
    state: _AdaptiveState | None = None


def _build_state(
    stage: SessionStage = SessionStage.MECHANISM_DISCOVERY,
) -> SessionState:
    return SessionState(
        session_id="session-1",
        cycle_id="cycle-1",
        user_id=1,
        mode=SessionMode.FREE,
        stage=stage,
    )


def _safe_decision() -> SafetyDecision:
    return SafetyDecision(level=SafetyLevel.SAFE)


def _premium_offer(
    reason: PremiumReason = PremiumReason.DEPTH_AVAILABLE,
) -> PremiumDecision:
    return PremiumDecision(
        should_offer=True,
        should_postpone=False,
        reason=reason,
        timing_ok=True,
        psychological_open=True,
        insight_quality=InsightQuality.MEDIUM,
    )


def _anti_loop_strategy(
    strategy: RecoveryStrategy,
    *,
    should_soft_close: bool = False,
) -> AntiLoopDecision:
    return AntiLoopDecision(
        loop_detected=True,
        signal_strength=SignalStrength.CLEAR,
        strategy=strategy,
        should_soft_close=should_soft_close,
        reason="repeated_system_angle",
        stagnation_detected=True,
        requires_intervention=True,
    )


def test_crisis_interrupt_has_top_priority() -> None:
    """При кризисе planner должен прервать обычный путь."""
    state = _build_state()
    crisis = SafetyDecision(
        level=SafetyLevel.CRISIS,
        can_continue=False,
        must_switch_to_crisis_protocol=True,
        must_block_premium_trigger=True,
    )

    result = plan_response(
        state=state,
        safety_decision=crisis,
        adaptive_engine_result=_AdaptiveResult(),
        anti_loop_decision=_anti_loop_strategy(RecoveryStrategy.ANGLE_SHIFT),
        premium_decision=_premium_offer(),
    )

    assert result.reply_type == "crisis_interrupt"
    assert result.should_interrupt
    assert not result.use_reflection
    assert result.next_step_type is None
    assert not result.show_premium
    assert not result.should_complete


def test_anti_loop_strategy_used_when_loop_detected() -> None:
    """При loop planner должен выбрать anti-loop стратегию."""
    state = _build_state()
    anti_loop = _anti_loop_strategy(RecoveryStrategy.ANGLE_SHIFT)

    result = plan_response(
        state=state,
        safety_decision=_safe_decision(),
        adaptive_engine_result=_AdaptiveResult(
            response_plan=_AdaptiveResponsePlan(step_type="structured_progress")
        ),
        anti_loop_decision=anti_loop,
        premium_decision=_premium_offer(),
    )

    assert result.reply_type == "anti_loop"
    assert not result.use_reflection
    assert result.next_step_type == RecoveryStrategy.ANGLE_SHIFT.value
    assert not result.show_premium
    assert not result.should_complete


def test_anti_loop_not_activated_on_weak_requires_intervention_only() -> None:
    """Слабый сигнал без loop/soft-close/CLEAR не активирует anti-loop ветку."""
    state = _build_state()
    adaptive = _AdaptiveResult(
        reply_type="coach_message",
        response_plan=_AdaptiveResponsePlan(step_type="structured_progress"),
    )
    weak_intervention = AntiLoopDecision(
        loop_detected=False,
        signal_strength=SignalStrength.WEAK,
        strategy=RecoveryStrategy.INSIGHT_TRIGGER,
        should_soft_close=False,
        reason="stagnation_detected",
        stagnation_detected=True,
        requires_intervention=True,
    )

    result = plan_response(
        state=state,
        safety_decision=_safe_decision(),
        adaptive_engine_result=adaptive,
        anti_loop_decision=weak_intervention,
        premium_decision=None,
    )

    assert result.reply_type == "coach_message"
    assert result.use_reflection
    assert result.next_step_type == "structured_progress"


def test_anti_loop_activated_on_clear_signal_without_loop_detected() -> None:
    """Явный CLEAR-сигнал должен активировать anti-loop даже без loop_detected."""
    state = _build_state()
    clear_signal = AntiLoopDecision(
        loop_detected=False,
        signal_strength=SignalStrength.CLEAR,
        strategy=RecoveryStrategy.META_QUESTION,
        should_soft_close=False,
        reason="stagnation_detected",
        stagnation_detected=True,
        requires_intervention=True,
    )

    result = plan_response(
        state=state,
        safety_decision=_safe_decision(),
        adaptive_engine_result=_AdaptiveResult(),
        anti_loop_decision=clear_signal,
        premium_decision=None,
    )

    assert result.reply_type == "anti_loop"
    assert result.next_step_type == RecoveryStrategy.META_QUESTION.value


def test_anti_loop_activated_on_soft_close_without_loop_detected() -> None:
    """Soft-close должен активировать anti-loop независимо от loop_detected."""
    state = _build_state()
    soft_close = AntiLoopDecision(
        loop_detected=False,
        signal_strength=SignalStrength.WEAK,
        strategy=RecoveryStrategy.SOFT_CLOSE,
        should_soft_close=True,
        reason="stagnation_detected",
        stagnation_detected=True,
        requires_intervention=True,
    )

    result = plan_response(
        state=state,
        safety_decision=_safe_decision(),
        adaptive_engine_result=_AdaptiveResult(),
        anti_loop_decision=soft_close,
        premium_decision=None,
    )

    assert result.reply_type == "anti_loop"
    assert result.should_complete
    assert result.completion_stage == SessionStage.REFLECTION_SUMMARY.value


def test_anti_loop_soft_close_marks_completion() -> None:
    """При soft-close planner должен переводить в завершение."""
    state = _build_state()
    anti_loop = _anti_loop_strategy(
        RecoveryStrategy.SOFT_CLOSE,
        should_soft_close=True,
    )

    result = plan_response(
        state=state,
        safety_decision=_safe_decision(),
        adaptive_engine_result=_AdaptiveResult(),
        anti_loop_decision=anti_loop,
        premium_decision=_premium_offer(),
    )

    assert result.should_complete
    assert result.completion_stage == SessionStage.REFLECTION_SUMMARY.value
    assert not result.show_premium


def test_premium_shown_when_no_blockers() -> None:
    """Premium показывается только при отсутствии блокирующих условий."""
    state = _build_state()
    state.__dict__["reflection_depth"] = "medium"
    adaptive = _AdaptiveResult(
        reply_type="coach_message",
        response_plan=_AdaptiveResponsePlan(step_type="structured_progress"),
    )

    result = plan_response(
        state=state,
        safety_decision=_safe_decision(),
        adaptive_engine_result=adaptive,
        anti_loop_decision=None,
        premium_decision=_premium_offer(PremiumReason.DEPTH_AVAILABLE),
    )

    assert result.show_premium
    assert result.premium_reason == PremiumReason.DEPTH_AVAILABLE.value
    assert result.use_reflection
    assert result.reflection_depth == "medium"
    assert result.next_step_type == "structured_progress"


def test_premium_not_shown_in_crisis() -> None:
    """При кризисе premium не показывается."""
    state = _build_state()

    result = plan_response(
        state=state,
        safety_decision=SafetyDecision(level=SafetyLevel.CRISIS, can_continue=False),
        adaptive_engine_result=_AdaptiveResult(),
        anti_loop_decision=None,
        premium_decision=_premium_offer(),
    )

    assert not result.show_premium


def test_premium_not_shown_on_completion() -> None:
    """При завершении premium не показывается."""
    state = _build_state(stage=SessionStage.COMPLETION)

    result = plan_response(
        state=state,
        safety_decision=_safe_decision(),
        adaptive_engine_result=_AdaptiveResult(),
        anti_loop_decision=None,
        premium_decision=_premium_offer(),
    )

    assert result.should_complete
    assert result.completion_stage == SessionStage.COMPLETION.value
    assert not result.show_premium


def test_premium_not_shown_when_anti_loop_active() -> None:
    """При active anti-loop premium должен быть скрыт."""
    state = _build_state()

    result = plan_response(
        state=state,
        safety_decision=_safe_decision(),
        adaptive_engine_result=_AdaptiveResult(),
        anti_loop_decision=_anti_loop_strategy(RecoveryStrategy.META_QUESTION),
        premium_decision=_premium_offer(),
    )

    assert result.reply_type == "anti_loop"
    assert not result.show_premium


def test_normal_path_uses_reflection_and_adaptive_step() -> None:
    """На основном пути planner должен использовать reflection и adaptive step."""
    state = _build_state()
    state.__dict__["reflection_depth"] = "low"
    adaptive = _AdaptiveResult(
        reply_type="coach_message",
        response_plan=_AdaptiveResponsePlan(step_type="clarity_structuring"),
    )

    result = plan_response(
        state=state,
        safety_decision=_safe_decision(),
        adaptive_engine_result=adaptive,
        anti_loop_decision=None,
        premium_decision=None,
    )

    assert result.reply_type == "coach_message"
    assert result.use_reflection
    assert result.reflection_depth == "low"
    assert result.next_step_type == "clarity_structuring"
    assert not result.should_complete


def test_completion_stage_reflection_summary_supported() -> None:
    """Completion-ветка должна корректно поддерживать REFLECTION_SUMMARY stage."""
    state = _build_state(stage=SessionStage.REFLECTION_SUMMARY)

    result = plan_response(
        state=state,
        safety_decision=_safe_decision(),
        adaptive_engine_result=_AdaptiveResult(),
        anti_loop_decision=None,
        premium_decision=_premium_offer(),
    )

    assert result.should_complete
    assert result.completion_stage == SessionStage.REFLECTION_SUMMARY.value
    assert not result.show_premium
