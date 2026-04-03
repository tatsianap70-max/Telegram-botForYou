"""Доменный policy-слой планирования следующего ответа без генерации текста."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.services.coaching.contracts.enums import SafetyLevel, SessionStage

if TYPE_CHECKING:
    from src.services.coaching.contracts.safety import SafetyDecision
    from src.services.coaching.contracts.session import SessionState


@dataclass(slots=True, frozen=True)
class ResponsePlan:
    """Структурный план следующего действия без UI и текстовой генерации."""

    reply_type: str
    should_interrupt: bool
    use_reflection: bool
    reflection_depth: str | None
    next_step_type: str | None
    show_premium: bool
    premium_reason: str | None
    should_complete: bool
    completion_stage: str | None


def plan_response(
    state: SessionState,
    safety_decision: SafetyDecision | None,
    adaptive_engine_result: object | None,
    anti_loop_decision: object | None,
    premium_decision: object | None,
) -> ResponsePlan:
    """Собрать policy-план ответа из готовых решений других модулей."""
    resolved_safety = safety_decision or state.safety_decision
    if _is_crisis(resolved_safety):
        return ResponsePlan(
            reply_type="crisis_interrupt",
            should_interrupt=True,
            use_reflection=False,
            reflection_depth=None,
            next_step_type=None,
            show_premium=False,
            premium_reason=None,
            should_complete=False,
            completion_stage=None,
        )

    current_stage = _resolve_stage_value(state, adaptive_engine_result)
    anti_loop_active = _is_anti_loop_active(anti_loop_decision)
    anti_loop_soft_close = _is_anti_loop_soft_close(anti_loop_decision)
    if anti_loop_active:
        should_complete = anti_loop_soft_close
        completion_stage = (
            _resolve_completion_stage(current_stage)
            if should_complete
            else None
        )
        return ResponsePlan(
            reply_type="anti_loop",
            should_interrupt=False,
            use_reflection=False,
            reflection_depth=None,
            next_step_type=_extract_strategy(anti_loop_decision),
            show_premium=False,
            premium_reason=None,
            should_complete=should_complete,
            completion_stage=completion_stage,
        )

    if _is_completion_stage(current_stage):
        return ResponsePlan(
            reply_type="completion",
            should_interrupt=False,
            use_reflection=False,
            reflection_depth=None,
            next_step_type=None,
            show_premium=False,
            premium_reason=None,
            should_complete=True,
            completion_stage=current_stage,
        )

    show_premium = _should_show_premium(
        premium_decision=premium_decision,
        blocked_by_crisis=False,
        blocked_by_completion=False,
        blocked_by_anti_loop=anti_loop_soft_close,
    )
    premium_reason = (
        _enum_or_string(getattr(premium_decision, "reason", None))
        if show_premium
        else None
    )
    return ResponsePlan(
        reply_type=_extract_reply_type(adaptive_engine_result),
        should_interrupt=False,
        use_reflection=True,
        reflection_depth=_optional_string(getattr(state, "reflection_depth", None)),
        next_step_type=_extract_next_step_type(adaptive_engine_result),
        show_premium=show_premium,
        premium_reason=premium_reason,
        should_complete=False,
        completion_stage=None,
    )


def _is_crisis(safety_decision: SafetyDecision) -> bool:
    """Определить кризисный кейс как высший приоритет планирования."""
    return bool(
        safety_decision.level is SafetyLevel.CRISIS
        or safety_decision.must_switch_to_crisis_protocol
        or not safety_decision.can_continue
    )


def _is_anti_loop_active(anti_loop_decision: object | None) -> bool:
    """Проверить, что anti-loop потребовал policy-вмешательство."""
    if anti_loop_decision is None:
        return False
    if bool(getattr(anti_loop_decision, "loop_detected", False)):
        return True
    if bool(getattr(anti_loop_decision, "should_soft_close", False)):
        return True
    signal_strength = _enum_or_string(
        getattr(anti_loop_decision, "signal_strength", None)
    )
    return signal_strength == "clear"


def _is_anti_loop_soft_close(anti_loop_decision: object | None) -> bool:
    """Проверить, требует ли anti-loop мягкого завершения."""
    if anti_loop_decision is None:
        return False
    return bool(getattr(anti_loop_decision, "should_soft_close", False))


def _should_show_premium(
    *,
    premium_decision: object | None,
    blocked_by_crisis: bool,
    blocked_by_completion: bool,
    blocked_by_anti_loop: bool,
) -> bool:
    """Проверить, можно ли показывать premium-переход на этом ходе."""
    if blocked_by_crisis or blocked_by_completion or blocked_by_anti_loop:
        return False
    if premium_decision is None:
        return False
    return bool(getattr(premium_decision, "should_offer", False))


def _resolve_stage_value(
    state: SessionState,
    adaptive_engine_result: object | None,
) -> str:
    """Определить текущую stage в строковом виде."""
    stage_obj = getattr(getattr(adaptive_engine_result, "state", None), "stage", None)
    if stage_obj is None:
        stage_obj = getattr(
            getattr(adaptive_engine_result, "response_plan", None),
            "stage",
            None,
        )
    if stage_obj is None:
        stage_obj = state.stage
    return _enum_or_string(stage_obj) or SessionStage.TOPIC_DEFINITION.value


def _is_completion_stage(stage_value: str) -> bool:
    """Проверить stage завершения сессии."""
    return stage_value in {
        SessionStage.REFLECTION_SUMMARY.value,
        SessionStage.COMPLETION.value,
    }


def _resolve_completion_stage(stage_value: str) -> str:
    """Вернуть корректную completion-stage для soft-close."""
    if stage_value in {
        SessionStage.REFLECTION_SUMMARY.value,
        SessionStage.COMPLETION.value,
    }:
        return stage_value
    return SessionStage.REFLECTION_SUMMARY.value


def _extract_reply_type(adaptive_engine_result: object | None) -> str:
    """Безопасно извлечь reply_type из adaptive результата."""
    if adaptive_engine_result is None:
        return "coach_message"
    reply_type = getattr(adaptive_engine_result, "reply_type", None)
    extracted = _enum_or_string(reply_type)
    return extracted or "coach_message"


def _extract_next_step_type(adaptive_engine_result: object | None) -> str | None:
    """Безопасно извлечь next step type из adaptive результата."""
    if adaptive_engine_result is None:
        return None
    step_type = getattr(
        getattr(adaptive_engine_result, "response_plan", None),
        "step_type",
        None,
    )
    return _optional_string(step_type)


def _extract_strategy(anti_loop_decision: object | None) -> str | None:
    """Извлечь строковый тип anti-loop стратегии."""
    if anti_loop_decision is None:
        return None
    strategy = getattr(anti_loop_decision, "strategy", None)
    return _optional_string(_enum_or_string(strategy))


def _enum_or_string(value: object | None) -> str | None:
    """Преобразовать enum-like значение к строке."""
    if value is None:
        return None
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, str) and enum_value:
        return enum_value
    if isinstance(value, str) and value:
        return value
    return None


def _optional_string(value: object | None) -> str | None:
    """Вернуть строку или None."""
    if isinstance(value, str) and value:
        return value
    return None


__all__ = [
    "ResponsePlan",
    "plan_response",
]
