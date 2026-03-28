"""Доменная логика психологического перехода в Premium.

Модуль не зависит от transport/UI и не содержит интеграций с Telegram.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from src.services.coaching.contracts.enums import (
    EmotionState,
    SafetyLevel,
    ScenarioType,
    SessionMode,
    SessionStage,
)

if TYPE_CHECKING:
    from src.services.coaching.contracts.session import SessionState


class InsightQuality(StrEnum):
    """Качество инсайта для решения о переходе в Premium."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PremiumReason(StrEnum):
    """Причина, почему переход в Premium может быть релевантен сейчас."""

    REPEATING_PATTERN = "repeating_pattern"
    DEPTH_AVAILABLE = "depth_available"
    USER_WANTS_DEEPER = "user_wants_deeper"
    PATTERN_MEMORY_VALUE = "pattern_memory_value"


@dataclass(slots=True, frozen=True)
class TransitionBridge:
    """Переходный мост перед предложением Premium."""

    recognition: str
    depth_bridge: str


@dataclass(slots=True, frozen=True)
class PremiumDecision:
    """Решение доменной логики по Premium-trigger."""

    should_offer: bool
    should_postpone: bool
    reason: PremiumReason | None
    timing_ok: bool
    psychological_open: bool
    insight_quality: InsightQuality


def evaluate_insight_quality(state: SessionState) -> InsightQuality:
    """Оценить качество инсайта по сигналам SessionState/CompletionCriteria.

    Для MVP используются только уже доступные сигналы:
    - insight_count
    - mechanism_formulated
    - state_shift_confirmed
    - integration_summary_given
    """
    if state.insight_count <= 0:
        return InsightQuality.LOW

    criteria = state.completion_criteria
    quality_signals = 0
    if criteria.mechanism_formulated:
        quality_signals += 1
    if criteria.state_shift_confirmed:
        quality_signals += 1
    if criteria.integration_summary_given:
        quality_signals += 1

    if state.insight_count >= 2 and quality_signals >= 2:
        return InsightQuality.HIGH
    if quality_signals >= 1:
        return InsightQuality.MEDIUM
    return InsightQuality.LOW


def premium_readiness_check(state: SessionState) -> bool:
    """Проверить готовность к Premium по формальным доменным условиям."""
    if state.mode is not SessionMode.FREE:
        return False
    if state.premium_trigger_shown:
        return False
    if state.insight_count < 1:
        return False
    if state.contentful_message_count < 5:
        return False
    if state.safety_decision.level is not SafetyLevel.SAFE:
        return False
    if state.safety_decision.must_block_premium_trigger:
        return False
    if _has_forced_clarity_overload(state):
        return False
    if _has_user_resistance(state):
        return False
    return evaluate_insight_quality(state) is not InsightQuality.LOW


def is_bad_timing(state: SessionState) -> bool:
    """Проверить, является ли момент неподходящим для перехода."""
    overload = state.emotion_result.state is EmotionState.OVERLOAD
    forced_clarity = _has_forced_clarity_overload(state)
    short_answers = _has_short_answer_pattern(state)
    resistance = _has_user_resistance(state)
    return overload or forced_clarity or short_answers or resistance


def is_psychological_moment_open(state: SessionState) -> bool:
    """Проверить, открыт ли пользователь к углублению психологически."""
    if _has_user_resistance(state):
        return False
    if state.emotion_result.state in {EmotionState.OVERLOAD, EmotionState.CONFUSION}:
        return False
    if evaluate_insight_quality(state) is InsightQuality.LOW:
        return False

    open_stages = {
        SessionStage.PATTERN_HIGHLIGHT,
        SessionStage.REFLECTION_SUMMARY,
        SessionStage.INSIGHT,
        SessionStage.INFLUENCE_ZONE,
        SessionStage.ACTION_STEP,
    }
    if state.stage not in open_stages:
        return False
    return state.contentful_message_count >= 5


def determine_premium_reason(state: SessionState) -> PremiumReason:
    """Определить наиболее точную причину перехода в Premium."""
    if _user_wants_deeper(state):
        return PremiumReason.USER_WANTS_DEEPER
    if _pattern_memory_value_relevant(state):
        return PremiumReason.PATTERN_MEMORY_VALUE
    if _repeating_pattern_relevant(state):
        return PremiumReason.REPEATING_PATTERN
    return PremiumReason.DEPTH_AVAILABLE


def understanding_reflection(state: SessionState) -> str:
    """Короткое отражение «вас поняли» перед переходом."""
    if state.stage is SessionStage.PATTERN_HIGHLIGHT:
        return "Похоже, вы точно увидели, где реакция повторяется."
    if state.stage is SessionStage.REFLECTION_SUMMARY:
        return "Похоже, вы уже собрали ключевую логику этой ситуации."
    return "Похоже, вы уже проделали важный шаг в этой теме."


def build_transition_bridge(
    state: SessionState,
    reason: PremiumReason,
) -> TransitionBridge:
    """Собрать мягкий мост из recognition и depth bridge."""
    recognition = understanding_reflection(state)
    depth_bridge_map: dict[PremiumReason, str] = {
        PremiumReason.REPEATING_PATTERN: (
            "Иногда за повтором стоит слой глубже, который полезно разобрать отдельно."
        ),
        PremiumReason.DEPTH_AVAILABLE: (
            "Похоже, сейчас есть окно, чтобы пойти в более глубокий разбор."
        ),
        PremiumReason.USER_WANTS_DEEPER: (
            "Вы уже движетесь в глубину, и это можно продолжить в расширенном формате."
        ),
        PremiumReason.PATTERN_MEMORY_VALUE: (
            "Здесь особенно полезно удерживать паттерны "
            "и возвращаться к прошлым инсайтам."
        ),
    }
    depth_bridge = depth_bridge_map[reason]
    return TransitionBridge(
        recognition=_compact_text(recognition),
        depth_bridge=_compact_text(depth_bridge),
    )


def build_premium_message(state: SessionState, reason: PremiumReason) -> str:
    """Сформировать основной текст перехода в Premium без давления."""
    _ = state
    reason_messages: dict[PremiumReason, str] = {
        PremiumReason.REPEATING_PATTERN: (
            "В расширенной версии можно подробнее разобрать повторяющийся сценарий "
            "и постепенно менять его опорными шагами."
        ),
        PremiumReason.DEPTH_AVAILABLE: (
            "В расширенной версии можно идти глубже по той же теме, "
            "без спешки и с более точной структурой."
        ),
        PremiumReason.USER_WANTS_DEEPER: (
            "В расширенной версии можно продолжить тот же вектор глубины "
            "и удержать логику процесса между сессиями."
        ),
        PremiumReason.PATTERN_MEMORY_VALUE: (
            "В расширенной версии бот запоминает паттерны, "
            "возвращает к прошлым инсайтам и помогает выстраивать устойчивость, "
            "а не разовые решения."
        ),
    }
    return _compact_text(reason_messages[reason])


def evaluate_premium_trigger(state: SessionState) -> PremiumDecision:
    """Главное решение: предложить, отложить или не предлагать Premium."""
    timing_ok = not is_bad_timing(state)
    psychological_open = is_psychological_moment_open(state)
    insight_quality = evaluate_insight_quality(state)
    readiness_ok = premium_readiness_check(state)

    if readiness_ok and timing_ok and psychological_open:
        reason = determine_premium_reason(state)
        return PremiumDecision(
            should_offer=True,
            should_postpone=False,
            reason=reason,
            timing_ok=timing_ok,
            psychological_open=psychological_open,
            insight_quality=insight_quality,
        )

    if readiness_ok and (not timing_ok or not psychological_open):
        postpone_count = _get_postpone_count(state)
        if postpone_count < 1:
            _set_postpone_count(state, postpone_count + 1)
            return PremiumDecision(
                should_offer=False,
                should_postpone=True,
                reason=None,
                timing_ok=timing_ok,
                psychological_open=psychological_open,
                insight_quality=insight_quality,
            )

    return PremiumDecision(
        should_offer=False,
        should_postpone=False,
        reason=None,
        timing_ok=timing_ok,
        psychological_open=psychological_open,
        insight_quality=insight_quality,
    )


def mark_trigger_shown(state: SessionState) -> SessionState:
    """Пометить, что Premium-trigger уже показан в текущем цикле."""
    state.premium_trigger_shown = True
    _set_postpone_count(state, 0)
    return state


def should_offer_premium(state: SessionState) -> bool:
    """Совместимость: вернуть только флаг показа Premium."""
    return evaluate_premium_trigger(state).should_offer


def pre_premium_reflection(
    state: SessionState,
    last_insight: str | None = None,
) -> str | None:
    """Короткая формулировка перед основным Premium-сообщением."""
    decision = evaluate_premium_trigger(state)
    if not decision.should_offer or decision.reason is None:
        return None

    bridge = build_transition_bridge(state, decision.reason)
    insight = _build_insight_prefix(last_insight)
    text = f"{insight}{bridge.recognition} {bridge.depth_bridge}"
    return _compact_text(text)


def build_premium_reflection(
    state: SessionState,
    last_insight: str | None = None,
) -> str | None:
    """Совместимость: собрать мягкий полный переход в Premium."""
    decision = evaluate_premium_trigger(state)
    if not decision.should_offer or decision.reason is None:
        return None

    bridge = build_transition_bridge(state, decision.reason)
    premium_message = build_premium_message(state, decision.reason)
    insight = _build_insight_prefix(last_insight)
    text = f"{insight}{bridge.recognition} {bridge.depth_bridge} {premium_message}"
    return _compact_text(text)


def _has_forced_clarity_overload(state: SessionState) -> bool:
    """Определить forced clarity fallback, вызванный перегрузом."""
    if state.scenario_type is not ScenarioType.CLARITY_FALLBACK:
        return False
    overload = state.emotion_result.state is EmotionState.OVERLOAD
    return overload or state.load_limiter_active


def _has_user_resistance(state: SessionState) -> bool:
    """Проверить наличие выраженного сопротивления в state."""
    return any(
        (
            bool(getattr(state, "has_user_resistance", False)),
            bool(getattr(state, "user_resistance", False)),
            bool(getattr(state, "resistance_detected", False)),
        )
    )


def _has_short_answer_pattern(state: SessionState) -> bool:
    """Выявить паттерн коротких/обрывочных ответов."""
    if state.low_engagement_turns >= 2:
        return True
    if state.question_count < 4:
        return False
    threshold = state.question_count // 2
    return state.contentful_message_count <= threshold


def _user_wants_deeper(state: SessionState) -> bool:
    """Проверить явный запрос пользователя на более глубокую работу."""
    return any(
        (
            bool(getattr(state, "user_wants_deeper", False)),
            bool(getattr(state, "wants_deeper", False)),
            bool(getattr(state, "requested_deeper", False)),
        )
    )


def _pattern_memory_value_relevant(state: SessionState) -> bool:
    """Проверить, важна ли сейчас ценность памяти паттернов."""
    explicit_flag = any(
        (
            bool(getattr(state, "pattern_memory_relevant", False)),
            bool(getattr(state, "needs_pattern_memory", False)),
        )
    )
    if explicit_flag:
        return True

    criteria = state.completion_criteria
    return state.insight_count >= 2 and criteria.mechanism_formulated


def _repeating_pattern_relevant(state: SessionState) -> bool:
    """Проверить, релевантна ли причина повторяющегося паттерна."""
    return (
        state.emotion_result.state is EmotionState.REPEATING_PATTERN
        or state.stage is SessionStage.PATTERN_HIGHLIGHT
    )


def _build_insight_prefix(last_insight: str | None) -> str:
    """Построить мягкий префикс с последним инсайтом, если он передан."""
    if not last_insight:
        return ""
    cleaned = _clean_insight(last_insight)
    if not cleaned:
        return ""
    return f"Вы уже заметили: '{cleaned}'. "


def _clean_insight(last_insight: str) -> str:
    """Очистить и укоротить текст инсайта для безопасной вставки."""
    compact = re.sub(r"\s+", " ", last_insight.strip())
    if not compact:
        return ""
    if len(compact) <= 120:
        return compact
    short = compact[:120].rsplit(" ", 1)[0].rstrip(",;:-")
    if not short:
        short = compact[:120]
    return f"{short}..."


def _compact_text(text: str) -> str:
    """Сделать текст компактным и коротким."""
    compact = re.sub(r"\s+", " ", text.strip())
    if len(compact) <= 420:
        return compact
    short = compact[:420].rsplit(" ", 1)[0].rstrip(",;:-")
    if not short:
        short = compact[:420]
    return f"{short}."


def _get_postpone_count(state: SessionState) -> int:
    """Получить число отложенных решений в текущем цикле."""
    value = state.__dict__.get("premium_postpone_count", 0)
    if not isinstance(value, int):
        return 0
    return max(value, 0)


def _set_postpone_count(state: SessionState, value: int) -> None:
    """Сохранить число отложенных решений в state без изменения контракта."""
    state.__dict__["premium_postpone_count"] = max(value, 0)
