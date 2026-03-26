"""Перечисления и декларативные ограничения для коучинговых сессий."""

from __future__ import annotations

from enum import StrEnum
from typing import Final


class SessionMode(StrEnum):
    """Режим сессии."""

    FREE = "free"
    PREMIUM = "premium"


class SessionStage(StrEnum):
    """Этапы сессии (FSM на уровне домена)."""

    TOPIC_DEFINITION = "topic_definition"
    TENSION_REDUCTION = "tension_reduction"
    MECHANISM_DISCOVERY = "mechanism_discovery"
    PATTERN_HIGHLIGHT = "pattern_highlight"
    DEEP_ANALYSIS = "deep_analysis"
    INSIGHT = "insight"
    INFLUENCE_ZONE = "influence_zone"
    ACTION_STEP = "action_step"
    DECISION_NODE = "decision_node"
    REFLECTION_SUMMARY = "reflection_summary"
    COMPLETION = "completion"
    CRISIS = "crisis"


class ScenarioType(StrEnum):
    """Тип сценария внутри сессии."""

    A_STRUCTURED = "a_structured"
    B_CLEAN_LANGUAGE = "b_clean_language"
    C_SECONDARY_GAIN = "c_secondary_gain"
    CLARITY_FALLBACK = "clarity_fallback"


class SafetyLevel(StrEnum):
    """Уровень риска для safety-решения."""

    SAFE = "safe"
    ELEVATED = "elevated"
    CRISIS = "crisis"


class EmotionState(StrEnum):
    """Распознаваемые эмоциональные состояния."""

    OVERLOAD = "overload"
    ANXIETY = "anxiety"
    CONFUSION = "confusion"
    SELF_CRITICISM = "self_criticism"
    CONFLICT = "conflict"
    DECISION_FEAR = "decision_fear"
    REPEATING_PATTERN = "repeating_pattern"
    UNKNOWN = "unknown"


# Декларативные ограничения сценариев для v1.
FREE_PRIMARY_SCENARIO: Final[ScenarioType] = ScenarioType.A_STRUCTURED

FREE_ALLOWED_SCENARIOS: Final[frozenset[ScenarioType]] = frozenset(
    {
        ScenarioType.A_STRUCTURED,
        ScenarioType.CLARITY_FALLBACK,
    }
)

PREMIUM_ALLOWED_SCENARIOS: Final[frozenset[ScenarioType]] = frozenset(
    {
        ScenarioType.A_STRUCTURED,
        ScenarioType.B_CLEAN_LANGUAGE,
        ScenarioType.CLARITY_FALLBACK,
    }
)

# Зарезервировано для будущих релизов, в v1 не активируется.
DISABLED_V1_SCENARIOS: Final[frozenset[ScenarioType]] = frozenset(
    {
        ScenarioType.C_SECONDARY_GAIN,
    }
)
