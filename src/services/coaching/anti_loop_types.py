"""Типы и структуры решения для anti-loop policy-слоя."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

MAX_REPEAT_WITHOUT_PROGRESS = 3


class SignalStrength(StrEnum):
    """Сила anti-loop сигнала."""

    NONE = "none"
    WEAK = "weak"
    CLEAR = "clear"


class RecoveryStrategy(StrEnum):
    """Рекомендованный тип anti-loop сдвига."""

    NONE = "none"
    ANGLE_SHIFT = "angle_shift"
    SIMPLIFY = "simplify"
    META_QUESTION = "meta_question"
    INSIGHT_TRIGGER = "insight_trigger"
    SOFT_CLOSE = "soft_close"


# Совместимость с предыдущим именованием внутри модуля.
AntiLoopStrategy = RecoveryStrategy


@dataclass(slots=True, frozen=True)
class LoopSignals:
    """Сырые policy-сигналы для anti-loop решения."""

    loop_detected: bool
    stagnation_detected: bool
    repeated_user_angle: bool
    repeated_system_angle: bool
    short_repetitive_answers: bool
    no_new_information: bool
    has_new_fact: bool
    has_new_insight: bool
    has_new_step: bool
    no_progress: bool
    signal_strength: SignalStrength
    reason: str
    evidence_count: int


@dataclass(slots=True, frozen=True)
class _SignalFlags:
    """Внутренний контейнер флагов для вычисления силы сигнала и причины."""

    loop_detected: bool
    stagnation_detected: bool
    repeated_user_angle: bool
    repeated_system_angle: bool
    short_repetitive_answers: bool
    no_new_information: bool
    no_progress: bool


@dataclass(slots=True, frozen=True)
class AntiLoopDecision:
    """Финальное решение anti-loop слоя для текущего хода."""

    loop_detected: bool
    signal_strength: SignalStrength
    strategy: RecoveryStrategy
    should_soft_close: bool
    reason: str
    stagnation_detected: bool
    requires_intervention: bool


__all__ = [
    "MAX_REPEAT_WITHOUT_PROGRESS",
    "AntiLoopDecision",
    "AntiLoopStrategy",
    "LoopSignals",
    "RecoveryStrategy",
    "SignalStrength",
    "_SignalFlags",
]

