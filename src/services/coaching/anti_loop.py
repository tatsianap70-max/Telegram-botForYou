"""Публичный facade для anti-loop policy-слоя.

Модуль сохраняет прежний API и переэкспортирует функции/типы,
а внутренняя логика вынесена в отдельные файлы.
"""

from src.services.coaching.anti_loop_policy import (
    detect_loop_signals,
    evaluate_anti_loop,
    select_recovery_strategy,
    should_soft_close,
)
from src.services.coaching.anti_loop_types import (
    MAX_REPEAT_WITHOUT_PROGRESS,
    AntiLoopDecision,
    AntiLoopStrategy,
    LoopSignals,
    RecoveryStrategy,
    SignalStrength,
)

__all__ = [
    "MAX_REPEAT_WITHOUT_PROGRESS",
    "AntiLoopDecision",
    "AntiLoopStrategy",
    "LoopSignals",
    "RecoveryStrategy",
    "SignalStrength",
    "detect_loop_signals",
    "evaluate_anti_loop",
    "select_recovery_strategy",
    "should_soft_close",
]
