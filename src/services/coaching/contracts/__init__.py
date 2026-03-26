"""Контракты коучингового домена."""

from src.services.coaching.contracts.completion import CompletionCriteria
from src.services.coaching.contracts.emotion import EmotionDetectionResult
from src.services.coaching.contracts.enums import (
    DISABLED_V1_SCENARIOS,
    FREE_ALLOWED_SCENARIOS,
    FREE_PRIMARY_SCENARIO,
    PREMIUM_ALLOWED_SCENARIOS,
    EmotionState,
    SafetyLevel,
    ScenarioType,
    SessionMode,
    SessionStage,
)
from src.services.coaching.contracts.safety import SafetyDecision
from src.services.coaching.contracts.session import SessionState

__all__ = [
    "DISABLED_V1_SCENARIOS",
    "FREE_ALLOWED_SCENARIOS",
    "FREE_PRIMARY_SCENARIO",
    "PREMIUM_ALLOWED_SCENARIOS",
    "CompletionCriteria",
    "EmotionDetectionResult",
    "EmotionState",
    "SafetyDecision",
    "SafetyLevel",
    "ScenarioType",
    "SessionMode",
    "SessionStage",
    "SessionState",
]
