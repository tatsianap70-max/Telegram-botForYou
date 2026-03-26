"""Сервисы коучингового домена."""

from src.services.coaching.scenario_selector import ScenarioSelector
from src.services.coaching.session_manager import (
    DeepSessionLimitError,
    InvalidStageTransitionError,
    SessionCompletionError,
    SessionManager,
    SessionManagerError,
    TopicSwitchPolicyError,
)

__all__ = [
    "DeepSessionLimitError",
    "InvalidStageTransitionError",
    "ScenarioSelector",
    "SessionCompletionError",
    "SessionManager",
    "SessionManagerError",
    "TopicSwitchPolicyError",
]
