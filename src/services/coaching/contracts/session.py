"""Контракт состояния коучинговой сессии."""

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from .completion import CompletionCriteria
from .emotion import EmotionDetectionResult
from .enums import ScenarioType, SessionMode, SessionStage
from .safety import SafetyDecision


class SessionState(BaseModel):
    """Состояние одной сессии (одного цикла по одной теме)."""

    session_id: str
    cycle_id: str
    user_id: int

    mode: SessionMode
    stage: SessionStage = SessionStage.TOPIC_DEFINITION
    scenario_type: ScenarioType = ScenarioType.A_STRUCTURED

    # Правило: одна тема = один цикл.
    topic_id: str | None = None
    pending_topic_id: str | None = None
    decision_node_required_for_topic_switch: bool = True

    question_count: int = 0
    contentful_message_count: int = 0
    insight_count: int = 0
    premium_trigger_shown: bool = False
    recent_step_types: list[str] = Field(default_factory=list)
    no_progress_turns: int = 0
    low_engagement_turns: int = 0
    load_limiter_active: bool = False

    # Лимит глубокой premium-сессии: максимум одна в день.
    deep_session_allowed_today: bool = True

    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    timezone: str = "Europe/Oslo"

    safety_decision: SafetyDecision = Field(default_factory=SafetyDecision)
    emotion_result: EmotionDetectionResult = Field(
        default_factory=EmotionDetectionResult
    )
    completion_criteria: CompletionCriteria = Field(default_factory=CompletionCriteria)
