"""Контракт результата распознавания эмоционального состояния."""

from pydantic import BaseModel, Field

from .enums import EmotionState


class EmotionDetectionResult(BaseModel):
    """Результат распознавания эмоции для маршрутизации сценария."""

    state: EmotionState = EmotionState.UNKNOWN
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    is_confident: bool = False
    fallback_to_clarity: bool = True
    clarification_attempts: int = Field(default=0, ge=0)
    markers: list[str] = Field(default_factory=list)
