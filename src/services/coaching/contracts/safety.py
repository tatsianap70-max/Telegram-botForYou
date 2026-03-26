"""Контракт решения safety-слоя."""

from pydantic import BaseModel, Field

from .enums import SafetyLevel


class SafetyDecision(BaseModel):
    """Решение safety-слоя для текущего шага диалога."""

    level: SafetyLevel = SafetyLevel.SAFE
    can_continue: bool = True
    must_switch_to_crisis_protocol: bool = False
    must_block_premium_trigger: bool = False
    reason_codes: list[str] = Field(default_factory=list)
    message_template_key: str | None = None
