"""Контракт критериев завершения сессии."""

from pydantic import BaseModel


class CompletionCriteria(BaseModel):
    """Критерии завершения сессии."""

    mechanism_formulated: bool = False
    influence_zone_defined: bool = False
    state_shift_confirmed: bool = False
    integration_summary_given: bool = False
    is_completed: bool = False
