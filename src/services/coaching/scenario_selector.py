"""Селектор сценариев коучинговой сессии.

Модуль отвечает только за выбор безопасного сценария в рамках policy v1:
- Free: A_STRUCTURED + CLARITY_FALLBACK
- Premium: A_STRUCTURED + B_CLEAN_LANGUAGE + CLARITY_FALLBACK
- C_SECONDARY_GAIN: зарезервирован, но отключён в v1
"""

from src.services.coaching.contracts.emotion import EmotionDetectionResult
from src.services.coaching.contracts.enums import (
    DISABLED_V1_SCENARIOS,
    FREE_ALLOWED_SCENARIOS,
    FREE_PRIMARY_SCENARIO,
    PREMIUM_ALLOWED_SCENARIOS,
    ScenarioType,
    SessionMode,
)


class ScenarioSelector:
    """Выбирает сценарий с учётом режима и fallback-правил."""

    def __init__(self, max_clarification_attempts: int = 2) -> None:
        """Создать селектор сценариев.

        Args:
            max_clarification_attempts: Максимум попыток уточнения эмоции
                перед принудительным переходом в clarity fallback.
        """
        if max_clarification_attempts < 0:
            msg = "Параметр max_clarification_attempts не может быть отрицательным."
            raise ValueError(msg)
        self._max_clarification_attempts = max_clarification_attempts

    def select_default(self, mode: SessionMode) -> ScenarioType:
        """Получить сценарий по умолчанию для режима сессии."""
        # В v1 базовый сценарий одинаковый для Free и Premium.
        _ = mode
        return FREE_PRIMARY_SCENARIO

    def is_allowed(self, mode: SessionMode, scenario: ScenarioType) -> bool:
        """Проверить, разрешён ли сценарий в текущем режиме."""
        return scenario in self._allowed_scenarios(mode)

    def select(
        self,
        mode: SessionMode,
        requested: ScenarioType,
        emotion_result: EmotionDetectionResult,
    ) -> ScenarioType:
        """Выбрать безопасный сценарий для текущего шага.

        Приоритет правил:
        1. Если требуется fallback по эмоции -> CLARITY_FALLBACK.
        2. Если сценарий отключён в v1 -> сценарий по умолчанию.
        3. Если сценарий разрешён в режиме -> requested.
        4. Иначе -> сценарий по умолчанию.
        """
        if self._must_force_clarity_fallback(emotion_result):
            return ScenarioType.CLARITY_FALLBACK

        if requested in DISABLED_V1_SCENARIOS:
            return self.select_default(mode)

        if self.is_allowed(mode, requested):
            return requested

        return self.select_default(mode)

    def _allowed_scenarios(self, mode: SessionMode) -> frozenset[ScenarioType]:
        if mode is SessionMode.FREE:
            return FREE_ALLOWED_SCENARIOS
        return PREMIUM_ALLOWED_SCENARIOS

    def _must_force_clarity_fallback(
        self, emotion_result: EmotionDetectionResult
    ) -> bool:
        return emotion_result.fallback_to_clarity or (
            emotion_result.clarification_attempts >= self._max_clarification_attempts
        )
