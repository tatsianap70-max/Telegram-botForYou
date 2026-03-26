"""Тесты для ScenarioSelector."""

from src.services.coaching.contracts.emotion import EmotionDetectionResult
from src.services.coaching.contracts.enums import ScenarioType, SessionMode
from src.services.coaching.scenario_selector import ScenarioSelector


def test_free_mode_rejects_clean_language_scenario() -> None:
    """В Free нельзя выбрать B_CLEAN_LANGUAGE."""
    selector = ScenarioSelector(max_clarification_attempts=2)
    emotion = EmotionDetectionResult(
        fallback_to_clarity=False,
        clarification_attempts=0,
        is_confident=True,
        confidence=0.9,
    )

    selected = selector.select(
        mode=SessionMode.FREE,
        requested=ScenarioType.B_CLEAN_LANGUAGE,
        emotion_result=emotion,
    )

    assert selected == ScenarioType.A_STRUCTURED


def test_premium_mode_accepts_clean_language_scenario() -> None:
    """В Premium сценарий B доступен."""
    selector = ScenarioSelector(max_clarification_attempts=2)
    emotion = EmotionDetectionResult(
        fallback_to_clarity=False,
        clarification_attempts=0,
        is_confident=True,
        confidence=0.8,
    )

    selected = selector.select(
        mode=SessionMode.PREMIUM,
        requested=ScenarioType.B_CLEAN_LANGUAGE,
        emotion_result=emotion,
    )

    assert selected == ScenarioType.B_CLEAN_LANGUAGE


def test_disabled_secondary_gain_falls_back_to_default() -> None:
    """Отключённый в v1 сценарий C откатывается к сценарию по умолчанию."""
    selector = ScenarioSelector(max_clarification_attempts=2)
    emotion = EmotionDetectionResult(
        fallback_to_clarity=False,
        clarification_attempts=0,
        is_confident=True,
        confidence=0.8,
    )

    selected = selector.select(
        mode=SessionMode.PREMIUM,
        requested=ScenarioType.C_SECONDARY_GAIN,
        emotion_result=emotion,
    )

    assert selected == ScenarioType.A_STRUCTURED


def test_forces_clarity_fallback_when_emotion_marks_fallback() -> None:
    """Явный fallback флаг переводит сценарий в CLARITY_FALLBACK."""
    selector = ScenarioSelector(max_clarification_attempts=2)
    emotion = EmotionDetectionResult(
        fallback_to_clarity=True,
        clarification_attempts=1,
        is_confident=False,
        confidence=0.3,
    )

    selected = selector.select(
        mode=SessionMode.PREMIUM,
        requested=ScenarioType.B_CLEAN_LANGUAGE,
        emotion_result=emotion,
    )

    assert selected == ScenarioType.CLARITY_FALLBACK


def test_forces_clarity_fallback_after_two_clarifications() -> None:
    """После двух уточнений включается принудительный fallback."""
    selector = ScenarioSelector(max_clarification_attempts=2)
    emotion = EmotionDetectionResult(
        fallback_to_clarity=False,
        clarification_attempts=2,
        is_confident=False,
        confidence=0.4,
    )

    selected = selector.select(
        mode=SessionMode.PREMIUM,
        requested=ScenarioType.A_STRUCTURED,
        emotion_result=emotion,
    )

    assert selected == ScenarioType.CLARITY_FALLBACK
