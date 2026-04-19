"""Тесты для изолированного safety-layer модуля."""

from src.services.coaching.contracts.enums import SafetyLevel
from src.services.coaching.safety_layer import (
    CRISIS_RESPONSE_TEXT,
    build_crisis_response,
    detect_crisis_markers,
    evaluate_safety,
    should_interrupt_session,
)

CANONICAL_CRISIS_RESPONSE = (
    "Мне очень жаль, что вам сейчас так тяжело. Если есть риск для вашей жизни, "
    "безопасности или безопасности другого человека, я не продолжаю этот разговор "
    "в обычном формате. Сейчас важно как можно скорее обратиться за срочной помощью "
    "в вашей стране: в экстренные службы, к близкому человеку рядом или к "
    "специалисту, который может быть с вами в реальном времени. Когда ситуация "
    "станет безопаснее, вы сможете вернуться сюда позже."
)


def test_safe_reflective_message_returns_safe() -> None:
    """Обычное рефлексивное сообщение должно классифицироваться как SAFE."""
    decision = evaluate_safety("Мне сложно, но я хочу понять свои реакции спокойнее.")

    assert decision.level is SafetyLevel.SAFE
    assert decision.can_continue
    assert not decision.must_switch_to_crisis_protocol
    assert not should_interrupt_session(decision)


def test_self_harm_intent_returns_crisis_interrupt() -> None:
    """Намерение самоповреждения должно давать CRISIS и interrupt."""
    decision = evaluate_safety("Я хочу покончить с собой.")

    assert decision.level is SafetyLevel.CRISIS
    assert should_interrupt_session(decision)
    assert "self_harm_intent" in decision.reason_codes


def test_plan_means_timeframe_returns_crisis_interrupt() -> None:
    """План/срок/средство должны детектироваться как кризис."""
    decision = evaluate_safety(
        "У меня есть план и есть способ сделать это сегодня ночью."
    )

    assert decision.level is SafetyLevel.CRISIS
    assert should_interrupt_session(decision)
    assert "plan_means_timeframe" in decision.reason_codes


def test_negated_know_how_does_not_trigger_crisis() -> None:
    """Фраза "не знаю как" не должна считаться признаком плана/средства."""
    decision = evaluate_safety("Мне тревожно, я не знаю как успокоиться и собраться.")

    assert decision.level is SafetyLevel.SAFE
    assert not should_interrupt_session(decision)
    assert "plan_means_timeframe" not in decision.reason_codes


def test_positive_know_how_still_triggers_crisis() -> None:
    """Явная формулировка "знаю как" должна сохранять кризисный interrupt."""
    decision = evaluate_safety("Я знаю, как это сделать сегодня ночью.")

    assert decision.level is SafetyLevel.CRISIS
    assert should_interrupt_session(decision)
    assert "plan_means_timeframe" in decision.reason_codes


def test_harm_to_others_returns_crisis_interrupt() -> None:
    """Намерение причинить вред другому должно давать crisis-решение."""
    decision = evaluate_safety("Я хочу причинить вред другому человеку.")

    assert decision.level is SafetyLevel.CRISIS
    assert should_interrupt_session(decision)
    assert "harm_to_others" in decision.reason_codes


def test_cannot_guarantee_safety_returns_crisis_interrupt() -> None:
    """Фраза про невозможность гарантировать безопасность должна прерывать flow."""
    decision = evaluate_safety("Я не могу гарантировать безопасность.")

    assert decision.level is SafetyLevel.CRISIS
    assert should_interrupt_session(decision)
    assert "cannot_guarantee_safety" in decision.reason_codes


def test_violence_or_immediate_threat_returns_crisis_interrupt() -> None:
    """Актуальная угроза или насилие должны давать CRISIS и interrupt."""
    decision = evaluate_safety("Мне угрожают прямо сейчас, это актуальное насилие.")

    assert decision.level is SafetyLevel.CRISIS
    assert should_interrupt_session(decision)
    assert "immediate_violence_or_threat" in decision.reason_codes


def test_crisis_response_text_exactly_matches_canonical() -> None:
    """Кризисный ответ должен совпадать с каноническим текстом посимвольно."""
    assert CRISIS_RESPONSE_TEXT == CANONICAL_CRISIS_RESPONSE
    assert build_crisis_response() == CANONICAL_CRISIS_RESPONSE


def test_crisis_response_does_not_contain_forbidden_words() -> None:
    """Канонический кризисный ответ не должен содержать запрещенные слова."""
    response = build_crisis_response().lower()
    forbidden_words = (
        "психолог",
        "психотерапия",
        "терапия",
        "лечение",
        "диагноз",
    )
    for word in forbidden_words:
        assert word not in response


def test_safe_case_does_not_interrupt() -> None:
    """В safe-кейсе interruption не должен включаться."""
    decision = evaluate_safety("Я хочу спокойно разобрать свою ситуацию.")

    assert decision.level is SafetyLevel.SAFE
    assert not should_interrupt_session(decision)


def test_detect_crisis_markers_empty_for_safe_text() -> None:
    """Детектор кризисных маркеров должен возвращать пустой список для safe-текста."""
    markers = detect_crisis_markers("Я замечаю тревогу и хочу мягко замедлиться.")

    assert markers == []
