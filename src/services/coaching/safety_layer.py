"""Изолированный policy-модуль безопасности для коучингового домена.

Safety-layer работает как interrupt-first слой:
- детектирует кризисные маркеры в тексте;
- возвращает структурированное решение SafetyDecision;
- при кризисе останавливает обычный flow на policy-уровне.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from src.services.coaching.contracts.enums import SafetyLevel
from src.services.coaching.contracts.safety import SafetyDecision

if TYPE_CHECKING:
    from src.services.coaching.contracts.session import SessionState


CRISIS_RESPONSE_TEXT = (
    "Мне очень жаль, что вам сейчас так тяжело. Если есть риск для вашей жизни, "
    "безопасности или безопасности другого человека, я не продолжаю этот разговор "
    "в обычном формате. Сейчас важно как можно скорее обратиться за срочной помощью "
    "в вашей стране: в экстренные службы, к близкому человеку рядом или к "
    "специалисту, который может быть с вами в реальном времени. Когда ситуация "
    "станет безопаснее, вы сможете вернуться сюда позже."
)

_CRISIS_MARKERS: dict[str, tuple[str, ...]] = {
    "self_harm_intent": (
        "хочу умереть",
        "покончить с собой",
        "убью себя",
        "причинить себе вред",
        "навредить себе",
    ),
    "plan_means_timeframe": (
        "есть план",
        "есть способ",
        "знаю как",
        "сегодня ночью",
        "в ближайший час",
    ),
    "harm_to_others": (
        "причинить вред другому",
        "навредить другому",
        "убью его",
        "убью ее",
        "напасть на",
    ),
    "cannot_guarantee_safety": (
        "не могу гарантировать безопасность",
        "не могу обещать безопасность",
        "не могу гарантировать что буду в безопасности",
    ),
    "disorientation_with_risk": (
        "дезориентирован",
        "не понимаю где я",
        "не понимаю что происходит и мне опасно",
    ),
    "immediate_violence_or_threat": (
        "мне угрожают прямо сейчас",
        "меня сейчас бьют",
        "сейчас есть угроза",
        "актуальное насилие",
    ),
}

_KNOW_HOW_PATTERN = re.compile(r"\b(?:не\s+)?знаю\s*,?\s*как\b")


def detect_crisis_markers(user_text: str) -> list[str]:
    """Определить сработавшие кризисные reason-коды по тексту пользователя."""
    normalized = _normalize_text(user_text)
    if not normalized:
        return []

    reason_codes: list[str] = []
    for reason_code, markers in _CRISIS_MARKERS.items():
        if any(
            _contains_crisis_marker(
                normalized_text=normalized,
                reason_code=reason_code,
                marker=marker,
            )
            for marker in markers
        ):
            reason_codes.append(reason_code)
    return reason_codes


def evaluate_safety(
    user_text: str,
    *,
    state: SessionState | None = None,
) -> SafetyDecision:
    """Оценить безопасность сообщения и вернуть доменное safety-решение.

    Параметр state оставлен как опциональный контекст для будущей интеграции.
    В текущей версии решение детерминированно строится только по user_text.
    """
    _ = state
    reason_codes = detect_crisis_markers(user_text)
    if not reason_codes:
        return SafetyDecision(
            level=SafetyLevel.SAFE,
            can_continue=True,
            must_switch_to_crisis_protocol=False,
            must_block_premium_trigger=False,
            reason_codes=[],
            message_template_key=None,
        )

    return SafetyDecision(
        level=SafetyLevel.CRISIS,
        can_continue=False,
        must_switch_to_crisis_protocol=True,
        must_block_premium_trigger=True,
        reason_codes=reason_codes,
        message_template_key="crisis_support_v1",
    )


def should_interrupt_session(decision: SafetyDecision) -> bool:
    """Проверить, должен ли safety-layer прервать обычный сценарий сессии."""
    return (
        decision.level is SafetyLevel.CRISIS
        or decision.must_switch_to_crisis_protocol
        or not decision.can_continue
    )


def build_crisis_response() -> str:
    """Вернуть канонический кризисный ответ без адаптаций."""
    return CRISIS_RESPONSE_TEXT


def _normalize_text(text: str) -> str:
    """Нормализовать входной текст для policy-детекции маркеров."""
    return re.sub(r"\s+", " ", text.strip().lower())


def _contains_crisis_marker(
    normalized_text: str,
    reason_code: str,
    marker: str,
) -> bool:
    """Проверить маркер с точечной защитой от ложных срабатываний."""
    if reason_code == "plan_means_timeframe" and marker == "знаю как":
        return _contains_positive_know_how(normalized_text)
    return marker in normalized_text


def _contains_positive_know_how(normalized_text: str) -> bool:
    """Распознать "знаю как", игнорируя форму "не знаю как"."""
    for match in _KNOW_HOW_PATTERN.finditer(normalized_text):
        if not match.group(0).startswith("не "):
            return True
    return False
