"""Правила и справочники для AdaptiveSessionEngine."""

from __future__ import annotations

import re
from enum import StrEnum
from hashlib import sha1

from src.services.coaching.contracts.enums import EmotionState, SessionStage

CRISIS_RESPONSE_TEXT = (
    "Мне очень жаль, что вам сейчас так тяжело. "
    "Если есть риск для вашей жизни, безопасности или безопасности другого человека, "
    "я не продолжаю этот разговор в обычном формате. "
    "Сейчас важно как можно скорее обратиться за срочной помощью в вашей стране: "
    "в экстренные службы, к близкому человеку рядом или к специалисту, "
    "который может быть с вами в реальном времени. "
    "Когда ситуация станет безопаснее, вы сможете вернуться сюда позже."
)

CRISIS_MARKERS: tuple[str, ...] = (
    "хочу умереть",
    "покончу с собой",
    "суицид",
    "убью себя",
    "причинить себе вред",
    "навредить другому",
)

ELEVATED_MARKERS: tuple[str, ...] = (
    "невыносимо",
    "не справляюсь",
    "очень тяжело",
    "паника",
)

EMOTION_MARKERS: dict[EmotionState, tuple[str, ...]] = {
    EmotionState.OVERLOAD: ("перегруз", "выгор", "не тяну", "давит"),
    EmotionState.ANXIETY: (
        "тревог",
        "боюсь",
        "страшно",
        "паник",
        "волнуюсь",
        "волнует",
        "переживаю",
        "тревожусь",
    ),
    EmotionState.CONFUSION: ("не понимаю", "запутал", "неясно"),
    EmotionState.SELF_CRITICISM: ("виню себя", "я плох", "неудач"),
    EmotionState.CONFLICT: ("конфликт", "ссора", "спор"),
    EmotionState.DECISION_FEAR: ("не могу решиться", "боюсь выбрать"),
    EmotionState.REPEATING_PATTERN: ("опять", "снова", "по кругу", "повторяется"),
}

CLEAN_LANGUAGE_HINTS: tuple[str, ...] = (
    "как будто",
    "словно",
    "метафора",
    "образ",
)

REQUEST_MARKERS: dict[str, tuple[str, ...]] = {
    "decision": ("решение", "выбрать", "вариант", "что делать"),
    "pattern": ("опять", "снова", "по кругу", "повторяется"),
    "confusion": (
        "не понимаю",
        "не поняла",
        "запутал",
        "неясно",
        "не поняла вопрос",
        "не понимаю вопрос",
        "переформулируйте",
        "объясните иначе",
    ),
    "overload": ("перегруз", "не тяну", "слишком много"),
    "calm": ("размышляю", "осмысляю", "спокойно", "хочу понять себя"),
}

CLARIFICATION_MARKERS: tuple[str, ...] = (
    "не поняла",
    "не понимаю",
    "не поняла вопрос",
    "не понимаю вопрос",
    "переформулируйте",
    "объясните иначе",
)

TOPIC_SWITCH_MARKERS: tuple[str, ...] = (
    "другая тема",
    "новая тема",
    "другая ситуация",
    "еще один вопрос",
)

APPROVE_SWITCH_MARKERS: tuple[str, ...] = (
    "да",
    "перейдем",
    "сменим тему",
    "новый цикл",
)

DECLINE_SWITCH_MARKERS: tuple[str, ...] = ("нет", "не сейчас", "позже")

RESISTANCE_MARKERS: tuple[str, ...] = (
    "не хочу",
    "бесполезно",
    "хватит",
    "раздражает",
)

INSIGHT_MARKERS: tuple[str, ...] = (
    "осознал",
    "понял",
    "заметил",
    "увидел",
    "инсайт",
)

MECHANISM_MARKERS: tuple[str, ...] = ("механизм", "паттерн", "сценарий")
INFLUENCE_MARKERS: tuple[str, ...] = (
    "могу повлиять",
    "в моей зоне",
    "под моим контролем",
)
STATE_SHIFT_MARKERS: tuple[str, ...] = ("стало легче", "спокойнее", "яснее")

LINEAR_STAGE_TRANSITIONS: dict[SessionStage, SessionStage] = {
    SessionStage.TOPIC_DEFINITION: SessionStage.TENSION_REDUCTION,
    SessionStage.TENSION_REDUCTION: SessionStage.MECHANISM_DISCOVERY,
    SessionStage.MECHANISM_DISCOVERY: SessionStage.PATTERN_HIGHLIGHT,
    SessionStage.DEEP_ANALYSIS: SessionStage.INSIGHT,
    SessionStage.INSIGHT: SessionStage.INFLUENCE_ZONE,
    SessionStage.INFLUENCE_ZONE: SessionStage.ACTION_STEP,
    SessionStage.ACTION_STEP: SessionStage.REFLECTION_SUMMARY,
}

TO_REFLECTION_PATH: dict[SessionStage, SessionStage] = {
    SessionStage.TOPIC_DEFINITION: SessionStage.TENSION_REDUCTION,
    SessionStage.TENSION_REDUCTION: SessionStage.MECHANISM_DISCOVERY,
    SessionStage.MECHANISM_DISCOVERY: SessionStage.PATTERN_HIGHLIGHT,
    SessionStage.PATTERN_HIGHLIGHT: SessionStage.REFLECTION_SUMMARY,
    SessionStage.DEEP_ANALYSIS: SessionStage.INSIGHT,
    SessionStage.INSIGHT: SessionStage.INFLUENCE_ZONE,
    SessionStage.INFLUENCE_ZONE: SessionStage.ACTION_STEP,
    SessionStage.ACTION_STEP: SessionStage.REFLECTION_SUMMARY,
    SessionStage.DECISION_NODE: SessionStage.REFLECTION_SUMMARY,
}


class RequestType(StrEnum):
    """Тип пользовательского запроса для маршрутизации."""

    SITUATION = "situation"
    OVERLOAD = "overload"
    CONFUSION = "confusion"
    REPEATING_PATTERN = "repeating_pattern"
    DECISION_CHOICE = "decision_choice"
    CALM_REFLECTION = "calm_reflection"


def normalize_text(text: str) -> str:
    """Привести входной текст к нормализованному виду."""
    return re.sub(r"\s+", " ", text.strip().lower())


def is_contentful(normalized_text: str) -> bool:
    """Содержательное ли сообщение для учета в метриках цикла."""
    return len(normalized_text) >= 5 and len(normalized_text.split()) >= 2


def contains_any(normalized_text: str, markers: tuple[str, ...]) -> bool:
    """Проверить, содержит ли текст хотя бы один маркер."""
    return any(marker in normalized_text for marker in markers)


def build_topic_id(normalized_text: str) -> str:
    """Построить deterministic topic_id из текста."""
    digest = sha1(normalized_text.encode("utf-8"), usedforsecurity=False).hexdigest()
    return f"topic-{digest[:12]}"


def has_progress_signal(normalized_text: str) -> bool:
    """Есть ли в сообщении признаки прогресса/инсайта."""
    markers = INSIGHT_MARKERS + MECHANISM_MARKERS
    return any(marker in normalized_text for marker in markers)


def is_low_engagement(normalized_text: str, contentful: bool) -> bool:
    """Низкая ли вовлеченность на текущем ходу."""
    return (not contentful) or len(normalized_text.split()) <= 2
