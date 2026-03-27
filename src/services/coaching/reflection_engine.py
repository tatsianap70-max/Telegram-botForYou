"""Доменный движок рефлексивных формулировок.

Модуль не зависит от Telegram/handler-flow и работает только
с доменным контекстом SessionState.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import TYPE_CHECKING

from src.services.coaching.contracts.enums import (
    EmotionState,
    ScenarioType,
    SessionMode,
    SessionStage,
)

if TYPE_CHECKING:
    from src.services.coaching.contracts.session import SessionState


class ReflectionDepth(StrEnum):
    """Глубина отражения."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ReflectionTone(StrEnum):
    """Тон формулировки."""

    NEUTRAL = "neutral"
    SUPPORTIVE = "supportive"
    SLOWING = "slowing"


_BANNED_WORDS: tuple[str, ...] = (
    "психолог",
    "психотерапия",
    "лечение",
    "диагноз",
)

_RESISTANCE_MARKERS: tuple[str, ...] = (
    "не хочу",
    "бесполезно",
    "хватит",
    "раздражает",
    "устал",
    "надоело",
)

_STOP_WORDS: frozenset[str] = frozenset(
    {
        "и",
        "в",
        "на",
        "но",
        "а",
        "это",
        "как",
        "что",
        "мне",
        "меня",
        "очень",
        "просто",
        "сейчас",
        "уже",
        "все",
        "всё",
    }
)


class ReflectionEngine:
    """Генерирует короткие спокойные отражения для коучингового домена."""

    def __init__(self, max_chars: int = 220) -> None:
        if max_chars < 80:
            msg = "Параметр max_chars должен быть >= 80."
            raise ValueError(msg)
        self._max_chars = max_chars

    def mirror_user_language(self, state: SessionState, user_text: str) -> str:
        """Зеркалит смысл фразы, используя отдельные слова пользователя."""
        _ = state
        keywords = self._extract_keywords(user_text, limit=3)
        if not keywords:
            return self._apply_style_guardrails("Похоже, это сейчас для вас важно.")
        if len(keywords) == 1:
            text = f"Похоже, здесь много про «{keywords[0]}»."
        else:
            text = f"Похоже, здесь много про «{keywords[0]}» и «{keywords[1]}»."
        return self._apply_style_guardrails(text)

    def short_reflection(self, state: SessionState, user_text: str) -> str:
        """Возвращает короткое отражение текущего смысла."""
        depth = self._resolve_depth(state, user_text)
        tone = self._resolve_tone(state)
        resistance = self._has_resistance(user_text)
        if resistance:
            text = "Похоже, сейчас лучше держаться одной ясной мысли."
            return self._apply_style_guardrails(text)

        mirrored = self.mirror_user_language(state, user_text)
        if tone is ReflectionTone.SLOWING:
            text = f"{mirrored} Важно двигаться короткими шагами."
            return self._apply_style_guardrails(text)
        if tone is ReflectionTone.SUPPORTIVE:
            text = f"{mirrored} Возможно, здесь уже появляется ясность."
            return self._apply_style_guardrails(text)
        if depth is ReflectionDepth.HIGH:
            text = f"{mirrored} Иногда в этом виден более глубокий слой."
            return self._apply_style_guardrails(text)
        return self._apply_style_guardrails(mirrored)

    def insight_hypothesis(
        self,
        state: SessionState,
        user_text: str | None = None,
    ) -> str:
        """Формирует мягкую гипотезу без категоричных выводов."""
        text = user_text or ""
        depth = self._resolve_depth(state, text)
        prefix = self._hedge_prefix(depth)

        if self._has_resistance(text):
            hypothesis = (
                f"{prefix}, здесь есть важный узел, который лучше разбирать постепенно."
            )
            return self._apply_style_guardrails(hypothesis)

        if state.stage in {
            SessionStage.PATTERN_HIGHLIGHT,
            SessionStage.MECHANISM_DISCOVERY,
        }:
            base = "это может повторяться в похожих ситуациях"
        elif state.stage in {SessionStage.DEEP_ANALYSIS, SessionStage.INSIGHT}:
            base = (
                "здесь может быть внутренний механизм, который раньше оставался в фоне"
            )
        elif state.stage == SessionStage.DECISION_NODE:
            base = "в этом месте могут сталкиваться два важных приоритета"
        else:
            base = "в этой реакции может быть устойчивый способ защиты"

        return self._apply_style_guardrails(f"{prefix}, {base}.")

    def next_question_prompt(
        self,
        state: SessionState,
        user_text: str | None = None,
    ) -> str:
        """Возвращает следующий вопрос по стадии и эмоциональному контексту."""
        text = (user_text or "").strip().lower()
        if self._has_resistance(text):
            prompt = "Какой один вопрос стоит оставить главным сейчас?"
            return self._apply_style_guardrails(prompt)

        prompts: dict[SessionStage, str] = {
            SessionStage.TOPIC_DEFINITION: "Что здесь для вас сейчас самое важное?",
            SessionStage.TENSION_REDUCTION: (
                "Что в этом факт, а что ваша интерпретация?"
            ),
            SessionStage.MECHANISM_DISCOVERY: (
                "Что запускается первым: мысль, эмоция или реакция?"
            ),
            SessionStage.PATTERN_HIGHLIGHT: "Где этот сценарий уже повторялся?",
            SessionStage.DEEP_ANALYSIS: "Что в этом ощущается самым точным образом?",
            SessionStage.INSIGHT: "Что вы заметили нового про себя в этой точке?",
            SessionStage.INFLUENCE_ZONE: "На что вы реально можете повлиять сегодня?",
            SessionStage.ACTION_STEP: "Какой самый маленький шаг вы готовы сделать?",
            SessionStage.REFLECTION_SUMMARY: "Что из этого разговора важно сохранить?",
            SessionStage.COMPLETION: "Что вы хотите унести из этой сессии дальше?",
            SessionStage.DECISION_NODE: (
                "Остаемся в этой теме или открываем новый цикл?"
            ),
            SessionStage.CRISIS: (
                "Кто может быть рядом с вами прямо сейчас в реальном времени?"
            ),
        }
        return self._apply_style_guardrails(prompts[state.stage])

    def reflection_summary(
        self,
        state: SessionState,
        user_text: str | None = None,
    ) -> str:
        """Собирает краткий итог шага без советов."""
        _ = user_text
        criteria = state.completion_criteria
        parts: list[str] = []

        if criteria.mechanism_formulated:
            parts.append("механизм стал понятнее")
        if criteria.influence_zone_defined:
            parts.append("зона влияния прояснилась")
        if criteria.state_shift_confirmed:
            parts.append("состояние немного сдвинулось")

        if not parts:
            text = "Похоже, вы уже выделили главное в этой теме."
            return self._apply_style_guardrails(text)
        if len(parts) == 1:
            return self._apply_style_guardrails(f"Похоже, {parts[0]}.")
        return self._apply_style_guardrails(f"Похоже, {parts[0]}, и {parts[1]}.")

    def soft_landing(
        self,
        state: SessionState,
        user_text: str | None = None,
    ) -> str:
        """Мягко завершает ход, если сильного инсайта нет."""
        text = (user_text or "").strip().lower()
        if self._has_resistance(text) or state.load_limiter_active:
            result = "Похоже, на сегодня достаточно. Можно оставить один ясный вывод."
            return self._apply_style_guardrails(result)
        result = "Похоже, на этом шаге уже есть опора. К теме можно вернуться позже."
        return self._apply_style_guardrails(result)

    def exit_reflection_question(
        self,
        state: SessionState,
        user_text: str | None = None,
    ) -> str:
        """Возвращает финальный вопрос после сессии."""
        text = (user_text or "").strip().lower()
        if self._has_resistance(text):
            prompt = (
                "Какой самый маленький шаг вы готовы сохранить после этого разговора?"
            )
            return self._apply_style_guardrails(prompt)
        if state.completion_criteria.is_completed:
            prompt = "Что из сегодняшнего вам важно удержать в ближайшие сутки?"
            return self._apply_style_guardrails(prompt)
        prompt = "Что сейчас выглядит для вас следующим спокойным шагом?"
        return self._apply_style_guardrails(prompt)

    def progress_reflection(
        self,
        state: SessionState,
        user_text: str | None = None,
    ) -> str:
        """Отражает ощущение движения и накопленного прогресса."""
        text = (user_text or "").strip().lower()
        if self._has_resistance(text):
            result = "Похоже, движение уже есть, даже если шаг пока маленький."
            return self._apply_style_guardrails(result)
        if state.insight_count >= 2:
            result = "Похоже, у вас уже формируется устойчивое понимание."
            return self._apply_style_guardrails(result)
        if state.insight_count >= 1:
            result = "Похоже, появился заметный сдвиг в понимании."
            return self._apply_style_guardrails(result)
        if state.contentful_message_count >= 3:
            result = "Похоже, ясность постепенно растет от шага к шагу."
            return self._apply_style_guardrails(result)
        result = "Иногда движение начинается с одного точного уточнения."
        return self._apply_style_guardrails(result)

    def pre_premium_reflection(
        self,
        state: SessionState,
        user_text: str | None = None,
    ) -> str | None:
        """Возвращает pre-premium текст только при строгом наборе условий."""
        _ = user_text
        has_overload = (
            state.load_limiter_active
            or state.emotion_result.state == EmotionState.OVERLOAD
        )
        can_show = (
            state.mode == SessionMode.FREE
            and not state.premium_trigger_shown
            and state.insight_count >= 1
            and state.contentful_message_count >= 5
            and not state.safety_decision.must_block_premium_trigger
            and not has_overload
        )
        if not can_show:
            return None
        text = "Похоже, здесь есть слой глубже. Иногда его полезно разобрать отдельно."
        return self._apply_style_guardrails(text)

    def _resolve_depth(self, state: SessionState, user_text: str) -> ReflectionDepth:
        """Определяет глубину отражения по контексту сессии."""
        resistance = self._has_resistance(user_text)
        emotion = state.emotion_result
        forced_low = (
            resistance
            or state.load_limiter_active
            or not emotion.is_confident
            or emotion.fallback_to_clarity
            or emotion.state in {EmotionState.OVERLOAD, EmotionState.CONFUSION}
        )
        if forced_low:
            return ReflectionDepth.LOW

        if (
            state.mode == SessionMode.PREMIUM
            and state.scenario_type == ScenarioType.B_CLEAN_LANGUAGE
            and state.stage in {SessionStage.DEEP_ANALYSIS, SessionStage.INSIGHT}
            and emotion.confidence >= 0.7
        ):
            return ReflectionDepth.HIGH
        return ReflectionDepth.MEDIUM

    @staticmethod
    def _resolve_tone(state: SessionState) -> ReflectionTone:
        """Выбирает тон на основе эмоции и нагрузки."""
        emotion = state.emotion_result.state
        slowing = state.load_limiter_active or emotion in {
            EmotionState.OVERLOAD,
            EmotionState.CONFUSION,
        }
        if slowing:
            return ReflectionTone.SLOWING

        supportive = emotion in {
            EmotionState.ANXIETY,
            EmotionState.SELF_CRITICISM,
            EmotionState.CONFLICT,
        }
        if supportive:
            return ReflectionTone.SUPPORTIVE
        return ReflectionTone.NEUTRAL

    @staticmethod
    def _hedge_prefix(depth: ReflectionDepth) -> str:
        """Возвращает мягкую вводную для гипотезы."""
        if depth is ReflectionDepth.HIGH:
            return "иногда"
        if depth is ReflectionDepth.MEDIUM:
            return "похоже"
        return "возможно"

    @staticmethod
    def _has_resistance(text: str) -> bool:
        """Определяет маркеры сопротивления/раздражения."""
        normalized = text.strip().lower()
        if not normalized:
            return False
        return any(marker in normalized for marker in _RESISTANCE_MARKERS)

    @staticmethod
    def _extract_keywords(user_text: str, limit: int = 3) -> list[str]:
        """Выделяет несколько смысловых слов пользователя."""
        words = re.findall(r"[A-Za-zА-Яа-яЁё0-9-]+", user_text.lower())
        result: list[str] = []
        for word in words:
            if len(word) <= 2 or word in _STOP_WORDS:
                continue
            if word not in result:
                result.append(word)
            if len(result) >= limit:
                break
        return result

    def _apply_style_guardrails(self, text: str) -> str:
        """Ограничивает длину и очищает формулировку от перегруза."""
        normalized = re.sub(r"\s+", " ", text.strip())
        if not normalized:
            return "Похоже, это важная точка."

        normalized = self._remove_banned_words(normalized)
        normalized = self._remove_cliches(normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip(" ,.;")
        if not normalized:
            return "Похоже, это важная точка."

        compact = self._limit_sentences(normalized)
        compact = self._limit_commas(compact)
        compact = self._truncate(compact)
        return compact or "Похоже, это важная точка."

    @staticmethod
    def _remove_banned_words(text: str) -> str:
        """Удаляет запрещенные слова без замены."""
        cleaned = text
        for word in _BANNED_WORDS:
            cleaned = re.sub(
                rf"\b{re.escape(word)}\b",
                "",
                cleaned,
                flags=re.IGNORECASE,
            )
        return cleaned

    @staticmethod
    def _remove_cliches(text: str) -> str:
        """Удаляет шаблонные клише, создающие «ИИ-стиль»."""
        clichés = (
            "я здесь, чтобы",
            "давайте попробуем",
            "как искусственный интеллект",
            "в конечном итоге",
        )
        cleaned = text
        lowered = cleaned.lower()
        for cliché in clichés:
            if cliché in lowered:
                idx = lowered.find(cliché)
                cleaned = cleaned[:idx] + cleaned[idx + len(cliché) :]
                lowered = cleaned.lower()
        return cleaned

    @staticmethod
    def _limit_sentences(text: str) -> str:
        """Оставляет максимум два предложения."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        compact = [sentence.strip() for sentence in sentences if sentence.strip()]
        return " ".join(compact[:2])

    @staticmethod
    def _limit_commas(text: str) -> str:
        """Убирает перегруженные конструкции с большим числом запятых."""
        if text.count(",") <= 2:
            return text
        parts = [part.strip() for part in text.split(",") if part.strip()]
        reduced = ", ".join(parts[:2])
        if reduced and not reduced.endswith((".", "!", "?")):
            reduced += "."
        return reduced

    def _truncate(self, text: str) -> str:
        """Ограничивает длину финальной формулировки."""
        compact = text.strip()
        if len(compact) <= self._max_chars:
            return compact
        trimmed = compact[: self._max_chars].rsplit(" ", 1)[0].rstrip(",;:-")
        if trimmed and not trimmed.endswith((".", "!", "?")):
            trimmed += "."
        return trimmed
