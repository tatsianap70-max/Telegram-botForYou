"""Доменный in-memory модуль памяти инсайтов.

Модуль не интегрирован в transport-слой и не зависит от Telegram.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol

from typing_extensions import override

from src.services.coaching.contracts.enums import SessionMode

if TYPE_CHECKING:
    from src.services.coaching.contracts.session import SessionState


class InsightMemoryPort(Protocol):
    """Порт доменного слоя для работы с памятью инсайтов."""

    def save_insight(
        self,
        state: SessionState,
        insight_text: str,
        topic_id: str | None = None,
    ) -> InsightRecord:
        """Сохранить инсайт пользователя с учетом контекста сессии."""

    def get_last_insight(self, user_id: int) -> InsightRecord | None:
        """Вернуть последний инсайт пользователя."""

    def get_recent_insights(
        self,
        user_id: int,
        limit: int = 5,
    ) -> list[InsightRecord]:
        """Вернуть последние N инсайтов пользователя."""

    def build_memory_reflection(self, user_id: int) -> str | None:
        """Построить короткое напоминание о прошлом инсайте."""


@dataclass(slots=True, frozen=True)
class InsightRecord:
    """Структурная запись инсайта."""

    user_id: int
    insight_text: str
    created_at: datetime
    topic_id: str | None
    cycle_id: str
    session_id: str
    mode: SessionMode


class InMemoryInsightMemory(InsightMemoryPort):
    """In-memory реализация памяти инсайтов с policy Free/Premium."""

    def __init__(
        self,
        *,
        max_insight_chars: int = 320,
        default_recent_limit: int = 5,
        reflection_preview_chars: int = 110,
    ) -> None:
        if max_insight_chars < 80:
            msg = "Параметр max_insight_chars должен быть >= 80."
            raise ValueError(msg)
        if default_recent_limit < 1:
            msg = "Параметр default_recent_limit должен быть >= 1."
            raise ValueError(msg)
        if reflection_preview_chars < 30:
            msg = "Параметр reflection_preview_chars должен быть >= 30."
            raise ValueError(msg)

        self._max_insight_chars = max_insight_chars
        self._default_recent_limit = default_recent_limit
        self._reflection_preview_chars = reflection_preview_chars
        self._records_by_user: dict[int, list[InsightRecord]] = {}

    @override
    def save_insight(
        self,
        state: SessionState,
        insight_text: str,
        topic_id: str | None = None,
    ) -> InsightRecord:
        """Сохранить инсайт в контексте пользователя/сессии/цикла/темы.

        Политика:
        - Free: храним только инсайты текущего session/cycle.
        - Premium: храним долгосрочно.
        """
        cleaned_text = self._normalize_insight_text(insight_text)
        record = InsightRecord(
            user_id=state.user_id,
            insight_text=cleaned_text,
            created_at=datetime.now(UTC),
            topic_id=topic_id or state.topic_id,
            cycle_id=state.cycle_id,
            session_id=state.session_id,
            mode=state.mode,
        )

        records = self._records_by_user.setdefault(state.user_id, [])
        if state.mode is SessionMode.FREE:
            records[:] = self._filter_free_records_for_active_cycle(
                records,
                session_id=state.session_id,
                cycle_id=state.cycle_id,
            )
        records.append(record)
        return record

    @override
    def get_last_insight(self, user_id: int) -> InsightRecord | None:
        """Вернуть последний инсайт пользователя независимо от режима."""
        records = self._records_by_user.get(user_id, [])
        if not records:
            return None
        return records[-1]

    @override
    def get_recent_insights(
        self,
        user_id: int,
        limit: int = 5,
    ) -> list[InsightRecord]:
        """Вернуть список последних инсайтов в порядке от нового к старому."""
        if limit < 1:
            msg = "Параметр limit должен быть >= 1."
            raise ValueError(msg)

        effective_limit = limit or self._default_recent_limit
        records = self._records_by_user.get(user_id, [])
        reversed_records = list(reversed(records))
        return reversed_records[:effective_limit]

    @override
    def build_memory_reflection(self, user_id: int) -> str | None:
        """Сформировать короткое напоминание по последнему инсайту."""
        last_record = self.get_last_insight(user_id)
        if last_record is None:
            return None

        excerpt = self._build_excerpt(last_record.insight_text)
        if last_record.topic_id:
            text = (
                f"Ранее вы отмечали по теме '{last_record.topic_id}': "
                f"'{excerpt}'. Похоже, это может быть опорой и сейчас."
            )
            return self._normalize_reflection_text(text)

        text = (
            f"Ранее вы отмечали: '{excerpt}'. Похоже, это может быть опорой и сейчас."
        )
        return self._normalize_reflection_text(text)

    @staticmethod
    def _filter_free_records_for_active_cycle(
        records: list[InsightRecord],
        *,
        session_id: str,
        cycle_id: str,
    ) -> list[InsightRecord]:
        """Оставить только актуальные free-записи текущего session/cycle."""
        filtered: list[InsightRecord] = []
        for record in records:
            if record.mode is SessionMode.PREMIUM:
                filtered.append(record)
                continue

            is_same_session = record.session_id == session_id
            is_same_cycle = record.cycle_id == cycle_id
            if is_same_session and is_same_cycle:
                filtered.append(record)
        return filtered

    def _normalize_insight_text(self, insight_text: str) -> str:
        """Очистить и ограничить текст инсайта.

        Модуль хранит только короткую структурную формулировку,
        а не длинные сырые диалоги.
        """
        normalized = re.sub(r"\s+", " ", insight_text.strip())
        if not normalized:
            msg = "Текст инсайта не может быть пустым."
            raise ValueError(msg)

        if len(normalized) <= self._max_insight_chars:
            return normalized

        truncated = normalized[: self._max_insight_chars].rsplit(" ", 1)[0]
        truncated = truncated.rstrip(",;:-")
        if not truncated:
            truncated = normalized[: self._max_insight_chars]
        return f"{truncated}..."

    def _build_excerpt(self, insight_text: str) -> str:
        """Построить короткий фрагмент инсайта для напоминания."""
        if len(insight_text) <= self._reflection_preview_chars:
            return insight_text

        preview = insight_text[: self._reflection_preview_chars].rsplit(" ", 1)[0]
        preview = preview.rstrip(",;:-")
        if not preview:
            preview = insight_text[: self._reflection_preview_chars]
        return f"{preview}..."

    @staticmethod
    def _normalize_reflection_text(text: str) -> str:
        """Сделать напоминание компактным и читаемым."""
        compact = re.sub(r"\s+", " ", text.strip())
        sentences = re.split(r"(?<=[.!?])\s+", compact)
        prepared = [sentence.strip() for sentence in sentences if sentence.strip()]
        return " ".join(prepared[:2])
