"""Тесты для доменного модуля памяти инсайтов."""

from src.services.coaching.contracts.emotion import EmotionDetectionResult
from src.services.coaching.contracts.enums import (
    EmotionState,
    ScenarioType,
    SessionMode,
    SessionStage,
)
from src.services.coaching.contracts.safety import SafetyDecision
from src.services.coaching.contracts.session import SessionState
from src.services.coaching.insight_memory import InMemoryInsightMemory


def _build_state(
    *,
    user_id: int = 1,
    mode: SessionMode = SessionMode.FREE,
    session_id: str = "session-1",
    cycle_id: str = "cycle-1",
    topic_id: str | None = "topic-1",
) -> SessionState:
    return SessionState(
        session_id=session_id,
        cycle_id=cycle_id,
        user_id=user_id,
        topic_id=topic_id,
        mode=mode,
        stage=SessionStage.MECHANISM_DISCOVERY,
        scenario_type=ScenarioType.A_STRUCTURED,
        safety_decision=SafetyDecision(),
        emotion_result=EmotionDetectionResult(
            state=EmotionState.ANXIETY,
            confidence=0.8,
            is_confident=True,
            fallback_to_clarity=False,
        ),
    )


def test_save_insight_keeps_structured_context() -> None:
    """Сохранение инсайта должно учитывать user/session/cycle/topic/mode."""
    memory = InMemoryInsightMemory()
    state = _build_state()

    record = memory.save_insight(state, "Я замечаю, что критикую себя автоматически.")

    assert record.user_id == 1
    assert record.session_id == "session-1"
    assert record.cycle_id == "cycle-1"
    assert record.topic_id == "topic-1"
    assert record.mode is SessionMode.FREE
    assert len(record.insight_text) > 0


def test_save_insight_allows_topic_override() -> None:
    """Явный topic_id должен переопределять topic_id из state."""
    memory = InMemoryInsightMemory()
    state = _build_state(topic_id="topic-from-state")

    record = memory.save_insight(
        state,
        "Замечаю повторение одной и той же реакции.",
        topic_id="topic-override",
    )

    assert record.topic_id == "topic-override"


def test_get_last_insight_returns_latest_record() -> None:
    """Последний инсайт должен возвращаться корректно."""
    memory = InMemoryInsightMemory()
    state = _build_state(mode=SessionMode.PREMIUM)
    memory.save_insight(state, "Первый инсайт.")
    last = memory.save_insight(state, "Второй инсайт.")

    result = memory.get_last_insight(state.user_id)

    assert result is not None
    assert result.insight_text == last.insight_text


def test_get_recent_insights_applies_limit_and_order() -> None:
    """Список должен быть ограничен limit и отсортирован от нового к старому."""
    memory = InMemoryInsightMemory()
    state = _build_state(mode=SessionMode.PREMIUM)
    memory.save_insight(state, "Инсайт 1")
    memory.save_insight(state, "Инсайт 2")
    memory.save_insight(state, "Инсайт 3")

    recent = memory.get_recent_insights(state.user_id, limit=2)

    assert len(recent) == 2
    assert recent[0].insight_text == "Инсайт 3"
    assert recent[1].insight_text == "Инсайт 2"


def test_build_memory_reflection_returns_none_when_no_data() -> None:
    """Если инсайтов нет, напоминание не строится."""
    memory = InMemoryInsightMemory()

    assert memory.build_memory_reflection(user_id=101) is None


def test_build_memory_reflection_uses_last_insight() -> None:
    """Напоминание должно опираться на последний инсайт пользователя."""
    memory = InMemoryInsightMemory()
    state = _build_state(mode=SessionMode.PREMIUM, topic_id="responsibility")
    memory.save_insight(state, "Я могу выбирать более мягкий внутренний тон.")

    reflection = memory.build_memory_reflection(state.user_id)

    assert reflection is not None
    assert "Ранее вы отмечали" in reflection
    assert "опорой" in reflection


def test_free_policy_keeps_only_current_cycle_memory() -> None:
    """В Free должны сохраняться только записи текущего session/cycle."""
    memory = InMemoryInsightMemory()
    state_cycle_1 = _build_state(
        mode=SessionMode.FREE,
        user_id=2,
        session_id="session-a",
        cycle_id="cycle-1",
    )
    state_cycle_2 = _build_state(
        mode=SessionMode.FREE,
        user_id=2,
        session_id="session-a",
        cycle_id="cycle-2",
    )

    memory.save_insight(state_cycle_1, "Инсайт первого цикла.")
    memory.save_insight(state_cycle_2, "Инсайт второго цикла.")
    recent = memory.get_recent_insights(user_id=2, limit=5)

    assert len(recent) == 1
    assert recent[0].cycle_id == "cycle-2"
    assert recent[0].insight_text == "Инсайт второго цикла."


def test_premium_policy_keeps_long_term_memory() -> None:
    """В Premium записи должны храниться долгосрочно между циклами."""
    memory = InMemoryInsightMemory()
    state_cycle_1 = _build_state(
        mode=SessionMode.PREMIUM,
        user_id=3,
        cycle_id="cycle-1",
    )
    state_cycle_2 = _build_state(
        mode=SessionMode.PREMIUM,
        user_id=3,
        cycle_id="cycle-2",
    )

    memory.save_insight(state_cycle_1, "Premium инсайт 1.")
    memory.save_insight(state_cycle_2, "Premium инсайт 2.")
    recent = memory.get_recent_insights(user_id=3, limit=5)

    assert len(recent) == 2
    assert recent[0].insight_text == "Premium инсайт 2."
    assert recent[1].insight_text == "Premium инсайт 1."


def test_save_insight_rejects_empty_text() -> None:
    """Пустой инсайт должен быть отклонен."""
    memory = InMemoryInsightMemory()
    state = _build_state()

    try:
        memory.save_insight(state, "   ")
    except ValueError:
        pass
    else:
        raise AssertionError("Ожидался ValueError для пустого insight_text")


def test_save_insight_truncates_long_raw_text() -> None:
    """Слишком длинный текст должен сохраняться в сокращенном виде."""
    memory = InMemoryInsightMemory(max_insight_chars=90)
    state = _build_state(mode=SessionMode.PREMIUM)
    long_text = (
        "Это очень длинный сырой текст с множеством повторений и переносов строк.\n"
        "Он не должен храниться как полный диалог, только короткая формулировка."
    )

    record = memory.save_insight(state, long_text)

    assert len(record.insight_text) <= 91
    assert "\n" not in record.insight_text
    assert record.insight_text.endswith("...")
