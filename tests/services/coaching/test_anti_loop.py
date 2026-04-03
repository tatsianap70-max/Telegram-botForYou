"""Тесты для изолированного anti-loop policy-модуля (усиленная версия)."""

from pathlib import Path

from src.services.coaching.anti_loop import (
    RecoveryStrategy,
    SignalStrength,
    evaluate_anti_loop,
)
from src.services.coaching.contracts.enums import (
    ScenarioType,
    SessionMode,
    SessionStage,
)
from src.services.coaching.contracts.session import SessionState


def _build_state(
    *,
    question_count: int = 4,
    contentful_message_count: int = 4,
    insight_count: int = 1,
    no_progress_turns: int = 0,
    low_engagement_turns: int = 0,
    recent_step_types: list[str] | None = None,
    topic_id: str | None = "topic-1",
    pending_topic_id: str | None = None,
) -> SessionState:
    return SessionState(
        session_id="session-1",
        cycle_id="cycle-1",
        user_id=17,
        mode=SessionMode.FREE,
        stage=SessionStage.MECHANISM_DISCOVERY,
        scenario_type=ScenarioType.A_STRUCTURED,
        topic_id=topic_id,
        pending_topic_id=pending_topic_id,
        question_count=question_count,
        contentful_message_count=contentful_message_count,
        insight_count=insight_count,
        no_progress_turns=no_progress_turns,
        low_engagement_turns=low_engagement_turns,
        recent_step_types=recent_step_types or [],
    )


def test_normal_progress_returns_no_loop() -> None:
    """При живом прогрессе anti-loop не должен вмешиваться."""
    state = _build_state(
        no_progress_turns=0,
        recent_step_types=["topic_definition", "tension_reduction"],
    )
    turns = [
        "Я увидела, что тревога поднимается после конкретного комментария.",
        "Сейчас я лучше понимаю, где начинается мой автоматический ответ.",
        "Хочу выбрать один реальный шаг на сегодня.",
    ]

    decision = evaluate_anti_loop(state, turns)

    assert not decision.requires_intervention
    assert not decision.loop_detected
    assert decision.signal_strength is SignalStrength.NONE
    assert decision.strategy is RecoveryStrategy.NONE


def test_repeated_thought_two_three_times_detects_loop() -> None:
    """Повтор одной мысли 2-3 раза должен давать loop."""
    state = _build_state(
        no_progress_turns=1,
        recent_step_types=["mechanism_discovery", "pattern_highlight"],
    )
    turns = [
        "Меня задевает критика на работе в похожих ситуациях.",
        "Меня очень задевает критика на работе в похожих ситуациях.",
        "Все равно задевает критика на работе в похожих ситуациях.",
    ]

    decision = evaluate_anti_loop(state, turns)

    assert decision.loop_detected
    assert decision.requires_intervention
    assert decision.signal_strength in {SignalStrength.WEAK, SignalStrength.CLEAR}
    assert decision.strategy is RecoveryStrategy.META_QUESTION


def test_no_new_fact_detects_stagnation() -> None:
    """Если нового факта/инсайта/шага нет, должна фиксироваться stagnation."""
    state = _build_state(
        insight_count=0,
        no_progress_turns=2,
        contentful_message_count=3,
        recent_step_types=["mechanism_discovery", "mechanism_discovery"],
    )
    turns = [
        "В этой ситуации я снова реагирую одинаково, без изменений в шаге.",
        "В этой ситуации я снова реагирую одинаково, без изменений в шаге.",
    ]

    decision = evaluate_anti_loop(state, turns)

    assert decision.stagnation_detected
    assert decision.loop_detected
    assert decision.signal_strength in {SignalStrength.WEAK, SignalStrength.CLEAR}


def test_repeated_system_angle_requires_shift() -> None:
    """Повтор system-angle должен вести к ANGLE_SHIFT по приоритету."""
    state = _build_state(
        no_progress_turns=1,
        recent_step_types=[
            "mechanism_discovery",
            "mechanism_discovery",
            "mechanism_discovery",
        ],
    )
    turns = [
        "Я вижу факт и эмоцию, но застреваю в одном и том же месте.",
        "Снова говорю про то же и движения пока нет.",
    ]

    decision = evaluate_anti_loop(state, turns)

    assert decision.requires_intervention
    assert decision.strategy is RecoveryStrategy.ANGLE_SHIFT


def test_short_answers_with_stagnation_recommend_simplify_or_soft_close() -> None:
    """Короткие ответы + stagnation дают simplify, а не случайную стратегию."""
    state = _build_state(
        insight_count=0,
        no_progress_turns=2,
        recent_step_types=["tension_reduction", "pattern_highlight"],
    )
    turns = ["не знаю", "не знаю", "ничего не меняется"]

    decision = evaluate_anti_loop(state, turns)

    assert decision.requires_intervention
    assert decision.strategy in {
        RecoveryStrategy.SIMPLIFY,
        RecoveryStrategy.SOFT_CLOSE,
    }


def test_short_answer_alone_does_not_create_loop() -> None:
    """Короткий ответ сам по себе не должен включать anti-loop."""
    state = _build_state(
        no_progress_turns=0,
        recent_step_types=["tension_reduction", "mechanism_discovery"],
    )
    turns = ["не знаю", "не знаю"]

    decision = evaluate_anti_loop(state, turns)

    assert not decision.requires_intervention
    assert not decision.loop_detected
    assert decision.signal_strength is SignalStrength.NONE
    assert decision.strategy is RecoveryStrategy.NONE


def test_max_repeat_policy_leads_to_soft_close() -> None:
    """После нескольких повторов без прогресса включается soft-close."""
    state = _build_state(
        insight_count=0,
        no_progress_turns=3,
        recent_step_types=["pattern_highlight", "pattern_highlight"],
    )
    turns = ["не знаю", "все то же самое", "ничего не меняется"]

    decision = evaluate_anti_loop(state, turns)

    assert decision.requires_intervention
    assert decision.signal_strength is SignalStrength.CLEAR
    assert decision.strategy is RecoveryStrategy.SOFT_CLOSE
    assert decision.should_soft_close


def test_soft_close_not_triggered_on_weak_signal() -> None:
    """SOFT_CLOSE не должен срабатывать при слабом сигнале."""
    state = _build_state(
        no_progress_turns=2,
        recent_step_types=["mechanism_discovery", "mechanism_discovery"],
    )
    turns = [
        "Сегодня в разговоре с коллегой я заметила новый факт про ситуацию.",
        "После разговора я выберу один спокойный шаг и проверю реакцию.",
    ]

    decision = evaluate_anti_loop(state, turns)

    assert decision.signal_strength in {SignalStrength.NONE, SignalStrength.WEAK}
    assert decision.strategy is not RecoveryStrategy.SOFT_CLOSE
    assert not decision.should_soft_close


def test_soft_close_requires_clear_signal_and_confirmed_repeat() -> None:
    """SOFT_CLOSE включается только при CLEAR и подтвержденном repeat/stagnation."""
    state = _build_state(
        no_progress_turns=3,
        recent_step_types=["mechanism_discovery", "pattern_highlight"],
    )
    turns = [
        "Сегодня на встрече появился новый факт про мои границы.",
        "Я поняла связь и выберу следующий шаг для этой ситуации.",
    ]

    decision = evaluate_anti_loop(state, turns)

    assert decision.signal_strength is SignalStrength.NONE
    assert decision.strategy is RecoveryStrategy.NONE
    assert not decision.should_soft_close


def test_anti_loop_not_activated_too_early() -> None:
    """До минимального контекста anti-loop не должен срабатывать."""
    state = _build_state(
        question_count=1,
        contentful_message_count=1,
        no_progress_turns=0,
        recent_step_types=["topic_definition"],
    )
    turns = ["Мне тревожно, хочу разобраться."]

    decision = evaluate_anti_loop(state, turns)

    assert not decision.requires_intervention
    assert decision.signal_strength is SignalStrength.NONE
    assert decision.strategy is RecoveryStrategy.NONE
    assert decision.reason == "insufficient_context"


def test_anti_loop_not_activated_on_second_turn_uncertainty_entry() -> None:
    """На 2-м ходе 'запуталась -> не знаю' anti-loop еще не должен вмешиваться."""
    state = _build_state(
        question_count=2,
        contentful_message_count=2,
        no_progress_turns=2,
        recent_step_types=["state_clarification", "state_clarification"],
    )
    turns = ["Я запуталась, не знаю, что делать.", "Не знаю"]

    decision = evaluate_anti_loop(state, turns)

    assert not decision.requires_intervention
    assert decision.signal_strength is SignalStrength.NONE
    assert decision.strategy is RecoveryStrategy.NONE
    assert decision.reason == "early_uncertainty_entry"


def test_anti_loop_disabled_for_all_early_turns_t1_t3() -> None:
    """На T1-T3 anti-loop должен быть выключен даже при повторе угла."""
    state = _build_state(
        question_count=3,
        contentful_message_count=3,
        no_progress_turns=3,
        recent_step_types=["clarity_structuring", "clarity_structuring"],
    )
    turns = [
        "Опять одно и то же в этой ситуации.",
        "Опять одно и то же в этой ситуации.",
    ]

    decision = evaluate_anti_loop(state, turns)

    assert not decision.requires_intervention
    assert decision.signal_strength is SignalStrength.NONE
    assert decision.strategy is RecoveryStrategy.NONE
    assert decision.reason == "early_turn_guard"


def test_anti_loop_disabled_for_clarification_chain() -> None:
    """Clarification-цепочка не должна включать anti-loop."""
    state = _build_state(
        question_count=5,
        contentful_message_count=5,
        no_progress_turns=3,
        recent_step_types=["state_clarification", "state_clarification"],
    )
    turns = ["Не поняла вопрос.", "Переформулируйте, пожалуйста."]

    decision = evaluate_anti_loop(state, turns)

    assert not decision.requires_intervention
    assert decision.signal_strength is SignalStrength.NONE
    assert decision.strategy is RecoveryStrategy.NONE
    assert decision.reason == "clarification_chain_guard"


def test_anti_loop_does_not_interfere_with_new_topic() -> None:
    """Если pending_topic уже задан, anti-loop не должен вмешиваться в цикл."""
    state = _build_state(
        insight_count=0,
        no_progress_turns=3,
        topic_id="topic-a",
        pending_topic_id="topic-b",
        recent_step_types=["pattern_highlight", "pattern_highlight"],
    )
    cycle_before = state.cycle_id
    topic_before = state.topic_id

    decision = evaluate_anti_loop(state, ["не знаю", "все то же самое"])

    assert not decision.requires_intervention
    assert decision.strategy is RecoveryStrategy.NONE
    assert decision.reason == "topic_switch_pending"
    assert state.cycle_id == cycle_before
    assert state.topic_id == topic_before


def test_low_engagement_alone_does_not_trigger_anti_loop() -> None:
    """Усталость сама по себе не должна включать anti-loop без stagnation сигнала."""
    state = _build_state(
        no_progress_turns=0,
        low_engagement_turns=3,
        recent_step_types=["tension_reduction", "mechanism_discovery"],
    )
    turns = [
        "Я устала, но сейчас могу ответить на один короткий вопрос.",
        "Пока хочу просто немного замедлиться и сохранить фокус.",
    ]

    decision = evaluate_anti_loop(state, turns)

    assert not decision.requires_intervention
    assert decision.signal_strength is SignalStrength.NONE
    assert decision.strategy is RecoveryStrategy.NONE


def test_reason_is_neutral_and_non_interpretive() -> None:
    """Reason должен описывать динамику, а не давать интерпретации пользователя."""
    state = _build_state(
        no_progress_turns=2,
        recent_step_types=["pattern_highlight", "pattern_highlight"],
    )
    turns = [
        "В этой ситуации я снова говорю о том же самом без нового шага.",
        "В этой ситуации я снова говорю о том же самом без нового шага.",
    ]

    decision = evaluate_anti_loop(state, turns)

    allowed_reasons = {
        "none",
        "insufficient_context",
        "early_uncertainty_entry",
        "early_turn_guard",
        "clarification_chain_guard",
        "weak_single_signal",
        "repeated_user_angle",
        "repeated_system_angle",
        "no_new_information",
        "stagnation_detected",
        "topic_switch_pending",
    }
    assert decision.reason in allowed_reasons
    lower_reason = decision.reason.lower()
    assert "избег" not in lower_reason
    assert "боит" not in lower_reason
    assert "сопротив" not in lower_reason


def test_module_has_no_user_facing_generated_phrases() -> None:
    """Модуль anti-loop не должен содержать user-facing формулировки ответа."""
    path = Path("src/services/coaching/anti_loop.py")
    source = path.read_text(encoding="utf-8").lower()

    assert "мы ходим по кругу" not in source
    assert "давайте попробуем иначе" not in source
    assert "кажется, ничего не меняется" not in source


def test_missing_state_fields_do_not_break_evaluation() -> None:
    """Модуль должен быть устойчив к отсутствующим полям state через getattr."""
    state = _build_state(
        no_progress_turns=2,
        recent_step_types=["mechanism_discovery", "pattern_highlight"],
    )
    del state.__dict__["recent_step_types"]
    del state.__dict__["no_progress_turns"]

    decision = evaluate_anti_loop(
        state,
        ["Я повторяю ту же мысль.", "Я повторяю ту же мысль."],
    )

    assert decision.signal_strength in {SignalStrength.NONE, SignalStrength.WEAK}
