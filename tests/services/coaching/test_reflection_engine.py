"""Тесты для ReflectionEngine."""

from src.services.coaching.contracts.emotion import EmotionDetectionResult
from src.services.coaching.contracts.enums import (
    EmotionState,
    SafetyLevel,
    ScenarioType,
    SessionMode,
    SessionStage,
)
from src.services.coaching.contracts.safety import SafetyDecision
from src.services.coaching.contracts.session import SessionState
from src.services.coaching.reflection_engine import ReflectionEngine


def _build_state(
    *,
    mode: SessionMode = SessionMode.FREE,
    stage: SessionStage = SessionStage.MECHANISM_DISCOVERY,
    scenario: ScenarioType = ScenarioType.A_STRUCTURED,
) -> SessionState:
    return SessionState(
        session_id="sess-reflect-1",
        cycle_id="cycle-reflect-1",
        user_id=10,
        mode=mode,
        stage=stage,
        scenario_type=scenario,
        topic_id="topic-1",
    )


def test_mirror_user_language_reflects_meaning_without_copying_full_text() -> None:
    """Зеркалирование берет слова пользователя, но не копирует фразу целиком."""
    engine = ReflectionEngine()
    state = _build_state()
    user_text = "Мне страшно ошибиться перед командой и потерять доверие."

    result = engine.mirror_user_language(state, user_text)

    assert result != user_text
    lowered = result.lower()
    assert ("страшно" in lowered) or ("доверие" in lowered) or ("командой" in lowered)


def test_insight_hypothesis_uses_soft_prefix() -> None:
    """Гипотеза должна начинаться с мягкой вводной."""
    engine = ReflectionEngine()
    state = _build_state(stage=SessionStage.PATTERN_HIGHLIGHT)
    state.emotion_result = EmotionDetectionResult(
        state=EmotionState.ANXIETY,
        confidence=0.75,
        is_confident=True,
        fallback_to_clarity=False,
    )

    result = engine.insight_hypothesis(state, "Это снова повторяется.")

    lowered = result.lower()
    assert lowered.startswith(("похоже", "возможно", "иногда"))


def test_low_confidence_forces_soft_low_depth_wording() -> None:
    """При низкой уверенности формулировка должна быть максимально мягкой."""
    engine = ReflectionEngine()
    state = _build_state()
    state.emotion_result = EmotionDetectionResult(
        state=EmotionState.UNKNOWN,
        confidence=0.2,
        is_confident=False,
        fallback_to_clarity=True,
    )

    result = engine.insight_hypothesis(state, "Не понимаю, что происходит.")

    assert result.lower().startswith("возможно")


def test_style_guardrails_remove_banned_words_and_limit_output() -> None:
    """Guardrails должны удалять запрещенные слова и не давать перегруженный текст."""
    engine = ReflectionEngine(max_chars=120)
    raw = (
        "Похоже, это похоже на диагноз, и тут нужна психотерапия, "
        "а еще лечение, и это длинная длинная длинная длинная формулировка."
    )

    result = engine._apply_style_guardrails(raw)
    lowered = result.lower()

    assert "диагноз" not in lowered
    assert "психотерапия" not in lowered
    assert "лечение" not in lowered
    assert len(result) <= 120
    sentence_count = len(
        [
            part
            for part in result.replace("?", ".").replace("!", ".").split(".")
            if part.strip()
        ]
    )
    assert sentence_count <= 2


def test_pre_premium_reflection_returns_text_only_when_all_conditions_met() -> None:
    """Pre-premium текст должен появляться только при полном выполнении условий."""
    engine = ReflectionEngine()
    state = _build_state(mode=SessionMode.FREE)
    state.insight_count = 1
    state.contentful_message_count = 5
    state.premium_trigger_shown = False
    state.load_limiter_active = False
    state.safety_decision = SafetyDecision(
        level=SafetyLevel.SAFE,
        must_block_premium_trigger=False,
    )
    state.emotion_result = EmotionDetectionResult(
        state=EmotionState.ANXIETY,
        confidence=0.8,
        is_confident=True,
        fallback_to_clarity=False,
    )

    result = engine.pre_premium_reflection(state, "Понял важный паттерн.")

    assert result is not None
    assert len(result) > 0


def test_pre_premium_reflection_returns_none_when_conditions_fail() -> None:
    """Если хотя бы одно условие нарушено, pre-premium возвращает None."""
    engine = ReflectionEngine()
    state = _build_state(mode=SessionMode.FREE)
    state.insight_count = 1
    state.contentful_message_count = 5
    state.safety_decision = SafetyDecision(must_block_premium_trigger=False)

    state.premium_trigger_shown = True
    assert engine.pre_premium_reflection(state) is None

    state.premium_trigger_shown = False
    state.load_limiter_active = True
    assert engine.pre_premium_reflection(state) is None

    state.load_limiter_active = False
    state.emotion_result.state = EmotionState.OVERLOAD
    assert engine.pre_premium_reflection(state) is None


def test_progress_reflection_mentions_movement() -> None:
    """Progress reflection должен показывать ощущение движения."""
    engine = ReflectionEngine()
    state = _build_state()
    state.insight_count = 1

    result = engine.progress_reflection(state, "Я понял, что делаю это по кругу.")

    assert "сдвиг" in result.lower() or "движ" in result.lower()


def test_next_question_prompt_follows_stage() -> None:
    """Вопрос должен соответствовать стадии и оставаться коротким."""
    engine = ReflectionEngine()
    state = _build_state(stage=SessionStage.INFLUENCE_ZONE)

    result = engine.next_question_prompt(state)

    assert "повлиять" in result.lower()
    assert "?" in result


def test_resistance_simplifies_short_reflection() -> None:
    """При сопротивлении формулировка должна упрощаться и не углубляться."""
    engine = ReflectionEngine()
    state = _build_state(
        mode=SessionMode.PREMIUM, scenario=ScenarioType.B_CLEAN_LANGUAGE
    )

    result = engine.short_reflection(state, "Бесполезно, не хочу в это углубляться.")

    assert "глубок" not in result.lower()
    assert len(result) <= 220


def test_all_public_methods_return_compact_text_or_none() -> None:
    """Публичные методы должны возвращать короткий текст (или None для pre-premium)."""
    engine = ReflectionEngine()
    state = _build_state()
    state.insight_count = 1
    state.contentful_message_count = 5
    state.safety_decision = SafetyDecision(must_block_premium_trigger=False)

    outputs = [
        engine.mirror_user_language(state, "Мне тревожно и сложно выбрать."),
        engine.short_reflection(state, "Мне тревожно и сложно выбрать."),
        engine.insight_hypothesis(state, "Мне тревожно и сложно выбрать."),
        engine.next_question_prompt(state, "Мне тревожно и сложно выбрать."),
        engine.reflection_summary(state),
        engine.soft_landing(state),
        engine.exit_reflection_question(state),
        engine.progress_reflection(state),
    ]
    pre_premium = engine.pre_premium_reflection(state)
    if pre_premium is not None:
        outputs.append(pre_premium)

    for output in outputs:
        sentence_count = len(
            [
                part
                for part in output.replace("?", ".").replace("!", ".").split(".")
                if part.strip()
            ]
        )
        assert sentence_count <= 2
        assert len(output) <= 220
