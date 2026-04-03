"""Тесты для AdaptiveSessionEngine после pipeline-рефактора."""

from src.services.coaching.adaptive_session_engine import (
    AdaptiveSessionEngine,
    EngineReplyType,
)
from src.services.coaching.adaptive_session_rules import RequestType
from src.services.coaching.contracts.enums import (
    ScenarioType,
    SessionMode,
    SessionStage,
)
from src.services.coaching.contracts.session import SessionState


def _build_state(
    *,
    mode: SessionMode = SessionMode.FREE,
    stage: SessionStage = SessionStage.TOPIC_DEFINITION,
    topic_id: str | None = None,
) -> SessionState:
    return SessionState(
        session_id="sess-1",
        cycle_id="cycle-1",
        user_id=1,
        mode=mode,
        stage=stage,
        topic_id=topic_id,
    )


def test_crisis_precheck_switches_to_crisis_protocol() -> None:
    """Кризисный маркер завершает обычный flow и включает кризисный протокол."""
    engine = AdaptiveSessionEngine()
    state = _build_state(topic_id="topic-1")

    result = engine.process_turn(state, "Я хочу умереть, мне невыносимо.")

    assert result.state.stage == SessionStage.CRISIS
    assert result.reply_type == EngineReplyType.CRISIS_PROTOCOL
    assert result.response_plan.template_key == "crisis_support_v1"
    assert not hasattr(result, "response_text")


def test_requested_scenario_is_ignored_in_regular_flow() -> None:
    """requested_scenario не должен влиять на обычный пользовательский путь."""
    engine = AdaptiveSessionEngine()
    state = _build_state(
        mode=SessionMode.PREMIUM,
        stage=SessionStage.MECHANISM_DISCOVERY,
        topic_id="topic-1",
    )

    result = engine.process_turn(
        state,
        "Мне тревожно и страшно, хочу структурно разобрать ситуацию.",
        requested_scenario=ScenarioType.B_CLEAN_LANGUAGE,
        internal_call=False,
    )

    assert result.selected_scenario == ScenarioType.A_STRUCTURED


def test_requested_scenario_is_allowed_for_internal_calls() -> None:
    """requested_scenario разрешен для внутренних вызовов/тестов."""
    engine = AdaptiveSessionEngine()
    state = _build_state(
        mode=SessionMode.PREMIUM,
        stage=SessionStage.MECHANISM_DISCOVERY,
        topic_id="topic-1",
    )

    result = engine.process_turn(
        state,
        "Мне тревожно и страшно, но я готов работать глубже.",
        requested_scenario=ScenarioType.B_CLEAN_LANGUAGE,
        internal_call=True,
    )

    assert result.selected_scenario == ScenarioType.B_CLEAN_LANGUAGE


def test_free_mode_blocks_clean_language_even_for_internal_override() -> None:
    """В Free сценарий B запрещен policy независимо от override."""
    engine = AdaptiveSessionEngine()
    state = _build_state(
        mode=SessionMode.FREE,
        stage=SessionStage.MECHANISM_DISCOVERY,
        topic_id="topic-1",
    )

    result = engine.process_turn(
        state,
        "Мне тревожно и как будто внутри сильный шум.",
        requested_scenario=ScenarioType.B_CLEAN_LANGUAGE,
        internal_call=True,
    )

    assert result.selected_scenario in {
        ScenarioType.A_STRUCTURED,
        ScenarioType.CLARITY_FALLBACK,
    }
    assert result.selected_scenario != ScenarioType.B_CLEAN_LANGUAGE


def test_request_type_confusion_routes_to_clarity_fallback() -> None:
    """Когнитивная запутанность должна вести в безопасный clarity flow."""
    engine = AdaptiveSessionEngine()
    state = _build_state(
        mode=SessionMode.PREMIUM,
        stage=SessionStage.TENSION_REDUCTION,
        topic_id="topic-1",
    )

    result = engine.process_turn(
        state,
        "Я не понимаю, что делать, все запутано.",
        requested_scenario=ScenarioType.B_CLEAN_LANGUAGE,
        internal_call=True,
    )

    assert result.request_type == RequestType.CONFUSION
    assert result.selected_scenario == ScenarioType.CLARITY_FALLBACK


def test_topic_switch_moves_session_to_decision_node() -> None:
    """Попытка сменить тему должна переводить в decision node."""
    engine = AdaptiveSessionEngine()
    state = _build_state(
        mode=SessionMode.FREE,
        stage=SessionStage.TENSION_REDUCTION,
        topic_id="topic-old",
    )

    result = engine.process_turn(
        state,
        "У меня новая тема, это другая ситуация.",
    )

    assert result.state.stage == SessionStage.DECISION_NODE
    assert result.reply_type == EngineReplyType.DECISION_PROMPT
    assert result.response_plan.step_type == "topic_switch_decision_prompt"


def test_load_limiter_disables_deep_path_even_in_premium() -> None:
    """При срабатывании LoadLimiter сценарий B должен отключаться на текущем ходу."""
    engine = AdaptiveSessionEngine()
    state = _build_state(
        mode=SessionMode.PREMIUM,
        stage=SessionStage.MECHANISM_DISCOVERY,
        topic_id="topic-1",
    )
    state.low_engagement_turns = 1

    result = engine.process_turn(
        state,
        "тревожно",
        requested_scenario=ScenarioType.B_CLEAN_LANGUAGE,
        internal_call=True,
    )

    assert result.state.load_limiter_active
    assert result.selected_scenario != ScenarioType.B_CLEAN_LANGUAGE


def test_should_offer_premium_when_all_strict_conditions_met() -> None:
    """Premium-trigger показывается только при полном выполнении формальных условий."""
    engine = AdaptiveSessionEngine()
    state = _build_state(
        mode=SessionMode.FREE,
        stage=SessionStage.PATTERN_HIGHLIGHT,
        topic_id="topic-1",
    )
    state.contentful_message_count = 4

    result = engine.process_turn(
        state,
        "Теперь я понял паттерн и вижу, на что могу повлиять.",
        has_user_resistance=False,
    )

    assert result.should_offer_premium
    assert result.state.premium_trigger_shown


def test_should_not_offer_premium_when_clarity_overload_is_active() -> None:
    """Если активен clarity fallback из-за перегруза, trigger запрещен."""
    engine = AdaptiveSessionEngine()
    state = _build_state(
        mode=SessionMode.FREE,
        stage=SessionStage.PATTERN_HIGHLIGHT,
        topic_id="topic-1",
    )
    state.contentful_message_count = 4
    state.insight_count = 1

    result = engine.process_turn(state, "Перегруз, не тяну, все давит.")

    assert result.selected_scenario == ScenarioType.CLARITY_FALLBACK
    assert not result.should_offer_premium


def test_antiloop_changes_step_type_when_repeated() -> None:
    """Anti-loop не должен повторять один и тот же тип шага подряд."""
    engine = AdaptiveSessionEngine()
    state = _build_state(
        mode=SessionMode.PREMIUM,
        stage=SessionStage.MECHANISM_DISCOVERY,
        topic_id="topic-1",
    )
    state.recent_step_types = ["structured_progress"]

    result = engine.process_turn(
        state,
        "Мне тревожно и страшно, хочу немного ясности.",
    )

    assert result.response_plan.step_type != "structured_progress"


def test_first_turn_confusion_keeps_topic_definition_for_soft_entry() -> None:
    """На первом ходе с запутанностью не уходим сразу в fact/interpretation."""
    engine = AdaptiveSessionEngine()
    state = _build_state(mode=SessionMode.FREE, topic_id="topic-1")

    result = engine.process_turn(state, "Я запуталась, не знаю, что делать.")

    assert result.state.stage == SessionStage.TOPIC_DEFINITION
    assert result.response_plan.step_type == "state_clarification"


def test_first_turn_emotion_keeps_topic_definition_for_emotion_contact() -> None:
    """На первом эмоциональном ходе удерживаем мягкий вход в тему."""
    engine = AdaptiveSessionEngine()
    state = _build_state(mode=SessionMode.FREE, topic_id="topic-1")

    result = engine.process_turn(state, "Мне грустно и тревожно.")

    assert result.state.stage == SessionStage.TOPIC_DEFINITION
    assert result.response_plan.step_type == "emotion_contact"


def test_first_turn_pattern_input_keeps_structured_progression() -> None:
    """Осознанный паттерновый вход остается в структурной ветке."""
    engine = AdaptiveSessionEngine()
    state = _build_state(mode=SessionMode.FREE, topic_id="topic-1")

    result = engine.process_turn(
        state,
        "Хочу понять, почему повторяется один и тот же сценарий.",
    )

    assert result.state.stage == SessionStage.TENSION_REDUCTION
    assert result.request_type == RequestType.REPEATING_PATTERN


def test_first_turn_worry_is_recognized_as_emotional_soft_entry() -> None:
    """Маркер «волнуюсь» должен вести в мягкий эмоциональный вход."""
    engine = AdaptiveSessionEngine()
    state = _build_state(mode=SessionMode.FREE, topic_id="topic-1")

    result = engine.process_turn(state, "Я очень волнуюсь, что нет работы.")

    assert result.state.stage == SessionStage.TOPIC_DEFINITION
    assert result.response_plan.step_type == "emotion_contact"


def test_clarification_input_holds_stage_and_step() -> None:
    """Запрос «не поняла вопрос» не должен продвигать стадию и углублять ход."""
    engine = AdaptiveSessionEngine()
    state = _build_state(
        mode=SessionMode.FREE,
        stage=SessionStage.TENSION_REDUCTION,
        topic_id="topic-1",
    )

    result = engine.process_turn(state, "Не поняла вопрос.")

    assert result.request_type == RequestType.CONFUSION
    assert result.response_plan.step_type == "state_clarification"
    assert result.state.stage == SessionStage.TENSION_REDUCTION
    assert result.selected_scenario == ScenarioType.CLARITY_FALLBACK


def test_rephrase_request_is_treated_as_clarification() -> None:
    """Фраза «переформулируйте, пожалуйста» должна идти как clarification."""
    engine = AdaptiveSessionEngine()
    state = _build_state(
        mode=SessionMode.FREE,
        stage=SessionStage.MECHANISM_DISCOVERY,
        topic_id="topic-1",
    )

    result = engine.process_turn(state, "Переформулируйте, пожалуйста.")

    assert result.request_type == RequestType.CONFUSION
    assert result.response_plan.step_type == "state_clarification"
    assert result.state.stage == SessionStage.MECHANISM_DISCOVERY
