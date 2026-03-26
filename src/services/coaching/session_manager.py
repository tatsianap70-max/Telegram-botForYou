"""Менеджер сессии: инварианты, переходы и policy-проверки.

Модуль реализует «рельсы» для коучингового потока:
- валидные переходы между стадиями;
- правило «одна тема = один цикл»;
- лимит глубокой Premium-сессии (1 в день);
- критерии показа Premium-trigger;
- безопасный выбор сценария через ScenarioSelector.
"""

from src.services.coaching.contracts.completion import CompletionCriteria
from src.services.coaching.contracts.emotion import EmotionDetectionResult
from src.services.coaching.contracts.enums import (
    DISABLED_V1_SCENARIOS,
    SafetyLevel,
    ScenarioType,
    SessionMode,
    SessionStage,
)
from src.services.coaching.contracts.safety import SafetyDecision
from src.services.coaching.contracts.session import SessionState
from src.services.coaching.scenario_selector import ScenarioSelector


class SessionManagerError(Exception):
    """Базовое исключение менеджера сессии."""


class InvalidStageTransitionError(SessionManagerError):
    """Недопустимый переход между стадиями сессии."""


class TopicSwitchPolicyError(SessionManagerError):
    """Нарушение правила смены темы."""


class DeepSessionLimitError(SessionManagerError):
    """Нарушение дневного лимита глубокой Premium-сессии."""


class SessionCompletionError(SessionManagerError):
    """Ошибка завершения сессии по критериям."""


_ALLOWED_STAGE_TRANSITIONS: dict[SessionStage, frozenset[SessionStage]] = {
    SessionStage.TOPIC_DEFINITION: frozenset(
        {
            SessionStage.TENSION_REDUCTION,
            SessionStage.DECISION_NODE,
            SessionStage.CRISIS,
        }
    ),
    SessionStage.TENSION_REDUCTION: frozenset(
        {
            SessionStage.MECHANISM_DISCOVERY,
            SessionStage.DECISION_NODE,
            SessionStage.CRISIS,
        }
    ),
    SessionStage.MECHANISM_DISCOVERY: frozenset(
        {
            SessionStage.PATTERN_HIGHLIGHT,
            SessionStage.DECISION_NODE,
            SessionStage.CRISIS,
        }
    ),
    SessionStage.PATTERN_HIGHLIGHT: frozenset(
        {
            SessionStage.DEEP_ANALYSIS,
            SessionStage.REFLECTION_SUMMARY,
            SessionStage.DECISION_NODE,
            SessionStage.CRISIS,
        }
    ),
    SessionStage.DEEP_ANALYSIS: frozenset(
        {
            SessionStage.INSIGHT,
            SessionStage.DECISION_NODE,
            SessionStage.CRISIS,
        }
    ),
    SessionStage.INSIGHT: frozenset(
        {
            SessionStage.INFLUENCE_ZONE,
            SessionStage.DECISION_NODE,
            SessionStage.CRISIS,
        }
    ),
    SessionStage.INFLUENCE_ZONE: frozenset(
        {
            SessionStage.ACTION_STEP,
            SessionStage.REFLECTION_SUMMARY,
            SessionStage.DECISION_NODE,
            SessionStage.CRISIS,
        }
    ),
    SessionStage.ACTION_STEP: frozenset(
        {
            SessionStage.REFLECTION_SUMMARY,
            SessionStage.DECISION_NODE,
            SessionStage.CRISIS,
        }
    ),
    SessionStage.DECISION_NODE: frozenset(
        {
            SessionStage.TOPIC_DEFINITION,
            SessionStage.DEEP_ANALYSIS,
            SessionStage.REFLECTION_SUMMARY,
            SessionStage.CRISIS,
        }
    ),
    SessionStage.REFLECTION_SUMMARY: frozenset(
        {
            SessionStage.COMPLETION,
            SessionStage.DECISION_NODE,
            SessionStage.CRISIS,
        }
    ),
    SessionStage.COMPLETION: frozenset({SessionStage.DECISION_NODE}),
    SessionStage.CRISIS: frozenset(),
}


class SessionManager:
    """Управляет состоянием сессии и проверяет доменные правила."""

    def __init__(self, scenario_selector: ScenarioSelector | None = None) -> None:
        """Создать менеджер сессии.

        Args:
            scenario_selector: Селектор сценариев. Если не передан, будет
                создан стандартный ScenarioSelector для policy v1.
        """
        self._scenario_selector = scenario_selector or ScenarioSelector()

    def transition_stage(
        self, state: SessionState, next_stage: SessionStage
    ) -> SessionState:
        """Выполнить переход стадии с проверкой разрешённых переходов."""
        if state.stage == next_stage:
            return state

        allowed_stages = _ALLOWED_STAGE_TRANSITIONS[state.stage]
        if next_stage not in allowed_stages:
            msg = (
                f"Переход {state.stage.value} -> {next_stage.value} не разрешён "
                "в policy v1."
            )
            raise InvalidStageTransitionError(msg)

        state.stage = next_stage
        if next_stage == SessionStage.CRISIS:
            state.safety_decision = SafetyDecision(
                level=SafetyLevel.CRISIS,
                can_continue=False,
                must_switch_to_crisis_protocol=True,
                must_block_premium_trigger=True,
            )
        return state

    def set_topic(self, state: SessionState, topic_id: str) -> SessionState:
        """Установить тему цикла.

        Правило:
        - если тема ещё не установлена, сохраняем её;
        - если тема совпадает, состояние не меняется;
        - если тема новая, нужен decision node + новый цикл.
        """
        if not topic_id:
            msg = "topic_id не может быть пустым."
            raise ValueError(msg)

        if state.topic_id is None:
            state.topic_id = topic_id
            state.pending_topic_id = None
            return state

        if state.topic_id == topic_id:
            return state

        msg = (
            "Смена темы в текущем цикле запрещена. "
            "Сначала нужен переход в decision node."
        )
        raise TopicSwitchPolicyError(msg)

    def request_topic_switch(
        self, state: SessionState, new_topic_id: str
    ) -> SessionState:
        """Запросить смену темы через decision node."""
        if not new_topic_id:
            msg = "new_topic_id не может быть пустым."
            raise ValueError(msg)

        if state.stage == SessionStage.CRISIS:
            msg = "В кризисном состоянии смена темы запрещена."
            raise TopicSwitchPolicyError(msg)

        if state.topic_id is None:
            state.topic_id = new_topic_id
            state.pending_topic_id = None
            return state

        if state.topic_id == new_topic_id:
            state.pending_topic_id = None
            return state

        state.pending_topic_id = new_topic_id
        state.stage = SessionStage.DECISION_NODE
        return state

    def apply_topic_switch_decision(
        self,
        state: SessionState,
        approve_switch: bool,
        new_cycle_id: str | None = None,
    ) -> SessionState:
        """Применить решение по смене темы в decision node."""
        if state.stage != SessionStage.DECISION_NODE:
            msg = "Решение о смене темы доступно только на стадии decision_node."
            raise TopicSwitchPolicyError(msg)

        if not approve_switch:
            state.pending_topic_id = None
            state.stage = SessionStage.REFLECTION_SUMMARY
            return state

        if state.pending_topic_id is None:
            msg = "Нет ожидающей темы для переключения."
            raise TopicSwitchPolicyError(msg)
        if not new_cycle_id:
            msg = "При подтверждении смены темы нужен new_cycle_id."
            raise TopicSwitchPolicyError(msg)

        state.cycle_id = new_cycle_id
        state.topic_id = state.pending_topic_id
        state.pending_topic_id = None
        state.stage = SessionStage.TOPIC_DEFINITION

        state.question_count = 0
        state.contentful_message_count = 0
        state.insight_count = 0
        state.premium_trigger_shown = False

        state.scenario_type = self._scenario_selector.select_default(state.mode)
        state.safety_decision = SafetyDecision()
        state.emotion_result = EmotionDetectionResult()
        state.completion_criteria = CompletionCriteria()
        return state

    def update_emotion(
        self, state: SessionState, emotion_result: EmotionDetectionResult
    ) -> SessionState:
        """Обновить эмоцию и при необходимости переключить сценарий на fallback."""
        state.emotion_result = emotion_result
        state.scenario_type = self._scenario_selector.select(
            mode=state.mode,
            requested=state.scenario_type,
            emotion_result=emotion_result,
        )
        return state

    def select_scenario(
        self, state: SessionState, requested: ScenarioType
    ) -> SessionState:
        """Выбрать сценарий с учётом ограничений режима и fallback-правил."""
        state.scenario_type = self._scenario_selector.select(
            mode=state.mode,
            requested=requested,
            emotion_result=state.emotion_result,
        )
        return state

    def enforce_scenario_policy(self, state: SessionState) -> SessionState:
        """Применить policy к уже установленному сценарию."""
        if state.scenario_type in DISABLED_V1_SCENARIOS:
            state.scenario_type = self._scenario_selector.select_default(state.mode)
            return state
        return self.select_scenario(state, state.scenario_type)

    def increment_question_count(
        self, state: SessionState, *, contentful: bool = True
    ) -> SessionState:
        """Учесть очередной вопрос в рамках цикла."""
        state.question_count += 1
        if contentful:
            state.contentful_message_count += 1
        return state

    def register_insight(self, state: SessionState) -> SessionState:
        """Зарегистрировать инсайт пользователя в текущем цикле."""
        state.insight_count += 1
        return state

    def can_show_premium_trigger(
        self, state: SessionState, *, has_user_resistance: bool = False
    ) -> bool:
        """Проверить, можно ли безопасно показать Premium-trigger.

        Условия показа в v1:
        - режим Free;
        - минимум 1 инсайт;
        - минимум 5 содержательных сообщений;
        - нет кризиса и нет блокировки trigger со стороны safety;
        - нет сопротивления пользователя;
        - trigger ещё не показывался в текущем цикле.
        """
        if state.mode != SessionMode.FREE:
            return False
        if state.insight_count < 1:
            return False
        if state.contentful_message_count < 5:
            return False
        if state.premium_trigger_shown:
            return False
        if has_user_resistance:
            return False
        if state.safety_decision.must_block_premium_trigger:
            return False
        return state.safety_decision.level != SafetyLevel.CRISIS

    def mark_premium_trigger_shown(self, state: SessionState) -> SessionState:
        """Отметить, что Premium-trigger уже показан в текущем цикле."""
        state.premium_trigger_shown = True
        return state

    def can_start_deep_premium_session(self, state: SessionState) -> bool:
        """Проверить лимит глубокой Premium-сессии."""
        return (
            state.mode == SessionMode.PREMIUM and state.deep_session_allowed_today
        )

    def reserve_deep_premium_session(self, state: SessionState) -> SessionState:
        """Зарезервировать глубокую Premium-сессию на текущий день."""
        if not self.can_start_deep_premium_session(state):
            msg = "Лимит глубокой Premium-сессии на сегодня исчерпан."
            raise DeepSessionLimitError(msg)
        state.deep_session_allowed_today = False
        return state

    def is_completion_ready(self, state: SessionState) -> bool:
        """Проверить полноту критериев завершения сессии."""
        criteria = state.completion_criteria
        ready = (
            criteria.mechanism_formulated
            and criteria.influence_zone_defined
            and criteria.state_shift_confirmed
            and criteria.integration_summary_given
        )
        criteria.is_completed = ready
        return ready

    def finalize_if_ready(self, state: SessionState) -> SessionState:
        """Завершить сессию, если выполнены все критерии."""
        if not self.is_completion_ready(state):
            msg = "Критерии завершения сессии не выполнены."
            raise SessionCompletionError(msg)
        return self.transition_stage(state, SessionStage.COMPLETION)
