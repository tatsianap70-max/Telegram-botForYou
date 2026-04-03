"""Адаптивный движок коучинговой сессии.

Pipeline (фиксированный порядок):
1) SafetyPrecheckStage
2) EmotionDetectionStage
3) RequestTypeClassificationStage
4) TopicCycleGuardStage
5) ScenarioPolicyStage
6) AntiLoopStage
7) LoadLimiterStage
8) CompletionStage
9) ResponsePlannerStage
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field

from src.services.coaching.adaptive_session_rules import (
    APPROVE_SWITCH_MARKERS,
    CLARIFICATION_MARKERS,
    CLEAN_LANGUAGE_HINTS,
    CRISIS_MARKERS,
    DECLINE_SWITCH_MARKERS,
    ELEVATED_MARKERS,
    EMOTION_MARKERS,
    INFLUENCE_MARKERS,
    INSIGHT_MARKERS,
    LINEAR_STAGE_TRANSITIONS,
    MECHANISM_MARKERS,
    REQUEST_MARKERS,
    RESISTANCE_MARKERS,
    STATE_SHIFT_MARKERS,
    TO_REFLECTION_PATH,
    TOPIC_SWITCH_MARKERS,
    RequestType,
    build_topic_id,
    contains_any,
    has_progress_signal,
    is_contentful,
    is_low_engagement,
    normalize_text,
)
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
from src.services.coaching.session_manager import SessionManager


class EngineReplyType(StrEnum):
    """Тип ответа, который должен вернуть оркестратор."""

    COACH_MESSAGE = "coach_message"
    DECISION_PROMPT = "decision_prompt"
    CRISIS_PROTOCOL = "crisis_protocol"
    SESSION_COMPLETED = "session_completed"


class ResponsePlan(BaseModel):
    """Структурный план ответа без генерации текста."""

    stage: SessionStage
    scenario: ScenarioType
    request_type: RequestType
    step_type: str
    goals: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    template_key: str | None = None


class AdaptiveTurnResult(BaseModel):
    """Результат обработки одного пользовательского хода."""

    state: SessionState
    reply_type: EngineReplyType
    response_plan: ResponsePlan
    selected_scenario: ScenarioType
    request_type: RequestType
    should_offer_premium: bool = False


@dataclass(slots=True)
class _PipelineContext:
    normalized_text: str
    contentful: bool
    requested_scenario: ScenarioType | None
    internal_call: bool
    has_user_resistance: bool
    request_type: RequestType = RequestType.SITUATION
    step_type: str = "structured_progress"
    stop_processing: bool = False
    load_limiter_triggered: bool = False
    deep_path_allowed: bool = True


class AdaptiveSessionEngine:
    """Engine для выбора сценария и управления стадиями сессии."""

    def __init__(
        self,
        session_manager: SessionManager | None = None,
        confidence_threshold: float = 0.55,
        max_recent_steps: int = 3,
    ) -> None:
        if confidence_threshold <= 0 or confidence_threshold >= 1:
            msg = "Порог confidence_threshold должен быть в диапазоне (0, 1)."
            raise ValueError(msg)
        if max_recent_steps < 2:
            msg = "Параметр max_recent_steps должен быть >= 2."
            raise ValueError(msg)
        self._manager = session_manager or SessionManager()
        self._confidence_threshold = confidence_threshold
        self._max_recent_steps = max_recent_steps

    def process_turn(
        self,
        state: SessionState,
        user_text: str,
        *,
        requested_scenario: ScenarioType | None = None,
        internal_call: bool = False,
        has_user_resistance: bool = False,
    ) -> AdaptiveTurnResult:
        """Обработать вход пользователя по stage-пайплайну."""
        if not isinstance(state, SessionState):
            msg = "Аргумент state должен быть объектом SessionState."
            raise TypeError(msg)

        normalized = normalize_text(user_text)
        if not normalized:
            msg = "Сообщение пользователя не может быть пустым."
            raise ValueError(msg)

        context = _PipelineContext(
            normalized_text=normalized,
            contentful=is_contentful(normalized),
            requested_scenario=requested_scenario,
            internal_call=internal_call,
            has_user_resistance=has_user_resistance,
        )

        self._manager.increment_question_count(state, contentful=context.contentful)
        self._safety_precheck_stage(state, context)
        self._emotion_detection_stage(state, context)
        self._request_type_classification_stage(state, context)
        self._topic_cycle_guard_stage(state, context)
        self._scenario_policy_stage(state, context)
        self._anti_loop_stage(state, context)
        self._load_limiter_stage(state, context)
        self._completion_stage(state, context)
        return self._response_planner_stage(state, context)

    def process_user_message(
        self,
        state: SessionState,
        user_text: str,
        *,
        requested_scenario: ScenarioType | None = None,
        has_user_resistance: bool = False,
    ) -> AdaptiveTurnResult:
        """Backwards-compatible обертка: requested_scenario игнорируется."""
        return self.process_turn(
            state,
            user_text,
            requested_scenario=requested_scenario,
            internal_call=False,
            has_user_resistance=has_user_resistance,
        )

    def _safety_precheck_stage(
        self, state: SessionState, context: _PipelineContext
    ) -> None:
        state.safety_decision = self._analyze_safety(context.normalized_text)
        if state.safety_decision.must_switch_to_crisis_protocol:
            self._manager.transition_stage(state, SessionStage.CRISIS)
            context.step_type = "crisis_protocol"
            context.deep_path_allowed = False
            context.stop_processing = True

    def _emotion_detection_stage(
        self, state: SessionState, context: _PipelineContext
    ) -> None:
        if context.stop_processing:
            return
        emotion = self._analyze_emotion(context.normalized_text, state.emotion_result)
        self._manager.update_emotion(state, emotion)

    def _request_type_classification_stage(
        self,
        state: SessionState,
        context: _PipelineContext,
    ) -> None:
        if context.stop_processing:
            return
        context.request_type = self._classify_request_type(
            context.normalized_text,
            state.emotion_result,
        )

    def _topic_cycle_guard_stage(
        self, state: SessionState, context: _PipelineContext
    ) -> None:
        if context.stop_processing:
            return

        if state.stage == SessionStage.DECISION_NODE:
            if contains_any(context.normalized_text, APPROVE_SWITCH_MARKERS):
                self._manager.apply_topic_switch_decision(
                    state,
                    approve_switch=True,
                    new_cycle_id=f"cycle-{uuid4().hex[:12]}",
                )
                context.step_type = "topic_switch_approved"
                return
            if contains_any(context.normalized_text, DECLINE_SWITCH_MARKERS):
                self._manager.apply_topic_switch_decision(state, approve_switch=False)
                context.step_type = "topic_switch_declined"
                return
            context.step_type = "topic_switch_decision_prompt"
            context.stop_processing = True
            return

        if state.topic_id is None:
            self._manager.set_topic(state, build_topic_id(context.normalized_text))
            return

        if contains_any(context.normalized_text, TOPIC_SWITCH_MARKERS):
            new_topic_id = build_topic_id(context.normalized_text)
            if new_topic_id != state.topic_id:
                self._manager.request_topic_switch(state, new_topic_id)
                context.step_type = "topic_switch_decision_prompt"
                context.stop_processing = True

    def _scenario_policy_stage(
        self, state: SessionState, context: _PipelineContext
    ) -> None:
        if context.stop_processing:
            return

        requested = context.requested_scenario if context.internal_call else None
        preferred = self._choose_policy_scenario(state, context, requested)
        self._manager.select_scenario(state, preferred)

    def _anti_loop_stage(self, state: SessionState, context: _PipelineContext) -> None:
        if context.stop_processing:
            return

        if self._is_early_turn(state):
            if state.emotion_result.state == EmotionState.CONFUSION:
                self._set_step_type(state, context, "state_clarification")
                return
            if self._is_early_emotional_state(state.emotion_result.state):
                self._set_step_type(state, context, "emotion_contact")
                return

        step_type = self._derive_step_type(state, context.request_type)
        if (
            self._is_early_turn(state)
            and state.stage in {SessionStage.TOPIC_DEFINITION, SessionStage.TENSION_REDUCTION}
            and step_type == "clarity_structuring"
        ):
            step_type = "state_clarification"
        recent = state.recent_step_types[-self._max_recent_steps :]
        if recent and recent[-1] == step_type:
            step_type = self._alternate_step_type(step_type)

        progress = has_progress_signal(context.normalized_text)
        state.no_progress_turns = 0 if progress else state.no_progress_turns + 1
        if state.no_progress_turns >= 2:
            step_type = self._alternate_step_type(step_type)

        state.recent_step_types = [*recent, step_type][-self._max_recent_steps :]
        context.step_type = step_type

    def _load_limiter_stage(
        self, state: SessionState, context: _PipelineContext
    ) -> None:
        if context.stop_processing:
            return

        if self._is_early_turn(state):
            state.low_engagement_turns = 0
            context.load_limiter_triggered = False
            state.load_limiter_active = False
            return

        low_engagement = is_low_engagement(context.normalized_text, context.contentful)
        state.low_engagement_turns = (
            state.low_engagement_turns + 1 if low_engagement else 0
        )
        context.load_limiter_triggered = state.low_engagement_turns >= 2
        state.load_limiter_active = context.load_limiter_triggered

        if context.load_limiter_triggered:
            context.deep_path_allowed = False
            if state.scenario_type == ScenarioType.B_CLEAN_LANGUAGE:
                self._manager.select_scenario(state, ScenarioType.A_STRUCTURED)

    def _completion_stage(self, state: SessionState, context: _PipelineContext) -> None:
        if state.stage in {
            SessionStage.CRISIS,
            SessionStage.DECISION_NODE,
            SessionStage.COMPLETION,
        }:
            return

        if self._should_hold_first_turn_stage(state, context):
            return

        if self._is_early_turn(state) and state.stage in {
            SessionStage.TOPIC_DEFINITION,
            SessionStage.TENSION_REDUCTION,
            SessionStage.MECHANISM_DISCOVERY,
        }:
            self._advance_stage(state, allow_deep=False)
            return

        self._update_completion_flags(state, context.normalized_text)
        if contains_any(context.normalized_text, INSIGHT_MARKERS):
            self._manager.register_insight(state)

        if context.load_limiter_triggered:
            self._move_to_reflection_summary(state)
        else:
            self._advance_stage(state, allow_deep=context.deep_path_allowed)

        if state.stage == SessionStage.REFLECTION_SUMMARY:
            state.completion_criteria.integration_summary_given = True
            if self._manager.is_completion_ready(state):
                self._manager.finalize_if_ready(state)
            elif context.load_limiter_triggered:
                state.completion_criteria.is_completed = True
                state.stage = SessionStage.COMPLETION

    @staticmethod
    def _should_hold_first_turn_stage(
        state: SessionState,
        context: _PipelineContext,
    ) -> bool:
        """Удержать ранние ходы в безопасном входном режиме."""
        if not AdaptiveSessionEngine._is_early_turn(state):
            return False

        is_early_entry_stage = state.stage in {
            SessionStage.TOPIC_DEFINITION,
            SessionStage.TENSION_REDUCTION,
        }
        if is_early_entry_stage and context.step_type == "clarity_structuring":
            context.step_type = "state_clarification"

        needs_clarification = context.step_type == "state_clarification" or (
            state.emotion_result.state == EmotionState.CONFUSION
        )
        if needs_clarification:
            context.step_type = "state_clarification"
            context.deep_path_allowed = False
            if is_early_entry_stage and state.stage != SessionStage.TOPIC_DEFINITION:
                state.stage = SessionStage.TOPIC_DEFINITION
            return True

        if (
            context.step_type == "emotion_contact"
            or AdaptiveSessionEngine._is_early_emotional_state(
                state.emotion_result.state
            )
        ):
            context.step_type = "emotion_contact"
            context.deep_path_allowed = False
            if is_early_entry_stage and state.stage != SessionStage.TOPIC_DEFINITION:
                state.stage = SessionStage.TOPIC_DEFINITION
            return True
        return False

    def _set_step_type(
        self,
        state: SessionState,
        context: _PipelineContext,
        step_type: str,
    ) -> None:
        """Установить тип шага и синхронизировать anti-loop состояние."""
        context.step_type = step_type
        state.no_progress_turns = 0
        recent = state.recent_step_types[-self._max_recent_steps :]
        state.recent_step_types = [*recent, step_type][-self._max_recent_steps :]

    @staticmethod
    def _is_early_emotional_state(emotion_state: EmotionState) -> bool:
        """Эмоции для мягкого входа на первых ходах."""
        return emotion_state in {
            EmotionState.ANXIETY,
            EmotionState.DECISION_FEAR,
            EmotionState.OVERLOAD,
        }

    @staticmethod
    def _is_clarification_input(normalized_text: str) -> bool:
        """Определить запрос на переформулировку/упрощение вопроса."""
        return contains_any(normalized_text, CLARIFICATION_MARKERS)

    @staticmethod
    def _is_uncertainty_input(normalized_text: str) -> bool:
        """Определить ранний вход неопределенности/растерянности."""
        uncertainty_markers = ("не знаю", "запутал", "запуталась", "неясно")
        return contains_any(normalized_text, uncertainty_markers)

    @staticmethod
    def _is_early_turn(state: SessionState) -> bool:
        """Ранние ходы для мягкого входа: первые 3 сообщения цикла."""
        return state.question_count <= 3

    def _response_planner_stage(
        self,
        state: SessionState,
        context: _PipelineContext,
    ) -> AdaptiveTurnResult:
        should_offer = self._should_offer_premium(state, context)
        if should_offer:
            self._manager.mark_premium_trigger_shown(state)

        reply_type = EngineReplyType.COACH_MESSAGE
        template_key = None
        if state.stage == SessionStage.CRISIS:
            reply_type = EngineReplyType.CRISIS_PROTOCOL
            template_key = (
                state.safety_decision.message_template_key or "crisis_support_v1"
            )
            context.step_type = "crisis_protocol"
        elif state.stage == SessionStage.DECISION_NODE:
            reply_type = EngineReplyType.DECISION_PROMPT
            context.step_type = "topic_switch_decision_prompt"
        elif state.stage == SessionStage.COMPLETION:
            reply_type = EngineReplyType.SESSION_COMPLETED
            context.step_type = "completion_wrap"

        plan = ResponsePlan(
            stage=state.stage,
            scenario=state.scenario_type,
            request_type=context.request_type,
            step_type=context.step_type,
            goals=self._build_goals(state.stage, context.request_type),
            constraints=self._build_constraints(state, context),
            template_key=template_key,
        )
        return AdaptiveTurnResult(
            state=state,
            reply_type=reply_type,
            response_plan=plan,
            selected_scenario=state.scenario_type,
            request_type=context.request_type,
            should_offer_premium=should_offer,
        )

    def _choose_policy_scenario(
        self,
        state: SessionState,
        context: _PipelineContext,
        requested: ScenarioType | None,
    ) -> ScenarioType:
        if (
            state.emotion_result.fallback_to_clarity
            or not state.emotion_result.is_confident
            or context.request_type in {RequestType.OVERLOAD, RequestType.CONFUSION}
            or state.no_progress_turns >= 2
        ):
            return ScenarioType.CLARITY_FALLBACK
        if requested is not None:
            return requested
        if state.mode == SessionMode.FREE:
            return ScenarioType.A_STRUCTURED
        if contains_any(context.normalized_text, CLEAN_LANGUAGE_HINTS):
            return ScenarioType.B_CLEAN_LANGUAGE
        return ScenarioType.A_STRUCTURED

    def _advance_stage(self, state: SessionState, *, allow_deep: bool) -> None:
        if state.stage in {
            SessionStage.CRISIS,
            SessionStage.COMPLETION,
            SessionStage.DECISION_NODE,
        }:
            return
        if state.stage == SessionStage.PATTERN_HIGHLIGHT:
            if (
                allow_deep
                and state.scenario_type == ScenarioType.B_CLEAN_LANGUAGE
                and self._manager.can_start_deep_premium_session(state)
            ):
                self._manager.reserve_deep_premium_session(state)
                self._manager.transition_stage(state, SessionStage.DEEP_ANALYSIS)
            else:
                self._manager.transition_stage(state, SessionStage.REFLECTION_SUMMARY)
            return
        next_stage = LINEAR_STAGE_TRANSITIONS.get(state.stage)
        if next_stage is not None:
            self._manager.transition_stage(state, next_stage)

    def _move_to_reflection_summary(self, state: SessionState) -> None:
        while state.stage not in {
            SessionStage.REFLECTION_SUMMARY,
            SessionStage.CRISIS,
            SessionStage.COMPLETION,
        }:
            next_stage = TO_REFLECTION_PATH.get(state.stage)
            if next_stage is None:
                break
            self._manager.transition_stage(state, next_stage)

    def _analyze_safety(self, normalized_text: str) -> SafetyDecision:
        if contains_any(normalized_text, CRISIS_MARKERS):
            return SafetyDecision(
                level=SafetyLevel.CRISIS,
                can_continue=False,
                must_switch_to_crisis_protocol=True,
                must_block_premium_trigger=True,
                reason_codes=["crisis_marker_detected"],
                message_template_key="crisis_support_v1",
            )
        if contains_any(normalized_text, ELEVATED_MARKERS):
            return SafetyDecision(
                level=SafetyLevel.ELEVATED,
                can_continue=True,
                must_switch_to_crisis_protocol=False,
                must_block_premium_trigger=False,
                reason_codes=["elevated_distress_marker_detected"],
                message_template_key="elevated_support_v1",
            )
        return SafetyDecision()

    def _analyze_emotion(
        self,
        normalized_text: str,
        previous: EmotionDetectionResult,
    ) -> EmotionDetectionResult:
        scores: dict[EmotionState, int] = {}
        for emotion, markers in EMOTION_MARKERS.items():
            count = sum(1 for marker in markers if marker in normalized_text)
            if count:
                scores[emotion] = count

        if not scores:
            overload_markers = ("перегруз", "не тяну", "давит", "слишком много")
            if contains_any(normalized_text, overload_markers):
                return EmotionDetectionResult(
                    state=EmotionState.OVERLOAD,
                    confidence=0.7,
                    is_confident=True,
                    fallback_to_clarity=False,
                    clarification_attempts=0,
                )

            anxiety_markers = (
                "тревог",
                "боюсь",
                "страш",
                "волнуюсь",
                "волнует",
                "переживаю",
                "тревожусь",
                "груст",
            )
            if contains_any(normalized_text, anxiety_markers):
                return EmotionDetectionResult(
                    state=EmotionState.ANXIETY,
                    confidence=0.7,
                    is_confident=True,
                    fallback_to_clarity=False,
                    clarification_attempts=0,
                )

            if self._is_clarification_input(
                normalized_text
            ) or self._is_uncertainty_input(normalized_text):
                return EmotionDetectionResult(
                    state=EmotionState.CONFUSION,
                    confidence=0.65,
                    is_confident=True,
                    fallback_to_clarity=False,
                    clarification_attempts=0,
                )
            attempts = previous.clarification_attempts + 1
            return EmotionDetectionResult(
                state=EmotionState.UNKNOWN,
                confidence=0.0,
                is_confident=False,
                fallback_to_clarity=True,
                clarification_attempts=attempts,
            )

        best = max(scores, key=scores.__getitem__)
        confidence = min(0.5 + scores[best] * 0.15, 0.9)
        is_confident = confidence >= self._confidence_threshold
        attempts = 0 if is_confident else previous.clarification_attempts + 1
        return EmotionDetectionResult(
            state=best,
            confidence=confidence,
            is_confident=is_confident,
            fallback_to_clarity=(not is_confident) or attempts >= 2,
            clarification_attempts=attempts,
            markers=[m for m in EMOTION_MARKERS[best] if m in normalized_text],
        )

    @staticmethod
    def _classify_request_type(
        normalized_text: str,
        emotion: EmotionDetectionResult,
    ) -> RequestType:
        if emotion.state == EmotionState.OVERLOAD or contains_any(
            normalized_text, REQUEST_MARKERS["overload"]
        ):
            return RequestType.OVERLOAD
        if emotion.state == EmotionState.CONFUSION or contains_any(
            normalized_text, REQUEST_MARKERS["confusion"]
        ):
            return RequestType.CONFUSION
        if contains_any(normalized_text, REQUEST_MARKERS["pattern"]):
            return RequestType.REPEATING_PATTERN
        if contains_any(normalized_text, REQUEST_MARKERS["decision"]):
            return RequestType.DECISION_CHOICE
        if contains_any(normalized_text, REQUEST_MARKERS["calm"]):
            return RequestType.CALM_REFLECTION
        return RequestType.SITUATION

    @staticmethod
    def _update_completion_flags(state: SessionState, normalized_text: str) -> None:
        if contains_any(normalized_text, MECHANISM_MARKERS):
            state.completion_criteria.mechanism_formulated = True
        if contains_any(normalized_text, INFLUENCE_MARKERS):
            state.completion_criteria.influence_zone_defined = True
        if contains_any(normalized_text, STATE_SHIFT_MARKERS):
            state.completion_criteria.state_shift_confirmed = True

    @staticmethod
    def _derive_step_type(state: SessionState, request_type: RequestType) -> str:
        if state.scenario_type == ScenarioType.CLARITY_FALLBACK:
            return "clarity_structuring"
        if request_type == RequestType.DECISION_CHOICE:
            return "decision_structuring"
        if request_type == RequestType.REPEATING_PATTERN:
            return "pattern_interrupt"
        if request_type == RequestType.OVERLOAD:
            return "tension_regulation"
        if state.scenario_type == ScenarioType.B_CLEAN_LANGUAGE:
            return "clean_language_probe"
        return "structured_progress"

    @staticmethod
    def _alternate_step_type(step_type: str) -> str:
        mapping = {
            "structured_progress": "pattern_interrupt",
            "pattern_interrupt": "clarity_structuring",
            "clarity_structuring": "decision_structuring",
            "decision_structuring": "structured_progress",
            "clean_language_probe": "structured_progress",
            "tension_regulation": "clarity_structuring",
        }
        return mapping.get(step_type, "structured_progress")

    def _should_offer_premium(
        self, state: SessionState, context: _PipelineContext
    ) -> bool:
        resistance = context.has_user_resistance or contains_any(
            context.normalized_text,
            RESISTANCE_MARKERS,
        )
        clarity_overload_active = (
            state.scenario_type == ScenarioType.CLARITY_FALLBACK
            and (
                context.request_type == RequestType.OVERLOAD
                or state.emotion_result.state == EmotionState.OVERLOAD
                or context.load_limiter_triggered
            )
        )
        return (
            state.mode == SessionMode.FREE
            and state.safety_decision.level != SafetyLevel.CRISIS
            and not state.premium_trigger_shown
            and state.contentful_message_count >= 5
            and state.insight_count >= 1
            and not resistance
            and not clarity_overload_active
        )

    @staticmethod
    def _build_goals(stage: SessionStage, request_type: RequestType) -> list[str]:
        goals = [
            f"обработать запрос типа: {request_type.value}",
            "сохранить структуру цикла",
        ]
        if stage == SessionStage.REFLECTION_SUMMARY:
            goals.append("сделать мягкий reflective exit")
        if stage == SessionStage.COMPLETION:
            goals.append("завершить цикл одной темы")
        return goals

    @staticmethod
    def _build_constraints(state: SessionState, context: _PipelineContext) -> list[str]:
        constraints = ["не генерировать финальный текст внутри engine"]
        if state.mode == SessionMode.FREE:
            constraints.append(
                "в Free разрешены только A_STRUCTURED и CLARITY_FALLBACK"
            )
        if context.load_limiter_triggered:
            constraints.append("deep-path отключен на этом ходу")
        if state.stage == SessionStage.CRISIS:
            constraints.append("использовать кризисный протокол")
        return constraints
