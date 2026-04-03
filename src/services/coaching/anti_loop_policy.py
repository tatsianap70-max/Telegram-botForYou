"""Внутренняя policy-логика anti-loop без интеграций и side effects."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from src.services.coaching.anti_loop_types import (
    MAX_REPEAT_WITHOUT_PROGRESS,
    AntiLoopDecision,
    LoopSignals,
    RecoveryStrategy,
    SignalStrength,
    _SignalFlags,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.services.coaching.contracts.session import SessionState


_SHORT_REPEAT_MARKERS: tuple[str, ...] = (
    "не знаю",
    "все то же самое",
    "всё то же самое",
    "ничего не меняется",
    "без изменений",
    "никак",
)

_CLARIFICATION_MARKERS: tuple[str, ...] = (
    "не поняла",
    "не понимаю",
    "не поняла вопрос",
    "не понимаю вопрос",
    "переформулируйте",
    "объясните иначе",
)

_STAGNATION_MARKERS: tuple[str, ...] = (
    "ничего не меняется",
    "все то же самое",
    "всё то же самое",
    "снова то же",
    "без изменений",
)

_FACT_MARKERS: tuple[str, ...] = (
    "когда",
    "после",
    "сегодня",
    "вчера",
    "на работе",
    "дома",
    "в разговоре",
    "в ситуации",
)

_INSIGHT_MARKERS: tuple[str, ...] = (
    "я понял",
    "я поняла",
    "понимаю",
    "заметил",
    "заметила",
    "осознал",
    "осознала",
    "связь",
)

_STEP_MARKERS: tuple[str, ...] = (
    "сделаю",
    "попробую",
    "выберу",
    "решил",
    "решила",
    "мой шаг",
    "следующий шаг",
    "начну",
)

_STOP_WORDS: frozenset[str] = frozenset(
    {
        "и",
        "в",
        "на",
        "но",
        "а",
        "что",
        "это",
        "мне",
        "меня",
        "как",
        "уже",
        "очень",
        "тоже",
        "все",
        "всё",
    }
)


def detect_loop_signals(
    state: SessionState,
    recent_user_turns: Sequence[str] | None = None,
) -> LoopSignals:
    """Вычислить loop/stagnation сигналы по простым policy-правилам."""
    turns = _prepare_turns(recent_user_turns)
    repeated_user_angle = _is_repeated_user_angle(turns)
    repeated_system_angle = _is_repeated_system_angle(
        _state_list(state, "recent_step_types")
    )
    short_repetitive_answers = _has_short_repetitive_answers(turns)
    no_new_information = _has_no_new_information(turns)
    has_new_fact, has_new_insight, has_new_step = _detect_progress_novelty(turns)
    no_progress = not (has_new_fact or has_new_insight or has_new_step)

    no_progress_turns = _state_int(state, "no_progress_turns")
    no_progress_repeated = no_progress_turns >= 2
    stagnation_detected = no_progress_repeated and no_progress
    if not stagnation_detected and no_progress and no_new_information:
        stagnation_detected = repeated_user_angle or repeated_system_angle
    loop_detected = (
        repeated_user_angle
        or repeated_system_angle
        or (short_repetitive_answers and stagnation_detected)
    )

    flags = _SignalFlags(
        loop_detected=loop_detected,
        stagnation_detected=stagnation_detected,
        repeated_user_angle=repeated_user_angle,
        repeated_system_angle=repeated_system_angle,
        short_repetitive_answers=short_repetitive_answers,
        no_new_information=no_new_information,
        no_progress=no_progress,
    )
    signal_strength = _derive_signal_strength(state, flags)
    reason = _derive_reason(flags)
    evidence_count = _count_evidence(flags)

    return LoopSignals(
        loop_detected=loop_detected,
        stagnation_detected=stagnation_detected,
        repeated_user_angle=repeated_user_angle,
        repeated_system_angle=repeated_system_angle,
        short_repetitive_answers=short_repetitive_answers,
        no_new_information=no_new_information,
        has_new_fact=has_new_fact,
        has_new_insight=has_new_insight,
        has_new_step=has_new_step,
        no_progress=no_progress,
        signal_strength=signal_strength,
        reason=reason,
        evidence_count=evidence_count,
    )


def evaluate_anti_loop(
    state: SessionState,
    recent_user_turns: Sequence[str] | None = None,
) -> AntiLoopDecision:
    """Вернуть структурированное решение anti-loop для текущего хода."""
    if _topic_switch_pending(state):
        return _build_no_loop_decision("topic_switch_pending")

    turns = _prepare_turns(recent_user_turns)
    if len(turns) < 2:
        return _build_no_loop_decision("insufficient_context")
    if _is_clarification_chain(turns):
        return _build_no_loop_decision("clarification_chain_guard")
    if _is_early_uncertainty_entry(state, turns):
        return _build_no_loop_decision("early_uncertainty_entry")
    if _is_early_turn(state):
        return _build_no_loop_decision("early_turn_guard")
    if not _has_minimum_context(state, turns):
        return _build_no_loop_decision("insufficient_context")

    signals = detect_loop_signals(state, turns)
    if signals.signal_strength is SignalStrength.NONE:
        return _build_no_loop_decision("none")
    if (
        signals.signal_strength is SignalStrength.WEAK
        and signals.evidence_count < 2
        and not (signals.repeated_user_angle or signals.repeated_system_angle)
    ):
        return _build_no_loop_decision("weak_single_signal")

    strategy = select_recovery_strategy(state, signals)
    return AntiLoopDecision(
        loop_detected=signals.loop_detected,
        signal_strength=signals.signal_strength,
        strategy=strategy,
        should_soft_close=strategy is RecoveryStrategy.SOFT_CLOSE,
        reason=signals.reason,
        stagnation_detected=signals.stagnation_detected,
        requires_intervention=strategy is not RecoveryStrategy.NONE,
    )


def select_recovery_strategy(
    state: SessionState,
    signals: LoopSignals,
) -> RecoveryStrategy:
    """Выбрать следующий тип сдвига по фиксированному приоритету.

    Порядок выбора:
    ANGLE_SHIFT -> SIMPLIFY -> META_QUESTION -> INSIGHT_TRIGGER -> SOFT_CLOSE.

    Отдельно применяется max repeat policy: при превышении лимита сразу SOFT_CLOSE.
    """
    soft_close_now = should_soft_close(state, signals)

    if signals.repeated_system_angle and not soft_close_now:
        return RecoveryStrategy.ANGLE_SHIFT

    if (
        signals.short_repetitive_answers
        and signals.stagnation_detected
        and not soft_close_now
    ):
        return RecoveryStrategy.SIMPLIFY

    if signals.repeated_user_angle and not soft_close_now:
        return RecoveryStrategy.META_QUESTION

    if signals.stagnation_detected and not soft_close_now:
        return RecoveryStrategy.INSIGHT_TRIGGER

    if soft_close_now:
        return RecoveryStrategy.SOFT_CLOSE

    return RecoveryStrategy.NONE


def should_soft_close(state: SessionState, signals: LoopSignals) -> bool:
    """Определить мягкое завершение при устойчивом отсутствии прогресса."""
    no_progress_turns = _state_int(state, "no_progress_turns")
    return bool(
        no_progress_turns >= MAX_REPEAT_WITHOUT_PROGRESS
        and signals.signal_strength is SignalStrength.CLEAR
        and _has_confirmed_repeat_or_stagnation(signals)
    )


def _prepare_turns(recent_user_turns: Sequence[str] | None) -> list[str]:
    """Подготовить последние 2-3 user-turn для детекции динамики."""
    if not recent_user_turns:
        return []

    prepared: list[str] = []
    for text in recent_user_turns:
        normalized = re.sub(r"\s+", " ", text.strip().lower())
        if normalized:
            prepared.append(normalized)
    return prepared[-3:]


def _has_minimum_context(state: SessionState, turns: Sequence[str]) -> bool:
    """Проверить минимальный порог контекста до срабатывания anti-loop."""
    if len(turns) < 2:
        return False
    recent_step_types = _state_list(state, "recent_step_types")
    if len(recent_step_types) < 1:
        return False

    reflective_steps = sum(1 for step in recent_step_types[-3:] if step)
    if reflective_steps < 1:
        return False

    # Срабатывание должно опираться на повторяемость, а не единичный сбой.
    no_progress_turns = _state_int(state, "no_progress_turns")
    has_repeat_signal = no_progress_turns >= 2 or _is_repeated_user_angle(turns)
    return has_repeat_signal or _is_repeated_system_angle(recent_step_types)


def _is_early_uncertainty_entry(state: SessionState, turns: Sequence[str]) -> bool:
    """Блокировать anti-loop на раннем входе «запуталась/не знаю»."""
    if _state_int(state, "question_count") > 3:
        return False
    if len(turns) < 2:
        return False
    if _has_minimal_user_context(turns):
        return False

    uncertainty_markers = ("не знаю", "запутал", "не понимаю", "неясно")
    previous_turn = turns[-2]
    latest_turn = turns[-1]

    previous_uncertain = (
        _contains_any(previous_turn, uncertainty_markers)
        or _is_trivial_short_turn(previous_turn)
    )
    latest_uncertain = (
        _contains_any(latest_turn, uncertainty_markers)
        or _is_trivial_short_turn(latest_turn)
    )
    return previous_uncertain and latest_uncertain


def _is_early_turn(state: SessionState) -> bool:
    """Ранние ходы (T1-T3), где anti-loop выключен policy-правилом."""
    return _state_int(state, "question_count") <= 3


def _is_clarification_chain(turns: Sequence[str]) -> bool:
    """Определить clarification-цепочку для блокировки anti-loop."""
    if not turns:
        return False
    latest = turns[-1]
    if _contains_any(latest, _CLARIFICATION_MARKERS):
        return True
    if len(turns) >= 2:
        previous = turns[-2]
        previous_is_clarification = _contains_any(previous, _CLARIFICATION_MARKERS)
        return previous_is_clarification and _is_trivial_short_turn(latest)
    return False


def _has_minimal_user_context(turns: Sequence[str]) -> bool:
    """Проверить появление минимального контекста (факт/инсайт/шаг)."""
    has_new_fact, has_new_insight, has_new_step = _detect_progress_novelty(turns)
    return has_new_fact or has_new_insight or has_new_step


def _exceeded_max_repeat_policy(state: SessionState) -> bool:
    """Ограничить число повторов без прогресса (max repeat policy)."""
    no_progress_turns = _state_int(state, "no_progress_turns")
    return no_progress_turns >= MAX_REPEAT_WITHOUT_PROGRESS


def _derive_signal_strength(state: SessionState, flags: _SignalFlags) -> SignalStrength:
    """Определить силу сигнала: NONE / WEAK / CLEAR."""
    if not flags.loop_detected and not flags.stagnation_detected:
        return SignalStrength.NONE

    evidence_count = _count_evidence(flags)
    no_progress_turns = _state_int(state, "no_progress_turns")
    clear_condition = (
        (_exceeded_max_repeat_policy(state) and evidence_count >= 2)
        or (flags.repeated_system_angle and flags.repeated_user_angle)
        or (
            flags.stagnation_detected
            and flags.no_new_information
            and (flags.repeated_user_angle or flags.short_repetitive_answers)
        )
        or (no_progress_turns >= 3 and evidence_count >= 2)
    )
    if clear_condition:
        return SignalStrength.CLEAR
    return SignalStrength.WEAK


def _derive_reason(flags: _SignalFlags) -> str:
    """Выбрать первичную reason-метку для решения anti-loop."""
    if flags.repeated_system_angle:
        return "repeated_system_angle"
    if flags.repeated_user_angle:
        return "repeated_user_angle"
    if flags.no_new_information:
        return "no_new_information"
    if flags.stagnation_detected:
        return "stagnation_detected"
    return "none"


def _is_repeated_system_angle(step_types: Sequence[str]) -> bool:
    """Проверить повтор одного и того же system-angle на последних шагах."""
    recent_steps = [step for step in step_types if step][-3:]
    if len(recent_steps) < 2:
        return False
    if recent_steps[-1] == recent_steps[-2]:
        return True
    return len(recent_steps) == 3 and len(set(recent_steps)) == 1


def _is_repeated_user_angle(turns: Sequence[str]) -> bool:
    """Проверить повтор пользовательской мысли на 2-3 последних ходах."""
    if len(turns) < 2:
        return False

    if _is_trivial_short_turn(turns[-1]) and _is_trivial_short_turn(turns[-2]):
        return False

    if _similarity(turns[-1], turns[-2]) >= 0.8:
        return True

    if len(turns) < 3:
        return False

    return (
        _similarity(turns[-1], turns[-2]) >= 0.72
        and _similarity(turns[-2], turns[-3]) >= 0.72
    )


def _has_short_repetitive_answers(turns: Sequence[str]) -> bool:
    """Выявить короткие повторяющиеся ответы без семантической модели."""
    if len(turns) < 2:
        return False

    recent = turns[-2:]
    if recent[0] == recent[1] and len(recent[1].split()) <= 4:
        return True

    return any(text in _SHORT_REPEAT_MARKERS for text in recent)


def _has_no_new_information(turns: Sequence[str]) -> bool:
    """Проверить отсутствие новой информации между последними ходами."""
    if len(turns) < 2:
        return False

    if any(marker in turns[-1] for marker in _STAGNATION_MARKERS):
        return True

    latest_tokens = _tokenize(turns[-1])
    previous_tokens = _tokenize(turns[-2])
    if not latest_tokens or not previous_tokens:
        return False

    has_new_tokens = bool(latest_tokens - previous_tokens)
    if has_new_tokens:
        return False

    return _similarity(turns[-1], turns[-2]) >= 0.75


def _detect_progress_novelty(turns: Sequence[str]) -> tuple[bool, bool, bool]:
    """Определить появление нового факта, инсайта и шага в последнем ходе."""
    if len(turns) < 2:
        return (False, False, False)

    latest = turns[-1]
    previous = turns[-2]
    has_new_tokens = bool(_tokenize(latest) - _tokenize(previous))

    has_new_fact = _contains_any(latest, _FACT_MARKERS) and has_new_tokens
    has_new_insight = _contains_any(latest, _INSIGHT_MARKERS) and has_new_tokens
    has_new_step = _contains_any(latest, _STEP_MARKERS) and has_new_tokens
    return (has_new_fact, has_new_insight, has_new_step)


def _contains_any(text: str, markers: Sequence[str]) -> bool:
    """Проверить наличие любого маркера в тексте."""
    return any(marker in text for marker in markers)


def _is_trivial_short_turn(text: str) -> bool:
    """Проверить, что реплика слишком короткая для вывода о loop."""
    normalized = text.strip().lower()
    return bool(normalized in _SHORT_REPEAT_MARKERS or len(normalized.split()) <= 2)


def _count_evidence(flags: _SignalFlags) -> int:
    """Подсчитать количество независимых признаков loop/stagnation."""
    return sum(
        (
            flags.repeated_user_angle,
            flags.repeated_system_angle,
            flags.short_repetitive_answers and flags.stagnation_detected,
            flags.no_new_information,
            flags.stagnation_detected,
            flags.no_progress,
        )
    )


def _has_confirmed_repeat_or_stagnation(signals: LoopSignals) -> bool:
    """Проверить подтверждённость repeat/stagnation несколькими сигналами."""
    return bool(
        (
            signals.repeated_user_angle
            or signals.repeated_system_angle
            or signals.stagnation_detected
        )
        and signals.evidence_count >= 2
    )


def _build_no_loop_decision(reason: str) -> AntiLoopDecision:
    """Собрать нейтральное решение без anti-loop вмешательства."""
    return AntiLoopDecision(
        loop_detected=False,
        signal_strength=SignalStrength.NONE,
        strategy=RecoveryStrategy.NONE,
        should_soft_close=False,
        reason=reason,
        stagnation_detected=False,
        requires_intervention=False,
    )


def _state_int(state: SessionState, field: str, default: int = 0) -> int:
    """Безопасно получить целочисленное поле из state."""
    value = getattr(state, field, default)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    return default


def _state_list(
    state: SessionState,
    field: str,
    default: Sequence[str] = (),
) -> list[str]:
    """Безопасно получить список строк из state."""
    value = getattr(state, field, default)
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    if isinstance(value, tuple):
        return [item for item in value if isinstance(item, str)]
    return list(default)


def _similarity(first: str, second: str) -> float:
    """Вычислить простую token-based схожесть (Jaccard)."""
    first_tokens = _tokenize(first)
    second_tokens = _tokenize(second)
    if not first_tokens and not second_tokens:
        return 1.0
    if not first_tokens or not second_tokens:
        return 0.0

    intersection = len(first_tokens & second_tokens)
    union = len(first_tokens | second_tokens)
    return intersection / union


def _tokenize(text: str) -> set[str]:
    """Разбить текст на токены для policy-сравнения."""
    words = re.findall(r"[A-Za-zА-Яа-яЁё0-9-]+", text.lower())
    return {word for word in words if len(word) > 2 and word not in _STOP_WORDS}


def _topic_switch_pending(state: SessionState) -> bool:
    """Проверить, ожидается ли переход к новой теме через decision node."""
    topic_id = getattr(state, "topic_id", None)
    pending_topic_id = getattr(state, "pending_topic_id", None)
    return bool(topic_id and pending_topic_id and pending_topic_id != topic_id)


__all__ = [
    "detect_loop_signals",
    "evaluate_anti_loop",
    "select_recovery_strategy",
    "should_soft_close",
]
