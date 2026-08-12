"""Движок чередующихся предложений — сердце стенда.

Протокол сессии (дизайн-док §4.2):

1. разыгрывается сценарий $(v,c)$, роли назначаются;
2. каждой стороне выдаётся её конфигурация харнесса;
3. чередующиеся предложения, максимум $T$ раундов;
4. на каждом ходу агент выдаёт свободный текст плюс структурированное действие;
5. строгий парсер извлекает действие; невалидный вывод — до трёх повторов,
   затем сессия помечается техническим сбоем и исключается;
6. сделка при ``accept`` либо исчерпание раундов без сделки;
7. логируется всё: транскрипт, предложения, раунд соглашения, токены, латентность.

Учёт раундов: один раунд = по одному ходу каждой стороны. Соглашение в
раунде $t^*$ даёт аллокативную эффективность $E = \\delta^{t^*-1}$; отсутствие
сделки при $v > c$ — чистая потеря, $E = 0$.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from harness_asymmetry.agent import NegotiationAgent
from harness_asymmetry.config import HarnessConfig, RunnerConfig
from harness_asymmetry.harness.assembler import Harness, assemble_harness
from harness_asymmetry.harness.components.base import NegotiationState, SideIdentity
from harness_asymmetry.harness.components.commitment import opponent_commitment_block
from harness_asymmetry.harness.components.memory import MemoryStore, role_share
from harness_asymmetry.llm_client import LLMClient, LLMUnavailableError
from harness_asymmetry.observability import (
    EVENT_CIRCUIT_BREAKER,
    EVENT_SESSION_END,
    EVENT_SESSION_START,
    EVENT_STUCK,
    EVENT_TURN_END,
    CircuitBreaker,
    CircuitBreakerTripped,
    EventLog,
    TraceContext,
    new_id,
)
from harness_asymmetry.schemas import (
    Action,
    ActionType,
    HarnessVector,
    InfoRegime,
    MalformedActionError,
    Role,
    Scenario,
    SessionRecord,
    TurnRecord,
    anchor_reference,
    asymmetry,
    asymmetry_l1,
    rubinstein_price,
)


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SideSpec:
    """Конфигурация одной стороны на одну сессию."""

    side: str  # "A" | "B"
    role: Role
    vector: HarnessVector
    llm: LLMClient
    party_id: str
    memory_store: MemoryStore | None = None


@dataclass(slots=True)
class _SideRuntime:
    spec: SideSpec
    harness: Harness
    agent: NegotiationAgent
    reservation: float
    anchor: float
    identity: SideIdentity
    commitment: float | None = None
    plan: list[float] | None = None
    offers: list[float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.offers is None:
            self.offers = []


def run_session(
    *,
    session_id: str,
    run_id: str,
    experiment: str,
    cell_id: str,
    repeat_index: int,
    pair_id: str,
    scenario: Scenario,
    side_a: SideSpec,
    side_b: SideSpec,
    harness_config: HarnessConfig,
    runner_config: RunnerConfig,
    prompt_variant: str,
    first_mover: str = "A",
    meta: dict[str, Any] | None = None,
    event_log: EventLog | None = None,
) -> SessionRecord:
    """Прогоняет одну сессию переговоров и возвращает полную запись."""

    trace = TraceContext(trace_id=new_id("trace"))
    started = time.perf_counter()

    record = SessionRecord(
        session_id=session_id,
        run_id=run_id,
        experiment=experiment,
        cell_id=cell_id,
        repeat_index=repeat_index,
        pair_id=pair_id,
        scenario_id=scenario.scenario_id,
        v=scenario.v,
        c=scenario.c,
        surplus=scenario.surplus,
        discount=scenario.discount,
        max_rounds=scenario.max_rounds,
        info_regime=scenario.info_regime.value,
        harness_a=side_a.vector.code(),
        harness_b=side_b.vector.code(),
        model_a=getattr(side_a.llm, "model", "?"),
        model_b=getattr(side_b.llm, "model", "?"),
        role_a=side_a.role.value,
        first_mover_side=first_mover,
        delta_bits=asymmetry(side_a.vector, side_b.vector),
        abs_delta=asymmetry_l1(side_a.vector, side_b.vector),
        meta=dict(meta or {}),
    )

    if event_log is not None:
        event_log.emit(
            EVENT_SESSION_START,
            session_id=session_id,
            experiment=experiment,
            cell_id=cell_id,
            repeat=repeat_index,
            scenario_id=scenario.scenario_id,
            info_regime=scenario.info_regime.value,
            harness_a=record.harness_a,
            harness_b=record.harness_b,
            role_a=record.role_a,
            first_mover=first_mover,
            **trace.as_fields(),
        )

    runtimes = {
        spec.side: _build_runtime(
            spec=spec,
            other=(side_b if spec.side == side_a.side else side_a),
            scenario=scenario,
            harness_config=harness_config,
            prompt_variant=prompt_variant,
            session_id=session_id,
            event_log=event_log,
        )
        for spec in (side_a, side_b)
    }

    breaker = CircuitBreaker(
        max_turns=runner_config.max_turns_per_session,
        max_tokens=runner_config.max_tokens_per_session,
        max_wall_s=runner_config.max_wall_s_per_session,
        abort_on_stuck=runner_config.abort_on_stuck,
    )

    transcript: list[dict[str, Any]] = []
    standing: dict[str, float | None] = {"A": None, "B": None}  # что стоит ПЕРЕД стороной
    order = [first_mover, "B" if first_mover == "A" else "A"]

    try:
        _pre_session(runtimes, scenario, transcript, standing, event_log, session_id)

        turn_index = 0
        max_turns = scenario.max_rounds * 2
        while turn_index < max_turns:
            side = order[turn_index % 2]
            round_index = turn_index // 2 + 1
            runtime = runtimes[side]
            other = runtimes["B" if side == "A" else "A"]

            breaker.before_turn()
            state = _make_state(
                runtime=runtime,
                other=other,
                scenario=scenario,
                session_id=session_id,
                round_index=round_index,
                turn_index=turn_index,
                standing_offer=standing[side],
                transcript=transcript,
            )
            extra = _extra_blocks(other)
            output = runtime.agent.act(state, extra_blocks=extra, trace=trace)
            turn = output.record
            record.turns.append(turn)
            _accumulate(record, turn, side)

            if event_log is not None:
                event_log.emit(
                    EVENT_TURN_END,
                    session_id=session_id,
                    side=side,
                    role=runtime.spec.role.value,
                    round=round_index,
                    action=turn.action_type,
                    price=turn.price,
                    gate_violations=len(turn.gate_violations),
                    gate_clamped=turn.gate_clamped,
                    invalid_outputs=turn.invalid_outputs,
                    tokens=turn.total_tokens,
                    latency_ms=round(turn.latency_ms, 1),
                    **trace.child("turn").as_fields(),
                )

            transcript.append(
                {
                    "side": side,
                    "round_index": round_index,
                    "action_type": turn.action_type,
                    "price": turn.price,
                    "message": turn.message,
                }
            )
            # В отпечаток входит стоящее предложение контрагента: держаться
            # за свою цену, пока контрагент уступает, — это стратегия, а не
            # зависание, и такие ходы отпечаток различает.
            stuck = breaker.after_turn(
                tokens=turn.total_tokens,
                side=side,
                fingerprint=f"{turn.action_type}|{turn.price}|{standing[side]}",
            )
            if stuck:
                record.stuck_turns += 1
                if event_log is not None:
                    event_log.emit(
                        EVENT_STUCK,
                        session_id=session_id,
                        side=side,
                        round=round_index,
                        action=turn.action_type,
                        price=turn.price,
                    )

            action = output.action
            if action.type is ActionType.ACCEPT and action.price is not None:
                record.deal = True
                record.price = action.price
                record.agreement_round = round_index
                break
            if action.type is ActionType.OFFER and action.price is not None:
                runtime.offers.append(action.price)
                # Предложение встаёт «на стол» перед контрагентом; своё
                # собственное предложение сторона принять не может.
                standing["B" if side == "A" else "A"] = action.price
            turn_index += 1

    except CircuitBreakerTripped as exc:
        record.technical_failure = True
        record.failure_reason = f"circuit_breaker:{exc.reason}"
        record.stop_reason = exc.reason
        if event_log is not None:
            event_log.emit(
                EVENT_CIRCUIT_BREAKER,
                session_id=session_id,
                reason=exc.reason,
                detail=exc.detail,
            )
    except MalformedActionError as exc:
        if exc.turn is not None:
            record.turns.append(exc.turn)
            _accumulate(record, exc.turn, exc.turn.side)
        record.technical_failure = True
        record.failure_reason = f"malformed_action:{exc}"
        record.stop_reason = "malformed_action"
    except LLMUnavailableError as exc:
        record.technical_failure = True
        record.failure_reason = f"llm_unavailable:{exc}"
        record.stop_reason = "llm_unavailable"

    _finalize(record, runtimes, scenario, side_a, side_b)
    record.latency_ms = (time.perf_counter() - started) * 1000

    if not record.technical_failure:
        _write_memories(record, runtimes, scenario)

    if event_log is not None:
        event_log.emit(
            EVENT_SESSION_END,
            session_id=session_id,
            deal=record.deal,
            price=record.price,
            agreement_round=record.agreement_round,
            phi_a=record.phi_a,
            efficiency=record.efficiency,
            tokens=record.total_tokens,
            llm_calls=record.llm_calls,
            technical_failure=record.technical_failure,
            stop_reason=record.stop_reason,
            **trace.as_fields(),
        )
    return record


# ---------------------------------------------------------------------------
# Внутреннее
# ---------------------------------------------------------------------------


def _build_runtime(
    *,
    spec: SideSpec,
    other: SideSpec,
    scenario: Scenario,
    harness_config: HarnessConfig,
    prompt_variant: str,
    session_id: str,
    event_log: EventLog | None,
) -> _SideRuntime:
    reservation = scenario.c if spec.role is Role.SELLER else scenario.v
    opponent_reservation = scenario.v if spec.role is Role.SELLER else scenario.c
    anchor = anchor_reference(
        scenario,
        spec.role,
        opponent_reservation if scenario.info_regime is InfoRegime.FULL else None,
    )
    identity = SideIdentity(
        side=spec.side, party_id=spec.party_id, counterparty_id=other.party_id
    )
    harness = assemble_harness(
        spec.vector,
        config=harness_config,
        memory_store=spec.memory_store,
        event_log=event_log,
        session_id=session_id,
    )
    agent = NegotiationAgent(
        side=spec.side,
        llm=spec.llm,
        harness=harness,
        prompt_variant=prompt_variant,
        max_repairs=harness_config.verifier_max_repairs,
        event_log=event_log,
    )
    return _SideRuntime(
        spec=spec,
        harness=harness,
        agent=agent,
        reservation=reservation,
        anchor=anchor,
        identity=identity,
        offers=[],
    )


def _make_state(
    *,
    runtime: _SideRuntime,
    other: _SideRuntime,
    scenario: Scenario,
    session_id: str,
    round_index: int,
    turn_index: int,
    standing_offer: float | None,
    transcript: list[dict[str, Any]],
) -> NegotiationState:
    opponent_reservation = (
        (scenario.v if runtime.spec.role is Role.SELLER else scenario.c)
        if scenario.info_regime is InfoRegime.FULL
        else None
    )
    view = [
        {**turn, "mine": turn["side"] == runtime.spec.side} for turn in transcript
    ]
    return NegotiationState(
        session_id=session_id,
        scenario=scenario,
        identity=runtime.identity,
        role=runtime.spec.role,
        reservation=runtime.reservation,
        anchor_reference=runtime.anchor,
        opponent_reservation=opponent_reservation,
        round_index=round_index,
        turn_index=turn_index,
        max_rounds=scenario.max_rounds,
        standing_offer=standing_offer,
        own_last_offer=runtime.offers[-1] if runtime.offers else None,
        own_offers=list(runtime.offers),
        opponent_offers=list(other.offers),
        transcript=view,
        own_commitment=runtime.commitment,
        opponent_commitment=other.commitment,
        plan=list(runtime.plan or []),
    )


def _extra_blocks(other: _SideRuntime) -> list[str]:
    """Что сторона видит о контрагенте независимо от собственной обвязки.

    Публичное обязательство контрагента видно всем — иначе связывание рук не
    работало бы как механизм. Это принципиально: H3 у одной стороны меняет
    информационное поле обеих.
    """

    block = opponent_commitment_block(other.commitment, other.identity.party_id)
    return [block] if block else []


def _pre_session(
    runtimes: dict[str, _SideRuntime],
    scenario: Scenario,
    transcript: list[dict[str, Any]],
    standing: dict[str, float | None],
    event_log: EventLog | None,
    session_id: str,
) -> None:
    """Публикация обязательств (H3) и построение планов (H5) до торга."""

    for side, runtime in runtimes.items():
        other = runtimes["B" if side == "A" else "A"]
        if not runtime.harness.uses_commitment:
            continue
        state = _make_state(
            runtime=runtime,
            other=other,
            scenario=scenario,
            session_id=session_id,
            round_index=1,
            turn_index=0,
            standing_offer=None,
            transcript=transcript,
        )
        runtime.commitment = runtime.agent.request_commitment(state)

    for side, runtime in runtimes.items():
        other = runtimes["B" if side == "A" else "A"]
        if not runtime.harness.uses_planner:
            continue
        state = _make_state(
            runtime=runtime,
            other=other,
            scenario=scenario,
            session_id=session_id,
            round_index=1,
            turn_index=0,
            standing_offer=None,
            transcript=transcript,
        )
        runtime.plan = runtime.agent.request_plan(
            state, horizon=_planner_horizon(runtime)
        )


def _planner_horizon(runtime: _SideRuntime) -> int:
    for component in runtime.harness.components:
        if component.key == "planner":
            return getattr(component, "horizon", 4)
    return 4


def _accumulate(record: SessionRecord, turn: TurnRecord, side: str) -> None:
    record.total_tokens += turn.total_tokens
    record.prompt_tokens += turn.prompt_tokens
    record.completion_tokens += turn.completion_tokens
    record.invalid_outputs += turn.invalid_outputs
    record.llm_calls += 1
    if side == "A":
        record.gate_violations_a += len(turn.gate_violations)
    else:
        record.gate_violations_b += len(turn.gate_violations)


def _finalize(
    record: SessionRecord,
    runtimes: dict[str, _SideRuntime],
    scenario: Scenario,
    side_a: SideSpec,
    side_b: SideSpec,
) -> None:
    """Считает исходные метрики §5.1 и вторичные §5.2."""

    rt_a, rt_b = runtimes["A"], runtimes["B"]
    record.commitment_a = rt_a.commitment
    record.commitment_b = rt_b.commitment
    record.anchor_a = rt_a.offers[0] if rt_a.offers else None
    record.anchor_b = rt_b.offers[0] if rt_b.offers else None

    first_mover_role = runtimes[record.first_mover_side].spec.role
    record.rubinstein_price = rubinstein_price(scenario, first_mover_role)

    if record.deal and record.price is not None:
        # Цена вне [c, v] означает, что одна из сторон согласилась хуже
        # собственной резервной величины. Не чиним и не отбрасываем: это
        # наблюдаемое нарушение бюджетного ограничения (ZI-U против ZI-C).
        record.budget_violation = not (scenario.c <= record.price <= scenario.v)
        record.phi_a = scenario.share_for(side_a.role, record.price)
        record.phi_b = scenario.share_for(side_b.role, record.price)
        rounds = record.agreement_round or 1
        discount_factor = scenario.discount ** (rounds - 1)
        record.efficiency = discount_factor
        record.phi_a_discounted = record.phi_a * discount_factor
        record.rubinstein_gap = abs(record.price - record.rubinstein_price)
    else:
        # Отсутствие сделки при v > c — чистая потеря. Доли не определены:
        # писать 0.5 или 0 было бы подменой отсутствующего наблюдения.
        record.efficiency = 0.0
        record.phi_a = None
        record.phi_b = None
        record.phi_a_discounted = None
        if not record.stop_reason or record.stop_reason == "completed":
            record.stop_reason = "no_deal"

    surplus = scenario.surplus
    if record.anchor_a is not None and surplus > 0:
        record.anchor_aggressiveness_a = abs(record.anchor_a - rt_a.reservation) / surplus
    record.concession_rate_a = _concession_rate(rt_a.offers, surplus)
    record.concession_rate_b = _concession_rate(rt_b.offers, surplus)


def _concession_rate(offers: list[float], surplus: float) -> float | None:
    """Средний шаг уступки в долях излишка — вторичная метрика §5.2."""

    if len(offers) < 2 or surplus <= 0:
        return None
    steps = [abs(offers[i + 1] - offers[i]) for i in range(len(offers) - 1)]
    return sum(steps) / len(steps) / surplus


def _write_memories(
    record: SessionRecord, runtimes: dict[str, _SideRuntime], scenario: Scenario
) -> None:
    """Раздаёт исход компонентам памяти — каждой стороне её собственный вид.

    Сторона видит СВОЮ долю и предложения контрагента, но не его резервную
    величину: память — это лог наблюдаемого, а не окно в чужие книги.
    """

    for side, runtime in runtimes.items():
        other = runtimes["B" if side == "A" else "A"]
        share = (
            role_share(runtime.spec.role, record.price, scenario.v, scenario.c)
            if record.deal and record.price is not None
            else None
        )
        outcome = {
            "session_id": record.session_id,
            "deal": record.deal,
            "price": record.price,
            "my_share": share,
            "rounds": record.agreement_round or record.max_rounds,
            "counterparty_first_offer": other.offers[0] if other.offers else None,
            "counterparty_final_offer": other.offers[-1] if other.offers else None,
        }
        state = _make_state(
            runtime=runtime,
            other=other,
            scenario=scenario,
            session_id=record.session_id,
            round_index=record.agreement_round or record.max_rounds,
            turn_index=0,
            standing_offer=None,
            transcript=[],
        )
        runtime.harness.on_session_end(state, outcome)
