"""Агент: вызов модели, строгий разбор действия, ремонт через гейт.

Здесь реализованы два требования дизайн-дока, на которых ломаются наивные
реализации:

* **§4.2 п.4–5.** Модель выдаёт свободный текст ПЛЮС структурированное
  действие; цену извлекает парсер из поля JSON, а не регулярка из текста.
  Невалидный вывод — до трёх повторов, затем ход считается техническим
  сбоем, а сессия исключается из анализа с публикацией доли сбоев.
* **§7.2.** Доля невалидных выводов логируется по конфигурациям: сама по
  себе интересная метрика (компоненты обвязки удлиняют промпт и могут
  ухудшать следование формату).

Ремонт через гейт устроен так: гейт отклоняет действие → агент получает
структурированную обратную связь и переспрашивает модель (до
``verifier_max_repairs`` раз) → если модель так и не исправилась, гейт сам
зажимает цену. Различие «модель исправилась» / «зажал гейт» сохраняется в
``TurnRecord``: это разные экономические механизмы.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from harness_asymmetry.harness.assembler import Harness
from harness_asymmetry.harness.components.base import NegotiationState
from harness_asymmetry.harness.components.commitment import clamp_commitment
from harness_asymmetry.harness.components.planner import sanitize_plan
from harness_asymmetry.llm_client import (
    LLMClient,
    LLMUnavailableError,
    coerce_price,
    parse_json_object,
)
from harness_asymmetry.observability import (
    EVENT_PARSE_FAIL,
    EventLog,
    TraceContext,
)
from harness_asymmetry.prompts import (
    COMMITMENT_INSTRUCTION,
    PARSE_RETRY_INSTRUCTION,
    PLAN_INSTRUCTION,
    REPAIR_INSTRUCTION,
    format_state_block,
    system_prompt,
)
from harness_asymmetry.schemas import (
    Action,
    ActionType,
    InfoRegime,
    MalformedActionError,
    Role,
    Scenario,
    TurnRecord,
    anchor_reference,
)


logger = logging.getLogger(__name__)

MAX_PARSE_ATTEMPTS = 3  # §4.2 п.5: «до трёх повторов»


@dataclass(slots=True)
class TurnOutput:
    action: Action
    record: TurnRecord


class NegotiationAgent:
    """Одна сторона переговоров: модель + её обвязка. Состояния не держит.

    Всё состояние торга живёт в ``NegotiationState``, который собирает
    протокол. Агент — чистая функция «состояние → действие» с побочными
    эффектами в лог. Так его можно гонять параллельно и переигрывать по
    транскрипту.
    """

    def __init__(
        self,
        *,
        side: str,
        llm: LLMClient,
        harness: Harness,
        prompt_variant: str,
        max_repairs: int,
        event_log: EventLog | None = None,
    ) -> None:
        self.side = side
        self.llm = llm
        self.harness = harness
        self.prompt_variant = prompt_variant
        self.max_repairs = max_repairs
        self.event_log = event_log

    # -- публичный API ------------------------------------------------------

    def act(
        self,
        state: NegotiationState,
        *,
        extra_blocks: list[str] | None = None,
        trace: TraceContext | None = None,
    ) -> TurnOutput:
        """Один ход: промпт → действие → гейт → запись."""

        system = self._system(state)
        blocks = self.harness.context_blocks(state, extra_blocks=extra_blocks)
        user = self._user_prompt(state, blocks)

        record = TurnRecord(
            turn_index=state.turn_index,
            round_index=state.round_index,
            side=self.side,
            role=state.role.value,
            action_type="",
            price=None,
            components_active=self.harness.active_keys,
        )

        action, parse_failures = self._ask_action(system, user, state, record)
        record.invalid_outputs = parse_failures
        record.price_before_gate = action.price

        action = self._repair_through_gate(action, state, system, user, record)

        record.action_type = action.type.value
        record.price = action.price
        record.message = action.message
        return TurnOutput(action=action, record=record)

    def request_commitment(self, state: NegotiationState) -> float | None:
        """H3: попросить у модели связывающую резервную цену до начала торга."""

        system = self._system(state)
        payload = {**state.machine_state(), "request": "commitment"}
        user = "\n\n".join(
            [
                COMMITMENT_INSTRUCTION,
                format_state_block(payload),
            ]
        )
        try:
            response = self.llm.chat(system=system, user=user, tag="commitment")
        except LLMUnavailableError:
            logger.warning("Не удалось получить обязательство для стороны %s", self.side)
            return None
        parsed = parse_json_object(response.text)
        price = coerce_price(parsed.get("reserve_price"))
        if price is None:
            return None
        return clamp_commitment(
            state.role, price, state.reservation, state.anchor_reference
        )

    def request_plan(self, state: NegotiationState, *, horizon: int) -> list[float]:
        """H5: план уступок на горизонт, один вызов до начала торга."""

        system = self._system(state)
        payload = {**state.machine_state(), "request": "plan", "horizon": horizon}
        user = "\n\n".join(
            [
                PLAN_INSTRUCTION.format(horizon=horizon),
                format_state_block(payload),
            ]
        )
        try:
            response = self.llm.chat(system=system, user=user, tag="plan")
        except LLMUnavailableError:
            logger.warning("Не удалось получить план для стороны %s", self.side)
            return []
        parsed = parse_json_object(response.text)
        raw_plan = parsed.get("plan") or []
        if not isinstance(raw_plan, list):
            return []
        return sanitize_plan(
            raw_plan,
            reservation=state.reservation,
            anchor=state.anchor_reference,
            horizon=horizon,
            is_seller=state.role is Role.SELLER,
        )

    # -- внутреннее ---------------------------------------------------------

    def _system(self, state: NegotiationState) -> str:
        return system_prompt(
            variant=self.prompt_variant,
            role=state.role.value,
            reservation=state.reservation,
            unit=state.scenario.unit,
            good=state.scenario.good,
            max_rounds=state.max_rounds,
            discount=state.scenario.discount,
            info_block=_info_block(state.scenario, state.role, state.opponent_reservation),
        )

    def _user_prompt(self, state: NegotiationState, blocks: list[str]) -> str:
        parts = list(blocks)
        parts.append(format_state_block(state.machine_state()))
        parts.append(
            f"Ваш ход в раунде {state.round_index} из {state.max_rounds}. "
            "Верните JSON-объект с действием."
        )
        return "\n\n".join(parts)

    def _ask_action(
        self,
        system: str,
        user: str,
        state: NegotiationState,
        record: TurnRecord,
    ) -> tuple[Action, int]:
        """Запрос действия с повторами при неразобранном ответе."""

        failures = 0
        current_user = user
        last_reason = ""
        for attempt in range(1, MAX_PARSE_ATTEMPTS + 1):
            response = self.llm.chat(system=system, user=current_user, tag="act")
            record.prompt_tokens += response.usage.prompt
            record.completion_tokens += response.usage.completion
            record.total_tokens += response.usage.total
            record.latency_ms += response.latency_ms
            record.cached = response.cached
            record.stop_reason = response.stop_reason
            record.raw_text = response.text

            action, reason = _parse_action(response.text, state)
            if action is not None:
                return action, failures

            failures += 1
            last_reason = reason
            if self.event_log is not None:
                self.event_log.emit(
                    EVENT_PARSE_FAIL,
                    session_id=state.session_id,
                    side=self.side,
                    round=state.round_index,
                    attempt=attempt,
                    reason=reason,
                    response_chars=len(response.text),
                )
            current_user = user + "\n\n" + PARSE_RETRY_INSTRUCTION.format(reason=reason)

        record.action_type = "malformed"
        record.stop_reason = "malformed_action"
        raise MalformedActionError(
            f"Сторона {self.side}: не удалось разобрать действие за "
            f"{MAX_PARSE_ATTEMPTS} попыток (последняя причина: {last_reason}).",
            turn=record,
        )

    def _repair_through_gate(
        self,
        action: Action,
        state: NegotiationState,
        system: str,
        user: str,
        record: TurnRecord,
    ) -> Action:
        """Цикл ремонта: гейт → переспрос модели → в крайнем случае зажим."""

        for _ in range(self.max_repairs):
            result = self.harness.gate(action, state)
            if result.ok:
                return action
            record.gate_violations.extend(result.violations)
            repair_user = user + "\n\n" + REPAIR_INSTRUCTION.format(
                violations="; ".join(result.violations)
            )
            try:
                repaired_action, _ = self._ask_action(system, repair_user, state, record)
            except MalformedActionError:
                break
            record.gate_repaired = True
            action = repaired_action

        final = self.harness.gate(action, state)
        if final.ok:
            return action
        record.gate_violations.extend(final.violations)

        if action.type is ActionType.ACCEPT:
            # Запрещённый приём НЕЛЬЗЯ чинить подстановкой цены: контрагент
            # такой цены не предлагал, и «сделка» возникла бы из воздуха — в
            # том числе за пределами резервных величин обеих сторон. Гейт
            # говорит «принять нельзя», и единственный корректный исход —
            # отказ в этом раунде. Экономически это и есть работа механизма
            # обязательства: связанные руки могут стоить сделки.
            record.gate_clamped = True
            return Action(
                type=ActionType.REJECT,
                price=None,
                message=action.message,
            )

        # Предложение — другое дело: агент называет цену, обвязка её
        # ограничивает. Флаг отличает «модель уступила под давлением
        # регламента» от «регламент сам поставил цену».
        if final.repaired_price is not None:
            record.gate_clamped = True
            action = Action(
                type=action.type,
                price=final.repaired_price,
                message=action.message,
            )
        return action


def _parse_action(text: str, state: NegotiationState) -> tuple[Action | None, str]:
    """Строгий разбор действия. Возвращает ``(действие, причина отказа)``."""

    parsed = parse_json_object(text)
    if not parsed:
        return None, "в ответе нет JSON-объекта"

    raw_action = str(parsed.get("action", "")).strip().lower()
    if raw_action not in {a.value for a in ActionType}:
        return None, f"поле action={raw_action!r} не входит в offer|accept|reject"
    action_type = ActionType(raw_action)
    message = str(parsed.get("message", ""))[:600]

    if action_type is ActionType.REJECT:
        return Action(type=action_type, price=None, message=message), ""

    price = coerce_price(parsed.get("price"))
    if action_type is ActionType.ACCEPT:
        if state.standing_offer is None:
            # Принимать нечего: трактуем как невалидный вывод, а не как «отказ»,
            # иначе мы бы молча превратили ошибку модели в осмысленное действие.
            return None, "action=accept, но предложения на столе нет"
        # Цена приёма — всегда стоящее предложение контрагента, что бы модель
        # ни написала в price. Иначе «приём» стал бы скрытым контрпредложением.
        return Action(
            type=action_type, price=state.standing_offer, message=message
        ), ""

    if price is None:
        return None, "action=offer, но поле price пустое или не число"
    if price <= 0:
        return None, f"цена {price} неположительна"
    return Action(type=action_type, price=price, message=message), ""


def _info_block(
    scenario: Scenario, role: Role, opponent_reservation: float | None
) -> str:
    """Режим информации I0/I1 — контрольная ось §4.1.

    В I0 обе стороны знают величины друг друга: если эффект харнесса выживает
    здесь, его нельзя объяснить переодетой информационной асимметрией.
    В I1 известно только распределение — работает граница
    Майерсона–Сатертуэйта.
    """

    if scenario.info_regime is InfoRegime.FULL:
        if opponent_reservation is None:
            raise ValueError("Режим I0 требует известной резервной величины контрагента.")
        label = (
            "предельный бюджет закупщика"
            if role is Role.SELLER
            else "себестоимость поставщика"
        )
        return (
            f"Вам также известно: {label} составляет {opponent_reservation:,.0f}. "
            "Контрагенту ваша величина известна точно так же.".replace(",", " ")
        )

    if role is Role.SELLER:
        low, high = scenario.v_low, scenario.v_high
        label = "предельный бюджет закупщика"
    else:
        low, high = scenario.c_low, scenario.c_high
        label = "себестоимость поставщика"
    return (
        f"Точная величина контрагента вам неизвестна. Известно лишь, что {label} "
        f"равномерно распределён в диапазоне от {low:,.0f} до {high:,.0f}. "
        "Ваша величина контрагенту также неизвестна.".replace(",", " ")
    )


def make_state(
    *,
    session_id: str,
    scenario: Scenario,
    identity: Any,
    role: Role,
    round_index: int = 1,
    turn_index: int = 0,
    **overrides: Any,
) -> NegotiationState:
    """Хелпер для тестов: состояние с корректными границами информации."""

    reservation = scenario.c if role is Role.SELLER else scenario.v
    opponent_reservation = (
        (scenario.v if role is Role.SELLER else scenario.c)
        if scenario.info_regime is InfoRegime.FULL
        else None
    )
    state = NegotiationState(
        session_id=session_id,
        scenario=scenario,
        identity=identity,
        role=role,
        reservation=reservation,
        anchor_reference=anchor_reference(scenario, role, opponent_reservation),
        opponent_reservation=opponent_reservation,
        round_index=round_index,
        turn_index=turn_index,
        max_rounds=scenario.max_rounds,
    )
    for key, value in overrides.items():
        setattr(state, key, value)
    return state
