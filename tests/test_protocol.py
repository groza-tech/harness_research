"""Протокол: разбор действия, учёт исхода, границы информации, сбои.

§7.3 п.4 требует, чтобы при симметричных голых конфигурациях и полной
информации результат тяготел к теоретическому эталону. На уровне юнит-теста
проверяемо ровно одно: **учёт** протокола совпадает с аналитикой Рубинштейна,
если стороны сыграли равновесную цену. Сходится ли к ней живая модель —
эмпирический вопрос, на него отвечает Э1 и таблица `saturation` в отчёте.
"""

from __future__ import annotations

import json

import pytest

from harness_asymmetry.agent import make_state
from harness_asymmetry.config import HarnessConfig, RunnerConfig
from harness_asymmetry.harness.components.base import NegotiationState
from harness_asymmetry.llm_client import LLMResponse, TokenUsage, coerce_price, parse_json_object
from harness_asymmetry.protocol import SideSpec, run_session
from harness_asymmetry.schemas import (
    Action,
    ActionType,
    HarnessVector,
    InfoRegime,
    Role,
    Scenario,
    anchor_reference,
    rubinstein_price,
)


class ScriptedClient:
    """Клиент, отвечающий заранее заданной очередью реплик.

    Нужен, чтобы проверять протокол без сети и без вариативности модели:
    каждый ответ — ровно то, что мы хотим протолкнуть через парсер и гейт.
    """

    def __init__(self, responses: list[str], *, model: str = "test/scripted") -> None:
        self.model = model
        self.responses = list(responses)
        self.calls: list[str] = []

    def chat(self, *, system: str, user: str, tag: str = "") -> LLMResponse:
        self.calls.append(user)
        text = self.responses.pop(0) if self.responses else '{"action":"reject"}'
        return LLMResponse(
            text=text,
            usage=TokenUsage(prompt=10, completion=5, total=15),
            latency_ms=1.0,
            model=self.model,
        )

    def health(self) -> dict:
        return {"provider": "scripted", "calls_ok": len(self.calls)}


def _offer(price: float) -> str:
    return json.dumps({"action": "offer", "price": price, "message": "предложение"})


ACCEPT = json.dumps({"action": "accept", "price": None, "message": "принимаю"})


def _run(scenario, responses_a, responses_b, *, vector_a=None, vector_b=None, first_mover="A"):
    return run_session(
        session_id="t1",
        run_id="r1",
        experiment="TEST",
        cell_id="c1",
        repeat_index=0,
        pair_id="pair1",
        scenario=scenario,
        side_a=SideSpec(
            side="A",
            role=Role.SELLER,
            vector=vector_a or HarnessVector.bare(),
            llm=ScriptedClient(responses_a),
            party_id="pair1#A",
        ),
        side_b=SideSpec(
            side="B",
            role=Role.BUYER,
            vector=vector_b or HarnessVector.bare(),
            llm=ScriptedClient(responses_b),
            party_id="pair1#B",
        ),
        harness_config=HarnessConfig(verifier_max_repairs=1),
        runner_config=RunnerConfig(max_wall_s_per_session=30.0),
        prompt_variant="base",
        first_mover=first_mover,
    )


# ---------------------------------------------------------------------------
# Парсер действия
# ---------------------------------------------------------------------------


def test_price_comes_from_json_not_from_text():
    """Число в свободном тексте не должно попадать в цену (§4.2)."""

    text = 'Готов уступить до 999 999 руб. ```json\n{"action":"offer","price":1250000}\n```'
    parsed = parse_json_object(text)
    assert parsed["price"] == 1_250_000


@pytest.mark.parametrize(
    "value,expected",
    [
        (1_250_000, 1_250_000.0),
        ("1250000", 1_250_000.0),
        ("1 250 000 руб.", 1_250_000.0),
        ("1,250,000", None),  # запятая как разделитель тысяч неоднозначна
        (None, None),
        (True, None),
        ("дороже", None),
        (float("inf"), None),
    ],
)
def test_coerce_price(value, expected):
    result = coerce_price(value)
    if expected is None:
        assert result is None or result != result  # None или NaN
    else:
        assert result == pytest.approx(expected)


def test_accept_without_standing_offer_is_invalid(scenario):
    """«Принимаю», когда принимать нечего, — ошибка модели, а не действие."""

    record = _run(scenario, [ACCEPT, ACCEPT, ACCEPT], [_offer(1_200_000.0)])
    assert record.technical_failure
    assert record.stop_reason == "malformed_action"


# ---------------------------------------------------------------------------
# Учёт исхода
# ---------------------------------------------------------------------------


def test_accounting_matches_rubinstein(scenario):
    """Продавец называет равновесную цену, покупатель принимает в первом раунде."""

    price = rubinstein_price(scenario, Role.SELLER)
    record = _run(scenario, [_offer(price)], [ACCEPT], first_mover="A")

    assert record.deal is True
    assert record.price == pytest.approx(price)
    assert record.agreement_round == 1
    assert record.efficiency == pytest.approx(1.0)  # δ^0 — задержки не было
    assert record.phi_a == pytest.approx(1 / (1 + scenario.discount))
    assert record.phi_a + record.phi_b == pytest.approx(1.0)
    assert record.rubinstein_gap == pytest.approx(0.0)


def test_delay_discounts_efficiency(scenario):
    """Соглашение в раунде t даёт E = δ^(t−1) — задержка съедает излишек."""

    record = _run(
        scenario,
        [_offer(1_350_000.0), _offer(1_300_000.0)],
        [_offer(1_050_000.0), ACCEPT],
    )
    assert record.deal is True
    assert record.agreement_round == 2
    assert record.efficiency == pytest.approx(scenario.discount)
    assert record.phi_a_discounted == pytest.approx(record.phi_a * scenario.discount)


def test_no_deal_leaves_shares_undefined(scenario):
    """Несостоявшаяся сделка не имеет доли: писать 0 или 0,5 — подмена наблюдения."""

    offers = [_offer(1_390_000.0 - i * 1000) for i in range(scenario.max_rounds)]
    counters = [_offer(1_010_000.0 + i * 1000) for i in range(scenario.max_rounds)]
    record = _run(scenario, offers, counters)

    assert record.deal is False
    assert record.phi_a is None and record.phi_b is None
    assert record.efficiency == 0.0
    assert record.stop_reason == "no_deal"


def test_tokens_and_calls_are_accounted(scenario):
    record = _run(scenario, [_offer(1_200_000.0)], [ACCEPT])
    assert record.llm_calls == 2
    assert record.total_tokens == 30  # 2 вызова × 15 токенов у ScriptedClient
    assert record.latency_ms > 0


# ---------------------------------------------------------------------------
# Границы информации
# ---------------------------------------------------------------------------


def test_private_regime_does_not_leak(private_scenario, identity_a):
    """В I1 якорная точка — граница распределения, а не истинная величина."""

    anchor = anchor_reference(private_scenario, Role.SELLER, None)
    assert anchor == private_scenario.v_high
    assert anchor != private_scenario.v

    state = make_state(
        session_id="s", scenario=private_scenario, identity=identity_a, role=Role.SELLER
    )
    payload = json.dumps(state.machine_state(), ensure_ascii=False)
    assert str(int(private_scenario.v)) not in payload
    assert state.opponent_reservation is None


def test_full_regime_exposes_both_values(scenario, identity_a):
    state = make_state(
        session_id="s", scenario=scenario, identity=identity_a, role=Role.SELLER
    )
    assert state.opponent_reservation == scenario.v
    assert state.anchor_reference == scenario.v


# ---------------------------------------------------------------------------
# Предохранители
# ---------------------------------------------------------------------------


def _deadlock(scenario, runner_config):
    """Обе стороны намертво повторяют своё предложение до конца горизонта."""

    return run_session(
        session_id="t2",
        run_id="r1",
        experiment="TEST",
        cell_id="c1",
        repeat_index=0,
        pair_id="pair1",
        scenario=scenario,
        side_a=SideSpec(
            side="A",
            role=Role.SELLER,
            vector=HarnessVector.bare(),
            llm=ScriptedClient([_offer(1_300_000.0)] * 20),
            party_id="pair1#A",
        ),
        side_b=SideSpec(
            side="B",
            role=Role.BUYER,
            vector=HarnessVector.bare(),
            llm=ScriptedClient([_offer(1_050_000.0)] * 20),
            party_id="pair1#B",
        ),
        harness_config=HarnessConfig(),
        runner_config=runner_config,
        prompt_variant="base",
    )


def test_stuck_is_flagged_but_session_is_not_failed(scenario):
    """Залипание — сигнал, а не сбой.

    Досрочный обрыв подменял бы честное «сделки не было» техническим сбоем
    и смещал бы выборку против компонентов, чья суть — стоять на своём.
    Протокол и так ограничен $2T$ ходами.
    """

    record = _deadlock(
        scenario, RunnerConfig(max_turns_per_session=40, max_wall_s_per_session=30.0)
    )
    assert record.technical_failure is False
    assert record.stop_reason == "no_deal"
    assert record.deal is False
    assert record.stuck_turns > 0
    assert len(record.turns) == scenario.max_rounds * 2  # горизонт отработан целиком


def test_stuck_aborts_when_explicitly_enabled(scenario):
    """Кому важнее экономия — включает abort_on_stuck и получает обрыв."""

    record = _deadlock(
        scenario,
        RunnerConfig(
            max_turns_per_session=40, max_wall_s_per_session=30.0, abort_on_stuck=True
        ),
    )
    assert record.technical_failure
    assert record.stop_reason == "stuck_repeated_action"


def test_token_budget_breaker_fails_session(scenario):
    """А вот исчерпание бюджета токенов — настоящий технический сбой."""

    record = _deadlock(
        scenario,
        RunnerConfig(max_tokens_per_session=20, max_wall_s_per_session=30.0),
    )
    assert record.technical_failure
    assert record.stop_reason == "max_tokens"


def test_blocked_accept_becomes_reject_not_a_fabricated_deal(scenario):
    """Запрещённый приём не превращается в сделку по цене, которой никто не называл.

    Раньше гейт «чинил» приём подстановкой своей цены — и рождал сделку из
    воздуха, иногда за пределами резервных величин обеих сторон. Правильный
    исход: принять нельзя ⇒ отказ в этом раунде. В этом и состоит цена
    связанных рук.
    """

    commitment = json.dumps({"reserve_price": 1_300_000.0, "message": "связываю руки"})
    record = _run(
        scenario,
        # A — продавец с обязательством: сначала публикует резервную цену,
        # потом предлагает высоко и пытается принять заведомо худшую цену.
        [commitment, _offer(1_380_000.0)] + [ACCEPT] * 10,
        [_offer(1_020_000.0)] * 8,
        vector_a=HarnessVector(commitment=True),
    )
    assert record.commitment_a == pytest.approx(1_300_000.0)
    accepts = [t for t in record.turns if t.side == "A" and t.action_type == "accept"]
    assert not accepts, "приём хуже обязательства обязан быть заблокирован"
    blocked = [t for t in record.turns if t.side == "A" and t.gate_clamped]
    assert blocked and blocked[0].action_type == "reject"
    if record.deal:
        assert scenario.c <= record.price <= scenario.v
        assert record.budget_violation is False


def test_gate_clamps_when_model_will_not_fix(scenario):
    """Модель упорствует в нарушении — гейт зажимает цену и помечает это флагом."""

    below_cost = [json.dumps({"action": "offer", "price": 800_000.0})] * 6
    record = _run(
        scenario,
        below_cost,
        [_offer(1_050_000.0)] * 6,
        vector_a=HarnessVector(verifier=True),
    )
    first_turn = record.turns[0]
    assert first_turn.gate_violations
    assert first_turn.gate_clamped
    assert first_turn.price == pytest.approx(scenario.c)
