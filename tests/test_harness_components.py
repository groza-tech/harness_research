"""§7.3 пп.2–3: обязательство связывает, верификатор ловит нарушения.

Плюс инвариант сборки: для любого из 64 векторов множество активных
компонентов в точности совпадает с множеством единиц вектора.
"""

from __future__ import annotations

from itertools import product

import pytest

from harness_asymmetry.agent import make_state
from harness_asymmetry.harness.assembler import assemble_harness
from harness_asymmetry.harness.components.commitment import (
    CommitmentComponent,
    clamp_commitment,
    opponent_commitment_block,
)
from harness_asymmetry.harness.components.context import ContextComponent
from harness_asymmetry.harness.components.memory import MemoryStore
from harness_asymmetry.harness.components.planner import sanitize_plan
from harness_asymmetry.harness.components.verifier import VerifierComponent
from harness_asymmetry.schemas import (
    COMPONENT_KEYS,
    Action,
    ActionType,
    HarnessVector,
    Role,
)


# ---------------------------------------------------------------------------
# H3. Обязательство
# ---------------------------------------------------------------------------


def test_commitment_blocks_worse_offer(scenario, identity_a):
    """Продавец не может предложить ниже опубликованной резервной цены."""

    state = make_state(
        session_id="s", scenario=scenario, identity=identity_a, role=Role.SELLER
    )
    state.own_commitment = 1_250_000.0
    component = CommitmentComponent()

    result = component.gate(Action(type=ActionType.OFFER, price=1_100_000.0), state)
    assert not result.ok
    assert result.repaired_price == 1_250_000.0

    ok = component.gate(Action(type=ActionType.OFFER, price=1_300_000.0), state)
    assert ok.ok


def test_commitment_blocks_worse_accept(scenario, identity_a):
    """Связывание рук распространяется и на приём чужой цены — иначе оно фиктивно."""

    state = make_state(
        session_id="s", scenario=scenario, identity=identity_a, role=Role.SELLER
    )
    state.own_commitment = 1_250_000.0
    state.standing_offer = 1_120_000.0
    result = CommitmentComponent().gate(
        Action(type=ActionType.ACCEPT, price=1_120_000.0), state
    )
    assert not result.ok


def test_commitment_is_visible_to_opponent():
    block = opponent_commitment_block(1_250_000.0, "cellX#A")
    assert block is not None and "[ОБЯЗАТЕЛЬСТВО КОНТРАГЕНТА]" in block
    assert opponent_commitment_block(None, "cellX#A") is None


def test_clamp_commitment_keeps_it_executable():
    """Объявление за пределами диапазона превратило бы H3 в генератор отказов."""

    assert clamp_commitment(Role.SELLER, 9_000_000.0, 1_000_000.0, 1_400_000.0) == 1_400_000.0
    assert clamp_commitment(Role.SELLER, 100.0, 1_000_000.0, 1_400_000.0) == 1_000_000.0
    assert clamp_commitment(Role.BUYER, 100.0, 1_400_000.0, 1_000_000.0) == 1_000_000.0


# ---------------------------------------------------------------------------
# H4. Верификационный гейт
# ---------------------------------------------------------------------------


def test_verifier_catches_budget_violation(scenario, identity_a):
    """ZI-C: предложение ниже собственной себестоимости отклоняется."""

    state = make_state(
        session_id="s", scenario=scenario, identity=identity_a, role=Role.SELLER
    )
    component = VerifierComponent(max_concession_frac=0.35)
    result = component.gate(Action(type=ActionType.OFFER, price=900_000.0), state)
    assert not result.ok
    assert result.repaired_price == scenario.c
    assert any("резервной" in v for v in result.violations)


def test_verifier_catches_budget_violation_for_buyer(scenario, identity_b):
    state = make_state(
        session_id="s", scenario=scenario, identity=identity_b, role=Role.BUYER
    )
    result = VerifierComponent(max_concession_frac=0.35).gate(
        Action(type=ActionType.OFFER, price=1_500_000.0), state
    )
    assert not result.ok
    assert result.repaired_price == scenario.v


def test_verifier_caps_concession_step(scenario, identity_a):
    """Уступка больше допустимого шага зажимается, а не пропускается."""

    state = make_state(
        session_id="s", scenario=scenario, identity=identity_a, role=Role.SELLER
    )
    state.own_last_offer = 1_390_000.0
    component = VerifierComponent(max_concession_frac=0.10)  # 10% от span=400k → 40k
    result = component.gate(Action(type=ActionType.OFFER, price=1_100_000.0), state)
    assert not result.ok
    assert result.repaired_price == pytest.approx(1_350_000.0)


def test_verifier_does_not_penalise_accept(scenario, identity_a):
    """Приём стоящей цены — не уступка: запрещать его шагом значит запрещать сделку."""

    state = make_state(
        session_id="s", scenario=scenario, identity=identity_a, role=Role.SELLER
    )
    state.own_last_offer = 1_390_000.0
    state.standing_offer = 1_050_000.0
    result = VerifierComponent(max_concession_frac=0.10).gate(
        Action(type=ActionType.ACCEPT, price=1_050_000.0), state
    )
    assert result.ok


# ---------------------------------------------------------------------------
# H5/H6
# ---------------------------------------------------------------------------


def test_sanitize_plan_is_monotone_and_bounded():
    plan = sanitize_plan(
        [1_500_000, 1_600_000, 900_000, "мусор", 1_200_000],
        reservation=1_000_000,
        anchor=1_400_000,
        horizon=4,
        is_seller=True,
    )
    assert plan == sorted(plan, reverse=True)  # продавец не повышает цену
    assert all(1_000_000 <= p <= 1_400_000 for p in plan)


def test_compaction_is_shorter_than_full_log(scenario, identity_a):
    state = make_state(
        session_id="s", scenario=scenario, identity=identity_a, role=Role.SELLER
    )
    state.transcript = [
        {
            "side": "A" if i % 2 == 0 else "B",
            "mine": i % 2 == 0,
            "round_index": i // 2 + 1,
            "action_type": "offer",
            "price": 1_300_000 - i * 10_000,
            "message": "довольно длинное обоснование позиции стороны " * 3,
        }
        for i in range(8)
    ]
    state.own_offers = [1_300_000, 1_280_000]
    state.opponent_offers = [1_050_000, 1_090_000]

    full = ContextComponent(full_log=True, tail_turns=2).context_block(state) or ""
    compact = ContextComponent(full_log=False, tail_turns=2).context_block(state) or ""
    assert len(compact) < len(full)
    assert "свёрнуты" in compact


# ---------------------------------------------------------------------------
# Ассемблер
# ---------------------------------------------------------------------------


def test_every_vector_assembles(harness_config):
    """Все 64 конфигурации собираются, активные ключи совпадают с единицами."""

    for bits in product((0, 1), repeat=len(COMPONENT_KEYS)):
        vector = HarnessVector.from_bits(bits)
        store = MemoryStore(owner_id="p") if vector.memory else None
        harness = assemble_harness(vector, config=harness_config, memory_store=store)
        # H6 присутствует всегда как рендерер контекста; его бит переключает
        # режим (полный лог / компакция), а не наличие компонента.
        assert {c.key for c in harness.components} == set(vector.active()) - {"full_log"}
        assert harness.context_renderer is not None
        assert harness.context_renderer.full_log == vector.full_log
        assert harness.uses_commitment == vector.commitment
        assert harness.uses_planner == vector.planner


def test_harness_vector_roundtrip():
    vector = HarnessVector(memory=True, verifier=True, full_log=True)
    assert HarnessVector.from_code(vector.code()) == vector
    assert vector.level == 3
    assert set(vector.active()) == {"memory", "verifier", "full_log"}


def test_context_blocks_are_deterministic(scenario, identity_a, harness_config):
    """Один и тот же вход обязан давать побайтово один и тот же промпт."""

    vector = HarnessVector(market=True, verifier=True, full_log=True)
    harness = assemble_harness(vector, config=harness_config)
    state = make_state(
        session_id="s", scenario=scenario, identity=identity_a, role=Role.SELLER
    )
    first = harness.context_blocks(state)
    second = harness.context_blocks(state)
    assert first == second
