"""§7.3 п.1: изоляция памяти между сторонами.

Самая частая ошибка реализации: общая память утекает обеим сторонам, и вся
асимметрия схлопывается. Этот файл — страховка от неё.
"""

from __future__ import annotations

import pytest

from harness_asymmetry.agent import make_state
from harness_asymmetry.harness.assembler import assemble_harness
from harness_asymmetry.harness.components.memory import (
    DealMemory,
    MemoryIsolationError,
    MemoryStore,
    make_stores,
)
from harness_asymmetry.schemas import HarnessVector, Role


def _deal(counterparty: str, share: float) -> DealMemory:
    return DealMemory(
        session_id="s1",
        counterparty_id=counterparty,
        role="seller",
        deal=True,
        price=1_200_000.0,
        my_share=share,
        rounds=3,
        counterparty_first_offer=1_050_000.0,
        counterparty_final_offer=1_180_000.0,
    )


def test_store_is_per_side():
    stores = make_stores(["cellX#A", "cellX#B"])
    stores["cellX#A"].append("cellX#A", _deal("cellX#B", 0.6))

    assert stores["cellX#A"].size() == 1
    # Сторона B ничего не видит: её стор — отдельный объект.
    assert stores["cellX#B"].size() == 0
    assert stores["cellX#B"].recall("cellX#B", "cellX#A", window=6) == []


def test_cross_side_read_raises():
    """Чтение чужой памяти должно ПАДАТЬ, а не тихо возвращать пустоту.

    Тихий пустой список означал бы, что баг проходит незамеченным и молча
    обнуляет половину плана эксперимента.
    """

    store = MemoryStore(owner_id="cellX#A")
    store.append("cellX#A", _deal("cellX#B", 0.6))
    with pytest.raises(MemoryIsolationError):
        store.recall("cellX#B", "cellX#A", window=6)
    with pytest.raises(MemoryIsolationError):
        store.append("cellX#B", _deal("cellX#A", 0.4))


def test_memory_block_only_for_side_with_component(scenario, identity_a, identity_b, harness_config):
    """Блок памяти появляется в промпте только у стороны с включённым H1."""

    stores = make_stores([identity_a.party_id, identity_b.party_id])
    stores[identity_a.party_id].append(identity_a.party_id, _deal(identity_b.party_id, 0.62))

    with_memory = assemble_harness(
        HarnessVector(memory=True),
        config=harness_config,
        memory_store=stores[identity_a.party_id],
    )
    without_memory = assemble_harness(
        HarnessVector(),
        config=harness_config,
        memory_store=stores[identity_b.party_id],
    )

    state_a = make_state(session_id="s2", scenario=scenario, identity=identity_a, role=Role.SELLER)
    state_b = make_state(session_id="s2", scenario=scenario, identity=identity_b, role=Role.BUYER)

    blocks_a = "\n".join(with_memory.context_blocks(state_a))
    blocks_b = "\n".join(without_memory.context_blocks(state_b))

    assert "[ПАМЯТЬ О КОНТРАГЕНТЕ]" in blocks_a
    assert "[ПАМЯТЬ О КОНТРАГЕНТЕ]" not in blocks_b


def test_memory_requires_store(harness_config):
    """Включённая память без персонального стора — ошибка конфигурации."""

    with pytest.raises(ValueError, match="MemoryStore"):
        assemble_harness(HarnessVector(memory=True), config=harness_config, memory_store=None)


def test_memory_window_limits_history(scenario, identity_a, harness_config):
    store = MemoryStore(owner_id=identity_a.party_id)
    for i in range(10):
        store.append(identity_a.party_id, _deal(identity_a.counterparty_id, 0.5 + i / 100))
    recalled = store.recall(identity_a.party_id, identity_a.counterparty_id, window=4)
    assert len(recalled) == 4
