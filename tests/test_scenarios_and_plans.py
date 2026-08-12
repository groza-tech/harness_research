"""Общие случайные числа, планы экспериментов, теоретический эталон."""

from __future__ import annotations

import pytest

from harness_asymmetry.config import ScenarioConfig
from harness_asymmetry.runner.plans import (
    _population,
    ladder_vector,
    plackett_burman_12,
    plan_e2_screening,
    plan_e5_market,
    plan_pilot,
)
from harness_asymmetry.scenarios import build_scenario_pool, market_comparables
from harness_asymmetry.schemas import (
    COMPONENT_KEYS,
    InfoRegime,
    Role,
    rubinstein_price,
)


def test_common_random_numbers():
    """Один сид — один и тот же пул. Это и есть парный дизайн §4.4."""

    cfg = ScenarioConfig(n_scenarios=20, seed=123)
    first, second = build_scenario_pool(cfg), build_scenario_pool(cfg)
    assert first.fingerprint() == second.fingerprint()
    assert [s.v for s in first.scenarios] == [s.v for s in second.scenarios]

    other = build_scenario_pool(ScenarioConfig(n_scenarios=20, seed=124))
    assert other.fingerprint() != first.fingerprint()


def test_pool_guarantees_positive_surplus():
    pool = build_scenario_pool(ScenarioConfig(n_scenarios=50, min_surplus=50_000.0, seed=7))
    assert all(s.surplus >= 50_000.0 for s in pool.scenarios)


def test_regime_switch_keeps_same_draw():
    """I0 и I1 обязаны видеть идентичные (v, c) — иначе режим смешается с розыгрышем."""

    pool = build_scenario_pool(ScenarioConfig(n_scenarios=10, seed=5))
    base = pool.get(3)
    switched = pool.with_regime(3, InfoRegime.PRIVATE)
    assert (switched.v, switched.c) == (base.v, base.c)
    assert switched.info_regime is InfoRegime.PRIVATE


def test_market_comparables_are_deterministic():
    pool = build_scenario_pool(ScenarioConfig(n_scenarios=5, seed=11))
    scenario = pool.get(0)
    first = market_comparables(scenario, n=5, noise_frac=0.06)
    second = market_comparables(scenario, n=5, noise_frac=0.06)
    assert [c.price for c in first] == [c.price for c in second]


def test_rubinstein_benchmark_matches_analytics():
    """Первый ходящий получает 1/(1+δ) излишка — нулевая линия §3.1."""

    pool = build_scenario_pool(ScenarioConfig(n_scenarios=3, seed=1, discount=0.9))
    scenario = pool.get(0)
    price = rubinstein_price(scenario, Role.SELLER)
    share = scenario.share_seller(price)
    assert share == pytest.approx(1 / (1 + scenario.discount))

    buyer_price = rubinstein_price(scenario, Role.BUYER)
    assert scenario.share_buyer(buyer_price) == pytest.approx(1 / (1 + scenario.discount))


# ---------------------------------------------------------------------------
# Планы
# ---------------------------------------------------------------------------


def test_plackett_burman_is_balanced():
    """12 прогонов, каждый фактор ровно 6 раз включён — свойство плана."""

    design = plackett_burman_12(len(COMPONENT_KEYS))
    assert len(design) == 12
    for column in range(len(COMPONENT_KEYS)):
        ones = sum(row[column] for row in design)
        assert ones == 6, f"фактор {column}: {ones} единиц вместо 6"


def test_plackett_burman_columns_are_orthogonal():
    """Ортогональность главных эффектов — то, ради чего берётся PB."""

    design = plackett_burman_12(len(COMPONENT_KEYS))
    signs = [[1 if b else -1 for b in row] for row in design]
    for i in range(len(COMPONENT_KEYS)):
        for j in range(i + 1, len(COMPONENT_KEYS)):
            dot = sum(row[i] * row[j] for row in signs)
            assert dot == 0, f"колонки {i} и {j} не ортогональны: {dot}"


def test_roles_are_counterbalanced():
    """Каждые четыре повтора дают полный цикл роль × очередь хода (§6.4)."""

    specs = plan_pilot(model="m", repeats=8)
    cell = [s for s in specs if s.cell_id.startswith("full_vs_bare")]
    combos = {(s.role_a, s.first_mover) for s in cell[:4]}
    assert combos == {
        (Role.SELLER, "A"),
        (Role.BUYER, "A"),
        (Role.SELLER, "B"),
        (Role.BUYER, "B"),
    }
    assert sum(1 for s in cell if s.role_a is Role.SELLER) == len(cell) // 2


def test_screening_plan_keeps_side_b_bare():
    specs = plan_e2_screening(model="m", repeats=4, info_regimes=[InfoRegime.FULL])
    assert {s.harness_b.code() for s in specs} == {"000000"}
    assert len({s.harness_a.code() for s in specs}) == 12


def test_ladder_is_cumulative():
    for level in range(7):
        assert ladder_vector(level).level == level
    assert set(ladder_vector(2).active()) <= set(ladder_vector(3).active())


def test_market_population_keeps_mean_constant():
    """Разброс варьируется ПРИ НЕИЗМЕННОМ среднем — иначе H4a/H4b неразделимы."""

    import random

    for dispersion in (0.0, 0.5, 1.0):
        population = _population(
            n_agents=12, mean_level=3.0, dispersion=dispersion, rng=random.Random(0)
        )
        mean = sum(v.level for v in population) / len(population)
        assert mean == pytest.approx(3.0, abs=1e-9)


def test_market_plan_pairs_are_unique_per_period():
    specs = plan_e5_market(
        model="m",
        n_agents=6,
        periods=2,
        dispersions=(0.0, 1.0),
        mean_level=3.0,
        info_regime=InfoRegime.FULL,
        seed=1,
    )
    assert len(specs) == 2 * 2 * 3  # 2 разброса × 2 периода × 3 пары
    for spec in specs:
        assert spec.meta["agent_a"] != spec.meta["agent_b"]
