"""Общие фикстуры тестов."""

from __future__ import annotations

import pytest

from harness_asymmetry.config import HarnessConfig, RunConfig, RunnerConfig, ScenarioConfig
from harness_asymmetry.harness.components.base import SideIdentity
from harness_asymmetry.schemas import InfoRegime, Scenario


@pytest.fixture()
def scenario() -> Scenario:
    """Ровные числа: излишек 400 000, чтобы доли считались в уме."""

    return Scenario(
        scenario_id="S0000",
        v=1_400_000.0,
        c=1_000_000.0,
        discount=0.9,
        max_rounds=6,
        info_regime=InfoRegime.FULL,
        v_low=1_000_000.0,
        v_high=1_600_000.0,
        c_low=700_000.0,
        c_high=1_300_000.0,
    )


@pytest.fixture()
def private_scenario(scenario: Scenario) -> Scenario:
    return Scenario(
        scenario_id=scenario.scenario_id,
        v=scenario.v,
        c=scenario.c,
        discount=scenario.discount,
        max_rounds=scenario.max_rounds,
        info_regime=InfoRegime.PRIVATE,
        v_low=scenario.v_low,
        v_high=scenario.v_high,
        c_low=scenario.c_low,
        c_high=scenario.c_high,
    )


@pytest.fixture()
def identity_a() -> SideIdentity:
    return SideIdentity(side="A", party_id="cellX#A", counterparty_id="cellX#B")


@pytest.fixture()
def identity_b() -> SideIdentity:
    return SideIdentity(side="B", party_id="cellX#B", counterparty_id="cellX#A")


@pytest.fixture()
def harness_config() -> HarnessConfig:
    return HarnessConfig(verifier_max_repairs=1, planner_horizon=3, compaction_tail_turns=2)


@pytest.fixture()
def run_config(harness_config: HarnessConfig) -> RunConfig:
    return RunConfig(
        scenarios=ScenarioConfig(n_scenarios=8, max_rounds=6, seed=42),
        harness=harness_config,
        runner=RunnerConfig(repeats=4, max_workers=1, max_wall_s_per_session=60.0),
        info_regimes=("I0",),
    )
