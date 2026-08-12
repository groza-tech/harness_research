"""Генератор сценариев $(v,c)$ и общие случайные числа.

Ключевое требование дизайн-дока §4.4: **одни и те же розыгрыши $(v,c)$
используются во всех ячейках плана**. Это парный дизайн, он режет дисперсию
в разы и определяет разницу между «эффект значим» и «эффект в шуме».

Реализовано просто и проверяемо: пул сценариев строится один раз из
``ScenarioConfig.seed`` и раздаётся всем ячейкам по индексу повтора. Ни одна
ячейка не имеет права разыграть себе «свои» $(v,c)$ — за этим следит тест
``test_scenarios.py::test_common_random_numbers``.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Sequence

from harness_asymmetry.config import ScenarioConfig
from harness_asymmetry.schemas import InfoRegime, Scenario


def _stable_seed(*parts: object) -> int:
    """Детерминированный сид из произвольных частей — без ``hash()``.

    Встроенный ``hash()`` для строк рандомизируется между процессами
    (PYTHONHASHSEED), что тихо ломает воспроизводимость при параллельном
    прогоне. Поэтому SHA-256.
    """

    payload = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


@dataclass(slots=True)
class ScenarioPool:
    """Пул сценариев, общий для всех ячеек плана (CRN).

    Индексация циклическая: ``pool.get(repeat_index)`` при
    ``repeat_index >= len(pool)`` начинает пул заново. Так ячейка с 100
    повторами и пул из 40 сценариев остаются согласованными между собой.
    """

    config: ScenarioConfig
    scenarios: tuple[Scenario, ...]

    def __len__(self) -> int:
        return len(self.scenarios)

    def get(self, repeat_index: int) -> Scenario:
        return self.scenarios[repeat_index % len(self.scenarios)]

    def with_regime(self, repeat_index: int, regime: InfoRegime) -> Scenario:
        """Тот же розыгрыш $(v,c)$, но в другом режиме информации.

        I0 и I1 обязаны видеть идентичные $(v,c)$ — иначе разница между
        режимами смешается с разницей в розыгрышах.
        """

        base = self.get(repeat_index)
        if base.info_regime is regime:
            return base
        return Scenario(
            scenario_id=base.scenario_id,
            v=base.v,
            c=base.c,
            discount=base.discount,
            max_rounds=base.max_rounds,
            info_regime=regime,
            v_low=base.v_low,
            v_high=base.v_high,
            c_low=base.c_low,
            c_high=base.c_high,
            unit=base.unit,
            good=base.good,
        )

    def fingerprint(self) -> str:
        """SHA-256 пула — попадает в манифест как доказательство CRN."""

        payload = ";".join(
            f"{s.scenario_id}:{s.v:.4f}:{s.c:.4f}:{s.discount:.4f}" for s in self.scenarios
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_scenario_pool(config: ScenarioConfig) -> ScenarioPool:
    """Строит пул из ``config.n_scenarios`` розыгрышей с $v > c + \\text{min}$."""

    rng = random.Random(config.seed)
    scenarios: list[Scenario] = []
    attempts = 0
    max_attempts = config.n_scenarios * 200
    while len(scenarios) < config.n_scenarios:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                "Не удалось набрать пул сценариев: распределения v и c почти не "
                "пересекаются с запасом min_surplus. Проверьте ScenarioConfig."
            )
        v = rng.uniform(config.v_low, config.v_high)
        c = rng.uniform(config.c_low, config.c_high)
        if v - c < config.min_surplus:
            # Отбрасываем: сессия без взаимовыгодной сделки не несёт информации
            # о распределении излишка, но портит метрику доли сделок D.
            continue
        idx = len(scenarios)
        scenarios.append(
            Scenario(
                scenario_id=f"S{idx:04d}",
                v=round(v, 2),
                c=round(c, 2),
                discount=config.discount,
                max_rounds=config.max_rounds,
                info_regime=InfoRegime.FULL,
                v_low=config.v_low,
                v_high=config.v_high,
                c_low=config.c_low,
                c_high=config.c_high,
                unit=config.unit,
                good=config.good,
            )
        )
    return ScenarioPool(config=config, scenarios=tuple(scenarios))


# ---------------------------------------------------------------------------
# Рыночный ретрив (H2): таблица сопоставимых сделок
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Comparable:
    deal_id: str
    price: float
    volume_note: str


def market_comparables(
    scenario: Scenario,
    *,
    n: int,
    noise_frac: float,
    salt: str = "market",
) -> tuple[Comparable, ...]:
    """Сопоставимые сделки для компонента H2.

    Экономическая роль — снижение неопределённости о резервной цене
    контрагента (§3.3). Технически: выборка вокруг **конкурентной цены**
    $(v+c)/2$ с шумом. Конкурентная цена известна аналитически, поэтому
    ретрив честный — он показывает реальную статистику рынка, а не подсказку
    «предложи столько-то».

    Детерминирован по ``scenario_id``: одна и та же сессия в разных ячейках
    плана видит одни и те же сопоставимые.
    """

    center = (scenario.v + scenario.c) / 2.0
    rng = random.Random(_stable_seed(salt, scenario.scenario_id, n, noise_frac))
    out: list[Comparable] = []
    for i in range(n):
        price = center * (1.0 + rng.gauss(0.0, noise_frac))
        out.append(
            Comparable(
                deal_id=f"{scenario.scenario_id}-M{i:02d}",
                price=round(price, 2),
                volume_note=rng.choice(
                    ["сопоставимый объём", "объём −10%", "объём +15%", "сопоставимый объём"]
                ),
            )
        )
    return tuple(out)


def scenario_ids(pool: Sequence[Scenario]) -> list[str]:
    return [s.scenario_id for s in pool]
