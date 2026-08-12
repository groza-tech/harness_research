"""Планы экспериментов Э1–Э5 (дизайн-док §4.3).

План — это список полностью специфицированных сессий (:class:`SessionSpec`).
Никакой ячейке не позволено доразыгрывать что-то от себя: сценарий приходит
индексом в общий пул (CRN), роли и очередь хода расписаны заранее,
контрбалансировка встроена в генерацию, а не оставлена на волю случая.

Контрбалансировка (§6.4, обязательная): в литературе зафиксирована ролевая
асимметрия LLM-агентов — покупательские роли систематически переигрывают
поставщицкие за счёт якорения. Без контрбалансировки мы измерили бы её
вместо эффекта харнесса. Поэтому в каждой ячейке четыре повтора образуют
полный цикл: A-продавец/A-первый, A-покупатель/A-первый,
A-продавец/B-первый, A-покупатель/B-первый.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Iterable, Sequence

from harness_asymmetry.schemas import COMPONENT_KEYS, HarnessVector, InfoRegime, Role


#: Порядок наращивания обвязки в «лестнице» Э1. Зафиксирован и опубликован:
#: любой другой порядок дал бы другую кривую насыщения.
LADDER_ORDER: tuple[str, ...] = (
    "verifier",   # H4 — прямой аналог бюджетного ограничения ZI-C
    "memory",     # H1 — гипотеза H2 дизайн-дока о наибольшем вкладе
    "market",     # H2
    "planner",    # H5
    "commitment", # H3
    "full_log",   # H6 — кандидат на отрицательную отдачу
)


@dataclass(frozen=True, slots=True)
class SessionSpec:
    """Одна сессия плана — всё уже решено, случайности не осталось."""

    experiment: str
    cell_id: str
    repeat_index: int
    scenario_index: int
    info_regime: InfoRegime
    harness_a: HarnessVector
    harness_b: HarnessVector
    model_a: str
    model_b: str
    role_a: Role
    first_mover: str  # "A" | "B"
    pair_id: str
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def session_id(self) -> str:
        return f"{self.experiment}:{self.cell_id}:r{self.repeat_index:04d}"

    @property
    def party_a(self) -> str:
        return f"{self.pair_id}#A"

    @property
    def party_b(self) -> str:
        return f"{self.pair_id}#B"


def _counterbalance(repeat_index: int) -> tuple[Role, str]:
    """Полный цикл ролей и очереди хода на каждые четыре повтора."""

    role_a = Role.SELLER if repeat_index % 2 == 0 else Role.BUYER
    first_mover = "A" if (repeat_index // 2) % 2 == 0 else "B"
    return role_a, first_mover


def _expand_cell(
    *,
    experiment: str,
    cell_id: str,
    harness_a: HarnessVector,
    harness_b: HarnessVector,
    model_a: str,
    model_b: str,
    info_regime: InfoRegime,
    repeats: int,
    counterbalance: bool,
    meta: dict[str, Any] | None = None,
    pair_id: str | None = None,
) -> list[SessionSpec]:
    """Разворачивает ячейку плана в ``repeats`` сессий с общими сценариями."""

    out: list[SessionSpec] = []
    for repeat in range(repeats):
        if counterbalance:
            role_a, first_mover = _counterbalance(repeat)
        else:
            role_a, first_mover = Role.SELLER, "A"
        out.append(
            SessionSpec(
                experiment=experiment,
                cell_id=cell_id,
                repeat_index=repeat,
                # CRN: индекс повтора — он же индекс сценария во всех ячейках.
                scenario_index=repeat,
                info_regime=info_regime,
                harness_a=harness_a,
                harness_b=harness_b,
                model_a=model_a,
                model_b=model_b,
                role_a=role_a,
                first_mover=first_mover,
                pair_id=pair_id or f"{experiment}:{cell_id}",
                meta=dict(meta or {}),
            )
        )
    return out


def ladder_vector(level: int) -> HarnessVector:
    """Симметричная «лестница» Э1: первые ``level`` компонентов из LADDER_ORDER."""

    if not 0 <= level <= len(LADDER_ORDER):
        raise ValueError(f"Уровень обвязки должен быть 0..{len(LADDER_ORDER)}.")
    return HarnessVector.from_names(LADDER_ORDER[:level])


# ---------------------------------------------------------------------------
# Э1. Симметричная база (репликация Годе–Сандера)
# ---------------------------------------------------------------------------


def plan_e1_symmetric(
    *,
    models: dict[str, str],
    repeats: int,
    info_regimes: Sequence[InfoRegime],
    levels: Sequence[int] = (0, 1, 2, 3, 4, 5, 6),
) -> list[SessionSpec]:
    """Обе стороны получают одинаковую конфигурацию; уровень варьируется.

    Проверяет H5 (убывающая отдача и плато) и даёт нулевую линию для
    отклонения от эталона Рубинштейна при симметричных голых харнессах.
    """

    specs: list[SessionSpec] = []
    for model_key, level, regime in product(sorted(models), levels, info_regimes):
        vector = ladder_vector(level)
        specs.extend(
            _expand_cell(
                experiment="E1",
                cell_id=f"{model_key}_L{level}_{regime.value}",
                harness_a=vector,
                harness_b=vector,
                model_a=models[model_key],
                model_b=models[model_key],
                info_regime=regime,
                repeats=repeats,
                counterbalance=True,
                meta={"model_class": model_key, "level": level},
            )
        )
    return specs


# ---------------------------------------------------------------------------
# Э2. Скрининг компонентов (дробный факторный план Плакетта–Бёрмана)
# ---------------------------------------------------------------------------


#: Генерирующая строка плана Плакетта–Бёрмана на 12 прогонов (11 факторов).
_PB12_GENERATOR: tuple[int, ...] = (1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1)


def plackett_burman_12(n_factors: int) -> list[tuple[int, ...]]:
    """План PB-12: 12 прогонов, до 11 факторов. Возвращает 0/1-строки.

    Классическая конструкция: циклические сдвиги генерирующей строки плюс
    строка из одних минусов. Для $K=6$ компонентов это 12 конфигураций
    вместо 64 — ровно то, что просит §4.3 («12–16 конфигураций вместо 64»).

    Свойство, ради которого он берётся: главные эффекты ортогональны друг
    другу, поэтому вклад каждого компонента оценивается независимо, хотя
    двухфакторные взаимодействия с ними смешаны. Смешивание снимается на
    Э3, где выжившие компоненты гоняются полным факторным планом.
    """

    if not 1 <= n_factors <= 11:
        raise ValueError("PB-12 поддерживает от 1 до 11 факторов.")
    rows: list[tuple[int, ...]] = []
    gen = list(_PB12_GENERATOR)
    for shift in range(11):
        row = gen[-shift:] + gen[:-shift] if shift else list(gen)
        rows.append(tuple(1 if x > 0 else 0 for x in row[:n_factors]))
    rows.append(tuple(0 for _ in range(n_factors)))
    return rows


def plan_e2_screening(
    *,
    model: str,
    repeats: int,
    info_regimes: Sequence[InfoRegime],
) -> list[SessionSpec]:
    """Сторона A варьируется по PB-12, сторона B фиксирована на голой модели.

    Измеряется $\\phi^A$. Ожидаемый исход — два-три компонента со значимыми
    главными эффектами, остальные отсеиваются (проверка H2).
    """

    design = plackett_burman_12(len(COMPONENT_KEYS))
    bare = HarnessVector.bare()
    specs: list[SessionSpec] = []
    for run_idx, bits in enumerate(design):
        vector = HarnessVector.from_bits(bits)
        for regime in info_regimes:
            specs.extend(
                _expand_cell(
                    experiment="E2",
                    cell_id=f"pb{run_idx:02d}_{vector.code()}_{regime.value}",
                    harness_a=vector,
                    harness_b=bare,
                    model_a=model,
                    model_b=model,
                    info_regime=regime,
                    repeats=repeats,
                    counterbalance=True,
                    meta={"pb_run": run_idx, "design_row": list(bits)},
                )
            )
    return specs


# ---------------------------------------------------------------------------
# Э3. Градиент асимметрии (полный факторный план на выживших)
# ---------------------------------------------------------------------------


def plan_e3_gradient(
    *,
    survivors: Sequence[str],
    models: dict[str, str],
    repeats: int,
    info_regimes: Sequence[InfoRegime],
) -> list[SessionSpec]:
    """Ядро статьи: рост доли A по мере роста асимметрии обвязок.

    Уровень асимметрии $|\\Delta|_1 = 0..len(survivors)$: сторона A получает
    первые $k$ выживших компонентов, сторона B — ничего. Уровень 0 —
    симметричная база, от неё отсчитывается рента $\\rho(\\Delta)$.
    """

    unknown = set(survivors) - set(COMPONENT_KEYS)
    if unknown:
        raise ValueError(f"Неизвестные компоненты среди выживших: {sorted(unknown)}.")
    bare = HarnessVector.bare()
    specs: list[SessionSpec] = []
    for model_key, regime in product(sorted(models), info_regimes):
        for k in range(len(survivors) + 1):
            vector = HarnessVector.from_names(survivors[:k])
            specs.extend(
                _expand_cell(
                    experiment="E3",
                    cell_id=f"{model_key}_d{k}_{vector.code()}_{regime.value}",
                    harness_a=vector,
                    harness_b=bare,
                    model_a=models[model_key],
                    model_b=models[model_key],
                    info_regime=regime,
                    repeats=repeats,
                    counterbalance=True,
                    meta={
                        "model_class": model_key,
                        "asymmetry_level": k,
                        "survivors": list(survivors),
                    },
                )
            )
    return specs


# ---------------------------------------------------------------------------
# Э4. Курс обмена «модель ↔ харнесс»
# ---------------------------------------------------------------------------


def plan_e4_exchange_rate(
    *,
    models: dict[str, str],
    repeats: int,
    info_regimes: Sequence[InfoRegime],
    levels: Sequence[int] = (0, 3, 6),
) -> list[SessionSpec]:
    """Перебор пар (модель, уровень обвязки) по обеим сторонам.

    Искомое — точки, где $\\phi^A = 0{,}5$: кривая безразличия «модель ↔
    харнесс». Численный ответ на вопрос, скольким весовым классам модели
    эквивалентна полная обвязка (H3 дизайн-дока, самый цитируемый результат,
    если получится чисто).

    Полная решётка $3 \\times 3$ на сторону — 81 ячейка; на боевом прогоне
    имеет смысл ограничить ``--repeats`` и один режим информации.
    """

    specs: list[SessionSpec] = []
    model_keys = sorted(models)
    for (mk_a, lvl_a), (mk_b, lvl_b) in product(
        product(model_keys, levels), product(model_keys, levels)
    ):
        for regime in info_regimes:
            specs.extend(
                _expand_cell(
                    experiment="E4",
                    cell_id=f"{mk_a}L{lvl_a}_vs_{mk_b}L{lvl_b}_{regime.value}",
                    harness_a=ladder_vector(lvl_a),
                    harness_b=ladder_vector(lvl_b),
                    model_a=models[mk_a],
                    model_b=models[mk_b],
                    info_regime=regime,
                    repeats=repeats,
                    counterbalance=True,
                    meta={
                        "model_class_a": mk_a,
                        "model_class_b": mk_b,
                        "level_a": lvl_a,
                        "level_b": lvl_b,
                    },
                )
            )
    return specs


# ---------------------------------------------------------------------------
# Э5. Рыночный уровень
# ---------------------------------------------------------------------------


def _population(
    *,
    n_agents: int,
    mean_level: float,
    dispersion: float,
    rng: random.Random,
) -> list[HarnessVector]:
    """Популяция обвязок с заданным средним уровнем и разбросом.

    Ключевое требование §4.3: разброс варьируется **при неизменном среднем
    уровне**. Иначе эффект «разброса» смешается с эффектом «в среднем лучше
    оснащены», и разделение H4a/H4b станет невозможным.
    """

    k = len(LADDER_ORDER)
    half = dispersion * min(mean_level, k - mean_level)
    levels: list[int] = []
    for i in range(n_agents):
        # Симметричная пара вокруг среднего: сумма уровней сохраняется точно.
        offset = half if i % 2 == 0 else -half
        level = int(round(min(max(mean_level + offset, 0), k)))
        levels.append(level)
    # Компенсируем округление, чтобы средний уровень совпал с заданным.
    target_sum = int(round(mean_level * n_agents))
    while sum(levels) > target_sum:
        idx = max(range(n_agents), key=lambda i: levels[i])
        if levels[idx] == 0:
            break
        levels[idx] -= 1
    while sum(levels) < target_sum:
        idx = min(range(n_agents), key=lambda i: levels[i])
        if levels[idx] == k:
            break
        levels[idx] += 1
    rng.shuffle(levels)
    return [ladder_vector(lv) for lv in levels]


def plan_e5_market(
    *,
    model: str,
    n_agents: int,
    periods: int,
    dispersions: Sequence[float],
    mean_level: float,
    info_regime: InfoRegime,
    seed: int,
) -> list[SessionSpec]:
    """Популяция агентов, случайное паросочетание, много торговых периодов.

    Это абзац, ради которого пишется вся статья: регрессии совокупной
    эффективности $E$ и коэффициента Джини на разброс обвязок разделяют
    H4a («пирог растёт») и H4b («пирог перераспределяется»).
    """

    if n_agents % 2 != 0:
        raise ValueError("Число агентов должно быть чётным для паросочетания.")
    specs: list[SessionSpec] = []
    for disp_idx, dispersion in enumerate(dispersions):
        rng = random.Random(seed + disp_idx * 1000)
        population = _population(
            n_agents=n_agents, mean_level=mean_level, dispersion=dispersion, rng=rng
        )
        market_id = f"m{disp_idx}_disp{dispersion:.2f}".replace(".", "")
        for period in range(periods):
            order = list(range(n_agents))
            rng.shuffle(order)
            for pair_no in range(0, n_agents, 2):
                i, j = order[pair_no], order[pair_no + 1]
                repeat = period * (n_agents // 2) + pair_no // 2
                role_a, first_mover = _counterbalance(repeat)
                specs.append(
                    SessionSpec(
                        experiment="E5",
                        cell_id=f"{market_id}_p{period:03d}_pair{pair_no // 2:03d}",
                        repeat_index=repeat,
                        scenario_index=repeat,
                        info_regime=info_regime,
                        harness_a=population[i],
                        harness_b=population[j],
                        model_a=model,
                        model_b=model,
                        role_a=role_a,
                        first_mover=first_mover,
                        pair_id=f"E5:{market_id}:a{i:03d}-a{j:03d}",
                        meta={
                            "market_id": market_id,
                            "dispersion": dispersion,
                            "mean_level": mean_level,
                            "period": period,
                            "agent_a": i,
                            "agent_b": j,
                            "population_var": _variance(
                                [v.level for v in population]
                            ),
                            "population_mean": sum(v.level for v in population)
                            / n_agents,
                        },
                    )
                )
    return specs


def _variance(values: Iterable[float]) -> float:
    data = list(values)
    if len(data) < 2:
        return 0.0
    mean = sum(data) / len(data)
    return sum((x - mean) ** 2 for x in data) / (len(data) - 1)


# ---------------------------------------------------------------------------
# Пилот
# ---------------------------------------------------------------------------


def plan_pilot(
    *,
    model: str,
    repeats: int,
    info_regime: InfoRegime = InfoRegime.FULL,
) -> list[SessionSpec]:
    """Пилот §4.4: две крайние ячейки, по ``repeats`` сессий на каждую.

    Смысл — оценить фактическую дисперсию $\\phi^A$ и пересчитать требуемое
    $n$ ДО основного прогона. Неделя 6 — точка невозврата: если дисперсия
    требует непосильного числа повторов, надо менять сеттинг, а не гнать
    основной эксперимент в надежде.
    """

    bare, full = HarnessVector.bare(), HarnessVector.full()
    specs = _expand_cell(
        experiment="PILOT",
        cell_id=f"symmetric_bare_{info_regime.value}",
        harness_a=bare,
        harness_b=bare,
        model_a=model,
        model_b=model,
        info_regime=info_regime,
        repeats=repeats,
        counterbalance=True,
        meta={"cell": "symmetric_bare"},
    )
    specs += _expand_cell(
        experiment="PILOT",
        cell_id=f"full_vs_bare_{info_regime.value}",
        harness_a=full,
        harness_b=bare,
        model_a=model,
        model_b=model,
        info_regime=info_regime,
        repeats=repeats,
        counterbalance=True,
        meta={"cell": "full_vs_bare"},
    )
    return specs
