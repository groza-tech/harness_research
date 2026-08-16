"""Метрики §5: первичные, вторичные и рыночные.

Первичные (§5.1): доля излишка стороны A $\\phi^A$, рента от асимметрии
$\\rho(\\Delta)$, аллокативная эффективность $E$, доля состоявшихся сделок
$D$, задержка $t^*$.

Принцип, которого держимся везде: **несостоявшаяся сделка не имеет доли**.
Записать в $\\phi^A$ ноль или 0,5 при отсутствии сделки было бы подменой
отсутствующего наблюдения. Поэтому $\\phi^A$ считается только по сделкам, а
провал торга учитывается отдельной метрикой $D$ и нулевой $E$. Иначе
компонент, который просто чаще срывает сделки, выглядел бы «скромным
переговорщиком».
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from harness_asymmetry.schemas import COMPONENT_KEYS, PRICE_TOLERANCE


def load_sessions(path: str | Path) -> pd.DataFrame:
    """Читает ``sessions.csv``. Технические сбои остаются в таблице с флагом.

    Коды харнесса читаются строго как строки: ``"000000"`` иначе становится
    целым нулём, ``"010000"`` — десятью тысячами, и все ведущие нули (то есть
    половина информации о конфигурации) теряются. Это тихая порча данных,
    которая проявилась бы только в неверных таблицах отчёта.
    """

    df = pd.read_csv(path, dtype={"harness_a": str, "harness_b": str})
    if df.empty:
        return df
    for col in ("harness_a", "harness_b"):
        if col in df.columns:
            df[col] = df[col].fillna("").str.zfill(len(COMPONENT_KEYS))
    for col in ("deal", "technical_failure", "budget_violation"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().isin({"true", "1", "1.0", "yes"})
    return df


def analysis_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Отбрасывает технические сбои — но только их, и с явным следом.

    Доля отброшенного считается в :func:`failure_summary` и обязана попасть
    в отчёт (дизайн-док §4.2 п.5: «доля сбоев логируется и публикуется»).
    """

    if df.empty:
        return df
    clean = df.loc[~df["technical_failure"]].copy()
    if {"price", "v", "c"} <= set(clean.columns):
        # Флаг из записи пересчитываем: допуск PRICE_TOLERANCE мог измениться
        # после прогона, и старые сессии должны считаться по текущему правилу.
        # Иначе округление цены до целых рублей навсегда осталось бы в данных
        # как «нарушение бюджетного ограничения».
        priced = clean["price"].notna()
        clean.loc[priced, "budget_violation"] = (
            clean.loc[priced, "price"] < clean.loc[priced, "c"] - PRICE_TOLERANCE
        ) | (clean.loc[priced, "price"] > clean.loc[priced, "v"] + PRICE_TOLERANCE)
    return clean


def failure_summary(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {"sessions": 0, "failures": 0, "failure_rate_pct": 0.0}
    failures = int(df["technical_failure"].sum())
    return {
        "sessions": int(len(df)),
        "failures": failures,
        "failure_rate_pct": round(100 * failures / len(df), 2),
        "invalid_output_rate_pct": round(
            100 * float(df["invalid_outputs"].gt(0).mean()), 2
        ),
    }


# ---------------------------------------------------------------------------
# Агрегаты по ячейкам
# ---------------------------------------------------------------------------


def cell_metrics(df: pd.DataFrame, *, by: Sequence[str] = ("experiment", "cell_id")) -> pd.DataFrame:
    """Первичные метрики §5.1 по ячейкам плана."""

    clean = analysis_frame(df)
    if clean.empty:
        return pd.DataFrame()
    groups = clean.groupby(list(by), dropna=False)
    out = groups.apply(_cell_row, include_groups=False).reset_index()
    return out


def _cell_row(g: pd.DataFrame) -> pd.Series:
    deals = g.loc[g["deal"]]
    phi = deals["phi_a"].dropna()
    return pd.Series(
        {
            "n_sessions": len(g),
            "n_deals": len(deals),
            "deal_rate": len(deals) / len(g) if len(g) else np.nan,
            "phi_a_mean": phi.mean() if len(phi) else np.nan,
            "phi_a_std": phi.std(ddof=1) if len(phi) > 1 else np.nan,
            "phi_a_median": phi.median() if len(phi) else np.nan,
            "efficiency_mean": g["efficiency"].mean(),
            "agreement_round_mean": deals["agreement_round"].mean()
            if len(deals)
            else np.nan,
            "rubinstein_gap_mean": deals["rubinstein_gap"].mean()
            if len(deals)
            else np.nan,
            "anchor_aggressiveness_mean": g["anchor_aggressiveness_a"].mean(),
            # Доля сделок вне [c, v] — прямой замер бюджетного ограничения:
            # у конфигураций с включённым H4 она обязана быть нулевой.
            "budget_violation_rate": float(deals["budget_violation"].mean())
            if len(deals) and "budget_violation" in deals
            else np.nan,
            "tokens_mean": g["total_tokens"].mean(),
            "latency_ms_mean": g["latency_ms"].mean(),
            "invalid_outputs_total": int(g["invalid_outputs"].sum()),
            "harness_a": g["harness_a"].iloc[0],
            "harness_b": g["harness_b"].iloc[0],
            "abs_delta": g["abs_delta"].iloc[0],
            "info_regime": g["info_regime"].iloc[0],
            "model_a": g["model_a"].iloc[0],
            "model_b": g["model_b"].iloc[0],
        }
    )


# ---------------------------------------------------------------------------
# Рента от асимметрии
# ---------------------------------------------------------------------------


def harness_rent(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("model_a", "info_regime"),
    n_boot: int = 5000,
    seed: int = 7,
) -> pd.DataFrame:
    """$\\rho(\\Delta) = E[\\phi^A \\mid h^A, h^B] - E[\\phi^A \\mid h^A = h^B]$.

    Базой служит симметричная ячейка внутри той же группы (та же модель, тот
    же режим информации) — так модель и режим не смешиваются с асимметрией.
    Доверительный интервал — бутстрэп разности средних (§6.4).
    """

    clean = analysis_frame(df)
    clean = clean.loc[clean["deal"] & clean["phi_a"].notna()]
    if clean.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for keys, group in clean.groupby(list(group_cols), dropna=False):
        symmetric = group.loc[group["harness_a"] == group["harness_b"], "phi_a"]
        if symmetric.empty:
            continue
        base = float(symmetric.mean())
        for (h_a, h_b), cell in group.groupby(["harness_a", "harness_b"]):
            if h_a == h_b:
                continue
            values = cell["phi_a"].to_numpy(dtype=float)
            rho = float(values.mean()) - base
            low, high = bootstrap_diff_ci(
                values, symmetric.to_numpy(dtype=float), n_boot=n_boot, seed=seed
            )
            row = {col: key for col, key in zip(group_cols, _as_tuple(keys))}
            row.update(
                {
                    "harness_a": h_a,
                    "harness_b": h_b,
                    "abs_delta": int(cell["abs_delta"].iloc[0]),
                    # Знаковое превосходство A: сколько компонентов есть у A и
                    # нет у B, минус обратное. |Δ|₁ для градиента не годится —
                    # он одинаков и когда A сильнее, и когда слабее, так что
                    # усреднение по нему гасит эффект в ноль.
                    "net_delta": _net_delta(h_a, h_b),
                    "n": len(values),
                    "phi_a_mean": float(values.mean()),
                    "phi_a_symmetric_base": base,
                    "rho": rho,
                    "rho_ci_low": low,
                    "rho_ci_high": high,
                    "significant": bool(low > 0 or high < 0),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows).sort_values("rho", ascending=False) if rows else pd.DataFrame()


def _as_tuple(keys: object) -> tuple:
    return keys if isinstance(keys, tuple) else (keys,)


def _net_delta(code_a: str, code_b: str) -> int:
    """$\\sum_k \\Delta_k$ — знаковое превосходство A по числу компонентов."""

    return sum(int(x) - int(y) for x, y in zip(code_a, code_b))


# ---------------------------------------------------------------------------
# Бутстрэп и перестановочный тест (§6.4)
# ---------------------------------------------------------------------------


def bootstrap_ci(
    values: np.ndarray | Sequence[float],
    *,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 7,
) -> tuple[float, float]:
    """Перцентильный бутстрэп среднего."""

    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    return (
        float(np.quantile(draws, alpha / 2)),
        float(np.quantile(draws, 1 - alpha / 2)),
    )


def bootstrap_diff_ci(
    a: np.ndarray | Sequence[float],
    b: np.ndarray | Sequence[float],
    *,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 7,
) -> tuple[float, float]:
    """Бутстрэп разности средних ``mean(a) − mean(b)``."""

    arr_a = np.asarray(list(a), dtype=float)
    arr_b = np.asarray(list(b), dtype=float)
    arr_a, arr_b = arr_a[np.isfinite(arr_a)], arr_b[np.isfinite(arr_b)]
    if arr_a.size == 0 or arr_b.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws_a = rng.choice(arr_a, size=(n_boot, arr_a.size), replace=True).mean(axis=1)
    draws_b = rng.choice(arr_b, size=(n_boot, arr_b.size), replace=True).mean(axis=1)
    diff = draws_a - draws_b
    return (
        float(np.quantile(diff, alpha / 2)),
        float(np.quantile(diff, 1 - alpha / 2)),
    )


def permutation_test(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_perm: int = 5000,
    seed: int = 7,
) -> dict[str, float]:
    """Перестановочный тест на случайной переразметке конфигураций (§6.4).

    Проверяет ровно то, что нужно: если приписать сессиям метки ячеек
    случайно, воспроизводится ли наблюдаемая разность? Не требует
    предположений о распределении $\\phi^A$, которое заведомо не нормально
    (масса на границах диапазона).
    """

    arr_a = np.asarray(list(a), dtype=float)
    arr_b = np.asarray(list(b), dtype=float)
    arr_a, arr_b = arr_a[np.isfinite(arr_a)], arr_b[np.isfinite(arr_b)]
    if arr_a.size == 0 or arr_b.size == 0:
        return {"observed": float("nan"), "p_value": float("nan"), "n_perm": 0}
    observed = float(arr_a.mean() - arr_b.mean())
    pooled = np.concatenate([arr_a, arr_b])
    rng = np.random.default_rng(seed)
    n_a = arr_a.size
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        diff = pooled[:n_a].mean() - pooled[n_a:].mean()
        if abs(diff) >= abs(observed):
            count += 1
    return {
        "observed": observed,
        "p_value": (count + 1) / (n_perm + 1),  # добавка Дэвисона–Хинкли
        "n_perm": float(n_perm),
    }


# ---------------------------------------------------------------------------
# Рыночные метрики (§5.3, эксперимент Э5)
# ---------------------------------------------------------------------------


def gini(values: Sequence[float]) -> float:
    """Коэффициент Джини по долям агентов. 0 — равенство, 1 — максимум."""

    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan")
    if arr.min() < 0:
        arr = arr - arr.min()
    total = arr.sum()
    if total <= 0:
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    index = np.arange(1, n + 1)
    return float((2 * (index * arr).sum()) / (n * total) - (n + 1) / n)


def market_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Совокупные показатели рынка по уровням разброса обвязок (Э5).

    Считаем на уровне *агента*: каждой сессии соответствуют два наблюдения
    (доля A и доля B), и Джини строится по средним долям агентов, а не по
    сессиям. Иначе неравенство между агентами подменилось бы дисперсией
    между сделками.
    """

    market = analysis_frame(df)
    market = market.loc[market["experiment"] == "E5"]
    if market.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for market_id, group in market.groupby("market_id", dropna=False):
        deals = group.loc[group["deal"]]
        shares: dict[str, list[float]] = {}
        for _, row in deals.iterrows():
            if pd.notna(row.get("phi_a")):
                shares.setdefault(str(row["agent_a"]), []).append(float(row["phi_a"]))
            if pd.notna(row.get("phi_b")):
                shares.setdefault(str(row["agent_b"]), []).append(float(row["phi_b"]))
        agent_means = [float(np.mean(v)) for v in shares.values() if v]
        rows.append(
            {
                "market_id": market_id,
                "dispersion": float(group["dispersion"].iloc[0]),
                "harness_var": float(group["population_var"].iloc[0]),
                "harness_mean": float(group["population_mean"].iloc[0]),
                "n_sessions": len(group),
                "deal_rate": len(deals) / len(group) if len(group) else np.nan,
                "efficiency_mean": float(group["efficiency"].mean()),
                "realized_surplus": float((group["efficiency"] * group["surplus"]).sum()),
                "max_surplus": float(group["surplus"].sum()),
                "gini_shares": gini(agent_means),
                "n_agents_observed": len(agent_means),
                "lost_trade_rate": 1 - (len(deals) / len(group)) if len(group) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("harness_var")


# ---------------------------------------------------------------------------
# Мощность (пилот §4.4)
# ---------------------------------------------------------------------------


def required_n(
    *,
    observed_sd: float,
    effect_pp: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Требуемое $n$ на ячейку по фактической дисперсии пилота.

    Двухвыборочный t-критерий, эффект задаётся в процентных пунктах доли
    излишка. Именно этот пересчёт §4.4 называет условием прохождения точки
    невозврата на неделе 6.
    """

    from scipy import stats

    if observed_sd <= 0 or effect_pp <= 0:
        return 0
    delta = effect_pp / 100.0
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    n = 2 * ((z_alpha + z_beta) ** 2) * (observed_sd**2) / (delta**2)
    return int(np.ceil(n))


def pilot_power_report(df: pd.DataFrame, *, effect_pp: float = 7.5) -> pd.DataFrame:
    """Сводка пилота: дисперсия по крайним ячейкам и требуемое $n$."""

    clean = analysis_frame(df)
    clean = clean.loc[clean["deal"] & clean["phi_a"].notna()]
    if clean.empty:
        return pd.DataFrame()
    rows = []
    for cell_id, group in clean.groupby("cell_id"):
        sd = float(group["phi_a"].std(ddof=1)) if len(group) > 1 else float("nan")
        rows.append(
            {
                "cell_id": cell_id,
                "n_observed": len(group),
                "phi_a_mean": float(group["phi_a"].mean()),
                "phi_a_sd": sd,
                "required_n_per_cell": required_n(observed_sd=sd, effect_pp=effect_pp)
                if np.isfinite(sd)
                else None,
                "target_effect_pp": effect_pp,
            }
        )
    return pd.DataFrame(rows)


def budget_discipline(df: pd.DataFrame) -> pd.DataFrame:
    """Нарушения бюджетного ограничения по наличию верификационного гейта.

    Прямая проверка механизма Годе–Сандера на нашем объекте: ZI-C отличался
    от ZI-U ровно бюджетным ограничением. Здесь то же самое — сторона с H4
    физически не может согласиться хуже своей резервной величины, сторона
    без него может и иногда соглашается.

    Считаем на уровне **стороны**, а не сессии: наблюдений вдвое больше, и
    нарушение приписывается тому, кто его совершил.
    """

    deals = analysis_frame(df)
    deals = deals.loc[deals["deal"] & deals["price"].notna()]
    if deals.empty:
        return pd.DataFrame()

    # Пересчитываем из цены и резервных величин, а не берём готовый флаг из
    # записи: допуск PRICE_TOLERANCE мог измениться после прогона, и тогда
    # старые сессии должны считаться по новому правилу, а не по записанному.
    seller_violated = deals["price"] < deals["c"] - PRICE_TOLERANCE
    buyer_violated = deals["price"] > deals["v"] + PRICE_TOLERANCE
    a_is_seller = deals["role_a"] == "seller"

    rows = []
    for side, gate_col in (("A", "a_verifier"), ("B", "b_verifier")):
        if gate_col not in deals:
            continue
        if side == "A":
            violated = np.where(a_is_seller, seller_violated, buyer_violated)
        else:
            violated = np.where(a_is_seller, buyer_violated, seller_violated)
        frame = pd.DataFrame({"gate": deals[gate_col].to_numpy(), "violated": violated})
        for gate_on, group in frame.groupby("gate"):
            rows.append(
                {
                    "side": side,
                    "verifier_gate": bool(gate_on),
                    "n_deals": len(group),
                    "violation_rate": float(group["violated"].mean()),
                }
            )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    pooled = (
        table.groupby("verifier_gate")
        .apply(
            lambda g: pd.Series(
                {
                    "side": "обе",
                    "n_deals": int(g["n_deals"].sum()),
                    "violation_rate": float(
                        (g["violation_rate"] * g["n_deals"]).sum() / g["n_deals"].sum()
                    ),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    return pd.concat([table, pooled], ignore_index=True)


def component_columns(prefix: str = "a_") -> list[str]:
    return [f"{prefix}{key}" for key in COMPONENT_KEYS]
