"""Эконометрика §6: основная спецификация, разложение дисперсии, тест H4a/H4b.

Основная спецификация (§6.1):

$$\\phi^A_{ijs} = \\alpha + \\sum_k \\beta_k h^A_k + \\sum_k \\gamma_k h^B_k
+ \\sum_{k<l} \\theta_{kl} h^A_k h^A_l + \\mu_j + \\eta_s + \\varepsilon_{ijs}$$

где $\\mu_j$ — фиксированный эффект модели, $\\eta_s$ — фиксированный эффект
сценария $(v,c)$, стандартные ошибки кластеризуются по сценарию.

Ключевая проверка, ради которой всё это считается: $\\gamma_k \\approx
-\\beta_k$ означает строго распределительную игру — компонент у меня
прибавляет ровно столько, сколько отнимает у противника. Систематическое
отличие — признак, что пирог меняется, а не только делится. Этот тест
выведен в отдельную функцию :func:`distributive_test`.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from harness_asymmetry.analysis.metrics import analysis_frame
from harness_asymmetry.schemas import COMPONENT_CODES, COMPONENT_KEYS


def _deals_frame(df: pd.DataFrame) -> pd.DataFrame:
    clean = analysis_frame(df)
    if clean.empty:
        return clean
    return clean.loc[clean["deal"] & clean["phi_a"].notna()].copy()


# ---------------------------------------------------------------------------
# §6.1 Основная спецификация
# ---------------------------------------------------------------------------


def main_specification(
    df: pd.DataFrame,
    *,
    interactions: bool = True,
    scenario_fe: bool = True,
    model_fe: bool = True,
    role_covariate: bool = True,
) -> dict[str, Any]:
    """OLS доли излишка на компоненты обеих сторон с FE и кластеризацией.

    ``role_covariate`` включает роль стороны A в регрессоры: §8 требует
    учитывать ролевую асимметрию LLM-агентов ковариатой, иначе эффект
    харнесса рискует оказаться переодетым эффектом якорения покупателя.

    Возвращает словарь с таблицей коэффициентов, $R^2$ и диагностикой, а не
    объект statsmodels: отчёты должны собираться из данных, а не из
    ``.summary()``-строк.
    """

    import statsmodels.formula.api as smf

    deals = _deals_frame(df)
    if len(deals) < 10:
        return {"ok": False, "reason": f"слишком мало сделок: {len(deals)}"}

    terms: list[str] = []
    terms += [f"a_{k}" for k in COMPONENT_KEYS if deals[f"a_{k}"].nunique() > 1]
    terms += [f"b_{k}" for k in COMPONENT_KEYS if deals[f"b_{k}"].nunique() > 1]
    if not terms:
        return {"ok": False, "reason": "нет варьирующихся компонентов харнесса"}

    if interactions:
        own = [t for t in terms if t.startswith("a_")]
        for i, first in enumerate(own):
            for second in own[i + 1 :]:
                pair = f"{first}:{second}"
                # Пара, которая никогда не встречается вместе, даёт
                # вырожденный столбец — statsmodels его молча выкинет, но
                # лучше не подавать вовсе.
                if (deals[first] * deals[second]).nunique() > 1:
                    terms.append(pair)

    if role_covariate and deals["role_a"].nunique() > 1:
        terms.append("C(role_a)")
    if model_fe and deals["model_a"].nunique() > 1:
        terms.append("C(model_a)")
    if scenario_fe and deals["scenario_id"].nunique() > 1:
        terms.append("C(scenario_id)")

    formula = "phi_a ~ " + " + ".join(terms)
    model = smf.ols(formula, data=deals)
    if scenario_fe and deals["scenario_id"].nunique() > 1:
        fit = model.fit(cov_type="cluster", cov_kwds={"groups": deals["scenario_id"]})
        se_kind = "cluster(scenario_id)"
    else:
        fit = model.fit(cov_type="HC1")
        se_kind = "HC1"

    coefs = _coef_table(fit)
    return {
        "ok": True,
        "formula": formula,
        "se_kind": se_kind,
        "n_obs": int(fit.nobs),
        "r_squared": float(fit.rsquared),
        "r_squared_adj": float(fit.rsquared_adj),
        "coefficients": coefs,
        "component_effects": _component_effects(coefs),
    }


def _coef_table(fit: Any) -> pd.DataFrame:
    conf = fit.conf_int()
    return pd.DataFrame(
        {
            "term": fit.params.index,
            "coef": fit.params.to_numpy(),
            "std_err": fit.bse.to_numpy(),
            "t": fit.tvalues.to_numpy(),
            "p_value": fit.pvalues.to_numpy(),
            "ci_low": conf[0].to_numpy(),
            "ci_high": conf[1].to_numpy(),
        }
    ).reset_index(drop=True)


def _component_effects(coefs: pd.DataFrame) -> pd.DataFrame:
    """Сводит $\\beta_k$ (свой компонент) и $\\gamma_k$ (тот же у противника)."""

    indexed = coefs.set_index("term")
    rows = []
    for key in COMPONENT_KEYS:
        beta_term, gamma_term = f"a_{key}", f"b_{key}"
        beta = indexed.loc[beta_term] if beta_term in indexed.index else None
        gamma = indexed.loc[gamma_term] if gamma_term in indexed.index else None
        rows.append(
            {
                "component": key,
                "code": COMPONENT_CODES[key],
                "beta": float(beta["coef"]) if beta is not None else np.nan,
                "beta_se": float(beta["std_err"]) if beta is not None else np.nan,
                "beta_p": float(beta["p_value"]) if beta is not None else np.nan,
                "beta_ci_low": float(beta["ci_low"]) if beta is not None else np.nan,
                "beta_ci_high": float(beta["ci_high"]) if beta is not None else np.nan,
                "gamma": float(gamma["coef"]) if gamma is not None else np.nan,
                "gamma_se": float(gamma["std_err"]) if gamma is not None else np.nan,
                "gamma_p": float(gamma["p_value"]) if gamma is not None else np.nan,
            }
        )
    table = pd.DataFrame(rows)
    # β + γ ≈ 0 ⇔ строго распределительная игра.
    table["beta_plus_gamma"] = table["beta"] + table["gamma"]
    return table


def distributive_test(component_effects: pd.DataFrame) -> pd.DataFrame:
    """Насколько компонент «перекладывает» излишек, а не создаёт его.

    Для строго распределительной игры $\\gamma_k = -\\beta_k$. Отношение
    $|\\beta + \\gamma| / |\\beta|$ показывает долю эффекта, которая НЕ
    объясняется простым переносом доли от противника.
    """

    out = component_effects.copy()
    denom = out["beta"].abs().replace(0, np.nan)
    out["non_distributive_share"] = (out["beta"] + out["gamma"]).abs() / denom
    out["verdict"] = np.where(
        out["non_distributive_share"] < 0.25,
        "распределительный",
        np.where(out["non_distributive_share"] < 0.75, "смешанный", "меняет пирог"),
    )
    # Там, где сторона B не варьировалась (Э2 держит её голой), γ не оценён.
    # Пустой γ уходит в NaN, и формула выше молча объявила бы компонент
    # «меняющим пирог» — вывод, которого данные не поддерживают. Тест
    # применим только к планам, где обе стороны варьируются (Э3, Э4).
    out.loc[out["gamma"].isna(), "verdict"] = "γ не оценён: противник не варьировался"
    return out[["component", "code", "beta", "gamma", "beta_plus_gamma",
                "non_distributive_share", "verdict"]]


# ---------------------------------------------------------------------------
# §6.2 Разложение дисперсии — «главный слайд»
# ---------------------------------------------------------------------------


def variance_decomposition(
    df: pd.DataFrame, *, dependent: str = "auto"
) -> pd.DataFrame:
    """Двухфакторный ANOVA: сколько дисперсии от модели, сколько от обвязки.

    Это главный слайд статьи (§6.2). Если обвязка объясняет сопоставимо или
    больше — получен современный аналог результата Годе–Сандера.

    **Выбор зависимой переменной не косметика.** На симметричном плане (Э1,
    где обе стороны получают одинаковую конфигурацию) доля излишка
    $\\phi^A$ равна примерно 0,5 *по построению*: стороны оснащены
    одинаково, делить нечего. Раскладывать её дисперсию бессмысленно —
    получится разложение шума вокруг половины, и «харнесс объясняет 0,3%»
    будет выглядеть как содержательный результат, не будучи им.
    Осмысленный отклик на симметричных данных — аллокативная эффективность
    $E$: обвязка влияет на то, насколько быстро и надёжно стороны находят
    сделку, а не на то, кому достанется больше.

    ``dependent="auto"`` выбирает $\\phi^A$, если в данных есть асимметричные
    ячейки, и $E$ — если план целиком симметричен. Выбор пишется в колонку
    ``dependent``, чтобы отчёт не мог о нём умолчать.

    Считаем через ``anova_lm(typ=2)``: тип II корректен при
    несбалансированных ячейках, которые неизбежны после отсева технических
    сбоев. Доли — $\\eta^2$ от суммы квадратов.
    """

    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    clean = analysis_frame(df)
    if clean.empty:
        return pd.DataFrame()

    asymmetric_share = float((clean["harness_a"] != clean["harness_b"]).mean())
    if dependent == "auto":
        dependent = "phi_a" if asymmetric_share > 0.05 else "efficiency"

    if dependent == "phi_a":
        deals = _deals_frame(df)
    else:
        # Эффективность определена и для несостоявшихся сделок (там она ноль),
        # поэтому берём все сессии: провал торга — полноценное наблюдение.
        deals = clean.loc[clean["efficiency"].notna()]
    if deals.empty:
        return pd.DataFrame()

    deals = deals.copy()
    deals["harness_cfg"] = deals["harness_a"] + "|" + deals["harness_b"]
    factors = []
    if deals["model_a"].nunique() > 1:
        factors.append("C(model_a)")
    if deals["harness_cfg"].nunique() > 1:
        factors.append("C(harness_cfg)")
    if len(factors) < 1:
        return pd.DataFrame()

    formula = f"{dependent} ~ " + " + ".join(factors)
    if len(factors) == 2:
        formula += " + C(model_a):C(harness_cfg)"
    try:
        fit = smf.ols(formula, data=deals).fit()
        table = sm.stats.anova_lm(fit, typ=2)
    except Exception as exc:  # noqa: BLE001 - вырожденные планы бывают
        return pd.DataFrame([{"source": "error", "detail": str(exc)}])

    total_ss = float(table["sum_sq"].sum())
    labels = {
        "C(model_a)": "Модель",
        "C(harness_cfg)": "Харнесс",
        "C(model_a):C(harness_cfg)": "Модель × Харнесс",
        "Residual": "Остаток (сценарий, шум LLM)",
    }
    rows = []
    for source, row in table.iterrows():
        rows.append(
            {
                "source": labels.get(str(source), str(source)),
                "sum_sq": float(row["sum_sq"]),
                "df": float(row["df"]),
                "eta_sq_pct": round(100 * float(row["sum_sq"]) / total_ss, 2)
                if total_ss > 0
                else np.nan,
                "F": float(row.get("F", np.nan)),
                "p_value": float(row.get("PR(>F)", np.nan)),
                "dependent": dependent,
                "asymmetric_share": round(asymmetric_share, 3),
                "n_obs": int(len(deals)),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# §6.3 Разделение H4a / H4b
# ---------------------------------------------------------------------------


def h4_separation(market: pd.DataFrame) -> dict[str, Any]:
    """Две регрессии на данных Э5: $E$ и Джини на разброс обвязок.

    * $a_1 > 0$, $b_1 > 0$ → пирог растёт вместе с неравенством (H4a,
      Парето-аргумент: регулировать нечего);
    * $a_1 \\approx 0$, $b_1 > 0$ → чистое перераспределение (H4b,
      аргумент для регулятора);
    * $a_1 < 0$, $b_1 > 0$ → худший случай: асимметрия и уменьшает пирог, и
      перекашивает доли.

    Порог «$\\approx 0$» не выдумываем: считаем доверительный интервал и
    смотрим, накрывает ли он ноль.
    """

    if market.empty or len(market) < 3:
        return {"ok": False, "reason": f"мало рынков для регрессии: {len(market)}"}

    import statsmodels.formula.api as smf

    data = market.copy()
    results: dict[str, Any] = {"ok": True, "n_markets": len(data)}
    for label, dep in (("efficiency", "efficiency_mean"), ("gini", "gini_shares")):
        if data[dep].notna().sum() < 3:
            results[label] = {"ok": False, "reason": "нет данных"}
            continue
        fit = smf.ols(f"{dep} ~ harness_var", data=data).fit(cov_type="HC1")
        conf = fit.conf_int()
        results[label] = {
            "ok": True,
            "coef": float(fit.params["harness_var"]),
            "std_err": float(fit.bse["harness_var"]),
            "p_value": float(fit.pvalues["harness_var"]),
            "ci_low": float(conf.loc["harness_var", 0]),
            "ci_high": float(conf.loc["harness_var", 1]),
            "r_squared": float(fit.rsquared),
        }

    eff, gini_res = results.get("efficiency", {}), results.get("gini", {})
    results["verdict"] = _h4_verdict(eff, gini_res)
    return results


def _h4_verdict(eff: dict[str, Any], gini_res: dict[str, Any]) -> str:
    if not eff.get("ok") or not gini_res.get("ok"):
        return "недостаточно данных для разделения H4a/H4b"
    a1_zero = eff["ci_low"] <= 0 <= eff["ci_high"]
    b1_pos = gini_res["ci_low"] > 0
    b1_zero = gini_res["ci_low"] <= 0 <= gini_res["ci_high"]
    if b1_zero:
        return (
            "разброс обвязок не сдвигает неравенство долей значимо — "
            "ни H4a, ни H4b не подтверждаются на этих данных"
        )
    if not b1_pos:
        return "неравенство долей падает с разбросом обвязок — исход вне H4a/H4b"
    if a1_zero:
        return "H4b: чистое перераспределение — пирог не меняется, доли расходятся (аргумент для регулятора)"
    if eff["ci_low"] > 0:
        return "H4a: пирог растёт вместе с неравенством (Парето-аргумент, регулировать нечего)"
    return "худший случай: асимметрия и уменьшает пирог, и перекашивает доли"


# ---------------------------------------------------------------------------
# Э2. Главные эффекты скрининга Плакетта–Бёрмана
# ---------------------------------------------------------------------------


def screening_effects(df: pd.DataFrame, *, n_boot: int = 5000, seed: int = 7) -> pd.DataFrame:
    """Главные эффекты компонентов на $\\phi^A$ по плану PB-12.

    Главный эффект компонента — разность средних $\\phi^A$ между
    конфигурациями с включённым и выключенным компонентом. Ортогональность
    плана делает эти разности независимыми оценками.
    """

    from harness_asymmetry.analysis.metrics import bootstrap_diff_ci

    deals = _deals_frame(df)
    deals = deals.loc[deals["experiment"] == "E2"] if "experiment" in deals else deals
    if deals.empty:
        return pd.DataFrame()

    rows = []
    for key in COMPONENT_KEYS:
        col = f"a_{key}"
        on = deals.loc[deals[col] == 1, "phi_a"].to_numpy(dtype=float)
        off = deals.loc[deals[col] == 0, "phi_a"].to_numpy(dtype=float)
        if on.size == 0 or off.size == 0:
            continue
        effect = float(on.mean() - off.mean())
        low, high = bootstrap_diff_ci(on, off, n_boot=n_boot, seed=seed)
        rows.append(
            {
                "component": key,
                "code": COMPONENT_CODES[key],
                "n_on": int(on.size),
                "n_off": int(off.size),
                "phi_on": float(on.mean()),
                "phi_off": float(off.mean()),
                "main_effect_pp": 100 * effect,
                "ci_low_pp": 100 * low,
                "ci_high_pp": 100 * high,
                "survives": bool(low > 0 or high < 0),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.reindex(
        table["main_effect_pp"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)


def survivors_from_screening(effects: pd.DataFrame, *, max_survivors: int = 3) -> list[str]:
    """Компоненты, прошедшие скрининг: значимые, по убыванию |эффекта|."""

    if effects.empty:
        return []
    significant = effects.loc[effects["survives"]]
    if significant.empty:
        significant = effects.head(max_survivors)
    return list(significant["component"].head(max_survivors))


# ---------------------------------------------------------------------------
# Э4. Курс обмена «модель ↔ харнесс»
# ---------------------------------------------------------------------------


def exchange_rate(df: pd.DataFrame, *, level_col_a: str = "level_a", level_col_b: str = "level_b") -> pd.DataFrame:
    """Ищет уровень обвязки, при котором $\\phi^A = 0{,}5$ против сильной модели.

    Для каждой пары (класс модели A, класс модели B) строится зависимость
    средней $\\phi^A$ от разницы уровней обвязки и линейной интерполяцией
    находится пересечение с 0,5 — точка безразличия. Отношение «сколько
    уровней обвязки компенсируют переход на класс модели ниже» и есть
    искомый курс $K$ гипотезы H3.
    """

    deals = _deals_frame(df)
    deals = deals.loc[deals["experiment"] == "E4"] if "experiment" in deals else deals
    if deals.empty or level_col_a not in deals or level_col_b not in deals:
        return pd.DataFrame()

    rows = []
    for (mk_a, mk_b), group in deals.groupby(["model_class_a", "model_class_b"]):
        grid = (
            group.groupby([level_col_a, level_col_b])["phi_a"]
            .mean()
            .reset_index()
            .sort_values([level_col_b, level_col_a])
        )
        for lvl_b, sub in grid.groupby(level_col_b):
            crossing = _interpolate_crossing(
                sub[level_col_a].to_numpy(dtype=float),
                sub["phi_a"].to_numpy(dtype=float),
                target=0.5,
            )
            rows.append(
                {
                    "model_class_a": mk_a,
                    "model_class_b": mk_b,
                    "level_b": lvl_b,
                    "level_a_at_parity": crossing,
                    "n_cells": len(sub),
                    "phi_min": float(sub["phi_a"].min()),
                    "phi_max": float(sub["phi_a"].max()),
                }
            )
    return pd.DataFrame(rows)


def _interpolate_crossing(
    x: np.ndarray, y: np.ndarray, *, target: float
) -> float | None:
    """Линейная интерполяция первой точки пересечения ``y`` уровня ``target``."""

    order = np.argsort(x)
    xs, ys = x[order], y[order]
    for i in range(len(xs) - 1):
        y0, y1 = ys[i], ys[i + 1]
        if (y0 - target) * (y1 - target) <= 0 and y1 != y0:
            t = (target - y0) / (y1 - y0)
            return float(xs[i] + t * (xs[i + 1] - xs[i]))
    return None


# ---------------------------------------------------------------------------
# Э1. Насыщение (H5)
# ---------------------------------------------------------------------------


def saturation_curve(df: pd.DataFrame) -> pd.DataFrame:
    """Кривая «уровень симметричной обвязки → эффективность» с приростами.

    Убывающая отдача проверяется по приращению: если прирост эффективности
    на очередном компоненте становится статистически неотличим от нуля,
    достигнуто плато — конечная оптимальная конфигурация существует, и
    «включить всё» оптимумом не является (H5).
    """

    from harness_asymmetry.analysis.metrics import bootstrap_ci

    clean = analysis_frame(df)
    clean = clean.loc[clean["experiment"] == "E1"] if "experiment" in clean else clean
    if clean.empty or "level" not in clean:
        return pd.DataFrame()

    rows = []
    for level, group in clean.groupby("level"):
        eff = group["efficiency"].to_numpy(dtype=float)
        low, high = bootstrap_ci(eff)
        deals = group.loc[group["deal"]]
        rows.append(
            {
                "level": int(level),
                "n": len(group),
                "efficiency_mean": float(eff.mean()),
                "efficiency_ci_low": low,
                "efficiency_ci_high": high,
                "deal_rate": len(deals) / len(group),
                "rubinstein_gap_mean": float(deals["rubinstein_gap"].mean())
                if len(deals)
                else np.nan,
                "tokens_mean": float(group["total_tokens"].mean()),
            }
        )
    table = pd.DataFrame(rows).sort_values("level").reset_index(drop=True)
    table["marginal_gain"] = table["efficiency_mean"].diff()
    table["tokens_per_gain"] = table["tokens_mean"].diff() / table["marginal_gain"]
    return table


def role_balance_check(df: pd.DataFrame) -> pd.DataFrame:
    """Контроль ролевой асимметрии (§6.4): доля A по ролям и по очереди хода."""

    deals = _deals_frame(df)
    if deals.empty:
        return pd.DataFrame()
    return (
        deals.groupby(["role_a", "first_mover_side"])["phi_a"]
        .agg(["count", "mean", "std"])
        .reset_index()
        .rename(columns={"count": "n", "mean": "phi_a_mean", "std": "phi_a_sd"})
    )


def prompt_robustness(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Сравнение эффекта между вариантами промпта (§6.4).

    На вход — словарь ``{вариант промпта: sessions}``. Эффект обязан
    сохраняться на всех вариантах; расхождение знака означает, что измерялась
    формулировка, а не обвязка.
    """

    rows = []
    for variant, frame in frames.items():
        deals = _deals_frame(frame)
        if deals.empty:
            continue
        asym = deals.loc[deals["harness_a"] != deals["harness_b"], "phi_a"]
        sym = deals.loc[deals["harness_a"] == deals["harness_b"], "phi_a"]
        rows.append(
            {
                "prompt_variant": variant,
                "n_asymmetric": len(asym),
                "n_symmetric": len(sym),
                "phi_asymmetric": float(asym.mean()) if len(asym) else np.nan,
                "phi_symmetric": float(sym.mean()) if len(sym) else np.nan,
                "rent_pp": 100 * (float(asym.mean()) - float(sym.mean()))
                if len(asym) and len(sym)
                else np.nan,
            }
        )
    return pd.DataFrame(rows)
