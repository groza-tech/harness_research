"""Графики отчёта. Один файл — одна фигура, все пути возвращаются наверх.

Оформление намеренно скупое: чёрно-белая печать ВАК-журнала съедает
градиенты и полупрозрачность. Различаем ряды формой маркера и штрихом, а не
только цветом.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # отчёты собираются на сервере без дисплея

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from harness_asymmetry.schemas import COMPONENT_CODES, COMPONENT_LABELS_RU


plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 140,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.autolayout": True,
    }
)

_PALETTE = ["#22415f", "#c1663a", "#4f7f6a", "#8a6b9e", "#8c8c8c", "#a83a52"]


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_variance_decomposition(table: pd.DataFrame, path: Path) -> Path | None:
    """Главный слайд §6.2: сколько дисперсии от модели, сколько от обвязки."""

    if table.empty or "eta_sq_pct" not in table:
        return None
    data = table.loc[table["source"] != "error"].copy()
    if data.empty:
        return None
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(data))]
    bars = ax.barh(data["source"], data["eta_sq_pct"], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("доля объяснённой дисперсии $\\phi^A$, %")
    ax.set_title("Разложение дисперсии доли излишка")
    for bar, value in zip(bars, data["eta_sq_pct"]):
        ax.text(
            bar.get_width() + 0.6,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%",
            va="center",
            fontsize=8,
        )
    ax.set_xlim(0, max(100.0, float(data["eta_sq_pct"].max()) * 1.15))
    return _save(fig, path)


def fig_asymmetry_gradient(rent: pd.DataFrame, path: Path) -> Path | None:
    """Ядро статьи (Э3): рента $\\rho$ по знаковому превосходству A.

    Ось абсцисс — ``net_delta`` = (компонентов у A и нет у B) − (обратное), а
    не $|\\Delta|_1$. Модуль асимметрии одинаков и когда A сильнее, и когда
    слабее, поэтому усреднение по нему гасит эффект в ноль — на графике
    получилась бы горизонталь при живом эффекте.
    """

    if rent.empty or "net_delta" not in rent:
        return None
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    regimes = sorted(rent["info_regime"].dropna().unique()) if "info_regime" in rent else [None]
    markers = ["o", "s", "^", "D"]
    for idx, regime in enumerate(regimes):
        sub = rent if regime is None else rent.loc[rent["info_regime"] == regime]
        grouped = sub.groupby("net_delta").agg(
            rho=("rho", "mean"),
            low=("rho_ci_low", "mean"),
            high=("rho_ci_high", "mean"),
        ).reset_index()
        if grouped.empty:
            continue
        ax.errorbar(
            grouped["net_delta"],
            100 * grouped["rho"],
            yerr=[
                100 * (grouped["rho"] - grouped["low"]),
                100 * (grouped["high"] - grouped["rho"]),
            ],
            marker=markers[idx % len(markers)],
            capsize=3,
            linewidth=1.4,
            color=_PALETTE[idx % len(_PALETTE)],
            label=f"режим {regime}" if regime else "все режимы",
        )
    ax.axhline(0, color="black", linewidth=0.9, linestyle="--")
    ax.axvline(0, color="black", linewidth=0.7, alpha=0.4)
    ax.set_xlabel("превосходство A в обвязке, компонентов ($\\sum_k \\Delta_k$)")
    ax.set_ylabel("рента $\\rho(\\Delta)$, п.п. доли излишка")
    ax.set_title("Градиент асимметрии: рента от превосходства в обвязке")
    ax.legend(frameon=False)
    return _save(fig, path)


def fig_screening_effects(effects: pd.DataFrame, path: Path) -> Path | None:
    """Э2: торнадо главных эффектов компонентов с доверительными интервалами."""

    if effects.empty:
        return None
    data = effects.iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    y = np.arange(len(data))
    colors = ["#22415f" if s else "#b0b0b0" for s in data["survives"]]
    ax.barh(y, data["main_effect_pp"], color=colors)
    ax.errorbar(
        data["main_effect_pp"],
        y,
        xerr=[
            data["main_effect_pp"] - data["ci_low_pp"],
            data["ci_high_pp"] - data["main_effect_pp"],
        ],
        fmt="none",
        ecolor="black",
        capsize=3,
        linewidth=1.0,
    )
    ax.set_yticks(y)
    ax.set_yticklabels([COMPONENT_LABELS_RU.get(c, c) for c in data["component"]])
    ax.axvline(0, color="black", linewidth=0.9)
    ax.set_xlabel("главный эффект на $\\phi^A$, п.п.")
    ax.set_title("Скрининг компонентов (план Плакетта–Бёрмана)")
    return _save(fig, path)


def fig_saturation(curve: pd.DataFrame, path: Path) -> Path | None:
    """Э1: эффективность по уровню симметричной обвязки — проверка H5."""

    if curve.empty:
        return None
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.plot(
        curve["level"],
        curve["efficiency_mean"],
        marker="o",
        color=_PALETTE[0],
        linewidth=1.6,
        label="аллокативная эффективность $E$",
    )
    ax.fill_between(
        curve["level"],
        curve["efficiency_ci_low"],
        curve["efficiency_ci_high"],
        color=_PALETTE[0],
        alpha=0.15,
    )
    ax.plot(
        curve["level"],
        curve["deal_rate"],
        marker="s",
        linestyle="--",
        color=_PALETTE[1],
        linewidth=1.3,
        label="доля состоявшихся сделок $D$",
    )
    ax.set_xlabel("уровень симметричной обвязки (число компонентов)")
    ax.set_ylabel("значение метрики")
    ax.set_ylim(0, 1.05)
    ax.set_title("Насыщение: отдача от наращивания обвязки")
    ax.legend(frameon=False)
    return _save(fig, path)


def fig_market_h4(market: pd.DataFrame, path: Path) -> Path | None:
    """Э5: разделение H4a/H4b — эффективность и Джини против разброса обвязок."""

    if market.empty or len(market) < 2:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    for ax, col, label, color in (
        (axes[0], "efficiency_mean", "совокупная эффективность $E$", _PALETTE[0]),
        (axes[1], "gini_shares", "неравенство долей (Джини)", _PALETTE[1]),
    ):
        ax.scatter(market["harness_var"], market[col], color=color, s=36, zorder=3)
        if market["harness_var"].nunique() > 1 and market[col].notna().sum() > 1:
            coeffs = np.polyfit(market["harness_var"], market[col].fillna(0), 1)
            xs = np.linspace(market["harness_var"].min(), market["harness_var"].max(), 50)
            ax.plot(xs, np.polyval(coeffs, xs), color=color, linewidth=1.2, linestyle="--")
            ax.set_title(f"{label}\nнаклон {coeffs[0]:+.4f}", fontsize=9)
        else:
            ax.set_title(label, fontsize=9)
        ax.set_xlabel("Var(h) — разброс обвязок в популяции")
    fig.suptitle("Рыночный уровень: растёт пирог или перераспределяется", fontsize=10)
    return _save(fig, path)


def fig_component_effects(effects: pd.DataFrame, path: Path) -> Path | None:
    """§6.1: $\\beta_k$ против $\\gamma_k$ — распределительность компонентов."""

    if effects.empty or effects["beta"].isna().all():
        return None
    data = effects.dropna(subset=["beta"])
    if data.empty:
        return None
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    x = np.arange(len(data))
    width = 0.38
    ax.bar(x - width / 2, 100 * data["beta"], width, label=r"$\beta_k$ (свой компонент)", color=_PALETTE[0])
    ax.bar(x + width / 2, 100 * data["gamma"], width, label=r"$\gamma_k$ (тот же у противника)", color=_PALETTE[1])
    ax.set_xticks(x)
    ax.set_xticklabels([COMPONENT_CODES.get(c, c) for c in data["component"]])
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_ylabel("эффект на $\\phi^A$, п.п.")
    ax.set_title(r"Собственный и зеркальный эффекты: $\gamma_k \approx -\beta_k$ ⇒ игра распределительная")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, path)


def fig_cost_vs_rent(sessions: pd.DataFrame, rent: pd.DataFrame, path: Path) -> Path | None:
    """Сопоставление ренты компонента с его ценой в токенах (§5.2).

    Именно этот график отвечает на вопрос, которого нет в токеномической
    литературе: сколько излишка покупается экономией на контексте.
    """

    if rent.empty or sessions.empty:
        return None
    tokens = sessions.groupby("harness_a")["total_tokens"].mean()
    data = rent.copy()
    data["tokens_mean"] = data["harness_a"].map(tokens)
    data = data.dropna(subset=["tokens_mean"])
    if data.empty:
        return None
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    ax.scatter(data["tokens_mean"], 100 * data["rho"], s=38, color=_PALETTE[0], zorder=3)
    for _, row in data.iterrows():
        ax.annotate(
            row["harness_a"],
            (row["tokens_mean"], 100 * row["rho"]),
            fontsize=7,
            xytext=(3, 3),
            textcoords="offset points",
        )
    ax.axhline(0, color="black", linewidth=0.9, linestyle="--")
    ax.set_xlabel("средний расход токенов на сессию")
    ax.set_ylabel("рента $\\rho$, п.п.")
    ax.set_title("Цена компонента против захваченной ренты")
    return _save(fig, path)


def fig_run_health(sessions: pd.DataFrame, path: Path) -> Path | None:
    """Операционный график для отчёта о прогоне: причины остановки сессий."""

    if sessions.empty or "stop_reason" not in sessions:
        return None
    counts = sessions["stop_reason"].fillna("unknown").value_counts()
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    colors = [
        "#4f7f6a" if reason in {"completed", "no_deal"} else "#a83a52"
        for reason in counts.index
    ]
    ax.bar(counts.index.astype(str), counts.to_numpy(), color=colors)
    ax.set_ylabel("сессий")
    ax.set_title("Причины завершения сессий")
    ax.tick_params(axis="x", rotation=25)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    return _save(fig, path)


def build_all(
    *,
    figures_dir: Path,
    sessions: pd.DataFrame,
    rent: pd.DataFrame,
    screening: pd.DataFrame,
    saturation: pd.DataFrame,
    market: pd.DataFrame,
    variance: pd.DataFrame,
    component_effects: pd.DataFrame,
) -> dict[str, Path]:
    """Строит все фигуры, которые возможны на этих данных. Пустые пропускает."""

    figures_dir.mkdir(parents=True, exist_ok=True)
    candidates: dict[str, Any] = {
        "variance_decomposition": (fig_variance_decomposition, variance),
        "asymmetry_gradient": (fig_asymmetry_gradient, rent),
        "screening_effects": (fig_screening_effects, screening),
        "saturation": (fig_saturation, saturation),
        "market_h4": (fig_market_h4, market),
        "component_effects": (fig_component_effects, component_effects),
        "run_health": (fig_run_health, sessions),
    }
    built: dict[str, Path] = {}
    for name, (func, data) in candidates.items():
        try:
            result = func(data, figures_dir / f"{name}.png")
        except Exception:  # noqa: BLE001 - отсутствие графика не должно ронять отчёт
            result = None
        if result is not None:
            built[name] = result

    try:
        cost = fig_cost_vs_rent(sessions, rent, figures_dir / "cost_vs_rent.png")
        if cost is not None:
            built["cost_vs_rent"] = cost
    except Exception:  # noqa: BLE001
        pass
    return built
