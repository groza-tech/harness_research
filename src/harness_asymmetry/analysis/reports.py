"""Специальные отчёты: операционный, научный, дайджест логов и HTML.

Прогон производит четыре артефакта, у каждого свой читатель:

``RUN_REPORT.md``
    Операционный отчёт. Отвечает на вопрос «прогон вообще здоров?»: доля
    технических сбоев, невалидных выводов, сработавшие предохранители,
    здоровье провайдера, токены, латентность, стоимость. Именно эти цифры
    §4.2 и §7.2 требуют публиковать, а не прятать.

``RESULTS.md``
    Научный отчёт по разделам дизайн-дока: разложение дисперсии,
    скрининг компонентов, градиент асимметрии, курс обмена, рыночный
    уровень, робастность. Каждая таблица подписана тем, какую гипотезу она
    проверяет.

``LOG_DIGEST.md``
    Дайджест ``events.jsonl``: события по типам, самые долгие вызовы,
    ошибки, залипания. Полный лог остаётся машиночитаемым; дайджест — для
    человека.

``report.html``
    Самодостаточная страница со всеми таблицами и графиками (картинки
    вшиты как data-URI) — чтобы отправить научруку одним файлом.

Все отчёты честно помечают прогон на mock-провайдере: такие данные
синтетические, и выводы по ним делать нельзя.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from harness_asymmetry.analysis import econometrics as econ
from harness_asymmetry.analysis import figures as figs
from harness_asymmetry.analysis import metrics as met
from harness_asymmetry.observability import iter_events
from harness_asymmetry.schemas import (
    COMPONENT_CODES,
    COMPONENT_ECONOMICS_RU,
    COMPONENT_KEYS,
    COMPONENT_LABELS_RU,
)


MOCK_BANNER = (
    "> ⚠️ **ПРОГОН НА MOCK-ПРОВАЙДЕРЕ.** Ответы сгенерированы детерминированным "
    "скриптом, а не языковой моделью. Эффект харнесса здесь возникает по "
    "построению. Отчёт пригоден только для проверки инфраструктуры "
    "(протокол, гейты, чекпоинты, метрики, отчёты) и **не является "
    "результатом исследования**.\n"
)


@dataclass(slots=True)
class ReportBundle:
    """Что получилось собрать. Пути — для CLI и для тестов."""

    reports_dir: Path
    run_report: Path
    results_report: Path
    log_digest: Path
    html_report: Path
    tables: dict[str, Path]
    figures: dict[str, Path]
    analytics: dict[str, Any]


# ---------------------------------------------------------------------------
# Утилиты форматирования
# ---------------------------------------------------------------------------


def _md_table(df: pd.DataFrame | None, *, floatfmt: str = ".3f", max_rows: int = 40) -> str:
    if df is None or len(df) == 0:
        return "_нет данных_\n"
    view = df.head(max_rows)
    text = view.to_markdown(index=False, floatfmt=floatfmt)
    if len(df) > max_rows:
        text += f"\n\n_показаны первые {max_rows} из {len(df)} строк; полная таблица — в `tables/`._"
    return text + "\n"


def _pct(value: Any, digits: int = 1) -> str:
    try:
        return f"{float(value):.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


def _num(value: Any, digits: int = 3) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{f:.{digits}f}"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------------------
# Аналитика
# ---------------------------------------------------------------------------


def compute_analytics(sessions: pd.DataFrame) -> dict[str, Any]:
    """Считает всё, что можно посчитать на этих данных. Пустое — не ошибка."""

    analytics: dict[str, Any] = {}
    analytics["failures"] = met.failure_summary(sessions)
    analytics["cells"] = met.cell_metrics(sessions)
    analytics["rent"] = met.harness_rent(sessions)
    analytics["screening"] = econ.screening_effects(sessions)
    analytics["survivors"] = econ.survivors_from_screening(analytics["screening"])
    analytics["saturation"] = econ.saturation_curve(sessions)
    analytics["market"] = met.market_metrics(sessions)
    analytics["h4"] = econ.h4_separation(analytics["market"])
    analytics["variance"] = econ.variance_decomposition(sessions)
    analytics["spec"] = econ.main_specification(sessions)
    component_effects = (
        analytics["spec"].get("component_effects")
        if analytics["spec"].get("ok")
        else pd.DataFrame()
    )
    analytics["component_effects"] = (
        component_effects if isinstance(component_effects, pd.DataFrame) else pd.DataFrame()
    )
    analytics["distributive"] = (
        econ.distributive_test(analytics["component_effects"])
        if not analytics["component_effects"].empty
        else pd.DataFrame()
    )
    analytics["exchange_rate"] = econ.exchange_rate(sessions)
    analytics["budget_discipline"] = met.budget_discipline(sessions)
    analytics["role_balance"] = econ.role_balance_check(sessions)
    analytics["pilot_power"] = met.pilot_power_report(
        sessions.loc[sessions["experiment"] == "PILOT"]
        if "experiment" in sessions and not sessions.empty
        else sessions
    )
    return analytics


def digest_events(events_path: Path, *, top_n: int = 10) -> dict[str, Any]:
    """Сводка ``events.jsonl`` — без чтения всего файла в память дважды."""

    counts: dict[str, int] = {}
    errors: list[dict[str, Any]] = []
    breakers: list[dict[str, Any]] = []
    parse_fails: list[dict[str, Any]] = []
    latencies: list[tuple[float, str]] = []
    gate_by_component: dict[str, int] = {}
    tokens_total = 0

    for event in iter_events(events_path):
        etype = str(event.get("event_type", "?"))
        counts[etype] = counts.get(etype, 0) + 1
        if etype == "llm.call":
            tokens_total += int(event.get("input_tokens", 0) or 0)
            tokens_total += int(event.get("output_tokens", 0) or 0)
            latency = float(event.get("latency_ms", 0) or 0)
            latencies.append((latency, str(event.get("tag", ""))))
        elif etype in {"llm.error", "llm.retry"}:
            if len(errors) < top_n:
                errors.append(event)
        elif etype == "harness.circuit_breaker":
            breakers.append(event)
        elif etype == "action.parse_fail":
            if len(parse_fails) < top_n:
                parse_fails.append(event)
        elif etype == "harness.component" and event.get("kind") == "gate_violation":
            comp = str(event.get("component", "?"))
            gate_by_component[comp] = gate_by_component.get(comp, 0) + 1

    latencies.sort(reverse=True)
    return {
        "counts": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        "tokens_total": tokens_total,
        "slowest_calls_ms": latencies[:top_n],
        "errors": errors,
        "circuit_breakers": breakers,
        "parse_failures": parse_fails,
        "gate_violations_by_component": dict(
            sorted(gate_by_component.items(), key=lambda kv: -kv[1])
        ),
    }


# ---------------------------------------------------------------------------
# RUN_REPORT.md
# ---------------------------------------------------------------------------


def build_run_report(
    *,
    path: Path,
    manifest: dict[str, Any],
    run_meta: dict[str, Any],
    sessions: pd.DataFrame,
    analytics: dict[str, Any],
    events: dict[str, Any],
    figures: dict[str, Path],
    is_mock: bool,
) -> Path:
    """Операционный отчёт: здоров ли прогон и во что он обошёлся."""

    lines: list[str] = ["# Отчёт о прогоне\n"]
    if is_mock:
        lines.append(MOCK_BANNER)

    lines.append("## 1. Идентификация и пиннинг\n")
    lines.append(
        f"- **run_id:** `{manifest.get('run_id', '—')}`\n"
        f"- **дата:** {manifest.get('created_at', '—')}\n"
        f"- **провайдер:** `{manifest.get('provider', '—')}`\n"
        f"- **вариант промпта:** `{manifest.get('prompt_variant', '—')}`\n"
        f"- **отпечаток пула сценариев (CRN):** `{manifest.get('scenario_pool_fingerprint', '—')}` "
        f"({manifest.get('scenario_pool_size', '—')} розыгрышей)\n"
        f"- **python:** {manifest.get('environment', {}).get('python', '—')}, "
        f"{manifest.get('environment', {}).get('platform', '—')}\n"
    )
    models = manifest.get("models", {})
    if models:
        lines.append("\n**Модели (запиннены):**\n")
        lines.extend(f"- `{k}` → `{v}`" for k, v in models.items())
        lines.append("")
    hashes = manifest.get("prompt_hashes", {})
    if hashes:
        lines.append("\n**SHA-256 промптов** (правка формулировки меняет хеш):\n")
        lines.append(
            _md_table(
                pd.DataFrame(
                    {"шаблон": list(hashes), "sha256[:16]": list(hashes.values())}
                ),
                max_rows=100,
            )
        )
    pkgs = manifest.get("environment", {}).get("packages", {})
    if pkgs:
        lines.append("**Версии пакетов:** " + ", ".join(f"`{k}={v}`" for k, v in pkgs.items()) + "\n")

    lines.append("\n## 2. Объём и здоровье прогона\n")
    failures = analytics.get("failures", {})
    lines.append(
        f"- сессий всего: **{failures.get('sessions', 0)}**\n"
        f"- технических сбоев: **{failures.get('failures', 0)}** "
        f"({_pct(failures.get('failure_rate_pct', 0), 2)})\n"
        f"- сессий хотя бы с одним невалидным выводом модели: "
        f"{_pct(failures.get('invalid_output_rate_pct', 0), 2)}\n"
        f"- возобновлено из чекпоинтов: {run_meta.get('sessions_resumed', 0)}\n"
        f"- время прогона: {run_meta.get('elapsed_s', '—')} с\n"
    )
    if float(failures.get("failure_rate_pct", 0) or 0) > 10:
        lines.append(
            "\n> ⚠️ Доля технических сбоев выше 10%: провайдер деградировал, часть "
            "выборки потеряна. Потеряна честно — сбойные сессии исключены, а не "
            "заменены заглушками, — но интерпретировать результат нужно с оговоркой.\n"
        )

    if not sessions.empty:
        lines.append("\n### Причины завершения сессий\n")
        stop = sessions["stop_reason"].fillna("unknown").value_counts().reset_index()
        stop.columns = ["причина", "сессий"]
        stop["доля"] = (100 * stop["сессий"] / len(sessions)).round(2)
        lines.append(_md_table(stop, floatfmt=".2f"))

    lines.append("\n## 3. Издержки: токены, латентность, деньги\n")
    health = run_meta.get("llm_health", {})
    if health:
        rows = []
        for model, stats in health.items():
            rows.append(
                {
                    "модель": model,
                    "вызовов ок": stats.get("calls_ok", 0),
                    "вызовов провалено": stats.get("calls_failed", 0),
                    "доля сбоев, %": stats.get("failure_rate_pct", 0),
                    "попадания в кеш, %": stats.get("cache_hit_rate_pct", 0),
                    "prompt-токенов": stats.get("prompt_tokens", 0),
                    "completion-токенов": stats.get("completion_tokens", 0),
                    "средняя латентность, мс": stats.get("latency_ms_mean", 0),
                    "оценка стоимости, $": stats.get("estimated_cost_usd", 0),
                }
            )
        lines.append(_md_table(pd.DataFrame(rows), floatfmt=".2f"))
    if not sessions.empty:
        cost = (
            sessions.groupby(["experiment", "harness_a", "harness_b"])
            .agg(
                сессий=("session_id", "count"),
                токенов_на_сессию=("total_tokens", "mean"),
                вызовов_на_сессию=("llm_calls", "mean"),
                латентность_мс=("latency_ms", "mean"),
            )
            .reset_index()
            .sort_values("токенов_на_сессию", ascending=False)
        )
        lines.append("\n**Расход по конфигурациям обвязки** (для сопоставления ренты с ценой компонента):\n")
        lines.append(_md_table(cost, floatfmt=".1f", max_rows=25))

    lines.append("\n## 4. Наблюдаемость харнесса\n")
    counts = events.get("counts", {})
    if counts:
        lines.append("**События в `events.jsonl`:**\n")
        lines.append(
            _md_table(
                pd.DataFrame({"событие": list(counts), "штук": list(counts.values())}),
                max_rows=30,
            )
        )
    gate = events.get("gate_violations_by_component", {})
    if gate:
        lines.append("\n**Срабатывания гейтов по компонентам** (H3 и H4 — единственные, кто отклоняет действия):\n")
        lines.append(
            _md_table(
                pd.DataFrame(
                    {
                        "компонент": [COMPONENT_LABELS_RU.get(k, k) for k in gate],
                        "отклонений": list(gate.values()),
                    }
                )
            )
        )
    breakers = events.get("circuit_breakers", [])
    if breakers:
        lines.append(
            f"\n**Сработавшие предохранители: {len(breakers)}.** Это жёсткие лимиты харнесса "
            "(раунды, токены, время, залипание на повторяющемся действии), они живут в коде, "
            "а не в промпте.\n"
        )
        lines.append(
            _md_table(
                pd.DataFrame(
                    [
                        {"сессия": b.get("session_id"), "причина": b.get("reason")}
                        for b in breakers[:20]
                    ]
                )
            )
        )
    parse_fails = events.get("parse_failures", [])
    if parse_fails:
        lines.append("\n**Примеры невалидных выводов модели** (первые записи):\n")
        lines.append(
            _md_table(
                pd.DataFrame(
                    [
                        {
                            "сессия": p.get("session_id"),
                            "сторона": p.get("side"),
                            "раунд": p.get("round"),
                            "причина": p.get("reason"),
                        }
                        for p in parse_fails
                    ]
                )
            )
        )
    slowest = events.get("slowest_calls_ms", [])
    if slowest:
        lines.append("\n**Самые долгие вызовы, мс:** " + ", ".join(f"{ms:.0f} ({tag})" for ms, tag in slowest[:5]) + "\n")

    if "run_health" in figures:
        lines.append(f"\n![Причины завершения сессий](figures/{figures['run_health'].name})\n")

    lines.append("\n## 5. Контроль дизайна\n")
    role = analytics.get("role_balance")
    if isinstance(role, pd.DataFrame) and not role.empty:
        lines.append(
            "Контрбалансировка ролей (§6.4). В литературе зафиксирована ролевая асимметрия "
            "LLM-агентов: покупательские роли систематически переигрывают поставщицкие за счёт "
            "якорения. Ячейки ниже должны быть заполнены сопоставимо, иначе эффект харнесса "
            "смешан с эффектом роли.\n"
        )
        lines.append(_md_table(role))
    pilot = analytics.get("pilot_power")
    if isinstance(pilot, pd.DataFrame) and not pilot.empty:
        lines.append(
            "\n**Пилот и мощность (§4.4).** Требуемое $n$ пересчитано по фактической дисперсии; "
            "это условие прохождения точки невозврата на неделе 6.\n"
        )
        lines.append(_md_table(pilot))

    lines.append(
        "\n## 6. Где что лежит\n"
        "- `run_manifest.json` — пиннинг: модели, промпты, сиды, конфиг, окружение;\n"
        "- `events.jsonl` — полный лог событий харнесса (одно событие = одна строка);\n"
        "- `sessions/*.jsonl` — транскрипты сессий, они же чекпоинты для resume;\n"
        "- `sessions.csv` / `turns.csv` — плоские таблицы для аналитики;\n"
        "- `run_meta.json` — итоги прогона и здоровье провайдера.\n"
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# RESULTS.md
# ---------------------------------------------------------------------------


def build_results_report(
    *,
    path: Path,
    manifest: dict[str, Any],
    sessions: pd.DataFrame,
    analytics: dict[str, Any],
    figures: dict[str, Path],
    is_mock: bool,
) -> Path:
    """Научный отчёт по разделам дизайн-дока."""

    lines: list[str] = [
        "# Асимметрия агентного харнесса как источник рыночной власти\n",
        "## Результаты прогона\n",
    ]
    if is_mock:
        lines.append(MOCK_BANNER)

    lines.append(
        f"Прогон `{manifest.get('run_id', '—')}`, провайдер `{manifest.get('provider', '—')}`, "
        f"вариант промпта `{manifest.get('prompt_variant', '—')}`, "
        f"пул сценариев `{manifest.get('scenario_pool_fingerprint', '—')}`.\n"
    )

    lines.append("\n## 0. Компоненты харнесса и их экономические имена\n")
    lines.append(
        _md_table(
            pd.DataFrame(
                [
                    {
                        "код": COMPONENT_CODES[k],
                        "компонент": COMPONENT_LABELS_RU[k],
                        "экономическая категория": COMPONENT_ECONOMICS_RU[k],
                    }
                    for k in COMPONENT_KEYS
                ]
            ),
            max_rows=10,
        )
    )

    lines.append("\n## 1. Разложение дисперсии — главный результат (§6.2)\n")
    lines.append(
        "Вопрос Годе–Сандера на языке 2026 года: интеллект — это модель, институт — это харнесс. "
        "Что из них определяет экономический исход? Если обвязка объясняет сопоставимо или больше — "
        "получен современный аналог классического результата.\n"
    )
    variance = analytics.get("variance")
    if isinstance(variance, pd.DataFrame) and not variance.empty and "dependent" in variance:
        dep = str(variance["dependent"].iloc[0])
        asym = float(variance["asymmetric_share"].iloc[0])
        if dep == "efficiency":
            lines.append(
                f"> **Отклик — аллокативная эффективность $E$, не доля $\\phi^A$.** "
                f"Асимметричных ячеек в этих данных {asym:.0%}: план практически "
                f"целиком симметричен, а при одинаковой обвязке у обеих сторон "
                f"$\\phi^A \\approx 0{{,}}5$ **по построению** — делить нечего. "
                f"Раскладывать её дисперсию значило бы раскладывать шум вокруг "
                f"половины и выдавать его за результат. Осмысленный вопрос к "
                f"симметричным данным другой: влияет ли обвязка на то, насколько "
                f"быстро и надёжно стороны вообще находят сделку.\n"
            )
        else:
            lines.append(
                f"Отклик — доля излишка $\\phi^A$; асимметричных ячеек "
                f"{asym:.0%}.\n"
            )
    lines.append(_md_table(variance, floatfmt=".2f"))
    if isinstance(variance, pd.DataFrame) and not variance.empty and "eta_sq_pct" in variance:
        lines.append(_variance_verdict(variance))
    if "variance_decomposition" in figures:
        lines.append(f"\n![Разложение дисперсии](figures/{figures['variance_decomposition'].name})\n")

    lines.append("\n## 2. Основная спецификация (§6.1)\n")
    spec = analytics.get("spec", {})
    if spec.get("ok"):
        lines.append(
            f"Формула: `{spec['formula']}`\n\n"
            f"Наблюдений (сделок): **{spec['n_obs']}**; $R^2$ = {_num(spec['r_squared'])}; "
            f"стандартные ошибки — {spec['se_kind']}.\n"
        )
        effects = analytics.get("component_effects")
        if isinstance(effects, pd.DataFrame) and not effects.empty:
            table = effects.copy()
            table["компонент"] = table["component"].map(COMPONENT_LABELS_RU)
            view = table[
                ["код" if "код" in table else "code", "компонент", "beta", "beta_se", "beta_p", "gamma", "gamma_p", "beta_plus_gamma"]
            ].rename(
                columns={
                    "code": "код",
                    "beta": "β (свой)",
                    "beta_se": "s.e.",
                    "beta_p": "p(β)",
                    "gamma": "γ (у противника)",
                    "gamma_p": "p(γ)",
                    "beta_plus_gamma": "β+γ",
                }
            )
            lines.append(_md_table(view, floatfmt=".4f"))
        distributive = analytics.get("distributive")
        if isinstance(distributive, pd.DataFrame) and not distributive.empty:
            lines.append(
                "\n**Распределительность компонентов.** $\\gamma_k \\approx -\\beta_k$ означает "
                "строго распределительную игру: компонент отнимает у противника ровно столько, "
                "сколько прибавляет мне. Систематическое отличие — признак, что меняется сам пирог.\n"
            )
            lines.append(_md_table(distributive, floatfmt=".4f"))
    else:
        lines.append(f"_Регрессия не оценена: {spec.get('reason', 'нет данных')}._\n")
    if "component_effects" in figures:
        lines.append(f"\n![Собственный и зеркальный эффекты](figures/{figures['component_effects'].name})\n")

    lines.append("\n## 3. Э1. Симметричная база и насыщение (H5)\n")
    saturation = analytics.get("saturation")
    lines.append(
        "Обе стороны получают одинаковую конфигурацию, уровень растёт от голой модели к полной "
        "обвязке. Гипотеза H5: отдача убывающая и выходит на плато — «включить всё» оптимумом "
        "не является при учёте издержек компонентов.\n"
    )
    lines.append(_md_table(saturation, floatfmt=".3f"))
    if "saturation" in figures:
        lines.append(f"\n![Насыщение](figures/{figures['saturation'].name})\n")

    lines.append("\n## 3a. Бюджетное ограничение: ZI-U против ZI-C\n")
    lines.append(
        "Прямая проверка механизма Годе–Сандера на нашем объекте. Сторона с "
        "верификационным гейтом (H4) физически не может согласиться на условия хуже "
        "собственной резервной величины; сторона без него — может, и доля таких сделок "
        "измерима. Если у конфигураций с H4 нарушений нет, а без H4 они есть, "
        "современная реализация ZI-C воспроизведена.\n"
    )
    budget = analytics.get("budget_discipline")
    lines.append(_md_table(budget, floatfmt=".4f"))
    if isinstance(budget, pd.DataFrame) and not budget.empty:
        gated = budget.loc[(budget["side"] == "обе") & (budget["verifier_gate"])]
        ungated = budget.loc[(budget["side"] == "обе") & (~budget["verifier_gate"].astype(bool))]
        if not gated.empty and not ungated.empty:
            lines.append(
                f"\nС гейтом: {_pct(100 * float(gated['violation_rate'].iloc[0]), 2)} сделок вне "
                f"своей резервной величины; без гейта: "
                f"{_pct(100 * float(ungated['violation_rate'].iloc[0]), 2)}.\n"
            )

    lines.append("\n## 4. Э2. Скрининг компонентов (H2)\n")
    lines.append(
        "План Плакетта–Бёрмана на 12 конфигураций вместо 64. Сторона A варьируется, сторона B "
        "фиксирована на голой модели. Гипотеза H2: наибольший вклад даёт память о контрагенте, "
        "а не вычислительно более тяжёлые компоненты.\n"
    )
    screening = analytics.get("screening")
    lines.append(_md_table(screening, floatfmt=".2f"))
    survivors = analytics.get("survivors") or []
    if survivors:
        lines.append(
            "\n**Выжившие компоненты:** "
            + ", ".join(COMPONENT_LABELS_RU.get(s, s) for s in survivors)
            + ". Именно они идут в полный факторный план Э3.\n"
        )
    if "screening_effects" in figures:
        lines.append(f"\n![Скрининг компонентов](figures/{figures['screening_effects'].name})\n")

    lines.append("\n## 5. Э3. Градиент асимметрии — рента $\\rho(\\Delta)$ (H1)\n")
    lines.append(
        "Рента от асимметрии харнесса — центральная величина исследования: прирост доли излишка "
        "стороны A, объяснимый исключительно разницей в обвязке при неизменной модели и неизменной "
        "задаче. Доверительные интервалы — бутстрэп разности средних.\n"
    )
    rent = analytics.get("rent")
    if isinstance(rent, pd.DataFrame) and not rent.empty:
        view = rent.copy()
        view["ρ, п.п."] = (100 * view["rho"]).round(2)
        view["CI, п.п."] = view.apply(
            lambda r: f"[{100 * r['rho_ci_low']:.1f}; {100 * r['rho_ci_high']:.1f}]", axis=1
        )
        cols = [
            c
            for c in ("model_a", "info_regime", "harness_a", "harness_b", "net_delta", "abs_delta", "n")
            if c in view
        ]
        lines.append(_md_table(view[cols + ["ρ, п.п.", "CI, п.п.", "significant"]], floatfmt=".2f"))
        lines.append(_rent_verdict(rent))
    else:
        lines.append("_Асимметричных ячеек в данных нет — рента не определена._\n")
    if "asymmetry_gradient" in figures:
        lines.append(f"\n![Градиент асимметрии](figures/{figures['asymmetry_gradient'].name})\n")

    lines.append("\n## 6. Э4. Курс обмена «модель ↔ харнесс» (H3)\n")
    lines.append(
        "Ищем точки $\\phi^A = 0{,}5$ — кривую безразличия. Численный ответ: скольким весовым "
        "классам модели эквивалентна полная обвязка.\n"
    )
    lines.append(_md_table(analytics.get("exchange_rate"), floatfmt=".2f"))

    lines.append("\n## 7. Э5. Рыночный уровень — разделение H4a/H4b\n")
    lines.append(
        "Популяция агентов, случайное паросочетание, разброс обвязок варьируется **при неизменном "
        "среднем уровне**. Это абзац, ради которого пишется вся статья.\n"
    )
    market = analytics.get("market")
    lines.append(_md_table(market, floatfmt=".3f"))
    h4 = analytics.get("h4", {})
    if h4.get("ok"):
        eff, gini_res = h4.get("efficiency", {}), h4.get("gini", {})
        lines.append(
            f"\n- $a_1$ (эффективность на Var(h)): **{_num(eff.get('coef'), 4)}**, "
            f"CI [{_num(eff.get('ci_low'), 4)}; {_num(eff.get('ci_high'), 4)}], p={_num(eff.get('p_value'))}\n"
            f"- $b_1$ (Джини на Var(h)): **{_num(gini_res.get('coef'), 4)}**, "
            f"CI [{_num(gini_res.get('ci_low'), 4)}; {_num(gini_res.get('ci_high'), 4)}], p={_num(gini_res.get('p_value'))}\n"
        )
        lines.append(f"\n**Вывод:** {h4.get('verdict', '—')}\n")
    else:
        lines.append(f"\n_Разделение не проведено: {h4.get('reason', 'нет данных Э5')}._\n")
    if "market_h4" in figures:
        lines.append(f"\n![Рыночный уровень](figures/{figures['market_h4'].name})\n")

    lines.append("\n## 8. Издержки против ренты\n")
    lines.append(
        "Компакция контекста (H6) удешевляет прогон, но может ухудшать переговорную позицию. "
        "Если подтвердится — экономия на токенах покупается уступкой излишка, чего в "
        "токеномической литературе никто не считал.\n"
    )
    if "cost_vs_rent" in figures:
        lines.append(f"\n![Цена против ренты](figures/{figures['cost_vs_rent'].name})\n")

    lines.append("\n## 9. Методологические оговорки\n")
    failures = analytics.get("failures", {})
    lines.append(
        f"- Общие случайные числа: все ячейки плана используют один пул $(v,c)$, отпечаток "
        f"`{manifest.get('scenario_pool_fingerprint', '—')}` (парный дизайн, §4.4).\n"
        f"- Роли и очередь хода контрбалансированы внутри каждой ячейки (§6.4).\n"
        f"- $\\phi^A$ считается только по состоявшимся сделкам; провал торга учитывается "
        f"отдельно метрикой $D$ и нулевой $E$ — подставлять 0 или 0,5 вместо отсутствующего "
        f"наблюдения нельзя.\n"
        f"- Технические сбои ({_pct(failures.get('failure_rate_pct', 0), 2)}) исключены из "
        f"анализа и опубликованы в `RUN_REPORT.md`, а не заменены заглушками.\n"
        f"- Анализ ведётся по вектору компонентов, а не по скалярному индексу обвязки; "
        f"уровень используется только для подписей осей (§8).\n"
        f"- Промпты не содержат терминов теории игр — контроль контаминации (§8).\n"
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _variance_verdict(variance: pd.DataFrame) -> str:
    """Вердикт по разложению. Доли берём из таблицы, а не вычитаем из ста.

    Вычитание «100 − модель − харнесс» теряет член взаимодействия, а он как
    раз бывает самым крупным: отдача от обвязки зависит от того, какая модель
    в ней работает. Молча приписать это остатку — потерять результат.
    """

    dep = str(variance["dependent"].iloc[0]) if "dependent" in variance else "phi_a"
    what = "аллокативной эффективности" if dep == "efficiency" else "доли излишка"

    def share(name: str) -> float:
        try:
            return float(variance.loc[variance["source"] == name, "eta_sq_pct"].iloc[0])
        except (IndexError, KeyError, ValueError):
            return float("nan")

    model_share = share("Модель")
    harness_share = share("Харнесс")
    inter_share = share("Модель × Харнесс")
    residual = share("Остаток (сценарий, шум LLM)")

    if harness_share != harness_share:  # NaN
        return "\n_Вклад харнесса не выделен: в данных варьируется только один фактор._\n"
    if model_share != model_share:
        return (
            f"\n**Итог:** харнесс объясняет {harness_share:.1f}% дисперсии {what}. "
            "Модель в этом прогоне не варьировалась, поэтому сравнить вклады нельзя — "
            "нужен прогон Э1 с тремя весовыми классами.\n"
        )

    systematic = harness_share + model_share + (inter_share if inter_share == inter_share else 0.0)
    tail = (
        f"Остаток — {residual:.1f}% ({'розыгрыш сценария и собственный шум LLM'})."
        if residual == residual
        else ""
    )

    # Взаимодействие крупнее любого из главных эффектов — самостоятельный
    # результат: эффекты не складываются, отдача от обвязки зависит от модели.
    if inter_share == inter_share and inter_share > max(model_share, harness_share):
        return (
            f"\n**Итог:** главные эффекты сопоставимы и невелики — модель "
            f"{model_share:.2f}%, харнесс {harness_share:.2f}%, — но их "
            f"**взаимодействие ({inter_share:.2f}%) крупнее каждого из них**. "
            f"Это содержательный результат, а не шум: отдача от обвязки зависит "
            f"от того, какая модель в ней работает, то есть «модель» и «институт» "
            f"не складываются аддитивно. Ровно на этом стоит RQ3 о курсе обмена. "
            f"{tail} Систематически объяснено {systematic:.1f}%.\n"
        )
    if systematic < 5.0:
        return (
            f"\n**Итог:** ни модель ({model_share:.2f}%), ни харнесс "
            f"({harness_share:.2f}%) не объясняют сколько-нибудь заметной доли "
            f"дисперсии {what}. {tail} Сравнивать вклады здесь нельзя: оба тонут "
            "в шуме. Либо эффект существенно меньше межсценарной вариации, либо "
            "не хватает мощности — смотрите пересчёт требуемого $n$ в отчёте о "
            "прогоне.\n"
        )
    if harness_share >= model_share:
        return (
            f"\n**Итог:** харнесс объясняет {harness_share:.1f}% дисперсии против "
            f"{model_share:.1f}% у модели. Это и есть заголовок статьи: при равном "
            f"интеллекте выигрывает тот, у кого лучше обвязка. {tail}\n"
        )
    return (
        f"\n**Итог:** модель объясняет {model_share:.1f}% дисперсии против "
        f"{harness_share:.1f}% у харнесса. Содержательное расхождение с ожиданием — "
        f"его надо объяснять, а не прятать. {tail}\n"
    )


def _rent_verdict(rent: pd.DataFrame) -> str:
    significant = rent.loc[rent["significant"]]
    if significant.empty:
        return (
            "\n**Итог:** ни один доверительный интервал ренты не отделён от нуля. "
            "Либо эффекта нет, либо не хватает мощности — по §4.4 надо пересчитать "
            "требуемое $n$ по фактической дисперсии, прежде чем гнать основной прогон.\n"
        )
    best = significant.iloc[0]
    return (
        f"\n**Итог:** максимальная значимая рента — {100 * best['rho']:.1f} п.п. доли излишка "
        f"при конфигурации `{best['harness_a']}` против `{best['harness_b']}` "
        f"($|\\Delta|_1$ = {best['abs_delta']}). Значимых ячеек: {len(significant)} из {len(rent)}.\n"
    )


# ---------------------------------------------------------------------------
# LOG_DIGEST.md
# ---------------------------------------------------------------------------


def build_log_digest(*, path: Path, events: dict[str, Any], run_meta: dict[str, Any]) -> Path:
    lines = [
        "# Дайджест логов харнесса\n",
        "Полный машиночитаемый лог — `events.jsonl` в каталоге прогона: одно событие на строку, "
        "разбор через `grep`/`jq` без внешнего вендора обсервабилити. Промпты и ответы моделей "
        "в лог не пишутся — только их SHA-256 и длина.\n",
        "\n## События по типам\n",
    ]
    counts = events.get("counts", {})
    lines.append(
        _md_table(
            pd.DataFrame({"событие": list(counts), "штук": list(counts.values())}),
            max_rows=40,
        )
    )
    lines.append(f"\n**Суммарно токенов по событиям `llm.call`:** {events.get('tokens_total', 0):,}\n".replace(",", " "))

    slowest = events.get("slowest_calls_ms", [])
    if slowest:
        lines.append("\n## Самые долгие вызовы\n")
        lines.append(
            _md_table(
                pd.DataFrame(
                    [{"латентность, мс": round(ms, 1), "тег": tag} for ms, tag in slowest]
                ),
                floatfmt=".1f",
            )
        )

    errors = events.get("errors", [])
    lines.append("\n## Ошибки и ретраи провайдера\n")
    if errors:
        lines.append(
            _md_table(
                pd.DataFrame(
                    [
                        {
                            "тип": e.get("event_type"),
                            "попытка": e.get("attempt"),
                            "ошибка": str(e.get("error") or e.get("reason"))[:120],
                        }
                        for e in errors
                    ]
                )
            )
        )
    else:
        lines.append("_Ошибок провайдера не зафиксировано._\n")

    breakers = events.get("circuit_breakers", [])
    lines.append("\n## Предохранители\n")
    lines.append(
        "Жёсткие лимиты сессии (раунды, токены, время, залипание на повторяющемся действии) "
        "живут в коде харнесса, а не в промпте: модель не может уговорить себя выйти за них.\n"
    )
    if breakers:
        lines.append(
            _md_table(
                pd.DataFrame(
                    [
                        {
                            "сессия": b.get("session_id"),
                            "причина": b.get("reason"),
                            "детали": json.dumps(b.get("detail", {}), ensure_ascii=False),
                        }
                        for b in breakers
                    ]
                ),
                max_rows=50,
            )
        )
    else:
        lines.append("_Ни один предохранитель не сработал._\n")

    gate = events.get("gate_violations_by_component", {})
    lines.append("\n## Отклонения гейтов по компонентам\n")
    if gate:
        lines.append(
            _md_table(
                pd.DataFrame(
                    {
                        "компонент": [COMPONENT_LABELS_RU.get(k, k) for k in gate],
                        "отклонений": list(gate.values()),
                    }
                )
            )
        )
    else:
        lines.append("_Гейты не отклоняли действий (или компоненты H3/H4 были выключены)._\n")

    lines.append("\n## Здоровье провайдера\n")
    health = run_meta.get("llm_health", {})
    if health:
        lines.append("```json\n" + json.dumps(health, ensure_ascii=False, indent=2) + "\n```\n")
    else:
        lines.append("_Нет данных._\n")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------


_HTML_CSS = """
:root { color-scheme: light dark; --fg:#1a1a1a; --bg:#ffffff; --muted:#5c5c5c;
        --line:#e3e3e3; --accent:#22415f; --warn:#a83a52; }
@media (prefers-color-scheme: dark) {
  :root { --fg:#e8e8e8; --bg:#141618; --muted:#a0a0a0; --line:#2c2f33; --accent:#7fa8cc; }
}
* { box-sizing: border-box; }
body { margin:0; padding:2rem 1rem 4rem; background:var(--bg); color:var(--fg);
       font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
main { max-width: 62rem; margin: 0 auto; }
h1 { font-size:1.7rem; border-bottom:2px solid var(--accent); padding-bottom:.4rem; }
h2 { font-size:1.25rem; margin-top:2.2rem; color:var(--accent); }
h3 { font-size:1.05rem; margin-top:1.4rem; }
table { border-collapse:collapse; width:100%; font-size:13px; margin:.8rem 0; }
th,td { border:1px solid var(--line); padding:.35rem .55rem; text-align:right; }
th { background:color-mix(in srgb, var(--accent) 12%, transparent); text-align:left; }
td:first-child, th:first-child { text-align:left; }
.tablewrap { overflow-x:auto; }
figure { margin:1.2rem 0; }
img { max-width:100%; height:auto; border:1px solid var(--line); border-radius:4px; }
.banner { border-left:4px solid var(--warn); background:color-mix(in srgb, var(--warn) 10%, transparent);
          padding:.7rem 1rem; border-radius:0 4px 4px 0; margin:1rem 0; }
.kv { display:grid; grid-template-columns:max-content 1fr; gap:.2rem 1rem; font-size:14px; }
.kv dt { color:var(--muted); }
.kv dd { margin:0; font-variant-numeric:tabular-nums; }
code { background:color-mix(in srgb, var(--fg) 8%, transparent); padding:.1rem .3rem; border-radius:3px;
       font-size:13px; }
.verdict { border-left:4px solid var(--accent); padding:.6rem 1rem; margin:1rem 0;
           background:color-mix(in srgb, var(--accent) 8%, transparent); border-radius:0 4px 4px 0; }
footer { margin-top:3rem; padding-top:1rem; border-top:1px solid var(--line); color:var(--muted); font-size:13px; }
"""


def _img_tag(path: Path, caption: str) -> str:
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return (
        f'<figure><img src="data:image/png;base64,{encoded}" alt="{caption}">'
        f"<figcaption>{caption}</figcaption></figure>"
    )


def _html_table(df: pd.DataFrame | None, *, max_rows: int = 30) -> str:
    if df is None or len(df) == 0:
        return "<p><em>нет данных</em></p>"
    return (
        '<div class="tablewrap">'
        + df.head(max_rows).to_html(index=False, float_format=lambda x: f"{x:.3f}", border=0)
        + "</div>"
    )


def build_html_report(
    *,
    path: Path,
    manifest: dict[str, Any],
    run_meta: dict[str, Any],
    analytics: dict[str, Any],
    figures: dict[str, Path],
    is_mock: bool,
) -> Path:
    """Самодостаточная страница: таблицы + графики вшиты как data-URI."""

    failures = analytics.get("failures", {})
    h4 = analytics.get("h4", {})
    parts: list[str] = [
        "<!doctype html><html lang='ru'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>Отчёт прогона {manifest.get('run_id', '')}</title>",
        f"<style>{_HTML_CSS}</style></head><body><main>",
        "<h1>Асимметрия агентного харнесса: отчёт прогона</h1>",
    ]
    if is_mock:
        parts.append(
            "<div class='banner'><strong>Прогон на mock-провайдере.</strong> Ответы "
            "сгенерированы детерминированным скриптом. Эффект харнесса здесь возникает "
            "по построению; отчёт годится только для проверки инфраструктуры.</div>"
        )

    parts.append("<h2>Паспорт прогона</h2><dl class='kv'>")
    for label, value in (
        ("run_id", manifest.get("run_id", "—")),
        ("дата", manifest.get("created_at", "—")),
        ("провайдер", manifest.get("provider", "—")),
        ("вариант промпта", manifest.get("prompt_variant", "—")),
        ("пул сценариев (CRN)", manifest.get("scenario_pool_fingerprint", "—")),
        ("сессий", failures.get("sessions", 0)),
        ("технических сбоев", f"{failures.get('failures', 0)} ({failures.get('failure_rate_pct', 0)}%)"),
        ("время прогона, с", run_meta.get("elapsed_s", "—")),
    ):
        parts.append(f"<dt>{label}</dt><dd>{value}</dd>")
    parts.append("</dl>")

    sections: list[tuple[str, Any, str | None, str | None]] = [
        ("Разложение дисперсии (§6.2)", analytics.get("variance"), "variance_decomposition", "Сколько дисперсии доли излишка объясняет модель, а сколько — обвязка."),
        ("Эффекты компонентов (§6.1)", analytics.get("component_effects"), "component_effects", "β — свой компонент, γ — тот же компонент у противника."),
        ("Скрининг компонентов (Э2, H2)", analytics.get("screening"), "screening_effects", "Главные эффекты по плану Плакетта–Бёрмана."),
        ("Градиент асимметрии (Э3, H1)", analytics.get("rent"), "asymmetry_gradient", "Рента ρ(Δ) — прирост доли, объяснимый только разницей в обвязке."),
        ("Насыщение (Э1, H5)", analytics.get("saturation"), "saturation", "Отдача от наращивания симметричной обвязки."),
        ("Рыночный уровень (Э5, H4)", analytics.get("market"), "market_h4", "Совокупная эффективность и неравенство долей против разброса обвязок."),
        ("Курс обмена (Э4, H3)", analytics.get("exchange_rate"), None, "Уровень обвязки, уравнивающий стороны при разных классах моделей."),
        ("Здоровье прогона", None, "run_health", "Причины завершения сессий."),
    ]
    for title, table, fig_key, caption in sections:
        parts.append(f"<h2>{title}</h2>")
        if caption:
            parts.append(f"<p>{caption}</p>")
        if isinstance(table, pd.DataFrame):
            parts.append(_html_table(table))
        if fig_key and fig_key in figures:
            parts.append(_img_tag(figures[fig_key], title))

    if h4.get("ok"):
        parts.append(f"<div class='verdict'><strong>Разделение H4a/H4b:</strong> {h4.get('verdict')}</div>")

    parts.append(
        "<footer>Собрано пакетом <code>harness-asymmetry</code>. Полные логи — "
        "<code>events.jsonl</code>, транскрипты — <code>sessions/*.jsonl</code>, "
        "пиннинг — <code>run_manifest.json</code>.</footer></main></body></html>"
    )
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------


def render_reports(*, run_dir: Path, reports_dir: Path) -> ReportBundle:
    """Собирает все отчёты по каталогу прогона."""

    run_dir, reports_dir = Path(run_dir), Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = reports_dir / "tables"
    tables_dir.mkdir(exist_ok=True)

    sessions_csv = run_dir / "sessions.csv"
    if not sessions_csv.exists():
        raise FileNotFoundError(
            f"Не найден {sessions_csv} — сначала выполните прогон (`run`/`pilot`)."
        )
    sessions = met.load_sessions(sessions_csv)
    manifest = _read_json(run_dir / "run_manifest.json")
    run_meta = _read_json(run_dir / "run_meta.json")
    is_mock = str(manifest.get("provider", "")).lower() == "mock"

    analytics = compute_analytics(sessions)
    events = digest_events(run_dir / "events.jsonl")

    tables: dict[str, Path] = {}
    for name in (
        "cells",
        "rent",
        "screening",
        "saturation",
        "market",
        "variance",
        "component_effects",
        "distributive",
        "exchange_rate",
        "role_balance",
        "pilot_power",
        "budget_discipline",
    ):
        table = analytics.get(name)
        if isinstance(table, pd.DataFrame) and not table.empty:
            out = tables_dir / f"{name}.csv"
            table.to_csv(out, index=False)
            tables[name] = out
    spec_coefs = (analytics.get("spec") or {}).get("coefficients")
    if isinstance(spec_coefs, pd.DataFrame) and not spec_coefs.empty:
        out = tables_dir / "regression_coefficients.csv"
        spec_coefs.to_csv(out, index=False)
        tables["regression_coefficients"] = out

    figures = figs.build_all(
        figures_dir=reports_dir / "figures",
        sessions=met.analysis_frame(sessions),
        rent=analytics.get("rent", pd.DataFrame()),
        screening=analytics.get("screening", pd.DataFrame()),
        saturation=analytics.get("saturation", pd.DataFrame()),
        market=analytics.get("market", pd.DataFrame()),
        variance=analytics.get("variance", pd.DataFrame()),
        component_effects=analytics.get("component_effects", pd.DataFrame()),
    )

    run_report = build_run_report(
        path=reports_dir / "RUN_REPORT.md",
        manifest=manifest,
        run_meta=run_meta,
        sessions=sessions,
        analytics=analytics,
        events=events,
        figures=figures,
        is_mock=is_mock,
    )
    results_report = build_results_report(
        path=reports_dir / "RESULTS.md",
        manifest=manifest,
        sessions=sessions,
        analytics=analytics,
        figures=figures,
        is_mock=is_mock,
    )
    log_digest = build_log_digest(
        path=reports_dir / "LOG_DIGEST.md", events=events, run_meta=run_meta
    )
    html_report = build_html_report(
        path=reports_dir / "report.html",
        manifest=manifest,
        run_meta=run_meta,
        analytics=analytics,
        figures=figures,
        is_mock=is_mock,
    )

    return ReportBundle(
        reports_dir=reports_dir,
        run_report=run_report,
        results_report=results_report,
        log_digest=log_digest,
        html_report=html_report,
        tables=tables,
        figures=figures,
        analytics=analytics,
    )
