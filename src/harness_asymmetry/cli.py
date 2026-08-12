"""CLI стенда.

    harness-asymmetry plan   --experiment E2            # смета: сессии, вызовы, деньги
    harness-asymmetry run    --experiment pilot --provider mock --output outputs/smoke
    harness-asymmetry report outputs/smoke reports/smoke
    harness-asymmetry doctor                            # проверка ключа и провайдера

Разделение `run` и `report` намеренное: отчёты пересобираются из
``sessions.csv`` сколько угодно раз, не трогая API. Повторный `run` с тем же
``--output`` подхватывает чекпоинты и догоняет недостающие сессии.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from harness_asymmetry.config import (
    ConfigError,
    DEFAULT_CONFIG_PATH,
    RunConfig,
    RunnerConfig,
    ScenarioConfig,
    load_llm_settings,
    load_run_config,
    model_registry,
)
from harness_asymmetry.runner.plans import (
    SessionSpec,
    plan_e1_symmetric,
    plan_e2_screening,
    plan_e3_gradient,
    plan_e4_exchange_rate,
    plan_e5_market,
    plan_pilot,
)
from harness_asymmetry.runner.runner import ExperimentRunner
from harness_asymmetry.schemas import InfoRegime


EXPERIMENTS = ("pilot", "E1", "E2", "E3", "E4", "E5")


def _apply_overrides(config: RunConfig, args: argparse.Namespace) -> RunConfig:
    """CLI-флаги перекрывают YAML. Что не задано — остаётся из конфига."""

    scenarios = config.scenarios
    runner = config.runner
    if args.rounds:
        scenarios = replace(scenarios, max_rounds=args.rounds)
    if args.repeats:
        runner = replace(runner, repeats=args.repeats)
    if args.workers:
        runner = replace(runner, max_workers=args.workers)
    regimes = tuple(args.info_regimes) if args.info_regimes else config.info_regimes
    return RunConfig(
        scenarios=scenarios,
        harness=config.harness,
        runner=runner,
        info_regimes=regimes,
        prompt_variant=args.prompt_variant or config.prompt_variant,
    )


def build_plan(
    experiment: str, *, config: RunConfig, models: dict[str, str], args: argparse.Namespace
) -> list[SessionSpec]:
    """Собирает план сессий выбранного эксперимента."""

    regimes = [InfoRegime(r) for r in config.info_regimes]
    repeats = config.runner.repeats
    # Одномодельные эксперименты (пилот, Э2, Э5) идут на «рабочей лошадке» из
    # HA_MODEL, а не на лёгком классе: §4.5 велит гнать основную массу дёшево,
    # но модель всё же должна надёжно держать формат действия. Разброс весовых
    # классов нужен только там, где он предмет измерения — в Э1 и Э4.
    default_model = getattr(args, "model", None) or os.getenv("HA_MODEL") or models["mid"]

    if experiment == "pilot":
        return plan_pilot(model=default_model, repeats=repeats, info_regime=regimes[0])
    if experiment == "E1":
        return plan_e1_symmetric(models=models, repeats=repeats, info_regimes=regimes)
    if experiment == "E2":
        return plan_e2_screening(model=default_model, repeats=repeats, info_regimes=regimes)
    if experiment == "E3":
        survivors = args.survivors or ["memory", "verifier", "market"]
        return plan_e3_gradient(
            survivors=survivors, models=models, repeats=repeats, info_regimes=regimes
        )
    if experiment == "E4":
        return plan_e4_exchange_rate(models=models, repeats=repeats, info_regimes=regimes)
    if experiment == "E5":
        return plan_e5_market(
            model=default_model,
            n_agents=args.agents,
            periods=args.periods,
            dispersions=tuple(args.dispersions),
            mean_level=args.mean_level,
            info_regime=regimes[0],
            seed=config.scenarios.seed,
        )
    raise ConfigError(f"Неизвестный эксперимент: {experiment}. Доступны: {EXPERIMENTS}.")


def _plan_summary(specs: Sequence[SessionSpec], config: RunConfig) -> dict[str, float]:
    """Смета §4.5: сессии → вызовы → порядок стоимости."""

    rounds = config.scenarios.max_rounds
    # Два агента × раунды, плюс по одному вызову на обязательство и план.
    extras = sum(
        int(s.harness_a.commitment) + int(s.harness_b.commitment)
        + int(s.harness_a.planner) + int(s.harness_b.planner)
        for s in specs
    )
    calls = len(specs) * 2 * rounds + extras
    return {
        "sessions": len(specs),
        "cells": len({(s.experiment, s.cell_id) for s in specs}),
        "max_llm_calls": calls,
        "est_tokens": calls * 1400,  # порядок величины: ~1,2k промпт + ~0,2k ответ
    }


# ---------------------------------------------------------------------------
# Команды
# ---------------------------------------------------------------------------


def cmd_plan(args: argparse.Namespace) -> int:
    config = _apply_overrides(load_run_config(args.config), args)
    models = model_registry()
    total = {"sessions": 0, "cells": 0, "max_llm_calls": 0, "est_tokens": 0}
    print(f"Конфиг: {args.config or DEFAULT_CONFIG_PATH}")
    print(f"Повторов на ячейку: {config.runner.repeats}; раундов: {config.scenarios.max_rounds}; "
          f"режимы информации: {', '.join(config.info_regimes)}\n")
    experiments = EXPERIMENTS if args.experiment == "all" else [args.experiment]
    for experiment in experiments:
        specs = build_plan(experiment, config=config, models=models, args=args)
        summary = _plan_summary(specs, config)
        for key in total:
            total[key] += summary[key]
        print(
            f"{experiment:>6}: ячеек {summary['cells']:>5}, сессий {summary['sessions']:>6}, "
            f"вызовов LLM ≤ {summary['max_llm_calls']:>8}, токенов ≈ {summary['est_tokens']:>10,}".replace(",", " ")
        )
    print(
        f"\nИТОГО: сессий {total['sessions']}, вызовов LLM ≤ {total['max_llm_calls']}, "
        f"токенов ≈ {total['est_tokens']:,}".replace(",", " ")
    )
    print(
        "\nОценка §4.5 дизайн-дока — 150–250 тыс. вызовов на все эксперименты, несколько сотен "
        "долларов на дешёвых моделях. Основную массу гоните на лёгкой модели; тяжёлые — только "
        "в Э1 и Э4, где нужен разброс весовых классов."
    )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )
    config = _apply_overrides(load_run_config(args.config), args)
    models = model_registry()

    llm_settings = None
    if args.provider == "openrouter":
        llm_settings = load_llm_settings(model=models.get("light"))
    elif args.provider == "mock":
        print(
            "⚠️  Провайдер mock: ответы генерирует детерминированный скрипт. Это смоук "
            "инфраструктуры, а не эксперимент — отчёты будут помечены баннером.\n"
        )

    experiments = EXPERIMENTS if args.experiment == "all" else [args.experiment]
    specs: list[SessionSpec] = []
    for experiment in experiments:
        specs.extend(build_plan(experiment, config=config, models=models, args=args))
    if args.limit:
        specs = specs[: args.limit]
    if not specs:
        raise SystemExit("План пуст — проверьте --experiment и конфиг.")

    summary = _plan_summary(specs, config)
    print(
        f"План: {summary['sessions']} сессий в {summary['cells']} ячейках, "
        f"вызовов LLM ≤ {summary['max_llm_calls']}."
    )

    output_dir = Path(args.output).expanduser().resolve()
    progress = _make_progress(total=len(specs), enabled=not args.no_progress)
    runner = ExperimentRunner(
        output_dir=output_dir,
        config=config,
        provider=args.provider,
        models=models,
        llm_settings=llm_settings,
        run_id=args.run_id,
        progress_cb=progress,
    )
    result = runner.run(specs)
    print(
        f"\nГотово: {len(result.records)} сессий "
        f"(сбоев {result.failed}, из чекпоинтов {result.resumed}) → {result.output_dir}"
    )

    if args.report:
        from harness_asymmetry.analysis.reports import render_reports

        reports_dir = Path(args.report).expanduser().resolve()
        bundle = render_reports(run_dir=output_dir, reports_dir=reports_dir)
        _print_report_paths(bundle)
    else:
        print(f"Отчёты: harness-asymmetry report {output_dir} reports/<имя>")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    from harness_asymmetry.analysis.reports import render_reports

    run_dir = Path(args.run_dir).expanduser().resolve()
    reports_dir = Path(args.output).expanduser().resolve()
    bundle = render_reports(run_dir=run_dir, reports_dir=reports_dir)
    _print_report_paths(bundle)
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """Проверяет окружение и живость провайдера до дорогого прогона."""

    print("Проверка окружения\n" + "-" * 40)
    try:
        config = load_run_config(args.config)
        print(f"✓ конфиг: {args.config or DEFAULT_CONFIG_PATH}")
        print(f"  раундов={config.scenarios.max_rounds}, повторов={config.runner.repeats}, "
              f"сценариев в пуле={config.scenarios.n_scenarios}")
    except ConfigError as exc:
        print(f"✗ конфиг: {exc}")
        return 1

    from harness_asymmetry.scenarios import build_scenario_pool

    pool = build_scenario_pool(config.scenarios)
    print(f"✓ пул сценариев: {len(pool)} розыгрышей, отпечаток {pool.fingerprint()}")

    print(f"  модели: {model_registry()}")
    try:
        settings = load_llm_settings()
    except ConfigError as exc:
        print(f"✗ провайдер: {exc}")
        print("  (для смоука инфраструктуры используйте --provider mock)")
        return 1
    print(f"✓ ключ OPENROUTER_API_KEY найден, base_url={settings.base_url}")

    if args.ping:
        from harness_asymmetry.llm_client import OpenRouterClient

        client = OpenRouterClient(settings)
        try:
            response = client.chat(
                system="Отвечай строго JSON-объектом.",
                user='Верни {"ok": true}',
                tag="doctor",
            )
            print(f"✓ ответ модели за {response.latency_ms:.0f} мс: {response.text[:80]!r}")
        except Exception as exc:  # noqa: BLE001
            print(f"✗ вызов не прошёл: {exc}")
            return 1
    return 0


def _print_report_paths(bundle) -> None:
    print("\nОтчёты собраны:")
    print(f"  операционный : {bundle.run_report}")
    print(f"  научный      : {bundle.results_report}")
    print(f"  дайджест логов: {bundle.log_digest}")
    print(f"  HTML         : {bundle.html_report}")
    print(f"  таблиц: {len(bundle.tables)}, графиков: {len(bundle.figures)}")


def _make_progress(*, total: int, enabled: bool):
    if not enabled:
        return None
    try:
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            TextColumn,
            TimeElapsedColumn,
        )
    except ImportError:  # pragma: no cover
        return None

    state: dict[str, object] = {}

    def callback(kind: str, payload: dict) -> None:
        if kind == "run_start":
            progress = Progress(
                TextColumn("[bold]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
            )
            progress.start()
            task = progress.add_task("сессии", total=payload.get("total", total))
            progress.update(task, advance=payload.get("resumed", 0))
            state["progress"], state["task"] = progress, task
        elif kind == "session_done" and "progress" in state:
            state["progress"].update(state["task"], advance=1)  # type: ignore[index]
        elif kind == "run_end" and "progress" in state:
            state["progress"].stop()  # type: ignore[union-attr]

    return callback


# ---------------------------------------------------------------------------
# Парсер
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="harness-asymmetry")
    subs = parser.add_subparsers(dest="command", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--log-level", default="info")
        sp.add_argument("--config", default=None, help="YAML-конфиг прогона.")
        sp.add_argument("--experiment", default="pilot", choices=(*EXPERIMENTS, "all"))
        sp.add_argument("--repeats", type=int, default=None, help="Повторов на ячейку (n ≥ 40 в основном прогоне).")
        sp.add_argument("--rounds", type=int, default=None, help="Максимум раундов торга T.")
        sp.add_argument("--workers", type=int, default=None, help="Параллельных ячеек.")
        sp.add_argument("--info-regimes", nargs="+", default=None, choices=("I0", "I1"))
        sp.add_argument("--prompt-variant", default=None, choices=("base", "terse", "formal"))
        sp.add_argument("--survivors", nargs="+", default=None, help="Компоненты для Э3.")
        sp.add_argument(
            "--model",
            default=None,
            help="Модель для одномодельных экспериментов (pilot/E2/E5). По умолчанию HA_MODEL.",
        )
        sp.add_argument("--agents", type=int, default=12, help="Э5: агентов в популяции.")
        sp.add_argument("--periods", type=int, default=4, help="Э5: торговых периодов.")
        sp.add_argument("--dispersions", nargs="+", type=float, default=(0.0, 0.5, 1.0))
        sp.add_argument("--mean-level", type=float, default=3.0, help="Э5: средний уровень обвязки.")

    plan = subs.add_parser("plan", help="Смета плана без единого вызова LLM.")
    add_common(plan)
    plan.set_defaults(func=cmd_plan)

    run = subs.add_parser("run", help="Прогон эксперимента с чекпоинтами и resume.")
    add_common(run)
    run.add_argument("--output", required=True, help="Каталог прогона.")
    run.add_argument("--provider", default="openrouter", choices=("openrouter", "mock"))
    run.add_argument("--run-id", default=None)
    run.add_argument("--limit", type=int, default=None, help="Обрезать план (для отладки).")
    run.add_argument("--report", default=None, help="Сразу собрать отчёты в этот каталог.")
    run.add_argument("--no-progress", action="store_true")
    run.set_defaults(func=cmd_run)

    report = subs.add_parser("report", help="Собрать отчёты из готового прогона.")
    report.add_argument("run_dir")
    report.add_argument("output")
    report.add_argument("--log-level", default="info")
    report.set_defaults(func=cmd_report)

    doctor = subs.add_parser("doctor", help="Проверить конфиг, ключ и провайдера.")
    doctor.add_argument("--config", default=None)
    doctor.add_argument("--log-level", default="info")
    doctor.add_argument("--ping", action="store_true", help="Сделать один реальный вызов.")
    doctor.set_defaults(func=cmd_doctor)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ConfigError as exc:
        print(f"Ошибка конфигурации: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
