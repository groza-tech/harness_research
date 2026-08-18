"""Смета прогона по ФАКТИЧЕСКИМ токенам, а не по верхней границе плана.

Оценка внутри ``cli plan`` считает максимум: все раунды до упора, каждый
раунд — полный контекст. Реальность вчетверо дешевле, потому что сделки
случаются на первых раундах. Считать бюджет по верхней границе — значит
отказаться от экспериментов, которые на самом деле по карману.

Здесь модель расхода строится по уже прошедшим сессиям: среднее число
токенов как функция модели и суммарного уровня обвязки обеих сторон
(``harness_level_a + harness_level_b``) — именно он определяет длину
контекста. Цены берутся живьём у провайдера.

    python scripts/estimate_cost.py --measured outputs/night_20260812 outputs/night_20260816 \
        --config configs/main.yaml --survivors-from outputs/night_20260812/E2
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from harness_asymmetry.cli import build_plan  # noqa: E402
from harness_asymmetry.config import load_run_config, model_registry  # noqa: E402
from harness_asymmetry.pricing import fetch_prices  # noqa: E402

EXPERIMENTS = ("pilot", "E1", "E2", "E3", "E4", "E5")


def measured_token_model(dirs: list[str]) -> dict[str, dict[int, tuple[float, float]]]:
    """``{модель: {сумма уровней: (вход, выход)}}`` по прошедшим сессиям."""

    agg: dict[tuple[str, int], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    for d in dirs:
        for path in pathlib.Path(d).rglob("sessions/*.jsonl"):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    s = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if s.get("technical_failure"):
                    continue
                key = (s.get("model_a"), int(s.get("harness_level_a", 0)) + int(s.get("harness_level_b", 0)))
                cell = agg[key]
                cell[0] += 1
                cell[1] += s.get("prompt_tokens", 0) or 0
                cell[2] += s.get("completion_tokens", 0) or 0
    out: dict[str, dict[int, tuple[float, float]]] = defaultdict(dict)
    for (model, level), (n, tin, tout) in agg.items():
        if n:
            out[model][level] = (tin / n, tout / n)
    return out


def lookup(model_table: dict[int, tuple[float, float]], level: int) -> tuple[float, float]:
    """Ближайший замеренный уровень; между замерами — линейная интерполяция."""

    if not model_table:
        return (12_000.0, 1_200.0)  # грубый общий ориентир, если модель новая
    if level in model_table:
        return model_table[level]
    below = [k for k in model_table if k < level]
    above = [k for k in model_table if k > level]
    if below and above:
        lo, hi = max(below), min(above)
        w = (level - lo) / (hi - lo)
        return tuple(model_table[lo][i] + w * (model_table[hi][i] - model_table[lo][i]) for i in range(2))
    nearest = min(model_table, key=lambda k: abs(k - level))
    return model_table[nearest]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/main.yaml")
    ap.add_argument("--measured", nargs="+", default=["outputs"])
    ap.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS))
    ap.add_argument("--survivors-from")
    ap.add_argument("--survivors", nargs="+")
    ap.add_argument("--model-classes", nargs="+")
    ap.add_argument("--agents", type=int, default=24)
    ap.add_argument("--periods", type=int, default=8)
    ap.add_argument("--dispersions", nargs="+", type=float, default=(0.0, 0.5, 1.0))
    ap.add_argument("--mean-level", type=float, default=3.0)
    ap.add_argument("--model")
    ap.add_argument("--overhead", type=float, default=1.15,
                    help="Запас на повторы после невалидного вывода и ретраи, по умолчанию 15%%.")
    args = ap.parse_args()

    tokens = measured_token_model(args.measured)
    if not tokens:
        print("Нет замеров: укажите --measured на каталог с прошедшими сессиями.", file=sys.stderr)
        return 1

    config = load_run_config(args.config)
    registry = model_registry()
    models = {k: registry[k] for k in args.model_classes} if args.model_classes else registry
    prices, source = fetch_prices(set(registry.values()))

    print(f"Замеры: {sum(len(v) for v in tokens.values())} точек по {len(tokens)} моделям")
    print(f"Прайс: {source}; запас на ретраи ×{args.overhead}\n")
    print(f"{'этап':>6} {'сессий':>8} {'токенов вход':>14} {'выход':>10} {'$':>9}")

    grand_sessions = 0
    grand_cost = 0.0
    per_model: dict[str, float] = defaultdict(float)
    for experiment in args.experiments:
        try:
            specs = build_plan(experiment, config=config, models=models, args=args)
        except Exception as exc:  # noqa: BLE001 — этап без выживших просто пропускаем
            print(f"{experiment:>6} — пропущен: {exc}")
            continue
        tin = tout = 0.0
        cost = 0.0
        for spec in specs:
            level = spec.harness_a.level + spec.harness_b.level
            model = spec.model_a
            ti, to = lookup(tokens.get(model, {}), level)
            ti *= args.overhead
            to *= args.overhead
            pin, pout = prices.get(model, (0.14, 0.28))
            c = ti / 1e6 * pin + to / 1e6 * pout
            tin += ti
            tout += to
            cost += c
            per_model[model] += c
        print(f"{experiment:>6} {len(specs):>8} {tin:>14,.0f} {tout:>10,.0f} {cost:>9.2f}".replace(",", " "))
        grand_sessions += len(specs)
        grand_cost += cost

    print(f"\n{'ИТОГО':>6} {grand_sessions:>8} сессий {'':>14} {'':>10} ${grand_cost:.2f}")
    print("\nПо моделям:")
    for model, cost in sorted(per_model.items(), key=lambda kv: -kv[1]):
        pin, pout = prices.get(model, (0.0, 0.0))
        share = 100 * cost / grand_cost if grand_cost else 0
        print(f"  {model:38} ${cost:>8.2f}  {share:>5.1f}%   (${pin}/${pout} за 1M)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
