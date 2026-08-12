"""Прогон плана: чекпоинты, resume, параллелизация, манифест.

Три инженерных требования §7.2, реализованных здесь:

* **Чекпоинты.** Прогон на 200 тыс. вызовов упадёт посередине. Каждая
  завершённая сессия немедленно дописывается в ``sessions/<ячейка>.jsonl`` с
  ``flush``; повторный запуск той же команды подхватывает готовое и гонит
  только недостающее.
* **Пиннинг всего.** ``run_manifest.json`` фиксирует версии моделей,
  хеши промптов, сиды, отпечаток пула сценариев, версии пакетов и полный
  конфиг. Без этого воспроизводимость потеряна, и рецензент это заметит.
* **Отдельный лог издержек.** Токены и латентность пишутся по каждому ходу,
  чтобы соотнести ренту с ценой компонента.

Параллелизация — на уровне **ячеек**, а не сессий. Внутри ячейки сессии идут
строго последовательно: компонент памяти (H1) накапливает историю пары от
повтора к повтору, и перестановка повторов сломала бы репутационный механизм.
"""

from __future__ import annotations

import json
import logging
import platform
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from harness_asymmetry.config import (
    HarnessConfig,
    LLMSettings,
    RunConfig,
    RunnerConfig,
)
from harness_asymmetry.harness.components.memory import MemoryStore
from harness_asymmetry.llm_client import LLMClient, build_client
from harness_asymmetry.observability import (
    EVENT_RUN_END,
    EVENT_RUN_START,
    EventLog,
    new_id,
)
from harness_asymmetry.prompts import prompt_hashes
from harness_asymmetry.protocol import SideSpec, run_session
from harness_asymmetry.runner.plans import SessionSpec
from harness_asymmetry.scenarios import ScenarioPool, build_scenario_pool
from harness_asymmetry.schemas import Role, SessionRecord


logger = logging.getLogger(__name__)


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)


# ---------------------------------------------------------------------------
# Манифест
# ---------------------------------------------------------------------------


def build_manifest(
    *,
    run_id: str,
    provider: str,
    config: RunConfig,
    models: dict[str, str],
    llm_settings: LLMSettings | None,
    pool: ScenarioPool,
    specs: Sequence[SessionSpec],
) -> dict[str, Any]:
    """Всё, что нужно, чтобы повторить прогон через год."""

    by_experiment: dict[str, int] = defaultdict(int)
    for spec in specs:
        by_experiment[spec.experiment] += 1
    return {
        "run_id": run_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "provider": provider,
        "models": dict(models),
        "llm_settings": llm_settings.redacted() if llm_settings else None,
        "prompt_variant": config.prompt_variant,
        "prompt_hashes": prompt_hashes(),
        "scenario_pool_fingerprint": pool.fingerprint(),
        "scenario_pool_size": len(pool),
        "config": config.as_dict(),
        "plan": {
            "total_sessions": len(specs),
            "by_experiment": dict(by_experiment),
            "cells": len({(s.experiment, s.cell_id) for s in specs}),
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": _package_versions(),
        },
    }


def _package_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str] = {}
    for name in ("openai", "pandas", "numpy", "scipy", "statsmodels", "matplotlib"):
        try:
            out[name] = version(name)
        except PackageNotFoundError:  # pragma: no cover
            out[name] = "absent"
    return out


# ---------------------------------------------------------------------------
# Раннер
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RunResult:
    run_id: str
    output_dir: Path
    records: list[SessionRecord]
    manifest: dict[str, Any]
    health: dict[str, Any] = field(default_factory=dict)
    resumed: int = 0
    failed: int = 0


class ExperimentRunner:
    """Исполняет план сессий с чекпоинтами и возобновлением."""

    def __init__(
        self,
        *,
        output_dir: Path,
        config: RunConfig,
        provider: str,
        models: dict[str, str],
        llm_settings: LLMSettings | None,
        run_id: str | None = None,
        progress_cb: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "sessions").mkdir(exist_ok=True)
        self.config = config
        self.provider = provider
        self.models = models
        self.llm_settings = llm_settings
        self.run_id = run_id or new_id("run")
        self.progress_cb = progress_cb

        self.event_log = EventLog(self.output_dir / "events.jsonl", run_id=self.run_id)
        self.pool = build_scenario_pool(config.scenarios)
        self._clients: dict[str, LLMClient] = {}
        self._client_lock = threading.Lock()
        self._write_lock = threading.Lock()

    # -- клиенты ------------------------------------------------------------

    def client_for(self, model: str) -> LLMClient:
        """Один клиент на модель: кеш промптов общий, что и нужно при CRN."""

        with self._client_lock:
            client = self._clients.get(model)
            if client is None:
                settings = (
                    replace(self.llm_settings, model=model)
                    if self.llm_settings is not None
                    else None
                )
                client = build_client(
                    provider=self.provider,
                    settings=settings,
                    event_log=self.event_log,
                    seed=self.config.scenarios.seed,
                )
                self._clients[model] = client
            return client

    # -- прогон -------------------------------------------------------------

    def run(self, specs: Sequence[SessionSpec]) -> RunResult:
        manifest = build_manifest(
            run_id=self.run_id,
            provider=self.provider,
            config=self.config,
            models=self.models,
            llm_settings=self.llm_settings,
            pool=self.pool,
            specs=specs,
        )
        (self.output_dir / "run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        done = self._load_checkpoints()
        pending = [s for s in specs if s.session_id not in done]
        resumed = len(specs) - len(pending)

        self.event_log.emit(
            EVENT_RUN_START,
            total=len(specs),
            pending=len(pending),
            resumed=resumed,
            provider=self.provider,
            experiments=sorted({s.experiment for s in specs}),
        )
        self._notify(
            "run_start",
            {"total": len(specs), "pending": len(pending), "resumed": resumed},
        )

        pending = _apply_pair_chunking(pending, self.config.runner.pair_chunk)
        cells: dict[tuple[str, str, str], list[SessionSpec]] = defaultdict(list)
        for spec in pending:
            cells[(spec.experiment, spec.cell_id, spec.pair_id)].append(spec)
        for bucket in cells.values():
            bucket.sort(key=lambda s: s.repeat_index)

        # Ячейку приходится гнать последовательно только там, где включена
        # память (H1): она копит историю пары от повтора к повтору, и
        # перестановка повторов сломала бы репутационный механизм. Если H1
        # выключена у обеих сторон, межсессионного состояния нет вовсе —
        # такие ячейки распараллеливаем посессионно. На планах вроде Э2, где
        # половина конфигураций без памяти, это разница между «ночь» и
        # «сутки»: узким местом перестаёт быть число ячеек.
        sequential = {k: v for k, v in cells.items() if _needs_memory_order(v)}
        parallel = [s for k, v in cells.items() if k not in sequential for s in v]

        records: list[SessionRecord] = list(done.values())
        failed = sum(1 for r in records if r.technical_failure)

        workers = max(1, int(self.config.runner.max_workers))
        started = time.perf_counter()
        if workers == 1:
            for key, bucket in cells.items():
                for record in self._run_cell(key, bucket):
                    records.append(record)
                    failed += int(record.technical_failure)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool_exec:
                futures: dict[Any, str] = {}
                for key, bucket in sequential.items():
                    futures[pool_exec.submit(self._run_cell_collect, key, bucket)] = str(key)
                for spec in parallel:
                    futures[pool_exec.submit(self._run_single, spec)] = spec.session_id
                for future in as_completed(futures):
                    label = futures[future]
                    try:
                        batch = future.result()
                    except Exception:  # noqa: BLE001
                        logger.exception("Задача %s упала целиком", label)
                        continue
                    records.extend(batch)
                    failed += sum(1 for r in batch if r.technical_failure)

        elapsed = time.perf_counter() - started
        health = {model: client.health() for model, client in self._clients.items()}
        self.event_log.emit(
            EVENT_RUN_END,
            total=len(records),
            failed=failed,
            elapsed_s=round(elapsed, 1),
            health=health,
        )
        self._notify("run_end", {"total": len(records), "failed": failed})

        self._write_tables(records)
        meta = {
            "run_id": self.run_id,
            "sessions_total": len(records),
            "sessions_failed": failed,
            "sessions_resumed": resumed,
            "elapsed_s": round(elapsed, 1),
            "event_counts": self.event_log.counts(),
            "llm_health": health,
        }
        (self.output_dir / "run_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.event_log.close()
        return RunResult(
            run_id=self.run_id,
            output_dir=self.output_dir,
            records=records,
            manifest=manifest,
            health=health,
            resumed=resumed,
            failed=failed,
        )

    # -- ячейка -------------------------------------------------------------

    def _run_cell_collect(
        self, key: tuple[str, str, str], bucket: list[SessionSpec]
    ) -> list[SessionRecord]:
        return list(self._run_cell(key, bucket))

    def _run_single(self, spec: SessionSpec) -> list[SessionRecord]:
        """Одна сессия ячейки без памяти — порядок повторов не важен."""

        return list(self._run_cell((spec.experiment, spec.cell_id, spec.pair_id), [spec]))

    def _run_cell(
        self, key: tuple[str, str, str], bucket: list[SessionSpec]
    ) -> Iterable[SessionRecord]:
        """Все сессии одной ячейки — строго последовательно (память копится)."""

        experiment, cell_id, _pair = key
        path = self.output_dir / "sessions" / f"{_safe_name(experiment)}__{_safe_name(cell_id)}.jsonl"
        stores: dict[str, MemoryStore] = {}
        consecutive_failures = 0

        for spec in bucket:
            for party in (spec.party_a, spec.party_b):
                stores.setdefault(party, MemoryStore(owner_id=party))
            try:
                record = self._run_one(spec, stores)
            except Exception as exc:  # noqa: BLE001
                # Фатальная ошибка сессии не должна ронять ячейку: фиксируем
                # её как технический сбой и идём дальше — доля сбоев потом
                # публикуется в отчёте о прогоне.
                logger.exception("Сессия %s упала: %s", spec.session_id, exc)
                record = _failure_record(spec, self.run_id, self.pool, reason=str(exc))

            self._append_session(path, record)
            consecutive_failures = consecutive_failures + 1 if record.technical_failure else 0
            self._notify(
                "session_done",
                {
                    "session_id": record.session_id,
                    "experiment": experiment,
                    "cell_id": cell_id,
                    "deal": record.deal,
                    "failed": record.technical_failure,
                },
            )
            if consecutive_failures >= self.config.runner.max_consecutive_failures:
                logger.error(
                    "Ячейка %s: %d подряд неудач — прекращаем её досрочно.",
                    cell_id,
                    consecutive_failures,
                )
                yield record
                return
            yield record

    def _run_one(
        self, spec: SessionSpec, stores: dict[str, MemoryStore]
    ) -> SessionRecord:
        scenario = self.pool.with_regime(spec.scenario_index, spec.info_regime)
        role_a = spec.role_a
        side_a = SideSpec(
            side="A",
            role=role_a,
            vector=spec.harness_a,
            llm=self.client_for(spec.model_a),
            party_id=spec.party_a,
            memory_store=stores[spec.party_a],
        )
        side_b = SideSpec(
            side="B",
            role=role_a.opposite(),
            vector=spec.harness_b,
            llm=self.client_for(spec.model_b),
            party_id=spec.party_b,
            memory_store=stores[spec.party_b],
        )
        return run_session(
            session_id=spec.session_id,
            run_id=self.run_id,
            experiment=spec.experiment,
            cell_id=spec.cell_id,
            repeat_index=spec.repeat_index,
            pair_id=spec.pair_id,
            scenario=scenario,
            side_a=side_a,
            side_b=side_b,
            harness_config=self.config.harness,
            runner_config=self.config.runner,
            prompt_variant=self.config.prompt_variant,
            first_mover=spec.first_mover,
            meta=spec.meta,
            event_log=self.event_log,
        )

    # -- ввод/вывод ---------------------------------------------------------

    def _append_session(self, path: Path, record: SessionRecord) -> None:
        payload = record.as_dict()
        with self._write_lock:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
                fh.flush()

    def _load_checkpoints(self) -> dict[str, SessionRecord]:
        """Читает готовые сессии для resume; битые строки пропускает."""

        done: dict[str, SessionRecord] = {}
        for path in sorted((self.output_dir / "sessions").glob("*.jsonl")):
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # прогон убили сигналом на середине строки
                    record = _record_from_dict(data)
                    if record is not None:
                        done[record.session_id] = record
        return done

    def _write_tables(self, records: Sequence[SessionRecord]) -> None:
        """Плоские таблицы: ``sessions.csv`` и ``turns.csv`` для аналитики."""

        import pandas as pd

        sessions = pd.DataFrame([r.flat() for r in records])
        if not sessions.empty:
            sessions = sessions.sort_values(["experiment", "cell_id", "repeat_index"])
        sessions.to_csv(self.output_dir / "sessions.csv", index=False)

        turn_rows: list[dict[str, Any]] = []
        for record in records:
            for turn in record.turns:
                row = turn.as_dict()
                row.pop("raw_text", None)  # сырьё живёт в sessions/*.jsonl
                row["session_id"] = record.session_id
                row["experiment"] = record.experiment
                row["cell_id"] = record.cell_id
                row["harness_a"] = record.harness_a
                row["harness_b"] = record.harness_b
                row["components_active"] = ",".join(row.get("components_active") or [])
                row["gate_violations"] = len(row.get("gate_violations") or [])
                turn_rows.append(row)
        pd.DataFrame(turn_rows).to_csv(self.output_dir / "turns.csv", index=False)

    def _notify(self, kind: str, payload: dict[str, Any]) -> None:
        if self.progress_cb is not None:
            try:
                self.progress_cb(kind, payload)
            except Exception:  # noqa: BLE001 - прогресс не должен ронять прогон
                logger.debug("progress_cb упал", exc_info=True)


def _apply_pair_chunking(specs: list[SessionSpec], chunk: int) -> list[SessionSpec]:
    """Разбивает повторы ячейки на независимые пары контрагентов.

    Компонент памяти привязан к ``party_id``, а тот выводится из ``pair_id``.
    Пока вся ячейка — одна пара, её 40 повторов обязаны идти строго по
    очереди, и самые дорогие конфигурации становятся узким местом всего
    прогона. Разбивая ячейку на пары по ``chunk`` встреч, мы сохраняем
    репутационный механизм (глубина истории ≥ ``memory_window``, дальше окно
    всё равно не читается) и возвращаем параллелизм.

    Методологически это не подмена: дизайн-док требует, чтобы память
    превращала разовую игру в повторяющуюся для одной стороны, а не чтобы
    конкретная пара встретилась ровно сорок раз.
    """

    if chunk <= 0:
        return specs
    out: list[SessionSpec] = []
    for spec in specs:
        pair_index = spec.repeat_index // chunk
        out.append(replace(spec, pair_id=f"{spec.pair_id}:p{pair_index:03d}"))
    return out


def _needs_memory_order(bucket: list[SessionSpec]) -> bool:
    """Нужен ли строгий порядок повторов внутри ячейки.

    Нужен ровно тогда, когда хотя бы у одной стороны включён компонент
    памяти: только он переносит состояние между сессиями одной пары.
    """

    return any(s.harness_a.memory or s.harness_b.memory for s in bucket)


def _failure_record(
    spec: SessionSpec, run_id: str, pool: ScenarioPool, *, reason: str
) -> SessionRecord:
    scenario = pool.with_regime(spec.scenario_index, spec.info_regime)
    return SessionRecord(
        session_id=spec.session_id,
        run_id=run_id,
        experiment=spec.experiment,
        cell_id=spec.cell_id,
        repeat_index=spec.repeat_index,
        pair_id=spec.pair_id,
        scenario_id=scenario.scenario_id,
        v=scenario.v,
        c=scenario.c,
        surplus=scenario.surplus,
        discount=scenario.discount,
        max_rounds=scenario.max_rounds,
        info_regime=scenario.info_regime.value,
        harness_a=spec.harness_a.code(),
        harness_b=spec.harness_b.code(),
        model_a=spec.model_a,
        model_b=spec.model_b,
        role_a=spec.role_a.value,
        first_mover_side=spec.first_mover,
        meta=dict(spec.meta),
        technical_failure=True,
        failure_reason=f"runner_exception:{reason}"[:500],
        stop_reason="runner_exception",
    )


def _record_from_dict(data: dict[str, Any]) -> SessionRecord | None:
    """Восстанавливает запись из чекпоинта (без транскрипта — он не нужен для resume)."""

    try:
        record = SessionRecord(
            session_id=data["session_id"],
            run_id=data.get("run_id", ""),
            experiment=data.get("experiment", ""),
            cell_id=data.get("cell_id", ""),
            repeat_index=int(data.get("repeat_index", 0)),
            pair_id=data.get("pair_id", ""),
            scenario_id=data.get("scenario_id", ""),
            v=float(data.get("v", 0.0)),
            c=float(data.get("c", 0.0)),
            surplus=float(data.get("surplus", 0.0)),
            discount=float(data.get("discount", 0.0)),
            max_rounds=int(data.get("max_rounds", 0)),
            info_regime=data.get("info_regime", ""),
            harness_a=data.get("harness_a", "000000"),
            harness_b=data.get("harness_b", "000000"),
            model_a=data.get("model_a", ""),
            model_b=data.get("model_b", ""),
            role_a=data.get("role_a", ""),
            first_mover_side=data.get("first_mover_side", "A"),
        )
    except (KeyError, TypeError, ValueError):
        return None

    for key in (
        "abs_delta",
        "deal",
        "price",
        "agreement_round",
        "phi_a",
        "phi_b",
        "phi_a_discounted",
        "efficiency",
        "rubinstein_price",
        "rubinstein_gap",
        "anchor_a",
        "anchor_b",
        "anchor_aggressiveness_a",
        "concession_rate_a",
        "concession_rate_b",
        "total_tokens",
        "prompt_tokens",
        "completion_tokens",
        "latency_ms",
        "llm_calls",
        "invalid_outputs",
        "gate_violations_a",
        "gate_violations_b",
        "commitment_a",
        "commitment_b",
        "budget_violation",
        "technical_failure",
        "failure_reason",
        "stop_reason",
        "stuck_turns",
    ):
        if key in data and data[key] is not None:
            setattr(record, key, data[key])
    record.delta_bits = tuple(data.get("delta_bits", ()) or ())
    record.meta = dict(data.get("meta") or {})
    return record
