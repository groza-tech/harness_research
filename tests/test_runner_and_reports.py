"""Раннер и отчёты end-to-end на mock-провайдере.

Проверяем то, что дороже всего чинить постфактум: чекпоинты и resume, состав
манифеста, полноту логов и то, что все четыре отчёта собираются и несут
баннер mock-прогона.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from harness_asymmetry.analysis.metrics import gini, load_sessions, required_n
from harness_asymmetry.analysis.reports import render_reports
from harness_asymmetry.observability import iter_events
from harness_asymmetry.runner.plans import plan_pilot
from harness_asymmetry.runner.runner import ExperimentRunner
from harness_asymmetry.schemas import InfoRegime


def _runner(tmp_path, run_config, run_id="run_test"):
    return ExperimentRunner(
        output_dir=tmp_path / "run",
        config=run_config,
        provider="mock",
        models={"light": "mock/scripted-negotiator"},
        llm_settings=None,
        run_id=run_id,
    )


@pytest.fixture()
def smoke_run(tmp_path, run_config):
    specs = plan_pilot(
        model="mock/scripted-negotiator", repeats=4, info_regime=InfoRegime.FULL
    )
    result = _runner(tmp_path, run_config).run(specs)
    return tmp_path, result


def test_run_produces_all_artifacts(smoke_run):
    tmp_path, result = smoke_run
    run_dir = tmp_path / "run"
    for name in ("run_manifest.json", "run_meta.json", "sessions.csv", "turns.csv", "events.jsonl"):
        assert (run_dir / name).exists(), f"нет артефакта {name}"
    assert list((run_dir / "sessions").glob("*.jsonl")), "нет чекпоинтов сессий"
    assert len(result.records) == 8


def test_manifest_pins_everything(smoke_run):
    tmp_path, _ = smoke_run
    manifest = json.loads((tmp_path / "run" / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["provider"] == "mock"
    assert manifest["scenario_pool_fingerprint"]
    assert manifest["prompt_hashes"], "хеши промптов обязаны быть в манифесте"
    assert manifest["config"]["scenarios"]["seed"]
    assert manifest["environment"]["python"]


def test_resume_skips_completed_sessions(tmp_path, run_config):
    specs = plan_pilot(model="mock/scripted-negotiator", repeats=4)
    first = _runner(tmp_path, run_config).run(specs)
    assert first.resumed == 0

    second = _runner(tmp_path, run_config, run_id="run_test2").run(specs)
    assert second.resumed == len(specs), "второй прогон обязан подхватить чекпоинты"
    assert len(second.records) == len(first.records)


def test_events_log_is_structured(smoke_run):
    tmp_path, _ = smoke_run
    events = list(iter_events(tmp_path / "run" / "events.jsonl"))
    assert events
    types = {e["event_type"] for e in events}
    assert {"run.start", "session.start", "turn.end", "session.end", "run.end"} <= types
    for event in events:
        assert "run_id" in event and "ts" in event
    # Сырьё промптов в лог не пишется — только хеши.
    llm_calls = [e for e in events if e["event_type"] == "llm.call"]
    assert llm_calls
    assert all("prompt_hash" in e and "user" not in e for e in llm_calls)


def test_sessions_table_has_component_dummies(smoke_run):
    tmp_path, _ = smoke_run
    df = load_sessions(tmp_path / "run" / "sessions.csv")
    for col in ("a_memory", "b_memory", "d_memory", "phi_a", "efficiency", "abs_delta"):
        assert col in df.columns
    assert df["harness_a"].nunique() == 2  # голая и полная ячейки пилота


def test_reports_are_built_and_marked_mock(smoke_run):
    tmp_path, _ = smoke_run
    bundle = render_reports(run_dir=tmp_path / "run", reports_dir=tmp_path / "reports")

    for path in (bundle.run_report, bundle.results_report, bundle.log_digest, bundle.html_report):
        assert path.exists() and path.stat().st_size > 500, f"отчёт пуст: {path}"

    run_text = bundle.run_report.read_text(encoding="utf-8")
    results_text = bundle.results_report.read_text(encoding="utf-8")
    assert "MOCK-ПРОВАЙДЕРЕ" in run_text and "MOCK-ПРОВАЙДЕРЕ" in results_text
    assert "run_id" in run_text
    assert "Разложение дисперсии" in results_text

    html = bundle.html_report.read_text(encoding="utf-8")
    assert html.startswith("<!doctype html>")
    assert "mock-провайдере" in html

    assert bundle.tables, "ни одна таблица не выгружена"
    assert (tmp_path / "reports" / "tables").exists()


def test_reports_fail_loudly_without_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="sessions.csv"):
        render_reports(run_dir=tmp_path / "nothing", reports_dir=tmp_path / "reports")


# ---------------------------------------------------------------------------
# Метрики
# ---------------------------------------------------------------------------


def test_gini_bounds():
    assert gini([1, 1, 1, 1]) == pytest.approx(0.0, abs=1e-9)
    assert gini([0, 0, 0, 4]) == pytest.approx(0.75, abs=0.01)
    assert 0 <= gini([0.2, 0.5, 0.9]) <= 1


def test_required_n_grows_with_variance():
    small = required_n(observed_sd=0.10, effect_pp=7.5)
    large = required_n(observed_sd=0.20, effect_pp=7.5)
    assert large > small > 0
    assert required_n(observed_sd=0.0, effect_pp=7.5) == 0


def test_analysis_frame_drops_only_failures():
    from harness_asymmetry.analysis.metrics import analysis_frame, failure_summary

    df = pd.DataFrame(
        {
            "technical_failure": [False, True, False],
            "deal": [True, False, False],
            "phi_a": [0.5, None, None],
            "invalid_outputs": [0, 3, 0],
        }
    )
    assert len(analysis_frame(df)) == 2
    summary = failure_summary(df)
    assert summary["failures"] == 1
    assert summary["failure_rate_pct"] == pytest.approx(33.33, abs=0.01)
