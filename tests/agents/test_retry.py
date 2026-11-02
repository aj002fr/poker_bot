"""Spec 9.2: infrastructure failures invalidate a bot-only table, which is rerun with
the same seed under a declared policy; every attempt is counted; agent crashes only
fall back."""

from __future__ import annotations

import json

import pytest

from arena.agents.baselines import BASELINES
from arena.benchmark.runner import (
    INFRASTRUCTURE_CAUSES,
    BenchmarkManifest,
    Entrant,
    RetryPolicy,
    run_benchmark,
    schedule,
)
from tests.agents.conftest import bot, needs_sandbox, small_config


def _baselines(*names: str) -> list[Entrant]:
    return [Entrant(n, BASELINES[n], True) for n in names]


FIELD = ("legal_random", "calling_station", "check_fold")


EXACT_BUDGET_NS = 50_000_000


def _run(entrants=None, *, blocks=1, seed="retry", isolation="in_process", **kw):
    entrants = entrants or _baselines(*FIELD)
    return run_benchmark(config=small_config(len(entrants)), entrants=entrants, blocks=blocks, base_seed=seed,
                         isolation=isolation, **kw)


def _seat_of(entrants, entrant: int, index: int = 0, seed: str = "retry", blocks: int = 1) -> int:
    a = schedule(small_config(len(entrants)), len(entrants), blocks, seed)[index]
    return next(s for s, e in a.seats.items() if e == entrant)


def test_infrastructure_failure_reruns_with_same_seed_and_counts_both_attempts(tmp_path):
    clean = _run()
    run = _run(_faults={(1, 1): "start_fail"}, out_dir=tmp_path / "run", save_histories=True)
    s, m = run.manifest.summary, run.manifest

    assert [r.table_id for r in run.results] == [r.table_id for r in clean.results]
    assert [r.points for r in run.results] == [r.points for r in clean.results]
    assert all(r.valid for r in run.results) and run.results[1].attempt == 2
    assert [(r.index, r.attempt) for r in run.attempts] == [(0, 1), (1, 1), (1, 2), (2, 1)]

    first = run.attempts[1]
    assert first.invalid_reason == "worker_start_failed" and not first.valid
    assert first.attempt_id == f"{first.table_id}-a1" and run.results[1].attempt_id == f"{first.table_id}-a2"
    assert s["tables_scheduled"] == 3 and s["tables_completed"] == 3
    assert s["attempts_total"] == 4 and s["reruns"] == 1
    assert [x["attempt_id"] for x in s["invalid_attempts"]] == [first.attempt_id]
    assert s["invalid_attempts"][0]["reason"] == "worker_start_failed"
    assert s["tables_invalid"] == [] and s["tables_failed"] == []

    assert m.retry["max_attempts"] == 3 and m.retry["same_seed"] is True
    assert m.retry["invalidating_causes"] == list(INFRASTRUCTURE_CAUSES)
    assert "same seed" in m.retry_policy and "3 attempts" in m.retry_policy
    saved = BenchmarkManifest.model_validate_json((tmp_path / "run" / "manifest.json").read_text())
    assert saved.retry == m.retry and saved.summary["invalid_attempts"] == s["invalid_attempts"]
    assert len(list((tmp_path / "run" / "tables").glob("*.json"))) == 3


def test_policy_limits_attempts_and_reports_tables_still_invalid():
    run = _run(_faults={(0, 1): "start_fail", (0, 2): "start_fail"}, retry_policy=RetryPolicy(max_attempts=2))
    s, m = run.manifest.summary, run.manifest
    assert [(r.index, r.attempt) for r in run.attempts] == [(0, 1), (0, 2), (1, 1), (2, 1)]
    assert s["attempts_total"] == 4 and s["tables_completed"] == 2
    assert s["tables_invalid"] == [{"table_id": run.results[0].table_id, "attempt_id": run.results[0].attempt_id,
                                    "reason": "worker_start_failed"}]
    failed = {f.table_id: f.error for f in m.results.failed_tables}
    assert failed[run.results[0].table_id].startswith("invalid after 2 attempts: worker_start_failed")
    assert run.results[0].table_id not in m.results.completed_table_ids


def test_max_attempts_one_records_but_never_reruns():
    run = _run(_faults={(2, 1): "start_fail"}, retry_policy=RetryPolicy(max_attempts=1))
    s = run.manifest.summary
    assert s["attempts_total"] == 3 and s["reruns"] == 0 and len(s["tables_invalid"]) == 1
    assert run.manifest.retry["max_attempts"] == 1
    with pytest.raises(ValueError):
        RetryPolicy(max_attempts=0)


def test_lost_job_process_in_serial_mode_is_rerun():
    run = _run(_faults={(0, 1): "job_exit"})
    assert run.attempts[0].invalid_reason == "job_process_lost" and run.results[0].valid
    assert run.manifest.summary["reruns"] == 1


def test_non_infrastructure_errors_fail_without_rerun(monkeypatch):
    import arena.table.service as service

    real = service.seeded_table
    calls = {"n": 0}

    def flaky(config, seed, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("engine bug")
        return real(config, seed, **kw)

    monkeypatch.setattr(service, "seeded_table", flaky)
    run = _run()
    s = run.manifest.summary
    assert s["attempts_total"] == 3 and s["reruns"] == 0 and s["invalid_attempts"] == []
    assert s["tables_failed"] == [{"table_id": run.results[0].table_id, "error": "RuntimeError: engine bug"}]
    assert run.results[0].invalid_reason is None


def test_parallel_lost_job_process_is_rerun_and_matches_serial():
    clean = _run(blocks=2)
    run = _run(blocks=2, jobs=2, _faults={(3, 1): "job_exit"})
    s = run.manifest.summary
    assert [r.points for r in run.results] == [r.points for r in clean.results]
    assert all(r.valid for r in run.results)
    lost = [r for r in run.attempts if r.invalid_reason == "job_process_lost"]
    assert any(r.index == 3 for r in lost)
    # A lost pool interrupts at most the tables in flight: one per job process.
    assert 1 <= len(lost) <= 2 and s["reruns"] == len(lost)
    assert s["attempts_total"] == 6 + len(lost)


def test_old_manifests_without_retry_field_still_load(tmp_path):
    run = _run(out_dir=tmp_path / "run")
    data = json.loads((tmp_path / "run" / "manifest.json").read_text())
    data.pop("retry")
    assert BenchmarkManifest.model_validate_json(json.dumps(data)).retry == {}
    assert run.manifest.summary["attempts_total"] == 3 and run.attempts == run.results


@needs_sandbox
def test_infrastructure_kill_of_explicit_state_bot_is_restored_not_invalidated():
    entrants = _baselines(*FIELD)
    # Exact points across two sandboxed runs: a 1 ms host-stall timeout (a forced fold
    # in one run only) would break the comparison, so both runs get a 50 ms budget.
    clean = _run(entrants, isolation="sandbox", budget_ns=EXACT_BUDGET_NS)
    seat = _seat_of(entrants, 0)
    run = _run(entrants, isolation="sandbox", budget_ns=EXACT_BUDGET_NS, _faults={(0, 1): f"kill:{seat}:2"})
    s = run.manifest.summary
    assert s["attempts_total"] == 3 and s["invalid_attempts"] == []
    assert [r.points for r in run.results] == [r.points for r in clean.results]
    assert s["infrastructure_restored"][0]["attempt_id"] == run.results[0].attempt_id
    assert s["infrastructure_restored"][0]["infrastructure"][0]["event"] == "infrastructure_kill"
    assert [r[1:] for r in run.results[0].restarts[0]] == [("infrastructure_kill", True)]


@needs_sandbox
def test_infrastructure_kill_of_instance_state_bot_invalidates_and_reruns(tmp_path):
    entrants = [Entrant("instance", bot("instance_bot"))] + _baselines("calling_station", "check_fold")
    seat = _seat_of(entrants, 0)
    run = _run(entrants, isolation="sandbox", _faults={(0, 1): f"kill:{seat}:2"},
               out_dir=tmp_path / "run", save_histories=True)
    s = run.manifest.summary
    first = run.attempts[0]
    assert first.invalid_reason == "instance_state_lost_to_infrastructure" and first.completed
    assert run.results[0].valid and run.results[0].attempt == 2 and run.results[0].restarts[0] == []
    # The invalid attempt's results stay in the report, whatever they were.
    reported = s["invalid_attempts"][0]
    assert reported["completed"] and reported["points_by_entrant"] == {str(k): v for k, v in first.points.items()}
    assert s["entrants"][0]["mean_points_including_invalid_attempts"] is not None
    assert s["entrants"][0]["decisions"] == sum(r.decisions[0] for r in run.attempts)
    assert len(list((tmp_path / "run" / "tables").glob("*.json"))) == 3
    assert len(list((tmp_path / "run" / "tables" / "invalid" / "a1").rglob("*.json"))) == 1


@needs_sandbox
def test_worker_that_cannot_be_restarted_invalidates_and_the_pool_recovers():
    entrants = _baselines(*FIELD)
    seat = _seat_of(entrants, 1)
    run = _run(entrants, isolation="sandbox", _faults={(0, 1): f"spawn_fail:{seat}:2"})
    first = run.attempts[0]
    assert first.invalid_reason == "worker_start_failed"
    assert first.fallbacks[1].get("worker_unavailable", 0) > 0
    assert all(r.valid for r in run.results) and run.manifest.summary["reruns"] == 1


@needs_sandbox
def test_agent_crashes_fall_back_and_do_not_invalidate():
    entrants = [Entrant("crash", bot("crash_bot"))] + _baselines("calling_station", "check_fold")
    run = _run(entrants, isolation="sandbox")
    s = run.manifest.summary
    assert s["attempts_total"] == 3 and s["invalid_attempts"] == [] and s["tables_completed"] == 3
    crash = s["entrants"][0]
    assert crash["restarts"] > 0
    assert crash["fallbacks"].get("worker_crash", 0) + crash["fallbacks"].get("timeout", 0) > 0
