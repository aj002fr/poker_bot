"""Benchmark runner: seat balancing, paired seeds, intervals, manifest and CLI."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from arena.benchmark.runner import (
    BenchmarkManifest,
    Entrant,
    bootstrap_interval,
    run_benchmark,
    schedule,
)
from arena.agents.baselines import BASELINES
from arena.cli import main
from tests.agents.conftest import needs_sandbox, small_config

REPO = Path(__file__).resolve().parents[2]
TEMPLATE = REPO / "examples" / "template_bot.py"


def _entrants(n: int) -> list[Entrant]:
    names = ["calling_station", "legal_random", "check_fold", "nlhe_heuristic"]
    return [Entrant("template_bot", TEMPLATE)] + [Entrant(names[k % 4], BASELINES[names[k % 4]], True)
                                                   for k in range(n - 1)]


def test_schedule_rotates_every_entrant_through_every_seat_per_block():
    config = small_config(5)
    plan = schedule(config, 5, blocks=3, base_seed="s")
    assert len(plan) == 15
    for b in range(3):
        block = [a for a in plan if a.block == b]
        assert len({a.seed for a in block}) == 1
        for entrant in range(5):
            assert Counter(next(s for s, e in a.seats.items() if e == entrant) for a in block) == \
                Counter(config.resolved_seats)
        assert len({a.table_id for a in block}) == 5
    assert len({a.seed for a in plan}) == 3
    assert all(a.seed not in a.table_id for a in plan)


def test_bootstrap_interval_is_deterministic_and_brackets_the_mean():
    values = [10.0, 20.0, 30.0, 40.0, 50.0, 25.0]
    lo, hi = bootstrap_interval(values, "x")
    assert (lo, hi) == bootstrap_interval(values, "x")
    assert lo <= sum(values) / len(values) <= hi
    assert bootstrap_interval([1.0], "x") is None


def test_in_process_benchmark_manifest(tmp_path):
    config = small_config(3, hand_cap=10)
    run = run_benchmark(config=config, entrants=_entrants(3), blocks=2, base_seed="m", isolation="in_process",
                        out_dir=tmp_path / "run")
    m = run.manifest
    assert BenchmarkManifest.model_validate_json((tmp_path / "run" / "manifest.json").read_text()) == m
    assert m.competitive is False and "DIAGNOSTIC" in m.report_note
    assert m.scheduled_tables == 6 and m.seed_blocks == 2 and m.rotations_per_block == 3
    assert len(m.results.completed_table_ids) == 6
    s = m.summary
    assert abs(sum(e["mean_points"] for e in s["entrants"]) - 100) < 1e-9
    seeds = json.loads((tmp_path / "run" / "private" / "seeds.json").read_text())["seeds"]
    assert len(seeds) == 2 and not any(seed in (tmp_path / "run" / "manifest.json").read_text() for seed in seeds)
    again = run_benchmark(config=config, entrants=_entrants(3), blocks=2, base_seed="m", isolation="in_process")
    assert [r.points for r in again.results] == [r.points for r in run.results]


@needs_sandbox
def test_sandboxed_benchmark_reports_timing_and_fallbacks(tmp_path):
    config = small_config(3, hand_cap=8)
    run = run_benchmark(config=config, entrants=_entrants(3), blocks=1, base_seed="sb", out_dir=tmp_path / "run",
                        save_histories=True)
    m = run.manifest
    assert m.competitive is True and m.isolation == "sandbox" and m.budget_ns == 1_000_000
    assert len(m.results.completed_table_ids) == 3
    e = m.summary["entrants"][0]
    t = e["timing"]
    assert t["samples"] == e["decisions"] - e["fallbacks"].get("worker_unavailable", 0)
    assert 0 < t["end_to_end_s"]["p50"] <= t["end_to_end_s"]["p99"] <= t["end_to_end_s"]["max"]
    assert t["worker_wall_s"]["p50"] < t["end_to_end_s"]["p50"]
    assert m.sandbox["gvisor"] == "not used"
    histories = sorted((tmp_path / "run" / "tables").glob("*.json"))
    assert len(histories) == 3
    from arena.history.canonical import load_history
    from arena.history.replay import replay

    assert all(replay(load_history(p)).complete for p in histories)


def test_cli_match_in_process(capsys):
    code = main(["match", "--bot", str(TEMPLATE), "--players", "3", "--tables", "3", "--stack", "300",
                 "--in-process", "--jobs", "1", "--quiet"])
    out = capsys.readouterr().out
    assert code == 0
    assert "template_bot" in out and "p99" in out and "DIAGNOSTIC" in out and "fallbacks" in out


@needs_sandbox
def test_cli_match_sandboxed_and_rejects_unsupported_ruleset(capsys):
    code = main(["match", "--bot", str(TEMPLATE), "--players", "3", "--tables", "3", "--stack", "200",
                 "--jobs", "1", "--quiet"])
    out = capsys.readouterr().out
    assert code == 0 and "isolation sandbox" in out and "end-to-end round trip" in out
    assert "baselines combined" in out and "1 ms timeout fallbacks" in out
    bad = REPO / "tests" / "agents" / "bots" / "plo_only_bot.py"
    assert main(["match", "--bot", str(bad), "--players", "3", "--tables", "3", "--quiet"]) == 5
    assert "rejected before the table started" in capsys.readouterr().err
