from types import SimpleNamespace

import pytest

from arena.domain.models import TournamentConfig
from arena.domain.rng import SeededRng
from arena.table.driver import legal_random, play_out
from arena.table.service import seeded_table
from strategy.eval.execution import evaluate_gates, merge_seat_timeouts, seat_timeouts, timeout_integrity
from strategy.eval.outcomes import depth_bucket, horizon_bucket, opportunities
from strategy.eval.parallel import capacity, cgroup_cpu_limit, oversubscription_warning, physical_cores, safe_jobs


@pytest.mark.parametrize("ruleset", ["practice_nlhe", "uncertainty_plo4"])
def test_opportunities_count_every_decision(ruleset):
    config = TournamentConfig(ruleset_id=ruleset, initial_player_count=5, starting_stack=600, hand_cap=40)
    table = seeded_table(config, 11)
    play_out(table, legal_random, SeededRng(11, "players"))
    events = [e.model_dump(mode="json") for e in table.log]
    assert sum(1 for e in events if e["type"] == "DecisionRequested") > 20
    for seat in table.seat_ring:
        decisions = sum(1 for e in events if e["type"] == "DecisionRequested" and e["data"]["seat"] == seat)
        opp = opportunities(events, seat, config.hand_cap)
        for slice_name in ("blind_level", "hands_to_cap", "players_in_hand"):
            assert sum(v for k, v in opp.items() if k.startswith(slice_name + "=") and k.endswith("|*")) == decisions
        street = sum(v for k, v in opp.items() if k.startswith("street=") and k.endswith("|*"))
        assert street == decisions
        for key, total in opp.items():
            if key.endswith("|*"):
                head = key[:-2]
                assert sum(v for k, v in opp.items() if k.startswith(head + "|") and k != key) == total


def test_buckets():
    assert depth_bucket(7.9) == "<8bb" and depth_bucket(8) == "8-15bb" and depth_bucket(500) == "50bb+"
    assert horizon_bucket(0) == "0-5" and horizon_bucket(6) == "6-20" and horizon_bucket(150) == "101+"


def test_execution_gates():
    execution = {"illegal_actions": 1, "errors": 0, "timeouts": 3, "fallback_rate": 0.01,
                 "latency_end_to_end_s": {"p99": 0.2}, "artifact_bytes": 10, "tables_failed": 0,
                 "tables_invalid": 0, "competitive": False}
    checks = {c["gate"]: c["passed"] for c in evaluate_gates(execution, {
        "max_illegal_actions": 0, "max_errors": 0, "max_timeouts": None, "max_fallback_rate": 0.05,
        "max_p99_latency_s": 0.1, "max_artifact_bytes": 100, "max_failed_tables": 0, "require_competitive": True})}
    assert checks == {"illegal_actions": False, "errors": True, "fallback_rate": True, "p99_latency_s": False,
                      "artifact_bytes": True, "failed_or_invalid_tables": True, "competitive_timing": False}


def _cap(cores, limit=None):
    usable = cores if limit is None else min(cores, max(1, int(limit)))
    return {"cpus": list(range(cores)), "physical_cores": cores, "cgroup_cpu_limit": limit, "usable_cores": usable}


def test_safe_jobs_leave_each_sandboxed_job_four_cores():
    assert [safe_jobs("sandbox", _cap(n)) for n in (1, 4, 8, 9, 16, 32)] == [1, 1, 1, 2, 3, 7]
    assert [safe_jobs("in_process", _cap(n)) for n in (1, 2, 8)] == [1, 1, 7]
    assert safe_jobs("sandbox", _cap(32, limit=8.5)) == 1
    assert oversubscription_warning("sandbox", 1, _cap(8)) is None
    assert oversubscription_warning("in_process", 8, _cap(8)) is None
    assert "exceeds the safe 1" in oversubscription_warning("sandbox", 4, _cap(8))
    assert capacity()["usable_cores"] >= 1 and safe_jobs("sandbox") >= 1


def test_capacity_reads_cgroup_quota_and_core_topology(tmp_path):
    assert cgroup_cpu_limit(tmp_path) is None
    (tmp_path / "cpu.max").write_text("max 100000\n")
    assert cgroup_cpu_limit(tmp_path) is None
    (tmp_path / "cpu.max").write_text("250000 100000\n")
    assert cgroup_cpu_limit(tmp_path) == 2.5
    sys_cpu = tmp_path / "sys"
    for cpu, core in enumerate([0, 1, 0, 1]):
        topo = sys_cpu / f"cpu{cpu}" / "topology"
        topo.mkdir(parents=True)
        (topo / "physical_package_id").write_text("0\n")
        (topo / "core_id").write_text(f"{core}\n")
    assert physical_cores([0, 1, 2, 3], sys_cpu) == [[0, 2], [1, 3]]
    cap = capacity([0, 1, 2, 3], sys_cpu=sys_cpu, cgroup_root=tmp_path)
    assert cap["physical_cores"] == 2 and cap["usable_cores"] == 2
    assert physical_cores([5], tmp_path / "missing") == [[5]]


def _attempt(decisions, timeouts):
    return SimpleNamespace(decisions=decisions, fallbacks={e: {"timeout": n} for e, n in timeouts.items()})


def test_timeout_integrity_pools_every_seat_and_flags_single_entrants():
    labels = {0: "candidate", 1: "opponent 1"}
    gates = {"max_timeout_rate": 0.005, "max_entrant_timeout_rate": 0.02}
    clean = merge_seat_timeouts([seat_timeouts([_attempt({0: 1000, 1: 1000}, {1: 2})], labels)])
    assert clean["decisions"] == 2000 and clean["timeouts"] == 2
    assert clean["by_entrant"]["opponent 1"]["timeout_rate"] == 0.002
    assert timeout_integrity({"A": clean, "B": clean}, gates)["passed"]

    pooled = merge_seat_timeouts([seat_timeouts([_attempt({0: 1000, 1: 1000}, {0: 6, 1: 6})], labels)])
    check = timeout_integrity({"A": clean, "B": pooled}, gates)
    assert not check["passed"] and check["failures"][0].startswith("arm B: 12/2000")

    # one entrant above its rate limit, pooled rate still under the pooled limit
    lone = merge_seat_timeouts([seat_timeouts([_attempt({0: 10_000, 1: 200}, {1: 5})], labels)])
    assert lone["timeout_rate"] < 0.005
    check = timeout_integrity({"A": lone}, gates)
    assert check["failures"] == ["arm A: entrant opponent 1 timed out on 5/200 decisions (2.50% > 2.00%)"]
    few = merge_seat_timeouts([seat_timeouts([_attempt({0: 10_000, 1: 100}, {1: 4})], labels)])
    assert timeout_integrity({"A": few}, gates)["passed"], "under the minimum count, one entrant is noise"
    assert timeout_integrity({"A": pooled}, {"max_timeout_rate": None, "max_entrant_timeout_rate": None})["passed"]
