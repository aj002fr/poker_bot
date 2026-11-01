import pytest

from arena.domain.models import TournamentConfig
from arena.domain.rng import SeededRng
from arena.table.driver import legal_random, play_out
from arena.table.service import seeded_table
from strategy.eval.execution import evaluate_gates
from strategy.eval.outcomes import depth_bucket, horizon_bucket, opportunities


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
