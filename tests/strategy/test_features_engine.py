import ast
import json
import math
from pathlib import Path

import pytest

from strategy.features import bench
from strategy.features.engine import CARD_FIELDS, RANGE_FIELDS, FeatureEngine
from strategy.features.state import NUMERIC_FIELDS
from tests.strategy.feature_support import RULESETS, recorded

STRATEGY = Path(__file__).resolve().parents[2] / "strategy"
# The decision path works from public observations only: the bot SDK and observation
# contracts, plus the public-event table-reading stats it reuses.
ALLOWED_ARENA = {"arena.agents.sdk", "arena.domain.observations", "arena.table.player_stats"}
OFFLINE = {"bench.py"}
"""Harness modules that run tables offline and are never imported by the decision path."""


@pytest.mark.parametrize("ruleset_id", RULESETS)
def test_engine_on_recorded_play(ruleset_id: str) -> None:
    width = len(NUMERIC_FIELDS) + len(CARD_FIELDS) + len(RANGE_FIELDS)
    decisions = 0
    for seat, seats, observations in recorded(ruleset_id):
        eng = FeatureEngine(ruleset_id, seat, seats)
        for obs in observations:
            f = eng.update(obs)
            v = f.vector()
            assert len(v) == width and all(math.isfinite(x) for x in v)
            assert 0.0 <= f.cards.equity <= 1.0 and 0 <= f.cards.bucket <= 9
            assert 0.0 <= f.ranges.own_rank <= 1.0 and 0.0 <= f.ranges.fold_all <= 1.0
            assert {o.seat for o in f.ranges.opponents} <= {p.seat for p in obs.players if p.seat != seat}
            for w in eng.ranges.weights.values():
                assert sum(w) == pytest.approx(1.0, abs=1e-6)
            decisions += 1
        state = eng.to_state()
        assert len(json.dumps(state, separators=(",", ":"))) <= 64 * 1024
        rates = eng.stats_rates(seats[0])
        assert rates["hands"] >= 1
    assert decisions > 50


def test_update_is_idempotent_for_seen_events() -> None:
    seat, seats, observations = recorded("practice_nlhe")[0]
    eng = FeatureEngine("practice_nlhe", seat, seats)
    for obs in observations[:20]:
        eng.update(obs)
    before = json.dumps(eng.to_state(), sort_keys=True)
    eng.update(observations[19])
    after = eng.to_state()
    after_posterior = json.dumps(after["posterior"], sort_keys=True)
    assert after_posterior == json.dumps(json.loads(before)["posterior"], sort_keys=True)


@pytest.mark.parametrize("ruleset_id", RULESETS)
def test_state_round_trip_mid_hand_matches_continuous_engine(ruleset_id: str) -> None:
    seat, seats, observations = max(recorded(ruleset_id), key=lambda r: len(r[2]))
    # split after a decision whose hand continues with another hero decision
    split = next(i for i in range(len(observations) // 2, len(observations) - 1)
                 if observations[i].hand_id == observations[i + 1].hand_id)
    live = FeatureEngine(ruleset_id, seat, seats)
    for obs in observations[:split + 1]:
        live.update(obs)
    restored = FeatureEngine.from_state(json.loads(json.dumps(live.to_state())), seats)
    for obs in observations[split + 1:split + 30]:
        a, b = live.update(obs), restored.update(obs)
        assert b.vector() == pytest.approx(a.vector(), abs=2e-3)


def test_state_stays_within_budget() -> None:
    seat, seats, observations = recorded("practice_nlhe")[0]
    eng = FeatureEngine("practice_nlhe", seat, seats)
    for obs in observations:
        eng.update(obs)
    small = eng.to_state(max_bytes=3000)
    assert len(json.dumps(small, separators=(",", ":"))) <= 3000 or not eng.posterior.cell


def _imports(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_decision_path_imports_only_public_modules() -> None:
    files = [p for d in ("features", "beliefs") for p in sorted((STRATEGY / d).glob("*.py")) if p.name not in OFFLINE]
    assert len(files) >= 8
    for path in files:
        for name in _imports(path):
            assert name.split(".")[0] != "pokerkit", (path.name, name)
            if name.split(".")[0] == "arena":
                assert name in ALLOWED_ARENA, (path.name, name)


def test_microbenchmark_smoke() -> None:
    result = bench.run("practice_nlhe", tables=1, hand_cap=30, repeat=1)
    assert result["decisions"] > 20
    assert result["mean_us"] < 10 * bench.TARGET_US  # generous: shared CI machines; the target itself is reported
    assert result["max_state_bytes"] <= 64 * 1024
