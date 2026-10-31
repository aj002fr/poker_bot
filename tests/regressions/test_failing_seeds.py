"""Every failure the release gates have found stays fixed.

``failing_seeds.json`` retains each failing generated-hand seed, and
``fixtures/`` holds the minimized hand for it (``python -m tests.release
minimize ... --write``). Fixtures expect the settlement of the independent
oracles and run through the golden-fixture runner, which also re-checks the
finished hand against the oracles.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.property.generated import play_generated_hand
from tests.release.repro import minimize, record_generated, replay_fixture
from tests.support.fixtures import run_fixture

HERE = Path(__file__).resolve().parent
SEEDS = json.loads((HERE / "failing_seeds.json").read_text())
FIXTURES = sorted((HERE / "fixtures").glob("*.json"))


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_minimized_fixture(path):
    fx = json.loads(path.read_text())
    assert "expect" in fx, "a regression fixture must state the expected settlement"
    run_fixture(fx)


@pytest.mark.parametrize("entry", SEEDS, ids=lambda e: f"{e['ruleset']}-{e['players']}-{e['seed']}")
def test_failing_seed_now_passes(entry):
    assert (HERE.parent.parent / entry["fixture"]).exists()
    play_generated_hand(entry["seed"], entry["ruleset"], entry["players"])


def test_minimizer_shrinks_while_the_property_holds():
    """Self-test on a property that holds, not on a bug: 'some pot is split between tied winners'."""
    seed = next(s for s in range(2000, 4000)
                if _split_pot(record_generated(s, "practice_nlhe", 8)[0]) and
                len(record_generated(s, "practice_nlhe", 8)[0]["setup"]["stacks"]) == 8)
    fx, error = record_generated(seed, "practice_nlhe", 8)
    assert error is None
    small = minimize(fx, _split_pot)
    assert _split_pot(small)
    assert len(small["setup"]["stacks"]) < len(fx["setup"]["stacks"])


def _split_pot(fx) -> bool:
    status, hand = replay_fixture(fx)
    return status == "ok" and any(sum(1 for v in p.awards.values() if v) > 1 for p in hand.result().pots)
