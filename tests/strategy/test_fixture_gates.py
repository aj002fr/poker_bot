"""Every Stage 2 fixture gate: analytic values and thresholds within declared tolerance."""

import pytest

from strategy.games.fixtures import FAST_RESOLUTION, FIXTURES, FULL_RESOLUTION, gate_table, run_fixture
from strategy.solve import nash_conv
from strategy.solve.evaluate import expected_values

ES_FIXTURES = sorted(fid for fid, fx in FIXTURES.items() if fx.es_iterations is not None)
ES_FAST = ["matching-pennies", "clairvoyance", "limit-akq", "kuhn", "marked-draw", "half-street-no-fold"]


def _assert_gates(run):
    assert run.passed, "\n" + gate_table(run.failures())


@pytest.mark.parametrize("fixture_id", sorted(FIXTURES))
def test_fixture_gates(fixture_id):
    _assert_gates(run_fixture(fixture_id, resolution=FAST_RESOLUTION))


@pytest.mark.slow
@pytest.mark.parametrize("fixture_id", sorted(fid for fid, fx in FIXTURES.items() if fx.uses_resolution))
def test_fixture_gates_full_resolution(fixture_id):
    _assert_gates(run_fixture(fixture_id, resolution=FULL_RESOLUTION))


def _assert_sampled_agrees(fixture_id: str, seed: int, method: str = "es"):
    """ES-MCCFR converges and agrees with full traversal.

    For two-player zero-sum games |v(sigma) - v*| <= max unilateral gain, so
    the two methods' values may differ by at most the sum of their gains.
    """
    fx = FIXTURES[fixture_id]
    full = run_fixture(fixture_id)
    sampled = run_fixture(fixture_id, method=method, seed=seed)
    tree = full.tree
    gain_full = nash_conv(tree, full.policy).max_gain
    gain_sampled = nash_conv(tree, sampled.policy).max_gain
    assert gain_sampled <= fx.es_tolerance, f"{fixture_id}: ES gain {gain_sampled:.4g} > {fx.es_tolerance}"
    gap = abs(float(expected_values(tree, full.policy)[0] - expected_values(tree, sampled.policy)[0]))
    assert gap <= gain_full + gain_sampled + 1e-9


@pytest.mark.parametrize("fixture_id", ES_FAST)
def test_sampled_and_full_traversal_agree(fixture_id):
    _assert_sampled_agrees(fixture_id, seed=0)


@pytest.mark.slow
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("method", ["es", "es-linear"])
@pytest.mark.parametrize("fixture_id", ES_FIXTURES)
def test_sampled_and_full_traversal_agree_all(fixture_id, method, seed):
    _assert_sampled_agrees(fixture_id, seed=seed, method=method)
