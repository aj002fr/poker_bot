"""The deployed strategy is the reach-weighted average, not the last iterate."""

import numpy as np
import pytest

from strategy.games.fixtures import run_fixture
from strategy.solve import PRESETS, CFRSolver, ExternalSamplingMCCFR, Fault, nash_conv
from tests.strategy.helpers import average_weighting_error, kuhn_tree, leduc_tree


def test_vanilla_cfr_average_converges_but_last_iterate_does_not():
    tree = kuhn_tree()
    solver = CFRSolver(tree, PRESETS["cfr"]).run(1000)
    average = nash_conv(tree, solver.average_policy()).nash_conv
    last = nash_conv(tree, solver.current_policy()).nash_conv
    assert average < 0.002
    assert last > 0.05
    assert last > 20 * average


def test_es_mccfr_average_beats_its_current_strategy():
    tree = kuhn_tree()
    solver = ExternalSamplingMCCFR(tree, seed=3).run(20_000)
    average = nash_conv(tree, solver.average_policy()).nash_conv
    last = nash_conv(tree, solver.current_policy()).nash_conv
    assert average < 0.02
    assert last > 3 * average


@pytest.mark.parametrize("solver_name", sorted(PRESETS))
@pytest.mark.parametrize("tree_fn", [kuhn_tree, leduc_tree], ids=["kuhn", "leduc"])
def test_average_is_the_reach_weighted_mixture_of_iterates(tree_fn, solver_name):
    assert average_weighting_error(tree_fn(), solver_name, iterations=15) < 1e-10


def test_first_iteration_strategy_sum_is_own_reach_times_uniform():
    tree = kuhn_tree()
    solver = CFRSolver(tree, PRESETS["cfr"]).run(1)
    # Uniform play: player 1's infosets have own reach 1, player 0's second decision has 1/2.
    # Like OpenSpiel, the sum is accumulated per history (two per Kuhn infoset).
    for key, own in [((0, "J", ""), 1.0), ((1, "Q", "b"), 1.0), ((0, "K", "pb"), 0.5)]:
        i = tree.infoset(key)
        histories = int((tree.node_infoset == i).sum())
        assert histories == 2
        assert np.allclose(solver.strategy_sum[tree.slots(i)], histories * own * 0.5)


def test_last_iterate_fault_reports_the_current_strategy():
    tree = kuhn_tree()
    solver = CFRSolver(tree, PRESETS["cfr"], fault=Fault.LAST_ITERATE).run(50)
    assert np.array_equal(solver.average_policy().probs, solver.current_policy().probs)


def test_fixture_runs_measure_the_average_policy():
    run = run_fixture("kuhn", iterations=200)
    solver = CFRSolver(run.tree, PRESETS["cfr+"]).run(200)
    assert np.allclose(run.policy.probs, solver.average_policy().probs)
