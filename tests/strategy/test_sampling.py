"""External-sampling MCCFR: unbiased regret estimates, determinism, convergence."""

import numpy as np
import pytest

from strategy.games import akq, marked_draw
from strategy.solve import ExternalSamplingMCCFR, Fault, compile_game, nash_conv
from tests.strategy.helpers import kuhn_tree, leduc_tree, max_regret_z, random_policy

Z_LIMIT = 6.0


def _trees():
    return {
        "kuhn": kuhn_tree(),
        "limit-akq": compile_game(akq.LimitAKQ()),
        "marked-draw": compile_game(marked_draw.MarkedDraw()),
        "leduc": leduc_tree(),
    }


@pytest.mark.parametrize("game", ["kuhn", "limit-akq", "marked-draw", "leduc"])
@pytest.mark.parametrize("traverser", [0, 1])
def test_sampled_regrets_are_unbiased(game, traverser):
    tree = _trees()[game]
    policy = random_policy(tree, seed=11)
    samples = 100_000 if game == "leduc" else 40_000
    assert max_regret_z(tree, policy, traverser, samples) < Z_LIMIT


@pytest.mark.parametrize(
    "fault, game",
    [
        (Fault.ES_UNIFORM_OPPONENT, "kuhn"),
        (Fault.ES_UNIFORM_OPPONENT, "leduc"),
        (Fault.ES_UNIFORM_CHANCE, "marked-draw"),
        (Fault.ES_NO_EXPLORATION, "kuhn"),
        (Fault.ES_NO_EXPLORATION, "leduc"),
    ],
)
def test_biased_sampling_variants_are_detected(fault, game):
    tree = _trees()[game]
    policy = random_policy(tree, seed=11)
    samples = 100_000 if game == "leduc" else 40_000
    worst = max(max_regret_z(tree, policy, p, samples, fault=fault) for p in (0, 1))
    assert worst > 3 * Z_LIMIT


def test_uniform_chance_is_harmless_only_when_chance_is_uniform():
    tree = kuhn_tree()
    policy = random_policy(tree, seed=5)
    assert max_regret_z(tree, policy, 0, 40_000, fault=Fault.ES_UNIFORM_CHANCE) < Z_LIMIT


def test_runs_are_deterministic_per_seed_and_resumable():
    tree = kuhn_tree()
    a = ExternalSamplingMCCFR(tree, seed=7).run(3000)
    b = ExternalSamplingMCCFR(tree, seed=7).run(3000)
    c = ExternalSamplingMCCFR(tree, seed=8).run(3000)
    assert np.array_equal(a.regret, b.regret) and np.array_equal(a.strategy_sum, b.strategy_sum)
    assert not np.array_equal(a.regret, c.regret)
    assert a.nodes_touched == b.nodes_touched
    split = ExternalSamplingMCCFR(tree, seed=7).run(1000).run(2000)
    assert split.iterations == 3000
    assert nash_conv(tree, split.average_policy()).nash_conv < 0.1


def test_sampling_touches_far_fewer_nodes_per_iteration():
    tree = leduc_tree()
    solver = ExternalSamplingMCCFR(tree, seed=0).run(1000)
    per_iteration = solver.nodes_touched / solver.iterations
    assert per_iteration < tree.num_nodes / 50


def test_lone_acting_player_still_accumulates_an_average():
    from strategy.games import uniform

    tree = compile_game(uniform.HalfStreetNoFold(n=20))
    assert set(tree.infoset_player.tolist()) == {1}
    solver = ExternalSamplingMCCFR(tree, seed=0).run(5000)
    assert solver.strategy_sum.sum() > 0
    assert nash_conv(tree, solver.average_policy()).max_gain < 0.02


@pytest.mark.slow
@pytest.mark.parametrize("linear", [False, True])
def test_long_es_run_on_leduc(linear):
    tree = leduc_tree()
    solver = ExternalSamplingMCCFR(tree, seed=1, linear=linear).run(2_000_000)
    assert nash_conv(tree, solver.average_policy()).nash_conv < 0.04
