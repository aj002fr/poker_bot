"""Endgame tests that need numba: the evaluator kernel and the cross-check with strategy.solve."""

from __future__ import annotations

import random

import numpy as np
import pytest

pytest.importorskip("numba")

from strategy.endgame import kernels  # noqa: E402
from strategy.endgame.showdown import ShowdownModel  # noqa: E402
from strategy.endgame.solver import PushFoldSolver  # noqa: E402
from strategy.endgame.subgame import SubgameTree  # noqa: E402
from strategy.features.cards import capabilities, evaluate_any  # noqa: E402
from strategy.solve import CFR_PLUS, CFRSolver, compile_game, nash_conv  # noqa: E402
from strategy.solve.game import Chance, Decision, Terminal  # noqa: E402
from tests.strategy.test_endgame import _game, synthetic_model  # noqa: E402


@pytest.mark.parametrize("ruleset", ["practice_nlhe", "uncertainty_shortdeck"])
def test_numba_evaluator_matches_the_feature_evaluator(ruleset):
    caps = capabilities(ruleset)
    rng = random.Random(7)
    hands = [rng.sample(list(caps.deck), 7) for _ in range(3000)]
    got = kernels.evaluate_many(np.array(hands, np.int64), np.array(caps.straight_high, np.int64),
                                np.array(caps.strength, np.int64))
    want = [evaluate_any(h, caps) for h in hands]
    assert got.tolist() == want


class _TreeGame:
    """The same heads-up push/fold game as an ``ExtensiveGame`` for :mod:`strategy.solve`."""

    name = "pushfold-hu"
    num_players = 2
    player_names = ("SB", "BB")

    def __init__(self, tree: SubgameTree, model: ShowdownModel) -> None:
        self.tree, self.model = tree, model

    def root(self):
        return ()

    def expand(self, state):
        m = self.model
        if state == ():
            h = m.h
            return Chance(tuple(((i, j), float(m.joint[i, j]), (i, j, 0)) for i in range(h) for j in range(h)))
        i, j, node = state
        nd = self.tree.nodes[node]
        cls = (i, j)
        if nd.player >= 0:
            acts = [("f", (i, j, nd.children["f"]))] + [(k, (i, j, c)) for k, c in nd.children.items() if k != "f"]
            return Decision(nd.player, (nd.player, node, cls[nd.player]), tuple(acts))
        t = self.tree.terminals[node]
        u = t.utility
        if len(t.showdown) == 1:
            return Terminal(tuple(float(x) for x in u[0]))
        a, b = t.showdown
        pw, pt, pl = m.p_win[cls[a], cls[b]], m.p_tie[cls[a], cls[b]], m.p_lose[cls[a], cls[b]]
        return Terminal(tuple(float(pt * u[0, p] + pw * u[1, p] + pl * u[2, p]) for p in range(2)))


def _as_policy(gtree, solver: PushFoldSolver, sigma):
    probs = np.zeros(gtree.num_slots)
    for key, idx in gtree.key_to_infoset.items():
        p, node, c = key
        probs[gtree.slots(idx)] = [sigma[node][0][c], sigma[node][1][c]]
    return probs


@pytest.mark.parametrize("stacks", [(800, 1200), (2500, 400)])
def test_solver_best_response_matches_strategy_solve_on_the_same_game(stacks):
    model = synthetic_model()
    tree = SubgameTree(_game(stacks, utility="chips"))
    solver = PushFoldSolver(tree, model).run(60)
    sigma = solver.average()
    gtree = compile_game(_TreeGame(tree, model))
    report = nash_conv(gtree, _as_policy(gtree, solver, sigma))
    mine = np.array([solver.report(sigma, q).gain for q in range(2)])
    assert np.allclose(report.gains, mine, atol=1e-9)
    assert np.allclose(report.values, solver.values(sigma), atol=1e-9)
    # Both solvers converge to the same game value.
    ref = CFRSolver(gtree, CFR_PLUS).run(3000).average_policy()
    full = PushFoldSolver(tree, model).run(3000)
    assert nash_conv(gtree, ref).nash_conv < 0.05
    assert full.values(full.average())[0] == pytest.approx(nash_conv(gtree, ref).values[0], abs=0.05)
