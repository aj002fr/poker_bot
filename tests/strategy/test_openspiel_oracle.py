"""Live cross-checks against OpenSpiel (``make verify-oracle``; dev-only dependency)."""

import json

import numpy as np
import pytest

from strategy.games import kuhn, leduc
from strategy.solve import PRESETS, CFRSolver, Chance, Decision, Terminal, nash_conv
from tests.strategy.helpers import REFERENCE, kuhn_tree, leduc_tree, random_policy

pytestmark = pytest.mark.oracle
pyspiel = pytest.importorskip("pyspiel")

from open_spiel.python import policy as os_policy  # noqa: E402
from open_spiel.python.algorithms import cfr, discounted_cfr, exploitability  # noqa: E402

GAMES = {
    "kuhn": (kuhn.KuhnPoker, kuhn_tree, "kuhn_poker", kuhn.CARDS.index, {"p": 0, "b": 1}),
    "leduc": (leduc.LeducPoker, leduc_tree, "leduc_poker", int, {"f": 0, "c": 1, "r": 2}),
}
OS_SOLVERS = {
    "cfr": cfr.CFRSolver,
    "cfr+": cfr.CFRPlusSolver,
    "linear-cfr": discounted_cfr.LCFRSolver,
    "dcfr": discounted_cfr.DCFRSolver,
}


def lockstep_infoset_map(name: str) -> dict:
    """Walk our game and OpenSpiel's together; return our key -> (OpenSpiel infostate, action ids)."""
    game_cls, _, short, chance_action, action_id = GAMES[name]
    game = game_cls()
    os_game = pyspiel.load_game(short)
    mapping: dict = {}
    reverse: dict = {}
    stack = [(game.root(), os_game.new_initial_state())]
    while stack:
        state, os_state = stack.pop()
        node = game.expand(state)
        if isinstance(node, Terminal):
            assert os_state.is_terminal()
            assert np.allclose(node.utility, os_state.returns())
            continue
        if isinstance(node, Chance):
            assert os_state.is_chance_node()
            ours = {chance_action(label): (p, child) for label, p, child in node.outcomes}
            theirs = dict(os_state.chance_outcomes())
            assert set(ours) == set(theirs)
            for a, (p, child) in ours.items():
                assert p == pytest.approx(theirs[a], abs=1e-15)
                stack.append((child, os_state.child(a)))
            continue
        assert isinstance(node, Decision)
        assert os_state.current_player() == node.player
        ids = tuple(action_id[label] for label, _ in node.actions)
        assert sorted(ids) == os_state.legal_actions()
        info = os_state.information_state_string()
        assert mapping.setdefault(node.infoset, (info, ids)) == (info, ids)
        assert reverse.setdefault(info, node.infoset) == node.infoset
        for (_, child), a in zip(node.actions, ids, strict=True):
            stack.append((child, os_state.child(a)))
    return mapping


def to_openspiel(name: str, tree, probs) -> os_policy.TabularPolicy:
    os_game = pyspiel.load_game(GAMES[name][2])
    out = os_policy.TabularPolicy(os_game)
    for key, (info, ids) in lockstep_infoset_map(name).items():
        row = out.policy_for_key(info)
        row[:] = 0.0
        row[list(ids)] = probs[tree.slots(tree.infoset(key))]
    return out


@pytest.mark.parametrize("name", ["kuhn", "leduc"])
def test_trees_are_isomorphic(name):
    mapping = lockstep_infoset_map(name)
    tree = GAMES[name][1]()
    assert len(mapping) == tree.num_infosets


@pytest.mark.parametrize("solver_name", sorted(OS_SOLVERS))
@pytest.mark.parametrize("name, checkpoints", [("kuhn", (1, 2, 5, 50)), ("leduc", (1, 2, 5))])
def test_average_policies_match_openspiel_iterate_by_iterate(name, checkpoints, solver_name):
    tree = GAMES[name][1]()
    mapping = lockstep_infoset_map(name)
    os_game = pyspiel.load_game(GAMES[name][2])
    theirs = OS_SOLVERS[solver_name](os_game)
    ours = CFRSolver(tree, PRESETS[solver_name])
    done = 0
    for k in checkpoints:
        for _ in range(k - done):
            theirs.evaluate_and_update_policy()
        ours.run(k - done)
        done = k
        avg = theirs.average_policy()
        mine = ours.average_policy().probs
        worst = 0.0
        for key, (info, ids) in mapping.items():
            expected = np.asarray(avg.policy_for_key(info))[list(ids)]
            worst = max(worst, float(np.abs(mine[tree.slots(tree.infoset(key))] - expected).max()))
        assert worst < 1e-10, f"{name}/{solver_name} after {k}: {worst}"


@pytest.mark.parametrize("name", ["kuhn", "leduc"])
@pytest.mark.parametrize("seed", [0, 1])
def test_nash_conv_matches_openspiel(name, seed):
    tree = GAMES[name][1]()
    policy = random_policy(tree, seed)
    os_game = pyspiel.load_game(GAMES[name][2])
    expected = exploitability.nash_conv(os_game, to_openspiel(name, tree, policy.probs))
    assert nash_conv(tree, policy).nash_conv == pytest.approx(expected, abs=1e-10)


def test_frozen_reference_matches_a_fresh_kuhn_run():
    from tests.strategy.oracle import openspiel_reference as gen

    frozen = json.loads(REFERENCE.read_text())["games"]["kuhn"]
    os_game = pyspiel.load_game("kuhn_poker")
    assert gen.tree_counts(os_game) == frozen["counts"]
    solver = cfr.CFRSolver(os_game)
    for _ in range(10):
        solver.evaluate_and_update_policy()
    row = next(r for r in frozen["solvers"]["cfr"] if r["iterations"] == 10)
    assert exploitability.nash_conv(os_game, solver.average_policy()) == pytest.approx(row["average_nash_conv"], abs=1e-12)
