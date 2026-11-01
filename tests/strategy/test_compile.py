import numpy as np
import pytest

from strategy.games.fixtures import FIXTURES
from strategy.solve import CHANCE, TERMINAL, GameDefinitionError, compile_game
from tests.strategy.helpers import SpecGame, kuhn_tree, leduc_tree, reference


def _counts(tree):
    return {
        "nodes": tree.num_nodes,
        "terminal": int((tree.node_player == TERMINAL).sum()),
        "chance": int((tree.node_player == CHANCE).sum()),
        "decision": int((tree.node_player >= 0).sum()),
        "infosets": tree.num_infosets,
    }


@pytest.mark.parametrize("name, tree_fn", [("kuhn", kuhn_tree), ("leduc", leduc_tree)])
def test_tree_sizes_match_openspiel(name, tree_fn):
    assert _counts(tree_fn()) == reference()["games"][name]["counts"]


def test_kuhn_and_leduc_counts_are_the_known_ones():
    assert _counts(kuhn_tree()) == {"nodes": 58, "terminal": 30, "chance": 4, "decision": 24, "infosets": 12}
    assert _counts(leduc_tree()) == {"nodes": 9457, "terminal": 5520, "chance": 157, "decision": 3780, "infosets": 936}


@pytest.mark.parametrize("fixture_id", sorted(FIXTURES))
def test_compiled_tree_invariants(fixture_id):
    tree = compile_game(FIXTURES[fixture_id].build(8))
    n = tree.num_nodes
    for h in range(n):
        lo, k = int(tree.first_child[h]), int(tree.num_children[h])
        kids = np.arange(lo, lo + k)
        assert np.all(kids > h)
        assert np.all(tree.node_parent[kids] == h)
        assert np.all(tree.node_depth[kids] == tree.node_depth[h] + 1)
        p = tree.node_player[h]
        if p == TERMINAL:
            assert k == 0 and tree.node_terminal[h] >= 0
        elif p == CHANCE:
            assert k > 0 and np.all(tree.node_slot[kids] == -1)
            assert tree.node_chance_prob[kids].sum() == pytest.approx(1.0, abs=1e-12)
            assert tree.node_chance_cdf[kids[-1]] == 1.0
            assert np.all(np.diff(tree.node_chance_cdf[kids]) > 0)
        else:
            i = tree.node_infoset[h]
            assert tree.infoset_player[i] == p
            assert np.array_equal(tree.node_slot[kids], np.arange(tree.infoset_offset[i], tree.infoset_offset[i + 1]))
            assert np.all(tree.node_chance_prob[kids] == 1.0)
    assert np.all(tree.slot_infoset[tree.infoset_offset[:-1]] == np.arange(tree.num_infosets))
    assert sorted(tree.node_terminal[tree.node_terminal >= 0]) == list(range(tree.terminal_utility.shape[0]))


def test_infoset_lookup_round_trips():
    tree = kuhn_tree()
    for i, key in enumerate(tree.infoset_keys):
        assert tree.infoset(key) == i
    assert tree.action_labels[tree.infoset((1, "Q", "b"))] == ("p", "b")


def test_imperfect_recall_is_rejected():
    # Player 0 forgets its own first action: both continuations share infoset "b".
    second = ("D", 0, "b", (("x", ("T", (1, -1))), ("y", ("T", (-1, 1)))))
    game = SpecGame(("D", 0, "a", (("l", second), ("r", second))))
    with pytest.raises(GameDefinitionError, match="imperfect recall"):
        compile_game(game)


def test_shared_infoset_after_opponent_or_chance_is_fine():
    leaf = ("D", 0, "b", (("x", ("T", (1, -1))), ("y", ("T", (-1, 1)))))
    game = SpecGame(("C", (("h", 0.5, ("D", 1, "o", (("u", leaf), ("v", leaf)))), ("t", 0.5, leaf))))
    tree = compile_game(game)
    assert tree.num_infosets == 2


def test_inconsistent_actions_are_rejected():
    a = ("D", 1, "same", (("x", ("T", (0, 0))),))
    b = ("D", 1, "same", (("y", ("T", (0, 0))),))
    with pytest.raises(GameDefinitionError, match="inconsistent"):
        compile_game(SpecGame(("C", (("1", 0.5, a), ("2", 0.5, b)))))


@pytest.mark.parametrize(
    "spec, message",
    [
        (("C", (("a", 0.5, ("T", (0, 0))), ("b", 0.4, ("T", (0, 0))))), "sum to"),
        (("C", (("a", 1.0, ("T", (0, 0))), ("b", 0.0, ("T", (0, 0))))), "positive"),
        (("T", (1, 2, 3)), "entries"),
        (("D", 2, "k", (("a", ("T", (0, 0))),)), "invalid player"),
        (("D", 0, "k", ()), "without actions"),
    ],
)
def test_malformed_games_are_rejected(spec, message):
    with pytest.raises(GameDefinitionError, match=message):
        compile_game(SpecGame(spec))


def test_node_budget_is_enforced():
    from strategy.games import leduc

    with pytest.raises(GameDefinitionError, match="exceeds"):
        compile_game(leduc.LeducPoker(), max_nodes=1000)
