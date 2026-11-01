"""Reach probabilities, cross-street posteriors and card removal."""

from dataclasses import dataclass, replace

import numpy as np
import pytest

from strategy.games import akq, leduc, marked_draw
from strategy.games.fixtures import FIXTURES, run_fixture
from strategy.solve import CHANCE, TabularPolicy, compile_game, expected_values, infoset_reach, nash_conv
from strategy.solve.evaluate import reach_probabilities
from tests.strategy.helpers import kuhn_tree, leduc_tree, reference


def policy_with(tree, overrides):
    probs = TabularPolicy.uniform(tree).probs.copy()
    for key, dist in overrides.items():
        i = tree.infoset(key)
        probs[tree.slots(i)] = [dist[a] for a in tree.action_labels[i]]
    return TabularPolicy(tree, probs)


def histories(tree, key):
    return np.flatnonzero(tree.node_infoset == tree.infoset(key))


def first_chance_label(tree, h):
    """Index of the root chance outcome on the path to node ``h``."""
    while tree.node_parent[h] != 0:
        h = tree.node_parent[h]
    return int(h - tree.first_child[0])


@dataclass(frozen=True)
class AKQWithReplacement(akq.LimitAKQ):
    def deals(self):
        return tuple((x, y, 1.0 / 9.0) for x in akq.CARDS for y in akq.CARDS)


@dataclass(frozen=True)
class LeducPublicFromFullDeck(leduc.LeducPoker):
    def deck_for_public(self, state):
        return list(range(6))


def test_akq_reach_and_posterior_facing_a_bet():
    tree = compile_game(akq.LimitAKQ(pot=2.0))
    policy = policy_with(tree, {("Y", "A"): {"check": 0, "bet1.0": 1}, ("Y", "K"): {"check": 1, "bet1.0": 0},
                                ("Y", "Q"): {"check": 2 / 3, "bet1.0": 1 / 3}})
    reach = infoset_reach(tree, policy)
    own, external, total = reach.of(tree, ("X", "K", "bet1.0"))
    assert own == 1.0
    assert external == pytest.approx(1 / 6 + 1 / 6 * 1 / 3)
    assert total == pytest.approx(external)
    full = reach_probabilities(tree, policy).prod(axis=1)
    hs = histories(tree, ("X", "K", "bet1.0"))
    # Y holds A in 3/4 of the weight once Q bluffs 1/3.
    weights = {first_chance_label(tree, h): full[h] for h in hs}
    deals = akq.LimitAKQ().deals()
    by_y = {deals[i][1]: w for i, w in weights.items()}
    assert set(by_y) == {"A", "Q"}
    assert by_y["A"] / sum(by_y.values()) == pytest.approx(0.75)


def test_akq_card_removal():
    tree = compile_game(akq.LimitAKQ())
    for x in akq.CARDS:
        hs = histories(tree, ("X", x, "bet1.0"))
        assert len(hs) == 2
        deals = akq.LimitAKQ().deals()
        assert all(deals[first_chance_label(tree, h)][1] != x for h in hs)
    broken = compile_game(AKQWithReplacement())
    assert len(histories(broken, ("X", "K", "bet1.0"))) == 3


def test_akq_dealt_with_replacement_fails_its_gate():
    fx = replace(FIXTURES["limit-akq"], id="limit-akq-with-replacement", build=lambda n: AKQWithReplacement())
    run = run_fixture(fx)
    assert not run.passed
    assert {r.check for r in run.failures()} & {"value_y", "y_q_bluff", "x_k_call"}


def test_kuhn_reach_includes_chance_and_opponent():
    tree = kuhn_tree()
    reach = infoset_reach(tree, TabularPolicy.uniform(tree))
    # P1 holds Q facing a bet: two deals of 1/6 each, P0 bets half the time.
    assert reach.of(tree, (1, "Q", "b")) == pytest.approx((1.0, 1 / 6, 1 / 6))
    # P0 holds K after check-bet: own reach 1/2 (it checked), chance 2/6, P1 bets 1/2.
    assert reach.of(tree, (0, "K", "pb")) == pytest.approx((0.5, 1 / 6, 1 / 12))


def test_leduc_public_card_excludes_private_cards():
    tree = leduc_tree()
    second_round = np.flatnonzero((tree.node_player == CHANCE) & (tree.node_depth > 2))
    assert len(second_round) == 157 - 1 - 6
    for h in second_round:
        kids = np.arange(tree.first_child[h], tree.first_child[h] + tree.num_children[h])
        assert len(kids) == 4
        assert np.allclose(tree.node_chance_prob[kids], 0.25)


def test_leduc_uniform_value_matches_openspiel():
    tree = leduc_tree()
    ref = reference()["games"]["leduc"]
    uniform = TabularPolicy.uniform(tree)
    assert float(expected_values(tree, uniform)[0]) == pytest.approx(ref["uniform_value_p0"], abs=1e-12)
    assert nash_conv(tree, uniform).nash_conv == pytest.approx(ref["uniform_nash_conv"], abs=1e-12)


def test_leduc_public_card_from_full_deck_is_caught():
    tree = compile_game(LeducPublicFromFullDeck())
    ref = reference()["games"]["leduc"]
    uniform = TabularPolicy.uniform(tree)
    assert abs(nash_conv(tree, uniform).nash_conv - ref["uniform_nash_conv"]) > 1e-3
    fx = replace(FIXTURES["leduc"], id="leduc-full-deck", build=lambda n: LeducPublicFromFullDeck())
    assert not run_fixture(fx).passed


def test_marked_draw_river_posterior_depends_on_turn_calls():
    """X's flush-river posterior is set by Y's turn calling range: 7/8 draws at equilibrium."""
    run = run_fixture("marked-draw")
    tree, policy = run.tree, run.policy
    game = marked_draw.MarkedDraw()
    c_draw = policy.prob(("Y", "draw", "turn"), "call")
    c_dead = policy.prob(("Y", "dead", "turn"), "call")
    b_draw = policy.prob(("Y", "draw", "flush"), "bet")
    b_dead = policy.prob(("Y", "dead", "flush"), "bet")
    f, d = game.flush_probability, game.draw_fraction
    own, external, _ = infoset_reach(tree, policy).of(tree, ("X", "flush"))
    assert own == 1.0
    assert external == pytest.approx(f * (d * c_draw * b_draw + (1 - d) * c_dead * b_dead), rel=1e-12)
    full = reach_probabilities(tree, policy).prod(axis=1)
    hs = histories(tree, ("X", "flush"))
    draw_weight = sum(full[h] for h in hs if first_chance_label(tree, h) == 0)
    posterior = draw_weight / full[hs].sum()
    assert posterior == pytest.approx(7 / 8, abs=0.01)
