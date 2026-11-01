import numpy as np
import pytest

from strategy.games import akq, clairvoyance, uniform
from strategy.solve import (
    TabularPolicy,
    best_response,
    compile_game,
    expected_values,
    exploitability,
    nash_conv,
)
from tests.strategy.helpers import brute_force_best_response, kuhn_tree, random_policy, reference


def _small_trees():
    return {
        "kuhn": kuhn_tree(),
        "limit-akq": compile_game(akq.LimitAKQ()),
        "two-street": compile_game(clairvoyance.TwoStreetClairvoyant(fractions=(0.5, 1.0))),
        "dry-side-pot": compile_game(uniform.DrySidePot(n=4, pot=9.0)),
    }


@pytest.mark.parametrize("game", ["kuhn", "limit-akq", "two-street", "dry-side-pot"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_best_response_matches_brute_force(game, seed):
    tree = _small_trees()[game]
    policy = random_policy(tree, seed)
    for player in range(tree.num_players):
        if not np.any(tree.infoset_player == player):
            continue
        value, response = best_response(tree, policy, player)
        assert value == pytest.approx(brute_force_best_response(tree, policy, player), abs=1e-12)
        mask = tree.player_slot_mask(player)
        played = np.where(mask, response.probs, policy.probs)
        assert float(expected_values(tree, played)[player]) == pytest.approx(value, abs=1e-12)
        assert np.all(np.isin(response.probs[mask], (0.0, 1.0)))
        assert np.array_equal(response.probs[~mask], policy.probs[~mask])


def test_uniform_kuhn_nash_conv_matches_openspiel():
    tree = kuhn_tree()
    ref = reference()["games"]["kuhn"]
    assert nash_conv(tree, TabularPolicy.uniform(tree)).nash_conv == pytest.approx(ref["uniform_nash_conv"], abs=1e-12)


def test_known_equilibrium_has_zero_nash_conv():
    tree = kuhn_tree()
    alpha = 0.2
    probs = TabularPolicy.uniform(tree).probs.copy()
    eq = {
        (0, "J", ""): (1 - alpha, alpha), (0, "Q", ""): (1, 0), (0, "K", ""): (1 - 3 * alpha, 3 * alpha),
        (0, "J", "pb"): (1, 0), (0, "Q", "pb"): (2 / 3 - alpha, 1 / 3 + alpha), (0, "K", "pb"): (0, 1),
        (1, "J", "p"): (2 / 3, 1 / 3), (1, "Q", "p"): (1, 0), (1, "K", "p"): (0, 1),
        (1, "J", "b"): (1, 0), (1, "Q", "b"): (2 / 3, 1 / 3), (1, "K", "b"): (0, 1),
    }
    for key, dist in eq.items():
        probs[tree.slots(tree.infoset(key))] = dist
    report = nash_conv(tree, probs)
    assert report.nash_conv == pytest.approx(0.0, abs=1e-12)
    assert report.values[0] == pytest.approx(-1 / 18, abs=1e-12)
    assert exploitability(tree, probs) == pytest.approx(0.0, abs=1e-12)


def test_exploitability_is_half_nash_conv_in_two_player_zero_sum():
    tree = kuhn_tree()
    policy = random_policy(tree, 4)
    assert exploitability(tree, policy) == pytest.approx(nash_conv(tree, policy).nash_conv / 2)


def test_exploitability_is_refused_for_three_players():
    tree = compile_game(uniform.DrySidePot(n=4, pot=9.0))
    assert tree.num_players == 3
    with pytest.raises(ValueError, match="two-player zero-sum"):
        exploitability(tree, TabularPolicy.uniform(tree))
    gains = nash_conv(tree, TabularPolicy.uniform(tree)).gains
    assert gains.shape == (3,) and np.all(gains >= -1e-12)
