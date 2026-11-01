"""Exact evaluation of policies on a `GameTree`.

* `expected_values` — utility vector of a profile.
* `reach_probabilities` / `infoset_reach` — per-node and per-infoset reach.
* `best_response` — exact pure best response for one player.
* `nash_conv` — sum of unilateral best-response gains. For two-player
  zero-sum games this is twice the exploitability; for other games it is
  reported as a deviation gain, never as exploitability.
* `counterfactual_regrets` — the exact one-iteration regret increment that
  sampled estimators must match in expectation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from strategy.solve import kernels as K
from strategy.solve.game import GameTree
from strategy.solve.policy import TabularPolicy


def _probs(policy: TabularPolicy | np.ndarray) -> np.ndarray:
    return policy.probs if isinstance(policy, TabularPolicy) else np.asarray(policy, dtype=np.float64)


def edge_weights(tree: GameTree, probs: np.ndarray) -> np.ndarray:
    out = np.empty(tree.num_nodes)
    K.edge_weights(tree.node_slot, tree.node_chance_prob, probs, out)
    return out


def node_values(tree: GameTree, policy: TabularPolicy | np.ndarray) -> np.ndarray:
    values = np.empty((tree.num_nodes, tree.num_players))
    K.backward_values(tree.node_parent, tree.node_terminal, tree.terminal_utility, edge_weights(tree, _probs(policy)), values)
    return values


def expected_values(tree: GameTree, policy: TabularPolicy | np.ndarray) -> np.ndarray:
    return node_values(tree, policy)[0].copy()


def reach_probabilities(tree: GameTree, policy: TabularPolicy | np.ndarray) -> np.ndarray:
    """``[n, P + 1]``: each player's own reach contribution, then chance."""
    reach = np.empty((tree.num_nodes, tree.num_players + 1))
    K.forward_reach(tree.node_player, tree.node_parent, edge_weights(tree, _probs(policy)), 0, reach)
    return reach


@dataclass(frozen=True)
class InfosetReach:
    own: np.ndarray  # pi_i(I): acting player's own contribution (equal across I under perfect recall)
    external: np.ndarray  # sum over h in I of opponents' and chance's contribution
    total: np.ndarray  # sum over h in I of the full reach probability

    def of(self, tree: GameTree, key: object) -> tuple[float, float, float]:
        i = tree.infoset(key)
        return float(self.own[i]), float(self.external[i]), float(self.total[i])


def infoset_reach(tree: GameTree, policy: TabularPolicy | np.ndarray) -> InfosetReach:
    reach = reach_probabilities(tree, policy)
    decision = np.flatnonzero(tree.node_player >= 0)
    iset = tree.node_infoset[decision]
    players = tree.node_player[decision]
    full = reach.prod(axis=1)[decision]
    own_r = reach[decision, players]
    external_r = _external_product(reach[decision], players)
    own = np.zeros(tree.num_infosets)
    np.maximum.at(own, iset, own_r)
    external = np.bincount(iset, weights=external_r, minlength=tree.num_infosets)
    total = np.bincount(iset, weights=full, minlength=tree.num_infosets)
    return InfosetReach(own, external, total)


def _external_product(reach_rows: np.ndarray, players: np.ndarray) -> np.ndarray:
    masked = reach_rows.copy()
    masked[np.arange(len(players)), players] = 1.0
    return masked.prod(axis=1)


def best_response(tree: GameTree, policy: TabularPolicy | np.ndarray, player: int) -> tuple[float, TabularPolicy]:
    """Value of ``player``'s exact best response to the others' policy, and that response."""
    probs = _probs(policy)
    reach = reach_probabilities(tree, probs)
    sigma = probs.copy()
    values = np.empty((tree.num_nodes, tree.num_players))
    edge_w = np.empty(tree.num_nodes)
    mine = tree.infoset_player == player
    if mine.any():
        q = np.zeros(tree.num_slots)
        for level in range(int(tree.infoset_level[mine].max()), -1, -1):
            K.edge_weights(tree.node_slot, tree.node_chance_prob, sigma, edge_w)
            K.backward_values(tree.node_parent, tree.node_terminal, tree.terminal_utility, edge_w, values)
            q[:] = 0.0
            K.best_response_level(
                tree.node_player, tree.node_infoset, tree.first_child, tree.num_children,
                tree.infoset_offset, tree.infoset_player, tree.infoset_level,
                reach, values, player, level, q, sigma,
            )
    K.edge_weights(tree.node_slot, tree.node_chance_prob, sigma, edge_w)
    K.backward_values(tree.node_parent, tree.node_terminal, tree.terminal_utility, edge_w, values)
    return float(values[0, player]), TabularPolicy(tree, sigma)


@dataclass(frozen=True)
class NashConvReport:
    values: np.ndarray  # profile value per player
    best_response_values: np.ndarray
    gains: np.ndarray  # unilateral deviation gain per player (>= 0 up to rounding)

    @property
    def nash_conv(self) -> float:
        return float(self.gains.sum())

    @property
    def max_gain(self) -> float:
        return float(self.gains.max())


def nash_conv(tree: GameTree, policy: TabularPolicy | np.ndarray) -> NashConvReport:
    probs = _probs(policy)
    values = expected_values(tree, probs)
    br = np.array([best_response(tree, probs, p)[0] for p in range(tree.num_players)])
    return NashConvReport(values=values, best_response_values=br, gains=br - values)


def exploitability(tree: GameTree, policy: TabularPolicy | np.ndarray) -> float:
    """Mean best-response gain; defined only for two-player zero-sum games."""
    if tree.num_players != 2 or not tree.is_zero_sum:
        raise ValueError("exploitability is only meaningful for two-player zero-sum games; use nash_conv gains")
    return nash_conv(tree, policy).nash_conv / 2.0


def counterfactual_regrets(tree: GameTree, policy: TabularPolicy | np.ndarray, player: int) -> np.ndarray:
    """Exact vanilla-CFR regret increment for ``player`` under a fixed profile."""
    probs = _probs(policy)
    edge_w = edge_weights(tree, probs)
    reach = np.empty((tree.num_nodes, tree.num_players + 1))
    K.forward_reach(tree.node_player, tree.node_parent, edge_w, 0, reach)
    values = np.empty((tree.num_nodes, tree.num_players))
    K.backward_values(tree.node_parent, tree.node_terminal, tree.terminal_utility, edge_w, values)
    regret = np.zeros(tree.num_slots)
    K.accumulate_regrets(
        tree.node_player, tree.node_infoset, tree.first_child, tree.num_children, tree.infoset_offset,
        reach, values, probs, player, 0.0, 0, regret, np.zeros(tree.num_slots),
    )
    return regret
