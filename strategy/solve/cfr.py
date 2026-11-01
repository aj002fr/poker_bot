"""Full-traversal counterfactual regret minimisation on a `GameTree`.

One solver covers vanilla CFR, CFR+, linear CFR and discounted CFR; the
variants differ only in `CFRConfig`. With alternating updates the iteration
order matches OpenSpiel's `CFRSolver`/`CFRPlusSolver`/`DCFRSolver`
exactly: for each player in turn, traverse under the current profile, add
that player's regrets and reach-weighted strategy, discount/floor its
regrets, then refresh its regret-matching strategy before the next player.

Discounting follows Brown & Sandholm (2019): after iteration t, positive
cumulative regrets are multiplied by t^alpha / (t^alpha + 1), negative ones by
t^beta / (t^beta + 1), and iteration t's strategy contribution is weighted by
t^gamma.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from strategy.solve import kernels as K
from strategy.solve.faults import Fault
from strategy.solve.game import GameTree
from strategy.solve.policy import TabularPolicy


@dataclass(frozen=True)
class CFRConfig:
    name: str
    alternating: bool = True
    regret_matching_plus: bool = False
    alpha: float | None = None
    beta: float | None = None
    gamma: float = 0.0


VANILLA = CFRConfig("cfr")
CFR_PLUS = CFRConfig("cfr+", regret_matching_plus=True, gamma=1.0)
LINEAR = CFRConfig("linear-cfr", alpha=1.0, beta=1.0, gamma=1.0)
DISCOUNTED = CFRConfig("dcfr", alpha=1.5, beta=0.0, gamma=2.0)
PRESETS = {c.name: c for c in (VANILLA, CFR_PLUS, LINEAR, DISCOUNTED)}


class CFRSolver:
    """Deterministic full-traversal solver. ``nodes_touched`` counts node visits."""

    def __init__(self, tree: GameTree, config: CFRConfig = VANILLA, *, fault: Fault = Fault.NONE):
        self.tree = tree
        self.config = config
        self.fault = int(fault)
        self.iterations = 0
        self.nodes_touched = 0
        self.regret = np.zeros(tree.num_slots)
        self.strategy_sum = np.zeros(tree.num_slots)
        self.sigma = TabularPolicy.uniform(tree).probs.copy()
        self._slot_player = tree.infoset_player[tree.slot_infoset].astype(np.int32)
        self._players = [p for p in range(tree.num_players) if np.any(tree.infoset_player == p)]
        self._edge_w = np.empty(tree.num_nodes)
        self._reach = np.empty((tree.num_nodes, tree.num_players + 1))
        self._values = np.empty((tree.num_nodes, tree.num_players))

    def run(self, iterations: int) -> CFRSolver:
        for _ in range(iterations):
            self._iterate()
        return self

    def _iterate(self) -> None:
        self.iterations += 1
        t = float(self.iterations)
        cfg = self.config
        avg_weight = t**cfg.gamma if cfg.gamma else 1.0
        pos = t**cfg.alpha / (t**cfg.alpha + 1.0) if cfg.alpha is not None else 1.0
        neg = t**cfg.beta / (t**cfg.beta + 1.0) if cfg.beta is not None else 1.0
        passes = self._players if cfg.alternating else [-1]
        tr = self.tree
        for player in passes:
            K.edge_weights(tr.node_slot, tr.node_chance_prob, self.sigma, self._edge_w)
            K.forward_reach(tr.node_player, tr.node_parent, self._edge_w, self.fault, self._reach)
            K.backward_values(tr.node_parent, tr.node_terminal, tr.terminal_utility, self._edge_w, self._values)
            K.accumulate_regrets(
                tr.node_player, tr.node_infoset, tr.first_child, tr.num_children, tr.infoset_offset,
                self._reach, self._values, self.sigma, player, avg_weight, self.fault,
                self.regret, self.strategy_sum,
            )
            K.update_cumulative_regret(self.regret, self._slot_player, player, cfg.regret_matching_plus, pos, neg)
            K.regret_matching(self.regret, tr.infoset_offset, tr.infoset_player, player, self.sigma)
            self.nodes_touched += tr.num_nodes

    def current_policy(self) -> TabularPolicy:
        return TabularPolicy(self.tree, self.sigma.copy())

    def average_policy(self) -> TabularPolicy:
        if self.fault == Fault.LAST_ITERATE:
            return self.current_policy()
        return TabularPolicy.from_strategy_sum(self.tree, self.strategy_sum)
