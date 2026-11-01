"""External-sampling Monte Carlo CFR (Lanctot et al., 2009) on a `GameTree`.

Each iteration traverses once per player that owns an information set. The
traverser's actions are all expanded; chance and every other player's action
are sampled from their probabilities, so the sampled counterfactual value
needs no importance weight. The other players' current strategies are added
to their strategy sums at each sampled visit ("simple" averaging), which is
reach-proportional in two-player games. ``linear=True`` weights iteration t's
regret and strategy contributions by t (linear MCCFR, as used by Pluribus).

Randomness comes from numba's generator, reseeded deterministically at the
start of every `run` call from ``seed`` and the iteration count.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from strategy.solve import kernels as K
from strategy.solve.faults import Fault
from strategy.solve.game import GameTree
from strategy.solve.policy import TabularPolicy, normalize_per_infoset


class ExternalSamplingMCCFR:
    def __init__(self, tree: GameTree, *, seed: int = 0, linear: bool = False, fault: Fault = Fault.NONE):
        self.tree = tree
        self.seed = int(seed)
        self.linear = bool(linear)
        self.fault = int(fault)
        self.iterations = 0
        self.regret = np.zeros(tree.num_slots)
        self.strategy_sum = np.zeros(tree.num_slots)
        self._players = np.array(
            [p for p in range(tree.num_players) if np.any(tree.infoset_player == p)], dtype=np.int64
        )
        self._scratch_sigma = np.zeros((tree.max_depth + 1, max(tree.max_actions, 1)))
        self._scratch_vals = np.zeros_like(self._scratch_sigma)
        self._counter = np.zeros(1, dtype=np.int64)

    @property
    def name(self) -> str:
        return "es-mccfr-linear" if self.linear else "es-mccfr"

    @property
    def nodes_touched(self) -> int:
        return int(self._counter[0])

    def run(self, iterations: int) -> ExternalSamplingMCCFR:
        tr = self.tree
        K.seed_rng((self.seed * 1_000_003 + self.iterations) % (2**31 - 1))
        K.es_iterations(
            self.iterations, iterations, self._players, self.linear,
            tr.node_player, tr.node_infoset, tr.first_child, tr.num_children, tr.node_chance_cdf,
            tr.node_terminal, tr.terminal_utility, tr.infoset_offset,
            self.regret, self.strategy_sum, self.fault, self._scratch_sigma, self._scratch_vals, self._counter,
        )
        self.iterations += iterations
        return self

    def current_policy(self) -> TabularPolicy:
        return TabularPolicy(self.tree, normalize_per_infoset(self.tree, self.regret))

    def average_policy(self) -> TabularPolicy:
        if self.fault == Fault.LAST_ITERATE:
            return self.current_policy()
        return TabularPolicy.from_strategy_sum(self.tree, self.strategy_sum)


@dataclass(frozen=True)
class SampledRegret:
    mean: np.ndarray
    stderr: np.ndarray
    samples: int


def sampled_regret_estimate(
    tree: GameTree,
    policy: TabularPolicy,
    traverser: int,
    *,
    samples: int,
    seed: int = 0,
    fault: Fault = Fault.NONE,
) -> SampledRegret:
    """Mean and standard error of one-traversal external-sampling regrets under a fixed profile."""
    total = np.zeros(tree.num_slots)
    total_sq = np.zeros(tree.num_slots)
    scratch = np.zeros((tree.max_depth + 1, max(tree.max_actions, 1)))
    K.seed_rng(seed)
    K.es_regret_samples(
        samples, traverser, tree.node_player, tree.node_infoset, tree.first_child, tree.num_children,
        tree.node_chance_cdf, tree.node_terminal, tree.terminal_utility, tree.infoset_offset,
        policy.probs, int(fault), scratch, np.zeros_like(scratch), total, total_sq, np.zeros(1, dtype=np.int64),
    )
    mean = total / samples
    var = np.maximum(total_sq / samples - mean**2, 0.0)
    return SampledRegret(mean=mean, stderr=np.sqrt(var / samples), samples=samples)
