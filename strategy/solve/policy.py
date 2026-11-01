"""Tabular behaviour policies over a `GameTree`'s flat action slots."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass

import numpy as np

from strategy.solve.game import GameTree


@dataclass(frozen=True, eq=False)
class TabularPolicy:
    tree: GameTree
    probs: np.ndarray

    @classmethod
    def uniform(cls, tree: GameTree) -> TabularPolicy:
        counts = np.diff(tree.infoset_offset)
        return cls(tree, (1.0 / counts)[tree.slot_infoset].astype(np.float64))

    @classmethod
    def from_strategy_sum(cls, tree: GameTree, strategy_sum: np.ndarray) -> TabularPolicy:
        return cls(tree, normalize_per_infoset(tree, strategy_sum))

    def infoset_probs(self, infoset: int) -> np.ndarray:
        return self.probs[self.tree.slots(infoset)]

    def action_probs(self, key: Hashable) -> dict[Hashable, float]:
        i = self.tree.infoset(key)
        return dict(zip(self.tree.action_labels[i], self.infoset_probs(i).tolist(), strict=True))

    def prob(self, key: Hashable, action: Hashable) -> float:
        return self.action_probs(key)[action]

    def check_normalized(self, atol: float = 1e-9) -> None:
        if np.any(self.probs < -atol):
            raise ValueError("negative action probability")
        sums = np.add.reduceat(self.probs, self.tree.infoset_offset[:-1]) if self.tree.num_infosets else np.zeros(0)
        if not np.allclose(sums, 1.0, atol=atol):
            raise ValueError("policy is not normalised at every information set")


def normalize_per_infoset(tree: GameTree, weights: np.ndarray) -> np.ndarray:
    """Normalise nonnegative slot weights per infoset; uniform where all are zero."""
    if tree.num_infosets == 0:
        return np.zeros(0)
    w = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    sums = np.add.reduceat(w, tree.infoset_offset[:-1])[tree.slot_infoset]
    counts = np.diff(tree.infoset_offset)[tree.slot_infoset]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(sums > 0.0, w / sums, 1.0 / counts)
    return out
