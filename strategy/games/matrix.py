"""Two-player zero-sum matrix games, as normal-form payoffs and as extensive-form games."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from strategy.solve.game import Decision, Terminal

MATCHING_PENNIES = np.array([[1.0, -1.0], [-1.0, 1.0]])
ROCK_PAPER_SCISSORS = np.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])


@dataclass(frozen=True)
class MatrixGame:
    """Row player (0) moves, then column player (1) moves without observing it."""

    name: str
    payoff: np.ndarray = field(repr=False)
    row_labels: tuple[str, ...]
    col_labels: tuple[str, ...]
    num_players: int = 2
    player_names: tuple[str, ...] = ("row", "col")

    def root(self) -> tuple:
        return ()

    def expand(self, state: tuple):
        if not state:
            return Decision(0, "row", tuple((label, (i,)) for i, label in enumerate(self.row_labels)))
        if len(state) == 1:
            return Decision(1, "col", tuple((label, (state[0], j)) for j, label in enumerate(self.col_labels)))
        u = float(self.payoff[state[0], state[1]])
        return Terminal((u, -u))


def matching_pennies() -> MatrixGame:
    return MatrixGame("matching-pennies", MATCHING_PENNIES, ("heads", "tails"), ("heads", "tails"))


def rock_paper_scissors() -> MatrixGame:
    labels = ("rock", "paper", "scissors")
    return MatrixGame("rock-paper-scissors", ROCK_PAPER_SCISSORS, labels, labels)
