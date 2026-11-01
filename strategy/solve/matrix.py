"""Regret matching self-play on two-player normal-form games.

The smallest solver checkpoint: both players run regret matching
simultaneously against each other's current mixed strategy. The time-averaged
strategies converge to equilibrium in zero-sum games; the last iterate need
not, which the Stage 2 tests use to show the average is what gets deployed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MatrixSelfPlay:
    average: tuple[np.ndarray, np.ndarray]
    last: tuple[np.ndarray, np.ndarray]
    average_value: float  # row player's value of the average profile
    external_regret: np.ndarray  # [T, 2]: max_a sum_t u(a, s_-i^t) - sum_t u(s^t), divided by t


def _regret_matching(regret: np.ndarray) -> np.ndarray:
    pos = np.maximum(regret, 0.0)
    total = pos.sum()
    return pos / total if total > 0 else np.full(regret.shape, 1.0 / regret.size)


def regret_matching_selfplay(
    payoff_row: np.ndarray,
    payoff_col: np.ndarray | None = None,
    *,
    iterations: int,
    initial_regret: tuple[np.ndarray, np.ndarray] | None = None,
) -> MatrixSelfPlay:
    a = np.asarray(payoff_row, dtype=np.float64)
    b = -a if payoff_col is None else np.asarray(payoff_col, dtype=np.float64)
    r_row = np.zeros(a.shape[0]) if initial_regret is None else np.asarray(initial_regret[0], dtype=np.float64).copy()
    r_col = np.zeros(a.shape[1]) if initial_regret is None else np.asarray(initial_regret[1], dtype=np.float64).copy()
    sum_row = np.zeros(a.shape[0])
    sum_col = np.zeros(a.shape[1])
    cum_action_row = np.zeros(a.shape[0])
    cum_action_col = np.zeros(a.shape[1])
    realized = np.zeros(2)
    ext = np.empty((iterations, 2))
    s_row = s_col = None
    for t in range(iterations):
        s_row = _regret_matching(r_row)
        s_col = _regret_matching(r_col)
        u_row = a @ s_col
        u_col = s_row @ b
        v_row = float(s_row @ u_row)
        v_col = float(u_col @ s_col)
        r_row += u_row - v_row
        r_col += u_col - v_col
        sum_row += s_row
        sum_col += s_col
        cum_action_row += u_row
        cum_action_col += u_col
        realized += (v_row, v_col)
        ext[t] = (cum_action_row.max() - realized[0], cum_action_col.max() - realized[1])
        ext[t] /= t + 1
    avg_row = sum_row / iterations
    avg_col = sum_col / iterations
    return MatrixSelfPlay(
        average=(avg_row, avg_col),
        last=(s_row, s_col),
        average_value=float(avg_row @ a @ avg_col),
        external_regret=ext,
    )
