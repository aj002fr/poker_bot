import numpy as np
import pytest

from strategy.games import matrix
from strategy.solve import regret_matching_selfplay
from strategy.solve import kernels as K


def test_regret_matching_kernel_uses_positive_part_and_falls_back_to_uniform():
    regret = np.array([2.0, -1.0, 6.0, -3.0, -2.0, 0.0])
    offset = np.array([0, 3, 6], dtype=np.int32)
    players = np.array([0, 0], dtype=np.int32)
    sigma = np.zeros(6)
    K.regret_matching(regret, offset, players, 0, sigma)
    assert sigma[:3] == pytest.approx([0.25, 0.0, 0.75])
    assert sigma[3:] == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_regret_matching_kernel_only_touches_the_requested_player():
    regret = np.array([1.0, 0.0, 0.0, 1.0])
    offset = np.array([0, 2, 4], dtype=np.int32)
    players = np.array([0, 1], dtype=np.int32)
    sigma = np.full(4, -1.0)
    K.regret_matching(regret, offset, players, 1, sigma)
    assert list(sigma) == [-1.0, -1.0, 0.0, 1.0]


@pytest.mark.parametrize("payoff", [matrix.MATCHING_PENNIES, matrix.ROCK_PAPER_SCISSORS], ids=["pennies", "rps"])
def test_selfplay_average_converges_while_last_iterate_does_not(payoff):
    k = payoff.shape[0]
    start = (np.eye(k)[0] * 3.0, np.eye(k)[min(1, k - 1)] * 2.0)
    out = regret_matching_selfplay(payoff, iterations=20_000, initial_regret=start)
    uniform = np.full(k, 1 / k)
    assert np.abs(out.average[0] - uniform).max() < 0.01
    assert np.abs(out.average[1] - uniform).max() < 0.01
    assert abs(out.average_value) < 1e-3
    assert max(np.abs(out.last[0] - uniform).max(), np.abs(out.last[1] - uniform).max()) > 0.25


@pytest.mark.parametrize("payoff", [matrix.MATCHING_PENNIES, matrix.ROCK_PAPER_SCISSORS], ids=["pennies", "rps"])
def test_average_external_regret_respects_the_regret_matching_bound(payoff):
    # Hart & Mas-Colell: average external regret <= range * sqrt(|A| / T) from zero initial regret.
    iterations = 5000
    out = regret_matching_selfplay(payoff, iterations=iterations)
    t = np.arange(1, iterations + 1)
    bound = float(payoff.max() - payoff.min()) * np.sqrt(payoff.shape[0] / t)
    assert np.all(out.external_regret <= bound[:, None] + 1e-12)


def test_general_sum_column_payoffs_are_respected():
    # Prisoner's dilemma: defection (action 1) strictly dominates for both players.
    row = np.array([[3.0, 0.0], [5.0, 1.0]])
    out = regret_matching_selfplay(row, row.T, iterations=2000)
    assert out.average[0][1] > 0.99 and out.average[1][1] > 0.99
