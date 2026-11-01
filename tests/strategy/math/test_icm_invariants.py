"""T005: invariants of the arena's cap ICM scorer as used by strategy.math.points."""

import random

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from arena.tournament.scoring import icm

stacks_st = st.lists(st.integers(1, 5000), min_size=2, max_size=7)
prizes_st = st.lists(st.integers(0, 100), min_size=1, max_size=7).map(lambda p: sorted(p, reverse=True))


@settings(max_examples=150, deadline=None)
@given(stacks_st, prizes_st)
def test_icm_allocations_sum_to_the_contested_prizes(stacks, prizes):
    assert sum(icm(stacks, prizes)) == pytest.approx(sum(prizes[: len(stacks)]))


@settings(max_examples=60, deadline=None)
@given(st.integers(2, 7), st.integers(1, 5000), prizes_st)
def test_equal_stacks_get_equal_shares(n, stack, prizes):
    values = icm([stack] * n, prizes)
    assert values == pytest.approx([values[0]] * n)


@settings(max_examples=100, deadline=None)
@given(stacks_st, prizes_st, st.integers(2, 50))
def test_scaling_all_stacks_leaves_shares_unchanged(stacks, prizes, k):
    assert icm([s * k for s in stacks], prizes) == pytest.approx(icm(stacks, prizes))


@settings(max_examples=100, deadline=None)
@given(stacks_st, prizes_st, st.randoms(use_true_random=False))
def test_permuting_players_permutes_values(stacks, prizes, rng: random.Random):
    order = list(range(len(stacks)))
    rng.shuffle(order)
    values = icm(stacks, prizes)
    permuted = icm([stacks[i] for i in order], prizes)
    assert permuted == pytest.approx([values[i] for i in order])


@settings(max_examples=100, deadline=None)
@given(stacks_st, prizes_st)
def test_more_chips_never_means_fewer_points(stacks, prizes):
    values = icm(stacks, prizes)
    for i, si in enumerate(stacks):
        for j, sj in enumerate(stacks):
            if si > sj:
                assert values[i] >= values[j] - 1e-9


def test_winner_take_all_is_chip_share():
    assert icm([1, 2, 3], [1]) == pytest.approx([1 / 6, 2 / 6, 3 / 6])
