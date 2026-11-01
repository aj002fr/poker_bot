"""Property tests for point-denominated call thresholds and the ICM risk premium (test spec T004)."""

from __future__ import annotations

from fractions import Fraction

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from arena.tournament.scoring import icm
from strategy.math import (
    call_point_gain,
    icm_all_in_call,
    point_call_threshold,
    pot_odds_threshold,
    risk_premium,
)

TOL = 1e-9

utility = st.fractions(min_value=0, max_value=100, max_denominator=100)


@given(u_fold=utility, u_win=utility, u_lose=utility)
def test_point_threshold_is_the_indifference_win_probability(u_fold, u_win, u_lose):
    assume(u_win > u_lose)
    q = point_call_threshold(u_fold, u_win, u_lose)
    assert q * u_win + (1 - q) * u_lose == u_fold
    if u_lose <= u_fold <= u_win:
        assert 0 <= q <= 1


@given(u_fold=utility, u_win=utility, u_lose=utility, p=st.fractions(min_value=0, max_value=1, max_denominator=500))
def test_call_point_gain_sign_matches_threshold(u_fold, u_win, u_lose, p):
    assume(u_win > u_lose)
    gain = call_point_gain([(p, u_win), (1 - p, u_lose)], u_fold)
    q = point_call_threshold(u_fold, u_win, u_lose)
    assert (gain > 0) == (p > q)


@given(
    pot=st.fractions(min_value=1, max_value=1000, max_denominator=10),
    call=st.fractions(min_value=1, max_value=1000, max_denominator=10),
    stack=st.fractions(min_value=0, max_value=10_000, max_denominator=10),
    k=st.fractions(min_value=Fraction(1, 10), max_value=10, max_denominator=10),
    c=st.fractions(min_value=-50, max_value=50, max_denominator=10),
)
def test_linear_point_utility_has_no_risk_premium(pot, call, stack, k, c):
    """Points that are an affine function of chips reproduce chip pot odds exactly."""
    assume(stack >= call)

    def u(chips):
        return k * chips + c

    u_fold = u(stack)
    u_win = u(stack + pot)
    u_lose = u(stack - call)
    assert risk_premium(u_fold, u_win, u_lose, pot, call) == 0


def test_call_point_gain_handles_ties():
    gain = call_point_gain([(Fraction(1, 2), 30), (Fraction(1, 10), 20), (Fraction(2, 5), 0)], 15)
    assert gain == 15 + 2 - 15


all_in_states = st.integers(min_value=2, max_value=8).flatmap(
    lambda n: st.tuples(
        st.lists(st.integers(min_value=1, max_value=6000), min_size=n, max_size=n),
        st.lists(st.integers(min_value=0, max_value=100), min_size=n, max_size=n),
        st.integers(min_value=0, max_value=n - 1),
        st.integers(min_value=1, max_value=n - 1),
        st.integers(min_value=0, max_value=200),
        st.integers(min_value=0, max_value=200),
    )
)


def _call(state, prizes_override=None):
    stacks, prizes, villain, offset, hero_committed, dead = state
    n = len(stacks)
    hero = (villain + offset) % n
    villain_committed = stacks[villain] + hero_committed + 1
    stacks = list(stacks)
    stacks[villain] = 0
    prizes = sorted(prizes if prizes_override is None else prizes_override(n), reverse=True)
    return icm_all_in_call(stacks, prizes, hero, villain, hero_committed, villain_committed, dead), stacks, prizes, hero


@given(all_in_states)
def test_icm_call_utilities_are_ordered(state):
    stacks, prizes, *_ = state
    assume(len(set(prizes)) > 1)
    try:
        r, *_ = _call(state)
    except ValueError:
        assume(False)
    assert r.u_lose <= r.u_fold + TOL
    assert r.u_fold <= r.u_win + TOL
    q = r.point_threshold
    assert q * r.u_win + (1 - q) * r.u_lose == pytest.approx(r.u_fold, abs=1e-7)


@given(all_in_states, st.integers(min_value=1, max_value=1000))
def test_winner_take_all_points_have_no_risk_premium(state, first):
    r, *_ = _call(state, prizes_override=lambda n: [first] + [0] * (n - 1))
    assert r.risk_premium == pytest.approx(0.0, abs=TOL)


@given(all_in_states)
def test_two_survivors_have_no_risk_premium(state):
    stacks, prizes, villain, offset, hero_committed, dead = state
    state = (stacks[:2], prizes[:2], villain % 2, 1, hero_committed, dead)
    assume(len(set(prizes[:2])) > 1)
    r, *_ = _call(state)
    assert r.risk_premium == pytest.approx(0.0, abs=TOL)


@given(all_in_states)
def test_chip_threshold_is_pot_odds_on_matched_chips(state):
    stacks, prizes, *_ = state
    assume(len(set(prizes)) > 1)
    try:
        r, *_ = _call(state)
    except ValueError:
        assume(False)
    assert r.chip_threshold == pot_odds_threshold(r.contested_pot - r.matched_call, r.matched_call)


@given(all_in_states, st.randoms(use_true_random=False))
def test_relabelling_bystanders_leaves_hero_values_unchanged(state, rng):
    stacks, prizes, villain, offset, hero_committed, dead = state
    assume(len(set(prizes)) > 1)
    try:
        r, stacks0, prizes0, hero = _call(state)
    except ValueError:
        assume(False)
    order = list(range(len(stacks0)))
    rng.shuffle(order)
    permuted = [stacks0[i] for i in order]
    moved = icm_all_in_call(
        permuted,
        prizes0,
        order.index(hero),
        order.index(villain),
        hero_committed,
        stacks[villain] + hero_committed + 1,
        dead,
    )
    assert (moved.u_fold, moved.u_win, moved.u_lose) == pytest.approx((r.u_fold, r.u_win, r.u_lose), abs=TOL)


def test_covered_hero_who_loses_takes_the_last_remaining_prize():
    r = icm_all_in_call([0, 500, 900], [50, 30, 20], hero=1, villain=0, hero_committed=0, villain_committed=2000)
    assert r.u_lose == 20
    assert r.matched_call == 500
    assert r.contested_pot == 1000


def test_short_villain_returns_nothing_and_busts_on_the_win_branch():
    r = icm_all_in_call([0, 3000, 900], [50, 30, 20], hero=1, villain=0, hero_committed=20, villain_committed=600)
    assert r.matched_call == 580
    assert r.contested_pot == 1200
    assert r.u_win == pytest.approx(icm([3620, 900], [50, 30])[0])


def test_medium_stack_covered_by_leader_needs_a_large_premium():
    """Regression: a covered medium stack with a short stack still alive.

    Leader shoves 6,000 total; hero has 3,000 behind, a 700 short stack waits, and
    300 of blinds are dead. Chip odds need under half the pot; points need about 73%.
    """
    r = icm_all_in_call([0, 4000, 700, 3000], [50, 30, 20, 0], hero=3, villain=0, hero_committed=0,
                        villain_committed=6000, dead=300)
    assert r.chip_threshold == pytest.approx(3000 / 6300)
    assert r.point_threshold == pytest.approx(0.7322381795, abs=1e-9)
    assert r.risk_premium > 0.25


def test_value_fn_hook_replaces_icm():
    def chips_only(stacks, prizes):
        total = sum(stacks)
        return [sum(prizes) * s / total for s in stacks]

    r = icm_all_in_call([0, 4000, 700, 3000], [50, 30, 20, 0], 3, 0, 0, 6000, 300, value_fn=chips_only)
    assert r.u_lose == 0.0
    assert r.u_fold > 0


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(stacks=[0, 100], prizes=[1, 0], hero=0, villain=0, hero_committed=0, villain_committed=10),
        dict(stacks=[0, 0], prizes=[1, 0], hero=1, villain=0, hero_committed=0, villain_committed=10),
        dict(stacks=[5, 100], prizes=[1, 0], hero=1, villain=0, hero_committed=0, villain_committed=10),
        dict(stacks=[0, 100], prizes=[1, 0], hero=1, villain=0, hero_committed=10, villain_committed=10),
    ],
)
def test_icm_all_in_call_rejects_invalid_states(kwargs):
    with pytest.raises(ValueError):
        icm_all_in_call(**kwargs)


def test_point_threshold_requires_win_above_lose():
    with pytest.raises(ValueError):
        point_call_threshold(1, 1, 1)
