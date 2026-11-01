"""Property tests for chip-denominated betting identities (test spec T001-T003)."""

from __future__ import annotations

from fractions import Fraction

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from strategy.math import (
    bluff_break_even,
    bluff_ev,
    bluff_share_of_betting_range,
    bluff_to_value_ratio,
    call_ev,
    effective_pot_odds,
    geometric_bet_fraction,
    minimum_defense_frequency,
    multi_street_bluff_ev,
    multi_street_bluff_last_fold_break_even,
    pot_fraction_bets,
    pot_odds_threshold,
    semi_bluff_break_even,
    semi_bluff_ev,
)

chips = st.fractions(min_value=0, max_value=10_000, max_denominator=50)
positive = st.fractions(min_value=Fraction(1, 50), max_value=10_000, max_denominator=50)
prob = st.fractions(min_value=0, max_value=1, max_denominator=1000)


@given(pot=positive, call=chips)
def test_pot_odds_threshold_is_the_zero_of_call_ev(pot, call):
    q = pot_odds_threshold(pot, call)
    assert 0 <= q < 1
    assert call_ev(q, pot, call) == 0


@given(pot=positive, call=positive, equity=prob)
def test_call_ev_sign_matches_threshold(pot, call, equity):
    q = pot_odds_threshold(pot, call)
    ev = call_ev(equity, pot, call)
    assert (ev > 0) == (equity > q)
    assert (ev < 0) == (equity < q)


@given(pot=positive, call=positive, k=positive)
def test_pot_odds_are_scale_invariant(pot, call, k):
    assert pot_odds_threshold(k * pot, k * call) == pot_odds_threshold(pot, call)


def test_pot_odds_worked_values():
    assert pot_odds_threshold(100, 50) == Fraction(1, 3)
    assert pot_odds_threshold(Fraction(300), Fraction(100)) == Fraction(1, 4)
    assert pot_odds_threshold(10, 0) == 0


@given(pot_before=chips, bet=positive, stack=positive)
def test_effective_pot_odds_ignores_unmatched_chips(pot_before, bet, stack):
    pot = pot_before + bet
    matched = min(bet, stack)
    assert effective_pot_odds(pot, bet, stack) == pot_odds_threshold(pot_before + matched, matched)
    if stack >= bet:
        assert effective_pot_odds(pot, bet, stack) == pot_odds_threshold(pot, bet)


def test_effective_overbet_case_from_fixture_catalogue():
    # 400 in the middle, a 2,200 shove, but hero can only put in 750 more.
    pot = 400 + 2_200
    assert effective_pot_odds(pot, 2_200, 750) == Fraction(750, 400 + 750 + 750)
    assert effective_pot_odds(pot, 2_200, 750) < pot_odds_threshold(pot, 2_200)


@given(bet=positive, pot=positive)
def test_defence_and_bluff_identities(bet, pot):
    be = bluff_break_even(bet, pot)
    assert be + minimum_defense_frequency(bet, pot) == 1
    assert bluff_ev(be, bet, pot) == 0
    s = bet / pot
    assert be == s / (1 + s)
    assert minimum_defense_frequency(bet, pot) == 1 / (1 + s)
    ratio = bluff_to_value_ratio(bet, pot)
    share = bluff_share_of_betting_range(bet, pot)
    assert share == ratio / (1 + ratio)
    # The bluff-catcher risks `bet` to win `pot + bet` and is indifferent at this mix.
    assert share * (pot + bet) - (1 - share) * bet == 0


def test_clairvoyance_half_street_at_pot_two():
    # Pot 2, unit bet: bluff one third as often as value, defender folds one third.
    assert bluff_to_value_ratio(1, 2) == Fraction(1, 3)
    assert minimum_defense_frequency(1, 2) == Fraction(2, 3)
    assert bluff_share_of_betting_range(1, 2) == Fraction(1, 4)


@given(bet=positive, pot=positive, fold=prob)
def test_semi_bluff_with_zero_equity_is_a_bluff(bet, pot, fold):
    assert semi_bluff_ev(fold, bet, pot, 0) == bluff_ev(fold, bet, pot)
    assert semi_bluff_break_even(bet, pot, 0) == bluff_break_even(bet, pot)


@given(bet=positive, pot=positive, equity=prob)
def test_semi_bluff_break_even_is_minimal(bet, pot, equity):
    f = semi_bluff_break_even(bet, pot, equity)
    assert 0 <= f <= 1
    assert semi_bluff_ev(f, bet, pot, equity) >= 0
    if f > 0:
        assert semi_bluff_ev(f, bet, pot, equity) == 0


@given(bet=positive, pot=positive, fold=prob, e1=prob, e2=prob)
def test_semi_bluff_ev_increases_with_called_equity(bet, pot, fold, e1, e2):
    lo, hi = sorted((e1, e2))
    assert semi_bluff_ev(fold, bet, pot, lo) <= semi_bluff_ev(fold, bet, pot, hi)


def _line_ev_by_enumeration(pot, bets, folds):
    """Independent recursive evaluation of the same heads-up zero-equity line."""

    def go(i, opponent_called):
        if i == len(bets):
            return -sum(bets)
        win_now = pot + opponent_called
        return folds[i] * win_now + (1 - folds[i]) * go(i + 1, opponent_called + bets[i])

    return go(0, 0)


lines = st.integers(min_value=1, max_value=4).flatmap(
    lambda n: st.tuples(st.lists(positive, min_size=n, max_size=n), st.lists(prob, min_size=n, max_size=n))
)


@given(pot=positive, line=lines)
def test_multi_street_ev_matches_tree_enumeration(pot, line):
    bets, folds = line
    assert multi_street_bluff_ev(pot, bets, folds) == _line_ev_by_enumeration(pot, bets, folds)


@given(pot=positive, bet=positive, fold=prob)
def test_one_street_line_is_a_single_bluff(pot, bet, fold):
    assert multi_street_bluff_ev(pot, [bet], [fold]) == bluff_ev(fold, bet, pot)


@given(pot=positive, line=lines, k=positive)
def test_multi_street_ev_boundaries_and_scaling(pot, line, k):
    bets, folds = line
    assert multi_street_bluff_ev(pot, bets, [Fraction(1)] + folds[1:]) == pot
    assert multi_street_bluff_ev(pot, bets, [Fraction(0)] * len(bets)) == -sum(bets)
    assert multi_street_bluff_ev(k * pot, [k * b for b in bets], folds) == k * multi_street_bluff_ev(
        pot, bets, folds
    )


@given(pot=positive, line=lines)
def test_last_fold_break_even_zeroes_the_line(pot, line):
    bets, folds = line
    assume(all(f < 1 for f in folds[:-1]))
    f = multi_street_bluff_last_fold_break_even(pot, bets, folds[:-1])
    if 0 <= f <= 1:
        assert multi_street_bluff_ev(pot, bets, [*folds[:-1], f]) == 0
    elif f < 0:
        assert multi_street_bluff_ev(pot, bets, [*folds[:-1], 0]) > 0
    else:
        assert multi_street_bluff_ev(pot, bets, [*folds[:-1], 1]) < 0


def test_pot_sized_triple_barrel_needs_far_more_than_river_local_folds():
    bets = pot_fraction_bets(Fraction(1), [1, 1, 1])
    assert bets == [1, 3, 9]
    assert multi_street_bluff_last_fold_break_even(Fraction(1), bets, [0, 0]) == Fraction(13, 18)
    # The river viewed alone only needs the one-street break-even.
    assert bluff_break_even(9, 9) == Fraction(1, 2)


@given(pot=positive, fractions=st.lists(st.fractions(min_value=0, max_value=3, max_denominator=20), max_size=4))
def test_pot_fraction_bets_track_called_pot_growth(pot, fractions):
    bets = pot_fraction_bets(pot, fractions)
    running = pot
    for f, b in zip(fractions, bets, strict=True):
        assert b == f * running
        running += 2 * b


@given(
    stack=st.integers(min_value=0, max_value=10_000),
    pot=st.integers(min_value=1, max_value=10_000),
    streets=st.integers(min_value=1, max_value=4),
)
def test_geometric_fraction_commits_the_stack_exactly(stack, pot, streets):
    f = geometric_bet_fraction(stack, pot, streets)
    final_pot = pot * (1 + 2 * f) ** streets
    assert final_pot == pytest.approx(pot + 2 * stack, rel=1e-9)


def test_geometric_two_streets_stack_four_is_pot_sized():
    assert geometric_bet_fraction(4, 1, 2) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "call",
    [
        lambda: pot_odds_threshold(0, 0),
        lambda: pot_odds_threshold(-1, 1),
        lambda: call_ev(Fraction(3, 2), 1, 1),
        lambda: bluff_ev(Fraction(-1, 2), 1, 1),
        lambda: multi_street_bluff_ev(1, [1, 2], [0]),
        lambda: multi_street_bluff_ev(1, [], []),
        lambda: effective_pot_odds(10, 100, 50),
        lambda: geometric_bet_fraction(1, 0, 1),
    ],
)
def test_rejects_invalid_inputs(call):
    with pytest.raises(ValueError):
        call()
