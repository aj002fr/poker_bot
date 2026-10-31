import random

import pytest

from tests.oracles.cards import parse_cards
from tests.oracles.pots import clockwise_after, settle
from tests.oracles.standard import best_nlhe

A, B, C, D = 1, 2, 3, 4


def test_g13_three_all_ins_refund_and_side_pot():
    board = parse_cards("2c 3d 7h 9s Jc")
    hole = {A: "As Ad", B: "Ks Kd", C: "Qs Qd"}
    strengths = {seat: best_nlhe(parse_cards(h), board).value for seat, h in hole.items()}
    assert strengths[A] > strengths[B] > strengths[C]
    stacks = {A: 100, B: 250, C: 400}
    s = settle(stacks, folded=(), strengths=strengths, button=C)
    assert s.refunds == {C: 150}
    assert [(p.amount, p.eligible, p.winners) for p in s.pots] == [(300, (A, B, C), (A,)), (300, (B, C), (B,))]
    assert s.main_pot.amount == 300
    final = s.final_stacks(stacks, stacks)
    assert final == {A: 300, B: 300, C: 150}
    assert sum(final.values()) == 750


def test_g14_folded_dead_money_stays_in_lower_layers():
    contributions = {A: 100, B: 250, C: 400, D: 50}
    s = settle(contributions, folded={D}, strengths={A: 3, B: 2, C: 1}, button=D)
    assert s.refunds == {C: 150}
    assert [(p.amount, p.contributors, p.eligible, p.winners) for p in s.pots] == [
        (200, (A, B, C, D), (A, B, C), (A,)),
        (150, (A, B, C), (A, B, C), (A,)),
        (300, (B, C), (B, C), (B,)),
    ]
    assert sum(p.amount for p in s.pots if p.winners == (A,)) == 350
    assert s.totals == {A: 350, B: 300, C: 150, D: 0}
    assert sum(s.totals.values()) == 800


def test_g14_merged_layers_give_same_totals():
    contributions = {A: 100, B: 250, C: 400, D: 50}
    s = settle(contributions, folded={D}, strengths={A: 3, B: 2, C: 1}, button=D, merge_equal_eligibility=True)
    assert [p.amount for p in s.pots] == [350, 300]
    assert s.totals == {A: 350, B: 300, C: 150, D: 0}


def test_g15_odd_chip_to_first_tied_winner_clockwise_after_button():
    button = 0
    a, b, c = 1, 2, 0
    s = settle({a: 34, b: 34, c: 33}, folded={c}, strengths={a: 5, b: 5}, button=button)
    assert [(p.amount, p.shares) for p in s.pots] == [(99, {a: 50, b: 49}), (2, {a: 1, b: 1})]
    assert s.totals == {a: 51, b: 50, c: 0}
    assert s.refunds == {}


def test_g15_odd_chip_follows_button_not_seat_number():
    s = settle({1: 34, 2: 34, 3: 33}, folded={3}, strengths={1: 5, 2: 5}, button=1)
    assert s.pots[0].shares == {2: 50, 1: 49}
    assert s.totals == {1: 50, 2: 51, 3: 0}


def test_g22_ties_in_different_side_pots_split_separately():
    # Seats 0-5, button 2: clockwise order after the button is 3, 4, 5, 0, 1.
    contributions = {0: 21, 1: 21, 2: 46, 3: 46, 4: 60, 5: 8}
    strengths = {0: 3, 1: 3, 2: 2, 3: 2, 4: 1}
    s = settle(contributions, folded={5}, strengths=strengths, button=2)
    assert [(p.amount, p.eligible, p.winners, dict(p.shares)) for p in s.pots] == [
        (48, (0, 1, 2, 3, 4), (0, 1), {0: 24, 1: 24}),
        (65, (0, 1, 2, 3, 4), (0, 1), {0: 33, 1: 32}),
        (75, (2, 3, 4), (3, 2), {3: 38, 2: 37}),
    ]
    assert s.refunds == {4: 14}
    assert s.totals == {0: 57, 1: 56, 2: 37, 3: 38, 4: 14, 5: 0}
    assert sum(s.totals.values()) == sum(contributions.values()) == 202


THREE_WAY_104 = {0: 26, 1: 26, 2: 26, 3: 26}


def test_g22_three_way_tie_remainder_one_each_clockwise():
    # 104 chips three ways is 34 r2; the spec gives one each to the first two clockwise.
    s = settle(THREE_WAY_104, folded={3}, strengths={0: 1, 1: 1, 2: 1}, button=0)
    assert s.pots[0].amount == 104 and s.refunds == {}
    assert s.pots[0].winners == (1, 2, 0)
    assert dict(s.pots[0].shares) == {1: 35, 2: 35, 0: 34}
    contributions = {0: 33, 1: 34, 2: 34, 3: 1}
    s = settle(contributions, folded={3}, strengths={0: 1, 1: 1, 2: 1}, button=1)
    assert [p.amount for p in s.pots] == [4, 96, 2]
    assert s.pots[0].shares == {2: 2, 0: 1, 1: 1}


def test_layering_versus_merging_changes_odd_chips():
    # Two folded short contributors create three layers with identical eligibility.
    contributions = {0: 1, 1: 2, 2: 3, 3: 3, 4: 3}
    strengths = {2: 9, 3: 9, 4: 1}
    layered = settle(contributions, folded={0, 1}, strengths=strengths, button=4)
    merged = settle(contributions, folded={0, 1}, strengths=strengths, button=4, merge_equal_eligibility=True)
    assert [p.amount for p in layered.pots] == [5, 4, 3]
    assert [p.amount for p in merged.pots] == [12]
    assert (layered.totals[2], layered.totals[3]) == (7, 5)
    assert (merged.totals[2], merged.totals[3]) == (6, 6)


def test_all_remainder_to_first_variant():
    s = settle(THREE_WAY_104, folded={3}, strengths={0: 1, 1: 1, 2: 1}, button=0, odd_chips="all_to_first_clockwise")
    assert dict(s.pots[0].shares) == {1: 36, 2: 34, 0: 34}
    with pytest.raises(ValueError, match="odd-chip"):
        settle(THREE_WAY_104, folded={3}, strengths={0: 1, 1: 1, 2: 1}, button=0, odd_chips="random")


def test_uncalled_bet_is_returned_not_a_side_pot():
    s = settle({0: 300, 1: 100}, folded=(), strengths={0: 1, 1: 2}, button=0)
    assert s.refunds == {0: 200}
    assert [(p.amount, p.winners) for p in s.pots] == [(200, (1,))]
    assert s.totals == {0: 200, 1: 200}


def test_everyone_folds_to_one_player():
    s = settle({0: 10, 1: 20, 2: 60}, folded={0, 1}, strengths={2: 0}, button=2)
    assert s.refunds == {2: 40}
    assert s.totals == {0: 0, 1: 0, 2: 90}


def test_zero_contributor_never_wins():
    s = settle({0: 0, 1: 50, 2: 50}, folded=(), strengths={0: 99, 1: 1, 2: 2}, button=0)
    assert all(0 not in p.eligible for p in s.pots)
    assert s.totals == {0: 0, 1: 0, 2: 100}


@pytest.mark.parametrize("bad", [-1, 1.5, True, "10", None])
def test_rejects_non_integer_chips(bad):
    with pytest.raises(ValueError):
        settle({0: bad, 1: 10}, folded=(), strengths={0: 1, 1: 1}, button=0)


def test_rejects_missing_strength_and_unknown_folds():
    with pytest.raises(ValueError, match="strength"):
        settle({0: 10, 1: 10}, folded=(), strengths={0: 1}, button=0)
    with pytest.raises(ValueError, match="ledger"):
        settle({0: 10, 1: 10}, folded={7}, strengths={0: 1, 1: 1}, button=0)
    with pytest.raises(ValueError, match="no eligible"):
        settle({0: 10, 1: 10, 2: 5}, folded={0, 1}, strengths={2: 1}, button=0)


def test_clockwise_after_button():
    assert clockwise_after(5, [0, 2, 5, 7]) == (7, 0, 2, 5)
    assert clockwise_after(7, [0, 2, 5, 7]) == (0, 2, 5, 7)


def test_randomized_conservation_and_eligibility():
    rng = random.Random(2026)
    for _ in range(2000):
        n = rng.randint(2, 8)
        seats = rng.sample(range(8), n)
        contributions = {s: rng.choice([0, 1, rng.randint(1, 50), rng.randint(1, 5000)]) for s in seats}
        positive = [s for s in seats if contributions[s] > 0]
        if not positive:
            continue
        top = max(contributions.values())
        must_live = rng.choice([s for s in positive if contributions[s] == top])
        folded = {s for s in seats if s != must_live and rng.random() < 0.3}
        strengths = {s: rng.randint(0, 3) for s in seats if s not in folded}
        merge = rng.random() < 0.5
        result = settle(contributions, folded, strengths, button=rng.choice(seats), merge_equal_eligibility=merge)
        assert sum(result.totals.values()) == sum(contributions.values())
        assert all(v >= 0 for v in result.totals.values())
        for f in folded:
            assert result.winnings.get(f, 0) == 0
        for pot in result.pots:
            assert set(pot.winners) <= set(pot.eligible) <= set(pot.contributors)
            assert all(contributions[w] >= pot.level_to for w in pot.winners)
            assert sum(pot.shares.values()) == pot.amount
            assert max(pot.shares.values()) - min(pot.shares.values()) <= 1
        assert len(result.refunds) <= 1
        for seat, chips in result.refunds.items():
            others = [contributions[s] for s in seats if s != seat]
            assert chips == contributions[seat] - max(others)
