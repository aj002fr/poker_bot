import random
from itertools import combinations

import pytest

from tests.oracles.cards import InvalidCards, parse_cards, standard_deck
from tests.oracles.omaha import best_omaha, omaha_combinations
from tests.oracles.ranking import (
    FLUSH,
    FULL_HOUSE,
    HIGH_CARD,
    ONE_PAIR,
    STRAIGHT,
    STRAIGHT_FLUSH,
    THREE_OF_A_KIND,
    rank_five_unchecked,
)
from tests.oracles.standard import STANDARD, best_nlhe


def omaha(hole, board):
    return best_omaha(parse_cards(hole), parse_cards(board))


def test_river_enumerates_exactly_60_combinations():
    hole, board = parse_cards("2h 3d 4c 5h"), parse_cards("As Ks Qs Js Ts")
    combos = list(omaha_combinations(hole, board))
    assert len(combos) == 60 == len(set(combos))
    assert all(len(h) == 2 and len(b) == 3 for h, b in combos)
    assert omaha("2h 3d 4c 5h", "As Ks Qs Js Ts").combinations_checked == 60


@pytest.mark.parametrize("board, expected", [("As Ks Qs", 6), ("As Ks Qs Js", 24)])
def test_flop_and_turn_combination_counts(board, expected):
    assert omaha("2h 3d 4c 5h", board).combinations_checked == expected


def test_e09_cannot_play_a_royal_flush_board():
    result = omaha("2h 3d 4c 5h", "As Ks Qs Js Ts")
    assert result.value.category == HIGH_CARD
    assert result.value.tiebreak == (14, 13, 12, 5, 4)
    assert len(result.hole_used) == 2 and len(result.board_used) == 3
    holdem_style = best_nlhe(parse_cards("2h 3d"), parse_cards("As Ks Qs Js Ts")).value
    assert holdem_style.category == STRAIGHT_FLUSH


def test_e10_four_board_spades_one_hole_spade_is_no_flush():
    result = omaha("Qs Jh Th 9c", "As Ks 7s 4s 2d")
    assert result.value.category != FLUSH
    assert result.value.category == HIGH_CARD
    assert result.value.tiebreak == (14, 13, 12, 11, 7)
    assert best_nlhe(parse_cards("Qs Jh"), parse_cards("As Ks 7s 4s 2d")).value.category == FLUSH


def test_two_hole_spades_with_three_board_spades_is_a_flush():
    result = omaha("Qs 3s Th 9c", "As Ks 7s 4d 2d")
    assert result.value.category == FLUSH
    assert result.value.tiebreak == (14, 13, 12, 7, 3)


def test_one_hole_card_cannot_complete_a_board_straight():
    result = omaha("9c 2d 2h 3c", "Tc Jd Qh Ks 4s")
    assert result.value.category != STRAIGHT
    assert omaha("9c 8d 2h 3c", "Tc Jd Qh Ks 4s").value.category == STRAIGHT


def test_board_trips_with_hole_pair_is_full_house_but_board_boat_is_not():
    assert omaha("Ac Ad 3h 4s", "Kc Kd Kh 7c 2s").value.category == FULL_HOUSE
    result = omaha("Ah 2c 3d 4s", "Ks Kd Kh Qs Qd")
    assert result.value.category == THREE_OF_A_KIND
    assert result.value.tiebreak == (13, 14, 4)


def test_double_paired_hole_cannot_make_two_pair_from_hole_alone():
    result = omaha("As Ad Ks Kd", "2c 7h 9d Jc 3s")
    assert result.value.category == ONE_PAIR
    assert result.value.tiebreak == (14, 11, 9, 7)
    assert set(result.hole_used) == set(parse_cards("As Ad"))


@pytest.mark.parametrize(
    "hole, board",
    [
        ("As Kd", "2c 3d 4h 5s 9c"),
        ("As Kd Qc Jh Th", "2c 3d 4h 5s 9c"),
        ("As Kd Qc Jh", "2c 3d"),
        ("As Kd Qc Jh", "As 3d 4h 5s 9c"),
        ("As Kd Qc Jh", "2c 3d 4h 5s 9c 8c"),
    ],
)
def test_invalid_omaha_inputs_rejected(hole, board):
    with pytest.raises(InvalidCards):
        omaha(hole, board)


def test_sampled_against_nine_card_subset_brute_force():
    rng = random.Random(4)
    deck = standard_deck()
    for _ in range(1500):
        cards = rng.sample(deck, 9)
        hole, board = cards[:4], cards[4:]
        hole_set = set(hole)
        expected = max(
            rank_five_unchecked(five, STANDARD)
            for five in combinations(cards, 5)
            if sum(c in hole_set for c in five) == 2
        )
        assert best_omaha(hole, board).value == expected
