import random
from collections import Counter

import pytest

from tests.oracles.cards import InvalidCards, parse_cards, standard_deck
from tests.oracles.census import five_card_census, seven_card_category_counts
from tests.oracles.helpers import category_by_counting, class_representatives
from tests.oracles.ranking import (
    FLUSH,
    FOUR_OF_A_KIND,
    FULL_HOUSE,
    HIGH_CARD,
    ONE_PAIR,
    STRAIGHT,
    STRAIGHT_FLUSH,
    THREE_OF_A_KIND,
    TWO_PAIR,
    best_of,
)
from tests.oracles.standard import STANDARD, best_nlhe, rank5

# Spec section 13 category totals.
FIVE_CARD_TOTALS = {
    HIGH_CARD: 1_302_540,
    ONE_PAIR: 1_098_240,
    TWO_PAIR: 123_552,
    THREE_OF_A_KIND: 54_912,
    STRAIGHT: 10_200,
    FLUSH: 5_108,
    FULL_HOUSE: 3_744,
    FOUR_OF_A_KIND: 624,
    STRAIGHT_FLUSH: 40,
}

FIVE_CARD_CLASSES = {
    HIGH_CARD: 1277,
    ONE_PAIR: 2860,
    TWO_PAIR: 858,
    THREE_OF_A_KIND: 858,
    STRAIGHT: 10,
    FLUSH: 1277,
    FULL_HOUSE: 156,
    FOUR_OF_A_KIND: 156,
    STRAIGHT_FLUSH: 10,
}

# Published seven-card totals over C(52,7) = 133,784,560 hands.
SEVEN_CARD_TOTALS = {
    HIGH_CARD: 23_294_460,
    ONE_PAIR: 58_627_800,
    TWO_PAIR: 31_433_400,
    THREE_OF_A_KIND: 6_461_620,
    STRAIGHT: 6_180_020,
    FLUSH: 4_047_644,
    FULL_HOUSE: 3_473_184,
    FOUR_OF_A_KIND: 224_848,
    STRAIGHT_FLUSH: 41_584,
}

# Cactus Kev equivalence-class numbers: 1 = royal flush, 7462 = 7-5-4-3-2 offsuit.
CACTUS_KEV = [
    ("As Ks Qs Js Ts", 1),
    ("5s 4s 3s 2s As", 10),
    ("As Ad Ah Ac Ks", 11),
    ("As Ad Ah Ac Qs", 12),
    ("Ks Kd Kh Kc As", 23),
    ("2s 2d 2h 2c 3s", 166),
    ("As Ad Ah Ks Kd", 167),
    ("2s 2d 2h 3s 3d", 322),
    ("As Ks Qs Js 9s", 323),
    ("7s 5s 4s 3s 2s", 1599),
    ("As Kd Qs Js Ts", 1600),
    ("5s 4d 3s 2s As", 1609),
    ("As Ad Ah Ks Qd", 1610),
    ("2s 2d 2h 4s 3d", 2467),
    ("As Ad Ks Kd Qh", 2468),
    ("3s 3d 2s 2d 4h", 3325),
    ("As Ad Ks Qd Jh", 3326),
    ("2s 2d 5s 4d 3h", 6185),
    ("As Kd Qs Js 9s", 6186),
    ("7s 5d 4s 3s 2s", 7462),
]


def hv(text):
    return rank5(parse_cards(text))


@pytest.mark.parametrize(
    "text, category, tiebreak",
    [
        ("As Ks Qs Js Ts", STRAIGHT_FLUSH, (14,)),
        ("5h 4h 3h 2h Ah", STRAIGHT_FLUSH, (5,)),
        ("9c 9d 9h 9s 2c", FOUR_OF_A_KIND, (9, 2)),
        ("Jc Jd Jh 4s 4c", FULL_HOUSE, (11, 4)),
        ("Kd 9d 7d 4d 2d", FLUSH, (13, 9, 7, 4, 2)),
        ("Tc 9d 8h 7s 6c", STRAIGHT, (10,)),
        ("7c 7d 7h Ks 2c", THREE_OF_A_KIND, (7, 13, 2)),
        ("Qc Qd 3h 3s Ac", TWO_PAIR, (12, 3, 14)),
        ("8c 8d Ah 5s 3c", ONE_PAIR, (8, 14, 5, 3)),
        ("Ac Jd 9h 5s 3c", HIGH_CARD, (14, 11, 9, 5, 3)),
    ],
)
def test_rank_tuples(text, category, tiebreak):
    value = hv(text)
    assert value.category == category
    assert value.tiebreak == tiebreak
    assert value.strength == STANDARD.strength[category]


def test_e01_wheel_loses_to_six_high_straight():
    wheel, six_high = hv("As 2d 3c 4h 5s"), hv("2c 3d 4s 5h 6d")
    assert wheel.category == six_high.category == STRAIGHT
    assert wheel.tiebreak == (5,) and six_high.tiebreak == (6,)
    assert six_high > wheel


def test_e01_ace_is_low_only_in_the_wheel():
    assert hv("Ks As 2d 3c 4h").category == HIGH_CARD
    assert hv("Qs Ks As 2d 3c").category == HIGH_CARD
    assert hv("As Kd Qc Jh Ts") > hv("Ks Qd Jc Th 9s")


def test_e02_royal_on_board_ties_everyone():
    board = parse_cards("As Ks Qs Js Ts")
    results = [best_nlhe(parse_cards(h), board) for h in ("2c 3d", "Ah Ad", "9s 8s", "Kh Qh")]
    assert {r.value for r in results} == {hv("As Ks Qs Js Ts")}
    assert all(r.cards == board for r in results)
    assert hv("Ah Kh Qh Jh Th") == hv("Ad Kd Qd Jd Td")


def test_e03_board_quads_best_fifth_card_decides():
    board = parse_cards("7c 7d 7h 7s 2c")
    a = best_nlhe(parse_cards("Kd 3h"), board).value
    b = best_nlhe(parse_cards("Qd Jh"), board).value
    c = best_nlhe(parse_cards("3s 4s"), board).value
    assert a.tiebreak == (7, 13) and b.tiebreak == (7, 12) and c.tiebreak == (7, 4)
    assert a > b > c
    tie_board = parse_cards("7c 7d 7h 7s Ac")
    assert best_nlhe(parse_cards("Kd 3h"), tie_board).value == best_nlhe(parse_cards("2d 3c"), tie_board).value


def test_e04_two_triplets_higher_trips_forms_full_house():
    value = best_nlhe(parse_cards("9s 9d"), parse_cards("9h 4c 4d 4s Kc")).value
    assert value.category == FULL_HOUSE and value.tiebreak == (9, 4)
    assert best_nlhe(parse_cards("Ks Kd"), parse_cards("Kh Qc Qd Qs 2c")).value.tiebreak == (13, 12)


def test_e05_six_suited_cards_best_five_and_kickers():
    board = parse_cards("Ah 9h 7h 4h 2h")
    a = best_nlhe(parse_cards("Kh 3c"), board)
    b = best_nlhe(parse_cards("Qh Jh"), board)
    assert a.value.category == b.value.category == FLUSH
    assert a.value.tiebreak == (14, 13, 9, 7, 4)
    assert b.value.tiebreak == (14, 12, 11, 9, 7)
    assert a.value > b.value
    assert best_nlhe(parse_cards("3h 5c"), board).value.tiebreak == (14, 9, 7, 4, 3)


def test_e06_impossible_hands_rejected():
    with pytest.raises(InvalidCards):
        rank5(parse_cards("As As Kd Qc Jh"))
    with pytest.raises(InvalidCards):
        best_nlhe(parse_cards("As Kd"), parse_cards("As 2c 3d 4h 5s"))
    with pytest.raises(InvalidCards):
        rank5(parse_cards("As Kd Qc Jh"))
    with pytest.raises(InvalidCards):
        best_nlhe(parse_cards("As Kd Qc"), parse_cards("2c 3d 4h 5s 7s"))


def test_seven_cards_checks_all_21_combinations():
    result = best_nlhe(parse_cards("Ah Kd"), parse_cards("2c 3d 4h 5s 9c"))
    assert result.combinations_checked == 21
    assert result.value.category == STRAIGHT and result.value.tiebreak == (5,)


def test_hole_cards_may_be_unused_or_both_used():
    board = parse_cards("Tc Jd Qh Ks 2c")
    assert best_nlhe(parse_cards("As 3d"), board).value.tiebreak == (14,)
    assert best_nlhe(parse_cards("9s 8d"), board).value.tiebreak == (13,)


def test_every_equivalence_class_is_distinct_and_ordered():
    values = sorted({rank5(h) for h in class_representatives(STANDARD)})
    assert len(values) == 7462
    assert dict(Counter(v.category for v in values)) == FIVE_CARD_CLASSES
    for text, expected in CACTUS_KEV:
        assert len(values) - values.index(hv(text)) == expected, text


def test_seven_card_category_totals():
    counts = seven_card_category_counts(STANDARD)
    assert counts == SEVEN_CARD_TOTALS
    assert sum(counts.values()) == 133_784_560


def test_sampled_seven_card_best_of_21_matches_counting_classifier():
    rng = random.Random(20261028)
    deck = standard_deck()
    for _ in range(3000):
        cards = rng.sample(deck, 7)
        assert best_of(cards, STANDARD).value.category == category_by_counting(cards, STANDARD)


@pytest.mark.release
def test_exhaustive_five_card_census():
    census = five_card_census(STANDARD)
    assert census.hands == 2_598_960
    assert census.category_counts == FIVE_CARD_TOTALS
    assert census.class_counts == FIVE_CARD_CLASSES
    assert len(census.values) == 7462
    for text, expected in CACTUS_KEV:
        assert census.class_index(hv(text)) == expected, text


@pytest.mark.release
def test_sampled_100k_seven_card_hands_match_counting_classifier():
    rng = random.Random(7)
    deck = standard_deck()
    for _ in range(100_000):
        cards = rng.sample(deck, 7)
        assert best_of(cards, STANDARD).value.category == category_by_counting(cards, STANDARD)
