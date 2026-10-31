import random
from collections import Counter
from math import comb

import pytest

from tests.oracles.cards import InvalidCards, parse_cards, short_deck
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
from tests.oracles.shortdeck import SPEC_ORDER, ShortDeckOrder, best_shortdeck, rank5, shortdeck_ranking
from tests.oracles.standard import best_nlhe
from tests.oracles.standard import rank5 as standard_rank5

SPEC = ShortDeckOrder.STRAIGHT_OVER_TRIPS
TRITON = ShortDeckOrder.TRIPS_OVER_STRAIGHT
BOTH = pytest.mark.parametrize("order", list(ShortDeckOrder), ids=lambda o: o.value)

# Hand-derived over 9 ranks x 4 suits, C(36,5) = 376,992. The 5-card category of
# a hand does not depend on the straight/trips order, so both orders share it.
FIVE_CARD_TOTALS = {
    STRAIGHT_FLUSH: 6 * 4,
    FOUR_OF_A_KIND: 9 * 32,
    FULL_HOUSE: 9 * 4 * 8 * 6,
    FLUSH: 4 * (comb(9, 5) - 6),
    STRAIGHT: 6 * (4**5 - 4),
    THREE_OF_A_KIND: 9 * 4 * comb(8, 2) * 16,
    TWO_PAIR: comb(9, 2) * 6 * 6 * 7 * 4,
    ONE_PAIR: 9 * 6 * comb(8, 3) * 4**3,
    HIGH_CARD: (comb(9, 5) - 6) * (4**5 - 4),
}

FIVE_CARD_CLASSES = {
    STRAIGHT_FLUSH: 6,
    FOUR_OF_A_KIND: 72,
    FULL_HOUSE: 72,
    FLUSH: 120,
    STRAIGHT: 6,
    THREE_OF_A_KIND: 252,
    TWO_PAIR: 252,
    ONE_PAIR: 504,
    HIGH_CARD: 120,
}

# Seven-card best-hand categories over C(36,7) = 8,347,680, from the counting
# method that reproduces the published standard-deck table exactly; cross-checked
# below against best-of-21 on sampled hands. Only straight and trips move: 30,360
# seven-card hands hold both (and nothing better).
SEVEN_CARD_TOTALS = {
    SPEC: {
        HIGH_CARD: 233_100,
        ONE_PAIR: 2_316_600,
        TWO_PAIR: 3_157_056,
        THREE_OF_A_KIND: 607_200,
        STRAIGHT: 1_169_940,
        FULL_HOUSE: 633_024,
        FLUSH: 175_560,
        FOUR_OF_A_KIND: 44_640,
        STRAIGHT_FLUSH: 10_560,
    },
    TRITON: {
        HIGH_CARD: 233_100,
        ONE_PAIR: 2_316_600,
        TWO_PAIR: 3_157_056,
        THREE_OF_A_KIND: 637_560,
        STRAIGHT: 1_139_580,
        FULL_HOUSE: 633_024,
        FLUSH: 175_560,
        FOUR_OF_A_KIND: 44_640,
        STRAIGHT_FLUSH: 10_560,
    },
}

# 1 = best class. Shared boundaries, then the two order-dependent blocks.
COMMON_BOUNDARIES = [
    ("As Ks Qs Js Ts", 1),
    ("Ah 6h 7h 8h 9h", 6),
    ("As Ad Ah Ac Ks", 7),
    ("6s 6d 6h 6c 7s", 78),
    ("As Ks Qs Js 9s", 79),
    ("Js 9s 8s 7s 6s", 198),
    ("As Ad Ah Ks Kd", 199),
    ("6s 6d 6h 7s 7d", 270),
    ("As Ad Ks Kd Qh", 529),
    ("7s 7d 6s 6d 8h", 780),
    ("As Ad Ks Qd Jh", 781),
    ("6s 6d 9s 8d 7h", 1284),
    ("As Kd Qs Js 9s", 1285),
    ("Js 9d 8s 7s 6s", 1404),
]
ORDER_BOUNDARIES = {
    SPEC: [("As Kd Qs Js Ts", 271), ("Ah 6d 7s 8s 9s", 276), ("As Ad Ah Ks Qd", 277), ("6s 6d 6h 8s 7d", 528)],
    TRITON: [("As Ad Ah Ks Qd", 271), ("6s 6d 6h 8s 7d", 522), ("As Kd Qs Js Ts", 523), ("Ah 6d 7s 8s 9s", 528)],
}


def hv(text, order=SPEC_ORDER):
    return rank5(parse_cards(text), order)


def test_spec_order_is_straight_over_trips():
    assert SPEC_ORDER is SPEC
    assert shortdeck_ranking().outranks(STRAIGHT, THREE_OF_A_KIND)
    assert shortdeck_ranking(TRITON).outranks(THREE_OF_A_KIND, STRAIGHT)


@BOTH
def test_order_parameter_accepts_value_strings(order):
    assert shortdeck_ranking(order.value) is shortdeck_ranking(order)


@BOTH
def test_e07_flush_beats_full_house_reversed_in_standard(order):
    flush, boat = "Ah Jh 9h 8h 6h", "Ks Kd Kh 7c 7d"
    assert hv(flush, order) > hv(boat, order)
    assert standard_rank5(parse_cards(boat)) > standard_rank5(parse_cards(flush))
    board = parse_cards("Kh Kd 7h 9h 6c")
    p1 = best_shortdeck(parse_cards("Ah Jh"), board, order).value
    p2 = best_shortdeck(parse_cards("Ks 7c"), board, order).value
    assert (p1.category, p2.category) == (FLUSH, FULL_HOUSE)
    assert p1 > p2
    assert best_nlhe(parse_cards("Ks 7c"), board).value > best_nlhe(parse_cards("Ah Jh"), board).value


@BOTH
def test_e08_ace_six_to_nine_is_lowest_straight(order):
    low = hv("Ah 6c 7d 8s 9h", order)
    assert low.category == STRAIGHT and low.tiebreak == (9,)
    assert hv("6c 7d 8s 9h Tc", order) > low
    assert hv("Ah 6h 7h 8h 9h", order).category == STRAIGHT_FLUSH
    assert hv("Ah 6h 7h 8h 9h", order) < hv("6h 7h 8h 9h Th", order)
    assert standard_rank5(parse_cards("Ah 6c 7d 8s 9h")).category == HIGH_CARD


def test_e08_straight_versus_trips_follows_declared_order():
    straight, trips = "Ah 6c 7d 8s 9h", "As Ad Ac Ks Qd"
    assert hv(straight, SPEC) > hv(trips, SPEC)
    assert hv(trips, TRITON) > hv(straight, TRITON)


def test_e08_seven_cards_with_both_straight_and_trips():
    hole, board = parse_cards("9s 9d"), parse_cards("9h 6c 7d 8s As")
    spec = best_shortdeck(hole, board, SPEC).value
    triton = best_shortdeck(hole, board, TRITON).value
    assert (spec.category, spec.tiebreak) == (STRAIGHT, (9,))
    assert (triton.category, triton.tiebreak) == (THREE_OF_A_KIND, (9, 14, 8))


@BOTH
@pytest.mark.parametrize("cards", ["As 2d 3c 4h 5s", "5c 6d 7h 8s 9c", "Ks Kd 2c 2d Ah"])
def test_e08_low_ranks_rejected(order, cards):
    with pytest.raises(InvalidCards):
        hv(cards, order)


@BOTH
def test_every_equivalence_class_is_distinct_and_ordered(order):
    ranking = shortdeck_ranking(order)
    values = sorted({rank5(h, order) for h in class_representatives(ranking)})
    assert len(values) == 1404
    assert dict(Counter(v.category for v in values)) == FIVE_CARD_CLASSES
    for text, expected in COMMON_BOUNDARIES + ORDER_BOUNDARIES[order]:
        assert len(values) - values.index(hv(text, order)) == expected, text


def test_hand_derived_five_card_totals_sum_to_deck():
    assert sum(FIVE_CARD_TOTALS.values()) == comb(36, 5) == 376_992


@BOTH
def test_seven_card_category_totals(order):
    counts = seven_card_category_counts(shortdeck_ranking(order))
    assert counts == SEVEN_CARD_TOTALS[order]
    assert sum(counts.values()) == comb(36, 7)


def test_seven_card_orders_differ_only_by_straight_and_trips():
    spec, triton = SEVEN_CARD_TOTALS[SPEC], SEVEN_CARD_TOTALS[TRITON]
    moved = spec[STRAIGHT] - triton[STRAIGHT]
    assert moved == triton[THREE_OF_A_KIND] - spec[THREE_OF_A_KIND] == 30_360
    assert {c for c in spec if spec[c] != triton[c]} == {STRAIGHT, THREE_OF_A_KIND}


@BOTH
def test_sampled_seven_card_best_of_21_matches_counting_classifier(order):
    ranking = shortdeck_ranking(order)
    rng = random.Random(36)
    deck = short_deck()
    for _ in range(3000):
        cards = rng.sample(deck, 7)
        assert best_of(cards, ranking).value.category == category_by_counting(cards, ranking)


@pytest.mark.release
@BOTH
def test_exhaustive_five_card_census(order):
    census = five_card_census(shortdeck_ranking(order))
    assert census.hands == 376_992
    assert census.category_counts == FIVE_CARD_TOTALS
    assert census.class_counts == FIVE_CARD_CLASSES
    for text, expected in COMMON_BOUNDARIES + ORDER_BOUNDARIES[order]:
        assert census.class_index(hv(text, order)) == expected, text
    by_category = {}
    for v in census.values:
        lo, hi = by_category.get(v.category, (v, v))
        by_category[v.category] = (min(lo, v), max(hi, v))
    if order is SPEC:
        assert by_category[STRAIGHT][0] > by_category[THREE_OF_A_KIND][1]
    else:
        assert by_category[THREE_OF_A_KIND][0] > by_category[STRAIGHT][1]
    assert by_category[FLUSH][0] > by_category[FULL_HOUSE][1]


@pytest.mark.release
@BOTH
def test_sampled_100k_seven_card_hands_match_counting_classifier(order):
    ranking = shortdeck_ranking(order)
    rng = random.Random(9)
    deck = short_deck()
    for _ in range(100_000):
        cards = rng.sample(deck, 7)
        assert best_of(cards, ranking).value.category == category_by_counting(cards, ranking)
