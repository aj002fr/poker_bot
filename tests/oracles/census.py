"""Combinatorial checksums: exhaustive five-card census and seven-card category counts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import combinations
from math import comb, prod

from tests.oracles.cards import make_deck
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
    HandValue,
    Ranking,
    rank_five_unchecked,
)


@dataclass(frozen=True)
class FiveCardCensus:
    hands: int
    category_counts: dict[str, int]
    class_counts: dict[str, int]
    values: tuple[HandValue, ...]

    def class_index(self, value: HandValue) -> int:
        """1 = best class (Cactus Kev numbering), len(values) = worst."""
        return len(self.values) - self.values.index(value)


def five_card_census(ranking: Ranking) -> FiveCardCensus:
    """Evaluate every five-card hand of the declared deck."""
    deck = make_deck(ranking.ranks)
    categories: Counter[str] = Counter()
    values: set[HandValue] = set()
    hands = 0
    for combo in combinations(deck, 5):
        value = rank_five_unchecked(combo, ranking)
        categories[value.category] += 1
        values.add(value)
        hands += 1
    classes = Counter(v.category for v in values)
    return FiveCardCensus(hands, dict(categories), dict(classes), tuple(sorted(values)))


def _rank_vectors(total: int, slots: int, cap: int = 4) -> Iterator[tuple[int, ...]]:
    if slots == 0:
        if total == 0:
            yield ()
        return
    for k in range(min(cap, total) + 1):
        for rest in _rank_vectors(total - k, slots - 1, cap):
            yield (k, *rest)


def _ways_one_suit_on_at_least(multiplicities: tuple[int, ...], threshold: int) -> int:
    """Suit assignments in which one fixed suit appears on >= ``threshold`` ranks."""
    poly = [1]
    for k in multiplicities:
        if not k:
            continue
        without, with_ = comb(3, k), comb(3, k - 1)
        nxt = [0] * (len(poly) + 1)
        for d, coef in enumerate(poly):
            nxt[d] += coef * without
            nxt[d + 1] += coef * with_
        poly = nxt
    return sum(poly[threshold:])


def _best_rank_only_category(multiplicities: tuple[int, ...], ranks: tuple[int, ...], ranking: Ranking) -> str:
    present = frozenset(r for r, k in zip(ranks, multiplicities) if k)
    trips_or_more = sum(1 for k in multiplicities if k >= 3)
    pairs_or_more = sum(1 for k in multiplicities if k >= 2)
    made = [HIGH_CARD]
    if pairs_or_more >= 1:
        made.append(ONE_PAIR)
    if pairs_or_more >= 2:
        made.append(TWO_PAIR)
    if trips_or_more:
        made.append(THREE_OF_A_KIND)
    if any(s <= present for s in ranking.straights):
        made.append(STRAIGHT)
    if trips_or_more >= 2 or (trips_or_more and pairs_or_more >= 2):
        made.append(FULL_HOUSE)
    if max(multiplicities) >= 4:
        made.append(FOUR_OF_A_KIND)
    return max(made, key=ranking.strength.__getitem__)


def seven_card_category_counts(ranking: Ranking) -> dict[str, int]:
    """Best-hand category of every seven-card set, counted without enumerating hands.

    Non-flush hands are classified from their rank multiset alone and weighted
    by the number of suit assignments without five of one suit. Flush hands
    are classified from the flush suit's rank set: with five or more cards in
    one suit at most two cards lie elsewhere, so quads and full houses cannot
    co-occur and every other made category ranks below a flush in all the
    profiles this supports.
    """
    for weaker in (THREE_OF_A_KIND, STRAIGHT, TWO_PAIR, ONE_PAIR, HIGH_CARD):
        assert ranking.outranks(FLUSH, weaker), (ranking.name, weaker)
    ranks = ranking.ranks
    counts = Counter({c: 0 for c in ranking.order})
    flush_from_multisets = 0
    for m in _rank_vectors(7, len(ranks)):
        assignments = prod(comb(4, k) for k in m)
        flushes = 4 * _ways_one_suit_on_at_least(m, 5)
        flush_from_multisets += flushes
        if assignments > flushes:
            counts[_best_rank_only_category(m, ranks, ranking)] += assignments - flushes
    flush_direct = 0
    for k in (5, 6, 7):
        others = comb(3 * len(ranks), 7 - k)
        for suited in combinations(ranks, k):
            suited_set = frozenset(suited)
            sf = any(s <= suited_set for s in ranking.straights)
            counts[STRAIGHT_FLUSH if sf else FLUSH] += 4 * others
            flush_direct += 4 * others
    assert flush_direct == flush_from_multisets, (flush_direct, flush_from_multisets)
    assert sum(counts.values()) == comb(4 * len(ranks), 7)
    return dict(counts)
