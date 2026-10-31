"""Shared helpers for the oracle self-tests."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Sequence

from tests.oracles.cards import SUITS, Card
from tests.oracles.census import _best_rank_only_category
from tests.oracles.ranking import FLUSH, STRAIGHT_FLUSH, Ranking


def class_representatives(ranking: Ranking) -> Iterator[tuple[Card, ...]]:
    """One five-card hand per equivalence class: every rank multiset, plus a flush for distinct ranks."""
    ranks = ranking.ranks

    def multisets(start: int, left: int) -> Iterator[list[int]]:
        if left == 0:
            yield []
            return
        for i in range(start, len(ranks)):
            for rest in multisets(i, left - 1):
                yield [ranks[i], *rest]

    for ms in multisets(0, 5):
        counts = Counter(ms)
        if max(counts.values()) > 4:
            continue
        if len(counts) == 5:
            yield tuple(Card(r, SUITS[i % 2]) for i, r in enumerate(ms))
            yield tuple(Card(r, "s") for r in ms)
        else:
            yield tuple(Card(r, s) for r, k in counts.items() for s in SUITS[:k])


def category_by_counting(cards: Sequence[Card], ranking: Ranking) -> str:
    """Seven-card category the way ``seven_card_category_counts`` classifies it."""
    by_suit = Counter(c.suit for c in cards)
    suit, n = by_suit.most_common(1)[0]
    if n >= 5:
        suited = frozenset(c.rank for c in cards if c.suit == suit)
        return STRAIGHT_FLUSH if any(s <= suited for s in ranking.straights) else FLUSH
    counts = Counter(c.rank for c in cards)
    return _best_rank_only_category(tuple(counts.get(r, 0) for r in ranking.ranks), ranking.ranks, ranking)
