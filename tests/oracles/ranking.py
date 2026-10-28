"""Shared five-card ranking core, driven by an explicitly declared ``Ranking``.

Each ruleset declares its own deck ranks, straight table (including any
ace-low straight) and category order low-to-high. Category *membership* of a
five-card hand depends only on its rank pattern and suits; *strength* comes
from the declared order, so the same pattern code serves every profile
without any profile inheriting another's order or straights.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import NamedTuple

from tests.oracles.cards import Card, InvalidCards, validate_cards

HIGH_CARD = "high_card"
ONE_PAIR = "one_pair"
TWO_PAIR = "two_pair"
THREE_OF_A_KIND = "three_of_a_kind"
STRAIGHT = "straight"
FLUSH = "flush"
FULL_HOUSE = "full_house"
FOUR_OF_A_KIND = "four_of_a_kind"
STRAIGHT_FLUSH = "straight_flush"

CATEGORIES = frozenset(
    {
        HIGH_CARD,
        ONE_PAIR,
        TWO_PAIR,
        THREE_OF_A_KIND,
        STRAIGHT,
        FLUSH,
        FULL_HOUSE,
        FOUR_OF_A_KIND,
        STRAIGHT_FLUSH,
    }
)


class HandValue(NamedTuple):
    """Comparable hand value: higher ``strength`` wins, then ``tiebreak`` lexicographically.

    ``tiebreak`` holds ranks (2..14) in significance order: the straight's top
    card for straights (5 for the standard wheel, 9 for short-deck A-6-7-8-9),
    otherwise ranks grouped by multiplicity then by rank, e.g. two pair
    ``(high_pair, low_pair, kicker)``. Suits never appear.
    """

    strength: int
    tiebreak: tuple[int, ...]
    category: str


@dataclass(frozen=True)
class Ranking:
    name: str
    ranks: tuple[int, ...]
    straights: Mapping[frozenset[int], int]
    order: tuple[str, ...]
    strength: Mapping[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if set(self.order) != CATEGORIES or len(self.order) != len(CATEGORIES):
            raise ValueError(f"{self.name}: order must list each category once")
        for ranks, high in self.straights.items():
            if len(ranks) != 5 or not ranks <= set(self.ranks) or high not in ranks:
                raise ValueError(f"{self.name}: bad straight {sorted(ranks)}")
        object.__setattr__(self, "strength", {c: i for i, c in enumerate(self.order)})

    def outranks(self, better: str, worse: str) -> bool:
        return self.strength[better] > self.strength[worse]


_PATTERN_CATEGORY = {
    (4, 1): FOUR_OF_A_KIND,
    (3, 2): FULL_HOUSE,
    (3, 1, 1): THREE_OF_A_KIND,
    (2, 2, 1): TWO_PAIR,
    (2, 1, 1, 1): ONE_PAIR,
}


def classify_five(cards: Sequence[Card], ranking: Ranking) -> tuple[str, tuple[int, ...]]:
    """Category and tiebreak of exactly five already-validated cards."""
    counts: dict[int, int] = {}
    for card in cards:
        counts[card.rank] = counts.get(card.rank, 0) + 1
    groups = sorted(counts.items(), key=lambda rc: (rc[1], rc[0]), reverse=True)
    ordered = tuple(r for r, _ in groups)
    s0 = cards[0].suit
    is_flush = cards[1].suit == s0 and cards[2].suit == s0 and cards[3].suit == s0 and cards[4].suit == s0
    if len(counts) == 5:
        high = ranking.straights.get(frozenset(counts))
        if high is not None:
            return (STRAIGHT_FLUSH if is_flush else STRAIGHT), (high,)
        return (FLUSH if is_flush else HIGH_CARD), ordered
    return _PATTERN_CATEGORY[tuple(c for _, c in groups)], ordered


def rank_five_unchecked(cards: Sequence[Card], ranking: Ranking) -> HandValue:
    category, tiebreak = classify_five(cards, ranking)
    return HandValue(ranking.strength[category], tiebreak, category)


def rank_five(cards: Iterable[Card], ranking: Ranking) -> HandValue:
    cards = validate_cards(cards, ranking.ranks)
    if len(cards) != 5:
        raise InvalidCards(f"need exactly 5 cards, got {len(cards)}")
    return rank_five_unchecked(cards, ranking)


class BestHand(NamedTuple):
    value: HandValue
    cards: tuple[Card, ...]
    combinations_checked: int


def best_of(cards: Iterable[Card], ranking: Ranking) -> BestHand:
    """Best five of any ``n >= 5`` cards by exhaustive ``C(n, 5)`` enumeration."""
    cards = validate_cards(cards, ranking.ranks)
    if len(cards) < 5:
        raise InvalidCards(f"need at least 5 cards, got {len(cards)}")
    best: tuple[HandValue, tuple[Card, ...]] | None = None
    checked = 0
    for combo in combinations(cards, 5):
        checked += 1
        value = rank_five_unchecked(combo, ranking)
        if best is None or value > best[0]:
            best = (value, combo)
    assert best is not None
    return BestHand(best[0], best[1], checked)


def best_holdem(hole: Iterable[Card], board: Iterable[Card], ranking: Ranking) -> BestHand:
    """Hold'em: best five from two hole cards plus the board, using 0, 1 or 2 hole cards."""
    hole = tuple(hole)
    board = tuple(board)
    if len(hole) != 2:
        raise InvalidCards(f"Hold'em needs 2 hole cards, got {len(hole)}")
    if len(board) not in (3, 4, 5):
        raise InvalidCards(f"board must have 3-5 cards, got {len(board)}")
    return best_of(hole + board, ranking)
