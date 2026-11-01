"""Variant metadata and capabilities (decision-time safe: no arena engine imports).

A :class:`VariantSpec` describes a poker variant as data: deck, deal, showdown
selection, hand-ranking order and betting structure. Strategy code asks for
capabilities instead of switching on ruleset ids, so a surprise variant either
declares what it supports or fails loudly; Hold'em assets are never reused
silently. :mod:`strategy.rules.conformance` checks these declarations against the
arena engine offline.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from strategy.rules.canonical import canonical_hash

VARIANT_SCHEMA_VERSION = 1

Betting = Literal["no_limit", "pot_limit"]
HandSelection = Literal["best_five_of_seven", "two_hole_three_board"]

STANDARD_CATEGORY_ORDER = (
    "high_card", "one_pair", "two_pair", "three_of_a_kind", "straight", "flush", "full_house",
    "four_of_a_kind", "straight_flush",
)
SHORT_DECK_CATEGORY_ORDER = (
    "high_card", "one_pair", "two_pair", "three_of_a_kind", "straight", "full_house", "flush",
    "four_of_a_kind", "straight_flush",
)

# Capability names. Strategy assets declare the capabilities they need.
CAP_NO_LIMIT = "betting.no_limit"
CAP_POT_LIMIT = "betting.pot_limit"
CAP_COMMUNITY_BOARD = "deal.community_board"
CAP_TWO_HOLE = "deal.two_hole_cards"
CAP_FOUR_HOLE = "deal.four_hole_cards"
CAP_ANY_FIVE = "showdown.best_five_any"
CAP_EXACTLY_TWO_HOLE = "showdown.exactly_two_hole"
CAP_DECK_52 = "deck.standard_52"
CAP_DECK_36 = "deck.short_36"
CAP_STANDARD_RANKING = "ranking.standard_high"
CAP_FLUSH_OVER_FULL_HOUSE = "ranking.flush_over_full_house"
CAP_NLHE_ASSETS = "assets.nlhe"
"""Hold'em charts, tables and priors (Pluribus, Harrington) apply only to exact NLHE."""


class UnsupportedVariant(ValueError):
    """The variant is unknown, or lacks a capability an asset requires."""


@dataclass(frozen=True)
class VariantSpec:
    ruleset_id: str
    title: str
    family: str
    ranks: str
    suits: str
    hole_cards: int
    board_streets: tuple[int, ...]
    """Board cards dealt on each postflop street."""
    burn_before_board: bool
    hand_selection: HandSelection
    category_order: tuple[str, ...]
    """Five-card categories, weakest first."""
    lowest_straight: str
    """Ranks of the lowest straight, ace first when it plays low (``A2345``)."""
    betting: Betting
    streets: tuple[str, ...] = ("preflop", "flop", "turn", "river")
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.board_streets) != len(self.streets) - 1:
            raise ValueError("board_streets needs one entry per postflop street")
        if len(set(self.ranks)) != len(self.ranks) or len(set(self.suits)) != len(self.suits):
            raise ValueError("ranks and suits must be distinct")

    @property
    def deck_size(self) -> int:
        return len(self.ranks) * len(self.suits)

    @property
    def board_cards(self) -> int:
        return sum(self.board_streets)

    @property
    def capabilities(self) -> frozenset[str]:
        caps = {CAP_NO_LIMIT if self.betting == "no_limit" else CAP_POT_LIMIT}
        if self.board_cards:
            caps.add(CAP_COMMUNITY_BOARD)
        if self.hole_cards == 2:
            caps.add(CAP_TWO_HOLE)
        if self.hole_cards == 4:
            caps.add(CAP_FOUR_HOLE)
        caps.add(CAP_ANY_FIVE if self.hand_selection == "best_five_of_seven" else CAP_EXACTLY_TWO_HOLE)
        if self.ranks == "23456789TJQKA" and self.suits == "cdhs":
            caps.add(CAP_DECK_52)
        if self.ranks == "6789TJQKA" and self.suits == "cdhs":
            caps.add(CAP_DECK_36)
        if self.category_order == STANDARD_CATEGORY_ORDER:
            caps.add(CAP_STANDARD_RANKING)
        if self.category_order.index("flush") > self.category_order.index("full_house"):
            caps.add(CAP_FLUSH_OVER_FULL_HOUSE)
        if caps >= {CAP_NO_LIMIT, CAP_TWO_HOLE, CAP_ANY_FIVE, CAP_DECK_52, CAP_STANDARD_RANKING} \
                and self.board_streets == (3, 1, 1):
            caps.add(CAP_NLHE_ASSETS)
        return frozenset(caps)

    def supports(self, *capabilities: str) -> bool:
        return set(capabilities) <= self.capabilities

    def require(self, *capabilities: str, asset: str = "asset") -> None:
        missing = sorted(set(capabilities) - self.capabilities)
        if missing:
            raise UnsupportedVariant(f"{asset} needs {', '.join(missing)}; {self.ruleset_id} lacks them")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": VARIANT_SCHEMA_VERSION,
            "ruleset_id": self.ruleset_id,
            "title": self.title,
            "family": self.family,
            "ranks": self.ranks,
            "suits": self.suits,
            "deck_size": self.deck_size,
            "hole_cards": self.hole_cards,
            "streets": list(self.streets),
            "board_streets": list(self.board_streets),
            "burn_before_board": self.burn_before_board,
            "hand_selection": self.hand_selection,
            "category_order": list(self.category_order),
            "lowest_straight": self.lowest_straight,
            "betting": self.betting,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> VariantSpec:
        """Build a spec for a revealed variant, e.g. from an official starter kit."""
        if data.get("schema_version", VARIANT_SCHEMA_VERSION) != VARIANT_SCHEMA_VERSION:
            raise UnsupportedVariant(f"variant schema_version {data.get('schema_version')!r}")
        spec = cls(
            ruleset_id=str(data["ruleset_id"]), title=str(data["title"]), family=str(data["family"]),
            ranks=str(data["ranks"]), suits=str(data["suits"]), hole_cards=int(data["hole_cards"]),
            streets=tuple(data.get("streets", ("preflop", "flop", "turn", "river"))),
            board_streets=tuple(int(x) for x in data["board_streets"]),
            burn_before_board=bool(data["burn_before_board"]), hand_selection=data["hand_selection"],
            category_order=tuple(data["category_order"]), lowest_straight=str(data["lowest_straight"]),
            betting=data["betting"], extra=dict(data.get("extra", {})),
        )
        if "deck_size" in data and int(data["deck_size"]) != spec.deck_size:
            raise UnsupportedVariant("deck_size disagrees with ranks x suits")
        return spec

    def variant_hash(self) -> str:
        """Identifies the game itself, independent of the tournament structure."""
        return canonical_hash(self.as_dict(), domain="strategy-variant-v1")


KNOWN_VARIANTS: dict[str, VariantSpec] = {
    "practice_nlhe": VariantSpec(
        ruleset_id="practice_nlhe", title="No-limit Texas Hold'em", family="holdem",
        ranks="23456789TJQKA", suits="cdhs", hole_cards=2, board_streets=(3, 1, 1), burn_before_board=True,
        hand_selection="best_five_of_seven", category_order=STANDARD_CATEGORY_ORDER, lowest_straight="A2345",
        betting="no_limit",
    ),
    "uncertainty_shortdeck": VariantSpec(
        ruleset_id="uncertainty_shortdeck", title="Short-deck Hold'em", family="holdem",
        ranks="6789TJQKA", suits="cdhs", hole_cards=2, board_streets=(3, 1, 1), burn_before_board=True,
        hand_selection="best_five_of_seven", category_order=SHORT_DECK_CATEGORY_ORDER, lowest_straight="A6789",
        betting="no_limit",
    ),
    "uncertainty_plo4": VariantSpec(
        ruleset_id="uncertainty_plo4", title="Four-card pot-limit Omaha", family="omaha",
        ranks="23456789TJQKA", suits="cdhs", hole_cards=4, board_streets=(3, 1, 1), burn_before_board=True,
        hand_selection="two_hole_three_board", category_order=STANDARD_CATEGORY_ORDER, lowest_straight="A2345",
        betting="pot_limit",
    ),
}


def get_variant(ruleset_id: str) -> VariantSpec:
    try:
        return KNOWN_VARIANTS[ruleset_id]
    except KeyError:
        raise UnsupportedVariant(
            f"unknown variant {ruleset_id!r}; describe it with VariantSpec.from_dict before use") from None
