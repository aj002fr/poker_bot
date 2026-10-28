"""Small rules-adapter protocol used by the table service."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from arena.domain.models import Action, LegalActions

STREETS = ("preflop", "flop", "turn", "river")


class IllegalAction(ValueError):
    """The submitted action is not legal for the acting seat. State is unchanged."""


class EngineDiscrepancy(AssertionError):
    """PokerKit and the adapter's declared local rules disagree. Release blocker."""


@dataclass(frozen=True)
class HandEvent:
    type: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PotResult:
    amount: int
    eligible: tuple[int, ...]
    awards: dict[int, int]


@dataclass(frozen=True)
class PlayerPublic:
    seat: int
    stack: int
    street_committed: int
    hand_committed: int
    status: str  # "active" | "folded" | "all_in"


@dataclass(frozen=True)
class HandResult:
    starting_stacks: dict[int, int]
    final_stacks: dict[int, int]
    contributions: dict[int, int]
    pots: tuple[PotResult, ...]
    board: tuple[str, ...]
    revealed: dict[int, tuple[str, ...]]
    folded: tuple[int, ...]
    ended_by_fold: bool


class RulesHand(Protocol):
    @property
    def actor_seat(self) -> int | None: ...

    @property
    def is_complete(self) -> bool: ...

    def legal_actions(self) -> LegalActions: ...

    def apply(self, action: Action) -> list[HandEvent]: ...

    def result(self) -> HandResult: ...
