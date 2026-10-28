"""Pot-layer settlement oracle (spec section 4.3).

Input is a settlement ledger: each seat's total hand contribution, which seats
folded, a comparable showdown strength for every live contributor, and the
button seat. Pots are built from distinct contribution levels; a level with a
single contributor is unmatched excess and is returned to its owner. Folded
chips stay in every layer they reached, but folded seats are never eligible.

Two seams are parameters because the spec and PokerKit 0.7.7 differ:

- ``merge_equal_eligibility``: the spec builds one pot per contribution level
  (G15: "main 99, side 2"); PokerKit merges adjacent levels with identical
  eligible sets. Only odd-chip placement can differ.
- ``odd_chips``: the spec gives remainder chips one each to tied winners
  clockwise after the button (``"one_each_clockwise"``); PokerKit 0.7.7 gives
  the whole remainder to the first tied winner (``"all_to_first_clockwise"``).
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

OddChips = Literal["one_each_clockwise", "all_to_first_clockwise"]


@dataclass(frozen=True)
class PotResult:
    amount: int
    level_from: int
    level_to: int
    contributors: tuple[int, ...]
    eligible: tuple[int, ...]
    winners: tuple[int, ...]
    shares: Mapping[int, int]


@dataclass(frozen=True)
class Settlement:
    pots: tuple[PotResult, ...]
    refunds: Mapping[int, int]
    winnings: Mapping[int, int]
    totals: Mapping[int, int]

    @property
    def main_pot(self) -> PotResult:
        return self.pots[0]

    def final_stacks(self, starting_stacks: Mapping[int, int], contributions: Mapping[int, int]) -> dict[int, int]:
        return {s: starting_stacks[s] - contributions.get(s, 0) + self.totals.get(s, 0) for s in starting_stacks}


def _chip(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{what} must be a nonnegative integer chip amount, got {value!r}")
    return value


def clockwise_after(button: int, seats: Iterable[int]) -> tuple[int, ...]:
    seats = tuple(seats)
    ring = max((*seats, button)) + 1
    return tuple(sorted(seats, key=lambda s: (s - button - 1) % ring))


def settle(
    contributions: Mapping[int, int],
    folded: Iterable[int],
    strengths: Mapping[int, Hashable],
    button: int,
    *,
    merge_equal_eligibility: bool = False,
    odd_chips: OddChips = "one_each_clockwise",
) -> Settlement:
    folded = frozenset(folded)
    contributions = {seat: _chip(c, f"contribution of seat {seat}") for seat, c in contributions.items()}
    unknown = folded - contributions.keys()
    if unknown:
        raise ValueError(f"folded seats without a ledger entry: {sorted(unknown)}")
    live = [s for s, c in contributions.items() if c > 0 and s not in folded]
    missing = [s for s in live if s not in strengths]
    if missing:
        raise ValueError(f"live contributors without a showdown strength: {missing}")
    if odd_chips not in ("one_each_clockwise", "all_to_first_clockwise"):
        raise ValueError(f"unknown odd-chip rule {odd_chips!r}")

    layers: list[list[Any]] = []
    refunds: dict[int, int] = {}
    previous = 0
    for level in sorted({c for c in contributions.values() if c > 0}):
        contributors = tuple(sorted(s for s, c in contributions.items() if c >= level))
        amount = (level - previous) * len(contributors)
        if len(contributors) == 1:
            owner = contributors[0]
            refunds[owner] = refunds.get(owner, 0) + amount
        else:
            eligible = tuple(s for s in contributors if s not in folded)
            if not eligible:
                raise ValueError(f"layer ({previous}, {level}] has no eligible player")
            if merge_equal_eligibility and layers and layers[-1][4] == eligible:
                layers[-1][0] += amount
                layers[-1][2] = level
            else:
                layers.append([amount, previous, level, contributors, eligible])
        previous = level

    pots: list[PotResult] = []
    winnings: dict[int, int] = {}
    for amount, lo, hi, contributors, eligible in layers:
        best = max(strengths[s] for s in eligible)
        winners = clockwise_after(button, (s for s in eligible if strengths[s] == best))
        share, remainder = divmod(amount, len(winners))
        shares = {w: share for w in winners}
        if odd_chips == "one_each_clockwise":
            for w in winners[:remainder]:
                shares[w] += 1
        else:
            shares[winners[0]] += remainder
        for w, chips in shares.items():
            winnings[w] = winnings.get(w, 0) + chips
        pots.append(PotResult(amount, lo, hi, contributors, eligible, winners, shares))

    totals = {s: winnings.get(s, 0) + refunds.get(s, 0) for s in contributions}
    assert sum(totals.values()) == sum(contributions.values())
    return Settlement(tuple(pots), refunds, winnings, totals)
