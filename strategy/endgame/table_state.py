"""From physical seats to the subgame's preflop action order (decision-time safe).

The arena uses a dead button: the button and small-blind markers may sit on busted
seats, the big blind is always live (``arena.table.positions``). Heads-up, the button
posts the small blind and acts first preflop.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

from strategy.endgame.keys import Snapshot


def _clockwise_after(seat: int, ring: Sequence[int]) -> list[int]:
    i = ring.index(seat)
    return list(ring[i + 1:]) + list(ring[:i + 1])


def preflop_order(ring: Sequence[int], live: Collection[int], button: int, big_blind: int,
                  heads_up: bool) -> list[int]:
    """Live seats in preflop action order, big blind last."""
    if heads_up:
        return [button, big_blind]
    return [s for s in _clockwise_after(big_blind, ring) if s in live]


def odd_chip_order(ring: Sequence[int], live: Collection[int], button: int) -> list[int]:
    """Live seats clockwise from the first seat after the button (the arena's odd-chip order)."""
    return [s for s in _clockwise_after(button, ring) if s in live]


@dataclass(frozen=True)
class SeatedSnapshot:
    snapshot: Snapshot
    seats: tuple[int, ...]
    """Physical seat of each player index (preflop action order)."""
    odd_order: tuple[int, ...]
    """Player indices in odd-chip order."""

    def index(self, seat: int) -> int:
        return self.seats.index(seat)


def seated_snapshot(*, ring: Sequence[int], start_stacks: Mapping[int, int], button: int, small_blind_seat: int | None,
                    big_blind: int, heads_up: bool, blinds: tuple[int, int], payouts: Sequence[float],
                    hand_number: int, hand_cap: int) -> SeatedSnapshot:
    live = [s for s in ring if start_stacks.get(s, 0) > 0]
    order = preflop_order(ring, live, button, big_blind, heads_up)
    sb, bb = blinds
    sb_seat = button if heads_up else small_blind_seat
    posts = tuple(min(sb, start_stacks[s]) if s == sb_seat and s != big_blind else
                  min(bb, start_stacks[s]) if s == big_blind else 0 for s in order)
    snap = Snapshot(tuple(int(start_stacks[s]) for s in order), posts, bb, tuple(float(p) for p in payouts),
                    hand_number, hand_cap)
    odd = tuple(order.index(s) for s in odd_chip_order(ring, live, button))
    return SeatedSnapshot(snap, tuple(order), odd)
