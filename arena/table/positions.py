"""Button and blind progression across hands on a closed table (spec section 4.1).

Markers are physical seat ids. The button and small-blind markers may rest on
seats that have busted (dead button / dead small blind); the big blind is
always a live seat.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass

from arena.domain.rng import Rng, choice


@dataclass(frozen=True)
class Positions:
    button: int
    small_blind: int
    big_blind: int
    heads_up: bool

    def small_blind_seat(self, live: Collection[int]) -> int | None:
        """The seat that posts the small blind, or None when the SB is dead."""
        return self.small_blind if self.small_blind in live else None

    def as_dict(self) -> dict[str, int | bool]:
        return {
            "button": self.button,
            "small_blind": self.small_blind,
            "big_blind": self.big_blind,
            "heads_up": self.heads_up,
        }


def _clockwise_after(seat: int, ring: Sequence[int]) -> list[int]:
    i = ring.index(seat)
    return list(ring[i + 1 :]) + list(ring[: i + 1])


def first_live_after(seat: int, ring: Sequence[int], live: Collection[int]) -> int:
    for s in _clockwise_after(seat, ring):
        if s in live:
            return s
    raise ValueError("no live seat")


def initial_positions(ring: Sequence[int], live: Collection[int], rng: Rng) -> Positions:
    occupied = [s for s in ring if s in live]
    if len(occupied) < 2:
        raise ValueError("need at least two players")
    button = choice(occupied, rng)
    return positions_from_button(ring, live, button)


def positions_from_button(ring: Sequence[int], live: Collection[int], button: int) -> Positions:
    if button not in live:
        raise ValueError("initial button must be an occupied seat")
    if len([s for s in ring if s in live]) == 2:
        other = first_live_after(button, ring, live)
        return Positions(button=button, small_blind=button, big_blind=other, heads_up=True)
    sb = first_live_after(button, ring, live)
    bb = first_live_after(sb, ring, live)
    return Positions(button=button, small_blind=sb, big_blind=bb, heads_up=False)


def next_positions(prev: Positions, ring: Sequence[int], survivors: Collection[int]) -> Positions:
    live = [s for s in ring if s in survivors]
    if len(live) < 2:
        raise ValueError("table is finished")
    if len(live) == 2:
        if prev.heads_up:
            if {prev.button, prev.big_blind} != set(live):
                raise ValueError("heads-up survivors changed without an elimination")
            return Positions(button=prev.big_blind, small_blind=prev.big_blind, big_blind=prev.button, heads_up=True)
        bb = first_live_after(prev.big_blind, ring, live)
        button = next(s for s in live if s != bb)
        return Positions(button=button, small_blind=button, big_blind=bb, heads_up=True)
    if prev.heads_up:
        raise ValueError("a closed table cannot grow from heads-up")
    bb = first_live_after(prev.big_blind, ring, live)
    pos = Positions(button=prev.small_blind, small_blind=prev.big_blind, big_blind=bb, heads_up=False)
    check_no_live_seat_skipped(pos, ring, live)
    return pos


def check_no_live_seat_skipped(pos: Positions, ring: Sequence[int], live: Collection[int]) -> None:
    """Assert no live seat sits strictly between the SB marker and the BB."""
    for s in _clockwise_after(pos.small_blind, ring):
        if s == pos.big_blind:
            return
        if s in live:
            raise AssertionError(f"live seat {s} between SB marker {pos.small_blind} and BB {pos.big_blind}")
    raise AssertionError("big blind not found clockwise of the SB marker")
