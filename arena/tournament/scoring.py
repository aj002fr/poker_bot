"""Placements and ICM (spec section 4.4).

Local simultaneous-bust rule: among players eliminated on the same hand, the
larger stack at the start of that hand finishes higher; equal starting stacks
split the points of the finishing positions they jointly occupy. At the hand
cap only positive-stack survivors share the remaining top prizes by ICM.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache

SCORE_TOLERANCE = 1e-9


def icm(stacks: Sequence[int], prizes: Sequence[float]) -> list[float]:
    """Malmuth-Harville expected prize per player, by memoized recursion over subsets.

    ``prizes[k]`` is the prize for finishing position ``k + 1`` among these
    players; missing positions pay zero.
    """
    n = len(stacks)
    if n == 0:
        return []
    if any(type(s) is not int or s <= 0 for s in stacks):
        raise ValueError("ICM stacks must be positive integers")
    if n > 16:
        raise ValueError("too many players for exact ICM")
    prize = list(prizes)[:n] + [0.0] * max(0, n - len(prizes))
    full = (1 << n) - 1

    @lru_cache(maxsize=None)
    def expected(mask: int) -> tuple[float, ...]:
        """Expected prize per player for the players in ``mask`` competing for the next places."""
        out = [0.0] * n
        members = [i for i in range(n) if mask >> i & 1]
        place = n - len(members)
        if not members or place >= n:
            return tuple(out)
        total = sum(stacks[i] for i in members)
        for winner in members:
            p = stacks[winner] / total
            out[winner] += p * prize[place]
            rest = expected(mask & ~(1 << winner))
            for i in members:
                if i != winner:
                    out[i] += p * rest[i]
        return tuple(out)

    return list(expected(full))


@dataclass(frozen=True)
class Placement:
    seat: int
    positions: tuple[int, ...]
    points: float
    method: str  # "elimination" | "winner" | "icm"
    hand_number: int


class Standings:
    def __init__(self, payouts: Sequence[int], seats: Sequence[int]) -> None:
        if len(payouts) != len(seats):
            raise ValueError("payouts must have one entry per initial seat")
        self.payouts = [float(p) for p in payouts]
        self.seats = list(seats)
        self.placements: dict[int, Placement] = {}

    @property
    def remaining(self) -> list[int]:
        return [s for s in self.seats if s not in self.placements]

    def record_eliminations(
        self,
        hand_number: int,
        start_stacks: Mapping[int, int],
        end_stacks: Mapping[int, int],
    ) -> list[Placement]:
        """Award positions to seats that finished this hand with zero chips."""
        busted = [s for s in self.remaining if end_stacks.get(s, 0) == 0]
        if not busted:
            return []
        k = len(self.remaining)
        busted.sort(key=lambda s: -start_stacks[s])
        out: list[Placement] = []
        position = k - len(busted) + 1
        i = 0
        while i < len(busted):
            j = i
            while j < len(busted) and start_stacks[busted[j]] == start_stacks[busted[i]]:
                j += 1
            group = busted[i:j]
            group_positions = tuple(range(position, position + len(group)))
            share = sum(self.payouts[p - 1] for p in group_positions) / len(group)
            for seat in group:
                out.append(Placement(seat, group_positions, share, "elimination", hand_number))
            position += len(group)
            i = j
        for p in out:
            self.placements[p.seat] = p
        return out

    def finalize(self, hand_number: int, stacks: Mapping[int, int]) -> list[Placement]:
        """Score survivors: the sole survivor wins outright, otherwise ICM over the top prizes."""
        survivors = self.remaining
        if any(stacks[s] <= 0 for s in survivors):
            raise ValueError("record eliminations before finalizing")
        out: list[Placement]
        if len(survivors) == 1:
            out = [Placement(survivors[0], (1,), self.payouts[0], "winner", hand_number)]
        else:
            prizes = self.payouts[: len(survivors)]
            ev = icm([stacks[s] for s in survivors], prizes)
            positions = tuple(range(1, len(survivors) + 1))
            out = [Placement(s, positions, v, "icm", hand_number) for s, v in zip(survivors, ev)]
        for p in out:
            self.placements[p.seat] = p
        total = sum(p.points for p in self.placements.values())
        if abs(total - sum(self.payouts)) > SCORE_TOLERANCE * max(1.0, sum(self.payouts)):
            raise AssertionError(f"awarded {total} points, expected {sum(self.payouts)}")
        return out

    def points(self) -> dict[int, float]:
        return {s: self.placements[s].points for s in self.seats}
