"""ICM permutation oracle and tournament point allocation (spec section 4.4).

``icm_permutations`` walks every complete finishing order (k! leaves, 40,320
at eight players). The probability of an order is the product of each
player's stack over the sum of stacks not yet placed. This deliberately
differs from the production algorithm (memoized recursion over subsets).
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

MAX_PLAYERS = 8


@dataclass(frozen=True)
class IcmResult:
    points: dict[Hashable, Fraction | float]
    orders: int


def _positive_int(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{what} must be a positive integer, got {value!r}")
    return value


def icm_permutations(stacks: Mapping[Hashable, int], prizes: Sequence[int | Fraction], *, exact: bool = True) -> IcmResult:
    """Expected prize per player; ``prizes[i]`` goes to finishing position ``i + 1``.

    ``exact=True`` uses rationals; ``exact=False`` uses doubles.
    """
    players = list(stacks)
    if not 1 <= len(players) <= MAX_PLAYERS:
        raise ValueError(f"need 1..{MAX_PLAYERS} survivors, got {len(players)}")
    chips = [_positive_int(stacks[p], f"stack of {p!r}") for p in players]
    if len(prizes) < len(players):
        raise ValueError("fewer prizes than survivors")
    num = Fraction if exact else float
    prize = [num(x) for x in prizes[: len(players)]]
    zero = num(0)
    totals = [zero] * len(players)
    orders = 0

    def walk(remaining: list[int], remaining_chips: int, probability: Any, prefix: list[int]) -> None:
        nonlocal orders
        if not remaining:
            orders += 1
            for position, i in enumerate(prefix):
                totals[i] += probability * prize[position]
            return
        for j, i in enumerate(remaining):
            p = probability * num(chips[i]) / num(remaining_chips)
            prefix.append(i)
            walk(remaining[:j] + remaining[j + 1 :], remaining_chips - chips[i], p, prefix)
            prefix.pop()

    walk(list(range(len(players))), sum(chips), num(1), [])
    return IcmResult({p: totals[i] for i, p in enumerate(players)}, orders)


@dataclass(frozen=True)
class Bust:
    player: Hashable
    start_stack: int


def table_points(
    payout: Sequence[int],
    bust_hands: Sequence[Sequence[Bust]],
    survivors: Mapping[Hashable, int],
    *,
    exact: bool = True,
) -> dict[Hashable, Fraction | float]:
    """Final points for every initial player.

    ``bust_hands`` lists, in hand order, the players eliminated on each hand
    together with their stacks at the start of that hand. Busts on one hand
    rank by start stack (larger finishes higher); equal start stacks share the
    points of the positions they jointly occupy. Every bust ranks below every
    survivor. Surviving positive stacks split the remaining top prizes by ICM,
    or take first place outright if only one survives.
    """
    n = len(payout)
    num = Fraction if exact else float
    busted = [b.player for hand in bust_hands for b in hand]
    if len(set(busted)) != len(busted) or set(busted) & set(survivors):
        raise ValueError("each player busts at most once and never also survives")
    if len(busted) + len(survivors) != n:
        raise ValueError(f"payout has {n} positions for {len(busted) + len(survivors)} players")
    if not survivors:
        raise ValueError("at least one player must survive")

    points: dict[Hashable, Any] = {}
    next_position = n  # 1-based; earliest busts take the lowest positions
    for hand in bust_hands:
        by_stack: dict[int, list[Hashable]] = {}
        for b in hand:
            by_stack.setdefault(_positive_int(b.start_stack, f"start stack of {b.player!r}"), []).append(b.player)
        occupied = next_position - len(hand) + 1
        position = occupied
        for stack in sorted(by_stack, reverse=True):
            group = by_stack[stack]
            share = num(sum(payout[position - 1 : position - 1 + len(group)])) / len(group)
            for player in group:
                points[player] = share
            position += len(group)
        next_position = occupied - 1

    assert next_position == len(survivors)
    if len(survivors) == 1:
        (winner,) = survivors
        points[winner] = num(payout[0])
    else:
        points.update(icm_permutations(survivors, payout[: len(survivors)], exact=exact).points)
    return points
