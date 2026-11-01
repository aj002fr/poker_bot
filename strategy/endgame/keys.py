"""Abstract decision keys for endgame tables, shared by the generator and the runtime.

Decision-time safe: pure Python, no numpy. The generator computes a key from the
subgame it solved and the bot computes it from its observation through the same
functions, so the two cannot drift.

A :class:`Snapshot` is the hand at its start in preflop action order (big blind
last). Three decision kinds are keyed:

``open``      everyone before the hero folded; the hero may shove.
``call``      exactly one player shoved and everyone else before the hero folded.
``overcall``  one shove and one call before the hero.

Each kind has a ladder of keys from fine to coarse; the table stores every level and
the lookup takes the finest level with enough weight. Fields:

* ``n`` live players, ``behind`` players still to act after the hero, ``shover_behind``
  players who were behind the shover when it shoved (its position);
* ``depth``: chips at risk in big blinds (hero vs the largest stack behind for
  ``open``; ``min(hero, shover)`` for ``call``), bucketed by :data:`DEPTH_EDGES`;
* ``covered``: the hero can be eliminated by that confrontation;
* ``pressure``: the hero's ICM risk premium in that all-in confrontation,
  ``q_icm - q_chip``, where ``q`` is the equity needed to break even against folding
  (:func:`pressure`), bucketed by :data:`PRESSURE_EDGES`;
* ``horizon``: 0 on the hand-cap hand, 1 when the cap is less than an orbit away, else 2.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass

DEPTH_EDGES = (1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 20, 23, 26, 30)
PRESSURE_EDGES = (0.01, 0.03, 0.06, 0.10, 0.15, 0.22)
MAX_DEPTH_BB = 30.0
"""Tables cover decisions with at most this many big blinds at risk."""
KINDS = ("open", "call", "overcall")
KIND_CODE = {k: i for i, k in enumerate(KINDS)}
LEVELS = 4


@dataclass(frozen=True)
class Snapshot:
    stacks: tuple[int, ...]
    """Chips at the start of the hand (before posting), preflop action order."""
    posts: tuple[int, ...]
    bb: int
    payouts: tuple[float, ...]
    hand_number: int
    hand_cap: int

    @property
    def n(self) -> int:
        return len(self.stacks)

    @property
    def prizes(self) -> tuple[float, ...]:
        return tuple(self.payouts[: self.n])


# --------------------------------------------------------------------------- ICM


def icm_player(stacks: Sequence[float], hero: int, prizes: Sequence[float]) -> float:
    """Exact Malmuth-Harville ICM value of one player (zero stacks are out of the race).

    Cost is about ``n ** (paid - 1)`` terms, where ``paid`` counts the nonzero prizes.
    """
    live = [i for i, s in enumerate(stacks) if s > 0]
    if stacks[hero] <= 0:
        return 0.0
    k = len(live)
    prize = list(prizes[:k]) + [0.0] * max(0, k - len(prizes))
    paid = max((i + 1 for i, p in enumerate(prize) if p), default=0)
    total = float(sum(stacks[i] for i in live))
    sh = float(stacks[hero])
    others = [float(stacks[i]) for i in live if i != hero]

    def rec(place: int, removed: float, used: int, prob: float) -> float:
        rest = total - removed
        value = prob * sh / rest * prize[place]
        if place + 1 < paid:
            for j, sj in enumerate(others):
                if not used >> j & 1:
                    value += rec(place + 1, removed + sj, used | 1 << j, prob * sj / rest)
        return value

    return rec(0, 0.0, 0, 1.0) if paid else 0.0


def pressure(stacks: Sequence[int], hero: int, villain: int, prizes: Sequence[float]) -> float:
    """ICM risk premium of an all-in for ``min(hero, villain)`` chips between two players.

    ``q = (U_fold - U_lose) / (U_win - U_lose)`` is the equity the hero needs to prefer the
    all-in to keeping its stack, with points from :func:`icm_player`; chip EV needs 0.5.
    """
    e = min(stacks[hero], stacks[villain])
    if e <= 0:
        return 0.0
    win = list(stacks)
    lose = list(stacks)
    win[hero] += e
    win[villain] -= e
    lose[hero] -= e
    lose[villain] += e
    u_fold = icm_player(stacks, hero, prizes)
    u_win = icm_player(win, hero, prizes)
    u_lose = icm_player(lose, hero, prizes)
    span = u_win - u_lose
    if span <= 1e-12:
        return 0.0
    return (u_fold - u_lose) / span - 0.5


# -------------------------------------------------------------------------- keys


def depth_bucket(chips: float, bb: int) -> int:
    return bisect_right(DEPTH_EDGES, chips / bb)


def pressure_bucket(rp: float) -> int:
    return bisect_right(PRESSURE_EDGES, rp)


def horizon_bucket(snap: Snapshot) -> int:
    left = snap.hand_cap - snap.hand_number
    if left <= 0:
        return 0
    return 1 if left < snap.n else 2


def _pack(kind: str, level: int, fields: Sequence[int]) -> int:
    key = KIND_CODE[kind]
    key = (key << 3) | level
    for f in fields:
        if not 0 <= f < 64:
            raise ValueError(f"key field {f} out of range")
        key = (key << 6) | f
    return key


def in_range(chips_at_risk: float, bb: int) -> bool:
    return chips_at_risk / bb <= MAX_DEPTH_BB


def open_context(snap: Snapshot, hero: int) -> tuple[int, int, float, bool, float]:
    """``(behind, villain, chips_at_risk, covered, pressure)`` for an open shove."""
    behind = list(range(hero + 1, snap.n))
    villain = max(behind, key=lambda p: (snap.stacks[p], -p))
    risk = min(snap.stacks[hero], snap.stacks[villain])
    return len(behind), villain, risk, snap.stacks[hero] <= snap.stacks[villain], \
        pressure(snap.stacks, hero, villain, snap.prizes)


def open_keys(snap: Snapshot, hero: int) -> tuple[int, ...]:
    behind, _, risk, covered, rp = open_context(snap, hero)
    n, d, c, p, h = snap.n, depth_bucket(risk, snap.bb), int(covered), pressure_bucket(rp), horizon_bucket(snap)
    return (_pack("open", 0, (n, behind, d, c, p, h)), _pack("open", 1, (n, behind, d, p)),
            _pack("open", 2, (n, behind, d)), _pack("open", 3, (behind, d)))


def call_keys(snap: Snapshot, hero: int, shover: int) -> tuple[int, ...]:
    behind = snap.n - 1 - hero
    shover_behind = snap.n - 1 - shover
    risk = min(snap.stacks[hero], snap.stacks[shover])
    covered = int(snap.stacks[hero] <= snap.stacks[shover])
    n, d, p, h = snap.n, depth_bucket(risk, snap.bb), \
        pressure_bucket(pressure(snap.stacks, hero, shover, snap.prizes)), horizon_bucket(snap)
    return (_pack("call", 0, (n, behind, shover_behind, d, covered, p, h)),
            _pack("call", 1, (n, behind, shover_behind, d, covered, p)),
            _pack("call", 2, (n, behind, shover_behind, d, covered)), _pack("call", 3, (behind, d, covered)))


def overcall_keys(snap: Snapshot, hero: int, shover: int, caller: int) -> tuple[int, ...]:
    behind = snap.n - 1 - hero
    big = max((shover, caller), key=lambda p: snap.stacks[p])
    risk = min(snap.stacks[hero], snap.stacks[big])
    covered = int(snap.stacks[hero] <= snap.stacks[big])
    n, d, p = snap.n, depth_bucket(risk, snap.bb), pressure_bucket(pressure(snap.stacks, hero, big, snap.prizes))
    return (_pack("overcall", 0, (n, behind, d, covered, p)), _pack("overcall", 1, (n, behind, d, covered)),
            _pack("overcall", 2, (behind, d, covered)), _pack("overcall", 3, (d, covered)))


def keys_for(kind: str, snap: Snapshot, hero: int, shover: int | None = None,
             caller: int | None = None) -> tuple[int, ...]:
    if kind == "open":
        return open_keys(snap, hero)
    if kind == "call":
        assert shover is not None
        return call_keys(snap, hero, shover)
    assert shover is not None and caller is not None
    return overcall_keys(snap, hero, shover, caller)
