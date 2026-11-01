"""Preflop all-in subgames over the full stack vector.

A :class:`Subgame` is one hand from its start: every live player's stack in preflop
action order (first to act ... small blind, big blind), the blinds actually posted, the
odd-chip order, the payout vector and whether this is the hand-cap hand. Its push/fold
tree has these rules:

* Before anyone shoves, a player may fold or shove (go all-in). The big blind, if
  nobody shoved, checks: a walk, or a showdown with players all-in from posting.
* After a shove a player may fold or call. A call commits the caller's whole stack
  (an isolation all-in); chips above the second-largest commitment come back at
  settlement, so this equals a plain call whenever nobody behind can still act.
* Players all-in from posting never act and always reach showdown.
* At most ``max_in_pot`` players reach showdown; once that many are in, the rest fold.

Each terminal's utility is computed for every showdown result (a weak ordering of the
showdown players' hands) by exact side-pot settlement, then mapped to points:
eliminated players get their finishing points under the arena's simultaneous-bust rule
and survivors get the continuation value of the resulting stacks. On the hand-cap hand
the continuation value is the official cap ICM, so those states are solved to the
terminal score exactly. ``utility="chips"`` replaces points by the chip change
(the chip-EV baseline).
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from strategy.endgame.points import eliminations, icm_batch, settle

FOLD, SHOVE, CALL = "f", "s", "c"
ValueFn = Callable[[np.ndarray, Sequence[float]], np.ndarray]
"""``(survivor_stacks[m, k], prizes) -> values[m, k]``: continuation points of survivors."""


def icm_value(survivors: np.ndarray, prizes: Sequence[float]) -> np.ndarray:
    return icm_batch(survivors, prizes)


@dataclass(frozen=True)
class Subgame:
    stacks: tuple[int, ...]
    """Chips at the start of the hand, preflop action order (big blind last)."""
    posts: tuple[int, ...]
    """Blinds posted (``min(stack, blind)``); zero for non-blind players."""
    payouts: tuple[float, ...]
    """The tournament's full payout vector by finishing place."""
    bb: int
    odd_order: tuple[int, ...] | None = None
    """Player indices clockwise from the first seat after the button (default: SB first)."""
    final_hand: bool = False
    max_in_pot: int = 3
    utility: str = "points"
    seats: tuple[int, ...] | None = None
    value_fn: ValueFn = field(default=icm_value, compare=False, repr=False)

    def __post_init__(self) -> None:
        n = len(self.stacks)
        if n < 2 or len(self.posts) != n:
            raise ValueError("need at least two players and one post per player")
        if any(s <= 0 for s in self.stacks) or any(not 0 <= p <= s for p, s in zip(self.posts, self.stacks)):
            raise ValueError("stacks must be positive and posts within stacks")
        if self.utility not in ("points", "chips"):
            raise ValueError("utility is 'points' or 'chips'")
        if self.odd_order is not None and sorted(self.odd_order) != list(range(n)):
            raise ValueError("odd_order must permute the players")
        if len(self.payouts) < n:
            raise ValueError("payouts must cover every live player's place")

    @property
    def n(self) -> int:
        return len(self.stacks)

    @property
    def order_for_odd_chips(self) -> tuple[int, ...]:
        if self.odd_order is not None:
            return self.odd_order
        n = self.n
        if n == 2:
            return (1, 0)
        return (n - 2, n - 1, *range(n - 2))

    @property
    def auto_all_in(self) -> frozenset[int]:
        return frozenset(i for i in range(self.n) if self.posts[i] == self.stacks[i])

    @property
    def prizes(self) -> tuple[float, ...]:
        return tuple(float(p) for p in self.payouts[: self.n])


@dataclass
class Node:
    idx: int
    player: int
    """Acting player, or -1 for a terminal."""
    parent: int
    label: str
    children: dict[str, int] = field(default_factory=dict)
    shover: int | None = None
    in_pot: tuple[int, ...] = ()
    folded: tuple[int, ...] = ()


@dataclass
class Terminal:
    node: int
    showdown: tuple[int, ...]
    """Players reaching showdown (or the sole winner), ascending."""
    folded: tuple[int, ...]
    contrib: tuple[int, ...]
    codes: tuple[int, ...]
    """Weak-ordering codes of the showdown players (base-k digits of 'players strictly better')."""
    utility: np.ndarray
    """``utility[k, p]``: player p's utility under showdown result ``codes[k]``."""


def ordering_codes(k: int) -> list[tuple[int, tuple[int, ...]]]:
    """Every weak ordering of ``k`` players as ``(code, better_counts)``."""
    seen: dict[int, tuple[int, ...]] = {}
    for ranks in itertools.product(range(k), repeat=k):
        better = tuple(sum(1 for r in ranks if r > x) for x in ranks)
        code = 0
        for b in better:
            code = code * k + b
        seen[code] = better
    return sorted(seen.items())


class SubgameTree:
    """The push/fold tree of a :class:`Subgame` with per-outcome terminal utilities."""

    def __init__(self, game: Subgame) -> None:
        self.game = game
        self.nodes: list[Node] = []
        self.terminals: dict[int, Terminal] = {}
        auto = game.auto_all_in
        if len(auto) >= game.max_in_pot:
            raise ValueError("too many players all-in from posting for this subgame")
        self._auto = auto
        self._build(0, -1, "", None, (), ())
        self._utilities()
        self.player_nodes = [[nd.idx for nd in self.nodes if nd.player == p] for p in range(game.n)]

    # ----------------------------------------------------------------- tree
    def _new(self, player: int, parent: int, label: str, shover, in_pot, folded) -> int:
        idx = len(self.nodes)
        self.nodes.append(Node(idx, player, parent, label, {}, shover, in_pot, folded))
        if parent >= 0:
            self.nodes[parent].children[label] = idx
        return idx

    def _build(self, i: int, parent: int, label: str, shover, in_pot, folded) -> None:
        g = self.game
        n = g.n
        while i < n:
            if i in self._auto:
                i += 1
                continue
            if shover is None and i == n - 1:
                i = n
                break
            if shover is not None and len(in_pot) + len(self._auto) >= g.max_in_pot:
                folded = (*folded, i)
                i += 1
                continue
            break
        if i >= n:
            self._new(-1, parent, label, shover, in_pot, folded)
            return
        node = self._new(i, parent, label, shover, in_pot, folded)
        if shover is None:
            self._build(i + 1, node, FOLD, None, in_pot, (*folded, i))
            self._build(i + 1, node, SHOVE, i, (*in_pot, i), folded)
        else:
            self._build(i + 1, node, FOLD, shover, in_pot, (*folded, i))
            self._build(i + 1, node, CALL, shover, (*in_pot, i), folded)

    # ------------------------------------------------------------ utilities
    def _utilities(self) -> None:
        g = self.game
        n = g.n
        start = list(g.stacks)
        rows: list[tuple[int, int, list[int]]] = []
        pending: list[tuple[int, tuple[int, ...], tuple[int, ...], tuple[int, ...], list]] = []
        for nd in self.nodes:
            if nd.player != -1:
                continue
            live = set(nd.in_pot) | set(self._auto)
            if nd.shover is None:
                live.add(n - 1)
            showdown = tuple(sorted(live))
            folded = tuple(p for p in range(n) if p not in live)
            contrib = tuple(g.stacks[p] if (p in nd.in_pot or p in self._auto) else g.posts[p] for p in range(n))
            outcomes = ordering_codes(len(showdown)) if len(showdown) > 1 else [(0, (0,))]
            ends = []
            for code, better in outcomes:
                strength = [0] * n
                for p, b in zip(showdown, better):
                    strength[p] = -b
                fold_flags = [p not in live for p in range(n)]
                awards = settle(contrib, fold_flags, strength, g.order_for_odd_chips)
                end = [start[p] - contrib[p] + awards[p] for p in range(n)]
                if sum(end) != sum(start):
                    raise AssertionError("settlement does not conserve chips")
                ends.append(end)
            pending.append((nd.idx, showdown, folded, contrib, [(c, e) for (c, _), e in zip(outcomes, ends)]))
            for k, (_, e) in enumerate(pending[-1][4]):
                rows.append((nd.idx, k, e))
        values = self._values([r[2] for r in rows])
        cursor = 0
        for idx, showdown, folded, contrib, outs in pending:
            u = values[cursor:cursor + len(outs)]
            cursor += len(outs)
            self.terminals[idx] = Terminal(idx, showdown, folded, contrib, tuple(c for c, _ in outs), u)

    def _values(self, ends: list[list[int]]) -> np.ndarray:
        g = self.game
        start = np.array(g.stacks, np.float64)
        e = np.array(ends, np.float64)
        if g.utility == "chips":
            return e - start[None, :]
        out = np.zeros_like(e)
        alive = e > 0
        groups: dict[tuple[bool, ...], list[int]] = {}
        for r in range(len(ends)):
            groups.setdefault(tuple(alive[r]), []).append(r)
        prizes = g.prizes
        for pattern, rs in groups.items():
            surv = [p for p in range(g.n) if pattern[p]]
            k = len(surv)
            for r in rs:
                for p, pts in eliminations(g.stacks, ends[r], g.payouts, g.n).items():
                    out[r, p] = pts
            if k == 1:
                out[rs, surv[0]] = float(g.payouts[0])
                continue
            vals = icm_batch(e[np.ix_(rs, surv)], prizes[:k]) if g.final_hand else \
                g.value_fn(e[np.ix_(rs, surv)], prizes[:k])
            out[np.ix_(rs, surv)] = vals
        return out

    # ---------------------------------------------------------- inspection
    def path(self, idx: int) -> str:
        labels = []
        while idx > 0:
            nd = self.nodes[idx]
            labels.append(f"{self.nodes[nd.parent].player}{nd.label}")
            idx = nd.parent
        return " ".join(reversed(labels))

    def find(self, actions: Sequence[str]) -> int:
        """Node reached from the root by the acting players' action labels."""
        idx = 0
        for a in actions:
            idx = self.nodes[idx].children[a]
        return idx
