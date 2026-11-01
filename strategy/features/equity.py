"""Decision-time equity lookups from the offline tables, with a labelled fallback.

Preflop rows are keyed by :func:`~strategy.features.cards.canonical_hole_key` and give
equity against 1..7 random hands, the spread of heads-up equity across flops, the
combo-weighted percentile and the strength decile (the latent range bucket). Postflop
cells are keyed by :func:`~strategy.features.cards.postflop_key` and give equity against
1..3 random hands; more opponents extrapolate geometrically from the last two
(``eq_k ~ e^k`` for independent showdowns).

When a table is missing (the PLO4 preflop table is built on demand, see
``strategy/tables/README.md``) or a cell is empty, a monotone heuristic stands in and
the result's ``source`` says ``prior`` so callers can discount it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from strategy.features.cards import CardCapabilities, canonical_hole_key

PREFLOP_OPPONENTS = 7
POSTFLOP_OPPONENTS = 3
BUCKETS = 10
SOURCES = ("table", "parent", "parent", "parent", "prior")
"""Postflop ``src`` codes: exact cell, three coarser aggregates, none."""


@dataclass(frozen=True, slots=True)
class PreflopEquity:
    equity: float
    sd: float
    """Spread of heads-up equity across flops (how much the flop moves this hand)."""
    tie: float
    percentile: float
    bucket: int
    source: str


@dataclass(frozen=True, slots=True)
class PostflopEquity:
    equity: float
    sd: float
    """Spread of heads-up equity among hands in the same bucket."""
    tie: float
    source: str


def _extrapolate(eqs: Sequence[float], k: int) -> float:
    if k <= len(eqs):
        return eqs[k - 1]
    a, b = eqs[-2], eqs[-1]
    ratio = b / a if a > 0 else 0.0
    return b * ratio ** (k - len(eqs))


def heuristic_percentile(hole: Sequence[int]) -> float:
    """Monotone stand-in for the preflop percentile: high cards, pairs, suits, connectivity."""
    ranks = sorted((c >> 2 for c in hole), reverse=True)
    suits = [c & 3 for c in hole]
    n = len(ranks)
    hi = ranks[0] / 12
    second = ranks[1] / 12
    pair = max((r for r in ranks if ranks.count(r) >= 2), default=-1)
    suited = sum(1 for s in set(suits) if suits.count(s) >= 2)
    distinct = sorted(set(ranks), reverse=True)
    gaps = [a - b for a, b in zip(distinct, distinct[1:])]
    connected = sum(1 for g in gaps if g <= 2) / max(1, n - 1)
    s = 0.32 * hi + 0.18 * second + 0.06 * min(suited, 2) + 0.08 * connected
    if pair >= 0:
        s += 0.22 + 0.14 * pair / 12
    return min(1.0, max(0.0, s))


class EquityTables:
    """Lookups for one variant. Load once per table; every lookup is O(1)."""

    __slots__ = ("caps", "has_preflop", "has_postflop", "_index", "_pre_eq", "_pre_sd", "_pre_tie", "_pct",
                 "_bucket", "_post_eq", "_post_sd", "_post_tie", "_post_src", "bucket_combos", "bucket_card",
                 "provenance")

    def __init__(self, caps: CardCapabilities, directory: str | None = None) -> None:
        from strategy.tables import store
        from strategy.tables.postflop import table_name as post_name
        from strategy.tables.preflop import table_name as pre_name

        self.caps = caps
        self.provenance: dict[str, dict] = {}
        pre = store.load(pre_name(caps.ruleset_id), directory)
        self.has_preflop = pre is not None
        self.bucket_combos: list[int] | None = None
        self.bucket_card: list[list[int]] | None = None
        if pre is not None:
            prov, d = pre
            self.provenance["preflop"] = prov
            sc = d["scale"]
            keys = d["keys"]
            self._index = {k: i for i, k in enumerate(keys)}
            self._pre_eq = [tuple(d["eq"][j][i] / sc for j in range(d["max_opponents"])) for i in range(len(keys))]
            self._pre_sd = [max(0.0, fsq / sc - (fm / sc) ** 2) ** 0.5 for fm, fsq in zip(d["flop_mean"], d["flop_sq"])]
            self._pre_tie = [t / sc for t in d["tie"]]
            self._pct = [p / sc for p in d["percentile"]]
            self._bucket = list(d["bucket"])
            self.bucket_combos = list(d["bucket_combos"])
            self.bucket_card = [list(r) for r in d["bucket_card"]]
        post = store.load(post_name(caps.ruleset_id), directory)
        self.has_postflop = post is not None
        if post is not None:
            prov, d = post
            self.provenance["postflop"] = prov
            sc = d["scale"]
            eq = d["eq"]
            self._post_eq = [tuple(eq[k][i] / sc for k in range(d["opponents"])) if d["src"][i] < 4 else None
                             for i in range(d["cells"])]
            self._post_sd = [x / sc if x >= 0 else 0.0 for x in d["sd"]]
            self._post_tie = [x / sc if x >= 0 else 0.0 for x in d["tie"]]
            self._post_src = list(d["src"])

    def preflop(self, hole: Sequence[int], opponents: int) -> PreflopEquity:
        k = min(PREFLOP_OPPONENTS, max(1, opponents))
        if self.has_preflop:
            i = self._index.get(canonical_hole_key(hole))
            if i is not None:
                return PreflopEquity(self._pre_eq[i][k - 1], self._pre_sd[i], self._pre_tie[i], self._pct[i],
                                     self._bucket[i], "table")
        pct = heuristic_percentile(hole)
        base = 1.0 / (k + 1)
        return PreflopEquity(base * (0.5 + pct), 0.15, 0.0, pct, min(BUCKETS - 1, int(pct * BUCKETS)), "prior")

    def postflop(self, cell: int, made_level: int, draw_cls: int, opponents: int) -> PostflopEquity:
        k = max(1, opponents)
        if self.has_postflop:
            eqs = self._post_eq[cell]
            if eqs is not None:
                return PostflopEquity(_extrapolate(eqs, k), self._post_sd[cell], self._post_tie[cell],
                                      SOURCES[self._post_src[cell]])
        e = min(0.95, 0.2 + 0.075 * made_level + 0.03 * draw_cls)
        return PostflopEquity(e ** k, 0.2, 0.0, "prior")
