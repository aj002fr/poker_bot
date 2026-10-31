"""Starting-hand classes, opening charts and table positions for the style bots.

Pure functions over card strings and public view fields, shared by the style bots,
the learning aids and ``GET /styles``. The charts are our own simplified tables in
the spirit of the opening requirements in *Harrington on Hold'em, Vol. I*, Part Two;
they are not copied from the book.

Position groups count non-blind seats back from the button: the button and the seat
before it are ``late``, the two before those ``middle``, anything earlier ``early``.
The blinds are ``blinds``; when the action is folded to a blind it opens from the
``late`` chart.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

RANKS = "23456789TJQKA"
GRID_RANKS = RANKS[::-1]
RANK_VALUE = {r: i for i, r in enumerate(RANKS)}
GROUPS = ("early", "middle", "late")


def hand_class(cards: Sequence[str]) -> str:
    """``"AKs"``, ``"T9o"`` or ``"77"`` for two hole cards."""
    a, b = cards[0], cards[1]
    hi, lo = sorted((a, b), key=lambda c: RANK_VALUE[c[0]], reverse=True)
    if hi[0] == lo[0]:
        return hi[0] + lo[0]
    return hi[0] + lo[0] + ("s" if hi[1] == lo[1] else "o")


def grid() -> list[list[str]]:
    """The 13x13 grid, A..2 on both axes: pairs on the diagonal, suited above it."""
    rows = []
    for i, r in enumerate(GRID_RANKS):
        row = []
        for j, c in enumerate(GRID_RANKS):
            if i == j:
                row.append(r + c)
            elif j > i:
                row.append(r + c + "s")
            else:
                row.append(c + r + "o")
        rows.append(row)
    return rows


ALL_CLASSES: tuple[str, ...] = tuple(c for row in grid() for c in row)


def _pairs_from(low: str, high: str = "A") -> set[str]:
    return {r + r for r in RANKS[RANK_VALUE[low] : RANK_VALUE[high] + 1]}


def expand(spec: str) -> frozenset[str]:
    """Expand a comma-separated range such as ``"77+, ATs+, KQ, 54s-T9s, any"``.

    ``77+`` pairs from 77 up; ``99-66`` a band of pairs; ``ATs+`` A with kickers T..K
    suited; ``KQ`` both suits; ``54s-T9s`` a band of suited connectors (same gap);
    ``Axs`` any suited ace (``Ax`` any ace); ``any`` every hand.
    """
    out: set[str] = set()
    for raw in spec.split(","):
        tok = raw.strip()
        if not tok:
            continue
        if tok == "any":
            out.update(ALL_CLASSES)
            continue
        if "-" in tok:
            a, b = (t.strip() for t in tok.split("-"))
            if len(a) == 2 and a[0] == a[1]:
                lo, hi = sorted((a[0], b[0]), key=RANK_VALUE.__getitem__)
                out |= _pairs_from(lo, hi)
                continue
            suffix = a[2:]
            gap = RANK_VALUE[a[0]] - RANK_VALUE[a[1]]
            lo, hi = sorted((RANK_VALUE[a[0]], RANK_VALUE[b[0]]))
            for top in range(lo, hi + 1):
                out.add(RANKS[top] + RANKS[top - gap] + suffix)
            continue
        plus = tok.endswith("+")
        tok = tok.rstrip("+")
        if len(tok) == 2 and tok[0] == tok[1]:
            out |= _pairs_from(tok[0]) if plus else {tok}
            continue
        hi, lo, suits = tok[0], tok[1], tok[2:]
        kinds = [suits] if suits in ("s", "o") else ["s", "o"]
        if lo == "x":
            lows = RANKS[: RANK_VALUE[hi]]
        elif plus:
            lows = RANKS[RANK_VALUE[lo] : RANK_VALUE[hi]]
        else:
            lows = lo
        for low in lows:
            for k in kinds:
                out.add(hi + low + k)
    unknown = out - set(ALL_CLASSES)
    if unknown:
        raise ValueError(f"bad hand classes {sorted(unknown)} in {spec!r}")
    return frozenset(out)


CHART_SPECS: dict[str, dict[str, str]] = {
    "conservative": {
        "early": "99+, AQ+, AJs, KQs",
        "middle": "77+, AJ+, ATs, KQ, KJs, QJs",
        "late": "22+, A8+, A2s+, KT+, K9s, QT+, Q9s, JT, J9s, T9s, 98s",
    },
    "aggressive": {
        "early": "22+, Ax, KQ, KJ, QJ, 32s-JTs",
        "middle": "22+, Ax, KQ, KJ, QJ, 32s-JTs, KTs, QTs",
        "late": "22+, Ax, KT+, QT+, JT, K2s+, Q8s+, J8s+, 32s-JTs, 42s-T8s, 52s-96s",
    },
    "weak_tight": {
        "early": "99+, AQ+, AJs, KQs",
        "middle": "77+, AJ+, ATs, KQ, KJs",
        "late": "55+, AT+, A8s+, KJ+, KTs, QJ, QTs, JTs",
    },
    "station": {
        "early": "22+, Ax, K2s+, Q2s+, J7s+, T7s+, 96s+, 85s+, 74s+, 63s+, 52s+, 42s+, 32s, KT+, QT+, JT, T9o, 98o, 87o",
        "middle": "22+, Ax, K2s+, Q2s+, J7s+, T7s+, 96s+, 85s+, 74s+, 63s+, 52s+, 42s+, 32s, KT+, QT+, JT, T9o, 98o, 87o",
        "late": "22+, Ax, K2s+, Q2s+, J5s+, T6s+, 96s+, 85s+, 74s+, 63s+, 52s+, 42s+, 32s, K8+, Q9+, J9+, T9o, 98o, 87o, 76o",
    },
    "push": {
        "early": "77+, AT+, KQ",
        "middle": "55+, A7+, A2s+, KT+, QJs",
        "late": "22+, Ax, K8+, K2s+, Q9+, Q8s+, J9s+, T9s",
    },
}

CHARTS: dict[str, dict[str, frozenset[str]]] = {
    name: {group: expand(spec) for group, spec in groups.items()} for name, groups in CHART_SPECS.items()
}
PUBLIC_CHARTS = ("conservative", "aggressive")


def chart_group(group: str) -> str:
    return "late" if group == "blinds" else group


def in_chart(chart: str, group: str, hand: str) -> bool:
    return hand in CHARTS[chart][chart_group(group)]


def in_range(spec: str, hand: str) -> bool:
    return hand in _cached(spec)


_RANGE_CACHE: dict[str, frozenset[str]] = {}


def _cached(spec: str) -> frozenset[str]:
    r = _RANGE_CACHE.get(spec)
    if r is None:
        r = _RANGE_CACHE[spec] = expand(spec)
    return r


# ------------------------------------------------------------------ positions


def _clockwise_from(seat: int, ring: Sequence[int]) -> list[int]:
    i = ring.index(seat)
    return list(ring[i + 1 :]) + list(ring[: i + 1])


def preflop_order(ring: Sequence[int], dealt: Iterable[int], big_blind: int) -> list[int]:
    """Dealt seats in preflop acting order: from the seat after the big blind round to it."""
    dealt_set = set(dealt)
    return [s for s in _clockwise_from(big_blind, ring) if s in dealt_set]


def seat_positions(ring: Sequence[int], dealt: Iterable[int], button: int | None, small_blind: int | None,
                   big_blind: int) -> dict[int, dict[str, str]]:
    """Position label (UTG, MP, CO, BTN, SB, BB) and group for every dealt seat."""
    order = preflop_order(ring, dealt, big_blind)
    out: dict[int, dict[str, str]] = {big_blind: {"label": "BB", "group": "blinds"}}
    if len(order) == 2:
        other = order[0]
        out[other] = {"label": "BTN", "group": "late"}
        return out
    rest = order[:-1]
    if small_blind is not None and small_blind in rest and rest[-1] == small_blind:
        out[small_blind] = {"label": "SB", "group": "blinds"}
        rest = rest[:-1]
    k = len(rest)
    for i, seat in enumerate(rest):
        d = k - 1 - i
        if d == 0:
            label = "BTN"
        elif d == 1:
            label = "CO"
        elif i == 0:
            label = "UTG"
        else:
            label = "MP"
        group = "late" if d <= 1 else ("middle" if d <= 3 else "early")
        out[seat] = {"label": label, "group": group}
    return out


def positions_from_events(seats: Sequence[int], events: Iterable[Mapping]) -> dict[int, dict[str, str]] | None:
    """Positions for the hand whose public ``HandStarted`` event is in ``events``."""
    for e in events:
        etype = e["type"] if isinstance(e, Mapping) else e.type
        if etype != "HandStarted":
            continue
        d = e["data"] if isinstance(e, Mapping) else e.data
        pos = d["positions"]
        dealt = [int(s) for s in d["start_stacks"]]
        return seat_positions(list(seats), dealt, pos["button"], d.get("small_blind_seat"), pos["big_blind"])
    return None
