"""Per-seat table-reading stats built from public events only.

The four questions of *Harrington on Hold'em, Vol. I*, Part Three ("Reading the
Table"), as numbers with their sample sizes:

* **Hands played** (``vpip``): how often a seat voluntarily puts chips in preflop.
* **Raise-first** (``pfr`` and ``rfi``): how often it raises preflop, and how often it
  opens with a raise when the action is folded to it.
* **Callers or raisers** (``aggression``): bets and raises against calls, all streets.
* **Reraises** (``reraise``): after betting or raising and then facing a raise in the
  same betting round, how often it folds, calls or raises again.
* **Showdowns**: the hands it showed, with the line it took.

Feed it events as a spectator sees them (:class:`~arena.domain.observations.PublicEvent`
objects or their JSON dicts). It never needs hidden cards: only ``CardsRevealed`` shows
a hand. Events are de-duplicated by ``seq``, so a bot can call :meth:`observe` with each
observation's ``action_history`` (the current hand so far) and keep the tracker in its
explicit state with :meth:`to_state` / :meth:`from_state`. A bot only sees each hand up
to its own last decision, so its numbers cover fewer events than the table's.

Automatic actions (``auto``) are not choices and are skipped.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from arena.agents.styles.handeval import describe, evaluate, preflop_words

STREETS = ("preflop", "flop", "turn", "river")
RECENT_SHOWDOWNS = 3


def _rate(count: int, of: int) -> float | None:
    return round(count / of, 4) if of else None


@dataclass
class SeatCounts:
    hands: int = 0
    vpip: int = 0
    pfr: int = 0
    rfi: int = 0
    rfi_of: int = 0
    bets_raises: int = 0
    calls: int = 0
    faced: int = 0
    folded: int = 0
    called: int = 0
    reraised: int = 0
    showdowns: list[dict[str, Any]] = field(default_factory=list)
    showdown_count: int = 0

    def payload(self, seat: int, showdowns: int | None) -> dict[str, Any]:
        items = self.showdowns if showdowns is None else self.showdowns[-showdowns:] if showdowns else []
        return {
            "seat": seat,
            "hands": self.hands,
            "vpip": {"count": self.vpip, "of": self.hands, "rate": _rate(self.vpip, self.hands)},
            "pfr": {"count": self.pfr, "of": self.hands, "rate": _rate(self.pfr, self.hands)},
            "rfi": {"count": self.rfi, "of": self.rfi_of, "rate": _rate(self.rfi, self.rfi_of)},
            "aggression": {"bets_raises": self.bets_raises, "calls": self.calls,
                           "factor": round(self.bets_raises / self.calls, 4) if self.calls else None},
            "reraise": {"faced": self.faced, "folded": self.folded, "called": self.called,
                        "reraised": self.reraised, "fold_rate": _rate(self.folded, self.faced)},
            "showdowns": {"count": self.showdown_count, "items": [dict(x) for x in items]},
        }


@dataclass
class _Hand:
    hand_id: str | None
    hand_number: int
    dealt: list[int]
    street: str = "preflop"
    level: int = 0
    voluntary: bool = False
    vpip: set[int] = field(default_factory=set)
    pfr: set[int] = field(default_factory=set)
    acted_pre: set[int] = field(default_factory=set)
    aggressors: set[int] = field(default_factory=set)
    facing: set[int] = field(default_factory=set)
    lines: dict[int, dict[str, list[str]]] = field(default_factory=dict)
    board: list[str] = field(default_factory=list)
    shown: dict[int, list[str]] = field(default_factory=dict)
    won: dict[int, int] = field(default_factory=dict)

    def to_state(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in self.__dict__.items():
            if isinstance(v, set):
                out[k] = sorted(v)
            elif isinstance(v, dict):
                out[k] = {str(s): x for s, x in v.items()}
            else:
                out[k] = v
        return out

    @classmethod
    def from_state(cls, d: Mapping[str, Any]) -> _Hand:
        h = cls(d["hand_id"], d["hand_number"], list(d["dealt"]))
        h.street, h.level, h.voluntary, h.board = d["street"], d["level"], d["voluntary"], list(d["board"])
        for k in ("vpip", "pfr", "acted_pre", "aggressors", "facing"):
            setattr(h, k, set(d[k]))
        h.lines = {int(s): {st: list(t) for st, t in v.items()} for s, v in d["lines"].items()}
        h.shown = {int(s): list(v) for s, v in d["shown"].items()}
        h.won = {int(s): int(v) for s, v in d["won"].items()}
        return h


def _fields(event: Any) -> tuple[int, str | None, str, Mapping[str, Any]]:
    if isinstance(event, Mapping):
        return event["seq"], event.get("hand_id"), event["type"], event["data"]
    return event.seq, event.hand_id, event.type, event.data


class StatsTracker:
    """Incremental per-seat stats. ``max_showdowns`` caps the stored showdowns per seat."""

    def __init__(self, seats: Iterable[int] = (), *, max_showdowns: int | None = None) -> None:
        self.counts: dict[int, SeatCounts] = {int(s): SeatCounts() for s in seats}
        self.max_showdowns = max_showdowns
        self.last_seq = -1
        self.hands_tracked = 0
        self._hand: _Hand | None = None

    # ------------------------------------------------------------- input

    def observe(self, observation: Any) -> None:
        self.feed(observation.action_history)

    def feed(self, events: Iterable[Any]) -> None:
        for event in events:
            seq, hand_id, etype, data = _fields(event)
            if seq <= self.last_seq:
                continue
            self.last_seq = seq
            self._apply(hand_id, etype, data)

    def _seat(self, seat: int) -> SeatCounts:
        c = self.counts.get(seat)
        if c is None:
            c = self.counts[seat] = SeatCounts()
        return c

    def _apply(self, hand_id: str | None, etype: str, d: Mapping[str, Any]) -> None:
        if etype == "TableStarted":
            for s in d.get("seat_ring", ()):
                self._seat(int(s))
            return
        if etype == "HandStarted":
            self._finish()
            dealt = sorted(int(s) for s in d["start_stacks"])
            self._hand = _Hand(hand_id, d["hand_number"], dealt, level=0)
            self.hands_tracked += 1
            for s in dealt:
                self._seat(s).hands += 1
            return
        h = self._hand
        if h is None:
            return
        if etype == "BlindPosted":
            h.level = max(h.level, d["amount"])
        elif etype == "BoardDealt":
            h.street, h.level = d["street"], 0
            h.board.extend(d["cards"])
            h.aggressors.clear()
            h.facing.clear()
        elif etype == "ActionApplied":
            if d.get("auto"):
                return
            self._action(h, d)
        elif etype == "CardsRevealed":
            h.shown[d["seat"]] = list(d["cards"])
        elif etype == "PotAwarded":
            for s, amount in d["awards"].items():
                h.won[int(s)] = h.won.get(int(s), 0) + int(amount)
        elif etype == "HandCompleted":
            self._finish()

    def _action(self, h: _Hand, d: Mapping[str, Any]) -> None:
        seat, street, kind = d["seat"], d["street"], d["action"]["type"]
        c = self._seat(seat)
        if street == "preflop":
            if seat not in h.acted_pre:
                h.acted_pre.add(seat)
                if not h.voluntary:
                    c.rfi_of += 1
                    if kind == "raise_to":
                        c.rfi += 1
            if kind in ("call", "raise_to"):
                h.voluntary = True
                if seat not in h.vpip:
                    h.vpip.add(seat)
                    c.vpip += 1
            if kind == "raise_to" and seat not in h.pfr:
                h.pfr.add(seat)
                c.pfr += 1
        if kind == "raise_to":
            token = "raise" if street == "preflop" or h.level > 0 else "bet"
            c.bets_raises += 1
        else:
            token = kind
            if kind == "call":
                c.calls += 1
        if seat in h.facing:
            h.facing.discard(seat)
            c.faced += 1
            if kind == "fold":
                c.folded += 1
            elif kind == "raise_to":
                c.reraised += 1
            else:
                c.called += 1
        if kind == "raise_to":
            h.facing |= h.aggressors - {seat}
            h.aggressors.add(seat)
            h.level = max(h.level, d["street_committed"])
        elif kind == "fold":
            h.aggressors.discard(seat)
        h.lines.setdefault(seat, {st: [] for st in STREETS})[street].append(token)

    def _finish(self) -> None:
        h, self._hand = self._hand, None
        if h is None:
            return
        for seat, cards in sorted(h.shown.items()):
            if len(h.board) >= 3:
                words = describe(evaluate([*cards, *h.board]))
            else:
                words = preflop_words(cards) if len(cards) == 2 else ""
            entry = {
                "hand_number": h.hand_number,
                "hand_id": h.hand_id,
                "cards": list(cards),
                "board": list(h.board),
                "hand_in_words": words,
                "line": {st: list(t) for st, t in h.lines.get(seat, {s: [] for s in STREETS}).items()},
                "won": h.won.get(seat, 0),
            }
            c = self._seat(seat)
            c.showdown_count += 1
            c.showdowns.append(entry)
            if self.max_showdowns is not None and len(c.showdowns) > self.max_showdowns:
                del c.showdowns[: len(c.showdowns) - self.max_showdowns]

    # ------------------------------------------------------------ output

    def payload(self, *, showdowns: int | None = None) -> dict[str, Any]:
        """The ``stats`` wire shape; ``showdowns`` limits each seat's list (None: all)."""
        return {"hands_tracked": self.hands_tracked,
                "seats": {str(s): c.payload(s, showdowns) for s, c in sorted(self.counts.items())}}

    def seat(self, seat: int) -> dict[str, Any]:
        return self._seat(seat).payload(seat, None)

    def rates(self, seat: int) -> dict[str, float | None]:
        """Compact numbers for a bot's decisions: vpip, pfr, rfi, aggression, fold_to_reraise."""
        p = self.seat(seat)
        return {"hands": p["hands"], "vpip": p["vpip"]["rate"], "pfr": p["pfr"]["rate"], "rfi": p["rfi"]["rate"],
                "aggression": p["aggression"]["factor"], "fold_to_reraise": p["reraise"]["fold_rate"]}

    def to_state(self) -> dict[str, Any]:
        return {
            "v": 1,
            "last_seq": self.last_seq,
            "hands_tracked": self.hands_tracked,
            "max_showdowns": self.max_showdowns,
            "counts": {str(s): {**c.__dict__, "showdowns": list(c.showdowns)} for s, c in self.counts.items()},
            "hand": self._hand.to_state() if self._hand is not None else None,
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> StatsTracker:
        t = cls(max_showdowns=state.get("max_showdowns"))
        t.last_seq = state["last_seq"]
        t.hands_tracked = state["hands_tracked"]
        t.counts = {int(s): SeatCounts(**{**c, "showdowns": list(c["showdowns"])}) for s, c in state["counts"].items()}
        t._hand = _Hand.from_state(state["hand"]) if state.get("hand") is not None else None
        return t
