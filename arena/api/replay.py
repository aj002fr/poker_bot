"""Hand-by-hand replay frames for one table, re-simulated from its private log.

The table is rebuilt the way :mod:`arena.history.replay` does it: from the
``TableStarted`` record, each hand's stored deck and the accepted commands. After
every command the replaying table is projected for the requested viewer with the
same :func:`~arena.table.projections.table_view` the live table uses, so a replay
frame can never show more than that viewer saw (or could have seen) live. One
extra frame is captured as each hand completes, before its result is recorded,
so showdowns and awards are visible.

Perspectives: a seat sees its own cards, during and after the table; a
spectator sees public information; the reviewer additionally sees every hole
card, only once the table is complete and only with the host credential.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, Literal

from arena.domain.events import EventEnvelope
from arena.domain.models import Action, TournamentConfig
from arena.rules.profiles import RuleProfile
from arena.table.projections import ProjectionDenied, Viewer, project_events, table_view
from arena.table.service import Table

Perspective = Literal["seat", "spectator", "reviewer"]
FrameKind = Literal["deal", "action", "hand_end", "complete"]
CACHE_SIZE = 16


class ReplayError(RuntimeError):
    pass


class _FrameTable(Table):
    def __init__(self, *args: Any, on_hand_end: Callable[[Table], None], **kwargs: Any) -> None:
        self._on_hand_end = on_hand_end
        super().__init__(*args, **kwargs)

    def _record_result(self) -> None:
        self._on_hand_end(self)
        super()._record_result()


def _commands(log: list[EventEnvelope]) -> list[tuple[int, Action, str | None, Any]]:
    out = []
    pending: tuple[str, Any] | None = None
    for event in log:
        if event.type == "ActionFallback":
            pending = (event.data["reason"], event.data["rejected"])
        elif event.type == "ActionApplied" and event.data["auto"] is None:
            action = Action.model_validate({k: v for k, v in event.data["action"].items() if v is not None}, strict=True)
            reason, rejected = pending if pending is not None else (None, None)
            out.append((event.data["seat"], action, reason, rejected))
            pending = None
    return out


def _frame(table: Table, perspective: Perspective, seat: int | None, kind: FrameKind) -> dict[str, Any]:
    viewer = Viewer.for_seat(seat) if perspective == "seat" and seat is not None else Viewer.spectator()
    view = table_view(table, viewer).model_dump(mode="json")
    view["legal"] = None
    out: dict[str, Any] = {"kind": kind, "seq": table.log[-1].seq, "hand_number": table.hand_number,
                           "view": view, "hole_cards": None}
    if perspective == "reviewer" and table.hand is not None:
        out["hole_cards"] = {str(s): list(c) for s, c in sorted(table.hand.hole_cards.items())}
    return out


def build_frames(log: list[EventEnvelope], perspective: Perspective, seat: int | None,
                 notes: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """``notes`` (reviewer only): bot reasoning notes by decision id, attached to the frame each decision made."""
    started = log[0].data
    config = TournamentConfig.model_validate(started["config"], strict=True)
    hand_starts = [e for e in log if e.type == "HandStarted"]
    decks = {e.data["hand_number"]: tuple(e.data["deck"]) for e in hand_starts}

    def deck_source(hand_number: int, profile: RuleProfile) -> tuple[str, ...]:
        if hand_number not in decks:
            raise ReplayError(f"log has no deck for hand {hand_number}")
        return decks[hand_number]

    frames: list[dict[str, Any]] = []
    table = _FrameTable(
        config,
        table_id=log[0].table_id,
        deck_source=deck_source,
        initial_button=hand_starts[0].data["positions"]["button"] if hand_starts else None,
        start_stacks={int(s): v for s, v in started["start_stacks"].items()},
        rng_record=started["rng"],
        on_hand_end=lambda t: frames.append(_frame(t, perspective, seat, "hand_end")),
    )
    if table.hand is not None:
        frames.append(_frame(table, perspective, seat, "deal"))
    notes = notes if perspective == "reviewer" else None
    for actor, action, reason, rejected in _commands(log):
        hand_before = table.hand_number
        decision_id = table.decision_id
        first = len(frames)
        if reason is not None:
            table.apply_fallback(actor, reason, rejected)
        else:
            table.submit(actor, action)
        if table.complete:
            frames.append(_frame(table, perspective, seat, "complete"))
        else:
            frames.append(_frame(table, perspective, seat, "deal" if table.hand_number != hand_before else "action"))
        if notes and decision_id in notes and reason is None:
            frames[first]["bot_note"] = dict(notes[decision_id])
        if table.complete:
            break
    if table.log != log:
        raise ReplayError("re-simulated log differs from the stored log")
    for f in frames:
        f.setdefault("bot_note", None)
    return frames


class ReplayCache:
    """Frames per (table, committed log length, perspective); the log only grows."""

    def __init__(self, size: int = CACHE_SIZE) -> None:
        self.size = size
        self._items: OrderedDict[tuple, dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, log: list[EventEnvelope], perspective: Perspective, seat: int | None,
            notes: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
        key = (log[0].table_id, len(log), perspective, seat, len(notes) if notes else 0)
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key]
        built = _index(log, perspective, seat, notes)
        with self._lock:
            self._items[key] = built
            while len(self._items) > self.size:
                self._items.popitem(last=False)
        return built


def _index(log: list[EventEnvelope], perspective: Perspective, seat: int | None,
           notes: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    frames = build_frames(log, perspective, seat, notes)
    viewer = {"seat": Viewer.for_seat(seat) if seat is not None else Viewer.spectator(),
              "spectator": Viewer.spectator(), "reviewer": Viewer.reviewer()}[perspective]
    hands: dict[int, dict[str, Any]] = {}
    for f in frames:
        number = f["hand_number"]
        hand = hands.setdefault(number, {"hand_number": number, "hand_id": f["view"]["hand_id"], "frames": []})
        hand["hand_id"] = hand["hand_id"] or f["view"]["hand_id"]
        f["index"] = len(hand["frames"])
        hand["frames"].append(f)
    for hand in hands.values():
        segment = [e for e in log if e.hand_id == hand["hand_id"]]
        if hand is hands[max(hands)]:
            segment += [e for e in log if e.hand_id is None and e.type == "TableScored"]
        hand["events"] = [e.model_dump(mode="json") for e in project_events(segment, viewer)]
    return {"hands": [hands[n] for n in sorted(hands)]}


def check_perspective(perspective: str, *, seat: int | None, host: bool, complete: bool) -> Perspective:
    if perspective == "seat":
        if seat is None:
            raise ProjectionDenied("the seat perspective needs this table's seat credential")
        return "seat"
    if perspective == "spectator":
        return "spectator"
    if perspective == "reviewer":
        if not host:
            raise ProjectionDenied("the reviewer perspective needs the host credential")
        if not complete:
            raise ProjectionDenied("the reviewer perspective opens only after the table is complete")
        return "reviewer"
    raise ValueError(perspective)
