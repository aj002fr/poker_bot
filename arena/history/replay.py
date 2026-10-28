"""Replay a stored history through the pinned engine and verify it.

Replay rebuilds the table from the ``TableStarted`` record, feeds each hand the
deck stored in its ``HandStarted`` event (never a reshuffle), and re-submits
the accepted commands in order: each manual ``ActionApplied`` and each
``ActionFallback``. Automatic actions are not commands; the engine must
reproduce them by itself.

The regenerated log must equal the stored log event for event. That covers
every hand-start hash, every decision-boundary state hash, all dealing,
settlement, eliminations and scoring. The first difference halts replay with
:class:`ReplayMismatch`. Nothing is repaired or skipped.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import arena
from arena.domain.models import Action, TournamentConfig
from arena.history.canonical import TableHistory, export_history
from arena.rules.pokerkit_adapter import POKERKIT_VERSION
from arena.rules.profiles import RuleProfile
from arena.table.service import ActionRejected, Table


class ReplayIncompatible(RuntimeError):
    """The history was produced by an engine build this checkout cannot vouch for."""


class ReplayMismatch(RuntimeError):
    def __init__(self, index: int, message: str, *, hand_id: str | None = None, decision_id: str | None = None):
        where = f"event {index}"
        if hand_id:
            where += f" (hand {hand_id}"
            where += f", decision {decision_id})" if decision_id else ")"
        super().__init__(f"{where}: {message}")
        self.index = index
        self.hand_id = hand_id
        self.decision_id = decision_id


@dataclass
class ReplayReport:
    table_id: str
    complete: bool
    events: int
    hands: int
    decisions_verified: int
    hand_starts_verified: int
    final_stacks: dict[int, int]
    points: dict[int, float] | None
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _Command:
    seat: int
    action: Action
    fallback_reason: str | None
    rejected: object


def _commands(history: TableHistory) -> list[_Command]:
    commands: list[_Command] = []
    pending: tuple[str, object] | None = None
    for event in history.events:
        if event.type == "ActionFallback":
            pending = (event.data["reason"], event.data["rejected"])
        elif event.type == "ActionApplied" and event.data["auto"] is None:
            action = Action.model_validate(
                {k: v for k, v in event.data["action"].items() if v is not None}, strict=True
            )
            reason, rejected = pending if pending is not None else (None, None)
            commands.append(_Command(event.data["seat"], action, reason, rejected))
            pending = None
    return commands


def _check_engine(history: TableHistory, *, allow_engine_mismatch: bool) -> list[str]:
    engine = history.table_started.data["engine"]
    notes = []
    if engine["pokerkit_version"] != POKERKIT_VERSION:
        message = f"history engine PokerKit {engine['pokerkit_version']}, this build {POKERKIT_VERSION}"
        if not allow_engine_mismatch:
            raise ReplayIncompatible(message)
        notes.append(message)
    if engine["arena_version"] != arena.__version__:
        notes.append(f"history arena {engine['arena_version']}, this build {arena.__version__}")
    return notes


def _diff(index: int, expected, got) -> ReplayMismatch:
    hand_id = expected.hand_id if expected is not None else (got.hand_id if got is not None else None)
    decision_id = None
    if expected is not None and expected.type == "DecisionRequested":
        decision_id = expected.data["decision_id"]
    if expected is None:
        return ReplayMismatch(index, f"replay produced extra event {got.type}", hand_id=hand_id)
    if got is None:
        return ReplayMismatch(index, f"replay stopped before stored event {expected.type}", hand_id=hand_id)
    if expected.type != got.type:
        return ReplayMismatch(index, f"stored {expected.type}, replay {got.type}", hand_id=hand_id)
    if expected.type == "DecisionRequested" and expected.data["state_hash"] != got.data["state_hash"]:
        return ReplayMismatch(index, "decision state hash differs", hand_id=hand_id, decision_id=decision_id)
    if expected.type == "HandStarted" and expected.data["start_hash"] != got.data["start_hash"]:
        return ReplayMismatch(index, "hand start hash differs", hand_id=hand_id)
    keys = sorted(k for k in set(expected.data) | set(got.data) if expected.data.get(k) != got.data.get(k))
    other = [k for k in ("table_id", "hand_id", "seq") if getattr(expected, k) != getattr(got, k)]
    return ReplayMismatch(index, f"{expected.type} differs in {', '.join(other + keys)}", hand_id=hand_id,
                          decision_id=decision_id)


def _first_difference(stored, replayed) -> ReplayMismatch | None:
    for index in range(max(len(stored), len(replayed))):
        a = stored[index] if index < len(stored) else None
        b = replayed[index] if index < len(replayed) else None
        if a != b:
            return _diff(index, a, b)
    return None


def replay(history: TableHistory, *, allow_engine_mismatch: bool = False) -> ReplayReport:
    notes = _check_engine(history, allow_engine_mismatch=allow_engine_mismatch)
    started = history.table_started.data
    config = TournamentConfig.model_validate(started["config"], strict=True)
    if config.config_hash() != started["config_hash"]:
        raise ReplayMismatch(0, "stored config does not match its config_hash")
    hand_starts = [e for e in history.events if e.type == "HandStarted"]
    decks = {e.data["hand_number"]: tuple(e.data["deck"]) for e in hand_starts}

    def deck_source(hand_number: int, profile: RuleProfile) -> tuple[str, ...]:
        if hand_number not in decks:
            raise ReplayMismatch(len(history.events), f"replay reached hand {hand_number}, which the history lacks")
        return decks[hand_number]

    initial_button = hand_starts[0].data["positions"]["button"] if hand_starts else None
    try:
        table = Table(
            config,
            table_id=history.table_id,
            deck_source=deck_source,
            initial_button=initial_button,
            start_stacks=started["start_stacks"],
            rng_record=started["rng"],
        )
        for command in _commands(history):
            if table.complete or table.actor_seat != command.seat:
                break
            if command.fallback_reason is not None:
                receipt = table.apply_fallback(command.seat, command.fallback_reason, command.rejected)
                if receipt.action != command.action:
                    break
            else:
                table.submit(command.seat, command.action)
    except ActionRejected as exc:
        mismatch = _first_difference(history.events, table.log)
        raise mismatch or ReplayMismatch(len(table.log), f"stored action rejected on replay: {exc.code}") from None
    except ReplayMismatch:
        raise
    stored = history.events
    if allow_engine_mismatch and stored and table.log:
        # The replaying build records its own engine; every other field must still match.
        first = stored[0].model_copy(update={"data": {**stored[0].data, "engine": table.log[0].data["engine"]}})
        stored = [first, *stored[1:]]
    mismatch = _first_difference(stored, table.log)
    if mismatch is not None:
        raise mismatch
    regenerated = export_history(table)
    if not history.summary.complete and table.complete:
        raise ReplayMismatch(len(table.log), "stored history is marked incomplete but replay completed the table")
    if history.summary != regenerated.summary:
        raise ReplayMismatch(len(table.log), "stored summary differs from the replayed table")
    if not history.summary.complete:
        notes.append("table incomplete: verified up to the last committed event")
    return ReplayReport(
        table_id=history.table_id,
        complete=table.complete,
        events=len(table.log),
        hands=len(hand_starts),
        decisions_verified=sum(1 for e in table.log if e.type == "DecisionRequested"),
        hand_starts_verified=len(hand_starts),
        final_stacks=dict(table.stacks),
        points=regenerated.summary.points,
        notes=notes,
    )
