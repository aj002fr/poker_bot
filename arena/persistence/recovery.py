"""Rebuild a live table from its durable record (spec 9.2).

Recovery starts from the newest hand-start snapshot, has the pinned engine deal
that hand again, and re-submits every committed command after it. The rebuilt
log must equal the stored log event for event: the hand-start hash, every
decision hash, every card and payout. Any difference raises and the caller
halts the table. Nothing is reshuffled, repaired or re-decided: seeded tables
re-deal from the checkpointed dealer RNG and must reproduce the stored deck;
unseeded tables take the stored deck.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import arena
from arena.domain.cards import full_deck
from arena.domain.events import EventEnvelope
from arena.domain.models import TournamentConfig
from arena.domain.rng import Rng, SeededRng, SystemRng, shuffled
from arena.history.replay import ReplayIncompatible, ReplayMismatch, _first_difference, commands_from_events
from arena.rules.pokerkit_adapter import POKERKIT_VERSION
from arena.rules.profiles import RuleProfile
from arena.table.service import ActionRejected, DeckSource, Table


def check_engine(events: Sequence[EventEnvelope]) -> None:
    engine = events[0].data["engine"]
    if engine["pokerkit_version"] != POKERKIT_VERSION:
        raise ReplayIncompatible(f"table recorded on PokerKit {engine['pokerkit_version']}, this build {POKERKIT_VERSION}")
    if engine["arena_version"] != arena.__version__:
        raise ReplayIncompatible(f"table recorded on arena {engine['arena_version']}, this build {arena.__version__}")


def _deck_source(events: Sequence[EventEnvelope], rng: Rng, seeded: bool) -> DeckSource:
    stored = {e.data["hand_number"]: tuple(e.data["deck"]) for e in events if e.type == "HandStarted"}

    def source(hand_number: int, profile: RuleProfile) -> tuple[str, ...]:
        if not seeded and hand_number in stored:
            return stored[hand_number]
        return tuple(shuffled(full_deck(profile.ranks), rng))

    return source


def rebuild_table(
    config: TournamentConfig,
    events: Sequence[EventEnvelope],
    point: Mapping[str, Any] | None,
    *,
    seed: str | None,
    on_hand_start: Callable[[Table], None] | None = None,
) -> Table:
    """Return a table whose log equals ``events``, or raise :class:`ReplayMismatch`.

    ``point`` is the newest stored resume point (``None`` replays from the start).
    ``seed`` is the table's private deck seed, or ``None`` for system-random decks.
    """
    if not events or events[0].type != "TableStarted":
        raise ReplayMismatch(0, "stored log does not start with TableStarted")
    check_engine(events)
    if config.config_hash() != events[0].data["config_hash"]:
        raise ReplayMismatch(0, "stored config does not match its config_hash")
    seeded = seed is not None
    hand_starts = [e for e in events if e.type == "HandStarted"]
    try:
        if point is not None:
            if point["seq"] > len(events) or events[point["seq"]].type != "HandStarted":
                raise ReplayMismatch(point["seq"], "snapshot does not point at a stored HandStarted")
            if seeded:
                if point["env_rng"] is None:
                    raise ReplayMismatch(point["seq"], "seeded table snapshot lacks its dealer RNG state")
                rng: Rng = SeededRng.from_state(point["env_rng"])
            else:
                rng = SystemRng()
            point = dict(point)
            if point["positions"] is None and not seeded:
                point["initial_button"] = events[point["seq"]].data["positions"]["button"]
            table = Table.resume_at_hand(
                config, point, list(events[: point["seq"]]),
                env_rng=rng, deck_source=_deck_source(events, rng, seeded), on_hand_start=on_hand_start,
            )
            tail = events[point["seq"]:]
        else:
            rng = SeededRng(seed, "env") if seeded else SystemRng()
            started = events[0].data
            table = Table(
                config,
                table_id=events[0].table_id,
                env_rng=rng,
                deck_source=_deck_source(events, rng, seeded),
                initial_button=None if seeded or not hand_starts else hand_starts[0].data["positions"]["button"],
                start_stacks={int(s): v for s, v in started["start_stacks"].items()},
                rng_record=started["rng"],
                on_hand_start=on_hand_start,
            )
            tail = events
        for command in commands_from_events(tail):
            if table.complete or table.actor_seat != command.seat:
                break
            if command.fallback_reason is not None:
                receipt = table.apply_fallback(command.seat, command.fallback_reason, command.rejected)
                if receipt.action != command.action:
                    break
            else:
                table.submit(command.seat, command.action)
    except ActionRejected as exc:
        mismatch = _first_difference(list(events), table.log)
        raise mismatch or ReplayMismatch(len(table.log), f"stored action rejected on recovery: {exc.code}") from None
    except ValueError as exc:
        raise ReplayMismatch(0, f"snapshot rejected: {exc}") from None
    mismatch = _first_difference(list(events), table.log)
    if mismatch is not None:
        raise mismatch
    return table
