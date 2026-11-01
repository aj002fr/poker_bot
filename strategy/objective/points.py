"""The training and evaluation objective: official terminal points.

A seat's utility is the points it is awarded: its finishing position's points when
it is eliminated (shared on simultaneous busts with equal starting stacks), the
first prize when it is last alive, or its exact Malmuth-Harville ICM share of the
still-unassigned top prizes when the hand cap ends the tournament with several
survivors. All scoring goes through :mod:`arena.tournament.scoring`, the same code
the authoritative table uses; nothing here re-implements it.

Chips are diagnostics only. :func:`chip_diagnostics` labels them as such, and the
experiment schema rejects any primary metric other than terminal points.

Offline module: it imports arena internals and is not for decision-time code.
"""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from arena.tournament import scoring
from arena.tournament.scoring import SCORE_TOLERANCE, Standings, icm

OBJECTIVE_ID = "terminal_points_v1"
OBJECTIVE_DESCRIPTION = (
    "official terminal points: finish-position points on elimination or as last survivor, exact ICM "
    "(Malmuth-Harville, arena.tournament.scoring) of the unassigned top prizes among survivors at the hand cap"
)


def scorer_fingerprint() -> str:
    """SHA-256 of the scorer source, so a manifest pins the exact scoring rules."""
    return hashlib.sha256(inspect.getsource(scoring).encode()).hexdigest()


def icm_points(stacks: Sequence[int], prizes: Sequence[float]) -> list[float]:
    """Exact ICM expected prize per stack (up to 16 players)."""
    return icm(list(stacks), list(prizes))


def cap_allocation(survivor_stacks: Mapping[int, int], payouts: Sequence[int]) -> dict[int, float]:
    """Points for the survivors at the cap: they hold the top ``len(survivors)`` prizes, split by ICM."""
    seats = sorted(survivor_stacks)
    if len(seats) == 1:
        return {seats[0]: float(payouts[0])}
    values = icm([survivor_stacks[s] for s in seats], [float(p) for p in payouts[: len(seats)]])
    return dict(zip(seats, values))


@dataclass(frozen=True)
class Finish:
    seat: int
    positions: tuple[int, ...]
    points: float
    method: str
    """``elimination``, ``winner`` (last alive) or ``icm`` (survivor at the cap)."""
    hand_number: int

    @property
    def position(self) -> float:
        """Finishing position, averaged over shared positions; ICM survivors average their prize places."""
        return sum(self.positions) / len(self.positions)


@dataclass(frozen=True)
class TerminalOutcome:
    payouts: tuple[int, ...]
    hand_cap: int
    hands_played: int
    start_stacks: Mapping[int, int]
    final_stacks: Mapping[int, int]
    finishes: Mapping[int, Finish]

    @property
    def seats(self) -> list[int]:
        return sorted(self.finishes)

    def points(self, seat: int) -> float:
        return self.finishes[seat].points

    def utility(self, seat: int) -> float:
        """The objective. Identical to :meth:`points`; named for call sites that optimize it."""
        return self.finishes[seat].points

    @property
    def ended_at_cap(self) -> bool:
        return any(f.method == "icm" for f in self.finishes.values())

    @property
    def survivors(self) -> list[int]:
        return sorted(s for s, f in self.finishes.items() if f.method in ("icm", "winner"))

    def cap_icm_points(self, seat: int) -> float:
        f = self.finishes[seat]
        return f.points if f.method == "icm" else 0.0

    def validate(self) -> None:
        total = sum(f.points for f in self.finishes.values())
        if abs(total - sum(self.payouts)) > SCORE_TOLERANCE * max(1.0, sum(self.payouts)):
            raise AssertionError(f"awarded {total} points, expected {sum(self.payouts)}")
        if sum(self.final_stacks.values()) != sum(self.start_stacks.values()):
            raise AssertionError("chips not conserved")
        for s, f in self.finishes.items():
            busted = self.final_stacks.get(s, 0) == 0
            if busted != (f.method == "elimination"):
                raise AssertionError(f"seat {s}: method {f.method} with {self.final_stacks.get(s)} chips")


def chip_diagnostics(outcome: TerminalOutcome, seat: int) -> dict[str, Any]:
    """Chip numbers for reports. Not an objective: never optimize or promote on these."""
    start = outcome.start_stacks[seat]
    end = outcome.final_stacks.get(seat, 0)
    return {"diagnostic_only": True, "start_chips": start, "final_chips": end, "chip_delta": end - start}


def _event_fields(e: Any) -> tuple[str, Mapping[str, Any]]:
    if isinstance(e, Mapping):
        return e["type"], e["data"]
    return e.type, e.data


def outcome_from_events(events: Iterable[Any]) -> TerminalOutcome:
    """From a complete private event log (``EventEnvelope`` objects or their JSON dicts)."""
    config: Mapping[str, Any] | None = None
    start: dict[int, int] = {}
    final: dict[int, int] = {}
    hands = 0
    scored: Mapping[str, Any] | None = None
    for e in events:
        kind, data = _event_fields(e)
        if kind == "TableStarted":
            config = data["config"]
            start = {int(s): int(v) for s, v in data["start_stacks"].items()}
            final = dict(start)
        elif kind == "HandStarted":
            hands = int(data["hand_number"])
        elif kind == "HandCompleted":
            final.update({int(s): int(v) for s, v in data["final_stacks"].items()})
        elif kind == "TableScored":
            scored = data
    if config is None or scored is None:
        raise ValueError("event log has no TableStarted/TableScored: the table did not finish")
    finishes = {int(p["seat"]): Finish(int(p["seat"]), tuple(p["positions"]), float(p["points"]), p["method"],
                                       int(p["hand_number"])) for p in scored["placements"]}
    out = TerminalOutcome(tuple(config["payout_points"]), int(config["hand_cap"]), hands, start, final, finishes)
    out.validate()
    return out


def outcome_from_table(table: Any) -> TerminalOutcome:
    """From a finished :class:`arena.table.service.Table`."""
    if not table.complete:
        raise ValueError("table not complete")
    return outcome_from_events(table.log)


def rescore(payouts: Sequence[int], seats: Sequence[int], start_stacks: Mapping[int, int],
            hands: Iterable[tuple[int, Mapping[int, int], Mapping[int, int]]], hand_cap: int) -> TerminalOutcome:
    """Score a tournament from ``(hand_number, start_stacks, end_stacks)`` per hand with the arena scorer.

    For rollouts that do not run a :class:`Table` (solvers, value models): they must
    reach the same points the table would.
    """
    standings = Standings(list(payouts), list(seats))
    stacks = dict(start_stacks)
    last = 0
    for hand_number, before, after in hands:
        standings.record_eliminations(hand_number, before, after)
        stacks.update(after)
        last = hand_number
    survivors = {s: stacks[s] for s in standings.remaining}
    standings.finalize(last, survivors)
    finishes = {s: Finish(s, p.positions, p.points, p.method, p.hand_number) for s, p in standings.placements.items()}
    out = TerminalOutcome(tuple(payouts), hand_cap, last, dict(start_stacks), stacks, finishes)
    out.validate()
    return out


def rescore_events(events: Iterable[Any]) -> TerminalOutcome:
    """Independently rescore a logged tournament from its hand start/end stacks."""
    payouts: Sequence[int] = ()
    seats: list[int] = []
    start: dict[int, int] = {}
    hand_cap = 0
    hands: list[tuple[int, dict[int, int], dict[int, int]]] = []
    current: tuple[int, dict[int, int]] | None = None
    for e in events:
        kind, data = _event_fields(e)
        if kind == "TableStarted":
            payouts, seats = data["config"]["payout_points"], list(data["seat_ring"])
            hand_cap = int(data["config"]["hand_cap"])
            start = {int(s): int(v) for s, v in data["start_stacks"].items()}
        elif kind == "HandStarted":
            current = (int(data["hand_number"]), {int(s): int(v) for s, v in data["start_stacks"].items()})
        elif kind == "HandCompleted" and current is not None:
            hands.append((current[0], current[1], {int(s): int(v) for s, v in data["final_stacks"].items()}))
            current = None
    return rescore(payouts, seats, start, hands, hand_cap)
