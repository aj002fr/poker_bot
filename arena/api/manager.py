"""Table manager behind the HTTP/WebSocket API: the single command writer per table.

Each table has a FIFO command queue; joins, starts, human actions, timeout
fallbacks and bot decisions run through it one at a time. A command mutates
the in-memory engine (which validates it), then commits everything it changed
in one store transaction: events, hand rows, hand-start snapshots, the action
receipt, bot RNG checkpoints, awards, the scheduler and an outbox row. Only
after the commit does the API answer, and only outbox versions that committed
are broadcast. If a commit fails, the in-memory table is thrown away and
rebuilt from the store, so memory never runs ahead of what is durable.

Ownership is a writer id stored on the table row. Every commit is a
compare-and-set on (owner, version), so a writer that lost the table cannot
append to it. Running several server processes against one database
(multi-writer fencing tokens, lease handover) is deliberately not built yet.

Nothing here renders state directly: every view comes from
:mod:`arena.table.projections`.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import secrets
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, ValidationError

from arena.api.bots import BOT_POLICIES
from arena.api.clock import Clock, MonotonicClock
from arena.domain.canonical import canonical_json
from arena.domain.events import EventEnvelope
from arena.domain.models import RULESET_IDS, ActionEnvelope, TournamentConfig
from arena.domain.rng import SeededRng
from arena.history.replay import ReplayIncompatible, ReplayMismatch
from arena.persistence.models import (
    AwardRow,
    CheckpointRow,
    CommitBatch,
    HandRow,
    ParticipantRow,
    PauseRow,
    ReceiptRow,
    SnapshotRow,
    StoredTable,
    TableRow,
)
from arena.persistence.recovery import rebuild_table
from arena.persistence.store import MemoryStore, OwnershipLost, Store, StoreError, StoreUnavailable
from arena.table.driver import policy_rngs
from arena.table.projections import (
    Viewer,
    observation,
    project_events,
    table_history_view,
    table_view,
)
from arena.table.service import ActionRejected, Table, seeded_table

SeatKind = Literal["human", "bot"]
Role = Literal["seat", "host", "anonymous"]

REJECTION_STATUS = {
    "table_complete": 409,
    "wrong_hand": 409,
    "stale_decision": 409,
    "stale_version": 409,
    "out_of_turn": 409,
    "no_decision": 409,
    "invalid_schema": 422,
    "illegal_action": 422,
}

FAULT_POINTS = ("before_tx", "in_tx", "after_commit", "before_broadcast")


class SimulatedCrash(BaseException):
    """Raised by fault hooks to stand in for the process dying at that point."""


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, details: Any = None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.details = details

    def body(self) -> dict[str, Any]:
        out: dict[str, Any] = {"error": {"code": self.code, "message": self.message}}
        if self.details is not None:
            out["error"]["details"] = self.details
        return out


def storage_unavailable() -> ApiError:
    return ApiError(503, "storage_unavailable", "storage is unavailable; the table is paused and will resume")


def validation_details(exc: ValidationError) -> list[dict[str, Any]]:
    """Field locations and messages only; the submitted input is never echoed back."""
    return [{"loc": [str(p) for p in e["loc"]], "msg": e["msg"]} for e in exc.errors()]


class SeatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: SeatKind
    policy: StrictStr | None = None


class CreateTableRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    config: dict[str, Any] = Field(default_factory=dict)
    seats: list[SeatRequest] = Field(min_length=2, max_length=8)
    seed: StrictStr | None = Field(None, min_length=1, max_length=128)
    """Optional private seed for reproducible decks; never returned by any endpoint."""


class JoinRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seat: int | None = None


@dataclass(frozen=True)
class Principal:
    role: Role
    table_id: str | None = None
    seat: int | None = None

    @property
    def viewer(self) -> Viewer:
        return Viewer.for_seat(self.seat) if self.role == "seat" and self.seat is not None else Viewer.spectator()


ANONYMOUS = Principal("anonymous")


@dataclass
class SeatSlot:
    seat: int
    kind: SeatKind
    policy: str | None
    claimed: bool = False
    token_hash: str | None = None

    def public(self, viewer_seat: int | None) -> dict[str, Any]:
        bot = BOT_POLICIES.get(self.policy) if self.policy else None
        return {
            "seat": self.seat,
            "kind": self.kind,
            "policy": self.policy,
            "label": bot.label if bot else "Human",
            "artifact_hash": bot.artifact_hash if bot else None,
            "claimed": self.claimed if self.kind == "human" else True,
            "you": viewer_seat == self.seat,
        }

    def row(self) -> ParticipantRow:
        bot = BOT_POLICIES.get(self.policy) if self.policy else None
        return ParticipantRow(self.seat, self.kind, self.policy, bot.artifact_hash if bot else None, self.claimed,
                              self.token_hash)


@dataclass
class StoredReceipt:
    seat: int
    fingerprint: str
    body: dict[str, Any]


class Subscriber:
    """Latest-frame mailbox for one WebSocket. Frames only ever supersede each other."""

    def __init__(self, principal: Principal, loop: asyncio.AbstractEventLoop) -> None:
        self.principal = principal
        self.loop = loop
        self.event = asyncio.Event()
        self.latest: dict[str, Any] | None = None

    def push(self, frame: dict[str, Any]) -> None:
        self.latest = frame
        self.loop.call_soon_threadsafe(self.event.set)

    async def next(self) -> dict[str, Any]:
        while True:
            await self.event.wait()
            self.event.clear()
            frame, self.latest = self.latest, None
            if frame is not None:
                return frame


class CommandQueue:
    """FIFO mutual exclusion: one table's commands run one at a time, in arrival order."""

    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._next = 0
        self._serving = 0
        self._owner: int | None = None
        self._depth = 0

    def __enter__(self) -> None:
        me = threading.get_ident()
        with self._cv:
            if self._owner == me:
                self._depth += 1
                return
            ticket = self._next
            self._next += 1
            while ticket != self._serving:
                self._cv.wait()
            self._owner, self._depth = me, 1

    def __exit__(self, *exc: object) -> None:
        with self._cv:
            self._depth -= 1
            if self._depth == 0:
                self._owner = None
                self._serving += 1
                self._cv.notify_all()


@dataclass
class LiveTable:
    table_id: str
    config: TournamentConfig
    slots: dict[int, SeatSlot]
    seed: str | None
    bot_seed: str = ""
    host_token_hash: str = ""
    status: Literal["waiting", "running", "complete"] = "waiting"
    version: int = 0
    table: Table | None = None
    bot_rngs: dict[int, SeededRng] = field(default_factory=dict)
    bot_decisions: dict[int, int] = field(default_factory=dict)
    scheduled_decision: str | None = None
    scheduled_hand: str | None = None
    deadline_ns: int | None = None
    bot_due_ns: int | None = None
    timed_out: set[str] = field(default_factory=set)
    receipts: dict[str, StoredReceipt] = field(default_factory=dict)
    subscribers: list[Subscriber] = field(default_factory=list)
    inflight: list[int] = field(default_factory=list)
    results: dict[str, Any] | None = None
    queue: CommandQueue = field(default_factory=CommandQueue)
    # Durability bookkeeping: what the store already holds and what is still pending.
    committed_version: int = 0
    persisted_seq: int = 0
    pending_points: list[dict[str, Any]] = field(default_factory=list)
    dirty_bots: set[int] = field(default_factory=set)
    hand_rows: dict[str, HandRow] = field(default_factory=dict)
    unpublished: list[int] = field(default_factory=list)
    published_version: int = 0
    marked_version: int = 0
    dirty: bool = False
    """A commit failed after the engine moved: memory is ahead of the store until reloaded."""
    fenced: bool = False
    halted_reason: str | None = None

    def setup_public(self, viewer_seat: int | None) -> dict[str, Any]:
        return {
            "config": self.config.resolved(),
            "config_hash": self.config.config_hash(),
            "seats": [self.slots[s].public(viewer_seat) for s in sorted(self.slots)],
        }

    def capture_point(self, table: Table) -> None:
        self.pending_points.append(table.resume_point())


def _token_key(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def policy_catalog() -> list[dict[str, Any]]:
    return [
        {"id": p.id, "label": p.label, "description": p.description, "artifact_hash": p.artifact_hash,
         "rulesets": list(RULESET_IDS)}
        for p in BOT_POLICIES.values()
    ]


class TableManager:
    def __init__(
        self,
        clock: Clock | None = None,
        *,
        bot_delay_ms: int = 700,
        hand_pause_ms: int = 1500,
        store: Store | None = None,
        writer_id: str = "local-writer",
        lease_s: float = 15.0,
        heartbeat_ms: int = 1000,
        recover: bool = True,
        force_claim: bool = False,
    ) -> None:
        self.clock = clock or MonotonicClock()
        self.bot_delay_ns = int(bot_delay_ms * 1_000_000)
        self.hand_pause_ns = int(hand_pause_ms * 1_000_000)
        self.store: Store = store if store is not None else MemoryStore()
        self.writer_id = writer_id
        self.lease_s = lease_s
        self.heartbeat_ns = int(heartbeat_ms * 1_000_000)
        self.tables: dict[str, LiveTable] = {}
        self._tokens: dict[str, Principal] = {}
        self._registry = threading.Lock()
        self._inflight_lock = threading.Lock()
        self.faults: dict[str, Callable[[str, str], None]] = {}
        """Test hooks keyed by :data:`FAULT_POINTS`; called with (table_id, command kind)."""
        self._last_ok_ns = self.clock.now_ns()
        self._last_heartbeat_ns = self._last_ok_ns
        self.outage_since: float | None = None
        if recover:
            self.recover_all(force=force_claim)

    # ----------------------------------------------------------- configuration

    @staticmethod
    def build_config(raw: dict[str, Any], seat_count: int) -> TournamentConfig:
        data = dict(raw)
        data.setdefault("initial_player_count", seat_count)
        try:
            config = TournamentConfig.model_validate(data)
        except ValidationError as exc:
            raise ApiError(422, "invalid_config", "configuration rejected", validation_details(exc)) from None
        if config.initial_player_count != seat_count:
            raise ApiError(422, "invalid_config", "seat assignments must match initial_player_count")
        return config

    def resolve_config(self, body: CreateTableRequest) -> dict[str, Any]:
        config = self.build_config(body.config, len(body.seats))
        self._validate_seats(body.seats)
        return {"config": config.resolved(), "config_hash": config.config_hash(), "policies": policy_catalog()}

    @staticmethod
    def _validate_seats(seats: list[SeatRequest]) -> None:
        for i, s in enumerate(seats):
            if s.kind == "bot" and s.policy not in BOT_POLICIES:
                raise ApiError(422, "invalid_seat", f"seat {i}: unknown bot policy")
            if s.kind == "human" and s.policy is not None:
                raise ApiError(422, "invalid_seat", f"seat {i}: human seats take no policy")

    # -------------------------------------------------------- identity / auth

    def _get(self, table_id: str) -> LiveTable:
        with self._registry:
            live = self.tables.get(table_id)
        if live is None:
            raise ApiError(404, "table_not_found", "no such table")
        return live

    def _register(self, live: LiveTable) -> None:
        with self._registry:
            self.tables[live.table_id] = live
            self._tokens[live.host_token_hash] = Principal("host", live.table_id)
            for slot in live.slots.values():
                if slot.token_hash:
                    self._tokens[slot.token_hash] = Principal("seat", live.table_id, slot.seat)

    def authenticate(self, table_id: str, token: str | None) -> Principal:
        self._get(table_id)
        if not token:
            return ANONYMOUS
        key = _token_key(token)
        with self._registry:
            principal = next((p for k, p in self._tokens.items() if hmac.compare_digest(k, key)), None)
        if principal is None:
            raise ApiError(401, "invalid_credential", "credential not recognized")
        if principal.table_id != table_id:
            raise ApiError(403, "credential_wrong_table", "credential belongs to a different table")
        return principal

    @staticmethod
    def _require_seat(principal: Principal) -> int:
        if principal.role != "seat" or principal.seat is None:
            raise ApiError(401, "seat_credential_required", "acting requires this table's seat credential")
        return principal.seat

    def _check_usable(self, live: LiveTable) -> None:
        if live.halted_reason is not None:
            raise ApiError(503, "table_halted", "this table is halted for diagnosis: its stored record could not "
                                                "be reproduced exactly, so it will not continue")
        if live.fenced:
            raise ApiError(409, "writer_fenced", "this server no longer owns the table")
        if self.outage_since is not None or live.dirty:
            raise storage_unavailable()

    # --------------------------------------------------------------- lobby

    def create_table(self, body: CreateTableRequest) -> dict[str, Any]:
        config = self.build_config(body.config, len(body.seats))
        self._validate_seats(body.seats)
        if self.outage_since is not None:
            raise storage_unavailable()
        table_id = "t_" + secrets.token_hex(6)
        slots = {seat: SeatSlot(seat, s.kind, s.policy) for seat, s in zip(config.resolved_seats, body.seats,
                                                                             strict=True)}
        token = secrets.token_urlsafe(24)
        bot_seed = body.seed if body.seed is not None else secrets.token_hex(16)
        live = LiveTable(table_id, config, slots, body.seed, bot_seed=bot_seed,
                         host_token_hash=_token_key(token), version=1)
        with live.queue:
            self._commit(live, "create", participants=[s.row() for s in slots.values()])
            self._register(live)
            self._publish(live)
            return {"table_id": table_id, "host_token": token,
                    "snapshot": self._snapshot(live, Principal("host", table_id))}

    def join(self, table_id: str, seat: int | None) -> dict[str, Any]:
        live = self._get(table_id)
        with live.queue:
            self._check_usable(live)
            if live.status != "waiting":
                raise ApiError(409, "already_started", "seats can only be claimed before the table starts")
            open_seats = [s for s in sorted(live.slots) if live.slots[s].kind == "human" and not live.slots[s].claimed]
            if seat is None:
                if not open_seats:
                    raise ApiError(409, "no_open_seat", "every human seat is taken")
                seat = open_seats[0]
            elif seat not in open_seats:
                raise ApiError(409, "seat_unavailable", "that seat is not an open human seat")
            token = secrets.token_urlsafe(24)
            slot = live.slots[seat]
            slot.claimed, slot.token_hash = True, _token_key(token)
            live.version += 1
            self._commit(live, "join", participants=[slot.row()])
            self._register(live)
            self._publish(live)
            principal = Principal("seat", table_id, seat)
            return {"seat": seat, "seat_token": token, "snapshot": self._snapshot(live, principal)}

    def start(self, table_id: str, principal: Principal) -> dict[str, Any]:
        live = self._get(table_id)
        with live.queue:
            self._check_usable(live)
            if principal.role != "host":
                raise ApiError(403, "host_credential_required", "only the table host can start the table")
            if live.status != "waiting":
                raise ApiError(409, "already_started", "table already started")
            unclaimed = [s for s, slot in live.slots.items() if slot.kind == "human" and not slot.claimed]
            if unclaimed:
                raise ApiError(409, "seats_unclaimed", f"{len(unclaimed)} human seat(s) not yet claimed")
            seats = list(live.config.resolved_seats)
            if live.seed is not None:
                live.table = seeded_table(live.config, live.seed, table_id=table_id, on_hand_start=live.capture_point)
            else:
                live.table = Table(live.config, table_id=table_id, on_hand_start=live.capture_point)
            live.bot_rngs = {s: r for s, r in policy_rngs(live.bot_seed, seats).items()
                             if live.slots[s].kind == "bot"}
            live.bot_decisions = {s: 0 for s in live.bot_rngs}
            live.dirty_bots = set(live.bot_rngs)
            live.status = "running"
            now = self.clock.now_ns()
            self._schedule(live, now)
            live.version += 1
            self._commit(live, "start")
            self._run_due_bots(live, now)
            self._publish(live)
            return self._snapshot(live, principal)

    # ------------------------------------------------------------ snapshots

    def snapshot(self, table_id: str, principal: Principal) -> dict[str, Any]:
        live = self._get(table_id)
        with live.queue:
            self._check_usable(live)
            return self._snapshot(live, principal)

    def _snapshot(self, live: LiveTable, principal: Principal) -> dict[str, Any]:
        viewer = principal.viewer
        out: dict[str, Any] = {
            "table_id": live.table_id,
            "status": live.status,
            "version": live.version,
            "compatibility_status": "practice_assumptions",
            "viewer": {"role": principal.role, "seat": principal.seat},
            "setup": live.setup_public(principal.seat),
            "view": None,
            "turn": None,
            "last_hand": None,
            "results": None,
        }
        table = live.table
        if table is None:
            return out
        out["view"] = table_view(table, viewer).model_dump(mode="json")
        actor = table.actor_seat
        if actor is not None:
            kind = live.slots[actor].kind
            remaining = None
            if kind == "human" and live.deadline_ns is not None:
                remaining = max(0, (live.deadline_ns - self.clock.now_ns()) // 1_000_000)
            out["turn"] = {
                "seat": actor,
                "kind": kind,
                "decision_id": table.decision_id,
                "turn_ms": live.config.human_turn_ms if kind == "human" else None,
                "remaining_ms": remaining,
            }
        out["last_hand"] = self._last_hand(table, viewer)
        if live.status == "complete":
            out["results"] = self._results(live)
        return out

    @staticmethod
    def _last_hand(table: Table, viewer: Viewer) -> dict[str, Any] | None:
        """The most recently completed hand, as this viewer may see it."""
        log = table.log
        i = (table.hand_log_start if table.hand is not None else len(log)) - 1
        while i >= 0 and log[i].hand_id is None:
            i -= 1
        if i < 0:
            return None
        hand_id = log[i].hand_id
        j = i
        while j >= 0 and log[j].hand_id == hand_id:
            j -= 1
        events = project_events(log[j + 1 : i + 1], viewer)
        started = next((e for e in events if e.type == "HandStarted"), None)
        return {
            "hand_id": hand_id,
            "hand_number": started.data["hand_number"] if started else None,
            "events": [e.model_dump(mode="json") for e in events],
        }

    def _results(self, live: LiveTable) -> dict[str, Any]:
        if live.results is None:
            assert live.table is not None
            events = project_events(live.table.log, Viewer.spectator())
            scored = next(e for e in reversed(events) if e.type == "TableScored")
            fallbacks: dict[str, Counter] = {}
            for e in events:
                if e.type == "ActionFallback":
                    fallbacks.setdefault(str(e.data["seat"]), Counter())[e.data["reason"]] += 1
            live.results = {
                "placements": scored.data["placements"],
                "points": scored.data["points"],
                "payouts": list(live.config.resolved_payouts),
                "hands_played": live.table.hand_number,
                "hand_cap": live.config.hand_cap,
                "eliminations": [e.data for e in events if e.type == "PlayerEliminated"],
                "fallbacks": {s: dict(c) for s, c in fallbacks.items()},
            }
        return live.results

    def history(self, table_id: str, principal: Principal, perspective: str | None) -> dict[str, Any]:
        live = self._get(table_id)
        with live.queue:
            self._check_usable(live)
            if live.table is None:
                raise ApiError(409, "not_started", "table has not started")
            if perspective == "reviewer":
                if principal.role != "host":
                    raise ApiError(403, "reviewer_permission_required", "reviewer history requires the host credential")
                viewer = Viewer.reviewer()
            else:
                viewer = principal.viewer
            return table_history_view(live.table, viewer).model_dump(mode="json")

    # -------------------------------------------------------------- actions

    def register_action(self, table_id: str) -> int:
        """Timestamp an arriving action and mark it in flight, atomically with respect to timeouts.

        The timeout check takes the same lock, so an action counts as on time exactly
        when it registered no later than the deadline, however long it then queues.
        """
        live = self._get(table_id)
        with self._inflight_lock:
            received_ns = self.clock.now_ns()
            live.inflight.append(received_ns)
        return received_ns

    def submit_action(self, table_id: str, principal: Principal, raw: bytes,
                      received_ns: int | None = None, *, registered: bool = False) -> dict[str, Any]:
        """Apply one human action; pass ``registered=True`` with the value from :meth:`register_action`."""
        live = self._get(table_id)
        if not registered:
            with self._inflight_lock:
                if received_ns is None:
                    received_ns = self.clock.now_ns()
                live.inflight.append(received_ns)
        assert received_ns is not None
        try:
            with live.queue:
                self._check_usable(live)
                return self._submit(live, principal, raw, received_ns)
        finally:
            with self._inflight_lock:
                live.inflight.remove(received_ns)

    def _submit(self, live: LiveTable, principal: Principal, raw: bytes, received_ns: int) -> dict[str, Any]:
        seat = self._require_seat(principal)
        if live.table is None:
            raise ApiError(409, "not_started", "table has not started")
        try:
            envelope = ActionEnvelope.model_validate_json(raw)
        except ValidationError as exc:
            raise ApiError(422, "invalid_schema", "action envelope rejected", validation_details(exc)) from None
        if envelope.table_id != live.table_id:
            raise ApiError(422, "table_mismatch", "envelope table_id does not match the URL")
        fingerprint = canonical_json(envelope.model_dump(mode="json"))
        prior = live.receipts.get(envelope.action_id)
        if prior is not None:
            if prior.seat == seat and prior.fingerprint == fingerprint:
                return {"receipt": prior.body, "replayed": True, "snapshot": self._snapshot(live, principal)}
            raise ApiError(409, "action_id_conflict", "action_id already used with a different payload")
        # A deadline that expired before this request arrived is applied first.
        if self._expire(live, received_ns):
            self._publish(live)
        if envelope.decision_id in live.timed_out:
            raise ApiError(409, "deadline_passed", "the decision timed out before this action arrived")
        table = live.table
        try:
            receipt = table.submit(
                seat,
                envelope.action,
                hand_id=envelope.hand_id,
                decision_id=envelope.decision_id,
                expected_state_version=envelope.expected_state_version,
            )
        except ActionRejected as exc:
            raise ApiError(REJECTION_STATUS.get(exc.code, 409), exc.code, str(exc)) from None
        now = self.clock.now_ns()
        self._schedule(live, now)
        live.version += 1
        body = {
            "action_id": envelope.action_id,
            "table_id": live.table_id,
            "seat": seat,
            "hand_id": receipt.hand_id,
            "decision_id": receipt.decision_id,
            "action": receipt.action.model_dump(mode="json", exclude_none=True),
            "state_version": receipt.state_version,
            "version": live.version,
        }
        self._commit(live, "action", receipt=ReceiptRow(envelope.action_id, seat, receipt.decision_id, fingerprint,
                                                        body, live.version))
        live.receipts[envelope.action_id] = StoredReceipt(seat, fingerprint, body)
        self._run_due_bots(live, now)
        self._publish(live)
        return {"receipt": body, "replayed": False, "snapshot": self._snapshot(live, principal)}

    # ----------------------------------------------------- scheduling loop

    def _schedule(self, live: LiveTable, now: int, *, remaining_ns: int | None = None) -> None:
        """Set the clock for the current decision: a human deadline or a (paced) bot move."""
        table = live.table
        assert table is not None
        if table.complete:
            live.status = "complete"
            live.deadline_ns = live.bot_due_ns = None
            live.scheduled_decision = None
            return
        if live.scheduled_decision == table.decision_id:
            return
        seat = table.actor_seat
        assert seat is not None and table.decision_id is not None
        hand_id = table.hands[-1].hand_id
        new_hand = live.scheduled_hand not in (None, hand_id)
        live.scheduled_decision, live.scheduled_hand = table.decision_id, hand_id
        if live.slots[seat].kind == "human":
            turn_ns = live.config.human_turn_ms * 1_000_000
            live.deadline_ns = now + (turn_ns if remaining_ns is None else max(0, min(remaining_ns, turn_ns)))
            live.bot_due_ns = None
            return
        live.deadline_ns = None
        live.bot_due_ns = now + self.bot_delay_ns + (self.hand_pause_ns if new_hand else 0)

    def _run_due_bots(self, live: LiveTable, now: int) -> int:
        commits = 0
        while live.status == "running" and live.bot_due_ns is not None and live.bot_due_ns <= now:
            self._bot_act(live, now)
            commits += 1
        return commits

    def _bot_act(self, live: LiveTable, now: int) -> None:
        table = live.table
        assert table is not None
        seat = table.actor_seat
        assert seat is not None
        obs = observation(table, seat)
        policy = BOT_POLICIES[live.slots[seat].policy or ""]
        try:
            action = policy.decide(obs, live.bot_rngs[seat])
            table.submit(seat, action, hand_id=obs.hand_id, decision_id=obs.decision_id,
                         expected_state_version=obs.state_version)
        except ActionRejected as exc:
            table.apply_fallback(seat, "invalid_action", {"code": exc.code})
        except Exception as exc:  # noqa: BLE001 - any bot failure consumes the decision once
            table.apply_fallback(seat, "bot_error", {"error": type(exc).__name__})
        live.bot_decisions[seat] += 1
        live.dirty_bots.add(seat)
        live.bot_due_ns = None
        self._schedule(live, now)
        live.version += 1
        self._commit(live, "bot")

    def _expire(self, live: LiveTable, now: int) -> bool:
        """Apply the timeout fallback if the human deadline is strictly before ``now``."""
        table = live.table
        if table is None or table.complete or live.deadline_ns is None or now <= live.deadline_ns:
            return False
        seat, decision = table.actor_seat, table.decision_id
        assert seat is not None and decision is not None
        table.apply_fallback(seat, "timeout", None)
        live.timed_out.add(decision)
        live.deadline_ns = None
        self._schedule(live, max(now, self.clock.now_ns()))
        live.version += 1
        self._commit(live, "timeout")
        return True

    def pump(self, now: int | None = None) -> int:
        """Run due bot moves and human timeouts, heartbeat, publish. Returns the number of commits."""
        mono = self.clock.now_ns() if now is None else now
        if mono - self._last_heartbeat_ns >= self.heartbeat_ns:
            self.heartbeat(mono)
        if self.outage_since is not None:
            self._try_restore()
            if self.outage_since is not None:
                return 0
        commits = 0
        with self._registry:
            tables = list(self.tables.values())
        for live in tables:
            if live.status != "running":
                continue
            with live.queue:
                if live.fenced or live.dirty or live.halted_reason is not None or self.outage_since is not None:
                    continue
                try:
                    commits += self._pump_table(live, now)
                except ApiError:
                    continue
                self._publish(live)
        self._flush_published()
        return commits

    def _pump_table(self, live: LiveTable, now: int | None) -> int:
        commits = 0
        while live.status == "running":
            with self._inflight_lock:
                t = self.clock.now_ns() if now is None else now
                on_time_pending = live.deadline_ns is not None and any(r <= live.deadline_ns for r in live.inflight)
            if live.deadline_ns is not None:
                if on_time_pending or not self._expire(live, t):
                    break
            elif live.bot_due_ns is not None and t >= live.bot_due_ns:
                self._bot_act(live, t)
            else:
                break
            commits += 1
        return commits

    # ------------------------------------------------------- durable commits

    def _fault(self, point: str, live: LiveTable, kind: str) -> None:
        hook = self.faults.get(point)
        if hook is not None:
            hook(live.table_id, kind)

    def _scheduler(self, live: LiveTable, now: int) -> dict[str, Any] | None:
        table = live.table
        if live.status != "running" or table is None or table.decision_id is None:
            return None
        seat = table.actor_seat
        assert seat is not None
        kind = live.slots[seat].kind
        remaining = None
        if kind == "human" and live.deadline_ns is not None:
            remaining = max(0, (live.deadline_ns - now) // 1_000_000)
        return {"decision_id": table.decision_id, "seat": seat, "kind": kind, "remaining_ms": remaining}

    def _batch(self, live: LiveTable, kind: str, *, receipt: ReceiptRow | None,
               participants: Iterable[ParticipantRow], pauses: Iterable[PauseRow]) -> CommitBatch:
        table = live.table
        new_events = table.log[live.persisted_seq:] if table is not None else []
        hands: list[HandRow] = []
        awards: list[AwardRow] = []
        for e in new_events:
            if e.type == "HandStarted":
                row = HandRow(e.hand_id or "", e.data["hand_number"], e.seq, e.data["start_hash"])
                live.hand_rows[row.hand_id] = row
                hands.append(row)
            elif e.type == "HandCompleted":
                row = live.hand_rows[e.hand_id or ""]
                row.end_seq, row.final_stacks = e.seq, {str(s): v for s, v in e.data["final_stacks"].items()}
                if row not in hands:
                    hands.append(row)
            elif e.type == "TableScored":
                awards = [AwardRow(p["seat"], list(p["positions"]), float(p["points"]), p["method"], p["hand_number"])
                          for p in e.data["placements"]]
        snapshots = []
        if table is not None:
            engine = table.log[0].data["engine"]
            for p in live.pending_points:
                snapshots.append(SnapshotRow(p["hand_number"], p["seq"], table.log[p["seq"]].data["start_hash"],
                                             p, dict(engine)))
        checkpoints = []
        for seat in sorted(live.dirty_bots):
            policy = BOT_POLICIES[live.slots[seat].policy or ""]
            checkpoints.append(CheckpointRow(seat, policy.id, policy.artifact_hash, live.bot_rngs[seat].getstate(),
                                             live.bot_decisions[seat], live.version))
        row = TableRow(
            table_id=live.table_id,
            config=live.config.resolved(),
            config_hash=live.config.config_hash(),
            deck_seed=live.seed,
            bot_seed=live.bot_seed,
            host_token_hash=live.host_token_hash,
            status=live.status,
            version=live.version,
            state_version=table.state_version if table is not None else 0,
            event_count=len(table.log) if table is not None else 0,
            scheduler=self._scheduler(live, self.clock.now_ns()),
        )
        return CommitBatch(
            table=row,
            expected_version=live.committed_version or None,
            kind=kind,
            events=[e.to_json_obj() for e in new_events],
            participants=list(participants),
            hands=hands,
            snapshots=snapshots,
            receipt=receipt,
            checkpoints=checkpoints,
            awards=awards,
            pauses=list(pauses),
        )

    def _commit(self, live: LiveTable, kind: str, *, receipt: ReceiptRow | None = None,
                participants: Iterable[ParticipantRow] = (), pauses: Iterable[PauseRow] = ()) -> None:
        """Write everything this command changed, or discard the in-memory change and raise."""
        batch = self._batch(live, kind, receipt=receipt, participants=participants, pauses=pauses)
        self._fault("before_tx", live, kind)
        try:
            self.store.commit(batch, self.writer_id, self.lease_s,
                              in_tx=lambda: self._fault("in_tx", live, kind))
        except OwnershipLost:
            live.fenced = True
            raise ApiError(409, "writer_fenced", "this server no longer owns the table") from None
        except StoreUnavailable:
            self._outage()
            live.dirty = live.committed_version > 0
            raise storage_unavailable() from None
        except StoreError:
            live.dirty = live.committed_version > 0
            if live.dirty:
                self._reload(live, reason="commit_rejected")
            raise ApiError(500, "commit_rejected", "the store rejected the change; the table was reloaded") from None
        self._last_ok_ns = self.clock.now_ns()
        live.committed_version = live.version
        if live.table is not None:
            live.persisted_seq = len(live.table.log)
        live.pending_points.clear()
        live.dirty_bots.clear()
        live.unpublished.append(live.version)
        self._fault("after_commit", live, kind)

    # ------------------------------------------------------------- outbox

    def _publish(self, live: LiveTable) -> None:
        """Broadcast committed outbox versions as each subscriber's own projection."""
        if not live.unpublished:
            return
        if live.halted_reason is not None:
            live.unpublished.clear()
            return
        self._fault("before_broadcast", live, "publish")
        for sub in list(live.subscribers):
            sub.push(self.frame(live, sub.principal))
        live.published_version = max(live.published_version, *live.unpublished)
        live.unpublished.clear()

    def _flush_published(self) -> None:
        with self._registry:
            tables = list(self.tables.values())
        for live in tables:
            if live.published_version > live.marked_version and not live.fenced:
                try:
                    self.store.mark_published(live.table_id, live.published_version)
                except StoreUnavailable:
                    self._outage()
                    return
                live.marked_version = live.published_version

    # --------------------------------------------------- ownership / health

    def heartbeat(self, now: int | None = None) -> None:
        """Renew this writer's leases and record each pending human decision's remaining time."""
        now = self.clock.now_ns() if now is None else now
        self._last_heartbeat_ns = now
        schedulers: dict[str, dict[str, Any] | None] = {}
        with self._registry:
            tables = list(self.tables.values())
        for live in tables:
            if live.fenced or live.dirty or live.halted_reason is not None or live.status == "complete":
                continue
            schedulers[live.table_id] = self._scheduler(live, now)
        if self.outage_since is not None:
            return
        try:
            lost = self.store.heartbeat(self.writer_id, self.lease_s, schedulers)
        except StoreUnavailable:
            self._outage()
            return
        self._last_ok_ns = now
        for table_id in lost:
            self.tables[table_id].fenced = True

    def _outage(self) -> None:
        if self.outage_since is None:
            stale_s = max(0, self.clock.now_ns() - self._last_ok_ns) / 1e9
            self.outage_since = time.time() - stale_s

    def _try_restore(self) -> None:
        """After an outage, reload every table from the store and give humans back their think time."""
        try:
            self.store.ping()
        except StoreUnavailable:
            return
        since = self.outage_since
        with self._registry:
            tables = list(self.tables.values())
        # Until each table is reloaded its deadlines are stale; keep it closed so no request expires them.
        for live in tables:
            live.dirty = True
        self.outage_since = None
        for live in tables:
            if live.fenced or live.halted_reason is not None:
                continue
            with live.queue:
                try:
                    self._reload(live, reason="storage_outage", paused_at=since)
                except ApiError:
                    return

    def _reload(self, live: LiveTable, *, reason: str, paused_at: float | None = None) -> None:
        try:
            stored = self.store.load(live.table_id)
        except StoreUnavailable:
            self._outage()
            live.dirty = True
            raise storage_unavailable() from None
        if stored is None:
            raise ApiError(404, "table_not_found", "no such table")
        fresh = self._load(stored, reason=reason, paused_at=paused_at)
        keep = {"subscribers", "inflight", "queue"}
        for name, value in fresh.__dict__.items():
            if name not in keep:
                setattr(live, name, value)
        if live.table is not None:
            live.table.on_hand_start = live.capture_point
        self._resume(live, stored, reason=reason, paused_at=paused_at)
        self._publish(live)

    def recover_all(self, *, force: bool = False) -> list[str]:
        """Load every stored table this writer can claim. Returns the ids that were halted."""
        halted = []
        for table_id in self.store.table_ids():
            claimed = self.store.claim(table_id, self.writer_id, self.lease_s, force=force)
            stored = self.store.load(table_id)
            if stored is None:
                continue
            live = self._load(stored, reason="process_restart")
            if not claimed:
                live.fenced = True
            self._register(live)
            if live.halted_reason is not None:
                halted.append(table_id)
                continue
            if claimed:
                with live.queue:
                    try:
                        self._resume(live, stored, reason="process_restart")
                        self._publish(live)
                    except ApiError:
                        pass
        self._flush_published()
        return halted

    def _load(self, stored: StoredTable, *, reason: str, paused_at: float | None = None) -> LiveTable:
        t = stored.table
        config = TournamentConfig.model_validate(t.config)
        slots = {p.seat: SeatSlot(p.seat, p.kind, p.policy, p.claimed, p.token_hash)  # type: ignore[arg-type]
                 for p in stored.participants}
        live = LiveTable(t.table_id, config, slots, t.deck_seed, bot_seed=t.bot_seed,
                         host_token_hash=t.host_token_hash, version=t.version, committed_version=t.version)
        live.receipts = {r.action_id: StoredReceipt(r.seat, r.fingerprint, r.body) for r in stored.receipts}
        live.unpublished = list(stored.unpublished)
        if t.status == "halted":
            live.halted_reason = t.halted_reason or "halted"
            return live
        live.status = "waiting" if t.status == "waiting" else "running"
        if t.status == "waiting":
            return live
        try:
            if config.config_hash() != t.config_hash:
                raise ReplayMismatch(0, "stored config does not match its config_hash")
            events = [EventEnvelope.from_json_obj(e) for e in stored.events]
            if len(events) != t.event_count:
                raise ReplayMismatch(len(events), "stored event count differs from the table row")
            point = stored.snapshot.point if stored.snapshot is not None else None
            table = rebuild_table(config, events, point, seed=t.deck_seed)
            if table.state_version != t.state_version:
                raise ReplayMismatch(len(events), "rebuilt state version differs from the table row")
            if table.complete != (t.status == "complete"):
                raise ReplayMismatch(len(events), "rebuilt completion differs from the table row")
            for c in stored.checkpoints:
                policy = BOT_POLICIES.get(c.policy)
                if policy is None or policy.artifact_hash != c.artifact_hash or slots[c.seat].policy != c.policy:
                    raise ReplayMismatch(len(events), f"bot artifact for seat {c.seat} changed since the checkpoint")
                live.bot_rngs[c.seat] = SeededRng.from_state(c.rng_state)
                live.bot_decisions[c.seat] = c.decisions
            missing = {s for s, slot in slots.items() if slot.kind == "bot"} - set(live.bot_rngs)
            if missing:
                raise ReplayMismatch(len(events), f"no bot checkpoint for seats {sorted(missing)}")
        except (ReplayMismatch, ReplayIncompatible, ValueError) as exc:
            live.halted_reason = f"{type(exc).__name__}: {exc}"
            try:
                self.store.halt(t.table_id, live.halted_reason)
            except StoreError:
                pass
            return live
        live.table = table
        table.on_hand_start = live.capture_point
        live.persisted_seq = len(events)
        decision = None
        for e in events:
            if e.type == "DecisionRequested":
                decision = e.data["decision_id"]
            elif e.type == "ActionFallback" and e.data["reason"] == "timeout" and decision is not None:
                live.timed_out.add(decision)
            if e.type == "HandStarted":
                live.hand_rows[e.hand_id or ""] = HandRow(e.hand_id or "", e.data["hand_number"], e.seq,
                                                          e.data["start_hash"])
            elif e.type == "HandCompleted":
                row = live.hand_rows[e.hand_id or ""]
                row.end_seq, row.final_stacks = e.seq, {str(s): v for s, v in e.data["final_stacks"].items()}
        if table.complete:
            live.status = "complete"
        return live

    def _resume(self, live: LiveTable, stored: StoredTable, *, reason: str, paused_at: float | None = None) -> None:
        """Reschedule a recovered running table and record the pause it just came out of."""
        table = live.table
        if live.status != "running" or table is None or live.halted_reason is not None:
            return
        now = self.clock.now_ns()
        wall = time.time()
        decision = table.decision_id
        seat = table.actor_seat
        assert decision is not None and seat is not None
        human = live.slots[seat].kind == "human"
        sched = stored.table.scheduler or {}
        pauses: list[PauseRow] = []
        remaining_ms: int | None = None
        open_pause = stored.open_pause
        if open_pause is not None:
            pauses.append(replace(open_pause, resumed_at=wall))
            if open_pause.decision_id == decision:
                remaining_ms = open_pause.remaining_ms
        if remaining_ms is None and sched.get("decision_id") == decision:
            remaining_ms = sched.get("remaining_ms")
        if remaining_ms is None and human:
            remaining_ms = live.config.human_turn_ms
        if open_pause is None or open_pause.decision_id != decision:
            started = paused_at if paused_at is not None else wall - (stored.scheduler_age_s or 0.0)
            pauses.append(PauseRow(reason, decision, remaining_ms if human else None, started, wall))
        live.scheduled_decision = None
        live.scheduled_hand = table.hands[-1].hand_id
        self._schedule(live, now, remaining_ns=remaining_ms * 1_000_000 if human and remaining_ms is not None
                       else None)
        live.version += 1
        self._commit(live, "resume", pauses=pauses)
        self._run_due_bots(live, now)

    def close(self) -> None:
        """Graceful shutdown: record a pause for every pending human decision so its time is restored."""
        wall = time.time()
        now = self.clock.now_ns()
        with self._registry:
            tables = list(self.tables.values())
        for live in tables:
            with live.queue:
                if live.fenced or live.dirty or live.halted_reason is not None or live.status != "running":
                    continue
                sched = self._scheduler(live, now)
                if sched is None or sched["kind"] != "human":
                    continue
                try:
                    self.store.record_pause(live.table_id, self.writer_id,
                                            PauseRow("shutdown", sched["decision_id"], sched["remaining_ms"], wall),
                                            sched)
                except StoreError:
                    pass

    # ------------------------------------------------------------ streaming

    def subscribe(self, table_id: str, principal: Principal, loop: asyncio.AbstractEventLoop) -> Subscriber:
        live = self._get(table_id)
        with live.queue:
            self._check_usable(live)
            sub = Subscriber(principal, loop)
            live.subscribers.append(sub)
            sub.push(self.frame(live, principal))
            return sub

    def unsubscribe(self, table_id: str, sub: Subscriber) -> None:
        with self._registry:
            live = self.tables.get(table_id)
        if live is not None:
            with live.queue:
                if sub in live.subscribers:
                    live.subscribers.remove(sub)

    def frame(self, live: LiveTable, principal: Principal) -> dict[str, Any]:
        return {"type": "snapshot", **self._snapshot(live, principal)}
