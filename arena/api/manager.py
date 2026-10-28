"""Single-process, in-memory table manager behind the HTTP/WebSocket API.

One process and one lock mean one writer per table by construction. Every
mutation (join, start, human action, timeout fallback, bot decision) happens
under the lock, bumps the table's ``version`` and is then broadcast to each
subscriber as that subscriber's own projection. Nothing here renders state
directly: every view comes from :mod:`arena.table.projections`.

Persistence, receipts at rest and recovery are milestone 5; this layer keeps the
same API contract so they can slide in underneath.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import secrets
import threading
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, ValidationError

from arena.api.bots import BOT_POLICIES
from arena.api.clock import Clock, MonotonicClock
from arena.domain.canonical import canonical_json
from arena.domain.models import RULESET_IDS, ActionEnvelope, TournamentConfig
from arena.domain.rng import Rng
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


@dataclass
class LiveTable:
    table_id: str
    config: TournamentConfig
    slots: dict[int, SeatSlot]
    seed: str | None
    status: Literal["waiting", "running", "complete"] = "waiting"
    version: int = 0
    table: Table | None = None
    bot_rngs: dict[int, Rng] = field(default_factory=dict)
    scheduled_decision: str | None = None
    scheduled_hand: str | None = None
    deadline_ns: int | None = None
    bot_due_ns: int | None = None
    timed_out: set[str] = field(default_factory=set)
    receipts: dict[str, StoredReceipt] = field(default_factory=dict)
    subscribers: list[Subscriber] = field(default_factory=list)
    inflight: list[int] = field(default_factory=list)
    results: dict[str, Any] | None = None

    def setup_public(self, viewer_seat: int | None) -> dict[str, Any]:
        return {
            "config": self.config.resolved(),
            "config_hash": self.config.config_hash(),
            "seats": [self.slots[s].public(viewer_seat) for s in sorted(self.slots)],
        }


def _token_key(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def policy_catalog() -> list[dict[str, Any]]:
    return [
        {"id": p.id, "label": p.label, "description": p.description, "artifact_hash": p.artifact_hash,
         "rulesets": list(RULESET_IDS)}
        for p in BOT_POLICIES.values()
    ]


class TableManager:
    def __init__(self, clock: Clock | None = None, *, bot_delay_ms: int = 700, hand_pause_ms: int = 1500) -> None:
        self.clock = clock or MonotonicClock()
        self.bot_delay_ns = int(bot_delay_ms * 1_000_000)
        self.hand_pause_ns = int(hand_pause_ms * 1_000_000)
        self.tables: dict[str, LiveTable] = {}
        self._tokens: dict[str, Principal] = {}
        self._lock = threading.RLock()
        self._inflight_lock = threading.Lock()

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

    def _issue(self, principal: Principal) -> str:
        token = secrets.token_urlsafe(24)
        self._tokens[_token_key(token)] = principal
        return token

    def _get(self, table_id: str) -> LiveTable:
        live = self.tables.get(table_id)
        if live is None:
            raise ApiError(404, "table_not_found", "no such table")
        return live

    def authenticate(self, table_id: str, token: str | None) -> Principal:
        with self._lock:
            self._get(table_id)
            if not token:
                return ANONYMOUS
            key = _token_key(token)
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

    # --------------------------------------------------------------- lobby

    def create_table(self, body: CreateTableRequest) -> dict[str, Any]:
        config = self.build_config(body.config, len(body.seats))
        self._validate_seats(body.seats)
        with self._lock:
            table_id = "t_" + secrets.token_hex(6)
            slots = {
                seat: SeatSlot(seat, s.kind, s.policy)
                for seat, s in zip(config.resolved_seats, body.seats, strict=True)
            }
            live = LiveTable(table_id, config, slots, body.seed)
            self.tables[table_id] = live
            host = Principal("host", table_id)
            token = self._issue(host)
            live.version += 1
            return {"table_id": table_id, "host_token": token, "snapshot": self._snapshot(live, host)}

    def join(self, table_id: str, seat: int | None) -> dict[str, Any]:
        with self._lock:
            live = self._get(table_id)
            if live.status != "waiting":
                raise ApiError(409, "already_started", "seats can only be claimed before the table starts")
            open_seats = [s for s in sorted(live.slots) if live.slots[s].kind == "human" and not live.slots[s].claimed]
            if seat is None:
                if not open_seats:
                    raise ApiError(409, "no_open_seat", "every human seat is taken")
                seat = open_seats[0]
            elif seat not in open_seats:
                raise ApiError(409, "seat_unavailable", "that seat is not an open human seat")
            live.slots[seat].claimed = True
            principal = Principal("seat", table_id, seat)
            token = self._issue(principal)
            live.version += 1
            self._broadcast(live)
            return {"seat": seat, "seat_token": token, "snapshot": self._snapshot(live, principal)}

    def start(self, table_id: str, principal: Principal) -> dict[str, Any]:
        with self._lock:
            live = self._get(table_id)
            if principal.role != "host":
                raise ApiError(403, "host_credential_required", "only the table host can start the table")
            if live.status != "waiting":
                raise ApiError(409, "already_started", "table already started")
            unclaimed = [s for s, slot in live.slots.items() if slot.kind == "human" and not slot.claimed]
            if unclaimed:
                raise ApiError(409, "seats_unclaimed", f"{len(unclaimed)} human seat(s) not yet claimed")
            seats = list(live.config.resolved_seats)
            if live.seed is not None:
                live.table = seeded_table(live.config, live.seed, table_id=table_id)
                live.bot_rngs = policy_rngs(live.seed, seats)
            else:
                live.table = Table(live.config, table_id=table_id)
                live.bot_rngs = policy_rngs(secrets.token_hex(16), seats)
            live.status = "running"
            live.version += 1
            self._advance(live, self.clock.now_ns())
            self._broadcast(live)
            return self._snapshot(live, principal)

    # ------------------------------------------------------------ snapshots

    def snapshot(self, table_id: str, principal: Principal) -> dict[str, Any]:
        with self._lock:
            return self._snapshot(self._get(table_id), principal)

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
        with self._lock:
            live = self._get(table_id)
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

    def submit_action(self, table_id: str, principal: Principal, raw: bytes, received_ns: int) -> dict[str, Any]:
        """Apply one human action. ``received_ns`` is taken once the full request is in."""
        live = self._get(table_id)
        with self._inflight_lock:
            live.inflight.append(received_ns)
        try:
            with self._lock:
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
            self._broadcast(live)
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
        live.receipts[envelope.action_id] = StoredReceipt(seat, fingerprint, body)
        self._advance(live, self.clock.now_ns())
        self._broadcast(live)
        return {"receipt": body, "replayed": False, "snapshot": self._snapshot(live, principal)}

    # ----------------------------------------------------- scheduling loop

    def _advance(self, live: LiveTable, now: int) -> None:
        """Schedule the next decision: a human deadline, or a (paced) bot move."""
        table = live.table
        assert table is not None
        while not table.complete:
            if live.scheduled_decision == table.decision_id:
                return
            seat = table.actor_seat
            assert seat is not None and table.decision_id is not None
            hand_id = table.hands[-1].hand_id
            new_hand = live.scheduled_hand not in (None, hand_id)
            live.scheduled_decision, live.scheduled_hand = table.decision_id, hand_id
            if live.slots[seat].kind == "human":
                live.deadline_ns = now + live.config.human_turn_ms * 1_000_000
                live.bot_due_ns = None
                return
            live.deadline_ns = None
            delay = self.bot_delay_ns + (self.hand_pause_ns if new_hand else 0)
            if delay > 0:
                live.bot_due_ns = now + delay
                return
            self._bot_act(live, seat)
        live.status = "complete"
        live.deadline_ns = live.bot_due_ns = None
        live.scheduled_decision = None

    def _bot_act(self, live: LiveTable, seat: int) -> None:
        table = live.table
        assert table is not None
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
        live.bot_due_ns = None
        live.version += 1

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
        live.version += 1
        self._advance(live, now)
        return True

    def pump(self, now: int | None = None) -> int:
        """Run due bot moves and human timeouts. Returns the number of commits."""
        now = self.clock.now_ns() if now is None else now
        commits = 0
        with self._lock:
            for live in list(self.tables.values()):
                if live.status != "running":
                    continue
                changed = False
                while live.status == "running":
                    if live.deadline_ns is not None:
                        with self._inflight_lock:
                            on_time_pending = any(r <= live.deadline_ns for r in live.inflight)
                        if on_time_pending or not self._expire(live, now):
                            break
                    elif live.bot_due_ns is not None and now >= live.bot_due_ns:
                        seat = live.table.actor_seat if live.table else None
                        assert seat is not None
                        self._bot_act(live, seat)
                        self._advance(live, now)
                    else:
                        break
                    changed = True
                    commits += 1
                if changed:
                    self._broadcast(live)
        return commits

    # ------------------------------------------------------------ streaming

    def subscribe(self, table_id: str, principal: Principal, loop: asyncio.AbstractEventLoop) -> Subscriber:
        with self._lock:
            live = self._get(table_id)
            sub = Subscriber(principal, loop)
            live.subscribers.append(sub)
            sub.push(self.frame(live, principal))
            return sub

    def unsubscribe(self, table_id: str, sub: Subscriber) -> None:
        with self._lock:
            live = self.tables.get(table_id)
            if live is not None and sub in live.subscribers:
                live.subscribers.remove(sub)

    def frame(self, live: LiveTable, principal: Principal) -> dict[str, Any]:
        return {"type": "snapshot", **self._snapshot(live, principal)}

    def _broadcast(self, live: LiveTable) -> None:
        for sub in list(live.subscribers):
            sub.push(self.frame(live, sub.principal))
