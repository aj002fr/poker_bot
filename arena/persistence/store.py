"""The store protocol and its in-process implementation.

A store owns durability and table ownership. The command writer hands it one
:class:`CommitBatch` per command; the store applies the whole batch or none of
it, and only if the writer still owns the table at the version it last wrote.
That compare-and-set is the fencing available today: a second writer that
took the table over makes every later commit from the old writer fail with
:class:`OwnershipLost`.

:class:`MemoryStore` keeps the same semantics in process memory, for fast tests
and ``make dev``. It stores JSON round-tripped copies so rows read back from it
look exactly like rows read back from Postgres.
"""

from __future__ import annotations

import copy
import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from arena.persistence.models import (
    AwardRow,
    CheckpointRow,
    CommitBatch,
    HandRow,
    NoteRow,
    ParticipantRow,
    PauseRow,
    ReceiptRow,
    SnapshotRow,
    StoredTable,
    TableRow,
)


class StoreError(Exception):
    pass


class StoreUnavailable(StoreError):
    """The store could not be reached; whether the last commit landed is unknown."""


class OwnershipLost(StoreError):
    """Another writer owns the table, or it moved past the version this writer knows."""


class Store(Protocol):
    kind: str

    def ping(self) -> None: ...

    def commit(self, batch: CommitBatch, writer_id: str, lease_s: float,
               in_tx: Callable[[], None] | None = None) -> None: ...

    def load(self, table_id: str) -> StoredTable | None: ...

    def table_ids(self) -> list[str]: ...

    def claim(self, table_id: str, writer_id: str, lease_s: float, *, force: bool = False) -> bool: ...

    def heartbeat(self, writer_id: str, lease_s: float, schedulers: dict[str, dict[str, Any] | None]) -> set[str]:
        """Renew leases (and pending-decision clocks); return the tables this writer no longer owns."""
        ...

    def record_pause(self, table_id: str, writer_id: str, pause: PauseRow,
                     scheduler: dict[str, Any] | None) -> None: ...

    def halt(self, table_id: str, reason: str) -> None: ...

    def mark_published(self, table_id: str, up_to_version: int) -> None: ...

    def close(self) -> None: ...


def _j(value: Any) -> Any:
    return json.loads(json.dumps(value))


@dataclass
class _Outbox:
    version: int
    kind: str
    published: bool = False


@dataclass
class _MemTable:
    table: TableRow
    owner_id: str | None
    lease_until: float
    scheduler_at: float
    participants: dict[int, ParticipantRow] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    hands: dict[str, HandRow] = field(default_factory=dict)
    snapshots: dict[int, SnapshotRow] = field(default_factory=dict)
    receipts: dict[str, ReceiptRow] = field(default_factory=dict)
    checkpoints: dict[int, CheckpointRow] = field(default_factory=dict)
    awards: dict[int, AwardRow] = field(default_factory=dict)
    notes: dict[str, NoteRow] = field(default_factory=dict)
    pauses: list[PauseRow] = field(default_factory=list)
    outbox: list[_Outbox] = field(default_factory=list)


class MemoryStore:
    kind = "memory"

    def __init__(self, now_s: Callable[[], float] = time.monotonic) -> None:
        self._now = now_s
        self._lock = threading.Lock()
        self._tables: dict[str, _MemTable] = {}
        self._pause_ids = 0
        self.available = True
        """Set False to simulate an outage: every call raises :class:`StoreUnavailable`."""
        self.lose_next_ack = False
        """Apply the next commit, then report it as failed (an ambiguous commit)."""

    def _check(self) -> None:
        if not self.available:
            raise StoreUnavailable("memory store marked unavailable")

    def ping(self) -> None:
        self._check()

    def commit(self, batch: CommitBatch, writer_id: str, lease_s: float,
               in_tx: Callable[[], None] | None = None) -> None:
        with self._lock:
            self._check()
            now = self._now()
            table_id = batch.table.table_id
            current = self._tables.get(table_id)
            if batch.expected_version is None:
                if current is not None:
                    raise StoreError(f"table {table_id} already exists")
                staged = _MemTable(replace(batch.table), writer_id, now + lease_s, now)
            else:
                if current is None:
                    raise StoreError(f"table {table_id} does not exist")
                if current.owner_id != writer_id or current.table.version != batch.expected_version:
                    raise OwnershipLost(table_id)
                staged = copy.copy(current)
                for name in ("participants", "hands", "snapshots", "receipts", "checkpoints", "awards", "notes"):
                    setattr(staged, name, dict(getattr(current, name)))
                staged.events = list(current.events)
                staged.pauses = list(current.pauses)
                staged.outbox = list(current.outbox)
            row = TableRow(**_j(batch.table.__dict__))
            if row.event_count != len(staged.events) + len(batch.events):
                raise StoreError("event_count does not match the appended events")
            for event in batch.events:
                if event["seq"] != len(staged.events):
                    raise StoreError("events must be appended in sequence")
                staged.events.append(_j(event))
            for p in batch.participants:
                if p.token_hash and any(o.token_hash == p.token_hash and o.seat != p.seat
                                        for o in staged.participants.values()):
                    raise StoreError("duplicate token hash")
                staged.participants[p.seat] = ParticipantRow(**_j(p.__dict__))
            for h in batch.hands:
                staged.hands[h.hand_id] = HandRow(**_j(h.__dict__))
            for s in batch.snapshots:
                staged.snapshots[s.hand_number] = SnapshotRow(**_j(s.__dict__))
            if batch.receipt is not None:
                r = batch.receipt
                if r.action_id in staged.receipts or any(o.decision_id == r.decision_id
                                                         for o in staged.receipts.values()):
                    raise StoreError("duplicate receipt")
                staged.receipts[r.action_id] = ReceiptRow(**_j(r.__dict__))
            for c in batch.checkpoints:
                staged.checkpoints[c.seat] = CheckpointRow(**_j(c.__dict__))
            for a in batch.awards:
                staged.awards[a.seat] = AwardRow(**_j(a.__dict__))
            for n in batch.notes:
                if n.decision_id in staged.notes:
                    raise StoreError("duplicate bot note")
                staged.notes[n.decision_id] = NoteRow(**_j(n.__dict__))
            for pause in batch.pauses:
                self._put_pause(staged, pause)
            if any(o.version == row.version for o in staged.outbox):
                raise StoreError("duplicate outbox version")
            staged.outbox.append(_Outbox(row.version, batch.kind))
            staged.table = row
            staged.owner_id = writer_id
            staged.lease_until = now + lease_s
            staged.scheduler_at = now
            if in_tx is not None:
                in_tx()
            self._tables[table_id] = staged
            if self.lose_next_ack:
                self.lose_next_ack = False
                raise StoreUnavailable("commit applied but the acknowledgement was lost")

    def _put_pause(self, staged: _MemTable, pause: PauseRow) -> None:
        if pause.pause_id is None:
            self._pause_ids += 1
            staged.pauses.append(replace(pause, pause_id=self._pause_ids))
        else:
            staged.pauses = [replace(pause) if p.pause_id == pause.pause_id else p for p in staged.pauses]

    def load(self, table_id: str) -> StoredTable | None:
        with self._lock:
            self._check()
            t = self._tables.get(table_id)
            if t is None:
                return None
            snapshot = t.snapshots[max(t.snapshots)] if t.snapshots else None
            open_pause = next((replace(p) for p in reversed(t.pauses) if p.resumed_at is None), None)
            return StoredTable(
                table=TableRow(**_j(t.table.__dict__)),
                owner_id=t.owner_id,
                participants=[ParticipantRow(**_j(p.__dict__)) for _, p in sorted(t.participants.items())],
                events=_j(t.events),
                snapshot=SnapshotRow(**_j(snapshot.__dict__)) if snapshot else None,
                receipts=[ReceiptRow(**_j(r.__dict__)) for r in t.receipts.values()],
                checkpoints=[CheckpointRow(**_j(c.__dict__)) for _, c in sorted(t.checkpoints.items())],
                awards=[AwardRow(**_j(a.__dict__)) for _, a in sorted(t.awards.items())],
                open_pause=open_pause,
                scheduler_age_s=self._now() - t.scheduler_at,
                unpublished=[o.version for o in t.outbox if not o.published],
                notes=[NoteRow(**_j(n.__dict__)) for _, n in sorted(t.notes.items())],
            )

    def table_ids(self) -> list[str]:
        with self._lock:
            self._check()
            return list(self._tables)

    def claim(self, table_id: str, writer_id: str, lease_s: float, *, force: bool = False) -> bool:
        with self._lock:
            self._check()
            t = self._tables.get(table_id)
            if t is None:
                return False
            now = self._now()
            if not force and t.owner_id not in (None, writer_id) and t.lease_until >= now:
                return False
            t.owner_id, t.lease_until = writer_id, now + lease_s
            return True

    def heartbeat(self, writer_id: str, lease_s: float, schedulers: dict[str, dict[str, Any] | None]) -> set[str]:
        with self._lock:
            self._check()
            now = self._now()
            lost = set()
            for table_id, scheduler in schedulers.items():
                t = self._tables.get(table_id)
                if t is None or t.owner_id != writer_id:
                    lost.add(table_id)
                    continue
                t.lease_until = now + lease_s
                if scheduler is not None:
                    t.table = replace(t.table, scheduler=_j(scheduler))
                    t.scheduler_at = now
            return lost

    def record_pause(self, table_id: str, writer_id: str, pause: PauseRow,
                     scheduler: dict[str, Any] | None) -> None:
        with self._lock:
            self._check()
            t = self._tables.get(table_id)
            if t is None or t.owner_id != writer_id:
                raise OwnershipLost(table_id)
            self._put_pause(t, pause)
            t.table = replace(t.table, scheduler=_j(scheduler))
            t.scheduler_at = self._now()

    def halt(self, table_id: str, reason: str) -> None:
        with self._lock:
            self._check()
            t = self._tables.get(table_id)
            if t is not None:
                t.table = replace(t.table, status="halted", halted_reason=reason)

    def mark_published(self, table_id: str, up_to_version: int) -> None:
        with self._lock:
            self._check()
            t = self._tables.get(table_id)
            if t is not None:
                for o in t.outbox:
                    if o.version <= up_to_version:
                        o.published = True

    def close(self) -> None:
        pass

    # Test inspection helpers.

    def pauses(self, table_id: str) -> list[PauseRow]:
        with self._lock:
            return [replace(p) for p in self._tables[table_id].pauses]

    def outbox(self, table_id: str) -> list[tuple[int, str, bool]]:
        with self._lock:
            return [(o.version, o.kind, o.published) for o in self._tables[table_id].outbox]
