"""PostgreSQL store (psycopg 3). Same semantics as :class:`MemoryStore`.

Each commit is one transaction. The table row is locked first (``FOR UPDATE``)
and checked against this writer's ownership and last version, so a fenced
writer appends nothing. The lease only decides when another writer may claim
the table; ownership itself is the fence. Unique keys on events, receipts and outbox rows
are a second line of defence against a double append.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from importlib import resources
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

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
from arena.persistence.store import OwnershipLost, StoreError, StoreUnavailable

MIGRATIONS_PACKAGE = "arena.persistence.migrations"
_TABLE_COLUMNS = ("config", "config_hash", "deck_seed", "bot_seed", "host_token_hash", "status", "version",
                  "state_version", "event_count", "scheduler", "halted_reason")


def _ts(epoch: float | None) -> datetime | None:
    return None if epoch is None else datetime.fromtimestamp(epoch, UTC)


def _epoch(ts: datetime | None) -> float | None:
    return None if ts is None else ts.timestamp()


def migrations() -> list[tuple[str, str]]:
    files = sorted(f for f in resources.files(MIGRATIONS_PACKAGE).iterdir() if f.name.endswith(".sql"))
    return [(f.name.removesuffix(".sql"), f.read_text()) for f in files]


def migrate(conn: psycopg.Connection) -> list[str]:
    """Apply pending migrations in order, each in its own transaction. Returns the applied names."""
    with conn.transaction():
        conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations "
                     "(name text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())")
    applied = []
    for name, sql in migrations():
        with conn.transaction():
            conn.execute("SELECT pg_advisory_xact_lock(hashtext('arena-migrations'))")
            if conn.execute("SELECT 1 FROM schema_migrations WHERE name = %s", (name,)).fetchone():
                continue
            conn.execute(sql)
            conn.execute("INSERT INTO schema_migrations (name) VALUES (%s)", (name,))
            applied.append(name)
    return applied


class PostgresStore:
    kind = "postgres"

    def __init__(self, dsn: str, *, min_size: int = 1, max_size: int = 8, timeout_s: float = 5.0) -> None:
        self.dsn = dsn
        self._pool = ConnectionPool(dsn, min_size=min_size, max_size=max_size, timeout=timeout_s, open=False,
                                    kwargs={"autocommit": True, "connect_timeout": max(1, int(timeout_s))})
        try:
            self._pool.open(wait=True, timeout=timeout_s)
        except Exception as exc:  # noqa: BLE001 - surface as an availability error
            self._pool.close()
            raise StoreUnavailable(f"cannot reach Postgres: {type(exc).__name__}") from None
        with self._conn() as conn:
            migrate(conn)

    def _conn(self):
        return _Connection(self._pool)

    def ping(self) -> None:
        with self._conn() as conn:
            conn.execute("SELECT 1")

    # -------------------------------------------------------------- commits

    def commit(self, batch: CommitBatch, writer_id: str, lease_s: float,
               in_tx: Callable[[], None] | None = None) -> None:
        row = batch.table
        with self._conn() as conn, conn.transaction():
            if batch.expected_version is None:
                conn.execute(
                    f"INSERT INTO arena_tables (table_id, {', '.join(_TABLE_COLUMNS)}, scheduler_at, owner_id, "
                    f"lease_until) VALUES (%s, {', '.join(['%s'] * len(_TABLE_COLUMNS))}, now(), %s, "
                    "now() + make_interval(secs => %s))",
                    (row.table_id, *self._table_values(row), writer_id, lease_s),
                )
                base_events = 0
            else:
                current = conn.execute(
                    "SELECT version, owner_id, event_count FROM arena_tables WHERE table_id = %s FOR UPDATE", (row.table_id,),
                ).fetchone()
                if current is None:
                    raise StoreError(f"table {row.table_id} does not exist")
                if current["owner_id"] != writer_id or current["version"] != batch.expected_version:
                    raise OwnershipLost(row.table_id)
                base_events = current["event_count"]
                sets = ", ".join(f"{c} = %s" for c in _TABLE_COLUMNS)
                conn.execute(
                    f"UPDATE arena_tables SET {sets}, scheduler_at = now(), updated_at = now(), "
                    "lease_until = now() + make_interval(secs => %s) WHERE table_id = %s",
                    (*self._table_values(row), lease_s, row.table_id),
                )
            if row.event_count != base_events + len(batch.events):
                raise StoreError("event_count does not match the appended events")
            if [e["seq"] for e in batch.events] != list(range(base_events, row.event_count)):
                raise StoreError("events must be appended in sequence")
            tid = row.table_id
            if batch.events:
                with conn.cursor() as cur:
                    cur.executemany(
                        "INSERT INTO events (table_id, seq, hand_id, schema_version, type, data) "
                        "VALUES (%s, %s, %s, %s, %s, %s)",
                        [(tid, e["seq"], e["hand_id"], e["schema_version"], e["type"], Jsonb(e["data"]))
                         for e in batch.events],
                    )
            for p in batch.participants:
                conn.execute(
                    "INSERT INTO participants (table_id, seat, kind, policy, artifact_hash, claimed, token_hash) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT (table_id, seat) DO UPDATE SET "
                    "kind = EXCLUDED.kind, policy = EXCLUDED.policy, artifact_hash = EXCLUDED.artifact_hash, "
                    "claimed = EXCLUDED.claimed, token_hash = EXCLUDED.token_hash",
                    (tid, p.seat, p.kind, p.policy, p.artifact_hash, p.claimed, p.token_hash),
                )
            for h in batch.hands:
                conn.execute(
                    "INSERT INTO hands (table_id, hand_id, hand_number, start_seq, start_hash, end_seq, final_stacks) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT (table_id, hand_id) DO UPDATE SET "
                    "end_seq = EXCLUDED.end_seq, final_stacks = EXCLUDED.final_stacks",
                    (tid, h.hand_id, h.hand_number, h.start_seq, h.start_hash, h.end_seq,
                     Jsonb(h.final_stacks) if h.final_stacks is not None else None),
                )
            for s in batch.snapshots:
                conn.execute(
                    "INSERT INTO snapshots (table_id, hand_number, seq, start_hash, point, engine) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    (tid, s.hand_number, s.seq, s.start_hash, Jsonb(s.point), Jsonb(s.engine)),
                )
            if batch.receipt is not None:
                r = batch.receipt
                conn.execute(
                    "INSERT INTO action_receipts (table_id, action_id, seat, decision_id, fingerprint, body, version) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (tid, r.action_id, r.seat, r.decision_id, r.fingerprint, Jsonb(r.body), r.version),
                )
            for c in batch.checkpoints:
                conn.execute(
                    "INSERT INTO bot_checkpoints (table_id, seat, policy, artifact_hash, rng_state, decisions, version) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT (table_id, seat) DO UPDATE SET "
                    "rng_state = EXCLUDED.rng_state, decisions = EXCLUDED.decisions, version = EXCLUDED.version",
                    (tid, c.seat, c.policy, c.artifact_hash, Jsonb(c.rng_state), c.decisions, c.version),
                )
            for a in batch.awards:
                conn.execute(
                    "INSERT INTO awards (table_id, seat, positions, points, method, hand_number) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    (tid, a.seat, Jsonb(a.positions), a.points, a.method, a.hand_number),
                )
            for pause in batch.pauses:
                self._put_pause(conn, tid, pause)
            conn.execute("INSERT INTO outbox (table_id, version, kind) VALUES (%s, %s, %s)",
                         (tid, row.version, batch.kind))
            if in_tx is not None:
                in_tx()

    @staticmethod
    def _table_values(row: TableRow) -> tuple[Any, ...]:
        return (Jsonb(row.config), row.config_hash, row.deck_seed, row.bot_seed, row.host_token_hash, row.status,
                row.version, row.state_version, row.event_count,
                Jsonb(row.scheduler) if row.scheduler is not None else None, row.halted_reason)

    @staticmethod
    def _put_pause(conn, table_id: str, pause: PauseRow) -> None:
        if pause.pause_id is None:
            conn.execute(
                "INSERT INTO table_pauses (table_id, reason, decision_id, remaining_ms, paused_at, resumed_at) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (table_id, pause.reason, pause.decision_id, pause.remaining_ms, _ts(pause.paused_at),
                 _ts(pause.resumed_at)),
            )
        else:
            conn.execute("UPDATE table_pauses SET resumed_at = %s WHERE pause_id = %s AND table_id = %s",
                         (_ts(pause.resumed_at), pause.pause_id, table_id))

    # ---------------------------------------------------------------- reads

    def load(self, table_id: str) -> StoredTable | None:
        with self._conn() as conn, conn.transaction():
            conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            t = conn.execute(
                "SELECT *, extract(epoch FROM now() - scheduler_at)::float8 AS scheduler_age "
                "FROM arena_tables WHERE table_id = %s", (table_id,),
            ).fetchone()
            if t is None:
                return None
            table = TableRow(table_id=table_id, **{c: t[c] for c in _TABLE_COLUMNS})
            participants = [ParticipantRow(**{k: r[k] for k in ParticipantRow.__dataclass_fields__})
                            for r in conn.execute("SELECT * FROM participants WHERE table_id = %s ORDER BY seat",
                                                  (table_id,))]
            events = [
                {"schema_version": r["schema_version"], "table_id": table_id, "hand_id": r["hand_id"],
                 "seq": r["seq"], "type": r["type"], "data": r["data"]}
                for r in conn.execute("SELECT * FROM events WHERE table_id = %s ORDER BY seq", (table_id,))
            ]
            s = conn.execute("SELECT * FROM snapshots WHERE table_id = %s ORDER BY hand_number DESC LIMIT 1",
                             (table_id,)).fetchone()
            snapshot = SnapshotRow(**{k: s[k] for k in SnapshotRow.__dataclass_fields__}) if s else None
            receipts = [ReceiptRow(**{k: r[k] for k in ReceiptRow.__dataclass_fields__})
                        for r in conn.execute("SELECT * FROM action_receipts WHERE table_id = %s", (table_id,))]
            checkpoints = [CheckpointRow(**{k: r[k] for k in CheckpointRow.__dataclass_fields__})
                           for r in conn.execute("SELECT * FROM bot_checkpoints WHERE table_id = %s ORDER BY seat",
                                                 (table_id,))]
            awards = [AwardRow(**{k: r[k] for k in AwardRow.__dataclass_fields__})
                      for r in conn.execute("SELECT * FROM awards WHERE table_id = %s ORDER BY seat", (table_id,))]
            p = conn.execute("SELECT * FROM table_pauses WHERE table_id = %s AND resumed_at IS NULL "
                             "ORDER BY pause_id DESC LIMIT 1", (table_id,)).fetchone()
            open_pause = (PauseRow(p["reason"], p["decision_id"], p["remaining_ms"], _epoch(p["paused_at"]),
                                   None, p["pause_id"]) if p else None)
            unpublished = [r["version"] for r in conn.execute(
                "SELECT version FROM outbox WHERE table_id = %s AND published_at IS NULL ORDER BY version",
                (table_id,))]
            return StoredTable(table, t["owner_id"], participants, events, snapshot, receipts, checkpoints, awards,
                               open_pause, t["scheduler_age"], unpublished)

    def table_ids(self) -> list[str]:
        with self._conn() as conn:
            return [r["table_id"] for r in conn.execute("SELECT table_id FROM arena_tables ORDER BY created_at")]

    # ------------------------------------------------------------- ownership

    def claim(self, table_id: str, writer_id: str, lease_s: float, *, force: bool = False) -> bool:
        with self._conn() as conn:
            r = conn.execute(
                "UPDATE arena_tables SET owner_id = %s, lease_until = now() + make_interval(secs => %s) "
                "WHERE table_id = %s AND (%s OR owner_id IS NULL OR owner_id = %s OR lease_until < now()) "
                "RETURNING table_id",
                (writer_id, lease_s, table_id, force, writer_id),
            ).fetchone()
            return r is not None

    def heartbeat(self, writer_id: str, lease_s: float, schedulers: dict[str, dict[str, Any] | None]) -> set[str]:
        if not schedulers:
            self.ping()
            return set()
        with self._conn() as conn, conn.transaction():
            renewed = set()
            for table_id, scheduler in schedulers.items():
                if scheduler is None:
                    r = conn.execute(
                        "UPDATE arena_tables SET lease_until = now() + make_interval(secs => %s) "
                        "WHERE table_id = %s AND owner_id = %s RETURNING table_id",
                        (lease_s, table_id, writer_id)).fetchone()
                else:
                    r = conn.execute(
                        "UPDATE arena_tables SET lease_until = now() + make_interval(secs => %s), scheduler = %s, "
                        "scheduler_at = now() WHERE table_id = %s AND owner_id = %s RETURNING table_id",
                        (lease_s, Jsonb(scheduler), table_id, writer_id)).fetchone()
                if r is not None:
                    renewed.add(table_id)
            return set(schedulers) - renewed

    def record_pause(self, table_id: str, writer_id: str, pause: PauseRow,
                     scheduler: dict[str, Any] | None) -> None:
        with self._conn() as conn, conn.transaction():
            r = conn.execute(
                "UPDATE arena_tables SET scheduler = %s, scheduler_at = now() "
                "WHERE table_id = %s AND owner_id = %s RETURNING table_id",
                (Jsonb(scheduler) if scheduler is not None else None, table_id, writer_id)).fetchone()
            if r is None:
                raise OwnershipLost(table_id)
            self._put_pause(conn, table_id, pause)

    def halt(self, table_id: str, reason: str) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE arena_tables SET status = 'halted', halted_reason = %s, updated_at = now() "
                         "WHERE table_id = %s", (reason, table_id))

    def mark_published(self, table_id: str, up_to_version: int) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE outbox SET published_at = now() WHERE table_id = %s AND version <= %s "
                         "AND published_at IS NULL", (table_id, up_to_version))

    def close(self) -> None:
        self._pool.close()

    # Test inspection helpers.

    def pauses(self, table_id: str) -> list[PauseRow]:
        with self._conn() as conn:
            return [PauseRow(p["reason"], p["decision_id"], p["remaining_ms"], _epoch(p["paused_at"]),
                             _epoch(p["resumed_at"]), p["pause_id"])
                    for p in conn.execute("SELECT * FROM table_pauses WHERE table_id = %s ORDER BY pause_id",
                                          (table_id,))]

    def outbox(self, table_id: str) -> list[tuple[int, str, bool]]:
        with self._conn() as conn:
            return [(r["version"], r["kind"], r["published_at"] is not None) for r in conn.execute(
                "SELECT version, kind, published_at FROM outbox WHERE table_id = %s ORDER BY version", (table_id,))]

    def count(self, table: str, table_id: str) -> int:
        if table not in {"events", "action_receipts", "awards", "hands", "snapshots", "outbox", "table_pauses"}:
            raise ValueError(table)
        with self._conn() as conn:
            return conn.execute(f"SELECT count(*) AS n FROM {table} WHERE table_id = %s", (table_id,)).fetchone()["n"]


class _Connection:
    """Borrow a pooled connection; any driver-level failure becomes :class:`StoreUnavailable`."""

    def __init__(self, pool: ConnectionPool) -> None:
        self._pool = pool
        self._cm = None

    def __enter__(self) -> psycopg.Connection:
        try:
            self._cm = self._pool.connection()
            conn = self._cm.__enter__()
        except Exception as exc:  # noqa: BLE001
            raise StoreUnavailable(f"Postgres unavailable: {type(exc).__name__}") from None
        conn.row_factory = dict_row
        return conn

    def __exit__(self, exc_type, exc, tb) -> bool:
        assert self._cm is not None
        try:
            self._cm.__exit__(exc_type, exc, tb)
        except Exception as close_exc:  # noqa: BLE001
            if exc is None:
                raise StoreUnavailable(f"Postgres unavailable: {type(close_exc).__name__}") from None
        if isinstance(exc, (psycopg.OperationalError, psycopg.InterfaceError)) and not isinstance(
                exc, psycopg.errors.QueryCanceled):
            raise StoreUnavailable(f"Postgres unavailable: {type(exc).__name__}") from None
        if isinstance(exc, psycopg.IntegrityError):
            raise StoreError(f"constraint violated: {type(exc).__name__}") from None
        return False

