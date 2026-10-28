"""Both stores honour the same contract: atomic batches, ownership compare-and-set, claims and the outbox."""

from __future__ import annotations

import time

import pytest

from arena.persistence.models import CommitBatch, ParticipantRow, PauseRow, ReceiptRow, TableRow
from arena.persistence.store import OwnershipLost, StoreError


def row(version: int, events: int = 0, **kw) -> TableRow:
    return TableRow(table_id="t_contract", config={"ruleset_id": "practice_nlhe"}, config_hash="h", deck_seed=None,
                    bot_seed="b", host_token_hash="host", status="waiting", version=version, event_count=events, **kw)


def event(seq: int) -> dict:
    return {"schema_version": 1, "table_id": "t_contract", "hand_id": None, "seq": seq, "type": "TableStarted",
            "data": {"n": seq, "nested": {"1": [1, 2]}}}


def create(store, writer: str = "w1") -> None:
    store.commit(CommitBatch(row(1), None, "create",
                             participants=[ParticipantRow(0, "human", None, None, False, None)]), writer, 15)


def test_round_trip_and_outbox(new_store):
    store = new_store()
    create(store)
    store.commit(CommitBatch(row(2, 2, scheduler={"decision_id": "d_00001", "remaining_ms": 10}), 1, "start",
                             events=[event(0), event(1)],
                             receipt=ReceiptRow("a1", 0, "d_00001", "fp", {"ok": True}, 2)), "w1", 15)
    loaded = store.load("t_contract")
    assert loaded.table.version == 2 and loaded.table.scheduler["remaining_ms"] == 10
    assert loaded.events == [event(0), event(1)]
    assert [(r.action_id, r.body) for r in loaded.receipts] == [("a1", {"ok": True})]
    assert loaded.unpublished == [1, 2] and loaded.owner_id == "w1"
    store.mark_published("t_contract", 1)
    assert store.load("t_contract").unpublished == [2]
    assert store.outbox("t_contract") == [(1, "create", True), (2, "start", False)]


def test_stale_version_or_foreign_writer_is_fenced(new_store):
    store = new_store()
    create(store)
    with pytest.raises(OwnershipLost):
        store.commit(CommitBatch(row(2), 0, "join"), "w1", 15)
    with pytest.raises(OwnershipLost):
        store.commit(CommitBatch(row(2), 1, "join"), "w2", 15)
    assert store.load("t_contract").table.version == 1


def test_failed_batch_leaves_no_trace(new_store):
    store = new_store()
    create(store)

    def boom() -> None:
        raise RuntimeError("in transaction")

    with pytest.raises(RuntimeError):
        store.commit(CommitBatch(row(2, 1), 1, "start", events=[event(0)],
                                 receipt=ReceiptRow("a1", 0, "d1", "fp", {}, 2)), "w1", 15, in_tx=boom)
    loaded = store.load("t_contract")
    assert loaded.table.version == 1 and loaded.events == [] and loaded.receipts == [] and loaded.unpublished == [1]


def test_duplicate_receipts_and_gapped_events_are_rejected(new_store):
    store = new_store()
    create(store)
    store.commit(CommitBatch(row(2), 1, "a", receipt=ReceiptRow("a1", 0, "d1", "fp", {}, 2)), "w1", 15)
    with pytest.raises(StoreError):
        store.commit(CommitBatch(row(3), 2, "a", receipt=ReceiptRow("a1", 0, "d2", "fp", {}, 3)), "w1", 15)
    with pytest.raises(StoreError):
        store.commit(CommitBatch(row(3), 2, "a", receipt=ReceiptRow("a2", 0, "d1", "fp", {}, 3)), "w1", 15)
    with pytest.raises(StoreError):
        store.commit(CommitBatch(row(3, 1), 2, "a", events=[event(1)]), "w1", 15)
    assert store.load("t_contract").table.version == 2


def test_claims_respect_live_leases(new_store):
    store = new_store()
    create(store)
    assert store.claim("t_contract", "w1", 15) is True
    assert store.claim("t_contract", "w2", 15) is False
    assert store.heartbeat("w2", 15, {"t_contract": None}) == {"t_contract"}
    assert store.heartbeat("w1", 0.05, {"t_contract": {"remaining_ms": 5}}) == set()
    time.sleep(0.12)
    assert store.claim("t_contract", "w2", 15) is True
    assert store.heartbeat("w1", 15, {"t_contract": None}) == {"t_contract"}
    assert store.claim("t_contract", "w1", 15, force=True) is True
    assert store.load("t_contract").owner_id == "w1"


def test_pauses_open_and_close(new_store):
    store = new_store()
    create(store)
    store.record_pause("t_contract", "w1", PauseRow("shutdown", "d1", 1234, time.time()), {"remaining_ms": 1234})
    open_pause = store.load("t_contract").open_pause
    assert (open_pause.reason, open_pause.remaining_ms) == ("shutdown", 1234)
    closed = PauseRow(open_pause.reason, open_pause.decision_id, open_pause.remaining_ms, open_pause.paused_at,
                      time.time(), open_pause.pause_id)
    store.commit(CommitBatch(row(2), 1, "resume", pauses=[closed]), "w1", 15)
    assert store.load("t_contract").open_pause is None
    with pytest.raises(OwnershipLost):
        store.record_pause("t_contract", "w2", PauseRow("shutdown", "d1", 1, time.time()), None)


def test_postgres_migrations_are_idempotent(new_store, store_kind):
    if store_kind != "postgres":
        pytest.skip("migrations exist only for PostgreSQL")
    store = new_store()
    from arena.persistence.postgres import migrate

    with store.inner._conn() as conn:
        assert migrate(conn) == []
        names = [r["name"] for r in conn.execute("SELECT name FROM schema_migrations ORDER BY name")]
    assert names == ["0001_initial"]
