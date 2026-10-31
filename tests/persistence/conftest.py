"""Store fixtures: every persistence test runs against the memory store and PostgreSQL.

PostgreSQL tests use ``ARENA_TEST_DATABASE_URL`` (default: the docker compose
database ``arena_test``). When it is unreachable they are skipped with a
visible reason, unless ``ARENA_REQUIRE_POSTGRES=1`` turns that into a failure.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from arena.persistence.store import MemoryStore, StoreUnavailable

TEST_DSN = os.environ.get("ARENA_TEST_DATABASE_URL", "postgresql://arena:arena@127.0.0.1:54329/arena_test")
_TABLES = ("outbox", "table_pauses", "awards", "snapshots", "bot_checkpoints", "action_receipts", "events", "hands",
           "participants", "arena_tables")
_pg_error: str | None = None


def _postgres():
    global _pg_error
    if _pg_error is not None:
        return None
    try:
        from arena.persistence.postgres import PostgresStore

        return PostgresStore(TEST_DSN, timeout_s=2.0)
    except StoreUnavailable as exc:
        _pg_error = str(exc)
        return None


def _require_postgres() -> bool:
    return os.environ.get("ARENA_REQUIRE_POSTGRES") == "1"


class FlakyStore:
    """Wraps a store so tests can take it offline; every call then raises StoreUnavailable."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.kind = inner.kind
        self.available = True
        self.calls = 0
        self._lock = threading.Lock()

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self.inner, name)
        if not callable(attr):
            return attr

        def call(*args: Any, **kwargs: Any) -> Any:
            with self._lock:
                self.calls += 1
            if not self.available:
                raise StoreUnavailable("test outage")
            return attr(*args, **kwargs)

        return call


@pytest.fixture(params=["memory", "postgres"])
def store_kind(request) -> str:
    return request.param


@pytest.fixture
def new_store(store_kind) -> Iterator[Callable[[], FlakyStore]]:
    """A factory for store handles over one shared database: calling it again is a process restart."""
    opened: list[Any] = []
    if store_kind == "memory":
        shared = MemoryStore()

        def make() -> FlakyStore:
            return FlakyStore(shared)
    else:
        first = _postgres()
        if first is None:
            if _require_postgres():
                pytest.fail(f"PostgreSQL required but unavailable at {TEST_DSN}: {_pg_error}")
            pytest.skip(f"PostgreSQL unavailable at {TEST_DSN} ({_pg_error}); run `make db-up`")
        with first._conn() as conn:
            conn.execute(f"TRUNCATE {', '.join(_TABLES)} RESTART IDENTITY CASCADE")
        opened.append(first)
        handles = iter([first])

        def make() -> FlakyStore:
            from arena.persistence.postgres import PostgresStore

            store = next(handles, None) or PostgresStore(TEST_DSN, timeout_s=2.0)
            opened.append(store)
            return FlakyStore(store)
    yield make
    for store in opened:
        store.close()
