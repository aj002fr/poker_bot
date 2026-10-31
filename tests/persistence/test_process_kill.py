"""A real server process on PostgreSQL, killed with SIGKILL mid-table, restarts into the same table."""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time

import httpx
import pytest

from tests.api.helpers import auth
from tests.persistence.conftest import TEST_DSN, _postgres, _require_postgres


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def serve(port: int) -> subprocess.Popen:
    env = {**os.environ, "ARENA_STORE": "postgres", "ARENA_DATABASE_URL": TEST_DSN, "ARENA_BOT_DELAY_MS": "0",
           "ARENA_HAND_PAUSE_MS": "0"}
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "--factory", "arena.api.app:create_app", "--host", "127.0.0.1",
         "--port", str(port), "--log-level", "warning"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    for _ in range(100):
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=0.2).status_code == 200:
                return proc
        except httpx.HTTPError:
            time.sleep(0.1)
    proc.kill()
    raise AssertionError(proc.stderr.read().decode() if proc.stderr else "server did not start")


@pytest.fixture
def pg_clean():
    store = _postgres()
    if store is None:
        if _require_postgres():
            pytest.fail("PostgreSQL required but unavailable")
        pytest.skip(f"PostgreSQL unavailable at {TEST_DSN}; run `make db-up`")
    with store._conn() as conn:
        conn.execute("TRUNCATE arena_tables CASCADE")
    yield store
    store.close()


def test_sigkill_and_restart_keeps_the_table(pg_clean):
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    proc = serve(port)
    try:
        r = httpx.post(f"{base}/tables", json={"seats": [{"kind": "human"}, {"kind": "human"},
                                                         {"kind": "bot", "policy": "check_or_call"}],
                                               "seed": "sigkill", "config": {"hand_cap": 4}})
        table_id, host = r.json()["table_id"], r.json()["host_token"]
        tokens = {}
        for seat in (0, 1):
            tokens[seat] = httpx.post(f"{base}/tables/{table_id}/join", json={"seat": seat}).json()["seat_token"]
        assert httpx.post(f"{base}/tables/{table_id}/start", headers=auth(host)).status_code == 200
        for i in range(3):
            snap = httpx.get(f"{base}/tables/{table_id}").json()
            seat = snap["turn"]["seat"]
            view = httpx.get(f"{base}/tables/{table_id}", headers=auth(tokens[seat])).json()["view"]
            action = {"type": "check"} if view["legal"]["check"] else {"type": "call"}
            body = {"action_id": f"k{i}", "table_id": table_id, "hand_id": view["hand_id"],
                    "decision_id": view["decision_id"], "expected_state_version": view["state_version"],
                    "action": action}
            assert httpx.post(f"{base}/tables/{table_id}/actions", headers=auth(tokens[seat]), json=body).status_code == 200
        before = {s: httpx.get(f"{base}/tables/{table_id}", headers=auth(t)).json() for s, t in tokens.items()}
    finally:
        proc.send_signal(signal.SIGKILL)
        proc.wait(10)

    proc = serve(port)
    try:
        after = {s: httpx.get(f"{base}/tables/{table_id}", headers=auth(t)).json() for s, t in tokens.items()}
        for s in tokens:
            assert after[s]["view"] == before[s]["view"]
            assert after[s]["version"] > before[s]["version"]
        replayed = httpx.post(f"{base}/tables/{table_id}/actions", headers=auth(tokens[seat]), json=body)
        assert replayed.status_code == 200 and replayed.json()["replayed"] is True
        pauses = pg_clean.pauses(table_id)
        assert [p.reason for p in pauses] == ["process_restart"]
    finally:
        proc.send_signal(signal.SIGTERM)
        proc.wait(10)
    pauses = pg_clean.pauses(table_id)
    assert [p.reason for p in pauses] == ["process_restart", "shutdown"] and pauses[-1].resumed_at is None
