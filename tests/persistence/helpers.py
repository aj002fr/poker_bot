"""Boot, crash and restart table managers over a shared store."""

from __future__ import annotations

import json
import threading
from typing import Any

from fastapi.testclient import TestClient

from arena.api.app import create_app
from arena.api.clock import FakeClock
from arena.api.manager import SimulatedCrash, TableManager
from arena.domain.events import EventEnvelope
from tests.api.helpers import HUMAN, Harness, bot


class TickingClock(FakeClock):
    """A fake clock that also moves 1 ns per read, so concurrent readers get distinct, ordered times."""

    def __init__(self, start_ns: int = 1_000_000_000) -> None:
        super().__init__(start_ns)
        self._tick_lock = threading.Lock()

    def now_ns(self) -> int:
        with self._tick_lock:
            self.t += 1
            return self.t - 1


def boot(store, clock: FakeClock, *, carry: Harness | None = None, writer_id: str = "local-writer",
         force: bool = False, bot_delay_ms: int = 0) -> Harness:
    """A fresh process: new manager (which recovers from ``store``) and a client without lifespan hooks."""
    manager = TableManager(clock, store=store, bot_delay_ms=bot_delay_ms, hand_pause_ms=0, writer_id=writer_id,
                           force_claim=force)
    h = Harness(TestClient(create_app(manager, background_pump=False)), manager, clock)
    if carry is not None:
        h.table_id, h.host, h.tokens, h._ids = carry.table_id, carry.host, dict(carry.tokens), carry._ids
    return h


def safe_action(view: dict[str, Any]) -> dict[str, Any]:
    return {"type": "check"} if view["legal"]["check"] else {"type": "call"}


def raw_action(h: Harness, seat: int, action: dict[str, Any] | None = None, action_id: str | None = None) -> bytes:
    view = h.snap(seat)["view"]
    return json.dumps(h.envelope(view, action or safe_action(view), action_id)).encode()


def principal(h: Harness, seat: int):
    return h.manager.authenticate(h.table_id, h.tokens[seat])


def stored_events(store, table_id: str) -> list[EventEnvelope]:
    return [EventEnvelope.from_json_obj(e) for e in store.load(table_id).events]


def crash_on(kind: str, *, times: int = 1):
    """A fault hook that raises SimulatedCrash for the first ``times`` commands of ``kind``."""
    left = [times]

    def hook(table_id: str, command: str) -> None:
        if command == kind and left[0] > 0:
            left[0] -= 1
            raise SimulatedCrash(f"{kind} on {table_id}")

    return hook


def mixed_table(h: Harness, *, humans: int = 2, bots: int = 2, **config: Any) -> None:
    h.setup([HUMAN] * humans + [bot("check_or_call")] * bots, **config)


def play_humans(h: Harness, decisions: int) -> int:
    """Make up to ``decisions`` human actions (bots act inline at zero delay). Returns how many were made."""
    made = 0
    while made < decisions:
        snap = h.snap()
        if snap["status"] != "running":
            break
        seat = snap["turn"]["seat"]
        r = h.act(seat, safe_action(h.snap(seat)["view"]))
        assert r.status_code == 200, r.text
        made += 1
    return made


def chips_conserved(events: list[EventEnvelope]) -> bool:
    total = sum(events[0].data["start_stacks"].values())
    finals = [e for e in events if e.type == "HandCompleted"]
    return all(sum(e.data["final_stacks"].values()) == total for e in finals)
