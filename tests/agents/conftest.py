"""Shared helpers for agent-runtime tests.

Tests that start real sandboxed workers need util-linux ``unshare``, unprivileged
user namespaces and Landlock. Where those are missing they are skipped with the
reason; the milestone report records where they ran.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from arena.agents import sandbox
from arena.agents.baselines import baseline_path
from arena.agents.sdk import TableInfo
from arena.agents.worker import Decision, WorkerPool, WorkerSeat, play_table
from arena.domain.models import TournamentConfig
from arena.table.service import seeded_table

BOTS = Path(__file__).resolve().parent / "bots"


def _sandbox_unavailable() -> str | None:
    if shutil.which("unshare") is None:
        return "util-linux unshare not installed"
    if sandbox.landlock_abi() < 1:
        return "Landlock not available"
    probe = subprocess.run(["unshare", *sandbox.UNSHARE_ARGS, "true"], capture_output=True)
    if probe.returncode != 0:
        return f"unprivileged namespaces unavailable: {probe.stderr.decode()[:200]}"
    return None


SANDBOX_SKIP = _sandbox_unavailable()
needs_sandbox = pytest.mark.skipif(SANDBOX_SKIP is not None, reason=SANDBOX_SKIP or "")


@pytest.fixture(scope="session")
def pool():
    if SANDBOX_SKIP:
        pytest.skip(SANDBOX_SKIP)
    with WorkerPool(size=6) as p:
        yield p


def small_config(players: int = 3, **kw) -> TournamentConfig:
    kw.setdefault("hand_cap", 6)
    kw.setdefault("hands_per_level", -(-kw["hand_cap"] // 8))
    return TournamentConfig(initial_player_count=players, **kw)


def info_for(config: TournamentConfig, seat: int, table_id: str = "t_test", seed: str = "p") -> TableInfo:
    return TableInfo(table_id=table_id, hero_seat=seat, ruleset_id=config.ruleset_id, config=config.resolved(),
                     config_hash=config.config_hash(), seats=config.resolved_seats,
                     decision_budget_ns=config.bot_budget_ns, policy_seed=f"{seed}-{seat}")


def bot(name: str) -> Path:
    path = BOTS / f"{name}.py"
    return path if path.exists() else baseline_path(name)


@dataclass
class Played:
    table: object
    seats: dict[int, WorkerSeat]
    records: list[tuple[int, str, str | None]] = field(default_factory=list)
    """(seat, decision_id, fallback) for every decision."""

    def seat_records(self, seat: int) -> list[tuple[str, str | None]]:
        return [(d, f) for s, d, f in self.records if s == seat]


def play_sandboxed(pool: WorkerPool, bots: list[str], *, config: TournamentConfig | None = None, seed: str = "faults",
                   budget_ns: int | None = None, before_decision=None, **seat_kw) -> Played:
    config = config or small_config(len(bots))
    seats: dict[int, WorkerSeat] = {}
    for s, name in zip(config.resolved_seats, bots):
        seats[s] = WorkerSeat(s, pool.stage(bot(name)), info_for(config, s, seed=seed), pool.acquire,
                              budget_ns=budget_ns or config.bot_budget_ns, **seat_kw)
        seats[s].start()
    table = seeded_table(config, seed, table_id="t_test")
    played = Played(table, seats)
    pending: dict[str, str] = {}

    def before(index: int, seat: int) -> None:
        if before_decision is not None:
            before_decision(index, seat, seats)

    def on_decision(seat: int, decision: Decision, table) -> None:
        played.records.append((seat, pending.pop("id"), decision.fallback))

    def wrapped_before(index: int, seat: int) -> None:
        pending["id"] = table.decision_id
        before(index, seat)

    try:
        play_table(table, seats, before_decision=wrapped_before, on_decision=on_decision)
    finally:
        for agent in seats.values():
            agent.close()
    return played
