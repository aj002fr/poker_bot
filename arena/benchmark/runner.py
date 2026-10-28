"""Seat-balanced, paired-seed benchmark runner for bots (spec section 10).

Schedule
    ``blocks`` paired seed blocks. Every block has one private table seed and
    ``N`` rotations: in rotation ``r`` entrant ``i`` sits at seat ``(i + r) mod N``.
    All rotations of a block share the seed (same decks and first button), so each
    entrant meets the same cards from every seat. ``tables = blocks * N``.

Scores
    Mean tournament points per table, with a percentile bootstrap interval that
    resamples whole blocks (the block is the independent unit, not the hand or the
    table). Failed tables are listed and never silently dropped from the counts.

Isolation
    ``sandbox`` (default) runs every seat in a pooled, sandboxed worker under the
    1 ms deadline. ``in_process`` calls the bots directly: a diagnostic mode for
    trusted code with no isolation and no enforced deadline, labelled as such and
    not comparable with sandboxed results.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from array import array
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Literal

from pydantic import StrictInt, StrictStr

from arena.agents.sdk import BotInfo, TableInfo, load_bot, require_supported
from arena.agents.worker import (
    AgentLoadError,
    Artifact,
    InProcessSeat,
    SeatAgent,
    WorkerPool,
    WorkerSeat,
    artifact_hash,
    play_table,
)
from arena.benchmark.manifest import (
    RunManifest,
    RunResults,
    FailedTable,
    Participant,
    code_info,
    derive_seeds,
    host_info,
    seed_list_hash,
)
from arena.domain.canonical import canonical_hash, canonical_json
from arena.domain.models import TournamentConfig
from arena.domain.rng import SeededRng

Isolation = Literal["sandbox", "in_process"]
BOOTSTRAP_RESAMPLES = 2000


@dataclass(frozen=True)
class Entrant:
    name: str
    path: Path
    builtin: bool = False


@dataclass(frozen=True)
class Assignment:
    index: int
    block: int
    rotation: int
    seed: str
    seats: dict[int, int]
    """seat -> entrant index"""

    @property
    def table_id(self) -> str:
        digest = hashlib.sha256(f"arena-anon-table-v1|{self.seed}|{self.rotation}".encode()).hexdigest()
        return "t_" + digest[:16]


def schedule(config: TournamentConfig, n_entrants: int, blocks: int, base_seed: str) -> list[Assignment]:
    seats = config.resolved_seats
    if n_entrants != len(seats):
        raise ValueError("one entrant per seat")
    seeds = derive_seeds(base_seed, blocks)
    out = []
    for b, seed in enumerate(seeds):
        for r in range(len(seats)):
            mapping = {seats[(i + r) % len(seats)]: i for i in range(n_entrants)}
            out.append(Assignment(len(out), b, r, seed, dict(sorted(mapping.items()))))
    return out


def policy_seed(seed: str, seat: int) -> str:
    rng = SeededRng(seed, f"policy-seat-{seat}")
    return "".join(f"{rng.randbelow(1 << 32):08x}" for _ in range(4))


def table_info(config: TournamentConfig, assignment: Assignment, seat: int, budget_ns: int) -> TableInfo:
    return TableInfo(
        table_id=assignment.table_id,
        hero_seat=seat,
        ruleset_id=config.ruleset_id,
        config=config.resolved(),
        config_hash=config.config_hash(),
        seats=config.resolved_seats,
        decision_budget_ns=budget_ns,
        policy_seed=policy_seed(assignment.seed, seat),
    )


# ------------------------------------------------------------------ one table


@dataclass
class TableResult:
    index: int
    block: int
    rotation: int
    table_id: str
    seats: dict[int, int]
    completed: bool
    error: str | None
    hands: int
    wall_s: float
    points: dict[int, float] = field(default_factory=dict)
    decisions: dict[int, int] = field(default_factory=dict)
    fallbacks: dict[int, dict[str, int]] = field(default_factory=dict)
    restarts: dict[int, list[tuple[int, str, bool]]] = field(default_factory=dict)
    late_discarded: dict[int, int] = field(default_factory=dict)
    max_state_bytes: dict[int, int] = field(default_factory=dict)
    e2e_ns: dict[int, bytes] = field(default_factory=dict)
    worker_wall_ns: dict[int, bytes] = field(default_factory=dict)
    worker_cpu_ns: dict[int, bytes] = field(default_factory=dict)

    @property
    def state_lost(self) -> list[int]:
        return sorted(e for e, rs in self.restarts.items() if any(not restored for _, _, restored in rs))


class TableRunner:
    """Plays assignments in one process. Owns a worker pool in sandbox mode."""

    def __init__(self, config: TournamentConfig, entrants: Sequence[Entrant], *, isolation: Isolation = "sandbox",
                 budget_ns: int | None = None, history_dir: Path | None = None, pool_size: int | None = None) -> None:
        self.config = config
        self.entrants = list(entrants)
        self.isolation = isolation
        self.budget_ns = budget_ns or config.bot_budget_ns
        self.history_dir = history_dir
        self.pool: WorkerPool | None = None
        self.artifacts: list[Artifact] = []
        if isolation == "sandbox":
            self.pool = WorkerPool(size=pool_size or len(config.resolved_seats) + 2)
            self.artifacts = [self.pool.stage(e.path) for e in self.entrants]
        else:
            self.artifacts = [Artifact(e.name, e.path, e.path.parent, artifact_hash(e.path)) for e in self.entrants]

    def close(self) -> None:
        if self.pool is not None:
            self.pool.close()

    def _agent(self, seat: int, entrant: int, info: TableInfo) -> SeatAgent:
        art = self.artifacts[entrant]
        if self.pool is not None:
            return WorkerSeat(seat, art, info, self.pool.acquire, budget_ns=self.budget_ns)
        bot, bot_info = load_bot(art.entry, module_name=f"arena_inproc_{entrant}_{seat}")
        return InProcessSeat(seat, bot, bot_info, info, art.sha256)

    def run(self, a: Assignment) -> TableResult:
        from arena.history.canonical import write_history
        from arena.table.service import seeded_table

        t0 = time.perf_counter()
        agents: dict[int, SeatAgent] = {}
        result = TableResult(a.index, a.block, a.rotation, a.table_id, dict(a.seats), False, None, 0, 0.0)
        try:
            for seat, entrant in a.seats.items():
                agent = self._agent(seat, entrant, table_info(self.config, a, seat, self.budget_ns))
                agents[seat] = agent
                agent.start()
            table = seeded_table(self.config, a.seed, table_id=a.table_id)
            play_table(table, agents)
            result.hands = len(table.hands)
            result.completed = table.complete
            if table.complete:
                result.points = {a.seats[s]: float(p) for s, p in table.points().items()}
            if self.history_dir is not None:
                write_history(table, self.history_dir)
        except Exception as exc:  # recorded, never hidden
            result.error = f"{type(exc).__name__}: {exc}"[:500]
        finally:
            for agent in agents.values():
                agent.close()
        for seat, agent in agents.items():
            e, st = a.seats[seat], agent.stats
            result.decisions[e] = st.decisions
            result.fallbacks[e] = dict(st.fallbacks)
            result.restarts[e] = [(r.decision_index, r.cause, r.restored) for r in st.restarts]
            result.late_discarded[e] = st.late_discarded
            result.max_state_bytes[e] = st.max_state_bytes
            result.e2e_ns[e] = array("q", st.e2e_ns).tobytes()
            result.worker_wall_ns[e] = array("q", st.worker_wall_ns).tobytes()
            result.worker_cpu_ns[e] = array("q", st.worker_cpu_ns).tobytes()
        result.wall_s = time.perf_counter() - t0
        return result


_RUNNER: TableRunner | None = None


def _job_init(config_json: str, entrants: list[Entrant], isolation: Isolation, budget_ns: int,
              history_dir: str | None) -> None:
    global _RUNNER
    config = TournamentConfig.model_validate_json(config_json)
    _RUNNER = TableRunner(config, entrants, isolation=isolation, budget_ns=budget_ns,
                          history_dir=Path(history_dir) if history_dir else None)


def _job_run(a: Assignment) -> TableResult:
    assert _RUNNER is not None
    return _RUNNER.run(a)


# ----------------------------------------------------------------- preflight


def inspect_entrants(config: TournamentConfig, entrants: Sequence[Entrant], isolation: Isolation) -> list[BotInfo]:
    """Load each entrant once and check its declared rulesets before any table starts."""
    infos = []
    dummy = Assignment(0, 0, 0, "preflight", {})
    if isolation == "sandbox":
        with WorkerPool(size=1) as pool:
            for e in entrants:
                seat = WorkerSeat(config.resolved_seats[0], pool.stage(e.path),
                                  table_info(config, dummy, config.resolved_seats[0], config.bot_budget_ns),
                                  pool.acquire)
                try:
                    infos.append(seat.start())
                finally:
                    seat.close()
    else:
        for k, e in enumerate(entrants):
            _, info = load_bot(e.path, module_name=f"arena_preflight_{k}")
            require_supported(info, config.ruleset_id)
            infos.append(info)
    return infos


# ------------------------------------------------------------------- summary


def _quantile(sorted_xs: Sequence[int], q: float) -> int | None:
    if not sorted_xs:
        return None
    return sorted_xs[min(len(sorted_xs) - 1, max(0, math.ceil(q * len(sorted_xs)) - 1))]


def timing_summary(e2e: Sequence[int], wall: Sequence[int], cpu: Sequence[int], budget_ns: int) -> dict[str, Any]:
    e, w, c = sorted(e2e), sorted(wall), sorted(cpu)
    to_s = lambda v: None if v is None else v / 1e9  # noqa: E731

    def q(xs: Sequence[int]) -> dict[str, float | None]:
        return {"p50": to_s(_quantile(xs, 0.50)), "p95": to_s(_quantile(xs, 0.95)),
                "p99": to_s(_quantile(xs, 0.99)), "max": to_s(xs[-1] if xs else None)}

    return {
        "samples": len(e),
        "end_to_end_s": q(e),
        "over_budget_samples": sum(1 for v in e if v > budget_ns),
        "worker_wall_s": q(w),
        "worker_cpu_s": q(c),
    }


def bootstrap_interval(block_values: Sequence[float], seed: str, resamples: int = BOOTSTRAP_RESAMPLES,
                       level: float = 0.95) -> tuple[float, float] | None:
    n = len(block_values)
    if n < 2:
        return None
    rng = random.Random(hashlib.sha256(f"arena-bootstrap-v1|{seed}".encode()).digest())
    means = sorted(sum(block_values[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    lo = means[int((1 - level) / 2 * resamples)]
    hi = means[min(resamples - 1, int((1 + level) / 2 * resamples))]
    return lo, hi


def summarize(results: Sequence[TableResult], entrants: Sequence[Entrant], infos: Sequence[BotInfo],
              budget_ns: int, base_seed: str) -> dict[str, Any]:
    completed = [r for r in results if r.completed and r.error is None]
    per_entrant = []
    for i, e in enumerate(entrants):
        blocks: dict[int, list[float]] = {}
        for r in completed:
            blocks.setdefault(r.block, []).append(r.points[i])
        block_means = [sum(v) / len(v) for _, v in sorted(blocks.items())]
        pts = [r.points[i] for r in completed]
        e2e = [x for r in results for x in array("q", r.e2e_ns.get(i, b""))]
        wall = [x for r in results for x in array("q", r.worker_wall_ns.get(i, b""))]
        cpu = [x for r in results for x in array("q", r.worker_cpu_ns.get(i, b""))]
        fallbacks: Counter = Counter()
        for r in results:
            fallbacks.update(r.fallbacks.get(i, {}))
        decisions = sum(r.decisions.get(i, 0) for r in results)
        restarts = [x for r in results for x in r.restarts.get(i, [])]
        per_entrant.append({
            "entrant": i,
            "name": e.name,
            "bot_name": infos[i].name if i < len(infos) else e.name,
            "state_mode": infos[i].state_mode if i < len(infos) else None,
            "builtin": e.builtin,
            "tables": len(pts),
            "mean_points": sum(pts) / len(pts) if pts else None,
            "points_ci95": bootstrap_interval(block_means, f"{base_seed}|{i}"),
            "decisions": decisions,
            "fallbacks": dict(sorted(fallbacks.items())),
            "fallback_total": sum(fallbacks.values()),
            "timeout_fallbacks": fallbacks.get("timeout", 0),
            "restarts": len(restarts),
            "restarts_not_restorable": sum(1 for _, _, ok in restarts if not ok),
            "tables_with_state_loss": sum(1 for r in results if i in r.state_lost),
            "late_replies_discarded": sum(r.late_discarded.get(i, 0) for r in results),
            "max_state_bytes": max((r.max_state_bytes.get(i, 0) for r in results), default=0),
            "timing": timing_summary(e2e, wall, cpu, budget_ns),
        })
    base = [p for p in per_entrant if p["builtin"]]
    base_e2e = sorted(x for r in results for i, p in enumerate(per_entrant) if p["builtin"]
                      for x in array("q", r.e2e_ns.get(i, b"")))
    base_decisions = sum(p["decisions"] for p in base)
    base_timeouts = sum(p["timeout_fallbacks"] for p in base)
    return {
        "tables_scheduled": len(results),
        "tables_completed": len(completed),
        "tables_failed": [{"table_id": r.table_id, "error": r.error} for r in results if r.error],
        "tables_incomplete": [r.table_id for r in results if not r.completed and not r.error],
        "blocks": len({r.block for r in results}),
        "entrants": per_entrant,
        "baselines_combined": {
            "decisions": base_decisions,
            "timeout_fallbacks": base_timeouts,
            "timeout_rate": base_timeouts / base_decisions if base_decisions else None,
            "fallback_total": sum(p["fallback_total"] for p in base),
            "timing": timing_summary(base_e2e, [], [], budget_ns),
        },
        "table_wall_s_mean": sum(r.wall_s for r in results) / len(results) if results else None,
    }


# ------------------------------------------------------------------ manifest


class BenchmarkManifest(RunManifest):
    """The section 10 run manifest, extended for agent benchmarks."""

    manifest_kind: Literal["agent_benchmark"] = "agent_benchmark"
    isolation: Literal["sandbox", "in_process"]
    competitive: bool
    budget_ns: StrictInt
    seed_blocks: StrictInt
    rotations_per_block: StrictInt
    entrants: list[dict[str, Any]]
    process_layout: dict[str, Any]
    sandbox: dict[str, Any]
    summary: dict[str, Any]
    report_note: StrictStr

    def content_hash(self) -> str:
        data = self.model_dump(mode="json")
        data.pop("created_at")
        return canonical_hash(data, domain="arena-benchmark-manifest-v1")


SANDBOX_DESCRIPTION = {
    "namespaces": "user (unprivileged, no root mapping), network (loopback only, down), ipc, uts, pid, mount (+private /proc)",
    "landlock": "read-only Python runtime and site-packages, plus the bot's own staged bundle; no writes; TCP bind/connect denied; signals and abstract UNIX sockets scoped",
    "seccomp": "deny list: socket creation, fork/vfork/non-thread clone, exec, ptrace, mount/namespace changes, BPF, io_uring, keyrings, SysV/POSIX IPC creation",
    "rlimits": "address space, CPU seconds, file size 0, open files, threads (NPROC per user namespace), no core dumps",
    "gvisor": "not used",
}


@dataclass
class BenchmarkRun:
    manifest: BenchmarkManifest
    results: list[TableResult]
    out_dir: Path | None


def run_benchmark(
    *,
    config: TournamentConfig,
    entrants: Sequence[Entrant],
    blocks: int,
    base_seed: str,
    isolation: Isolation = "sandbox",
    budget_ns: int | None = None,
    jobs: int = 1,
    out_dir: Path | None = None,
    save_histories: bool = False,
    run_id: str | None = None,
    progress: Callable[[int, int, TableResult], None] | None = None,
) -> BenchmarkRun:
    budget_ns = budget_ns or config.bot_budget_ns
    infos = inspect_entrants(config, entrants, isolation)
    plan = schedule(config, len(entrants), blocks, base_seed)
    seeds = derive_seeds(base_seed, blocks)
    run_id = run_id or "b_" + canonical_hash(
        [config.config_hash(), base_seed, blocks, [artifact_hash(e.path) for e in entrants], isolation],
        domain="arena-benchmark-id-v1")[:12]
    history_dir = (out_dir / "tables") if (out_dir is not None and save_histories) else None
    results: list[TableResult] = []
    if jobs <= 1:
        runner = TableRunner(config, entrants, isolation=isolation, budget_ns=budget_ns, history_dir=history_dir)
        try:
            for a in plan:
                results.append(runner.run(a))
                if progress:
                    progress(len(results), len(plan), results[-1])
        finally:
            runner.close()
    else:
        ctx = get_context("spawn")
        with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx, initializer=_job_init,
                                 initargs=(config.model_dump_json(), list(entrants), isolation, budget_ns,
                                           str(history_dir) if history_dir else None)) as ex:
            for r in ex.map(_job_run, plan, chunksize=1):
                results.append(r)
                if progress:
                    progress(len(results), len(plan), r)
    results.sort(key=lambda r: r.index)
    summary = json.loads(canonical_json(summarize(results, entrants, infos, budget_ns, base_seed)))
    competitive = isolation == "sandbox" and budget_ns == config.bot_budget_ns
    seats = config.resolved_seats
    seat_points = {s: 0.0 for s in seats}
    completed = [r for r in results if r.completed and not r.error]
    for r in completed:
        for s, e in r.seats.items():
            seat_points[s] += r.points[e]
    manifest = BenchmarkManifest(
        run_id=run_id,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        code=code_info(),
        ruleset_id=config.ruleset_id,
        config=config.resolved(),
        config_hash=config.config_hash(),
        initial_player_count=config.initial_player_count,
        participants=[Participant(seat=i, name=e.name, kind="in_process_baseline" if e.builtin and isolation == "in_process"
                                  else "agent_artifact", artifact_sha256=artifact_hash(e.path))
                      for i, e in enumerate(entrants)],
        rng_implementation=SeededRng.implementation,
        seed_count=len(seeds),
        seed_list_sha256=seed_list_hash(seeds),
        seat_allocation_policy=("paired seed blocks with full rotation: in rotation r of a block, entrant i sits at "
                                "seat (i + r) mod N; all N rotations share the block's seed; participant.seat is "
                                "the entrant index"),
        budget_semantics=(
            f"parent-observed round trip from dispatch to complete reply; on time iff received_ns <= dispatched_ns "
            f"+ {budget_ns}; late, invalid, crashed or illegal replies fall back to check, otherwise fold"
            if isolation == "sandbox" else
            "in-process diagnostic: bots called directly, no isolation, deadline not enforced; not competitive"
        ),
        host=host_info(),
        scheduled_tables=len(plan),
        retry_policy="none: failed or incomplete tables are listed, not rerun; agent crashes use fallbacks and do "
                     "not invalidate a table; instance-state restarts are counted per table",
        results=RunResults(
            completed_table_ids=[r.table_id for r in completed],
            failed_tables=[FailedTable(table_id=r.table_id, error=r.error or "") for r in results if r.error],
            incomplete_table_ids=[r.table_id for r in results if not r.completed and not r.error],
            mean_points_by_seat={s: (v / len(completed) if completed else 0.0) for s, v in seat_points.items()},
        ),
        isolation=isolation,
        competitive=competitive,
        budget_ns=budget_ns,
        seed_blocks=blocks,
        rotations_per_block=len(seats),
        entrants=[{"entrant": i, "name": e.name, "artifact_sha256": artifact_hash(e.path), "builtin": e.builtin,
                   **(infos[i].as_dict() if i < len(infos) else {})} for i, e in enumerate(entrants)],
        process_layout={"jobs": max(1, jobs), "tables_in_parallel": max(1, jobs),
                        "workers_per_table": len(seats), "pool_size_per_job": len(seats) + 2,
                        "worker_reuse": "none: one worker process per seat per table"},
        sandbox=SANDBOX_DESCRIPTION if isolation == "sandbox" else {"isolation": "none (in-process diagnostic)"},
        summary=summary,
        report_note="practice assumptions; not official Sequoia rules"
                    + ("" if competitive else "; DIAGNOSTIC run, excluded from competitive comparison"),
    )
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "manifest.json").write_text(canonical_json(manifest.model_dump(mode="json")) + "\n", encoding="ascii")
        private = out_dir / "private"
        private.mkdir(exist_ok=True)
        (private / "seeds.json").write_text(canonical_json({"run_id": run_id, "seeds": seeds}) + "\n", encoding="ascii")
    return BenchmarkRun(manifest, results, out_dir)


__all__ = [
    "AgentLoadError",
    "Assignment",
    "BenchmarkManifest",
    "Entrant",
    "TableResult",
    "TableRunner",
    "bootstrap_interval",
    "inspect_entrants",
    "run_benchmark",
    "schedule",
    "summarize",
]
