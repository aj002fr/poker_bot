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

Infrastructure failures and reruns (spec section 9.2)
    A table interrupted by infrastructure is marked invalid and rerun with the same
    seed and table ID under a new attempt ID (``<table_id>-a<n>``), up to the
    declared :class:`RetryPolicy`, which is recorded in the manifest. Infrastructure
    means a worker that could not be started, a host OS or memory error in the
    table process, a lost job process, or an infrastructure kill that left an
    instance-state bot without its state. An infrastructure kill of an
    explicit-state bot is restored exactly and does not invalidate the table.
    Agent faults (crashes, timeouts, bad output) use the normal fallback and never
    invalidate a table. The rerun decision looks only at those infrastructure
    events, never at results, and every attempt is reported.

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
import os
import random
import time
from array import array
from collections import Counter, deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from datetime import UTC, datetime
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, StrictInt, StrictStr

from arena.agents.sdk import BotInfo, TableInfo, load_bot, require_supported
from arena.agents.worker import (
    AgentLoadError,
    Artifact,
    InProcessSeat,
    SeatAgent,
    WorkerPool,
    WorkerSeat,
    WorkerStartError,
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

INFRASTRUCTURE_CAUSES = (
    "worker_start_failed",                    # no sandboxed worker could be started for a seat
    "host_os_error",                          # OSError in the table process (fds, disk, fork)
    "host_memory_error",                      # MemoryError in the table process
    "job_process_lost",                       # the job process running the table died
    "job_error",                              # the job could not run or return the table
    "instance_state_lost_to_infrastructure",  # infrastructure kill; instance state not restorable
)


@dataclass(frozen=True)
class RetryPolicy:
    """Declared rerun policy for tables invalidated by infrastructure (spec 9.2).

    ``max_attempts`` counts the first attempt, so 1 means invalid tables are
    reported but never rerun.
    """

    max_attempts: int = 3

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy": "rerun_infrastructure_invalid_tables",
            "max_attempts": self.max_attempts,
            "same_seed": True,
            "same_table_id": True,
            "attempt_id_format": "<table_id>-a<attempt>",
            "invalidating_causes": list(INFRASTRUCTURE_CAUSES),
            "not_invalidating": ["agent faults (crash, timeout, exception, invalid or illegal output): normal "
                                 "fallback", "infrastructure kill of an explicit-state bot: restored exactly"],
            "not_retried": "engine or other non-infrastructure errors: listed as failed",
            "selection": "decided only from infrastructure events, never from results; every attempt is reported",
        }

    def describe(self) -> str:
        return (f"rerun tables invalidated by infrastructure ({', '.join(INFRASTRUCTURE_CAUSES)}) with the same seed "
                f"and table ID, up to {self.max_attempts} attempts in total; attempt IDs <table_id>-a<n>; every "
                "attempt is counted and reported; agent crashes use fallbacks and do not invalidate a table; "
                "other errors are listed as failed and not rerun")


def _infrastructure_cause(exc: BaseException) -> str | None:
    if isinstance(exc, WorkerStartError):
        return "worker_start_failed"
    if isinstance(exc, MemoryError):
        return "host_memory_error"
    if isinstance(exc, OSError):
        return "host_os_error"
    return None


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
    attempt: int = 1
    invalid_reason: str | None = None
    """One of :data:`INFRASTRUCTURE_CAUSES` when infrastructure invalidated this attempt."""
    infrastructure: list[dict[str, Any]] = field(default_factory=list)

    @property
    def state_lost(self) -> list[int]:
        return sorted(e for e, rs in self.restarts.items() if any(not restored for _, _, restored in rs))

    @property
    def attempt_id(self) -> str:
        return f"{self.table_id}-a{self.attempt}"

    @property
    def valid(self) -> bool:
        return self.completed and self.error is None and self.invalid_reason is None


def _lost_attempt(a: Assignment, attempt: int, cause: str, detail: str) -> TableResult:
    return TableResult(a.index, a.block, a.rotation, a.table_id, dict(a.seats), False, detail[:500], 0, 0.0,
                       attempt=attempt, invalid_reason=cause)


def _seat_invalidation(stats: Any) -> str | None:
    events = [ev for _, ev in stats.infrastructure]
    if "worker_start_failed" in events:
        return "worker_start_failed"
    kills = {ev for ev in events if ev != "worker_start_failed"}
    if any(r.cause in kills and not r.restored for r in stats.restarts):
        return "instance_state_lost_to_infrastructure"
    return None


class _InjectedFaults:
    """Test-only infrastructure faults: ``start_fail``, ``kill:<seat>:<decision>``
    (an infrastructure kill) and ``spawn_fail:<seat>:<decision>`` (a kill after
    which no worker can be started for that seat)."""

    def __init__(self, spec: str | None) -> None:
        self.start_fail = spec == "start_fail"
        self.kill: tuple[str, int, int] | None = None
        if spec and spec.split(":")[0] in ("kill", "spawn_fail"):
            kind, seat, index = spec.split(":")
            self.kill = (kind, int(seat), int(index))

    def before_decision(self, agents: Mapping[int, SeatAgent]) -> Callable[[int, int], None] | None:
        if self.kill is None:
            return None
        kind, seat, at = self.kill

        def hook(index: int, _actor: int) -> None:
            if index != at:
                return
            agent: Any = agents[seat]
            if kind == "spawn_fail":
                def refuse() -> Any:
                    raise WorkerStartError("injected: no worker can be started")
                agent.spawn = refuse
            agent.kill_worker()

        return hook


class TableRunner:
    """Plays assignments in one process. Owns a worker pool in sandbox mode."""

    def __init__(self, config: TournamentConfig, entrants: Sequence[Entrant], *, isolation: Isolation = "sandbox",
                 budget_ns: int | None = None, history_dir: Path | None = None, pool_size: int | None = None) -> None:
        self.config = config
        self.entrants = list(entrants)
        self.isolation = isolation
        self.budget_ns = budget_ns or config.bot_budget_ns
        self.history_dir = history_dir
        self.pool_size = pool_size or len(config.resolved_seats) + 2
        self.pool: WorkerPool | None = None
        self.artifacts: list[Artifact] = []
        if isolation == "sandbox":
            self._start_pool()
        else:
            self.artifacts = [Artifact(e.name, e.path, e.path.parent, artifact_hash(e.path)) for e in self.entrants]

    def _start_pool(self) -> None:
        self.pool = WorkerPool(size=self.pool_size)
        self.artifacts = [self.pool.stage(e.path) for e in self.entrants]

    def _reset_pool(self) -> None:
        """Replace the worker pool after an infrastructure failure."""
        if self.pool is not None:
            self.pool.close()
            self.pool = None
        try:
            self._start_pool()
        except (WorkerStartError, OSError):
            self.pool = None

    def close(self) -> None:
        if self.pool is not None:
            self.pool.close()

    def _agent(self, seat: int, entrant: int, info: TableInfo) -> SeatAgent:
        if self.isolation == "sandbox":
            if self.pool is None:
                self._start_pool()
            assert self.pool is not None
            return WorkerSeat(seat, self.artifacts[entrant], info, self.pool.acquire, budget_ns=self.budget_ns)
        art = self.artifacts[entrant]
        bot, bot_info = load_bot(art.entry, module_name=f"arena_inproc_{entrant}_{seat}")
        return InProcessSeat(seat, bot, bot_info, info, art.sha256)

    def run(self, a: Assignment, attempt: int = 1, fault: str | None = None) -> TableResult:
        """Play one attempt of ``a``. Every attempt of a table uses the same seed and table ID."""
        from arena.history.canonical import write_history
        from arena.table.service import seeded_table

        t0 = time.perf_counter()
        faults = _InjectedFaults(fault)
        agents: dict[int, SeatAgent] = {}
        result = TableResult(a.index, a.block, a.rotation, a.table_id, dict(a.seats), False, None, 0, 0.0,
                             attempt=attempt)
        table = None
        try:
            if faults.start_fail:
                raise WorkerStartError("injected: worker failed to start")
            for seat, entrant in a.seats.items():
                agent = self._agent(seat, entrant, table_info(self.config, a, seat, self.budget_ns))
                agents[seat] = agent
                agent.start()
            table = seeded_table(self.config, a.seed, table_id=a.table_id)
            play_table(table, agents, before_decision=faults.before_decision(agents))
            result.hands = len(table.hands)
            result.completed = table.complete
            if table.complete:
                result.points = {a.seats[s]: float(p) for s, p in table.points().items()}
        except Exception as exc:  # recorded, never hidden
            result.error = f"{type(exc).__name__}: {exc}"[:500]
            result.invalid_reason = _infrastructure_cause(exc)
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
            result.infrastructure += [{"entrant": e, "seat": seat, "decision": i, "event": ev}
                                      for i, ev in st.infrastructure]
            if result.invalid_reason is None:
                result.invalid_reason = _seat_invalidation(st)
        if self.history_dir is not None and table is not None and result.error is None:
            target = self.history_dir if result.invalid_reason is None else (
                self.history_dir / "invalid" / f"a{attempt}")
            try:
                write_history(table, target)
            except OSError as exc:
                result.error = f"{type(exc).__name__}: {exc}"[:500]
                result.invalid_reason = result.invalid_reason or "host_os_error"
        if self.isolation == "sandbox" and result.invalid_reason in (
                "worker_start_failed", "host_os_error", "host_memory_error"):
            self._reset_pool()
        result.wall_s = time.perf_counter() - t0
        return result


_RUNNER: TableRunner | None = None


def _job_init(config_json: str, entrants: list[Entrant], isolation: Isolation, budget_ns: int,
              history_dir: str | None) -> None:
    global _RUNNER
    config = TournamentConfig.model_validate_json(config_json)
    _RUNNER = TableRunner(config, entrants, isolation=isolation, budget_ns=budget_ns,
                          history_dir=Path(history_dir) if history_dir else None)


def _job_run(a: Assignment, attempt: int = 1, fault: str | None = None) -> TableResult:
    assert _RUNNER is not None
    if fault == "job_exit":
        os._exit(70)
    return _RUNNER.run(a, attempt, fault)


def _play_with_retries(
    plan: Sequence[Assignment],
    policy: RetryPolicy,
    *,
    make_runner: Callable[[], TableRunner],
    jobs: int,
    job_initargs: tuple[Any, ...],
    faults: Mapping[tuple[int, int], str],
    progress: Callable[[int, int, TableResult], None] | None,
) -> tuple[list[TableResult], list[TableResult]]:
    """Play every assignment, rerunning infrastructure-invalid attempts. Returns
    (final attempt per table, every attempt)."""
    pending: deque[tuple[Assignment, int]] = deque((a, 1) for a in plan)
    final: dict[int, TableResult] = {}
    attempts: list[TableResult] = []

    def record(a: Assignment, r: TableResult) -> None:
        attempts.append(r)
        if r.invalid_reason is not None and r.attempt < policy.max_attempts:
            pending.appendleft((a, r.attempt + 1))
            return
        final[a.index] = r
        if progress:
            progress(len(final), len(plan), r)

    if jobs <= 1:
        runner = make_runner()
        try:
            while pending:
                a, n = pending.popleft()
                fault = faults.get((a.index, n))
                if fault == "job_exit":
                    runner.close()
                    record(a, _lost_attempt(a, n, "job_process_lost", "injected: job process lost"))
                    runner = make_runner()
                    continue
                record(a, runner.run(a, n, fault))
        finally:
            runner.close()
    else:
        ctx = get_context("spawn")

        def executor() -> ProcessPoolExecutor:
            return ProcessPoolExecutor(max_workers=jobs, mp_context=ctx, initializer=_job_init, initargs=job_initargs)

        ex = executor()
        inflight: dict[Future[TableResult], tuple[Assignment, int]] = {}
        try:
            while pending or inflight:
                # At most one table per job process in flight, so a lost pool only
                # interrupts tables that were actually running.
                while pending and len(inflight) < jobs:
                    a, n = pending.popleft()
                    inflight[ex.submit(_job_run, a, n, faults.get((a.index, n)))] = (a, n)
                done, _ = wait(inflight, return_when=FIRST_COMPLETED)
                broken = False
                for f in done:
                    a, n = inflight.pop(f)
                    try:
                        record(a, f.result())
                    except BrokenProcessPool as exc:
                        broken = True
                        record(a, _lost_attempt(a, n, "job_process_lost", f"BrokenProcessPool: {exc}"))
                    except Exception as exc:
                        record(a, _lost_attempt(a, n, "job_error", f"{type(exc).__name__}: {exc}"))
                if broken:
                    for f, (a, n) in list(inflight.items()):
                        try:
                            record(a, f.result())
                        except Exception as exc:
                            record(a, _lost_attempt(a, n, "job_process_lost", f"{type(exc).__name__}: {exc}"))
                    inflight.clear()
                    ex.shutdown(wait=True, cancel_futures=True)
                    ex = executor()
        finally:
            ex.shutdown(wait=True, cancel_futures=True)
    attempts.sort(key=lambda r: (r.index, r.attempt))
    return [final[a.index] for a in plan], attempts


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
              budget_ns: int, base_seed: str, attempts: Sequence[TableResult] | None = None) -> dict[str, Any]:
    """``results`` holds the final attempt per table. Points come from valid final
    attempts. Decisions, fallbacks, timing and restarts count every attempt in
    ``attempts`` (defaults to ``results``), including invalidated ones."""
    attempts = list(attempts) if attempts is not None else list(results)
    completed = [r for r in results if r.valid]
    invalid_attempts = [r for r in attempts if r.invalid_reason is not None]
    per_entrant = []
    for i, e in enumerate(entrants):
        blocks: dict[int, list[float]] = {}
        for r in completed:
            blocks.setdefault(r.block, []).append(r.points[i])
        block_means = [sum(v) / len(v) for _, v in sorted(blocks.items())]
        pts = [r.points[i] for r in completed]
        pts_all = pts + [r.points[i] for r in invalid_attempts if r.completed and i in r.points]
        e2e = [x for r in attempts for x in array("q", r.e2e_ns.get(i, b""))]
        wall = [x for r in attempts for x in array("q", r.worker_wall_ns.get(i, b""))]
        cpu = [x for r in attempts for x in array("q", r.worker_cpu_ns.get(i, b""))]
        fallbacks: Counter = Counter()
        for r in attempts:
            fallbacks.update(r.fallbacks.get(i, {}))
        decisions = sum(r.decisions.get(i, 0) for r in attempts)
        restarts = [x for r in attempts for x in r.restarts.get(i, [])]
        per_entrant.append({
            "entrant": i,
            "name": e.name,
            "bot_name": infos[i].name if i < len(infos) else e.name,
            "state_mode": infos[i].state_mode if i < len(infos) else None,
            "builtin": e.builtin,
            "tables": len(pts),
            "mean_points": sum(pts) / len(pts) if pts else None,
            "points_ci95": bootstrap_interval(block_means, f"{base_seed}|{i}"),
            "mean_points_including_invalid_attempts": sum(pts_all) / len(pts_all) if pts_all else None,
            "decisions": decisions,
            "fallbacks": dict(sorted(fallbacks.items())),
            "fallback_total": sum(fallbacks.values()),
            "timeout_fallbacks": fallbacks.get("timeout", 0),
            "restarts": len(restarts),
            "restarts_not_restorable": sum(1 for _, _, ok in restarts if not ok),
            "tables_with_state_loss": sum(1 for r in attempts if i in r.state_lost),
            "late_replies_discarded": sum(r.late_discarded.get(i, 0) for r in attempts),
            "max_state_bytes": max((r.max_state_bytes.get(i, 0) for r in attempts), default=0),
            "timing": timing_summary(e2e, wall, cpu, budget_ns),
        })
    base = [p for p in per_entrant if p["builtin"]]
    base_e2e = sorted(x for r in attempts for i, p in enumerate(per_entrant) if p["builtin"]
                      for x in array("q", r.e2e_ns.get(i, b"")))
    base_decisions = sum(p["decisions"] for p in base)
    base_timeouts = sum(p["timeout_fallbacks"] for p in base)
    return {
        "tables_scheduled": len(results),
        "tables_completed": len(completed),
        "tables_failed": [{"table_id": r.table_id, "error": r.error} for r in results if r.error],
        "tables_invalid": [{"table_id": r.table_id, "attempt_id": r.attempt_id, "reason": r.invalid_reason}
                           for r in results if r.invalid_reason],
        "tables_incomplete": [r.table_id for r in results if not r.completed and not r.error and not r.invalid_reason],
        "attempts_total": len(attempts),
        "reruns": len(attempts) - len(results),
        "invalid_attempts": [{
            "attempt_id": r.attempt_id,
            "table_id": r.table_id,
            "attempt": r.attempt,
            "reason": r.invalid_reason,
            "error": r.error,
            "completed": r.completed,
            "hands": r.hands,
            "points_by_entrant": {str(k): v for k, v in sorted(r.points.items())},
            "infrastructure": r.infrastructure,
        } for r in invalid_attempts],
        "infrastructure_restored": [{"attempt_id": r.attempt_id, "infrastructure": r.infrastructure}
                                    for r in attempts if r.infrastructure and r.invalid_reason is None],
        "blocks": len({r.block for r in results}),
        "entrants": per_entrant,
        "baselines_combined": {
            "decisions": base_decisions,
            "timeout_fallbacks": base_timeouts,
            "timeout_rate": base_timeouts / base_decisions if base_decisions else None,
            "fallback_total": sum(p["fallback_total"] for p in base),
            "timing": timing_summary(base_e2e, [], [], budget_ns),
        },
        "table_wall_s_mean": sum(r.wall_s for r in attempts) / len(attempts) if attempts else None,
    }


# ------------------------------------------------------------------ manifest


class BenchmarkManifest(RunManifest):
    """The section 10 run manifest, extended for agent benchmarks."""

    manifest_kind: Literal["agent_benchmark"] = "agent_benchmark"
    isolation: Literal["sandbox", "in_process"]
    competitive: bool
    budget_ns: StrictInt
    retry: dict[str, Any] = Field(default_factory=dict)
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
    """The final attempt of every scheduled table, in schedule order."""
    out_dir: Path | None
    attempts: list[TableResult] = field(default_factory=list)
    """Every attempt, including those invalidated by infrastructure and rerun."""


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
    retry_policy: RetryPolicy | None = None,
    _faults: Mapping[tuple[int, int], str] | None = None,
) -> BenchmarkRun:
    """Run the benchmark. ``progress`` is called once per scheduled table, with its
    final attempt. ``_faults`` maps (assignment index, attempt) to an injected
    infrastructure fault and exists for tests only."""
    budget_ns = budget_ns or config.bot_budget_ns
    infos = inspect_entrants(config, entrants, isolation)
    plan = schedule(config, len(entrants), blocks, base_seed)
    seeds = derive_seeds(base_seed, blocks)
    run_id = run_id or "b_" + canonical_hash(
        [config.config_hash(), base_seed, blocks, [artifact_hash(e.path) for e in entrants], isolation],
        domain="arena-benchmark-id-v1")[:12]
    history_dir = (out_dir / "tables") if (out_dir is not None and save_histories) else None
    policy = retry_policy or RetryPolicy()
    results, attempts = _play_with_retries(
        plan, policy,
        make_runner=lambda: TableRunner(config, entrants, isolation=isolation, budget_ns=budget_ns,
                                        history_dir=history_dir),
        jobs=jobs,
        job_initargs=(config.model_dump_json(), list(entrants), isolation, budget_ns,
                      str(history_dir) if history_dir else None),
        faults=_faults or {},
        progress=progress,
    )
    summary = json.loads(canonical_json(summarize(results, entrants, infos, budget_ns, base_seed,
                                                  attempts=attempts)))
    competitive = isolation == "sandbox" and budget_ns == config.bot_budget_ns
    seats = config.resolved_seats
    seat_points = {s: 0.0 for s in seats}
    completed = [r for r in results if r.valid]
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
        retry_policy=policy.describe(),
        results=RunResults(
            completed_table_ids=[r.table_id for r in completed],
            failed_tables=[FailedTable(table_id=r.table_id, error=_failure_text(r)) for r in results
                           if r.error or r.invalid_reason],
            incomplete_table_ids=[r.table_id for r in results
                                  if not r.completed and not r.error and not r.invalid_reason],
            mean_points_by_seat={s: (v / len(completed) if completed else 0.0) for s, v in seat_points.items()},
        ),
        isolation=isolation,
        competitive=competitive,
        budget_ns=budget_ns,
        retry=policy.as_dict(),
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
    return BenchmarkRun(manifest, results, out_dir, attempts)


def _failure_text(r: TableResult) -> str:
    if r.invalid_reason is None:
        return r.error or ""
    text = f"invalid after {r.attempt} attempt{'s' if r.attempt > 1 else ''}: {r.invalid_reason}"
    return f"{text}; {r.error}" if r.error else text


__all__ = [
    "INFRASTRUCTURE_CAUSES",
    "AgentLoadError",
    "Assignment",
    "BenchmarkManifest",
    "Entrant",
    "RetryPolicy",
    "TableResult",
    "TableRunner",
    "bootstrap_interval",
    "inspect_entrants",
    "run_benchmark",
    "schedule",
    "summarize",
]
