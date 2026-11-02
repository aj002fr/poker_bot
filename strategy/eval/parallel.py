"""How many benchmark jobs a host can run without corrupting the 1 ms decision deadline.

Why sandboxed runs need room
    The deadline is wall clock: a reply counts only if the parent has received it
    within ``budget_ns`` of dispatch, and a late reply is a forced check or fold. One
    sandboxed job (an arena benchmark job process) keeps two to three cores busy:

    - the job's parent busy-spins for up to the whole budget while it waits for each
      reply, so it does not pay for a wake-up;
    - the deciding worker needs its own core at the same moment;
    - workers are never reused across tables: each table takes one fresh worker per
      seat and the pool immediately starts replacements, one after another, each a
      new sandboxed Python process (about 130 ms of start-up). That start-up runs
      during the first hands of every table.

Measurements (8-CPU VM, 6 seats, 1 ms budget; pooled timeout rate over all seats)
    - Arena styles and nlhe_heuristic, 24 tables: one job averaged 1.8 busy cores and
      peaked at 2.4. Timeouts: 0-0.02% with 1 job, 0.03-0.3% with 2, 30-33% with 4,
      66% with 8. One job confined to 2 CPUs: 43%; to 3 CPUs: under 0.1%.
    - strategy.population lineups, about 25,000 decisions per setting: 0.02% with 1
      job; 0.12-0.24% with 2 jobs, whether unpinned, pinned to 3 cores each, or with
      the harness process pinned too. With 2 jobs the timeouts concentrate in the first
      75 decisions of each table (0.49%, against 0.08% mid-table): the replacement
      start-ups, not other jobs, take the deciding worker's core. Pinning removes
      cross-job interference but not that, so the harness does not pin.

Policy (:mod:`arena.benchmark.capacity`, shared with the arena CLI)
    A sandboxed job gets ``CORES_PER_SANDBOX_JOB`` (4) physical cores: parent, deciding
    worker, a starting replacement and slack. One core is reserved for the harness
    process, which parses every table history. Physical cores, not hardware threads:
    SMT siblings share execution units. That is 1 job on 8 cores, 3 on 16 and 7 on 32.
    In-process runs enforce no deadline (their timing is labelled not competitive), so
    they may use every core but one. The usable set honours the process's CPU
    affinity and any cgroup v2 CPU quota.

A larger ``--jobs`` is allowed with a warning; the timeout integrity gate
(:func:`strategy.eval.execution.timeout_integrity`) withholds the points of any run
whose deadline did not hold.
"""

from __future__ import annotations

from typing import Any

from arena.benchmark.capacity import (
    CORES_PER_SANDBOX_JOB,
    RESERVED_CORES,
    capacity,
    cgroup_cpu_limit,
    physical_cores,
    safe_jobs,
    usable_cpus,
)


def oversubscription_warning(isolation: str, jobs: int, cap: dict[str, Any] | None = None) -> str | None:
    cap = cap or capacity()
    safe = safe_jobs(isolation, cap)
    if isolation != "sandbox" or jobs <= safe:
        return None
    return (f"--jobs {jobs} exceeds the safe {safe} for sandboxed runs on this host ({cap['usable_cores']} usable "
            f"cores, {CORES_PER_SANDBOX_JOB} per job + {RESERVED_CORES} reserved): deadline timeouts (forced folds) "
            f"will rise; the timeout integrity gate withholds the points if they pass its limit")


__all__ = ["CORES_PER_SANDBOX_JOB", "RESERVED_CORES", "capacity", "cgroup_cpu_limit", "oversubscription_warning",
           "physical_cores", "safe_jobs", "usable_cpus"]
