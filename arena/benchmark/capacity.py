"""How many sandboxed benchmark jobs a host can run without breaking the 1 ms deadline.

One sandboxed job keeps two to three cores busy: its parent busy-spins for the whole
budget while it waits for each reply, the deciding worker needs its own core at the
same moment, and the pool starts a fresh sandboxed worker for every seat of every
table during the table's first hands. So a sandboxed job gets
``CORES_PER_SANDBOX_JOB`` (4) physical cores, and ``RESERVED_CORES`` (1) is kept for
the process that drives the run. Physical cores, not hardware threads: SMT siblings
share execution units. That is 1 job on 8 cores, 3 on 16 and 7 on 32. In-process
runs enforce no deadline, so they may use every core but the reserved one. The
usable set honours the process's CPU affinity and any cgroup v2 CPU quota.

Measurements behind the policy are in ``docs/agent-sandbox.md``.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

CORES_PER_SANDBOX_JOB = 4
RESERVED_CORES = 1
_SYS_CPU = Path("/sys/devices/system/cpu")


def usable_cpus() -> list[int]:
    try:
        return sorted(os.sched_getaffinity(0))
    except AttributeError:
        return list(range(os.cpu_count() or 1))


def cgroup_cpu_limit(root: Path = Path("/sys/fs/cgroup")) -> float | None:
    """CPUs allowed by a cgroup v2 ``cpu.max`` quota, or None when unlimited or unknown."""
    try:
        quota, period = (root / "cpu.max").read_text().split()[:2]
    except (OSError, ValueError):
        return None
    if quota == "max":
        return None
    return int(quota) / int(period)


def physical_cores(cpus: Sequence[int] | None = None, sys_cpu: Path = _SYS_CPU) -> list[list[int]]:
    """``cpus`` grouped by physical core (SMT siblings together), in CPU order."""
    cpus = list(cpus if cpus is not None else usable_cpus())
    groups: dict[tuple[str, str], list[int]] = {}
    for c in cpus:
        topo = sys_cpu / f"cpu{c}" / "topology"
        try:
            key = ((topo / "physical_package_id").read_text().strip(), (topo / "core_id").read_text().strip())
        except OSError:
            key = ("?", str(c))
        groups.setdefault(key, []).append(c)
    return sorted(groups.values(), key=lambda g: g[0])


def capacity(cpus: Sequence[int] | None = None, sys_cpu: Path = _SYS_CPU,
             cgroup_root: Path = Path("/sys/fs/cgroup")) -> dict[str, Any]:
    cpus = list(cpus if cpus is not None else usable_cpus())
    cores = physical_cores(cpus, sys_cpu)
    limit = cgroup_cpu_limit(cgroup_root)
    usable = len(cores) if limit is None else min(len(cores), max(1, math.floor(limit)))
    return {"cpus": cpus, "physical_cores": len(cores), "cgroup_cpu_limit": limit, "usable_cores": usable}


def safe_jobs(isolation: str, cap: dict[str, Any] | None = None) -> int:
    """The default number of parallel benchmark jobs for ``isolation`` on this host."""
    usable = (cap or capacity())["usable_cores"]
    if isolation == "sandbox":
        return max(1, (usable - RESERVED_CORES) // CORES_PER_SANDBOX_JOB)
    return max(1, usable - RESERVED_CORES)


__all__ = ["CORES_PER_SANDBOX_JOB", "RESERVED_CORES", "capacity", "cgroup_cpu_limit", "physical_cores", "safe_jobs",
           "usable_cpus"]
