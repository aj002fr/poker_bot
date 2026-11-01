"""Execution metrics for one entrant of a benchmark run: latency, fallbacks, errors, size, state.

Latency quantiles include p99.9, computed from the runner's raw per-decision
timings across every attempt (invalidated attempts included, as the runner does).
In-process runs measure a direct call, not the sandboxed round trip, and are
labelled not competitive. Resident memory is not measured by the arena; the
sandbox enforces a 1 GiB address-space ceiling, recorded here as a limit only.
"""

from __future__ import annotations

import math
from array import array
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

ILLEGAL_REASONS = ("illegal_action", "invalid_schema", "invalid_output", "oversized_output")
ERROR_REASONS = ("bot_exception", "bot_memory_error", "worker_crash", "worker_unavailable", "state_invalid",
                 "state_too_large")
SANDBOX_ADDRESS_SPACE_LIMIT = 1 << 30
QUANTILES = (("p50", 0.50), ("p95", 0.95), ("p99", 0.99), ("p99_9", 0.999))


def _nearest_rank(xs: Sequence[int], q: float) -> int | None:
    if not xs:
        return None
    return xs[min(len(xs) - 1, max(0, math.ceil(q * len(xs)) - 1))]


def latency(samples_ns: Sequence[int]) -> dict[str, float | int | None]:
    s = sorted(samples_ns)
    out: dict[str, float | int | None] = {"samples": len(s)}
    for name, q in QUANTILES:
        v = _nearest_rank(s, q)
        out[name] = None if v is None else v / 1e9
    out["max"] = s[-1] / 1e9 if s else None
    return out


def entrant_execution(attempts: Sequence[Any], entrant: int, *, budget_ns: int, isolation: str, competitive: bool,
                      artifact_bytes: int | None = None, failed_tables: int = 0, invalid_tables: int = 0,
                      reruns: int = 0) -> dict[str, Any]:
    """``attempts`` are runner ``TableResult`` objects (every attempt)."""
    e2e: list[int] = []
    wall: list[int] = []
    cpu: list[int] = []
    fallbacks: Counter = Counter()
    decisions = restarts = not_restorable = late = 0
    max_state = 0
    for r in attempts:
        e2e.extend(array("q", r.e2e_ns.get(entrant, b"")))
        wall.extend(array("q", r.worker_wall_ns.get(entrant, b"")))
        cpu.extend(array("q", r.worker_cpu_ns.get(entrant, b"")))
        fallbacks.update(r.fallbacks.get(entrant, {}))
        decisions += r.decisions.get(entrant, 0)
        rs = r.restarts.get(entrant, [])
        restarts += len(rs)
        not_restorable += sum(1 for _, _, ok in rs if not ok)
        late += r.late_discarded.get(entrant, 0)
        max_state = max(max_state, r.max_state_bytes.get(entrant, 0))
    total = sum(fallbacks.values())
    return {
        "isolation": isolation,
        "competitive": competitive,
        "timing_note": None if competitive else "in-process or non-default budget: direct-call timing, not competitive",
        "decisions": decisions,
        "latency_end_to_end_s": latency(e2e),
        "latency_worker_wall_s": latency(wall),
        "latency_worker_cpu_s": latency(cpu),
        "budget_s": budget_ns / 1e9,
        "over_budget_decisions": sum(1 for v in e2e if v > budget_ns),
        "fallbacks": dict(sorted(fallbacks.items())),
        "fallback_total": total,
        "fallback_rate": total / decisions if decisions else None,
        "timeouts": fallbacks.get("timeout", 0),
        "illegal_actions": sum(fallbacks.get(k, 0) for k in ILLEGAL_REASONS),
        "errors": sum(fallbacks.get(k, 0) for k in ERROR_REASONS),
        "restarts": restarts,
        "restarts_not_restorable": not_restorable,
        "late_replies_discarded": late,
        "max_state_bytes": max_state,
        "artifact_bytes": artifact_bytes,
        "memory": {"measured": False, "sandbox_address_space_limit_bytes": SANDBOX_ADDRESS_SPACE_LIMIT
                   if isolation == "sandbox" else None},
        "tables_failed": failed_tables,
        "tables_invalid": invalid_tables,
        "reruns": reruns,
    }


ENTRANT_TIMEOUT_MIN_COUNT = 5


def seat_timeouts(attempts: Sequence[Any], labels: Mapping[int, str]) -> dict[str, Any]:
    """Decisions and timeouts of every entrant (candidate and opponents) over every attempt."""
    by: dict[str, dict[str, int]] = {}
    for r in attempts:
        for e, n in r.decisions.items():
            row = by.setdefault(labels.get(e, str(e)), {"decisions": 0, "timeouts": 0})
            row["decisions"] += n
            row["timeouts"] += r.fallbacks.get(e, {}).get("timeout", 0)
    return by


def merge_seat_timeouts(parts: Sequence[Mapping[str, Mapping[str, int]]]) -> dict[str, Any]:
    by: dict[str, dict[str, int]] = {}
    for part in parts:
        for name, row in part.items():
            acc = by.setdefault(name, {"decisions": 0, "timeouts": 0})
            acc["decisions"] += row["decisions"]
            acc["timeouts"] += row["timeouts"]
    decisions = sum(r["decisions"] for r in by.values())
    timeouts = sum(r["timeouts"] for r in by.values())
    return {"decisions": decisions, "timeouts": timeouts, "timeout_rate": timeouts / decisions if decisions else 0.0,
            "by_entrant": {k: {**v, "timeout_rate": v["timeouts"] / v["decisions"] if v["decisions"] else 0.0}
                           for k, v in sorted(by.items())}}


def timeout_integrity(per_arm: Mapping[str, Mapping[str, Any]], gates: Mapping[str, Any]) -> dict[str, Any]:
    """Whether the deadline held well enough for the points to mean anything.

    ``per_arm`` maps an arm label to :func:`merge_seat_timeouts` output. Fails when an
    arm's pooled timeout rate over all seats exceeds ``max_timeout_rate``, or any entrant
    has at least ``ENTRANT_TIMEOUT_MIN_COUNT`` timeouts and a rate above
    ``max_entrant_timeout_rate``.
    """
    pooled_limit, entrant_limit = gates.get("max_timeout_rate"), gates.get("max_entrant_timeout_rate")
    failures = []
    for arm, t in sorted(per_arm.items()):
        if pooled_limit is not None and t["timeout_rate"] > pooled_limit:
            failures.append(f"arm {arm}: {t['timeouts']}/{t['decisions']} decisions over all seats timed out "
                            f"({t['timeout_rate']:.2%} > {pooled_limit:.2%})")
        if entrant_limit is not None:
            for name, row in t["by_entrant"].items():
                if row["timeouts"] >= ENTRANT_TIMEOUT_MIN_COUNT and row["timeout_rate"] > entrant_limit:
                    failures.append(f"arm {arm}: entrant {name} timed out on {row['timeouts']}/{row['decisions']} "
                                    f"decisions ({row['timeout_rate']:.2%} > {entrant_limit:.2%})")
    return {"passed": not failures, "failures": failures, "max_timeout_rate": pooled_limit,
            "max_entrant_timeout_rate": entrant_limit, "entrant_min_count": ENTRANT_TIMEOUT_MIN_COUNT,
            "arms": {arm: dict(t) for arm, t in sorted(per_arm.items())}}


def merge_execution(parts: Sequence[Mapping[str, Any]], raw: Mapping[str, Sequence[int]]) -> dict[str, Any]:
    """Combine per-cell summaries; ``raw`` holds the pooled timing samples for exact quantiles."""
    if not parts:
        return {}
    out = dict(parts[0])
    for key in ("decisions", "over_budget_decisions", "fallback_total", "timeouts", "illegal_actions", "errors",
                "restarts", "restarts_not_restorable", "late_replies_discarded", "tables_failed", "tables_invalid",
                "reruns"):
        out[key] = sum(p[key] for p in parts)
    fb: Counter = Counter()
    for p in parts:
        fb.update(p["fallbacks"])
    out["fallbacks"] = dict(sorted(fb.items()))
    out["fallback_rate"] = out["fallback_total"] / out["decisions"] if out["decisions"] else None
    out["max_state_bytes"] = max(p["max_state_bytes"] for p in parts)
    out["competitive"] = all(p["competitive"] for p in parts)
    for key in ("latency_end_to_end_s", "latency_worker_wall_s", "latency_worker_cpu_s"):
        out[key] = latency(raw.get(key, ()))
    return out


def evaluate_gates(execution: Mapping[str, Any], gates: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Each gate: name, limit, observed, passed. ``None`` limits are not checked."""
    checks = []

    def add(name: str, limit: Any, observed: Any, ok: bool) -> None:
        checks.append({"gate": name, "limit": limit, "observed": observed, "passed": ok})

    if gates.get("max_illegal_actions") is not None:
        add("illegal_actions", gates["max_illegal_actions"], execution["illegal_actions"],
            execution["illegal_actions"] <= gates["max_illegal_actions"])
    if gates.get("max_errors") is not None:
        add("errors", gates["max_errors"], execution["errors"], execution["errors"] <= gates["max_errors"])
    if gates.get("max_timeouts") is not None:
        add("timeouts", gates["max_timeouts"], execution["timeouts"], execution["timeouts"] <= gates["max_timeouts"])
    if gates.get("max_fallback_rate") is not None:
        rate = execution["fallback_rate"] or 0.0
        add("fallback_rate", gates["max_fallback_rate"], rate, rate <= gates["max_fallback_rate"])
    if gates.get("max_p99_latency_s") is not None:
        p99 = execution["latency_end_to_end_s"]["p99"]
        add("p99_latency_s", gates["max_p99_latency_s"], p99, p99 is not None and p99 <= gates["max_p99_latency_s"])
    if gates.get("max_artifact_bytes") is not None and execution.get("artifact_bytes") is not None:
        add("artifact_bytes", gates["max_artifact_bytes"], execution["artifact_bytes"],
            execution["artifact_bytes"] <= gates["max_artifact_bytes"])
    if gates.get("max_failed_tables") is not None:
        bad = execution["tables_failed"] + execution["tables_invalid"]
        add("failed_or_invalid_tables", gates["max_failed_tables"], bad, bad <= gates["max_failed_tables"])
    if gates.get("require_competitive"):
        add("competitive_timing", True, execution["competitive"], bool(execution["competitive"]))
    return checks
