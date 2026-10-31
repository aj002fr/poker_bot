"""Shared pieces for gate workloads: case runner, failure records, digests."""

from __future__ import annotations

import hashlib
import json
import shlex
import traceback
from collections import Counter
from collections.abc import Callable
from typing import Any

MAX_RECORDED_FAILURES = 25


def case_json(case: dict[str, Any]) -> str:
    return json.dumps(case, sort_keys=True, separators=(",", ":"))


def repro_command(case: dict[str, Any]) -> str:
    return f"uv run python -m tests.release repro {shlex.quote(case_json(case))}"


def digest(obj: Any) -> str:
    """Stable digest of JSON-like data (int dict keys are stringified)."""
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode()).hexdigest()


class CaseRunner:
    """Runs cases, records every failure with the case that reproduces it, and hashes results.

    The result hash folds in each case's digest (or a failure marker) in order,
    so a rerun of the same shard on the same code gives the same hash.
    """

    def __init__(self, gate: str) -> None:
        self.gate = gate
        self.counts: Counter = Counter()
        self.exclusions: Counter = Counter()
        self.failures: list[dict[str, Any]] = []
        self.failed_cases = 0
        self.cases = 0
        self._hash = hashlib.sha256(f"arena-release-v1|{gate}".encode())

    def run(self, case: dict[str, Any], fn: Callable[[dict[str, Any]], str]) -> bool:
        case = {"gate": self.gate, **case}
        self.cases += 1
        try:
            result = fn(case)
        except Exception as exc:  # every failure is recorded, never swallowed
            self._record(case, f"{type(exc).__name__}: {exc}", "".join(traceback.format_exception(exc)))
            return False
        self._hash.update(f"{result}\n".encode())
        return True

    def fail(self, case: dict[str, Any], message: str) -> None:
        """Record a failure found outside :meth:`run` (for example a cross-check)."""
        self._record({"gate": self.gate, **case}, message, "")

    def _record(self, case: dict[str, Any], error: str, tb: str) -> None:
        self.failed_cases += 1
        if len(self.failures) < MAX_RECORDED_FAILURES:
            self.failures.append({
                "case": case,
                "error": error[:2000],
                "traceback": tb[-4000:],
                "repro": repro_command(case),
            })
        self._hash.update(f"FAIL|{case_json(case)}\n".encode())

    def result(self, **extra: Any) -> dict[str, Any]:
        return {
            "cases": self.cases,
            "failed_cases": self.failed_cases,
            "counts": dict(sorted(self.counts.items())),
            "exclusions": dict(sorted(self.exclusions.items())),
            "failures": self.failures,
            "result_hash": self._hash.hexdigest(),
            **extra,
        }
