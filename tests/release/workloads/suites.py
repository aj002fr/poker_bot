"""Gates that run the existing pytest and Playwright suites: fast, fault, browser, isolation.

Each shard is one suite invocation with JUnit output. Every test case is
parsed: failures and errors fail the shard, and so does any skip that is not
on the allowlist below. Allowed skips are structural (a test parameter that
does not exist for one store) and are reported as exclusions with their
reason, never as passes. Tests a gate requires by name must appear and pass.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]

# (test id, substring of the skip reason) -> why the skip is structural.
ALLOWED_SKIPS: dict[tuple[str, str], str] = {
    ("tests.persistence.test_faults::test_ambiguous_commit_is_resolved_from_the_store[postgres]",
     "lost-acknowledgement injection exists only in the memory store"):
        "the lost-acknowledgement fault is injected inside the memory store; PostgreSQL commit ambiguity is "
        "covered by tests.persistence.test_process_kill",
    ("tests.persistence.test_store_contract::test_postgres_migrations_are_idempotent[memory]",
     "migrations exist only for PostgreSQL"):
        "the memory store has no schema migrations",
    ("::tests.strategy.test_openspiel_oracle", "could not import 'pyspiel'"):
        "OpenSpiel is the opt-in `oracle` dependency group; make verify-oracle runs these cross-checks",
}

PG = {"ARENA_REQUIRE_POSTGRES": "1"}
# The solver tests need numba from the opt-in `train` group. ARENA_REQUIRE_NUMBA makes pytest
# exit with a usage error when it is missing instead of leaving those modules uncollected.
TRAIN = {"uv_args": ["--group", "train"], "env": {"ARENA_REQUIRE_NUMBA": "1"}}
AGENT_FAULTS = ["tests/agents/test_worker_faults.py", "tests/agents/test_deadline.py", "tests/agents/test_retry.py",
                "tests/agents/test_restore.py", "tests/agents/test_t12.py"]

SHARDS: dict[str, list[dict[str, Any]]] = {
    "fast": [
        {"name": "verify-fast", "kind": "pytest", "args": ["-m", "not release"],
         "uv_args": TRAIN["uv_args"], "env": {**PG, **TRAIN["env"]},
         "require": ["tests.golden.test_core_fixtures::", "tests.property.test_generated_hands::",
                     "tests.golden.test_tournament_fixtures::test_icm_fixture",
                     "tests.metamorphic.test_metamorphic::", "tests.regressions.test_failing_seeds::",
                     "tests.strategy.test_fixture_gates::", "tests.strategy.test_regret_matching::"]},
    ],
    "fault": [
        {"name": "persistence (memory + PostgreSQL)", "kind": "pytest", "args": ["tests/persistence"], "env": PG,
         "require": ["tests.persistence.test_faults::test_action_racing_timeout_100_times[memory]",
                     "tests.persistence.test_faults::test_action_racing_timeout_100_times[postgres]",
                     "tests.persistence.test_faults::test_obsolete_writer_is_fenced_and_cannot_fork_the_table[postgres]",
                     "tests.persistence.test_faults::test_storage_outage_pauses_and_restores_human_think_time[postgres]",
                     "tests.persistence.test_process_kill::"]},
        {"name": "API on the memory store", "kind": "pytest", "args": ["tests/api"],
         "env": {"ARENA_API_TEST_STORE": "memory"},
         "require": ["tests.api.test_tables_api::test_action_id_idempotency",
                     "tests.api.test_tables_api::test_on_time_action_waiting_in_queue_beats_a_later_timeout"]},
        {"name": "API on PostgreSQL", "kind": "pytest", "args": ["tests/api"],
         "env": {"ARENA_API_TEST_STORE": "postgres", **PG},
         "require": ["tests.api.test_tables_api::test_action_id_idempotency",
                     "tests.api.test_tables_api::test_on_time_action_waiting_in_queue_beats_a_later_timeout"]},
        {"name": "worker faults, deadlines, retries, restore", "kind": "pytest", "args": AGENT_FAULTS,
         "require": ["tests.agents.test_worker_faults::test_crashing_worker_is_restored_from_committed_state",
                     "tests.agents.test_deadline::test_every_decision_consumed_exactly_once_under_mixed_faults",
                     "tests.agents.test_retry::test_parallel_lost_job_process_is_rerun_and_matches_serial"]},
    ],
    "browser": [
        {"name": "Playwright e2e (deterministic test clock)", "kind": "playwright",
         "require": ["mixes.spec.ts::", "layout.spec.ts::", "reconnect.spec.ts::", "keyboard.spec.ts::"]},
    ],
    "isolation": [
        {"name": "sandbox negative tests", "kind": "pytest", "args": ["tests/agents/test_sandbox.py"],
         "require": ["tests.agents.test_sandbox::test_network_denied", "tests.agents.test_sandbox::test_filesystem_reads_denied",
                     "tests.agents.test_sandbox::test_process_signal_and_ipc_access_denied"]},
        {"name": "projection, credential and replay authorization", "kind": "pytest",
         "args": ["tests/api/test_leaks.py", "tests/property/test_information_sets.py", "tests/api/test_replay.py",
                  "tests/history/test_replay.py", "tests/api/test_tables_api.py", "-k",
                  "leak or views or visible or review or perspective or replay or credential or projection or "
                  "stream or requires or snapshot"],
         "require": ["tests.api.test_leaks::test_no_unauthorized_cards_in_any_body_frame_or_error",
                     "tests.property.test_information_sets::test_views_identical_across_hidden_card_worlds",
                     "tests.api.test_tables_api::test_credentials_are_bound_to_their_table",
                     "tests.api.test_replay::test_perspective_permissions"]},
    ],
}


def shards(gate: str) -> list[dict[str, Any]]:
    return [{"gate": gate, **s} for s in SHARDS[gate]]


def _env(extra: dict[str, str]) -> dict[str, str]:
    env = {**os.environ, **extra}
    uv = shutil.which("uv") or str(Path.home() / ".local/bin/uv")
    env["PATH"] = f"{Path(uv).parent}:{env.get('PATH', '')}"
    env["UV"] = uv
    env.pop("PYTHONHASHSEED", None)
    return env


def _command(params: dict[str, Any], junit: Path) -> tuple[list[list[str]], Path]:
    if params["kind"] == "pytest":
        return [["uv", "run", *params.get("uv_args", []), "pytest", "-q", "-p", "no:cacheprovider", "-o", "junit_family=xunit1",
                 f"--junitxml={junit}", *params["args"]]], ROOT
    return [["npx", "tsc", "-b"], ["npx", "playwright", "test", "--reporter=junit,list"]], ROOT / "web"


def _cases(junit: Path, kind: str) -> list[dict[str, Any]]:
    out = []
    for tc in ET.parse(junit).getroot().iter("testcase"):
        name = tc.get("name", "")
        cls = tc.get("classname", "")
        if kind == "playwright":
            test_id = f"{cls}::{name}"
            node = f"e2e/{cls} -g {name!r}"
        else:
            test_id = f"{cls}::{name}"
            node = f"{tc.get('file') or cls.replace('.', '/') + '.py'}::{name}"
        outcome, message = "passed", ""
        for child in tc:
            if child.tag in ("failure", "error"):
                outcome, message = "failed", (child.get("message") or child.text or "")[:2000]
            elif child.tag == "skipped" and outcome == "passed":
                outcome, message = "skipped", " ".join(filter(None, (child.get("message"), child.text)))
        out.append({"id": test_id, "node": node, "outcome": outcome, "message": message,
                    "time_s": float(tc.get("time") or 0)})
    return out


def _allowed(test_id: str, message: str) -> str | None:
    for (tid, reason), why in ALLOWED_SKIPS.items():
        if tid == test_id and reason in message:
            return why
    return None


def run_shard(params: dict[str, Any]) -> dict[str, Any]:
    t0 = time.perf_counter()
    env = _env(params.get("env", {}))
    with tempfile.TemporaryDirectory() as tmp:
        junit = Path(tmp) / "junit.xml"
        if params["kind"] == "playwright":
            env["PLAYWRIGHT_JUNIT_OUTPUT_NAME"] = str(junit)
        commands, cwd = _command(params, junit)
        steps = []
        for cmd in commands:
            proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=3 * 3600)
            steps.append({"command": " ".join(cmd), "returncode": proc.returncode,
                          "output_tail": (proc.stdout + proc.stderr)[-3000:]})
            if proc.returncode != 0 and cmd[:2] == ["npx", "tsc"]:
                break
        cases = _cases(junit, params["kind"]) if junit.exists() else []
    failures: list[dict[str, Any]] = []
    counts = {"tests": len(cases), "tests_passed": 0, "tests_failed": 0, "tests_skipped_allowed": 0,
              "tests_skipped_unexpected": 0}
    exclusions: dict[str, int] = {}
    prefix = "cd web && " if params["kind"] == "playwright" else ""
    runner = "npx playwright test" if params["kind"] == "playwright" else " ".join(
        ["uv", "run", *params.get("uv_args", []), "pytest"])
    for c in cases:
        if c["outcome"] == "passed":
            counts["tests_passed"] += 1
        elif c["outcome"] == "failed":
            counts["tests_failed"] += 1
            failures.append({"case": {"gate": params["gate"], "test": c["id"]}, "error": c["message"],
                             "traceback": "", "repro": f"{prefix}{runner} '{c['node']}'"})
        elif (why := _allowed(c["id"], c["message"])) is not None:
            counts["tests_skipped_allowed"] += 1
            exclusions[f"skipped {c['id']}: {why}"] = 1
        else:
            counts["tests_skipped_unexpected"] += 1
            failures.append({"case": {"gate": params["gate"], "test": c["id"]},
                             "error": f"unexpected skip (a skip is never a pass): {c['message'][:500]}",
                             "traceback": "", "repro": f"{prefix}{runner} -rs '{c['node']}'"})
    for want in params.get("require", []):
        hits = [c for c in cases if (c["id"].startswith(want) if want.endswith("::")
                                     else c["id"] == want or c["id"].startswith(want + "["))]
        if not hits or any(c["outcome"] != "passed" for c in hits):
            failures.append({"case": {"gate": params["gate"], "required": want},
                             "error": f"required test {want!r} " + ("did not run" if not hits else "did not pass"),
                             "traceback": "", "repro": ""})
    bad_exit = [s for s in steps if s["returncode"] != 0]
    if bad_exit and not failures:
        failures.append({"case": {"gate": params["gate"], "suite": params["name"]},
                         "error": f"{bad_exit[0]['command']} exited {bad_exit[0]['returncode']} with no failing test "
                                  f"case recorded: {bad_exit[0]['output_tail'][-1500:]}", "traceback": "", "repro": ""})
    if not cases and not failures:
        failures.append({"case": {"gate": params["gate"], "suite": params["name"]}, "error": "no test cases ran",
                         "traceback": "", "repro": ""})
    h = hashlib.sha256()
    for c in sorted(cases, key=lambda c: c["id"]):
        h.update(f"{c['id']}|{c['outcome']}\n".encode())
    return {
        "cases": len(cases),
        "failed_cases": len(failures),
        "counts": counts,
        "exclusions": exclusions,
        "failures": failures[:50],
        "result_hash": h.hexdigest(),
        "suite": {"name": params["name"], "steps": [{k: v for k, v in s.items() if k != "output_tail"} | (
            {"output_tail": s["output_tail"][-800:]} if s["returncode"] else {}) for s in steps],
                  "wall_s": round(time.perf_counter() - t0, 1),
                  "slowest": sorted(({"id": c["id"], "time_s": c["time_s"]} for c in cases),
                                    key=lambda c: -c["time_s"])[:5]},
    }


def finalize(gate: str, results: list[dict[str, Any]], all_results: dict[str, Any]) -> list[dict[str, Any]]:
    tests = sum(r["counts"].get("tests", 0) for r in results)
    unexpected = sum(r["counts"].get("tests_skipped_unexpected", 0) for r in results)
    return [
        {"check": "every suite shard ran test cases", "passed": all(r["cases"] > 0 for r in results),
         "detail": {r["params"]["name"]: r["cases"] for r in results}},
        {"check": "no unexpected skips", "passed": unexpected == 0, "detail": {"tests": tests, "unexpected": unexpected}},
    ]
