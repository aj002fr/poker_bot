"""Aggregate every shard into ``artifacts/test-report.json`` and ``artifacts/test-report.md``.

A gate passes only when every one of its shards is present, up to date and
passed, and every gate-level check over the merged shards passes. Missing,
stale, failed or errored shards make the gate ``incomplete`` or ``failed``, and
the process exits 1. Inapplicable gates (the official-kit gate) are reported
as ``not_applicable`` with their reason; they are never counted as passed.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from collections import Counter
from datetime import UTC, datetime
from typing import Any

from tests.release.harness import ARTIFACTS, GATES, RELEASE_DIR, ROOT, inputs_fingerprint, shard_refs

SCHEMA = "arena-test-report.v1"


def _cmd(args: list[str]) -> str | None:
    try:
        return subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=20).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def environment() -> dict[str, Any]:
    import arena
    from arena.rules.pokerkit_adapter import POKERKIT_VERSION

    mem = None
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal:"):
                mem = f"{int(line.split()[1]) / 1024 / 1024:.1f} GiB"
    except OSError:
        pass
    cpu = None
    try:
        cpu = next(line.split(":", 1)[1].strip() for line in open("/proc/cpuinfo") if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    return {
        "platform": platform.platform(),
        "cpu_model": cpu,
        "cpu_count": os.cpu_count(),
        "memory": mem,
        "python": platform.python_version(),
        "pokerkit": POKERKIT_VERSION,
        "arena": arena.__version__,
        "git_commit": _cmd(["git", "rev-parse", "HEAD"]),
        "git_branch": _cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": bool(_cmd(["git", "status", "--porcelain", "--untracked-files=no"])),
        "node": _cmd(["node", "--version"]),
        "docker": _cmd(["docker", "--version"]),
    }


def _merge_counts(results: list[dict[str, Any]]) -> dict[str, Any]:
    total: Counter = Counter()
    maxima: dict[str, int] = {}
    for r in results:
        for k, v in r.get("counts", {}).items():
            if not isinstance(v, (int, float)):
                continue
            if k.startswith("max_"):
                maxima[k] = max(maxima.get(k, v), v)
            else:
                total[k] += v
    return {**dict(sorted(total.items())), **maxima}


def gate_report(gate, all_results: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    if not gate.applicable:
        return {"id": gate.id, "title": gate.title, "section": gate.section, "requirement": gate.requirement,
                "status": "not_applicable", "reason": gate.not_applicable_reason}
    refs = shard_refs(gate)
    states: Counter = Counter()
    records = []
    problems = []
    for ref in refs:
        state = ref.state()
        states[state] += 1
        rec = ref.load()
        if state in ("passed", "failed") and rec is not None:
            records.append(rec)
        if state != "passed":
            problems.append({"shard": ref.id, "state": state, "params": ref.params,
                             "error": (rec or {}).get("error")})
    results = [{**r["result"], "params": r["params"]} for r in records if "result" in r]
    failures = [f for r in results for f in r.get("failures", [])]
    exclusions: Counter = Counter()
    for r in results:
        exclusions.update(r.get("exclusions", {}))
    checks: list[dict[str, Any]] = []
    finalize = getattr(gate.workload(), "finalize", None)
    if finalize is not None and results:
        try:
            checks = finalize(gate.id, results, all_results)
        except Exception as exc:
            checks = [{"check": "gate-level checks", "passed": False, "detail": f"{type(exc).__name__}: {exc}"}]
    failed_cases = sum(r.get("failed_cases", 0) for r in results)
    if states["failed"] or states["error"] or failed_cases:
        status = "failed"
    elif states["missing"] or states["stale"] or not refs:
        status = "incomplete"
    elif any(not c["passed"] for c in checks):
        status = "failed"
    else:
        status = "passed"
    runtimes = [r.get("runtime_s", 0) for r in records]
    starts = sorted(r["started_at"] for r in records if r.get("started_at"))
    ends = sorted(r["finished_at"] for r in records if r.get("finished_at"))
    details: dict[str, Any] = {}
    for r in results:
        for k, v in r.items():
            if k in ("summary", "timing", "environment", "mutants", "suite", "relations"):
                details.setdefault(k, []).append(v)
    return {
        "id": gate.id, "title": gate.title, "section": gate.section, "requirement": gate.requirement,
        "status": status,
        "inputs_fingerprint": inputs_fingerprint(gate.inputs),
        "shards": {"expected": len(refs), **dict(sorted(states.items()))},
        "shard_problems": problems[:50],
        "cases": sum(r.get("cases", 0) for r in results),
        "failed_cases": failed_cases,
        "counts": _merge_counts(results),
        "checks": checks,
        "exclusions": dict(sorted(exclusions.items())),
        "failures": failures[:50],
        "cpu_time_s": round(sum(runtimes), 1),
        "first_started_at": starts[0] if starts else None,
        "last_finished_at": ends[-1] if ends else None,
        "result_hashes": {r["shard"]: r["result"].get("result_hash") for r in records if "result" in r},
        "details": details,
    }


def build_report() -> int:
    all_results: dict[str, list[dict[str, Any]]] = {}
    for gate in GATES:
        if gate.applicable:
            all_results[gate.id] = [{**r["result"], "params": r["params"]}
                                    for ref in shard_refs(gate) if ref.state() in ("passed", "failed")
                                    for r in [ref.load()] if r and "result" in r]
    gates = [gate_report(g, all_results) for g in GATES]
    applicable = [g for g in gates if g["status"] != "not_applicable"]
    verdict = "passed" if all(g["status"] == "passed" for g in applicable) else (
        "failed" if any(g["status"] == "failed" for g in applicable) else "incomplete")
    failing = [{"gate": g["id"], **f} for g in gates for f in g.get("failures", [])]
    report = {
        "schema": SCHEMA,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "verdict": verdict,
        "exit_code": 0 if verdict == "passed" else 1,
        "compatibility_status": "practice_assumptions",
        "environment": environment(),
        "gates": gates,
        "failing_seeds": failing,
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "test-report.json").write_text(json.dumps(report, indent=1, sort_keys=False, default=str) + "\n")
    (RELEASE_DIR).mkdir(parents=True, exist_ok=True)
    (RELEASE_DIR / "failing-seeds.json").write_text(json.dumps(failing, indent=1, default=str) + "\n")
    (ARTIFACTS / "test-report.md").write_text(render_markdown(report))
    print(f"verdict: {verdict}")
    for g in gates:
        print(f"  {g['id']:22s} {g['status']}")
    print(f"wrote {(ARTIFACTS / 'test-report.json').relative_to(ROOT)} and {(ARTIFACTS / 'test-report.md').relative_to(ROOT)}")
    return report["exit_code"]


def _fmt(v: Any) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        return f"{v:,.6g}"
    if isinstance(v, dict):
        return ", ".join(f"{k}: {_fmt(x)}" for k, x in v.items())
    if isinstance(v, list):
        return "; ".join(_fmt(x) for x in v[:10])
    return str(v)


def render_markdown(report: dict[str, Any]) -> str:
    env = report["environment"]
    out = [
        "# Release verification report",
        "",
        f"Verdict: **{report['verdict'].upper()}** (exit code {report['exit_code']}). Generated "
        f"{report['generated_at']} on commit `{(env.get('git_commit') or '?')[:12]}`"
        f"{' (dirty)' if env.get('git_dirty') else ''}. Practice assumptions, not official Sequoia rules.",
        "",
        f"Host: {env.get('cpu_model')}, {env.get('cpu_count')} vCPUs, {env.get('memory')}; {env.get('platform')}; "
        f"Python {env.get('python')}, PokerKit {env.get('pokerkit')}, Node {env.get('node')}.",
        "",
        "| Gate | Section | Status | Shards | Cases | CPU time |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for g in report["gates"]:
        if g["status"] == "not_applicable":
            out.append(f"| {g['title']} | {g['section']} | not applicable | - | - | - |")
            continue
        sh = g["shards"]
        out.append(f"| {g['title']} | {g['section']} | {g['status']} | {sh.get('passed', 0)}/{sh['expected']} | "
                   f"{g['cases']:,} | {g['cpu_time_s'] / 60:,.1f} min |")
    for g in report["gates"]:
        out += ["", f"## {g['title']} ({g['id']})", "", f"Requirement: {g['requirement']}", ""]
        if g["status"] == "not_applicable":
            out.append(f"Not applicable: {g['reason']}")
            continue
        out.append(f"Status: **{g['status']}**. Shards: {_fmt(g['shards'])}. Inputs fingerprint "
                   f"`{g['inputs_fingerprint'][:16]}`.")
        if g["checks"]:
            out += ["", "| Check | Result | Detail |", "| --- | --- | --- |"]
            for c in g["checks"]:
                out.append(f"| {c['check']} | {'pass' if c['passed'] else 'FAIL'} | {_fmt(c['detail'])[:400]} |")
        if g["counts"]:
            out += ["", "Counts: " + _fmt(g["counts"])[:3000]]
        if g["exclusions"]:
            out += ["", "Exclusions (counted, never passed): " + _fmt(g["exclusions"])]
        for p in g["shard_problems"][:10]:
            out.append(f"- shard {p['shard']}: {p['state']}{' - ' + str(p['error'])[:300] if p.get('error') else ''}")
        for f in g["failures"][:10]:
            out.append(f"- failing case `{json.dumps(f['case'], sort_keys=True)}`: {f['error'][:300]}  ")
            out.append(f"  reproduce: `{f['repro']}`")
    return "\n".join(out) + "\n"
