"""Release gate registry, fingerprints, checkpointed shards and the scheduler.

Every gate is a list of shards. A shard runs in its own subprocess and writes
``artifacts/release/shards/<gate>/<shard>.json`` atomically when it finishes.
A shard counts only if its file says ``passed`` *and* its fingerprint (the
hash of the gate's code inputs plus the shard parameters) equals the current
one. Anything else is missing, stale, failed or errored, and the report then
exits nonzero. Rerunning ``run`` resumes: shards that are already up to date
are not repeated, everything else is.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
RELEASE_DIR = ARTIFACTS / "release"
SHARD_DIR = RELEASE_DIR / "shards"
LOG_DIR = RELEASE_DIR / "logs"

ENGINE_INPUTS = ("arena/**/*.py", "arena/**/*.json", "arena/**/*.sql", "pyproject.toml", "uv.lock")
ORACLE_INPUTS = ("tests/oracles/*.py", "tests/support/__init__.py", "tests/support/oracle_bridge.py",
                 "tests/__init__.py", "tests/property/__init__.py", "tests/property/generated.py",
                 "tests/release/workloads/__init__.py", "tests/release/workloads/common.py")
SUITE_INPUTS = ENGINE_INPUTS + ("tests/**/*.py", "tests/**/*.json", "examples/*.py", "scripts/**/*",
                                "docker-compose.yml")
WEB_INPUTS = ("web/src/**/*", "web/e2e/**/*", "web/package-lock.json", "web/playwright.config.ts",
              "web/vite.config.ts", "web/index.html", "web/tsconfig*.json")
# Orchestration and reporting never change what a shard measured.
NOT_INPUTS = ("tests/release/harness.py", "tests/release/report.py", "tests/release/__main__.py",
              "tests/release/repro.py")


@dataclass(frozen=True)
class Gate:
    id: str
    title: str
    section: str
    requirement: str
    module: str
    inputs: tuple[str, ...]
    exclusive: bool = False
    """Run alone on an otherwise idle machine (timing-sensitive or real-time suites)."""
    applicable: bool = True
    not_applicable_reason: str = ""
    cost: str = "players"

    def workload(self):
        return importlib.import_module(self.module)

    def shards(self) -> list[dict[str, Any]]:
        if not self.applicable:
            return []
        mod = self.workload()
        try:
            return mod.shards(self.id)
        except TypeError:
            return mod.shards()


def _bulk(gate_id: str, module: str) -> tuple[str, ...]:
    return ENGINE_INPUTS + ORACLE_INPUTS + (module.replace(".", "/") + ".py",)


EVAL = "tests.release.workloads.evaluation"
GATES: list[Gate] = [
    Gate("fast", "Fast suite", "13 (fast, each change)",
         "Schema checks, golden fixtures, 100 generated hands per N per scenario, ICM fixtures, key races; "
         "zero failures, deterministic repeat with fixed seeds", "tests.release.workloads.suites",
         SUITE_INPUTS, exclusive=True),
    Gate("nlhe_eval", "NLHE evaluation", "13",
         "All 2,598,960 five-card combinations and >= 100,000 seven-card samples against the independent "
         "21-combination oracle; exact rank and category agreement including ties", EVAL, _bulk("nlhe_eval", EVAL)),
    Gate("shortdeck_eval", "Short-deck evaluation", "13",
         "All 376,992 five-card combinations and >= 100,000 seven-card samples against the independent short-deck "
         "oracle; exact agreement under the declared ranking", EVAL, _bulk("shortdeck_eval", EVAL)),
    Gate("plo_eval", "PLO evaluation", "13",
         ">= 100,000 distinct hole/board samples against the independent 60-combination oracle, including targeted "
         "false-flush and board-only cases; exact agreement", EVAL, _bulk("plo_eval", EVAL)),
    Gate("state_machine", "State machine", "13",
         ">= 100,000 complete NLHE hands for each initial N=2..8 and >= 25,000 per N for each other scenario "
         "(>= 1,050,000 total); zero invariant, legality, settlement or replay mismatches",
         "tests.release.workloads.state_machine", _bulk("state_machine", "tests.release.workloads.state_machine")),
    Gate("tournaments", "Tournaments", "13",
         "NLHE 10,000 eight-seat plus 2,000 each at N=2..7; each other scenario 1,000 per N=2..8 (>= 36,000); "
         "caps, levels, placements and ICM correct; no unexplained invalid tables",
         "tests.release.workloads.tournaments", _bulk("tournaments", "tests.release.workloads.tournaments")),
    Gate("icm", "ICM", "13",
         "Permutation-oracle comparison on >= 1,000 randomized cases across N=2..8 including sparse/flat prizes and "
         "unequal stacks; <= 1e-9 absolute error; conservation and symmetry",
         "tests.release.workloads.icm", _bulk("icm", "tests.release.workloads.icm")),
    Gate("fault", "Fault injection", "13 / 12.3",
         "Every 12.3 injection plus 100 repeated action/timeout races under each supported writer configuration "
         "(memory and PostgreSQL stores); exactly-once mutation and settlement; no privacy violations",
         "tests.release.workloads.suites", SUITE_INPUTS, exclusive=True),
    Gate("browser", "Browser", "13",
         "Human/bot mixtures at N=2,3,4,5,6,8; eight-to-seven elimination; all scenarios; 360/768/1440 px; reconnect "
         "and keyboard flows; no clipped essential information or hidden-card leakage",
         "tests.release.workloads.suites", SUITE_INPUTS + WEB_INPUTS, exclusive=True),
    Gate("isolation", "Isolation", "13",
         "Negative sandbox tests and observer/seat authorization tests; no unauthorized data, network or "
         "filesystem access", "tests.release.workloads.suites", SUITE_INPUTS, exclusive=True),
    Gate("isolated_tournaments", "Isolated-worker tournaments", "build plan wave 6",
         "About 500 eight-bot tournaments through real sandboxed workers under the 1 ms deadline; every table "
         "valid, replayed, hash-verified and re-settled by the oracles",
         "tests.release.workloads.isolated", _bulk("isolated", "tests.release.workloads.isolated") + (
             "tests/release/workloads/tournaments.py",), exclusive=True),
    Gate("performance", "Performance", "13",
         ">= 100,000 baseline decisions in sandboxed workers; p50/p95/p99/max, 1 ms fallback rate, worker/IPC split, "
         "throughput, peak memory; no silent deadline violations; declared baseline suitability",
         "tests.release.workloads.performance", _bulk("performance", "tests.release.workloads.performance") + (
             "tests/release/workloads/isolated.py", "tests/release/workloads/tournaments.py"), exclusive=True),
    Gate("metamorphic", "Metamorphic relations", "12.2",
         "Suit renaming, seat/button rotation, chip scaling without rounding effects, anonymous label relabeling, "
         "serial versus parallel scheduling of one manifest",
         "tests.release.workloads.metamorphic",
         _bulk("metamorphic", "tests.release.workloads.metamorphic") + ("tests/metamorphic/*.py",)),
    Gate("mutation", "Seeded-defect mutation checks", "12.4",
         "Each seeded defect in a test-only copy of the code is caught by its named test, and the same tests pass on "
         "the unmodified code", "tests.release.mutation", SUITE_INPUTS),
    Gate("determinism", "Deterministic repeat", "13 / 10",
         "Bulk-gate shards rerun in fresh processes with different hash seeds give identical result hashes",
         "tests.release.workloads.determinism", ENGINE_INPUTS + ORACLE_INPUTS + (
             "tests/release/workloads/*.py",)),
    Gate("official_compat", "Official compatibility", "13", "After the official kit arrives: curated traces, "
         ">= 10,000 matched hands and >= 1,000 full tables against its simulator", "", (), applicable=False,
         not_applicable_reason="The official Sequoia kit has not been released; arena/adapters/official.py is "
                               "disabled and every table stays labelled practice assumptions (milestone 8)."),
]
GATE_BY_ID = {g.id: g for g in GATES}


# ------------------------------------------------------------ fingerprints


def _files(patterns: tuple[str, ...]) -> list[Path]:
    out: set[Path] = set()
    for pattern in patterns:
        for p in ROOT.glob(pattern):
            rel = p.relative_to(ROOT).as_posix()
            if p.is_file() and "__pycache__" not in rel and "node_modules" not in rel and rel not in NOT_INPUTS:
                out.add(p)
    return sorted(out)


_FP_CACHE: dict[tuple[str, ...], str] = {}


def inputs_fingerprint(patterns: tuple[str, ...]) -> str:
    if patterns not in _FP_CACHE:
        h = hashlib.sha256(b"arena-release-inputs-v1")
        for p in _files(patterns):
            h.update(p.relative_to(ROOT).as_posix().encode() + b"\0")
            h.update(hashlib.sha256(p.read_bytes()).digest())
        _FP_CACHE[patterns] = h.hexdigest()
    return _FP_CACHE[patterns]


def shard_fingerprint(gate: Gate, params: dict[str, Any]) -> str:
    text = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{inputs_fingerprint(gate.inputs)}|{text}".encode()).hexdigest()


# ----------------------------------------------------------------- shards


@dataclass
class ShardRef:
    gate: Gate
    index: int
    params: dict[str, Any]

    @property
    def id(self) -> str:
        return f"{self.gate.id}-{self.index:03d}"

    @property
    def path(self) -> Path:
        return SHARD_DIR / self.gate.id / f"{self.id}.json"

    @property
    def log(self) -> Path:
        return LOG_DIR / self.gate.id / f"{self.id}.log"

    def load(self) -> dict[str, Any] | None:
        try:
            return json.loads(self.path.read_text())
        except (OSError, ValueError):
            return None

    def state(self) -> str:
        rec = self.load()
        if rec is None:
            return "missing"
        if rec.get("fingerprint") != shard_fingerprint(self.gate, self.params) or rec.get("params") != self.params:
            return "stale"
        return rec.get("status", "error")


def shard_refs(gate: Gate) -> list[ShardRef]:
    return [ShardRef(gate, k, p) for k, p in enumerate(gate.shards())]


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


def execute_shard(gate_id: str, index: int) -> int:
    """Body of ``python -m tests.release shard GATE INDEX``: run one shard and checkpoint it."""
    gate = GATE_BY_ID[gate_id]
    ref = shard_refs(gate)[index]
    started = datetime.now(UTC)
    t0 = time.perf_counter()
    record: dict[str, Any] = {
        "gate": gate.id, "shard": ref.id, "params": ref.params,
        "fingerprint": shard_fingerprint(gate, ref.params),
        "inputs_fingerprint": inputs_fingerprint(gate.inputs),
        "started_at": started.isoformat(timespec="seconds"), "pid": os.getpid(),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
    }
    try:
        result = gate.workload().run_shard(ref.params)
        status = "passed" if result.get("failed_cases", 0) == 0 and not result.get("failures") else "failed"
        if result.get("incomplete"):
            status = "failed"
        record.update(result=result, status=status)
    except Exception as exc:
        import traceback

        record.update(status="error", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()[-6000:])
    record["finished_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    record["runtime_s"] = round(time.perf_counter() - t0, 3)
    _write_json(ref.path, record)
    print(f"{ref.id}: {record['status']} in {record['runtime_s']:.1f}s", flush=True)
    return 0 if record["status"] == "passed" else 1


def _cost(ref: ShardRef) -> float:
    p = ref.params
    weight = {"uncertainty_plo4": 2.5, "uncertainty_shortdeck": 1.0, "practice_nlhe": 1.0}.get(p.get("ruleset"), 1.0)
    return weight * p.get("players", 4) * p.get("count", 1)


def _launch(ref: ShardRef, env_extra: dict[str, str] | None = None) -> subprocess.Popen:
    ref.log.parent.mkdir(parents=True, exist_ok=True)
    ref.path.unlink(missing_ok=True)
    env = {**os.environ, **(env_extra or {})}
    log = open(ref.log, "w")
    return subprocess.Popen([sys.executable, "-m", "tests.release", "shard", ref.gate.id, str(ref.index)],
                            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)


def _record_crash(ref: ShardRef, code: int) -> None:
    rec = ref.load()
    if rec is not None and rec.get("fingerprint") == shard_fingerprint(ref.gate, ref.params) \
            and rec.get("finished_at") and code in (0, 1):
        return
    _write_json(ref.path, {
        "gate": ref.gate.id, "shard": ref.id, "params": ref.params, "status": "error",
        "fingerprint": shard_fingerprint(ref.gate, ref.params),
        "error": f"shard process exited with code {code} without a result; see {ref.log.relative_to(ROOT)}",
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
    })


@dataclass
class RunPlan:
    exclusive: list[ShardRef] = field(default_factory=list)
    parallel: list[ShardRef] = field(default_factory=list)
    up_to_date: int = 0


def plan(gate_ids: list[str] | None, *, fresh: bool = False) -> RunPlan:
    out = RunPlan()
    for gate in GATES:
        if gate_ids and gate.id not in gate_ids:
            continue
        for ref in shard_refs(gate):
            if not fresh and ref.state() == "passed":
                out.up_to_date += 1
                continue
            (out.exclusive if gate.exclusive else out.parallel).append(ref)
    out.parallel.sort(key=_cost, reverse=True)
    return out


def run(gate_ids: list[str] | None, *, jobs: int, fresh: bool = False, exclusive_first: bool = True) -> None:
    p = plan(gate_ids, fresh=fresh)
    print(f"{p.up_to_date} shards up to date; {len(p.exclusive)} exclusive and {len(p.parallel)} parallel to run "
          f"with {jobs} jobs", flush=True)

    def run_exclusive() -> None:
        for ref in p.exclusive:
            proc = _launch(ref)
            code = proc.wait()
            _record_crash(ref, code)
            print(f"  {ref.id}: {ref.state()}", flush=True)

    if exclusive_first:
        run_exclusive()
    pending = list(p.parallel)
    running: dict[subprocess.Popen, ShardRef] = {}
    done = 0
    t0 = time.time()
    while pending or running:
        while pending and len(running) < jobs:
            ref = pending.pop(0)
            running[_launch(ref)] = ref
        time.sleep(0.5)
        for proc in [pr for pr in running if pr.poll() is not None]:
            ref = running.pop(proc)
            _record_crash(ref, proc.returncode)
            done += 1
            print(f"  [{done}/{len(p.parallel)} {time.time() - t0:.0f}s] {ref.id}: {ref.state()}", flush=True)
    if not exclusive_first:
        run_exclusive()
