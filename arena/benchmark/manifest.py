"""Immutable run manifest (spec section 10) and the in-process baseline run.

A manifest records everything needed to reproduce or audit a batch of tables:
code commit, dependency lock hash, ruleset and every config value, participant
artifact hashes, seat count, a hash of the seed list, the seat-allocation
policy, budget semantics, host details, the scheduled table count, the retry
policy, and which tables completed, failed or did not finish.

Seeds themselves are private. They live in ``private/seeds.json`` inside the
run directory and in each table's private history, never in the manifest.
"""

from __future__ import annotations

import hashlib
import inspect
import os
import platform
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, StrictInt, StrictStr

import arena
from arena.domain.canonical import canonical_hash, canonical_json
from arena.domain.models import TournamentConfig
from arena.domain.rng import SeededRng
from arena.rules.pokerkit_adapter import POKERKIT_VERSION

MANIFEST_SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[2]


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class CodeInfo(_M):
    commit: StrictStr | None
    dirty: bool | None
    arena_version: StrictStr
    pokerkit_version: StrictStr
    python_version: StrictStr
    dependency_lock_sha256: StrictStr | None


class Participant(_M):
    seat: StrictInt
    name: StrictStr
    kind: Literal["in_process_baseline", "agent_artifact"]
    artifact_sha256: StrictStr


class HostInfo(_M):
    os: StrictStr
    machine: StrictStr
    processor: StrictStr
    cpu_count: StrictInt | None
    container: bool


class FailedTable(_M):
    table_id: StrictStr
    error: StrictStr


class RunResults(_M):
    completed_table_ids: list[StrictStr]
    failed_tables: list[FailedTable]
    incomplete_table_ids: list[StrictStr]
    mean_points_by_seat: dict[int, float]


class RunManifest(_M):
    manifest_schema_version: Literal[1] = MANIFEST_SCHEMA_VERSION
    compatibility_status: Literal["practice_assumptions"] = "practice_assumptions"
    run_id: StrictStr
    created_at: StrictStr
    code: CodeInfo
    ruleset_id: StrictStr
    config: dict[str, Any]
    config_hash: StrictStr
    initial_player_count: StrictInt
    participants: list[Participant]
    rng_implementation: StrictStr
    seed_count: StrictInt
    seed_list_sha256: StrictStr
    seat_allocation_policy: StrictStr
    budget_semantics: StrictStr
    host: HostInfo
    scheduled_tables: StrictInt
    retry_policy: StrictStr
    results: RunResults

    def content_hash(self) -> str:
        """Hash of everything except ``created_at``, for comparing reruns."""
        data = self.model_dump(mode="json")
        data.pop("created_at")
        return canonical_hash(data, domain="arena-run-manifest-v1")


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def code_info() -> CodeInfo:
    lock = REPO_ROOT / "uv.lock"
    status = _git("status", "--porcelain", "--untracked-files=no")
    return CodeInfo(
        commit=_git("rev-parse", "HEAD"),
        dirty=None if status is None else bool(status),
        arena_version=arena.__version__,
        pokerkit_version=POKERKIT_VERSION,
        python_version=platform.python_version(),
        dependency_lock_sha256=hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else None,
    )


def host_info() -> HostInfo:
    return HostInfo(
        os=platform.platform(),
        machine=platform.machine(),
        processor=platform.processor() or "unknown",
        cpu_count=os.cpu_count(),
        container=Path("/.dockerenv").exists() or Path("/run/.containerenv").exists(),
    )


def policy_artifact_hash(policy: Callable[..., Any]) -> str:
    """Hash of an in-process baseline's source (agent artifacts hash their bundle instead)."""
    return hashlib.sha256(inspect.getsource(policy).encode()).hexdigest()


def derive_seeds(base_seed: str, count: int) -> list[str]:
    return [hashlib.sha256(f"arena-run-seed-v1|{base_seed}|{k}".encode()).hexdigest()[:20] for k in range(count)]


def seed_list_hash(seeds: list[str]) -> str:
    return canonical_hash(seeds, domain="arena-seed-list-v1")


def default_run_id(config: TournamentConfig, base_seed: str, tables: int) -> str:
    return "r_" + canonical_hash([config.config_hash(), base_seed, tables], domain="arena-run-id-v1")[:12]


def run_baseline(
    *,
    config: TournamentConfig,
    tables: int,
    base_seed: str,
    policies: dict[int, tuple[str, Callable[..., Any]]],
    out_dir: Path,
    run_id: str | None = None,
) -> RunManifest:
    """Play ``tables`` seeded tables in-process and write histories plus the manifest.

    Layout: ``<out_dir>/manifest.json``, ``<out_dir>/tables/<table_id>.json`` (private
    histories), ``<out_dir>/private/seeds.json``.
    """
    from arena.history.canonical import write_history
    from arena.table.driver import play_out, policy_rngs
    from arena.table.service import seeded_table

    seeds = derive_seeds(base_seed, tables)
    run_id = run_id or default_run_id(config, base_seed, tables)
    seats = config.resolved_seats
    if sorted(policies) != seats:
        raise ValueError("assign exactly one policy per seat")
    completed, failed, incomplete = [], [], []
    point_sums = {s: 0.0 for s in seats}
    for k, seed in enumerate(seeds):
        table_id = f"{run_id}_{k:05d}"
        table = seeded_table(config, seed, table_id=table_id)
        try:
            play_out(table, {s: p for s, (_, p) in policies.items()}, policy_rngs(seed, seats))
        except Exception as exc:  # recorded, never hidden
            failed.append(FailedTable(table_id=table_id, error=f"{type(exc).__name__}: {exc}"))
            write_history(table, out_dir / "tables")
            continue
        write_history(table, out_dir / "tables")
        if table.complete:
            completed.append(table_id)
            for s, v in table.points().items():
                point_sums[s] += v
        else:
            incomplete.append(table_id)
    n_done = len(completed)
    manifest = RunManifest(
        run_id=run_id,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        code=code_info(),
        ruleset_id=config.ruleset_id,
        config=config.resolved(),
        config_hash=config.config_hash(),
        initial_player_count=config.initial_player_count,
        participants=[
            Participant(seat=s, name=name, kind="in_process_baseline", artifact_sha256=policy_artifact_hash(fn))
            for s, (name, fn) in sorted(policies.items())
        ],
        rng_implementation=SeededRng.implementation,
        seed_count=len(seeds),
        seed_list_sha256=seed_list_hash(seeds),
        seat_allocation_policy="fixed: each seat keeps its assigned policy for every table (no rotation)",
        budget_semantics=(
            "in-process baseline policies, no decision deadline enforced; "
            f"configured bot budget {config.bot_budget_ns} ns applies from milestone 4 workers"
        ),
        host=host_info(),
        scheduled_tables=tables,
        retry_policy="none: failed or incomplete tables are listed, not rerun",
        results=RunResults(
            completed_table_ids=completed,
            failed_tables=failed,
            incomplete_table_ids=incomplete,
            mean_points_by_seat={s: (v / n_done if n_done else 0.0) for s, v in point_sums.items()},
        ),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(canonical_json(manifest.model_dump(mode="json")) + "\n", encoding="ascii")
    private = out_dir / "private"
    private.mkdir(exist_ok=True)
    (private / "seeds.json").write_text(canonical_json({"run_id": run_id, "seeds": seeds}) + "\n", encoding="ascii")
    return manifest


def load_manifest(path: Path) -> RunManifest:
    return RunManifest.model_validate_json(path.read_text(encoding="ascii"))
