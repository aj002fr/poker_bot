"""Operator benchmark runs: the milestone 4 runner, launched in the background.

A run is one or more cells (scenario = ruleset, and seat count). Each cell is
one :func:`~arena.benchmark.runner.run_benchmark` call with the chosen bots
cycled to fill the seats, so results stay separated by scenario and seat count.
Runs execute one at a time on a single background thread; cells run in order.
Manifests and private seed lists are written under ``artifacts/runs/<run_id>/``.
"""

from __future__ import annotations

import copy
import queue
import secrets
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, ValidationError

from arena.api.seats import BUILTIN_SOURCES, BotRegistry, BotSpec
from arena.benchmark.runner import Entrant, run_benchmark
from arena.domain.models import RulesetId, TournamentConfig

RunStatus = Literal["queued", "running", "complete", "failed"]
RESERVED_CONFIG_KEYS = ("ruleset_id", "initial_player_count", "seats")
MANIFEST_FIELDS = (
    "run_id", "config", "config_hash", "competitive", "isolation", "budget_ns", "seed_blocks",
    "rotations_per_block", "scheduled_tables", "entrants", "summary", "report_note", "code",
    "seed_list_sha256", "budget_semantics", "process_layout",
)


class RunCell(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ruleset_id: RulesetId
    players: int = Field(ge=2, le=8)


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cells: list[RunCell] = Field(min_length=1, max_length=12)
    bots: list[StrictStr] = Field(min_length=1, max_length=8)
    blocks: int = Field(2, ge=1, le=500)
    isolation: Literal["sandbox", "in_process"] = "sandbox"
    jobs: int = Field(1, ge=1, le=8)
    seed: StrictStr | None = Field(None, min_length=1, max_length=128)
    config: dict[str, Any] = Field(default_factory=dict)
    """Overrides applied to every cell, e.g. ``starting_stack`` or ``hand_cap``."""


class RunRejected(Exception):
    def __init__(self, code: str, message: str, details: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


@dataclass
class _Cell:
    cell_id: str
    ruleset_id: str
    players: int
    config: TournamentConfig
    entrants: list[Entrant]
    status: RunStatus = "queued"
    done: int = 0
    total: int = 0
    fallbacks_so_far: Counter = field(default_factory=Counter)
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    manifest: dict[str, Any] | None = None

    def public(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "ruleset_id": self.ruleset_id,
            "players": self.players,
            "config_hash": self.config.config_hash(),
            "entrants": [{"name": e.name, "builtin": e.builtin} for e in self.entrants],
            "status": self.status,
            "done": self.done,
            "total": self.total,
            "wall_s": (self.finished_at or time.monotonic()) - self.started_at if self.started_at else None,
            "error": self.error,
            "manifest": self.manifest,
        }


@dataclass
class _Run:
    run_id: str
    request: dict[str, Any]
    seed: str
    cells: list[_Cell]
    bots: list[dict[str, Any]]
    created_at: str
    status: RunStatus = "queued"
    error: str | None = None

    def public(self) -> dict[str, Any]:
        cells = [c.public() for c in self.cells]
        return {
            "run_id": self.run_id,
            "status": self.status,
            "created_at": self.created_at,
            "request": {k: v for k, v in self.request.items() if k != "seed"},
            "bots": self.bots,
            "progress": {"done": sum(c["done"] for c in cells), "total": sum(c["total"] for c in cells)},
            "cells": cells,
            "error": self.error,
            "compatibility_status": "practice_assumptions",
        }

    def brief(self) -> dict[str, Any]:
        full = self.public()
        return {k: full[k] for k in ("run_id", "status", "created_at", "progress", "error")} | {
            "scenarios": [f"{c.ruleset_id} · {c.players} seats" for c in self.cells],
            "bots": [b["label"] for b in self.bots],
        }


def _entrants(specs: list[BotSpec], players: int) -> list[Entrant]:
    picks = [specs[k % len(specs)] for k in range(players)]
    totals = Counter(s.id for s in picks)
    seen: Counter = Counter()
    out = []
    for s in picks:
        seen[s.id] += 1
        name = s.id if totals[s.id] == 1 else f"{s.id}#{seen[s.id]}"
        assert s.run_path is not None
        out.append(Entrant(name=name, path=s.run_path, builtin=s.source in BUILTIN_SOURCES))
    return out


class RunManager:
    def __init__(self, bots: BotRegistry, out_root: Path = Path("artifacts/runs")) -> None:
        self.bots = bots
        self.out_root = out_root
        self.runs: dict[str, _Run] = {}
        self._lock = threading.Lock()
        self._queue: queue.Queue[str] = queue.Queue()
        self._thread = threading.Thread(target=self._work, name="arena-runs", daemon=True)
        self._thread.start()

    # -- submission
    def submit(self, body: RunRequest) -> dict[str, Any]:
        bad = [k for k in RESERVED_CONFIG_KEYS if k in body.config]
        if bad:
            raise RunRejected("invalid_config", f"set {', '.join(bad)} per cell, not in config overrides")
        specs = []
        for i, bot_id in enumerate(body.bots):
            spec = self.bots.resolve(bot_id)
            if spec is None:
                raise RunRejected("unknown_bot", f"bot {i}: no baseline or registered bot with that id")
            specs.append(spec)
        if body.isolation == "in_process" and any(s.source not in BUILTIN_SOURCES for s in specs):
            raise RunRejected("isolation_required", "registered bot files only run sandboxed")
        cells = []
        for c in body.cells:
            try:
                config = TournamentConfig.model_validate(
                    {**body.config, "ruleset_id": c.ruleset_id, "initial_player_count": c.players})
            except ValidationError as exc:
                details = [{"loc": [str(p) for p in e["loc"]], "msg": e["msg"]} for e in exc.errors()]
                raise RunRejected("invalid_config", f"{c.ruleset_id} with {c.players} seats rejected", details) from None
            unsupported = [s.label for s in specs if c.ruleset_id not in s.rulesets]
            if unsupported:
                raise RunRejected("unsupported_ruleset", f"{', '.join(unsupported)} cannot play {c.ruleset_id}",
                                  {"ruleset_id": c.ruleset_id, "bots": unsupported})
            cell_id = f"{c.ruleset_id}-{c.players}p"
            if any(x.cell_id == cell_id for x in cells):
                raise RunRejected("duplicate_cell", f"{cell_id} listed twice")
            cells.append(_Cell(cell_id, c.ruleset_id, c.players, config, _entrants(specs, c.players),
                               total=body.blocks * c.players))
        run = _Run(
            run_id="r_" + secrets.token_hex(6),
            request=body.model_dump(mode="json"),
            seed=body.seed or secrets.token_hex(16),
            cells=cells,
            bots=[s.public() for s in specs],
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        with self._lock:
            self.runs[run.run_id] = run
        self._queue.put(run.run_id)
        return self.get(run.run_id)

    # -- reads
    def get(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            run = self.runs.get(run_id)
            return copy.deepcopy(run.public()) if run is not None else None

    def summaries(self) -> list[dict[str, Any]]:
        with self._lock:
            return [r.brief() for r in sorted(self.runs.values(), key=lambda r: r.created_at, reverse=True)]

    def wait(self, run_id: str, timeout_s: float = 60.0) -> dict[str, Any]:
        """Block until the run finishes (tests and scripts)."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            run = self.get(run_id)
            if run is None or run["status"] in ("complete", "failed"):
                return run or {}
            time.sleep(0.02)
        raise TimeoutError(run_id)

    # -- execution
    def _work(self) -> None:
        while True:
            run_id = self._queue.get()
            with self._lock:
                run = self.runs[run_id]
                run.status = "running"
            failed = False
            for index, cell in enumerate(run.cells):
                failed |= not self._run_cell(run, index, cell)
            with self._lock:
                run.status = "failed" if failed else "complete"
                if failed:
                    run.error = "one or more scenarios failed; see each scenario's error"

    def _run_cell(self, run: _Run, index: int, cell: _Cell) -> bool:
        with self._lock:
            cell.status = "running"
            cell.started_at = time.monotonic()

        def progress(done: int, total: int, result: Any) -> None:
            with self._lock:
                cell.done, cell.total = done, total
                for counts in result.fallbacks.values():
                    cell.fallbacks_so_far.update(counts)

        try:
            out = run_benchmark(
                config=cell.config, entrants=cell.entrants, blocks=int(run.request["blocks"]),
                base_seed=f"{run.seed}|{cell.cell_id}", isolation=run.request["isolation"],
                jobs=int(run.request["jobs"]), out_dir=self.out_root / run.run_id / cell.cell_id,
                run_id=f"{run.run_id}-{index}", progress=progress,
            )
        except Exception as exc:  # recorded and shown, never hidden
            with self._lock:
                cell.status, cell.error, cell.finished_at = "failed", f"{type(exc).__name__}: {exc}"[:500], time.monotonic()
            return False
        data = out.manifest.model_dump(mode="json")
        manifest = {k: data[k] for k in MANIFEST_FIELDS if k in data}
        manifest["content_hash"] = out.manifest.content_hash()
        manifest["mean_points_by_seat"] = data["results"]["mean_points_by_seat"]
        with self._lock:
            cell.manifest = manifest
            cell.total = cell.total or out.manifest.scheduled_tables
            cell.done = len(out.results)
            cell.status = "complete"
            cell.finished_at = time.monotonic()
        return True
