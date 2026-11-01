"""Common-random-number evaluation of parameter vectors on full in-process tournaments.

A *task* is one table: a seed (cards, button, every seat's policy seed), a rotation
(the candidate's seat) and an opponent lineup, with optional mirror seats played by
the incumbent's parameters. Every vector in a generation plays the same task list,
so the candidates see the same cards, seats, opponents and opponent random streams;
the only difference is the candidate's own parameters (common random numbers). The
candidate also draws from the same per-seat policy stream in every arm.

The candidate runs through :class:`strategy.policy.runtime.RuntimeBot`, the deployed
path, inside the arena's in-process seat (explicit state re-decoded from canonical
JSON each decision, as in the sandbox). Results are official terminal points.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from strategy.rules.canonical import canonical_hash

MIRROR = "mirror:incumbent"


@dataclass(frozen=True)
class Task:
    seed: str
    rotation: int
    opponents: tuple[str, ...]
    """Policy specs for entrants 1..N-1 (``baseline:``, ``style:``, a bot path, or :data:`MIRROR`)."""
    families: tuple[str, ...]

    def key(self) -> list[Any]:
        return [self.seed, self.rotation, list(self.opponents)]


def seed_set_hash(tasks: Sequence[Task]) -> str:
    return canonical_hash([t.key() for t in tasks], domain="strategy-search-seed-set-v1")


def make_tasks(source: Any, partition: str, *, ruleset_id: str, players: int, count: int, namespace: str,
               mirror_rate: float = 0.0, rotations: int | None = None) -> list[Task]:
    """``count`` tables with fresh seeds and lineups from ``partition``; rotation cycles through the seats."""
    part = source.partition(partition)
    lineups = source.sample_lineups(part, ruleset_id=ruleset_id, opponents=players - 1, count=count,
                                    seed=f"search-lineups|{namespace}")
    rot = rotations or players
    out = []
    for i, lineup in enumerate(lineups):
        seed = hashlib.sha256(f"strategy-search-table-v1|{namespace}|{i}".encode()).hexdigest()[:24]
        opps, fams = [], []
        for k, o in enumerate(lineup.opponents):
            h = int.from_bytes(hashlib.sha256(f"mirror|{namespace}|{i}|{k}".encode()).digest()[:8], "big") / 2 ** 64
            if h < mirror_rate:
                opps.append(MIRROR)
                fams.append("mirror")
            else:
                opps.append(o.policy)
                fams.append(o.family)
        out.append(Task(seed, i % rot, tuple(opps), tuple(fams)))
    return out


# ------------------------------------------------------------------ worker side

_W: dict[str, Any] = {}


def _init(config_json: str) -> None:
    from arena.domain.models import TournamentConfig

    _W["config"] = TournamentConfig.model_validate_json(config_json)
    _W["modules"] = {}
    _W["paths"] = {}


def _opponent(spec: str) -> Any:
    from arena.agents.sdk import load_bot
    from strategy.eval.policies import resolve_policy

    import sys

    path = _W["paths"].get(spec)
    if path is None:
        p = resolve_policy(spec).path
        path = _W["paths"][spec] = p / "bot.py" if p.is_dir() else p
    name = "search_opp_" + hashlib.sha256(str(path).encode()).hexdigest()[:16]
    mod = sys.modules.get(name)
    if mod is None:
        load_bot(path, module_name=name)
        mod = sys.modules[name]
    return mod.make_bot()


def play(task: Task, params: Mapping[str, float], incumbent: Mapping[str, float] | None) -> dict[str, Any]:
    """One table; returns the candidate's points and execution counters."""
    from arena.agents.sdk import bot_info, decode_state
    from arena.agents.worker import InProcessSeat, play_table
    from arena.benchmark.runner import Assignment, table_info
    from arena.table.service import seeded_table
    from strategy.policy.runtime import RuntimeBot

    config = _W["config"]
    seats = config.resolved_seats
    n = len(seats)
    mapping = {seats[(i + task.rotation) % n]: i for i in range(n)}
    a = Assignment(0, 0, task.rotation, task.seed, dict(sorted(mapping.items())))
    agents = {}
    cand_seat = -1
    for seat, entrant in a.seats.items():
        if entrant == 0:
            bot = RuntimeBot(params)
            cand_seat = seat
        else:
            spec = task.opponents[entrant - 1]
            bot = RuntimeBot(incumbent, name="mirror") if spec == MIRROR else _opponent(spec)
        info = bot_info(bot)
        agent = InProcessSeat(seat, bot, info, table_info(config, a, seat, config.bot_budget_ns), "search")
        agent.start()
        agents[seat] = agent
    table = seeded_table(config, task.seed, table_id=a.table_id)
    play_table(table, agents)
    cand = agents[cand_seat]
    state = decode_state(cand.committed)
    return {"points": float(table.points()[cand_seat]), "hands": len(table.hands),
            "decisions": cand.stats.decisions, "arena_fallbacks": sum(cand.stats.fallbacks.values()),
            "internal_fallbacks": int(state.get("fb", 0)) if isinstance(state, dict) else 0}


def _job(args: tuple[int, int, Task, dict[str, float], dict[str, float] | None]) -> tuple[int, int, dict[str, Any]]:
    ci, ti, task, params, incumbent = args
    return ci, ti, play(task, params, incumbent)


class Evaluator:
    """Parallel CRN evaluation. ``evaluate(vectors, tasks)`` -> per-vector lists of per-task results."""

    def __init__(self, config: Any, jobs: int | None = None) -> None:
        self.config = config
        self.jobs = jobs or os.cpu_count() or 1
        self._pool: ProcessPoolExecutor | None = None

    def _executor(self) -> ProcessPoolExecutor:
        if self._pool is None:
            self._pool = ProcessPoolExecutor(self.jobs, mp_context=get_context("spawn"), initializer=_init,
                                             initargs=(self.config.model_dump_json(),))
        return self._pool

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(cancel_futures=True)
            self._pool = None

    def __enter__(self) -> Evaluator:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def evaluate(self, vectors: Sequence[Mapping[str, float]], tasks: Sequence[Task],
                 incumbent: Mapping[str, float] | None = None,
                 progress: Callable[[int, int], None] | None = None) -> list[list[dict[str, Any]]]:
        out: list[list[dict[str, Any] | None]] = [[None] * len(tasks) for _ in vectors]
        work = [(ci, ti, t, dict(v), dict(incumbent) if incumbent is not None else None)
                for ci, v in enumerate(vectors) for ti, t in enumerate(tasks)]
        total, done = len(work), 0
        if self.jobs <= 1:
            _init(self.config.model_dump_json())
            results = map(_job, work)
        else:
            results = self._executor().map(_job, work, chunksize=max(1, min(8, total // (self.jobs * 8) or 1)))
        for ci, ti, r in results:
            out[ci][ti] = r
            done += 1
            if progress is not None and done % 200 == 0:
                progress(done, total)
        return out  # type: ignore[return-value]


def summarize(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    pts = [r["points"] for r in results]
    n = len(pts)
    mean = sum(pts) / n if n else 0.0
    var = sum((x - mean) ** 2 for x in pts) / (n - 1) if n > 1 else 0.0
    return {"tables": n, "mean": mean, "se": (var / n) ** 0.5 if n else 0.0,
            "decisions": sum(r["decisions"] for r in results),
            "arena_fallbacks": sum(r["arena_fallbacks"] for r in results),
            "internal_fallbacks": sum(r["internal_fallbacks"] for r in results)}


def paired(a: Sequence[dict[str, Any]], b: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Mean of b - a over common tasks with its standard error and t."""
    d = [y["points"] - x["points"] for x, y in zip(a, b)]
    n = len(d)
    mean = sum(d) / n
    var = sum((x - mean) ** 2 for x in d) / (n - 1) if n > 1 else 0.0
    se = (var / n) ** 0.5
    return {"n": n, "mean": mean, "se": se, "t": mean / se if se > 0 else 0.0}


def dump_tasks(tasks: Sequence[Task], path: Path) -> None:
    path.write_text(json.dumps([{"seed": t.seed, "rotation": t.rotation, "opponents": list(t.opponents),
                                 "families": list(t.families)} for t in tasks]) + "\n", encoding="ascii")
