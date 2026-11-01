"""Complete tournament rollouts in-process, recorded as between-hand states plus terminal points.

Each rollout plays one seeded table of the arena (the authoritative engine and scorer)
with in-process bots and records, for every hand, the start-of-hand stacks, positions
and blinds, then the official terminal points of every seat. These are the training
and validation data of the continuation-value model and the state sample of the
endgame table generator.

Offline module: it drives :class:`arena.table.service.Table` directly.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from arena.agents.worker import play_table
from arena.benchmark.runner import Entrant, TableRunner, schedule, table_info
from arena.domain.models import TournamentConfig
from strategy.eval.policies import resolve_policy


@dataclass(frozen=True)
class HandRow:
    hand_number: int
    start_stacks: dict[int, int]
    button: int
    small_blind_seat: int | None
    big_blind: int
    heads_up: bool
    blinds: tuple[int, int]
    end_stacks: dict[int, int]


@dataclass(frozen=True)
class Rollout:
    ruleset_id: str
    seed: str
    rotation: int
    seat_policies: dict[int, str]
    hands: list[HandRow]
    points: dict[int, float]
    hand_cap: int
    payouts: tuple[int, ...]
    ring: tuple[int, ...]

    def as_json(self) -> dict:
        d = asdict(self)
        d["hands"] = [{**asdict(h), "start_stacks": {str(k): v for k, v in h.start_stacks.items()},
                       "end_stacks": {str(k): v for k, v in h.end_stacks.items()}} for h in self.hands]
        d["seat_policies"] = {str(k): v for k, v in self.seat_policies.items()}
        d["points"] = {str(k): v for k, v in self.points.items()}
        return d

    @classmethod
    def from_json(cls, d: dict) -> Rollout:
        hands = [HandRow(h["hand_number"], {int(k): v for k, v in h["start_stacks"].items()}, h["button"],
                         h["small_blind_seat"], h["big_blind"], h["heads_up"], tuple(h["blinds"]),
                         {int(k): v for k, v in h["end_stacks"].items()}) for h in d["hands"]]
        return cls(d["ruleset_id"], d["seed"], d["rotation"], {int(k): v for k, v in d["seat_policies"].items()},
                   hands, {int(k): float(v) for k, v in d["points"].items()}, d["hand_cap"], tuple(d["payouts"]),
                   tuple(d["ring"]))


_RUNNER: TableRunner | None = None
_POLICIES: list[str] = []


def _init(config_json: str, policies: list[str]) -> None:
    global _RUNNER, _POLICIES
    config = TournamentConfig.model_validate_json(config_json)
    entrants = []
    for i, spec in enumerate(policies):
        p = resolve_policy(spec)
        entrants.append(Entrant(f"e{i}:{p.name}", p.path / "bot.py" if p.path.is_dir() else p.path, p.builtin))
    _RUNNER = TableRunner(config, entrants, isolation="in_process")
    _POLICIES = list(policies)


def _play(a) -> Rollout:
    from arena.table.service import seeded_table

    runner = _RUNNER
    assert runner is not None
    config = runner.config
    agents = {}
    for seat, entrant in a.seats.items():
        agent = runner._agent(seat, entrant, table_info(config, a, seat, runner.budget_ns))
        agent.start()
        agents[seat] = agent
    table = seeded_table(config, a.seed, table_id=a.table_id)
    try:
        play_table(table, agents)
    finally:
        for agent in agents.values():
            agent.close()
    rows = []
    stacks_after = {}
    for rec in table.hands:
        pos = rec.positions
        live = set(rec.start_stacks)
        rows.append(HandRow(rec.hand_number, dict(rec.start_stacks), pos.button,
                            pos.small_blind if pos.small_blind in live else None, pos.big_blind, pos.heads_up,
                            tuple(rec.blinds), {}))
    for i, row in enumerate(rows):
        nxt = rows[i + 1].start_stacks if i + 1 < len(rows) else None
        if nxt is None:
            stacks_after = {s: table.stacks.get(s, 0) for s in row.start_stacks}
        else:
            stacks_after = {s: nxt.get(s, 0) for s in row.start_stacks}
        row.end_stacks.update(stacks_after)
    return Rollout(config.ruleset_id, a.seed, a.rotation, {s: _POLICIES[e] for s, e in a.seats.items()}, rows,
                   {s: float(p) for s, p in table.points().items()}, config.hand_cap,
                   tuple(config.resolved_payouts), tuple(config.resolved_seats))


def run_rollouts(config: TournamentConfig, policies: Sequence[str], *, blocks: int, base_seed: str,
                 jobs: int = 1) -> list[Rollout]:
    """``blocks`` seeded blocks, each with every seat rotation of ``policies`` (one per seat)."""
    assignments = schedule(config, len(policies), blocks, base_seed)
    if jobs > 1:
        with ProcessPoolExecutor(jobs, initializer=_init, initargs=(config.model_dump_json(), list(policies))) as ex:
            return list(ex.map(_play, assignments, chunksize=4))
    _init(config.model_dump_json(), list(policies))
    return [_play(a) for a in assignments]


def save(rollouts: Sequence[Rollout], path: Path) -> str:
    """JSON lines; returns the SHA-256 of the file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(r.as_json(), sort_keys=True, separators=(",", ":")) + "\n" for r in rollouts)
    path.write_text(body, encoding="utf-8")
    return hashlib.sha256(body.encode()).hexdigest()


def load(path: Path) -> list[Rollout]:
    return [Rollout.from_json(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line]
