"""T12: the same bot artifact starting a new table gets a clean context.

Fresh process and module globals, a fresh initial explicit state, a new table ID,
and nothing in its table info or observations that identifies an opponent's
artifact or a previous table."""

from __future__ import annotations

import json

from arena.agents.sdk import TableInfo
from arena.benchmark.runner import Entrant, TableRunner, schedule
from tests.agents.conftest import bot, needs_sandbox, small_config

pytestmark = needs_sandbox


class Recorder:
    """Wraps a seat agent and keeps every observation it was sent."""

    def __init__(self, agent, sink):
        self.agent, self.sink = agent, sink

    def __getattr__(self, name):
        return getattr(self.agent, name)

    def decide(self, obs):
        self.sink.append(obs.model_dump_json())
        return self.agent.decide(obs)


def test_t12_same_artifact_new_table_is_clean():
    config = small_config(3, hand_cap=4)
    entrants = [Entrant("t12_probe", bot("t12_bot")), Entrant("calling_station", bot("calling_station"), True),
                Entrant("legal_random", bot("legal_random"), True)]
    runner = TableRunner(config, entrants)
    seen: list[str] = []
    probes = []
    original = runner._agent

    def agent(seat, entrant, info):
        a = original(seat, entrant, info)
        if entrant == 0:
            probes.append(a)
            return Recorder(a, seen)
        return a

    runner._agent = agent
    try:
        plan = schedule(config, 3, blocks=2, base_seed="t12")
        results = [runner.run(a) for a in plan]
    finally:
        runner.close()
    assert all(r.completed and r.error is None for r in results)
    starts = [json.loads(p.committed_history[0]) for p in probes]
    assert len(starts) == len(plan)
    # Fresh process every table: the module-level counter restarts at 1 and the bot
    # instance has made no decisions yet.
    assert all(s["module_contexts"] == 1 and s["instance_decisions_at_start"] == 0 for s in starts)
    assert all(s["decisions"] == 0 for s in starts)
    # Fresh table IDs, and the bot sat in every seat across a block's rotations.
    assert len({s["table_id"] for s in starts}) == len(plan)
    assert {s["seat"] for s in starts[:3]} == set(config.resolved_seats)
    assert all(s["fields"] == sorted(TableInfo.model_fields) for s in starts)
    # No opponent identity in anything the bot received.
    identities = [e.name for e in entrants[1:]] + [a.sha256 for a in runner.artifacts[1:]] + ["calling_station"]
    blob = "\n".join(seen)
    assert seen and not any(i in blob for i in identities)
    assert not any(a.root.as_posix() in blob for a in runner.artifacts)
