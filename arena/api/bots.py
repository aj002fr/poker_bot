"""In-process baseline seats for the playable table.

These are placeholders until the agent SDK lands: each one sees only the acting
seat's :class:`~arena.domain.observations.Observation` and returns an action.
The artifact hash is the SHA-256 of the policy's source, so a table records
exactly which code sat in each seat.
"""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass

from arena.domain.models import Action
from arena.domain.observations import Observation
from arena.domain.rng import Rng
from arena.table.driver import check_or_call, check_or_fold, legal_random


@dataclass(frozen=True)
class BotPolicy:
    id: str
    label: str
    description: str
    fn: object

    @property
    def artifact_hash(self) -> str:
        return hashlib.sha256(inspect.getsource(self.fn).encode()).hexdigest()

    def decide(self, obs: Observation, rng: Rng) -> Action:
        return self.fn(obs.hero_seat, obs.legal, rng)


BOT_POLICIES: dict[str, BotPolicy] = {
    p.id: p
    for p in (
        BotPolicy("legal_random", "Legal random", "Uniform over legal action kinds and raise sizes", legal_random),
        BotPolicy("check_or_fold", "Check / fold", "Checks when free, otherwise folds", check_or_fold),
        BotPolicy("check_or_call", "Calling station", "Checks when free, otherwise calls", check_or_call),
    )
}
