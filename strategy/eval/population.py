"""The partition interface evaluation needs from the opponent population.

The real registry lives in ``strategy.population`` (another workstream). Evaluation
depends only on :class:`PartitionSource`: a registry hash, named partitions with
member hashes and a seed namespace, and a deterministic lineup sampler. Any object
with these methods can be named in an experiment spec as ``"module:attribute"``
(a source, or a zero-argument factory returning one).

:class:`StubPartitionSource` stands in until then. It splits the arena's existing
baselines and six styles by family and gives each partition its own seed namespace.
It is clearly labelled ``stub`` in every manifest: it is not the plan's 70/15/15
registry and must not be used for promotion claims.
"""

from __future__ import annotations

import hashlib
import importlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from strategy.rules.canonical import canonical_hash

PARTITIONS = ("train", "validation", "holdout")


@dataclass(frozen=True)
class OpponentRef:
    member_id: str
    """Stable registry id; never shown to a learner at the table."""
    family: str
    policy: str
    """Policy spec understood by :func:`strategy.eval.policies.resolve_policy`."""
    rulesets: tuple[str, ...]
    params_hash: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"member_id": self.member_id, "family": self.family, "policy": self.policy,
                "rulesets": list(self.rulesets), "params_hash": self.params_hash}


@dataclass(frozen=True)
class Partition:
    name: str
    registry_id: str
    registry_hash: str
    seed_namespace: str
    members: tuple[OpponentRef, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def partition_hash(self) -> str:
        return canonical_hash({"name": self.name, "registry_id": self.registry_id,
                               "registry_hash": self.registry_hash, "seed_namespace": self.seed_namespace,
                               "members": [m.as_dict() for m in self.members], "metadata": self.metadata},
                              domain="strategy-partition-v1")

    def eligible(self, ruleset_id: str) -> list[OpponentRef]:
        return [m for m in self.members if ruleset_id in m.rulesets]


@dataclass(frozen=True)
class Lineup:
    lineup_id: str
    opponents: tuple[OpponentRef, ...]
    """One per non-candidate seat, in entrant order 1..N-1."""

    @property
    def families(self) -> tuple[str, ...]:
        return tuple(sorted({o.family for o in self.opponents}))


class PartitionSource(Protocol):
    registry_id: str

    def registry_hash(self) -> str: ...

    def partition(self, name: str) -> Partition: ...

    def sample_lineups(self, partition: Partition, *, ruleset_id: str, opponents: int, count: int,
                       seed: str) -> list[Lineup]: ...


def _draw(seed: str, k: int, n: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}|{k}".encode()).digest()[:8], "big") % n


def sample_lineups(partition: Partition, *, ruleset_id: str, opponents: int, count: int, seed: str) -> list[Lineup]:
    """Deterministic lineups: lineup ``j`` draws each seat uniformly, with replacement, from eligible members."""
    pool = sorted(partition.eligible(ruleset_id), key=lambda m: m.member_id)
    if not pool:
        raise ValueError(f"partition {partition.name!r} has no members that play {ruleset_id}")
    out = []
    for j in range(count):
        base = f"strategy-lineup-v1|{partition.partition_hash()}|{ruleset_id}|{seed}|{j}"
        opps = tuple(pool[_draw(base, k, len(pool))] for k in range(opponents))
        lid = "L_" + canonical_hash([o.member_id for o in opps] + [ruleset_id, j], domain="strategy-lineup-id-v1")[:12]
        out.append(Lineup(lid, opps))
    return out


ALL_RULESETS = ("practice_nlhe", "uncertainty_shortdeck", "uncertainty_plo4")
NLHE = ("practice_nlhe",)

_STUB_MEMBERS: dict[str, tuple[OpponentRef, ...]] = {
    "train": (
        OpponentRef("stub.style.rock", "rock", "style:rock", NLHE),
        OpponentRef("stub.style.aggressive", "aggressive", "style:aggressive", NLHE),
        OpponentRef("stub.style.station", "station", "style:station", NLHE),
        OpponentRef("stub.baseline.calling_station", "calling_station", "baseline:calling_station", ALL_RULESETS),
        OpponentRef("stub.baseline.legal_random", "legal_random", "baseline:legal_random", ALL_RULESETS),
    ),
    "validation": (
        OpponentRef("stub.style.weak_tight", "weak_tight", "style:weak_tight", NLHE),
        OpponentRef("stub.style.super_aggressive", "super_aggressive", "style:super_aggressive", NLHE),
        OpponentRef("stub.baseline.nlhe_heuristic", "nlhe_heuristic", "baseline:nlhe_heuristic", NLHE),
        OpponentRef("stub.baseline.calling_station.v", "calling_station", "baseline:calling_station", ALL_RULESETS),
        OpponentRef("stub.baseline.check_fold", "check_fold", "baseline:check_fold", ALL_RULESETS),
    ),
    "holdout": (
        OpponentRef("stub.style.switcher", "switcher", "style:switcher", NLHE),
        OpponentRef("stub.baseline.legal_random.h", "legal_random", "baseline:legal_random", ALL_RULESETS),
        OpponentRef("stub.baseline.check_fold.h", "check_fold", "baseline:check_fold", ALL_RULESETS),
    ),
}


class StubPartitionSource:
    """Existing baselines and styles split by family. Labelled ``stub``; not for promotion claims."""

    registry_id = "stub-arena-baselines-v1"

    def registry_hash(self) -> str:
        return canonical_hash({p: [m.as_dict() for m in ms] for p, ms in _STUB_MEMBERS.items()},
                              domain="strategy-stub-registry-v1")

    def partition(self, name: str) -> Partition:
        if name not in _STUB_MEMBERS:
            raise KeyError(f"unknown partition {name!r}; choose from {PARTITIONS}")
        return Partition(name, self.registry_id, self.registry_hash(), f"stub-{name}", _STUB_MEMBERS[name],
                         {"stub": True})

    def sample_lineups(self, partition: Partition, *, ruleset_id: str, opponents: int, count: int,
                       seed: str) -> list[Lineup]:
        return sample_lineups(partition, ruleset_id=ruleset_id, opponents=opponents, count=count, seed=seed)


def load_partition_source(name: str) -> PartitionSource:
    """``"stub"`` or ``"package.module:attribute"`` naming a source or a factory for one."""
    if name == "stub":
        return StubPartitionSource()
    module, _, attr = name.partition(":")
    if not attr:
        raise ValueError(f"population source {name!r} must be 'stub' or 'module:attribute'")
    obj = getattr(importlib.import_module(module), attr)
    source = obj() if isinstance(obj, type) or (callable(obj) and not hasattr(obj, "partition")) else obj
    for method in ("registry_hash", "partition", "sample_lineups"):
        if not callable(getattr(source, method, None)):
            raise TypeError(f"{name} does not implement PartitionSource.{method}")
    return source


def explicit_lineup(policies: Sequence[str], ruleset_id: str) -> Lineup:
    """A fixed lineup from policy specs, for hand-picked comparisons outside any partition."""
    from strategy.eval.policies import resolve_policy

    opps = []
    for spec in policies:
        p = resolve_policy(spec)
        opps.append(OpponentRef(f"explicit.{p.name}", p.family, spec, p.rulesets, p.artifact_sha256))
    lid = "L_" + canonical_hash([*policies, ruleset_id], domain="strategy-lineup-id-v1")[:12]
    return Lineup(lid, tuple(opps))
