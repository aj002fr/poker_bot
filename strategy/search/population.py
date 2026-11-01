"""Opponent source for the search: the frozen ``strategy.population`` registry.

Training draws lineups only from the ``train`` partition; ``validation`` is for
promotion (``strategy.search.validate``); ``holdout`` is never used here.
"""

from __future__ import annotations

import hashlib
from functools import cache
from typing import Any

DEFAULT_SOURCE = "strategy.population:PopulationSource"
SEARCH_PARTITIONS = ("train",)


@cache
def load_population(name: str = DEFAULT_SOURCE) -> Any:
    """A ``PartitionSource`` by ``module:attribute`` name (``stub`` for the harness stub)."""
    from strategy.eval.population import load_partition_source

    return load_partition_source(DEFAULT_SOURCE if name == "auto" else name)


def population_digest(source: Any, partition: str) -> dict[str, Any]:
    part = source.partition(partition)
    return {"registry_id": source.registry_id, "registry_hash": source.registry_hash(), "partition": partition,
            "partition_hash": part.partition_hash(), "members": len(part.members),
            "families": sorted({m.family for m in part.members}),
            "member_params_hash": hashlib.sha256("|".join(f"{m.member_id}:{m.params_hash}" for m in part.members)
                                                  .encode()).hexdigest()}
