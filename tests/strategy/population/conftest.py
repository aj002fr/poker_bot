"""Shared censuses of frozen registry members.

The NLHE census seats five members of every family (three train, one validation and one
holdout member; five holdout members for holdout-only families) at official-format
tables. Censuses are seeded, so every statistic the tests read is reproducible.
"""

from __future__ import annotations

import os

import pytest

from strategy.population.families import FAMILY_LIST, HOLDOUT_ONLY
from strategy.population.measure import Tally, census
from strategy.population.registry import CheckpointLedger, Member, Registry

CENSUS_TABLES = 136
VARIANT_TABLES = 24
JOBS = max(1, min(6, (os.cpu_count() or 2) - 1))


@pytest.fixture(scope="session")
def registry() -> Registry:
    return Registry(ledger=CheckpointLedger(None))


def census_members(registry: Registry, per_family: int = 5) -> list[Member]:
    picks = {"train": 3, "validation": 1, "holdout": 1}
    out: list[Member] = []
    for fam in FAMILY_LIST:
        mine = sorted((m for m in registry.members.values() if m.family == fam.id),
                      key=lambda m: (m.partition, m.index))
        if fam.id in HOLDOUT_ONLY:
            out += mine[:per_family]
            continue
        for partition, n in picks.items():
            out += [m for m in mine if m.partition == partition][:n]
    return out


@pytest.fixture(scope="session")
def members(registry: Registry) -> dict[str, Member]:
    return {m.member_id: m for m in census_members(registry)}


def by_family(tallies: dict[str, Tally], members: dict[str, Member]) -> dict[str, list[Tally]]:
    out: dict[str, list[Tally]] = {}
    for key, tally in tallies.items():
        out.setdefault(members[key].family, []).append(tally)
    return out


@pytest.fixture(scope="session")
def nlhe_census(members: dict[str, Member]) -> dict[str, Tally]:
    entries = [(mid, m.spec) for mid, m in members.items()]
    return census(entries, tables=CENSUS_TABLES, seed="test-population", jobs=JOBS)


@pytest.fixture(scope="session", params=["uncertainty_shortdeck", "uncertainty_plo4"])
def variant_census(request, registry: Registry) -> tuple[str, dict[str, Member], dict[str, Tally]]:
    chosen = {m.member_id: m for m in census_members(registry) if m.partition in ("validation", "holdout")}
    entries = [(mid, m.spec) for mid, m in chosen.items()]
    tallies = census(entries, tables=VARIANT_TABLES, seed=f"test-{request.param}", ruleset=request.param,
                     config={"hand_cap": 100, "hands_per_level": 13}, jobs=JOBS)
    return request.param, chosen, tallies
