"""Partitions are disjoint and frozen; the registry serves the evaluation harness."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from arena.agents.baselines import baseline_path
from arena.domain.models import RULESET_IDS
from strategy.eval.policies import resolve_policy
from strategy.eval.population import load_partition_source
from strategy.population.families import CHECKPOINT_FAMILY, FAMILIES, HOLDOUT_ONLY
from strategy.population.params import PARTITION_BINS, PARTITIONS, partition_of_bin, region_of
from strategy.population.registry import (
    COUNTS,
    LOCK_PATH,
    VALIDATION_ONLY_PAIRS,
    CheckpointLedger,
    Registry,
    get_lineup_pool,
    verify_lock,
)
from strategy.population.source import PopulationSource


def test_lock_matches(registry: Registry):
    verify_lock(registry=registry)


def test_lock_detects_changes(registry: Registry, tmp_path: Path):
    lock = json.loads(LOCK_PATH.read_text())
    some = next(iter(lock["members"]))
    lock["members"][some]["hash"] = "0" * 64
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(lock))
    with pytest.raises(AssertionError, match="members"):
        verify_lock(path, registry)


def test_counts_and_holdout_only(registry: Registry):
    assert len(HOLDOUT_ONLY) >= 2
    for fam in FAMILIES:
        got = {p: sum(1 for m in registry.partition(p) if m.family == fam) for p in PARTITIONS}
        if fam in HOLDOUT_ONLY:
            assert got["train"] == got["validation"] == 0 and got["holdout"] > 0
        else:
            assert got == dict(COUNTS)
    for p in ("train", "validation"):
        assert not set(registry.families_in(p)) & set(HOLDOUT_ONLY)


def test_parameter_regions_are_disjoint(registry: Registry):
    assert not set(PARTITION_BINS["train"]) & set(PARTITION_BINS["validation"])
    assert not set(PARTITION_BINS["train"]) & set(PARTITION_BINS["holdout"])
    assert not set(PARTITION_BINS["validation"]) & set(PARTITION_BINS["holdout"])
    for m in registry.members.values():
        assert partition_of_bin(region_of(FAMILIES[m.family].prior, m.params)) == m.partition


def test_ids_hashes_and_seeds_are_disjoint(registry: Registry):
    members = list(registry.members.values())
    assert len({m.member_id for m in members}) == len(members)
    assert len({m.hash for m in members}) == len(members)
    for m in members:
        assert m.family not in m.member_id and m.partition not in m.member_id
    seeds = {p: set(get_lineup_pool(p, v, registry=registry).table_seeds(200)) for p in PARTITIONS for v in ("nlhe",)}
    assert not seeds["train"] & seeds["validation"] and not seeds["train"] & seeds["holdout"]
    assert not seeds["validation"] & seeds["holdout"]
    variants = [set(get_lineup_pool("train", v, registry=registry).table_seeds(200)) for v in RULESET_IDS]
    assert not variants[0] & variants[1] and not variants[0] & variants[2]


def test_lineup_pools_respect_partition_rules(registry: Registry):
    for variant in RULESET_IDS:
        for p in PARTITIONS:
            pool = get_lineup_pool(p, variant, registry=registry)
            assert pool.hash == get_lineup_pool(p, variant, registry=registry).hash
            for lu in pool.lineups:
                fams = [pool.label(e) for e in lu.entries]
                assert len(fams) == 7 and all(pool.members[e].partition == p for e in lu.entries)
                pairs = [a in fams and b in fams for a, b in VALIDATION_ONLY_PAIRS]
                if p == "train":
                    assert not any(pairs) and not set(fams) & set(HOLDOUT_ONLY)
                elif p == "validation":
                    assert any(pairs)
                else:
                    assert set(fams) & set(HOLDOUT_ONLY)


def _checkpoint(tmp_path: Path, partition: str) -> tuple[CheckpointLedger, str]:
    art = tmp_path / "learner.py"
    art.write_bytes(baseline_path("calling_station").read_bytes())
    ledger = CheckpointLedger(tmp_path / "checkpoints.json")
    slot = ledger.register(art, kind="learner_checkpoint", partition=partition, rulesets=list(RULESET_IDS))
    return ledger, slot.slot_id


def test_checkpoint_slots(tmp_path: Path):
    ledger, slot_id = _checkpoint(tmp_path, "train")
    reg = Registry(ledger=ledger)
    assert not reg.lineup_pool("train", "nlhe").checkpoints
    ledger.promote(slot_id, "promotion-run-1")
    reloaded = CheckpointLedger(tmp_path / "checkpoints.json")
    assert reloaded.slots[slot_id].status == "promoted"
    pool = Registry(ledger=reloaded).lineup_pool("train", "nlhe")
    assert slot_id in pool.checkpoints and pool.label(slot_id) == "learner_checkpoint"
    assert any(slot_id in lu.entries for lu in pool.lineups)
    assert not Registry(ledger=reloaded).lineup_pool("validation", "nlhe").checkpoints
    Path(reloaded.slots[slot_id].path).write_text("# changed\n")
    with pytest.raises(ValueError, match="changed"):
        reloaded.slots[slot_id].verify()


# --------------------------------------------------------------- PartitionSource


@pytest.fixture(scope="module")
def source(registry: Registry, tmp_path_factory) -> PopulationSource:
    return PopulationSource(registry, tmp_path_factory.mktemp("pop-bots"))


def test_source_loads_by_spec_name():
    source = load_partition_source("strategy.population:PopulationSource")
    assert isinstance(source, PopulationSource)
    assert source.registry_id == "strategy.population/pop-v1"
    assert source.registry_hash() == json.loads(LOCK_PATH.read_text())["registry_hash"]


def test_source_partitions(source: PopulationSource, registry: Registry):
    parts = {p: source.partition(p) for p in PARTITIONS}
    ids = [{m.member_id for m in parts[p].members} for p in PARTITIONS]
    assert not ids[0] & ids[1] and not ids[0] & ids[2] and not ids[1] & ids[2]
    assert len({parts[p].seed_namespace for p in PARTITIONS}) == 3
    assert len({parts[p].partition_hash() for p in PARTITIONS}) == 3
    for p, part in parts.items():
        assert part.registry_hash == registry.hash and part.metadata["stub"] is False
        assert len(part.members) == len(registry.partition(p))
    ref = parts["validation"].members[0]
    member = registry.members[ref.member_id]
    assert ref.params_hash == member.hash and ref.family == member.family
    resolved = resolve_policy(ref.policy)
    assert resolved.name == ref.member_id and set(resolved.rulesets) == set(RULESET_IDS)
    assert resolved.artifact_sha256 == resolve_policy(ref.policy).artifact_sha256
    with pytest.raises(KeyError):
        source.partition("test")


def test_source_lineups(source: PopulationSource):
    for p in PARTITIONS:
        part = source.partition(p)
        for ruleset in RULESET_IDS:
            lineups = source.sample_lineups(part, ruleset_id=ruleset, opponents=5, count=12, seed="s1")
            again = source.sample_lineups(part, ruleset_id=ruleset, opponents=5, count=12, seed="s1")
            other = source.sample_lineups(part, ruleset_id=ruleset, opponents=5, count=12, seed="s2")
            assert [lu.lineup_id for lu in lineups] == [lu.lineup_id for lu in again]
            assert [lu.lineup_id for lu in lineups] != [lu.lineup_id for lu in other]
            ids = {m.member_id for m in part.members}
            for lu in lineups:
                fams = [o.family for o in lu.opponents]
                assert len(fams) == 5 == len(set(fams)) and all(o.member_id in ids for o in lu.opponents)
                if p == "train":
                    assert not any(a in fams and b in fams for a, b in VALIDATION_ONLY_PAIRS)
                elif p == "validation":
                    assert any(a in fams and b in fams for a, b in VALIDATION_ONLY_PAIRS)
                else:
                    assert set(fams) & set(HOLDOUT_ONLY)
    with pytest.raises(ValueError, match="belongs to"):
        from strategy.eval.population import StubPartitionSource

        stub = StubPartitionSource()
        source.sample_lineups(stub.partition("train"), ruleset_id="practice_nlhe", opponents=3, count=1, seed="x")


def test_harness_materializes_a_population_spec(tmp_path: Path):
    from strategy.experiments.runner import materialize
    from strategy.experiments.schema import load_spec

    spec = load_spec(Path(__file__).resolve().parents[3] / "strategy/experiments/specs/population-smoke.json")
    plan = materialize(spec, tmp_path / "plan")
    again = materialize(spec, tmp_path / "plan2")
    pop = plan.manifest.population
    assert pop.stub is False and pop.registry_id == "strategy.population/pop-v1"
    assert pop.partition.name == "validation" and len(pop.partition.member_ids) == 45
    assert plan.manifest.manifest_id == again.manifest.manifest_id
    assert len(plan.cells) == 9
    for cell in plan.cells:
        assert len(cell.lineup.opponents) == 5
        assert all(o.member_id.startswith("pm-") for o in cell.lineup.opponents)


def test_source_includes_promoted_checkpoints(tmp_path: Path):
    ledger, slot_id = _checkpoint(tmp_path, "validation")
    ledger.promote(slot_id, "promotion-run-2")
    source = PopulationSource(Registry(ledger=ledger), tmp_path / "bots")
    part = source.partition("validation")
    refs = [m for m in part.members if m.family == CHECKPOINT_FAMILY]
    assert [r.member_id for r in refs] == [slot_id] and part.metadata["checkpoints"] == [slot_id]
    lineups = source.sample_lineups(part, ruleset_id="practice_nlhe", opponents=7, count=40, seed="ck")
    assert any(o.member_id == slot_id for lu in lineups for o in lu.opponents)
    assert slot_id not in {m.member_id for m in source.partition("train").members}
