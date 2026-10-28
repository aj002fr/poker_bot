"""Spec section 10 run manifest for an in-process baseline run."""

import hashlib
import json

from arena.benchmark.manifest import REPO_ROOT, load_manifest, run_baseline, seed_list_hash
from arena.domain.models import TournamentConfig
from arena.history.canonical import load_history
from arena.history.replay import replay
from arena.table.driver import POLICIES


def _run(tmp_path, name):
    config = TournamentConfig(ruleset_id="uncertainty_shortdeck", initial_player_count=4, starting_stack=300)
    policies = {s: ("legal_random", POLICIES["legal_random"]) for s in config.resolved_seats}
    return run_baseline(config=config, tables=6, base_seed="manifest", policies=policies, out_dir=tmp_path / name)


def test_manifest_contents_and_private_seeds(tmp_path):
    manifest = _run(tmp_path, "a")
    run = tmp_path / "a"
    assert load_manifest(run / "manifest.json") == manifest
    assert manifest.code.dependency_lock_sha256 == hashlib.sha256((REPO_ROOT / "uv.lock").read_bytes()).hexdigest()
    assert manifest.code.pokerkit_version == "0.7.7"
    assert manifest.config["ruleset_id"] == manifest.ruleset_id == "uncertainty_shortdeck"
    assert manifest.initial_player_count == 4 and manifest.scheduled_tables == 6
    assert [p.seat for p in manifest.participants] == [0, 1, 2, 3]
    assert all(len(p.artifact_sha256) == 64 for p in manifest.participants)
    assert manifest.seat_allocation_policy and manifest.budget_semantics and manifest.retry_policy
    assert manifest.host.os and manifest.rng_implementation == "sha256-ctr-v1"
    results = manifest.results
    assert len(results.completed_table_ids) + len(results.failed_tables) + len(results.incomplete_table_ids) == 6
    assert abs(sum(results.mean_points_by_seat.values()) - 100) < 1e-9

    seeds = json.loads((run / "private" / "seeds.json").read_text())["seeds"]
    assert seed_list_hash(seeds) == manifest.seed_list_sha256
    text = (run / "manifest.json").read_text()
    assert not any(seed in text for seed in seeds)

    for table_id in results.completed_table_ids:
        history = load_history(run / "tables" / f"{table_id}.json")
        assert history.table_started.data["rng"]["seed"] in seeds
        assert replay(history).complete


def test_rerun_reproduces_manifest_content(tmp_path):
    a, b = _run(tmp_path, "a"), _run(tmp_path, "b")
    assert a.content_hash() == b.content_hash()
    for table_id in a.results.completed_table_ids:
        assert (tmp_path / "a" / "tables" / f"{table_id}.json").read_bytes() == \
            (tmp_path / "b" / "tables" / f"{table_id}.json").read_bytes()
