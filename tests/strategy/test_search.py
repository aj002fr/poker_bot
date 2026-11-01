"""Parameter search (strategy.search): optimizers, task sets, persistence, resume and validation."""

from __future__ import annotations

import json

import numpy as np
import pytest

from strategy.policy.baseline import SPACE
from strategy.policy.params import vector_hash
from strategy.search.cem import CEM, CMAES, SearchConfig, run_search
from strategy.search.evaluate import make_tasks, paired, seed_set_hash
from strategy.search.population import load_population


def _optimize(cls, gens=40):
    n = 6
    cfg = SearchConfig(run_id="toy", population=12, elite_frac=0.25, smoothing=0.7, sigma_floor=0.01)
    target = np.linspace(0.2, 0.8, n)
    opt = cls(np.full(n, 0.5), 0.2, cfg, np.ones(n))
    rng = np.random.default_rng(0)
    for _ in range(gens):
        z = opt.ask(rng)
        opt.tell(z, [-float(np.sum((row - target) ** 2)) for row in z])
    return float(np.sum((opt.mean - target) ** 2))


@pytest.mark.parametrize("cls", [CEM, CMAES])
def test_optimizers_converge_on_a_toy_objective(cls):
    assert _optimize(cls) < 1e-3


def test_frozen_coordinates_stay_put():
    cfg = SearchConfig(run_id="toy", population=8)
    free = np.array([1.0, 0.0, 1.0])
    for cls in (CEM, CMAES):
        opt = cls(np.array([0.5, 0.3, 0.5]), 0.2, cfg, free)
        z = opt.ask(np.random.default_rng(1))
        assert np.all(z[:, 1] == 0.3)
        opt.tell(z, list(range(len(z))))
        assert opt.mean[1] == pytest.approx(0.3)


def test_search_trains_only_on_train():
    for part in ("validation", "holdout"):
        with pytest.raises(ValueError, match="trains only"):
            SearchConfig(run_id="x", partition=part)
    with pytest.raises(ValueError):
        SearchConfig(run_id="x", frozen=["nope"])


def test_tasks_are_deterministic_and_use_the_partition():
    src = load_population()
    a = make_tasks(src, "train", ruleset_id="practice_nlhe", players=8, count=6, namespace="t", mirror_rate=0.3)
    b = make_tasks(src, "train", ruleset_id="practice_nlhe", players=8, count=6, namespace="t", mirror_rate=0.3)
    assert a == b and seed_set_hash(a) == seed_set_hash(b)
    assert seed_set_hash(a) != seed_set_hash(make_tasks(src, "train", ruleset_id="practice_nlhe", players=8,
                                                        count=6, namespace="u"))
    members = {m.policy for m in src.partition("train").members}
    for t in a:
        assert len(t.opponents) == 7
        assert all(o == "mirror:incumbent" or o in members for o in t.opponents)


def test_paired_statistics():
    a = [{"points": x} for x in (10.0, 20.0, 30.0)]
    b = [{"points": x} for x in (12.0, 21.0, 33.0)]
    p = paired(a, b)
    assert p["n"] == 3 and p["mean"] == pytest.approx(2.0)
    assert p["t"] > 0


def test_tiny_search_persists_everything_and_resumes(tmp_path):
    cfg = SearchConfig(run_id="tiny", generations=1, population=4, tables_per_gen=2, confirm_tables=2,
                       players=4, hand_cap=12, seed="tiny-search", mirror_rate=0.5)
    run = run_search(cfg, root=tmp_path, jobs=1, log=lambda s: None)
    gens = [json.loads(x) for x in (run / "generations.jsonl").read_text().splitlines()]
    evals = [json.loads(x) for x in (run / "evaluations.jsonl").read_text().splitlines()]
    assert [g["gen"] for g in gens] == [0]
    select = [e for e in evals if e["stage"] == "select"]
    assert len(select) == 2 + cfg.population
    for e in evals:
        assert (run / "vectors" / f"{e['vector_hash']}.json").exists()
        assert (run / "tasks" / f"{e['seed_set_hash']}.json").exists()
        assert e["population_hash"] and e["registry_hash"] == load_population().registry_hash()
        assert len(e["points"]) == (cfg.tables_per_gen if e["stage"] == "select" else cfg.confirm_tables)
    v = json.loads((run / "vectors" / f"{select[0]['vector_hash']}.json").read_text())
    assert vector_hash(v["params"]) == v["vector_hash"]
    assert select[0]["role"] == "incumbent" and v["params"] == pytest.approx(SPACE.defaults())
    assert gens[0]["fallbacks"] == 0

    cfg2 = SearchConfig(**{**cfg.__dict__, "generations": 2})
    run_search(cfg2, root=tmp_path, jobs=1, log=lambda s: None)
    gens = [json.loads(x) for x in (run / "generations.jsonl").read_text().splitlines()]
    assert [g["gen"] for g in gens] == [0, 1]
    with pytest.raises(RuntimeError, match="different search"):
        run_search(SearchConfig(**{**cfg.__dict__, "seed": "other"}), root=tmp_path, jobs=1, log=lambda s: None)


def test_phase_equity_adds_up_to_official_points(tmp_path):
    from strategy.search import validate as v

    spec = v.build_spec("baseline-validation-nlhe", overrides={
        "experiment_id": "phase-test", "table": {"players": 4, "starting_stack": 2000, "hand_cap": 30},
        "population": {"source": "strategy.population:PopulationSource", "partition": "validation",
                       "lineups_per_variant": 1},
        "blocks_per_lineup": 2})
    assert spec.stage == "confirmation" and spec.population.partition == "validation"
    import strategy.eval.paired as paired
    from strategy.experiments.runner import run_experiment
    from strategy.policy.package import build_bundle
    from unittest import mock

    spec = spec.model_copy(update={"arms": spec.arms.model_copy(update={
        "A": spec.arms.A.model_copy(update={"policy": str(tmp_path / "a")}),
        "B": spec.arms.B.model_copy(update={"policy": str(tmp_path / "b")})})})
    build_bundle(tmp_path / "a", None)
    build_bundle(tmp_path / "b", SPACE.from_unit([0.6] * len(SPACE)))
    phases = v.PhaseCollector()
    with mock.patch.object(paired, "table_record", phases.wrap(paired.table_record)):
        _, report = run_experiment(spec, root=tmp_path / "lab", jobs=1)
    prim = report["analysis"]["primary"]
    s = phases.summary()
    assert s["blocks"] == prim["blocks"] == 2
    start = 100 / 4
    for arm, key in (("A", "mean_points_A"), ("B", "mean_points_B")):
        total = start + sum(p[f"mean_equity_change_{arm}"] for p in s["by_phase"].values())
        assert total == pytest.approx(prim[key], abs=1e-6)
