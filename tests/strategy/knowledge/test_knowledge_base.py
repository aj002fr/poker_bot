import json
import shutil
from pathlib import Path

import pytest
import yaml

from strategy.knowledge import lint as lint_module
from strategy.knowledge import triage
from strategy.knowledge.lint import lint
from strategy.knowledge.registry import ROOT, UnsupportedDefaultError, load, parameter_default


def test_repository_knowledge_base_lints_clean():
    assert lint() == []


def test_lint_cli_reports_success(capsys):
    assert lint_module.main([]) == 0
    assert "knowledge lint: ok" in capsys.readouterr().out


def test_every_digest_hypothesis_is_covered_and_counts_are_stable():
    kb = load()
    index = kb.digest_index["records"]
    assert len(index) == 148
    cited = {
        key
        for doc in kb.records
        for src in doc.data["sources"]
        for key in src.get("digest_records", [])
    }
    assert cited == set(index)
    kinds = [doc.data["kind"] for doc in kb.records]
    assert kinds.count("invariant") == len(kb.test_specs) == 24
    assert len(kb.records) - kinds.count("invariant") == 72


def test_every_strategy_record_has_a_spec_and_every_spec_has_records():
    kb = load()
    specs = {d.data["id"]: d.data for d in kb.experiments + kb.test_specs}
    for doc in kb.records:
        spec = specs[doc.data["experiment"]["id"]]
        assert doc.data["id"] in spec["records"]
    assert all(spec["records"] for spec in specs.values())


def test_triage_index_is_current():
    assert triage.check(load()) == []


def test_top_fifteen_starts_with_evaluation_and_objective():
    top = triage.build(load())["top15"]
    assert len(top) == 15
    assert top[:3] == ["E010", "E001", "E003"]


def test_record_score_is_benefits_minus_penalties():
    t = dict(objective=3, reach=3, effect=3, transfer=3, readiness=3, power=3, risk=3,
             rule_uncertainty=1, leakage=2, complexity=3)
    assert triage.record_score(t) == 21 - 6


# --- parameter defaults -------------------------------------------------------------


def _record(status="proposed", artifact=None, **param):
    return {"id": "H999", "status": status, "experiment": {"result_artifact": artifact},
            "parameters": [{"name": "p", **param}]}


def test_analytic_and_neutral_defaults_are_available():
    assert parameter_default(_record(basis="analytic", default=255), "p") == 255
    assert parameter_default(_record(basis="neutral", default=0), "p") == 0


def test_sweep_only_parameters_have_no_default():
    with pytest.raises(UnsupportedDefaultError):
        parameter_default(_record(basis="sweep_only", default=None), "p")


def test_measured_defaults_need_a_supported_result():
    with pytest.raises(UnsupportedDefaultError):
        parameter_default(_record(basis="measured", default=6), "p")
    supported = _record(status="supported", artifact="results/E006.json", basis="measured", default=6)
    assert parameter_default(supported, "p") == 6


def test_unknown_parameter_raises_key_error():
    with pytest.raises(KeyError):
        parameter_default(_record(basis="neutral", default=0), "missing")


def test_repository_book_numbers_never_resolve_to_defaults():
    kb = load()
    for doc in kb.records:
        rec = doc.data
        for p in rec["parameters"]:
            for num in rec["source_numbers"]:
                if num["use"] == "analytic" or p.get("default") != num["value"]:
                    continue
                with pytest.raises(UnsupportedDefaultError):
                    parameter_default(rec, p["name"])


# --- mutation tests: the linter must catch each broken knowledge base ------------------


@pytest.fixture
def kb_dir(tmp_path: Path) -> Path:
    target = tmp_path / "knowledge"
    shutil.copytree(ROOT, target, ignore=shutil.ignore_patterns("*.py", "__pycache__"))
    return target


def _edit(path: Path, change) -> None:
    data = yaml.safe_load(path.read_text())
    change(data)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))


def _one(directory: Path, prefix: str) -> Path:
    (path,) = directory.glob(f"{prefix}-*.yaml")
    return path


def _errors(kb_dir: Path, check_index: bool = False) -> str:
    return "\n".join(lint(load(kb_dir), check_index=check_index))


def test_copy_is_clean(kb_dir):
    assert _errors(kb_dir, check_index=True) == ""


def test_source_number_labelled_analytic_cannot_become_a_default(kb_dir):
    def change(rec):
        rec["parameters"].append({"name": "jam_at", "basis": "analytic", "default": 6,
                                  "derivation": "Taken from the book"})

    _edit(_one(kb_dir / "records", "H011"), change)
    assert "unsupported source numbers cannot become defaults" in _errors(kb_dir)


def test_sweep_only_parameter_with_a_default_is_rejected(kb_dir):
    def change(rec):
        rec["parameters"][0]["default"] = 4

    _edit(_one(kb_dir / "records", "H011"), change)
    assert "sweep_only, so its default must be null" in _errors(kb_dir)


def test_measured_default_on_an_unsupported_record_is_rejected(kb_dir):
    def change(rec):
        rec["parameters"][0].update(basis="measured", default=6, evidence="a hunch")

    _edit(_one(kb_dir / "records", "H011"), change)
    assert "measured default but the record is not supported" in _errors(kb_dir)


def test_neutral_default_must_be_a_no_op_value(kb_dir):
    def change(rec):
        rec["parameters"][0]["default"] = 0.2

    _edit(_one(kb_dir / "records", "H039"), change)
    assert "neutral default must be one of" in _errors(kb_dir)


def test_concluded_status_needs_a_result_artifact(kb_dir):
    def change(rec):
        rec["status"] = "supported"
        rec["decision_log"].append({"date": "2026-11-02", "status": "supported", "by": "test",
                                    "note": "no artifact"})

    _edit(_one(kb_dir / "records", "H001"), change)
    assert "requires `experiment.result_artifact`" in _errors(kb_dir)


def test_record_must_link_to_an_existing_experiment(kb_dir):
    def change(rec):
        rec["experiment"]["id"] = "E999"

    _edit(_one(kb_dir / "records", "H001"), change)
    errors = _errors(kb_dir)
    assert "links to E999, which has no valid spec" in errors
    assert "lists H001, but H001 links to E999" in errors


def test_record_metric_must_match_its_spec(kb_dir):
    def change(rec):
        rec["experiment"]["minimum_effect"]["value"] = 0.5

    _edit(_one(kb_dir / "records", "H001"), change)
    assert "experiment.minimum_effect differs" in _errors(kb_dir)


def test_final_holdout_cannot_be_used(kb_dir):
    def change(spec):
        spec["partitions"]["final_holdout"] = "confirm"

    _edit(_one(kb_dir / "experiment_specs", "E001"), change)
    assert "final holdout is never used" in _errors(kb_dir)


def test_offline_only_features_are_not_observable_inputs(kb_dir):
    def change(rec):
        rec["observable_inputs"].append("opponent_hole_cards")

    _edit(_one(kb_dir / "records", "H034"), change)
    assert "not observable at decision time" in _errors(kb_dir)


def test_unknown_vocabulary_terms_are_rejected(kb_dir):
    def change(rec):
        rec["observable_inputs"].append("vibes")

    _edit(_one(kb_dir / "records", "H034"), change)
    assert "'vibes' is not in the vocabulary" in _errors(kb_dir)


def test_dropping_a_digest_citation_is_detected(kb_dir):
    def change(rec):
        rec["sources"] = [s for s in rec["sources"] if s["source"] != "mop"]

    _edit(_one(kb_dir / "records", "H060"), change)
    errors = _errors(kb_dir)
    assert "mop/H15 is not cited by any record" in errors


def test_citing_an_unknown_digest_record_is_detected(kb_dir):
    def change(rec):
        rec["sources"][0]["digest_records"].append("mop/H99")

    _edit(_one(kb_dir / "records", "H001"), change)
    assert "mop/H99 is not in digest_index.yaml" in _errors(kb_dir)


def test_implemented_test_spec_must_point_at_real_tests(kb_dir):
    def change(spec):
        spec["implemented_by"].append("tests/strategy/math/test_odds.py::test_does_not_exist")

    _edit(_one(kb_dir / "test_specs", "T001"), change)
    assert "test_does_not_exist is not defined" in _errors(kb_dir)


def test_factorial_experiments_need_two_factors(kb_dir):
    def change(spec):
        spec["factors"] = spec["factors"][:1]

    _edit(_one(kb_dir / "experiment_specs", "E003"), change)
    assert "factorial experiments need at least two factors" in _errors(kb_dir)


def test_arm_toggles_must_use_declared_levels(kb_dir):
    def change(spec):
        spec["arms"][1]["toggles"]["training_return"] = "chips_squared"

    _edit(_one(kb_dir / "experiment_specs", "E001"), change)
    assert "is not a declared level" in _errors(kb_dir)


def _run_spec(directory: Path, **overrides) -> Path:
    spec = {
        "experiment_id": "e001-points-vs-chips-screen",
        "title": "E001 screen",
        "hypothesis_id": "H001",
        "stage": "discovery",
        "min_effect": 0.4,
        "arms": {"A": {"policy": "style:station"}, "B": {"policy": "baseline:nlhe_heuristic"}},
        "variants": ["practice_nlhe"],
        "population": {"partition": "validation"},
        "blocks_per_lineup": 8,
        "base_seed": "knowledge/E001",
    } | overrides
    path = directory / f"{spec['experiment_id']}.json"
    path.write_text(json.dumps(spec))
    return path


def _link_run_spec(kb_dir: Path, path: Path) -> None:
    def change(spec):
        spec["run_specs"] = [str(path)]

    _edit(_one(kb_dir / "experiment_specs", "E001"), change)


def test_a_matching_runnable_spec_links_cleanly(kb_dir, tmp_path):
    _link_run_spec(kb_dir, _run_spec(tmp_path))
    assert _errors(kb_dir) == ""


def test_runnable_spec_must_name_the_hypothesis_and_meet_the_minimum_effect(kb_dir, tmp_path):
    _link_run_spec(kb_dir, _run_spec(tmp_path, hypothesis_id="H999", min_effect=0.1,
                                     variants=["practice_nlhe", "uncertainty_plo4"]))
    errors = _errors(kb_dir)
    assert "hypothesis_id 'H999', which names neither E001" in errors
    assert "below the declared minimum effect (0.004 x 100-point pool = 0.4)" in errors


def test_runnable_spec_cannot_play_undeclared_variants_or_the_holdout(kb_dir, tmp_path):
    def change(spec):
        spec["variants"]["rehearsal"] = []

    _edit(_one(kb_dir / "experiment_specs", "E001"), change)
    _link_run_spec(kb_dir, _run_spec(tmp_path, stage="final_holdout",
                                     population={"partition": "holdout"},
                                     variants=["uncertainty_shortdeck"]))
    errors = _errors(kb_dir)
    assert "plays uncertainty_shortdeck, which E001 does not declare" in errors
    assert "uses the final holdout" in errors


def test_missing_runnable_spec_is_reported(kb_dir, tmp_path):
    _link_run_spec(kb_dir, tmp_path / "nope.json")
    assert "could not be loaded" in _errors(kb_dir)


def test_ready_experiments_need_a_runnable_spec(kb_dir):
    def change(spec):
        spec["status"] = "ready"
        spec["blocked_by"] = []

    _edit(_one(kb_dir / "experiment_specs", "E001"), change)
    assert "status ready needs at least one entry in `run_specs`" in _errors(kb_dir)


def test_stale_index_is_detected(kb_dir):
    def change(rec):
        rec["triage"]["reach"] = 0

    _edit(_one(kb_dir / "records", "H056"), change)
    assert "index.md is stale" in _errors(kb_dir, check_index=True)


def test_invalid_yaml_is_reported(kb_dir):
    _one(kb_dir / "records", "H001").write_text("id: [unterminated\n")
    assert "invalid YAML" in _errors(kb_dir)


def test_lint_cli_fails_on_a_broken_base(kb_dir, capsys):
    _one(kb_dir / "records", "H001").unlink()
    assert lint_module.main([str(kb_dir)]) == 1
    assert "error(s)" in capsys.readouterr().err
