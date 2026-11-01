"""Controlled vocabularies and structural validation for the knowledge base.

Three document types live under ``strategy/knowledge``:

* hypothesis records (``records/*.yaml``): one paraphrased, testable claim each;
* experiment specs (``experiment_specs/E###.yaml``): the research design for testing
  strategy and method records (arms, factors, lineups, slices, guardrails, gates).
  Each design is executed by one or more runnable ``strategy.experiments`` specs
  (one A/B pair each), which it lists under ``run_specs``;
* invariant test specs (``test_specs/T###.yaml``): mandatory correctness tests that
  bypass strategy ranking.

Each ``validate_*`` function checks one document in isolation and returns error
strings. Cross-document rules (links, coverage, vocabulary, defaults) live in
``strategy.knowledge.lint``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from numbers import Real
from typing import Any

STATUSES = ("proposed", "screening", "supported", "rejected", "inconclusive", "superseded")
CONCLUDED_STATUSES = ("supported", "rejected", "inconclusive")
RECORD_KINDS = ("strategy", "method", "invariant")
TRANSFER_SCOPES = ("generic", "family", "variant_specific")
VARIANTS = ("all", "nlhe", "short_deck", "plo", "limit_holdem", "stud", "draw")
SOURCE_NUMBER_USES = (
    "analytic",  # follows from rules or an exact derivation; may be a default
    "sweep_bound",  # may bound a sweep range, never a default
    "initialization_candidate",  # may seed a search as one candidate, never a default
    "illustration",  # worked example only
    "rejected",  # known not to transfer
)
PARAMETER_BASES = ("neutral", "analytic", "measured", "sweep_only")
NEUTRAL_DEFAULTS = (0, 1, False, "off", "baseline")
EXPERIMENT_KINDS = (
    "ab_test",
    "factorial",
    "sweep",
    "ablation",
    "oracle_comparison",
    "null_simulation",
    "rehearsal",
)
EXPERIMENT_MODES = ("discovery", "confirmation")
EXPERIMENT_STATUSES = ("specified", "ready", "running", "complete", "blocked")
RUNNABLE_STATUSES = ("ready", "running", "complete")
VARIANT_RULESETS = {
    "nlhe": "practice_nlhe",
    "short_deck": "uncertainty_shortdeck",
    "plo": "uncertainty_plo4",
}
"""Knowledge variant names and the arena ruleset ids runnable specs use for them."""
TEST_SPEC_STATUSES = ("specified", "implemented")
MINIMUM_EFFECT_BASES = ("provisional", "plan_gate", "analytic")
ARM_ROLES = ("control", "treatment")
WORKSTREAMS = (
    "M1-A", "M1-B", "M1-C",
    "M2-A", "M2-B", "M2-C",
    "M3-A", "M3-B", "M3-C", "M3-D",
    "M4", "M5", "M6", "M7", "M8", "M9",
)  # fmt: skip

EXPERIMENT_PARTITIONS = {
    "fit": "train",
    "screen": "validation/screen",
    "confirm": "validation/confirm",
    "final_holdout": "excluded",
}
TEST_PARTITIONS = {"fixtures": "deterministic", "final_holdout": "excluded"}

TRIAGE_BENEFITS = ("objective", "reach", "effect", "transfer", "readiness", "power", "risk")
TRIAGE_PENALTIES = ("rule_uncertainty", "leakage", "complexity")

RECORD_ID = re.compile(r"^[HI]\d{3}$")
EXPERIMENT_ID = re.compile(r"^E\d{3}$")
TEST_ID = re.compile(r"^T\d{3}$")
DIGEST_KEY = re.compile(r"^[a-z0-9-]+/H\d{1,2}$")
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SPEC_SCHEMA = {"experiment": "knowledge.experiment/v1", "test": "knowledge.invariant-test/v1"}

RECORD_FIELDS = (
    "id", "title", "kind", "status", "cluster", "sources", "situation",
    "eligible_opportunity", "observable_inputs", "training_signals", "intervention",
    "baseline", "mechanism", "exceptions", "variant_scope", "source_numbers",
    "parameters", "experiment", "triage", "owner", "implementation", "decision_log",
    "related",
)  # fmt: skip
OPTIONAL_RECORD_FIELDS = ("superseded_by", "notes")
EXPERIMENT_FIELDS = (
    "schema", "id", "title", "kind", "mode", "records", "question", "arms", "factors",
    "toggles", "lineups", "variants", "seeds", "design", "primary_metric",
    "minimum_effect", "metrics", "slices", "guardrails", "partitions", "budget", "gates",
    "run_specs", "blocked_by", "status", "result_artifact",
)  # fmt: skip
TEST_SPEC_FIELDS = (
    "schema", "id", "title", "kind", "records", "owner_workstream", "target",
    "test_location", "status", "implemented_by", "primary_metric", "minimum_effect",
    "partitions", "cases", "properties", "mutations",
)  # fmt: skip


class _Checker:
    def __init__(self, doc: Any, label: str):
        self.doc = doc if isinstance(doc, Mapping) else {}
        self.label = label
        self.errors: list[str] = []
        if not isinstance(doc, Mapping):
            self.fail("document must be a mapping")

    def fail(self, message: str) -> None:
        self.errors.append(f"{self.label}: {message}")

    def fields(self, required: tuple[str, ...], optional: tuple[str, ...] = ()) -> None:
        for name in required:
            if name not in self.doc:
                self.fail(f"missing field `{name}`")
        for name in self.doc:
            if name not in required and name not in optional:
                self.fail(f"unknown field `{name}`")

    def text(self, value: Any, where: str, *, nullable: bool = False) -> bool:
        if value is None and nullable:
            return True
        if not isinstance(value, str) or not value.strip():
            self.fail(f"`{where}` must be non-empty text")
            return False
        return True

    def choice(self, value: Any, allowed: tuple[Any, ...], where: str) -> bool:
        if value not in allowed:
            self.fail(f"`{where}` is {value!r}; expected one of {', '.join(map(str, allowed))}")
            return False
        return True

    def text_list(self, value: Any, where: str, *, nonempty: bool = True) -> list[str]:
        if not isinstance(value, list) or (nonempty and not value):
            self.fail(f"`{where}` must be a {'non-empty ' if nonempty else ''}list")
            return []
        good = []
        for i, item in enumerate(value):
            if self.text(item, f"{where}[{i}]"):
                good.append(item)
        if len(set(good)) != len(good):
            self.fail(f"`{where}` has duplicates")
        return good

    def mapping(self, value: Any, where: str) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            self.fail(f"`{where}` must be a mapping")
            return {}
        return value

    def items(self, value: Any, where: str, *, nonempty: bool) -> list[Mapping[str, Any]]:
        if not isinstance(value, list) or (nonempty and not value):
            self.fail(f"`{where}` must be a {'non-empty ' if nonempty else ''}list")
            return []
        return [self.mapping(item, f"{where}[{i}]") for i, item in enumerate(value)]


def is_number(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _minimum_effect(c: _Checker, value: Any, where: str) -> None:
    effect = c.mapping(value, where)
    for key in ("value", "unit", "basis"):
        if key not in effect:
            c.fail(f"`{where}.{key}` is required")
    if "value" in effect and not is_number(effect["value"]):
        c.fail(f"`{where}.value` must be a number")
    elif "value" in effect and effect["value"] <= 0:
        c.fail(f"`{where}.value` must be positive")
    if "unit" in effect:
        c.text(effect["unit"], f"{where}.unit")
    if "basis" in effect:
        c.choice(effect["basis"], MINIMUM_EFFECT_BASES, f"{where}.basis")


def _partitions(c: _Checker, value: Any, expected: Mapping[str, str], where: str) -> None:
    if dict(c.mapping(value, where)) != dict(expected):
        c.fail(f"`{where}` must be exactly {dict(expected)} (final holdout is never used here)")


def validate_record(doc: Any, label: str) -> list[str]:
    c = _Checker(doc, label)
    d = c.doc
    c.fields(RECORD_FIELDS, OPTIONAL_RECORD_FIELDS)

    rid = d.get("id")
    if not (isinstance(rid, str) and RECORD_ID.match(rid)):
        c.fail(f"`id` {rid!r} must match H### or I###")
    kind = d.get("kind")
    c.choice(kind, RECORD_KINDS, "kind")
    if isinstance(rid, str) and RECORD_ID.match(rid):
        if (kind == "invariant") != rid.startswith("I"):
            c.fail("invariant records use I### ids and only invariants may")
    status = d.get("status")
    c.choice(status, STATUSES, "status")
    for name in (
        "title", "cluster", "situation", "eligible_opportunity", "intervention",
        "baseline", "mechanism", "owner",
    ):  # fmt: skip
        c.text(d.get(name), name)

    for i, src in enumerate(c.items(d.get("sources"), "sources", nonempty=True)):
        where = f"sources[{i}]"
        for key in src:
            if key not in ("source", "location", "digest_records", "digest_sections",
                           "transfer_caveat"):  # fmt: skip
                c.fail(f"unknown field `{where}.{key}`")
        c.text(src.get("source"), f"{where}.source")
        c.text(src.get("location"), f"{where}.location")
        c.text(src.get("transfer_caveat"), f"{where}.transfer_caveat")
        records = src.get("digest_records", [])
        sections = src.get("digest_sections", [])
        if not records and not sections:
            c.fail(f"`{where}` needs digest_records or digest_sections")
        for key in c.text_list(records, f"{where}.digest_records", nonempty=False):
            if not DIGEST_KEY.match(key):
                c.fail(f"`{where}.digest_records` entry {key!r} must look like 'mop/H01'")
        c.text_list(sections, f"{where}.digest_sections", nonempty=False)

    c.text_list(d.get("observable_inputs"), "observable_inputs")
    c.text_list(d.get("training_signals"), "training_signals", nonempty=False)
    c.text_list(d.get("exceptions"), "exceptions")
    c.text_list(d.get("related"), "related", nonempty=False)

    scope = c.mapping(d.get("variant_scope"), "variant_scope")
    if scope:
        transfer = scope.get("transfer")
        c.choice(transfer, TRANSFER_SCOPES, "variant_scope.transfer")
        variants = c.text_list(scope.get("variants"), "variant_scope.variants")
        for v in variants:
            c.choice(v, VARIANTS, "variant_scope.variants")
        if transfer == "generic" and variants != ["all"]:
            c.fail("generic records must list variants: [all]")
        if transfer in ("family", "variant_specific") and "all" in variants:
            c.fail("family or variant-specific records must name their variants")
        c.text(scope.get("notes"), "variant_scope.notes")

    for i, num in enumerate(c.items(d.get("source_numbers"), "source_numbers", nonempty=False)):
        where = f"source_numbers[{i}]"
        if not (is_number(num.get("value")) or isinstance(num.get("value"), str)):
            c.fail(f"`{where}.value` must be a number or text")
        c.text(num.get("context"), f"{where}.context")
        c.choice(num.get("use"), SOURCE_NUMBER_USES, f"{where}.use")

    names = []
    for i, p in enumerate(c.items(d.get("parameters"), "parameters", nonempty=False)):
        where = f"parameters[{i}]"
        if c.text(p.get("name"), f"{where}.name"):
            names.append(p["name"])
        if "default" not in p:
            c.fail(f"`{where}.default` is required (use null when unset)")
        basis = p.get("basis")
        if not c.choice(basis, PARAMETER_BASES, f"{where}.basis"):
            continue
        default = p.get("default")
        if basis == "sweep_only":
            if default is not None:
                c.fail(f"`{where}` is sweep_only, so its default must be null")
            if not p.get("sweep"):
                c.fail(f"`{where}` is sweep_only and needs a `sweep`")
        elif basis == "neutral":
            if default not in NEUTRAL_DEFAULTS or default is None:
                c.fail(f"`{where}` neutral default must be one of {NEUTRAL_DEFAULTS}")
            c.text(p.get("neutral_meaning"), f"{where}.neutral_meaning")
        elif basis == "analytic":
            c.text(p.get("derivation"), f"{where}.derivation")
        elif basis == "measured":
            c.text(p.get("evidence"), f"{where}.evidence")
            if status != "supported":
                c.fail(f"`{where}` has a measured default but the record is not supported")
    if len(set(names)) != len(names):
        c.fail("parameter names must be unique")

    exp = c.mapping(d.get("experiment"), "experiment")
    if exp:
        for key in exp:
            if key not in ("id", "primary_metric", "minimum_effect", "partitions",
                           "result_artifact"):  # fmt: skip
                c.fail(f"unknown field `experiment.{key}`")
        eid = exp.get("id")
        pattern = TEST_ID if kind == "invariant" else EXPERIMENT_ID
        if not (isinstance(eid, str) and pattern.match(eid)):
            c.fail(f"`experiment.id` {eid!r} must match {'T###' if kind == 'invariant' else 'E###'}")
        c.text(exp.get("primary_metric"), "experiment.primary_metric")
        _minimum_effect(c, exp.get("minimum_effect"), "experiment.minimum_effect")
        _partitions(
            c,
            exp.get("partitions"),
            TEST_PARTITIONS if kind == "invariant" else EXPERIMENT_PARTITIONS,
            "experiment.partitions",
        )
        if "result_artifact" not in exp:
            c.fail("`experiment.result_artifact` is required (null until a run concludes)")
        artifact = exp.get("result_artifact")
        c.text(artifact, "experiment.result_artifact", nullable=True)
        if status in CONCLUDED_STATUSES and not artifact:
            c.fail(f"status {status} requires `experiment.result_artifact`")

    tri = c.mapping(d.get("triage"), "triage")
    if tri:
        c.text(tri.get("rationale"), "triage.rationale")
        if kind == "invariant":
            if tri.get("mandatory") is not True or set(tri) != {"mandatory", "rationale"}:
                c.fail("invariant triage must be exactly {mandatory: true, rationale}")
        else:
            expected = set(TRIAGE_BENEFITS + TRIAGE_PENALTIES + ("rationale",))
            if set(tri) != expected:
                c.fail(f"strategy triage needs exactly {sorted(expected)}")
            for key in TRIAGE_BENEFITS + TRIAGE_PENALTIES:
                value = tri.get(key)
                if not (isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 3):
                    c.fail(f"`triage.{key}` must be an integer 0-3")

    impl = c.mapping(d.get("implementation"), "implementation")
    if impl:
        if set(impl) != {"module", "version_hash"}:
            c.fail("`implementation` needs exactly module and version_hash (null when absent)")
        c.text(impl.get("module"), "implementation.module", nullable=True)
        c.text(impl.get("version_hash"), "implementation.version_hash", nullable=True)

    log = c.items(d.get("decision_log"), "decision_log", nonempty=True)
    for i, entry in enumerate(log):
        where = f"decision_log[{i}]"
        if not (isinstance(entry.get("date"), str) and DATE.match(entry["date"])):
            c.fail(f"`{where}.date` must be a quoted YYYY-MM-DD string")
        c.choice(entry.get("status"), STATUSES, f"{where}.status")
        c.text(entry.get("by"), f"{where}.by")
        c.text(entry.get("note"), f"{where}.note")
    if log and log[-1].get("status") != status:
        c.fail("the last decision_log entry must record the current status")

    if status == "superseded":
        sup = d.get("superseded_by")
        if not (isinstance(sup, str) and RECORD_ID.match(sup)):
            c.fail("superseded records need `superseded_by: H###`")
    elif "superseded_by" in d:
        c.fail("`superseded_by` is only allowed on superseded records")
    if "notes" in d:
        c.text(d["notes"], "notes")
    return c.errors


def _common_spec(c: _Checker, schema: str, pattern: re.Pattern[str]) -> None:
    d = c.doc
    if d.get("schema") != schema:
        c.fail(f"`schema` must be {schema!r}")
    sid = d.get("id")
    if not (isinstance(sid, str) and pattern.match(sid)):
        c.fail(f"`id` {sid!r} has the wrong form")
    c.text(d.get("title"), "title")
    for rid in c.text_list(d.get("records"), "records"):
        if not RECORD_ID.match(rid):
            c.fail(f"`records` entry {rid!r} is not a record id")
    c.text(d.get("primary_metric"), "primary_metric")
    _minimum_effect(c, d.get("minimum_effect"), "minimum_effect")


def validate_experiment(doc: Any, label: str) -> list[str]:
    c = _Checker(doc, label)
    d = c.doc
    c.fields(EXPERIMENT_FIELDS)
    _common_spec(c, SPEC_SCHEMA["experiment"], EXPERIMENT_ID)
    kind = d.get("kind")
    c.choice(kind, EXPERIMENT_KINDS, "kind")
    c.choice(d.get("mode"), EXPERIMENT_MODES, "mode")
    c.choice(d.get("status"), EXPERIMENT_STATUSES, "status")
    c.text(d.get("question"), "question")

    arms = c.items(d.get("arms"), "arms", nonempty=True)
    if len(arms) < 2:
        c.fail("an experiment needs at least two arms")
    toggles = c.mapping(d.get("toggles"), "toggles")
    arm_names = []
    for i, arm in enumerate(arms):
        where = f"arms[{i}]"
        if c.text(arm.get("name"), f"{where}.name"):
            arm_names.append(arm["name"])
        c.choice(arm.get("role"), ARM_ROLES, f"{where}.role")
        c.text(arm.get("description"), f"{where}.description")
        for name, value in c.mapping(arm.get("toggles"), f"{where}.toggles").items():
            if name not in toggles:
                c.fail(f"`{where}.toggles.{name}` is not declared in `toggles`")
            elif isinstance(toggles[name], list) and value not in toggles[name]:
                c.fail(f"`{where}.toggles.{name}` value {value!r} is not a declared level")
    if len(set(arm_names)) != len(arm_names):
        c.fail("arm names must be unique")
    if sum(1 for arm in arms if arm.get("role") == "control") != 1:
        c.fail("exactly one arm must be the control")

    factors = c.items(d.get("factors"), "factors", nonempty=False)
    if kind == "factorial" and len(factors) < 2:
        c.fail("factorial experiments need at least two factors")
    records = d.get("records") if isinstance(d.get("records"), list) else []
    for i, factor in enumerate(factors):
        if factor.get("toggle") not in toggles:
            c.fail(f"`factors[{i}].toggle` must name a declared toggle")
        for rid in c.text_list(factor.get("records_tested"), f"factors[{i}].records_tested"):
            if rid not in records:
                c.fail(f"`factors[{i}].records_tested` names {rid}, which is not in `records`")

    lineups = c.mapping(d.get("lineups"), "lineups")
    for key in ("train", "evaluate"):
        if key not in lineups:
            c.fail(f"`lineups.{key}` is required")
    variants = c.mapping(d.get("variants"), "variants")
    for v in c.text_list(variants.get("primary"), "variants.primary"):
        c.choice(v, VARIANTS, "variants.primary")
    for v in c.text_list(variants.get("rehearsal", []), "variants.rehearsal", nonempty=False):
        c.choice(v, VARIANTS, "variants.rehearsal")
    seeds = c.mapping(d.get("seeds"), "seeds")
    c.text(seeds.get("namespace"), "seeds.namespace")
    design = c.mapping(d.get("design"), "design")
    for key in ("unit", "interval", "sample_size"):
        c.text(design.get(key), f"design.{key}")

    c.text_list(d.get("metrics"), "metrics", nonempty=False)
    c.text_list(d.get("slices"), "slices", nonempty=False)
    c.text_list(d.get("guardrails"), "guardrails")
    c.text_list(d.get("blocked_by"), "blocked_by", nonempty=False)
    run_specs = c.text_list(d.get("run_specs"), "run_specs", nonempty=False)
    if d.get("status") in RUNNABLE_STATUSES and not run_specs:
        c.fail(f"status {d.get('status')} needs at least one entry in `run_specs`")
    _partitions(c, d.get("partitions"), EXPERIMENT_PARTITIONS, "partitions")
    budget = c.mapping(d.get("budget"), "budget")
    if not is_number(budget.get("cpu_core_hours")):
        c.fail("`budget.cpu_core_hours` must be a number")
    gates = c.mapping(d.get("gates"), "gates")
    for key in ("promote", "reject", "execution"):
        if key not in gates:
            c.fail(f"`gates.{key}` is required")
    if "result_artifact" not in d:
        c.fail("`result_artifact` is required (null until complete)")
    elif d.get("status") == "complete" and not d.get("result_artifact"):
        c.fail("complete experiments need a result_artifact")
    return c.errors


def validate_test_spec(doc: Any, label: str) -> list[str]:
    c = _Checker(doc, label)
    d = c.doc
    c.fields(TEST_SPEC_FIELDS)
    _common_spec(c, SPEC_SCHEMA["test"], TEST_ID)
    if d.get("kind") != "invariant_test":
        c.fail("`kind` must be invariant_test")
    c.choice(d.get("owner_workstream"), WORKSTREAMS, "owner_workstream")
    c.text(d.get("target"), "target")
    c.text(d.get("test_location"), "test_location")
    status = d.get("status")
    c.choice(status, TEST_SPEC_STATUSES, "status")
    implemented = c.text_list(d.get("implemented_by"), "implemented_by", nonempty=False)
    if status == "implemented" and not implemented:
        c.fail("implemented test specs must list implemented_by test node ids")
    if status == "specified" and implemented:
        c.fail("a spec with implemented_by entries must have status implemented")
    _partitions(c, d.get("partitions"), TEST_PARTITIONS, "partitions")
    cases = c.items(d.get("cases"), "cases", nonempty=True)
    for i, case in enumerate(cases):
        c.text(case.get("name"), f"cases[{i}].name")
        if "given" not in case or "expect" not in case:
            c.fail(f"`cases[{i}]` needs `given` and `expect`")
    c.text_list(d.get("properties"), "properties", nonempty=False)
    c.text_list(d.get("mutations"), "mutations")
    return c.errors
