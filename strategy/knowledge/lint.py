"""Lint the knowledge base.

Run ``uv run python -m strategy.knowledge.lint``; exits nonzero on any error.

Beyond per-document schema checks (``strategy.knowledge.schema``) this enforces:

* every record links to an experiment spec (strategy and method records) or an
  invariant test spec (invariants), the spec lists the record back, and the record's
  primary metric, minimum effect and partitions match the spec;
* every source, digest record and vocabulary term a record cites exists, and every
  hypothesis in the digest index is cited by at least one record;
* observable inputs are observable at decision time (offline-only quantities may only
  appear as training signals);
* no parameter default equals a source number unless that number is analytic, so an
  untested book or article figure cannot silently become behaviour;
* implemented test specs point at test functions that exist;
* the generated triage index is up to date.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from strategy.knowledge import schema
from strategy.knowledge.registry import REPO_ROOT, Document, KnowledgeBase, load


def _same_number(a: Any, b: Any) -> bool:
    if schema.is_number(a) and schema.is_number(b):
        return abs(float(a) - float(b)) <= 1e-12 * max(1.0, abs(float(a)), abs(float(b)))
    return isinstance(a, str) and isinstance(b, str) and a.strip() == b.strip()


def _docs(kb: KnowledgeBase) -> Iterable[tuple[str, Document]]:
    for doc in kb.records:
        yield "record", doc
    for doc in kb.experiments:
        yield "experiment", doc
    for doc in kb.test_specs:
        yield "test", doc


VALIDATORS = {
    "record": schema.validate_record,
    "experiment": schema.validate_experiment,
    "test": schema.validate_test_spec,
}


def _check_test_node(node: str) -> str | None:
    path, sep, name = node.partition("::")
    if not sep or not name:
        return f"implemented_by entry {node!r} must look like 'tests/x.py::test_name'"
    file = REPO_ROOT / path
    if not file.is_file():
        return f"implemented_by file {path} does not exist"
    if f"def {name.split('[')[0]}(" not in file.read_text(encoding="utf-8"):
        return f"implemented_by test {name} is not defined in {path}"
    return None


def lint(kb: KnowledgeBase | None = None, *, check_index: bool = True) -> list[str]:
    kb = kb or load()
    errors: list[str] = []
    valid: dict[str, list[Document]] = {"record": [], "experiment": [], "test": []}
    seen: dict[str, str] = {}

    for kind, doc in _docs(kb):
        if doc.error:
            errors.append(f"{doc.label}: {doc.error}")
            continue
        found = VALIDATORS[kind](doc.data, doc.label)
        errors.extend(found)
        data = doc.data if isinstance(doc.data, dict) else {}
        did = data.get("id")
        if isinstance(did, str):
            if did in seen:
                errors.append(f"{doc.label}: duplicate id {did} (also in {seen[did]})")
            seen[did] = doc.label
            if not doc.path.name.startswith(f"{did}-") and doc.path.stem != did:
                errors.append(f"{doc.label}: file name must start with its id {did}")
        if not found:
            valid[kind].append(doc)

    features = kb.vocabulary.get("features") or {}
    metrics = kb.vocabulary.get("metrics") or {}
    sources = kb.sources.get("sources") or {}
    digests = kb.sources.get("digests") or {}
    digest_index = kb.digest_index.get("records") or {}
    if not features or not metrics:
        errors.append("vocabulary.yaml: needs `features` and `metrics`")
    if not sources:
        errors.append("sources.yaml: needs `sources`")
    if not digest_index:
        errors.append("digest_index.yaml: needs `records`")
    for key, entry in digest_index.items():
        prefix = key.split("/")[0]
        if prefix not in digests:
            errors.append(f"digest_index.yaml: {key} uses unknown digest {prefix!r}")
        if not isinstance(entry, dict) or not entry.get("title") or not entry.get("location"):
            errors.append(f"digest_index.yaml: {key} needs a title and location")
    for key, entry in sources.items():
        if isinstance(entry, dict) and entry.get("digest") and entry["digest"] not in digests:
            errors.append(f"sources.yaml: {key} names unknown digest {entry['digest']!r}")

    records = {d.data["id"]: d for d in valid["record"]}
    experiments = {d.data["id"]: d for d in valid["experiment"]}
    tests = {d.data["id"]: d for d in valid["test"]}
    all_record_ids = {d.data.get("id") for d in kb.records if isinstance(d.data, dict)}
    cited: set[str] = set()

    def metric_unit(label: str, metric: Any, effect: Any) -> None:
        if metric not in metrics:
            errors.append(f"{label}: unknown metric {metric!r}")
        elif isinstance(effect, dict) and effect.get("unit") != metrics[metric].get("unit"):
            errors.append(
                f"{label}: minimum effect unit {effect.get('unit')!r} does not match "
                f"{metric}'s unit {metrics[metric].get('unit')!r}"
            )

    for rid, doc in records.items():
        r, label = doc.data, doc.label
        for src in r["sources"]:
            if src["source"] not in sources:
                errors.append(f"{label}: unknown source {src['source']!r}")
            for key in src.get("digest_records", []):
                if key not in digest_index:
                    errors.append(f"{label}: digest record {key} is not in digest_index.yaml")
                cited.add(key)
        for name in r["observable_inputs"]:
            if name not in features:
                errors.append(f"{label}: observable input {name!r} is not in the vocabulary")
            elif features[name].get("observable") is not True:
                errors.append(
                    f"{label}: {name!r} is not observable at decision time; "
                    "list it under training_signals"
                )
        for name in r["training_signals"]:
            if name not in features:
                errors.append(f"{label}: training signal {name!r} is not in the vocabulary")
        for other in r["related"]:
            if other not in all_record_ids:
                errors.append(f"{label}: related record {other} does not exist")
            if other == rid:
                errors.append(f"{label}: a record cannot relate to itself")
        if r.get("superseded_by") and r["superseded_by"] not in all_record_ids:
            errors.append(f"{label}: superseded_by {r['superseded_by']} does not exist")

        # Schema rules already force sweep_only defaults to null, neutral defaults to
        # no-op values and measured defaults onto supported records with artifacts, so
        # the remaining route for a book figure to become behaviour is an "analytic"
        # label on a number the record itself classifies as non-analytic.
        for p in r["parameters"]:
            if p["basis"] != "analytic" or p.get("default") is None:
                continue
            for num in r["source_numbers"]:
                if num["use"] != "analytic" and _same_number(p["default"], num["value"]):
                    errors.append(
                        f"{label}: parameter {p['name']} defaults to source number "
                        f"{num['value']!r} ({num['use']}); unsupported source numbers "
                        "cannot become defaults"
                    )
        link = r["experiment"]
        target = (tests if r["kind"] == "invariant" else experiments).get(link["id"])
        if target is None:
            errors.append(f"{label}: links to {link['id']}, which has no valid spec")
            continue
        spec = target.data
        if rid not in spec["records"]:
            errors.append(f"{label}: {link['id']} does not list {rid} in its records")
        for key in ("primary_metric", "minimum_effect", "partitions"):
            if link[key] != spec[key]:
                errors.append(f"{label}: experiment.{key} differs from {target.label}")
        if r["kind"] == "invariant" and spec["status"] == "implemented":
            if not r["implementation"].get("module"):
                errors.append(f"{label}: {link['id']} is implemented; set implementation.module")

    for key in sorted(set(digest_index) - cited):
        errors.append(f"digest_index.yaml: {key} is not cited by any record")

    for specs, kind in ((experiments, "experiment"), (tests, "test")):
        for sid, doc in specs.items():
            spec, label = doc.data, doc.label
            metric_unit(label, spec["primary_metric"], spec["minimum_effect"])
            for m in spec.get("metrics", []):
                if m not in metrics:
                    errors.append(f"{label}: unknown metric {m!r}")
            for rid in spec["records"]:
                rec = records.get(rid)
                if rec is None:
                    if rid not in all_record_ids:
                        errors.append(f"{label}: lists unknown record {rid}")
                    continue
                if rec.data["experiment"]["id"] != sid:
                    errors.append(f"{label}: lists {rid}, but {rid} links to "
                                  f"{rec.data['experiment']['id']}")
                if (rec.data["kind"] == "invariant") != (kind == "test"):
                    errors.append(f"{label}: {rid} is the wrong record kind for this spec")
            if kind == "test":
                for node in spec["implemented_by"]:
                    problem = _check_test_node(node)
                    if problem:
                        errors.append(f"{label}: {problem}")

    if check_index and not errors:
        from strategy.knowledge import triage

        errors.extend(triage.check(kb))
    return errors


def main(argv: list[str] | None = None) -> int:
    root = Path(argv[0]) if argv else None
    kb = load(root) if root else load()
    errors = lint(kb)
    for error in errors:
        print(error, file=sys.stderr)
    if errors:
        print(f"knowledge lint: {len(errors)} error(s)", file=sys.stderr)
        return 1
    print(
        f"knowledge lint: ok ({len(kb.records)} records, {len(kb.experiments)} experiment "
        f"specs, {len(kb.test_specs)} test specs)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
