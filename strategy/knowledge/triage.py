"""Score hypothesis records and rank the experiments that test them.

Run ``uv run python -m strategy.knowledge.triage --write`` after editing records, and
``--check`` (also run by the linter) to confirm the committed index is current.

Record score (plan section 6): the sum of seven 0-3 benefit scores minus three 0-3
penalties, so the range is -9 to 21. Invariants are not scored; they are mandatory
tests. Correlated records share one experiment, so the ranking is over experiments:
priority is the best member score, ties broken by mean member score, then by number
of member records (more independent sources pointing at one mechanism), then by id.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from strategy.knowledge.registry import KnowledgeBase, load
from strategy.knowledge.schema import TRIAGE_BENEFITS, TRIAGE_PENALTIES

TOP_N = 15

# The plan's provisional first batch (section 6, "Prioritized first batch: top 15"),
# mapped to the experiment spec that tests each row, in the plan's order.
PLAN_TOP15 = (
    "E001",  # terminal points vs chip-EV training
    "E004",  # cap-aware horizon features
    "E003",  # full stack vector and point-risk premium for all-in calls
    "E009",  # opponent-specific effective stacks and side-pot eligibility
    "E010",  # paired-block promotion
    "E006",  # full compact preflop actions vs forced shove/fold
    "E007",  # point-aware push and call tables
    "E008",  # joint multiway caller subsets
    "E011",  # contextual Bayesian beliefs
    "E012",  # robust vs raw vs confidence-capped exploitation
    "E013",  # frozen league vs latest-only self-play
    "E014",  # blocker-aware distributional card buckets
    "E015",  # small adaptive action menu and off-tree translation
    "E016",  # trajectory learning vs isolated-street labels
    "E017",  # byte-quantized mixed policy vs float teacher
)


def record_score(triage: dict[str, Any]) -> int:
    return sum(triage[k] for k in TRIAGE_BENEFITS) - sum(triage[k] for k in TRIAGE_PENALTIES)


@dataclass(frozen=True)
class RankedExperiment:
    id: str
    title: str
    kind: str
    priority: int
    mean: float
    members: tuple[tuple[str, int], ...]
    plan_rank: int | None

    def sort_key(self) -> tuple[Any, ...]:
        return (-self.priority, -self.mean, -len(self.members), self.id)


def _records(kb: KnowledgeBase) -> dict[str, dict[str, Any]]:
    return {d.data["id"]: d.data for d in kb.records if isinstance(d.data, dict)}


def rank(kb: KnowledgeBase) -> list[RankedExperiment]:
    records = _records(kb)
    ranked = []
    for doc in kb.experiments:
        spec = doc.data
        members = []
        for rid in spec["records"]:
            rec = records.get(rid)
            if rec and rec["kind"] != "invariant":
                members.append((rid, record_score(rec["triage"])))
        if not members:
            continue
        scores = [s for _, s in members]
        plan = spec["id"] in PLAN_TOP15
        ranked.append(
            RankedExperiment(
                id=spec["id"],
                title=spec["title"],
                kind=spec["kind"],
                priority=max(scores),
                mean=round(sum(scores) / len(scores), 2),
                members=tuple(sorted(members, key=lambda m: (-m[1], m[0]))),
                plan_rank=PLAN_TOP15.index(spec["id"]) + 1 if plan else None,
            )
        )
    return sorted(ranked, key=RankedExperiment.sort_key)


def _verdict(position: int, item: RankedExperiment) -> str:
    if position <= TOP_N:
        return "confirmed" if item.plan_rank else "added"
    return "dropped" if item.plan_rank else ""


def build(kb: KnowledgeBase) -> dict[str, Any]:
    records = _records(kb)
    ranked = rank(kb)
    top = [r.id for r in ranked[:TOP_N]]
    experiments = []
    for pos, item in enumerate(ranked, 1):
        experiments.append(
            {
                "rank": pos,
                "id": item.id,
                "title": item.title,
                "kind": item.kind,
                "priority": item.priority,
                "mean_score": item.mean,
                "records": [{"id": rid, "score": s} for rid, s in item.members],
                "plan_rank": item.plan_rank,
                "top15": _verdict(pos, item) or None,
            }
        )
    scored = sorted(
        (r for r in records.values() if r["kind"] != "invariant"),
        key=lambda r: (-record_score(r["triage"]), r["id"]),
    )
    invariants = sorted(
        (r for r in records.values() if r["kind"] == "invariant"), key=lambda r: r["id"]
    )
    tests = {d.data["id"]: d.data for d in kb.test_specs if isinstance(d.data, dict)}
    return {
        "generated_by": "python -m strategy.knowledge.triage --write",
        "top15": top,
        "plan_top15": list(PLAN_TOP15),
        "confirmed": [e for e in PLAN_TOP15 if e in top],
        "added": [e for e in top if e not in PLAN_TOP15],
        "dropped": [e for e in PLAN_TOP15 if e not in top],
        "experiments": experiments,
        "records": [
            {
                "id": r["id"],
                "title": r["title"],
                "cluster": r["cluster"],
                "score": record_score(r["triage"]),
                "triage": {k: r["triage"][k] for k in TRIAGE_BENEFITS + TRIAGE_PENALTIES},
                "experiment": r["experiment"]["id"],
                "status": r["status"],
            }
            for r in scored
        ],
        "mandatory_tests": [
            {
                "id": r["id"],
                "title": r["title"],
                "test_spec": r["experiment"]["id"],
                "test_status": tests.get(r["experiment"]["id"], {}).get("status"),
            }
            for r in invariants
        ],
    }


def render_yaml(index: dict[str, Any]) -> str:
    header = "# Generated file; do not edit. Regenerate with:\n#   uv run python -m strategy.knowledge.triage --write\n"
    return header + yaml.safe_dump(index, sort_keys=False, allow_unicode=True, width=100)


def render_markdown(index: dict[str, Any]) -> str:
    by_id = {e["id"]: e for e in index["experiments"]}
    lines = [
        "# Knowledge triage index",
        "",
        "Generated by `uv run python -m strategy.knowledge.triage --write`; do not edit.",
        "",
        "Record score = objective + reach + effect + transfer + readiness + power + risk",
        "(each 0-3) minus rule uncertainty, leakage and complexity (each 0-3). Experiment",
        "priority is the best member score; ties go to the higher mean, then more member",
        "records, then the lower id. Invariants are mandatory tests and are not ranked.",
        "",
        f"## Top {TOP_N} experiments",
        "",
        "| Rank | Experiment | Priority | Mean | Records (score) | Plan rank | Verdict |",
        "|---:|---|---:|---:|---|---:|---|",
    ]
    for e in index["experiments"][:TOP_N]:
        lines.append(_experiment_row(e))
    lines += [
        "",
        f"Confirmed from the plan: {len(index['confirmed'])} of {TOP_N}. "
        f"Added: {', '.join(index['added']) or 'none'}. "
        f"Dropped: {', '.join(index['dropped']) or 'none'}.",
        "",
    ]
    if index["dropped"]:
        lines += ["Dropped plan rows and where they now rank:", ""]
        for eid in index["dropped"]:
            e = by_id[eid]
            lines.append(f"- {eid} {e['title']}: rank {e['rank']}, priority {e['priority']}, "
                         f"plan rank {e['plan_rank']}.")
        lines.append("")
    lines += [
        "## All experiments",
        "",
        "| Rank | Experiment | Priority | Mean | Records (score) | Plan rank | Verdict |",
        "|---:|---|---:|---:|---|---:|---|",
    ]
    lines += [_experiment_row(e) for e in index["experiments"]]
    lines += [
        "",
        "## Records by score",
        "",
        "| Record | Title | Cluster | Score | obj/reach/eff/xfer/ready/power/risk − rule/leak/cplx | Experiment |",
        "|---|---|---|---:|---|---|",
    ]
    for r in index["records"]:
        t = r["triage"]
        benefits = "/".join(str(t[k]) for k in TRIAGE_BENEFITS)
        penalties = "/".join(str(t[k]) for k in TRIAGE_PENALTIES)
        lines.append(f"| {r['id']} | {r['title']} | {r['cluster']} | {r['score']} | "
                     f"{benefits} − {penalties} | {r['experiment']} |")
    lines += [
        "",
        "## Mandatory invariant tests",
        "",
        "| Record | Title | Test spec | Status |",
        "|---|---|---|---|",
    ]
    for r in index["mandatory_tests"]:
        lines.append(f"| {r['id']} | {r['title']} | {r['test_spec']} | {r['test_status']} |")
    return "\n".join(lines) + "\n"


def _experiment_row(e: dict[str, Any]) -> str:
    members = ", ".join(f"{m['id']} ({m['score']})" for m in e["records"])
    plan = str(e["plan_rank"]) if e["plan_rank"] else ""
    return (f"| {e['rank']} | {e['id']} {e['title']} | {e['priority']} | {e['mean_score']:g} | "
            f"{members} | {plan} | {e['top15'] or ''} |")


def outputs(kb: KnowledgeBase) -> dict[Path, str]:
    index = build(kb)
    return {
        kb.root / "index.md": render_markdown(index),
        kb.root / "index.yaml": render_yaml(index),
    }


def check(kb: KnowledgeBase) -> list[str]:
    errors = []
    for path, text in outputs(kb).items():
        if not path.exists() or path.read_text(encoding="utf-8") != text:
            errors.append(f"{path.name} is stale; run `uv run python -m strategy.knowledge.triage --write`")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--write", action="store_true", help="regenerate index.md and index.yaml")
    group.add_argument("--check", action="store_true", help="fail if the index is stale")
    args = parser.parse_args(argv)
    kb = load()
    from strategy.knowledge.lint import lint

    errors = lint(kb, check_index=False)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        print("fix lint errors before triage", file=sys.stderr)
        return 1
    if args.write:
        for path, text in outputs(kb).items():
            path.write_text(text, encoding="utf-8")
        print(f"wrote {', '.join(sorted(p.name for p in outputs(kb)))}")
        return 0
    if args.check:
        stale = check(kb)
        print("\n".join(stale) or "triage index: up to date", file=sys.stderr if stale else sys.stdout)
        return 1 if stale else 0
    for e in build(kb)["experiments"][:TOP_N]:
        print(f"{e['rank']:>2}. {e['id']} {e['title']} (priority {e['priority']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
