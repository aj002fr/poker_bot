"""Metamorphic gate (spec 12.2) at release scale; the relations live in ``tests/metamorphic/relations.py``.

Per ruleset and initial N=2..8, ``BASES_PER_CELL`` seeded base tables (hand cap
32, four hands per level so blinds escalate into all-ins). Every base is checked
under two suit permutations, two seat/button rotations, two chip-scaling factors
and one table relabel, each parameter drawn from the case seed. Chip scaling
stops at the first odd-chip hand; the excluded hands are counted. One shard per
ruleset also runs the serial-versus-parallel scheduling relation.
"""

from __future__ import annotations

import itertools
import random
from collections import Counter
from typing import Any

from tests.metamorphic import relations as R
from tests.property.generated import RULESETS
from tests.release.workloads.common import CaseRunner, digest

BASES_PER_CELL = 200
SUIT_PERMS = ["".join(p) for p in itertools.permutations("cdhs") if "".join(p) != "cdhs"]
SCALES = [2, 3, 7, 10, 1000]
SVP = [(2, 4), (5, 2), (8, 2)]


def case_seed(ruleset: str, n: int, index: int) -> str:
    return f"meta|{ruleset}|{n}|{index}"


def variants(ruleset: str, n: int, index: int) -> list[tuple[str, Any]]:
    rng = random.Random(f"{case_seed(ruleset, n, index)}|params")
    return ([("suits", p) for p in rng.sample(SUIT_PERMS, 2)] +
            [("rotation", r) for r in rng.sample(range(1, 8), 2)] +
            [("scaling", k) for k in rng.sample(SCALES, 2)] +
            [("table_id", f"t_relabel_{index}")])


def check(case: dict[str, Any], counts: Counter, base: R.Base | None = None) -> str:
    ruleset, n, index = case["ruleset"], case["players"], case["index"]
    base = base or R.play_base(R.base_config(ruleset, n), case_seed(ruleset, n, index))
    relation, param = case["relation"], case["param"]
    if relation == "suits":
        h = R.suit_renaming(base, param)
    elif relation == "rotation":
        h = R.seat_rotation(base, param)
    elif relation == "scaling":
        h, cut = R.chip_scaling(base, param)
        if cut is not None:
            counts["scaling_tables_truncated_at_odd_chip"] += 1
            counts["scaling_hands_compared"] += cut - 1
            counts["scaling_hands_excluded"] += len(base.table.hands) - cut + 1
        else:
            counts["scaling_hands_compared"] += len(base.table.hands)
    elif relation == "table_id":
        h = R.table_relabel(base, param)
    else:
        raise ValueError(relation)
    counts[f"relation:{relation}"] += 1
    counts["variant_hands"] += len(base.table.hands)
    return h


def run_case(case: dict[str, Any]) -> str:
    if case["relation"] == "scheduling":
        return digest(R.serial_vs_parallel(case["ruleset"], case["players"], case["blocks"], case["seed"]))
    return check(case, Counter())


def run_shard(params: dict[str, Any]) -> dict[str, Any]:
    runner = CaseRunner("metamorphic")
    ruleset = params["ruleset"]
    if params["kind"] == "scheduling":
        for n, blocks in SVP:
            case = {"relation": "scheduling", "ruleset": ruleset, "players": n, "blocks": blocks,
                    "seed": f"meta-svp|{ruleset}|{n}"}

            def svp(c: dict[str, Any]) -> str:
                out = R.serial_vs_parallel(c["ruleset"], c["players"], c["blocks"], c["seed"])
                runner.counts["relation:scheduling"] += 1
                runner.counts["scheduling_tables_compared"] += out["tables"]
                return digest(out)

            runner.run(case, svp)
        return runner.result()
    n = params["players"]
    for index in range(params["count"]):
        base = None

        def make_base() -> R.Base:
            return R.play_base(R.base_config(ruleset, n), case_seed(ruleset, n, index))

        try:
            base = make_base()
        except Exception as exc:  # recorded as a failing case of its own
            runner.fail({"relation": "base", "ruleset": ruleset, "players": n, "index": index}, repr(exc))
            continue
        runner.counts["base_tables"] += 1
        runner.counts["base_hands"] += len(base.table.hands)
        for relation, param in variants(ruleset, n, index):
            case = {"relation": relation, "ruleset": ruleset, "players": n, "index": index, "param": param}
            runner.run(case, lambda c, b=base: check(c, runner.counts, b))
    if runner.counts["scaling_hands_excluded"]:
        runner.exclusions["chip scaling: hands from the first odd-chip split on (cannot scale exactly)"] = \
            runner.counts["scaling_hands_excluded"]
    return runner.result()


def shards() -> list[dict[str, Any]]:
    out = [{"gate": "metamorphic", "kind": "tables", "ruleset": r, "players": n, "count": BASES_PER_CELL}
           for r in RULESETS for n in range(2, 9)]
    out += [{"gate": "metamorphic", "kind": "scheduling", "ruleset": r} for r in RULESETS]
    return out


def finalize(gate: str, results: list[dict[str, Any]], all_results: dict) -> list[dict[str, Any]]:
    totals: Counter = Counter()
    for r in results:
        totals.update(r["counts"])
    need = BASES_PER_CELL * len(RULESETS) * 7
    return [
        {"check": f"base tables >= {need:,} ({BASES_PER_CELL} per ruleset and N=2..8)",
         "passed": totals["base_tables"] >= need,
         "detail": f"{totals['base_tables']:,} base tables, {totals['base_hands']:,} hands"},
        {"check": "every relation exercised", "passed": all(totals[f"relation:{k}"] > 0 for k in
                                                            ("suits", "rotation", "scaling", "table_id", "scheduling")),
         "detail": {k: totals[f"relation:{k}"] for k in ("suits", "rotation", "scaling", "table_id", "scheduling")}},
        {"check": "chip scaling compared hands, odd-chip exclusions counted",
         "passed": totals["scaling_hands_compared"] > 0,
         "detail": f"{totals['scaling_hands_compared']:,} hands compared exactly; "
                   f"{totals['scaling_hands_excluded']:,} hands in {totals['scaling_tables_truncated_at_odd_chip']:,} "
                   "tables excluded from the first odd-chip split on"},
    ]
