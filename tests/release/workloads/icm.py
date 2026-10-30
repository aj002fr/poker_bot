"""ICM gate: the engine's memoized subset recursion against the exact permutation oracle.

Randomized cases at N=2..8 mix unequal, equal, one-dominant and one-chip
stacks with top-heavy, flat, sparse, winner-take-all and random nonincreasing
prize vectors (100-point pools). Every case checks:

- absolute error <= 1e-9 against ``icm_permutations(exact=True)`` (rationals,
  all k! finishing orders);
- conservation: the expected prizes sum to the prize pool;
- symmetry: permuting the stacks permutes the result, and equal stacks get
  equal shares.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Any

from arena.tournament.scoring import SCORE_TOLERANCE, icm
from tests.oracles.icm import icm_permutations
from tests.release.workloads.common import CaseRunner, digest

CASES_PER_N = 1_000
SHARD_CASES = 250

STACK_KINDS = ("unequal", "equal", "one_dominant", "one_chip", "near_equal")
PRIZE_KINDS = ("top_three", "flat", "sparse", "winner_take_all", "random")


def make_case(n: int, index: int) -> dict[str, Any]:
    rng = random.Random(f"arena-release|icm|{n}|{index}")
    stack_kind = STACK_KINDS[index % len(STACK_KINDS)]
    prize_kind = PRIZE_KINDS[(index // len(STACK_KINDS)) % len(PRIZE_KINDS)]
    if stack_kind == "equal":
        stacks = [rng.randint(1, 16_000)] * n
    elif stack_kind == "one_dominant":
        stacks = [rng.randint(1, 50) for _ in range(n)]
        stacks[rng.randrange(n)] = rng.randint(10_000, 16_000)
    elif stack_kind == "one_chip":
        stacks = [rng.randint(1, 16_000) for _ in range(n)]
        stacks[rng.randrange(n)] = 1
    elif stack_kind == "near_equal":
        base = rng.randint(1_000, 4_000)
        stacks = [base + rng.randint(-2, 2) for _ in range(n)]
    else:
        stacks = [rng.randint(1, 16_000) for _ in range(n)]
    if prize_kind == "top_three":
        prizes = ([65, 35] if n == 2 else ([50, 30, 20] + [0] * n)[:n])
    elif prize_kind == "flat":
        prizes = [100 // n] * n
        prizes[0] += 100 - sum(prizes)
    elif prize_kind == "sparse":
        k = rng.randint(1, max(1, n - 1))
        cuts = sorted(rng.sample(range(1, 100), k - 1)) if k > 1 else []
        parts = sorted((b - a for a, b in zip([0, *cuts], [*cuts, 100])), reverse=True)
        prizes = parts + [0] * (n - len(parts))
    elif prize_kind == "winner_take_all":
        prizes = [100] + [0] * (n - 1)
    else:
        cuts = sorted(rng.sample(range(0, 101), n - 1))
        prizes = sorted((b - a for a, b in zip([0, *cuts], [*cuts, 100])), reverse=True)
    return {"players": n, "index": index, "stacks": stacks, "prizes": prizes,
            "stack_kind": stack_kind, "prize_kind": prize_kind}


def check_case(case: dict[str, Any], counts: Counter) -> str:
    stacks, prizes = case["stacks"], case["prizes"]
    n = len(stacks)
    engine = icm(stacks, prizes)
    oracle = icm_permutations(dict(enumerate(stacks)), prizes, exact=True)
    worst = 0.0
    for i in range(n):
        err = abs(engine[i] - float(oracle.points[i]))
        worst = max(worst, err)
        assert err <= SCORE_TOLERANCE, f"player {i}: engine {engine[i]!r}, oracle {oracle.points[i]} (error {err:.3e})"
    pool = sum(prizes[:n])
    assert abs(sum(engine) - pool) <= SCORE_TOLERANCE, f"sum {sum(engine)!r} != pool {pool}"
    perm = list(range(n))
    random.Random(f"{case['players']}|{case['index']}|perm").shuffle(perm)
    permuted = icm([stacks[j] for j in perm], prizes)
    for k, j in enumerate(perm):
        assert abs(permuted[k] - engine[j]) <= 1e-12, "permuting stacks does not permute the result"
    for i in range(n):
        for j in range(i + 1, n):
            if stacks[i] == stacks[j]:
                assert abs(engine[i] - engine[j]) <= 1e-12, f"equal stacks {i},{j} get {engine[i]} and {engine[j]}"
    counts["cases"] += 1
    counts[f"stacks:{case['stack_kind']}"] += 1
    counts[f"prizes:{case['prize_kind']}"] += 1
    counts["orders_enumerated"] += oracle.orders
    counts["max_abs_error_e18"] = max(counts["max_abs_error_e18"], int(worst * 1e18))
    return digest([round(x, 12) for x in engine])


def run_case(case: dict[str, Any]) -> str:
    return check_case(case, Counter())


def run_shard(params: dict[str, Any]) -> dict[str, Any]:
    runner = CaseRunner("icm")
    for index in range(params["start"], params["start"] + params["count"]):
        case = make_case(params["players"], index)
        runner.run(case, lambda c: check_case(c, runner.counts))
    return runner.result()


def shards() -> list[dict[str, Any]]:
    return [{"gate": "icm", "players": n, "start": k * SHARD_CASES, "count": SHARD_CASES}
            for n in range(2, 9) for k in range(CASES_PER_N // SHARD_CASES)]


def finalize(gate: str, results: list[dict[str, Any]], all_results: dict) -> list[dict[str, Any]]:
    per_n: Counter = Counter()
    totals: Counter = Counter()
    worst = 0
    for r in results:
        per_n[r["params"]["players"]] += r["counts"].get("cases", 0)
        worst = max(worst, r["counts"].get("max_abs_error_e18", 0))
        totals.update({k: v for k, v in r["counts"].items() if k != "max_abs_error_e18"})
    return [
        {"check": ">= 1,000 randomized cases across N=2..8", "passed": totals["cases"] >= 1_000 and all(per_n[n] for n in range(2, 9)),
         "detail": {str(n): per_n[n] for n in range(2, 9)}},
        {"check": "sparse and flat prize vectors and unequal stacks exercised",
         "passed": all(totals[f"prizes:{k}"] for k in ("sparse", "flat")) and totals["stacks:unequal"] > 0,
         "detail": {k: v for k, v in sorted(totals.items()) if ":" in k}},
        {"check": "max absolute error <= 1e-9 (100-point pools)", "passed": worst <= 1e9,
         "detail": f"{worst / 1e18:.3e}"},
    ]
