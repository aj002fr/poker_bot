"""Deterministic-repeat gate: bulk shards rerun in fresh processes give identical result hashes.

Each shard picks one already-passed shard of a bulk gate, reruns its workload
in a new interpreter with a different ``PYTHONHASHSEED``, and compares the
result hash (which folds in a digest of every case: event streams, histories,
evaluator classes, ICM values) with the checkpointed one. A target shard that
is not present and passed makes this shard fail: an unverified repeat is never
counted as a pass.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any

TARGETS = {
    "nlhe_eval": [0, 6, 13], "shortdeck_eval": [0, 2], "plo_eval": [0, 9],
    "state_machine": [0, 104, 209], "tournaments": [36], "icm": [0, 27], "metamorphic": [0, 21],
}
HASH_SEEDS = ("1", "987654321")


def shards() -> list[dict[str, Any]]:
    return [{"gate": "determinism", "target_gate": g, "target_index": i, "hash_seed": h}
            for g, idx in TARGETS.items() for i in idx for h in HASH_SEEDS]


_CHILD = """
import json, sys
from tests.release.harness import GATE_BY_ID, shard_refs
ref = shard_refs(GATE_BY_ID[sys.argv[1]])[int(sys.argv[2])]
r = ref.gate.workload().run_shard(ref.params)
print(json.dumps({"result_hash": r["result_hash"], "cases": r["cases"], "failed_cases": r["failed_cases"]}))
"""


def run_shard(params: dict[str, Any]) -> dict[str, Any]:
    from tests.release.harness import GATE_BY_ID, ROOT, shard_refs

    ref = shard_refs(GATE_BY_ID[params["target_gate"]])[params["target_index"]]
    case = {"gate": "determinism", "target": ref.id, "hash_seed": params["hash_seed"]}
    state = ref.state()
    base = {"cases": 1, "counts": {}, "exclusions": {}, "repeat": {"target": ref.id}}
    if state != "passed":
        return {**base, "failed_cases": 1, "result_hash": "",
                "failures": [{"case": case, "error": f"target shard {ref.id} is {state}; run its gate first",
                              "traceback": "", "repro": ""}]}
    stored = ref.load()["result"]
    env = {**os.environ, "PYTHONHASHSEED": params["hash_seed"]}
    proc = subprocess.run([sys.executable, "-c", _CHILD, ref.gate.id, str(ref.index)], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=6 * 3600)
    if proc.returncode != 0:
        return {**base, "failed_cases": 1, "result_hash": "",
                "failures": [{"case": case, "error": f"rerun crashed: {proc.stderr[-1500:]}", "traceback": "",
                              "repro": ""}]}
    got = json.loads(proc.stdout.strip().splitlines()[-1])
    same = got["result_hash"] == stored["result_hash"] and got["cases"] == stored["cases"]
    failures = [] if same else [{
        "case": case, "error": f"result hash {got['result_hash']} ({got['cases']} cases) differs from the "
                               f"checkpointed {stored['result_hash']} ({stored['cases']} cases)",
        "traceback": "", "repro": f"PYTHONHASHSEED={params['hash_seed']} uv run python -m tests.release shard "
                                  f"{ref.gate.id} {ref.index}"}]
    return {**base, "failed_cases": len(failures), "failures": failures, "result_hash": got["result_hash"],
            "counts": {"repeats": 1, "repeated_cases": got["cases"], "identical": int(same)},
            "repeat": {"target": ref.id, "hash_seed": params["hash_seed"], "result_hash": got["result_hash"],
                       "checkpointed_hash": stored["result_hash"], "checkpointed_hash_seed":
                           ref.load().get("python_hash_seed")}}


def finalize(gate: str, results: list[dict[str, Any]], all_results: dict) -> list[dict[str, Any]]:
    gates = sorted({r["params"]["target_gate"] for r in results})
    return [{"check": "every repeated shard reproduced its result hash under a different hash seed",
             "passed": all(r["counts"].get("identical") == 1 for r in results),
             "detail": f"{sum(r['counts'].get('repeated_cases', 0) for r in results):,} cases in "
                       f"{len(results)} repeats across {', '.join(gates)}"}]
