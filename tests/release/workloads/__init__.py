"""Gate workloads. Each module exposes ``run_shard(params)`` and ``run_case(case)``.

A shard result is a JSON-compatible dict with ``counts``, ``failures`` (each
carrying the exact ``case`` that reproduces it), ``exclusions`` and a
``result_hash`` that is identical whenever the same shard is rerun on the same
code. Workload modules are part of the gate fingerprint; orchestration code
(``tests/release/harness.py``, ``report.py``) is not, so changing the report
does not invalidate finished shards.
"""
