"""Decision-time policies.

* :mod:`strategy.policy.baseline`: the Stage 1 interpretable parameterized policy.
* :mod:`strategy.policy.runtime`: the single deployed path (an explicit-state SDK bot).
* :mod:`strategy.policy.fallback`: the constant-time legal fallback and intent legalization.
* :mod:`strategy.policy.spot`: the slim per-decision features the runtime reads.

Offline helpers (not on the decision path): :mod:`strategy.policy.package` (bundle a
parameter vector into a self-contained bot), :mod:`strategy.policy.bench` and
:mod:`strategy.policy.gate` (latency, legality and the sandboxed deployed-path gate).
"""
