"""Classify deadline timeouts as bot-caused or stall-like, and gate each class.

Why
    On a virtualised host, a no-op bot still misses the 1 ms deadline on about 0.04%
    of decisions: the host deschedules a vCPU for a millisecond or more. A rate
    limit alone cannot separate those stalls from a bot that is too slow, so every
    timeout is classified and the two classes are gated differently.

Classes (:func:`classify`)
    ``stall_like``
        * transport: the late reply reports ``parse + wall`` within the budget minus
          the entrant's median harness overhead (``median e2e - median worker
          wall``). The bot's own code fitted; the time was lost outside it.
        * replay: the timed-out decision, replayed from its exact observation and
          committed state in a fresh worker (explicit-state bots only), has a median
          ``parse + wall`` under ``STALL_REPLAY_FRACTION`` of the budget.
    ``bot_caused``
        * replay median at or above that fraction, or any replay that missed even the
          100 ms replay budget. A bot that needs half the budget is responsible for
          the margin it leaves.
    ``unclassified``
        * no transport evidence and no replay: instance-state bots, timeouts past
          the per-table replay cap, or a replay that could not run. Counted with the
          stall-like ones (never exempted).

Gates (:func:`assess`)
    1. Bot-caused timeouts are strict: more than ``max_bot_timeouts`` (default 0)
       on a strict entrant fails it, with no rerun.
    2. Host ceiling: the interleaved no-op control must not run significantly above
       ``max_control_timeout_rate`` (0.05%): one-sided Poisson test,
       ``P(X >= k | n * rate) < CEILING_ALPHA`` means the host was invalid.
    3. Stall-like plus unclassified timeouts are compared with the control by a
       conditional two-Poisson test (the C-test). Exposure is the time spent
       waiting for replies (sum of end-to-end times), so under H0 the entrant's
       share of the pooled count is Binomial(n, E_entrant / (E_entrant + E_control)).
       A one-sided p-value under ``stall_p_min`` (0.01) is a stall excess. The
       exposure model ignores that a slower decision needs a shorter stall to miss
       the deadline, so it is strict for bots that use much of the budget.

Reruns (:func:`select_attempt`)
    A host-invalid or stall-excess attempt is rerun with the same seeds, up to
    ``max_reruns`` (2) times. Every attempt is reported; the first passing attempt
    is used; a bot-caused failure in any attempt is final.
"""

from __future__ import annotations

import math
import statistics
from array import array
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

CONTROL_POLICY = "baseline:calling_station"
STALL_REPLAY_FRACTION = 0.5
CEILING_ALPHA = 0.01
MAX_CONTROL_TIMEOUT_RATE = 0.0005
STALL_P_MIN = 0.01
MAX_RERUNS = 2
CONTROL_EVERY = 4
TIMEOUT_REPLAYS_PER_SEAT = 3
RECORDS_KEPT = 50

CLASSES = ("bot_caused", "stall_like", "unclassified")


def overhead_ns(e2e: Sequence[int], wall: Sequence[int]) -> int | None:
    """Median parent-observed round trip minus median worker wall time."""
    if not e2e or not wall:
        return None
    return max(0, int(statistics.median(e2e) - statistics.median(wall)))


def _worker_ns(row: Mapping[str, Any]) -> int | None:
    wall = row.get("worker_wall_ns")
    if not isinstance(wall, int):
        return None
    parse = row.get("worker_parse_ns")
    return wall + (parse if isinstance(parse, int) else 0)


def classify(record: Mapping[str, Any], budget_ns: int, overhead: int | None) -> tuple[str, str]:
    """``(class, rule)`` for one timeout record from ``TableResult.timeouts``."""
    own = _worker_ns(record)
    if record.get("outcome") == "late_reply" and own is not None and overhead is not None \
            and own <= budget_ns - overhead:
        return "stall_like", "transport"
    replay = record.get("replay")
    if not replay:
        return "unclassified", "no_replay"
    if "error" in replay:
        return "unclassified", "replay_error"
    runs = replay.get("runs") or []
    if any(r.get("fallback") == "timeout" for r in runs):
        return "bot_caused", "replay_timeout"
    times = [t for t in (_worker_ns(r) for r in runs) if t is not None]
    if not times:
        return "unclassified", "replay_untimed"
    if statistics.median(times) < STALL_REPLAY_FRACTION * budget_ns:
        return "stall_like", "replay_fast"
    return "bot_caused", "replay_slow"


def tally(attempts: Sequence[Any], entrant: int, budget_ns: int) -> dict[str, Any]:
    """Decisions, exposure and classified timeouts of one entrant over runner ``TableResult``s."""
    return tally_samples(
        decisions=sum(r.decisions.get(entrant, 0) for r in attempts),
        timeouts=sum(r.fallbacks.get(entrant, {}).get("timeout", 0) for r in attempts),
        e2e=[x for r in attempts for x in array("q", r.e2e_ns.get(entrant, b""))],
        wall=[x for r in attempts for x in array("q", r.worker_wall_ns.get(entrant, b""))],
        records=[(f"{r.table_id}-a{r.attempt}", rec) for r in attempts for rec in r.timeouts.get(entrant, [])],
        budget_ns=budget_ns)


def tally_samples(*, decisions: int, timeouts: int, e2e: Sequence[int], wall: Sequence[int],
                  records: Sequence[tuple[str, Mapping[str, Any]]], budget_ns: int) -> dict[str, Any]:
    """:func:`tally` from raw samples; ``records`` pairs a table label with each timeout record."""
    over = overhead_ns(e2e, wall)
    classes: Counter = Counter()
    rules: Counter = Counter()
    kept: list[dict[str, Any]] = []
    for table, rec in records:
        cls, rule = classify(rec, budget_ns, over)
        classes[cls] += 1
        rules[rule] += 1
        if len(kept) < RECORDS_KEPT:
            kept.append({"table": table, "class": cls, "rule": rule,
                         **{k: v for k, v in rec.items() if k != "replay"},
                         "replay_worker_ns": [_worker_ns(x) for x in (rec.get("replay") or {}).get("runs", [])]})
    return {
        "decisions": decisions,
        "timeouts": timeouts,
        "timeout_rate": timeouts / decisions if decisions else None,
        "exposure_s": sum(min(x, budget_ns) for x in e2e) / 1e9,
        "overhead_ns": over,
        "classes": {c: classes.get(c, 0) for c in CLASSES},
        "rules": dict(sorted(rules.items())),
        "unrecorded": max(0, timeouts - sum(classes.values())),
        "records": kept,
    }


def merge_tallies(parts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"decisions": 0, "timeouts": 0, "exposure_s": 0.0, "classes": Counter(),
                           "rules": Counter(), "unrecorded": 0, "records": []}
    for p in parts:
        for k in ("decisions", "timeouts", "exposure_s", "unrecorded"):
            out[k] += p[k]
        out["classes"].update(p["classes"])
        out["rules"].update(p["rules"])
        out["records"] += p["records"][: max(0, RECORDS_KEPT - len(out["records"]))]
    out["classes"] = {c: out["classes"].get(c, 0) for c in CLASSES}
    out["rules"] = dict(sorted(out["rules"].items()))
    out["timeout_rate"] = out["timeouts"] / out["decisions"] if out["decisions"] else None
    return out


def _log_binom_pmf(k: int, n: int, p: float) -> float:
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            + k * math.log(p) + (n - k) * math.log1p(-p))


def binomial_upper_tail(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p)."""
    if k <= 0:
        return 1.0
    if k > n or p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    return min(1.0, sum(math.exp(_log_binom_pmf(i, n, p)) for i in range(k, n + 1)))


def poisson_upper_tail(k: int, mu: float) -> float:
    """P(X >= k) for X ~ Poisson(mu)."""
    if k <= 0:
        return 1.0
    if mu <= 0.0:
        return 0.0
    below = sum(math.exp(i * math.log(mu) - mu - math.lgamma(i + 1)) for i in range(k))
    return max(0.0, 1.0 - below)


def excess_p_value(k_entrant: int, exposure_entrant: float, k_control: int, exposure_control: float) -> float | None:
    """One-sided C-test p-value that the entrant's timeout rate per unit exposure
    exceeds the control's; None without exposure on both sides."""
    if exposure_entrant <= 0 or exposure_control <= 0:
        return None
    share = exposure_entrant / (exposure_entrant + exposure_control)
    return binomial_upper_tail(k_entrant, k_entrant + k_control, share)


def assess(entrant: Mapping[str, Any], control: Mapping[str, Any] | None, *, strict: bool,
           max_bot_timeouts: int | None = 0, max_control_timeout_rate: float | None = MAX_CONTROL_TIMEOUT_RATE,
           stall_p_min: float = STALL_P_MIN) -> dict[str, Any]:
    """Verdict for one entrant tally against the control tally of the same run:
    ``fail`` (bot-caused, strict entrants only), ``host_invalid``, ``stall_excess`` or ``pass``."""
    bot = entrant["classes"]["bot_caused"]
    stall = entrant["classes"]["stall_like"] + entrant["classes"]["unclassified"] + entrant.get("unrecorded", 0)
    out: dict[str, Any] = {"strict": strict, "bot_caused": bot, "stall_like_or_unclassified": stall,
                           "max_bot_timeouts": max_bot_timeouts, "max_control_timeout_rate": max_control_timeout_rate,
                           "stall_p_min": stall_p_min, "ceiling_p_value": None, "excess_p_value": None}
    if strict and max_bot_timeouts is not None and bot > max_bot_timeouts:
        return {**out, "verdict": "fail",
                "reason": f"{bot} bot-caused timeouts (limit {max_bot_timeouts}): the bot's own decision time, "
                          "measured by replay, used at least half the budget"}
    if control is None:
        return {**out, "verdict": "pass" if stall == 0 else "stall_excess",
                "reason": "no control run" if stall == 0 else f"{stall} stall-like timeouts and no control to test them"}
    kc, nc = control["timeouts"], control["decisions"]
    if max_control_timeout_rate is not None:
        p_ceiling = poisson_upper_tail(kc, nc * max_control_timeout_rate)
        out["ceiling_p_value"] = p_ceiling
        if p_ceiling < CEILING_ALPHA:
            return {**out, "verdict": "host_invalid",
                    "reason": f"control timed out on {kc}/{nc} decisions, significantly above the "
                              f"{max_control_timeout_rate:.3%} host ceiling (p={p_ceiling:.2g})"}
    p_excess = excess_p_value(stall, entrant["exposure_s"], kc, control["exposure_s"])
    out["excess_p_value"] = p_excess
    if stall and (p_excess is None or p_excess < stall_p_min):
        return {**out, "verdict": "stall_excess",
                "reason": f"{stall} stall-like or unclassified timeouts against {kc} in the control "
                          f"(exposure {entrant['exposure_s']:.3g} s vs {control['exposure_s']:.3g} s, "
                          f"p={'n/a' if p_excess is None else f'{p_excess:.2g}'} < {stall_p_min})"}
    return {**out, "verdict": "pass", "reason": "within the host-stall rate of the interleaved control"}


def needs_rerun(assessment: Mapping[str, Any]) -> bool:
    return assessment["verdict"] in ("host_invalid", "stall_excess")


def select_attempt(assessments: Sequence[Mapping[str, Any]], max_reruns: int = MAX_RERUNS) -> dict[str, Any]:
    """Apply the rerun policy to the attempts so far (oldest first).

    ``action`` is ``use`` (with ``used``, the attempt index), ``rerun``, ``fail`` or
    ``invalid`` (reruns exhausted)."""
    for i, a in enumerate(assessments):
        if a["verdict"] == "fail":
            return {"action": "fail", "used": i, "reason": a["reason"]}
    for i, a in enumerate(assessments):
        if a["verdict"] == "pass":
            return {"action": "use", "used": i, "reason": a["reason"]}
    if len(assessments) <= max_reruns:
        return {"action": "rerun", "used": None, "reason": assessments[-1]["reason"] if assessments else ""}
    return {"action": "invalid", "used": None,
            "reason": f"{len(assessments)} attempts, none passed: {assessments[-1]['reason']}"}


__all__ = ["CLASSES", "CONTROL_POLICY", "STALL_REPLAY_FRACTION", "assess", "binomial_upper_tail", "classify",
           "excess_p_value", "merge_tallies", "needs_rerun", "overhead_ns", "poisson_upper_tail", "select_attempt",
           "tally", "tally_samples"]
