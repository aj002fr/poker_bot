"""Offline diagnostic: how often do our bots choose what Pluribus chose in the same spot?

Each bot is seated, in-process, at every Pluribus decision of every imported hand and
asked to act on the frozen observation Pluribus saw (one fresh bot per hand, its
explicit state chained across that hand's Pluribus decisions). The hand then follows
Pluribus's real line, whatever the bot chose, so later spots are always Pluribus's.

Reported per bot, as the mean over ``--replicates`` independent policy seeds:

* action agreement: the bot's action kind (fold, check, call, bet/raise) equals
  Pluribus's, overall, by street and by Pluribus's kind (a confusion matrix);
* balanced agreement: the mean over Pluribus's four kinds of the share the bot matched,
  so a bot that always folds cannot score well just because most spots are folds;
* size agreement: when both raise, the two amounts map to the same canonical size
  option (:func:`strategy.data.phh_import.map_size`).

This is a diagnostic, not a training target or a strength measure. Pluribus sampled
one action from a mixed strategy, so a single recorded action is not "the" right
answer, and it played six-max 100 BB cash, not the eight-player tournament points our
bots are tuned for.

    python -m strategy.data.agreement                     # the cached corpus, all bots
    python -m strategy.data.agreement --source fixture --replicates 2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from arena.agents.baselines import BASELINES
from arena.agents.sdk import Action, TableInfo, load_bot, split_note
from arena.agents.styles import STYLES
from strategy.data.fetch import FetchError
from strategy.data.phh_import import (ImportedHand, ImportRejected, category_of, import_hand, iter_source,
                                      map_size, provenance_for, source_dir)

HERO = "Pluribus"
REPO = Path(__file__).resolve().parents[2]
BASELINE_VECTOR = REPO / "strategy" / "policy" / "vectors" / "nlhe-v2.json"
KINDS = ("fold", "check", "call", "raise")
STREETS = ("preflop", "flop", "turn", "river")
DEFAULT_BOTS = ("baseline:nlhe-v2", "baseline:control", "arena:nlhe_heuristic", "style:rock", "style:aggressive",
                "style:super_aggressive", "style:station", "style:weak_tight", "arena:calling_station",
                "arena:check_fold", "arena:legal_random")


@cache
def factory(spec: str) -> Callable[[], Any]:
    """``baseline:nlhe-v2`` / ``baseline:control`` (strategy.policy), ``arena:<baseline>`` or ``style:<style>``."""
    kind, _, name = spec.partition(":")
    if kind == "baseline":
        from strategy.policy.runtime import RuntimeBot

        if name == "control":
            return lambda: RuntimeBot(None, name="baseline_control")
        if name == "nlhe-v2":
            params = json.loads(BASELINE_VECTOR.read_text())["params"]
            return lambda: RuntimeBot(params, name="baseline_nlhe_v2")
        raise ValueError(f"unknown baseline vector {name!r}")
    paths = {"arena": BASELINES, "style": STYLES}.get(kind)
    if paths is None or name not in paths:
        raise ValueError(f"unknown bot {spec!r}")
    module = "agreement_" + hashlib.sha256(spec.encode()).hexdigest()[:12]
    load_bot(paths[name], module_name=module)
    return sys.modules[module].make_bot


def table_info(hand: ImportedHand, seat: int, seed: str) -> TableInfo:
    cfg = hand.config
    return TableInfo(table_id=hand.table.table_id, hero_seat=seat, ruleset_id=cfg.ruleset_id, config=cfg.resolved(),
                     config_hash=cfg.config_hash(), seats=list(cfg.resolved_seats),
                     decision_budget_ns=cfg.bot_budget_ns, policy_seed=seed)


def _as_action(raw: Any) -> Action | None:
    payload, _ = split_note(raw.model_dump(exclude_none=True) if isinstance(raw, Action) else raw)
    try:
        return Action.model_validate(payload)
    except Exception:  # noqa: BLE001 - an unparseable reply counts as illegal
        return None


@dataclass
class Tally:
    decisions: float = 0.0
    agree: float = 0.0
    illegal: float = 0.0
    both_raise: float = 0.0
    size_agree: float = 0.0
    errors: float = 0.0

    def add(self, other: Tally) -> None:
        self.decisions += other.decisions
        self.agree += other.agree
        self.illegal += other.illegal
        self.errors += other.errors
        self.both_raise += other.both_raise
        self.size_agree += other.size_agree


def evaluate_hand(hand: ImportedHand, bots: Sequence[str], replicates: int) -> dict[str, Any]:
    """Tallies for one hand: ``{bot: {"all": Tally, street: Tally, "confusion": Counter}}``."""
    hero = hand.seat_of(HERO)
    decisions = [d for d in hand.decisions if d.seat == hero]
    out: dict[str, Any] = {}
    for spec in bots:
        make = factory(spec)
        tallies: dict[str, Any] = defaultdict(Tally)
        confusion: Counter = Counter()
        for r in range(replicates):
            seed = hashlib.sha256(f"agreement-v1|{spec}|{hand.hand_id}|{r}".encode()).hexdigest()[:32]
            bot = make()
            info = table_info(hand, hero, seed)
            state = bot.initial_state(info) if bot.state_mode == "explicit" else bot.start_table(info)
            w = 1.0 / replicates
            for d in decisions:
                obs = d.observation
                t = Tally(decisions=w)
                try:
                    if bot.state_mode == "explicit":
                        raw, state = bot.decide(obs, state)
                    else:
                        raw = bot.decide(obs)
                    action = _as_action(raw)
                except Exception:  # noqa: BLE001 - a crash counts as illegal, as a worker fault would
                    action = None
                    t.errors = w
                if action is None or not obs.legal.allows(action):
                    t.illegal = w
                    kind = "illegal"
                else:
                    kind = category_of(action)
                    t.agree = w if kind == d.category else 0.0
                    if kind == "raise" and d.size is not None:
                        assert action.amount is not None
                        t.both_raise = w
                        t.size_agree = w if map_size(action.amount, obs, d.menu).option.key == \
                            d.size.option.key else 0.0
                tallies["all"].add(t)
                tallies[d.street].add(t)
                confusion[(d.category, kind)] += w
        out[spec] = {"tallies": dict(tallies), "confusion": confusion}
    return out


def _work(args: tuple[str, list[tuple[str, bytes]], tuple[str, ...], int]) -> tuple[dict[str, Any], int, int]:
    _, chunk, bots, replicates = args
    merged: dict[str, Any] = {}
    hands = rejected = 0
    for rel, data in chunk:
        try:
            hand = import_hand(data.decode("utf-8"), provenance_for(rel, data))
        except ImportRejected:
            rejected += 1
            continue
        if hand.seat_of(HERO) is None:
            continue
        hands += 1
        for spec, res in evaluate_hand(hand, bots, replicates).items():
            m = merged.setdefault(spec, {"tallies": defaultdict(Tally), "confusion": Counter()})
            for key, t in res["tallies"].items():
                m["tallies"][key].add(t)
            m["confusion"].update(res["confusion"])
    return merged, hands, rejected


def _rate(num: float, den: float) -> float | None:
    return round(num / den, 4) if den else None


def run(root: Path, bots: Sequence[str], *, replicates: int = 4, jobs: int = 1, limit: int | None = None,
        label: str | None = None) -> dict[str, Any]:
    texts = list(iter_source(root))[:limit]
    size = max(1, len(texts) // max(1, jobs * 8))
    chunks = [(str(i), texts[i:i + size], tuple(bots), replicates) for i in range(0, len(texts), size)]
    merged: dict[str, Any] = {}
    hands = rejected = 0
    if jobs > 1:
        with ProcessPoolExecutor(jobs) as pool:
            results = list(pool.map(_work, chunks))
    else:
        results = [_work(c) for c in chunks]
    for part, h, rj in results:
        hands += h
        rejected += rj
        for spec, res in part.items():
            m = merged.setdefault(spec, {"tallies": defaultdict(Tally), "confusion": Counter()})
            for key, t in res["tallies"].items():
                m["tallies"][key].add(t)
            m["confusion"].update(res["confusion"])
    rows = []
    for spec in bots:
        m = merged.get(spec)
        if m is None:
            continue
        a = m["tallies"]["all"]
        conf = m["confusion"]
        per_kind = {k: _rate(conf[(k, k)], sum(v for (p, _), v in conf.items() if p == k)) for k in KINDS}
        seen = [v for v in per_kind.values() if v is not None]
        rows.append({
            "bot": spec,
            "decisions": round(a.decisions),
            "action_agreement": _rate(a.agree, a.decisions),
            "balanced_agreement": round(sum(seen) / len(seen), 4) if seen else None,
            "size_agreement": _rate(a.size_agree, a.both_raise),
            "both_raise": round(a.both_raise, 2),
            "illegal": _rate(a.illegal, a.decisions),
            "errors": round(a.errors, 2),
            "by_street": {s: _rate(m["tallies"][s].agree, m["tallies"][s].decisions) for s in STREETS
                          if s in m["tallies"]},
            "by_pluribus_kind": per_kind,
            "confusion": {p: {b: round(conf[(p, b)], 2) for b in (*KINDS, "illegal") if conf[(p, b)]}
                          for p in KINDS},
        })
    return {"source": label or str(root), "hands": hands, "rejected": rejected, "replicates": replicates,
            "hero": HERO, "bots": rows,
            "caveat": "Diagnostic only: one sampled Pluribus action per spot, six-max 100 BB cash, teacher-forced "
                      "lines. Not a training target or a strength measure."}


def markdown(report: dict[str, Any]) -> str:
    lines = [f"{report['hands']} hands, {report['bots'][0]['decisions'] if report['bots'] else 0} {HERO} "
             f"decisions, {report['replicates']} seeds per bot.", "",
             "| Bot | Action agreement | Balanced | Preflop | Flop | Turn | River | Size agreement (both raise) | Illegal |",
             "|---|---|---|---|---|---|---|---|---|"]

    def pct(x: float | None) -> str:
        return "–" if x is None else f"{100 * x:.1f}%"

    for r in report["bots"]:
        st = r["by_street"]
        lines.append(f"| {r['bot']} | {pct(r['action_agreement'])} | {pct(r['balanced_agreement'])} | {pct(st.get('preflop'))} | "
                     f"{pct(st.get('flop'))} | {pct(st.get('turn'))} | {pct(st.get('river'))} | "
                     f"{pct(r['size_agreement'])} ({r['both_raise']:.0f}) | {pct(r['illegal'])} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="How often bots agree with Pluribus's recorded actions (diagnostic).")
    ap.add_argument("--source", default="cache", help="cache (default), fixture, or a directory of .phh files")
    ap.add_argument("--bots", nargs="+", default=list(DEFAULT_BOTS))
    ap.add_argument("--replicates", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None, help="write the report as JSON")
    args = ap.parse_args(argv)
    try:
        root = source_dir(args.source)
    except FetchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    report = run(root, args.bots, replicates=args.replicates, jobs=args.jobs, limit=args.limit, label=args.source)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
