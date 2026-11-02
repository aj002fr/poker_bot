"""Population response curves for push/fold spots, fitted on the train partition only (offline).

Data: complete tournaments of training-partition lineups of the opponent registry (our
point-aware bot in one seat, seven members in the rest). We run the tables, so every
preflop decision a member takes while the hand still fits the push/fold abstraction is
recorded with its hole cards:

``open``      everyone before folded: shove (the all-in raise) or anything else;
``call``      facing exactly one all-in: commit (call or re-shove) or fold;
``overcall``  facing an all-in and a call: commit or fold.

In the abstraction a call commits the caller's whole stack, so a re-shove over a shove
is a commit. A non-shove open (limp, small raise) leaves the abstraction; in the solved
tree it is a fold, the only other action there.

Context features (shared with the solver through :func:`context`): log chips at risk in
big blinds (capped at 100), the ICM pressure of the confrontation
(:func:`~strategy.endgame.keys.pressure`), the share of players still behind, whether the
actor is covered, small and big blind flags, dead money over the risk, and the live count.
Hand strength is the class's quantile ``u`` by equity against a random hand (0 weakest).

Three forms per kind, chosen by held-out log loss on train-partition tournaments split
by lineup:

``base``       a constant rate;
``threshold``  a logistic frequency on the context, spread over hands by the soft
               threshold range model of :mod:`strategy.beliefs.ranges` (the strongest
               part of the range commits, with its bluff share);
``logistic``   a logistic in context and strength with strength interactions;
``classes``    the same plus one offset per hand class, shrunk towards the smooth curve.

    python -m strategy.endgame.response collect --ruleset practice_nlhe --jobs 8
    python -m strategy.endgame.response fit --ruleset practice_nlhe
"""

from __future__ import annotations

import argparse
import math
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np

from strategy.endgame.keys import MAX_DEPTH_BB, Snapshot, pressure
from strategy.tables import store

GENERATOR = "strategy.endgame.response"
INPUTS = ["population/registry.lock.json", "endgame/classes.py", "endgame/keys.py", "endgame/response.py",
          "endgame/table_state.py", "beliefs/ranges.py", "value/rollouts.py", "rules/variants.py"]
KINDS = ("open", "call", "overcall")
FORMS = ("base", "threshold", "logistic", "classes")
CONTEXT = ("log_risk_bb", "pressure", "behind_share", "covered", "small_blind", "big_blind", "dead_ratio", "live")
STRENGTH = ("u", "u2", "u3")
RISK_CAP_BB = 100.0
COLLECT_MAX_BB = 60.0
"""Decisions with more than this many big blinds at risk are not recorded (tables stop at 30)."""
OUR_BOT = "strategy/endgame/bots/points.py"
DEFAULTS = {"lineups": 200, "blocks": 1, "seed": 11}
TEST_SHARE = 0.3
RIDGE = 1.0
CLASS_RIDGE = 1.0
DATA_DIR = Path("artifacts/endgame")


def table_name(ruleset_id: str) -> str:
    return f"response-{ruleset_id}"


def data_path(ruleset_id: str) -> Path:
    return DATA_DIR / f"responses-{ruleset_id}.npz"


# ----------------------------------------------------------------- features


def villain_of(snap: Snapshot, kind: str, actor: int, shover: int | None, caller: int | None) -> int:
    st = snap.stacks
    if kind == "open":
        return max(range(actor + 1, snap.n), key=lambda p: (st[p], -p))
    if kind == "call":
        assert shover is not None
        return shover
    assert shover is not None and caller is not None
    return max((shover, caller), key=lambda p: st[p])


def context(snap: Snapshot, kind: str, actor: int, shover: int | None = None, caller: int | None = None) -> tuple:
    st = snap.stacks
    v = villain_of(snap, kind, actor, shover, caller)
    risk = min(st[actor], st[v])
    small = float(actor != snap.n - 1 and snap.posts[actor] > 0)
    return (math.log(min(max(risk / snap.bb, 0.5), RISK_CAP_BB)), pressure(st, actor, v, snap.prizes),
            (snap.n - 1 - actor) / max(1, snap.n - 1), float(st[actor] <= st[v]), small,
            float(actor == snap.n - 1), sum(snap.posts) / max(1, risk), (snap.n - 2) / 6.0)


def risk_bb(snap: Snapshot, kind: str, actor: int, shover: int | None, caller: int | None) -> float:
    v = villain_of(snap, kind, actor, shover, caller)
    return min(snap.stacks[actor], snap.stacks[v]) / snap.bb


def strength_quantiles(model) -> np.ndarray:
    """``u[i]``: combo share of classes weaker than class i plus half its own, by equity vs random."""
    eq, prior = np.asarray(model.eq_random, float), np.asarray(model.prior, float)
    weaker = np.array([prior[eq < e].sum() for e in eq])
    return weaker + prior / 2


# ------------------------------------------------------------------- curves


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _ctx_matrix(ctx: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((len(ctx), 1)), ctx])


def design(form: str, ctx: np.ndarray, cls: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Design rows of a logistic form (``threshold`` models the frequency only)."""
    x = _ctx_matrix(ctx)
    if form == "threshold":
        return x
    ur = u[cls]
    smooth = np.hstack([x, x * ur[:, None], (ur ** 2)[:, None], (ur ** 3)[:, None]])
    if form == "logistic":
        return smooth
    onehot = np.zeros((len(cls), len(u)))
    onehot[np.arange(len(cls)), cls] = 1.0
    return np.hstack([smooth, onehot])


def penalties(form: str, width: int, classes: int) -> np.ndarray:
    """Ridge weights: none on the intercept, light on the smooth terms, more on class offsets."""
    pen = np.full(width, RIDGE)
    pen[0] = 0.0
    if form == "classes":
        pen[width - classes:] = CLASS_RIDGE
    return pen


def fit_logistic(xfun, n: int, y: np.ndarray, pen: np.ndarray, iters: int = 50, chunk: int = 50_000) -> np.ndarray:
    """Penalised logistic regression by Newton steps; ``xfun(slice)`` builds design rows in chunks."""
    beta = np.zeros(len(pen))
    for _ in range(iters):
        grad = pen * beta
        hess = np.diag(pen + 1e-9)
        for lo in range(0, n, chunk):
            sl = slice(lo, min(n, lo + chunk))
            x = xfun(sl)
            p = _sigmoid(x @ beta)
            grad = grad + x.T @ (p - y[sl])
            hess = hess + (x * (p * (1 - p) + 1e-9)[:, None]).T @ x
        step = np.linalg.solve(hess, grad)
        beta -= step
        if np.abs(step).max() < 1e-7:
            break
    return beta


def threshold_rows(freq: np.ndarray, u: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """Per-class commit probability of a player committing with frequency ``freq``, for each row.

    The class shape is :func:`strategy.beliefs.ranges.threshold_likelihood` for the aggressive
    category over the prior range ordered weakest first.
    """
    from strategy.beliefs.ranges import threshold_likelihood

    order = np.argsort(u, kind="stable")
    w = prior[order].tolist()
    out = np.empty((len(freq), len(u)))
    for r, f in enumerate(freq):
        lik = threshold_likelihood(w, (1.0 - f, 0.0, f), 2, street=0)
        row = np.empty(len(u))
        row[order] = lik
        out[r] = row
    return np.clip(out, 1e-4, 1 - 1e-4)


class Curve:
    def __init__(self, kind: str, form: str, coef: np.ndarray, u: np.ndarray, prior: np.ndarray) -> None:
        self.kind, self.form, self.coef, self.u, self.prior = kind, form, np.asarray(coef, float), u, prior

    def probs(self, ctx: np.ndarray, cls: np.ndarray) -> np.ndarray:
        """Commit probability for rows of context with the given class indices."""
        ctx = np.atleast_2d(ctx)
        if self.form == "base":
            return np.full(len(ctx), float(self.coef[0]))
        if self.form == "threshold":
            freq = _sigmoid(_ctx_matrix(ctx) @ self.coef)
            return threshold_rows(freq, self.u, self.prior)[np.arange(len(ctx)), cls]
        out = np.empty(len(ctx))
        for lo in range(0, len(ctx), 50_000):
            sl = slice(lo, lo + 50_000)
            out[sl] = _sigmoid(design(self.form, ctx[sl], cls[sl], self.u) @ self.coef)
        return out

    def row(self, ctx: tuple) -> np.ndarray:
        """Commit probability of every class in one context."""
        c = np.asarray(ctx, float)[None, :]
        h = len(self.u)
        if self.form == "base":
            return np.full(h, float(self.coef[0]))
        if self.form == "threshold":
            return threshold_rows(_sigmoid(_ctx_matrix(c) @ self.coef), self.u, self.prior)[0]
        return _sigmoid(design(self.form, np.repeat(c, h, 0), np.arange(h), self.u) @ self.coef)


def fit_form(form: str, ctx: np.ndarray, cls: np.ndarray, y: np.ndarray, u: np.ndarray) -> np.ndarray:
    if form == "base":
        return np.array([y.mean()])
    width = design(form, ctx[:1], cls[:1], u).shape[1]
    return fit_logistic(lambda sl: design(form, ctx[sl], cls[sl], u), len(y), y, penalties(form, width, len(u)))


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


class ResponseModel:
    """The fitted curves of one variant, checked against its hand classes."""

    def __init__(self, provenance: dict, arrays: dict) -> None:
        from strategy.endgame.classes import hand_classes

        ruleset_id = provenance["ruleset_id"]
        hc = hand_classes(ruleset_id)
        if provenance.get("variant_hash") != hc.variant_hash or tuple(arrays["class_keys"].tolist()) != hc.keys:
            raise store.TableError(f"{provenance['name']}: built for different hand classes than {ruleset_id}")
        self.provenance = provenance
        self.ruleset_id = ruleset_id
        u, prior = arrays["u"], arrays["prior"]
        self.curves = {k: Curve(k, provenance["forms"][k], arrays[f"{k}_coef"], u, prior) for k in KINDS}

    def row(self, snap: Snapshot, kind: str, actor: int, shover: int | None = None,
            caller: int | None = None) -> np.ndarray:
        return self.curves[kind].row(context(snap, kind, actor, shover, caller))


@lru_cache(maxsize=None)
def load_response(ruleset_id: str, directory: str | None = None) -> ResponseModel:
    loaded = store.load_arrays(table_name(ruleset_id), directory)
    if loaded is None:
        raise FileNotFoundError(f"{table_name(ruleset_id)}.npz not found; run `make tables-response`")
    return ResponseModel(*loaded)


def population_profile(solver, snap: Snapshot, model: ResponseModel) -> dict[int, np.ndarray]:
    """Every decision node of a push/fold subgame played by the curves: ``(2, H)`` fold/commit."""
    from strategy.endgame.harvest import decision_nodes

    out = {}
    for kind, node, actor, shover, caller in decision_nodes(solver.tree):
        p = model.row(snap, kind, actor, shover, caller)
        out[node] = np.stack([1.0 - p, p])
    if set(out) != set(solver.decisions):
        raise AssertionError("decision_nodes does not cover the tree")
    return out


# ---------------------------------------------------------------- decisions


def hand_decisions(snap: Snapshot, seats, events, members) -> list[tuple]:
    """``(kind, actor, shover, caller, seat, commit)`` for members' in-abstraction preflop decisions.

    Mirrors :func:`strategy.endgame.lookup.preflop_spot`: folds keep the hand in the
    abstraction, the first all-in raise is the shover, the next call or raise the caller,
    and anything else ends it after being recorded.
    """
    index = {s: i for i, s in enumerate(seats)}
    committed = list(snap.posts)
    shover = caller = None
    out = []
    for e in events:
        if e.type != "ActionApplied" or e.data.get("street") != "preflop":
            continue
        i = index.get(e.data["seat"])
        if i is None:
            break
        act = e.data["action"]["type"]
        committed[i] += int(e.data.get("added") or 0)
        all_in = committed[i] >= snap.stacks[i]
        kind = "open" if shover is None else "call" if caller is None else "overcall"
        if kind == "open":
            commit = int(act == "raise_to" and all_in)
        else:
            commit = int(act in ("call", "raise_to"))
        if e.data["seat"] in members and not (kind == "open" and i == snap.n - 1):
            out.append((kind, i, shover, caller, e.data["seat"], commit))
        if act == "fold":
            continue
        if kind == "open" and commit:
            shover = i
            continue
        if kind == "call":
            caller = i
            continue
        break
    return out


def _collect(a) -> list[tuple]:
    from arena.agents.worker import play_table
    from arena.benchmark.runner import table_info
    from arena.table.service import seeded_table
    from strategy.endgame.classes import hand_classes
    from strategy.endgame.table_state import seated_snapshot
    from strategy.features.cards import card_ints
    from strategy.value import rollouts

    runner = rollouts._RUNNER
    assert runner is not None
    config = runner.config
    hc = hand_classes(config.ruleset_id)
    agents = {}
    for seat, entrant in a.seats.items():
        agent = runner._agent(seat, entrant, table_info(config, a, seat, runner.budget_ns))
        agent.start()
        agents[seat] = agent
    table = seeded_table(config, a.seed, table_id=a.table_id)
    try:
        play_table(table, agents)
    finally:
        for agent in agents.values():
            agent.close()
    members = {s for s, e in a.seats.items() if rollouts._POLICIES[e] != OUR_BOT}
    rows = []
    for rec in table.hands:
        pos = rec.positions
        live = set(rec.start_stacks)
        seated = seated_snapshot(ring=tuple(config.resolved_seats), start_stacks=dict(rec.start_stacks),
                                 button=pos.button, small_blind_seat=pos.small_blind if pos.small_blind in live else None,
                                 big_blind=pos.big_blind, heads_up=pos.heads_up, blinds=tuple(rec.blinds),
                                 payouts=tuple(config.resolved_payouts), hand_number=rec.hand_number,
                                 hand_cap=config.hand_cap)
        snap = seated.snapshot
        holes: dict[int, list[str]] = {}
        for e in rec.events:
            if e.type == "HoleDealt":
                holes.setdefault(e.data["seat"], []).append(e.data["card"])
        for kind, actor, shover, caller, seat, commit in hand_decisions(snap, seated.seats, rec.events, members):
            if risk_bb(snap, kind, actor, shover, caller) > COLLECT_MAX_BB:
                continue
            cls = hc.index(card_ints(holes[seat]))
            rows.append((KINDS.index(kind), *context(snap, kind, actor, shover, caller), cls, commit,
                         rec.hand_number))
    return rows


def collect(ruleset_id: str, *, lineups: int, blocks: int, seed: int, jobs: int) -> dict[str, np.ndarray]:
    from arena.benchmark.runner import schedule
    from arena.domain.models import TournamentConfig
    from strategy.value import rollouts

    config = TournamentConfig(ruleset_id=ruleset_id)
    teams = rollouts.training_lineups(ruleset_id, config.initial_player_count, lineups, seed, fixed=(OUR_BOT,))
    kinds, ctx, cls, y, hand, group = [], [], [], [], [], []
    for li, team in enumerate(teams):
        assignments = schedule(config, len(team), blocks, f"responses-{seed}|lineup{li}")
        if jobs > 1:
            with ProcessPoolExecutor(jobs, initializer=rollouts._init,
                                     initargs=(config.model_dump_json(), list(team))) as ex:
                parts = list(ex.map(_collect, assignments, chunksize=1))
        else:
            rollouts._init(config.model_dump_json(), list(team))
            parts = [_collect(a) for a in assignments]
        for rows in parts:
            for r in rows:
                kinds.append(r[0])
                ctx.append(r[1:1 + len(CONTEXT)])
                cls.append(r[1 + len(CONTEXT)])
                y.append(r[2 + len(CONTEXT)])
                hand.append(r[3 + len(CONTEXT)])
                group.append(li)
    return {"kind": np.array(kinds, np.int8), "ctx": np.array(ctx, np.float64).reshape(-1, len(CONTEXT)),
            "cls": np.array(cls, np.int16), "commit": np.array(y, np.int8), "hand": np.array(hand, np.int16),
            "lineup": np.array(group, np.int16)}


# --------------------------------------------------------------------- fit


def fit(ruleset_id: str, data: dict[str, np.ndarray], *, test_share: float = TEST_SHARE, seed: int = 1) -> tuple:
    """Choose each kind's form by held-out log loss (lineups split), then refit it on all data."""
    from strategy.endgame.showdown import load_showdown

    model = load_showdown(ruleset_id)
    u, prior = strength_quantiles(model), np.asarray(model.prior, float)
    rng = np.random.default_rng(seed)
    groups = np.unique(data["lineup"])
    test_groups = set(rng.choice(groups, max(1, int(round(test_share * len(groups)))), replace=False).tolist())
    is_test = np.isin(data["lineup"], list(test_groups))
    arrays: dict[str, np.ndarray] = {"u": u, "prior": prior}
    forms, diag = {}, {}
    for k, kind in enumerate(KINDS):
        m = data["kind"] == k
        ctx, cls, y = data["ctx"][m], data["cls"][m].astype(int), data["commit"][m].astype(float)
        tr, te = ~is_test[m], is_test[m]
        losses, calib = {}, {}
        for form in FORMS:
            coef = fit_form(form, ctx[tr], cls[tr], y[tr], u)
            p = Curve(kind, form, coef, u, prior).probs(ctx[te], cls[te])
            losses[form] = log_loss(p, y[te])
            calib[form] = calibration(p, y[te])
        best = min(FORMS, key=lambda f: losses[f])
        forms[kind] = best
        arrays[f"{kind}_coef"] = fit_form(best, ctx, cls, y, u)
        diag[kind] = {"decisions": int(m.sum()), "test_decisions": int(te.sum()), "commit_rate": float(y.mean()),
                      "held_out_log_loss": losses, "form": best, "calibration": calib[best]}
    return arrays, forms, diag


def calibration(p: np.ndarray, y: np.ndarray, bins: int = 10) -> list[list[float]]:
    """``[mean predicted, observed, count]`` per predicted-probability decile."""
    if len(p) == 0:
        return []
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    return [[float(p[idx == b].mean()), float(y[idx == b].mean()), int((idx == b).sum())]
            for b in range(bins) if (idx == b).any()]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--ruleset", required=True)
    for k, v in DEFAULTS.items():
        c.add_argument("--" + k, type=int, default=v)
    c.add_argument("--jobs", type=int, default=8)
    f = sub.add_parser("fit")
    f.add_argument("--ruleset", required=True)
    f.add_argument("--out", help="output directory (default: strategy/tables/data)")
    a = ap.parse_args(argv)
    if a.cmd == "collect":
        t0 = time.perf_counter()
        data = collect(a.ruleset, lineups=a.lineups, blocks=a.blocks, seed=a.seed, jobs=a.jobs)
        path = data_path(a.ruleset)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, params=np.array([a.lineups, a.blocks, a.seed]), **data)
        print(f"wrote {path}: {len(data['commit'])} decisions "
              f"({', '.join(f'{k} {(data['kind'] == i).sum()}' for i, k in enumerate(KINDS))}), "
              f"{time.perf_counter() - t0:.0f}s")
        return
    from strategy.endgame.classes import hand_classes

    t0 = time.perf_counter()
    raw = np.load(data_path(a.ruleset))
    data = {k: raw[k] for k in raw.files if k != "params"}
    lineups, blocks, seed = (int(x) for x in raw["params"])
    arrays, forms, diag = fit(a.ruleset, data)
    hc = hand_classes(a.ruleset)
    arrays["class_keys"] = np.array(hc.keys, np.int64)
    showdown_prov = store.load_arrays(f"showdown-{a.ruleset}")[0]
    extra = {"ruleset_id": a.ruleset, "variant_hash": hc.variant_hash, "forms": forms, "diagnostics": diag,
             "context": list(CONTEXT), "partition": "train", "showdown_sha256": showdown_prov["data_sha256"],
             "decisions_sha256": store.arrays_sha256(data), "collect_max_bb": COLLECT_MAX_BB,
             "max_depth_bb": MAX_DEPTH_BB}
    path = store.write_arrays(table_name(a.ruleset), arrays, generator=GENERATOR, inputs=INPUTS,
                              params={"ruleset_id": a.ruleset, "lineups": lineups, "blocks": blocks, "seed": seed,
                                      "test_share": TEST_SHARE, "ridge": RIDGE, "class_ridge": CLASS_RIDGE},
                              seconds=time.perf_counter() - t0, extra=extra,
                              directory=Path(a.out) if a.out else None)
    print(f"wrote {path}")
    for kind, d in diag.items():
        ll = ", ".join(f"{f} {v:.4f}" for f, v in d["held_out_log_loss"].items())
        print(f"  {kind}: {d['decisions']} decisions, commit rate {d['commit_rate']:.3f}, held-out log loss {ll}"
              f" -> {d['form']}")


if __name__ == "__main__":
    main()
