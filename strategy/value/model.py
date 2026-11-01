"""Continuation value V(s) of a start-of-hand state: ICM plus a residual fitted to complete rollouts.

``V_i(s) = ICM_i(s) + r_i(s)``, where the residual is linear in a few features of each
live player (blind pressure, hands until it posts the big blind, chip share, the cap
horizon) with one weight vector per live-player count. Features are centred across
the live players, so residuals sum to zero and V conserves the remaining prize pool;
residuals are shrunk towards zero until every value lies within the remaining prizes.
Initialised at ICM (all weights zero), refreshed by least squares on the terminal
points of complete tournament rollouts (:mod:`strategy.value.rollouts`).

The state order is *next-big-blind first*: the player who posts the big blind on this
hand, then clockwise. A subgame's survivors listed in its preflop order are already in
that order for the next hand (the big blind moves to the first live player after the
current one), so :meth:`ValueModel.for_snapshot` plugs into
:class:`~strategy.endgame.subgame.Subgame` as its ``value_fn``.

On the hand-cap hand the subgames use the official cap ICM, i.e. they are solved to the
terminal score; :func:`late_values` gives those exact values of a cap-hand state.

    python -m strategy.value.model rollouts --ruleset practice_nlhe --lineups 50 --blocks 5 --jobs 8
    python -m strategy.value.model fit --ruleset practice_nlhe
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from strategy.endgame.points import icm_batch
from strategy.tables import store

GENERATOR = "strategy.value.model"
INPUTS = ["value/model.py", "value/rollouts.py", "endgame/points.py", "endgame/table_state.py", "endgame/keys.py"]
FEATURES = ("pressure", "pressure_x_wait", "posts_now", "posts_next", "posts_last", "pressure_x_posts_now",
            "spread", "spread_x_horizon", "icm_x_horizon")
MAX_PLAYERS = 8
ROLLOUT_DIR = Path(__import__("os").environ.get("VALUE_ROLLOUT_DIR", "artifacts/value"))


def table_name(ruleset_id: str) -> str:
    return f"value-{ruleset_id}"


def features(stacks: np.ndarray, bb: float, hands_left: float, icm: np.ndarray, first_prize: float) -> np.ndarray:
    """``(m, k, F)`` centred residual features of ``m`` states with ``k`` live players each."""
    s = np.asarray(stacks, np.float64)
    m, k = s.shape
    total = s.sum(1, keepdims=True)
    x = s / total
    p = np.minimum(1.5 * bb / s, 1.0)
    wait = np.broadcast_to(np.arange(k) / max(k - 1, 1), (m, k))
    d = np.arange(k)
    h = np.full((m, k), min(hands_left, 200.0) / 200.0) if np.isscalar(hands_left) else \
        np.broadcast_to(np.minimum(np.asarray(hands_left, np.float64), 200.0)[:, None] / 200.0, (m, k))
    v = icm / first_prize
    spread = x * (1 - x)
    f = np.stack([p, p * wait, np.broadcast_to(d == 0, (m, k)), np.broadcast_to(d == 1, (m, k)),
                  np.broadcast_to(d == k - 1, (m, k)), p * (d == 0), spread, spread * h, v * h], axis=2)
    return f - f.mean(1, keepdims=True)


def _shrink(icm: np.ndarray, r: np.ndarray, prizes: Sequence[float]) -> np.ndarray:
    """The largest ``a`` in [0, 1] per state with ``icm + a r`` inside [last prize, first prize]."""
    k = icm.shape[1]
    lo, hi = float(min(prizes[:k])), float(max(prizes[:k]))
    with np.errstate(divide="ignore", invalid="ignore"):
        up = np.where(r > 0, (hi - icm) / r, np.inf)
        down = np.where(r < 0, (lo - icm) / r, np.inf)
    a = np.minimum(1.0, np.minimum(up, down).min(1))
    return np.clip(a, 0.0, 1.0)


@dataclass(frozen=True)
class ValueModel:
    weights: dict[int, np.ndarray]
    """Per live-player count, ``len(FEATURES)`` weights in first-prize units."""
    blind_levels: tuple[tuple[int, int], ...]
    hands_per_level: int
    hand_cap: int
    payouts: tuple[float, ...]

    @classmethod
    def icm(cls, config) -> ValueModel:
        return cls({}, tuple(map(tuple, config.blind_levels)), config.hands_per_level, config.hand_cap,
                   tuple(float(p) for p in config.resolved_payouts))

    def big_blind(self, hand_number: int) -> int:
        level = min((hand_number - 1) // self.hands_per_level, len(self.blind_levels) - 1)
        return self.blind_levels[level][1]

    def values(self, stacks: np.ndarray, prizes: Sequence[float], *, hand_number: int) -> np.ndarray:
        """V of states at the start of ``hand_number`` (rows: next-big-blind-first stacks)."""
        s = np.asarray(stacks, np.float64)
        if s.ndim == 1:
            return self.values(s[None, :], prizes, hand_number=hand_number)[0]
        k = s.shape[1]
        base = icm_batch(s, prizes)
        w = self.weights.get(k)
        if w is None or hand_number > self.hand_cap:
            return base
        f = features(s, self.big_blind(hand_number), self.hand_cap - hand_number + 1, base, self.payouts[0])
        r = (f @ w) * self.payouts[0]
        return base + _shrink(base, r, prizes)[:, None] * r

    def for_snapshot(self, snap):
        """The ``value_fn`` of a subgame for ``snap``'s hand: survivors start the next hand."""
        nxt = snap.hand_number + 1

        def value_fn(survivors: np.ndarray, prizes: Sequence[float]) -> np.ndarray:
            return self.values(survivors, prizes, hand_number=nxt)

        return value_fn


def load_value_model(ruleset_id: str, directory: str | None = None) -> ValueModel:
    loaded = store.load_arrays(table_name(ruleset_id), directory)
    if loaded is None:
        raise FileNotFoundError(f"{table_name(ruleset_id)}.npz is missing: make value-rollouts value-fit")
    prov, a = loaded
    if prov.get("ruleset_id") != ruleset_id:
        raise store.TableError(f"value table built for {prov.get('ruleset_id')}, not {ruleset_id}")
    weights = {int(k): a["weights"][i].copy() for i, k in enumerate(a["players"].tolist())}
    return ValueModel(weights, tuple(map(tuple, prov["blind_levels"])), int(prov["hands_per_level"]),
                      int(prov["hand_cap"]), tuple(prov["payouts"]))


def load_value_fn(name: str) -> ValueModel:
    """``value-<ruleset>`` -> the fitted model (the generator's ``--value``)."""
    return load_value_model(name.removeprefix("value-"))


# --------------------------------------------------------------------- data


@dataclass
class States:
    """Start-of-hand states of complete rollouts, grouped by live-player count."""

    tournament: np.ndarray
    hand_number: np.ndarray
    stacks: np.ndarray
    """``(m, k)`` next-big-blind-first."""
    target: np.ndarray
    """``(m, k)`` the players' terminal points."""


def rollout_states(rollouts) -> dict[int, States]:
    from strategy.endgame.table_state import seated_snapshot

    rows: dict[int, list] = {}
    for t, ro in enumerate(rollouts):
        for h in ro.hands:
            s = seated_snapshot(ring=ro.ring, start_stacks=h.start_stacks, button=h.button,
                                small_blind_seat=h.small_blind_seat, big_blind=h.big_blind, heads_up=h.heads_up,
                                blinds=h.blinds, payouts=ro.payouts, hand_number=h.hand_number,
                                hand_cap=ro.hand_cap)
            order = list(s.seats)
            order = order[-1:] + order[:-1]
            k = len(order)
            rows.setdefault(k, []).append((t, h.hand_number, [h.start_stacks[x] for x in order],
                                           [ro.points[x] for x in order]))
    out = {}
    for k, rs in rows.items():
        out[k] = States(np.array([r[0] for r in rs]), np.array([r[1] for r in rs]),
                        np.array([r[2] for r in rs], np.float64), np.array([r[3] for r in rs], np.float64))
    return out


def design(model: ValueModel, st: States, k: int) -> tuple[np.ndarray, np.ndarray]:
    prizes = model.payouts[:k]
    base = icm_batch(st.stacks, prizes)
    bb = np.array([model.big_blind(int(h)) for h in st.hand_number], np.float64)
    out = np.zeros(st.stacks.shape + (len(FEATURES),))
    for b in np.unique(bb):
        sel = bb == b
        out[sel] = features(st.stacks[sel], float(b), model.hand_cap - st.hand_number[sel] + 1, base[sel],
                            model.payouts[0])
    return out, base


def fit(model: ValueModel, states: dict[int, States], train: np.ndarray, *, ridge: float = 1e-3,
        min_rows: int = 200) -> ValueModel:
    weights = {}
    for k, st in sorted(states.items()):
        if k < 2:
            continue
        sel = np.isin(st.tournament, train)
        if sel.sum() < min_rows:
            continue
        f, base = design(model, States(st.tournament[sel], st.hand_number[sel], st.stacks[sel], st.target[sel]), k)
        y = (st.target[sel] - base) / model.payouts[0]
        x = f.reshape(-1, f.shape[2])
        a = x.T @ x + ridge * len(x) * np.eye(x.shape[1])
        weights[k] = np.linalg.solve(a, x.T @ y.reshape(-1))
    return ValueModel(weights, model.blind_levels, model.hands_per_level, model.hand_cap, model.payouts)


def evaluate(models: dict[str, ValueModel], states: dict[int, States], test: np.ndarray, *, resamples: int = 1000,
             seed: int = 0) -> dict:
    """Held-out squared error of each model vs ICM, with a tournament-clustered bootstrap CI."""
    per: dict[str, list[np.ndarray]] = {name: [] for name in models}
    owners = []
    by_k = {}
    for k, st in sorted(states.items()):
        sel = np.isin(st.tournament, test)
        if not sel.any() or k < 2:
            continue
        stacks, target, hands = st.stacks[sel], st.target[sel], st.hand_number[sel]
        errs = {}
        for name, m in models.items():
            v = np.zeros_like(stacks)
            for hn in np.unique(hands):
                rows = hands == hn
                v[rows] = m.values(stacks[rows], m.payouts[:k], hand_number=int(hn))
            e = ((v - target) ** 2).sum(1)
            per[name].append(e)
            errs[name] = float(e.mean())
        owners.append(st.tournament[sel])
        by_k[k] = {"states": int(sel.sum()), **{f"mse_{n}": v for n, v in errs.items()}}
    owner = np.concatenate(owners)
    out = {"by_players": by_k, "states": int(len(owner)), "tournaments": int(len(np.unique(owner)))}
    ref = np.concatenate(per["icm"])
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(owner, return_inverse=True)
    for name in models:
        if name == "icm":
            continue
        diff = np.concatenate(per[name]) - ref
        sums = np.bincount(inv, diff, len(uniq))
        counts = np.bincount(inv, None, len(uniq))
        boots = []
        for _ in range(resamples):
            pick = rng.integers(0, len(uniq), len(uniq))
            boots.append(sums[pick].sum() / counts[pick].sum())
        out[f"mse_diff_{name}_vs_icm"] = {"mean": float(diff.mean()), "ci95": [float(np.quantile(boots, 0.025)),
                                                                              float(np.quantile(boots, 0.975))]}
        out[f"mse_ratio_{name}_vs_icm"] = float(np.concatenate(per[name]).mean() / ref.mean())
    out["mse_icm"] = float(ref.mean())
    return out


def drift(models: dict[str, ValueModel], rollouts, test: set[int]) -> dict:
    """Mean one-hand change of each player's value, by hands until it posts the big blind.

    A correct continuation value is a martingale along play, so the mean change given
    the state is zero; ICM ignores the blinds, so it drifts down for the player about to
    post. The last state of a tournament steps to the terminal points.
    """
    from strategy.endgame.table_state import seated_snapshot

    acc: dict[str, dict[int, list[float]]] = {n: {} for n in models}
    for t, ro in enumerate(rollouts):
        if t not in test:
            continue
        seq = []
        for h in ro.hands:
            s = seated_snapshot(ring=ro.ring, start_stacks=h.start_stacks, button=h.button,
                                small_blind_seat=h.small_blind_seat, big_blind=h.big_blind, heads_up=h.heads_up,
                                blinds=h.blinds, payouts=ro.payouts, hand_number=h.hand_number,
                                hand_cap=ro.hand_cap)
            order = list(s.seats)
            seq.append((h.hand_number, order[-1:] + order[:-1], h.start_stacks))
        for name, m in models.items():
            vals = []
            for hn, order, stacks in seq:
                v = m.values(np.array([[stacks[x] for x in order]], np.float64), m.payouts[:len(order)],
                             hand_number=hn)[0]
                vals.append(dict(zip(order, v.tolist())))
            for i, (hn, order, _) in enumerate(seq):
                nxt = vals[i + 1] if i + 1 < len(seq) else None
                for d, seat in enumerate(order):
                    after = nxt.get(seat, ro.points[seat]) if nxt is not None else ro.points[seat]
                    acc[name].setdefault(d, []).append(after - vals[i][seat])
    return {name: {str(d): {"mean": float(np.mean(x)), "se": float(np.std(x) / np.sqrt(len(x))), "n": len(x)}
                   for d, x in sorted(by.items())} for name, by in acc.items()}


def late_values(snap, model, *, iterations: int = 300) -> np.ndarray:
    """Exact (for the class model) points of a hand-cap state: its final hand solved to the score."""
    from strategy.endgame.harvest import solve_snapshot

    if snap.hand_number < snap.hand_cap:
        raise ValueError("late_values solves the hand-cap hand only")
    solved = solve_snapshot(snap, model, iterations=iterations, with_gains=False)
    return solved.solver.values(solved.sigma)


def cap_value_fn(snap, model, *, iterations: int = 300):
    """Continuation value for the hand before the cap: each survivor vector's cap hand, solved.

    Survivors arrive next-big-blind first; the cap hand's preflop order starts after
    that big blind, so it is the same list rotated by one.
    """
    from strategy.endgame.keys import Snapshot

    if snap.hand_number + 1 != snap.hand_cap:
        raise ValueError("cap_value_fn is for the hand just before the cap")
    sb, bb = None, None
    cache: dict[tuple[int, ...], np.ndarray] = {}

    def value_fn(survivors: np.ndarray, prizes: Sequence[float]) -> np.ndarray:
        nonlocal sb, bb
        out = np.zeros_like(survivors, dtype=np.float64)
        for r, row in enumerate(np.asarray(survivors, np.int64)):
            key = tuple(int(x) for x in row)
            if key not in cache:
                k = len(key)
                order = key[1:] + key[:1]
                blind = snap.bb
                posts = [0] * k
                posts[k - 1] = min(blind, order[k - 1])
                posts[k - 2] = min(blind // 2, order[k - 2])
                cap = Snapshot(order, tuple(posts), blind, snap.payouts, snap.hand_cap, snap.hand_cap)
                v = late_values(cap, model, iterations=iterations)
                cache[key] = np.concatenate([v[-1:], v[:-1]])
            out[r] = cache[key]
        return out

    return value_fn


# ---------------------------------------------------------------------- CLI


POINT_BOT = "strategy/endgame/bots/points.py"


def value_lineups(ruleset_id: str, count: int, seed: int) -> list[list[str]]:
    """Our point-aware bot in two seats, training-partition members of the registry in the rest."""
    from strategy.value.rollouts import training_lineups

    return training_lineups(ruleset_id, MAX_PLAYERS, count, seed, fixed=(POINT_BOT, POINT_BOT))


def rollout_path(ruleset_id: str) -> Path:
    return ROLLOUT_DIR / f"rollouts-{ruleset_id}.jsonl"


def _split(n: int, seed: int, test_share: float) -> tuple[np.ndarray, np.ndarray]:
    ids = np.arange(n)
    rng = np.random.default_rng(seed)
    test = np.sort(rng.choice(n, int(round(n * test_share)), replace=False))
    return np.setdiff1d(ids, test), test


def main(argv: list[str] | None = None) -> None:
    from strategy.tables.endgame import default_config
    from strategy.value.rollouts import load, run_lineups, save

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rollouts")
    r.add_argument("--ruleset", required=True)
    r.add_argument("--lineups", type=int, default=50)
    r.add_argument("--blocks", type=int, default=5, help="seeded blocks per lineup (each: every seat rotation)")
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--jobs", type=int, default=8)
    f = sub.add_parser("fit")
    f.add_argument("--ruleset", required=True)
    f.add_argument("--test-share", type=float, default=0.3)
    f.add_argument("--seed", type=int, default=1)
    f.add_argument("--ridge", type=float, default=1e-3)
    f.add_argument("--out", help="output directory (default: strategy/tables/data)")
    a = ap.parse_args(argv)
    config = default_config(a.ruleset)
    if a.cmd == "rollouts":
        t0 = time.perf_counter()
        ros = run_lineups(config, value_lineups(a.ruleset, a.lineups, a.seed), blocks=a.blocks,
                          base_seed=f"value-v1|{a.seed}", jobs=a.jobs)
        sha = save(ros, rollout_path(a.ruleset))
        print(f"wrote {rollout_path(a.ruleset)} ({len(ros)} tournaments, sha256 {sha[:12]}, "
              f"{time.perf_counter() - t0:.0f}s)")
        return
    t0 = time.perf_counter()
    path = rollout_path(a.ruleset)
    ros = load(path)
    states = rollout_states(ros)
    train, test = _split(len(ros), a.seed, a.test_share)
    icm_model = ValueModel.icm(config)
    model = fit(icm_model, states, train, ridge=a.ridge)
    report = evaluate({"icm": icm_model, "fitted": model}, states, test)
    report["drift"] = drift({"icm": icm_model, "fitted": model}, ros, set(test.tolist()))
    players = np.array(sorted(model.weights), np.int64)
    arrays = {"players": players, "weights": np.array([model.weights[int(k)] for k in players], np.float64)}
    body = path.read_bytes()
    extra = {"ruleset_id": a.ruleset, "features": list(FEATURES), "blind_levels": [list(x) for x in config.blind_levels],
             "hands_per_level": config.hands_per_level, "hand_cap": config.hand_cap,
             "payouts": list(icm_model.payouts), "rollouts_sha256": hashlib.sha256(body).hexdigest(),
             "rollouts": len(ros), "train_tournaments": int(len(train)), "test_tournaments": int(len(test)),
             "heldout": report}
    out = store.write_arrays(table_name(a.ruleset), arrays, generator=GENERATOR, inputs=INPUTS,
                             params={"ruleset_id": a.ruleset, "test_share": a.test_share, "seed": a.seed,
                                     "ridge": a.ridge}, seconds=time.perf_counter() - t0, extra=extra,
                             directory=Path(a.out) if a.out else None)
    print(f"wrote {out}")
    print(json.dumps({k: v for k, v in report.items() if k != "drift"}, indent=1))
    print(json.dumps(report["drift"], indent=1))


if __name__ == "__main__":
    main()
