"""CFR+ over hand-class vectors for push/fold subgames, with exact best responses.

Every decision node holds a ``(2, H)`` strategy (fold vs shove/call per hand class).
Counterfactual values are computed for all classes at once:

* a heads-up showdown multiplies the opponent's reach by the conditional win / tie /
  lose matrices (exact card removal between the two hands);
* a three-way showdown works at bucket level on the joint three-hand tensor;
* players who folded are integrated out with their class prior (their cards' removal
  effect on the others is ignored, as in standard push/fold solvers);
* a bystander's value (a folded player's points still move with the showdown result,
  through ICM) uses the joint distribution of the showdown hands.

In two-player subgames the reported best response and per-information-set action
regrets are exact for the model (class matrices). With three or more acting players
CFR has no equilibrium guarantee; :meth:`PushFoldSolver.deviation_gains` reports each
player's exact unilateral best-response gain against the averaged profile instead.
Averages are full-traversal, reach-weighted (linear weights by default), so the
stochastic-averaging caveat of external-sampling MCCFR does not apply here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from strategy.endgame.showdown import ShowdownModel
from strategy.endgame.subgame import SubgameTree, ordering_codes


@dataclass
class _Term:
    node: int
    showdown: tuple[int, ...]
    others: tuple[int, ...]
    u: np.ndarray


@dataclass(frozen=True)
class PlayerReport:
    player: int
    value: float
    best_response_value: float
    gain: float
    max_action_regret: float
    """Largest conditional action regret over information sets with cf reach >= the floor."""
    mean_action_regret: float
    """Reach-weighted mean conditional action regret."""


class PushFoldSolver:
    def __init__(self, tree: SubgameTree, model: ShowdownModel) -> None:
        self.tree = tree
        self.model = model
        g = tree.game
        self.n = g.n
        self.h = model.h
        self.num_nodes = len(tree.nodes)
        if tuple(c for c, _ in ordering_codes(3)) != model.codes3:
            raise AssertionError("three-way outcome codes disagree with the showdown table")
        self.parent = np.array([nd.parent for nd in tree.nodes])
        self.player = np.array([nd.player for nd in tree.nodes])
        self.action_index = np.zeros(self.num_nodes, np.int64)
        for nd in tree.nodes[1:]:
            self.action_index[nd.idx] = 0 if nd.label == "f" else 1
        self.decisions = [nd.idx for nd in tree.nodes if nd.player >= 0]
        self.child = {d: (tree.nodes[d].children["f"], tree.nodes[d].children[[k for k in tree.nodes[d].children
                                                                                if k != "f"][0]])
                      for d in self.decisions}
        self.terms = [_Term(t.node, t.showdown, tuple(p for p in range(self.n) if p not in t.showdown), t.utility)
                      for t in tree.terminals.values()]
        self.mw, self.mt, self.ml, *_ = model.matrices()
        self.prior = model.prior
        self.regret = {d: np.zeros((2, self.h)) for d in self.decisions}
        self.strat_sum = {d: np.zeros((2, self.h)) for d in self.decisions}
        self.sigma = {d: np.full((2, self.h), 0.5) for d in self.decisions}
        self.iterations = 0
        b = model.bucket
        self._bucket = b
        self._j3c = np.ascontiguousarray(model.j3.transpose(0, 1, 3, 2))  # (a, b, k, c)
        self._j3b = np.ascontiguousarray(model.j3.transpose(0, 2, 3, 1))  # (a, c, k, b)

    # ---------------------------------------------------------------- core
    def reach(self, sigma) -> np.ndarray:
        r = np.ones((self.num_nodes, self.n, self.h))
        for idx in range(1, self.num_nodes):
            par = self.parent[idx]
            r[idx] = r[par]
            p = self.player[par]
            r[idx, p] *= sigma[par][self.action_index[idx]]
        return r

    def terminal_values(self, r: np.ndarray, ones: bool = False) -> dict[int, np.ndarray]:
        """``out[t][q]``: player q's counterfactual value at terminal t for each of q's classes.

        Given q's class i, every other player's class is drawn with card removal against
        q (``cond[i]``) and independently of the rest, so q's value is
        ``prod_{p != q} rho_p(i) * E[u_q | i, the others' actions]`` with
        ``rho_p = cond @ reach_p``. Using the same conditioning at every terminal keeps
        counterfactual values consistent across q's actions (a constant shift in utility
        leaves every regret unchanged).
        """
        out: dict[int, np.ndarray] = {}
        prior = self.prior
        nodes = [t.node for t in self.terms]
        rho_all = r[nodes] @ self.model.cond.T  # (T, n, H): rho[t, p, i] = sum_j cond[i, j] r[t, p, j]
        eps = 1e-300
        for ti, t in enumerate(self.terms):
            rt = r[t.node]
            rho = rho_all[ti]
            u = np.ones_like(t.u) if ones else t.u
            v = np.zeros((self.n, self.h))
            s = t.showdown
            fold_prod = np.ones(self.h)
            for p in t.others:
                fold_prod = fold_prod * rho[p]

            def all_but(q: int) -> np.ndarray:
                prod = np.ones(self.h)
                for p in range(self.n):
                    if p != q:
                        prod = prod * rho[p]
                return prod

            if len(s) == 1:
                for q in range(self.n):
                    v[q] = u[0, q] * all_but(q)
            elif len(s) == 2:
                a, b = s
                wb, tb, lb = self.mw @ rt[b], self.mt @ rt[b], self.ml @ rt[b]
                wa, ta, la = self.mw @ rt[a], self.mt @ rt[a], self.ml @ rt[a]
                v[a] = fold_prod * (u[1, a] * wb + u[0, a] * tb + u[2, a] * lb)
                v[b] = fold_prod * (u[2, b] * wa + u[0, b] * ta + u[1, b] * la)
                if t.others:
                    ra = rt[a] * prior
                    xw, xt, xl = ra @ wb, ra @ tb, ra @ lb
                    mass = xw + xt + xl
                    for q in t.others:
                        e = (u[1, q] * xw + u[0, q] * xt + u[2, q] * xl) / mass if mass > eps else \
                            (u[1, q] + u[2, q]) / 2
                        v[q] = all_but(q) * e
            else:
                rb = [self.model.bucket_avg @ rt[p] for p in s]
                x = self._j3c @ rb[2]  # (a, b, k)
                y = self._j3b @ rb[1]  # (a, c, k)
                gs = (np.einsum("abk,b->ak", x, rb[1]), np.einsum("abk,a->bk", x, rb[0]),
                      np.einsum("ack,a->ck", y, rb[0]))
                for role, gk in enumerate(gs):
                    q = s[role]
                    mass = gk.sum(1)
                    num = gk @ u[:, q]
                    e = np.where(mass > eps, num / np.maximum(mass, eps), u[:, q].mean())
                    v[q] = all_but(q) * e[self._bucket]
                if t.others:
                    total = gs[0].T @ rb[0]
                    mass = total.sum()
                    for q in t.others:
                        e = (total @ u[:, q]) / mass if mass > eps else u[:, q].mean()
                        v[q] = all_but(q) * e
            out[t.node] = v
        return out

    def backward(self, sigma, term: dict[int, np.ndarray]) -> np.ndarray:
        v = np.zeros((self.num_nodes, self.n, self.h))
        for idx in range(self.num_nodes - 1, -1, -1):
            if idx in term:
                v[idx] = term[idx]
                continue
            p = self.player[idx]
            c0, c1 = self.child[idx]
            v[idx] = v[c0] + v[c1]
            v[idx, p] = sigma[idx][0] * v[c0, p] + sigma[idx][1] * v[c1, p]
        return v

    # ----------------------------------------------------------------- CFR
    def iterate(self, linear: bool = True) -> None:
        self.iterations += 1
        t = self.iterations
        r = self.reach(self.sigma)
        v = self.backward(self.sigma, self.terminal_values(r))
        w = float(t) if linear else 1.0
        for d in self.decisions:
            p = self.player[d]
            c0, c1 = self.child[d]
            reg = self.regret[d]
            reg[0] += v[c0, p] - v[d, p]
            reg[1] += v[c1, p] - v[d, p]
            np.maximum(reg, 0.0, out=reg)
            self.strat_sum[d] += w * r[d, p] * self.sigma[d]
        for d in self.decisions:
            reg = self.regret[d]
            tot = reg.sum(0)
            pos = tot > 0
            s = np.full((2, self.h), 0.5)
            s[:, pos] = reg[:, pos] / tot[pos]
            self.sigma[d] = s

    def run(self, iterations: int, linear: bool = True) -> PushFoldSolver:
        for _ in range(iterations):
            self.iterate(linear)
        return self

    def average(self) -> dict[int, np.ndarray]:
        out = {}
        for d in self.decisions:
            s = self.strat_sum[d]
            tot = s.sum(0)
            a = np.full((2, self.h), 0.5)
            pos = tot > 0
            a[:, pos] = s[:, pos] / tot[pos]
            out[d] = a
        return out

    # ---------------------------------------------------------- evaluation
    def values(self, sigma) -> np.ndarray:
        """Expected utility of every player under ``sigma``."""
        v = self.backward(sigma, self.terminal_values(self.reach(sigma)))
        return v[0] @ self.prior

    def best_response(self, sigma, q: int) -> tuple[float, dict[int, np.ndarray]]:
        """Exact best-response value of player q (classes chosen independently) and the BR strategy."""
        r = self.reach(sigma)
        term = self.terminal_values(r)
        v = np.zeros((self.num_nodes, self.h))
        br: dict[int, np.ndarray] = {}
        for idx in range(self.num_nodes - 1, -1, -1):
            if idx in term:
                v[idx] = term[idx][q]
                continue
            c0, c1 = self.child[idx]
            if self.player[idx] == q:
                pick = v[c1] > v[c0]
                v[idx] = np.where(pick, v[c1], v[c0])
                br[idx] = np.stack([~pick, pick]).astype(np.float64)
            else:
                v[idx] = v[c0] + v[c1]
        return float(v[0] @ self.prior), br

    def report(self, sigma, q: int, reach_floor: float = 1e-3) -> PlayerReport:
        r = self.reach(sigma)
        v = self.backward(sigma, self.terminal_values(r))
        w = self.backward(sigma, self.terminal_values(r, ones=True))
        value = float(v[0, q] @ self.prior)
        br_value, _ = self.best_response(sigma, q)
        worst = 0.0
        num = den = 0.0
        for d in self.tree.player_nodes[q]:
            c0, c1 = self.child[d]
            best = np.maximum(v[c0, q], v[c1, q])
            reg = best - v[d, q]
            cf = w[d, q]
            ok = cf >= reach_floor
            if ok.any():
                worst = max(worst, float((reg[ok] / cf[ok]).max()))
            num += float((reg * self.prior).sum())
            den += float((cf * self.prior).sum())
        return PlayerReport(q, value, br_value, br_value - value, worst, num / den if den > 0 else 0.0)

    def deviation_gains(self, sigma) -> np.ndarray:
        vals = self.values(sigma)
        return np.array([self.best_response(sigma, q)[0] - vals[q] for q in range(self.n)])

    # ------------------------------------------------------------ helpers
    def strategy_at(self, sigma, actions) -> np.ndarray:
        """Probability of the aggressive action (shove or call) per class at a node."""
        return sigma[self.tree.find(actions)][1]
