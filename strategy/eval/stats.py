"""Statistics for paired tournament evaluation. Pure Python, deterministic, no SciPy.

The independent unit is a complete paired block (one seed, every seat rotation, both
arms), never a hand or a decision. Intervals:

* :func:`paired_t` — Student t on block differences ``D_i = U_i(B) - U_i(A)``.
* :func:`cluster_bootstrap` — percentile bootstrap that resamples whole clusters
  (optionally within strata), for skewed or nonlinear statistics.
* :func:`wilson` / :func:`newcombe_difference` — opportunity rates. Decisions are
  clustered inside tournaments, so these are nominal and labelled as such in reports.

Sample size: ``N = (z_{1-a/2} + z_{power})^2 sigma_D^2 / delta^2`` (7.85 sigma^2/delta^2
at 5%/80%), optionally refined with t quantiles.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import Any, TypeVar

T = TypeVar("T")
_NORMAL = NormalDist()


def mean(xs: Sequence[float]) -> float:
    if not xs:
        raise ValueError("mean of no values")
    return math.fsum(xs) / len(xs)


def sample_sd(xs: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(math.fsum((x - m) ** 2 for x in xs) / (n - 1))


def quantile(xs: Sequence[float], q: float) -> float:
    """Linear interpolation between order statistics (R type 7)."""
    if not xs:
        raise ValueError("quantile of no values")
    s = sorted(xs)
    h = (len(s) - 1) * q
    lo = math.floor(h)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (h - lo) * (s[hi] - s[lo])


# ------------------------------------------------------------- distributions


def _betacf(a: float, b: float, x: float) -> float:
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c if abs(c) > tiny else 1.0 + aa / tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c if abs(c) > tiny else 1.0 + aa / tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta ``I_x(a, b)``."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    front = math.exp(ln)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _bisect(f: Callable[[float], float], target: float, lo: float, hi: float, iters: int = 200) -> float:
    for _ in range(iters):
        mid = (lo + hi) / 2
        if f(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def t_cdf(t: float, df: float) -> float:
    x = df / (df + t * t)
    tail = 0.5 * betainc(df / 2.0, 0.5, x)
    return 1.0 - tail if t > 0 else tail


def t_ppf(p: float, df: float) -> float:
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    if p == 0.5:
        return 0.0
    if p < 0.5:
        return -t_ppf(1.0 - p, df)
    hi = 1.0
    while t_cdf(hi, df) < p:
        hi *= 2.0
    return _bisect(lambda t: t_cdf(t, df), p, 0.0, hi)


def normal_ppf(p: float) -> float:
    return _NORMAL.inv_cdf(p)


def binomial_sf(k: int, n: int, p: float) -> float:
    """``P(X >= k)`` for ``X ~ Binomial(n, p)``."""
    if k <= 0:
        return 1.0
    if k > n or p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    lp, lq = math.log(p), math.log1p(-p)
    lg = math.lgamma(n + 1)
    return min(1.0, math.fsum(math.exp(lg - math.lgamma(j + 1) - math.lgamma(n - j + 1) + j * lp + (n - j) * lq)
                              for j in range(k, n + 1)))


def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Exact binomial interval for ``k`` successes in ``n``."""
    if n <= 0:
        raise ValueError("n must be positive")
    a = (1 - level) / 2
    lo = 0.0 if k == 0 else _bisect(lambda x: betainc(k, n - k + 1, x), a, 0.0, 1.0)
    hi = 1.0 if k == n else _bisect(lambda x: betainc(k + 1, n - k, x), 1 - a, 0.0, 1.0)
    return lo, hi


# ----------------------------------------------------------------- intervals


@dataclass(frozen=True)
class PairedT:
    n: int
    mean: float
    sd: float
    se: float | None
    df: int
    level: float
    t_crit: float | None
    lo: float | None
    hi: float | None
    t_stat: float | None
    p_value: float | None
    """Two-sided p-value for a zero mean difference."""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def paired_t(diffs: Sequence[float], level: float = 0.95) -> PairedT:
    n = len(diffs)
    if n == 0:
        raise ValueError("no paired differences")
    m = mean(diffs)
    if n < 2:
        return PairedT(n, m, 0.0, None, 0, level, None, None, None, None, None)
    sd = sample_sd(diffs)
    se = sd / math.sqrt(n)
    tc = t_ppf(0.5 + level / 2, n - 1)
    if se == 0.0:
        return PairedT(n, m, 0.0, 0.0, n - 1, level, tc, m, m, None, 1.0 if m == 0 else 0.0)
    t = m / se
    p = 2.0 * (1.0 - t_cdf(abs(t), n - 1))
    return PairedT(n, m, sd, se, n - 1, level, tc, m - tc * se, m + tc * se, t, p)


@dataclass(frozen=True)
class Welch:
    mean_difference: float
    lo: float | None
    hi: float | None
    df: float | None
    p_value: float | None


def welch_t(a: Sequence[float], b: Sequence[float], level: float = 0.95) -> Welch:
    """Unpaired ``mean(b) - mean(a)``; used only to show what ignoring the pairing costs."""
    d = mean(b) - mean(a)
    va, vb = sample_sd(a) ** 2 / len(a), sample_sd(b) ** 2 / len(b)
    se = math.sqrt(va + vb)
    if se == 0.0 or len(a) < 2 or len(b) < 2:
        return Welch(d, d if se == 0 else None, d if se == 0 else None, None, None)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    tc = t_ppf(0.5 + level / 2, df)
    p = 2.0 * (1.0 - t_cdf(abs(d / se), df))
    return Welch(d, d - tc * se, d + tc * se, df, p)


def _rng(seed: str) -> random.Random:
    return random.Random(hashlib.sha256(f"strategy-bootstrap-v1|{seed}".encode()).digest())


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lo: float | None
    hi: float | None
    level: float
    resamples: int
    clusters: int
    strata: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def cluster_bootstrap(clusters: Sequence[T], statistic: Callable[[Sequence[T]], float], *, seed: str,
                      strata: Sequence[Hashable] | None = None, resamples: int = 2000,
                      level: float = 0.95) -> BootstrapInterval:
    """Percentile interval from resampling whole clusters with replacement.

    With ``strata`` (one label per cluster), clusters are resampled within each stratum,
    preserving the design's lineup and variant balance.
    """
    n = len(clusters)
    if n == 0:
        raise ValueError("no clusters")
    estimate = statistic(clusters)
    groups: dict[Hashable, list[int]] = {}
    for i, s in enumerate(strata if strata is not None else [0] * n):
        groups.setdefault(s, []).append(i)
    if n < 2:
        return BootstrapInterval(estimate, None, None, level, 0, n, len(groups))
    rng = _rng(seed)
    stats = []
    for _ in range(resamples):
        sample: list[T] = []
        for members in groups.values():
            sample.extend(clusters[members[rng.randrange(len(members))]] for _ in members)
        stats.append(statistic(sample))
    stats.sort()
    lo = stats[int((1 - level) / 2 * resamples)]
    hi = stats[min(resamples - 1, int((1 + level) / 2 * resamples))]
    return BootstrapInterval(estimate, lo, hi, level, resamples, n, len(groups))


def bootstrap_mean(values: Sequence[float], *, seed: str, strata: Sequence[Hashable] | None = None,
                   resamples: int = 2000, level: float = 0.95) -> BootstrapInterval:
    return cluster_bootstrap(list(values), lambda xs: math.fsum(xs) / len(xs), seed=seed, strata=strata,
                             resamples=resamples, level=level)


def wilson(k: int, n: int, level: float = 0.95) -> tuple[float, float] | None:
    if n <= 0:
        return None
    z = normal_ppf(0.5 + level / 2)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def newcombe_difference(k1: int, n1: int, k2: int, n2: int, level: float = 0.95) -> tuple[float, float] | None:
    """Interval for ``k2/n2 - k1/n1`` from the two Wilson intervals (Newcombe's method 10)."""
    w1, w2 = wilson(k1, n1, level), wilson(k2, n2, level)
    if w1 is None or w2 is None:
        return None
    p1, p2 = k1 / n1, k2 / n2
    d = p2 - p1
    lo = d - math.sqrt((p2 - w2[0]) ** 2 + (w1[1] - p1) ** 2)
    hi = d + math.sqrt((w2[1] - p2) ** 2 + (p1 - w1[0]) ** 2)
    return lo, hi


def holm(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values (family-wise error control for screening many hypotheses)."""
    order = sorted(p_values, key=lambda k: p_values[k])
    m = len(order)
    out: dict[str, float] = {}
    running = 0.0
    for i, key in enumerate(order):
        running = max(running, min(1.0, (m - i) * p_values[key]))
        out[key] = running
    return out


# --------------------------------------------------------------- sample size


@dataclass(frozen=True)
class SampleSize:
    blocks: int
    sigma_d: float
    delta: float
    alpha: float
    power: float
    two_sided: bool
    normal_blocks: float
    multiplier: float
    """``(z_a + z_b)^2``: 7.85 at a two-sided 5% test with 80% power."""
    method: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def sample_size(sigma_d: float, delta: float, *, alpha: float = 0.05, power: float = 0.80, two_sided: bool = True,
                t_correction: bool = True) -> SampleSize:
    """Paired blocks needed to detect a mean difference ``delta`` given a pilot block SD ``sigma_d``."""
    if delta <= 0 or sigma_d < 0:
        raise ValueError("delta must be positive and sigma_d nonnegative")
    za = normal_ppf(1 - alpha / 2) if two_sided else normal_ppf(1 - alpha)
    zb = normal_ppf(power)
    mult = (za + zb) ** 2
    n_normal = mult * sigma_d ** 2 / delta ** 2
    n = max(2, math.ceil(n_normal))
    method = "normal"
    if t_correction:
        method = "t-refined"
        for _ in range(100):
            df = max(1, n - 1)
            ta = t_ppf(1 - alpha / 2, df) if two_sided else t_ppf(1 - alpha, df)
            tb = t_ppf(power, df)
            nxt = max(2, math.ceil((ta + tb) ** 2 * sigma_d ** 2 / delta ** 2))
            if nxt == n:
                break
            n = nxt
    return SampleSize(n, sigma_d, delta, alpha, power, two_sided, n_normal, mult, method)


def minimum_detectable_effect(sigma_d: float, blocks: int, *, alpha: float = 0.05, power: float = 0.80) -> float:
    if blocks < 2:
        return math.inf
    df = blocks - 1
    return (t_ppf(1 - alpha / 2, df) + t_ppf(power, df)) * sigma_d / math.sqrt(blocks)
