"""Hierarchical, correlated parameter sampling with disjoint partition regions.

A family declares a :class:`Prior`: for each named parameter a plausible interval
``[lo, hi]``, a family centre and a spread, plus a few correlations. One individual is
one draw::

    z ~ N(0, I), each coordinate truncated to |z_i| <= TRUNCATE
    eta = logit(centre) + spread * (L z)          L = cholesky(correlation matrix)
    x   = lo + (hi - lo) * sigmoid(eta)

so parameters move together (a wide opener also continuation-bets more, say) and stay
inside their bounds. A table-local regime is a shift of ``eta`` along named parameters
(:meth:`Prior.shifted`), applied from a hand chosen at table start.

Partitions own disjoint regions of the latent space. Each family has a fixed unit
direction ``w`` (from its id); ``u = Phi(w . z)`` is cut into :data:`BINS` equal-probability
bins and every bin belongs to exactly one partition (:data:`PARTITION_BINS`), interleaved
so each partition spans the whole range. ``z -> x`` is a bijection, so the regions are
disjoint in parameter space too; :func:`region_of` recovers a member's bin from its
stored parameters. Draws closer than :data:`EDGE_MARGIN` to a bin edge are rejected so
rounding can never move a member across a boundary.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

TRUNCATE = 2.5
BINS = 20
EDGE_MARGIN = 0.002
DIGITS = 6
PARTITIONS = ("train", "validation", "holdout")
PARTITION_BINS: dict[str, frozenset[int]] = {
    "validation": frozenset({3, 10, 17}),
    "holdout": frozenset({6, 13, 19}),
}
PARTITION_BINS["train"] = frozenset(set(range(BINS)) - PARTITION_BINS["validation"] - PARTITION_BINS["holdout"])


@dataclass(frozen=True)
class Param:
    name: str
    lo: float
    hi: float
    centre: float
    spread: float = 0.6
    """Standard deviation of the individual draw in logit units."""
    doc: str = ""

    def __post_init__(self) -> None:
        if not self.lo < self.centre < self.hi:
            raise ValueError(f"{self.name}: centre {self.centre} not inside ({self.lo}, {self.hi})")


def choice(name: str, options: Sequence[str], doc: str = "") -> Param:
    """A categorical parameter: a real value in ``[0, len(options))`` read with ``int()``.

    The spread is wide enough that the options are close to equally likely.
    """
    return Param(name, 0.0, float(len(options)), len(options) / 2, spread=1.7, doc=doc or " / ".join(options))


def _logit(p: float) -> float:
    p = min(1 - 1e-12, max(1e-12, p))
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def cholesky(m: list[list[float]]) -> list[list[float]]:
    n = len(m)
    out = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = m[i][j] - sum(out[i][k] * out[j][k] for k in range(j))
            if i == j:
                if s <= 1e-9:
                    raise ValueError("correlation matrix is not positive definite")
                out[i][j] = math.sqrt(s)
            else:
                out[i][j] = s / out[j][j]
    return out


@dataclass(frozen=True)
class Prior:
    """A family-level centre and covariance over named, bounded parameters."""

    family: str
    params: tuple[Param, ...]
    correlations: tuple[tuple[str, str, float], ...] = ()
    names: tuple[str, ...] = field(init=False)
    chol: tuple[tuple[float, ...], ...] = field(init=False)
    direction: tuple[float, ...] = field(init=False)

    def __post_init__(self) -> None:
        names = tuple(p.name for p in self.params)
        if len(set(names)) != len(names):
            raise ValueError(f"{self.family}: duplicate parameter names")
        index = {n: i for i, n in enumerate(names)}
        corr = [[1.0 if i == j else 0.0 for j in range(len(names))] for i in range(len(names))]
        for a, b, rho in self.correlations:
            if not -1 < rho < 1:
                raise ValueError(f"{self.family}: correlation {a}/{b} out of range")
            corr[index[a]][index[b]] = corr[index[b]][index[a]] = rho
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "chol", tuple(tuple(r) for r in cholesky(corr)))
        rng = random.Random(f"pop-region-direction-v1|{self.family}")
        w = [rng.gauss(0.0, 1.0) for _ in names]
        norm = math.sqrt(sum(x * x for x in w))
        object.__setattr__(self, "direction", tuple(x / norm for x in w))

    @property
    def centre(self) -> dict[str, float]:
        return {p.name: p.centre for p in self.params}

    def correlation(self, a: str, b: str) -> float:
        i, j = self.names.index(a), self.names.index(b)
        return sum(x * y for x, y in zip(self.chol[i], self.chol[j]))

    # -- latent <-> parameters
    def eta(self, z: Sequence[float]) -> list[float]:
        v = [sum(self.chol[i][k] * z[k] for k in range(i + 1)) for i in range(len(z))]
        return [_logit((p.centre - p.lo) / (p.hi - p.lo)) + p.spread * v[i] for i, p in enumerate(self.params)]

    def from_eta(self, eta: Sequence[float]) -> dict[str, float]:
        return {p.name: round(p.lo + (p.hi - p.lo) * _sigmoid(e), DIGITS) for p, e in zip(self.params, eta)}

    def latent(self, x: Mapping[str, float]) -> list[float]:
        """Invert :meth:`from_eta` and the correlation transform: the draw ``z`` behind ``x``."""
        v = []
        for p in self.params:
            e = _logit((x[p.name] - p.lo) / (p.hi - p.lo))
            v.append((e - _logit((p.centre - p.lo) / (p.hi - p.lo))) / p.spread)
        z: list[float] = []
        for i in range(len(v)):
            z.append((v[i] - sum(self.chol[i][k] * z[k] for k in range(i))) / self.chol[i][i])
        return z

    def region(self, z: Sequence[float]) -> tuple[int, float]:
        """(bin, distance in ``u`` to the nearest bin edge)."""
        u = _phi(sum(w * x for w, x in zip(self.direction, z)))
        b = min(BINS - 1, int(u * BINS))
        edge = min(u - b / BINS, (b + 1) / BINS - u)
        return b, edge

    def shifted(self, x: Mapping[str, float], delta: Mapping[str, float]) -> dict[str, float]:
        """``x`` with ``eta`` moved by ``delta`` (logit units) on the named parameters."""
        out = dict(x)
        for p in self.params:
            if p.name in delta:
                e = _logit((x[p.name] - p.lo) / (p.hi - p.lo)) + delta[p.name]
                out[p.name] = round(p.lo + (p.hi - p.lo) * _sigmoid(e), DIGITS)
        return out

    # -- sampling
    def sample(self, seed: str, partition: str | None = None) -> dict[str, float]:
        """One individual. With a partition, only draws inside that partition's region."""
        rng = random.Random(f"pop-sample-v1|{seed}")
        bins = PARTITION_BINS[partition] if partition is not None else None
        for _ in range(10_000):
            z = [_truncated(rng) for _ in self.params]
            if bins is not None:
                b, edge = self.region(z)
                if b not in bins or edge < EDGE_MARGIN:
                    continue
            x = self.from_eta(self.eta(z))
            if bins is not None and region_of(self, x) not in bins:
                continue
            return x
        raise RuntimeError(f"{self.family}: no draw found in partition {partition}")


def _truncated(rng: random.Random) -> float:
    while True:
        v = rng.gauss(0.0, 1.0)
        if abs(v) <= TRUNCATE:
            return v


def region_of(prior: Prior, x: Mapping[str, float]) -> int:
    return prior.region(prior.latent(x))[0]


def partition_of_bin(b: int) -> str:
    return next(p for p, bins in PARTITION_BINS.items() if b in bins)


def unit_hash(*parts: object) -> float:
    """A uniform number in [0, 1) from its inputs (table-local draws that must not use a stream)."""
    h = hashlib.blake2b("|".join(str(p) for p in parts).encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") / 18446744073709551616.0
