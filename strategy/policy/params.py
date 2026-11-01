"""Named, bounded policy parameters and their search encoding.

A :class:`ParamSpace` is an ordered tuple of :class:`Param`. A parameter vector is a
plain ``dict[name, float]``; the search works on the unit cube, where coordinate
``i`` maps linearly onto ``[lo_i, hi_i]``. Integer-like parameters are still floats:
the policy decides how to use them. Decision-time safe (standard library only).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Param:
    name: str
    default: float
    """The hand-written value: the control policy is every parameter at its default."""
    lo: float
    hi: float
    group: str
    doc: str

    def __post_init__(self) -> None:
        if not self.lo <= self.default <= self.hi or self.lo >= self.hi:
            raise ValueError(f"parameter {self.name}: default {self.default} outside [{self.lo}, {self.hi}]")

    def to_unit(self, value: float) -> float:
        return (value - self.lo) / (self.hi - self.lo)

    def from_unit(self, u: float) -> float:
        return self.lo + min(1.0, max(0.0, u)) * (self.hi - self.lo)


def canonical(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def vector_hash(values: Mapping[str, float], *, digits: int = 6) -> str:
    """Content hash of a parameter vector, rounded so float noise does not change it."""
    body = canonical({k: round(float(v), digits) for k, v in sorted(values.items())})
    return hashlib.sha256(b"strategy-policy-params-v1|" + body).hexdigest()


class ParamSpace:
    def __init__(self, params: Iterable[Param]) -> None:
        self.params: tuple[Param, ...] = tuple(params)
        self.index = {p.name: i for i, p in enumerate(self.params)}
        if len(self.index) != len(self.params):
            raise ValueError("duplicate parameter names")

    def __len__(self) -> int:
        return len(self.params)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.params)

    def defaults(self) -> dict[str, float]:
        return {p.name: p.default for p in self.params}

    def space_hash(self) -> str:
        body = canonical([[p.name, p.default, p.lo, p.hi, p.group] for p in self.params])
        return hashlib.sha256(b"strategy-param-space-v1|" + body).hexdigest()

    def complete(self, values: Mapping[str, float] | None) -> dict[str, float]:
        """Defaults overlaid with ``values``, clipped to bounds. Unknown names are an error."""
        out = self.defaults()
        for k, v in (values or {}).items():
            if k not in self.index:
                raise KeyError(f"unknown parameter {k!r}")
            p = self.params[self.index[k]]
            out[k] = min(p.hi, max(p.lo, float(v)))
        return out

    def to_unit(self, values: Mapping[str, float]) -> list[float]:
        full = self.complete(values)
        return [p.to_unit(full[p.name]) for p in self.params]

    def from_unit(self, z: Sequence[float]) -> dict[str, float]:
        if len(z) != len(self.params):
            raise ValueError(f"expected {len(self.params)} coordinates, got {len(z)}")
        return {p.name: p.from_unit(float(u)) for p, u in zip(self.params, z)}

    def groups(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for p in self.params:
            out.setdefault(p.group, []).append(p.name)
        return out
