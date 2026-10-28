"""Versioned environment randomness.

Benchmarks and tests use :class:`SeededRng`, a SHA-256 counter-mode generator
whose output is fixed by its version string, so a stored seed reproduces the
same decks across Python and library upgrades. Interactive play uses
:class:`SystemRng`. Fixtures should still store explicit decks.
"""

from __future__ import annotations

import hashlib
import secrets
from collections.abc import Sequence
from typing import Protocol, TypeVar

T = TypeVar("T")


class Rng(Protocol):
    implementation: str

    def randbelow(self, n: int) -> int: ...


class SeededRng:
    implementation = "sha256-ctr-v1"

    def __init__(self, seed: int | str, stream: str = "env") -> None:
        # Kept for the private audit record only; projections never read these.
        self.seed = str(seed)
        self.stream = stream
        self._key = hashlib.sha256(f"{self.implementation}|{stream}|{seed}".encode()).digest()
        self._counter = 0
        self._buffer = b""

    def _bytes(self, n: int) -> bytes:
        while len(self._buffer) < n:
            block = hashlib.sha256(self._key + self._counter.to_bytes(8, "big")).digest()
            self._counter += 1
            self._buffer += block
        out, self._buffer = self._buffer[:n], self._buffer[n:]
        return out

    def randbelow(self, n: int) -> int:
        if n <= 0:
            raise ValueError("n must be positive")
        limit = (1 << 64) - ((1 << 64) % n)
        while True:
            value = int.from_bytes(self._bytes(8), "big")
            if value < limit:
                return value % n


class SystemRng:
    implementation = "secrets-system"

    def randbelow(self, n: int) -> int:
        return secrets.randbelow(n)


def shuffled(items: Sequence[T], rng: Rng) -> list[T]:
    """Unbiased Fisher-Yates shuffle."""
    out = list(items)
    for i in range(len(out) - 1, 0, -1):
        j = rng.randbelow(i + 1)
        out[i], out[j] = out[j], out[i]
    return out


def choice(items: Sequence[T], rng: Rng) -> T:
    return items[rng.randbelow(len(items))]
