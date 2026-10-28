"""Monotonic clocks for server-side deadlines (spec 6.3).

Deadlines are integer nanoseconds on a monotonic clock. Tests drive a
:class:`FakeClock`; the server uses :class:`MonotonicClock`. Wall-clock time is
never used for game decisions.
"""

from __future__ import annotations

import time
from typing import Protocol


class Clock(Protocol):
    def now_ns(self) -> int: ...


class MonotonicClock:
    def now_ns(self) -> int:
        return time.monotonic_ns()


class FakeClock:
    def __init__(self, start_ns: int = 1_000_000_000) -> None:
        self.t = start_ns

    def now_ns(self) -> int:
        return self.t

    def advance_ms(self, ms: float) -> None:
        self.t += int(ms * 1_000_000)

    def set_ns(self, t: int) -> None:
        self.t = t
