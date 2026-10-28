"""Built-in baseline bots. Each is a self-contained bot file that imports only the SDK,
so the same file runs in-process or staged into a sandboxed worker.

``nlhe_heuristic`` is a simple hand-written heuristic for ``practice_nlhe`` only. None
of these are strong players; beating them says little about poker strength.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

from arena.agents.sdk import BotInfo, load_bot

BASELINE_DIR = Path(__file__).resolve().parent
BASELINES: dict[str, Path] = {
    "check_fold": BASELINE_DIR / "check_fold.py",
    "calling_station": BASELINE_DIR / "calling_station.py",
    "legal_random": BASELINE_DIR / "legal_random.py",
    "nlhe_heuristic": BASELINE_DIR / "nlhe_heuristic.py",
}


def baseline_path(name: str) -> Path:
    try:
        return BASELINES[name]
    except KeyError:
        raise KeyError(f"unknown baseline {name!r}; choose from {sorted(BASELINES)}") from None


def load_baseline(name: str) -> tuple[object, BotInfo]:
    """A fresh in-process instance (trusted code only)."""
    return load_bot(baseline_path(name), module_name=f"arena_baseline_{name}")


@cache
def baseline_info(name: str) -> BotInfo:
    return load_baseline(name)[1]
