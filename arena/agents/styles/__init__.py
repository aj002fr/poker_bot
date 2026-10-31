"""Opponent styles modelled on the playing styles in *Harrington on Hold'em, Vol. I*.

Each style is a small bot file (``rock.py``, ``aggressive.py``, ...) whose ``make_bot()``
returns an :class:`~arena.agents.styles.engine.StyleBot`. The shared rule engine, charts
and evaluator are trusted arena code: sandboxed workers import them before the sandbox
is applied, the same way they import the SDK, so a staged style file still loads.

Profiles (description, book reference, expected stats, counter-strategy) live in
:mod:`arena.agents.styles.profiles`.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

from arena.agents.styles.engine import STYLE_IDS
from arena.agents.styles.profiles import POLICY_PREFIX, PROFILES, RANDOM_POLICY

STYLE_DIR = Path(__file__).resolve().parent
STYLES: dict[str, Path] = {sid: STYLE_DIR / f"{sid}.py" for sid in STYLE_IDS}
ALIASES = {"superaggressive": "super_aggressive", "super": "super_aggressive",
           "weaktight": "weak_tight", "scared_money": "weak_tight", "conservative": "rock"}
SUPPORT_FILES = ("engine.py", "charts.py", "handeval.py", "profiles.py", "__init__.py")

__all__ = ["ALIASES", "POLICY_PREFIX", "PROFILES", "RANDOM_POLICY", "STYLES", "STYLE_IDS", "resolve_style",
           "style_artifact_hash"]


def resolve_style(name: str) -> str | None:
    """A style id from ``rock``, ``style_rock`` or an alias such as ``conservative``."""
    name = name.strip().lower().removeprefix("style:").removeprefix(POLICY_PREFIX)
    name = ALIASES.get(name, name)
    return name if name in STYLES else None


@cache
def style_artifact_hash(style: str) -> str:
    """Covers the style file and the shared engine, so an engine change changes every style's hash."""
    import hashlib

    h = hashlib.sha256(b"arena-style-artifact-v1")
    for p in (STYLES[style], *(STYLE_DIR / f for f in SUPPORT_FILES)):
        data = p.read_bytes()
        h.update(p.name.encode() + len(data).to_bytes(8, "big") + data)
    return h.hexdigest()
