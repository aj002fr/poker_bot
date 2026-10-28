"""Operator credential: registers bot files, seats them and launches benchmark runs.

One bearer token per server process. It comes from ``ARENA_OPERATOR_TOKEN`` or is
generated at start-up and written (mode 0600) to ``artifacts/operator-token`` so a
local operator can paste it into the UI. It is never returned by any endpoint.
"""

from __future__ import annotations

import hmac
import os
import secrets
from pathlib import Path

DEFAULT_TOKEN_FILE = Path("artifacts/operator-token")


def operator_token_from_env(token_file: Path = DEFAULT_TOKEN_FILE) -> str:
    token = os.environ.get("ARENA_OPERATOR_TOKEN")
    if token:
        return token
    token = secrets.token_urlsafe(24)
    try:
        token_file.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(token_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as fh:
            fh.write(token + "\n")
    except OSError:
        pass
    return token


def is_operator(expected: str | None, presented: str | None) -> bool:
    return bool(expected and presented and hmac.compare_digest(expected.encode(), presented.encode()))
