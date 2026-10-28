"""Rows written by the command writer and read back by recovery.

Every JSON-like field holds plain JSON values (string map keys included), so a
row read from either store compares equal to the same row read from the other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

TableStatus = Literal["waiting", "running", "complete", "halted"]


@dataclass
class TableRow:
    table_id: str
    config: dict[str, Any]
    config_hash: str
    deck_seed: str | None
    bot_seed: str
    host_token_hash: str
    status: TableStatus
    version: int
    state_version: int = 0
    event_count: int = 0
    scheduler: dict[str, Any] | None = None
    """The pending decision as of the last commit or heartbeat: decision_id, seat, kind, remaining_ms."""
    halted_reason: str | None = None


@dataclass
class ParticipantRow:
    seat: int
    kind: str
    policy: str | None
    artifact_hash: str | None
    claimed: bool
    token_hash: str | None


@dataclass
class HandRow:
    hand_id: str
    hand_number: int
    start_seq: int
    start_hash: str
    end_seq: int | None = None
    final_stacks: dict[str, int] | None = None


@dataclass
class SnapshotRow:
    hand_number: int
    seq: int
    start_hash: str
    point: dict[str, Any]
    engine: dict[str, Any]


@dataclass
class ReceiptRow:
    action_id: str
    seat: int
    decision_id: str
    fingerprint: str
    body: dict[str, Any]
    version: int


@dataclass
class CheckpointRow:
    seat: int
    policy: str
    artifact_hash: str
    rng_state: dict[str, Any]
    """The seat's committed bot state: ``{"mode": "explicit", "state": <canonical JSON>}`` or ``{"mode": "instance"}``."""
    decisions: int
    version: int


@dataclass
class AwardRow:
    seat: int
    positions: list[int]
    points: float
    method: str
    hand_number: int


@dataclass
class PauseRow:
    reason: str
    decision_id: str | None
    remaining_ms: int | None
    paused_at: float
    """Wall-clock epoch seconds (audit only; deadlines run on the monotonic clock)."""
    resumed_at: float | None = None
    pause_id: int | None = None


@dataclass
class CommitBatch:
    """Everything one command changes, written in one transaction or not at all."""

    table: TableRow
    expected_version: int | None
    """The version this writer last committed; ``None`` creates the table."""
    kind: str
    events: list[dict[str, Any]] = field(default_factory=list)
    participants: list[ParticipantRow] = field(default_factory=list)
    hands: list[HandRow] = field(default_factory=list)
    snapshots: list[SnapshotRow] = field(default_factory=list)
    receipt: ReceiptRow | None = None
    checkpoints: list[CheckpointRow] = field(default_factory=list)
    awards: list[AwardRow] = field(default_factory=list)
    pauses: list[PauseRow] = field(default_factory=list)
    """New pauses are inserted; rows with a ``pause_id`` are updated (to close them)."""


@dataclass
class StoredTable:
    table: TableRow
    owner_id: str | None
    participants: list[ParticipantRow]
    events: list[dict[str, Any]]
    snapshot: SnapshotRow | None
    """The newest hand-start snapshot."""
    receipts: list[ReceiptRow]
    checkpoints: list[CheckpointRow]
    awards: list[AwardRow]
    open_pause: PauseRow | None
    scheduler_age_s: float | None
    """Seconds since the scheduler was last written; how stale ``remaining_ms`` can be."""
    unpublished: list[int]
    """Outbox versions committed but not yet broadcast."""
