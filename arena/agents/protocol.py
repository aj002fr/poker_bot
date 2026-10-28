"""Framing between the parent and a worker over a pair of pipes.

A frame is an 8-byte header (``>II``: meta length, blob length), a JSON ``meta``
object and an opaque ``blob`` (a bot state in canonical JSON, or empty). Worker
replies are capped: a header announcing more than :data:`MAX_REPLY_META` or
:data:`MAX_REPLY_BLOB` bytes is rejected before anything else is read, and the
worker is replaced because the stream can no longer be trusted.
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass
from typing import Any

from arena.agents.sdk import MAX_STATE_BYTES

HEADER = struct.Struct(">II")
MAX_REPLY_META = 16 * 1024
MAX_REPLY_BLOB = MAX_STATE_BYTES
MAX_REQUEST_BYTES = 8 * 1024 * 1024
DETAIL_CHARS = 300


class FrameError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def encode_frame(meta: dict[str, Any] | bytes, blob: bytes = b"") -> bytes:
    raw = meta if isinstance(meta, bytes) else json.dumps(meta, separators=(",", ":")).encode()
    return HEADER.pack(len(raw), len(blob)) + raw + blob


@dataclass
class Frame:
    meta: dict[str, Any]
    blob: bytes
    size: int


class FrameReader:
    """Incremental, size-capped frame parser for the parent's non-blocking reads."""

    def __init__(self, max_meta: int = MAX_REPLY_META, max_blob: int = MAX_REPLY_BLOB) -> None:
        self.max_meta = max_meta
        self.max_blob = max_blob
        self.buf = bytearray()

    def feed(self, data: bytes) -> None:
        self.buf += data

    def next_frame(self) -> Frame | None:
        if len(self.buf) < HEADER.size:
            return None
        meta_len, blob_len = HEADER.unpack_from(self.buf)
        if meta_len > self.max_meta or blob_len > self.max_blob:
            raise FrameError("oversized_output", f"reply announces {meta_len}+{blob_len} bytes")
        total = HEADER.size + meta_len + blob_len
        if len(self.buf) < total:
            return None
        raw = bytes(self.buf[HEADER.size:HEADER.size + meta_len])
        blob = bytes(self.buf[HEADER.size + meta_len:total])
        del self.buf[:total]
        try:
            meta = json.loads(raw)
        except (UnicodeDecodeError, ValueError):
            raise FrameError("invalid_output", "reply meta is not JSON") from None
        if not isinstance(meta, dict):
            raise FrameError("invalid_output", "reply meta is not an object")
        return Frame(meta, blob, total)


def read_exact(fd: int, n: int) -> bytes:
    """Blocking read used on the worker side, where the parent is trusted."""
    chunks = []
    while n:
        chunk = os.read(fd, n)
        if not chunk:
            raise EOFError
        chunks.append(chunk)
        n -= len(chunk)
    return b"".join(chunks)


def read_frame_blocking(fd: int) -> tuple[dict[str, Any], bytes]:
    meta_len, blob_len = HEADER.unpack(read_exact(fd, HEADER.size))
    if meta_len + blob_len > MAX_REQUEST_BYTES:
        raise FrameError("oversized_request", "request too large")
    meta = json.loads(read_exact(fd, meta_len))
    blob = read_exact(fd, blob_len) if blob_len else b""
    return meta, blob


def write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        n = os.write(fd, view)
        view = view[n:]
