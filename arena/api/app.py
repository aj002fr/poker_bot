"""HTTP and WebSocket transport for the playable table (spec 9.1).

Seat and host credentials travel in ``Authorization: Bearer ...`` headers, and over
WebSocket in the first message, never in URLs. HTTP is the only mutation path; the
stream only publishes committed projections. Run with::

    uv run uvicorn --factory arena.api.app:create_app --port 8317

``ARENA_STORE=postgres`` (with ``ARENA_DATABASE_URL``) keeps tables in PostgreSQL;
the default ``memory`` store lasts as long as the process.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import threading
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from arena.api.clock import Clock
from arena.api.manager import (
    ApiError,
    CreateTableRequest,
    JoinRequest,
    Principal,
    TableManager,
    policy_catalog,
)
from arena.domain.models import RULESET_IDS
from arena.persistence.store import MemoryStore, Store
from arena.table.projections import ProjectionDenied

log = logging.getLogger("arena.api")

DEFAULT_ORIGINS = ("http://localhost:5317", "http://127.0.0.1:5317")
WS_AUTH_TIMEOUT_S = 5.0
DEFAULT_DATABASE_URL = "postgresql://arena:arena@127.0.0.1:54329/arena"


def store_from_env() -> Store:
    kind = os.environ.get("ARENA_STORE", "memory")
    if kind == "memory":
        return MemoryStore()
    if kind == "postgres":
        from arena.persistence.postgres import PostgresStore

        return PostgresStore(os.environ.get("ARENA_DATABASE_URL", DEFAULT_DATABASE_URL))
    raise ValueError(f"ARENA_STORE must be memory or postgres, not {kind!r}")


def _bearer(request: Request) -> str | None:
    header = request.headers.get("authorization")
    if not header:
        return None
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise ApiError(401, "invalid_credential", "use Authorization: Bearer <credential>")
    return token.strip()


def create_app(
    manager: TableManager | None = None,
    *,
    clock: Clock | None = None,
    allowed_origins: tuple[str, ...] | None = None,
    background_pump: bool = True,
    pump_interval_s: float = 0.02,
) -> FastAPI:
    if manager is None:
        manager = TableManager(
            clock,
            bot_delay_ms=int(os.environ.get("ARENA_BOT_DELAY_MS", "700")),
            hand_pause_ms=int(os.environ.get("ARENA_HAND_PAUSE_MS", "1500")),
            store=store_from_env(),
            writer_id=os.environ.get("ARENA_WRITER_ID", "local-writer"),
        )
    if allowed_origins is None:
        extra = tuple(o for o in os.environ.get("ARENA_ALLOWED_ORIGINS", "").split(",") if o)
        allowed_origins = DEFAULT_ORIGINS + extra

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        stop = threading.Event()
        thread = None
        if background_pump:
            def pump_forever() -> None:
                while not stop.wait(pump_interval_s):
                    try:
                        manager.pump()
                    except Exception:  # noqa: BLE001 - the pump must outlive any one failure
                        log.exception("pump iteration failed")

            thread = threading.Thread(target=pump_forever, name="arena-pump", daemon=True)
            thread.start()
        try:
            yield
        finally:
            stop.set()
            if thread is not None:
                await run_in_threadpool(thread.join, 5.0)
            await run_in_threadpool(manager.close)

    app = FastAPI(title="arena practice table", version="0.1.0", lifespan=lifespan)
    app.state.manager = manager

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(exc.body(), status_code=exc.status)

    @app.exception_handler(ProjectionDenied)
    async def _denied(request: Request, exc: ProjectionDenied) -> JSONResponse:
        return JSONResponse({"error": {"code": "projection_denied", "message": str(exc)}}, status_code=403)

    @app.exception_handler(RequestValidationError)
    async def _invalid(request: Request, exc: RequestValidationError) -> JSONResponse:
        details = [{"loc": [str(p) for p in e.get("loc", ())], "msg": e.get("msg", "")} for e in exc.errors()]
        return JSONResponse(
            {"error": {"code": "invalid_request", "message": "request rejected", "details": details}},
            status_code=422,
        )

    def principal_for(request: Request, table_id: str) -> Principal:
        return manager.authenticate(table_id, _bearer(request))

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"ok": True, "compatibility_status": "practice_assumptions"}

    @app.get("/catalog")
    async def catalog() -> dict[str, Any]:
        return {"rulesets": list(RULESET_IDS), "policies": policy_catalog(), "max_seats": 8, "min_seats": 2}

    @app.post("/configs/resolve")
    def resolve(body: CreateTableRequest) -> dict[str, Any]:
        return manager.resolve_config(body)

    @app.post("/tables", status_code=201)
    def create_table(body: CreateTableRequest) -> dict[str, Any]:
        return manager.create_table(body)

    @app.post("/tables/{table_id}/join")
    def join(table_id: str, body: JoinRequest) -> dict[str, Any]:
        return manager.join(table_id, body.seat)

    @app.post("/tables/{table_id}/start")
    def start(table_id: str, request: Request) -> dict[str, Any]:
        return manager.start(table_id, principal_for(request, table_id))

    @app.get("/tables/{table_id}")
    def get_table(table_id: str, request: Request) -> dict[str, Any]:
        return manager.snapshot(table_id, principal_for(request, table_id))

    @app.post("/tables/{table_id}/actions")
    async def act(table_id: str, request: Request) -> dict[str, Any]:
        principal = await run_in_threadpool(principal_for, request, table_id)
        raw = await request.body()
        received_ns = manager.register_action(table_id)
        return await run_in_threadpool(manager.submit_action, table_id, principal, raw, received_ns,
                                       registered=True)

    @app.get("/tables/{table_id}/history")
    def history(table_id: str, request: Request, perspective: str | None = None) -> dict[str, Any]:
        return manager.history(table_id, principal_for(request, table_id), perspective)

    @app.websocket("/tables/{table_id}/stream")
    async def stream(ws: WebSocket, table_id: str) -> None:
        origin = ws.headers.get("origin")
        if origin is not None and origin not in allowed_origins:
            await ws.close(code=4403)
            return
        await ws.accept()
        try:
            hello = json.loads(await asyncio.wait_for(ws.receive_text(), WS_AUTH_TIMEOUT_S))
            if not isinstance(hello, dict) or hello.get("type") != "auth":
                raise ApiError(401, "auth_required", "first message must be {type: auth, token}")
            token = hello.get("token")
            principal = manager.authenticate(table_id, token if isinstance(token, str) else None)
            sub = await run_in_threadpool(manager.subscribe, table_id, principal, asyncio.get_running_loop())
        except ApiError as exc:
            await ws.send_json({"type": "error", **exc.body()})
            await ws.close(code=4000 + exc.status)
            return
        except (asyncio.TimeoutError, ValueError, WebSocketDisconnect):
            with contextlib.suppress(Exception):
                await ws.close(code=4401)
            return

        async def drain_client() -> None:
            while True:
                await ws.receive_text()

        reader = asyncio.create_task(drain_client())
        try:
            while True:
                frame_task = asyncio.create_task(sub.next())
                done, _ = await asyncio.wait({frame_task, reader}, return_when=asyncio.FIRST_COMPLETED)
                if reader in done:
                    frame_task.cancel()
                    break
                await ws.send_json(frame_task.result())
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            reader.cancel()
            await run_in_threadpool(manager.unsubscribe, table_id, sub)

    return app
