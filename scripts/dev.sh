#!/usr/bin/env bash
# Starts the API (uvicorn) and the Vite table together; Ctrl-C stops both.
# ARENA_STORE=postgres keeps tables in PostgreSQL (see `make dev-postgres`); default is in-memory.
set -euo pipefail
cd "$(dirname "$0")/.."

API_PORT="${ARENA_API_PORT:-8317}"
WEB_PORT="${ARENA_WEB_PORT:-5317}"
UV="${UV:-uv}"

if [ ! -d web/node_modules ]; then
  (cd web && npm ci)
fi

trap 'kill 0' EXIT INT TERM

"$UV" run uvicorn --factory arena.api.app:create_app --host 127.0.0.1 --port "$API_PORT" &
(cd web && ARENA_API_PORT="$API_PORT" ARENA_WEB_PORT="$WEB_PORT" npx vite) &

echo "API  http://127.0.0.1:${API_PORT}  (store: ${ARENA_STORE:-memory})"
echo "Web  http://127.0.0.1:${WEB_PORT}"
wait
