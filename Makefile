UV ?= uv

.PHONY: setup dev dev-postgres db-up db-down verify-fast verify-postgres verify-release verify-web test cli-demo

setup:
	$(UV) sync --frozen
	cd web && npm ci

dev:
	UV=$(UV) ./scripts/dev.sh

db-up:
	docker compose up -d --wait postgres

db-down:
	docker compose down

dev-postgres: db-up
	ARENA_STORE=postgres UV=$(UV) ./scripts/dev.sh

verify-fast:
	$(UV) run pytest -m "not release" -q

verify-postgres: db-up
	ARENA_REQUIRE_POSTGRES=1 $(UV) run pytest tests/persistence -q
	ARENA_API_TEST_STORE=postgres $(UV) run pytest tests/api -q

verify-release:
	$(UV) run pytest -m release -q

verify-web:
	cd web && npx tsc -b && npx playwright test

test: verify-fast

cli-demo:
	$(UV) run python -m arena.cli tournament --ruleset practice_nlhe --players 6 --seed 1
