UV ?= uv

.PHONY: setup dev dev-postgres db-up db-down verify-fast verify-postgres verify-release verify-release-status \
	verify-release-report verify-release-smoke verify-web test cli-demo

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

# Every section 13 gate as checkpointed shards; resumes, then writes artifacts/test-report.{json,md}.
# Exit 1 unless every applicable gate passed. JOBS defaults to the CPU count; see docs/release-verification.md.
JOBS ?= $(shell nproc 2>/dev/null || echo 4)
verify-release:
	$(UV) run python -m tests.release run --jobs $(JOBS) --exclusive-last

verify-release-status:
	$(UV) run python -m tests.release status

verify-release-report:
	$(UV) run python -m tests.release report

# The older marker-based release smoke tests (a subset of the gates above).
verify-release-smoke:
	$(UV) run pytest -m release -q

verify-web:
	cd web && npx tsc -b && npx playwright test

test: verify-fast

cli-demo:
	$(UV) run python -m arena.cli tournament --ruleset practice_nlhe --players 6 --seed 1
