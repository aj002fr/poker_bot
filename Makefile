UV ?= uv

.PHONY: setup dev dev-postgres db-up db-down verify-fast verify-postgres verify-release verify-release-status \
	verify-release-report verify-release-smoke verify-web test cli-demo \
	verify-solve-slow verify-oracle gates-stage2 bench-solve

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

# Solver test modules (anything importing numba, strategy.solve or strategy.games) are not
# collected unless numba is installed; verify-strategy runs them.
verify-fast:
	$(UV) run pytest -m "not release and not slow and not oracle" -q

# The offline solvers need numba from the opt-in `train` group (no Intel-macOS wheels).
# ARENA_REQUIRE_NUMBA=1 makes pytest fail at startup instead of dropping the solver tests.
TRAIN = --group train
REQUIRE_NUMBA = ARENA_REQUIRE_NUMBA=1

# Stage 2 solver gates at full resolution, long ES-MCCFR runs and sampled-vs-full agreement.
verify-solve-slow:
	$(REQUIRE_NUMBA) $(UV) run $(TRAIN) pytest -m slow tests/strategy -q

# Kuhn/Leduc cross-checks against OpenSpiel (dev-only dependency group).
verify-oracle:
	$(REQUIRE_NUMBA) $(UV) run $(TRAIN) --group oracle pytest -m oracle tests/strategy -q

gates-stage2:
	$(UV) run $(TRAIN) python -m strategy.games.fixtures --resolution 200

bench-solve:
	$(UV) run $(TRAIN) python -m strategy.solve.bench

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

# Strategy lab (docs/strategy-evaluation.md). Runs land in artifacts/lab/<experiment>/<manifest_id>/;
# a manifest that already ran is not re-run.
.PHONY: lab-conformance lab-report lab-experiment lab-null lab-null-variants lab-null-synthetic verify-strategy
LAB_JOBS ?= $(JOBS)
LAB_SPEC ?= lab-smoke

lab-conformance:
	$(UV) run python -m strategy.cli conformance

lab-report:
	$(UV) run python -m strategy.cli experiment lab-smoke --jobs $(LAB_JOBS)

lab-experiment:
	$(UV) run python -m strategy.cli experiment $(LAB_SPEC) --jobs $(LAB_JOBS)

lab-null:
	$(UV) run python -m strategy.cli experiment null-aa-nlhe --jobs $(LAB_JOBS)

lab-null-variants:
	$(UV) run python -m strategy.cli experiment null-aa-variants --jobs $(LAB_JOBS)

lab-null-synthetic:
	$(UV) run python -m strategy.cli null-synthetic --replicates 4000 --blocks 20

verify-strategy:
	$(REQUIRE_NUMBA) $(UV) run $(TRAIN) pytest tests/strategy -m "not slow and not oracle" -q
