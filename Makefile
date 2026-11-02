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

# Stage 3 endgame (docs: strategy/tables/README.md). Heavy: about 30 minutes on 8 cores.
.PHONY: tables-showdown tables-endgame gates-endgame value-rollouts value-fit lab-endgame
ENDGAME_RULESETS ?= practice_nlhe uncertainty_shortdeck
ENDGAME_JOBS ?= $(JOBS)

# Only the showdown generator needs numba (the evaluator kernel); everything else runs without it.
tables-showdown:
	for r in $(ENDGAME_RULESETS); do $(UV) run $(TRAIN) python -m strategy.tables.showdown --ruleset $$r --jobs $(ENDGAME_JOBS) || exit 1; done

tables-endgame:
	for r in $(ENDGAME_RULESETS); do for m in points chips chart; do \
		$(UV) run python -m strategy.tables.endgame --ruleset $$r --mode $$m --jobs $(ENDGAME_JOBS) || exit 1; done; done

gates-endgame:
	$(UV) run python -m strategy.endgame.gates --jobs $(ENDGAME_JOBS)

value-rollouts:
	for r in $(ENDGAME_RULESETS); do $(UV) run python -m strategy.value.model rollouts --ruleset $$r --jobs $(ENDGAME_JOBS) || exit 1; done

value-fit:
	for r in $(ENDGAME_RULESETS); do $(UV) run python -m strategy.value.model fit --ruleset $$r || exit 1; done

lab-endgame:
	for s in endgame-points-vs-chips endgame-points-vs-chart endgame-points-vs-chips-shortdeck endgame-points-vs-none \
		endgame-points-vs-chips-confirm; do \
		$(UV) run python -m strategy.cli experiment $$s --jobs $(LAB_JOBS) || exit 1; done

# Baseline policy search (Stage 1 B/D; docs/baseline-policy.md). Long runs: never in CI.
# Runs land in artifacts/search/<run_id>/ and resume where they stopped. nlhe-cem-v2 starts from
# nlhe-cem-v1's state, so run SEARCH_CONFIG=strategy/search/configs/nlhe-cem.json first.
.PHONY: search-baseline search-status package-baseline validate-baseline validate-baseline-variants gate-baseline \
	gate-control bench-policy
SEARCH_CONFIG ?= strategy/search/configs/nlhe-cem-v2.json
SEARCH_RUN ?= artifacts/search/nlhe-cem-v2
SEARCH_JOBS ?= $(shell n=$$(nproc 2>/dev/null || echo 4); echo $$((n > 1 ? n - 1 : 1)))
BASELINE_CANDIDATE ?= $(SEARCH_RUN)/incumbent.json
GATE_DECISIONS ?= 50000

search-baseline:
	$(UV) run python -m strategy.search run $(SEARCH_CONFIG) --jobs $(SEARCH_JOBS)

search-status:
	$(UV) run python -m strategy.search status $(SEARCH_RUN)

package-baseline:
	$(UV) run python -m strategy.policy.package --params $(BASELINE_CANDIDATE) --out artifacts/bots/baseline-candidate \
		--name strategy_baseline

# Paired promotion check on the validation partition (in-process; official points; phases.json added).
validate-baseline:
	$(UV) run python -m strategy.search.validate baseline-validation-nlhe --candidate $(BASELINE_CANDIDATE) \
		--jobs $(LAB_JOBS)

# Rehearsal: the same NLHE-searched vector on short deck and PLO4 (no claims, discovery stage).
validate-baseline-variants:
	$(UV) run python -m strategy.search.validate baseline-rehearsal-variants --candidate $(BASELINE_CANDIDATE) \
		--jobs $(LAB_JOBS)

# The packaged bot through the real sandbox worker, one table at a time (run on an otherwise idle machine).
gate-baseline: package-baseline
	$(UV) run python -m strategy.policy.gate --bundle artifacts/bots/baseline-candidate --decisions $(GATE_DECISIONS) \
		--out artifacts/gate/baseline-candidate.json

# The same gate with a trivial bot: the machine's own timeout floor.
gate-control:
	$(UV) run python -m strategy.policy.gate --bundle arena/agents/baselines/calling_station.py \
		--decisions $(GATE_DECISIONS) --out artifacts/gate/control-calling-station.json

bench-policy: package-baseline
	$(UV) run python -m strategy.policy.bench --bundle artifacts/bots/baseline-candidate --tables 60 \
		--out artifacts/gate/baseline-bench.json
