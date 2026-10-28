UV ?= uv

.PHONY: setup dev verify-fast verify-release verify-web test cli-demo

setup:
	$(UV) sync --frozen
	cd web && npm ci

dev:
	UV=$(UV) ./scripts/dev.sh

verify-fast:
	$(UV) run pytest -m "not release" -q

verify-release:
	$(UV) run pytest -m release -q

verify-web:
	cd web && npx tsc -b && npx playwright test

test: verify-fast

cli-demo:
	$(UV) run python -m arena.cli tournament --ruleset practice_nlhe --players 6 --seed 1
