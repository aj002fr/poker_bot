UV ?= uv

.PHONY: setup verify-fast verify-release test cli-demo

setup:
	$(UV) sync --frozen

verify-fast:
	$(UV) run pytest -m "not release" -q

verify-release:
	$(UV) run pytest -m release -q

test: verify-fast

cli-demo:
	$(UV) run python -m arena.cli fixture G20
