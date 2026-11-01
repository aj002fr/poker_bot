import importlib.util

# The solver kernels need numba from the opt-in `train` dependency group, which has no
# Intel-macOS wheels. Without it these modules cannot import; `make verify-strategy` runs them.
NEEDS_NUMBA = [
    "test_averaging.py",
    "test_bench.py",
    "test_best_response.py",
    "test_compile.py",
    "test_fixture_gates.py",
    "test_mutations.py",
    "test_openspiel_oracle.py",
    "test_openspiel_reference.py",
    "test_reach_and_card_removal.py",
    "test_regret_matching.py",
    "test_sampling.py",
]

HAVE_NUMBA = importlib.util.find_spec("numba") is not None
collect_ignore = [] if HAVE_NUMBA else NEEDS_NUMBA


def pytest_report_header(config):
    if not HAVE_NUMBA:
        return f"numba not installed: {len(NEEDS_NUMBA)} solver test modules not collected (make verify-strategy)"
