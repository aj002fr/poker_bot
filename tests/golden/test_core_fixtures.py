import pytest

from tests.support.fixtures import load_fixtures, run_fixture

CORE = load_fixtures("core")


@pytest.mark.parametrize("fixture", CORE, ids=[f["id"] for f in CORE])
def test_core_fixture(fixture):
    run_fixture(fixture)
