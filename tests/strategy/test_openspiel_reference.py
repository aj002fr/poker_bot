"""Our CFR variants reproduce OpenSpiel's frozen Kuhn/Leduc checkpoints (no OpenSpiel import)."""

import pytest

from strategy.solve import PRESETS
from tests.strategy.helpers import reference, reference_mismatch


def test_reference_file_records_its_provenance():
    ref = reference()
    assert ref["open_spiel"] == "2.0.2"
    assert ref["games"]["leduc"]["parameters"] == {
        "action_mapping": "False", "players": "2", "starting_player": "0", "suit_isomorphism": "False",
    }
    assert set(ref["games"]["kuhn"]["solvers"]) == set(PRESETS)


@pytest.mark.parametrize("solver", sorted(PRESETS))
@pytest.mark.parametrize("game", ["kuhn", "leduc"])
def test_iterates_match_openspiel(game, solver):
    assert reference_mismatch(game, solver) < 1e-8
