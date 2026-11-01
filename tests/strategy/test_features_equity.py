import pytest

from strategy.features.cards import capabilities, card_ints, made_hand, postflop_key, texture
from strategy.features.equity import EquityTables, heuristic_percentile
from strategy.tables import build, store
from tests.strategy.feature_support import RULESETS


def c(*names: str) -> tuple[int, ...]:
    return card_ints(names)


@pytest.mark.parametrize("ruleset_id", RULESETS)
def test_committed_tables_are_fresh_and_verified(ruleset_id: str) -> None:
    for name, _ in build.tables(ruleset_id):
        if name in build.OPTIONAL and store.load(name) is None:
            continue
        prov, _ = store.load(name)  # verifies data_sha256
        assert not store.stale(name), f"{name} is stale: python -m strategy.tables.build"
        assert prov["generator"].startswith("strategy.tables.") and prov["params"]


def test_nlhe_preflop_lookup() -> None:
    eq = EquityTables(capabilities("practice_nlhe"))
    aa, trash = eq.preflop(c("Ah", "As"), 1), eq.preflop(c("7d", "2c"), 1)
    assert aa.source == "table" and 0.83 < aa.equity < 0.87 and aa.bucket == 9
    assert trash.equity < 0.37 and trash.bucket <= 1 and aa.percentile > trash.percentile
    assert eq.preflop(c("Ah", "As"), 3).equity < aa.equity
    assert eq.preflop(c("Ah", "As"), 12).equity == eq.preflop(c("Ah", "As"), 7).equity
    assert sum(eq.bucket_combos) == 1326


@pytest.mark.parametrize("ruleset_id", RULESETS)
def test_postflop_lookup_orders_made_hands(ruleset_id: str) -> None:
    caps = capabilities(ruleset_id)
    eq = EquityTables(caps)
    board = c("Kd", "8s", "3c")
    hole_strong = c("Kh", "Ks") if caps.hole_count == 2 else c("Kh", "Ks", "7d", "6d")
    hole_weak = c("Jh", "Tc") if caps.hole_count == 2 else c("Jh", "Tc", "9c", "6h")
    vals = []
    for hole in (hole_strong, hole_weak):
        m = made_hand(hole, board, caps)
        key = postflop_key(1, m.level, 0, texture(board, caps).key)
        vals.append(eq.postflop(key, m.level, 0, 1))
    assert vals[0].source != "prior" and vals[0].equity > vals[1].equity
    three = eq.postflop(key, 0, 0, 3).equity
    assert eq.postflop(key, 0, 0, 5).equity <= three <= eq.postflop(key, 0, 0, 1).equity


def test_missing_tables_fall_back_to_a_labelled_prior(tmp_path) -> None:
    eq = EquityTables(capabilities("uncertainty_plo4"), str(tmp_path))
    assert not eq.has_preflop and eq.bucket_combos is None
    strong = eq.preflop(c("Ah", "As", "Kh", "Ks"), 1)
    weak = eq.preflop(c("7d", "2c", "9h", "4s"), 1)
    assert strong.source == "prior" and strong.percentile > weak.percentile and strong.bucket > weak.bucket
    assert eq.postflop(0, 5, 0, 2).source == "prior"
    assert heuristic_percentile(c("Ah", "As")) > heuristic_percentile(c("7d", "2c"))
