"""SDK contracts, state encoding, ruleset declarations and the baseline bots."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from arena.agents.baselines import BASELINES, baseline_info, load_baseline
from arena.agents.sdk import (
    MAX_STATE_BYTES,
    BotLoadError,
    StateError,
    TableInfo,
    UnsupportedRuleset,
    clamp_raise_to,
    decode_state,
    encode_state,
    load_bot,
    require_supported,
)
from arena.agents.worker import InProcessSeat, play_table
from arena.domain.models import RULESET_IDS, LegalActions
from arena.domain.observations import Observation
from arena.table.service import seeded_table
from tests.agents.conftest import info_for, small_config

REPO = Path(__file__).resolve().parents[2]
TEMPLATE = REPO / "examples" / "template_bot.py"


def test_state_cap_is_exactly_64_kib():
    assert len(encode_state("x" * (MAX_STATE_BYTES - 2))) == MAX_STATE_BYTES
    with pytest.raises(StateError) as exc:
        encode_state("x" * (MAX_STATE_BYTES - 1))
    assert exc.value.code == "state_too_large"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), {1, 2}, object(), b"bytes"])
def test_state_must_be_json(bad):
    with pytest.raises(StateError) as exc:
        encode_state(bad)
    assert exc.value.code == "state_invalid"


def test_state_is_canonical_and_round_trips_to_what_the_bot_sees():
    raw = encode_state({"b": [1, 2.5, None], "a": (1, 2)})
    assert raw == b'{"a":[1,2],"b":[1,2.5,null]}'
    assert decode_state(raw) == {"a": [1, 2], "b": [1, 2.5, None]}
    assert decode_state(encode_state({3: "int-key"})) == {"3": "int-key"}
    with pytest.raises(StateError):
        encode_state({3: "mixed", "a": "keys"})


def test_load_bot_requires_make_bot_and_rulesets(tmp_path):
    (tmp_path / "none.py").write_text("x = 1\n")
    with pytest.raises(BotLoadError):
        load_bot(tmp_path / "none.py")
    (tmp_path / "norules.py").write_text(
        "from arena.agents.sdk import ExplicitStateBot\n"
        "class B(ExplicitStateBot):\n    pass\n"
        "def make_bot():\n    return B()\n")
    with pytest.raises(BotLoadError):
        load_bot(tmp_path / "norules.py")
    (tmp_path / "badrules.py").write_text(
        "from arena.agents.sdk import ExplicitStateBot\n"
        "class B(ExplicitStateBot):\n    rulesets = ('limit_holdem',)\n"
        "def make_bot():\n    return B()\n")
    with pytest.raises(BotLoadError):
        load_bot(tmp_path / "badrules.py")
    (tmp_path / "notabot.py").write_text("def make_bot():\n    return 42\n")
    with pytest.raises(BotLoadError):
        load_bot(tmp_path / "notabot.py")


def test_unsupported_ruleset_rejected_before_table_in_process():
    bot, info = load_baseline("nlhe_heuristic")
    assert info.rulesets == ("practice_nlhe",)
    with pytest.raises(UnsupportedRuleset):
        require_supported(info, "uncertainty_plo4")
    config = small_config(3, ruleset_id="uncertainty_plo4")
    with pytest.raises(UnsupportedRuleset):
        InProcessSeat(0, bot, info, info_for(config, 0), "0" * 64).start()


def test_clamp_raise_to():
    legal = LegalActions(fold=True, check=False, call_amount=20, can_raise=True, min_full_raise_to=40,
                         max_raise_to=500, short_all_in_raise_to=None, all_in_raise_to=500, pot_raise_to=70)
    assert clamp_raise_to(legal, 10).amount == 40
    assert clamp_raise_to(legal, 100).amount == 100
    assert clamp_raise_to(legal, 10_000).amount == 500
    short = LegalActions(fold=True, check=False, call_amount=20, can_raise=True, min_full_raise_to=None,
                         max_raise_to=None, short_all_in_raise_to=30, all_in_raise_to=30, pot_raise_to=None)
    assert clamp_raise_to(short, 100).amount == 30
    none = LegalActions(fold=True, check=False, call_amount=20, can_raise=False, min_full_raise_to=None,
                        max_raise_to=None, short_all_in_raise_to=None, all_in_raise_to=None, pot_raise_to=None)
    assert clamp_raise_to(none, 100) is None


def test_table_info_has_no_identity_fields():
    assert set(TableInfo.model_fields) == {"schema_version", "table_id", "hero_seat", "ruleset_id", "config",
                                           "config_hash", "seats", "decision_budget_ns", "policy_seed"}


def test_warmup_observation_fixture_matches_the_contract():
    raw = (REPO / "arena" / "agents" / "warmup_observation.json").read_bytes()
    obs = Observation.model_validate_json(raw)
    assert obs.legal is not None and len(obs.seats) == 8


def _bots_for(ruleset: str) -> list[tuple[str, object]]:
    out = []
    for name in sorted(BASELINES):
        if ruleset in baseline_info(name).rulesets:
            out.append((name, BASELINES[name]))
    return out + [("template_bot", TEMPLATE)]


@pytest.mark.parametrize("ruleset", RULESET_IDS)
def test_baselines_and_template_play_legal_actions(ruleset):
    bots = _bots_for(ruleset)
    for players in (2, 5, 8):
        config = small_config(players, ruleset_id=ruleset, hand_cap=12)
        seats = {}
        for k, s in enumerate(config.resolved_seats):
            name, path = bots[k % len(bots)]
            bot, info = load_bot(path, module_name=f"arena_sdk_test_{name}_{s}")
            seats[s] = InProcessSeat(s, bot, info, info_for(config, s, seed=f"{ruleset}-{players}"), "0" * 64)
            seats[s].start()
        table = seeded_table(config, f"sdk-{ruleset}-{players}", table_id="t_test")
        play_table(table, seats)
        assert table.complete
        for s, agent in seats.items():
            assert sum(agent.stats.fallbacks.values()) == 0, (s, dict(agent.stats.fallbacks))
        assert not any(e.type == "ActionFallback" for e in table.log)


def test_heuristic_is_nlhe_only_and_others_support_all():
    assert baseline_info("nlhe_heuristic").rulesets == ("practice_nlhe",)
    for name in ("check_fold", "calling_station", "legal_random"):
        assert set(baseline_info(name).rulesets) == set(RULESET_IDS)
    assert {"check_fold", "calling_station", "legal_random", "nlhe_heuristic"} <= set(BASELINES)


def test_baselines_import_only_the_sdk():
    for path in [*BASELINES.values(), TEMPLATE]:
        imports = [line for line in path.read_text().splitlines() if line.startswith(("import ", "from "))]
        for line in imports:
            assert line.startswith(("from arena.agents.sdk", "import hashlib", "from collections")), (path, line)


def test_template_policy_randomness_comes_from_state():
    bot, info = load_bot(TEMPLATE, "arena_template_rng")
    config = small_config(3)
    state = bot.initial_state(info_for(config, 0, seed="abc"))
    assert state["seed"] == "abc-0" and state["draws"] == 0
    assert json.loads(encode_state(state)) == state
