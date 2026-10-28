"""Versioned event schema: every logged event validates; bad payloads and versions are rejected."""

import json

import pytest
from pydantic import ValidationError

from arena.domain.events import EVENT_MODELS, EVENT_SCHEMA_VERSION, EventEnvelope, UnknownEventType
from arena.history.canonical import TableHistory, export_history
from arena.table.projections import OWNER_FIELDS, PUBLIC_FIELDS
from tests.support.tables import RULESETS, play_table


@pytest.fixture(scope="module")
def history():
    return export_history(play_table("practice_nlhe", 6, "schema", fallback_every=7))


def test_every_event_type_has_an_explicit_projection_rule():
    assert set(PUBLIC_FIELDS) == set(EVENT_MODELS)
    for event_type, fields in PUBLIC_FIELDS.items():
        if fields is not None:
            assert set(fields) <= set(EVENT_MODELS[event_type].model_fields), event_type
    for event_type, fields in OWNER_FIELDS.items():
        assert set(fields) <= set(EVENT_MODELS[event_type].model_fields)


def test_log_keys_and_order(history):
    events = history.events
    assert events[0].type == "TableStarted" and events[-1].type == "TableScored"
    assert [e.seq for e in events] == list(range(len(events)))
    assert all(e.schema_version == EVENT_SCHEMA_VERSION and e.table_id == history.table_id for e in events)
    types = {e.type for e in events}
    for required in ("HandStarted", "BlindPosted", "HoleDealt", "ActionApplied", "ActionFallback", "BoardDealt",
                     "PotAwarded", "PlayerEliminated", "TableScored", "DecisionRequested", "CardBurned"):
        assert required in types
    for e in events:
        assert (e.hand_id is None) == (e.type in ("TableStarted", "TableScored"))


@pytest.mark.parametrize("ruleset", RULESETS)
def test_explicit_decks_persisted_and_dealt(ruleset):
    table = play_table(ruleset, 4, "decks")
    starts = [e for e in table.log if e.type == "HandStarted"]
    assert len(starts) == len(table.hands)
    for start, record in zip(starts, table.hands):
        assert tuple(start.data["deck"]) == record.deck
        dealt = [e.data["card"] for e in table.log if e.hand_id == start.hand_id and e.type == "HoleDealt"]
        assert dealt == list(record.deck[: len(dealt)])


def test_canonical_dump_is_stable(history):
    text = history.dumps()
    again = TableHistory.loads(text)
    assert again == history
    assert again.dumps() == text
    assert text.isascii() and "NaN" not in text


def _obj(history):
    return json.loads(history.dumps())


def test_rejects_unknown_versions_and_formats(history):
    for key, value in (("schema_version", 2), ("format", "other"), ("event_schema_version", 2)):
        obj = _obj(history)
        obj[key] = value
        with pytest.raises((ValueError, ValidationError)):
            TableHistory.loads(json.dumps(obj))
    obj = _obj(history)
    obj["events"][3]["schema_version"] = 2
    with pytest.raises(ValueError):
        TableHistory.loads(json.dumps(obj))


def test_rejects_unknown_event_types_and_bad_payloads(history):
    obj = _obj(history)
    obj["events"][2]["type"] = "SecretPeek"
    with pytest.raises(UnknownEventType):
        TableHistory.loads(json.dumps(obj))
    blind = next(i for i, e in enumerate(obj["events"]) if e["type"] == "BlindPosted")
    for bad in ({"seat": 1, "amount": True}, {"seat": 1, "amount": 1.5}, {"seat": 1, "amount": 10, "extra": 1},
                {"seat": "1", "amount": 10}, {"amount": 10}):
        obj = _obj(history)
        obj["events"][blind]["data"] = bad
        with pytest.raises((ValueError, ValidationError)):
            TableHistory.loads(json.dumps(obj))


def test_rejects_sequence_gaps_and_foreign_events(history):
    obj = _obj(history)
    del obj["events"][5]
    with pytest.raises(ValueError, match="sequence gap"):
        TableHistory.loads(json.dumps(obj))
    obj = _obj(history)
    obj["events"][5]["table_id"] = "t_other"
    with pytest.raises(ValueError, match="another table"):
        TableHistory.loads(json.dumps(obj))


def test_append_time_validation():
    with pytest.raises(UnknownEventType):
        EventEnvelope.make("t", "h", 0, "Nope", {})
    with pytest.raises(ValidationError):
        EventEnvelope.make("t", "h", 0, "BlindPosted", {"seat": 0, "amount": -0.5})


def test_each_accepted_action_appears_once_and_rejections_log_nothing():
    from arena.domain.rng import SeededRng
    from arena.table.driver import legal_random
    from arena.table.service import ActionRejected

    table = play_table("practice_nlhe", 5, "once", fallback_every=6)
    manual = [e for e in table.log if e.type == "ActionApplied" and e.data["auto"] is None]
    assert len(manual) == len(table.receipts)
    for receipt, event in zip(table.receipts, manual):
        assert (receipt.seat, receipt.action.type) == (event.data["seat"], event.data["action"]["type"])
    decisions = [e.data["decision_id"] for e in table.log if e.type == "DecisionRequested"]
    assert decisions == [r.decision_id for r in table.receipts]
    fallbacks = [e for e in table.log if e.type == "ActionFallback"]
    assert len(fallbacks) == sum(1 for r in table.receipts if r.fallback_reason)

    from arena.domain.models import TournamentConfig
    from arena.table.service import seeded_table

    live = seeded_table(TournamentConfig(initial_player_count=3), "reject")
    rng = SeededRng("reject", "p")
    for _ in range(10):
        if live.complete:
            break
        before = (len(live.log), live.state_version, live.decision_id)
        for bad in ({"type": "raise_to", "amount": 10**9}, {"type": "raise_to"}, {"type": "call", "amount": 1}):
            try:
                live.submit(live.actor_seat, bad)
            except ActionRejected:
                pass
        assert (len(live.log), live.state_version, live.decision_id) == before
        live.submit(live.actor_seat, legal_random(live.actor_seat, live.legal_actions(), rng))
