"""Replay frames: re-simulated from the private log, projected per authorized perspective."""

from __future__ import annotations

import json
from collections import defaultdict

from tests.api.helpers import HUMAN, auth, bot, harness


def replay(h, token=None, **params):
    return h.client.get(f"/tables/{h.table_id}/replay", params=params, headers=auth(token))


def private_holes(h) -> dict[str, dict[int, set[str]]]:
    """hand_id -> seat -> hole cards, read straight from the private log."""
    out: dict[str, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
    for e in h.manager.tables[h.table_id].table.log:
        if e.type == "HoleDealt":
            out[e.hand_id][e.data["seat"]].add(e.data["card"])
    return out


def revealed(h) -> dict[str, set[int]]:
    out: dict[str, set[int]] = defaultdict(set)
    for e in h.manager.tables[h.table_id].table.log:
        if e.type == "CardsRevealed":
            out[e.hand_id].add(e.data["seat"])
    return out


def play_until(h, seat, hands):
    """The human always calls/checks; bots act instantly. Stops once ``hands`` hands started."""
    for _ in range(2000):
        snap = h.snap(seat)
        if snap["status"] != "running" or snap["view"]["hand_number"] > hands:
            return snap
        if snap["turn"] and snap["turn"]["seat"] == seat:
            legal = snap["view"]["legal"]
            assert h.act(seat, {"type": "check"} if legal["check"] else {"type": "call"}).status_code == 200
    raise AssertionError("table did not progress")


def assert_no_hidden_cards(h, body, viewer_seat):
    holes, shown = private_holes(h), revealed(h)
    hand = body["hand"]
    text = json.dumps(hand)
    for seat, cards in holes[hand["hand_id"]].items():
        if seat == viewer_seat or seat in shown[hand["hand_id"]]:
            continue
        for card in cards:
            assert f'"{card}"' not in text, f"seat {seat} card {card} leaked into a replay frame"


def test_seat_replay_during_the_table_matches_live_and_leaks_nothing():
    with harness() as h:
        h.setup([HUMAN, bot("calling_station"), bot("check_fold")], hand_cap=40)
        snap = play_until(h, 0, 3)
        r = replay(h, h.tokens[0])
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["perspective"] == "seat" and body["viewer_seat"] == 0
        assert body["available_perspectives"] == ["seat", "spectator"]
        assert body["table_complete"] is False
        assert [x["hand_number"] for x in body["hands"]][:3] == [1, 2, 3]
        for entry in body["hands"]:
            hand = replay(h, h.tokens[0], hand=entry["hand_number"]).json()
            assert len(hand["hand"]["frames"]) == entry["frames"]
            assert_no_hidden_cards(h, hand, viewer_seat=0)
            for frame in hand["hand"]["frames"]:
                assert frame["hole_cards"] is None and frame["view"]["legal"] is None
                assert frame["view"]["perspective"] == "seat"
        last = replay(h, h.tokens[0], hand=body["hands"][-1]["hand_number"]).json()["hand"]["frames"][-1]
        live = dict(snap["view"], legal=None)
        assert last["view"] == live, "the last replay frame is the live projection"


def test_hand_end_frames_show_the_finished_hand():
    with harness() as h:
        h.setup([bot("calling_station"), bot("calling_station")], hand_cap=3)
        body = replay(h).json()
        assert body["perspective"] == "spectator" and body["table_complete"]
        frames = body["hand"]["frames"]
        assert frames[0]["kind"] == "deal" and frames[-1]["kind"] == "hand_end"
        assert any(p["shown_cards"] for p in frames[-1]["view"]["players"]), "calling stations reach showdown"
        assert [e["type"] for e in body["hand"]["events"]][0] == "HandStarted"
        final = replay(h, hand=3).json()["hand"]
        assert final["frames"][-1]["kind"] == "complete"
        assert final["events"][-1]["type"] == "TableScored"


def test_perspective_permissions():
    with harness() as h:
        h.setup([HUMAN, bot("check_fold")], hand_cap=20)
        assert replay(h, perspective="seat").status_code == 403
        assert replay(h, h.tokens[0], perspective="reviewer").json()["error"]["code"] == "projection_denied"
        running = replay(h, h.host, perspective="reviewer")
        assert running.status_code == 403 and "complete" in running.json()["error"]["message"]
        assert replay(h, h.host).json()["available_perspectives"] == ["spectator"]
        assert replay(h, h.tokens[0], perspective="sideways").status_code == 422
        assert replay(h, h.tokens[0], hand=999).status_code == 404
        assert replay(h, "forged").status_code == 401

        for _ in range(400):
            if h.snap()["status"] == "complete":
                break
            h.clock.advance_ms(h.manager.tables[h.table_id].config.human_turn_ms + 1)
            h.manager.pump()
        assert h.snap()["status"] == "complete"
        assert replay(h, h.tokens[0], perspective="reviewer").status_code == 403
        assert replay(h, perspective="reviewer").status_code == 403
        rv = replay(h, h.host, perspective="reviewer")
        assert rv.status_code == 200
        assert rv.json()["available_perspectives"] == ["spectator", "reviewer"]
        frame = rv.json()["hand"]["frames"][0]
        assert set(frame["hole_cards"]) == {"0", "1"} and all(len(c) == 2 for c in frame["hole_cards"].values())
        assert frame["view"]["perspective"] == "spectator" and frame["view"]["hole_cards"] is None


def test_not_started_table_has_no_replay():
    with harness() as h:
        h.create([HUMAN, bot()])
        assert replay(h).status_code == 409
