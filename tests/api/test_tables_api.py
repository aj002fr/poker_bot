"""HTTP/WebSocket behaviour of the in-memory table server (spec 9.1, 6.2, 6.3, 12.3)."""

from __future__ import annotations

import json

import pytest
from starlette.websockets import WebSocketDisconnect

from tests.api.helpers import HUMAN, ORIGIN, auth, bot, harness

SIX = [HUMAN, HUMAN, bot(), bot(), bot("check_or_fold"), bot("check_or_call")]


def safe_action(view: dict) -> dict:
    legal = view["legal"]
    return {"type": "check"} if legal["check"] else {"type": "call"}


def history_events(h, token: str | None = None, perspective: str | None = None) -> list[dict]:
    params = {"perspective": perspective} if perspective else None
    r = h.client.get(f"/tables/{h.table_id}/history", headers=auth(token), params=params)
    assert r.status_code == 200, r.text
    return r.json()["events"]


# --------------------------------------------------------------- create / join / start


def test_create_validates_config_and_seats_without_echoing_input():
    with harness() as h:
        cases = [
            {"config": {"starting_stack": True}, "seats": [HUMAN, HUMAN]},
            {"config": {"starting_stack": 2000.5}, "seats": [HUMAN, HUMAN]},
            {"config": {"surprise": 1}, "seats": [HUMAN, HUMAN]},
            {"config": {"initial_player_count": 3}, "seats": [HUMAN, HUMAN]},
            {"config": {}, "seats": [HUMAN] * 9},
            {"config": {}, "seats": [HUMAN]},
            {"config": {}, "seats": [HUMAN, bot("nonexistent")]},
            {"config": {}, "seats": [HUMAN, {"kind": "human", "policy": "legal_random"}]},
            {"config": {"ruleset_id": "texas_omaha"}, "seats": [HUMAN, HUMAN]},
        ]
        for body in cases:
            r = h.client.post("/tables", json=body)
            assert r.status_code == 422, (body, r.text)
            err = r.json()["error"]
            assert err["code"] in {"invalid_config", "invalid_seat", "invalid_request"}
            assert '"input"' not in json.dumps(err) and "2000.5" not in json.dumps(err)
        assert h.manager.tables == {}


def test_resolve_returns_resolved_payouts_and_hash():
    with harness() as h:
        r = h.client.post("/configs/resolve", json={"config": {"hand_cap": 50}, "seats": [HUMAN] * 2 + [bot()]})
        assert r.status_code == 200
        body = r.json()
        assert body["config"]["payout_points"] == [50, 30, 20]
        assert body["config"]["initial_player_count"] == 3
        assert len(body["config_hash"]) == 64
        assert {p["id"] for p in body["policies"]} >= {"legal_random", "check_or_fold"}


def test_join_claims_human_seats_once_and_only_before_start():
    with harness() as h:
        h.create([HUMAN, bot(), HUMAN])
        first = h.client.post(f"/tables/{h.table_id}/join", json={})
        assert first.status_code == 200 and first.json()["seat"] == 0
        assert h.client.post(f"/tables/{h.table_id}/join", json={"seat": 0}).status_code == 409
        assert h.client.post(f"/tables/{h.table_id}/join", json={"seat": 1}).json()["error"]["code"] == "seat_unavailable"
        second = h.client.post(f"/tables/{h.table_id}/join", json={})
        assert second.json()["seat"] == 2
        assert h.client.post(f"/tables/{h.table_id}/join", json={}).json()["error"]["code"] == "no_open_seat"
        assert first.json()["seat_token"] != second.json()["seat_token"]
        h.start()
        assert h.client.post(f"/tables/{h.table_id}/join", json={}).json()["error"]["code"] == "already_started"


def test_start_requires_host_and_every_human_seat_claimed():
    with harness() as h:
        h.create([HUMAN, HUMAN, bot()])
        seat0 = h.client.post(f"/tables/{h.table_id}/join", json={"seat": 0}).json()["seat_token"]
        url = f"/tables/{h.table_id}/start"
        assert h.client.post(url).json()["error"]["code"] == "host_credential_required"
        assert h.client.post(url, headers=auth(seat0)).status_code == 403
        r = h.client.post(url, headers=auth(h.host))
        assert r.status_code == 409 and r.json()["error"]["code"] == "seats_unclaimed"
        h.client.post(f"/tables/{h.table_id}/join", json={"seat": 1})
        assert h.client.post(url, headers=auth(h.host)).status_code == 200
        assert h.client.post(url, headers=auth(h.host)).json()["error"]["code"] == "already_started"


# ------------------------------------------------------------------- identity


def test_snapshot_is_projected_for_the_caller_only():
    with harness() as h:
        h.setup(SIX, starting_stack=500)
        me = h.snap(0)
        assert me["viewer"] == {"role": "seat", "seat": 0}
        assert me["view"]["perspective"] == "seat" and len(me["view"]["hole_cards"]) == 2
        other = h.snap(1)
        assert set(other["view"]["hole_cards"]).isdisjoint(me["view"]["hole_cards"])
        spectator = h.snap()
        assert spectator["view"]["hole_cards"] is None and spectator["view"]["legal"] is None
        for p in spectator["view"]["players"]:
            assert p["shown_cards"] is None
        assert "seed" not in json.dumps(spectator)
        assert [s["you"] for s in me["setup"]["seats"]] == [True] + [False] * 5


def test_credentials_are_bound_to_their_table():
    with harness() as h:
        h.setup([HUMAN, bot()])
        token_a = h.tokens[0]
        first_table = h.table_id
        h.setup([HUMAN, bot()])
        assert h.get(token_a).status_code == 403
        assert h.get(token_a).json()["error"]["code"] == "credential_wrong_table"
        assert h.get("not-a-real-token").status_code == 401
        r = h.client.get(f"/tables/{first_table}", headers={"Authorization": "Basic abc"})
        assert r.status_code == 401
        assert h.client.get("/tables/t_missing").status_code == 404


def test_action_requires_the_acting_seat_credential():
    with harness() as h:
        h.setup([HUMAN, HUMAN])
        actor = h.actor()
        other = 1 - actor
        view = h.snap(actor)["view"]
        env = h.envelope(view, safe_action(view))
        assert h.post_action(None, env).json()["error"]["code"] == "seat_credential_required"
        assert h.post_action(None, env, token=h.host).status_code == 401
        r = h.post_action(other, env)
        assert r.status_code == 409 and r.json()["error"]["code"] == "out_of_turn"
        assert h.post_action(actor, {**env, "seat": other}).status_code == 422
        assert h.snap()["version"] == h.manager.tables[h.table_id].version
        assert h.post_action(actor, env).status_code == 200


# ---------------------------------------------------------------- actions


def test_invalid_and_illegal_actions_change_nothing():
    with harness() as h:
        h.setup([HUMAN, HUMAN])
        actor = h.actor()
        before = h.snap(actor)
        view = before["view"]
        legal = view["legal"]
        bad = [
            {"type": "raise_to", "amount": True},
            {"type": "raise_to", "amount": 25.0},
            {"type": "call", "amount": 20},
            {"type": "bet", "amount": 40},
            {"type": "raise_to", "amount": legal["min_full_raise_to"] - 1},
            {"type": "check"} if not legal["check"] else {"type": "call"},
        ]
        for action in bad:
            r = h.post_action(actor, h.envelope(view, action))
            assert r.status_code == 422, (action, r.text)
            assert r.json()["error"]["code"] in {"invalid_schema", "illegal_action"}
        r = h.post_action(actor, {**h.envelope(view, safe_action(view)), "table_id": "t_other"})
        assert r.json()["error"]["code"] == "table_mismatch"
        r = h.post_action(actor, {**h.envelope(view, safe_action(view)), "expected_state_version": view["state_version"] - 1})
        assert r.json()["error"]["code"] == "stale_version"
        after = h.snap(actor)
        assert after["version"] == before["version"]
        assert after["view"] == before["view"]


def test_raise_to_is_a_street_total():
    with harness() as h:
        h.setup([HUMAN, HUMAN], starting_stack=2000)
        actor = h.actor()
        view = h.snap(actor)["view"]
        me = next(p for p in view["players"] if p["seat"] == actor)
        target = view["legal"]["min_full_raise_to"]
        r = h.act(actor, {"type": "raise_to", "amount": target})
        assert r.status_code == 200, r.text
        assert r.json()["receipt"]["action"] == {"type": "raise_to", "amount": target}
        after = next(p for p in r.json()["snapshot"]["view"]["players"] if p["seat"] == actor)
        assert after["street_committed"] == target
        assert after["stack"] == me["stack"] - (target - me["street_committed"])


def test_action_id_idempotency():
    with harness() as h:
        h.setup([HUMAN, HUMAN])
        actor = h.actor()
        view = h.snap(actor)["view"]
        env = h.envelope(view, safe_action(view), action_id="a_same")
        first = h.post_action(actor, env)
        assert first.status_code == 200 and first.json()["replayed"] is False
        version = h.snap()["version"]
        again = h.post_action(actor, env)
        assert again.status_code == 200 and again.json()["replayed"] is True
        assert again.json()["receipt"] == first.json()["receipt"]
        assert h.snap()["version"] == version
        changed = h.post_action(actor, {**env, "action": {"type": "fold"}})
        assert changed.status_code == 409 and changed.json()["error"]["code"] == "action_id_conflict"
        nxt = h.actor()
        stolen = h.post_action(nxt, env)
        assert stolen.status_code == 409 and stolen.json()["error"]["code"] == "action_id_conflict"
        applied = [e for e in history_events(h, h.tokens[0]) if e["type"] == "ActionApplied" and e["data"]["seat"] == actor]
        assert len(applied) == 1


def test_two_tabs_with_different_ids_cannot_act_twice():
    with harness() as h:
        h.setup([HUMAN, HUMAN, bot("check_or_fold")])
        actor = h.actor()
        view = h.snap(actor)["view"]
        a = h.post_action(actor, h.envelope(view, safe_action(view), "a_tab_one"))
        b = h.post_action(actor, h.envelope(view, {"type": "fold"}, "a_tab_two"))
        assert a.status_code == 200
        assert b.status_code == 409
        assert b.json()["error"]["code"] in {"stale_decision", "out_of_turn", "wrong_hand", "stale_version"}


# --------------------------------------------------------------- deadlines


def heads_up(h, turn_ms: int = 5000) -> tuple[int, dict]:
    h.setup([HUMAN, HUMAN], human_turn_ms=turn_ms)
    actor = h.actor()
    return actor, h.snap(actor)["view"]


def last_action(h, seat: int) -> tuple[dict | None, dict]:
    events = history_events(h, h.tokens[seat])
    applied = [e for e in events if e["type"] == "ActionApplied" and e["data"]["seat"] == seat]
    fallbacks = [e for e in events if e["type"] == "ActionFallback" and e["data"]["seat"] == seat]
    return (fallbacks[-1] if fallbacks else None), applied[-1]


def test_action_received_exactly_at_deadline_is_on_time():
    with harness() as h:
        actor, view = heads_up(h)
        deadline = h.manager.tables[h.table_id].deadline_ns
        assert h.snap()["turn"]["remaining_ms"] == 5000
        h.clock.set_ns(deadline)
        r = h.post_action(actor, h.envelope(view, safe_action(view)))
        assert r.status_code == 200, r.text
        assert last_action(h, actor)[0] is None


def test_action_received_after_deadline_loses_to_the_timeout():
    with harness() as h:
        actor, view = heads_up(h)
        legal = view["legal"]
        deadline = h.manager.tables[h.table_id].deadline_ns
        h.clock.set_ns(deadline + 1)
        r = h.post_action(actor, h.envelope(view, {"type": "raise_to", "amount": legal["min_full_raise_to"]}))
        assert r.status_code == 409 and r.json()["error"]["code"] == "deadline_passed"
        fallback, applied = last_action(h, actor)
        assert fallback is not None and fallback["data"] == {"seat": actor, "reason": "timeout"}
        assert applied["data"]["action"]["type"] == ("check" if legal["check"] else "fold")


def test_on_time_action_waiting_in_queue_beats_a_later_timeout():
    with harness() as h:
        actor, view = heads_up(h)
        live = h.manager.tables[h.table_id]
        deadline = live.deadline_ns
        h.clock.set_ns(deadline + 250_000_000)
        live.inflight.append(deadline)
        assert h.manager.pump() == 0
        live.inflight.remove(deadline)
        principal = h.manager.authenticate(h.table_id, h.tokens[actor])
        raw = json.dumps(h.envelope(view, safe_action(view))).encode()
        out = h.manager.submit_action(h.table_id, principal, raw, received_ns=deadline)
        assert out["replayed"] is False
        assert last_action(h, actor)[0] is None


def test_timeout_falls_back_to_check_then_fold():
    with harness() as h:
        sb, view = heads_up(h, turn_ms=1000)
        assert view["legal"]["check"] is False
        h.clock.advance_ms(1000)
        assert h.manager.pump() == 0
        h.clock.advance_ms(0.001)
        assert h.manager.pump() == 1
        fallback, applied = last_action(h, sb)
        assert fallback["data"]["reason"] == "timeout" and applied["data"]["action"]["type"] == "fold"

        # Next hand: the small blind calls, so the big blind can check; its timeout checks.
        sb2 = h.actor()
        assert h.act(sb2, {"type": "call"}).status_code == 200
        bb = h.actor()
        assert bb != sb2 and h.snap(bb)["view"]["legal"]["check"] is True
        h.clock.advance_ms(1001)
        assert h.manager.pump() >= 1
        fallback, applied = last_action(h, bb)
        assert fallback["data"]["reason"] == "timeout" and applied["data"]["action"]["type"] == "check"


def test_bots_are_paced_by_the_server_clock():
    with harness(bot_delay_ms=500, hand_pause_ms=0) as h:
        h.setup([bot(), bot("check_or_call"), bot("check_or_fold")])
        assert h.snap()["turn"]["kind"] == "bot"
        version = h.snap()["version"]
        assert h.manager.pump() == 0
        h.clock.advance_ms(499)
        assert h.manager.pump() == 0
        h.clock.advance_ms(1)
        assert h.manager.pump() == 1
        assert h.snap()["version"] == version + 1


# ------------------------------------------------------------ history / 403


def test_projection_denied_maps_to_403():
    with harness() as h:
        h.setup([HUMAN, bot("check_or_call")], starting_stack=60, hand_cap=3)
        url = f"/tables/{h.table_id}/history"
        r = h.client.get(url, headers=auth(h.host), params={"perspective": "reviewer"})
        assert r.status_code == 403 and r.json()["error"]["code"] == "projection_denied"
        r = h.client.get(url, headers=auth(h.tokens[0]), params={"perspective": "reviewer"})
        assert r.status_code == 403 and r.json()["error"]["code"] == "reviewer_permission_required"
        while h.snap()["status"] != "complete":
            seat = h.actor()
            h.act(seat, safe_action(h.snap(seat)["view"]))
        r = h.client.get(url, headers=auth(h.host), params={"perspective": "reviewer"})
        assert r.status_code == 200
        assert r.json()["perspective"] == "reviewer"
        assert any("deck" in e["data"] for e in r.json()["events"] if e["type"] == "HandStarted")


def test_full_table_reaches_results_over_http():
    with harness() as h:
        h.setup(SIX, starting_stack=300, hand_cap=12)
        steps = 0
        while h.snap()["status"] != "complete":
            seat = h.actor()
            view = h.snap(seat)["view"]
            legal = view["legal"]
            action = {"type": "raise_to", "amount": legal["all_in_raise_to"]} if steps % 5 == 0 and legal["all_in_raise_to"] else safe_action(view)
            assert h.act(seat, action).status_code == 200
            steps += 1
        results = h.snap(0)["results"]
        assert sum(results["points"].values()) == pytest.approx(100.0, abs=1e-9)
        assert len(results["placements"]) == 6
        assert results["payouts"] == [50, 30, 20, 0, 0, 0]
        assert h.snap(0)["turn"] is None


# --------------------------------------------------------------- websocket


def stream(h, token: str | None, origin: str | None = ORIGIN):
    headers = {"origin": origin} if origin else {}
    ws = h.client.websocket_connect(f"/tables/{h.table_id}/stream", headers=headers)
    return ws


def test_stream_sends_authorized_snapshots_with_increasing_versions():
    with harness() as h:
        h.setup([HUMAN, HUMAN, bot()])
        with stream(h, h.tokens[0]) as ws:
            ws.send_json({"type": "auth", "token": h.tokens[0]})
            first = ws.receive_json()
            assert first["type"] == "snapshot" and first["viewer"]["seat"] == 0
            assert first["view"]["hole_cards"] and first["version"] == h.snap()["version"]
            versions = [first["version"]]
            for _ in range(3):
                seat = h.actor()
                if seat is None:
                    break
                target = h.act(seat, safe_action(h.snap(seat)["view"])).json()["snapshot"]["version"]
                while versions[-1] < target:
                    versions.append(ws.receive_json()["version"])
            assert versions == sorted(set(versions)) and len(versions) > 1


def test_stream_spectator_gets_public_view():
    with harness() as h:
        h.setup([HUMAN, bot()])
        with stream(h, None) as ws:
            ws.send_json({"type": "auth", "token": None})
            frame = ws.receive_json()
            assert frame["viewer"]["role"] == "anonymous"
            assert frame["view"]["hole_cards"] is None


@pytest.mark.parametrize(
    ("hello", "code"),
    [
        ({"type": "auth", "token": "forged"}, "invalid_credential"),
        ({"type": "hello"}, "auth_required"),
    ],
)
def test_stream_rejects_bad_credentials(hello, code):
    with harness() as h:
        h.setup([HUMAN, bot()])
        with stream(h, None) as ws:
            ws.send_json(hello)
            frame = ws.receive_json()
            assert frame["type"] == "error" and frame["error"]["code"] == code
            with pytest.raises(WebSocketDisconnect):
                ws.receive_json()


def test_stream_rejects_wrong_table_credential_and_foreign_origin():
    with harness() as h:
        h.setup([HUMAN, bot()])
        token = h.tokens[0]
        h.setup([HUMAN, bot()])
        with stream(h, None) as ws:
            ws.send_json({"type": "auth", "token": token})
            assert ws.receive_json()["error"]["code"] == "credential_wrong_table"
        with pytest.raises(WebSocketDisconnect):
            with stream(h, None, origin="https://evil.example") as ws:
                ws.receive_json()
