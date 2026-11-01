"""Opponent styles over the API: setup validation, hidden-seat privacy, stats, reveal,
learning aids and reviewer-only reasoning notes."""

from __future__ import annotations

import json
from contextlib import ExitStack

import pytest

from arena.agents.styles import PROFILES, STYLES, style_artifact_hash
from arena.api.manager import TableManager
from arena.api.styles import lineup_hash
from arena.domain.models import LegalActions
from arena.domain.rng import SeededRng
from arena.table.driver import legal_random
from tests.api.helpers import HUMAN, ORIGIN, auth, bot, harness


def create(h, seats, styles=None, *, seed="styles", status=201, **config):
    body = {"config": {"hand_cap": 12, **config}, "seats": seats, "seed": seed}
    if styles is not None:
        body["styles"] = styles
    r = h.client.post("/tables", json=body)
    assert r.status_code == status, r.text
    if status == 201:
        h.table_id, h.host = r.json()["table_id"], r.json()["host_token"]
        h.tokens.clear()
    return r


def play_out(h, rng_seed="driver", on_step=None):
    rng = SeededRng(rng_seed, "styles-driver")
    while (snap := h.snap())["status"] != "complete":
        seat = snap["turn"]["seat"]
        view = h.snap(seat)["view"]
        if on_step is not None:
            on_step(seat, view)
        action = legal_random(seat, LegalActions(**view["legal"]), rng).model_dump(exclude_none=True)
        r = h.act(seat, action)
        assert r.status_code == 200, r.text
    return h.snap()


def hidden(policy):
    return {"kind": "bot", "policy": policy, "reveal": False}


# ------------------------------------------------------------- catalog


def test_styles_endpoint_and_catalog():
    with harness() as h:
        body = h.client.get("/styles").json()
        assert [s["id"] for s in body["styles"]] == list(PROFILES)
        assert body["modes"] == ["labelled", "hidden", "mixed", "random"]
        assert "switcher" not in body["random_pool_default"]
        assert set(body["stat_definitions"]) == {"vpip", "pfr", "aggression", "fold_to_reraise", "showdowns"}
        assert len(body["grid"]) == 13 and body["grid"][0][:2] == ["AA", "AKs"]
        assert set(body["charts"]) == {"conservative", "aggressive"}
        assert "AA" in body["charts"]["conservative"]["early"]
        rock = body["styles"][0]
        assert rock["policy_id"] == "style_rock" and rock["book_ref"]["part"].startswith("Part Two")
        policies = h.client.get("/catalog").json()["policies"]
        styles = [p for p in policies if p.get("source") == "style"]
        assert {p["id"] for p in styles} == {f"style_{s}" for s in STYLES}
        assert all(p["style"] for p in styles)
        assert "style_random" not in {p["id"] for p in policies}


# ---------------------------------------------------------- validation


@pytest.mark.parametrize(("seats", "styles", "code"), [
    ([HUMAN, hidden("style_rock")], {"mode": "labelled"}, "invalid_styles"),
    ([HUMAN, {"kind": "bot", "policy": "style_rock", "reveal": True}], {"mode": "hidden"}, "invalid_styles"),
    ([HUMAN, bot("style_rock")], {"mode": "random"}, "invalid_styles"),
    ([HUMAN, {"kind": "bot", "policy": "style_random", "reveal": True}], {"mode": "mixed"}, "invalid_styles"),
    ([HUMAN, bot("style_random")], {"mode": "mixed", "random_pool": []}, "invalid_styles"),
    ([HUMAN, bot("style_random")], {"mode": "mixed", "random_pool": ["rock", "nit"]}, "invalid_styles"),
    ([{"kind": "human", "reveal": False}, bot("style_rock")], None, "invalid_seat"),
    ([HUMAN, bot("style_nope")], None, "invalid_seat"),
    ([HUMAN, {"kind": "bot"}], {"mode": "labelled"}, "invalid_seat"),
])
def test_invalid_style_setups_are_rejected(seats, styles, code):
    with harness() as h:
        r = create(h, seats, styles, status=422)
        assert r.json()["error"]["code"] == code


def test_style_seats_need_practice_nlhe_and_random_errors_do_not_name_a_pick():
    with harness() as h:
        r = create(h, [HUMAN, bot("style_rock")], ruleset_id="uncertainty_plo4", status=422)
        assert r.json()["error"]["code"] == "unsupported_ruleset"
        r = create(h, [HUMAN, bot("style_random")], {"mode": "mixed", "random_pool": ["station"]},
                   ruleset_id="uncertainty_plo4", status=422)
        text = json.dumps(r.json())
        assert r.json()["error"]["code"] == "unsupported_ruleset" and "station" not in text.lower()


def test_resolve_echoes_styles_without_picks():
    with harness() as h:
        r = h.client.post("/configs/resolve", json={"config": {}, "seats": [HUMAN, bot("style_random"),
                                                                             hidden("style_rock")],
                                                   "styles": {"mode": "mixed"}})
        assert r.status_code == 200, r.text
        assert r.json()["styles"]["hidden_seats"] == [1, 2]
        assert "style_" not in json.dumps(r.json()["styles"])


# ----------------------------------------------------------- modes


def test_labelled_mode_shows_styles_and_mixed_mode_masks_hidden_seats():
    with harness() as h:
        create(h, [HUMAN, bot("style_rock"), hidden("style_station"), bot("nlhe_heuristic")], {"mode": "mixed"})
        seats = {s["seat"]: s for s in h.get().json()["setup"]["seats"]}
        assert seats[1]["policy"] == "style_rock" and seats[1]["style"] == "rock" and seats[1]["source"] == "style"
        assert seats[1]["artifact_hash"] == style_artifact_hash("rock") and not seats[1]["hidden"]
        assert seats[2] | {"you": False} == {"seat": 2, "kind": "bot", "policy": None, "label": "Player 2",
                                            "artifact_hash": None, "source": None, "style": None, "hidden": True,
                                            "claimed": True, "you": False}
        assert seats[3]["style"] is None and seats[3]["policy"] == "nlhe_heuristic"
        assert seats[0]["style"] is None and not seats[0]["hidden"]
        styles = h.get().json()["setup"]["styles"]
        assert styles["mode"] == "mixed" and styles["hidden_seats"] == [2] and styles["revealed"] is False


def test_hidden_mode_hides_baselines_too():
    with harness() as h:
        create(h, [HUMAN, bot("style_rock"), bot("nlhe_heuristic")], {"mode": "hidden"})
        seats = h.get().json()["setup"]["seats"]
        assert [s["hidden"] for s in seats] == [False, True, True]
        assert all(s["policy"] is None for s in seats)


def test_random_picks_are_reproducible_with_a_seed_and_come_from_the_pool():
    picks = []
    for _ in range(2):
        with harness() as h:
            create(h, [HUMAN, {"kind": "bot"}, {"kind": "bot"}, {"kind": "bot"}],
                   {"mode": "random", "random_pool": ["rock", "station"]}, seed="same", hand_cap=2)
            live = h.manager.tables[h.table_id]
            picks.append([live.slots[s].policy for s in (1, 2, 3)])
    assert picks[0] == picks[1] and set(picks[0]) <= {"style_rock", "style_station"}


# ------------------------------------------------- privacy (hard rule)


def _secrets(live) -> tuple[set[str], set[str]]:
    """(exact strings, substrings) that must not appear while hidden seats are unrevealed."""
    exact, sub = set(), set()
    for slot in live.slots.values():
        if not slot.hidden:
            continue
        exact |= {slot.policy, slot.bot.label, slot.bot.style}
        sub |= {slot.policy, slot.bot.artifact_hash}
    return exact, sub


def _scan(body, exact, sub, where) -> list[str]:
    body = json.loads(json.dumps(body))
    setup = body.get("setup") or (body.get("snapshot") or {}).get("setup")
    if setup and "styles" in setup:
        setup["styles"].pop("random_pool")  # the host's public pool, not any seat's pick
    problems = []

    def walk(x, path):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, f"{path}.{k}")
        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(v, f"{path}[{i}]")
        elif isinstance(x, str):
            if x in exact or any(s in x for s in sub):
                problems.append(f"{where}{path}: {x[:80]}")

    walk(body, "")
    return problems


def test_hidden_seats_never_leak_style_policy_hash_or_notes_until_the_table_ends():
    seats = [HUMAN, HUMAN, hidden("style_rock"), hidden("style_super_aggressive"), bot("style_random"),
             bot("style_station")]
    with harness() as h:
        create(h, seats, {"mode": "mixed", "random_pool": ["weak_tight", "aggressive"]}, seed="privacy",
               hand_cap=25, starting_stack=400)
        h.join_all()
        live = h.manager.tables[h.table_id]
        exact, sub = _secrets(live)
        exact.discard("Calling station")  # the revealed station seat's label is public
        assert {"style_rock", "Rock", "rock", "Super-aggressive"} <= exact
        captured: list[tuple[str, object]] = []
        viewers = {0: h.tokens[0], 1: h.tokens[1], "host": h.host, "anon": None}
        with ExitStack() as stack:
            sockets = {}
            for name, tok in viewers.items():
                ws = stack.enter_context(h.client.websocket_connect(f"/tables/{h.table_id}/stream",
                                                                    headers={"origin": ORIGIN}))
                ws.send_json({"type": "auth", "token": tok})
                sockets[name] = ws
                captured.append((f"ws:{name}", ws.receive_json()))
            captured.append(("start", h.start()))

            def on_step(seat, view):
                for name, tok in viewers.items():
                    for path in ("", "/stats", "/aids", "/history", "/replay"):
                        r = h.client.get(f"/tables/{h.table_id}{path}", headers=auth(tok))
                        captured.append((f"{path or '/'}:{name}", r.json()))
                other = 1 - seat if seat in (0, 1) else 0
                r = h.post_action(other, h.envelope(view, {"type": "fold"}))
                captured.append(("error", r.json()))
                r = h.client.get(f"/tables/{h.table_id}/replay", headers=auth(h.host),
                                 params={"perspective": "reviewer"})
                assert r.status_code == 403
                captured.append(("reviewer-denied", r.json()))
                for name, ws in sockets.items():
                    captured.append((f"ws:{name}", ws.receive_json()))

            final = play_out(h, on_step=on_step)
        assert final["status"] == "complete"
        notes = h.manager.reviewer_notes(h.table_id)
        hidden_seats = {s for s, slot in live.slots.items() if slot.hidden}
        note_texts = {n["text"] for n in notes.values() if n["seat"] in hidden_seats and len(n["text"]) > 15}
        assert note_texts
        problems = []
        for where, body in captured:
            if isinstance(body, dict) and body.get("status") == "complete":
                continue
            problems += _scan(body, exact, sub | note_texts, where)
        assert problems == [], problems[:10]
        # The checks are not vacuous: the same identities are present once the table is complete.
        assert _scan(final, exact, sub, "final")


# --------------------------------------------------------- stats, reveal


def test_stats_snapshot_endpoint_and_reveal():
    seats = [HUMAN, hidden("style_rock"), bot("style_random"), bot("style_aggressive")]
    with harness() as h:
        create(h, seats, {"mode": "mixed", "random_pool": ["station"]}, hand_cap=20, starting_stack=400)
        assert h.get().json()["stats"] is None and h.get().json()["reveal"] is None
        assert h.client.get(f"/tables/{h.table_id}/stats").status_code == 409
        h.join_all()
        h.start()
        snap = play_out(h)
        stats = snap["stats"]
        assert set(stats["seats"]) == {"0", "1", "2", "3"} and stats["hands_tracked"] == snap["results"]["hands_played"]
        for s in stats["seats"].values():
            assert len(s["showdowns"]["items"]) <= 3
            assert s["vpip"]["of"] == s["hands"]
        full = h.client.get(f"/tables/{h.table_id}/stats").json()
        assert full["stats"]["seats"]["1"]["showdowns"]["count"] == len(full["stats"]["seats"]["1"]["showdowns"]["items"])

        reveal = snap["reveal"]
        assert snap["setup"]["styles"]["revealed"] is True
        by_seat = {s["seat"]: s for s in reveal["seats"]}
        assert by_seat[1]["policy"] == "style_rock" and by_seat[1]["hidden"] and not by_seat[1]["random_pick"]
        assert by_seat[2]["policy"] == "style_station" and by_seat[2]["random_pick"] and by_seat[2]["style"] == "station"
        assert by_seat[0]["style"] is None and by_seat[0]["book_ref"] is None
        assert [e["stat"] for e in by_seat[1]["evidence"]] == ["vpip", "pfr", "aggression", "fold_to_reraise"]
        assert by_seat[1]["evidence"][0]["expected"] == list(PROFILES["rock"].expected.vpip)
        assert by_seat[1]["summary"]
        assert reveal["lineup_hash"] == lineup_hash(snap["setup"]["config_hash"], reveal["lineup"])
        seats_after = {s["seat"]: s for s in snap["setup"]["seats"]}
        assert seats_after[1]["policy"] == "style_rock" and seats_after[1]["hidden"] is True


# ------------------------------------------------------------ aids


def test_aids_are_seat_only_and_use_the_seats_own_view():
    with harness() as h:
        create(h, [HUMAN, bot("style_rock"), bot("style_station")], hand_cap=6)
        h.join_all()
        assert h.client.get(f"/tables/{h.table_id}/aids", headers=auth(h.tokens[0])).status_code == 409
        h.start()
        assert h.client.get(f"/tables/{h.table_id}/aids").status_code == 401
        assert h.client.get(f"/tables/{h.table_id}/aids", headers=auth(h.host)).status_code == 401
        seen_coach = False
        while (snap := h.snap())["status"] != "complete":
            r = h.client.get(f"/tables/{h.table_id}/aids", headers=auth(h.tokens[0]))
            assert r.status_code == 200, r.text
            aids = r.json()
            view = h.snap(0)["view"]
            hero = aids["hero"]
            if view["hole_cards"]:
                assert hero["hole_cards"] == view["hole_cards"]
                assert hero["position"] == aids["positions"]["0"]
                assert 0.0 <= hero["equity_vs_random"]["vs_one"] <= 1.0
                assert set(hero["charts"]) == {"group", "conservative", "aggressive"}
            if snap["turn"]["seat"] == 0:
                assert hero["coach"]["action"] in {"fold", "check", "call", "raise_to"}
                assert hero["coach"]["reason"] and hero["coach"]["heuristic"] == "conservative"
                assert aids["decision_id"] == view["decision_id"]
                seen_coach = True
                legal = view["legal"]
                if legal["call_amount"]:
                    po = hero["pot_odds"]
                    assert po["to_call"] == legal["call_amount"]
                    assert po["break_even"] == round(po["to_call"] / (po["pot"] + po["to_call"]), 4)
                coach = hero["coach"]
                action = {"type": coach["action"]} | ({"amount": coach["amount"]} if coach["action"] == "raise_to" else {})
                assert h.act(0, action).status_code == 200, coach
            elif hero is not None:
                assert hero["coach"] is None and hero["pot_odds"] is None
                h.manager.pump()
        assert seen_coach


@pytest.mark.parametrize("ruleset", ["uncertainty_plo4", "uncertainty_shortdeck"])
def test_aids_are_refused_outside_nlhe(ruleset):
    with harness() as h:
        create(h, [HUMAN, bot("calling_station")], ruleset_id=ruleset, hand_cap=3)
        h.join_all()
        url = f"/tables/{h.table_id}/aids"
        assert h.client.get(url, headers=auth(h.tokens[0])).json()["error"]["code"] == "aids_unsupported_ruleset"
        h.start()
        r = h.client.get(url, headers=auth(h.tokens[0]))
        assert r.status_code == 409 and r.json()["error"]["code"] == "aids_unsupported_ruleset"
        assert r.json()["error"]["details"] == {"ruleset_id": ruleset, "supported": ["practice_nlhe"]}
        assert h.client.get(url).status_code == 401
        assert h.get().json()["stats"] is not None


def test_aids_can_be_switched_off():
    with harness() as h:
        create(h, [HUMAN, bot("style_rock")], {"learning_aids": False})
        h.join_all()
        h.start()
        r = h.client.get(f"/tables/{h.table_id}/aids", headers=auth(h.tokens[0]))
        assert r.status_code == 403 and r.json()["error"]["code"] == "learning_aids_disabled"
        assert h.get().json()["setup"]["styles"]["learning_aids"] is False


# ------------------------------------------------- notes in replay


def test_reasoning_notes_only_in_reviewer_replay_after_completion():
    with harness() as h:
        create(h, [HUMAN, hidden("style_rock"), bot("style_switcher")], {"mode": "mixed"}, hand_cap=4)
        h.join_all()
        h.start()
        play_out(h)
        replay = h.client.get(f"/tables/{h.table_id}/replay", headers=auth(h.host),
                              params={"perspective": "reviewer"}).json()
        notes = [f["bot_note"] for f in replay["hand"]["frames"] if f["bot_note"]]
        assert notes and all(n["seat"] in (1, 2) and 0 < len(n["text"]) <= 200 for n in notes)
        assert any(n["text"].startswith("[") for n in notes if n["seat"] == 2)
        for tok, perspective in ((h.tokens[0], "seat"), (None, "spectator")):
            frames = h.client.get(f"/tables/{h.table_id}/replay", headers=auth(tok),
                                  params={"perspective": perspective}).json()["hand"]["frames"]
            assert all(f["bot_note"] is None for f in frames)


def test_setup_and_notes_survive_a_restart():
    with harness() as h:
        create(h, [HUMAN, hidden("style_rock"), bot("style_random")], {"mode": "mixed", "random_pool": ["station"]},
               hand_cap=5)
        h.join_all()
        h.start()
        play_out(h)
        before = h.get(h.host).json()
        notes = h.manager.reviewer_notes(h.table_id)
        fresh = TableManager(h.clock, store=h.manager.store, writer_id="second", force_claim=True,
                             bots=h.manager.bots)
        live = fresh.tables[h.table_id]
        assert live.halted_reason is None
        assert fresh.reviewer_notes(h.table_id) == notes
        assert live.slots[1].hidden and live.slots[2].random_pick and live.slots[2].policy == "style_station"
        from arena.api.manager import Principal
        after = fresh.snapshot(h.table_id, Principal("host", h.table_id))
        assert after["setup"] == before["setup"] and after["reveal"] == before["reveal"]


def test_hidden_setup_survives_a_restart_mid_table():
    with harness() as h:
        create(h, [HUMAN, hidden("style_rock"), bot("style_aggressive")], {"mode": "mixed"}, hand_cap=5)
        h.join_all()
        h.start()
        fresh = TableManager(h.clock, store=h.manager.store, writer_id="second", force_claim=True,
                             bots=h.manager.bots)
        from arena.api.manager import Principal
        snap = fresh.snapshot(h.table_id, Principal("seat", h.table_id, 0))
        seats = {s["seat"]: s for s in snap["setup"]["seats"]}
        assert seats[1]["hidden"] and seats[1]["policy"] is None and seats[2]["policy"] == "style_aggressive"
        assert snap["stats"] is not None and snap["reveal"] is None
