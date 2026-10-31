"""Spec 12.2 / 12.1(7) leak checks over the wire: HTTP bodies, WebSocket frames and errors.

Full tables are played through the API while every human seat and a spectator hold
a WebSocket subscription. Every response body, stream frame and error is captured,
then checked against per-hand entitlements rebuilt from the reviewer log after the
table ends: a seat may see its own hole cards, the board dealt so far and showdown
reveals of that hand; a spectator only the latter two.
"""

from __future__ import annotations

import copy
import json
import re
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass, field

import pytest

from arena.domain.models import LegalActions
from arena.domain.rng import SeededRng
from arena.table.driver import legal_random
from tests.api.helpers import HUMAN, ORIGIN, auth, bot, harness

CARD_LOOSE = re.compile(r"(?<![A-Za-z0-9])([2-9TJQKA][cdhs])(?![A-Za-z0-9])")
CARD_EXACT = re.compile(r"^[2-9TJQKA][cdhs]$")
BOARD_LEN = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}


def cards_in(obj) -> set[str]:
    found: set[str] = set()

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(k)
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, str):
            if CARD_EXACT.match(x):
                found.add(x)
            else:
                found.update(CARD_LOOSE.findall(x))

    walk(obj)
    return found


class Ledger:
    """Per-hand card facts from the reviewer (full private) history."""

    def __init__(self, events: list[dict]):
        self.hole: dict[str, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
        self.board: dict[str, list[str]] = defaultdict(list)
        self.revealed: dict[str, set[str]] = defaultdict(set)
        self.deck: dict[str, set[str]] = {}
        for e in events:
            hid, d = e["hand_id"], e["data"]
            if e["type"] == "HandStarted":
                self.deck[hid] = set(d["deck"])
            elif e["type"] == "HoleDealt":
                self.hole[hid][d["seat"]].add(d["card"])
            elif e["type"] == "BoardDealt":
                self.board[hid].extend(d["cards"])
            elif e["type"] == "CardsRevealed":
                self.revealed[hid].update(d["cards"])

    def allowed(self, hand_id: str | None, seat: int | None) -> set[str]:
        if hand_id is None:
            return set()
        own = self.hole[hand_id].get(seat, set()) if seat is not None else set()
        return own | set(self.board[hand_id]) | self.revealed[hand_id]


@dataclass
class Capture:
    seat: int | None
    kind: str  # snapshot | receipt | error | history | frame
    body: dict


@dataclass
class Recorder:
    items: list[Capture] = field(default_factory=list)

    def add(self, seat: int | None, kind: str, body: dict) -> dict:
        self.items.append(Capture(seat, kind, body))
        return body


def showdown_lists(snap: dict) -> list[list[dict]]:
    """Showdown item lists in ``stats`` and ``reveal``: the only cards allowed outside view/last_hand."""
    out = []
    if snap.get("stats"):
        out += [s["showdowns"]["items"] for s in snap["stats"]["seats"].values()]
    if snap.get("reveal"):
        out += [s["showdowns"] for s in snap["reveal"]["seats"]]
    return out


def check_showdowns(snap: dict, ledger: Ledger) -> list[str]:
    problems = []
    for items in showdown_lists(snap):
        for item in items:
            hid = item["hand_id"]
            extra = cards_in(item) - ledger.revealed[hid] - set(ledger.board[hid])
            if extra:
                problems.append(f"showdown {hid} leaks {extra}")
    return problems


def check_snapshot(snap: dict, seat: int | None, ledger: Ledger) -> list[str]:
    problems: list[str] = check_showdowns(snap, ledger)
    rest = copy.deepcopy({k: v for k, v in snap.items() if k not in {"view", "last_hand"}})
    for items in showdown_lists(rest):
        items.clear()
    if cards_in(rest):
        problems.append(f"cards outside view/last_hand: {cards_in(rest)}")
    view = snap.get("view")
    if view is not None:
        hand_id = view["hand_id"]
        extra = cards_in(view) - ledger.allowed(hand_id, seat)
        if extra:
            problems.append(f"view {hand_id} leaks {extra} to seat {seat}")
        if hand_id is not None:
            if view["street"] is not None and len(view["board"]) != BOARD_LEN[view["street"]]:
                problems.append(f"board {view['board']} does not match street {view['street']}")
            if view["board"] != ledger.board[hand_id][: len(view["board"])]:
                problems.append("board is not a prefix of the dealt board")
            if seat is not None and set(view["hole_cards"] or []) != ledger.hole[hand_id].get(seat, set()):
                problems.append("hero hole cards mismatch")
    last = snap.get("last_hand")
    if last is not None:
        extra = cards_in(last) - ledger.allowed(last["hand_id"], seat)
        if extra:
            problems.append(f"last_hand {last['hand_id']} leaks {extra} to seat {seat}")
    return problems


def check(capture: Capture, ledger: Ledger) -> list[str]:
    body, seat = capture.body, capture.seat
    if capture.kind in {"snapshot", "frame"}:
        return check_snapshot(body, seat, ledger)
    if capture.kind == "receipt":
        problems = [] if not cards_in(body["receipt"]) else ["cards in receipt"]
        return problems + check_snapshot(body["snapshot"], seat, ledger)
    if capture.kind == "error":
        return [f"cards in error {body}"] if cards_in(body) else []
    if capture.kind == "history":
        if "events" not in body:
            return [f"cards in denial {body}"] if cards_in(body) else []
        out = []
        for e in body["events"]:
            extra = cards_in(e) - ledger.allowed(e["hand_id"], seat)
            if extra:
                out.append(f"history event {e['type']} {e['hand_id']} leaks {extra} to seat {seat}")
        return out
    raise AssertionError(capture.kind)


def play_recorded(h, rec: Recorder, seed: str) -> None:
    rng = SeededRng(seed, "leak-driver")
    humans = sorted(h.tokens)
    viewers: list[int | None] = [*humans, None]
    with ExitStack() as stack:
        sockets = {}
        for v in viewers:
            ws = stack.enter_context(
                h.client.websocket_connect(f"/tables/{h.table_id}/stream", headers={"origin": ORIGIN})
            )
            ws.send_json({"type": "auth", "token": h.tokens.get(v) if v is not None else None})
            sockets[v] = (ws, [rec.add(v, "frame", ws.receive_json())["version"]])

        def drain(target: int) -> None:
            for v, (ws, versions) in sockets.items():
                while versions[-1] < target:
                    frame = rec.add(v, "frame", ws.receive_json())
                    assert frame["version"] > versions[-1], "stream version went backwards"
                    versions.append(frame["version"])

        previous_env = None
        step = 0
        while True:
            snaps = {v: rec.add(v, "snapshot", h.get(h.tokens.get(v) if v is not None else None).json()) for v in viewers}
            spectator = snaps[None]
            if spectator["status"] == "complete":
                break
            actor = spectator["turn"]["seat"]
            view = snaps[actor]["view"]
            others = [s for s in humans if s != actor]
            if step % 6 == 0:
                if others:
                    r = h.post_action(others[0], h.envelope(snaps[others[0]]["view"] | {"decision_id": view["decision_id"], "hand_id": view["hand_id"]}, {"type": "fold"}))
                    rec.add(others[0], "error", r.json())
                r = h.post_action(actor, h.envelope(view, {"type": "raise_to", "amount": 1}))
                rec.add(actor, "error", r.json())
                if previous_env is not None:
                    r = h.post_action(previous_env[0], {**previous_env[1], "action_id": f"a_stale_{step}"})
                    rec.add(previous_env[0], "error", r.json())
            if step % 9 == 0:
                for v in viewers:
                    r = h.client.get(f"/tables/{h.table_id}/history", headers=auth(h.tokens.get(v) if v is not None else None))
                    rec.add(v, "history", r.json())
                r = h.client.get(f"/tables/{h.table_id}/history", headers=auth(h.host), params={"perspective": "reviewer"})
                assert r.status_code == 403
                rec.add(None, "history", r.json())
            action = legal_random(actor, LegalActions(**view["legal"]), rng).model_dump(exclude_none=True)
            env = h.envelope(view, action)
            r = h.post_action(actor, env)
            assert r.status_code == 200, r.text
            rec.add(actor, "receipt", r.json())
            previous_env = (actor, env)
            drain(r.json()["snapshot"]["version"])
            step += 1
        for v in viewers:
            r = h.client.get(f"/tables/{h.table_id}/history", headers=auth(h.tokens.get(v) if v is not None else None))
            rec.add(v, "history", r.json())


TABLES = [
    ("practice_nlhe", [HUMAN, HUMAN, bot(), bot(), bot("check_or_call"), bot("check_or_fold")]),
    ("practice_nlhe", [HUMAN, HUMAN]),
    ("uncertainty_plo4", [HUMAN, HUMAN, bot("check_or_call")]),
    ("uncertainty_shortdeck", [HUMAN, HUMAN, HUMAN, bot(), bot(), bot("check_or_call"), bot(), bot()]),
]


@pytest.mark.parametrize(("ruleset", "seats"), TABLES, ids=[f"{r}-{len(s)}" for r, s in TABLES])
def test_no_unauthorized_cards_in_any_body_frame_or_error(ruleset, seats):
    seed = f"leak-{ruleset}-{len(seats)}"
    with harness() as h:
        h.setup(seats, seed=seed, ruleset_id=ruleset, starting_stack=200, hand_cap=10)
        rec = Recorder()
        play_recorded(h, rec, seed)
        r = h.client.get(f"/tables/{h.table_id}/history", headers=auth(h.host), params={"perspective": "reviewer"})
        assert r.status_code == 200
        ledger = Ledger(r.json()["events"])

    problems = [p for c in rec.items for p in check(c, ledger)]
    assert problems == []

    kinds = {c.kind for c in rec.items}
    assert kinds == {"snapshot", "receipt", "error", "history", "frame"}
    frames = [c for c in rec.items if c.kind == "frame"]
    assert len(frames) > 10
    # The checks are not vacuous: seats saw their own cards, and opponents held cards they never saw.
    assert any(c.body.get("view") and c.body["view"]["hole_cards"] for c in frames if c.seat is not None)
    assert any(ledger.hole[h_][s] - ledger.allowed(h_, 0) for h_ in ledger.hole for s in ledger.hole[h_] if s != 0)
    assert any(ledger.revealed.values()), "no showdown was exercised"
    assert {c.body["error"]["code"] for c in rec.items if c.kind == "error"} >= {"out_of_turn", "illegal_action"}


def test_leak_checker_catches_an_injected_card():
    seed = "leak-canary"
    with harness() as h:
        h.setup([HUMAN, HUMAN, bot()], seed=seed, starting_stack=200, hand_cap=2)
        snap = h.snap(0)
        hand_id = snap["view"]["hand_id"]
        opponent_card = sorted(h.manager.tables[h.table_id].table.hands[-1].deck)[0]
        while h.snap()["status"] != "complete":
            seat = h.actor()
            h.act(seat, {"type": "fold"} if h.snap(seat)["view"]["legal"]["fold"] else {"type": "check"})
        ledger = Ledger(h.client.get(f"/tables/{h.table_id}/history", headers=auth(h.host),
                                     params={"perspective": "reviewer"}).json()["events"])
    hidden = sorted(ledger.deck[hand_id] - ledger.allowed(hand_id, 0))
    assert hidden, opponent_card
    assert check_snapshot(snap, 0, ledger) == []
    for mutate in (
        lambda s: s["view"]["players"][1].__setitem__("shown_cards", [hidden[0], hidden[1]]),
        lambda s: s["view"]["action_history"].append({"seq": 999, "hand_id": hand_id, "type": "HoleDealt",
                                                      "data": {"seat": 1, "card": hidden[0]}}),
        lambda s: s["setup"].__setitem__("note", f"debug deck {hidden[0]}"),
        lambda s: s["view"]["board"].append(hidden[0]),
        lambda s: s["stats"]["seats"]["1"]["showdowns"]["items"].append(
            {"hand_id": hand_id, "cards": [hidden[0], hidden[1]], "board": []}),
    ):
        leaked = copy.deepcopy(snap)
        mutate(leaked)
        assert check_snapshot(leaked, 0, ledger), json.dumps(leaked)[:200]
    assert check(Capture(0, "error", {"error": {"message": f"bad card {hidden[0]}"}}), ledger)
