"""Opponent styles: measured stats match each declared profile, seeded play is exact,
notes stay out of the event log, and the shared helpers (charts, evaluator, tracker) agree
with hand-checked cases."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pytest

from arena.agents.baselines import BASELINES
from arena.agents.sdk import MAX_NOTE_CHARS, TableInfo, load_bot, split_note, with_note
from arena.agents.styles import ALIASES, PROFILES, STYLES, resolve_style, style_artifact_hash
from arena.agents.styles.charts import ALL_CLASSES, CHARTS, expand, grid, hand_class, seat_positions
from arena.agents.styles.handeval import draws, equity_vs_random, evaluate, hand_words
from arena.agents.worker import InProcessSeat, WorkerSeat, play_table
from arena.benchmark.runner import policy_seed
from arena.domain.models import TournamentConfig
from arena.table.player_stats import StatsTracker
from arena.table.projections import Viewer, project_events
from arena.table.service import seeded_table
from tests.agents.conftest import info_for, needs_sandbox

ALL_STYLES = list(STYLES)


def _seat(path: Path, name: str, config: TournamentConfig, seat: int, seed: str) -> InProcessSeat:
    bot, info = load_bot(path, module_name=f"test_style_{name}")
    table_info = TableInfo(table_id="t_styles", hero_seat=seat, ruleset_id=config.ruleset_id, config=config.resolved(),
                           config_hash=config.config_hash(), seats=config.resolved_seats,
                           decision_budget_ns=config.bot_budget_ns, policy_seed=policy_seed(seed, seat))
    agent = InProcessSeat(seat, bot, info, table_info, "test")
    agent.start()
    return agent


def play(lineup: list[str], seed: str, *, hand_cap: int = 80):
    """One seeded table, in-process. Returns (table, {seat: name}, [(seat, decision_id, note)])."""
    config = TournamentConfig(initial_player_count=len(lineup), starting_stack=10000, hand_cap=hand_cap,
                              hands_per_level=hand_cap // 8 + 1)
    names = dict(zip(config.resolved_seats, lineup, strict=True))
    agents = {s: _seat(STYLES.get(n) or BASELINES[n], n, config, s, seed) for s, n in names.items()}
    table = seeded_table(config, seed, table_id="t_styles")
    notes: list[tuple[int, str, str | None]] = []
    pending: dict[str, str] = {}
    play_table(table, agents, before_decision=lambda i, s: pending.__setitem__("d", table.decision_id),
               on_decision=lambda s, d, t: notes.append((s, pending["d"], d.note)))
    return table, names, notes


def measure(lineup: list[str], tables: int, seed: str) -> dict[str, dict[str, int]]:
    totals: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for t in range(tables):
        rotated = [lineup[(i + t) % len(lineup)] for i in range(len(lineup))]
        table, names, _ = play(rotated, f"{seed}-{t}")
        tracker = StatsTracker(table.seat_ring)
        tracker.feed(project_events(table.log, Viewer.spectator()))
        for seat, name in names.items():
            c = tracker.counts[seat]
            for k in ("hands", "vpip", "pfr", "bets_raises", "calls", "faced", "folded"):
                totals[name][k] += getattr(c, k)
    return totals


def _assert_in_profile(name: str, c: dict[str, int]) -> None:
    e = PROFILES[name].expected
    measured = {
        "vpip": c["vpip"] / c["hands"],
        "pfr": c["pfr"] / c["hands"],
        "aggression": c["bets_raises"] / max(1, c["calls"]),
        "fold_to_reraise": c["folded"] / c["faced"] if c["faced"] >= 20 else None,
    }
    for stat, value in measured.items():
        band = getattr(e, stat)
        if band is None or value is None:
            continue
        assert band[0] <= value <= band[1], f"{name} {stat} {value:.3f} outside {band} ({dict(c)})"


@pytest.mark.parametrize(("lineup", "tables"), [
    (ALL_STYLES, 12),
    (["rock", "aggressive", "station", "nlhe_heuristic", "calling_station", "legal_random"], 10),
    (["super_aggressive", "weak_tight", "switcher", "nlhe_heuristic", "check_fold", "calling_station"], 10),
], ids=["all-styles", "with-baselines-a", "with-baselines-b"])
def test_measured_stats_fall_in_declared_profile_ranges(lineup, tables):
    totals = measure(lineup, tables, seed=f"profile-{lineup[0]}")
    for name in lineup:
        if name in PROFILES:
            assert totals[name]["hands"] >= 300, name
            _assert_in_profile(name, totals[name])


def test_styles_are_distinguishable_by_their_stats():
    totals = measure(ALL_STYLES, 6, seed="distinct")
    vpip = {n: c["vpip"] / c["hands"] for n, c in totals.items()}
    pfr = {n: c["pfr"] / c["hands"] for n, c in totals.items()}
    assert vpip["weak_tight"] < vpip["rock"] < vpip["aggressive"] < vpip["super_aggressive"]
    assert pfr["station"] < 0.05 < pfr["rock"]
    assert vpip["station"] > vpip["aggressive"]


def test_seeded_play_is_exact_and_policy_seed_matters():
    first = play(ALL_STYLES, "determinism", hand_cap=25)
    again = play(ALL_STYLES, "determinism", hand_cap=25)
    assert [e.to_json_obj() for e in first[0].log] == [e.to_json_obj() for e in again[0].log]
    assert first[2] == again[2]
    other = play(ALL_STYLES, "determinism-other", hand_cap=25)
    assert [e.to_json_obj() for e in other[0].log] != [e.to_json_obj() for e in first[0].log]


def test_every_decision_has_a_note_and_notes_stay_out_of_the_log():
    table, _, notes = play(ALL_STYLES, "notes", hand_cap=10)
    assert notes and all(n and len(n) <= MAX_NOTE_CHARS for _, _, n in notes)
    log_text = "".join(str(e.to_json_obj()) for e in table.log)
    assert not any(n in log_text for _, _, n in notes if len(n) > 12)
    assert any(n.startswith("[") for s, _, n in notes if s == 5), "switcher notes name their gear"


def test_with_note_and_split_note():
    payload = with_note({"type": "fold"}, "too weak \x00 here" + "x" * 300)
    action, note = split_note(payload)
    assert action == {"type": "fold"}
    assert note is not None and len(note) <= MAX_NOTE_CHARS and "\x00" not in note
    assert split_note({"type": "check", "note": 5}) == ({"type": "check"}, None)
    assert split_note({"type": "check"}) == ({"type": "check"}, None)


def test_profiles_ids_and_hashes():
    assert set(PROFILES) == set(STYLES)
    assert {p.policy_id for p in PROFILES.values()} == {f"style_{s}" for s in STYLES}
    assert resolve_style("style_rock") == "rock" and resolve_style("Super") == "super_aggressive"
    assert resolve_style("calling_station") is None, "the baseline keeps its name"
    assert all(resolve_style(a) == t for a, t in ALIASES.items())
    hashes = {style_artifact_hash(s) for s in STYLES}
    assert len(hashes) == len(STYLES)
    for p in PROFILES.values():
        e = p.public()["expected"]
        assert all(v is None or v[0] <= v[1] for v in e.values())
        assert p.book_ref.part.startswith(("Part Two", "Part Three"))
    assert [p.id for p in PROFILES.values() if p.advanced] == ["switcher"]


@needs_sandbox
def test_style_files_load_in_a_sandboxed_worker(pool):
    config = TournamentConfig(initial_player_count=2, hand_cap=3, hands_per_level=1)
    seats = {s: WorkerSeat(s, pool.stage(STYLES[name]), info_for(config, s), pool.acquire,
                           budget_ns=config.bot_budget_ns)
             for s, name in zip(config.resolved_seats, ["rock", "switcher"], strict=True)}
    for agent in seats.values():
        agent.start()
    table = seeded_table(config, "sandboxed-style", table_id="t_test")
    notes = []
    try:
        play_table(table, seats, on_decision=lambda s, d, t: notes.append((d.fallback, d.note)))
    finally:
        for agent in seats.values():
            agent.close()
    assert table.complete
    accepted = [n for f, n in notes if f is None]
    assert accepted and all(accepted)


# ------------------------------------------------------------------ charts


def test_grid_and_range_parser():
    g = grid()
    assert len(g) == 13 and all(len(r) == 13 for r in g)
    flat = [h for r in g for h in r]
    assert len(set(flat)) == 169 and set(flat) == set(ALL_CLASSES)
    assert g[0][0] == "AA" and g[0][1] == "AKs" and g[1][0] == "AKo" and g[12][12] == "22"
    assert expand("TT+") == {"TT", "JJ", "QQ", "KK", "AA"}
    assert expand("99-77") == {"99", "88", "77"}
    assert expand("ATs+") == {"ATs", "AJs", "AQs", "AKs"}
    assert expand("KQ") == {"KQs", "KQo"}
    assert expand("54s-87s") == {"54s", "65s", "76s", "87s"}
    assert len(expand("Ax")) == 24 and len(expand("any")) == 169
    assert hand_class(["Kd", "Ah"]) == "AKo" and hand_class(["7s", "7c"]) == "77" and hand_class(["2h", "Jh"]) == "J2s"
    for chart in ("conservative", "aggressive"):
        early, middle, late = (CHARTS[chart][g_] for g_ in ("early", "middle", "late"))
        assert early <= middle <= late, chart
    assert CHARTS["conservative"]["late"] <= CHARTS["aggressive"]["late"]


def test_seat_positions():
    ring = [0, 1, 2, 3, 4, 5]
    pos = seat_positions(ring, ring, button=3, small_blind=4, big_blind=5)
    assert {s: p["label"] for s, p in pos.items()} == {0: "UTG", 1: "MP", 2: "CO", 3: "BTN", 4: "SB", 5: "BB"}
    assert pos[0]["group"] == "early" and pos[2]["group"] == "late" and pos[4]["group"] == "blinds"
    hu = seat_positions([0, 1], [0, 1], button=0, small_blind=0, big_blind=1)
    assert hu[0]["label"] == "BTN" and hu[1]["label"] == "BB"


# --------------------------------------------------------------- evaluator


def test_evaluator_orders_and_describes_hands():
    flush = evaluate(["Ah", "2h", "7h", "9h", "Kh", "Qc", "3d"])
    straight = evaluate(["5c", "6d", "7h", "8s", "9c", "Kd", "2s"])
    wheel = evaluate(["Ac", "2d", "3h", "4s", "5c", "Kd", "9s"])
    trips = evaluate(["7c", "7d", "7h", "As", "Kc", "2d", "4s"])
    assert flush > straight > wheel > trips
    assert hand_words(["Ah", "Kd"], ["2c", "7d", "Ks"]) == ("Pair of kings, ace kicker", "one_pair")
    assert hand_words(["Qh", "Qd"], [])[1] == "one_pair"
    assert hand_words(["Ah", "Kd"], [])[1] == "high_card"


def test_draws_and_equity():
    found, outs = draws(["Ah", "Kh"], ["2h", "7h", "Qc"])
    assert [d["kind"] for d in found] == ["flush_draw"] and outs == 9
    found, outs = draws(["8c", "9d"], ["6h", "7s", "Kc"])
    assert [d["kind"] for d in found] == ["open_ended"] and outs == 8
    found, outs = draws(["8c", "9d"], ["5h", "7s", "Kc"])
    assert [d["kind"] for d in found] == ["gutshot"] and outs == 4
    aa = equity_vs_random(["As", "Ad"], [], 1)
    assert 0.82 < aa[0] < 0.88 and not aa[2]
    assert equity_vs_random(["As", "Ad"], [], 1) == aa, "seeded from the inputs"
    assert equity_vs_random(["As", "Ad"], [], 4)[0] < aa[0]
    river = equity_vs_random(["As", "Ad"], ["Ah", "Ac", "2d", "7s", "9h"], 1)
    assert river == (1.0, 990, True)


# ----------------------------------------------------------------- tracker


def test_tracker_counts_baseline_behaviour():
    table, names, _ = play(["check_fold", "calling_station", "rock"], "tracker", hand_cap=30)
    tracker = StatsTracker(table.seat_ring)
    tracker.feed(project_events(table.log, Viewer.spectator()))
    seat = {n: s for s, n in names.items()}
    folder = tracker.seat(seat["check_fold"])
    assert folder["vpip"]["count"] == 0 and folder["pfr"]["count"] == 0
    assert folder["aggression"] == {"bets_raises": 0, "calls": 0, "factor": None}
    caller = tracker.seat(seat["calling_station"])
    assert caller["pfr"]["count"] == 0 and caller["aggression"]["bets_raises"] == 0
    assert caller["vpip"]["count"] > 0 and caller["hands"] == table.hand_number
    shown = [i for s in table.seat_ring for i in tracker.seat(s)["showdowns"]["items"]]
    assert shown and all(len(i["cards"]) == 2 and i["hand_in_words"] for i in shown)


def test_tracker_is_incremental_dedupes_and_round_trips():
    table, _, _ = play(ALL_STYLES, "incremental", hand_cap=12)
    events = project_events(table.log, Viewer.spectator())
    whole = StatsTracker(table.seat_ring)
    whole.feed(events)
    pieces = StatsTracker(table.seat_ring)
    for i in range(0, len(events), 7):
        pieces.feed(events[max(0, i - 3): i + 7])
        pieces = StatsTracker.from_state(pieces.to_state())
    assert pieces.payload() == whole.payload()
    as_dicts = StatsTracker(table.seat_ring)
    as_dicts.feed([e.model_dump(mode="json") for e in events])
    assert as_dicts.payload() == whole.payload()
