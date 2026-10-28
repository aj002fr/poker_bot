import json
from fractions import Fraction

import pytest

from arena.domain.cards import full_deck, parse_cards
from arena.domain.models import TournamentConfig
from arena.domain.rng import SeededRng, shuffled
from arena.rules.pokerkit_adapter import HandSpec, engine_order
from arena.rules.profiles import get_profile
from arena.table.driver import check_or_call, check_or_fold, legal_random, play_out, shove
from arena.table.positions import Positions, next_positions
from arena.table.service import ActionRejected, Table, seeded_table
from arena.tournament.scoring import SCORE_TOLERANCE, Standings, icm
from tests.oracles.icm import Bust, icm_permutations, table_points
from tests.support.fixtures import FIXTURE_ROOT, build_deck, deal_order

POSITIONS = json.loads((FIXTURE_ROOT / "tournament" / "positions.json").read_text())
ICM = json.loads((FIXTURE_ROOT / "tournament" / "icm.json").read_text())


@pytest.mark.parametrize("fx", POSITIONS, ids=[f["id"] for f in POSITIONS])
def test_position_progression(fx):
    got = next_positions(Positions(**fx["prev"]), fx["ring"], fx["survivors"])
    assert got == Positions(**fx["expect"])
    if "posting_small_blind" in fx:
        assert got.small_blind_seat(fx["survivors"]) == fx["posting_small_blind"]
    spec = HandSpec(
        profile=get_profile("practice_nlhe"),
        seat_ring=tuple(fx["ring"]),
        stacks={s: 1000 for s in fx["survivors"]},
        button=got.button,
        small_blind_seat=got.small_blind_seat(fx["survivors"]),
        big_blind_seat=got.big_blind,
        small_blind=10,
        big_blind=20,
        deck=full_deck(get_profile("practice_nlhe").ranks),
    )
    assert list(engine_order(spec)) == fx["engine_order"]


@pytest.mark.parametrize("fx", ICM, ids=[f["id"] for f in ICM])
def test_icm_fixture(fx):
    got = icm(fx["stacks"], fx["prizes"])
    oracle = icm_permutations(dict(enumerate(fx["stacks"])), fx["prizes"]).points
    for i, value in enumerate(got):
        assert abs(value - float(oracle[i])) <= SCORE_TOLERANCE
    if "expect" in fx:
        assert all(abs(a - b) <= SCORE_TOLERANCE for a, b in zip(got, fx["expect"]))
    if "expect_fraction" in fx:
        want = [Fraction(n, d) for n, d in fx["expect_fraction"]]
        assert all(abs(a - float(b)) <= SCORE_TOLERANCE for a, b in zip(got, want))
        assert [oracle[i] for i in range(len(want))] == want
    if "expect_total" in fx:
        assert abs(sum(got) - fx["expect_total"]) <= SCORE_TOLERANCE


def test_icm_matches_permutation_oracle_randomized():
    rng = SeededRng(7, "icm-fast")
    for case in range(150):
        n = 2 + case % 7
        stacks = [1 + rng.randbelow(5000) for _ in range(n)]
        shape = case % 3
        if shape == 0:
            prizes = sorted((rng.randbelow(60) for _ in range(n)), reverse=True)
        elif shape == 1:
            prizes = [100] + [0] * (n - 1)
        else:
            prizes = [50, 50] + [0] * (n - 2)
        if sum(prizes) == 0:
            prizes[0] = 1
        got = icm(stacks, prizes)
        oracle = icm_permutations(dict(enumerate(stacks)), prizes, exact=False).points
        assert max(abs(got[i] - oracle[i]) for i in range(n)) <= SCORE_TOLERANCE
        assert abs(sum(got) - sum(prizes)) <= SCORE_TOLERANCE


def test_t09_icm_only_over_remaining_prizes():
    seats = list(range(6))
    s = Standings([50, 30, 20, 0, 0, 0], seats)
    s.record_eliminations(10, {5: 100}, {5: 0, 0: 1, 1: 1, 2: 1, 3: 1, 4: 1})
    s.record_eliminations(20, {4: 100}, {4: 0, 0: 1, 1: 1, 2: 1, 3: 1})
    s.record_eliminations(30, {3: 100}, {3: 0, 0: 1, 1: 1, 2: 1})
    final = s.finalize(200, {0: 3000, 1: 2000, 2: 1000})
    assert [p.points for p in s.placements.values() if p.method == "elimination"] == [0.0, 0.0, 0.0]
    assert abs(sum(p.points for p in final) - 100) <= SCORE_TOLERANCE
    want = icm([3000, 2000, 1000], [50, 30, 20])
    assert [p.points for p in final] == want


def test_t10_same_hand_busts_rank_by_start_stack():
    s = Standings([50, 30, 20, 0], [0, 1, 2, 3])
    out = s.record_eliminations(5, {1: 300, 2: 500, 3: 100}, {0: 8000, 1: 0, 2: 0, 3: 0})
    by_seat = {p.seat: p for p in out}
    assert by_seat[2].positions == (2,) and by_seat[2].points == 30
    assert by_seat[1].positions == (3,) and by_seat[1].points == 20
    assert by_seat[3].positions == (4,) and by_seat[3].points == 0
    s.finalize(5, {0: 8000})
    oracle = table_points([50, 30, 20, 0], [[Bust(1, 300), Bust(2, 500), Bust(3, 100)]], {0: 8000})
    assert s.points() == {k: float(v) for k, v in oracle.items()}


def test_t11_equal_start_stack_busts_split_positions():
    s = Standings([50, 30, 20], [0, 1, 2])
    out = s.record_eliminations(9, {1: 400, 2: 400}, {0: 6000, 1: 0, 2: 0})
    assert [(p.positions, p.points) for p in out] == [((2, 3), 25.0), ((2, 3), 25.0)]
    s.finalize(9, {0: 6000})
    assert sum(s.points().values()) == 100


def test_standings_match_oracle_randomized():
    rng = SeededRng(11, "standings")
    for case in range(200):
        n = 2 + case % 7
        payout = sorted((rng.randbelow(50) for _ in range(n)), reverse=True)
        payout[0] += 1
        players = list(range(n))
        s = Standings(payout, players)
        alive = list(players)
        bust_hands = []
        hand = 0
        while len(alive) > 1 and rng.randbelow(3):
            hand += 1
            k = 1 + rng.randbelow(len(alive) - 1)
            busted = [alive.pop(rng.randbelow(len(alive))) for _ in range(k)]
            start = {p: 1 + rng.randbelow(4) * 100 for p in busted}
            s.record_eliminations(hand, start, {**{p: 0 for p in busted}, **{p: 1 for p in alive}})
            bust_hands.append([Bust(p, start[p]) for p in busted])
        survivors = {p: 1 + rng.randbelow(3000) for p in alive}
        s.finalize(hand + 1, survivors)
        oracle = table_points(payout, bust_hands, survivors, exact=False)
        assert all(abs(s.points()[p] - oracle[p]) <= SCORE_TOLERANCE for p in players)


# ----------------------------------------------------------------- table level


def _config(**kw):
    return TournamentConfig(**kw)


def test_t01_level_changes_only_from_hand_26():
    t = seeded_table(_config(initial_player_count=3), "t01")
    play_out(t, check_or_fold, SeededRng(1, "policy"))
    by_number = {h.hand_number: h for h in t.hands}
    assert by_number[25].level == 0 and by_number[25].blinds == (10, 20)
    assert by_number[26].level == 1 and by_number[26].blinds == (15, 30)
    assert by_number[176].level == 7 and by_number[200].blinds == (200, 400)


def test_t02_cap_finishes_hand_200_and_scores_survivors_by_icm():
    t = seeded_table(_config(initial_player_count=4), "t02")
    play_out(t, check_or_fold, SeededRng(2, "policy"))
    assert t.hand_number == 200 and t.hands[-1].hand_number == 200
    assert all(h.result is not None for h in t.hands)
    icm_awards = [p for p in t.final_placements if p.method == "icm"]
    assert len(icm_awards) == len(t.live_seats) >= 2
    want = icm([t.stacks[p.seat] for p in icm_awards], [50, 30, 20, 0][: len(icm_awards)])
    assert [p.points for p in icm_awards] == want
    assert abs(sum(t.points().values()) - 100) <= SCORE_TOLERANCE


def test_t03_winner_before_cap_gets_first_place_once():
    t = seeded_table(_config(initial_player_count=6), "t03")
    play_out(t, shove, SeededRng(3, "policy"))
    assert t.hand_number < 200
    winners = [p for p in t.final_placements]
    assert len(winners) == 1 and winners[0].method == "winner" and winners[0].points == 50
    assert not any(p.method == "icm" for p in t.standings.placements.values())
    assert abs(sum(t.points().values()) - 100) <= SCORE_TOLERANCE


def test_g03_eight_players_then_one_busts():
    profile = get_profile("practice_nlhe")
    ring = list(range(8))
    hole = {0: parse_cards("AsAd"), 3: parse_cards("7c2d")}
    first = build_deck(profile.ranks, deal_order(ring, ring, 0, 2), 2, hole, parse_cards("KcQh9s5d4h"))
    rng = SeededRng(4, "env")

    def decks(n, p):
        return first if n == 1 else tuple(shuffled(full_deck(p.ranks), rng))

    stacks = {s: 2000 for s in ring} | {3: 20}
    t = Table(_config(initial_player_count=8), deck_source=decks, initial_button=0, start_stacks=stacks, env_rng=rng)
    assert len(t.hand.seats) == 8
    while t.hand_number == 1:
        t.submit(t.actor_seat, check_or_call(t.actor_seat, t.legal_actions(), rng))
    record = t.hands[-1]
    assert sorted(record.start_stacks) == [0, 1, 2, 4, 5, 6, 7]
    assert t.hands[0].eliminations[0].seat == 3 and t.hands[0].eliminations[0].positions == (8,)
    assert record.positions == Positions(button=1, small_blind=2, big_blind=4, heads_up=False)
    assert t.hand.seats == (2, 4, 5, 6, 7, 0, 1)
    play_out(t, legal_random, rng)
    assert t.complete and abs(sum(t.points().values()) - 100) <= SCORE_TOLERANCE


def test_t10_table_level_simultaneous_bust():
    profile = get_profile("practice_nlhe")
    ring = [0, 1, 2, 3]
    hole = {0: parse_cards("AsAd"), 1: parse_cards("7c2d"), 2: parse_cards("8c3d"), 3: parse_cards("9c4d")}
    deck = build_deck(profile.ranks, deal_order(ring, ring, 0, 2), 2, hole, parse_cards("KcQh Js 5h 6s"))
    stacks = {0: 5000, 1: 300, 2: 500, 3: 2200}
    t = Table(_config(initial_player_count=4), deck_source=lambda n, p: deck, initial_button=0, start_stacks=stacks,
              env_rng=SeededRng(5))
    t.submit(3, {"type": "raise_to", "amount": 2200})
    t.submit(0, {"type": "call"})
    t.submit(1, {"type": "call"})
    t.submit(2, {"type": "call"})
    elim = {p.seat: p for p in t.hands[0].eliminations}
    assert elim[3].positions == (2,) and elim[2].positions == (3,) and elim[1].positions == (4,)
    assert t.complete and t.points() == {0: 50.0, 3: 30.0, 2: 20.0, 1: 0.0}


def test_g24_attacks_rejected_without_mutation():
    t = seeded_table(_config(initial_player_count=3), "g24")
    seat = t.actor_seat
    hand_id, decision = t.hands[-1].hand_id, t.decision_id
    snapshot = (t.state_version, len(t.receipts), dict(t.stacks), seat, len(t.hands[-1].events), t.hand.chips_in_play())
    attacks = [
        ({"type": "raise_to", "amount": -40}, {}, "invalid_schema"),
        ({"type": "raise_to", "amount": 40.0}, {}, "invalid_schema"),
        ({"type": "raise_to", "amount": True}, {}, "invalid_schema"),
        ({"type": "raise_to", "amount": "40"}, {}, "invalid_schema"),
        ({"type": "raise_to", "amount": 10**15}, {}, "invalid_schema"),
        ({"type": "raise_to", "amount": float("nan")}, {}, "invalid_schema"),
        ({"type": "call", "amount": 20}, {}, "invalid_schema"),
        ({"type": "shove"}, {}, "invalid_schema"),
        ({"type": "fold", "seat": 0}, {}, "invalid_schema"),
        ({"type": "raise_to", "amount": 39}, {}, "illegal_action"),
        ({"type": "check"}, {}, "illegal_action"),
        ({"type": "call"}, {"hand_id": "h_999"}, "wrong_hand"),
        ({"type": "call"}, {"decision_id": "d_00000"}, "stale_decision"),
        ({"type": "call"}, {"expected_state_version": t.state_version + 5}, "stale_version"),
    ]
    for action, kw, code in attacks:
        with pytest.raises(ActionRejected) as exc:
            t.submit(seat, action, **kw)
        assert exc.value.code == code, (action, exc.value)
    other = next(s for s in t.live_seats if s != seat)
    with pytest.raises(ActionRejected) as exc:
        t.submit(other, {"type": "fold"})
    assert exc.value.code == "out_of_turn"
    after = (t.state_version, len(t.receipts), dict(t.stacks), t.actor_seat, len(t.hands[-1].events), t.hand.chips_in_play())
    assert after == snapshot
    t.submit(seat, {"type": "call"}, hand_id=hand_id, decision_id=decision)
    with pytest.raises(ActionRejected) as exc:
        t.submit(t.actor_seat, {"type": "fold"}, decision_id=decision)
    assert exc.value.code == "stale_decision"


def test_fallback_checks_when_possible_else_folds():
    t = seeded_table(_config(initial_player_count=3), "fallback")
    r = t.apply_fallback(t.actor_seat, "timeout")
    assert r.action.type == "fold" and r.fallback_reason == "timeout"
    t.submit(t.actor_seat, {"type": "call"})
    r = t.apply_fallback(t.actor_seat, "worker_crash")
    assert r.action.type == "check"


def test_t12_new_table_has_clean_state():
    cfg = _config(initial_player_count=3)
    a = seeded_table(cfg, "same")
    play_out(a, legal_random, SeededRng(1))
    b = seeded_table(cfg, "same")
    assert b.hand_number == 1 and b.receipts == [] and b.stacks == {0: 2000, 1: 2000, 2: 2000}
    assert b.standings is not a.standings and b.hands[0].deck == a.hands[0].deck


@pytest.mark.parametrize("ruleset", ["practice_nlhe", "uncertainty_shortdeck", "uncertainty_plo4"])
def test_t13_same_envelope_across_scenarios(ruleset):
    cfg = _config(ruleset_id=ruleset, initial_player_count=5)
    t = seeded_table(cfg, f"t13-{ruleset}")
    play_out(t, check_or_fold, SeededRng(13))
    assert t.hand_number == 200 and [h.blinds for h in t.hands[24:26]] == [(10, 20), (15, 30)]
    assert len(t.standings.placements) == 5 and abs(sum(t.points().values()) - 100) <= SCORE_TOLERANCE
    t2 = seeded_table(cfg, f"t13b-{ruleset}")
    play_out(t2, legal_random, SeededRng(14))
    assert t2.complete and abs(sum(t2.points().values()) - 100) <= SCORE_TOLERANCE
    assert sum(t2.stacks.values()) == 5 * 2000
