"""Spec 6.2 worked example: heads-up, preflop call/check, BB bets 20 on the flop."""

from arena.domain.models import Action, TournamentConfig
from arena.table.projections import Viewer, observation, table_view
from arena.table.service import seeded_table


def test_spec_observation_accounting():
    table = seeded_table(TournamentConfig(initial_player_count=2, seats=[0, 3]), "spec-6.2", initial_button=0)
    table.submit(0, Action(type="call"))
    table.submit(3, Action(type="check"))
    table.submit(3, Action(type="raise_to", amount=20))
    obs = observation(table, 0)
    assert (obs.hero_seat, obs.actor_seat, obs.button_seat, obs.small_blind_seat, obs.big_blind_seat) == (0, 0, 0, 0, 3)
    assert obs.street == "flop" and len(obs.board) == 3 and len(obs.hole_cards) == 2
    assert (obs.pot_total, obs.settled_pot, obs.current_bet_to) == (60, 40, 20)
    players = {p.seat: (p.stack, p.street_committed, p.hand_committed, p.status) for p in obs.players}
    assert players == {0: (1980, 0, 20, "active"), 3: (1960, 20, 40, "active")}
    assert sum(p.stack for p in obs.players) + obs.pot_total == 4000
    legal = obs.legal
    assert (legal.fold, legal.check, legal.call_amount, legal.can_raise) == (True, False, 20, True)
    assert (legal.min_full_raise_to, legal.max_raise_to, legal.short_all_in_raise_to) == (40, 1980, None)
    assert legal.pot_raise_to == 100 == 20 + 60 + 20

    table.submit(0, Action(type="raise_to", amount=100))
    view = table_view(table, Viewer.spectator())
    assert {p.seat: p.street_committed for p in view.players} == {0: 100, 3: 20}
    table.submit(3, Action(type="call"))
    view = table_view(table, Viewer.spectator())
    assert view.pot_total == 240 and view.street == "turn"
    assert view.hole_cards is None and all(p.shown_cards is None for p in view.players)
