"""Push/fold bots over one shared, variant-agnostic base policy (decision-time safe).

Every arm plays the same :class:`BasePolicy` and differs only in the push/fold layer,
so a paired comparison measures the tables alone:

``points``  point-aware tables (ICM continuation value, cap-hand ICM, eliminations);
``chips``   chip-EV tables over the same states and keys;
``chart``   a fixed chart: chip-EV, symmetric stacks, coarsest keys only;
``response``  best responses in points to the train-partition population curves;
``capped``  best responses to a half-curves, half-equilibrium mixture;
``none``    the base policy alone.

The layer takes over preflop when the hand fits the push/fold abstraction
(:func:`~strategy.endgame.lookup.preflop_spot`), an open has at most ``open_max_bb``
big blinds at risk (:data:`BLIND_OPEN_MAX_BB` from the small blind, where limping starts
to pay), and a call or overcall at most the tables' 30. Its action is a shove
(the all-in raise; a call when the hero cannot raise or nobody behind can act) or a
fold. Randomisation is keyed by the table's policy seed and the decision id, so play
replays exactly.

The base policy ranks hand classes by equity against one random hand from the
variant's own showdown table and plays a plain tight-aggressive game: fixed opening
and calling ranges preflop, made-hand strength postflop.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache

from arena.agents.sdk import ExplicitStateBot, call, check, clamp_raise_to, fold, raise_to
from strategy.endgame.classes import hand_classes
from strategy.endgame.keys import MAX_DEPTH_BB
from strategy.endgame.lookup import load_table, preflop_spot
from strategy.features.cards import capabilities, card_ints, made_hand
from strategy.tables import store

MODES = ("points", "chips", "chart", "response", "capped", "none")
OPEN_MAX_BB = 15.0
BLIND_OPEN_MAX_BB = 10.0
"""Small-blind opens (only the big blind behind) use push/fold up to here: beyond it the
measured equilibrium gain of limping exceeds 0.02 big blinds (:mod:`strategy.endgame.limp`)."""
RULESETS = ("practice_nlhe", "uncertainty_shortdeck")


@lru_cache(maxsize=None)
def class_percentiles(ruleset_id: str) -> tuple[float, ...]:
    """Share of combos stronger than each class (0 is the best hand), by equity vs a random hand."""
    loaded = store.load_arrays(f"showdown-{ruleset_id}")
    if loaded is None:
        raise LookupError(f"showdown-{ruleset_id}.npz is missing; run `make tables-showdown`")
    prov, a = loaded
    hc = hand_classes(ruleset_id)
    if prov.get("variant_hash") != hc.variant_hash or tuple(a["class_keys"].tolist()) != hc.keys:
        raise store.TableError(f"showdown-{ruleset_id} was built for other hand classes")
    eq = a["eq_random"].tolist()
    combos = a["class_combos"].tolist()
    total = float(sum(combos))
    out = []
    for i, e in enumerate(eq):
        better = sum(c for e2, c in zip(eq, combos) if e2 > e)
        out.append((better + combos[i] / 2) / total)
    return tuple(out)


def _uniform(seed: str, decision_id: str, salt: str = "") -> float:
    h = hashlib.sha256(f"endgame-bot-v1|{seed}|{decision_id}|{salt}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2.0 ** 64


class BasePolicy:
    def __init__(self, ruleset_id: str) -> None:
        self.ruleset_id = ruleset_id
        self.classes = hand_classes(ruleset_id)
        self.caps = capabilities(ruleset_id)
        self.pct = class_percentiles(ruleset_id)

    def hand_class(self, obs) -> int:
        return self.classes.index(card_ints(obs.hole_cards))

    def decide(self, obs):
        legal = obs.legal
        to_call = legal.call_amount or 0
        bb = obs.blinds.big_blind if obs.blinds else 1
        me = next(p for p in obs.players if p.seat == obs.hero_seat)
        if obs.street == "preflop":
            pct = self.pct[self.hand_class(obs)]
            behind = self._behind(obs)
            if obs.current_bet_to <= bb:
                if to_call == 0:
                    if pct <= 0.10:
                        return clamp_raise_to(legal, 3 * bb + obs.current_bet_to) or check()
                    return check()
                limped = any(p.hand_committed >= bb and p.seat != obs.big_blind_seat for p in obs.players
                             if p.seat != obs.hero_seat)
                if limped:
                    if pct <= 0.06:
                        return clamp_raise_to(legal, 4 * bb) or call()
                    return call() if pct <= 0.22 else fold()
                open_range = 0.35 if behind <= 1 else 0.25 if behind == 2 else 0.20 if behind == 3 else 0.15
                if pct <= open_range:
                    return clamp_raise_to(legal, int(2.5 * bb)) or call()
                return fold()
            if pct <= 0.04:
                return clamp_raise_to(legal, 3 * obs.current_bet_to) or call()
            if to_call * 3 > me.stack:
                return call() if pct <= 0.06 else fold()
            return call() if pct <= 0.12 else fold()
        made = made_hand(card_ints(obs.hole_cards), card_ints(obs.board), self.caps)
        pot = obs.pot_total
        if made.level >= 5:
            bet = clamp_raise_to(legal, obs.current_bet_to + (2 * (pot + to_call)) // 3)
            return bet or (call() if to_call else check())
        if made.level >= 3:
            if to_call == 0:
                return clamp_raise_to(legal, pot // 2) or check()
            return call() if to_call * 2 <= pot else fold()
        return check() if legal.check else fold()

    @staticmethod
    def _behind(obs) -> int:
        """Players still to act preflop after the hero in the first round."""
        ring = sorted(obs.seats)
        live = {p.seat for p in obs.players if p.status == "active" and p.stack > 0}
        bb = obs.big_blind_seat
        i = ring.index(obs.hero_seat)
        n = 0
        for s in ring[i + 1:] + ring[:i]:
            if s in live:
                n += 1
            if s == bb:
                break
        return n


class PushFoldBot(ExplicitStateBot):
    rulesets = RULESETS

    def __init__(self, mode: str, *, open_max_bb: float = OPEN_MAX_BB) -> None:
        if mode not in MODES:
            raise ValueError(f"mode is one of {MODES}")
        self.mode = mode
        self.name = f"endgame_{mode}"
        self.open_max_bb = open_max_bb
        self._base: dict[str, BasePolicy] = {}

    def initial_state(self, table):
        if self.mode != "none":
            load_table(self.mode, table.ruleset_id)
        self._policy(table.ruleset_id)
        return {"seed": table.policy_seed}

    def _policy(self, ruleset_id: str) -> BasePolicy:
        p = self._base.get(ruleset_id)
        if p is None:
            p = self._base[ruleset_id] = BasePolicy(ruleset_id)
        return p

    def push_fold(self, obs, base: BasePolicy, seed: str):
        """The push/fold action, or None when the layer does not apply."""
        if self.mode == "none":
            return None
        spot = preflop_spot(obs)
        if spot is None:
            return None
        if spot.kind == "open":
            limit = BLIND_OPEN_MAX_BB if spot.hero == spot.snapshot.n - 2 else self.open_max_bb
        else:
            limit = MAX_DEPTH_BB
        if spot.risk_bb > limit:
            return None
        table = load_table(self.mode, obs.ruleset_id)
        found = table.lookup(spot.kind, spot.keys(), base.hand_class(obs))
        if found is None:
            return None
        p, _ = found
        legal = obs.legal
        if _uniform(seed, obs.decision_id) < p:
            if legal.can_raise and (spot.kind == "open" or spot.behind_live) and legal.all_in_raise_to is not None:
                return raise_to(legal.all_in_raise_to)
            return call() if legal.call_amount else check()
        return check() if legal.check else fold()

    def decide(self, obs, state):
        base = self._policy(obs.ruleset_id)
        seed = (state or {}).get("seed", "")
        action = self.push_fold(obs, base, seed)
        return (action if action is not None else base.decide(obs)), state


def make_bot(mode: str) -> PushFoldBot:
    return PushFoldBot(mode)
