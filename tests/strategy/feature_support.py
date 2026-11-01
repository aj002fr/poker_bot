"""Shared fixtures for the strategy feature and belief tests: recorded seeded observations."""

from __future__ import annotations

from functools import cache

RULESETS = ("practice_nlhe", "uncertainty_shortdeck", "uncertainty_plo4")


@cache
def recorded(ruleset_id: str, hand_cap: int = 40, seed: int = 3) -> tuple[tuple[int, tuple[int, ...], tuple], ...]:
    """``(seat, seats, observations)`` for every seat of one seeded in-process table."""
    from strategy.features.bench import record

    return tuple((s, tuple(seats), tuple(obs)) for s, seats, obs in record(ruleset_id, tables=1, hand_cap=hand_cap,
                                                                           seed=seed))


def hand_events(start_stacks: dict[int, int], button: int, sb_seat: int | None, bb_seat: int, *, sb: int = 10,
                bb: int = 20, hand_number: int = 1, heads_up: bool = False, seq0: int = 0,
                hand_id: str = "h1") -> list[dict]:
    """Public events opening a hand: HandStarted plus the blinds."""
    events = [{"seq": seq0, "hand_id": hand_id, "type": "HandStarted",
               "data": {"hand_number": hand_number, "level": 0, "small_blind": sb, "big_blind": bb,
                        "positions": {"button": button, "big_blind": bb_seat, "heads_up": heads_up},
                        "small_blind_seat": sb_seat, "start_stacks": {str(s): v for s, v in start_stacks.items()}}}]
    if sb_seat is not None:
        events.append({"seq": seq0 + 1, "hand_id": hand_id, "type": "BlindPosted",
                       "data": {"seat": sb_seat, "amount": sb}})
    events.append({"seq": seq0 + 2, "hand_id": hand_id, "type": "BlindPosted", "data": {"seat": bb_seat, "amount": bb}})
    return events


class Actions:
    """Builds ActionApplied/BoardDealt events with consistent ``added``/``street_committed``."""

    def __init__(self, events: list[dict], blinds: dict[int, int], hand_id: str = "h1") -> None:
        self.events = events
        self.seq = events[-1]["seq"] + 1
        self.hand_id = hand_id
        self.street = "preflop"
        self.sc = dict(blinds)

    def _push(self, etype: str, data: dict) -> None:
        self.events.append({"seq": self.seq, "hand_id": self.hand_id, "type": etype, "data": data})
        self.seq += 1

    def act(self, seat: int, kind: str, to: int | None = None, *, auto: bool = False) -> Actions:
        prev = self.sc.get(seat, 0)
        level = max(self.sc.values(), default=0)
        new = {"fold": prev, "check": prev, "call": level, "raise_to": to}[kind]
        assert new is not None
        self.sc[seat] = new
        action = {"type": kind} if to is None else {"type": kind, "amount": to}
        self._push("ActionApplied", {"seat": seat, "street": self.street, "action": action, "added": new - prev,
                                     "street_committed": new, "auto": auto})
        return self

    def board(self, street: str, cards: list[str]) -> Actions:
        self.street = street
        self.sc = {}
        self._push("BoardDealt", {"street": street, "cards": cards})
        return self
