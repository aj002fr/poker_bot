"""Endgame bot: the shared base policy with chip-EV push/fold tables. See :mod:`strategy.endgame.bot`."""

from strategy.endgame.bot import PushFoldBot


def make_bot():
    return PushFoldBot("chips")
