"""Endgame bot: the shared base policy with a fixed chip-EV push/fold chart. See :mod:`strategy.endgame.bot`."""

from strategy.endgame.bot import PushFoldBot


def make_bot():
    return PushFoldBot("chart")
