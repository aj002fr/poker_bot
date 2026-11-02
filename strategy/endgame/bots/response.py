"""Endgame bot: the shared base policy with push/fold tables of best responses to the population response curves. See :mod:`strategy.endgame.bot`."""

from strategy.endgame.bot import PushFoldBot


def make_bot():
    return PushFoldBot("response")
