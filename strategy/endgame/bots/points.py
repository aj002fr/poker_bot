"""Endgame bot: the shared base policy with point-aware push/fold tables. See :mod:`strategy.endgame.bot`."""

from strategy.endgame.bot import PushFoldBot


def make_bot():
    return PushFoldBot("points")
