"""Endgame bot: the shared base policy with no push/fold layer. See :mod:`strategy.endgame.bot`."""

from strategy.endgame.bot import PushFoldBot


def make_bot():
    return PushFoldBot("none")
