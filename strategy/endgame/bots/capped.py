"""Endgame bot: the shared base policy with push/fold tables of best responses to a half-population, half-equilibrium mixture. See :mod:`strategy.endgame.bot`."""

from strategy.endgame.bot import PushFoldBot


def make_bot():
    return PushFoldBot("capped")
