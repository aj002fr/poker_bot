"""Style bot. Aggressive: wide opening charts and frequent continuation bets. NLHE only; see :mod:`arena.agents.styles.engine`."""

from arena.agents.styles.engine import StyleBot


def make_bot():
    return StyleBot("aggressive")
