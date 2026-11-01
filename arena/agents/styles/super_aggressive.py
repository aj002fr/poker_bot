"""Style bot. Super-aggressive: steals with any two cards late, reraises light, sets check-raise traps. NLHE only; see :mod:`arena.agents.styles.engine`."""

from arena.agents.styles.engine import StyleBot


def make_bot():
    return StyleBot("super_aggressive")
