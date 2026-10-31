"""Style bot. Weak-tight: tight, small raises, backs down to big reraises. NLHE only; see :mod:`arena.agents.styles.engine`."""

from arena.agents.styles.engine import StyleBot


def make_bot():
    return StyleBot("weak_tight")
