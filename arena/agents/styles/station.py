"""Style bot. Calling station: loose and passive. NLHE only; see :mod:`arena.agents.styles.engine`."""

from arena.agents.styles.engine import StyleBot


def make_bot():
    return StyleBot("station")
