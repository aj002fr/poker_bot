"""Style bot. Style switcher: conservative early, aggressive as blinds rise, push/fold when short. NLHE only; see :mod:`arena.agents.styles.engine`."""

from arena.agents.styles.engine import StyleBot


def make_bot():
    return StyleBot("switcher")
