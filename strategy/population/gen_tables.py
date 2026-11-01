"""Generate ``data/preflop.json``: starting-hand orderings for the three practice variants.

Run ``uv run python -m strategy.population.gen_tables`` to regenerate. The output is a
generated artifact: changing it changes every population bot's behaviour, so the
registry lock (``registry.lock.json``) records its hash and must be re-frozen after.

* ``practice_nlhe``: 169 classes with heads-up equity against a random hand, by seeded
  Monte Carlo with the style evaluator, strongest first, with combo counts.
* ``uncertainty_shortdeck``: the same for the 81 classes of a 36-card deck, with the
  short-deck ranking (:func:`strategy.population.cards.evaluate_shortdeck`).
* ``uncertainty_plo4``: quantiles of :func:`strategy.population.cards.plo_score` over
  seeded random four-card hands (PLO has too many starting hands for a class table).
"""

from __future__ import annotations

import json
import random

from arena.agents.styles.charts import ALL_CLASSES
from arena.agents.styles.handeval import evaluate
from strategy.population import cards

SAMPLES = 4000
PLO_HANDS = 40000
PLO_QUANTILES = 400
SUITS = "cdhs"


def _combos(cls: str, ranks: str) -> list[tuple[str, str]]:
    a, b = cls[0], cls[1]
    if a not in ranks or b not in ranks:
        return []
    if a == b:
        cs = [a + s for s in SUITS]
        return [(cs[i], cs[j]) for i in range(4) for j in range(i + 1, 4)]
    if cls.endswith("s"):
        return [(a + s, b + s) for s in SUITS]
    return [(a + s, b + t) for s in SUITS for t in SUITS if s != t]


def _equity(cls: str, ranks: str, ev, rng: random.Random) -> float:
    deck_all = [r + s for r in ranks for s in SUITS]
    combos = _combos(cls, ranks)
    total = 0.0
    for i in range(SAMPLES):
        hole = combos[i % len(combos)]
        deck = [c for c in deck_all if c not in hole]
        draw = rng.sample(deck, 7)
        board = draw[:5]
        mine, theirs = ev([*hole, *board]), ev([draw[5], draw[6], *board])
        total += 1.0 if mine > theirs else 0.5 if mine == theirs else 0.0
    return round(total / SAMPLES, 4)


def _table(ranks: str, ev, seed: str) -> list[list]:
    rng = random.Random(seed)
    rows = []
    for cls in ALL_CLASSES:
        combos = _combos(cls, ranks)
        if combos:
            rows.append([cls, _equity(cls, ranks, ev, rng), len(combos)])
    rows.sort(key=lambda r: (-r[1], r[0]))
    return rows


def main() -> None:
    out: dict = {"generator": "strategy.population.gen_tables", "version": 1, "samples_per_class": SAMPLES}
    out["practice_nlhe"] = _table("23456789TJQKA", evaluate, "pop-preflop-nlhe-v1")
    out["uncertainty_shortdeck"] = _table("6789TJQKA", cards.evaluate_shortdeck, "pop-preflop-shortdeck-v1")
    cards.DATA.parent.mkdir(parents=True, exist_ok=True)
    cards.DATA.write_text(json.dumps(out, separators=(",", ":")))
    cards._tables.cache_clear()
    cards._nlhe.cache_clear()
    rng = random.Random("pop-preflop-plo4-v1")
    deck = [r + s for r in "23456789TJQKA" for s in SUITS]
    scores = sorted(cards.plo_score(rng.sample(deck, 4)) for _ in range(PLO_HANDS))
    q = [round(scores[int((i + 0.5) * PLO_HANDS / PLO_QUANTILES)], 5) for i in range(PLO_QUANTILES)]
    out["uncertainty_plo4"] = {"hands": PLO_HANDS, "score_quantiles": q}
    cards.DATA.write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {cards.DATA}")


if __name__ == "__main__":
    main()
