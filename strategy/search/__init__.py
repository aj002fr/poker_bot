"""Parameter search for the interpretable baseline policy (Stage 1, workstream B).

* :mod:`strategy.search.population` - the interim perturbed-family opponent source
  (until the real ``strategy.population`` registry lands) and :func:`load_population`;
* :mod:`strategy.search.evaluate` - paired common-random-number table evaluation;
* :mod:`strategy.search.cem` - CEM / CMA-ES with a kept, confirmed incumbent and a
  persistent run store.

``python -m strategy.search --help`` lists the commands; long runs sit behind make
targets (``make search-baseline``), never CI.
"""
