"""Seeded determinism, hidden labels, sampling, regimes, and the behaviours census stats cannot show."""

from __future__ import annotations

import math
import random
from types import SimpleNamespace

import pytest

from arena.agents.sdk import TableInfo
from arena.domain.models import TournamentConfig
from strategy.population import cards
from strategy.population.behaviours import CALIBRATION, icm_equity, icm_threshold
from strategy.population.families import FAMILIES, FAMILY_LIST
from strategy.population.measure import play
from strategy.population.params import TRUNCATE
from strategy.population.policy import PopulationBot, scan
from strategy.population.registry import Member, Registry


def _firsts(registry: Registry, partition: str = "holdout") -> list[Member]:
    out: dict[str, Member] = {}
    for m in sorted(registry.partition(partition), key=lambda m: m.index):
        out.setdefault(m.family, m)
    return [out[f.id] for f in FAMILY_LIST if f.id in out]


def _actions(table) -> list:
    return [(e.data["seat"], e.data["action"]) for e in table.log if e.type == "ActionApplied"]


@pytest.mark.parametrize("ruleset", ["practice_nlhe", "uncertainty_shortdeck", "uncertainty_plo4"])
def test_deterministic_under_seed(registry: Registry, ruleset: str):
    lineup = _firsts(registry)[:8]
    cfg = {"hand_cap": 40, "hands_per_level": 5}

    def run(seed: str):
        table, _ = play([(m.member_id, m.make_bot()) for m in lineup], seed, ruleset, cfg)
        return _actions(table)

    first = run("det-1")
    assert first == run("det-1")
    assert first != run("det-2")


def test_labels_never_reach_the_table(registry: Registry):
    lineup = _firsts(registry)[8:16]
    table, seats = play([(m.member_id, m.make_bot()) for m in lineup], "labels", config={"hand_cap": 30,
                                                                                       "hands_per_level": 4},
                        keep=True)
    for m, (_, agent) in zip(lineup, seats.values()):
        assert agent.bot.name == m.member_id and m.member_id.startswith("pm-")
        for _obs, action in agent.records:
            assert set(action) <= {"type", "amount"}
    log = "".join(e.model_dump_json() for e in table.log)
    for fam in FAMILIES:
        assert fam not in log


def test_sampling_is_seeded_bounded_and_correlated():
    for fam in FAMILY_LIST:
        prior = fam.prior
        assert prior.sample("s") == prior.sample("s") != prior.sample("t")
        draws = [prior.sample(f"corr|{i}") for i in range(1500)]
        for p in prior.params:
            assert all(p.lo <= d[p.name] <= p.hi for d in draws)
        for a, b, rho in prior.correlations:
            ia, ib = prior.names.index(a), prior.names.index(b)
            v = [_shocks(prior, d) for d in draws]
            assert abs(_corr([x[ia] for x in v], [x[ib] for x in v]) - rho) < 0.1, (fam.id, a, b)
        assert all(max(abs(z) for z in prior.latent(d)) <= TRUNCATE + 1e-3 for d in draws[:200])


def _shocks(prior, x) -> list[float]:
    z = prior.latent(x)
    return [sum(prior.chol[i][k] * z[k] for k in range(i + 1)) for i in range(len(z))]


def _corr(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))


def _info(seed: str) -> TableInfo:
    cfg = TournamentConfig(initial_player_count=8)
    return TableInfo(table_id="t_pop", hero_seat=0, ruleset_id="practice_nlhe", config=cfg.resolved(),
                     config_hash=cfg.config_hash(), seats=cfg.resolved_seats, decision_budget_ns=cfg.bot_budget_ns,
                     policy_seed=seed)


def test_table_local_regime_switch(registry: Registry):
    for fam in FAMILY_LIST:
        member = next(m for m in registry.partition("holdout") if m.family == fam.id)
        bot = member.make_bot()
        switches = [bot.initial_state(_info(f"regime|{i}"))["sw"] for i in range(400)]
        on = [s for s in switches if s]
        assert 0.18 <= len(on) / 400 <= 0.32, fam.id
        assert all(30 <= s < 170 for s in on)
        base, switched = bot.active(False, False), bot.active(True, False)
        assert base[0] != switched[0], fam.id


def test_icm_equity():
    pays = [50.0, 30.0, 20.0]
    stacks = [4000.0, 2000.0, 1000.0, 1000.0]
    eq = [icm_equity(stacks, pays, i) for i in range(4)]
    assert sum(eq) == pytest.approx(100.0)
    assert eq[0] / 100 < stacks[0] / sum(stacks)
    assert eq[3] / 100 > stacks[3] / sum(stacks)
    assert icm_equity(stacks, [100.0], 1) == pytest.approx(25.0)


def test_icm_threshold_exceeds_chip_ev_near_the_money():
    players = [SimpleNamespace(seat=s, stack=st, street_committed=sc, status="active")
               for s, st, sc in ((0, 3000, 0), (1, 0, 3000), (2, 900, 0), (3, 1100, 0))]
    c = SimpleNamespace(obs=SimpleNamespace(players=players, pot_total=3030, payouts=[50, 30, 20, 0]),
                        s=SimpleNamespace(hero=0))
    chip, icm = icm_threshold(c, 3000)
    assert chip == pytest.approx(3000 / 6030)
    assert icm > chip + 0.03
    correct, over, ignored, inverted = CALIBRATION
    assert over > correct > ignored > 0 > inverted


def test_image_responsive_reraises_loose_openers_more(registry: Registry):
    loose = [m for m in registry.partition("train") if m.family == "chaotic_aggressor"][:3]
    tight = [m for m in registry.partition("train") if m.family == "selective_aggressor"][:2]
    image = [m for m in registry.partition("holdout") if m.family == "image_responsive"][:3]
    control = [m for m in registry.partition("train") if m.family == "wide_pressure"][:3]

    def rates(probe: list[Member]) -> dict[str, float]:
        hits = {"loose": [0, 0], "tight": [0, 0]}
        for t in range(5):
            lineup = [probe[0], loose[0], tight[0], probe[1], loose[1], tight[1], probe[2], loose[2]]
            _table, seats = play([(m.member_id, m.make_bot()) for m in lineup], f"image-{t}", keep=True)
            kinds = {m.member_id: "loose" for m in loose} | {m.member_id: "tight" for m in tight}
            group = {seat: kinds.get(key) for seat, (key, _) in seats.items()}
            for seat, (key, agent) in seats.items():
                if key not in {m.member_id for m in probe}:
                    continue
                for obs, action in agent.records:
                    x = scan(obs, seat)
                    if obs.street != "preflop" or x.opener is None or _raises(obs) != 1:
                        continue
                    g = group[x.opener]
                    if g is not None:
                        hits[g][0] += action["type"] == "raise_to"
                        hits[g][1] += 1
        assert min(n for _, n in hits.values()) >= 100, hits
        return {g: h / n for g, (h, n) in hits.items()}

    img, ctl = rates(image), rates(control)
    assert img["loose"] / max(img["tight"], 1e-3) > 1.5 * ctl["loose"] / max(ctl["tight"], 1e-3), (img, ctl)


def _raises(obs) -> int:
    return sum(1 for e in obs.action_history if e.type == "ActionApplied" and e.hand_id == obs.hand_id
               and e.data["street"] == "preflop" and e.data["action"]["type"] == "raise_to")


def test_variant_card_adapters():
    assert cards.evaluate_shortdeck(["Ah", "Kh", "Qh", "9h", "7h"]) > cards.evaluate_shortdeck(
        ["Ah", "Ad", "Ac", "Kh", "Kd"])
    assert cards.evaluate_shortdeck(["Ah", "6d", "7c", "8h", "9s"])[0] == cards.STRAIGHT
    hearts_board = cards.omaha_value(["Ah", "Kh", "Qh", "Jh"], ["2h", "3h", "4c", "5d", "9s"])
    assert hearts_board[0] < cards.FLUSH
    assert cards.omaha_value(["Ah", "Kh", "2c", "3d"], ["Qh", "Jh", "Th", "4c", "5d"])[0] == cards.STRAIGHT_FLUSH
    for ruleset, best, worst in (("practice_nlhe", ["As", "Ah"], ["7d", "2c"]),
                                 ("uncertainty_shortdeck", ["As", "Ah"], ["7d", "6c"]),
                                 ("uncertainty_plo4", ["As", "Ah", "Ks", "Kh"], ["9d", "6c", "3h", "2s"])):
        assert cards.preflop_top(best, ruleset) < 0.02 < 0.8 < cards.preflop_top(worst, ruleset), ruleset


def test_population_bot_rebuilds_from_spec(registry: Registry):
    rng = random.Random(3)
    for m in rng.sample(list(registry.members.values()), 10):
        bot = PopulationBot.from_spec(m.spec)
        assert bot.params == dict(m.params) and bot.family.id == m.family
