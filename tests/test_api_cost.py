"""API and plan seat prices (workingset.api_cost)."""
import math
import subprocess
import sys

import numpy as np
import pytest

from workingset import api_cost as A
from workingset import model as M


def test_clipped_partial_matches_monte_carlo():
    rng = np.random.default_rng(0)
    x = np.clip(rng.lognormal(math.log(31_000), 0.81, 2_000_000), 15_000, 180_000)
    p, e = A.clipped_partial(31_000, 0.81, 15_000, 180_000, -math.inf, math.inf)
    assert p == pytest.approx(1.0, abs=1e-6)
    assert e == pytest.approx(x.mean(), rel=2e-3)
    p, e = A.clipped_partial(31_000, 0.81, 15_000, 180_000, 100_000, math.inf)
    assert p == pytest.approx((x > 100_000).mean(), abs=1e-3)
    assert e == pytest.approx((x * (x > 100_000)).mean(), rel=5e-3)
    # the bands partition the distribution, atoms included
    lo = A.clipped_partial(31_000, 0.81, 15_000, 180_000, -math.inf, 100_000)
    hi = A.clipped_partial(31_000, 0.81, 15_000, 180_000, 100_000, math.inf)
    whole = A.clipped_partial(31_000, 0.81, 15_000, 180_000, -math.inf, math.inf)
    assert lo[0] + hi[0] == pytest.approx(whole[0], abs=1e-12)
    assert lo[1] + hi[1] == pytest.approx(whole[1], rel=1e-12)


def test_phi_is_a_cdf():
    assert A.phi(0) == pytest.approx(0.5, abs=1e-7)
    assert A.phi(1.959964) == pytest.approx(0.975, abs=2e-7)
    assert A.phi(-1.959964) == pytest.approx(0.025, abs=2e-7)


def test_expiry_miss_from_the_gap_mixture():
    # tool gaps rarely outlast 5 min, a sixth of human gaps do
    assert A.expiry_miss(300) == pytest.approx(0.0272, abs=2e-4)
    assert A.expiry_miss(3600) < A.expiry_miss(1800) < A.expiry_miss(300)
    assert A.expiry_miss(0) == 1.0


def _offer(read, write, inp=4.0, out=20.0, bands=None):
    b = bands or (A.PriceBand(0, inp, out, (A.CacheOption(300, read, write),)),)
    return A.ApiOffer("t", "t", "t", "t", b)


def test_without_a_cache_discount_every_prompt_token_bills_at_input():
    wl = M.Workload()
    rc = A.request_cost(_offer(4.0, 4.0), wl)
    lm = A.clipped_partial(wl.user_median, wl.user_sigma, wl.sys_user, wl.cap, -math.inf, math.inf)[1]
    ls = A.clipped_partial(wl.sub_median, wl.sub_sigma, wl.sys_sub, wl.cap, -math.inf, math.inf)[1]
    lbar = (lm + wl.sub_ratio * ls) / (1 + wl.sub_ratio)
    assert rc.usd == pytest.approx((4.0 * lbar + 20.0 * 400) / 1e6, rel=1e-12)
    assert rc.hit_rate == 0.0


def test_identical_bands_price_like_one():
    one = _offer(0.2, 5.0)
    same = A.PriceBand(100_000, 4.0, 20.0, (A.CacheOption(300, 0.2, 5.0),))
    two = _offer(0.2, 5.0, bands=one.bands + (same,))
    wl = M.Workload()
    assert A.request_cost(two, wl).usd == pytest.approx(A.request_cost(one, wl).usd, rel=1e-12)


def test_a_longer_ttl_is_taken_only_when_it_is_cheaper():
    wl = M.Workload()
    # a free 1 h write beats a 5 min one at the same price
    o = A.ApiOffer("t", "t", "t", "t", (A.PriceBand(0, 4, 20, (A.CacheOption(300, 0.2, 5.0),
                                                              A.CacheOption(3600, 0.2, 5.0))),))
    assert A.request_cost(o, wl).ttl_s == 3600
    # Anthropic's 2x write for 1 h loses at a 2,000-token fresh tail
    o = A.ApiOffer("t", "t", "t", "t", (A.PriceBand(0, 4, 20, (A.CacheOption(300, 0.2, 5.0),
                                                              A.CacheOption(3600, 0.2, 8.0))),))
    assert A.request_cost(o, wl).ttl_s == 300


def test_api_seat_is_linear_in_hours_and_per_session_divides_by_the_peak_share():
    o = A.catalogue()["apis"]["claude-sonnet-5.5"]
    wl = M.Workload()
    a = A.api_seat_cost(o, wl, active_hours=40)
    b = A.api_seat_cost(o, wl, active_hours=80)
    assert b.eur_seat == pytest.approx(2 * a.eur_seat, rel=1e-12)
    c = A.api_seat_cost(o, wl, active_hours=80, peak_active_share=0.25)
    assert c.eur_seat == pytest.approx(b.eur_seat, rel=1e-12)
    assert c.eur_session == pytest.approx(4 * b.eur_seat, rel=1e-12)


def test_catalogue_rows_carry_their_sources():
    cat = A.catalogue()
    assert cat["fx"]["usd_per_eur"] > 1
    for a in cat["apis"].values():
        assert a.meta["source"]["url"].startswith("https://") and a.meta["source"]["quote"]
        assert a.tb40 is not None and 0 <= a.tb40 <= 1
        assert round(a.tb40 * 198) == a.meta["quality"]["runs_passed"]
    for p in cat["plans"].values():
        assert p.api in cat["apis"]
        assert p.meta["price_source"]["url"].startswith("https://")
        assert p.meta["budget_source"]["quote"]


def test_a_plan_that_stops_needs_more_plans_and_one_that_bills_spills_at_api():
    cat = A.catalogue()
    wl = M.Workload()
    lite = A.plan_seat_cost(cat["plans"]["glm-coding-lite"], wl)
    assert not lite.fits and lite.binds == "5h" and lite.plans_per_seat >= 2
    assert lite.eur_seat == pytest.approx(18 * lite.plans_per_seat / A.usd_per_eur(), rel=1e-12)
    # Copilot Business's $19 a month is gone in a few days of agent use
    cb = A.plan_seat_cost(cat["plans"]["copilot-business"], wl)
    api = A.api_seat_cost(cat["apis"][cat["plans"]["copilot-business"].api], wl)
    assert not cb.fits and cb.binds == "month"
    # base price plus the API bill past the $19 pool: the API seat itself
    assert cb.eur_seat == pytest.approx(api.eur_seat, rel=1e-9)
    # at a few hours a month every plan fits
    for p in cat["plans"].values():
        assert A.plan_seat_cost(p, wl, active_hours=2).fits, p.key


def test_ws_api_prints_every_row():
    out = subprocess.run([sys.executable, "-m", "workingset.cli", "api", "--hours", "40"],
                         capture_output=True, text=True, check=True).stdout
    cat = A.catalogue()
    for a in cat["apis"].values():
        assert a.name in out
    assert "40 active hours" in out
