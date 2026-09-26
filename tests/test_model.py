"""The model's own self-checks, run as a test. They are the calibration
anchors (measured pools, MFU band, decode measurement) and take ~1 min."""
import sys
from pathlib import Path

import numpy as np
import pytest

from workingset import model as M


@pytest.mark.slow
def test_selfcheck():
    M._selfcheck()


def test_scripts_alias_is_the_same_module():
    """scripts/scenario_model.py must alias workingset.model, not copy it."""
    scripts = Path(__file__).resolve().parent.parent / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        import scenario_model as S  # noqa: F401
    finally:
        sys.path.remove(str(scripts))
    assert S is M
    assert S.MODELS is M.MODELS



def _ref_latency_args(**wl_kw):
    m = M.MODELS["27B"]
    topo = M.topology("tp", 2, "H200")
    return m, topo, M.Workload(**wl_kw)


KW = dict(turn_tokens=2000, per_pass_overhead=True)


def test_ttft_quantile_is_the_weighted_mixture_quantile():
    """c_p is the P-th quantile of the hit/miss service mixture, weights
    1 - m and m: at m = 0 it is the hits' own quantile, at m = 1 the misses'
    own, and in between the weighted CDF at c_p reaches P."""
    m, topo, _ = _ref_latency_args()
    for inval in (0.0, 0.01, 0.05, 0.3, 1.0):
        wl = M.Workload(invalidation=inval)
        cold, warm = M._prefill_service_arrays(m, topo, wl, 4096, 2000,
                                               per_pass_overhead=True)
        c = M.ttft_service_quantile(m, topo, wl, 4096, 95, **KW)
        cdf = inval * (cold <= c).mean() + (1 - inval) * (warm <= c).mean()
        below = inval * (cold < c).mean() + (1 - inval) * (warm < c).mean()
        assert below < 0.95 <= cdf + 1e-12, (inval, below, cdf)
    wl = M.Workload(invalidation=1.0)
    assert M.ttft_service_quantile(m, topo, wl, 4096, 95, **KW) == pytest.approx(
        M.miss_service_quantile(m, topo, wl, 4096, 95, **KW), rel=1e-4)


def _miss_dominates(m, topo, wl, turn):
    """First-order dominance of the miss service sample over the hit's:
    equal-size samples, so it is the elementwise order of the sorted pair."""
    cold, warm = M._prefill_service_arrays(m, topo, wl, 4096, turn,
                                           per_pass_overhead=True)
    return bool((np.sort(cold) >= np.sort(warm)).all())


def test_latency_ceiling_is_monotone_in_the_miss_share_when_misses_dominate():
    """Where the miss service distribution dominates the hit's, more misses
    never raise the ceiling, including across m = 1 - P where a hit/miss
    split used to jump. Without dominance there is no such claim (see the
    short-prompt test below)."""
    m, _, _ = _ref_latency_args()
    checked = 0
    for tp, turn, sla in ((1, 16_000, 1.0), (2, 16_000, 1.0),
                          (1, 16_000, 10.0), (2, 2_000, 10.0), (1, 500, 10.0)):
        topo = M.topology("tp", tp, "H200")
        if not _miss_dominates(m, topo, M.Workload(), turn):
            continue
        checked += 1
        ceil = [M.max_users_latency(m, topo, M.Workload(invalidation=x), 4096,
                                    sla, turn, percentile=95,
                                    per_pass_overhead=True)
                for x in (0.0, 0.03, 0.045, 0.05, 0.055, 0.07, 0.1, 0.3, 1.0)]
        assert all(a >= b for a, b in zip(ceil, ceil[1:])), (tp, turn, sla, ceil)
    assert checked >= 1


def test_latency_ceiling_rises_with_misses_when_a_miss_costs_less():
    """Short prompts and a long warm turn: a hit re-prefills 8,000 tokens
    over its cache while a miss re-prefills a ~2,000-token context, so the
    hits dominate and more misses LOWER the p95 own-prefill term. The
    ceiling then rises with the miss share. Expected, and pinned."""
    m = M.MODELS["27B"]
    topo = M.topology("tp", 1, "H200")
    wl = lambda x: M.Workload(user_median=1000, sys_user=1000, sub_ratio=0.0,
                              invalidation=x)
    assert not _miss_dominates(m, topo, wl(0.1), 8000)
    ceil = [M.max_users_latency(m, topo, wl(x), 4096, 10.0, 8000,
                                percentile=95, per_pass_overhead=True)
            for x in (0.0, 0.05, 0.10)]
    assert ceil[0] == pytest.approx(63.7, abs=0.1)
    assert ceil[2] == pytest.approx(69.2, abs=0.1)
    assert ceil[0] < ceil[1] < ceil[2]


def test_16k_turn_one_second_budget_is_unmeetable_past_the_split():
    """27B on one H200 with a 16,000-token turn: the mixture's p95 service is
    ~1.9 s, over a 1 s budget at zero load, at 5% and at 5.5% misses alike.
    A split that ranked every miss above every hit read 0.88 s at 5.5%."""
    m = M.MODELS["27B"]
    topo = M.topology("tp", 1, "H200")
    for inval in (0.05, 0.055):
        wl = M.Workload(invalidation=inval)
        c = M.ttft_service_quantile(m, topo, wl, 4096, 95, 16_000,
                                    per_pass_overhead=True)
        assert 1.8 < c < 2.1, (inval, c)
        assert M.max_users_latency(m, topo, wl, 4096, 1.0, 16_000,
                                   percentile=95, per_pass_overhead=True) == 0.0


def test_latency_ceiling_percentile_is_monotone():
    """A higher percentile is a stricter budget at any fixed miss share."""
    m, topo, _ = _ref_latency_args()
    for inval in (0.01, 0.1, 1.0):
        wl = M.Workload(invalidation=inval)
        ceil = [M.max_users_latency(m, topo, wl, 4096, 10.0, percentile=p, **KW)
                for p in (50, 90, 95, 99)]
        assert all(a >= b for a, b in zip(ceil, ceil[1:])), (inval, ceil)
    wl = M.Workload(invalidation=0.10)
    rate = M.request_rate(64, M.THINK_TIME_S)
    f = [M.sla_miss_rate(m, topo, wl, rate, 4096, 10.0, 2000,
                         per_pass_overhead=True, percentile=p) for p in (95, 99)]
    assert f[1] <= f[0]


def test_latency_ceiling_mean_is_the_old_behaviour():
    """percentile=None reproduces the mean-TTFT ceiling exactly: the closed
    form with c = E[S | miss], as before the percentile existed."""
    m, topo, wl = _ref_latency_args()
    sla, think = 10.0, M.THINK_TIME_S
    a_s, a_s2, c, _ = M.prefill_service_moments(m, topo, wl, 4096, 2000,
                                                per_pass_overhead=True)
    k = 2 * (sla - c)
    want = k / (a_s2 + k * a_s) * think / (1 + wl.sub_ratio)
    got = M.max_users_latency(m, topo, wl, 4096, sla, 2000, think,
                              per_pass_overhead=True)
    assert got == pytest.approx(want, rel=1e-12)
    op = M.operating_point(m, topo, wl, 64, chunk=4096, per_pass_overhead=True)
    assert op["ceilings"]["latency"] == pytest.approx(want, rel=1e-12)


def test_capped_workload_quantile_sits_at_the_cap():
    """A tight max_model_len truncates the tail: every miss past the cap
    costs the same, so a high miss quantile is the cost at the cap and the
    p99 ceiling equals the p95 one once both land on it."""
    m, topo, _ = _ref_latency_args()
    wl = M.Workload(invalidation=1.0, cap=16_000)
    at_cap = M.miss_context_seconds(m, topo, 16_000, 4096)
    q95 = M.ttft_service_quantile(m, topo, wl, 4096, 95, **KW)
    q99 = M.ttft_service_quantile(m, topo, wl, 4096, 99, **KW)
    assert q95 == pytest.approx(q99, rel=1e-9)
    assert q95 == pytest.approx(at_cap, rel=0.05)
    assert (M.max_users_latency(m, topo, wl, 4096, 10.0, percentile=95, **KW)
            == pytest.approx(M.max_users_latency(m, topo, wl, 4096, 10.0,
                                                 percentile=99, **KW)))


def test_percentile_rejects_out_of_range():
    m, topo, wl = _ref_latency_args()
    for bad in (0, 100, -5):
        with pytest.raises(ValueError):
            M.miss_service_quantile(m, topo, wl, 4096, bad)
        with pytest.raises(ValueError):
            M.ttft_service_quantile(m, topo, wl, 4096, bad)
