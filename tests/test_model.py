"""The model's own self-checks, run as a test. They are the calibration
anchors (measured pools, MFU band, decode measurement) and take ~1 min."""
import math
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


def test_miss_share_past_the_warm_pool():
    """f(U) = max(f0, 1 - W/U): the configured rate up to W / (1 - f0), then
    the users the cache cannot keep warm (issue #117)."""
    assert M.miss_share_at(100, 400, 0.01) == 0.01
    assert M.miss_share_at(404, 400, 0.01) == 0.01          # inside the knee
    assert M.miss_share_at(800, 400, 0.01) == pytest.approx(0.5)
    assert M.miss_share_at(800, None, 0.01) == 0.01
    assert M.miss_share_at(0, 400, 0.01) == 0.01
    assert M.cache_hit_rate(800, 400, 0.0) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        M.miss_share_at(-1, 400, 0.01)


def test_capacity_limits_are_fixed_points():
    """Each latency limit with W set is where its metric, priced at the miss
    share at that load, meets its budget; inside the knee it is untouched."""
    m, topo, wl = _ref_latency_args()
    w = 150.0
    for fn, args in ((M.max_users_cold_wait, (5.0, 2000)),
                     (M.max_users_saturation, (2000,)),
                     (M.max_users_slowed, (0.15, 2000)),
                     (M.max_users_latency, (10.0, 2000))):
        free = fn(m, topo, wl, 4096, *args, per_pass_overhead=True)
        u = fn(m, topo, wl, 4096, *args, per_pass_overhead=True, warm_users=w)
        assert w / (1 - wl.invalidation) <= u < free
        again = fn(m, topo, M.workload_at(wl, u, w), 4096, *args,
                   per_pass_overhead=True)
        assert again == pytest.approx(u, rel=1e-9)
        # a pool that holds everyone the limit allows changes nothing
        assert fn(m, topo, wl, 4096, *args, per_pass_overhead=True,
                  warm_users=2 * free) == free


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
    # the operating point prices the capacity misses past the warm pool
    # (issue #117); with an unbounded cache it is the closed form above
    assert op["ttft_ceiling"] == M.max_users_latency(
        m, topo, wl, 4096, sla, 2000, think, per_pass_overhead=True,
        warm_users=op["warm_users"])
    assert op["ttft_ceiling"] <= want
    assert M.max_users_latency(m, topo, wl, 4096, sla, 2000, think,
                               per_pass_overhead=True,
                               warm_users=float("inf")) == got


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


# ---- slowed generation and cold-request wait (issue #94) ------------------

def test_slowed_share_rises_with_load_and_caps_at_one():
    m, topo, wl = _ref_latency_args(invalidation=0.09)
    shares = [M.slowed_share(m, topo, wl, r, 4096, per_pass_overhead=True)
              for r in (0.0, 0.5, 1.0, 2.0, 50.0)]
    assert shares[0] == 0.0
    assert all(a < b for a, b in zip(shares[:-2], shares[1:-1]))
    assert shares[-1] == 1.0


def test_wait_quantile_is_exact_for_exponential_service():
    """M/M/1: P(W > t) = rho exp(-(mu - lam) t), the form the approximation
    assumes, so the two must agree; below p = 1 - rho the wait is zero."""
    lam, mu = 0.7, 1.0
    rho, mean = lam / mu, lam / (mu * (mu - lam))
    for p in (0.5, 0.9, 0.95, 0.99):
        exact = math.log(rho / (1 - p)) / (mu - lam)
        assert M.wait_quantile(mean, rho, p) == pytest.approx(exact, rel=1e-12)
    assert M.wait_quantile(mean, rho, 0.3) == 0.0
    assert M.wait_quantile(mean, 1.0, 0.95) == math.inf


@pytest.mark.parametrize("percentile", [None, 95.0])
def test_cold_wait_ceiling_solves_its_budget(percentile):
    m, topo, wl = _ref_latency_args(invalidation=0.09)
    users = M.max_users_cold_wait(m, topo, wl, 4096, 5.0, percentile=percentile,
                                  **KW)
    rate = M.request_rate(users, M.THINK_TIME_S, wl.sub_ratio)
    got = M.cold_wait_seconds(m, topo, wl, rate, 4096, percentile=percentile,
                              **KW)
    assert got == pytest.approx(5.0, rel=1e-9)


def test_slowed_ceiling_solves_its_limit():
    m, topo, wl = _ref_latency_args(invalidation=0.09)
    users = M.max_users_slowed(m, topo, wl, 4096, 0.05, **KW)
    rate = M.request_rate(users, M.THINK_TIME_S, wl.sub_ratio)
    got = M.slowed_share(m, topo, wl, rate, 4096, per_pass_overhead=True)
    assert got == pytest.approx(0.05, rel=1e-9)
    # no misses: nothing is ever slowed
    none = M.max_users_slowed(m, topo, M.Workload(invalidation=0.0), 4096, 0.05,
                              **KW)
    assert none == math.inf


def test_token_debt_counts_tokens_not_steps():
    """A drain costs each warm user the tokens it would have decoded: its
    per-user speed normally minus its speed while every step carries a chunk,
    over the drain. Each step hands out `tokens_per_step` tokens, so counting steps
    undercounts by the accepted length (the 27B's 2.94)."""
    m, t, wl = M.MODELS["27B"], M.topology_grid(1, 2, "H200"), M.Workload()
    assert m.tokens_per_step > 1                    # the case the step count got wrong
    drain, _, per_user, total = M.spike_token_debt(
        m, t, wl, 32, 64, 2.0, 32_768, 2_000, n_iter=200)
    _, p50, _, _ = M.decode_curves(m, t, wl, [64], n_iter=200, seed=0)
    d_ms, x_ms, _ = M.itl_spike(m, t, wl, 64, 32_768, n_iter=200)
    v_normal = p50[0]                   # tok/s per user, speculation included
    v_mixed = v_normal * d_ms / x_ms    # same tokens per step, longer steps
    assert per_user == pytest.approx(drain * (v_normal - v_mixed), rel=1e-9)
    assert total == pytest.approx(64 * per_user)


def test_expert_union_counts_speculative_verify_positions():
    """Issue #91: with draft depth k each sequence verifies 1 + k positions per
    step and every one routes through the experts, so the linear (no-overlap)
    union saturates at E/(k_e (1 + k)) sequences, not E/k_e."""
    m = M.MODELS["DSV41F"]                     # 384 experts / 6 routed, 5 drafts
    k = m.spec_tokens
    assert k == 5
    below, above = 10, 11                      # 10 x 6 = 60 < 64 <= 11 x 6
    assert m.w_decode(below * (1 + k), "linear") == pytest.approx(
        m.w_decode_shared + below * (1 + k) * m.w_route_pertok)
    assert m.w_decode(above * (1 + k), "linear") == pytest.approx(
        m.w_decode_shared + m.w_route_total)
    dense = M.MODELS["27B"]                    # no experts: drafts change nothing
    assert dense.w_decode(1) == dense.w_decode(600) == dense.w_decode_shared
