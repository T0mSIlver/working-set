"""The model's own self-checks, run as a test. They are the calibration
anchors (measured pools, MFU band, decode measurement) and take ~1 min."""
import sys
from pathlib import Path

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


def test_ttft_quantile_picks_the_all_request_path():
    """p-th TTFT over ALL requests: with m misses, the p-th request is a hit
    while m <= 1 - p, and a miss at the conditional quantile
    q = 1 - (1 - p)/m past it."""
    m, topo, _ = _ref_latency_args()
    for inval, pct, path in ((0.01, 95, "hit"), (0.05, 95, "hit"),
                             (0.10, 95, "miss"), (0.01, 99.5, "miss"),
                             (1.0, 50, "miss"), (0.0, 99, "hit")):
        wl = M.Workload(invalidation=inval)
        c, got = M.ttft_service_quantile(m, topo, wl, 4096, pct, **KW)
        assert got == path, (inval, pct)
        assert c >= 0
    # the miss path's quantile is the miss-conditional one
    wl = M.Workload(invalidation=0.10)
    c, _ = M.ttft_service_quantile(m, topo, wl, 4096, 95, **KW)
    q = 100 * (1 - 0.05 / 0.10)
    assert c == pytest.approx(M.miss_service_quantile(m, topo, wl, 4096, q, **KW))


def test_latency_ceiling_percentile_is_monotone_within_a_path():
    """Inside one path a higher percentile is a stricter budget: the ceiling
    never rises with it."""
    m, topo, _ = _ref_latency_args()
    for inval, pcts in ((1.0, (50, 90, 95, 99)), (0.01, (50, 90, 95, 98))):
        wl = M.Workload(invalidation=inval)
        ceil = [M.max_users_latency(m, topo, wl, 4096, 10.0, percentile=p, **KW)
                for p in pcts]
        assert all(a >= b for a, b in zip(ceil, ceil[1:])), (inval, ceil)
    # all misses: p95 is strictly tighter than p50
    wl = M.Workload(invalidation=1.0)
    assert (M.max_users_latency(m, topo, wl, 4096, 30.0, percentile=95, **KW)
            < M.max_users_latency(m, topo, wl, 4096, 30.0, percentile=50, **KW))
    # and the miss rate the budget allows falls with the percentile too
    wl = M.Workload(invalidation=0.10)
    rate = M.request_rate(64, M.THINK_TIME_S)
    f = [M.sla_miss_rate(m, topo, wl, rate, 4096, 10.0, 2000,
                         per_pass_overhead=True, percentile=p) for p in (99, 99.9)]
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
    q95, _ = M.ttft_service_quantile(m, topo, wl, 4096, 95, **KW)
    q99, _ = M.ttft_service_quantile(m, topo, wl, 4096, 99, **KW)
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
