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


def _ref_latency_args():
    m = M.MODELS["27B"]
    topo = M.topology("tp", 2, "H200")
    return m, topo, M.Workload()


def test_latency_ceiling_percentile_is_monotone():
    """A higher TTFT percentile is a stricter budget: the ceiling never
    rises with it, and p95 sits at or below p50."""
    m, topo, wl = _ref_latency_args()
    kw = dict(turn_tokens=2000, per_pass_overhead=True)
    ceil = {p: M.max_users_latency(m, topo, wl, 4096, 10.0, percentile=p, **kw)
            for p in (50, 90, 95, 99)}
    assert ceil[99] <= ceil[95] <= ceil[90] <= ceil[50]
    assert ceil[99] < ceil[50]
    # the miss's own prefill at a percentile is ordered the same way
    q = [M.miss_service_quantile(m, topo, wl, 4096, p, 2000,
                                 per_pass_overhead=True) for p in (50, 95, 99)]
    assert q[0] < q[1] < q[2]
    # and the miss rate the budget allows falls with it too
    rate = M.request_rate(64, M.THINK_TIME_S)
    f = {p: M.sla_miss_rate(m, topo, wl, rate, 4096, 10.0, 2000,
                            per_pass_overhead=True, percentile=p)
         for p in (None, 95)}
    assert f[95] <= f[None]


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


def test_percentile_rejects_out_of_range():
    m, topo, wl = _ref_latency_args()
    for bad in (0, 100, -5):
        with pytest.raises(ValueError):
            M.miss_service_quantile(m, topo, wl, 4096, bad)
