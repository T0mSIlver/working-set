"""Shared-endpoint mode: the safety rails, the covariate fit and its gates.

No test here needs a real endpoint. The streaming server is the same
`httpx.MockTransport` fake `test_probe.py` drives; the metrics source is a
scripted `running`/`waiting`/`kv` series, so a rail can be made to fire on a
chosen tick rather than by luck.
"""
from __future__ import annotations

import asyncio
import json
import math
import random
from dataclasses import replace

import httpx
import pytest

from workingset.config import RunConfig
from workingset.hypotheses import (NOT_ESTABLISHED, REGISTRY, RunContext,
                                   SUPPORTED, plan)
from workingset.predict import predict
from workingset.probe import RequestTrace, build_prefixes
from workingset.probe.request import EndpointSpec
from workingset.record import RunRecord, not_established_notes
from workingset.report import print_report
from workingset.shared import (LOAD_COLUMNS, TTFT_COLUMNS, BudgetAbort,
                               CovariateFit, ProbeBudget, ProbeGovernor,
                               SharedOptions, SharedResult, build_fits,
                               covariate_rows, cross_check, fit_covariates,
                               ladder_model_curve, natural_ladder,
                               operating_point_covariates, run_shared)

from test_probe import client_for, fake_server, small_cfg, small_opts


# ============================================================================
# fixtures
# ============================================================================
class ScriptedMetrics:
    """A metrics source with a written-down series.

    `at(t)` walks the script one step per call, so a test says exactly which
    reading the Nth gauge check sees; the last entry repeats forever. `window`
    hands back a `WindowDelta`-shaped double carrying TTFT / per-token
    histograms and the prefix-cache counters, which is all `cross_check`
    touches.
    """

    def __init__(self, script=None, window=None, ticks_before_window=0):
        self.script = list(script or [{"requests_running": 3,
                                       "requests_waiting": 0,
                                       "kv_cache_usage": 0.3}])
        self.i = 0
        self.at_calls = 0
        self.ticks = 0
        self._window = window
        self._need = ticks_before_window

    def at(self, t):
        self.at_calls += 1
        snap = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return dict(snap, t=t)

    async def next_tick(self):
        self.ticks += 1
        return True

    def window(self, t0, t1):
        if self.ticks < self._need:
            raise ValueError("no snapshot started after t1")
        if self._window is None:
            raise ValueError("window not covered")
        return self._window


class FakeHist:
    def __init__(self, qs: dict, n: int):
        self._qs, self.observations = qs, n

    def quantile(self, q):
        return self._qs[q]


class FakeWindow:
    """The slice of `WindowDelta` `cross_check` reads."""

    def __init__(self, ttft_p50=0.40, ttft_p95=0.90, tpot_s=0.020,
                 hit_rate=0.15, n=42):
        self.ttft = FakeHist({0.5: ttft_p50, 0.95: ttft_p95}, n)
        self.request_tpot = FakeHist({0.5: tpot_s}, n)
        self.prefix_hit_rate = hit_rate

    def to_dict(self):
        return {"n_snapshots": 7, "dt_s": 12.0}


def shared_opts(**kw) -> SharedOptions:
    base = dict(lengths="0.25,0.5,1.0", rounds=2, warm_turns=1, seed=3)
    base.update(kw)
    return SharedOptions(**base)


def budget(**kw) -> ProbeBudget:
    """A budget whose canary is off unless a test asks for it — the canary
    fires on a timer, and a test that did not ask for one should not race it."""
    base = dict(canary=False, max_extra_load=2)
    base.update(kw)
    return ProbeBudget.conservative(**base)


def synth_rows(n: int, c, noise: float = 0.0, seed: int = 0,
               running=None, kind: str = "miss") -> list[dict]:
    """Rows drawn from a KNOWN quadratic-in-L, linear-in-load surface."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        L = 5.0 + 35.0 * rng.random()
        r = running if running is not None else 2.0 + 20.0 * rng.random()
        wq = 4.0 * rng.random()
        y = (c[0] + c[1] * L + c[2] * L * L + c[3] * r + c[4] * wq
             + rng.gauss(0.0, noise))
        rows.append({"kind": kind, "L_ktok": L, "L_ktok2": L * L,
                     "running": r, "waiting": wq, "y": y,
                     "ttft": y, "itl_ms": y, "decode_tok_s": y})
    return rows


# ============================================================================
# the budget rails
# ============================================================================
def test_budget_describe_says_the_gauge_rails_need_metrics():
    lines = " ".join(budget(canary=True).describe(metrics=False))
    assert "NOT ENFORCEABLE" in lines and "--metrics-url" in lines
    assert "NOT ENFORCEABLE" not in " ".join(
        budget(canary=True).describe(metrics=True))


def test_exclusive_budget_takes_every_rail_off():
    b = ProbeBudget.for_exclusive()
    assert b.exclusive and not b.canary
    assert b.abort_if_waiting is None and b.abort_if_kv_above is None
    assert b.max_probe_tokens == 0 and b.effective_max_load == 0


def test_a_capped_budget_leaves_the_canary_a_slot():
    assert ProbeBudget.conservative(max_extra_load=1).effective_max_load == 2
    assert ProbeBudget.conservative(max_extra_load=1,
                                    canary=False).effective_max_load == 1


def test_budget_refuses_a_drift_ratio_that_cannot_discriminate():
    with pytest.raises(ValueError, match="canary-drift"):
        ProbeBudget.conservative(canary_drift=1.0)


def test_governor_aborts_on_the_waiting_gauge():
    gov = ProbeGovernor(budget(abort_if_waiting=2))
    gov.observe({"requests_waiting": 2, "kv_cache_usage": 0.1})   # not over
    with pytest.raises(BudgetAbort) as e:
        gov.observe({"requests_waiting": 3, "kv_cache_usage": 0.1})
    assert "queue reached 3" in e.value.reason
    assert e.value.detail["requests_waiting"] == 3
    assert gov.to_dict()["aborted"]["detail"]["limit"] == 2


def test_governor_aborts_on_kv_occupancy():
    gov = ProbeGovernor(budget(abort_if_waiting=None, abort_if_kv_above=0.8))
    gov.observe({"kv_cache_usage": 0.79})
    with pytest.raises(BudgetAbort) as e:
        gov.observe({"kv_cache_usage": 0.81})
    assert "KV occupancy" in e.value.reason
    assert gov.peak_kv == pytest.approx(0.81)


def test_governor_without_metrics_cannot_fire_the_gauge_rails():
    """No covariates is NOT a pass — it is a rail that cannot fire, and the
    governor records zero checks so the report can say so."""
    gov = ProbeGovernor(budget(abort_if_waiting=0))
    gov.observe(None)
    gov.observe({})
    assert gov.aborted is None and gov.n_gauge_checks == 0


def test_token_budget_refuses_the_request_that_would_exceed_it():
    gov = ProbeGovernor(budget(max_probe_tokens=100))
    gov.spend(60)
    with pytest.raises(BudgetAbort) as e:
        gov.spend(50)
    assert "prompt-token budget" in e.value.reason
    # the cap is a cap: the refused request never joined the total
    assert gov.tokens_spent == 60 and gov.n_requests == 1


def test_canary_drift_rule_needs_two_non_overlapping_windows():
    gov = ProbeGovernor(budget(canary=True, canary_baseline_s=10,
                               canary_window_s=10, canary_min_n=3,
                               canary_drift=3.0))
    # baseline: five fast canaries inside the first 10 s
    for i in range(5):
        gov.canary_ttft.append((i * 2.0, 0.10))
    # a slow canary while the windows still overlap cannot fire the rule
    gov.canary_ttft.append((12.0, 5.0))
    assert gov.canary_drift() is None
    # ...and once the trailing window clears the baseline window, it does
    for t in (21.0, 22.0, 23.0):
        gov.canary_ttft.append((t, 5.0))
    fired = gov.canary_drift()
    assert fired is not None
    base, recent = fired
    assert base == pytest.approx(0.10) and recent == pytest.approx(5.0)


def test_canary_drift_rule_stays_quiet_when_the_endpoint_stays_quiet():
    gov = ProbeGovernor(budget(canary=True, canary_baseline_s=10,
                               canary_window_s=10, canary_min_n=3))
    for i in range(6):
        gov.canary_ttft.append((i * 1.5, 0.10))
    for t in (21.0, 22.0, 23.0, 24.0):
        gov.canary_ttft.append((t, 0.12))
    assert gov.canary_drift() is None


def test_note_canary_raises_when_the_rule_fires():
    gov = ProbeGovernor(budget(canary=True, canary_baseline_s=10,
                               canary_window_s=10, canary_min_n=2,
                               canary_drift=2.0))
    for i in range(3):
        gov.canary_ttft.append((i * 2.0, 0.10))
    gov.canary_ttft.append((21.0, 1.0))
    gov.t0 = 0.0
    with pytest.raises(BudgetAbort) as e:
        gov.note_canary(22.0, 1.0)
    assert "canary TTFT drifted" in e.value.reason


# ============================================================================
# the rails, driven end to end against the fake endpoint
# ============================================================================
def _run_shared(handler, cfg, opts, sopts, bud, metrics=None):
    async def go():
        async with client_for(handler) as client:
            ep = EndpointSpec(base_url="http://x/v1", model="m")
            return await run_shared(client, ep, cfg, opts,
                                    build_prefixes(cfg.workload,
                                                   opts.chars_per_token),
                                    bud, sopts, metrics)
    return asyncio.run(go())


def test_shared_probe_stamps_covariates_and_fits():
    cfg, opts = small_cfg(), small_opts()
    metrics = ScriptedMetrics(
        [{"requests_running": 1 + (i % 9), "requests_waiting": (i * 3) % 5,
          "kv_cache_usage": 0.2 + 0.01 * (i % 5)} for i in range(200)],
        window=FakeWindow())
    res = _run_shared(fake_server(), cfg, opts, shared_opts(rounds=6),
                      budget(abort_if_waiting=None), metrics)
    assert res.aborted is None
    assert res.n_covariate_rows > 0
    assert res.sample is not None and res.sample.n_miss > 0
    # the shared probe produces the SAME Sample the plain cheap probe does
    assert math.isfinite(res.sample.ttft_miss_mean)
    # every request carried the load the server reported at send time, and the
    # load VARIED, so the load-only fits are well posed
    assert res.fits["itl"].usable, res.fits["itl"].refused
    assert res.fits["itl"].ranges["running"]["sd"] > 0
    assert res.fits["ttft_miss"].usable, res.fits["ttft_miss"].refused
    # this small test config saturates, so the model refuses an operating
    # point for it — the fit exists, the place to evaluate it does not, and
    # that is what the reading says rather than a number
    d = json.loads(json.dumps(res.to_dict()))
    assert set(d["fits"]) == {"ttft_miss", "ttft_hit", "itl", "decode"}
    assert "no steady state" in d["operating_point"]["refused"]
    assert not d["readings"]["ttft_miss"]["available"]
    assert "no steady state" in d["readings"]["ttft_miss"]["reason"]


def test_the_probe_aborts_when_the_server_queue_appears():
    cfg, opts = small_cfg(), small_opts()
    # the server is quiet, then somebody else's queue shows up
    metrics = ScriptedMetrics(
        [{"requests_running": 2, "requests_waiting": 0, "kv_cache_usage": 0.2}] * 3
        + [{"requests_running": 9, "requests_waiting": 4, "kv_cache_usage": 0.4}] * 50)
    with pytest.raises(BudgetAbort) as e:
        _run_shared(fake_server(), cfg, opts, shared_opts(rounds=4),
                    budget(abort_if_waiting=1, gauge_poll_s=0.01), metrics)
    assert "queue reached 4" in e.value.reason
    # the partial result rides on the exception, so the record still says what
    # HAD been measured when the rail tripped
    assert e.value.result is not None
    assert e.value.result.aborted == e.value.reason
    assert e.value.result.governor["aborted"]["reason"] == e.value.reason


def test_the_probe_aborts_on_kv_occupancy():
    cfg, opts = small_cfg(), small_opts()
    metrics = ScriptedMetrics(
        [{"requests_running": 2, "requests_waiting": 0, "kv_cache_usage": 0.5}]
        + [{"requests_running": 2, "requests_waiting": 0, "kv_cache_usage": 0.97}] * 50)
    with pytest.raises(BudgetAbort) as e:
        _run_shared(fake_server(), cfg, opts, shared_opts(rounds=3),
                    budget(abort_if_waiting=None, abort_if_kv_above=0.9,
                           gauge_poll_s=0.01), metrics)
    assert "KV occupancy" in e.value.reason


def test_the_probe_aborts_on_the_token_budget():
    cfg, opts = small_cfg(), small_opts()
    with pytest.raises(BudgetAbort) as e:
        _run_shared(fake_server(), cfg, opts, shared_opts(rounds=8),
                    budget(max_probe_tokens=1_500, abort_if_waiting=None,
                           abort_if_kv_above=None))
    assert "prompt-token budget exhausted" in e.value.reason
    spent = e.value.result.governor["tokens_spent"]
    assert spent <= 1_500


def test_the_in_flight_cap_admits_no_more_than_its_slots():
    """The rail that matters most on somebody else's endpoint: at no instant
    may more than `max_extra_load` of our requests be open."""
    live = {"now": 0, "peak": 0}
    gov = ProbeGovernor(budget(max_extra_load=3, canary=False))

    async def one():
        async with gov.slot():
            live["now"] += 1
            live["peak"] = max(live["peak"], live["now"])
            await asyncio.sleep(0.01)
            live["now"] -= 1

    asyncio.run(_gather(*(one() for _ in range(20))))
    assert live["peak"] == 3


def test_an_uncapped_budget_holds_no_slots_at_all():
    live = {"now": 0, "peak": 0}
    gov = ProbeGovernor(ProbeBudget.for_exclusive())

    async def one():
        async with gov.slot():
            live["now"] += 1
            live["peak"] = max(live["peak"], live["now"])
            await asyncio.sleep(0.01)
            live["now"] -= 1

    asyncio.run(_gather(*(one() for _ in range(12))))
    assert live["peak"] == 12


async def _gather(*aws):
    await asyncio.gather(*aws)


def test_the_probe_never_opens_more_than_the_cap_end_to_end():
    """The same rail through the real probe: the length ladder is sequential
    by design (a shared endpoint gets one extra request at a time) and the
    canary runs alongside it, so two slots is the whole of what is used."""
    cfg, opts = small_cfg(), small_opts()
    live = {"now": 0, "peak": 0}

    def handler(request):
        live["now"] += 1
        live["peak"] = max(live["peak"], live["now"])

        async def gen():
            # the decrement sits before the terminator, not in a `finally`:
            # the client BREAKS out of the stream on [DONE], and
            # MockTransport never closes the generator it was handed
            await asyncio.sleep(0.004)
            for _ in range(3):
                yield b'data: {"choices": [{"text": "tok "}]}\n\n'
                await asyncio.sleep(0.002)
            live["now"] -= 1
            yield b"data: [DONE]\n\n"

        return httpx.Response(200, content=gen(),
                              headers={"content-type": "text/event-stream"})

    _run_shared(handler, cfg, opts, shared_opts(rounds=3),
                budget(max_extra_load=2, canary=True, canary_every_s=0.001,
                       abort_if_waiting=None, abort_if_kv_above=None))
    assert 0 < live["peak"] <= 2


# ============================================================================
# the OLS fit
# ============================================================================
TRUE = (0.5, 0.02, 0.004, 0.11, 0.30)


def test_fit_recovers_known_coefficients():
    fit = fit_covariates(synth_rows(120, TRUE, noise=0.0, seed=1),
                         TTFT_COLUMNS, target="TTFT", unit="s")
    assert fit.usable, fit.refused
    for name, want in zip(TTFT_COLUMNS, TRUE):
        assert fit.coefficients[name] == pytest.approx(want, abs=1e-6)
    assert fit.n == 120 and fit.dof == 115
    assert fit.residual_std == pytest.approx(0.0, abs=1e-6)
    assert fit.r_squared == pytest.approx(1.0, abs=1e-9)
    assert math.isfinite(fit.condition_number)


def test_fit_reports_a_residual_and_a_prediction_se_under_noise():
    fit = fit_covariates(synth_rows(300, TRUE, noise=0.25, seed=2),
                         TTFT_COLUMNS, target="TTFT", unit="s")
    assert fit.usable
    assert fit.residual_std == pytest.approx(0.25, rel=0.25)
    point = {"const": 1.0, "L_ktok": 20.0, "L_ktok2": 400.0,
             "running": 12.0, "waiting": 2.0}
    want = (TRUE[0] + TRUE[1] * 20 + TRUE[2] * 400 + TRUE[3] * 12
            + TRUE[4] * 2)
    assert fit.predict(point) == pytest.approx(want, rel=0.05)
    # the standard error of the fitted mean is far below the residual spread
    assert 0 < fit.predict_se(point) < fit.residual_std


def test_fit_refuses_a_small_sample_and_says_what_would_fix_it():
    fit = fit_covariates(synth_rows(9, TRUE, seed=3), TTFT_COLUMNS)
    assert not fit.usable
    assert "n=9" in fit.refused and "15 needed" in fit.refused
    assert "--shared-rounds" in fit.refused
    with pytest.raises(ValueError, match="fit refused"):
        fit.predict({"const": 1, "L_ktok": 1, "L_ktok2": 1, "running": 1,
                     "waiting": 1})


def test_fit_refuses_a_rank_deficient_design_and_names_the_flat_column():
    """The canonical shape of a shared run with no metrics: `running` and
    `waiting` never vary, so their coefficients do not exist."""
    rows = synth_rows(60, TRUE, seed=4)
    for r in rows:
        r["running"] = 4.0
        r["waiting"] = 0.0
    fit = fit_covariates(rows, TTFT_COLUMNS)
    assert not fit.usable
    assert "rank-deficient" in fit.refused
    assert "running" in fit.refused and "waiting" in fit.refused


def test_fit_refuses_when_the_regressors_moved_together():
    """`waiting` is an exact multiple of `running`: the split between c3 and
    c4 is arbitrary however many digits numpy prints."""
    rows = synth_rows(80, TRUE, noise=0.01, seed=5)
    for r in rows:
        r["waiting"] = 0.5 * r["running"] + 1e-9
    fit = fit_covariates(rows, TTFT_COLUMNS, max_condition=1e6)
    assert not fit.usable
    assert "rank-deficient" in fit.refused or "condition number" in fit.refused


def test_fit_drops_rows_with_no_covariates_rather_than_imputing_them():
    rows = synth_rows(40, TRUE, seed=6)
    for r in rows[:25]:
        r["running"] = None                    # sent before any metrics scrape
    fit = fit_covariates(rows, TTFT_COLUMNS)
    assert fit.n == 15 and fit.usable


def test_extrapolation_distance_is_zero_inside_the_probed_range():
    fit = fit_covariates(synth_rows(120, TRUE, noise=0.05, seed=7),
                         TTFT_COLUMNS)
    inside = {"const": 1.0, "L_ktok": 20.0, "L_ktok2": 400.0,
              "running": 11.0, "waiting": 2.0}
    d, per = fit.extrapolation(inside)
    assert d == 0.0 and set(per) == {"L_ktok", "L_ktok2", "running", "waiting"}


def test_extrapolation_distance_counts_standard_deviations_outside():
    fit = fit_covariates(synth_rows(200, TRUE, noise=0.05, seed=8),
                         TTFT_COLUMNS)
    rng = fit.ranges["running"]
    far = {"const": 1.0, "L_ktok": 20.0, "L_ktok2": 400.0,
           "running": rng["max"] + 2.0 * rng["sd"], "waiting": 2.0}
    d, per = fit.extrapolation(far)
    assert per["running"] == pytest.approx(2.0, rel=1e-6)
    assert d == pytest.approx(2.0, rel=1e-6)


def test_a_regressor_with_no_spread_supports_no_extrapolation_at_all():
    fit = CovariateFit(target="t", unit="s", columns=LOAD_COLUMNS,
                       coefficients={"const": 1.0, "running": 0.0,
                                     "waiting": 0.0}, n=20, dof=17,
                       residual_std=0.1, condition_number=2.0,
                       ranges={"running": {"min": 4.0, "max": 4.0, "sd": 0.0},
                               "waiting": {"min": 0.0, "max": 1.0, "sd": 0.5}})
    d, per = fit.extrapolation({"const": 1.0, "running": 5.0, "waiting": 0.5})
    assert per["running"] == math.inf and d == math.inf


def test_build_fits_splits_misses_from_hits():
    rows = (synth_rows(40, TRUE, noise=0.02, seed=9, kind="miss")
            + synth_rows(40, TRUE, noise=0.02, seed=10, kind="hit"))
    fits = build_fits(rows)
    assert fits["ttft_miss"].n == 40 and fits["ttft_hit"].n == 40
    # the load-only fits see everything, and carry no length regressor
    assert fits["itl"].columns == LOAD_COLUMNS and fits["itl"].n == 80
    assert fits["decode"].n == 80


def test_covariate_rows_leave_the_load_absent_without_a_sampler():
    t = RequestTrace(kind="miss", ttft=1.0, ptok_achieved=8_000)
    t.covariates = None
    rows = covariate_rows([t])
    assert rows[0]["L_ktok"] == pytest.approx(8.0)
    assert rows[0]["running"] is None and rows[0]["waiting"] is None
    assert not fit_covariates([{**rows[0], "y": 1.0}] * 40,
                              TTFT_COLUMNS).usable


# ============================================================================
# the operating point and the gate
# ============================================================================
def test_operating_point_is_decoders_plus_prefill_occupancy():
    cfg = RunConfig()
    preds = predict(cfg, n_iter=64)
    op = operating_point_covariates(cfg, preds, n_iter=4_000)
    assert op.refused is None
    assert op.running == pytest.approx(preds.steady_decode_seqs
                                       + preds.prefill_duty)
    assert op.waiting >= 0 and op.L_ktok > 0
    # E[L^2] >= E[L]^2, strictly for a lognormal mixture — the second moment
    # is what makes the quadratic term evaluable at a DISTRIBUTION
    assert op.L2_ktok2 > op.L_ktok ** 2
    assert "running" in op.point(TTFT_COLUMNS)


def test_operating_point_refuses_when_the_model_has_no_steady_state():
    cfg = RunConfig()
    preds = predict(cfg, n_iter=64)
    saturated = replace(preds, prefill_duty=1.0, ttft_miss_s=math.inf)
    op = operating_point_covariates(cfg, saturated)
    assert op.refused and "no steady state" in op.refused
    no_decode = replace(preds, steady_decode_seqs=None)
    assert "steady decode point" in operating_point_covariates(
        cfg, no_decode).refused


def _result(fits, op, max_extrapolation=1.0) -> SharedResult:
    return SharedResult(fits=fits, op=op, max_extrapolation=max_extrapolation)


def test_reading_is_available_for_a_well_conditioned_interpolation():
    fits = build_fits(synth_rows(120, TRUE, noise=0.02, seed=11))
    fit = fits["ttft_miss"]
    op = operating_point_covariates(RunConfig(), predict(RunConfig(),
                                                         n_iter=64))
    # place the operating point in the middle of what was probed
    mid = replace(op, running=fit.ranges["running"]["mean"],
                  waiting=fit.ranges["waiting"]["mean"],
                  L_ktok=fit.ranges["L_ktok"]["mean"],
                  L2_ktok2=fit.ranges["L_ktok2"]["mean"])
    r = _result(fits, mid).reading("ttft_miss")
    assert r["available"] and r["reason"] is None
    assert r["extrapolation"] == 0.0 and math.isfinite(r["value"])
    assert r["at"]["running"] == pytest.approx(mid.running)


def test_reading_refuses_an_operating_point_beyond_the_probed_load():
    fits = build_fits(synth_rows(120, TRUE, noise=0.02, seed=12))
    fit = fits["ttft_miss"]
    rng = fit.ranges["running"]
    op = operating_point_covariates(RunConfig(), predict(RunConfig(),
                                                         n_iter=64))
    far = replace(op, running=rng["max"] + 5 * rng["sd"],
                  waiting=fit.ranges["waiting"]["mean"],
                  L_ktok=fit.ranges["L_ktok"]["mean"],
                  L2_ktok2=fit.ranges["L_ktok2"]["mean"])
    r = _result(fits, far).reading("ttft_miss")
    assert not r["available"]
    assert "outside the probed range of `running`" in r["reason"]
    assert "--max-extrapolation" in r["reason"]
    # raising the gate lets exactly the same fit through, which is the point
    # of the threshold being a flag rather than a constant
    assert _result(fits, far, max_extrapolation=99).reading(
        "ttft_miss")["available"]


def test_reading_carries_the_refusal_when_the_fit_itself_was_refused():
    fits = build_fits(synth_rows(6, TRUE, seed=13))
    op = operating_point_covariates(RunConfig(), predict(RunConfig(),
                                                         n_iter=64))
    r = _result(fits, op).reading("ttft_miss")
    assert not r["available"] and "n=6" in r["reason"]


def test_reading_refuses_when_the_operating_point_itself_is_refused():
    from workingset.shared import OperatingPoint
    fits = build_fits(synth_rows(120, TRUE, noise=0.02, seed=14))
    r = _result(fits, OperatingPoint(refused="no steady state")).reading(
        "ttft_miss")
    assert not r["available"] and r["reason"] == "no steady state"


# ============================================================================
# the natural ladder
# ============================================================================
def test_natural_ladder_bins_by_the_load_the_server_happened_to_carry():
    rows = ([{"kind": "miss", "running": 0.5, "ttft": 1.0, "itl_ms": 10.0,
              "decode_tok_s": 100.0} for _ in range(4)]
            + [{"kind": "miss", "running": 5.0, "ttft": 3.0, "itl_ms": 30.0,
                "decode_tok_s": 40.0} for _ in range(5)]
            + [{"kind": "hit", "running": 5.5, "ttft": 0.4, "itl_ms": 32.0,
                "decode_tok_s": 38.0} for _ in range(3)])
    bins = natural_ladder(rows)
    by_lo = {b["running_lo"]: b for b in bins}
    assert set(by_lo) == {0.0, 4.0}
    assert by_lo[0.0]["n"] == 4
    assert by_lo[0.0]["ttft_miss_p50_s"] == pytest.approx(1.0)
    assert by_lo[4.0]["n"] == 8
    assert by_lo[4.0]["ttft_miss_p50_s"] == pytest.approx(3.0)
    assert by_lo[4.0]["ttft_hit_p50_s"] == pytest.approx(0.4)
    assert by_lo[4.0]["running_mean"] == pytest.approx((5 * 5.0 + 3 * 5.5) / 8)


def test_natural_ladder_reports_a_thin_bin_as_a_count_and_nothing_else():
    rows = [{"kind": "miss", "running": 1.0, "ttft": 2.0, "itl_ms": 5.0,
             "decode_tok_s": 9.0}] * 2
    b = natural_ladder(rows, min_n=3)[0]
    assert b["n"] == 2 and not b["enough"]
    assert math.isnan(b["ttft_miss_p50_s"]) and math.isnan(b["itl_p50_ms"])


def test_natural_ladder_drops_readings_with_no_load_stamp():
    rows = [{"kind": "miss", "running": None, "ttft": 2.0}] * 5
    assert natural_ladder(rows) == []


def test_ladder_model_curve_quotes_the_model_at_that_concurrency():
    cfg = RunConfig()
    a = ladder_model_curve(cfg, 4.0, n_iter=48)
    b = ladder_model_curve(cfg, 32.0, n_iter=48)
    # more decoders share one HBM budget: per-user tok/s falls, ITL rises
    assert a["decode_tok_s"] > b["decode_tok_s"] > 0
    assert b["itl_ms"] > a["itl_ms"] > 0
    # and the load that produces the bigger batch is the heavier one
    assert b["rate_req_s"] > a["rate_req_s"] > 0
    assert b["ttft_miss_s"] >= a["ttft_miss_s"] > 0


# ============================================================================
# the server-side cross-check
# ============================================================================
def _traces_for_cross():
    out = []
    for ttft, cached, ptok in ((0.62, 0, 8_000), (0.70, 0, 8_000),
                               (0.66, 4_000, 8_000)):
        t = RequestTrace(kind="miss", ttft=ttft, cached_tokens=cached,
                         ptok_achieved=ptok)
        t.itl_p50 = 0.025
        out.append(t)
    return out


def test_cross_check_reports_client_minus_server_as_the_proxy_overhead():
    m = ScriptedMetrics(window=FakeWindow(ttft_p50=0.40, tpot_s=0.020))
    c = asyncio.run(cross_check(m, 10.0, 20.0, _traces_for_cross()))
    assert c["server_ttft_p50_s"] == pytest.approx(0.40)
    assert c["client_ttft_p50_s"] == pytest.approx(0.66)
    assert c["proxy_overhead_ttft_p50_s"] == pytest.approx(0.26)
    assert c["server_itl_p50_ms"] == pytest.approx(20.0)
    assert c["proxy_overhead_itl_p50_ms"] == pytest.approx(5.0)


def test_cross_check_confirms_forced_misses_from_the_cached_tokens_readback():
    m = ScriptedMetrics(window=FakeWindow(hit_rate=0.15))
    c = asyncio.run(cross_check(m, 10.0, 20.0, _traces_for_cross()))
    # two of three misses came back with no cached prefix; the third did not
    assert c["n_miss_with_cached_readback"] == 3
    assert c["forced_miss_clean_frac"] == pytest.approx(2 / 3)
    # the window rate is over ALL traffic and cannot attribute a hit
    assert c["prefix_hit_rate"] == pytest.approx(0.15)


def test_cross_check_awaits_a_tick_so_the_window_is_covered():
    """`window()` called the instant load stops has no enclosing high endpoint
    yet. The double refuses until `next_tick` has been awaited."""
    m = ScriptedMetrics(window=FakeWindow(), ticks_before_window=1)
    c = asyncio.run(cross_check(m, 10.0, 20.0, _traces_for_cross()))
    assert m.ticks == 1 and "error" not in c


def test_cross_check_reports_an_uncovered_window_as_the_finding():
    m = ScriptedMetrics(window=None)
    c = asyncio.run(cross_check(m, 10.0, 20.0, _traces_for_cross()))
    assert "window not covered" in c["error"]


def test_cross_check_is_none_without_a_sampler():
    assert asyncio.run(cross_check(None, 0.0, 1.0, [])) is None


# ============================================================================
# hypothesis gating
# ============================================================================
def _ctx(cfg, res: SharedResult, probes=frozenset({"shared"})) -> RunContext:
    preds = predict(cfg, n_iter=64)
    ctx = RunContext(cfg, preds, small_opts(), EndpointSpec(), probes=probes)
    return ctx.seed(shared=res)


def _shared_result_for(cfg, running, waiting, extrapolation_ok=True,
                       n=120) -> SharedResult:
    """A shared result whose fit is centred on (running, waiting), so the
    configured operating point either sits inside the probed cloud or well
    outside it."""
    preds = predict(cfg, n_iter=64)
    op = operating_point_covariates(cfg, preds, n_iter=4_000)
    rows = synth_rows(n, TRUE, noise=0.02, seed=21)
    fits = build_fits(rows)
    fit = fits["ttft_miss"]
    if extrapolation_ok:
        op = replace(op, running=fit.ranges["running"]["mean"],
                     waiting=fit.ranges["waiting"]["mean"],
                     L_ktok=fit.ranges["L_ktok"]["mean"],
                     L2_ktok2=fit.ranges["L_ktok2"]["mean"])
    else:
        r = fit.ranges["running"]
        op = replace(op, running=r["max"] + 9 * r["sd"],
                     waiting=fit.ranges["waiting"]["mean"],
                     L_ktok=fit.ranges["L_ktok"]["mean"],
                     L2_ktok2=fit.ranges["L_ktok2"]["mean"])
    from workingset.probe.population import Sample
    s = Sample(n=n, n_ok=n, n_miss=n // 2, n_hit=n // 2,
               ttft_miss_mean=3.0, ttft_miss_p50=3.0, ttft_hit_p50=0.5,
               decode_p50=57.0, decode_clean_p50=57.0, itl_p50_ms=51.0,
               itl_floor_ms=0.2, chunk_tok_ratio=1.0)
    return SharedResult(fits=fits, op=op, sample=s,
                        max_extrapolation=1.0 if extrapolation_ok else 1.0)


def test_a_well_conditioned_fit_lets_h_ttft_miss_reach_a_verdict():
    cfg = RunConfig()
    res = _shared_result_for(cfg, 12.0, 2.0)
    h = REGISTRY.get("H-ttft-miss")
    ctx = _ctx(cfg, res)
    m = asyncio.run(h.measure(ctx))
    assert m.data["source"] == "shared" and m.data["fit"]["available"]
    assert "@ op (fit)" in m.text
    v = h.verdict(h.predict(cfg, ctx.predictions), m)
    assert v.status != NOT_ESTABLISHED or "not corrected" not in v.text
    assert "covariate fit at the configured operating point" in v.text
    assert "extrapolation" in v.text


def test_a_far_extrapolation_keeps_h_ttft_miss_not_established():
    cfg = RunConfig()
    res = _shared_result_for(cfg, 12.0, 2.0, extrapolation_ok=False)
    h = REGISTRY.get("H-ttft-miss")
    ctx = _ctx(cfg, res)
    m = asyncio.run(h.measure(ctx))
    assert not m.data["fit"]["available"]
    # the raw prevailing-load number is still shown, and still not scored
    assert "@ prevailing" in m.text
    v = h.verdict(h.predict(cfg, ctx.predictions), m)
    assert v.status == NOT_ESTABLISHED
    assert "outside the probed range" in v.text


def test_a_shared_run_with_no_fit_keeps_the_old_cap_exactly():
    """The honesty floor: a plain sample with no covariates is capped at
    not_established, with the SAME reason it always gave."""
    cfg = RunConfig()
    res = _shared_result_for(cfg, 12.0, 2.0)
    res.fits = build_fits(synth_rows(6, TRUE, seed=22))     # too few
    h = REGISTRY.get("H-ttft-miss")
    ctx = _ctx(cfg, res)
    m = asyncio.run(h.measure(ctx))
    v = h.verdict(h.predict(cfg, ctx.predictions), m)
    assert v.status == NOT_ESTABLISHED
    assert "prevailing load" in v.text and "n=6" in v.text


def test_h_itl_mean_and_h_steady_gate_on_their_own_fits():
    cfg = RunConfig()
    res = _shared_result_for(cfg, 12.0, 2.0)
    ctx = _ctx(cfg, res)
    for key, which in (("H-itl-mean", "itl"), ("H-steady", "decode")):
        h = REGISTRY.get(key)
        m = asyncio.run(h.measure(ctx))
        assert m.data["source"] == "shared"
        assert m.data["fit"]["which"] == which
        v = h.verdict(h.predict(cfg, ctx.predictions), m)
        assert "covariate fit" in v.text


def test_h_steady_says_what_carries_the_batch_half_of_its_claim():
    cfg = RunConfig()
    res = _shared_result_for(cfg, 12.0, 2.0)
    h = REGISTRY.get("H-steady")
    ctx = _ctx(cfg, res)
    v = h.verdict(h.predict(cfg, ctx.predictions),
                  asyncio.run(h.measure(ctx)))
    assert "predicted decode batch" in v.text
    assert "extrapolation gate" in v.text


def test_h_itl_spike_needs_the_spike_evidence_as_well_as_the_fit():
    cfg = RunConfig()
    res = _shared_result_for(cfg, 12.0, 2.0)
    res.sample.spike = {"n": 0, "n_deep": 0, "deepest_ptok": 900,
                        "min_ptok": 90_000}
    h = REGISTRY.get("H-itl-spike")
    ctx = _ctx(cfg, res)
    v = h.verdict(h.predict(cfg, ctx.predictions),
                  asyncio.run(h.measure(ctx)))
    assert v.status == NOT_ESTABLISHED
    assert "did not occur" in v.text


def test_h_itl_spike_scores_when_the_evidence_and_the_fit_are_both_there():
    cfg = RunConfig()
    preds = predict(cfg, n_iter=64)
    res = _shared_result_for(cfg, 12.0, 2.0)
    res.sample.spike = {"n": 9, "n_deep": 4, "worst_p95_ms":
                        preds.itl_worst_freeze_ms, "worst_p50_ms": 3_000.0,
                        "worst_max_ms": 6_000.0, "normal_ms": 50.0,
                        "floor_ms": 0.4, "deepest_ptok": 180_000,
                        "min_ptok": 90_000, "cap_tokens": 180_000}
    h = REGISTRY.get("H-itl-spike")
    ctx = _ctx(cfg, res)
    m = asyncio.run(h.measure(ctx))
    assert m.data["source"] == "shared" and m.data["fit"]["available"]
    v = h.verdict(h.predict(cfg, preds), m)
    assert v.status == SUPPORTED
    assert "inside the MFU bracket" in v.text and "covariate fit" in v.text


def test_h_itl_spike_refuses_the_same_evidence_when_the_load_was_never_seen():
    cfg = RunConfig()
    preds = predict(cfg, n_iter=64)
    res = _shared_result_for(cfg, 12.0, 2.0, extrapolation_ok=False)
    res.sample.spike = {"n": 9, "n_deep": 4, "worst_p95_ms":
                        preds.itl_worst_freeze_ms, "worst_p50_ms": 3_000.0,
                        "worst_max_ms": 6_000.0, "normal_ms": 50.0,
                        "floor_ms": 0.4, "deepest_ptok": 180_000,
                        "min_ptok": 90_000, "cap_tokens": 180_000}
    h = REGISTRY.get("H-itl-spike")
    v = h.verdict(h.predict(cfg, preds),
                  asyncio.run(h.measure(_ctx(cfg, res))))
    assert v.status == NOT_ESTABLISHED
    assert "outside the probed range" in v.text


def test_the_exclusive_path_is_untouched_by_any_of_this():
    """A ladder run never consults a fit: the same rung reading, the same
    verdict text as before."""
    from workingset.probe.population import Rung
    cfg = RunConfig()
    preds = predict(cfg, n_iter=64)
    rung = Rung(pop=preds.operating_point_users, n_turns=40, n_miss=20,
                ttft_miss_mean=preds.ttft_miss_s, ttft_miss_p50=3.0)
    ctx = RunContext(cfg, preds, small_opts(), EndpointSpec(),
                     exclusive=True, probes=frozenset({"ladder"}))
    ctx.seed(rungs=[rung])
    h = REGISTRY.get("H-ttft-miss")
    m = asyncio.run(h.measure(ctx))
    assert m.data["source"] == "rung" and "fit" not in m.data
    v = h.verdict(h.predict(cfg, preds), m)
    assert v.status == SUPPORTED and "operating point" in v.text


# ============================================================================
# the record and the report
# ============================================================================
def test_run_record_round_trips_the_shared_block():
    cfg = RunConfig()
    res = _shared_result_for(cfg, 12.0, 2.0)
    pl = plan(REGISTRY.all(), exclusive=False, metrics=True, burst=0)
    rec = RunRecord.new(
        "0.0.0", mode="shared", config=cfg.to_dict(),
        predictions=predict(cfg, n_iter=64).to_dict(),
        options={**small_opts().to_dict(),
                 "probe_budget": budget().to_dict()},
        plan=pl.to_dict(), shared=res.to_dict(),
        sample=res.sample.to_dict(),
        not_established=not_established_notes(
            cfg, small_opts(), pl, sample=res.sample.to_dict(),
            exclusive=False, metrics=True, shared=res.to_dict()))
    back = RunRecord.from_dict(json.loads(rec.dumps()))
    assert back.shared["fits"]["ttft_miss"]["n"] == 120
    assert back.shared["readings"]["ttft_miss"]["available"]
    assert back.shared["operating_point"]["running"] is not None
    # the trailer names the corrected readings AND every refusal
    trailer = " ".join(back.not_established)
    assert "CORRECTED" in trailer and "extrapolation distance" in trailer


def test_report_prints_the_fit_the_ladder_and_the_cross_check(capsys):
    cfg = RunConfig()
    res = _shared_result_for(cfg, 12.0, 2.0)
    res.ladder = natural_ladder(
        [{"kind": "miss", "running": 5.0, "ttft": 3.0, "itl_ms": 30.0,
          "decode_tok_s": 40.0}] * 6)
    res.cross = asyncio.run(cross_check(ScriptedMetrics(window=FakeWindow()),
                                        1.0, 2.0, _traces_for_cross()))
    rec = RunRecord.new("0.0.0", mode="shared", config=cfg.to_dict(),
                        predictions=predict(cfg, n_iter=64).to_dict(),
                        options=small_opts().to_dict(),
                        shared=res.to_dict())
    print_report(rec)
    out = capsys.readouterr().out
    assert "SHARED-ENDPOINT FIT" in out
    assert "operating point:" in out and "SCORED" in out
    assert "NATURAL LADDER" in out
    assert "SERVER CROSS-CHECK" in out and "proxy" in out
    assert "forced misses confirmed cold" in out
    # no --shared-ladder, so no model columns were computed
    assert "model TTFT" not in out


def test_report_reads_the_ladder_against_the_model_curve(capsys):
    cfg = RunConfig()
    res = _shared_result_for(cfg, 12.0, 2.0)
    bins = natural_ladder(
        [{"kind": "miss", "running": 5.0, "ttft": 3.0, "itl_ms": 30.0,
          "decode_tok_s": 40.0}] * 6)
    bins[0]["model"] = ladder_model_curve(cfg, bins[0]["running_mean"],
                                          n_iter=48)
    res.ladder = bins
    rec = RunRecord.new("0.0.0", mode="shared", config=cfg.to_dict(),
                        predictions=predict(cfg, n_iter=64).to_dict(),
                        options=small_opts().to_dict(),
                        shared=res.to_dict())
    print_report(rec)
    out = capsys.readouterr().out
    assert "model TTFT" in out and "implied req/s" in out
    assert "the arrival rate that would PRODUCE that batch" in out


def test_report_of_an_aborted_run_leads_with_the_rail(capsys):
    cfg = RunConfig()
    res = _shared_result_for(cfg, 12.0, 2.0)
    res.aborted = "the server's queue reached 4 waiting requests"
    rec = RunRecord.new("0.0.0", mode="shared", config=cfg.to_dict(),
                        predictions=predict(cfg, n_iter=64).to_dict(),
                        options=small_opts().to_dict(),
                        shared=res.to_dict())
    print_report(rec)
    assert "ABORTED BY A SAFETY RAIL" in capsys.readouterr().out


# ============================================================================
# the CLI plan
# ============================================================================
def test_dry_run_prints_the_budget_and_the_shared_plan(tmp_path, capsys):
    from workingset.cli import main
    cfg_path = tmp_path / "ws.toml"
    cfg_path.write_text(RunConfig().dumps("toml"), encoding="utf-8")
    assert main(["test", str(cfg_path), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "PROBE BUDGET" in out and "at most 2 of our requests" in out
    assert "SHARED-MODE PLAN" in out
    assert "TTFT = c0 + c1 L + c2 L^2 + c3 running + c4 waiting" in out
    assert "--max-extrapolation" in out
    assert "NOT ENFORCEABLE" in out          # no --metrics-url was given


def test_dry_run_exclusive_says_the_rails_are_off(tmp_path, capsys):
    from workingset.cli import main
    cfg_path = tmp_path / "ws.toml"
    cfg_path.write_text(RunConfig().dumps("toml"), encoding="utf-8")
    assert main(["test", str(cfg_path), "--dry-run", "--exclusive"]) == 0
    out = capsys.readouterr().out
    assert "PROBE BUDGET" in out and "the rails are off" in out
    assert "SHARED-MODE PLAN" not in out
    assert "LOAD LADDER" in out


def test_dry_run_warns_when_the_plan_will_not_fit_its_own_token_budget(
        tmp_path, capsys):
    from workingset.cli import main
    cfg_path = tmp_path / "ws.toml"
    cfg_path.write_text(RunConfig().dumps("toml"), encoding="utf-8")
    assert main(["test", str(cfg_path), "--dry-run",
                 "--max-probe-tokens", "100000"]) == 0
    out = capsys.readouterr().out
    assert "exceed --max-probe-tokens 100,000" in out
    assert "abort partway through" in out


def test_dry_run_warns_when_the_length_ladder_collapses(tmp_path, capsys):
    """Every requested fraction clipping to the prefix floor leaves ONE
    length, and a quadratic in L cannot be identified from one length."""
    from workingset.cli import main
    cfg_path = tmp_path / "ws.toml"
    cfg_path.write_text(RunConfig().dumps("toml"), encoding="utf-8")
    assert main(["test", str(cfg_path), "--dry-run",
                 "--context-cap-tokens", "16000"]) == 0
    out = capsys.readouterr().out
    assert "the ladder collapsed to 2 distinct length" in out
    assert "at least three distinct lengths" in out


def test_flags_override_one_rail_at_a_time(tmp_path, capsys):
    from workingset.cli import main
    cfg_path = tmp_path / "ws.toml"
    cfg_path.write_text(RunConfig().dumps("toml"), encoding="utf-8")
    assert main(["test", str(cfg_path), "--dry-run", "--max-extra-load", "6",
                 "--abort-if-waiting", "12", "--no-canary"]) == 0
    out = capsys.readouterr().out
    assert "at most 6 of our requests" in out
    assert "requests_waiting > 12" in out
    assert "canary         : off" in out


# ============================================================================
# the real sampler, not a double
# ============================================================================
def test_the_shared_probe_stamps_real_gauges_off_a_real_sampler():
    """End to end over `MetricsSampler` itself: the clock conversion AND the
    semantic-gauge lookup have to both be right, or every request records an
    empty covariate set and the whole fit silently rests on nothing."""
    from test_metrics import FakeServer, _until

    from workingset.metrics import MetricsSampler

    cfg, opts = small_cfg(), small_opts()

    async def go():
        srv = FakeServer()
        async with MetricsSampler("http://fake/metrics", interval=0.02,
                                  client=srv.client()) as s:
            await _until(lambda: len(s) >= 3)
            async with client_for(fake_server()) as client:
                return await run_shared(
                    client, EndpointSpec(base_url="http://x/v1", model="m"),
                    cfg, opts,
                    build_prefixes(cfg.workload, opts.chars_per_token),
                    budget(abort_if_waiting=None, abort_if_kv_above=None),
                    shared_opts(rounds=4), s)

    res = asyncio.run(go())
    assert res.n_covariate_rows > 0, "no request carried a load stamp"
    running = [r["running"] for r in covariate_rows(res.sample.traces)]
    assert all(v is not None for v in running)
    # the fake's gauge counts scrapes, so a covariate that never moved would
    # mean every request read the same snapshot
    assert len(set(running)) > 1, running
    # and the governor saw those same gauges, so the rails were live
    assert res.governor["n_gauge_checks"] > 0
    assert res.governor["peak_kv_cache_usage"] is not None
