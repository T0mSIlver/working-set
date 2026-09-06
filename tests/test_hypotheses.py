"""Hypotheses: registry, requirement gating, verdict logic, the run record and
the end-to-end dry run."""
from __future__ import annotations

import asyncio
import json
import math
from dataclasses import replace

import pytest

from workingset import cli
from workingset.config import RunConfig
from workingset.hypotheses import (BOUNDED_BELOW, BURST, EXCLUSIVE, METRICS,
                                   NOT_ESTABLISHED, REFUTED, REGISTRY,
                                   STATUSES, SUPPORTED, Hypothesis, Prediction,
                                   Registry, RunContext, Verdict,
                                   bracket_verdict, plan)
from workingset.predict import predict
from workingset.probe import ProbeOptions, Rung, Sample
from workingset.probe.burst import BurstResult
from workingset.probe.request import EndpointSpec
from workingset.record import RunRecord, not_established_notes
from workingset.report import print_report

from test_probe import trace


# ============================================================================
# helpers
# ============================================================================
def rung(pop, passed=True, reasons=(), **kw) -> Rung:
    """A rung with eviction WITNESSED and warm by default — `warm_held` now
    needs evidence, so a fixture that means "held" has to say so."""
    r = Rung(pop=pop, n_sub=0, n_turns=10, n_hit=8, n_miss=2, evict_frac=0.0)
    r.passed, r.reasons = passed, list(reasons)
    for k, v in kw.items():
        setattr(r, k, v)
    return r


def spike(worst_p95, n=3, normal=5.0, floor=0.01, n_deep=1,
          deepest=95_000, min_ptok=90_000) -> dict:
    return {"n": n, "worst_p95_ms": worst_p95, "worst_p50_ms": worst_p95,
            "worst_max_ms": worst_p95 * 2, "normal_ms": normal,
            "floor_ms": floor, "deepest_ptok": deepest, "cap_tokens": 180_000,
            "min_ptok": min_ptok, "n_deep": n_deep}


NO_SPIKE = {"n": 0, "worst_p95_ms": None, "worst_p50_ms": None,
            "worst_max_ms": None, "normal_ms": None, "floor_ms": None,
            "deepest_ptok": 1_000, "cap_tokens": 180_000, "min_ptok": 90_000,
            "n_deep": 0}


def ctx_with(cfg=None, preds=None, opts=None, exclusive=False, burst=0,
             probes=frozenset(), ep=None, **seed) -> RunContext:
    cfg = cfg or RunConfig()
    preds = preds if preds is not None else predict(cfg, n_iter=40)
    c = RunContext(cfg, preds, opts or ProbeOptions(), ep or EndpointSpec(),
                   exclusive=exclusive, burst=burst, probes=frozenset(probes))
    return c.seed(**seed)


def ladder_ctx(preds=None, rungs=(), **kw) -> RunContext:
    return ctx_with(preds=preds, exclusive=True, probes={"ladder"},
                    rungs=list(rungs), **kw)


def shared_ctx(preds=None, sample=None, **kw) -> RunContext:
    return ctx_with(preds=preds, probes={"sample"}, sample=sample, **kw)


def score(h, ctx) -> tuple:
    p = h.predict(ctx.cfg, ctx.predictions)
    m = asyncio.run(h.measure(ctx))
    return p, m, h.verdict(p, m)


# ============================================================================
# registry / selection
# ============================================================================
def test_registry_carries_every_harness_hypothesis():
    assert REGISTRY.keys == ["H-cache", "H-decode", "H-latency",
                             "H-saturation", "H-binding", "H-ttft-miss",
                             "H-burst", "H-steady", "H-itl-spike",
                             "H-itl-mean"]
    for h in REGISTRY:
        assert h.title and h.requires <= {EXCLUSIVE, METRICS, BURST}


def test_select_resolves_keys_in_registry_order():
    got = REGISTRY.select(["H-itl-mean", "H-cache", "H-cache"])
    assert [h.key for h in got] == ["H-cache", "H-itl-mean"]
    assert len(REGISTRY.select(None)) == len(REGISTRY)


def test_select_rejects_a_typo_rather_than_testing_less():
    with pytest.raises(KeyError, match="unknown hypothesis"):
        REGISTRY.select(["H-cash"])


def test_duplicate_keys_are_a_registry_error():
    class A(Hypothesis):
        key = "dup"

    with pytest.raises(ValueError, match="duplicate"):
        Registry([A(), A()])


# ============================================================================
# requirement gating
# ============================================================================
def test_shared_mode_skips_every_exclusive_hypothesis():
    p = plan(REGISTRY.all(), exclusive=False, metrics=False, burst=0)
    assert [h.key for h in p.selected] == ["H-ttft-miss", "H-steady",
                                           "H-itl-spike", "H-itl-mean"]
    skipped = {h.key: r for h, r in p.skipped}
    assert set(skipped) == {"H-cache", "H-decode", "H-latency",
                            "H-saturation", "H-binding", "H-burst"}
    assert "--exclusive" in skipped["H-cache"]
    # the ladder is not run, so the cheap ones fall back to their own sample —
    # which in shared mode is the SHARED probe (the same handful of requests,
    # plus covariate stamping and the safety rails), never both
    assert p.probes == {"shared"} and not p.run_ladder


def test_exclusive_mode_runs_the_ladder_once_for_all_of_them():
    p = plan(REGISTRY.all(), exclusive=True, metrics=False, burst=0)
    assert p.run_ladder and "sample" not in p.probes
    assert [h.key for h, _ in p.skipped] == ["H-burst"]
    assert "--burst" in dict((h.key, r) for h, r in p.skipped)["H-burst"]


def test_permissions_are_not_probes():
    """H-burst carries the `exclusive` PERMISSION so it may generate a
    standing load. It does not ladder, and a plan that said so was planning
    work the run never did."""
    p = plan([REGISTRY.get("H-burst")], exclusive=True, burst=8)
    assert p.probes == {"burst"}
    assert not p.run_ladder


def test_itl_spike_uses_a_burst_only_when_one_is_already_planned():
    """It must never ADD a burst — and a 64-user standing load — to a plan
    whose dry-run printed `sample`."""
    alone = plan([REGISTRY.get("H-itl-spike")], exclusive=True, burst=8)
    assert alone.probes == {"sample"}
    assert alone.to_dict()["selected"][0]["probes"] == ["sample"]

    withburst = plan([REGISTRY.get("H-burst"), REGISTRY.get("H-itl-spike")],
                     exclusive=True, burst=8)
    assert withburst.probes == {"burst"}
    probes = {h["key"]: h["probes"] for h in withburst.to_dict()["selected"]}
    assert probes["H-itl-spike"] == ["burst"]


def test_cheap_hypotheses_read_a_ladder_that_already_exists():
    p = plan([REGISTRY.get("H-binding"), REGISTRY.get("H-itl-mean")],
             exclusive=True)
    assert p.probes == {"ladder"}
    probes = {h["key"]: h["probes"] for h in p.to_dict()["selected"]}
    assert probes["H-itl-mean"] == ["ladder"]


def test_burst_needs_both_the_flag_and_exclusive():
    only_flag = plan([REGISTRY.get("H-burst")], exclusive=False, burst=8)
    assert only_flag.skipped and "--exclusive" in only_flag.skipped[0][1]
    both = plan([REGISTRY.get("H-burst")], exclusive=True, burst=8)
    assert [h.key for h in both.selected] == ["H-burst"]
    assert "burst" in both.probes


def test_metrics_requirement_gates_on_the_sampler():
    class NeedsMetrics(Hypothesis):
        key, title, requires = "H-fake", "needs /metrics", frozenset({METRICS})

    p = plan([NeedsMetrics()], exclusive=True, metrics=False)
    assert p.skipped and "--metrics-url" in p.skipped[0][1]
    assert plan([NeedsMetrics()], exclusive=True, metrics=True).selected


def test_a_skip_names_every_unmet_requirement():
    p = plan([REGISTRY.get("H-burst")], exclusive=False, burst=0)
    reason = p.skipped[0][1]
    assert "--exclusive" in reason and "--burst" in reason


# ============================================================================
# bracket_verdict — one test per status
# ============================================================================
def test_bracket_verdict_supported_inside_the_bracket():
    v = bracket_verdict(400, 300, 500)
    assert v.status == SUPPORTED and "inside measured" in v.text


def test_bracket_verdict_refuted_outside_it():
    assert bracket_verdict(1000, 300, 500).status == REFUTED
    assert bracket_verdict(50, 300, 500).status == REFUTED


def test_bracket_verdict_not_established_within_25_percent():
    assert bracket_verdict(560, 300, 500).status == NOT_ESTABLISHED
    assert bracket_verdict(None, 1, 2).status == NOT_ESTABLISHED
    assert bracket_verdict(400, None, None).status == NOT_ESTABLISHED


def test_bracket_verdict_bounded_below_when_nothing_failed():
    v = bracket_verdict(400, 300, None)
    assert v.status == BOUNDED_BELOW and "not reached" in v.text
    v = bracket_verdict(400, 500, None)
    assert v.status == BOUNDED_BELOW and "within 25% above" in v.text
    # passing far above the prediction refutes it: the model was conservative
    assert bracket_verdict(100, 500, None).status == REFUTED


def test_bracket_verdict_upper_bound_only():
    assert bracket_verdict(100, None, 200).status == NOT_ESTABLISHED
    assert bracket_verdict(1000, None, 200).status == REFUTED


def test_verdict_rejects_an_unknown_status():
    with pytest.raises(ValueError, match="unknown verdict status"):
        Verdict("maybe", "x")
    assert set(STATUSES) == {SUPPORTED, REFUTED, BOUNDED_BELOW,
                             NOT_ESTABLISHED}


# ============================================================================
# the ceiling hypotheses, on synthetic ladders
# ============================================================================
def test_h_binding_reads_the_slo_capacity_bracket():
    preds = predict(RunConfig(), n_iter=40)
    lim = preds.predicted_limit_users
    ladder = [rung(lim - 5, passed=True), rung(lim + 5, passed=False,
                                               reasons=["p95 TTFT 12.00s > 10s"])]
    p, m, v = score(REGISTRY.get("H-binding"), ladder_ctx(preds, ladder))
    assert m.lo == lim - 5 and m.hi == lim + 5
    assert v.status == SUPPORTED


def test_h_decode_is_not_separable_without_a_floor_failure():
    ladder = [rung(10), rung(20, passed=False,
                             reasons=["p95 TTFT 12.00s > 10s"])]
    _, m, v = score(REGISTRY.get("H-decode"), ladder_ctx(rungs=ladder))
    assert m.text == "not separable"
    assert v.status == NOT_ESTABLISHED and "decode-floor" in v.text


def test_h_decode_brackets_on_a_floor_failure():
    preds = predict(RunConfig(), n_iter=40)
    dc = preds.decode_ceiling_users
    ladder = [rung(dc - 3), rung(dc + 3, passed=False,
                                 reasons=["decode p50 11.0 < 40 tok/s"])]
    _, m, v = score(REGISTRY.get("H-decode"), ladder_ctx(preds, ladder))
    assert (m.lo, m.hi) == (dc - 3, dc + 3) and v.status == SUPPORTED


def test_h_latency_brackets_on_a_ttft_failure():
    preds = predict(RunConfig(), n_iter=40)
    lc = preds.latency_ceiling_users
    ladder = [rung(lc - 2), rung(lc + 2, passed=False,
                                 reasons=["p95 TTFT 12.00s > 10s"])]
    _, m, v = score(REGISTRY.get("H-latency"), ladder_ctx(preds, ladder))
    assert (m.lo, m.hi) == (lc - 2, lc + 2) and v.status == SUPPORTED


def test_h_cache_is_bounded_below_when_nothing_evicts():
    preds = predict(RunConfig(), n_iter=40)
    wc = preds.warm_capacity_p5
    ladder = [rung(wc + 10, evict_frac=0.01)]
    _, m, v = score(REGISTRY.get("H-cache"), ladder_ctx(preds, ladder))
    assert m.lo == wc + 10 and m.hi is None
    assert v.status == BOUNDED_BELOW and "lower bound" in v.text


def test_h_cache_needs_a_witness_before_it_calls_a_rung_warm():
    """Inherited from the harness: a rung with NO forced miss leaves
    evict_frac nan, the classifier cannot run, and "no evidence of eviction"
    was counted as "held". Missing evidence is not evidence."""
    preds = predict(RunConfig(), n_iter=40)
    blind = rung(preds.warm_capacity_p5 + 10, n_hit=8, n_miss=0,
                 evict_frac=float("nan"), cached_frac=float("nan"))
    _, m, v = score(REGISTRY.get("H-cache"), ladder_ctx(preds, [blind]))
    assert m.text == "not separable" and m.lo is None
    assert v.status == NOT_ESTABLISHED and "never observed" in v.text

    # the server's own cached_tokens is a witness where the classifier is not
    witnessed = rung(preds.warm_capacity_p5 + 10, n_hit=8, n_miss=0,
                     evict_frac=float("nan"), cached_frac=1.0)
    _, m2, v2 = score(REGISTRY.get("H-cache"), ladder_ctx(preds, [witnessed]))
    assert v2.status == BOUNDED_BELOW and m2.lo == preds.warm_capacity_p5 + 10


def test_h_cache_is_refuted_when_eviction_arrives_far_early():
    preds = predict(RunConfig(), n_iter=40)
    wc = preds.warm_capacity_p5
    ladder = [rung(max(1, wc // 4), n_hit=8, evict_frac=0.4)]
    _, m, v = score(REGISTRY.get("H-cache"), ladder_ctx(preds, ladder))
    assert m.hi == max(1, wc // 4)
    assert v.status == REFUTED and "re-prefilled" in v.text


def test_h_saturation_never_rises_above_not_established():
    h = REGISTRY.get("H-saturation")
    plateau = rung(100, passed=False, reasons=["p95 TTFT 20.00s > 10s"],
                   achieved_rps=0.5, offered_rps=3.3)
    _, m, v = score(h, ladder_ctx(rungs=[plateau]))
    assert v.status == NOT_ESTABLISHED and "throughput plateau" in v.text
    _, _, v2 = score(h, ladder_ctx(rungs=[rung(100)]))
    assert v2.status == NOT_ESTABLISHED and "throttles" in v2.text


# ============================================================================
# the cheap hypotheses
# ============================================================================
def sample_with(**kw) -> Sample:
    s = Sample(n=4, n_ok=4, n_miss=2, n_hit=2)
    for k, v in kw.items():
        setattr(s, k, v)
    return s


# ---- item 1: a shared sample can support nothing ------------------------
def test_shared_mode_caps_every_cheap_verdict_at_not_established():
    """A prediction is made AT the configured operating point. A sample taken
    under unknown background load is a different experiment: a lower number is
    what a quieter server gives and a higher one what a busier server gives,
    so neither supports nor refutes. The number is still reported."""
    preds = predict(RunConfig(), n_iter=40)
    cases = {
        # each of these would score SUPPORTED on a ladder rung
        "H-ttft-miss": sample_with(ttft_miss_mean=preds.ttft_miss_s,
                                   ttft_miss_p50=1.0),
        "H-steady": sample_with(decode_clean_p50=preds.steady_decode_tok_s,
                                decode_seqs=preds.steady_decode_seqs,
                                n_seqs_obs=9),
        "H-itl-mean": sample_with(itl_p50_ms=preds.itl_normal_ms,
                                  itl_floor_ms=0.0001, chunk_tok_ratio=1.0),
        "H-itl-spike": sample_with(spike=spike(preds.itl_worst_freeze_ms)),
    }
    for key, s in cases.items():
        _, m, v = score(REGISTRY.get(key), shared_ctx(preds, s))
        assert v.status == NOT_ESTABLISHED, f"{key} -> {v}"
        assert "prevailing load" in v.text, key
        assert m.value is not None, key       # the measurement is kept


def test_shared_mode_caps_a_would_be_refutation_too():
    """The cap is symmetric: an unknown load cannot refute either."""
    preds = predict(RunConfig(), n_iter=40)
    cases = {
        "H-ttft-miss": sample_with(ttft_miss_mean=preds.ttft_miss_s * 20,
                                   ttft_miss_p50=1.0),
        "H-itl-mean": sample_with(itl_p50_ms=preds.itl_normal_ms * 10,
                                  itl_floor_ms=0.0001, chunk_tok_ratio=1.0),
        "H-itl-spike": sample_with(spike=spike(1.0)),
    }
    for key, s in cases.items():
        _, _, v = score(REGISTRY.get(key), shared_ctx(preds, s))
        assert v.status == NOT_ESTABLISHED, f"{key} -> {v}"
        assert "prevailing load" in v.text, key


def test_ttft_miss_statement_only_promises_a_rung_when_there_is_one():
    preds = predict(RunConfig(), n_iter=40)
    h = REGISTRY.get("H-ttft-miss")
    assert "ladder rung" in h.statement_for(RunConfig(), preds, {"ladder"})
    shared = h.statement_for(RunConfig(), preds, {"sample"})
    assert "ladder rung" not in shared and "cannot be scored" in shared


# ---- exclusive-mode scoring still works ---------------------------------
def test_h_ttft_miss_reads_the_rung_nearest_the_operating_point():
    preds = predict(RunConfig(), n_iter=40)
    op, tm = preds.operating_point_users, preds.ttft_miss_s
    ladder = [rung(op, n_miss=5, ttft_miss_mean=tm),
              rung(op * 4, n_miss=5, ttft_miss_mean=tm * 9)]
    _, m, v = score(REGISTRY.get("H-ttft-miss"), ladder_ctx(preds, ladder))
    assert m.data["at_users"] == op
    assert v.status == SUPPORTED and "operating point" in v.text


def test_h_ttft_miss_refuted_far_from_the_prediction_on_a_ladder():
    preds = predict(RunConfig(), n_iter=40)
    op = preds.operating_point_users
    ladder = [rung(op, n_miss=5, ttft_miss_mean=preds.ttft_miss_s * 20)]
    _, _, v = score(REGISTRY.get("H-ttft-miss"), ladder_ctx(preds, ladder))
    assert v.status == REFUTED


def test_h_ttft_miss_not_established_without_misses():
    preds = predict(RunConfig(), n_iter=40)
    _, m, v = score(REGISTRY.get("H-ttft-miss"),
                    shared_ctx(preds, sample_with()))
    assert m.text == "not separable" and v.status == NOT_ESTABLISHED


# ---- item 9: an exclusive run never falls back to a live sample ---------
def test_exclusive_run_reports_not_separable_instead_of_sampling():
    """After the ladder has drained, the endpoint is idle. Sampling it and
    scoring that against an operating-point prediction is the shared-mode
    error with an exclusive label on it — the harness said "not separable"."""
    preds = predict(RunConfig(), n_iter=40)
    ctx = ladder_ctx(preds, [rung(preds.operating_point_users, n_miss=0,
                                  ttft_miss_mean=float("nan"))])
    _, m, v = score(REGISTRY.get("H-ttft-miss"), ctx)
    assert m.text == "not separable" and v.status == NOT_ESTABLISHED
    assert ctx.cached()["sample"] is None      # nothing was fired


# ---- item 3: H-steady ---------------------------------------------------
def test_h_steady_needs_both_halves_before_it_scores():
    """A batch size is half the claim ("~n sequences at ~v tok/s each — NOT
    the whole warm pool"). Without /metrics there is no batch size."""
    preds = predict(RunConfig(), n_iter=40)
    op = preds.operating_point_users
    no_seqs = rung(op, decode_clean_p50=preds.steady_decode_tok_s,
                   decode_seqs=float("nan"), n_seqs_obs=0)
    _, m, v = score(REGISTRY.get("H-steady"), ladder_ctx(preds, [no_seqs]))
    assert not math.isfinite(m.data["seqs"])
    assert v.status == NOT_ESTABLISHED and "needs /metrics" in v.text


def test_h_steady_scores_the_freeze_excluded_rate_not_the_mean():
    """decode_p50 is a mean over the stream, freezes included;
    steady_decode_point predicts the clean speed between spikes. Scoring the
    mean charges every freeze to decode."""
    preds = predict(RunConfig(), n_iter=40)
    op = preds.operating_point_users
    r = rung(op, decode_p50=preds.steady_decode_tok_s / 10,
             decode_clean_p50=preds.steady_decode_tok_s,
             decode_seqs=preds.steady_decode_seqs, n_seqs_obs=12)
    _, m, v = score(REGISTRY.get("H-steady"), ladder_ctx(preds, [r]))
    assert m.value == pytest.approx(preds.steady_decode_tok_s)
    assert m.data["decode_p50_with_freezes"] < m.value
    assert v.status == SUPPORTED and "snapshots" in v.text


def test_h_steady_takes_the_weaker_of_the_two_halves():
    preds = predict(RunConfig(), n_iter=40)
    op = preds.operating_point_users
    r = rung(op, decode_clean_p50=preds.steady_decode_tok_s,
             decode_seqs=preds.steady_decode_seqs * 50, n_seqs_obs=12)
    _, _, v = score(REGISTRY.get("H-steady"), ladder_ctx(preds, [r]))
    assert v.status == REFUTED


# ---- item 4: H-itl-spike ------------------------------------------------
def test_h_itl_spike_scores_the_p95_against_the_mfu_bracket():
    preds = predict(RunConfig(), n_iter=40)
    h, op = REGISTRY.get("H-itl-spike"), preds.operating_point_users
    inside = rung(op, spike=spike(preds.itl_worst_freeze_ms))
    _, m, v = score(h, ladder_ctx(preds, [inside]))
    # the SCORED number is the p95, not the max-of-maxes the ported comment
    # calls a footnote — which is in data, and is twice as large here
    assert m.value == pytest.approx(preds.itl_worst_freeze_ms)
    assert m.data["spike_worst_max_ms"] == pytest.approx(
        2 * preds.itl_worst_freeze_ms)
    assert v.status == SUPPORTED and "p95" in v.text

    edge = rung(op, spike=spike(preds.itl_freeze_lo_ms * 0.8))
    _, _, v = score(h, ladder_ctx(preds, [edge]))
    assert v.status == NOT_ESTABLISHED

    small = rung(op, spike=spike(1.0))
    _, _, v = score(h, ladder_ctx(preds, [small]))
    assert v.status == REFUTED and "below" in v.text


def test_h_itl_spike_will_not_refute_an_experiment_that_did_not_run():
    """The prediction is the last chunk of a FULL-CONTEXT cold prefill. A run
    whose deepest cold prefill was 1k tokens has not performed it, and 1 ms
    gaps do not bound a 4-second event."""
    preds = predict(RunConfig(), n_iter=40)
    op = preds.operating_point_users
    shallow = rung(op, spike=dict(NO_SPIKE), itl_max_ms=1.0, itl_p50_ms=1.0)
    _, m, v = score(REGISTRY.get("H-itl-spike"), ladder_ctx(preds, [shallow]))
    assert m.value is None and m.text == "not observed"
    assert v.status == NOT_ESTABLISHED
    assert "did not occur" in v.text and "90,000 tokens" in v.text


def test_h_itl_spike_says_so_when_nothing_was_decoding_through_the_prefill():
    preds = predict(RunConfig(), n_iter=40)
    op = preds.operating_point_users
    ev = dict(NO_SPIKE, n_deep=2, deepest_ptok=150_000)
    _, m, v = score(REGISTRY.get("H-itl-spike"),
                    ladder_ctx(preds, [rung(op, spike=ev)]))
    assert v.status == NOT_ESTABLISHED
    assert "nothing was decoding through one" in v.text


def _burst_ctx(preds, b, probes=("burst",)) -> RunContext:
    c = RunContext(RunConfig(), preds, ProbeOptions(), EndpointSpec(),
                   exclusive=True, burst=b.n, burst_users=b.standing_users,
                   probes=frozenset(probes))
    return c.seed(burst=b)


def test_h_itl_spike_uses_the_burst_when_the_plan_ran_one():
    preds = predict(RunConfig(), n_iter=40)
    b = BurstResult(n=4, standing_users=8, n_ok=4, last_ttft_s=1.0,
                    standing_n=3, spike=spike(preds.itl_worst_freeze_ms))
    _, m, v = score(REGISTRY.get("H-itl-spike"), _burst_ctx(preds, b))
    assert m.data["source"] == "burst" and v.status == SUPPORTED


def test_h_itl_spike_does_not_fire_a_burst_the_plan_did_not_include():
    """It used to run one whenever `--burst N --exclusive` was passed, even on
    a plan whose dry-run printed `probes: sample` — a surprise 64-user
    standing load."""
    preds = predict(RunConfig(), n_iter=40)
    ctx = ctx_with(preds=preds, exclusive=True, burst=8, probes={"sample"},
                   sample=sample_with(spike=spike(preds.itl_worst_freeze_ms)))
    _, m, _ = score(REGISTRY.get("H-itl-spike"), ctx)
    assert m.data["source"] == "sample"
    assert ctx.cached()["burst"] is None


# ---- H-itl-mean ---------------------------------------------------------
def test_h_itl_mean_guards_on_the_client_floor():
    preds = predict(RunConfig(), n_iter=40)
    normal, op = preds.itl_normal_ms, preds.operating_point_users
    good = rung(op, itl_p50_ms=normal, itl_floor_ms=normal / 100,
                chunk_tok_ratio=1.0)
    _, _, v = score(REGISTRY.get("H-itl-mean"), ladder_ctx(preds, [good]))
    assert v.status == SUPPORTED

    floored = rung(op, itl_p50_ms=normal, itl_floor_ms=normal * 0.9,
                   chunk_tok_ratio=1.0)
    _, _, v = score(REGISTRY.get("H-itl-mean"), ladder_ctx(preds, [floored]))
    assert v.status == NOT_ESTABLISHED and "client floor" in v.text


def test_h_itl_mean_refuses_a_multi_token_event_comparison():
    preds = predict(RunConfig(), n_iter=40)
    op = preds.operating_point_users
    r = rung(op, itl_p50_ms=preds.itl_normal_ms, itl_floor_ms=0.0001,
             chunk_tok_ratio=4.0)
    _, _, v = score(REGISTRY.get("H-itl-mean"), ladder_ctx(preds, [r]))
    assert v.status == NOT_ESTABLISHED and "per SSE event" in v.text


def test_h_itl_mean_refuted_when_the_gap_is_nowhere_near():
    preds = predict(RunConfig(), n_iter=40)
    op = preds.operating_point_users
    r = rung(op, itl_p50_ms=preds.itl_normal_ms * 10, itl_floor_ms=0.0001,
             chunk_tok_ratio=1.0)
    _, _, v = score(REGISTRY.get("H-itl-mean"), ladder_ctx(preds, [r]))
    assert v.status == REFUTED


def test_h_steady_not_established_without_a_steady_point():
    preds = replace(predict(RunConfig(), n_iter=40), steady_decode_tok_s=None,
                    steady_decode_seqs=None, itl_normal_ms=None,
                    itl_worst_freeze_ms=None, itl_freeze_lo_ms=None,
                    itl_freeze_hi_ms=None)
    for key in ("H-steady", "H-itl-spike", "H-itl-mean"):
        _, _, v = score(REGISTRY.get(key), shared_ctx(preds, sample_with()))
        assert v.status == NOT_ESTABLISHED, key


# ============================================================================
# H-burst
# ============================================================================
def test_h_burst_supported_when_a_small_flush_drains_in_time():
    preds = replace(predict(RunConfig(), n_iter=40), bstar_misses=10.0)
    b = BurstResult(n=4, standing_users=8, n_ok=4, last_ttft_s=2.0,
                    drain_s=2.1, ttft_p50_s=1.0)
    _, m, v = score(REGISTRY.get("H-burst"), _burst_ctx(preds, b))
    assert m.value == 2.0
    assert v.status == SUPPORTED and "inside budget" in v.text


def test_h_burst_supported_when_a_big_flush_breaches_it():
    preds = replace(predict(RunConfig(), n_iter=40), bstar_misses=2.0)
    b = BurstResult(n=20, standing_users=8, n_ok=20, last_ttft_s=40.0,
                    drain_s=40.0)
    _, _, v = score(REGISTRY.get("H-burst"), _burst_ctx(preds, b))
    assert v.status == SUPPORTED and "as predicted" in v.text


def test_h_burst_refuted_when_the_outcome_contradicts_bstar():
    preds = replace(predict(RunConfig(), n_iter=40), bstar_misses=20.0)
    b = BurstResult(n=2, standing_users=8, n_ok=2, last_ttft_s=90.0)
    _, _, v = score(REGISTRY.get("H-burst"), _burst_ctx(preds, b))
    assert v.status == REFUTED


def test_h_burst_not_established_at_the_threshold_itself():
    preds = replace(predict(RunConfig(), n_iter=40), bstar_misses=8.0)
    b = BurstResult(n=9, standing_users=8, n_ok=9, last_ttft_s=1.0)
    _, _, v = score(REGISTRY.get("H-burst"), _burst_ctx(preds, b))
    assert v.status == NOT_ESTABLISHED and "not resolved" in v.text


def test_h_burst_refuses_to_score_a_partial_flush():
    """Inherited from the harness: `last_ttft_s` is the max over the requests
    that ANSWERED, so a burst of 20 where 19 failed reports the drain of the
    one that worked — and "supported"."""
    preds = replace(predict(RunConfig(), n_iter=40), bstar_misses=2.0)
    b = BurstResult(n=20, standing_users=8, n_ok=1, n_err=19, last_ttft_s=0.5)
    _, _, v = score(REGISTRY.get("H-burst"), _burst_ctx(preds, b))
    assert v.status == NOT_ESTABLISHED
    assert "19 of 20 burst requests failed" in v.text


def test_h_burst_not_measured_when_the_run_ended_first():
    preds = predict(RunConfig(), n_iter=40)
    ctx = RunContext(RunConfig(), preds, ProbeOptions(), EndpointSpec(),
                     exclusive=True, burst=4, probes=frozenset({"burst"}))
    ctx.freeze()
    _, m, v = score(REGISTRY.get("H-burst"), ctx)
    assert m.value is None and v.status == NOT_ESTABLISHED


# ============================================================================
# the probe cache
# ============================================================================
def test_cache_key_separates_endpoints_and_configs():
    """The key was (options, predictions) only: pointing the context at
    another endpoint handed back the sample measured against the first."""
    preds = predict(RunConfig(), n_iter=40)
    a = EndpointSpec(base_url="http://a/v1", model="m")
    b = EndpointSpec(base_url="http://b/v1", model="m")
    s = sample_with(ttft_miss_mean=1.0)
    ctx_a = ctx_with(preds=preds, probes={"sample"}, ep=a, sample=s)
    assert asyncio.run(ctx_a.sample()) is s

    ctx_b = ctx_with(preds=preds, probes={"sample"}, ep=b)
    ctx_b._cache = ctx_a._cache        # the same cache, a different endpoint
    ctx_b.freeze()
    assert asyncio.run(ctx_b.sample()) is None

    # and a different workload is a different experiment on the same endpoint
    other = replace(RunConfig(), workload=replace(RunConfig().workload,
                                                  warm_turn_tokens=99))
    ctx_c = ctx_with(cfg=other, preds=preds, probes={"sample"}, ep=a)
    ctx_c._cache = ctx_a._cache
    ctx_c.freeze()
    assert asyncio.run(ctx_c.sample()) is None


def test_prefixes_follow_the_workload():
    ctx = ctx_with()
    first = ctx.prefixes
    assert ctx.prefixes is first                    # cached
    ctx.cfg = replace(ctx.cfg, workload=replace(ctx.cfg.workload,
                                                system_prefix_tokens=99))
    assert ctx.prefixes is not first
    assert len(ctx.prefixes.user) != len(first.user)


def test_burst_standing_load_has_one_definition():
    """test_cmd's dry-run printed its own copy of the rule, which disagreed
    with the run whenever operating_point_users was 0."""
    preds = replace(predict(RunConfig(), n_iter=40), operating_point_users=0,
                    predicted_limit_users=80)
    ctx = RunContext(RunConfig(), preds, ProbeOptions(), EndpointSpec(),
                     burst=4)
    assert ctx._burst_pop() == 40                   # half the limit, not 1
    ctx2 = RunContext(RunConfig(), preds, ProbeOptions(), EndpointSpec(),
                      burst=4, burst_users=7)
    assert ctx2._burst_pop() == 7


# ============================================================================
# predictions the hypotheses needed
# ============================================================================
def test_steady_block_is_present_and_ordered():
    p = predict(RunConfig(), n_iter=40)
    assert p.steady_decode_seqs > 0 and p.steady_decode_tok_s > 0
    assert p.itl_normal_ms > 0
    # higher MFU = shorter freeze, so lo < point < hi
    assert p.itl_freeze_lo_ms < p.itl_worst_freeze_ms < p.itl_freeze_hi_ms


def test_freeze_ms_matches_the_explorer_formula():
    from workingset import model as M
    from workingset.predict import freeze_ms
    cfg = RunConfig()
    m, t = cfg.to_model(), cfg.to_topology()
    cap, chunk, pu = cfg.deployment.max_model_len, 4_096, 800.0
    step = min(chunk, cap)
    want = 1e3 * (m.mtp / pu
                  + M.prefill_seconds(m, t, step, M.MFU_DEFAULT, prior=cap - step))
    assert freeze_ms(m, t, cap, chunk, pu) == pytest.approx(want)
    # a chunk bigger than the context is one cap-sized pass, no negative prior
    assert freeze_ms(m, t, 1_000, 32_768, pu) == pytest.approx(
        1e3 * (m.mtp / pu + M.prefill_seconds(m, t, 1_000, M.MFU_DEFAULT,
                                              prior=0)))


def test_freeze_reproduces_the_reference_harness_block():
    """The explorer's reference CONFIG (27B / 4xH200 TP4 / chunk 4,096) shipped
    itl_normal_ms 2.2 and itl_worst_freeze_ms 171 — a prefill term of 168.8 ms
    behind one chunk of a 180k cold re-prefill.

    Only the PREFILL term is pinned. The mtp/pu term rides on the decode
    calibration, which moved (MBU 0.22, the 27B's measured mtp) after that
    block was generated: steady_decode_tok_s is 285 here where the reference
    said 817, exactly as decode_ceiling_users is 150 where it said 428. The
    prefill half is the part this port could get wrong, so that is what fails
    the test."""
    cfg = RunConfig.from_dict({"deployment": {"model": "27B", "gpu": "H200",
                                              "tensor_parallel": 4,
                                              "max_num_batched_tokens": 4096}})
    p = predict(cfg, n_iter=300)
    assert abs((p.itl_worst_freeze_ms - p.itl_normal_ms) - 168.8) < 1.0
    # the bracket straddles it, low edge = the HIGH MFU anchor
    assert p.itl_freeze_lo_ms < p.itl_worst_freeze_ms < p.itl_freeze_hi_ms


def test_steady_point_honours_the_calibration_mbu():
    """max_users_decode reads cfg.calibration.mbu; steady_decode_point stayed
    at the study default, so the decode ceiling and the steady point were
    priced at different decode efficiencies in the same run."""
    base = {"deployment": {"model": "27B", "gpu": "H200", "tensor_parallel": 4}}
    lo = predict(RunConfig.from_dict({**base, "calibration": {"mbu": 0.1}}),
                 n_iter=60)
    hi = predict(RunConfig.from_dict({**base, "calibration": {"mbu": 0.8}}),
                 n_iter=60)
    assert hi.decode_ceiling_users > lo.decode_ceiling_users
    # the steady point must move WITH it, not stay put
    assert hi.steady_decode_tok_s > 2 * lo.steady_decode_tok_s
    # and the normal gap, which is one decode step at that speed, with it
    assert hi.itl_normal_ms < lo.itl_normal_ms


def test_itl_normal_is_one_decode_step():
    p = predict(RunConfig(), n_iter=40)
    m = RunConfig().to_model()
    assert p.itl_normal_ms == pytest.approx(
        round(1e3 * m.mtp / p.steady_decode_tok_s, 1), abs=0.6)


# ============================================================================
# run record
# ============================================================================
def make_record() -> RunRecord:
    cfg = RunConfig()
    preds = predict(cfg, n_iter=40)
    opts = ProbeOptions(measure_s=5.0)
    h = REGISTRY.get("H-ttft-miss")
    t = trace(1, "hit", 1.0, 0.4, tps=120, ctok=10, ptok=100, intended=100)
    s = sample_with(ttft_miss_mean=preds.ttft_miss_s, ttft_miss_p50=1.0,
                    itl_p50_ms=preds.itl_normal_ms, itl_floor_ms=0.01,
                    decode_p50=preds.steady_decode_tok_s,
                    chunk_tok_ratio=1.0, ptok_ratio=0.98, traces=[t])
    ctx = shared_ctx(preds, s, cfg=cfg, opts=opts)
    p, m, v = score(h, ctx)
    pl = plan(REGISTRY.all(), exclusive=False)
    r = rung(31, passed=True, ttft_hit_p50=0.4, ttft_hit_pX=0.9,
             ttft_miss_p50=2.0, ttft_miss_pX=3.0, decode_p50=120.0,
             achieved_rps=1.9, offered_rps=2.1, itl_p50_ms=8.0,
             itl_worst_p50_ms=200.0, itl_worst_p95_ms=800.0,
             itl_max_ms=900.0, itl_floor_ms=0.2, freeze_per_ktok=2.0,
             stall_ms_per_ktok=40.0, stall_frac=0.04, chunk_tok_ratio=1.0,
             ptok_ratio=0.98,
             freeze_ladder=[{"threshold_ms": t, "per_ktok": 1.0,
                             "stall_ms_per_ktok": 3.0}
                            for t in (50.0, 100.0, 250.0, 500.0, 1000.0)])
    return RunRecord.new(
        "0.1.0", mode="shared", config=cfg.to_dict(),
        predictions=preds.to_dict(), options=opts.to_dict(),
        endpoint=EndpointSpec().redacted(), plan=pl.to_dict(),
        rungs=[r.to_dict()], sample=s.to_dict(),
        hypotheses=[{"key": h.key, "title": h.title,
                     "requires": sorted(h.requires),
                     "probes": sorted(h.conditional_probes(pl.probes)),
                     "statement": h.statement(cfg, preds),
                     "prediction": p.to_dict(), "measurement": m.to_dict(),
                     "verdict": v.to_dict()}],
        skipped=[{**hh.describe(), "reason": rr} for hh, rr in pl.skipped],
        not_established=not_established_notes(
            cfg, opts, pl, rungs=[r.to_dict()], sample=s.to_dict(),
            burst=None, hypotheses=[{"key": h.key,
                                     "verdict": v.to_dict()}],
            exclusive=False, metrics=False),
        measured_capacity_bracket=[31, None])


def test_run_record_round_trips(tmp_path):
    rec = make_record()
    p = rec.save(tmp_path / "run.json")
    back = RunRecord.load(p)
    assert back.to_dict() == json.loads(rec.dumps())
    assert back.hypotheses[0]["verdict"]["status"] in STATUSES
    assert back.measured_capacity_bracket == [31, None]


def test_run_record_rejects_a_foreign_schema(tmp_path):
    p = tmp_path / "run.json"
    p.write_text(json.dumps({"schema_version": 99}))
    with pytest.raises(ValueError, match="schema_version"):
        RunRecord.load(p)
    with pytest.raises(ValueError, match="unknown run-record keys"):
        RunRecord.from_dict({"nope": 1})


def test_non_finite_numbers_survive_the_json_round_trip(tmp_path):
    rec = make_record()
    rec.rungs[0]["decode_p50"] = float("nan")
    back = RunRecord.load(rec.save(tmp_path / "r.json"))
    assert back.rungs[0]["decode_p50"] is None


def test_not_established_names_the_untested():
    cfg, opts = RunConfig(), ProbeOptions()
    pl = plan(REGISTRY.all(), exclusive=False)
    notes = not_established_notes(cfg, opts, pl, exclusive=False,
                                  metrics=False)
    joined = "\n".join(notes)
    assert "No ladder was run" in joined
    assert "Shared mode" in joined
    assert "B*" in joined
    assert "H-cache was not tested" in joined


def test_trailer_is_built_from_results_not_from_the_plan():
    """A sample was PLANNED and every request failed. The old trailer said the
    run "establishes levels and gap distributions"; it establishes nothing."""
    cfg, opts = RunConfig(), ProbeOptions()
    pl = plan(REGISTRY.all(), exclusive=False)
    dead = {"n": 8, "n_ok": 0, "n_err": 8, "traces": []}
    notes = "\n".join(not_established_notes(cfg, opts, pl, sample=dead,
                                            exclusive=False, metrics=False))
    assert "answered 0 of 8 requests" in notes
    assert "rest on nothing" in notes
    # and with no usage readback anywhere, the token accounting is called out
    assert "No `usage` readback" in notes

    live = {"n": 8, "n_ok": 8, "n_err": 0,
            "traces": [{"ptok_achieved": 1234}]}
    ok = "\n".join(not_established_notes(cfg, opts, pl, sample=live,
                                         exclusive=False, metrics=False))
    assert "rest on nothing" not in ok
    assert "No `usage` readback" not in ok
    assert "chars/4 approximations" in ok


def test_trailer_reports_a_ladder_that_measured_nothing():
    cfg, opts = RunConfig(), ProbeOptions()
    pl = plan(REGISTRY.all(), exclusive=True, burst=1)
    rungs = [{"pop": 31, "n_turns": 0, "passed": False, "traces": []}]
    notes = "\n".join(not_established_notes(cfg, opts, pl, rungs=rungs,
                                            exclusive=True, metrics=True))
    assert "none produced a measured turn" in notes
    assert "No ladder was run" not in notes


def test_trailer_lists_the_hypotheses_that_reached_no_conclusion():
    cfg, opts = RunConfig(), ProbeOptions()
    pl = plan(REGISTRY.all(), exclusive=False)
    rows = [{"key": "H-itl-mean", "verdict": {"status": "not_established"}},
            {"key": "H-ttft-miss", "verdict": {"status": "supported"}}]
    notes = "\n".join(not_established_notes(cfg, opts, pl, hypotheses=rows,
                                            exclusive=False, metrics=False))
    assert "Not established by this run" in notes
    assert "H-itl-mean" in notes.split("Not established by this run")[1]


def test_degenerate_subagent_leg_is_called_out():
    cfg = RunConfig()
    cfg = replace(cfg, workload=replace(cfg.workload,
                                        subagent_prefix_tokens=9_000,
                                        subagent_median_tokens=8_000))
    notes = not_established_notes(cfg, ProbeOptions(),
                                  plan(REGISTRY.all(), exclusive=True),
                                  rungs=[{"n_turns": 3, "traces": []}],
                                  burst={"n": 4}, exclusive=True, metrics=True)
    assert any("DEGENERATE" in n for n in notes)


# ============================================================================
# report + CLI
# ============================================================================
def test_report_prints_the_tables_and_the_trailer(capsys):
    print_report(make_record())
    out = capsys.readouterr().out
    assert "VALIDATION REPORT" in out
    assert "INTER-TOKEN GAPS" in out
    assert "FREEZE LADDER" in out
    assert "MEASURED SLO CAPACITY" in out
    assert "PREDICTED vs MEASURED" in out
    assert "WHAT THIS RUN DOES NOT ESTABLISH" in out
    assert "H-ttft-miss" in out
    assert "skipped" in out


def test_ws_report_reads_a_record(tmp_path, capsys):
    p = make_record().save(tmp_path / "run.json")
    assert cli.main(["report", str(p)]) == 0
    assert "MEASURED SLO CAPACITY" in capsys.readouterr().out


def test_ws_test_dry_run_on_the_default_config(tmp_path, capsys):
    cfg = tmp_path / "workingset.toml"
    cfg.write_text(RunConfig().dumps("toml"))
    assert cli.main(["test", str(cfg), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "no requests sent" in out
    assert "mode     : shared" in out
    assert "sampler self-check PASSED" in out
    assert "SKIPPED (6)" in out
    assert "LOAD LADDER" not in out          # no ladder in shared mode


def test_ws_test_all_dry_run_exclusive(tmp_path, capsys):
    cfg = tmp_path / "workingset.toml"
    cfg.write_text(RunConfig().dumps("toml"))
    assert cli.main(["test", str(cfg), "--all", "--exclusive", "--burst", "8",
                     "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "LOAD LADDER" in out and "BURST PROBE" in out
    assert "HYPOTHESES SELECTED (10)" in out
    assert "SKIPPED" not in out


def test_ws_test_dry_run_honours_overrides(tmp_path, capsys):
    cfg = tmp_path / "workingset.toml"
    cfg.write_text(RunConfig().dumps("toml"))
    assert cli.main(["test", str(cfg), "H-itl-mean", "--dry-run",
                     "--base-url", "http://gpu:9000/v1", "--api", "chat",
                     "--ramp-s", "5", "--measure-s", "7"]) == 0
    out = capsys.readouterr().out
    assert "http://gpu:9000/v1" in out and "api=chat" in out
    assert "HYPOTHESES SELECTED (1)" in out


def test_an_interrupt_keeps_the_verdicts_already_computed():
    """The harness printed its PREDICTED vs MEASURED block over whatever had
    completed when Ctrl-C arrived. Discarding the rows threw away the only
    output a long ladder run had produced."""
    from workingset.test_cmd import _score, _score_from_cache

    preds = predict(RunConfig(), n_iter=40)
    op = preds.operating_point_users
    r = rung(op, n_miss=5, ttft_miss_mean=preds.ttft_miss_s,
             itl_p50_ms=preds.itl_normal_ms, itl_floor_ms=0.0001,
             chunk_tok_ratio=1.0)
    ctx = ladder_ctx(preds, [r])

    class Boom(Hypothesis):
        key, title, requires = "H-boom", "explodes", frozenset()

        def predict(self, cfg, p):
            return Prediction(value=1.0)

        async def measure(self, ctx):
            raise KeyboardInterrupt

        def verdict(self, pred, m):
            return Verdict(SUPPORTED, "never reached")

    pl = plan([REGISTRY.get("H-ttft-miss"), Boom(),
               REGISTRY.get("H-itl-mean")], exclusive=True)
    rows: list = []

    async def go():
        try:
            await _score(ctx, pl, rows)
        except KeyboardInterrupt:
            rows.extend(await _score_from_cache(ctx, pl,
                                                {x["key"] for x in rows}))
    asyncio.run(go())

    by_key = {x["key"]: x["verdict"]["status"] for x in rows}
    # the one that completed BEFORE the interrupt keeps its verdict
    assert by_key["H-ttft-miss"] == SUPPORTED
    # the ones after it are scored from the cache — the completed rung is
    # still evidence, so H-itl-mean gets a real verdict too
    assert by_key["H-itl-mean"] == SUPPORTED
    assert "H-boom" in by_key


def test_a_frozen_context_starts_no_new_probe():
    preds = predict(RunConfig(), n_iter=40)
    ctx = shared_ctx(preds)          # nothing seeded
    ctx.freeze()
    _, m, v = score(REGISTRY.get("H-ttft-miss"), ctx)
    assert m.text == "not separable" and v.status == NOT_ESTABLISHED
    assert ctx.cached()["sample"] is None


def test_ws_hypotheses_lists_requirements(capsys):
    assert cli.main(["hypotheses"]) == 0
    out = capsys.readouterr().out
    for h in REGISTRY:
        assert h.key in out
    assert "burst,exclusive" in out
