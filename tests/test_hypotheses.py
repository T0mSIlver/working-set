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
                                   STATUSES, SUPPORTED, Hypothesis, Registry,
                                   RunContext, Verdict, bracket_verdict, plan)
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
    r = Rung(pop=pop, n_sub=0, n_turns=10, n_hit=8, n_miss=2)
    r.passed, r.reasons = passed, list(reasons)
    for k, v in kw.items():
        setattr(r, k, v)
    return r


def ctx_with(cfg=None, preds=None, opts=None, exclusive=False, burst=0,
             run_ladder=False, **seed) -> RunContext:
    cfg = cfg or RunConfig()
    preds = preds if preds is not None else predict(cfg, n_iter=40)
    c = RunContext(cfg, preds, opts or ProbeOptions(),
                   EndpointSpec(), exclusive=exclusive, burst=burst,
                   run_ladder=run_ladder)
    return c.seed(**seed)


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
    # the ladder is not run, so the cheap ones fall back to their own sample
    assert p.probes == {"sample"} and not p.run_ladder


def test_exclusive_mode_runs_the_ladder_once_for_all_of_them():
    p = plan(REGISTRY.all(), exclusive=True, metrics=False, burst=0)
    assert p.run_ladder and "sample" not in p.probes
    assert [h.key for h, _ in p.skipped] == ["H-burst"]
    assert "--burst" in dict((h.key, r) for h, r in p.skipped)["H-burst"]


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
    p, m, v = score(REGISTRY.get("H-binding"),
                    ctx_with(preds=preds, exclusive=True, run_ladder=True,
                             rungs=ladder))
    assert m.lo == lim - 5 and m.hi == lim + 5
    assert v.status == SUPPORTED


def test_h_decode_is_not_separable_without_a_floor_failure():
    h = REGISTRY.get("H-decode")
    ladder = [rung(10), rung(20, passed=False,
                             reasons=["p95 TTFT 12.00s > 10s"])]
    _, m, v = score(h, ctx_with(exclusive=True, run_ladder=True, rungs=ladder))
    assert m.text == "not separable"
    assert v.status == NOT_ESTABLISHED and "decode-floor" in v.text


def test_h_decode_brackets_on_a_floor_failure():
    preds = predict(RunConfig(), n_iter=40)
    dc = preds.decode_ceiling_users
    ladder = [rung(dc - 3), rung(dc + 3, passed=False,
                                 reasons=["decode p50 11.0 < 40 tok/s"])]
    _, m, v = score(REGISTRY.get("H-decode"),
                    ctx_with(preds=preds, exclusive=True, run_ladder=True,
                             rungs=ladder))
    assert (m.lo, m.hi) == (dc - 3, dc + 3) and v.status == SUPPORTED


def test_h_latency_brackets_on_a_ttft_failure():
    preds = predict(RunConfig(), n_iter=40)
    lc = preds.latency_ceiling_users
    ladder = [rung(lc - 2), rung(lc + 2, passed=False,
                                 reasons=["p95 TTFT 12.00s > 10s"])]
    _, m, v = score(REGISTRY.get("H-latency"),
                    ctx_with(preds=preds, exclusive=True, run_ladder=True,
                             rungs=ladder))
    assert (m.lo, m.hi) == (lc - 2, lc + 2) and v.status == SUPPORTED


def test_h_cache_is_bounded_below_when_nothing_evicts():
    preds = predict(RunConfig(), n_iter=40)
    wc = preds.warm_capacity_p5
    ladder = [rung(wc + 10, evict_frac=0.01)]
    _, m, v = score(REGISTRY.get("H-cache"),
                    ctx_with(preds=preds, exclusive=True, run_ladder=True,
                             rungs=ladder))
    assert m.lo == wc + 10 and m.hi is None
    assert v.status == BOUNDED_BELOW and "lower bound" in v.text


def test_h_cache_is_refuted_when_eviction_arrives_far_early():
    preds = predict(RunConfig(), n_iter=40)
    wc = preds.warm_capacity_p5
    ladder = [rung(max(1, wc // 4), n_hit=8, evict_frac=0.4)]
    _, m, v = score(REGISTRY.get("H-cache"),
                    ctx_with(preds=preds, exclusive=True, run_ladder=True,
                             rungs=ladder))
    assert m.hi == max(1, wc // 4)
    assert v.status == REFUTED and "re-prefilled" in v.text


def test_h_saturation_never_rises_above_not_established():
    h = REGISTRY.get("H-saturation")
    plateau = rung(100, passed=False, reasons=["p95 TTFT 20.00s > 10s"],
                   achieved_rps=0.5, offered_rps=3.3)
    _, m, v = score(h, ctx_with(exclusive=True, run_ladder=True,
                                rungs=[plateau]))
    assert v.status == NOT_ESTABLISHED and "throughput plateau" in v.text
    _, _, v2 = score(h, ctx_with(exclusive=True, run_ladder=True,
                                 rungs=[rung(100)]))
    assert v2.status == NOT_ESTABLISHED and "throttles" in v2.text


# ============================================================================
# the cheap hypotheses
# ============================================================================
def sample_with(**kw) -> Sample:
    s = Sample(n=4, n_ok=4)
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def test_h_ttft_miss_reads_the_rung_nearest_the_operating_point():
    preds = predict(RunConfig(), n_iter=40)
    op, tm = preds.operating_point_users, preds.ttft_miss_s
    ladder = [rung(op, n_miss=5, ttft_miss_mean=tm),
              rung(op * 4, n_miss=5, ttft_miss_mean=tm * 9)]
    _, m, v = score(REGISTRY.get("H-ttft-miss"),
                    ctx_with(preds=preds, exclusive=True, run_ladder=True,
                             rungs=ladder))
    assert m.data["at_users"] == op
    assert v.status == SUPPORTED and "operating point" in v.text


def test_h_ttft_miss_falls_back_to_the_sample_in_shared_mode():
    preds = predict(RunConfig(), n_iter=40)
    s = sample_with(ttft_miss_mean=preds.ttft_miss_s, ttft_miss_p50=1.0)
    _, m, v = score(REGISTRY.get("H-ttft-miss"),
                    ctx_with(preds=preds, sample=s))
    assert m.data["source"] == "sample"
    assert v.status == SUPPORTED and "prevailing load" in v.text


def test_h_ttft_miss_refuted_far_from_the_prediction():
    preds = predict(RunConfig(), n_iter=40)
    s = sample_with(ttft_miss_mean=preds.ttft_miss_s * 20, ttft_miss_p50=1.0)
    _, _, v = score(REGISTRY.get("H-ttft-miss"), ctx_with(preds=preds, sample=s))
    assert v.status == REFUTED


def test_h_ttft_miss_not_established_without_misses():
    preds = predict(RunConfig(), n_iter=40)
    _, m, v = score(REGISTRY.get("H-ttft-miss"),
                    ctx_with(preds=preds, sample=sample_with()))
    assert m.text == "not separable" and v.status == NOT_ESTABLISHED


def test_h_steady_without_metrics_leaves_the_batch_size_open():
    preds = predict(RunConfig(), n_iter=40)
    s = sample_with(decode_p50=preds.steady_decode_tok_s)
    _, m, v = score(REGISTRY.get("H-steady"), ctx_with(preds=preds, sample=s))
    assert not math.isfinite(m.data["seqs"])
    assert v.status == SUPPORTED and "not established" in v.text


def test_h_steady_reads_the_decode_batch_from_metrics_covariates():
    preds = predict(RunConfig(), n_iter=40)
    seqs = preds.steady_decode_seqs
    t = trace(1, "hit", 1.0, 0.1, tps=preds.steady_decode_tok_s, ctok=10)
    t.covariates = {"requests_running": seqs}
    s = sample_with(decode_p50=preds.steady_decode_tok_s, traces=[t])
    _, m, v = score(REGISTRY.get("H-steady"), ctx_with(preds=preds, sample=s))
    assert m.data["seqs"] == pytest.approx(seqs)
    assert v.status == SUPPORTED and "batch" in v.text


def test_h_steady_takes_the_weaker_of_the_two_halves():
    preds = predict(RunConfig(), n_iter=40)
    t = trace(1, "hit", 1.0, 0.1, tps=1, ctok=10)
    t.covariates = {"requests_running": preds.steady_decode_seqs * 50}
    s = sample_with(decode_p50=preds.steady_decode_tok_s, traces=[t])
    _, _, v = score(REGISTRY.get("H-steady"), ctx_with(preds=preds, sample=s))
    assert v.status == REFUTED


def test_h_itl_mean_guards_on_the_client_floor():
    preds = predict(RunConfig(), n_iter=40)
    normal = preds.itl_normal_ms
    good = sample_with(itl_p50_ms=normal, itl_floor_ms=normal / 100,
                       chunk_tok_ratio=1.0)
    _, _, v = score(REGISTRY.get("H-itl-mean"), ctx_with(preds=preds, sample=good))
    assert v.status == SUPPORTED

    floored = sample_with(itl_p50_ms=normal, itl_floor_ms=normal * 0.9,
                          chunk_tok_ratio=1.0)
    _, _, v = score(REGISTRY.get("H-itl-mean"),
                    ctx_with(preds=preds, sample=floored))
    assert v.status == NOT_ESTABLISHED and "client floor" in v.text


def test_h_itl_mean_refuses_a_multi_token_event_comparison():
    preds = predict(RunConfig(), n_iter=40)
    s = sample_with(itl_p50_ms=preds.itl_normal_ms, itl_floor_ms=0.001,
                    chunk_tok_ratio=4.0)
    _, _, v = score(REGISTRY.get("H-itl-mean"), ctx_with(preds=preds, sample=s))
    assert v.status == NOT_ESTABLISHED and "per SSE event" in v.text


def test_h_itl_mean_refuted_when_the_gap_is_nowhere_near():
    preds = predict(RunConfig(), n_iter=40)
    s = sample_with(itl_p50_ms=preds.itl_normal_ms * 10, itl_floor_ms=0.001,
                    chunk_tok_ratio=1.0)
    _, _, v = score(REGISTRY.get("H-itl-mean"), ctx_with(preds=preds, sample=s))
    assert v.status == REFUTED


def test_h_itl_spike_scores_against_the_mfu_bracket():
    preds = predict(RunConfig(), n_iter=40)
    h = REGISTRY.get("H-itl-spike")
    inside = sample_with(itl_worst_max_ms=preds.itl_worst_freeze_ms,
                         itl_worst_p50_ms=preds.itl_worst_freeze_ms,
                         itl_p50_ms=preds.itl_normal_ms, itl_floor_ms=0.001,
                         itl_max_ms=preds.itl_worst_freeze_ms)
    _, _, v = score(h, ctx_with(preds=preds, sample=inside))
    assert v.status == SUPPORTED and "MFU bracket" in v.text

    tiny = sample_with(itl_worst_max_ms=1.0, itl_worst_p50_ms=1.0,
                       itl_p50_ms=preds.itl_normal_ms, itl_floor_ms=0.001)
    _, _, v = score(h, ctx_with(preds=preds, sample=tiny))
    assert v.status == REFUTED and "below" in v.text

    edge = sample_with(itl_worst_max_ms=preds.itl_freeze_lo_ms * 0.8,
                       itl_worst_p50_ms=1.0, itl_p50_ms=preds.itl_normal_ms,
                       itl_floor_ms=0.001)
    _, _, v = score(h, ctx_with(preds=preds, sample=edge))
    assert v.status == NOT_ESTABLISHED


def test_h_itl_spike_prefers_the_burst_probe():
    preds = predict(RunConfig(), n_iter=40)
    b = BurstResult(n=4, standing_users=8, n_ok=4, last_ttft_s=1.0,
                    standing_n=3,
                    standing_worst_max_ms=preds.itl_worst_freeze_ms,
                    standing_worst_p50_ms=preds.itl_worst_freeze_ms,
                    standing_itl_p50_ms=preds.itl_normal_ms,
                    standing_floor_ms=0.001)
    _, m, v = score(REGISTRY.get("H-itl-spike"), _burst_ctx(preds, b))
    assert m.data["source"] == "burst" and v.status == SUPPORTED


def _burst_ctx(preds, b) -> RunContext:
    c = RunContext(RunConfig(), preds, ProbeOptions(), EndpointSpec(),
                   exclusive=True, burst=b.n, burst_users=b.standing_users)
    return c.seed(burst=b)


def test_h_steady_not_established_without_a_steady_point():
    preds = replace(predict(RunConfig(), n_iter=40), steady_decode_tok_s=None,
                    steady_decode_seqs=None, itl_normal_ms=None,
                    itl_worst_freeze_ms=None, itl_freeze_lo_ms=None,
                    itl_freeze_hi_ms=None)
    for key in ("H-steady", "H-itl-spike", "H-itl-mean"):
        _, _, v = score(REGISTRY.get(key),
                        ctx_with(preds=preds, sample=sample_with()))
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
    s = sample_with(ttft_miss_mean=preds.ttft_miss_s, ttft_miss_p50=1.0,
                    itl_p50_ms=preds.itl_normal_ms, itl_floor_ms=0.01,
                    decode_p50=preds.steady_decode_tok_s,
                    chunk_tok_ratio=1.0, ptok_ratio=0.98)
    ctx = ctx_with(cfg=cfg, preds=preds, opts=opts, sample=s)
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
                     "statement": h.statement(cfg, preds),
                     "prediction": p.to_dict(), "measurement": m.to_dict(),
                     "verdict": v.to_dict()}],
        skipped=[{**hh.describe(), "reason": rr} for hh, rr in pl.skipped],
        not_established=not_established_notes(cfg, opts, pl, ran_ladder=True,
                                              ran_burst=False,
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


def test_not_established_names_the_untested(tmp_path):
    cfg, opts = RunConfig(), ProbeOptions()
    pl = plan(REGISTRY.all(), exclusive=False)
    notes = not_established_notes(cfg, opts, pl, ran_ladder=False,
                                  ran_burst=False, exclusive=False,
                                  metrics=False)
    joined = "\n".join(notes)
    assert "No ladder was run" in joined
    assert "Shared mode" in joined
    assert "B*" in joined
    assert "H-cache was not tested" in joined


def test_degenerate_subagent_leg_is_called_out():
    cfg = RunConfig()
    cfg = replace(cfg, workload=replace(cfg.workload,
                                        subagent_prefix_tokens=9_000,
                                        subagent_median_tokens=8_000))
    notes = not_established_notes(cfg, ProbeOptions(),
                                  plan(REGISTRY.all(), exclusive=True),
                                  ran_ladder=True, ran_burst=True,
                                  exclusive=True, metrics=True)
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


def test_ws_hypotheses_lists_requirements(capsys):
    assert cli.main(["hypotheses"]) == 0
    out = capsys.readouterr().out
    for h in REGISTRY:
        assert h.key in out
    assert "burst,exclusive" in out
