"""Tests for `workingset.metrics`.

The fixture (`tests/fixtures/vllm_metrics_v1.txt`) uses the metric names,
labels and bucket boundaries read out of vLLM's own source at v0.25.1 and at
a post-0.28 main -- the two agree exactly -- so a rename upstream should
break these tests rather than silently produce Nones in the field.
"""
from __future__ import annotations

import asyncio
import json
import math
import time
from pathlib import Path

import httpx
import pytest

from workingset.cli import main as ws_main
from workingset.metrics import (SEMANTIC_KEYS, GaugeStats, Histogram,
                                HistogramMismatch, HistogramReset,
                                MetricsSampler, Sample, SamplerStopped,
                                Snapshot, VLLMAdapter, WindowNotCovered,
                                adapter_for, detect_adapter, load_jsonl,
                                parse_families, parse_text, parse_types,
                                window_from_snapshots)
from workingset.metrics.adapter import MetricsAdapter

FIXTURE = Path(__file__).parent / "fixtures" / "vllm_metrics_v1.txt"
DUMP = FIXTURE.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------
def test_parses_a_real_dump():
    samples = parse_text(DUMP)
    assert len(samples) > 150
    names = {s.name for s in samples}
    # a plain counter, a gauge, and all three histogram members
    assert "vllm:prompt_tokens_total" in names
    assert "vllm:num_requests_running" in names
    for suffix in ("_bucket", "_sum", "_count"):
        assert f"vllm:time_to_first_token_seconds{suffix}" in names
    # comments never become samples
    assert not any(s.name.startswith("#") for s in samples)


def test_labels_and_escapes():
    by_name = [s for s in parse_text(DUMP) if s.name == "vllm:request_success_total"]
    reasons = {s.labels["finished_reason"] for s in by_name}
    assert {"stop", "length", "abort", "error", "repetition"} <= reasons
    # the escaped label value comes back unescaped, exactly once
    assert 'stop\\ "quoted"' in reasons
    for s in by_name:
        assert s.labels["model_name"] == "Qwen/Qwen3-27B"
        assert s.labels["engine"] == "0"


def test_nan_and_inf_survive():
    nan = [s for s in parse_text(DUMP) if s.name == "vllm:mm_cache_hit_rate"]
    assert len(nan) == 1 and math.isnan(nan[0].value)
    extra = parse_text('a_metric{x="1"} +Inf\nb_metric -Inf\nc_metric 1e3\n')
    assert [s.value for s in extra] == [math.inf, -math.inf, 1000.0]


def test_malformed_lines_are_skipped_not_fatal():
    text = ("good_metric 1.0\n"
            "this is not prometheus\n"
            "also_bad{unterminated 3\n"
            "\n"
            "another_good{a=\"b\"} 2.0\n")
    got = {s.name: s.value for s in parse_text(text)}
    assert got == {"good_metric": 1.0, "another_good": 2.0}


def test_types_are_read():
    types = parse_types(DUMP)
    assert types["vllm:iteration_tokens_total"] == "histogram"
    assert types["vllm:num_requests_running"] == "gauge"
    assert types["vllm:prompt_tokens"] == "counter"


def test_histogram_family_grouping_respects_type():
    """`vllm:iteration_tokens_total_count` is a HISTOGRAM member whose base
    ends in `_total`; `vllm:prompt_tokens_total` is a counter that merely
    looks similar. The two must not be confused."""
    fams = parse_families(DUMP)
    it = fams["vllm:iteration_tokens_total"]
    assert it.type == "histogram"
    assert {s.name for s in it.samples} >= {
        "vllm:iteration_tokens_total_bucket",
        "vllm:iteration_tokens_total_sum",
        "vllm:iteration_tokens_total_count"}
    assert "vllm:prompt_tokens_total" in fams
    assert fams["vllm:prompt_tokens_total"].type in ("counter", "untyped")


def _sum_bounds(h):
    """What a cumulative histogram's `_sum` is allowed to be.

    Every observation in bucket (lower, upper] contributes at least `lower`
    and at most `upper`, so
        SUM_i count_i * lower_i  <=  sum  <=  SUM_i count_i * upper_i
    with the +Inf bucket removing the upper bound. Violating the lower bound
    is not a rounding slip: it means the buckets and the sum describe
    different data, which is how the first version of this fixture claimed a
    5.2M token sum under buckets needing at least 14.9M.
    """
    lo = hi = 0.0
    prev_b, prev_c = 0.0, 0.0
    unbounded = False
    for b in h.bounds:
        c = h.buckets[b] - prev_c
        lo += c * prev_b
        if math.isinf(b):
            unbounded = unbounded or c > 0
        else:
            hi += c * b
        prev_b, prev_c = b, h.buckets[b]
    return lo, (math.inf if unbounded else hi)


HIST_KEYS = ("ttft_hist", "tpot_hist", "request_tpot_hist", "e2e_hist",
             "prefill_time_hist", "decode_time_hist", "queue_time_hist",
             "inference_time_hist", "iteration_tokens_hist",
             "request_prompt_tokens_hist", "request_generation_tokens_hist")


def test_every_fixture_histogram_has_a_feasible_sum():
    ad = adapter_for(DUMP)
    s = parse_text(DUMP)
    for key in HIST_KEYS:
        h = ad.histogram(s, key)
        assert h is not None, key
        lo, hi = _sum_bounds(h)
        assert lo <= h.sum <= hi, (
            f"{key}: _sum={h.sum:,.3f} outside the feasible "
            f"[{lo:,.3f}, {hi:,.3f}] implied by its own buckets")


def test_fixture_histograms_are_well_formed():
    ad = adapter_for(DUMP)
    s = parse_text(DUMP)
    for key in HIST_KEYS:
        h = ad.histogram(s, key)
        assert h.buckets[math.inf] == h.count, key
        assert all(h.buckets[a] <= h.buckets[b]
                   for a, b in zip(h.bounds, h.bounds[1:])), key


def test_fixture_cross_family_identities():
    """The identities a real engine would satisfy, asserted rather than
    trusted -- the fixture is generated, and a regeneration must keep them."""
    ad = adapter_for(DUMP)
    s = parse_text(DUMP)
    prompt = ad.counter(s, "prompt_tokens_total")
    gen = ad.counter(s, "generation_tokens_total")
    cached = ad.counter(s, "prompt_tokens_cached_total")
    by_source = {x.labels["source"]: x.value for x in s
                 if x.name == "vllm:prompt_tokens_by_source_total"}
    assert sum(by_source.values()) == pytest.approx(prompt)
    assert by_source["local_cache_hit"] == pytest.approx(cached)
    assert ad.counter(s, "prefix_cache_queries_total") == pytest.approx(prompt)
    assert ad.counter(s, "prefix_cache_hits_total") == pytest.approx(cached)
    # the per-request size histograms total the counters
    assert ad.histogram(s, "request_prompt_tokens_hist").sum == pytest.approx(prompt)
    assert ad.histogram(s, "request_generation_tokens_hist").sum == pytest.approx(gen)
    # an engine step processes uncached prompt tokens plus generated tokens
    assert ad.histogram(s, "iteration_tokens_hist").sum == pytest.approx(
        by_source["local_compute"] + gen)
    # phase accounting: queue + inference = e2e, prefill + decode = inference
    pre = ad.histogram(s, "prefill_time_hist").sum
    dec = ad.histogram(s, "decode_time_hist").sum
    inf = ad.histogram(s, "inference_time_hist").sum
    que = ad.histogram(s, "queue_time_hist").sum
    assert pre + dec == pytest.approx(inf)
    assert que + inf == pytest.approx(ad.histogram(s, "e2e_hist").sum)
    # spec decode: per-position acceptance sums to the total, and one verify
    # step per output event means drafts + accepted = generated tokens
    pos = ad.by_position(s, "spec_decode_accepted_per_pos")
    accepted = ad.counter(s, "spec_decode_num_accepted_tokens_total")
    drafts = ad.counter(s, "spec_decode_num_drafts_total")
    assert sum(pos.values()) == pytest.approx(accepted)
    assert drafts + accepted == pytest.approx(gen)


def test_itl_count_is_output_events_not_generated_tokens():
    """vLLM appends ONE inter-token latency per output EVENT (stats.py: the
    append is outside any per-token loop), so a spec-decode step accepting
    3 tokens records one observation, and the first output of a request
    books a TTFT instead. Asserting count == generation_tokens, as an
    earlier version of this test did, is wrong on both counts."""
    ad = adapter_for(DUMP)
    s = parse_text(DUMP)
    gen = ad.counter(s, "generation_tokens_total")
    events = ad.counter(s, "spec_decode_num_drafts_total")   # one per event
    requests = ad.counter(s, "request_success_total")
    itl = ad.histogram(s, "tpot_hist").observations
    assert itl == pytest.approx(events - requests)
    assert itl < gen                       # strictly fewer, under spec decode
    # the per-request mean is observed once per finished request instead
    assert ad.histogram(s, "request_tpot_hist").observations == pytest.approx(
        requests)
    assert ad.histogram(s, "e2e_hist").observations == pytest.approx(requests)


# ---------------------------------------------------------------------------
# histogram
# ---------------------------------------------------------------------------
def _hist(pairs, count=None, total=None):
    return Histogram("h", {}, dict(pairs), count, total)


def test_quantile_interpolates_inside_the_bucket():
    h = _hist({1.0: 0.0, 2.0: 10.0, math.inf: 10.0}, count=10.0)
    assert h.quantile(0.5) == pytest.approx(1.5)
    assert h.quantile(0.1) == pytest.approx(1.1)
    assert h.quantile(1.0) == pytest.approx(2.0)


def test_quantile_starts_from_zero_not_from_the_first_bound():
    """The first bucket's lower edge is 0, not its own `le`."""
    h = _hist({1.0: 5.0, 2.0: 10.0, math.inf: 10.0}, count=10.0)
    assert h.quantile(0.25) == pytest.approx(0.5)
    assert h.quantile(0.5) == pytest.approx(1.0)
    assert h.quantile(0.75) == pytest.approx(1.5)


def test_quantile_in_the_overflow_bucket_returns_the_largest_finite_bound():
    h = _hist({1.0: 5.0, math.inf: 10.0}, count=10.0)
    assert h.quantile(0.9) == pytest.approx(1.0)   # a lower bound, never inf


def test_quantile_of_an_empty_histogram_is_none():
    assert _hist({1.0: 0.0, math.inf: 0.0}, count=0.0).quantile(0.5) is None
    assert _hist({}).quantile(0.5) is None


def test_quantile_rejects_q_outside_zero_one():
    with pytest.raises(ValueError):
        _hist({1.0: 1.0}, count=1.0).quantile(1.5)


def test_mean_is_sum_over_count():
    h = _hist({1.0: 4.0, math.inf: 4.0}, count=4.0, total=6.0)
    assert h.mean() == pytest.approx(1.5)
    assert _hist({1.0: 4.0}, count=4.0).mean() is None    # no _sum exported


def test_fixture_ttft_quantiles_match_hand_computation():
    """The interpolation, recomputed by hand from the fixture's OWN buckets
    rather than against a frozen literal -- so regenerating the fixture
    cannot quietly invalidate the arithmetic this is here to check."""
    ad = adapter_for(DUMP)
    h = ad.histogram(parse_text(DUMP), "ttft_hist")
    assert h.observations == 1900.0
    assert h.mean() == pytest.approx(h.sum / 1900.0)

    target = 0.5 * h.observations
    lower, prev_c = 0.0, 0.0
    for b in h.bounds:                       # the bucket holding rank 950
        c = h.buckets[b]
        if c >= target:
            break
        lower, prev_c = b, c
    expected = lower + (target - prev_c) / (c - prev_c) * (b - lower)
    assert h.quantile(0.5) == pytest.approx(expected)
    # and it lands inside the bucket that holds the median
    assert lower <= h.quantile(0.5) <= b


def test_histogram_subtraction_is_bucketwise():
    a = _hist({1.0: 10.0, 2.0: 20.0, math.inf: 25.0}, count=25.0, total=30.0)
    b = _hist({1.0: 4.0, 2.0: 6.0, math.inf: 8.0}, count=8.0, total=9.0)
    d = a - b
    assert d.buckets == {1.0: 6.0, 2.0: 14.0, math.inf: 17.0}
    assert d.count == 17.0 and d.sum == 21.0


def test_histogram_subtraction_refuses_a_reset():
    """Going backwards is a restart, not a small negative to clamp away."""
    a = _hist({1.0: 10.0, math.inf: 25.0}, count=25.0, total=30.0)
    b = _hist({1.0: 4.0, math.inf: 8.0}, count=8.0, total=9.0)
    with pytest.raises(HistogramReset):
        _ = b - a
    # a reset visible only in _sum (buckets and count flat) is still caught
    x = _hist({1.0: 4.0, math.inf: 8.0}, count=8.0, total=9.0)
    y = _hist({1.0: 4.0, math.inf: 8.0}, count=8.0, total=2.0)
    with pytest.raises(HistogramReset):
        _ = y - x


def test_histogram_subtraction_refuses_a_changed_bucket_layout():
    """Zero-filling a boundary only one side exported invents observations:
    the delta would claim every observation fell in the smallest bucket."""
    full = _hist({1.0: 100.0, 2.0: 118.0, 8.0: 120.0, math.inf: 120.0},
                 count=120.0, total=182.0)
    truncated = _hist({1.0: 100.0, 2.0: 100.0}, count=100.0, total=50.0)
    with pytest.raises(HistogramMismatch):
        _ = full - truncated
    # and the nonsense it would otherwise have produced: 20 observations all
    # <= 2 s, whose mean of 6.6 s does not fit inside them
    naive_count = 120.0 - 100.0
    naive_mean = (182.0 - 50.0) / naive_count
    assert naive_mean > 2.0


# ---------------------------------------------------------------------------
# adapter
# ---------------------------------------------------------------------------
def test_adapter_resolves_the_required_keys_on_a_v1_dump():
    res = adapter_for(DUMP).resolution()
    assert res.engine == "vllm" and res.version_hint == "v1"
    required = ["prompt_tokens_total", "generation_tokens_total",
                "requests_running", "requests_waiting", "kv_cache_usage",
                "prefix_cache_queries_total", "prefix_cache_hits_total",
                "preemptions_total", "request_success_total",
                "ttft_hist", "tpot_hist", "e2e_hist", "prefill_time_hist",
                "decode_time_hist", "iteration_tokens_hist",
                "spec_decode_num_accepted_tokens_total",
                "spec_decode_num_draft_tokens_total",
                "spec_decode_accepted_per_pos", "estimated_flops_total"]
    assert not [k for k in required if k not in res.resolved]
    assert res.resolved["kv_cache_usage"] == "vllm:kv_cache_usage_perc"
    assert res.resolved["tpot_hist"] == "vllm:inter_token_latency_seconds"
    assert res.resolved["iteration_tokens_hist"] == "vllm:iteration_tokens_total"


def test_missing_keys_are_none_never_keyerror():
    ad = adapter_for(DUMP)
    samples = parse_text(DUMP)
    # V1 exports no per-pass timing histogram
    assert "forward_time_hist" in ad.resolution().missing
    assert ad.histogram(samples, "forward_time_hist") is None
    assert ad.counter(samples, "not_a_key_at_all") is None
    assert ad.by_position(samples, "not_a_key_at_all") is None
    for key in SEMANTIC_KEYS:
        ad.read(samples, key)             # must not raise for any key


def test_alias_resolution_prefers_the_newest_spelling():
    """A server exporting BOTH the old and the new spelling resolves to the
    new one; a server exporting only the old one still resolves."""
    both = parse_text('vllm:kv_cache_usage_perc{engine="0"} 0.5\n'
                      'vllm:gpu_cache_usage_perc{engine="0"} 0.5\n')
    assert VLLMAdapter(both).name_for("kv_cache_usage") == "vllm:kv_cache_usage_perc"
    old = parse_text('vllm:gpu_cache_usage_perc{engine="0"} 0.5\n'
                     'vllm:avg_generation_throughput_toks_per_s{engine="0"} 8.0\n')
    a = VLLMAdapter(old)
    assert a.name_for("kv_cache_usage") == "vllm:gpu_cache_usage_perc"
    assert a.resolution().version_hint == "v0"
    # a V0 server has no prefix-cache COUNTERS, so no window hit rate exists
    assert "prefix_cache_queries_total" in a.resolution().missing


def test_tpot_alias_falls_back_to_the_pre_rename_name():
    old = parse_text(
        'vllm:time_per_output_token_seconds_bucket{le="0.05"} 3\n'
        'vllm:time_per_output_token_seconds_bucket{le="+Inf"} 5\n'
        'vllm:time_per_output_token_seconds_count 5\n'
        'vllm:time_per_output_token_seconds_sum 0.2\n')
    a = VLLMAdapter(old)
    assert a.name_for("tpot_hist") == "vllm:time_per_output_token_seconds"
    assert a.histogram(old, "tpot_hist").observations == 5.0


def test_counters_sum_over_label_sets_and_engine_selects_one():
    ad = adapter_for(DUMP)
    samples = parse_text(DUMP)
    # 1834 + 61 + 4 + 0 + 0 + 1 (the escaped-label line)
    assert ad.counter(samples, "request_success_total") == 1900.0
    two = parse_text('vllm:prompt_tokens_total{engine="0"} 10\n'
                     'vllm:prompt_tokens_total{engine="1"} 25\n')
    assert VLLMAdapter(two).counter(two, "prompt_tokens_total") == 35.0
    assert VLLMAdapter(two, engine="1").counter(two, "prompt_tokens_total") == 25.0


def test_engine_selector_excludes_an_unlabelled_series():
    """Folding a series that carries no engine label into engine 0's total is
    the cross-engine mixing the selector exists to prevent."""
    mixed = parse_text('vllm:prompt_tokens_total{engine="0"} 10\n'
                       'vllm:prompt_tokens_total{engine="1"} 25\n'
                       'vllm:prompt_tokens_total 7\n')
    assert VLLMAdapter(mixed).counter(mixed, "prompt_tokens_total") == 42.0
    assert VLLMAdapter(mixed, engine="0").counter(
        mixed, "prompt_tokens_total") == 10.0
    assert VLLMAdapter(mixed, engine="1").counter(
        mixed, "prompt_tokens_total") == 25.0
    # a selector naming an engine that exports nothing gets None, not 0
    assert VLLMAdapter(mixed, engine="9").counter(
        mixed, "prompt_tokens_total") is None


def test_engine_selector_applies_to_histograms_and_positions():
    text = ('vllm:time_to_first_token_seconds_bucket{engine="0",le="+Inf"} 4\n'
            'vllm:time_to_first_token_seconds_count{engine="0"} 4\n'
            'vllm:time_to_first_token_seconds_bucket{engine="1",le="+Inf"} 6\n'
            'vllm:time_to_first_token_seconds_count{engine="1"} 6\n'
            'vllm:spec_decode_num_accepted_tokens_per_pos_total'
            '{engine="0",position="0"} 5\n'
            'vllm:spec_decode_num_accepted_tokens_per_pos_total'
            '{engine="1",position="0"} 9\n')
    s = parse_text(text)
    assert VLLMAdapter(s).histogram(s, "ttft_hist").observations == 10.0
    assert VLLMAdapter(s, engine="1").histogram(
        s, "ttft_hist").observations == 6.0
    assert VLLMAdapter(s).by_position(s, "spec_decode_accepted_per_pos") == {0: 14.0}
    assert VLLMAdapter(s, engine="0").by_position(
        s, "spec_decode_accepted_per_pos") == {0: 5.0}


def test_per_position_acceptance_keeps_the_label_as_the_measurement():
    ad = adapter_for(DUMP)
    s = parse_text(DUMP)
    pos = ad.by_position(s, "spec_decode_accepted_per_pos")
    assert sorted(pos) == [0, 1, 2]          # the label IS the measurement
    # acceptance decays with draft position -- that is what makes the
    # 1 + a + a^2 geometric model testable
    assert pos[0] > pos[1] > pos[2] > 0
    assert sum(pos.values()) == pytest.approx(
        ad.counter(s, "spec_decode_num_accepted_tokens_total"))


def test_gauge_aggregation_is_metric_specific():
    """Request counts add across engines; an occupancy FRACTION cannot --
    summing two engines at 0.7 would report the pool 140% full."""
    two = parse_text('vllm:num_requests_running{engine="0"} 8\n'
                     'vllm:num_requests_running{engine="1"} 5\n'
                     'vllm:kv_cache_usage_perc{engine="0"} 0.7\n'
                     'vllm:kv_cache_usage_perc{engine="1"} 0.7\n')
    both = VLLMAdapter(two)
    assert both.gauge(two, "requests_running") == 13.0     # extensive: sums
    assert both.gauge(two, "kv_cache_usage") is None       # intensive: cannot
    # selecting an engine makes the fraction answerable again
    one = VLLMAdapter(two, engine="1")
    assert one.gauge(two, "kv_cache_usage") == pytest.approx(0.7)
    assert one.gauge(two, "requests_running") == 5.0
    # a single-engine deployment is unaffected
    solo = parse_text('vllm:kv_cache_usage_perc{engine="0"} 0.64\n')
    assert VLLMAdapter(solo).gauge(solo, "kv_cache_usage") == pytest.approx(0.64)


def test_gauges_go_through_gauge_not_counter():
    """An adapter may combine or rescale a gauge differently from a counter
    (SGLang reports occupancy as a percent). Callers must dispatch on kind."""
    class PercentAdapter(VLLMAdapter):
        def gauge(self, samples, key):
            v = super().gauge(samples, key)
            if v is not None and key == "kv_cache_usage":
                return v / 100.0            # this engine reports 0..100
            return v

    s = parse_text('vllm:kv_cache_usage_perc{engine="0"} 64.0\n'
                   'vllm:num_requests_running{engine="0"} 4\n')
    ad = PercentAdapter(s)
    assert ad.counter(s, "kv_cache_usage") == 64.0          # raw
    assert ad.gauge(s, "kv_cache_usage") == pytest.approx(0.64)
    assert ad.read(s, "kv_cache_usage") == pytest.approx(0.64)
    snaps = [Snapshot(0.0, 0.01, s), Snapshot(2.0, 0.01, s)]
    w = window_from_snapshots(snaps, 0.5, 1.5, adapter=ad)
    assert w.kv_usage.mean == pytest.approx(0.64)


def test_detect_adapter_and_the_protocol():
    ad = detect_adapter(parse_text(DUMP))
    assert isinstance(ad, VLLMAdapter)
    assert isinstance(ad, MetricsAdapter)
    # an unrecognised dump still yields a usable adapter with everything None
    blank = detect_adapter(parse_text("some_other_exporter_metric 1.0\n"))
    assert blank.resolution().resolved == {}
    assert blank.counter([], "prompt_tokens_total") is None


def test_kv_usage_is_a_fraction_not_a_percent():
    ad = adapter_for(DUMP)
    v = ad.gauge(parse_text(DUMP), "kv_cache_usage")
    assert 0.0 <= v <= 1.0 and v == pytest.approx(0.6412)


# ---------------------------------------------------------------------------
# window arithmetic
# ---------------------------------------------------------------------------
def _dump(*, prompt=0.0, gen=0.0, running=0.0, waiting=0.0, kv=0.0,
          queries=0.0, hits=0.0, flops=0.0, drafts=0.0, draft_tok=0.0,
          accepted=0.0, steps=0, step_sum=0.0):
    """A minimal synthetic V1 dump, so a series can be written by hand."""
    return f"""# TYPE vllm:iteration_tokens_total histogram
vllm:prompt_tokens_total{{engine="0"}} {prompt}
vllm:generation_tokens_total{{engine="0"}} {gen}
vllm:num_requests_running{{engine="0"}} {running}
vllm:num_requests_waiting{{engine="0"}} {waiting}
vllm:kv_cache_usage_perc{{engine="0"}} {kv}
vllm:prefix_cache_queries_total{{engine="0"}} {queries}
vllm:prefix_cache_hits_total{{engine="0"}} {hits}
vllm:num_preemptions_total{{engine="0"}} 0.0
vllm:estimated_flops_per_gpu_total{{engine="0"}} {flops}
vllm:spec_decode_num_drafts_total{{engine="0"}} {drafts}
vllm:spec_decode_num_draft_tokens_total{{engine="0"}} {draft_tok}
vllm:spec_decode_num_accepted_tokens_total{{engine="0"}} {accepted}
vllm:iteration_tokens_total_bucket{{engine="0",le="32.0"}} {steps}
vllm:iteration_tokens_total_bucket{{engine="0",le="+Inf"}} {steps}
vllm:iteration_tokens_total_count{{engine="0"}} {steps}
vllm:iteration_tokens_total_sum{{engine="0"}} {step_sum}
"""


def _snap(t, rtt=0.02, **kw):
    text = _dump(**kw)
    return Snapshot(t_sent=t - rtt / 2, rtt=rtt, samples=parse_text(text),
                    lines=text.splitlines())


def _series():
    """Five ticks, one second apart, t = 0.0 .. 4.0 at the rtt MIDPOINT."""
    return [
        _snap(0.0, gen=0.0, prompt=0.0, running=2, kv=0.10, queries=0.0, hits=0.0),
        _snap(1.0, gen=100.0, prompt=1000.0, running=4, kv=0.20, queries=1000.0,
              hits=900.0),
        _snap(2.0, gen=300.0, prompt=2000.0, running=8, kv=0.40, queries=2000.0,
              hits=1800.0),
        _snap(3.0, gen=600.0, prompt=3000.0, running=6, kv=0.30, queries=3000.0,
              hits=2600.0),
        _snap(4.0, gen=1000.0, prompt=4000.0, running=1, kv=0.05, queries=4000.0,
              hits=3400.0),
    ]


def test_window_is_bounded_by_the_snapshots_outside_the_ask():
    snaps = _series()
    w = window_from_snapshots(snaps, 1.4, 2.6)
    assert (w.lo.t, w.hi.t) == (1.0, 3.0)      # nearest OUTSIDE, both sides
    assert w.dt == pytest.approx(2.0)
    assert w.tokens_out == pytest.approx(500.0)    # 600 - 100
    assert w.tokens_in == pytest.approx(2000.0)
    assert w.output_tok_s == pytest.approx(250.0)


def test_window_endpoints_carry_their_scrape_uncertainty():
    snaps = _series()
    w = window_from_snapshots(snaps, 1.4, 2.6)
    assert w.t_lo_uncertainty == pytest.approx(0.01)   # half of a 20 ms rtt
    assert w.t_hi_uncertainty == pytest.approx(0.01)
    assert w.dt_uncertainty == pytest.approx(0.02)


def test_an_uncovered_ask_raises_instead_of_answering_a_different_question():
    """Snapshots at 0..4 cannot answer "what happened during [10, 20]". The
    old behaviour clamped to the series and returned those numbers."""
    snaps = _series()
    with pytest.raises(WindowNotCovered, match="high side"):
        window_from_snapshots(snaps, 10.0, 20.0)
    with pytest.raises(WindowNotCovered, match="low side"):
        window_from_snapshots(snaps, -100.0, 2.0)
    with pytest.raises(WindowNotCovered, match="high side"):
        window_from_snapshots(snaps, 1.0, 100.0)


def test_endpoints_must_enclose_not_merely_neighbour():
    """lo's scrape must have COMPLETED before t0 and hi's must have STARTED
    after t1; the round trip is not a point."""
    # t_sent=1.0, rtt=0.5 -> this scrape is only known to be as-of [1.0, 1.5]
    snaps = [Snapshot(1.0, 0.5, parse_text(_dump(gen=10.0))),
             Snapshot(3.0, 0.5, parse_text(_dump(gen=90.0)))]
    # t0=1.2 falls INSIDE the first scrape's round trip: not covered
    with pytest.raises(WindowNotCovered, match="low side"):
        window_from_snapshots(snaps, 1.2, 2.0)
    # t0=1.5 is exactly when it completed: covered
    w = window_from_snapshots(snaps, 1.5, 2.0)
    assert (w.lo.t_sent, w.hi.t_sent) == (1.0, 3.0)
    assert w.tokens_out == pytest.approx(80.0)


def test_whole_log_needs_no_coverage():
    """An open end asks about the log itself, whose limit IS its endpoint."""
    snaps = _series()
    w = window_from_snapshots(snaps, None, None)
    assert (w.lo.t, w.hi.t) == (0.0, 4.0)
    assert w.tokens_out == pytest.approx(1000.0)


def test_window_between_two_ticks_still_yields_a_pair():
    """An ask entirely inside one scrape interval must not collapse."""
    w = window_from_snapshots(_series(), 2.2, 2.4)
    assert (w.lo.t, w.hi.t) == (2.0, 3.0)


def test_mid_request_flush_lands_in_the_window_containing_both_ticks():
    """A counter that flushes MID-request jumps between two ticks. The window
    bounded by those two ticks must carry the whole jump -- that is the entire
    reason endpoints are the snapshots OUTSIDE the ask."""
    snaps = [_snap(0.0, flops=0.0), _snap(1.0, flops=0.0),
             _snap(2.0, flops=0.0), _snap(3.0, flops=4.0e18),
             _snap(4.0, flops=4.0e18)]
    # the request ran from ~2.1 to ~2.9; its FLOPs flushed after the t=2 scrape
    w = window_from_snapshots(snaps, 2.1, 2.9)
    assert w.counters["estimated_flops_total"] == pytest.approx(4.0e18)
    # a window that ends before the flush and one that starts after it must
    # NOT also claim it -- the attribution is exclusive
    assert window_from_snapshots(snaps, 0.1, 1.9).counters[
        "estimated_flops_total"] == pytest.approx(0.0)
    assert window_from_snapshots(snaps, 3.1, 3.9).counters[
        "estimated_flops_total"] == pytest.approx(0.0)


def test_prefix_hit_and_miss_rate_are_window_rates():
    w = window_from_snapshots(_series(), 2.5, 3.5)
    # queries 2000 -> 4000, hits 1800 -> 3400 over [2.0, 4.0]
    assert w.prefix_hit_rate == pytest.approx(1600.0 / 2000.0)
    assert w.miss_rate == pytest.approx(0.2)


def test_gauge_stats_cover_every_snapshot_in_the_window():
    w = window_from_snapshots(_series(), 1.5, 3.5)
    assert (w.lo.t, w.hi.t) == (1.0, 4.0)
    assert w.running.n == 4                         # ticks at 1, 2, 3, 4
    assert w.running.max == 8.0
    assert w.running.mean == pytest.approx((4 + 8 + 6 + 1) / 4)
    assert w.kv_usage.max == pytest.approx(0.40)
    assert w.waiting.mean == pytest.approx(0.0)


def test_spec_decode_alpha_and_accepted_length():
    snaps = [_snap(0.0, drafts=0.0, draft_tok=0.0, accepted=0.0),
             _snap(1.0, drafts=100.0, draft_tok=300.0, accepted=180.0),
             _snap(2.0, drafts=200.0, draft_tok=600.0, accepted=360.0)]
    # the ask sits inside one interval, so the endpoints are t=0 and t=2:
    # 200 drafts, 600 tokens proposed, 360 accepted
    w = window_from_snapshots(snaps, 0.5, 1.5)
    assert (w.lo.t, w.hi.t) == (0.0, 2.0)
    assert w.alpha == pytest.approx(0.6)             # 360 / 600
    assert w.mean_accepted_len == pytest.approx(1.0 + 360.0 / 200.0)
    assert w.draft_width == pytest.approx(3.0)       # 600 / 200


def test_spec_decode_is_none_when_the_server_does_not_export_it():
    text = 'vllm:generation_tokens_total{engine="0"} 5\n'
    snaps = [Snapshot(0.0, 0.01, parse_text(text)),
             Snapshot(1.0, 0.01, parse_text(text))]
    w = window_from_snapshots(snaps, 0.5, 0.9)
    assert w.alpha is None and w.mean_accepted_len is None
    assert w.prefix_hit_rate is None and w.miss_rate is None


def test_step_counts_come_from_the_iteration_histogram():
    snaps = [_snap(0.0, steps=1000, step_sum=32000.0),
             _snap(2.0, steps=1200, step_sum=41600.0)]
    w = window_from_snapshots(snaps, 0.5, 1.5)
    assert w.steps == pytest.approx(200.0)
    assert w.tokens_per_step == pytest.approx((41600.0 - 32000.0) / 200.0)
    assert w.step_time_s == pytest.approx(2.0 / 200.0)


def test_window_histogram_delta_gives_window_quantiles():
    def dump(count, sum_s, b_small):
        return (f'# TYPE vllm:time_to_first_token_seconds histogram\n'
                f'vllm:time_to_first_token_seconds_bucket{{le="1.0"}} {b_small}\n'
                f'vllm:time_to_first_token_seconds_bucket{{le="2.0"}} {count}\n'
                f'vllm:time_to_first_token_seconds_bucket{{le="+Inf"}} {count}\n'
                f'vllm:time_to_first_token_seconds_count {count}\n'
                f'vllm:time_to_first_token_seconds_sum {sum_s}\n')

    snaps = [Snapshot(0.0, 0.01, parse_text(dump(100, 50.0, 100))),
             Snapshot(1.0, 0.01, parse_text(dump(120, 82.0, 100)))]
    w = window_from_snapshots(snaps, 0.5, 0.9)
    h = w.ttft
    # the 20 requests that finished IN the window all landed in (1, 2]
    assert h.observations == 20.0
    assert h.mean() == pytest.approx(32.0 / 20.0)
    assert h.quantile(0.5) == pytest.approx(1.5)


def test_window_needs_two_successful_snapshots():
    with pytest.raises(ValueError, match="at least|need >= 2|>= 2"):
        window_from_snapshots([_snap(0.0)], 0.0, 1.0)


def test_failed_scrapes_are_retained_and_never_become_endpoints():
    snaps = _series()
    snaps.insert(3, Snapshot(t_sent=2.5, rtt=float("nan"), samples=[], ok=False,
                             error="ReadTimeout: nope"))
    w = window_from_snapshots(snaps, 1.5, 3.5)
    assert (w.lo.t, w.hi.t) == (1.0, 4.0)
    assert w.n_failed == 1
    assert w.n_snapshots == 5              # the failure is counted, not dropped


def test_a_counter_reset_is_reported_not_reported_as_a_negative_rate():
    """An engine restart inside the window makes the delta unknowable: you
    know the end value but not how far the counter climbed before zeroing."""
    snaps = [_snap(0.0, gen=5000.0, prompt=9000.0, steps=800),
             _snap(1.0, gen=5400.0, prompt=9800.0, steps=850),
             _snap(2.0, gen=120.0, prompt=300.0, steps=20)]     # restarted
    w = window_from_snapshots(snaps, 0.5, 1.5)
    assert "generation_tokens_total" in w.counter_resets
    assert "prompt_tokens_total" in w.counter_resets
    assert "iteration_tokens_hist" in w.counter_resets
    assert w.tokens_out is None                  # never a negative token count
    assert w.output_tok_s is None                # and never a negative rate
    assert w.steps is None
    # a window that does NOT span the restart is unaffected
    clean = window_from_snapshots(snaps, 0.2, 0.8)
    assert clean.counter_resets == ()
    assert clean.tokens_out == pytest.approx(400.0)


def test_a_reset_that_recovers_between_the_endpoints_is_still_caught():
    """100 -> 10 -> 150 across three ticks: the two ENDPOINTS show a
    plausible +50, and only comparing consecutive snapshots sees the dip."""
    snaps = [_snap(0.0, gen=100.0), _snap(1.0, gen=10.0), _snap(2.0, gen=150.0)]
    w = window_from_snapshots(snaps, 0.5, 1.5)
    assert (w.lo.t, w.hi.t) == (0.0, 2.0)
    assert "generation_tokens_total" in w.invalid
    assert w.tokens_out is None            # not the bogus +50
    assert w.output_tok_s is None


def test_a_reset_hidden_inside_an_engine_sum_is_still_caught():
    """Engine 0 advancing while engine 1 restarts gives a RISING sum. Only
    a per-label-set comparison, before aggregation, can see the restart."""
    def dump(a, b):
        return (f'vllm:generation_tokens_total{{engine="0"}} {a}\n'
                f'vllm:generation_tokens_total{{engine="1"}} {b}\n')

    snaps = [Snapshot(0.0, 0.01, parse_text(dump(2_900_000, 700_000))),
             Snapshot(1.0, 0.01, parse_text(dump(4_200_000, 20_000))),
             Snapshot(2.0, 0.01, parse_text(dump(5_800_000, 90_000)))]
    # the naive endpoint sum looks fine: 3.6M -> 5.89M, a positive delta
    assert (5_800_000 + 90_000) - (2_900_000 + 700_000) > 0
    w = window_from_snapshots(snaps, 0.5, 1.5)
    assert "generation_tokens_total" in w.invalid
    assert w.tokens_out is None
    # engine 0 alone never restarted, so selecting it recovers the window
    w0 = window_from_snapshots(snaps, 0.5, 1.5, engine="0")
    assert w0.invalid == {}
    assert w0.tokens_out == pytest.approx(2_900_000.0)


def test_a_histogram_reset_visible_only_in_sum_is_caught():
    """Buckets and _count flat while _sum drops: still a restart."""
    def dump(count, total):
        return (f'vllm:time_to_first_token_seconds_bucket{{le="1.0"}} {count}\n'
                f'vllm:time_to_first_token_seconds_bucket{{le="+Inf"}} {count}\n'
                f'vllm:time_to_first_token_seconds_count {count}\n'
                f'vllm:time_to_first_token_seconds_sum {total}\n')

    snaps = [Snapshot(0.0, 0.01, parse_text(dump(50, 30.0))),
             Snapshot(1.0, 0.01, parse_text(dump(50, 4.0))),
             Snapshot(2.0, 0.01, parse_text(dump(50, 9.0)))]
    w = window_from_snapshots(snaps, 0.5, 1.5)
    assert "ttft_hist" in w.invalid
    assert w.ttft is None


def test_a_changed_bucket_layout_invalidates_rather_than_zero_fills():
    full = ('vllm:time_to_first_token_seconds_bucket{le="1.0"} 100\n'
            'vllm:time_to_first_token_seconds_bucket{le="2.0"} 118\n'
            'vllm:time_to_first_token_seconds_bucket{le="8.0"} 120\n'
            'vllm:time_to_first_token_seconds_bucket{le="+Inf"} 120\n'
            'vllm:time_to_first_token_seconds_count 120\n'
            'vllm:time_to_first_token_seconds_sum 182.0\n')
    truncated = ('vllm:time_to_first_token_seconds_bucket{le="1.0"} 100\n'
                 'vllm:time_to_first_token_seconds_bucket{le="+Inf"} 100\n'
                 'vllm:time_to_first_token_seconds_count 100\n'
                 'vllm:time_to_first_token_seconds_sum 50.0\n')
    snaps = [Snapshot(0.0, 0.01, parse_text(truncated)),
             Snapshot(2.0, 0.01, parse_text(full))]
    w = window_from_snapshots(snaps, 0.5, 1.5)
    assert "ttft_hist" in w.invalid
    assert "bucket layout" in w.invalid["ttft_hist"]
    assert w.ttft is None                  # not a 20-observation fiction


def test_reset_detection_works_without_the_bulk_read_fast_path():
    """The one-pass `all_series` is an optimisation; an adapter that only
    implements the per-key accessors must still get reset detection."""
    class SlowAdapter(VLLMAdapter):
        all_series = None                    # hide the fast path
        all_histogram_series = None

        def __getattribute__(self, name):
            if name in ("all_series", "all_histogram_series"):
                raise AttributeError(name)
            return super().__getattribute__(name)

    snaps = [_snap(0.0, gen=100.0), _snap(1.0, gen=10.0), _snap(2.0, gen=150.0)]
    ad = SlowAdapter(snaps[0].samples)
    assert not hasattr(ad, "all_series")
    w = window_from_snapshots(snaps, 0.5, 1.5, adapter=ad)
    assert "generation_tokens_total" in w.invalid
    assert w.tokens_out is None


def test_per_position_reset_is_reported_too():
    snaps = [_snap(0.0, accepted=0.0), _snap(1.0, accepted=0.0)]
    text_a = ('vllm:spec_decode_num_accepted_tokens_per_pos_total'
              '{engine="0",position="0"} 900\n')
    text_b = ('vllm:spec_decode_num_accepted_tokens_per_pos_total'
              '{engine="0",position="0"} 12\n')
    snaps = [Snapshot(0.0, 0.01, parse_text(text_a)),
             Snapshot(1.0, 0.01, parse_text(text_b))]
    w = window_from_snapshots(snaps, 0.5, 0.9)
    assert "spec_decode_accepted_per_pos" in w.counter_resets
    assert w.per_position is None


def test_window_to_dict_is_json_serialisable():
    d = window_from_snapshots(_series(), 1.0, 3.0).to_dict()
    json.dumps(d)                          # must not raise
    assert d["derived"]["output_tok_s"] == pytest.approx(250.0)
    assert d["endpoints"]["lo_uncertainty_s"] == pytest.approx(0.01)


def test_gauge_stats_of_nothing():
    g = GaugeStats.of([])
    assert g.n == 0 and g.mean is None and g.max is None


# ---------------------------------------------------------------------------
# snapshots and the JSONL round trip
# ---------------------------------------------------------------------------
def test_snapshot_json_round_trip_keeps_the_raw_lines(tmp_path):
    s = _snap(1.0, gen=42.0, running=3)
    back = Snapshot.from_json(s.to_json())
    assert back.t_sent == pytest.approx(s.t_sent)
    assert back.rtt == pytest.approx(s.rtt)
    ad = detect_adapter(back.samples)
    assert ad.counter(back.samples, "generation_tokens_total") == 42.0


def test_snapshot_json_round_trip_escapes_label_values():
    """The raw lines are the archive format, so a label value carrying a
    quote or a backslash has to survive the round trip intact."""
    nasty = Sample("vllm:request_success_total",
                   {"finished_reason": 'stop\\ "quoted"', "engine": "0"}, 3.0)
    s = Snapshot(t_sent=1.0, rtt=0.01, samples=[nasty])   # no `lines`: _render
    back = Snapshot.from_json(s.to_json())
    assert back.samples == [nasty]


def test_load_jsonl_skips_a_truncated_final_line(tmp_path):
    p = tmp_path / "log.jsonl"
    body = "\n".join(s.to_json() for s in _series())
    p.write_text(body + '\n{"t_sent": 5.0, "rtt": 0.0, "li')
    snaps = load_jsonl(p)
    assert len(snaps) == 5
    assert window_from_snapshots(snaps, 1.4, 2.6).tokens_out == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# the sampler, against an httpx.MockTransport fake
# ---------------------------------------------------------------------------
class FakeServer:
    """Serves a dump whose counters advance one step per request.

    `fail_every` makes every Nth request raise, so the sampler's
    keep-the-failure behaviour is exercised without a real socket.
    """

    def __init__(self, fail_every: int = 0, dump: str | None = None):
        self.n = 0
        self.fail_every = fail_every
        self.dump = dump

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.n += 1
        if self.fail_every and self.n % self.fail_every == 0:
            raise httpx.ReadTimeout("fake timeout", request=request)
        if self.dump is not None:
            return httpx.Response(200, text=self.dump)
        k = self.n
        return httpx.Response(200, text=_dump(
            prompt=1000.0 * k, gen=100.0 * k, running=k, waiting=0.0,
            kv=0.1 * k, queries=1000.0 * k, hits=900.0 * k, steps=50 * k,
            step_sum=1600.0 * k))

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=self.transport())


def test_sampler_collects_raw_snapshots():
    async def run():
        srv = FakeServer()
        async with MetricsSampler("http://fake/metrics", interval=0.01,
                                  client=srv.client()) as s:
            await _until(lambda: len(s) >= 5)
            return list(s.snapshots), s

    snaps, s = asyncio.run(run())
    assert len(snaps) >= 5
    assert all(sn.ok and sn.samples for sn in snaps[:5])
    assert all(sn.rtt >= 0 for sn in snaps[:5])
    # the RAW lines are kept, not just what the adapter understood
    assert any("vllm:kv_cache_usage_perc" in ln for ln in snaps[0].lines)
    # counters advance one step per scrape
    ad = detect_adapter(snaps[0].samples)
    a = ad.counter(snaps[0].samples, "generation_tokens_total")
    b = ad.counter(snaps[1].samples, "generation_tokens_total")
    assert b - a == pytest.approx(100.0)


def test_sampler_window_and_at():
    async def run():
        srv = FakeServer()
        async with MetricsSampler("http://fake/metrics", interval=0.01,
                                  client=srv.client()) as s:
            await _until(lambda: len(s) >= 6)
            t_lo = s.snapshots[1].t
            t_hi = s.snapshots[4].t
            return s.window(t_lo + 0.001, t_hi - 0.001), s.at(t_hi), s.snapshots

    w, at, snaps = asyncio.run(run())
    assert w.lo.t <= snaps[1].t and w.hi.t >= snaps[4].t
    assert w.tokens_out is not None and w.tokens_out > 0
    assert w.running.n >= 2
    assert at.t <= snaps[4].t


def test_sampler_retains_failures_as_snapshots():
    async def run():
        srv = FakeServer(fail_every=2)
        async with MetricsSampler("http://fake/metrics", interval=0.01,
                                  client=srv.client()) as s:
            await _until(lambda: len(s) >= 6)
            return list(s.snapshots), s.n_failed

    snaps, n_failed = asyncio.run(run())
    bad = [sn for sn in snaps if not sn.ok]
    assert bad and n_failed == len(bad)
    assert "ReadTimeout" in bad[0].error
    assert math.isnan(bad[0].rtt)
    # a failed scrape is still a snapshot, and its t is its send time
    assert bad[0].t == bad[0].t_sent


def test_sampler_writes_jsonl_that_replays(tmp_path):
    out = tmp_path / "tail.jsonl"

    async def run():
        srv = FakeServer()
        async with MetricsSampler("http://fake/metrics", interval=0.01,
                                  out=str(out), client=srv.client()) as s:
            await _until(lambda: len(s) >= 4)

    asyncio.run(run())
    snaps = load_jsonl(out)
    assert len(snaps) >= 4
    w = window_from_snapshots(snaps, None, None)
    assert w.tokens_out == pytest.approx(100.0 * (len(snaps) - 1))


def test_sampler_keep_filter_shrinks_the_record():
    async def run(keep):
        srv = FakeServer()
        async with MetricsSampler("http://fake/metrics", interval=0.01,
                                  keep=keep, client=srv.client()) as s:
            await _until(lambda: len(s) >= 2)
            return s.snapshots[0]

    full = asyncio.run(run(None))
    thin = asyncio.run(run(("num_requests_running",)))
    assert len(thin.lines) < len(full.lines)
    assert {s.name for s in thin.samples} == {"vllm:num_requests_running"}


def test_sampler_probe_reports_resolution():
    async def run():
        srv = FakeServer(dump=DUMP)
        s = MetricsSampler("http://fake/metrics", client=srv.client())
        try:
            return await s.probe()
        finally:
            await s.aclose()

    snap, res = asyncio.run(run())
    assert snap.ok
    assert res.version_hint == "v1"
    assert "ttft_hist" in res.resolved
    assert "forward_time_hist" in res.missing


def test_sampler_live_line():
    async def run():
        srv = FakeServer()
        async with MetricsSampler("http://fake/metrics", interval=0.01,
                                  client=srv.client()) as s:
            await _until(lambda: len(s) >= 3)
            return s.live()

    v = asyncio.run(run())
    assert v["running"] is not None
    assert v["tok_s"] is not None and v["tok_s"] > 0
    assert v["hit_rate"] == pytest.approx(0.9)


async def _until(pred, timeout=5.0, tick=0.005):
    """Wait for a condition, bounded. An unbounded `while ...: await sleep`
    hangs the whole suite when the thing under test stops advancing."""
    async with asyncio.timeout(timeout):
        while not pred():
            await asyncio.sleep(tick)


def test_a_dying_loop_surfaces_to_the_consumer():
    """An --out writer that starts failing must not look like a quiet run:
    the consumer waits on a snapshot count that will never rise again."""
    class Exploding:
        def __init__(self):
            self.n = 0

        def write(self, _s):
            self.n += 1
            if self.n > 2:
                raise OSError(28, "No space left on device")

        def close(self):
            pass

    async def run():
        srv = FakeServer()
        s = MetricsSampler("http://fake/metrics", interval=0.01,
                           client=srv.client())
        s._out = Exploding()
        await s.start()
        try:
            await _until(lambda: s.error is not None)
        finally:
            await s.aclose()
        return s

    s = asyncio.run(run())
    assert s.error is not None and "No space left" in s.error
    with pytest.raises(SamplerStopped, match="No space left"):
        s.window(0.0, 1.0)
    # and cleanup still ran despite the failure
    assert s._task is None and s._out is None


def test_aclose_is_idempotent_and_releases_everything():
    async def run():
        srv = FakeServer()
        s = MetricsSampler("http://fake/metrics", interval=0.01,
                           client=srv.client())
        await s.start()
        await _until(lambda: len(s) >= 2)
        await s.aclose()
        await s.aclose()               # second call must be a no-op, not a raise
        return s

    s = asyncio.run(run())
    assert s._task is None and s._out is None


def test_a_truncated_first_scrape_does_not_latch_a_poor_adapter():
    """Latching on scrape #1 would read the whole run through a half-resolved
    vocabulary, reporting keys as missing that the server exports."""
    class Ramping:
        """First response is truncated to two series, then the full dump."""
        def __init__(self):
            self.n = 0

        def handle(self, request):
            self.n += 1
            if self.n == 1:
                return httpx.Response(200, text=(
                    'vllm:num_requests_running{engine="0"} 1\n'
                    'vllm:prompt_tokens_total{engine="0"} 5\n'))
            return httpx.Response(200, text=DUMP)

    async def run():
        r = Ramping()
        client = httpx.AsyncClient(transport=httpx.MockTransport(r.handle))
        async with MetricsSampler("http://fake/metrics", interval=0.01,
                                  client=client) as s:
            await _until(lambda: len(s) >= 1)
            first = len(s.adapter.resolution().resolved)
            await _until(lambda: len(s.adapter.resolution().resolved) > first)
            return first, s.adapter.resolution()

    first, res = asyncio.run(run())
    assert first == 2                       # the truncated view
    assert "ttft_hist" in res.resolved       # the richer one won
    assert len(res.resolved) > first


def test_probe_keeps_the_resolution_it_computed():
    async def run():
        srv = FakeServer(dump=DUMP)
        s = MetricsSampler("http://fake/metrics", client=srv.client())
        try:
            _, res = await s.probe()
            return s.adapter, res
        finally:
            await s.aclose()

    adapter, res = asyncio.run(run())
    assert adapter is not None               # not discarded
    assert adapter.resolution().resolved == res.resolved


def test_wait_first_ignores_failed_scrapes():
    """A failed scrape is not an endpoint, so releasing `wait_first` on one
    hands the caller a sampler it cannot build a window from."""
    async def run():
        srv = FakeServer(fail_every=1)       # every scrape fails
        async with MetricsSampler("http://fake/metrics", interval=0.01,
                                  client=srv.client()) as s:
            await _until(lambda: len(s) >= 3)
            return await s.wait_first(timeout=0.05)

    assert asyncio.run(run()) is False


def test_next_tick_is_what_makes_a_live_window_coverable():
    """The documented pattern: t1 = now; await next_tick(); window(t0, t1)."""
    async def run():
        srv = FakeServer()
        async with MetricsSampler("http://fake/metrics", interval=0.02,
                                  client=srv.client()) as s:
            assert await s.wait_first(timeout=2.0)
            t0 = s.snapshots[0].t_sent + s.snapshots[0].rtt
            await _until(lambda: len(s) >= 2)
            t1 = time.time()
            # before the next tick lands there is no enclosing hi endpoint
            with pytest.raises(WindowNotCovered):
                s.window(t0, t1 + 0.05)
            assert await s.next_tick(timeout=2.0)
            return s.window(t0, t1)

    w = asyncio.run(run())
    assert w.tokens_out is not None and w.tokens_out > 0


def test_max_snapshots_trims_the_ring():
    async def run():
        srv = FakeServer()
        async with MetricsSampler("http://fake/metrics", interval=0.01,
                                  max_snapshots=3, client=srv.client()) as s:
            await _until(lambda: srv.n >= 6)
            return list(s.snapshots)

    snaps = asyncio.run(run())
    assert len(snaps) <= 3
    # it is the OLDEST that are dropped: counters rise across what remains
    ad = detect_adapter(snaps[0].samples)
    vals = [ad.counter(x.samples, "generation_tokens_total") for x in snaps]
    assert vals == sorted(vals)


def test_an_injected_client_still_honours_the_samplers_timeout():
    seen = {}

    def handle(request):
        seen["timeout"] = request.extensions.get("timeout")
        return httpx.Response(200, text=DUMP)

    async def run():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        s = MetricsSampler("http://fake/metrics", timeout=0.123, client=client)
        try:
            await s.scrape_once()
        finally:
            await s.aclose()

    asyncio.run(run())
    assert seen["timeout"]["read"] == pytest.approx(0.123)
    assert seen["timeout"]["connect"] == pytest.approx(0.123)


def test_scrape_times_ignore_a_wall_clock_step(monkeypatch):
    """`time.time()` can jump backwards (NTP), which would put the series out
    of order and make every bisect meaningless. Timestamps are anchored to
    `time.monotonic()` instead, so a step cannot reach them."""
    s = MetricsSampler("http://fake/metrics")
    before = s._now()
    monkeypatch.setattr(time, "time", lambda: 0.0)   # the clock lurches back
    after = s._now()
    assert after >= before
    assert abs(after - before) < 1.0                 # and barely moved


def test_window_sorts_snapshots_it_is_handed():
    """A replayed log can be concatenated out of order; the bisect must not
    be handed an unsorted list."""
    snaps = _series()
    shuffled = [snaps[3], snaps[0], snaps[4], snaps[1], snaps[2]]
    w = window_from_snapshots(shuffled, 1.4, 2.6)
    assert (w.lo.t, w.hi.t) == (1.0, 3.0)
    assert w.tokens_out == pytest.approx(500.0)
    assert w.dt > 0


def test_sampler_rejects_a_nonpositive_interval():
    with pytest.raises(ValueError):
        MetricsSampler("http://fake/metrics", interval=0.0)


def test_an_unwritable_out_path_fails_before_a_client_is_opened(tmp_path):
    """`__aenter__` raising means `__aexit__` never runs, so anything opened
    before the failure would leak."""
    async def run():
        # NO injected client, so start() is the thing that would build one
        s = MetricsSampler("http://fake/metrics", interval=0.01,
                           out=str(tmp_path / "no-such-dir" / "log.jsonl"))
        with pytest.raises(OSError):
            async with s:
                pass
        return s

    s = asyncio.run(run())
    assert s._client is None       # nothing to leak: it was never opened
    assert s._out is None
    assert s._task is None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_http(monkeypatch):
    """Point every `httpx.AsyncClient()` the CLI builds at the fake server."""
    srv = FakeServer(dump=DUMP)
    real = httpx.AsyncClient

    def factory(*a, **kw):
        kw.pop("verify", None)
        return real(*a, transport=srv.transport(), **kw)

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    return srv


def test_cli_metrics_probe(fake_http, capsys):
    assert ws_main(["metrics", "probe", "http://fake/metrics"]) == 0
    out = capsys.readouterr().out
    assert "vllm:time_to_first_token_seconds" in out
    assert "MISSING forward_time_hist" in out
    assert "semantic keys resolved" in out


def test_cli_metrics_probe_json(fake_http, capsys):
    assert ws_main(["metrics", "probe", "http://fake/metrics", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["engine"] == "vllm" and d["version_hint"] == "v1"
    assert d["resolved"]["kv_cache_usage"] == "vllm:kv_cache_usage_perc"
    assert "forward_time_hist" in d["missing"]


def test_cli_metrics_tail_writes_jsonl(fake_http, tmp_path, capsys):
    out = tmp_path / "tail.jsonl"
    rc = ws_main(["metrics", "tail", "http://fake/metrics", "--interval", "0.01",
                  "--count", "3", "--out", str(out)])
    assert rc == 0
    printed = capsys.readouterr().out
    assert printed.strip()                       # a live line per snapshot
    snaps = load_jsonl(out)
    assert len(snaps) >= 3
    assert detect_adapter(snaps[0].samples).counter(
        snaps[0].samples, "requests_running") == 24.0


def test_cli_metrics_tail_then_window(fake_http, tmp_path, capsys):
    out = tmp_path / "tail.jsonl"
    ws_main(["metrics", "tail", "http://fake/metrics", "--interval", "0.01",
             "--count", "4", "--out", str(out)])
    capsys.readouterr()
    assert ws_main(["metrics", "window", str(out), "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["dt_s"] > 0
    # the fake serves one constant dump, so every counter delta is zero
    assert d["counters"]["generation_tokens_total"] == 0.0
    assert d["gauges"]["requests_running"]["mean"] == pytest.approx(24.0)


def test_cli_metrics_window_table(tmp_path, capsys):
    p = tmp_path / "log.jsonl"
    p.write_text("\n".join(s.to_json() for s in _series()) + "\n")
    assert ws_main(["metrics", "window", str(p), "--from", "+1.4",
                    "--to", "-1.4"]) == 0
    out = capsys.readouterr().out
    assert "generation_tokens_total" in out
    assert "output tok/s" in out
    assert "snapshots ENCLOSING" in out


def test_cli_metrics_window_whole_log_table(tmp_path, capsys):
    """No --from/--to prints the table rather than tripping over the open
    bounds it passes down as None."""
    p = tmp_path / "log.jsonl"
    p.write_text("\n".join(s.to_json() for s in _series()) + "\n")
    assert ws_main(["metrics", "window", str(p)]) == 0
    out = capsys.readouterr().out
    assert "the whole log" in out
    assert "ENCLOSING" in out
    assert "generation_tokens_total" in out


def test_cli_metrics_window_uncovered_ask_is_a_clean_error(tmp_path, capsys):
    p = tmp_path / "log.jsonl"
    p.write_text("\n".join(s.to_json() for s in _series()) + "\n")
    # main() turns a ValueError into `ws: <message>` and exit 2
    assert ws_main(["metrics", "window", str(p), "--from", "+100",
                    "--to", "+200"]) == 2
    assert "not covered" in capsys.readouterr().err


def test_cli_metrics_window_needs_two_snapshots(tmp_path, capsys):
    p = tmp_path / "log.jsonl"
    p.write_text(_snap(0.0).to_json() + "\n")
    assert ws_main(["metrics", "window", str(p)]) == 1


def _dp_log(tmp_path):
    """A two-engine archive: engine 0 emits 10 tokens, engine 1 emits 100."""
    def dump(a, b):
        return (f'vllm:generation_tokens_total{{engine="0"}} {a}\n'
                f'vllm:generation_tokens_total{{engine="1"}} {b}\n'
                f'vllm:num_requests_running{{engine="0"}} 2\n'
                f'vllm:num_requests_running{{engine="1"}} 7\n')

    snaps = [Snapshot(0.0, 0.01, parse_text(dump(0, 0)),
                      lines=dump(0, 0).splitlines()),
             Snapshot(2.0, 0.01, parse_text(dump(10, 100)),
                      lines=dump(10, 100).splitlines())]
    p = tmp_path / "dp.jsonl"
    p.write_text("\n".join(x.to_json() for x in snaps) + "\n")
    return p, snaps


def test_window_engine_selector_changes_the_answer(tmp_path):
    _, snaps = _dp_log(tmp_path)
    assert window_from_snapshots(snaps, None, None,
                                 engine="0").tokens_out == pytest.approx(10.0)
    assert window_from_snapshots(snaps, None,
                                 None).tokens_out == pytest.approx(110.0)


def test_cli_window_engine_selector_matches_the_live_reading(tmp_path, capsys):
    """A `tail --engine 0` archive must replay as engine 0, not as the sum of
    every engine -- the same log otherwise reports 110 tokens where the live
    sampler reported 10."""
    p, _ = _dp_log(tmp_path)
    assert ws_main(["metrics", "window", str(p), "--engine", "0", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["counters"]["generation_tokens_total"] == pytest.approx(10.0)
    assert d["gauges"]["requests_running"]["mean"] == pytest.approx(2.0)


def test_cli_tail_exits_nonzero_when_every_scrape_failed(monkeypatch, capsys):
    """Three connection errors and rc 0 reads as a successful run."""
    srv = FakeServer(fail_every=1)
    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient",
                        lambda *a, **kw: real(*a, transport=srv.transport(),
                                              **{k: v for k, v in kw.items()
                                                 if k != "verify"}))
    rc = ws_main(["metrics", "tail", "http://fake/metrics",
                  "--interval", "0.01", "--count", "3"])
    assert rc == 1
    assert "every scrape failed" in capsys.readouterr().err


def test_cli_probe_reports_a_declared_but_idle_family(fake_http, capsys):
    """`# TYPE` with no samples means exported-and-idle, which is not the
    same as not exported -- the distinction probe exists to draw."""
    fams = parse_families(DUMP)
    assert "vllm:corrupted_requests" in fams
    assert fams["vllm:corrupted_requests"].samples == []
    assert fams["vllm:corrupted_requests"].type == "counter"


def test_parse_families_keeps_declared_empty_families():
    text = ("# HELP some_metric A metric with no samples right now.\n"
            "# TYPE some_metric counter\n"
            "# HELP other_metric Has one.\n"
            "# TYPE other_metric gauge\n"
            'other_metric{a="b"} 3\n')
    fams = parse_families(text)
    assert set(fams) == {"some_metric", "other_metric"}
    assert fams["some_metric"].samples == []
    assert fams["some_metric"].type == "counter"
    assert fams["some_metric"].help == "A metric with no samples right now."
    assert len(fams["other_metric"].samples) == 1


def test_cli_metrics_keep_and_tls_flags_parse():
    from workingset.cli import build_parser
    args = build_parser().parse_args(
        ["metrics", "tail", "http://x/metrics", "--keep", "decode",
         "--ca-bundle", "/etc/ssl/corp.pem"])
    assert args.keep == "decode" and args.ca_bundle == "/etc/ssl/corp.pem"
    args = build_parser().parse_args(
        ["metrics", "probe", "http://x/metrics", "--insecure"])
    assert args.insecure is True


def test_keep_filter_presets():
    from workingset.metrics import DECODE_KEEP, keep_filter
    assert keep_filter("all") is None
    assert keep_filter(None) is None
    assert keep_filter("decode") == DECODE_KEEP
    assert keep_filter("foo, bar ,") == ("foo", "bar")
