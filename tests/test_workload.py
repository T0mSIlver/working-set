"""Tests for `workingset.workload` — `ws workload`.

The Prometheus tests drive a fake `/api/v1` over `httpx.MockTransport` whose
histograms are generated from a KNOWN log-normal, so the fit is checked
against parameters the test chose rather than against itself. The rest of the
suite is about the two things this command must never do: invent a number the
metrics cannot support, and let a raw series value out of the building.
"""
from __future__ import annotations

import json
import math
import re
import statistics
import tomllib
from pathlib import Path

import httpx
import pytest

from workingset.cli import main as ws_main
from workingset.config import RunConfig, load_config
from workingset.metrics.parse import Histogram, parse_text
from workingset.metrics.sampler import Snapshot
from workingset.workload import (PROMQL, NotObservable, PrometheusClient,
                                 emit_json, emit_table, emit_toml, estimate,
                                 fit_lognormal, merge_into, parse_duration,
                                 promql, read_jsonl, read_metrics_text,
                                 read_prometheus)

FIXTURE = Path(__file__).parent / "fixtures" / "vllm_metrics_v1.txt"
DUMP = FIXTURE.read_text(encoding="utf-8")

# vLLM's own bucket layout for the two per-request size histograms.
SIZE_BOUNDS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000,
               10000, 20000, 50000, 100000, 200000]
E2E_BOUNDS = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0,
              40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0]


# ---------------------------------------------------------------------------
# synthetic distributions with parameters the test knows
# ---------------------------------------------------------------------------
def lognormal_buckets(n: int, median: float, sigma: float,
                      bounds=SIZE_BOUNDS) -> dict[float, float]:
    """Cumulative bucket counts of a log-normal, exactly as an exporter would
    have accumulated them (integer counts, `+Inf` = n)."""
    nd = statistics.NormalDist()
    out = {float(b): float(round(n * nd.cdf((math.log(b) - math.log(median)) / sigma)))
           for b in bounds}
    out[math.inf] = float(n)
    return out


def lognormal_hist(n: int, median: float, sigma: float,
                   bounds=SIZE_BOUNDS) -> Histogram:
    mean = median * math.exp(sigma ** 2 / 2.0)
    return Histogram("h", {}, lognormal_buckets(n, median, sigma, bounds),
                     float(n), mean * n)


# ---------------------------------------------------------------------------
# a fake Prometheus /api/v1
# ---------------------------------------------------------------------------
_NAME = re.compile(r"\b((?:vllm|sglang):[a-zA-Z0-9_:]*)")


class FakeProm:
    """Answers the five query shapes `PROMQL` builds, and nothing else.

    Every query it is asked is recorded in `.queries`, so a test can assert
    the PromQL that was actually built rather than the PromQL it expected to
    be built.
    """

    def __init__(self, *, names: set[str], counters: dict[str, float],
                 hists: dict[str, Histogram], gauge: dict[str, list[float]],
                 resets: dict[str, float] | None = None,
                 now: float = 1_700_000_000.0, step_s: float = 300.0):
        self.names = set(names)
        self.counters = counters
        self.hists = hists
        self.gauge = gauge
        self.resets = resets or {}
        self.now = now
        self.step_s = step_s
        self.queries: list[str] = []

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self._handle),
                            base_url="http://prom.test")

    # ---- the handler --------------------------------------------------
    def _handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/label/__name__/values"):
            return self._ok(sorted(self.names))
        q = request.url.params.get("query", "")
        self.queries.append(q)
        if q == "time()":
            return self._vector([({}, self.now)])
        m = _NAME.search(q)
        if m is None:
            return self._vector([])
        raw = m.group(1)
        base = raw
        for suffix in ("_bucket", "_sum", "_count"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        if path.endswith("/query_range"):
            vals = self.gauge.get(base)
            if vals is None:
                return self._matrix([])
            start = float(request.url.params["start"])
            return self._matrix([(start + i * self.step_s, v)
                                 for i, v in enumerate(vals)])
        if q.startswith("max(resets("):
            v = self.resets.get(base)
            return self._vector([] if v is None else [({}, v)])
        h = self.hists.get(base)
        if "by (le)" in q and h is not None:
            return self._vector([({"le": _le(b)}, c)
                                 for b, c in sorted(h.buckets.items())])
        if raw.endswith("_sum") and h is not None:
            return self._vector([({}, h.sum)])
        if raw.endswith("_count") and h is not None:
            return self._vector([({}, h.count)])
        if base in self.counters:
            return self._vector([({}, self.counters[base])])
        return self._vector([])

    @staticmethod
    def _ok(data) -> httpx.Response:
        return httpx.Response(200, json={"status": "success", "data": data})

    def _vector(self, pairs) -> httpx.Response:
        return self._ok({"resultType": "vector",
                         "result": [{"metric": lbl,
                                     "value": [self.now, repr(float(v))]}
                                    for lbl, v in pairs]})

    def _matrix(self, points) -> httpx.Response:
        return self._ok({"resultType": "matrix",
                         "result": [{"metric": {},
                                     "values": [[t, repr(float(v))]
                                                for t, v in points]}]})


def _le(b: float) -> str:
    return "+Inf" if math.isinf(b) else repr(b)


# the study's workload, as a Prometheus would hold it after 7 days
N_REQ = 39_412.0
PROMPT_MEDIAN, PROMPT_SIGMA = 47_400.0, 0.81
OUT_MEDIAN, OUT_SIGMA = 300.0, 0.80
SAVINGS = 0.875
# deliberately high-precision: a raw reading that survived into an emitted
# number would be visible verbatim, which is what the firewall test looks for
RUNNING = [3.1415926, 7.2360679, 11.0905356, 2.7182818, 0.5772156,
           19.1234567, 41.2718281, 5.6692016, 13.7314159, 1.0986122]


def study_prom(**kw) -> FakeProm:
    prompt = lognormal_hist(int(N_REQ), PROMPT_MEDIAN, PROMPT_SIGMA)
    out = lognormal_hist(int(N_REQ), OUT_MEDIAN, OUT_SIGMA)
    e2e = Histogram("e", {}, lognormal_buckets(int(N_REQ), 9.0, 1.1, E2E_BOUNDS),
                    N_REQ, 18.6931 * N_REQ)
    queries = prompt.sum
    names = {"vllm:prompt_tokens_total", "vllm:prompt_tokens_cached_total",
             "vllm:generation_tokens_total", "vllm:prefix_cache_queries_total",
             "vllm:prefix_cache_hits_total", "vllm:request_success_total",
             "vllm:num_requests_running", "vllm:num_requests_waiting",
             "vllm:request_prompt_tokens_bucket", "vllm:request_prompt_tokens_sum",
             "vllm:request_prompt_tokens_count",
             "vllm:request_generation_tokens_bucket",
             "vllm:request_generation_tokens_sum",
             "vllm:request_generation_tokens_count",
             "vllm:e2e_request_latency_seconds_bucket",
             "vllm:e2e_request_latency_seconds_sum",
             "vllm:e2e_request_latency_seconds_count"}
    spec = dict(
        names=names,
        counters={"vllm:request_success_total": N_REQ,
                  "vllm:prompt_tokens_total": queries,
                  "vllm:prompt_tokens_cached_total": SAVINGS * queries,
                  "vllm:generation_tokens_total": out.sum,
                  "vllm:prefix_cache_queries_total": queries,
                  "vllm:prefix_cache_hits_total": SAVINGS * queries},
        hists={"vllm:request_prompt_tokens": prompt,
               "vllm:request_generation_tokens": out,
               "vllm:e2e_request_latency_seconds": e2e},
        gauge={"vllm:num_requests_running": list(RUNNING)})
    spec.update(kw)
    return FakeProm(**spec)


def study_estimate(prom: FakeProm | None = None, **kw):
    prom = prom or study_prom()
    rd = read_prometheus("http://prom.test", range="7d", step="5m",
                         selector='model_name="Qwen/Qwen3-27B"',
                         now=prom.now, client=prom.client())
    return estimate(rd, **kw), prom


# ===========================================================================
# PromQL is built from the adapter's names
# ===========================================================================
def test_promql_shapes_carry_the_selector_and_the_engine():
    q = PROMQL(selector='model_name="m"', engine="0", range="7d")
    assert q.counter("vllm:prompt_tokens_total") == (
        'sum(increase(vllm:prompt_tokens_total'
        '{model_name="m",engine="0"}[7d]))')
    assert q.buckets("vllm:request_prompt_tokens") == (
        'sum by (le) (increase(vllm:request_prompt_tokens_bucket'
        '{model_name="m",engine="0"}[7d]))')
    assert q.gauge("vllm:num_requests_running") == (
        'sum(vllm:num_requests_running{model_name="m",engine="0"})')
    assert q.resets("vllm:request_success_total").startswith("max(resets(")


def test_promql_omits_an_empty_matcher():
    assert promql("counter", "vllm:x", range="1h") == "sum(increase(vllm:x[1h]))"
    assert promql("gauge", "vllm:x") == "sum(vllm:x)"


def test_promql_names_come_from_the_adapter_not_from_literals():
    """Every name queried was RESOLVED against what the server exports.

    The point of the seam: nothing in this module spells a `vllm:` string, so
    a rename upstream is absorbed by `ALIASES` and both the scrape path and
    this one follow it.
    """
    est, prom = study_estimate()
    resolved = est.provenance.resolved
    assert resolved["request_prompt_tokens_hist"] == "vllm:request_prompt_tokens"
    assert resolved["prefix_cache_hits_total"] == "vllm:prefix_cache_hits_total"
    exported = {n.rsplit("_bucket", 1)[0].rsplit("_sum", 1)[0].rsplit("_count", 1)[0]
                for n in prom.names}
    for q in prom.queries:
        m = _NAME.search(q)
        if m is None:                       # `time()` carries no metric name
            continue
        base = m.group(1)
        for suffix in ("_bucket", "_sum", "_count"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        assert base in exported, f"{q} names a metric the server does not export"
        assert base in set(resolved.values())


def test_parse_duration():
    assert parse_duration("7d") == 604800.0
    assert parse_duration("90m") == 5400.0
    assert parse_duration("30s") == 30.0
    with pytest.raises(ValueError):
        parse_duration("1h30m")
    with pytest.raises(ValueError):
        parse_duration("")


# ===========================================================================
# the log-normal fit
# ===========================================================================
@pytest.mark.parametrize("median,sigma,n", [(47_400.0, 0.81, 40_000),
                                            (300.0, 0.80, 40_000),
                                            (3_000.0, 0.60, 20_000)])
def test_fit_recovers_known_lognormal_parameters(median, sigma, n):
    """Bucket counts generated from a known log-normal; the fit must find it
    back to within 2% on both parameters — comfortably inside the study's own
    uncertainty, and far tighter than the buckets are wide."""
    f = fit_lognormal(lognormal_hist(n, median, sigma))
    assert f.median_tokens == pytest.approx(median, rel=0.02)
    assert f.sigma == pytest.approx(sigma, rel=0.02)
    assert f.residual_ln < 0.02          # a straight line, as a log-normal is
    assert f.n_points >= 5


def test_fit_reports_censoring_when_the_top_bucket_overflows():
    """vLLM's top size bucket is 200k tokens; a 47.4k-median, sigma-0.81
    workload puts ~4% of prompts above it."""
    f = fit_lognormal(lognormal_hist(40_000, 47_400.0, 0.81))
    assert f.censored
    assert f.censored_fraction == pytest.approx(0.038, abs=0.005)
    assert f.top_finite_bound == 200_000.0

    g = fit_lognormal(lognormal_hist(40_000, 3_000.0, 0.6))
    assert not g.censored
    assert g.censored_fraction == 0.0


def test_fit_refuses_rather_than_guessing():
    with pytest.raises(NotObservable, match="no such histogram"):
        fit_lognormal(None, "prompt length")
    empty = Histogram("h", {}, {1.0: 0.0, math.inf: 0.0}, 0.0, 0.0)
    with pytest.raises(NotObservable, match="no observation"):
        fit_lognormal(empty, "prompt length")
    # everything inside one bucket: no interior edge, so no shape
    one = Histogram("h", {}, {1.0: 0.0, 10.0: 100.0, math.inf: 100.0}, 100.0, 500.0)
    with pytest.raises(NotObservable, match="too coarse"):
        fit_lognormal(one, "prompt length")


def test_fit_mean_and_histogram_mean_are_separate_readings():
    h = lognormal_hist(40_000, 47_400.0, 0.81)
    f = fit_lognormal(h)
    assert f.mean_tokens == pytest.approx(h.mean())          # exact, from _sum
    assert f.fit_mean_tokens == pytest.approx(               # from the fit
        f.median_tokens * math.exp(f.sigma ** 2 / 2))


# ===========================================================================
# the Prometheus source, end to end
# ===========================================================================
def test_prometheus_estimate_recovers_the_synthetic_workload():
    est, _ = study_estimate()
    assert est.provenance.source == "prometheus"
    assert est.provenance.selector == 'model_name="Qwen/Qwen3-27B"'
    assert est.hours == pytest.approx(168.0)
    assert est.n_requests == pytest.approx(N_REQ)
    assert est.req_rate_s == pytest.approx(N_REQ / 604800.0, rel=1e-9)
    assert est.prompt.median_tokens == pytest.approx(PROMPT_MEDIAN, rel=0.02)
    assert est.prompt.sigma == pytest.approx(PROMPT_SIGMA, rel=0.02)
    assert est.output_mean_tokens == pytest.approx(
        OUT_MEDIAN * math.exp(OUT_SIGMA ** 2 / 2), rel=0.01)
    assert est.output.median_tokens == pytest.approx(OUT_MEDIAN, rel=0.02)
    assert est.e2e_mean_s == pytest.approx(18.6931, rel=1e-4)


def test_littles_law_on_a_scripted_series():
    """L = lambda * W, so W = mean(num_requests_running) / rate — computed
    here from a gauge series and a request count the test chose."""
    est, _ = study_estimate()
    mean_running = sum(RUNNING) / len(RUNNING)
    assert est.mean_running == pytest.approx(mean_running)
    assert est.p95_running == pytest.approx(
        float(__import__("numpy").percentile(RUNNING, 95)))
    assert est.little_w_s == pytest.approx(mean_running / est.req_rate_s)
    # the cycle is the session count over the rate; Z is what is left of it
    assert est.sessions == pytest.approx(est.p95_running)
    assert est.sessions_assumed is True
    assert est.cycle_s == pytest.approx(est.sessions / est.req_rate_s)
    assert est.think_time_s == pytest.approx(est.cycle_s - est.e2e_mean_s)


def test_sessions_override_changes_the_cycle_not_the_rate():
    est, _ = study_estimate(**{"sessions": 249.0})
    assert est.sessions == 249.0
    assert est.sessions_assumed is False
    assert est.cycle_s == pytest.approx(249.0 / est.req_rate_s)


def test_a_session_count_too_small_for_the_service_time_refuses():
    est, _ = study_estimate(**{"sessions": 0.001})
    assert est.think_time_s is None
    assert "no non-negative think time" in est.unobservable["think_time_s"]


# ===========================================================================
# the prefix cache: one observable, two unknowns
# ===========================================================================
def test_cache_savings_admits_exactly_two_readings():
    est, _ = study_estimate(turn_tokens=2_000.0, miss_rate=0.01)
    c = est.cache
    assert c.savings == pytest.approx(SAVINGS)
    C, T, f = c.mean_prompt_tokens, 2_000.0, 0.01
    assert c.miss_rate_given_turn == pytest.approx(
        ((1 - SAVINGS) * C - T) / (C - T))
    assert c.turn_tokens_given_miss == pytest.approx(
        C * (1 - SAVINGS - f) / (1 - f))
    # the study's 7-day reading, recovered from the counters
    assert c.miss_rate_given_turn == pytest.approx(0.093, abs=0.02)


def test_the_two_readings_are_each_labelled_with_their_assumption():
    est, _ = study_estimate()
    for text in (emit_table(est), emit_toml(est)):
        assert "IF the warm turn is" in text
        assert "IF the miss rate is" in text
        assert "cannot be separated" in text
    blob = emit_json(est)
    assert "cannot be separated" in json.loads(blob)["prefix_cache"]["note"]
    assert any("ONE observable over TWO unknowns" in c
               for c in json.loads(blob)["caveats"])


def test_a_server_without_prefix_cache_counters_says_so():
    prom = study_prom()
    for k in ("vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total",
              "vllm:prompt_tokens_cached_total"):
        prom.names.discard(k)
        prom.counters.pop(k, None)
    est, _ = study_estimate(prom)
    assert est.cache is None
    assert "prefix-cache counters" in est.unobservable["miss_rate"]
    assert "not observable from these metrics" in emit_toml(est)
    assert "miss_rate =" not in emit_toml(est)


# ===========================================================================
# gaps and resets
# ===========================================================================
def test_scrape_gaps_are_counted_and_named():
    est, _ = study_estimate()          # 10 steps out of 7d/5m = 2017
    assert est.gaps == 2017 - len(RUNNING)
    assert any("carry no num_requests_running sample" in c for c in est.caveats)


def test_counter_resets_are_reported_not_hidden():
    prom = study_prom(resets={"vllm:request_success_total": 2.0})
    est, _ = study_estimate(prom)
    assert "request_success_total" in est.resets
    assert "lower bound" in est.resets["request_success_total"]
    assert any("went backwards" in c for c in est.caveats)


# ===========================================================================
# source (b): a `ws metrics tail` archive
# ===========================================================================
def _zeroed(text: str) -> str:
    """The fixture with every counter and histogram member at zero — the
    'server just started' end of a window whose other end is the fixture."""
    keep = ("vllm:num_requests_running", "vllm:num_requests_waiting",
            "vllm:kv_cache_usage_perc")
    out = []
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            out.append(line)
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name in keep:
            out.append(line)
        else:
            head = line.rsplit(" ", 1)[0]
            out.append(f"{head} 0.0")
    return "\n".join(out)


def _archive(tmp_path: Path, dt: float = 3600.0) -> Path:
    lo = Snapshot(t_sent=0.0, rtt=0.0, samples=parse_text(_zeroed(DUMP)),
                  lines=_zeroed(DUMP).splitlines())
    hi = Snapshot(t_sent=dt, rtt=0.0, samples=parse_text(DUMP),
                  lines=DUMP.splitlines())
    p = tmp_path / "tail.jsonl"
    p.write_text("\n".join(s.to_json() for s in (lo, hi)) + "\n", encoding="utf-8")
    return p


def test_jsonl_archive_delta_matches_the_fixture(tmp_path):
    est = estimate(read_jsonl(_archive(tmp_path)))
    assert est.provenance.source == "jsonl"
    assert est.hours == pytest.approx(1.0)
    assert est.n_requests == pytest.approx(1900.0)     # the fixture's own count
    assert est.req_rate_s == pytest.approx(1900.0 / 3600.0)
    assert est.output_mean_tokens == pytest.approx(3894763.0 / 1900.0, rel=1e-6)
    assert est.e2e_mean_s == pytest.approx(35507.0905 / 1900.0, rel=1e-6)
    assert est.cache.savings == pytest.approx(54769778.0 / 58657565.0)
    # both endpoints carry the gauge, so it is a real (if short) series
    assert est.mean_running == pytest.approx(24.0)


def test_jsonl_archive_emits_a_readable_workload_block(tmp_path):
    est = estimate(read_jsonl(_archive(tmp_path)))
    block = emit_toml(est)
    parsed = tomllib.loads(block)["workload"]
    assert parsed["user_prompt_median_tokens"] > 0
    assert parsed["max_output_tokens"] > 0
    assert "warm_turn_tokens" in parsed


def test_jsonl_archive_needs_two_snapshots(tmp_path):
    p = tmp_path / "one.jsonl"
    p.write_text(Snapshot(0.0, 0.0, parse_text(DUMP)).to_json() + "\n",
                 encoding="utf-8")
    with pytest.raises(ValueError, match="needs >= 2"):
        read_jsonl(p)


# ===========================================================================
# source (c): one raw /metrics dump
# ===========================================================================
def test_metrics_text_gives_shape_but_refuses_every_rate():
    est = estimate(read_metrics_text(FIXTURE))
    assert est.hours is None
    assert est.prompt is not None                 # a distribution needs no clock
    assert est.cache is not None                  # a ratio needs no clock either
    assert est.req_rate_s is None
    assert est.little_w_s is None
    assert est.think_time_s is None
    assert "window length is unknown" in est.unobservable["request_rate"]
    assert "think_time_s" in est.unobservable
    assert any("CUMULATIVE SINCE SERVER START" in c for c in est.caveats)


# ===========================================================================
# what is never observable, whatever the source
# ===========================================================================
def test_the_unobservable_are_reasons_never_defaults():
    est, _ = study_estimate()
    for key in ("system_prefix_tokens", "subagent_ratio", "subagent_median_tokens",
                "subagent_sigma", "subagent_prefix_tokens", "sub_shares_prefix",
                "sessions_in_cache"):
        assert key in est.unobservable and len(est.unobservable[key]) > 20
    block = emit_toml(est)
    body = tomllib.loads(block)["workload"]
    for key in ("system_prefix_tokens", "subagent_ratio", "subagent_sigma",
                "subagent_median_tokens", "subagent_prefix_tokens",
                "sub_shares_prefix"):
        assert key not in body                     # a comment, not a default
        assert f"# {key}: not observable from these metrics" in block


# ===========================================================================
# rounding
# ===========================================================================
def test_every_emitted_number_is_rounded():
    est, _ = study_estimate()
    body = tomllib.loads(emit_toml(est))["workload"]
    # tokens: 3 significant figures, as an integer
    for key in ("user_prompt_median_tokens", "max_output_tokens",
                "warm_turn_tokens"):
        v = body[key]
        assert isinstance(v, int)
        assert v == int(float(f"{v:.3g}")), f"{key}={v} is not 3 s.f."
    assert body["user_prompt_sigma"] == round(body["user_prompt_sigma"], 2)
    assert body["think_time_s"] == round(body["think_time_s"], 1)
    assert isinstance(body["users"], int)
    blob = json.loads(emit_json(est))
    rate = blob["requests"]["rate_per_s"]
    assert rate == float(f"{rate:.2g}")           # rates: 2 significant figures
    assert blob["cycle"]["think_time_s"] == round(blob["cycle"]["think_time_s"], 1)


def test_an_unrounded_number_never_reaches_the_output():
    est, _ = study_estimate()
    exact_rate = repr(N_REQ / 604800.0)
    assert exact_rate not in emit_toml(est)
    assert exact_rate not in emit_json(est)
    assert exact_rate not in emit_table(est)


# ===========================================================================
# the firewall
# ===========================================================================
def test_no_raw_series_value_or_timestamp_is_emitted():
    """The employer-data firewall: every emitted number is an aggregate that
    has been rounded, so no timestamp-value pair from the source survives."""
    est, prom = study_estimate()
    outputs = (emit_toml(est), emit_json(est), emit_table(est))
    for text in outputs:
        # raw gauge readings, verbatim and rounded to more digits than we emit
        for v in RUNNING:
            assert repr(v) not in text, f"raw gauge value {v} leaked"
        assert "41.2718" not in text
        assert "13.7314" not in text
        # the evaluation instant and every step timestamp
        assert str(int(prom.now)) not in text
        for i in range(len(RUNNING)):
            assert repr(prom.now - 604800.0 + i * prom.step_s) not in text
        # raw counter totals and bucket counts
        assert repr(N_REQ) not in text
        prompt = prom.hists["vllm:request_prompt_tokens"]
        assert repr(prompt.sum) not in text
        for count in prompt.buckets.values():
            if count > 1000:                       # 0.0/1.0 are not identifying
                assert repr(count) not in text


def test_provenance_carries_no_time_at_all():
    est, _ = study_estimate()
    prov = json.loads(emit_json(est))["provenance"]
    assert set(prov) == {"source", "target", "range", "step", "selector",
                         "engine", "resolved", "missing"}
    assert prov["range"] == "7d"          # what the user typed, not an instant
    assert not any(isinstance(v, (int, float)) for v in prov.values())


# ===========================================================================
# --into
# ===========================================================================
def _config(tmp_path: Path) -> Path:
    p = tmp_path / "workingset.toml"
    p.write_text(RunConfig().dumps("toml"), encoding="utf-8")
    return p


def test_into_rewrites_only_the_workload_block(tmp_path):
    p = _config(tmp_path)
    before = p.read_text(encoding="utf-8")
    est, _ = study_estimate()
    after = merge_into(p, emit_toml(est))
    p.write_text(after, encoding="utf-8")

    for block in ("[deployment]", "[slo]", "[endpoint]", "[calibration]"):
        i, j = before.index(block), after.index(block)
        assert before[i:i + 200].split("[")[1] == after[j:j + 200].split("[")[1]
    old, new = tomllib.loads(before), tomllib.loads(after)
    for block in ("deployment", "slo", "endpoint", "calibration"):
        assert old[block] == new[block]
    assert new["schema_version"] == old["schema_version"]
    assert new["workload"] != old["workload"]
    assert new["workload"]["user_prompt_median_tokens"] == pytest.approx(
        47_400, rel=0.02)
    # and the result is still a config the package can read and price
    load_config(p).validate()


def test_into_appends_when_there_is_no_workload_block(tmp_path):
    p = tmp_path / "partial.toml"
    p.write_text('schema_version = 1\n\n[deployment]\nmodel = "27B"\n',
                 encoding="utf-8")
    est, _ = study_estimate()
    text = merge_into(p, emit_toml(est))
    assert tomllib.loads(text)["deployment"]["model"] == "27B"
    assert "user_prompt_median_tokens" in tomllib.loads(text)["workload"]


def test_into_refuses_a_block_the_schema_would_reject(tmp_path):
    p = _config(tmp_path)
    with pytest.raises(ValueError, match="unknown key workload"):
        merge_into(p, "[workload]\nnot_a_field = 1\n")


# ===========================================================================
# the CLI
# ===========================================================================
def test_cli_runs_the_metrics_text_source(capsys, tmp_path):
    assert ws_main(["workload", "--metrics-text", str(FIXTURE),
                    "--emit", "toml"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("[workload]")
    assert tomllib.loads(out)["workload"]["max_output_tokens"] > 0


def test_cli_json_is_parseable_and_into_writes_to_stderr(tmp_path, capsys):
    p = _config(tmp_path)
    assert ws_main(["workload", "--metrics-text", str(FIXTURE), "--json",
                    "--into", str(p)]) == 0
    cap = capsys.readouterr()
    assert "rewrote the [workload] block" in cap.err
    blob = json.loads(cap.out)                     # stdout stayed pure JSON
    assert blob["provenance"]["source"] == "metrics-text"
    assert tomllib.loads(p.read_text(encoding="utf-8"))["workload"]


def test_cli_needs_exactly_one_source(capsys):
    assert ws_main(["workload"]) == 2
    assert "exactly one source" in capsys.readouterr().err
    assert ws_main(["workload", "--metrics-text", str(FIXTURE),
                    "--jsonl", "x.jsonl"]) == 2


def test_cli_help_documents_every_source_and_the_caveats(capsys):
    with pytest.raises(SystemExit):
        ws_main(["workload", "--help"])
    raw = capsys.readouterr().out
    help_text = " ".join(raw.split())        # argparse wraps; the words matter
    for flag in ("--prometheus", "--jsonl", "--metrics-text", "--range", "--step",
                 "--selector", "--auth-header", "--ca-bundle", "--insecure",
                 "--assume-turn-tokens", "--assume-miss-rate", "--sessions",
                 "--emit", "--into", "--engine"):
        assert flag in help_text
    for caveat in ("CUMULATIVE SINCE SERVER START",
                   "ONE observable over TWO unknowns",
                   "sessions-in-cache is not observable from these metrics",
                   "LOWER BOUND", "ROUNDED AGGREGATES",
                   "increase() over the whole --range"):
        assert caveat in help_text, caveat


def test_cli_reports_a_dead_prometheus_as_a_message(capsys):
    assert ws_main(["workload", "--prometheus", "http://127.0.0.1:1/",
                    "--range", "1h"]) == 2
    assert "prometheus" in capsys.readouterr().err


def test_auth_header_must_be_a_header():
    from workingset.workload import _headers

    class A:
        auth_header = ["Authorization: Bearer x", "X-Scope: team"]
    assert _headers(A()) == {"Authorization": "Bearer x", "X-Scope": "team"}

    class B:
        auth_header = ["nonsense"]
    with pytest.raises(ValueError, match="Name: value"):
        _headers(B())


# ===========================================================================
# the HTTP client
# ===========================================================================
def test_prometheus_client_raises_on_an_unsuccessful_body():
    def handle(request):
        return httpx.Response(200, json={"status": "error", "error": "bad query"})

    pc = PrometheusClient("http://prom.test/api/v1",
                          client=httpx.Client(transport=httpx.MockTransport(handle)))
    assert pc.base == "http://prom.test"           # the /api/v1 suffix is stripped
    with pytest.raises(ValueError, match="bad query"):
        pc.query("up")


def test_metric_names_falls_back_when_match_is_rejected():
    seen = []

    def handle(request):
        seen.append(dict(request.url.params))
        if "match[]" in request.url.params:
            return httpx.Response(422, json={"status": "error", "error": "no"})
        return httpx.Response(200, json={"status": "success",
                                         "data": ["vllm:prompt_tokens_total"]})

    pc = PrometheusClient("http://prom.test",
                          client=httpx.Client(transport=httpx.MockTransport(handle)))
    assert pc.metric_names(match='{__name__=~"vllm:.*"}') == {
        "vllm:prompt_tokens_total"}
    assert len(seen) == 2                          # tried narrowed, then not


def test_non_monotone_increase_buckets_are_made_cumulative():
    """`increase()` extrapolates, so a bucket can come back a hair BELOW the
    one under it; a cumulative histogram that dips is not one."""
    from workingset.workload import _bucket_result
    result = [{"metric": {"le": "10.0"}, "value": [0, "100.0"]},
              {"metric": {"le": "20.0"}, "value": [0, "99.5"]},
              {"metric": {"le": "+Inf"}, "value": [0, "120.0"]}]
    got = _bucket_result(result)
    assert got == {10.0: 100.0, 20.0: 100.0, math.inf: 120.0}
