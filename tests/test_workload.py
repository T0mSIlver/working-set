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
import numpy as np
import pytest

from workingset.cli import main as ws_main
from workingset.config import RunConfig, load_config
from workingset.metrics.parse import Histogram, parse_text
from workingset.metrics.sampler import Snapshot
from workingset.workload import (PROMQL, NotObservable, PrometheusClient,
                                 Provenance, Reading, emit_json, emit_table,
                                 emit_toml, estimate, fit_lognormal,
                                 merge_into, parse_duration, promql,
                                 read_jsonl, read_metrics_text, read_prometheus)

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


# the identifying labels a real deployment's series carry. Every response
# below wears them, so the firewall test can assert that none of them — nor
# any raw value they hang off — reaches an emitted block.
LABELS = {"instance": "vllm-prod-7.internal.example:8000",
          "pod": "vllm-qwen3-27b-tp4-847d9c6b5-x2knq",
          "model_name": "Qwen/Qwen3-27B",
          "job": "vllm-serving-prod",
          "namespace": "ml-inference-prod"}


class FakeProm:
    """Answers the query shapes `PROMQL` builds, and nothing else.

    Every query is recorded in `.queries` as `(query, time param)` and every
    range window in `.windows`, so a test can assert the PromQL that was
    actually built and the instant it was evaluated at, rather than the ones
    it expected to be built.

    `resets` is keyed by the SERIES a reset query names, so a counter is
    `vllm:request_success_total` and a histogram is its `_count` member.
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
        self.queries: list[tuple[str, str | None]] = []
        self.windows: list[tuple[float, float]] = []

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self._handle),
                            base_url="http://prom.test")

    @property
    def promql(self) -> list[str]:
        return [q for q, _ in self.queries]

    # ---- the handler --------------------------------------------------
    def _handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/label/__name__/values"):
            return self._ok(sorted(self.names))
        q = request.url.params.get("query", "")
        self.queries.append((q, request.url.params.get("time")))
        if q == "time()":
            # the REAL shape: a `scalar` result is the bare pair [t, "v"],
            # not a list of {metric, value} series
            return self._ok({"resultType": "scalar",
                             "result": [self.now, repr(self.now)]})
        if q == "vector(time())":
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
            start = float(request.url.params["start"])
            self.windows.append((start, float(request.url.params["end"])))
            vals = self.gauge.get(base)
            if vals is None:
                return self._matrix([])
            return self._matrix([(start + i * self.step_s, v)
                                 for i, v in enumerate(vals)])
        if "resets(" in q:
            v = self.resets.get(raw)
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
                         "result": [{"metric": {**LABELS, **lbl},
                                     "value": [self.now, repr(float(v))]}
                                    for lbl, v in pairs]})

    def _matrix(self, points) -> httpx.Response:
        return self._ok({"resultType": "matrix",
                         "result": [{"metric": dict(LABELS),
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


def test_resets_are_summed_over_engines_not_maxed():
    """Two engines restarting once each is two restarts in the totals; `max`
    would report one and hide half a data-parallel deployment's damage."""
    q = PROMQL(range="7d")
    assert q.resets("vllm:request_success_total") == (
        "sum(resets(vllm:request_success_total[7d]))")
    assert "max(" not in q.resets("vllm:request_success_total")


def test_histograms_get_a_reset_query_too():
    """A restart inside a FITTED family corrupts the fit; `resets()` needs a
    plain counter, and the family's `_count` is one."""
    q = PROMQL(range="7d")
    assert q.hist_resets("vllm:request_prompt_tokens") == (
        "sum(resets(vllm:request_prompt_tokens_count[7d]))")


def test_matcher_strips_one_brace_pair_not_every_brace():
    braced = PROMQL(selector='{model_name="m"}').gauge("vllm:x")
    bare = PROMQL(selector='model_name="m"').gauge("vllm:x")
    assert braced == bare == 'sum(vllm:x{model_name="m"})'
    # a selector whose own value ends in a brace keeps it
    assert PROMQL(selector='model_name=~"m\\{2\\}"').gauge("vllm:x") == (
        'sum(vllm:x{model_name=~"m\\{2\\}"})')


@pytest.mark.parametrize("selector", ['engine="0"', 'model_name="m",engine="1"',
                                      'engine=~"[01]"', '{engine!="2"}'])
def test_engine_twice_is_refused_locally_not_by_a_400(selector):
    """`{engine="1",engine="0"}` is not a narrower selection, it is invalid
    PromQL; the error belongs here, not three round trips later."""
    with pytest.raises(ValueError, match="already constrains `engine`"):
        PROMQL(selector=selector, engine="0").gauge("vllm:x")
    # without --engine the same selector is perfectly fine
    assert "engine" in PROMQL(selector=selector).gauge("vllm:x")


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
    for q in prom.promql:
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


def test_a_scalar_result_does_not_crash_the_run():
    """`time()` returns Prometheus's `scalar` shape — the bare pair
    [t, "v"] — and reading it as a vector used to raise AttributeError on a
    float, uncaught, out of a helper."""
    from workingset.workload import _scalar
    assert _scalar([1_700_000_000.0, "1700000000"]) == pytest.approx(1.7e9)
    assert _scalar([{"metric": {}, "value": [0, "42"]}]) == 42.0
    assert _scalar([]) is None
    assert _scalar([{"metric": {}}]) is None
    assert _scalar(["nonsense", 3]) is None


def test_the_clock_is_asked_for_a_vector_and_the_run_survives_a_scalar():
    """End to end against a fake that answers `time()` with the real scalar
    shape: whichever form the query takes, the window still resolves."""
    prom = study_prom()
    rd = read_prometheus("http://prom.test", range="7d", step="5m",
                         client=prom.client())          # no `now=`: asks the server
    assert "vector(time())" in prom.promql
    assert prom.windows == [(prom.now - 604800.0, prom.now)]
    assert estimate(rd).hours == pytest.approx(168.0)

    # and a server that only understands the bare form is still usable
    from workingset.workload import PrometheusClient
    pc = PrometheusClient("http://prom.test", client=prom.client())
    assert pc.query("time()") == [prom.now, repr(prom.now)]
    from workingset.workload import _scalar as s
    assert s(pc.query("time()")) == pytest.approx(prom.now)


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


def sampled_hist(seed: int, n: int, median: float, sigma: float) -> Histogram:
    """A histogram accumulated from actual DRAWS, not from the exact CDF: the
    bucket counts carry real sampling noise, which is the only thing the
    weighting exists to handle."""
    rng = np.random.default_rng(seed)
    x = rng.lognormal(math.log(median), sigma, n)
    cum = {float(b): float((x <= b).sum()) for b in SIZE_BOUNDS}
    cum[math.inf] = float(n)
    return Histogram("h", {}, cum, float(n), float(x.sum()))


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_fit_recovers_a_noisily_sampled_lognormal(seed):
    """The exact-CDF test above cannot see a weighting bug: every point sits
    on the line by construction. These buckets come from 40,000 random draws,
    so the extreme edges (F ~ 5e-5, where one request moves the quantile a
    long way) are wrong by exactly as much as sampling makes them."""
    f = fit_lognormal(sampled_hist(seed, 40_000, 47_400.0, 0.81))
    assert f.median_tokens == pytest.approx(47_400.0, rel=0.01)
    assert f.sigma == pytest.approx(0.81, rel=0.02)


def test_the_fit_weights_the_edges_by_how_much_they_know():
    """Unweighted, the sparsest edge votes as loudly as the median and the
    recovered sigma wanders; weighted, it does not."""
    def unweighted(h):
        nd = statistics.NormalDist()
        xs, ys = [], []
        for b in SIZE_BOUNDS:
            p = h.buckets[float(b)] / h.observations
            if 0.0 < p < 1.0:
                xs.append(math.log(b))
                ys.append(nd.inv_cdf(p))
        slope, _ = np.polyfit(np.asarray(ys), np.asarray(xs), 1)
        return float(slope)

    worst_w = worst_u = 0.0
    for seed in range(12):
        h = sampled_hist(seed, 40_000, 47_400.0, 0.81)
        worst_w = max(worst_w, abs(fit_lognormal(h).sigma / 0.81 - 1))
        worst_u = max(worst_u, abs(unweighted(h) / 0.81 - 1))
    assert worst_w < worst_u
    assert worst_w < 0.02


def test_a_fit_through_too_few_edges_says_so():
    """Two edges make a line that fits exactly; the residual is then not
    evidence of anything."""
    h = Histogram("h", {}, {100.0: 10.0, 1000.0: 60.0, 10000.0: 95.0,
                            math.inf: 100.0}, 100.0, 100_000.0)
    est = estimate(Reading(provenance=Provenance("test", "t"),
                           histograms={"request_prompt_tokens_hist": h},
                           counters={"request_success_total": 100.0},
                           seconds=3600.0))
    assert est.prompt.n_points == 3
    assert any("only 3 bucket edges" in c or "only 3 " in c for c in est.caveats)
    assert any("fitted through only" in c for c in est.caveats)


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


def test_censoring_is_uncertainty_not_a_direction():
    """Censoring was described as making sigma a LOWER bound, which would
    license reading it as conservative. It is not a bound in either
    direction: the fit rests on the interior edges alone, and a workload
    whose long-context tail is heavier than log-normal — which is what
    agentic traffic looks like — fits WIDER than its body while overflowing
    the top bucket. Body sigma 0.46, fitted 0.8+."""
    rng = np.random.default_rng(7)
    n = 40_000
    body = rng.lognormal(math.log(30_000.0), 0.46, int(n * 0.8))
    tail = 60_000.0 * (rng.pareto(0.6, int(n * 0.2)) + 1)
    x = np.concatenate([body, tail])
    cum = {float(b): float((x <= b).sum()) for b in SIZE_BOUNDS}
    cum[math.inf] = float(n)
    h = Histogram("h", {}, cum, float(n), float(x.sum()))

    f = fit_lognormal(h)
    assert f.censored and f.censored_fraction > 0.05
    assert f.sigma > 0.46 * 1.5, "the reproduction: sigma came back HIGH"

    est = estimate(Reading(provenance=Provenance("test", "t"), seconds=3600.0,
                           counters={"request_success_total": float(n)},
                           histograms={"request_prompt_tokens_hist": h}))
    note = [c for c in est.caveats if "exceeded the largest finite bucket" in c]
    assert len(note) == 1
    assert "NEITHER direction" in note[0]
    for text in (emit_toml(est), emit_table(est), emit_json(est)):
        assert "sigma is a LOWER bound" not in text
        assert "sigma is a lower bound" not in text
        assert "a lower bound when the tail is censored" not in text


def test_finished_request_histograms_can_count_the_requests():
    """A server exporting the latency families but not request_success_total
    is not a server whose request count is unknowable: each of these takes one
    observation per FINISHED request."""
    hg = lognormal_hist(1_234, 300.0, 0.8)
    he = Histogram("e", {}, lognormal_buckets(1_234, 9.0, 1.1, E2E_BOUNDS),
                   1_234.0, 18.0 * 1_234)
    est = estimate(Reading(provenance=Provenance("test", "t"), seconds=1_234.0,
                           counters={"generation_tokens_total": 500_000.0},
                           histograms={"request_generation_tokens_hist": hg,
                                       "e2e_hist": he}))
    assert est.n_requests == pytest.approx(1_234)
    assert "request_generation_tokens" in est.n_requests_source
    assert est.req_rate_s == pytest.approx(1.0)
    assert "n_requests" not in est.unobservable


def test_disagreeing_request_counts_are_reported():
    hg = lognormal_hist(1_000, 300.0, 0.8)
    he = Histogram("e", {}, lognormal_buckets(1_500, 9.0, 1.1, E2E_BOUNDS),
                   1_500.0, 18.0 * 1_500)
    est = estimate(Reading(provenance=Provenance("test", "t"), seconds=1_000.0,
                           histograms={"request_generation_tokens_hist": hg,
                                       "e2e_hist": he}))
    assert est.n_requests == 1_500              # the largest
    assert any("disagree on how many requests" in c for c in est.caveats)


def test_no_finished_request_series_at_all_is_still_a_refusal():
    est = estimate(Reading(provenance=Provenance("test", "t"), seconds=100.0,
                           counters={"prompt_tokens_total": 5.0}))
    assert est.n_requests is None
    assert "no request_success_total" in est.unobservable["n_requests"]


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
def mixture_hist(n: int, ratio: float, main=(50_000.0, 0.5),
                 sub=(5_000.0, 0.5)) -> Histogram:
    """A prompt histogram from a MIXTURE of two request classes, which is what
    `request_prompt_tokens` actually accumulates."""
    nd = statistics.NormalDist()
    p_sub = ratio / (1.0 + ratio)
    cum: dict[float, float] = {}
    for b in SIZE_BOUNDS:
        f_main = nd.cdf((math.log(b) - math.log(main[0])) / main[1])
        f_sub = nd.cdf((math.log(b) - math.log(sub[0])) / sub[1])
        cum[float(b)] = float(round(n * ((1 - p_sub) * f_main + p_sub * f_sub)))
    cum[math.inf] = float(n)
    mean = ((1 - p_sub) * main[0] * math.exp(main[1] ** 2 / 2)
            + p_sub * sub[0] * math.exp(sub[1] ** 2 / 2))
    return Histogram("h", {}, cum, float(n), mean * n)


def test_a_mixed_prompt_histogram_is_not_the_configs_user_prompt():
    """`request_prompt_tokens` has no request-class label, so its fit is over
    main-user AND subagent requests. The config's `user_prompt_*` name ONE
    component of that mixture, and the model re-mixes `subagent_*` on top:
    assigning the aggregate hands it a distribution wider than the one
    measured, then widens it again. An equal mixture of 50k and 5k prompts at
    sigma 0.5 fits as a single log-normal near median 15k, sigma > 1.2 —
    neither component."""
    prom = study_prom()
    prom.hists["vllm:request_prompt_tokens"] = mixture_hist(int(N_REQ), 1.0)
    est, _ = study_estimate(prom)

    assert est.prompt is not None                    # the aggregate is reported
    assert est.prompt.median_tokens == pytest.approx(15_200, rel=0.15)
    assert est.prompt.sigma > 1.1
    assert not (0.45 < est.prompt.sigma < 0.55)      # neither component's sigma
    assert est.prompt_assignable is False

    block = emit_toml(est)
    body = tomllib.loads(block)["workload"]
    assert "user_prompt_median_tokens" not in body
    assert "user_prompt_sigma" not in body
    assert "prompt_median_all" in block and "prompt_sigma_all" in block
    for key in ("user_prompt_median_tokens", "user_prompt_sigma"):
        assert "mixes main-user and subagent requests" in est.unobservable[key]
        assert "--single-class" in est.unobservable[key]
    blob = json.loads(emit_json(est))["prompt_tokens_all_requests"]
    assert blob["prompt_median_all"] == _round3(est.prompt.median_tokens)
    assert blob["assignable_to_user_prompt"] is False


def _round3(x: float) -> int:
    return int(round(float(f"{x:.3g}")))


def test_single_class_asserts_the_mixture_away_and_unlocks_the_assignment():
    est, _ = study_estimate(**{"single_class": True})
    assert est.prompt_assignable is True
    body = tomllib.loads(emit_toml(est))["workload"]
    assert body["user_prompt_median_tokens"] == pytest.approx(47_400, rel=0.02)
    assert body["user_prompt_sigma"] == pytest.approx(0.81, rel=0.02)
    assert "user_prompt_median_tokens" not in est.unobservable
    assert "asserted single-class by --single-class" in emit_toml(est)


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
    assert est.p95_running == pytest.approx(float(np.percentile(RUNNING, 95)))
    assert est.little_w_s == pytest.approx(mean_running / est.req_rate_s)


def test_the_decoder_p95_is_a_diagnostic_and_nothing_is_derived_from_it():
    """A high quantile of an instantaneous execution count does not bound the
    time-average SESSION population in either direction: ten sessions working
    100 s and parked 900 s put the p95 at ten while averaging one, and
    `p95 / lambda` would then report a cycle — and a think time — several
    times the truth, in the unsafe direction."""
    est, _ = study_estimate()
    assert est.p95_running is not None            # still reported
    assert est.sessions is None
    assert est.cycle_s is None
    assert est.think_time_s is None
    assert "no quantile of it bounds" in est.unobservable["sessions"]
    assert "--sessions N" in est.unobservable["sessions"]
    for raw in (emit_toml(est), emit_table(est)):
        text = " ".join(raw.split())          # both wrap prose at 74 columns
        assert "think_time_s =" not in text
        assert ("bounds a session count in neither direction" in text
                or "bounds the session population in neither direction" in text)
    # and the diagnostic says what it is, wherever it is printed
    assert "DIAGNOSTIC" in emit_table(est)
    assert "p95_running_diagnostic" in emit_json(est)


def test_sessions_override_changes_the_cycle_not_the_rate():
    est, _ = study_estimate(**{"sessions": 249.0})
    assert est.sessions == 249.0
    assert est.cycle_s == pytest.approx(249.0 * 1.10 / est.req_rate_s)


def test_the_cycle_carries_the_subagent_ratio_the_model_defines():
    """`model.closed_request_rate` fixes `users = lam_total (Z + R) / (1 + r)`.
    The counters see lam_total over every class, so a cycle per SESSION is
    sessions x (1 + r) / lam_total; dividing by the raw rate would price each
    session as a single request stream."""
    from workingset import model as M

    base, _ = study_estimate(**{"sessions": 100.0, "subagent_ratio": 0.0})
    assert base.cycle_s == pytest.approx(100.0 / base.req_rate_s)

    est, _ = study_estimate(**{"sessions": 100.0, "subagent_ratio": 0.35,
                               "subagent_ratio_source": "--subagent-ratio"})
    assert est.cycle_s == pytest.approx(base.cycle_s * 1.35)
    assert est.think_time_s == pytest.approx(est.cycle_s - est.e2e_mean_s)
    # the model's own identity, run forwards on what we emitted
    lam_total = est.req_rate_s
    assert lam_total * (est.think_time_s + est.e2e_mean_s) / 1.35 == \
        pytest.approx(100.0)
    # and the assumption is printed wherever the number goes
    for text in (emit_toml(est), emit_table(est), emit_json(est)):
        assert "0.35" in text
        assert "--subagent-ratio" in text
    assert M.Workload().sub_ratio == pytest.approx(0.10)   # the module default


def test_a_zero_think_time_is_an_answer_not_a_refusal():
    """A fully autonomous fleet issues its next request the instant the last
    one lands; only a NEGATIVE Z is impossible."""
    est, _ = study_estimate()
    exact = est.e2e_mean_s * est.req_rate_s / 1.10        # cycle == R
    est, _ = study_estimate(**{"sessions": exact})
    assert est.think_time_s == pytest.approx(0.0, abs=1e-6)
    assert "think_time_s" not in est.unobservable
    assert tomllib.loads(emit_toml(est))["workload"]["think_time_s"] == 0.0


def test_no_session_count_writes_no_config_users():
    """`users` is the closed-loop operating point `ws predict` prices, and it
    is a SESSION count nothing here measures."""
    est, _ = study_estimate()
    block = emit_toml(est)
    body = tomllib.loads(block)["workload"]
    assert "users" not in body
    assert "think_time_s" not in body
    assert "# users: not observable from these metrics" in block
    assert "--sessions" in block
    # the diagnostic is still reported, as a comment naming the number
    assert f"{est.p95_running:,.0f}" in block


def test_an_explicit_session_count_does_write_users():
    est, _ = study_estimate(**{"sessions": 249.0})
    block = emit_toml(est)
    assert tomllib.loads(block)["workload"]["users"] == 249
    assert "as given by --sessions" in block
    assert "sessions given" in block


def test_an_emitted_block_never_under_prices_a_config_by_default(tmp_path):
    """The end-to-end shape of the same rule: `--into` without `--sessions`
    leaves the config's own `users` alone rather than lowering it."""
    p = _config(tmp_path)
    before = tomllib.loads(p.read_text(encoding="utf-8"))["workload"]["users"]
    est, _ = study_estimate()
    p.write_text(merge_into(p, est), encoding="utf-8")
    after = tomllib.loads(p.read_text(encoding="utf-8"))["workload"]
    assert after["users"] == before        # carried over, not reset
    assert load_config(p).workload.users == before


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


def test_the_identity_checks_that_both_counters_saw_the_same_requests():
    """The savings is priced against the mean prompt C, which is only
    legitimate if the cache counters queried C tokens per request."""
    est, _ = study_estimate()
    assert not any("not seeing the same requests" in c for c in est.caveats)

    prom = study_prom()
    prom.counters["vllm:prefix_cache_queries_total"] *= 0.7    # some traffic bypassed it
    prom.counters["vllm:prefix_cache_hits_total"] *= 0.7
    est, _ = study_estimate(prom)
    note = [c for c in est.caveats if "not seeing the same requests" in c]
    assert len(note) == 1
    assert "prefix_cache_queries_total / requests" in note[0]
    assert "request_prompt_tokens" in note[0]


def test_the_queue_cross_check_is_actually_run():
    """R - W is time spent not executing, and the queue histogram measures
    exactly that; the docstring promised the comparison, so make it."""
    est, _ = study_estimate()
    assert est.queue_mean_s is None            # not exported by the fake

    prom = study_prom()
    prom.names.update({"vllm:request_queue_time_seconds_bucket",
                       "vllm:request_queue_time_seconds_sum",
                       "vllm:request_queue_time_seconds_count"})
    # W - R is large and negative here, so any plausible queue time disagrees
    prom.hists["vllm:request_queue_time_seconds"] = Histogram(
        "q", {}, lognormal_buckets(int(N_REQ), 0.4, 0.9, E2E_BOUNDS),
        N_REQ, 0.5 * N_REQ)
    est, _ = study_estimate(prom)
    assert est.queue_mean_s == pytest.approx(0.5)
    note = [c for c in est.caveats if "against a measured mean queue time" in c]
    assert len(note) == 1
    assert "NOT consistent" in note[0]


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
    assert "LOWER bound" in est.resets["request_success_total"]
    assert any("went backwards" in c for c in est.caveats)


def test_a_reset_inside_a_fitted_histogram_is_reported():
    """The prompt and generation families are the two this command FITS: a
    restart inside one mixes two distributions into one bucket CDF."""
    prom = study_prom(resets={"vllm:request_prompt_tokens_count": 1.0})
    est, _ = study_estimate(prom)
    assert "sum(resets(vllm:request_prompt_tokens_count" in " ".join(prom.promql)
    assert "request_prompt_tokens_hist" in est.resets
    assert any("FITTED histogram" in c for c in est.caveats)
    # and every fitted family is asked, not just the one that answered
    for base in ("vllm:request_prompt_tokens", "vllm:request_generation_tokens"):
        assert f"sum(resets({base}_count" in " ".join(prom.promql)


def test_nothing_is_queried_that_nothing_consumes():
    """A fetched-and-unused series costs a round trip and, worse, shows up in
    `missing` as though its absence mattered."""
    _, prom = study_estimate()
    joined = " ".join(prom.promql)
    assert "time_to_first_token" not in joined
    assert "num_requests_waiting" not in joined
    est, _ = study_estimate()
    assert "ttft_hist" not in est.provenance.missing
    assert "requests_waiting" not in est.provenance.missing


def test_every_query_is_evaluated_at_the_same_instant():
    """Left to Prometheus's own `now`, each round trip pushes the next
    query's window later and the aggregates stop describing one window."""
    _, prom = study_estimate()
    instant = [(q, t) for q, t in prom.queries
               if "increase(" in q or "resets(" in q]
    assert len(instant) > 10
    assert {t for _, t in instant} == {repr(prom.now)}
    # and the gauge's range ends at that same instant
    assert prom.windows == [(prom.now - 604800.0, prom.now)]


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
    est = estimate(read_jsonl(_archive(tmp_path)), single_class=True)
    block = emit_toml(est)
    parsed = tomllib.loads(block)["workload"]
    assert parsed["user_prompt_median_tokens"] > 0
    assert parsed["max_output_tokens"] > 0
    assert "warm_turn_tokens" in parsed


def test_jsonl_resets_carry_no_raw_counter_readings(tmp_path):
    """`WindowDelta.invalid` is written for a terminal and quotes the counter
    on both sides of the jump; those are raw series values, and the
    classification is all a reader needs."""
    lo = Snapshot(t_sent=0.0, rtt=0.0, samples=parse_text(DUMP),
                  lines=DUMP.splitlines())
    restarted = DUMP.replace("} 1835.0", "} 11.0").replace(
        "vllm:prompt_tokens_total{engine=\"0\",model_name=\"Qwen/Qwen3-27B\"} ",
        "vllm:prompt_tokens_total{engine=\"0\",model_name=\"Qwen/Qwen3-27B\"} 7.0 #")
    hi = Snapshot(t_sent=3600.0, rtt=0.0, samples=parse_text(restarted),
                  lines=restarted.splitlines())
    p = tmp_path / "reset.jsonl"
    p.write_text("\n".join(s.to_json() for s in (lo, hi)) + "\n", encoding="utf-8")

    est = estimate(read_jsonl(p))
    assert est.resets, "the archive spans a restart"
    joined = " ".join(est.resets.values())
    assert "counter reset" in joined
    assert not re.search(r"\d{3,}", joined), f"a raw reading leaked: {joined}"
    for text in (emit_toml(est), emit_json(est), emit_table(est)):
        for raw in ("1835.0", "11.0", "58657565"):
            assert raw not in text


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


def test_one_scrape_is_not_a_window_of_the_gauge():
    """The fixture's `num_requests_running` reads 24. One reading of a gauge
    is the concurrency at that instant, not a mean, a p95 or a session count —
    and `users = 24` in a config would be exactly the invented number this
    command refuses to produce."""
    est = estimate(read_metrics_text(FIXTURE))
    assert est.mean_running is None
    assert est.p95_running is None
    assert est.sessions is None
    assert est.cycle_s is None
    for key in ("concurrency", "users"):
        assert "one INSTANT of the gauge" in est.unobservable[key]
    assert "sessions" in est.unobservable
    block = emit_toml(est)
    assert "users" not in tomllib.loads(block)["workload"]
    assert "# users: not observable from these metrics" in block
    # and no caveat bounding a session count that was never used
    assert not any("sessions defaults to" in c for c in est.caveats)


def test_one_scrape_still_refuses_when_sessions_is_given():
    """`--sessions` supplies the population, but the rate it would be divided
    by still does not exist here."""
    est = estimate(read_metrics_text(FIXTURE), sessions=249.0)
    assert est.cycle_s is None
    assert est.think_time_s is None
    assert "think_time_s" in est.unobservable
    # the population was asserted by the caller, so the block records it
    # rather than assigning it AND calling it unobservable in the same breath
    block = emit_toml(est)
    assert tomllib.loads(block)["workload"]["users"] == 249
    assert "# users: not observable" not in block
    assert "concurrency: not observable" in block


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
    est, _ = study_estimate(**{"single_class": True, "sessions": 249.0})
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
    has been rounded, so no timestamp-value pair from the source survives —
    and no label off the series it was read from."""
    est, prom = study_estimate()
    outputs = (emit_toml(est), emit_json(est), emit_table(est))
    for text in outputs:
        # the identifying labels every real series carries. `model_name` is
        # the exception and only because the USER typed it into --selector;
        # nothing reaches the output by having been read off a series.
        for label, value in LABELS.items():
            if label == "model_name":
                continue
            assert value not in text, f"the {label} label leaked"
            assert label not in text
        # raw gauge readings, verbatim and to more digits than we emit
        for v in RUNNING:
            assert repr(v) not in text, f"raw gauge value {v} leaked"
        assert "41.2718" not in text
        assert "13.7314" not in text
        # the evaluation instant and every step timestamp
        assert str(int(prom.now)) not in text
        for i in range(len(RUNNING)):
            assert repr(prom.now - 604800.0 + i * prom.step_s) not in text
        # raw counter totals, histogram sums and bucket counts
        assert repr(N_REQ) not in text
        for name, h in prom.hists.items():
            assert repr(h.sum) not in text, f"{name} _sum leaked"
            assert repr(h.count) not in text, f"{name} _count leaked"
            for count in h.buckets.values():
                if count > 1000:                   # 0.0/1.0 are not identifying
                    assert repr(count) not in text
        for name, v in prom.counters.items():
            if v > 1000:
                assert repr(v) not in text, f"{name} leaked"


def test_the_endpoint_hostname_never_reaches_an_emitted_block():
    """The Prometheus URL is the deployment's identity, and an emitted block
    is meant to be pasted into a config other people read. What the user
    typed into --selector is theirs to repeat; what this command learned by
    dialling is used and then forgotten."""
    url = "https://prometheus.ml-inference-prod.corp.example.internal:9090"
    prom = study_prom()
    rd = read_prometheus(url, range="7d", step="5m",
                         selector='model_name="Qwen/Qwen3-27B"',
                         now=prom.now, client=prom.client())
    est = estimate(rd)
    assert est.provenance.target == ""
    for text in (emit_toml(est), emit_json(est), emit_table(est)):
        assert "prometheus.ml-inference-prod" not in text
        assert "corp.example.internal" not in text
        assert "9090" not in text
        assert "prometheus" in text            # the SOURCE KIND still shows
        assert "7d" in text                    # and so does what was asked for
    assert json.loads(emit_json(est))["provenance"]["target"] == ""


def test_a_local_file_path_is_not_echoed_into_a_shared_block(tmp_path):
    """An emitted block is meant to be pasted into a config other people
    read; the archive's name identifies it, /home/<someone>/... does not."""
    p = _archive(tmp_path)
    est = estimate(read_jsonl(p))
    assert est.provenance.target == "tail.jsonl"
    for text in (emit_toml(est), emit_json(est), emit_table(est)):
        assert str(tmp_path) not in text
        assert "tail.jsonl" in text
    est = estimate(read_metrics_text(FIXTURE))
    assert est.provenance.target == "vllm_metrics_v1.txt"
    assert str(FIXTURE.parent) not in emit_json(est)


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


# A config whose [workload] is HAND-TUNED, no key at its dataclass default.
# Starting a preservation test from the defaults proves nothing: a dropped
# key and a preserved one read back the same.
TUNED = """\
schema_version = 1

[deployment]
model = "27B"
gpu = "H200"
tensor_parallel = 4
max_model_len = 180000

# the workload the team measured by hand in August
[workload]
system_prefix_tokens = 54321
user_prompt_median_tokens = 41000
user_prompt_sigma = 0.77
warm_turn_tokens = 3500
think_time_s = 42.5
subagent_ratio = 0.7
subagent_median_tokens = 9100
subagent_sigma = 0.95
subagent_prefix_tokens = 3300
sub_shares_prefix = true
miss_rate = 0.05
max_output_tokens = 640
users = 249

# the SLO the team committed to
[slo]
ttft_budget_s = 8.0
percentile = 99

[calibration]
mfu = 0.42
"""


def _tuned(tmp_path: Path, text: str = TUNED) -> Path:
    p = tmp_path / "tuned.toml"
    p.write_text(text, encoding="utf-8", newline="")   # no CRLF translation
    return p


def test_into_rewrites_only_the_workload_block(tmp_path):
    p = _config(tmp_path)
    before = p.read_text(encoding="utf-8")
    est, _ = study_estimate(**{"single_class": True})
    after = merge_into(p, est)
    p.write_text(after, encoding="utf-8")

    # whole surrounding blocks, compared textually
    for block in ("[deployment]", "[slo]", "[endpoint]", "[calibration]"):
        assert _block_text(before, block) == _block_text(after, block)
    old, new = tomllib.loads(before), tomllib.loads(after)
    for block in ("deployment", "slo", "endpoint", "calibration"):
        assert old[block] == new[block]
    assert new["schema_version"] == old["schema_version"]
    assert new["workload"] != old["workload"]
    assert new["workload"]["user_prompt_median_tokens"] == pytest.approx(
        47_400, rel=0.02)
    load_config(p).validate()


def _block_text(text: str, header: str) -> str:
    """From a table header to the line before the next one, verbatim."""
    lines = text.splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.strip() == header)
    j = next((k for k in range(i + 1, len(lines))
              if lines[k].lstrip().startswith("[")), len(lines))
    return "\n".join(lines[i:j]).rstrip()


def test_into_preserves_workload_keys_this_run_did_not_measure(tmp_path):
    """A command that derives four of the eleven [workload] keys must not
    reset the other seven: an omitted key reads back as the dataclass
    default, so dropping a hand-set 0.7 subagent ratio silently rewrites it
    to 0.1."""
    p = _tuned(tmp_path)
    old = tomllib.loads(p.read_text(encoding="utf-8"))["workload"]
    est, _ = study_estimate()                    # no --sessions, no --single-class
    p.write_text(merge_into(p, est), encoding="utf-8")
    new = tomllib.loads(p.read_text(encoding="utf-8"))["workload"]

    # every key this run could not observe survives with its tuned value
    for key in ("system_prefix_tokens", "subagent_ratio", "subagent_median_tokens",
                "subagent_sigma", "subagent_prefix_tokens", "sub_shares_prefix",
                "users", "user_prompt_median_tokens", "user_prompt_sigma",
                "think_time_s"):
        assert new[key] == old[key], key
    assert new["system_prefix_tokens"] == 54321
    assert new["subagent_ratio"] == 0.7
    assert new["users"] == 249
    # and the measured ones did move
    assert new["max_output_tokens"] != old["max_output_tokens"]
    assert new["miss_rate"] != old["miss_rate"]
    assert "preserved, not measured" in p.read_text(encoding="utf-8")
    load_config(p).validate()


def test_into_keeps_the_comment_that_introduces_the_next_table(tmp_path):
    p = _tuned(tmp_path)
    est, _ = study_estimate()
    after = merge_into(p, est)
    assert "# the SLO the team committed to" in after
    assert _block_text(after, "[slo]") == _block_text(TUNED, "[slo]")


def test_into_keeps_the_files_line_endings(tmp_path):
    p = _tuned(tmp_path, TUNED.replace("\n", "\r\n"))
    est, _ = study_estimate()
    after = merge_into(p, est)
    assert "\r\n" in after
    assert "\n" not in after.replace("\r\n", "")
    assert tomllib.loads(after)["workload"]["subagent_ratio"] == 0.7


def test_into_takes_the_subagent_ratio_from_the_config_it_writes(tmp_path,
                                                                 capsys):
    """`r` scales the cycle by (1 + r) and is not observable here, so the
    config being written to is the best source for it."""
    p = _tuned(tmp_path)
    assert ws_main(["workload", "--metrics-text", str(FIXTURE), "--json",
                    "--sessions", "100", "--into", str(p)]) == 0
    blob = json.loads(capsys.readouterr().out)
    assert blob["cycle"]["subagent_ratio"] == 0.7
    assert "tuned.toml" in blob["cycle"]["subagent_ratio_source"]


def test_into_appends_when_there_is_no_workload_block(tmp_path):
    p = tmp_path / "partial.toml"
    p.write_text('schema_version = 1\n\n[deployment]\nmodel = "27B"\n',
                 encoding="utf-8")
    est, _ = study_estimate(**{"single_class": True})
    text = merge_into(p, est)
    assert tomllib.loads(text)["deployment"]["model"] == "27B"
    assert "user_prompt_median_tokens" in tomllib.loads(text)["workload"]


def test_into_refuses_a_block_the_schema_would_reject(tmp_path):
    from workingset.workload import _splice_workload
    base = _config(tmp_path).read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="unknown key workload"):
        _splice_workload(base, "[workload]\nnot_a_field = 1\n")


def test_into_refuses_a_config_the_model_could_not_price(tmp_path):
    """`from_dict` accepts `users = -1`; only `validate()` rejects it, and a
    file this command wrote must be one `ws predict` can read."""
    from workingset.workload import _splice_workload
    base = _config(tmp_path).read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="users must be >= 0"):
        _splice_workload(base, "[workload]\nusers = -1\n")


def test_a_negative_session_count_is_refused_at_the_flag(capsys, tmp_path):
    p = _config(tmp_path)
    assert ws_main(["workload", "--metrics-text", str(FIXTURE),
                    "--sessions", "-1", "--into", str(p)]) == 2
    assert "--sessions must be >= 0" in capsys.readouterr().err
    # and the config was not touched on the way to the refusal
    assert p.read_text(encoding="utf-8") == RunConfig().dumps("toml")


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
    for flag in ("--single-class", "--subagent-ratio"):
        assert flag in help_text
    for caveat in ("CUMULATIVE SINCE SERVER START",
                   "ONE observable over TWO unknowns",
                   "NO DEFAULT: nothing on this surface counts sessions",
                   "bound the time-average population in neither direction",
                   "users = lambda_total (Z + R) / (1 + r)",
                   "carried over unchanged rather than reset",
                   "ROUNDED AGGREGATES",
                   "increase() over the whole --range"):
        assert caveat in help_text, caveat


def test_cli_refuses_a_double_engine_before_dialling(capsys):
    """The matcher is validated before the first socket, so the message is
    about the flags rather than about DNS."""
    assert ws_main(["workload", "--prometheus", "http://127.0.0.1:1/",
                    "--selector", 'engine="0"', "--engine", "1"]) == 2
    err = capsys.readouterr().err
    assert "already constrains `engine`" in err
    assert "Name or service" not in err and "Connect" not in err


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
    buckets, adjust = _bucket_result(result)
    assert buckets == {10.0: 100.0, 20.0: 100.0, math.inf: 120.0}
    assert adjust == pytest.approx(0.5)     # how far it had to be pushed up
    # a clean series reports no adjustment at all
    clean = [{"metric": {"le": "10.0"}, "value": [0, "100.0"]},
             {"metric": {"le": "+Inf"}, "value": [0, "120.0"]}]
    assert _bucket_result(clean)[1] == 0.0


def test_the_monotonisation_caveat_appears_only_when_it_did_something():
    """A correction that only ever raises counts, applied silently, is a
    number the reader cannot see; one that never fired is noise."""
    est, _ = study_estimate()
    assert not any("made cumulative by running maximum" in c for c in est.caveats)

    prom = study_prom()
    h = prom.hists["vllm:request_prompt_tokens"]
    dipped = dict(h.buckets)
    dipped[50_000.0] = dipped[20_000.0] - 37.0        # a dip increase() can make
    prom.hists["vllm:request_prompt_tokens"] = Histogram(
        "h", {}, dipped, h.count, h.sum)
    est, _ = study_estimate(prom)
    note = [c for c in est.caveats if "made cumulative by running maximum" in c]
    assert len(note) == 1
    assert "37.0 observations" in note[0]
    assert "request_prompt_tokens_hist" in note[0]


def test_count_and_the_inf_bucket_are_cross_checked():
    """`increase()` extrapolates the two independently and the fit divides one
    by the other, so a disagreement between them is not a detail."""
    prom = study_prom()
    h = prom.hists["vllm:request_generation_tokens"]
    prom.hists["vllm:request_generation_tokens"] = Histogram(
        "h", {}, dict(h.buckets), h.count * 1.20, h.sum)
    est, _ = study_estimate(prom)
    note = [c for c in est.caveats if "against a +Inf bucket of" in c]
    assert len(note) == 1
    assert "request_generation_tokens" in note[0]
    # a 1% wobble is what extrapolation explains, and stays quiet
    prom = study_prom()
    h = prom.hists["vllm:request_generation_tokens"]
    prom.hists["vllm:request_generation_tokens"] = Histogram(
        "h", {}, dict(h.buckets), h.count * 1.005, h.sum)
    est, _ = study_estimate(prom)
    assert not any("against a +Inf bucket of" in c for c in est.caveats)
