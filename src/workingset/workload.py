"""`ws workload` — characterise the real workload, emit the config's `[workload]`.

The study's `[workload]` block was last set BY HAND from a 7-day Grafana pull
(`research/workload_agentic_poc.md`: mean prompt 57.4k tokens, 87.5%
prefix-cache savings, 404 mean output tokens). This module makes that pull
reproducible, and doubles as the EMPLOYER-DATA FIREWALL: it reads a
Prometheus that scrapes vLLM, and emits only ROUNDED AGGREGATES — never a
timestamp, never a raw series value, never the instance's identity beyond the
selector you typed yourself.

Three sources behind one interface, all producing a `Reading`:

    --prometheus URL --range 7d [--step 5m] [--selector 'model_name="..."']
        PromQL over the HTTP API. `increase()` over the whole range for the
        counters and the histogram buckets, `query_range` for the gauges.
        Counter resets are absorbed by `increase()` and reported separately
        by `resets()`; scrape gaps show as missing steps in the range.
    --jsonl FILE
        a `ws metrics tail --out` archive, delta'd whole-log with
        `window_from_snapshots`. Counter resets and layout changes land in
        `WindowDelta.invalid` and are reported, never papered over.
    --metrics-text FILE
        one raw `/metrics` dump. Its counters are CUMULATIVE SINCE SERVER
        START, so distributions and ratios are available but every RATE is
        not: there is no duration to divide by.

What comes out (`WorkloadEstimate`, every field's units and formula in its
docstring):

    prompt length     log-normal fit to the `request_prompt_tokens` bucket CDF
    output tokens     mean + log-normal fit to `request_generation_tokens`
    request rate      `request_success_total` / window seconds
    residence time    Little's law on `num_requests_running`: W = L / lambda
    think time        Z = cycle - R, cycle = sessions / lambda, R = mean e2e
    prefix cache      hits / queries, and the TWO readings it admits

The prefix-cache reading is the point the research note makes and this module
refuses to blur: 87.5% savings is ONE observable over TWO unknowns (the miss
rate and the warm-turn size). Fixing either yields the other; nothing in
`/metrics` separates them. Both are printed, each labelled with the
assumption it needs.

Anything the metrics cannot answer is printed as
"not observable from these metrics: <reason>" and never as a default.
"""
from __future__ import annotations

import json
import math
import re
import statistics
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from .metrics.adapter import resolve_aliases
from .metrics.parse import Histogram, parse_text
from .metrics.sampler import load_jsonl, window_from_snapshots
from .metrics.vllm import ALIASES, detect_adapter

__all__ = [
    "Reading", "Provenance", "LogNormalFit", "CacheReadings", "WorkloadEstimate",
    "PrometheusClient", "read_prometheus", "read_jsonl", "read_metrics_text",
    "fit_lognormal", "estimate", "emit_toml", "emit_json", "emit_table",
    "merge_into", "parse_duration", "promql", "PROMQL",
    "DEFAULT_TURN_TOKENS", "DEFAULT_MISS_RATE",
]

# The study's two candidate assumptions, from research/workload_agentic_poc.md.
DEFAULT_TURN_TOKENS = 2_000.0     # the study's warm turn (tokens)
DEFAULT_MISS_RATE = 0.01          # the config's `miss_rate` default

# Semantic keys this command reads. Everything else in the adapter's
# vocabulary is irrelevant to a workload characterisation and is not queried,
# which keeps the Prometheus round trips down to one per key.
COUNTER_KEYS = ("prompt_tokens_total", "prompt_tokens_cached_total",
                "generation_tokens_total", "prefix_cache_queries_total",
                "prefix_cache_hits_total", "request_success_total")
HIST_KEYS = ("request_prompt_tokens_hist", "request_generation_tokens_hist",
             "e2e_hist", "queue_time_hist", "ttft_hist")
GAUGE_KEYS = ("requests_running", "requests_waiting")


# ===========================================================================
# PromQL, built from the adapter's metric names
# ===========================================================================
def _matcher(selector: str, engine: str | None) -> str:
    """The `{...}` label matcher, or `""` when nothing is selected.

    `selector` is passed through verbatim (it is PromQL the user typed);
    `engine` is appended as `engine="N"`, the label vLLM V1 puts on every
    per-engine series.
    """
    parts = [p for p in (selector.strip().strip("{}").strip() if selector else "",)
             if p]
    if engine is not None:
        parts.append(f'engine="{engine}"')
    return "{" + ",".join(parts) + "}" if parts else ""


@dataclass(frozen=True)
class PROMQL:
    """The five query shapes, built from an EXPORTED metric name.

    Nothing here hardcodes a `vllm:` string: the names come from
    `resolve_aliases(names, ALIASES, ...)` against what the server actually
    exposes, exactly as the scrape-side adapter resolves them. An engine
    rename upstream changes `ALIASES` and both paths follow.
    """

    selector: str = ""
    engine: str | None = None
    range: str = "7d"

    @property
    def m(self) -> str:
        return _matcher(self.selector, self.engine)

    def counter(self, name: str) -> str:
        """Counter increase over the whole range, summed over label sets."""
        return f"sum(increase({name}{self.m}[{self.range}]))"

    def resets(self, name: str) -> str:
        """How many times this counter went backwards inside the range."""
        return f"max(resets({name}{self.m}[{self.range}]))"

    def buckets(self, name: str) -> str:
        """Histogram bucket increase, keyed by `le` — the bucket-wise delta
        that makes a quantile over the range definable at all."""
        return f"sum by (le) (increase({name}_bucket{self.m}[{self.range}]))"

    def hist_sum(self, name: str) -> str:
        return f"sum(increase({name}_sum{self.m}[{self.range}]))"

    def hist_count(self, name: str) -> str:
        return f"sum(increase({name}_count{self.m}[{self.range}]))"

    def gauge(self, name: str) -> str:
        """A gauge summed across engines — the deployment-wide concurrency."""
        return f"sum({name}{self.m})"


def promql(kind: str, name: str, *, selector: str = "", engine: str | None = None,
           range: str = "7d") -> str:
    """One query string by kind: counter|resets|buckets|hist_sum|hist_count|gauge."""
    q = PROMQL(selector=selector, engine=engine, range=range)
    fn = getattr(q, kind, None)
    if fn is None:
        raise ValueError(f"unknown query kind {kind!r}")
    return fn(name)


_DUR = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(ms|s|m|h|d|w|y)\s*$")
_DUR_S = {"ms": 1e-3, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0,
          "w": 604800.0, "y": 31536000.0}


def parse_duration(spec: str) -> float:
    """A Prometheus duration (`7d`, `90m`, `30s`) as SECONDS.

    Only single-unit forms are accepted; `1h30m` is legal PromQL but two
    different windows spelled one way is exactly the ambiguity a measurement
    should not carry silently.
    """
    m = _DUR.match(spec or "")
    if not m:
        raise ValueError(f"bad duration {spec!r}: use one of "
                         f"{sorted(_DUR_S)} (e.g. 7d, 12h, 90m)")
    return float(m.group(1)) * _DUR_S[m.group(2)]


# ===========================================================================
# the Prometheus HTTP API
# ===========================================================================
class PrometheusClient:
    """The three `/api/v1` endpoints this command needs, and nothing else.

    url          Prometheus base URL (`http://prom:9090`, with or without
                 a trailing `/api/v1`)
    headers      extra request headers; `--auth-header 'Authorization: ...'`
    verify       True, False, or a CA bundle path (a corporate interception
                 proxy presents its own certificate: the bundle is the fix,
                 `--insecure` is the escape hatch)
    client       an injected `httpx.Client` (tests pass one built on
                 `httpx.MockTransport`); the client will not close what it
                 did not open.
    """

    def __init__(self, url: str, *, headers: dict[str, str] | None = None,
                 verify: bool | str = True, timeout: float = 30.0,
                 client: httpx.Client | None = None):
        base = url.rstrip("/")
        if base.endswith("/api/v1"):
            base = base[: -len("/api/v1")]
        self.base = base
        self._owned = client is None
        self._client = client or httpx.Client(headers=headers or {},
                                              verify=verify, timeout=timeout)

    def close(self) -> None:
        if self._owned:
            self._client.close()

    def __enter__(self) -> "PrometheusClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- raw calls ------------------------------------------------------
    def _get(self, path: str, params: dict[str, Any]) -> dict:
        r = self._client.get(f"{self.base}/api/v1/{path}", params=params)
        r.raise_for_status()
        body = r.json()
        if body.get("status") != "success":
            raise ValueError(f"prometheus {path}: "
                             f"{body.get('error', 'unsuccessful response')}")
        return body.get("data") or {}

    def metric_names(self, match: str | None = None) -> set[str]:
        """Every metric name Prometheus knows, optionally narrowed by a
        `match[]` series selector.

        The `match[]` narrowing is an optimisation, not a requirement: a
        Prometheus too old to accept it (or one that rejects the regex) is
        retried unfiltered, since the only use of this set is alias
        resolution.
        """
        if match:
            try:
                return set(self._get("label/__name__/values", {"match[]": match}))
            except (httpx.HTTPError, ValueError):
                pass
        return set(self._get("label/__name__/values", {}))

    def query(self, q: str) -> list[dict]:
        """Instant query -> the `result` list (empty when nothing matched)."""
        return list(self._get("query", {"query": q}).get("result") or [])

    def query_range(self, q: str, start: float, end: float, step: str) -> list[dict]:
        """Range query -> the `result` list of `{metric, values:[[t, "v"], ...]}`."""
        data = self._get("query_range", {"query": q, "start": repr(start),
                                         "end": repr(end), "step": step})
        return list(data.get("result") or [])


def _scalar(result: list[dict]) -> float | None:
    """The single value of an instant query, or None when it matched nothing."""
    for series in result:
        v = series.get("value")
        if v and len(v) == 2:
            try:
                x = float(v[1])
            except (TypeError, ValueError):
                continue
            if math.isfinite(x):
                return x
    return None


def _bucket_result(result: list[dict]) -> dict[float, float] | None:
    """`sum by (le) (increase(..._bucket[...]))` -> a `le` -> count mapping.

    `increase()` extrapolates at the range edges, so the counts are floats
    and can come back very slightly NON-monotone across `le`. A cumulative
    histogram that dips is not one, so the series is made monotone by running
    maximum; the distortion is bounded by the extrapolation and is recorded
    as a caveat by the caller.
    """
    raw: dict[float, float] = {}
    for series in result:
        le = (series.get("metric") or {}).get("le")
        v = series.get("value")
        if le is None or not v or len(v) != 2:
            continue
        try:
            b = math.inf if le in ("+Inf", "Inf") else float(le)
            x = float(v[1])
        except (TypeError, ValueError):
            continue
        if math.isfinite(x):
            raw[b] = raw.get(b, 0.0) + x
    if not raw:
        return None
    out: dict[float, float] = {}
    run = 0.0
    for b in sorted(raw):
        run = max(run, raw[b])
        out[b] = run
    return out


# ===========================================================================
# a source-agnostic reading
# ===========================================================================
@dataclass(frozen=True)
class Provenance:
    """Where a reading came from. Carries NO timestamp and no raw value —
    everything here is either something the user typed or a metric name."""

    source: str                       # "prometheus" | "jsonl" | "metrics-text"
    target: str                       # URL or file path, as given
    range: str | None = None          # "7d", or None for a file source
    step: str | None = None
    selector: str = ""
    engine: str | None = None
    resolved: dict[str, str] = field(default_factory=dict)
    missing: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {"source": self.source, "target": self.target, "range": self.range,
                "step": self.step, "selector": self.selector,
                "engine": self.engine, "resolved": dict(self.resolved),
                "missing": list(self.missing)}


@dataclass
class Reading:
    """Aggregates from one source, before any workload interpretation.

    counters    semantic key -> total over the window (units: `KEY_UNITS`)
    histograms  semantic key -> the window's bucket-wise `Histogram`
    running     the `num_requests_running` VALUES over the window (requests).
                Kept as a list only long enough to take a mean and a p95;
                nothing downstream sees it and nothing emitted contains it.
    seconds     window length in seconds, or None when the source cannot say
                (a single `/metrics` dump has no duration)
    gaps        scrape/step gaps observed inside the window
    resets      semantic key -> why no delta exists, or how many resets
    """

    provenance: Provenance
    counters: dict[str, float | None] = field(default_factory=dict)
    histograms: dict[str, Histogram | None] = field(default_factory=dict)
    running: list[float] = field(default_factory=list)
    seconds: float | None = None
    gaps: int = 0
    resets: dict[str, str] = field(default_factory=dict)
    caveats: list[str] = field(default_factory=list)

    @property
    def hours(self) -> float | None:
        return None if self.seconds is None else self.seconds / 3600.0


# ---------------------------------------------------------------------------
# source (a): Prometheus
# ---------------------------------------------------------------------------
def read_prometheus(url: str, *, range: str = "7d", step: str = "5m",
                    selector: str = "", engine: str | None = None,
                    headers: dict[str, str] | None = None,
                    verify: bool | str = True, timeout: float = 30.0,
                    now: float | None = None,
                    client: httpx.Client | None = None) -> Reading:
    """Pull the window's aggregates out of a Prometheus scraping vLLM.

    Counters and histogram buckets come from `increase(...[range])` evaluated
    once at the end of the window — one instant query per series, not a range
    of them, because the workload question is about the WHOLE window and a
    per-step series would be raw data this command exists not to move.

    Gauges are the exception: `num_requests_running` needs its distribution
    (a mean for Little's law, a p95 for the peak), so it is pulled as a
    `query_range` at `step` and reduced to two numbers here, in memory.

    `now` overrides the evaluation instant (tests pin it); the default is the
    server's own idea of now, which is what `time()` in PromQL would use.
    """
    q = PROMQL(selector=selector, engine=engine, range=range)
    seconds = parse_duration(range)
    step_s = parse_duration(step)
    owned = client is None
    pc = PrometheusClient(url, headers=headers, verify=verify, timeout=timeout,
                          client=client)
    try:
        names = pc.metric_names(match='{__name__=~"(vllm|sglang):.*"}')
        bases = {n[: -len("_bucket")] for n in names if n.endswith("_bucket")}
        resolved = resolve_aliases(names, ALIASES, bases)
        prov = Provenance(source="prometheus", target=url, range=range, step=step,
                          selector=selector, engine=engine, resolved=dict(resolved),
                          missing=tuple(k for k in COUNTER_KEYS + HIST_KEYS
                                        + GAUGE_KEYS if k not in resolved))
        rd = Reading(provenance=prov, seconds=seconds)

        for key in COUNTER_KEYS:
            name = resolved.get(key)
            if name is None:
                rd.counters[key] = None
                continue
            rd.counters[key] = _scalar(pc.query(q.counter(name)))
            n_reset = _scalar(pc.query(q.resets(name)))
            if n_reset:
                rd.resets[key] = (f"{n_reset:.0f} counter reset(s) inside the "
                                  f"range; increase() bridges them, the total "
                                  f"is a lower bound")

        for key in HIST_KEYS:
            name = resolved.get(key)
            if name is None:
                rd.histograms[key] = None
                continue
            buckets = _bucket_result(pc.query(q.buckets(name)))
            if buckets is None:
                rd.histograms[key] = None
                continue
            total = _scalar(pc.query(q.hist_sum(name)))
            count = _scalar(pc.query(q.hist_count(name)))
            rd.histograms[key] = Histogram(name=name, buckets=buckets,
                                           count=count, sum=total)

        end = float(now if now is not None else _server_now(pc))
        name = resolved.get("requests_running")
        if name is not None:
            series = pc.query_range(q.gauge(name), end - seconds, end, step)
            vals, n_points = _range_values(series)
            rd.running = vals
            expected = int(seconds // step_s) + 1
            rd.gaps = max(0, expected - n_points)
            if rd.gaps:
                rd.caveats.append(
                    f"{rd.gaps} of {expected} {step} steps carry no "
                    f"num_requests_running sample: the scrape was down, or the "
                    f"series had not started. Means and the p95 are over the "
                    f"{n_points} steps that exist.")
        rd.caveats.append(
            "Prometheus increase() extrapolates to the range edges, so counter "
            "totals and bucket counts are fractional and accurate to about one "
            "scrape interval at each end; bucket counts were made monotone.")
        return rd
    finally:
        if owned:
            pc.close()


def _server_now(pc: PrometheusClient) -> float:
    """Prometheus's own clock, so the range ends where the data does.

    Falls back to the local clock, which is what any client would use anyway
    and is wrong only by the machines' skew.
    """
    try:
        v = _scalar(pc.query("time()"))
        if v is not None:
            return v
    except (httpx.HTTPError, ValueError):
        pass
    import time as _t
    return _t.time()


def _range_values(series: list[dict]) -> tuple[list[float], int]:
    """A `query_range` result -> (values, n steps present).

    Timestamps are DROPPED here, at the boundary: nothing downstream can leak
    a when, because nothing downstream is given one.
    """
    by_step: dict[float, float] = {}
    for s in series:
        for pair in s.get("values") or []:
            if not pair or len(pair) != 2:
                continue
            try:
                t, v = float(pair[0]), float(pair[1])
            except (TypeError, ValueError):
                continue
            if math.isfinite(v):
                by_step[t] = by_step.get(t, 0.0) + v
    return [by_step[t] for t in sorted(by_step)], len(by_step)


# ---------------------------------------------------------------------------
# source (b): a `ws metrics tail` archive
# ---------------------------------------------------------------------------
def read_jsonl(path: str | Path, *, engine: str | None = None) -> Reading:
    """Delta a whole `ws metrics tail --out` log into one window.

    The window is the log's own endpoints (`t0=t1=None`): there is nothing
    outside the first and last snapshot to enclose with, and the archive IS
    the measurement.
    """
    snaps = load_jsonl(path)
    if len(snaps) < 2:
        raise ValueError(f"{path}: {len(snaps)} snapshot(s); a window needs >= 2")
    w = window_from_snapshots(snaps, None, None, engine=engine)
    adapter = detect_adapter(w.hi.samples, engine=engine)
    res = adapter.resolution()
    prov = Provenance(source="jsonl", target=str(path), selector="",
                      engine=engine, resolved=dict(res.resolved),
                      missing=tuple(k for k in COUNTER_KEYS + HIST_KEYS
                                    + GAUGE_KEYS if k not in res.resolved))
    rd = Reading(provenance=prov, seconds=w.dt)
    for key in COUNTER_KEYS:
        rd.counters[key] = w.counters.get(key)
    for key in HIST_KEYS:
        rd.histograms[key] = w.histograms.get(key)
    for key, why in w.invalid.items():
        rd.resets[key] = why
    rd.running = [v for v in (adapter.gauge(s.samples, "requests_running")
                              for s in snaps if s.ok and s.samples)
                  if v is not None]
    failed = sum(1 for s in snaps if not s.ok)
    rd.gaps = failed
    if failed:
        rd.caveats.append(f"{failed} of {len(snaps)} scrapes in the archive "
                          f"failed; the counter delta still spans the whole log, "
                          f"but the gauge mean and p95 skip those instants.")
    if w.dt > 0:
        rd.caveats.append(
            f"window endpoints are fuzzy by +/-{w.dt_uncertainty * 1e3:.0f} ms "
            f"(the two scrapes' round trips), so a rate over a short archive is "
            f"only as sharp as that.")
    return rd


# ---------------------------------------------------------------------------
# source (c): one raw /metrics dump
# ---------------------------------------------------------------------------
def read_metrics_text(path: str | Path, *, engine: str | None = None) -> Reading:
    """Read ONE `/metrics` dump. Cumulative since server start.

    Everything here is a since-boot total: the distributions and the ratios
    built from them are usable (they are shape, not rate), but the window
    length is unknown, so `seconds` is None and every rate — request rate,
    Little's law, think time — comes out "not observable". Point a
    `--prometheus` or a `--jsonl` at it if you need those.
    """
    text = Path(path).read_text(encoding="utf-8")
    samples = parse_text(text)
    adapter = detect_adapter(samples, engine=engine)
    res = adapter.resolution()
    prov = Provenance(source="metrics-text", target=str(path), engine=engine,
                      resolved=dict(res.resolved),
                      missing=tuple(k for k in COUNTER_KEYS + HIST_KEYS
                                    + GAUGE_KEYS if k not in res.resolved))
    rd = Reading(provenance=prov, seconds=None)
    for key in COUNTER_KEYS:
        rd.counters[key] = adapter.counter(samples, key)
    for key in HIST_KEYS:
        rd.histograms[key] = adapter.histogram(samples, key)
    g = adapter.gauge(samples, "requests_running")
    rd.running = [] if g is None else [g]
    rd.caveats.append(
        "a single /metrics dump is CUMULATIVE SINCE SERVER START: these totals "
        "cover the server's whole uptime, not a window you chose, and no rate "
        "is derivable from them because the duration is not exported.")
    return rd


# ===========================================================================
# the log-normal fit
# ===========================================================================
@dataclass(frozen=True)
class LogNormalFit:
    """A log-normal fitted to a bucketed CDF. Lengths in TOKENS.

    The fit: a log-normal has Phi^-1(F(b)) = (ln b - mu) / sigma, so plotting
    `ln b` against `Phi^-1(F(b))` over the histogram's bucket edges is a
    straight line whose SLOPE is sigma and whose INTERCEPT is mu. Ordinary
    least squares on those points gives both at once, and `residual_ln` is
    the RMS distance from the line in ln-token units — a straight-line fit
    with a small residual is evidence the distribution really is log-normal,
    which the study assumes and had never checked against buckets.

        median_tokens = exp(mu)
        fit_mean_tokens = exp(mu + sigma^2 / 2)
        mean_tokens = histogram _sum / _count  (exact, not from the fit)

    `censored` is True when observations fell above the largest FINITE bucket
    bound: the tail is then only bounded, not measured, and sigma is a lower
    bound on the true spread. vLLM's top prompt bucket is 200k tokens, so a
    deployment with a longer `max_model_len` censors.
    """

    median_tokens: float
    sigma: float
    mu: float
    fit_mean_tokens: float
    mean_tokens: float | None
    observations: float
    n_points: int
    residual_ln: float
    max_residual_ln: float
    censored: bool
    censored_fraction: float
    top_finite_bound: float

    @property
    def residual_pct(self) -> float:
        """RMS residual as a multiplicative error on a length, e.g. 0.07 = 7%."""
        return math.exp(self.residual_ln) - 1.0

    def to_dict(self) -> dict:
        return {"median_tokens": self.median_tokens, "sigma": self.sigma,
                "mu_ln": self.mu, "fit_mean_tokens": self.fit_mean_tokens,
                "mean_tokens": self.mean_tokens, "observations": self.observations,
                "n_fit_points": self.n_points, "residual_ln": self.residual_ln,
                "residual_pct": self.residual_pct,
                "max_residual_ln": self.max_residual_ln,
                "censored": self.censored,
                "censored_fraction": self.censored_fraction,
                "top_finite_bound_tokens": self.top_finite_bound}


class NotObservable(ValueError):
    """This quantity cannot be derived from the metrics at hand, and why."""


def fit_lognormal(h: Histogram | None, what: str = "distribution") -> LogNormalFit:
    """Least-squares log-normal over a cumulative histogram's bucket edges.

    Raises `NotObservable` with the reason when there is nothing to fit:
    the histogram is absent, it holds no observation in the window, or fewer
    than two bucket edges carry a strictly-interior CDF value (a distribution
    entirely inside one bucket has no shape to recover).
    """
    if h is None:
        raise NotObservable(f"{what}: the server exports no such histogram")
    n = h.observations
    if not n:
        raise NotObservable(f"{what}: the histogram holds no observation in "
                            f"this window")
    bounds = [b for b in h.bounds if math.isfinite(b) and b > 0]
    if not bounds:
        raise NotObservable(f"{what}: the histogram has no finite positive "
                            f"bucket bound")
    top = max(bounds)
    top_cum = h.buckets[top]
    censored_frac = max(0.0, (n - top_cum) / n)

    xs, ys = [], []
    for b in bounds:
        f = h.buckets[b] / n
        if not 0.0 < f < 1.0:
            continue
        xs.append(math.log(b))
        ys.append(statistics.NormalDist().inv_cdf(f))
    if len(xs) < 2:
        raise NotObservable(
            f"{what}: only {len(xs)} bucket edge(s) fall strictly inside the "
            f"distribution — the exporter's buckets are too coarse here to "
            f"recover a median and a sigma")

    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    sigma, mu = np.polyfit(y, x, 1)          # x = mu + sigma * y
    resid = x - (mu + sigma * y)
    rms = float(np.sqrt(float(np.mean(resid ** 2))))
    if not math.isfinite(sigma) or sigma <= 0:
        raise NotObservable(f"{what}: the bucket CDF is not monotone enough to "
                            f"fit (slope {sigma:.3g})")
    return LogNormalFit(
        median_tokens=math.exp(float(mu)), sigma=float(sigma), mu=float(mu),
        fit_mean_tokens=math.exp(float(mu) + float(sigma) ** 2 / 2.0),
        mean_tokens=h.mean(), observations=float(n), n_points=len(xs),
        residual_ln=rms, max_residual_ln=float(np.max(np.abs(resid))),
        censored=censored_frac > 0.0, censored_fraction=censored_frac,
        top_finite_bound=top)


# ===========================================================================
# the prefix-cache reading
# ===========================================================================
@dataclass(frozen=True)
class CacheReadings:
    """Prefix-cache savings, and the TWO workload readings it admits.

    `savings` S = prefix_cache_hits_total / prefix_cache_queries_total, in
    TOKENS: the fraction of prompt tokens that did not have to be prefilled.

    Under the study's turn model a request either matches nothing (a MISS,
    probability f, costing its whole context C tokens) or continues a warm
    session (costing only the new turn, T tokens). The computed tokens per
    request are then

        (1 - S) * C  =  f * C  +  (1 - f) * T

    which is ONE equation in TWO unknowns. Fix either and the other follows:

        miss_rate_given_turn = ((1 - S) * C - T) / (C - T)
        turn_tokens_given_miss = C * (1 - S - f) / (1 - f)

    They cannot be separated by any counter vLLM exports — the note in
    research/workload_agentic_poc.md §3 is the same finding, reached from a
    dashboard rather than from the counters. Separating them needs per-request
    hit/query attribution, which is not on the /metrics surface.

    C (`mean_prompt_tokens`) is the mean prompt length over the window, from
    the `request_prompt_tokens` histogram (or prompt_tokens_total / requests).
    """

    savings: float
    savings_source: str
    mean_prompt_tokens: float
    assumed_turn_tokens: float
    miss_rate_given_turn: float | None
    assumed_miss_rate: float
    turn_tokens_given_miss: float | None
    cached_savings: float | None          # the prompt_tokens_cached cross-check
    note: str = (
        "savings is ONE observable over TWO unknowns (miss rate, warm-turn "
        "size); each reading states the assumption it needs, and the two "
        "cannot be separated by any counter this surface exports")

    def to_dict(self) -> dict:
        return {"savings": self.savings, "savings_source": self.savings_source,
                "mean_prompt_tokens": self.mean_prompt_tokens,
                "assumed_turn_tokens": self.assumed_turn_tokens,
                "miss_rate_given_turn": self.miss_rate_given_turn,
                "assumed_miss_rate": self.assumed_miss_rate,
                "turn_tokens_given_miss": self.turn_tokens_given_miss,
                "cached_tokens_savings": self.cached_savings,
                "note": self.note}


def _cache_readings(rd: Reading, mean_prompt: float | None,
                    turn_tokens: float, miss_rate: float) -> CacheReadings | None:
    q = rd.counters.get("prefix_cache_queries_total")
    h = rd.counters.get("prefix_cache_hits_total")
    savings, source = None, ""
    if q and h is not None and q > 0:
        savings, source = h / q, "prefix_cache_hits_total / prefix_cache_queries_total"
    cached = rd.counters.get("prompt_tokens_cached_total")
    prompt = rd.counters.get("prompt_tokens_total")
    cached_savings = (cached / prompt if cached is not None and prompt else None)
    if savings is None:
        if cached_savings is None:
            return None
        savings, source = cached_savings, "prompt_tokens_cached_total / prompt_tokens_total"
    if mean_prompt is None or mean_prompt <= 0:
        return None
    c = mean_prompt
    f_given_t = (((1.0 - savings) * c - turn_tokens) / (c - turn_tokens)
                 if c > turn_tokens else None)
    t_given_f = (c * (1.0 - savings - miss_rate) / (1.0 - miss_rate)
                 if miss_rate < 1.0 else None)
    if f_given_t is not None and not 0.0 <= f_given_t <= 1.0:
        f_given_t = None
    if t_given_f is not None and t_given_f < 0.0:
        t_given_f = None
    return CacheReadings(savings=savings, savings_source=source,
                         mean_prompt_tokens=c, assumed_turn_tokens=turn_tokens,
                         miss_rate_given_turn=f_given_t,
                         assumed_miss_rate=miss_rate,
                         turn_tokens_given_miss=t_given_f,
                         cached_savings=cached_savings)


# ===========================================================================
# the estimate
# ===========================================================================
@dataclass(frozen=True)
class WorkloadEstimate:
    """Everything `ws workload` derives, with units and formulas.

    hours                window length, HOURS. None from a single dump.
    n_requests           requests that FINISHED in the window
                         (`request_success_total`, summed over
                         `finished_reason`; falls back to the prompt
                         histogram's observation count).
    req_rate_s           n_requests / window seconds, REQUESTS/SECOND. Over a
                         bursty multi-day range this is the 24/7 mean and is
                         several times below the office-hours rate.
    prompt               log-normal fit to `request_prompt_tokens`.
    output               log-normal fit to `request_generation_tokens`.
    output_mean_tokens   that histogram's _sum/_count, TOKENS/REQUEST. This,
                         not the fit, is what the config's `max_output_tokens`
                         is set from — the study prices a mean.
    mean_running         mean `num_requests_running` over the window,
                         REQUESTS. Concurrent requests IN EXECUTION BATCHES.
    p95_running          the 95th percentile of the same gauge, REQUESTS: the
                         peak DECODING concurrency.
    little_w_s           Little's law on the executing set, SECONDS:
                         L = lambda * W, so W = mean_running / req_rate_s.
                         Mean seconds a request spends being executed.
    e2e_mean_s           mean end-to-end request latency R, SECONDS, from the
                         e2e histogram's _sum/_count. Includes queueing;
                         `little_w_s` does not, so R - W is the queue share
                         (cross-checked against `queue_time_hist` when the
                         server exports it).
    sessions             the concurrent-SESSION count the cycle is computed
                         at. NOT observable: defaults to `p95_running`, which
                         counts DECODING requests and is therefore a LOWER
                         BOUND on sessions (a session thinking between turns
                         is not decoding and is invisible here).
    cycle_s              sessions / req_rate_s, SECONDS per session per
                         request — the closed-loop cycle.
    think_time_s         Z = cycle_s - e2e_mean_s, SECONDS: the same quantity
                         scripts/think_time_trace.py derives from a request
                         trace (Z = waiting per request, cycle = Z + R).
    cache                the prefix-cache readings, see `CacheReadings`.
    unobservable         field name -> why these metrics cannot answer it.
    caveats              everything a reader has to know before quoting a
                         number from this run.
    """

    provenance: Provenance
    hours: float | None
    n_requests: float | None
    n_requests_source: str
    req_rate_s: float | None
    prompt: LogNormalFit | None
    output: LogNormalFit | None
    output_mean_tokens: float | None
    mean_running: float | None
    p95_running: float | None
    little_w_s: float | None
    e2e_mean_s: float | None
    queue_mean_s: float | None
    sessions: float | None
    sessions_assumed: bool
    cycle_s: float | None
    think_time_s: float | None
    cache: CacheReadings | None
    gaps: int
    resets: dict[str, str]
    unobservable: dict[str, str]
    caveats: list[str]

    def to_dict(self) -> dict:
        return {
            "provenance": self.provenance.to_dict(),
            "window": {"hours": _r(self.hours, 3), "gaps": self.gaps,
                       "counter_resets": dict(self.resets)},
            "requests": {"n": _r(self.n_requests, 3),
                         "source": self.n_requests_source,
                         "rate_per_s": _r(self.req_rate_s, 2)},
            "prompt_tokens": None if self.prompt is None else _round_fit(self.prompt),
            "output_tokens": {
                "mean": _r(self.output_mean_tokens, 3),
                "fit": None if self.output is None else _round_fit(self.output)},
            "concurrency": {"mean_running": _r(self.mean_running, 3),
                            "p95_running": _r(self.p95_running, 3),
                            "little_w_s": _sec(self.little_w_s),
                            "e2e_mean_s": _sec(self.e2e_mean_s),
                            "queue_mean_s": _sec(self.queue_mean_s)},
            "cycle": {"sessions": _r(self.sessions, 3),
                      "sessions_assumed": self.sessions_assumed,
                      "cycle_s": _sec(self.cycle_s),
                      "think_time_s": None if self.think_time_s is None
                      else round(self.think_time_s, 1)},
            "prefix_cache": None if self.cache is None else _round_cache(self.cache),
            "not_observable": dict(self.unobservable),
            "caveats": list(self.caveats),
        }


# Every `[workload]` key this command cannot derive, and why. Printed instead
# of a default, so a reader never mistakes a dataclass default for a
# measurement.
UNOBSERVABLE: dict[str, str] = {
    "system_prefix_tokens":
        "the prefix-cache counters report how many tokens HIT, not which "
        "tokens form a shared system prefix; nothing on this surface "
        "attributes a hit to a prompt position",
    "subagent_ratio":
        "vLLM exports no request-class label, so the main/subagent split is "
        "the study's model of the client, not something /metrics distinguishes",
    "subagent_median_tokens":
        "same: no request-class label, so the subagent prompt distribution "
        "cannot be separated out of the single request_prompt_tokens histogram",
    "subagent_sigma":
        "same: no request-class label to split the prompt histogram by",
    "subagent_prefix_tokens":
        "same: no request-class label, and no per-position hit attribution",
    "sub_shares_prefix":
        "a client-side fact about how subagent prompts are built; the server "
        "sees tokens, not who assembled them",
}


def estimate(rd: Reading, *, turn_tokens: float = DEFAULT_TURN_TOKENS,
             miss_rate: float = DEFAULT_MISS_RATE,
             sessions: float | None = None) -> WorkloadEstimate:
    """Turn one `Reading` into the workload numbers, or into refusals.

    `turn_tokens` and `miss_rate` are the two ASSUMPTIONS the prefix-cache
    identity needs, one each; `sessions` overrides the concurrent-session
    count the think-time cycle is computed at (default: the running gauge's
    p95, a lower bound — see `WorkloadEstimate.sessions`).
    """
    unobs: dict[str, str] = dict(UNOBSERVABLE)
    caveats = list(rd.caveats)
    seconds = rd.seconds

    # ---- requests and rate ---------------------------------------------
    n = rd.counters.get("request_success_total")
    src = "request_success_total"
    hp = rd.histograms.get("request_prompt_tokens_hist")
    if n is None and hp is not None and hp.observations:
        n, src = hp.observations, "request_prompt_tokens count (request_success_total absent)"
    if n is None:
        src = "not observable"
        unobs["n_requests"] = ("neither request_success_total nor the "
                               "request_prompt_tokens histogram is exported here")
    rate = (n / seconds if n is not None and seconds and seconds > 0 else None)
    if rate is None and n is not None:
        unobs["request_rate"] = ("the window length is unknown: a single "
                                 "/metrics dump carries no duration")

    # ---- distributions ---------------------------------------------------
    prompt = output = None
    try:
        prompt = fit_lognormal(hp, "prompt length")
    except NotObservable as e:
        unobs["user_prompt_median_tokens"] = str(e)
        unobs["user_prompt_sigma"] = str(e)
    hg = rd.histograms.get("request_generation_tokens_hist")
    try:
        output = fit_lognormal(hg, "output length")
    except NotObservable as e:
        unobs["output_distribution"] = str(e)
    out_mean = hg.mean() if hg is not None else None
    if out_mean is None:
        g = rd.counters.get("generation_tokens_total")
        if g is not None and n:
            out_mean = g / n
            caveats.append("mean output tokens came from generation_tokens_total "
                           "/ requests, not from the request_generation_tokens "
                           "histogram: an E[X]/E[Y] ratio over the window.")
    if out_mean is None:
        unobs["max_output_tokens"] = ("no request_generation_tokens histogram "
                                      "and no generation_tokens_total to divide")

    if prompt is not None and prompt.censored:
        caveats.append(
            f"{prompt.censored_fraction:.1%} of prompts exceeded the largest "
            f"finite bucket ({prompt.top_finite_bound:,.0f} tokens): the tail is "
            f"bounded, not measured, and the fitted sigma is a LOWER bound.")
    if output is not None and output.censored:
        caveats.append(
            f"{output.censored_fraction:.1%} of outputs exceeded the largest "
            f"finite bucket ({output.top_finite_bound:,.0f} tokens): the output "
            f"sigma is a lower bound.")

    # ---- concurrency, Little's law, think time ---------------------------
    mean_run = float(np.mean(rd.running)) if rd.running else None
    p95_run = (float(np.percentile(np.asarray(rd.running, dtype=float), 95))
               if rd.running else None)
    if mean_run is None:
        unobs["concurrency"] = ("num_requests_running is not exported (or, "
                                "under DP>1 with no --engine, could not be "
                                "combined)")
    little_w = (mean_run / rate if mean_run is not None and rate else None)
    he = rd.histograms.get("e2e_hist")
    e2e = he.mean() if he is not None else None
    if e2e is None:
        unobs["e2e_mean_s"] = "the server exports no e2e_request_latency histogram"
    hq = rd.histograms.get("queue_time_hist")
    queue = hq.mean() if hq is not None else None

    sess, assumed = sessions, sessions is None
    if sess is None:
        sess = p95_run
    cycle = (sess / rate if sess is not None and rate else None)
    think = None
    if cycle is not None and e2e is not None:
        think = cycle - e2e
        if think <= 0:
            unobs["think_time_s"] = (
                f"the assumed session count ({sess:,.0f}) implies a cycle of "
                f"{cycle:,.1f} s, shorter than the {e2e:,.1f} s mean service "
                f"time: no non-negative think time is consistent with it. Pass "
                f"a larger --sessions, or narrow --range to the busy period")
            think = None
    elif cycle is None:
        unobs["think_time_s"] = ("needs a request rate and a session count; "
                                 + ("the window length is unknown"
                                    if not seconds else
                                    "num_requests_running is not exported"))
    elif e2e is None:
        unobs["think_time_s"] = ("Z = cycle - R needs the mean service time R, "
                                 "and no e2e_request_latency histogram is "
                                 "exported here")

    if sess is not None and assumed:
        caveats.append(
            "sessions defaults to the num_requests_running p95, which counts "
            "concurrent DECODING requests. A session thinking between turns is "
            "not decoding, so this is a LOWER BOUND on the population and the "
            "derived cycle and think time are lower bounds with it. "
            "Sessions-in-cache is NOT observable from these metrics: the KV "
            "pool's occupancy is a fraction, not a session count.")
    unobs["sessions_in_cache"] = (
        "no vLLM metric counts distinct sessions resident in the KV cache; "
        "kv_cache_usage_perc is an occupancy fraction and num_requests_running "
        "counts requests in execution batches, not warm sessions")

    if rate is not None and seconds and seconds > 3 * 3600:
        caveats.append(
            f"the rate is the 24/7 mean over {seconds / 3600:.1f} h. An "
            f"office-hours workload is bursty, so the peak rate is several "
            f"times this; narrow --range to the busy period for an operating "
            f"point rather than an average.")
    if little_w is not None and e2e is not None and e2e > 0:
        why = ("the gap is queueing, which the running gauge does not count "
               "and the e2e histogram does"
               if little_w <= e2e else
               "W above R means the gauge's mean is higher than the finished-"
               "request rate can explain: the window mixes idle and busy "
               "stretches, or requests were running that never finished inside "
               "it. Narrow --range to a homogeneous period")
        caveats.append(
            f"Little's law W = L / lambda = {little_w:,.2f} s against a mean "
            f"e2e R of {e2e:,.2f} s: {why}.")

    # ---- prefix cache ----------------------------------------------------
    mean_prompt = None
    if hp is not None and hp.mean() is not None:
        mean_prompt = hp.mean()
    elif rd.counters.get("prompt_tokens_total") is not None and n:
        mean_prompt = rd.counters["prompt_tokens_total"] / n
    cache = _cache_readings(rd, mean_prompt, turn_tokens, miss_rate)
    if cache is None:
        unobs["miss_rate"] = ("no prefix-cache counters (a vLLM V0 server "
                              "exports only a decaying hit-rate gauge, from "
                              "which no window rate is recoverable) or no mean "
                              "prompt length to price them against")
        unobs["warm_turn_tokens"] = unobs["miss_rate"]
    else:
        caveats.append(
            f"prefix-cache savings {cache.savings:.1%} is ONE observable over "
            f"TWO unknowns: the miss rate and the warm-turn size cannot be "
            f"separated from these counters (research/workload_agentic_poc.md "
            f"section 3). Each reading here carries the assumption it needs.")
        if (cache.cached_savings is not None
                and abs(cache.cached_savings - cache.savings) > 0.01
                and cache.savings_source.startswith("prefix_cache")):
            caveats.append(
                f"prompt_tokens_cached_total / prompt_tokens_total reads "
                f"{cache.cached_savings:.1%} against the prefix-cache counters' "
                f"{cache.savings:.1%}; the two count different things at the "
                f"block boundary, and the gap is the uncertainty on savings.")
    if rd.resets:
        caveats.append("counters went backwards or changed layout inside the "
                       "window: " + "; ".join(f"{k} ({why})"
                                              for k, why in sorted(rd.resets.items())))

    return WorkloadEstimate(
        provenance=rd.provenance, hours=rd.hours, n_requests=n,
        n_requests_source=src, req_rate_s=rate, prompt=prompt, output=output,
        output_mean_tokens=out_mean, mean_running=mean_run, p95_running=p95_run,
        little_w_s=little_w, e2e_mean_s=e2e, queue_mean_s=queue, sessions=sess,
        sessions_assumed=assumed, cycle_s=cycle, think_time_s=think, cache=cache,
        gaps=rd.gaps, resets=dict(rd.resets), unobservable=unobs,
        caveats=caveats)


# ===========================================================================
# rounding — the firewall's arithmetic
# ===========================================================================
def _sig(x: float | None, n: int) -> float | None:
    """`x` to `n` significant figures. The ONLY numbers this command emits.

    Rounding is the firewall: a rounded aggregate cannot be walked back to a
    scrape, and no emitted number is ever a raw series value.
    """
    if x is None or not math.isfinite(x):
        return None
    if x == 0:
        return 0.0
    return round(x, n - 1 - int(math.floor(math.log10(abs(x)))))


def _r(x: float | None, n: int) -> float | None:
    return _sig(x, n)


def _tok(x: float | None) -> int | None:
    """A token count: 3 significant figures, as an int."""
    v = _sig(x, 3)
    return None if v is None else int(round(v))


def _rate(x: float | None) -> float | None:
    """A rate: 2 significant figures."""
    return _sig(x, 2)


def _sec(x: float | None) -> float | None:
    """A duration in seconds: 3 significant figures. Not a rate — 2 figures
    would round a 18.7 s mean latency to 19 s and lose the comparison with
    the Little's-law residence time it exists to be checked against."""
    return _sig(x, 3)


def _frac(x: float | None) -> float | None:
    """A fraction in [0, 1]: 3 decimals (0.1 percentage-point resolution)."""
    return None if x is None else round(x, 3)


def _think(x: float | None) -> float | None:
    """A think time: 1 decimal second."""
    return None if x is None else round(x, 1)


def _round_fit(f: LogNormalFit) -> dict:
    d = f.to_dict()
    d["median_tokens"] = _tok(d["median_tokens"])
    d["fit_mean_tokens"] = _tok(d["fit_mean_tokens"])
    d["mean_tokens"] = _tok(d["mean_tokens"])
    d["top_finite_bound_tokens"] = _tok(d["top_finite_bound_tokens"])
    d["observations"] = _tok(d["observations"])
    d["sigma"] = round(d["sigma"], 2)
    d["mu_ln"] = round(d["mu_ln"], 3)
    d["residual_ln"] = round(d["residual_ln"], 4)
    d["max_residual_ln"] = round(d["max_residual_ln"], 4)
    d["residual_pct"] = round(d["residual_pct"], 4)
    d["censored_fraction"] = _frac(d["censored_fraction"])
    return d


def _round_cache(c: CacheReadings) -> dict:
    d = c.to_dict()
    d["savings"] = _frac(d["savings"])
    d["cached_tokens_savings"] = _frac(d["cached_tokens_savings"])
    d["mean_prompt_tokens"] = _tok(d["mean_prompt_tokens"])
    d["assumed_turn_tokens"] = _tok(d["assumed_turn_tokens"])
    d["miss_rate_given_turn"] = _frac(d["miss_rate_given_turn"])
    d["assumed_miss_rate"] = _frac(d["assumed_miss_rate"])
    d["turn_tokens_given_miss"] = _tok(d["turn_tokens_given_miss"])
    return d


# ===========================================================================
# emitters
# ===========================================================================
def _wrap_comment(text: str, width: int = 74, indent: str = "#   ",
                  cont: str | None = None) -> list[str]:
    """Wrap `text`, prefixing the first line with `indent` and the rest with
    `cont` (default: `indent` blanked out, so a bullet's marker is not
    repeated on every continuation line)."""
    if cont is None:
        # keep the comment marker, blank the bullet: "#   - " -> "#     "
        cont = ("#" + " " * (len(indent) - 1) if indent.lstrip().startswith("#")
                else " " * len(indent))
    words, lines, cur = text.split(), [], ""
    for w in words:
        pre = indent if not lines else cont
        if cur and len(pre) + len(cur) + 1 + len(w) > width:
            lines.append(pre + cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append((indent if not lines else cont) + cur)
    return lines


def _provenance_comments(est: WorkloadEstimate) -> list[str]:
    p = est.provenance
    head = f"measured by `ws workload` from {p.source} {p.target}"
    if p.range:
        head += f", range {p.range}"
    if p.selector:
        head += f", selector {p.selector}"
    if p.engine is not None:
        head += f", engine {p.engine}"
    out = _wrap_comment(head, indent="# ")
    facts = []
    if est.n_requests is not None:
        facts.append(f"{_tok(est.n_requests):,} requests")
    if est.hours is not None:
        facts.append(f"{est.hours:.1f} h")
    if est.req_rate_s is not None:
        facts.append(f"{_rate(est.req_rate_s):g} req/s (window mean)")
    if est.gaps:
        facts.append(f"{est.gaps} scrape gap(s)")
    if facts:
        out += _wrap_comment(" | ".join(facts), indent="# ")
    return out


def emit_toml(est: WorkloadEstimate) -> str:
    """A `[workload]` block, ready to paste or to `--into` an existing config.

    Only DERIVED keys are assigned. Everything else is a comment saying
    "not observable from these metrics: <reason>" — a key omitted from a TOML
    block reads back as the dataclass default, and a reader has to be able to
    tell a default from a measurement.
    """
    lines = ["[workload]"]
    lines += _provenance_comments(est)
    lines.append("#")

    def put(key: str, value, comment: str = "") -> None:
        if value is None:
            return
        v = ("true" if value is True else "false" if value is False
             else repr(value))
        lines.append(f"{key} = {v}" + (f"  # {comment}" if comment else ""))

    if est.prompt is not None:
        put("user_prompt_median_tokens", _tok(est.prompt.median_tokens),
            f"log-normal fit, {est.prompt.n_points} bucket edges, "
            f"residual {est.prompt.residual_pct:.1%}"
            + (" (TAIL CENSORED)" if est.prompt.censored else ""))
        put("user_prompt_sigma", round(est.prompt.sigma, 2),
            "same fit; a lower bound when the tail is censored"
            if est.prompt.censored else "same fit")
    put("max_output_tokens", _tok(est.output_mean_tokens),
        "mean of request_generation_tokens")
    if est.cache is not None and est.cache.miss_rate_given_turn is not None:
        put("miss_rate", _frac(est.cache.miss_rate_given_turn),
            f"EFFECTIVE total-miss rate, and only under warm_turn_tokens = "
            f"{_tok(est.cache.assumed_turn_tokens):,}")
        put("warm_turn_tokens", _tok(est.cache.assumed_turn_tokens),
            "ASSUMED, not measured (see the two readings below)")
    put("think_time_s", _think(est.think_time_s),
        "Z = sessions / rate - mean e2e")
    put("users", None if est.sessions is None else int(round(est.sessions)),
        "num_requests_running p95: concurrent DECODERS, a LOWER BOUND on users"
        if est.sessions_assumed else "as given by --sessions")

    lines.append("#")
    if est.cache is not None:
        lines += _wrap_comment(
            f"prefix-cache savings {est.cache.savings:.1%} over a mean prompt of "
            f"{_tok(est.cache.mean_prompt_tokens):,} tokens. ONE observable, TWO "
            f"unknowns:", indent="# ")
        if est.cache.miss_rate_given_turn is not None:
            lines += _wrap_comment(
                f"miss rate {est.cache.miss_rate_given_turn:.1%} IF the warm turn "
                f"is {_tok(est.cache.assumed_turn_tokens):,} tokens", indent="#   - ")
        if est.cache.turn_tokens_given_miss is not None:
            lines += _wrap_comment(
                f"warm turn {_tok(est.cache.turn_tokens_given_miss):,} tokens IF the "
                f"miss rate is {est.cache.assumed_miss_rate:.1%}", indent="#   - ")
        lines += _wrap_comment(
            "these cannot be separated by any counter vLLM exports "
            "(research/workload_agentic_poc.md section 3).", indent="#   ")
        lines.append("#")
    for key, why in sorted(est.unobservable.items()):
        lines += _wrap_comment(f"{key}: not observable from these metrics: {why}",
                               indent="# ", cont="#     ")
    if est.caveats:
        lines.append("#")
        lines.append("# caveats:")
        for c in est.caveats:
            lines += _wrap_comment(c, indent="#   - ")
    return "\n".join(lines) + "\n"


def emit_json(est: WorkloadEstimate) -> str:
    return json.dumps(est.to_dict(), indent=2, allow_nan=False) + "\n"


def _row(label: str, value: str, unit: str = "") -> str:
    return f"  {label:<34} {value:>16}  {unit}".rstrip()


def emit_table(est: WorkloadEstimate) -> str:
    """The human view: the derived numbers, then every refusal, then caveats."""
    p = est.provenance
    out: list[str] = []
    head = f"{p.source}: {p.target}"
    if p.range:
        head += f"  range {p.range}" + (f" step {p.step}" if p.step else "")
    if p.selector:
        head += f"  selector {p.selector}"
    if p.engine is not None:
        head += f"  engine {p.engine}"
    out.append(head)
    window = ("unknown (cumulative since server start)" if est.hours is None
              else f"{est.hours:.1f} h")
    line = f"  window {window}"
    if est.n_requests is not None:
        line += f"  |  {_tok(est.n_requests):,} requests ({est.n_requests_source})"
    out.append(line)
    if est.gaps:
        out.append(f"  {est.gaps} gap(s) in the gauge series")
    out.append("")
    out.append("  derived")
    if est.prompt is not None:
        out.append(_row("prompt median (log-normal)",
                        f"{_tok(est.prompt.median_tokens):,}", "tokens"))
        out.append(_row("prompt sigma", f"{est.prompt.sigma:.2f}",
                        f"fit over {est.prompt.n_points} edges, residual "
                        f"{est.prompt.residual_pct:.1%}"
                        + (", TAIL CENSORED" if est.prompt.censored else "")))
        if est.prompt.mean_tokens is not None:
            out.append(_row("prompt mean (histogram sum)",
                            f"{_tok(est.prompt.mean_tokens):,}", "tokens"))
    if est.output_mean_tokens is not None:
        out.append(_row("output mean", f"{_tok(est.output_mean_tokens):,}",
                        "tokens/request"))
    if est.output is not None:
        out.append(_row("output median (log-normal)",
                        f"{_tok(est.output.median_tokens):,}",
                        f"tokens, sigma {est.output.sigma:.2f}"
                        + (", TAIL CENSORED" if est.output.censored else "")))
    if est.req_rate_s is not None:
        out.append(_row("request rate", f"{_rate(est.req_rate_s):g}",
                        "req/s (window mean)"))
    if est.mean_running is not None:
        out.append(_row("num_requests_running mean",
                        f"{_r(est.mean_running, 3):g}", "requests in execution"))
    if est.p95_running is not None:
        out.append(_row("num_requests_running p95",
                        f"{_r(est.p95_running, 3):g}", "peak decoding concurrency"))
    if est.little_w_s is not None:
        out.append(_row("Little's law W = L / lambda",
                        f"{_sec(est.little_w_s):g}", "s in execution"))
    if est.e2e_mean_s is not None:
        out.append(_row("mean e2e latency R", f"{_sec(est.e2e_mean_s):g}", "s"))
    if est.queue_mean_s is not None:
        out.append(_row("mean queue time", f"{_sec(est.queue_mean_s):g}", "s"))
    if est.cycle_s is not None:
        out.append(_row("cycle = sessions / rate", f"{_sec(est.cycle_s):g}",
                        f"s at {int(round(est.sessions)):,} sessions"
                        + (" (ASSUMED = p95 decoders)" if est.sessions_assumed
                           else "")))
    if est.think_time_s is not None:
        out.append(_row("think time Z = cycle - R", f"{_think(est.think_time_s):g}", "s"))

    if est.cache is not None:
        c = est.cache
        out.append("")
        out.append("  prefix cache (ONE observable, TWO unknowns)")
        out.append(_row("savings (tokens hit / queried)", f"{c.savings:.1%}",
                        c.savings_source))
        out.append(_row("mean prompt", f"{_tok(c.mean_prompt_tokens):,}", "tokens"))
        if c.miss_rate_given_turn is not None:
            out.append(_row("miss rate", f"{c.miss_rate_given_turn:.1%}",
                            f"IF the warm turn is "
                            f"{_tok(c.assumed_turn_tokens):,} tokens"))
        if c.turn_tokens_given_miss is not None:
            out.append(_row("warm turn",
                            f"{_tok(c.turn_tokens_given_miss):,}",
                            f"tokens IF the miss rate is {c.assumed_miss_rate:.1%}"))
        out.append("  these two cannot be separated by any counter vLLM exports.")

    out.append("")
    out.append("  not observable from these metrics")
    for key, why in sorted(est.unobservable.items()):
        out.append(f"    {key}:")
        out += _wrap_comment(why, indent="      ")
    if est.caveats:
        out.append("")
        out.append("  caveats")
        for c in est.caveats:
            out += _wrap_comment(c, indent="    - ")
    return "\n".join(out) + "\n"


# ===========================================================================
# merging into an existing config
# ===========================================================================
_TABLE = re.compile(r"^\s*\[")


def merge_into(path: str | Path, block: str) -> str:
    """Rewrite ONLY the `[workload]` table of a TOML config, byte-for-byte
    elsewhere.

    A whole-file round trip through a TOML writer would reformat every other
    block, drop the comments the study's configs carry, and make the diff
    unreviewable. So this is a line splice: the `[workload]` header through
    the line before the next table header is replaced, and nothing else in
    the file is touched. An absent `[workload]` block is appended.

    The result is parsed before it is returned, so a block that would not
    read back fails here rather than in the next `ws predict`.
    """
    p = Path(path)
    lines = p.read_text(encoding="utf-8").splitlines()
    start = next((i for i, ln in enumerate(lines)
                  if ln.strip().startswith("[workload]")), None)
    new = block.rstrip("\n").split("\n")
    if start is None:
        out = lines + ([""] if lines and lines[-1].strip() else []) + new + [""]
    else:
        end = next((j for j in range(start + 1, len(lines))
                    if _TABLE.match(lines[j])), len(lines))
        # keep a blank separator line before the next table
        tail = lines[end:]
        out = lines[:start] + new + [""] + tail
    text = "\n".join(out).rstrip("\n") + "\n"
    parsed = tomllib.loads(text)          # refuses to write a file that won't read
    from .config import RunConfig
    RunConfig.from_dict(parsed)           # and refuses a key the schema lacks
    return text


# ===========================================================================
# CLI
# ===========================================================================
def _headers(args) -> dict[str, str]:
    out: dict[str, str] = {}
    for h in getattr(args, "auth_header", None) or []:
        if ":" not in h:
            raise ValueError(f"--auth-header must be 'Name: value', got {h!r}")
        k, v = h.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def _verify(args) -> bool | str:
    if getattr(args, "ca_bundle", None):
        return args.ca_bundle
    return not getattr(args, "insecure", False)


def cmd_workload(args) -> int:
    n_src = sum(bool(x) for x in (args.prometheus, args.jsonl, args.metrics_text))
    if n_src != 1:
        raise ValueError("pick exactly one source: --prometheus URL, "
                         "--jsonl FILE, or --metrics-text FILE")
    if args.prometheus:
        try:
            rd = read_prometheus(args.prometheus, range=args.range, step=args.step,
                                 selector=args.selector or "", engine=args.engine,
                                 headers=_headers(args), verify=_verify(args),
                                 timeout=args.timeout)
        except httpx.HTTPError as e:
            # a transport or status failure is a user-facing message about the
            # endpoint, not a traceback about httpx
            raise ValueError(f"prometheus {args.prometheus}: {e}") from e
    elif args.jsonl:
        rd = read_jsonl(args.jsonl, engine=args.engine)
    else:
        rd = read_metrics_text(args.metrics_text, engine=args.engine)

    est = estimate(rd, turn_tokens=args.assume_turn_tokens,
                   miss_rate=args.assume_miss_rate, sessions=args.sessions)

    emit = "json" if args.json else args.emit
    if args.into:
        text = merge_into(args.into, emit_toml(est))
        Path(args.into).write_text(text, encoding="utf-8")
        # stderr, so `--emit json --into cfg.toml | jq` still parses
        print(f"rewrote the [workload] block of {args.into}", file=sys.stderr)
    if emit == "toml":
        print(emit_toml(est), end="")
    elif emit == "json":
        print(emit_json(est), end="")
    else:
        print(emit_table(est), end="")
    return 0


def add_subparser(sub) -> None:
    """Attach `ws workload` to the top-level subparsers object."""
    p = sub.add_parser(
        "workload", help="characterise the workload, emit the [workload] block",
        description=__doc__,
        epilog="Examples:\n"
               "  ws workload --prometheus http://prom:9090 --range 7d \\\n"
               "      --selector 'model_name=\"Qwen/Qwen3-27B\"' --emit toml\n"
               "  ws workload --jsonl run.jsonl --emit toml "
               "--into workingset.toml\n"
               "  ws workload --metrics-text metrics.txt --json\n"
               "\nEvery emitted number is a ROUNDED AGGREGATE (tokens to 3\n"
               "significant figures, rates to 2, think time to 0.1 s). No raw\n"
               "series value and no timestamp leaves this command.\n",
        formatter_class=__import__("argparse").RawDescriptionHelpFormatter)

    g = p.add_argument_group(
        "sources (pick exactly one)",
        "each answers a different set of questions; a source that cannot "
        "answer one prints 'not observable from these metrics' and why")
    g.add_argument("--prometheus", metavar="URL",
                   help="Prometheus base URL. PromQL over the HTTP API: "
                        "increase() over the whole --range for counters and "
                        "histogram buckets, query_range at --step for the "
                        "num_requests_running gauge. Answers everything.")
    g.add_argument("--jsonl", metavar="FILE",
                   help="a `ws metrics tail --out` archive, delta'd whole-log. "
                        "Answers everything the archive's --keep retained; "
                        "counter resets and bucket-layout changes are reported, "
                        "never bridged.")
    g.add_argument("--metrics-text", metavar="FILE", dest="metrics_text",
                   help="one raw /metrics dump. Its counters are CUMULATIVE "
                        "SINCE SERVER START: distributions and ratios are "
                        "usable, every RATE (request rate, Little's law, think "
                        "time) is not observable, because the dump carries no "
                        "duration.")

    g = p.add_argument_group("prometheus options")
    g.add_argument("--range", default="7d",
                   help="window ending now, e.g. 7d / 12h / 90m (default 7d)")
    g.add_argument("--step", default="5m",
                   help="query_range step for the gauge series (default 5m). "
                        "Steps carrying no sample are counted as scrape gaps.")
    g.add_argument("--selector", default="",
                   help="extra PromQL label matcher, e.g. "
                        "'model_name=\"Qwen/Qwen3-27B\"'. Passed through "
                        "verbatim and echoed in the provenance.")
    g.add_argument("--auth-header", action="append", metavar="'Name: value'",
                   help="extra request header; repeatable. Prefer this over a "
                        "key in the URL, which lands in shell history.")
    g.add_argument("--ca-bundle", help="CA bundle to verify TLS against (the "
                                       "right fix for an interception proxy)")
    g.add_argument("--insecure", action="store_true",
                   help="skip TLS verification (internal endpoints only)")
    g.add_argument("--timeout", type=float, default=30.0,
                   help="per-query timeout, seconds (default 30)")

    g = p.add_argument_group(
        "assumptions",
        "the prefix-cache savings is ONE observable over TWO unknowns; each "
        "of these fixes one so the other can be read off, and both readings "
        "are always printed with the assumption they needed")
    g.add_argument("--assume-turn-tokens", type=float,
                   default=DEFAULT_TURN_TOKENS, metavar="N",
                   help=f"warm-turn size, tokens (default "
                        f"{DEFAULT_TURN_TOKENS:.0f}, the study's): fixes it so "
                        f"the effective miss rate can be read off")
    g.add_argument("--assume-miss-rate", type=float, default=DEFAULT_MISS_RATE,
                   metavar="F",
                   help=f"miss rate in [0, 1] (default {DEFAULT_MISS_RATE}, the "
                        f"config's): fixes it so the warm-turn size can be read "
                        f"off")
    g.add_argument("--sessions", type=float, metavar="N",
                   help="concurrent sessions the think-time cycle is computed "
                        "at. Default: the num_requests_running p95, which "
                        "counts concurrent DECODERS and is a LOWER BOUND — "
                        "sessions-in-cache is not observable from these metrics.")

    g = p.add_argument_group("output")
    g.add_argument("--emit", choices=("table", "toml", "json"), default="table",
                   help="table (default): the derived numbers, every refusal "
                        "and every caveat. toml: a [workload] block ready to "
                        "paste. json: the full estimate with provenance, fit "
                        "residuals and caveats.")
    g.add_argument("--json", action="store_true", help="shorthand for --emit json")
    g.add_argument("--into", metavar="workingset.toml",
                   help="rewrite just the [workload] block of this config, "
                        "leaving every other block byte-for-byte as it was")
    p.add_argument("--engine", help="engine index to select under DP>1 "
                                    "(default: sum every engine)")
    p.set_defaults(fn=cmd_workload)
