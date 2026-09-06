"""`MetricsSampler` — poll a `/metrics` endpoint and keep every raw snapshot.

The rule this module exists to enforce: **raw snapshots are kept, never only
derived values.** vLLM's counters flush MID-request -- a request's whole FLOP
count or prefill time lands in whichever scrape happens to follow the flush --
so a per-scrape RATE is noise and only a delta across a WHOLE window is a
measurement. You cannot know at scrape time which window you will want, so
every scrape is retained (in memory, and optionally appended to JSONL) and the
window arithmetic happens afterwards, offline, as many times as you like.

    async with MetricsSampler(url, interval=1.0) as s:
        await s.wait_first()
        t0 = time.time(); await drive_load(); t1 = time.time()
        await s.next_tick()                  # the enclosing endpoint
        w = s.window(t0, t1)
    print(w.tokens_out, w.ttft.quantile(0.95))

Window endpoints must ENCLOSE [t0, t1]: the low one's scrape COMPLETED before
t0 and the high one's STARTED after t1, so the counter delta provably covers
the whole interval. When no such pair exists the window raises
`WindowNotCovered` rather than falling back to a nearby pair and quietly
answering a different question -- which is why the `next_tick()` above is
part of the pattern and not a nicety. Each endpoint is still fuzzy within its
own round trip, so `WindowDelta` carries `t_lo_uncertainty` /
`t_hi_uncertainty` (seconds) and a caller dividing by `dt` knows how sharp
the divisor is.

Units: `t_sent` and every timestamp are UNIX SECONDS (`time.time()`); `rtt`,
`dt` and the uncertainties are SECONDS.
"""
from __future__ import annotations

import asyncio
import bisect
import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import httpx

from .adapter import KEY_KIND, SEMANTIC_KEYS, Resolution
from .parse import (Histogram, HistogramMismatch, HistogramReset, Sample,
                    parse_text)
from .vllm import detect_adapter

__all__ = ["Snapshot", "GaugeStats", "WindowDelta", "MetricsSampler",
           "DECODE_KEEP", "keep_filter", "load_jsonl", "window_from_snapshots",
           "WindowNotCovered", "SamplerStopped"]


class WindowNotCovered(ValueError):
    """No pair of snapshots encloses the requested interval.

    Deliberately not a number: the alternative is answering a question about
    a stretch of time the caller did not ask about.
    """


class SamplerStopped(RuntimeError):
    """The sampling loop died (a failing --out writer, typically)."""

# The series decode-side analysis reads -- the same preset
# scripts/scrape_metrics.py has always written, kept name-for-name so an old
# log and a new one answer the same questions. Writing only these shrinks a
# log ~8x, which matters when it has to travel: a full dump is ~100 series
# every interval, ~1 GB/day at 0.5 s. It DROPS the prefix-cache counters and
# the latency histograms, so a log recorded with it can give you decode
# speed but not a measured miss rate or a TTFT quantile.
#
# ENGINE-SPECIFIC, and knowingly so: these are vLLM substrings living in an
# otherwise engine-generic module, as is the unconditional `detect_adapter`
# import from `.vllm`. Both are the same debt -- vLLM is the only adapter
# that exists. When a second one lands, the preset moves onto the adapter
# (a `keep_preset()` classmethod) and detection moves to a registry module;
# neither changes a caller, which is what the adapter seam is for.
DECODE_KEEP: tuple[str, ...] = (
    "iteration_tokens_total", "num_requests_running", "num_requests_waiting",
    "prompt_tokens_total", "generation_tokens_total", "cache_usage_perc",
    "request_success_total", "spec_decode_", "model_forward_time",
)


def keep_filter(spec: str | None) -> tuple[str, ...] | None:
    """`--keep` -> substring tuple, or None for "keep everything".

    'all' keeps the dump whole (the FLOP reconciliation needs series the
    decode preset drops); 'decode' is `DECODE_KEEP`; anything else is a
    comma-separated list of name substrings.
    """
    if spec is None or spec == "all":
        return None
    if spec == "decode":
        return DECODE_KEEP
    subs = tuple(x.strip() for x in spec.split(",") if x.strip())
    return subs or None


@dataclass
class Snapshot:
    """One scrape. `samples` is the raw parse; nothing is derived here.

    t_sent  unix seconds, when the GET went out
    rtt     seconds the GET took. The counters are as-of SOME instant in
            [t_sent, t_sent + rtt]; `t` takes the midpoint, which is the
            unbiased choice and the best a client-side scrape can do.
    ok      False for a failed scrape; `samples` is then empty and `error`
            says why. Failures are RETAINED: a gap in the series is itself
            evidence about the run.
    """

    t_sent: float
    rtt: float
    samples: list[Sample] = field(default_factory=list)
    ok: bool = True
    error: str | None = None
    lines: list[str] | None = None      # raw text lines, when retained

    @property
    def t(self) -> float:
        """Midpoint of the scrape's round trip, unix seconds."""
        if not math.isfinite(self.rtt):
            return self.t_sent
        return self.t_sent + self.rtt / 2.0

    @property
    def uncertainty(self) -> float:
        """Half-width of this endpoint's timestamp, seconds."""
        return self.rtt / 2.0 if math.isfinite(self.rtt) else float("nan")

    def covers_before(self, t: float) -> bool:
        """This scrape had COMPLETED by `t`, so its counters are as-of <= t.

        The whole round trip has to be over, not just its midpoint: the
        server could have read its counters at the last instant before the
        response left.
        """
        rtt = 0.0 if math.isnan(self.rtt) else self.rtt
        return self.t_sent + rtt <= t

    def starts_after(self, t: float) -> bool:
        """This scrape had not STARTED by `t`, so its counters are as-of >= t."""
        return self.t_sent >= t

    def to_json(self) -> str:
        """One JSONL record. The RAW LINES are what is stored, so the file
        can be re-parsed by any later version of this package."""
        rec: dict[str, Any] = {"t_sent": round(self.t_sent, 4),
                               "rtt": None if not math.isfinite(self.rtt)
                               else round(self.rtt, 5),
                               "ok": self.ok}
        if self.error:
            rec["error"] = self.error
        rec["lines"] = self.lines if self.lines is not None else [
            _render(s) for s in self.samples]
        return json.dumps(rec, separators=(",", ":"))

    @classmethod
    def from_json(cls, line: str) -> "Snapshot":
        rec = json.loads(line)
        rtt = rec.get("rtt")
        lines = rec.get("lines") or []
        return cls(t_sent=float(rec["t_sent"]),
                   rtt=float("nan") if rtt is None else float(rtt),
                   samples=parse_text("\n".join(lines)),
                   ok=bool(rec.get("ok", True)), error=rec.get("error"),
                   lines=lines)


def _render(s: Sample) -> str:
    """A Sample back into a `/metrics` line, escaped so it re-parses.

    The raw lines ARE the archive format, so a round trip has to be exact:
    a label value carrying `"` or `\\` would otherwise re-parse into
    different labels, or be dropped. Only reached for a Snapshot built
    without `lines` -- the sampler always keeps the server's own text.
    """
    if s.labels:
        inner = ",".join(f'{k}="{_escape(v)}"' for k, v in s.labels.items())
        return f"{s.name}{{{inner}}} {s.value!r}"
    return f"{s.name} {s.value!r}"


def _escape(v: str) -> str:
    return v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


@dataclass(frozen=True)
class GaugeStats:
    """A gauge summarised over a window. Units: `KEY_UNITS[key]`.

    `n` is how many snapshots carried the series -- 0 means the gauge is
    unexported or every scrape in the window failed, and mean/max are None.
    """

    n: int = 0
    mean: float | None = None
    min: float | None = None
    max: float | None = None
    first: float | None = None
    last: float | None = None

    @classmethod
    def of(cls, values: Sequence[float]) -> "GaugeStats":
        vals = [v for v in values if v is not None and math.isfinite(v)]
        if not vals:
            return cls()
        return cls(n=len(vals), mean=sum(vals) / len(vals), min=min(vals),
                   max=max(vals), first=vals[0], last=vals[-1])

    def to_dict(self) -> dict:
        return {"n": self.n, "mean": self.mean, "min": self.min,
                "max": self.max, "first": self.first, "last": self.last}


@dataclass
class WindowDelta:
    """Everything that happened between two snapshots, and nothing derived
    from a single scrape.

    Endpoints are the nearest snapshots OUTSIDE the requested [t0, t1], so
    the counter deltas COVER the interval and may over-count by up to one
    scrape interval at each end. That direction is deliberate: a mid-request
    counter flush that lands just after t1 belongs to work started inside
    the window, and losing it would silently under-report.

    dt                 seconds between the two endpoints' round-trip midpoints
    t_lo_uncertainty   seconds; half the low endpoint's rtt
    t_hi_uncertainty   seconds; half the high endpoint's rtt
    counters           semantic key -> delta over the window
    histograms         semantic key -> bucket-wise Histogram delta
    gauges             semantic key -> GaugeStats over every snapshot in range
    invalid            semantic key -> why no delta exists for it here (a
                       counter that went backwards, a bucket layout that
                       changed). Those keys are None in `counters` /
                       `histograms` rather than carrying a fabricated
                       number, and a non-empty dict invalidates the window
                       for anything they feed.
    """

    t0: float
    t1: float
    lo: Snapshot
    hi: Snapshot
    counters: dict[str, float | None] = field(default_factory=dict)
    histograms: dict[str, Histogram | None] = field(default_factory=dict)
    gauges: dict[str, GaugeStats] = field(default_factory=dict)
    per_position: dict[int, float] | None = None
    n_snapshots: int = 0
    n_failed: int = 0
    version_hint: str = "unknown"
    invalid: dict[str, str] = field(default_factory=dict)

    @property
    def counter_resets(self) -> tuple[str, ...]:
        """The `invalid` keys whose reason was a counter going backwards."""
        return tuple(k for k, why in self.invalid.items() if "backwards" in why)

    # ---- timing ----------------------------------------------------------
    @property
    def dt(self) -> float:
        """Seconds covered by the endpoints. Divide counter deltas by this,
        but see the two uncertainties before believing the third digit."""
        return self.hi.t - self.lo.t

    @property
    def t_lo_uncertainty(self) -> float:
        return self.lo.uncertainty

    @property
    def t_hi_uncertainty(self) -> float:
        return self.hi.uncertainty

    @property
    def dt_uncertainty(self) -> float:
        """Worst-case error on `dt`, seconds (the two half-rtts add)."""
        a, b = self.t_lo_uncertainty, self.t_hi_uncertainty
        a = 0.0 if math.isnan(a) else a
        b = 0.0 if math.isnan(b) else b
        return a + b

    # ---- counters --------------------------------------------------------
    def counter(self, key: str) -> float | None:
        return self.counters.get(key)

    def rate(self, key: str) -> float | None:
        """Counter delta per second over the window. None if either is missing."""
        v = self.counters.get(key)
        if v is None or self.dt <= 0:
            return None
        return v / self.dt

    @property
    def tokens_in(self) -> float | None:
        """Prompt tokens processed in the window."""
        return self.counters.get("prompt_tokens_total")

    @property
    def tokens_out(self) -> float | None:
        """Generated tokens in the window."""
        return self.counters.get("generation_tokens_total")

    @property
    def output_tok_s(self) -> float | None:
        """Aggregate generation throughput, tokens/second."""
        return self.rate("generation_tokens_total")

    @property
    def requests_finished(self) -> float | None:
        return self.counters.get("request_success_total")

    @property
    def preemptions(self) -> float | None:
        return self.counters.get("preemptions_total")

    # ---- prefix cache ----------------------------------------------------
    @property
    def prefix_hit_rate(self) -> float | None:
        """Tokens hit / tokens queried over the window, in [0, 1].

        This is a WINDOW rate, not vLLM's decaying `hit_rate` gauge: the two
        counters are what make it computable, which is why V0 (gauge only)
        cannot answer this question at all.
        """
        q = self.counters.get("prefix_cache_queries_total")
        h = self.counters.get("prefix_cache_hits_total")
        if q is None or h is None or q <= 0:
            return None
        return h / q

    @property
    def miss_rate(self) -> float | None:
        """1 - prefix_hit_rate. The study's `workload.miss_rate`, measured."""
        r = self.prefix_hit_rate
        return None if r is None else 1.0 - r

    # ---- speculative decoding --------------------------------------------
    @property
    def alpha(self) -> float | None:
        """Per-DRAFT-TOKEN acceptance probability in [0, 1]:
        accepted tokens / drafted tokens over the window."""
        d = self.counters.get("spec_decode_num_draft_tokens_total")
        a = self.counters.get("spec_decode_num_accepted_tokens_total")
        if d is None or a is None or d <= 0:
            return None
        return a / d

    @property
    def mean_accepted_len(self) -> float | None:
        """Tokens emitted per sequence per forward pass: 1 + accepted/drafts.

        The `1 +` is the model's own token, which is always emitted; the
        drafts only add to it. This is the measured replacement for the
        study's `mtp` slider.
        """
        drafts = self.counters.get("spec_decode_num_drafts_total")
        a = self.counters.get("spec_decode_num_accepted_tokens_total")
        if drafts is None or a is None or drafts <= 0:
            return None
        return 1.0 + a / drafts

    @property
    def draft_width(self) -> float | None:
        """Tokens proposed per draft event (the speculator's k)."""
        drafts = self.counters.get("spec_decode_num_drafts_total")
        d = self.counters.get("spec_decode_num_draft_tokens_total")
        if drafts is None or d is None or drafts <= 0:
            return None
        return d / drafts

    # ---- engine steps ----------------------------------------------------
    @property
    def steps(self) -> float | None:
        """Engine steps THAT EMITTED OUTPUT, from the iteration-tokens
        histogram's observation count.

        A LOWER BOUND on forward passes, not a count of them: vLLM builds
        `IterationStats` only when a step produced outputs
        (`async_llm.py`: `IterationStats() if (log_stats and num_outputs)`),
        so a pure chunked-prefill chunk that emitted no token is never
        observed. Anything dividing by this -- `step_time_s` below -- is
        therefore an UPPER bound on time per pass.
        """
        h = self.histograms.get("iteration_tokens_hist")
        return None if h is None else h.observations

    @property
    def tokens_per_step(self) -> float | None:
        """Mean tokens processed per forward pass."""
        h = self.histograms.get("iteration_tokens_hist")
        return None if h is None else h.mean()

    @property
    def step_time_s(self) -> float | None:
        """Wall seconds per OUTPUT-EMITTING engine step -- an upper bound on
        time per forward pass, since `steps` under-counts (see above). Only
        as sharp as `dt_uncertainty`."""
        st = self.steps
        if not st or self.dt <= 0:
            return None
        return self.dt / st

    # ---- histograms ------------------------------------------------------
    def hist(self, key: str) -> Histogram | None:
        return self.histograms.get(key)

    @property
    def ttft(self) -> Histogram | None:
        """Time-to-first-token distribution of requests that FINISHED their
        prefill in the window. Seconds."""
        return self.histograms.get("ttft_hist")

    @property
    def tpot(self) -> Histogram | None:
        """Inter-token latency, seconds -- one observation per output EVENT,
        NOT per token.

        Under speculative decoding a step that accepts 3 tokens records ONE
        observation covering all three, so this is seconds-per-step and
        reads high; `request_tpot_hist` divides by the real token count and
        is the per-token figure. The first output of a request contributes
        to TTFT instead, so the count is (output events - requests started).
        """
        return self.histograms.get("tpot_hist")

    @property
    def request_tpot(self) -> Histogram | None:
        """Per-request mean time per output token, seconds: one observation
        per finished request, and a true per-token average even under
        speculative decoding."""
        return self.histograms.get("request_tpot_hist")

    @property
    def e2e(self) -> Histogram | None:
        """End-to-end request latency, seconds."""
        return self.histograms.get("e2e_hist")

    @property
    def prefill_time(self) -> Histogram | None:
        """Engine-side prefill time, seconds -- free of the proxy tax that
        inflates a client-measured TTFT."""
        return self.histograms.get("prefill_time_hist")

    @property
    def decode_time(self) -> Histogram | None:
        """Engine-side decode time, seconds."""
        return self.histograms.get("decode_time_hist")

    # ---- gauges ----------------------------------------------------------
    def gauge(self, key: str) -> GaugeStats:
        return self.gauges.get(key, GaugeStats())

    @property
    def running(self) -> GaugeStats:
        """Batch size over the window, requests."""
        return self.gauge("requests_running")

    @property
    def waiting(self) -> GaugeStats:
        """Queue depth over the window, requests."""
        return self.gauge("requests_waiting")

    @property
    def kv_usage(self) -> GaugeStats:
        """KV pool occupancy over the window, FRACTION in [0, 1]."""
        return self.gauge("kv_cache_usage")

    # ---- output ----------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "requested": {"t0": self.t0, "t1": self.t1},
            "endpoints": {"lo": self.lo.t, "hi": self.hi.t,
                          "lo_uncertainty_s": _num(self.t_lo_uncertainty),
                          "hi_uncertainty_s": _num(self.t_hi_uncertainty)},
            "dt_s": self.dt, "dt_uncertainty_s": _num(self.dt_uncertainty),
            "n_snapshots": self.n_snapshots, "n_failed": self.n_failed,
            "version_hint": self.version_hint,
            "invalid": dict(self.invalid),
            "counters": {k: v for k, v in self.counters.items() if v is not None},
            "gauges": {k: g.to_dict() for k, g in self.gauges.items() if g.n},
            "histograms": {k: {"count": h.observations, "mean": h.mean(),
                               "p50": h.quantile(0.5), "p95": h.quantile(0.95),
                               "p99": h.quantile(0.99)}
                           for k, h in self.histograms.items()
                           if h is not None and h.observations},
            "derived": {"output_tok_s": self.output_tok_s,
                        "prefix_hit_rate": self.prefix_hit_rate,
                        "miss_rate": self.miss_rate,
                        "alpha": self.alpha,
                        "mean_accepted_len": self.mean_accepted_len,
                        "draft_width": self.draft_width,
                        "steps": self.steps,
                        "tokens_per_step": self.tokens_per_step,
                        "step_time_s": self.step_time_s},
            "spec_decode_accepted_per_pos": self.per_position,
        }


def _num(x: float) -> float | None:
    return None if x is None or math.isnan(x) else x


# ---------------------------------------------------------------------------
# window arithmetic — pure, so it works on a live sampler or a replayed JSONL
# ---------------------------------------------------------------------------
def _detect_invalid(snaps: Sequence[Snapshot], adapter) -> dict[str, str]:
    """Keys no delta can be taken for, found PER SERIES across CONSECUTIVE
    snapshots -- the only view that can see a reset at all.

    Two endpoints cannot: a counter that goes 100 -> 10 -> 150 shows a
    perfectly plausible +50, and under DP a restarting engine (700k -> 20k)
    hides inside a rising sum from its neighbours. Both are invisible
    unless every retained snapshot is compared to its predecessor, and
    unless the comparison happens BEFORE label sets are aggregated.
    """
    bad: dict[str, str] = {}
    prev_c = prev_h = None
    for s in snaps:
        cur_c = _bulk(adapter, "all_series", s, "counter")
        cur_h = _bulk(adapter, "all_histogram_series", s, "histogram")
        if prev_c is not None:
            for key, sa in prev_c.items():
                if key in bad:
                    continue
                sb = cur_c.get(key)
                if not sb:
                    continue
                for lbl, va in sa.items():
                    vb = sb.get(lbl)
                    if vb is not None and vb < va:
                        bad[key] = (f"counter went backwards "
                                    f"({va:g} -> {vb:g}); engine restarted")
                        break
            for key, ha in prev_h.items():
                if key in bad:
                    continue
                hb = cur_h.get(key)
                if not hb:
                    continue
                for lbl, x in ha.items():
                    y = hb.get(lbl)
                    if y is None:
                        continue
                    try:
                        y - x
                    except (HistogramReset, HistogramMismatch) as e:
                        bad[key] = str(e).split(": ", 1)[-1]
                        break
        prev_c, prev_h = cur_c, cur_h
    return bad


def _bulk(adapter, method: str, snap: Snapshot, kind: str) -> dict[str, dict]:
    """One snapshot's per-label-set view, preferring the adapter's one-pass
    bulk read and falling back to the per-key accessors an adapter without
    one still provides."""
    fn = getattr(adapter, method, None)
    if fn is not None:
        return fn(snap.samples)
    per_key = (adapter.series if kind == "counter" else adapter.histogram_series)
    out = {}
    for key, k in KEY_KIND.items():
        if k != kind and not (kind == "counter" and k == "gauge"):
            continue
        got = per_key(snap.samples, key)
        if got:
            out[key] = got
    return out


def window_from_snapshots(snaps: Sequence[Snapshot], t0: float | None = None,
                          t1: float | None = None, adapter=None,
                          engine: str | None = None) -> WindowDelta:
    """The `WindowDelta` for [t0, t1] over an already-collected series.

    Endpoints must ENCLOSE the interval, which is stricter than "nearest
    outside" and is what makes the delta a measurement of [t0, t1] rather
    than of some nearby stretch:

        lo.t_sent + lo.rtt <= t0     lo's scrape COMPLETED before t0, so its
                                     counters are as-of a time <= t0
        hi.t_sent          >= t1     hi's scrape STARTED after t1, so its
                                     counters are as-of a time >= t1

    If no such pair exists the window is NOT covered and this raises
    `WindowNotCovered` -- it never falls back to an interior pair. Asking
    for [10, 20] of a run that only has snapshots at t=0 and t=1 used to
    return those two, silently answering a different question.

    The common way to hit this is calling `window()` the instant load
    finishes, before the next tick has landed: the enclosing `hi` does not
    exist yet. Await one more tick first (`await s.next_tick()`).

    `t0=None` / `t1=None` mean "the earliest / latest snapshot", for
    reporting on a whole log. No coverage is claimed for an open end,
    because the endpoint IS the limit of what was recorded.
    """
    ok = sorted((s for s in snaps if s.ok and s.samples), key=lambda s: s.t)
    if len(ok) < 2:
        raise ValueError(f"need >= 2 successful snapshots to make a window, "
                         f"have {len(ok)}")
    if t0 is not None and t1 is not None and t1 < t0:
        t0, t1 = t1, t0

    if t0 is None:
        i = 0
    else:
        covering = [k for k, s in enumerate(ok) if s.covers_before(t0)]
        if not covering:
            raise WindowNotCovered(
                f"no snapshot completed before t0={t0:.3f}: the earliest "
                f"finished at {ok[0].t_sent + _rtt(ok[0]):.3f}. The window is "
                f"not covered on the low side.")
        i = covering[-1]
    if t1 is None:
        j = len(ok) - 1
    else:
        after = [k for k, s in enumerate(ok) if s.starts_after(t1)]
        if not after:
            raise WindowNotCovered(
                f"no snapshot started after t1={t1:.3f}: the last began at "
                f"{ok[-1].t_sent:.3f}. The window is not covered on the high "
                f"side -- await one more scrape before asking for it.")
        j = after[0]
    if j <= i:
        raise WindowNotCovered(
            f"the enclosing snapshots collapse to one ({ok[i].t:.3f}); "
            f"[{t0}, {t1}] is shorter than the scrape interval")
    lo, hi = ok[i], ok[j]

    adapter = adapter or detect_adapter(hi.samples, engine=engine)
    inside = [s for s in snaps if lo.t <= s.t <= hi.t]
    inside_ok = [s for s in ok if lo.t <= s.t <= hi.t]

    invalid = _detect_invalid(inside_ok, adapter)

    counters: dict[str, float | None] = {}
    for key, kind in KEY_KIND.items():
        if kind != "counter":
            continue
        if key in invalid:
            counters[key] = None
            continue
        a = adapter.counter(lo.samples, key)
        b = adapter.counter(hi.samples, key)
        counters[key] = None if a is None or b is None else b - a

    histograms: dict[str, Histogram | None] = {}
    for key, kind in KEY_KIND.items():
        if kind != "histogram":
            continue
        if key in invalid:
            histograms[key] = None
            continue
        a = adapter.histogram(lo.samples, key)
        b = adapter.histogram(hi.samples, key)
        if a is None or b is None:
            histograms[key] = None
            continue
        try:
            histograms[key] = b - a
        except (HistogramReset, HistogramMismatch) as e:
            histograms[key] = None
            invalid[key] = str(e).split(": ", 1)[-1]

    gauges: dict[str, GaugeStats] = {}
    for key, kind in KEY_KIND.items():
        if kind != "gauge":
            continue
        vals = [v for v in (adapter.gauge(s.samples, key) for s in inside_ok)
                if v is not None]
        gauges[key] = GaugeStats.of(vals)

    per_pos = None
    if "spec_decode_accepted_per_pos" not in invalid:
        a_pos = adapter.by_position(lo.samples, "spec_decode_accepted_per_pos")
        b_pos = adapter.by_position(hi.samples, "spec_decode_accepted_per_pos")
        if a_pos is not None and b_pos is not None:
            d_pos = {p: b_pos.get(p, 0.0) - a_pos.get(p, 0.0) for p in sorted(b_pos)}
            if any(v < 0 for v in d_pos.values()):
                invalid["spec_decode_accepted_per_pos"] = "counter went backwards"
            else:
                per_pos = d_pos

    hint = adapter.resolution().version_hint if hasattr(adapter, "resolution") else "?"
    return WindowDelta(t0=lo.t if t0 is None else t0,
                       t1=hi.t if t1 is None else t1,
                       lo=lo, hi=hi, counters=counters,
                       histograms=histograms, gauges=gauges,
                       per_position=per_pos, n_snapshots=len(inside),
                       n_failed=len(inside) - len(inside_ok), version_hint=hint,
                       invalid=invalid)


def _rtt(s: Snapshot) -> float:
    return 0.0 if math.isnan(s.rtt) else s.rtt


def load_jsonl(path) -> list[Snapshot]:
    """Replay a `ws metrics tail --out` log. Malformed lines are skipped:
    a log truncated by Ctrl-C ends mid-line more often than not."""
    out: list[Snapshot] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(Snapshot.from_json(line))
            except Exception:
                continue
    out.sort(key=lambda s: s.t_sent)
    return out


# ---------------------------------------------------------------------------
# the sampler
# ---------------------------------------------------------------------------
class MetricsSampler:
    """Poll `/metrics` on a timer, keep every raw snapshot.

        async with MetricsSampler(url, interval=1.0) as s:
            await drive_the_load()
            w = s.window(t_start, t_end)

    url        the full `/metrics` URL (`Endpoint.metrics_url`)
    interval   seconds between scrape STARTS; a slow scrape shortens the
               following sleep rather than sliding the whole schedule, so
               ticks stay on a grid and a stall shows as a gap, not a drift
    headers    extra request headers (an `Authorization` bearer, typically)
    verify     TLS: True (system trust), False (skip -- internal endpoints
               only), or a path to a CA bundle. A corporate interception
               proxy presents its own certificate, so the system store is
               not enough and the bundle is the right fix.
    out        path to append raw JSONL to, one record per scrape
    keep       substring tuple limiting which series are retained
               (`keep_filter`); None keeps the dump whole
    max_snapshots  ring-buffer bound on the in-memory series; None = unbounded.
               The JSONL file, when given, is never trimmed.
    client     an injected `httpx.AsyncClient` (tests pass one built on
               `httpx.MockTransport`); the sampler will not close what it
               did not open.
    """

    def __init__(self, url: str, interval: float = 1.0, *,
                 headers: dict[str, str] | None = None,
                 verify: bool | str = True,
                 out: str | None = None,
                 keep: Iterable[str] | None = None,
                 timeout: float = 5.0,
                 max_snapshots: int | None = None,
                 engine: str | None = None,
                 client: httpx.AsyncClient | None = None):
        if interval <= 0:
            raise ValueError("interval must be > 0 seconds")
        self.url = url
        self.interval = float(interval)
        self.headers = dict(headers or {})
        self.verify = verify
        self.timeout = timeout
        self.keep = tuple(keep) if keep else None
        self.max_snapshots = max_snapshots
        self.engine = engine
        self.snapshots: list[Snapshot] = []
        self.adapter = None
        self.n_failed = 0
        self.error: str | None = None      # set when the loop dies terminally
        self._epoch_wall = time.time()
        self._epoch_mono = time.monotonic()
        self._out_path = out
        self._out = None
        self._client = client
        self._owns_client = client is None
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._first = asyncio.Event()      # set once one scrape has landed

    # ---- lifecycle -------------------------------------------------------
    async def __aenter__(self) -> "MetricsSampler":
        await self.start()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    async def start(self) -> None:
        if self._task is not None:
            return
        # The log file opens FIRST: an unwritable --out path must fail before
        # a client exists to leak. `__aenter__` raising means `__aexit__`
        # never runs, so nothing opened here would ever be closed.
        if self._out_path:
            self._out = open(self._out_path, "a", buffering=1, encoding="utf-8")
        if self._client is None:
            self._client = httpx.AsyncClient(verify=self.verify,
                                             timeout=self.timeout)
        self._stop = asyncio.Event()
        self._first = asyncio.Event()
        self._task = asyncio.create_task(self._loop(), name="metrics-sampler")

    async def aclose(self) -> None:
        """Stop the loop and release everything, whatever went wrong.

        Every step is in a `finally` chain: a task that refuses to end must
        not strand the client, and a client that fails to close must not
        strand the log file. `aclose` never raises -- inspect `.error` for
        a loop that died.
        """
        self._stop.set()
        try:
            if self._task is not None:
                task, self._task = self._task, None
                try:
                    await asyncio.wait_for(asyncio.shield(task),
                                           timeout=self.timeout + 1)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    task.cancel()
                    # a cancelled task is not finished until it is AWAITED
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        pass
                except Exception:
                    pass          # the loop's own failure is on self.error
        finally:
            try:
                if self._owns_client and self._client is not None:
                    client, self._client = self._client, None
                    await client.aclose()
            finally:
                if self._out is not None:
                    out, self._out = self._out, None
                    try:
                        out.close()
                    except OSError:
                        pass

    def raise_if_failed(self) -> None:
        """Re-raise the loop's terminal failure, if it had one."""
        if self.error is not None:
            raise SamplerStopped(self.error)

    async def wait_first(self, timeout: float = 10.0) -> bool:
        """Block until one SUCCESSFUL scrape has landed. Handy before
        starting load: a window needs an endpoint before t0, and a failed
        scrape is not an endpoint -- it carries no counters."""
        try:
            await asyncio.wait_for(self._first.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def next_tick(self, timeout: float | None = None) -> bool:
        """Wait for one more snapshot than are held right now.

        The pattern for closing a window on live load: the enclosing `hi`
        endpoint does not exist until a scrape STARTS after your t1, so

            t1 = time.time(); await s.next_tick(); w = s.window(t0, t1)

        is what turns a `WindowNotCovered` into a measurement.
        """
        target = len(self.snapshots) + 1
        deadline = (timeout if timeout is not None
                    else self.interval * 2 + self.timeout + 1)
        try:
            async with asyncio.timeout(deadline):
                while len(self.snapshots) < target:
                    if self.error is not None:
                        raise SamplerStopped(self.error)
                    await asyncio.sleep(min(0.01, self.interval / 4))
            return True
        except asyncio.TimeoutError:
            return False

    # ---- the loop --------------------------------------------------------
    def now(self) -> float:
        """THIS SAMPLER'S CLOCK, in wall-clock seconds that never run
        backwards.

        `time.time()` can step back (NTP), which would break the sort and
        the bisect that every window depends on. Anchoring an offset from
        `time.monotonic()` to one wall reading keeps timestamps comparable
        with a caller's own `time.time()` while staying ordered.

        PUBLIC because it is the only correct source of the `t` a caller
        hands to `at()` or `window()`. Every snapshot is stamped with it, so
        a caller timing its own work against `time.monotonic()` (which the
        probe layer does, for span arithmetic) must convert through this and
        not through its own clock -- the two differ by the unix epoch, and
        asking for a monotonic instant lands before the whole series.
        """
        return self._epoch_wall + (time.monotonic() - self._epoch_mono)

    # the private spelling this had before it needed to be part of the
    # sampler's contract; kept so an existing caller does not break
    _now = now

    async def _loop(self) -> None:
        try:
            while not self._stop.is_set():
                started = time.monotonic()
                await self.scrape_once()
                elapsed = time.monotonic() - started
                delay = max(0.0, self.interval - elapsed)
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=delay)
                except asyncio.TimeoutError:
                    pass
        except asyncio.CancelledError:
            raise
        except BaseException as e:            # noqa: BLE001
            # A dead loop must not look like a quiet one. Without this the
            # --out writer failing on a full disk stops the series while the
            # consumer waits forever on a snapshot count that never rises.
            self.error = f"{type(e).__name__}: {e}"
            self._stop.set()
            self._first.set()                 # release anyone blocked on it

    async def scrape_once(self) -> Snapshot:
        """One scrape, recorded. A FETCH failure never raises -- it becomes a
        snapshot with `ok=False`, because a gap in the series is evidence
        too. A RECORDING failure (the log writer) does raise: it means the
        archive is incomplete, which is not something to paper over."""
        t_sent = self._now()
        try:
            r = await self._client.get(self.url, headers=self.headers,
                                       timeout=self.timeout)
            r.raise_for_status()
            rtt = self._now() - t_sent
            snap = self._make(t_sent, rtt, r.text)
        except Exception as e:                       # noqa: BLE001 -- retained
            self.n_failed += 1
            snap = Snapshot(t_sent=t_sent, rtt=float("nan"), samples=[],
                            ok=False, error=f"{type(e).__name__}: {e}", lines=[])
        self._record(snap)
        return snap

    def _make(self, t_sent: float, rtt: float, text: str) -> Snapshot:
        lines = [ln for ln in text.splitlines()
                 if ln and not ln.startswith("#")
                 and (self.keep is None or any(k in ln for k in self.keep))]
        samples = parse_text("\n".join(lines))
        self._latch_adapter(samples)
        return Snapshot(t_sent=t_sent, rtt=rtt, samples=samples, ok=True, lines=lines)

    def _latch_adapter(self, samples: list[Sample]) -> None:
        """Keep the RICHEST resolution seen so far, not the first.

        Latching on scrape #1 is a trap: a truncated or mid-startup first
        response resolves half the vocabulary and every later window is read
        through that impoverished view, reporting keys as missing that the
        server exports. Re-detect while the resolution is still incomplete
        and keep whichever adapter resolved more.
        """
        if not samples:
            return
        if self.adapter is None:
            self.adapter = detect_adapter(samples, engine=self.engine)
            return
        have = len(self.adapter.resolution().resolved)
        if have >= len(SEMANTIC_KEYS):
            return                            # nothing left to improve on
        cand = detect_adapter(samples, engine=self.engine)
        if len(cand.resolution().resolved) > have:
            self.adapter = cand

    def _record(self, snap: Snapshot) -> None:
        self.snapshots.append(snap)
        if self.max_snapshots and len(self.snapshots) > self.max_snapshots:
            del self.snapshots[: len(self.snapshots) - self.max_snapshots]
        if self._out is not None:
            self._out.write(snap.to_json() + "\n")
        if snap.ok and snap.samples:
            self._first.set()

    # ---- reading ---------------------------------------------------------
    async def probe(self) -> tuple[Snapshot, Resolution]:
        """Fetch once and report which semantic keys this server exports.

        Usable without `start()`: it opens (and closes) its own client when
        the sampler is not running.
        """
        opened = False
        if self._client is None:
            self._client = httpx.AsyncClient(verify=self.verify, timeout=self.timeout)
            opened = True
        try:
            snap = await self.scrape_once()
        finally:
            if opened and self._owns_client:
                await self._client.aclose()
                self._client = None
        # A probe's resolution is worth keeping: it is the richest view we
        # have, and discarding it meant a later truncated scrape could still
        # be the one that latched the adapter for the whole run.
        self._latch_adapter(snap.samples)
        adapter = self.adapter or detect_adapter(snap.samples, engine=self.engine)
        return snap, adapter.resolution()

    def _sorted_ok(self) -> list[Snapshot]:
        """Successful snapshots in time order. Sorted rather than assumed:
        a replayed log can be concatenated out of order."""
        return sorted((s for s in self.snapshots if s.ok and s.samples),
                      key=lambda s: s.t)

    def at(self, t: float) -> Snapshot:
        """The nearest successful snapshot at or before `t` -- the covariate
        reading to attach to an event that happened at `t`. `t` is in THIS
        SAMPLER'S base (`now()`), not `time.monotonic()`. Falls back to the
        first snapshot when `t` precedes the series."""
        ok = self._sorted_ok()
        if not ok:
            self.raise_if_failed()
            raise ValueError("no successful snapshots yet")
        i = bisect.bisect_right([s.t for s in ok], t) - 1
        return ok[i] if i >= 0 else ok[0]

    def gauges_at(self, t: float) -> dict[str, float | None]:
        """The SEMANTIC gauges as of `t` -- what a caller attaching covariates
        to one request actually wants.

        `at()` hands back a raw `Snapshot`, which is the archive format: a
        list of parsed `/metrics` lines whose names are engine-specific
        (`vllm:num_requests_running`). The semantic names only exist through
        an adapter, so a consumer that read a Snapshot's attributes looking
        for `requests_running` found nothing and recorded an empty covariate
        set -- silently, since a snapshot WAS returned. This is the accessor
        that resolves them, so no consumer has to know the engine's spelling.

        Units: requests for `requests_running` / `requests_waiting`, FRACTION
        in [0, 1] for `kv_cache_usage`. A key the server does not export is
        None, never 0 -- an unexported queue is not an empty one.
        """
        snap = self.at(t)
        ad = self.adapter or detect_adapter(snap.samples, engine=self.engine)
        return {"t": snap.t,
                "requests_running": ad.gauge(snap.samples, "requests_running"),
                "requests_waiting": ad.gauge(snap.samples, "requests_waiting"),
                "kv_cache_usage": ad.gauge(snap.samples, "kv_cache_usage")}

    def window(self, t0: float | None = None, t1: float | None = None) -> WindowDelta:
        """Deltas and gauge stats over [t0, t1], bounded by snapshots that
        ENCLOSE it; raises `WindowNotCovered` if none do.

        Called the instant load finishes, the enclosing `hi` has not been
        scraped yet and this raises. `await s.next_tick()` first.
        """
        self.raise_if_failed()
        return window_from_snapshots(self.snapshots, t0, t1,
                                     adapter=self.adapter, engine=self.engine)

    def live(self) -> dict[str, float | None]:
        """A one-line view of the newest snapshot, for `ws metrics tail`.

        `tok_s` is the delta against the PREVIOUS snapshot, and is the one
        derived number in this module computed per scrape: it is a display
        aid, not a measurement. Anything you intend to publish comes from a
        `WindowDelta`.
        """
        ok = [s for s in self.snapshots if s.ok and s.samples]
        if not ok:
            return {}
        cur = ok[-1]
        ad = self.adapter or detect_adapter(cur.samples, engine=self.engine)
        out: dict[str, float | None] = {
            "t": cur.t,
            # gauges through .gauge(), not .counter(): the combination rule
            # differs (occupancy across engines is not a sum) and another
            # adapter may have to rescale a percent into a fraction
            "running": ad.gauge(cur.samples, "requests_running"),
            "waiting": ad.gauge(cur.samples, "requests_waiting"),
            "kv": ad.gauge(cur.samples, "kv_cache_usage"),
            "tok_s": None, "hit_rate": None,
        }
        if len(ok) >= 2:
            prev, dt = ok[-2], cur.t - ok[-2].t
            gen_a = ad.counter(prev.samples, "generation_tokens_total")
            gen_b = ad.counter(cur.samples, "generation_tokens_total")
            if gen_a is not None and gen_b is not None and dt > 0:
                out["tok_s"] = (gen_b - gen_a) / dt
            qa = ad.counter(prev.samples, "prefix_cache_queries_total")
            qb = ad.counter(cur.samples, "prefix_cache_queries_total")
            ha = ad.counter(prev.samples, "prefix_cache_hits_total")
            hb = ad.counter(cur.samples, "prefix_cache_hits_total")
            if None not in (qa, qb, ha, hb) and (qb - qa) > 0:
                out["hit_rate"] = (hb - ha) / (qb - qa)
        return out

    def __len__(self) -> int:
        return len(self.snapshots)
