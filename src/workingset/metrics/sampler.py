"""`MetricsSampler` — poll a `/metrics` endpoint and keep every raw snapshot.

The rule this module exists to enforce: **raw snapshots are kept, never only
derived values.** vLLM's counters flush MID-request -- a request's whole FLOP
count or prefill time lands in whichever scrape happens to follow the flush --
so a per-scrape RATE is noise and only a delta across a WHOLE window is a
measurement. You cannot know at scrape time which window you will want, so
every scrape is retained (in memory, and optionally appended to JSONL) and the
window arithmetic happens afterwards, offline, as many times as you like.

    async with MetricsSampler(url, interval=1.0) as s:
        ...                                  # drive load
        w = s.window(t0, t1)
    print(w.tokens_out, w.ttft.quantile(0.95))

Window endpoints are the nearest snapshots OUTSIDE [t0, t1], so a counter
delta covers the whole interval and can only OVER-count, never under-count.
Each endpoint is itself fuzzy: the counters are as-of some instant inside the
scrape's round trip. `WindowDelta` carries `t_lo_uncertainty` /
`t_hi_uncertainty` (each endpoint's rtt, in seconds) so a caller dividing by
`dt` knows how sharp the divisor is.

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

from .adapter import KEY_KIND, Resolution
from .parse import Histogram, Sample, parse_text
from .vllm import detect_adapter

__all__ = ["Snapshot", "GaugeStats", "WindowDelta", "MetricsSampler",
           "DECODE_KEEP", "keep_filter", "load_jsonl", "window_from_snapshots"]

# The series decode-side analysis reads -- the same preset
# scripts/scrape_metrics.py has always written, kept name-for-name so an old
# log and a new one answer the same questions. Writing only these shrinks a
# log ~8x, which matters when it has to travel: a full dump is ~100 series
# every interval, ~1 GB/day at 0.5 s. It DROPS the prefix-cache counters and
# the latency histograms, so a log recorded with it can give you decode
# speed but not a measured miss rate or a TTFT quantile.
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
    counter_resets     keys whose counter went BACKWARDS across the window --
                       the engine restarted inside it. Their delta is None,
                       because two endpoints cannot recover how far the
                       counter climbed before the reset. A non-empty tuple
                       invalidates the window for anything those keys feed.
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
    counter_resets: tuple[str, ...] = ()

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
        """Blocks hit / blocks queried over the window, in [0, 1].

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
        """Forward passes in the window, from the iteration-tokens histogram's
        observation count (one observation per engine step)."""
        h = self.histograms.get("iteration_tokens_hist")
        return None if h is None else h.observations

    @property
    def tokens_per_step(self) -> float | None:
        """Mean tokens processed per forward pass."""
        h = self.histograms.get("iteration_tokens_hist")
        return None if h is None else h.mean()

    @property
    def step_time_s(self) -> float | None:
        """Wall seconds per forward pass. Only as sharp as `dt_uncertainty`."""
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
        """Time per output token, seconds. (vLLM's inter-token latency.)"""
        return self.histograms.get("tpot_hist")

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
            "counter_resets": list(self.counter_resets),
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
def window_from_snapshots(snaps: Sequence[Snapshot], t0: float, t1: float,
                          adapter=None) -> WindowDelta:
    """The `WindowDelta` for [t0, t1] over an already-collected series.

    Endpoint choice: `lo` is the LAST snapshot at or before t0 (the nearest
    one outside the window on the low side), falling back to the first
    snapshot when the series starts inside the window; `hi` is the FIRST at
    or after t1, falling back to the last. Failed scrapes cannot be
    endpoints -- they carry no counters -- but they are counted in
    `n_failed` so a caller can see the coverage it actually got.
    """
    ok = [s for s in snaps if s.ok and s.samples]
    if len(ok) < 2:
        raise ValueError(f"need >= 2 successful snapshots to make a window, "
                         f"have {len(ok)}")
    if t1 < t0:
        t0, t1 = t1, t0
    times = [s.t for s in ok]

    i = max(0, bisect.bisect_right(times, t0) - 1)
    j = min(len(ok) - 1, bisect.bisect_left(times, t1))
    if j <= i:                            # degenerate ask: take the next pair
        j = i + 1
        if j >= len(ok):
            i, j = len(ok) - 2, len(ok) - 1
    lo, hi = ok[i], ok[j]

    adapter = adapter or detect_adapter(hi.samples)
    inside = [s for s in snaps if lo.t <= s.t <= hi.t]
    inside_ok = [s for s in inside if s.ok and s.samples]

    # A counter that went BACKWARDS means the engine restarted inside the
    # window. The delta is then unknowable from two endpoints -- you know
    # `b`, but not how far `a` climbed before the reset -- so the key is
    # reported as unmeasurable (None) and named in `counter_resets`, rather
    # than handed back as a negative rate or silently clamped to zero.
    resets: list[str] = []

    counters: dict[str, float | None] = {}
    for key, kind in KEY_KIND.items():
        if kind != "counter":
            continue
        a = adapter.counter(lo.samples, key)
        b = adapter.counter(hi.samples, key)
        if a is None or b is None:
            counters[key] = None
        elif b < a:
            counters[key] = None
            resets.append(key)
        else:
            counters[key] = b - a

    histograms: dict[str, Histogram | None] = {}
    for key, kind in KEY_KIND.items():
        if kind != "histogram":
            continue
        a = adapter.histogram(lo.samples, key)
        b = adapter.histogram(hi.samples, key)
        if a is None or b is None:
            histograms[key] = None
        elif (a.count is not None and b.count is not None and b.count < a.count):
            histograms[key] = None
            resets.append(key)
        else:
            histograms[key] = b - a

    gauges: dict[str, GaugeStats] = {}
    for key, kind in KEY_KIND.items():
        if kind != "gauge":
            continue
        vals = [v for v in (adapter.counter(s.samples, key) for s in inside_ok)
                if v is not None]
        gauges[key] = GaugeStats.of(vals)

    per_pos = None
    a_pos = adapter.by_position(lo.samples, "spec_decode_accepted_per_pos")
    b_pos = adapter.by_position(hi.samples, "spec_decode_accepted_per_pos")
    if a_pos is not None and b_pos is not None:
        d_pos = {p: b_pos.get(p, 0.0) - a_pos.get(p, 0.0) for p in sorted(b_pos)}
        if any(v < 0 for v in d_pos.values()):
            resets.append("spec_decode_accepted_per_pos")
        else:
            per_pos = d_pos

    hint = adapter.resolution().version_hint if hasattr(adapter, "resolution") else "?"
    return WindowDelta(t0=t0, t1=t1, lo=lo, hi=hi, counters=counters,
                       histograms=histograms, gauges=gauges,
                       per_position=per_pos, n_snapshots=len(inside),
                       n_failed=len(inside) - len(inside_ok), version_hint=hint,
                       counter_resets=tuple(resets))


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
        self._stop.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(self._task), timeout=self.timeout + 1)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()
            self._task = None
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None
        if self._out is not None:
            self._out.close()
            self._out = None

    async def wait_first(self, timeout: float = 10.0) -> bool:
        """Block until at least one scrape has landed. Handy before starting
        load: a window needs an endpoint BEFORE t0."""
        try:
            await asyncio.wait_for(self._first.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    # ---- the loop --------------------------------------------------------
    async def _loop(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            await self.scrape_once()
            elapsed = time.monotonic() - started
            delay = max(0.0, self.interval - elapsed)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass

    async def scrape_once(self) -> Snapshot:
        """One scrape, recorded. Never raises: a failure becomes a snapshot
        with `ok=False`, because a gap in the series is evidence too."""
        t_sent = time.time()
        try:
            r = await self._client.get(self.url, headers=self.headers)
            r.raise_for_status()
            rtt = time.time() - t_sent
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
        if self.adapter is None and samples:
            self.adapter = detect_adapter(samples, engine=self.engine)
        return Snapshot(t_sent=t_sent, rtt=rtt, samples=samples, ok=True, lines=lines)

    def _record(self, snap: Snapshot) -> None:
        self.snapshots.append(snap)
        if self.max_snapshots and len(self.snapshots) > self.max_snapshots:
            del self.snapshots[: len(self.snapshots) - self.max_snapshots]
        if self._out is not None:
            self._out.write(snap.to_json() + "\n")
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
        adapter = self.adapter or detect_adapter(snap.samples, engine=self.engine)
        return snap, adapter.resolution()

    def at(self, t: float) -> Snapshot:
        """The nearest successful snapshot at or before `t` -- the covariate
        reading to attach to an event that happened at `t`. Falls back to the
        first snapshot when `t` precedes the series."""
        ok = [s for s in self.snapshots if s.ok and s.samples]
        if not ok:
            raise ValueError("no successful snapshots yet")
        i = bisect.bisect_right([s.t for s in ok], t) - 1
        return ok[i] if i >= 0 else ok[0]

    def window(self, t0: float, t1: float) -> WindowDelta:
        """Counter/histogram deltas and gauge stats over [t0, t1], bounded by
        the nearest snapshots OUTSIDE it. See `window_from_snapshots`."""
        return window_from_snapshots(self.snapshots, t0, t1, adapter=self.adapter)

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
            "running": ad.counter(cur.samples, "requests_running"),
            "waiting": ad.counter(cur.samples, "requests_waiting"),
            "kv": ad.counter(cur.samples, "kv_cache_usage"),
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
