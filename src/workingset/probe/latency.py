"""The two latency limits, read off a probe's traces.

Both are pure functions of `RequestTrace`s, so a rung's statistics can be
tested on synthetic traces without an endpoint.

  slowed_stats   how much of the time streams spend in steps that carry a full
                 prefill chunk, and how fast they run there. H-itl-spike
                 measures the size of ONE spike; this measures how much of the
                 time the streams spend slowed.
  cold_wait_stats  a cold request's wait before its own prefill starts:
                 client TTFT minus the idle prefill time of that prompt.

Gaps are per SSE event and MTP delivers several tokens per event, so a slow
gap's tokens are the stream's mean tokens per event (completion_tokens over
events), never 1.
"""
from __future__ import annotations

import math

from .. import model as M
from .stats import pct

# a session's first request is a cold prefill exactly as a forced miss is
COLD_KINDS = ("miss", "first")


def default_slow_threshold_ms(cfg) -> float:
    """Half the predicted mixed-step time (decode step + one full-chunk
    prefill pass): it separates steps that carry a full chunk from normal
    steps, which sit at the decode step."""
    st = M.slowed_steps(cfg.to_model(), cfg.to_topology(), cfg.to_workload(),
                        cfg.deployment.max_num_batched_tokens,
                        cfg.slo.itl_floor_tok_s, cfg.calibration.mfu)
    return 0.5 * st["mixed_s"] * 1e3


def slow_threshold_ms(cfg, opts) -> tuple[float, str]:
    """(threshold, where it came from): `--freeze-threshold-ms` overrides the
    model's default."""
    given = getattr(opts, "slow_threshold_ms", None)
    if given is not None:
        return float(given), "--freeze-threshold-ms"
    return default_slow_threshold_ms(cfg), "half the predicted mixed step"


def _run_seconds(gaps_ms: list, threshold_ms: float) -> float:
    """Longest stretch of consecutive slow gaps in one stream, in seconds."""
    best = cur = 0.0
    for g in gaps_ms:
        cur = cur + g if g >= threshold_ms else 0.0
        best = max(best, cur)
    return best / 1e3


def slowed_stats(traces: list, threshold_ms: float,
                 percentile: float = 95.0) -> dict:
    """Slowed share, slowed speed and the longest slowed run over `traces`.

    share  sum of slow gaps / sum of stream spans (first to last event)
    speed  tokens delivered in slow gaps / time in slow gaps
    run    per stream, the longest run of consecutive slow gaps, in seconds;
           reported as the maximum and the p-th over streams that had one

    Empty dict when no stream carried gaps.
    """
    streams = [t for t in traces
               if not t.error and t.gaps_ms and t.span_s and t.span_s > 0]
    if not streams:
        return {}
    slow_s = slow_tok = span = 0.0
    runs = []
    for t in streams:
        per_event = (t.ctok / t.n_chunks) if t.ctok and t.n_chunks else 1.0
        slow = [g for g in t.gaps_ms if g >= threshold_ms]
        slow_s += sum(slow) / 1e3
        slow_tok += per_event * len(slow)
        span += t.span_s
        run = _run_seconds(t.gaps_ms, threshold_ms)
        if run > 0:
            runs.append(run)
    out = {"threshold_ms": threshold_ms, "n_streams": len(streams),
           "n_slow_streams": len(runs), "share": slow_s / span,
           "speed_tok_s": slow_tok / slow_s if slow_s > 0 else None,
           "longest_run_s": max(runs) if runs else None,
           "run_pX_s": pct(runs, percentile) if runs else None}
    return out


def model_idle_prefill(cfg):
    """`prompt tokens -> idle prefill seconds`, from the model at the
    configured MFU."""
    m, t = cfg.to_model(), cfg.to_topology()
    chunk, mfu = cfg.deployment.max_num_batched_tokens, cfg.calibration.mfu
    return lambda tokens: M.miss_context_seconds(m, t, tokens, chunk,
                                                 mfu_anchor=mfu)


def cold_waits(traces: list, idle_prefill) -> list[float]:
    """Wait of each cold request: client TTFT minus the idle prefill time of
    its prompt. Floored at 0: a request that started at once and prefilled
    faster than the idle estimate waited nothing."""
    out = []
    for t in traces:
        if t.kind not in COLD_KINDS or t.error or t.ttft is None:
            continue
        ptok = t.ptok_achieved or t.ptok_intended
        if not ptok:
            continue
        idle = idle_prefill(ptok)
        if not math.isfinite(idle):
            continue
        out.append(max(0.0, t.ttft - idle))
    return out


def cold_wait_stats(traces: list, idle_prefill, percentile: float,
                    mean: bool = False, source: str = "model") -> dict:
    """The p-th cold-request wait (the mean under `mean`), with which idle
    prefill time it was measured against. Empty dict with no cold request."""
    waits = cold_waits(traces, idle_prefill)
    if not waits:
        return {}
    return {"n": len(waits), "idle_source": source,
            "mean_s": sum(waits) / len(waits),
            "pX_s": pct(waits, percentile),
            "value_s": (sum(waits) / len(waits)) if mean
            else pct(waits, percentile)}


def queue_time(server: dict | None) -> dict:
    """vLLM's own queue time over the window: the delta of
    `request_queue_time_seconds`. vLLM counts it until the request is first
    SCHEDULED, which is not the same as until its prefill starts when
    prefills share a step, so it is reported next to the client wait and
    never forced to agree with it."""
    h = ((server or {}).get("histograms") or {}).get("queue_time_hist")
    if not h or not h.get("count"):
        return {}
    return {"n": h["count"], "mean_s": h.get("mean"), "p50_s": h.get("p50"),
            "p95_s": h.get("p95")}


def fit_idle_prefill(fit):
    """`prompt tokens -> idle prefill seconds` read off a shared run's TTFT
    fit at an idle server (running = waiting = 0), or None when the fit was
    refused. Floored at 0: a fitted intercept can dip below it."""
    if fit is None or not getattr(fit, "usable", False):
        return None

    def at(tokens):
        k = tokens / 1e3
        v = fit.predict({"L_ktok": k, "L_ktok2": k * k, "running": 0.0,
                         "waiting": 0.0})
        return max(0.0, v) if math.isfinite(v) else float("nan")
    return at
