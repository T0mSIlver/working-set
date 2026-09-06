"""`workingset.metrics` — sample a serving engine's `/metrics`, delta it honestly.

Four layers, each usable on its own:

    parse    Prometheus text format -> Sample / Histogram        (no deps)
    adapter  the engine seam: MetricsAdapter protocol, semantic keys, units
    vllm     the vLLM adapter and `detect_adapter`
    sampler  MetricsSampler / Snapshot / WindowDelta

The invariant the whole package exists to protect: **derive nothing at
scrape time.** vLLM's counters flush mid-request, so a per-scrape rate is
noise; only a delta across a whole window is a measurement. Every raw
snapshot is retained (memory, and optionally JSONL) and the arithmetic runs
afterwards, offline, as often as you like.

    from workingset.metrics import MetricsSampler

    async with MetricsSampler(url, interval=1.0) as s:
        await s.wait_first()
        t0 = time.time(); await drive_load(); t1 = time.time()
        await s.next_tick()          # window endpoints must ENCLOSE [t0, t1]
        w = s.window(t0, t1)
    print(w.output_tok_s, w.miss_rate, w.ttft.quantile(0.95))

A window whose endpoints do not enclose the interval raises
`WindowNotCovered` instead of returning numbers about some nearby stretch,
and any key whose counter went backwards mid-window lands in `w.invalid`
with a reason rather than carrying a fabricated delta.

This module deliberately knows nothing about `workingset.probe` or
`workingset.hypotheses`: it measures, they decide.
"""
from __future__ import annotations

from .adapter import (KEY_AGG, KEY_KIND, KEY_UNITS, SEMANTIC_KEYS,
                      MetricsAdapter, Resolution)
from .parse import (Histogram, HistogramMismatch, HistogramReset, MetricFamily,
                    Sample, group_histograms, parse_families, parse_text,
                    parse_types)
from .sampler import (DECODE_KEEP, GaugeStats, MetricsSampler, SamplerStopped,
                      Snapshot, WindowDelta, WindowNotCovered, keep_filter,
                      load_jsonl, window_from_snapshots)
from .vllm import ALIASES, VLLMAdapter, adapter_for, detect_adapter

__all__ = [
    # parse
    "Sample", "Histogram", "MetricFamily",
    "parse_text", "parse_types", "parse_families", "group_histograms",
    "HistogramMismatch", "HistogramReset",
    # adapter seam
    "MetricsAdapter", "Resolution", "SEMANTIC_KEYS", "KEY_KIND", "KEY_UNITS",
    "KEY_AGG",
    # vllm
    "VLLMAdapter", "ALIASES", "detect_adapter", "adapter_for",
    # sampling
    "MetricsSampler", "Snapshot", "WindowDelta", "GaugeStats",
    "window_from_snapshots", "load_jsonl", "keep_filter", "DECODE_KEEP",
    "WindowNotCovered", "SamplerStopped",
]
