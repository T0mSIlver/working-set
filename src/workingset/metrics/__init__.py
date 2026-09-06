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
        w = s.window(t0, t1)
    print(w.output_tok_s, w.miss_rate, w.ttft.quantile(0.95))

This module deliberately knows nothing about `workingset.probe` or
`workingset.hypotheses`: it measures, they decide.
"""
from __future__ import annotations

from .adapter import (KEY_KIND, KEY_UNITS, SEMANTIC_KEYS, MetricsAdapter,
                      Resolution)
from .parse import (Histogram, MetricFamily, Sample, group_histograms,
                    parse_families, parse_text, parse_types)
from .sampler import (DECODE_KEEP, GaugeStats, MetricsSampler, Snapshot,
                      WindowDelta, keep_filter, load_jsonl,
                      window_from_snapshots)
from .vllm import ALIASES, VLLMAdapter, adapter_for, detect_adapter

__all__ = [
    # parse
    "Sample", "Histogram", "MetricFamily",
    "parse_text", "parse_types", "parse_families", "group_histograms",
    # adapter seam
    "MetricsAdapter", "Resolution", "SEMANTIC_KEYS", "KEY_KIND", "KEY_UNITS",
    # vllm
    "VLLMAdapter", "ALIASES", "detect_adapter", "adapter_for",
    # sampling
    "MetricsSampler", "Snapshot", "WindowDelta", "GaugeStats",
    "window_from_snapshots", "load_jsonl", "keep_filter", "DECODE_KEEP",
]
