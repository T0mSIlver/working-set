"""The vLLM adapter: `vllm:*` names -> the semantic keys in `adapter.py`.

vLLM renamed a lot on the way to the V1 engine, and a deployment in the wild
is whatever the cluster happens to run. So every semantic key is
a CANDIDATE LIST resolved once per dump against what the server actually
exports, newest spelling first, and `Resolution` reports what was found --
`ws metrics probe` prints that table, which is how you learn before a run
whether this server can answer the question you are about to ask it.

Renames this adapter absorbs (see `ALIASES` for the literal strings). The
current-name column was read out of vLLM's `vllm/v1/metrics/loggers.py`,
`v1/spec_decode/metrics.py` and `v1/metrics/perf.py` at v0.25.1 and at a
post-0.28 main; the Prometheus surface is byte-identical between those two,
so every rename below completed BEFORE v0.25.1 and the older spellings are
here only for deployments still running them.

  gpu_cache_usage_perc            -> kv_cache_usage_perc
  gpu_prefix_cache_hit_rate       -> prefix_cache_queries + prefix_cache_hits
                                     (a decaying gauge became two counters,
                                     which is what makes a WINDOW hit rate
                                     computable at all)
  time_per_output_token_seconds   -> inter_token_latency_seconds
  model_forward_time_milliseconds -> gone; V1 exports no per-pass timing
                                     histogram, so `forward_time_hist`
                                     resolves only on a V0-era server and
                                     `step_time_s` falls back to wall clock

`vllm:iteration_tokens_total` is a HISTOGRAM whose declared name happens to
end in `_total`; its `_count` member is the forward-pass count, which is why
callers read it through `iteration_tokens_hist` and take `.observations`.

Labels: every V1 series carries `model_name` and `engine` (the engine index,
as a string). Per-engine series are summed unless `engine` selects one.

Units: token counters are tokens; latency histograms are SECONDS except the
`_milliseconds_` families; `kv_cache_usage_perc` is a FRACTION in [0, 1]
despite the name (vLLM's own docstring: "1 means 100 percent usage"); the
prefix-cache counters are in TOKENS queried / TOKENS hit, not blocks.
"""
from __future__ import annotations

from typing import Iterable

from .adapter import (KEY_KIND, SEMANTIC_KEYS, Resolution, _is_engine,
                      combine_gauge, group_by_position, histogram_bases,
                      pick_histogram, resolve_aliases, series_histograms,
                      series_samples, sum_samples)
from .parse import Histogram, Sample, group_histograms

__all__ = ["VLLMAdapter", "ALIASES", "detect_adapter", "adapter_for"]

# Ordered NEWEST FIRST. The first candidate present in the dump wins.
ALIASES: dict[str, tuple[str, ...]] = {
    # ---- token counters (stable since v0.2) ----------------------------
    "prompt_tokens_total": ("vllm:prompt_tokens_total",),
    "generation_tokens_total": ("vllm:generation_tokens_total",),
    # ---- scheduler gauges ----------------------------------------------
    "requests_running": ("vllm:num_requests_running",),
    "requests_waiting": ("vllm:num_requests_waiting",),
    # V1 renamed the KV gauge; the V0 CPU-side twin is deliberately ignored.
    "kv_cache_usage": ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"),
    # ---- prefix cache ---------------------------------------------------
    # V1 exports two counters (tokens queried / tokens hit) so a WINDOW hit
    # rate is computable. V0 exported only a decaying `..._hit_rate` gauge,
    # from which no window rate can be recovered at all; there is nothing to
    # alias it to, so on a V0 server these keys are simply missing and
    # `probe` says so.
    "prefix_cache_queries_total": ("vllm:prefix_cache_queries_total",),
    "prefix_cache_hits_total": ("vllm:prefix_cache_hits_total",),
    # ---- pressure -------------------------------------------------------
    "preemptions_total": ("vllm:num_preemptions_total",),
    "request_success_total": ("vllm:request_success_total",),
    # ---- latency histograms (seconds) -----------------------------------
    "ttft_hist": ("vllm:time_to_first_token_seconds",),
    # ITL: one observation per output EVENT, not per token. stats.py appends
    # a single interval per (request, engine step) that emitted anything, so
    # a spec-decode step accepting 3 tokens records ONE observation covering
    # all three -- this histogram reads seconds-per-step, high. The first
    # output of a request goes to TTFT instead, so
    #   count = (output events) - (requests that produced output).
    "tpot_hist": ("vllm:inter_token_latency_seconds",
                  "vllm:time_per_output_token_seconds"),
    # The per-REQUEST mean, one observation per finished request, computed
    # as decode_time / (generation_tokens - 1) -- a TRUE per-token average
    # that stays correct under spec decode. A different distribution, so it
    # gets its own key rather than silently standing in for `tpot_hist`.
    "request_tpot_hist": ("vllm:request_time_per_output_token_seconds",),
    "e2e_hist": ("vllm:e2e_request_latency_seconds",),
    "prefill_time_hist": ("vllm:request_prefill_time_seconds",),
    "decode_time_hist": ("vllm:request_decode_time_seconds",),
    "queue_time_hist": ("vllm:request_queue_time_seconds",),
    "inference_time_hist": ("vllm:request_inference_time_seconds",),
    # ---- per-request size histograms ------------------------------------
    "request_prompt_tokens_hist": ("vllm:request_prompt_tokens",),
    "request_generation_tokens_hist": ("vllm:request_generation_tokens",),
    # ---- engine-step histograms -----------------------------------------
    # Declared name ends in `_total` but the family IS a histogram; `_count`
    # is the forward-pass count.
    "iteration_tokens_hist": ("vllm:iteration_tokens_total",),
    # V0 only. V1 exports no per-pass timing histogram.
    "forward_time_hist": ("vllm:model_forward_time_milliseconds",),
    # ---- speculative decoding -------------------------------------------
    "spec_decode_num_drafts_total": ("vllm:spec_decode_num_drafts_total",),
    "spec_decode_num_draft_tokens_total": ("vllm:spec_decode_num_draft_tokens_total",),
    "spec_decode_num_accepted_tokens_total": ("vllm:spec_decode_num_accepted_tokens_total",),
    "spec_decode_accepted_per_pos": ("vllm:spec_decode_num_accepted_tokens_per_pos_total",
                                     "vllm:spec_decode_num_accepted_tokens_per_pos"),
    # ---- engine-side work accounting (PerfMetricsProm) -------------------
    # Only incremented when the scheduler produced perf stats, so a server
    # can export the series and never advance it.
    "estimated_flops_total": ("vllm:estimated_flops_per_gpu_total",),
    "estimated_read_bytes_total": ("vllm:estimated_read_bytes_per_gpu_total",),
    "estimated_write_bytes_total": ("vllm:estimated_write_bytes_per_gpu_total",),
    # tokens served straight from the prefix cache, no prefill compute
    "prompt_tokens_cached_total": ("vllm:prompt_tokens_cached_total",),
}

# Signals that place a dump on the V0 / V1 side of the rename. Used only for
# the human-readable `version_hint`; resolution never depends on it.
_V1_MARKERS = ("vllm:kv_cache_usage_perc", "vllm:prefix_cache_queries_total")
_V0_MARKERS = ("vllm:gpu_cache_usage_perc", "vllm:gpu_prefix_cache_hit_rate",
               "vllm:avg_generation_throughput_toks_per_s")


class VLLMAdapter:
    """Resolves the `vllm:*` vocabulary once, then answers semantic keys.

    Construct from ONE dump's samples (`VLLMAdapter(samples)`); the
    resolution is reused for every later snapshot of the same server, which
    is what makes a window's endpoints comparable even if a scrape came back
    partial.

    `engine` selects one engine of a data-parallel deployment (matched on the
    `engine` / `engine_index` label). Left None, per-engine series are SUMMED,
    which is right for token counts and wrong for anything divided by wall
    time -- engines step on independent clocks.
    """

    engine_name = "vllm"

    def __init__(self, samples: Iterable[Sample] | None = None,
                 engine: str | None = None):
        self.engine = engine
        samples = list(samples or [])
        names = {s.name for s in samples}
        bases = histogram_bases(samples)
        self._resolved = resolve_aliases(names, ALIASES, bases)
        self._version_hint = self._hint(names)

    # ---- detection -------------------------------------------------------
    @staticmethod
    def matches(samples: Iterable[Sample]) -> bool:
        """True when any series carries the `vllm:` prefix."""
        return any(s.name.startswith("vllm:") for s in samples)

    @staticmethod
    def _hint(names: set[str]) -> str:
        v1 = sum(m in names for m in _V1_MARKERS)
        v0 = sum(m in names for m in _V0_MARKERS)
        if v1 and not v0:
            return "v1"
        if v0 and not v1:
            return "v0"
        if v1 and v0:
            return "v1+v0-compat"       # hidden metrics re-enabled
        return "unknown"

    # ---- introspection ---------------------------------------------------
    def resolution(self) -> Resolution:
        missing = tuple(k for k in SEMANTIC_KEYS if k not in self._resolved)
        return Resolution(engine=self.engine_name, version_hint=self._version_hint,
                          resolved=dict(self._resolved), missing=missing)

    def name_for(self, key: str) -> str | None:
        """The exported metric name this server uses for a semantic key."""
        return self._resolved.get(key)

    def has(self, key: str) -> bool:
        return key in self._resolved

    # ---- reading ---------------------------------------------------------
    def counter(self, samples: list[Sample], key: str) -> float | None:
        """Counter value, summed over label sets. Units: `KEY_UNITS[key]`.
        None when the server does not export it or the scrape lacked it."""
        name = self._resolved.get(key)
        if name is None:
            return None
        return sum_samples(samples, name, self.engine)

    def gauge(self, samples: list[Sample], key: str) -> float | None:
        """Gauge value, combined per `KEY_AGG[key]`. Units: `KEY_UNITS[key]`.

        vLLM's gauges need no rescaling (`kv_cache_usage_perc` is already a
        fraction), so this differs from `counter` only in how it COMBINES:
        request counts add across engines, occupancy cannot, and an
        unweightable fraction spanning several engines returns None.
        """
        return combine_gauge(self.series(samples, key), key,
                             engine_selected=self.engine is not None)

    def series(self, samples: list[Sample], key: str) -> dict | None:
        """Per-label-set values for a counter or gauge, before aggregation."""
        name = self._resolved.get(key)
        if name is None:
            return None
        return series_samples(samples, name, self.engine)

    def histogram(self, samples: list[Sample], key: str) -> Histogram | None:
        """Cumulative histogram for a `*_hist` key, label sets summed."""
        name = self._resolved.get(key)
        if name is None:
            return None
        return pick_histogram(self._families(samples), name, self.engine)

    def histogram_series(self, samples: list[Sample], key: str) -> dict | None:
        """Per-label-set histograms for a `*_hist` key, before aggregation."""
        name = self._resolved.get(key)
        if name is None:
            return None
        return series_histograms(self._families(samples), name, self.engine)

    @staticmethod
    def _families(samples: list[Sample]) -> dict[str, list[Histogram]]:
        types = {b: "histogram" for b in histogram_bases(samples)}
        return group_histograms(samples, types)

    # ---- bulk reads ------------------------------------------------------
    # Reset detection walks every retained snapshot, so the per-key accessors
    # above would re-group the same dump once per key -- 11 histogram keys x
    # 600 snapshots of a 10-minute run was 3.1 s of pure regrouping. These do
    # one pass and hand back every key at once.
    def all_series(self, samples: list[Sample]) -> dict[str, dict]:
        """Per-label-set values for every counter/gauge key, in one pass."""
        wanted = {name: key for key, name in self._resolved.items()
                  if KEY_KIND.get(key) in ("counter", "gauge")}
        out: dict[str, dict] = {}
        for s in samples:
            key = wanted.get(s.name)
            if key is None:
                continue
            if self.engine is not None and not _is_engine(s.labels, self.engine):
                continue
            bucket = out.setdefault(key, {})
            lk = s.label_key()
            bucket[lk] = bucket.get(lk, 0.0) + s.value
        return out

    def all_histogram_series(self, samples: list[Sample]) -> dict[str, dict]:
        """Per-label-set histograms for every histogram key, in one pass."""
        fams = self._families(samples)
        out: dict[str, dict] = {}
        for key, kind in KEY_KIND.items():
            if kind != "histogram":
                continue
            name = self._resolved.get(key)
            if name is None:
                continue
            g = series_histograms(fams, name, self.engine)
            if g:
                out[key] = g
        return out

    def by_position(self, samples: list[Sample], key: str) -> dict[int, float] | None:
        """`position`-labelled counter as {draft position: tokens accepted}.

        This is the one place a label IS the measurement: position 0 is the
        first drafted token, and the ratio between consecutive positions
        tests the geometric 1 + a + a^2 acceptance model.
        """
        name = self._resolved.get(key)
        if name is None:
            return None
        return group_by_position(samples, name, self.engine)

    def read(self, samples: list[Sample], key: str):
        """Dispatch on `KEY_KIND[key]` -- float, Histogram, dict or None."""
        kind = KEY_KIND.get(key)
        if kind == "histogram":
            return self.histogram(samples, key)
        if kind == "counter_by_position":
            return self.by_position(samples, key)
        if kind == "gauge":
            return self.gauge(samples, key)
        return self.counter(samples, key)


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
# An SGLang adapter is added by appending its class here; no caller changes.
ADAPTERS: list[type] = [VLLMAdapter]


def detect_adapter(samples: Iterable[Sample], engine: str | None = None):
    """The first registered adapter that recognises this dump.

    Falls back to `VLLMAdapter` on an unrecognised dump so the caller still
    gets an object with everything resolved to None (and a `probe` table
    that says so) instead of an exception.
    """
    samples = list(samples)
    for cls in ADAPTERS:
        if cls.matches(samples):
            return cls(samples, engine=engine)
    return VLLMAdapter(samples, engine=engine)


def adapter_for(text: str, engine: str | None = None):
    """`detect_adapter` straight from a `/metrics` body."""
    from .parse import parse_text
    return detect_adapter(parse_text(text), engine=engine)
