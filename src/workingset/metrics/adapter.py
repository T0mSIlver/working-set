"""The engine-adapter seam: exported metric names -> stable semantic keys.

Everything above this module (the sampler, the window arithmetic, the
hypotheses layer) speaks only in SEMANTIC KEYS -- `generation_tokens_total`,
`ttft_hist` -- and never in `vllm:`-prefixed strings. That is the whole
point: vLLM renamed half of these between 0.6, 0.10 and V1, and an SGLang
adapter has to be addable without a caller changing.

A key that this server does not export resolves to None. Reading a missing
key is never a KeyError; it is a None you can report, which is what `ws
metrics probe` prints so a user knows what a run against this server can
actually measure.

Units are carried in `KEY_UNITS` and repeated in every docstring that
returns one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol, runtime_checkable

from .parse import Histogram, Sample

__all__ = ["SEMANTIC_KEYS", "KEY_UNITS", "KEY_KIND", "MetricsAdapter",
           "Resolution", "resolve_aliases", "sum_samples", "pick_histogram",
           "histogram_bases", "group_by_position"]

# ---------------------------------------------------------------------------
# the vocabulary. Order is presentation order for `ws metrics probe`.
# ---------------------------------------------------------------------------
KEY_KIND: dict[str, str] = {
    "prompt_tokens_total": "counter",
    "prompt_tokens_cached_total": "counter",
    "generation_tokens_total": "counter",
    "requests_running": "gauge",
    "requests_waiting": "gauge",
    "kv_cache_usage": "gauge",
    "prefix_cache_queries_total": "counter",
    "prefix_cache_hits_total": "counter",
    "preemptions_total": "counter",
    "request_success_total": "counter",
    "ttft_hist": "histogram",
    "tpot_hist": "histogram",
    "request_tpot_hist": "histogram",
    "e2e_hist": "histogram",
    "prefill_time_hist": "histogram",
    "decode_time_hist": "histogram",
    "queue_time_hist": "histogram",
    "inference_time_hist": "histogram",
    "request_prompt_tokens_hist": "histogram",
    "request_generation_tokens_hist": "histogram",
    "iteration_tokens_hist": "histogram",
    "forward_time_hist": "histogram",
    "spec_decode_num_drafts_total": "counter",
    "spec_decode_num_draft_tokens_total": "counter",
    "spec_decode_num_accepted_tokens_total": "counter",
    "spec_decode_accepted_per_pos": "counter_by_position",
    "estimated_flops_total": "counter",
    "estimated_read_bytes_total": "counter",
    "estimated_write_bytes_total": "counter",
}

SEMANTIC_KEYS: tuple[str, ...] = tuple(KEY_KIND)

KEY_UNITS: dict[str, str] = {
    "prompt_tokens_total": "tokens",
    "prompt_tokens_cached_total": "tokens served from cache",
    "generation_tokens_total": "tokens",
    "requests_running": "requests",
    "requests_waiting": "requests",
    "kv_cache_usage": "fraction of the KV pool in [0, 1]",
    "prefix_cache_queries_total": "tokens queried",
    "prefix_cache_hits_total": "tokens hit",
    "preemptions_total": "preemption events",
    "request_success_total": "requests",
    "ttft_hist": "seconds",
    "tpot_hist": "seconds between output tokens",
    "request_tpot_hist": "seconds per output token, per-request mean",
    "e2e_hist": "seconds",
    "prefill_time_hist": "seconds",
    "decode_time_hist": "seconds",
    "queue_time_hist": "seconds",
    "inference_time_hist": "seconds",
    "request_prompt_tokens_hist": "tokens per request",
    "request_generation_tokens_hist": "tokens per request",
    "iteration_tokens_hist": "tokens per engine step",
    "forward_time_hist": "milliseconds",
    "spec_decode_num_drafts_total": "draft events",
    "spec_decode_num_draft_tokens_total": "tokens proposed",
    "spec_decode_num_accepted_tokens_total": "tokens accepted",
    "spec_decode_accepted_per_pos": "tokens accepted, by draft position",
    "estimated_flops_total": "FLOP per GPU",
    "estimated_read_bytes_total": "bytes read per GPU",
    "estimated_write_bytes_total": "bytes written per GPU",
}


@dataclass(frozen=True)
class Resolution:
    """What an adapter made of one dump: which keys it found and under which
    exported name, and which it could not find at all."""

    engine: str
    version_hint: str
    resolved: dict[str, str]              # semantic key -> exported metric name
    missing: tuple[str, ...]              # semantic keys this server lacks

    @property
    def found(self) -> tuple[str, ...]:
        return tuple(k for k in SEMANTIC_KEYS if k in self.resolved)

    def to_dict(self) -> dict:
        return {"engine": self.engine, "version_hint": self.version_hint,
                "resolved": dict(self.resolved), "missing": list(self.missing)}


@runtime_checkable
class MetricsAdapter(Protocol):
    """The seam an engine backend implements. Callers use only this.

    An implementation is constructed from one dump's samples (so it can
    resolve names against what the server actually exports) and then answers
    `counter` / `gauge` / `histogram` for the semantic keys in
    `SEMANTIC_KEYS`. Every accessor returns None for an unexported key.

    `engine_name` is the BACKEND ("vllm", "sglang"), not to be confused with
    an implementation's per-instance engine INDEX selector for DP>1.
    """

    engine_name: str

    @staticmethod
    def matches(samples: Iterable[Sample]) -> bool:
        """True when this dump looks like it came from this engine."""

    def resolution(self) -> Resolution:
        """Which semantic keys this server exports, and under which names."""

    def counter(self, samples: list[Sample], key: str) -> float | None:
        """The counter's value, summed over label sets. Units: `KEY_UNITS[key]`."""

    def gauge(self, samples: list[Sample], key: str) -> float | None:
        """The gauge's value, summed over label sets. Units: `KEY_UNITS[key]`."""

    def histogram(self, samples: list[Sample], key: str) -> Histogram | None:
        """The cumulative histogram, summed over label sets."""

    def by_position(self, samples: list[Sample], key: str) -> dict[int, float] | None:
        """A counter whose `position` label IS the measurement (spec-decode
        per-position acceptance): position index -> value."""


# ---------------------------------------------------------------------------
# helpers shared by concrete adapters
# ---------------------------------------------------------------------------
def resolve_aliases(names: set[str], aliases: dict[str, tuple[str, ...]],
                    hist_bases: set[str]) -> dict[str, str]:
    """First alias present in `names` wins, per semantic key.

    Alias tuples are ordered NEWEST FIRST, so a server that exports both a
    current name and a deprecated one (vLLM keeps some behind
    `--show-hidden-metrics-for-version`) resolves to the current one.
    Histogram keys match against `hist_bases` -- the base names recovered by
    stripping `_bucket`/`_sum`/`_count` -- not against the raw sample names.
    """
    out: dict[str, str] = {}
    for key, cands in aliases.items():
        pool = hist_bases if KEY_KIND.get(key) == "histogram" else names
        for c in cands:
            if c in pool:
                out[key] = c
                break
    return out


def sum_samples(samples: Iterable[Sample], name: str,
                engine: str | None = None) -> float | None:
    """Sum one metric over its label sets. None when the series is absent.

    Summing is right for a single engine and WRONG across data-parallel
    engines, which step on independent clocks -- pass `engine` to select one
    (matched against the `engine` or `engine_index` label).

    A selector EXCLUDES a series carrying no engine label at all. Folding an
    unlabelled series into engine 0's total is exactly the cross-engine
    mixing the selector exists to prevent, and vLLM labels every per-engine
    series, so an unlabelled one is not engine 0's to claim.
    """
    total, seen = 0.0, False
    for s in samples:
        if s.name != name:
            continue
        if engine is not None and not _is_engine(s.labels, engine):
            continue
        total += s.value
        seen = True
    return total if seen else None


def _is_engine(labels: dict[str, str], engine: str) -> bool:
    eid = labels.get("engine", labels.get("engine_index"))
    return eid is not None and eid == str(engine)


def pick_histogram(hists: dict[str, list[Histogram]], base: str,
                   engine: str | None = None) -> Histogram | None:
    """Collapse a histogram family's label sets into one histogram.

    Same caveat as `sum_samples`: bucket-wise addition across engines is
    meaningful for a latency distribution (the observations are per-request
    either way) but the caller may still want one engine.
    """
    group = hists.get(base)
    if not group:
        return None
    chosen = [h for h in group
              if engine is None or _is_engine(h.labels, engine)]
    if not chosen:
        return None
    acc = chosen[0]
    for h in chosen[1:]:
        acc = acc + h
    return Histogram(base, {}, dict(acc.buckets), acc.count, acc.sum)


def histogram_bases(samples: Iterable[Sample]) -> set[str]:
    """Base names of every histogram family present, recovered from members.

    Works without `# TYPE` lines, which a `--keep`-filtered scrape drops:
    a `_bucket` member is unambiguous, and `_sum`/`_count` are only credited
    to a base whose `_bucket` we also saw.
    """
    bases = {s.name[: -len("_bucket")] for s in samples if s.name.endswith("_bucket")}
    return bases


def group_by_position(samples: Iterable[Sample], name: str,
                      engine: str | None = None) -> dict[int, float] | None:
    """A `position`-labelled counter as {position: value}. None if absent.

    Same engine selection as `sum_samples`: values are summed across engines
    unless `engine` picks one.
    """
    out: dict[int, float] = {}
    for s in samples:
        if s.name != name:
            continue
        if engine is not None and not _is_engine(s.labels, engine):
            continue
        pos = s.labels.get("position")
        if pos is None:
            continue
        try:
            p = int(pos)
        except ValueError:
            continue
        out[p] = out.get(p, 0.0) + s.value
    return out or None
