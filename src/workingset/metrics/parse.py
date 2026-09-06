"""Prometheus text-format parser — no dependency, no exporter client.

A `/metrics` dump is a flat list of `name{labels} value` lines. This module
turns one into `Sample` objects and groups the histogram families
(`_bucket` / `_sum` / `_count`) into `Histogram`, because every latency
question the study asks -- "what was p95 TTFT during THIS window" -- is a
question about a histogram DELTA, and a delta is only definable bucket-wise.

Why not `prometheus_client.parser`: it is a runtime dependency for a
20-line grammar, it discards the `# TYPE` lines we use to tell a histogram
from a counter that merely ends in `_count`, and it raises on the malformed
lines a half-written scrape produces. We keep everything and skip garbage.

Units are whatever the exporter says. vLLM's latency histograms are in
SECONDS (the `_milliseconds_` families are the exception and say so in the
name); token counters are tokens; `kv_cache_usage_perc` is a FRACTION in
[0, 1] despite the name.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Iterable

__all__ = ["Sample", "Histogram", "MetricFamily", "parse_text", "group_histograms"]

# name{label="value",...} 1.234 [timestamp]
# The value is taken as "the rest of the line" and float()-ed, so NaN, +Inf,
# -Inf and exponent forms all fall out of Python's own float parser rather
# than a regex that has to enumerate them.
_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{.*\})?[ \t]+(.+)$")
_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')
_TYPE = re.compile(r"^#\s*TYPE\s+([a-zA-Z_:][a-zA-Z0-9_:]*)\s+(\w+)")
_HELP = re.compile(r"^#\s*HELP\s+([a-zA-Z_:][a-zA-Z0-9_:]*)\s*(.*)$")

_UNESCAPE = {"\\": "\\", "n": "\n", '"': '"', "t": "\t"}


def _unescape(s: str) -> str:
    """Prometheus label-value escapes: \\\\, \\n, \\" (we also accept \\t)."""
    if "\\" not in s:
        return s
    out, i, n = [], 0, len(s)
    while i < n:
        c = s[i]
        if c == "\\" and i + 1 < n:
            nxt = s[i + 1]
            out.append(_UNESCAPE.get(nxt, "\\" + nxt))
            i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _labels(blob: str | None) -> dict[str, str]:
    if not blob:
        return {}
    return {k: _unescape(v) for k, v in _LABEL.findall(blob)}


def _to_float(tok: str) -> float | None:
    """Prometheus values: a float, NaN, +Inf, -Inf. Returns None if unparseable."""
    tok = tok.strip()
    if not tok:
        return None
    # a trailing millisecond timestamp is legal; the value is the first field
    head = tok.split()[0]
    try:
        return float(head)
    except ValueError:
        pass
    low = head.lower().lstrip("+")
    if low == "nan":
        return math.nan
    if low in ("inf", "infinity"):
        return math.inf
    if low in ("-inf", "-infinity"):
        return -math.inf
    return None


@dataclass(frozen=True)
class Sample:
    """One `name{labels} value` line. `value` is float; NaN and +/-Inf survive."""

    name: str
    labels: dict[str, str]
    value: float

    def label_key(self, drop: Iterable[str] = ()) -> tuple[tuple[str, str], ...]:
        """Hashable label identity, optionally dropping keys (e.g. `le`)."""
        drop = set(drop)
        return tuple(sorted((k, v) for k, v in self.labels.items() if k not in drop))


@dataclass
class Histogram:
    """A cumulative Prometheus histogram, or the DELTA of two of them.

    `buckets` maps upper bound (`le`, inclusive) -> cumulative count at or
    below that bound; `+Inf` is always present after `from_samples`. `count`
    and `sum` come from the `_count` / `_sum` series when they exist, else
    from the `+Inf` bucket (count) and are None (sum).

    A delta is bucket-wise: subtract each bucket, each of `count` and `sum`.
    That stays a valid cumulative histogram as long as the counters did not
    reset, so `WindowDelta` clamps negatives to zero rather than reporting a
    negative quantile.

    Units are the exporter's. vLLM latency families are in seconds.
    """

    name: str
    labels: dict[str, str] = field(default_factory=dict)
    buckets: dict[float, float] = field(default_factory=dict)
    count: float | None = None
    sum: float | None = None

    # ---- construction --------------------------------------------------
    @classmethod
    def from_samples(cls, name: str, samples: Iterable[Sample],
                     labels: dict[str, str] | None = None) -> "Histogram":
        """Build from the `_bucket` / `_sum` / `_count` members of one family."""
        h = cls(name=name, labels=dict(labels or {}))
        for s in samples:
            if s.name.endswith("_bucket"):
                le = _to_float(s.labels.get("le", ""))
                if le is not None:
                    h.buckets[le] = s.value
            elif s.name.endswith("_sum"):
                h.sum = s.value
            elif s.name.endswith("_count"):
                h.count = s.value
        if math.inf not in h.buckets and h.buckets:
            h.buckets[math.inf] = h.count if h.count is not None else max(h.buckets.values())
        if h.count is None and math.inf in h.buckets:
            h.count = h.buckets[math.inf]
        return h

    # ---- arithmetic ----------------------------------------------------
    def __sub__(self, other: "Histogram") -> "Histogram":
        """Bucket-wise delta. Negative results (a counter reset, or a bound
        that only one side exported) clamp to 0."""
        bounds = set(self.buckets) | set(other.buckets)
        buckets = {b: max(0.0, self.buckets.get(b, 0.0) - other.buckets.get(b, 0.0))
                   for b in bounds}
        count = None
        if self.count is not None and other.count is not None:
            count = max(0.0, self.count - other.count)
        total = None
        if self.sum is not None and other.sum is not None:
            total = max(0.0, self.sum - other.sum)
        return Histogram(self.name, dict(self.labels), buckets, count, total)

    def __add__(self, other: "Histogram") -> "Histogram":
        bounds = set(self.buckets) | set(other.buckets)
        buckets = {b: self.buckets.get(b, 0.0) + other.buckets.get(b, 0.0)
                   for b in bounds}
        count = None
        if self.count is not None and other.count is not None:
            count = self.count + other.count
        total = None
        if self.sum is not None and other.sum is not None:
            total = self.sum + other.sum
        return Histogram(self.name, dict(self.labels), buckets, count, total)

    # ---- reading -------------------------------------------------------
    @property
    def bounds(self) -> list[float]:
        return sorted(self.buckets)

    @property
    def observations(self) -> float:
        """How many observations this histogram (or delta) holds."""
        if self.count is not None:
            return self.count
        return self.buckets.get(math.inf, 0.0)

    def mean(self) -> float | None:
        """sum / count, in the exporter's units. None when either is missing
        or no observation fell in the window."""
        if self.sum is None or not self.observations:
            return None
        return self.sum / self.observations

    def quantile(self, q: float) -> float | None:
        """Linearly-interpolated quantile, the way Prometheus `histogram_quantile`
        does it: find the bucket holding rank q*count, interpolate uniformly
        inside it between its lower and upper bound.

        Returns None for an empty histogram. If the target rank falls in the
        `+Inf` bucket the answer is unbounded above; we return the largest
        FINITE bound, which is the tightest thing the data supports, so a
        caller reading p99 off an overflowing histogram gets a lower bound
        rather than `inf`. Bucket resolution is the accuracy limit: this is
        never better than the exporter's bucket layout.
        """
        if not 0.0 <= q <= 1.0:
            raise ValueError(f"quantile q must be in [0, 1], got {q}")
        total = self.observations
        if not total or not self.buckets:
            return None
        bounds = self.bounds
        target = q * total
        lower = 0.0          # lower bound of the bucket under inspection
        prev_c = 0.0         # cumulative count below that bucket
        for b in bounds:
            c = self.buckets[b]
            if c >= target:
                if math.isinf(b):
                    finite = [x for x in bounds if math.isfinite(x)]
                    return max(finite) if finite else None
                if c == prev_c:          # empty bucket: the rank sits at its edge
                    return b
                frac = (target - prev_c) / (c - prev_c)
                return lower + frac * (b - lower)
            lower, prev_c = b, c
        finite = [x for x in bounds if math.isfinite(x)]
        return max(finite) if finite else None

    def to_dict(self) -> dict:
        return {"name": self.name, "labels": dict(self.labels),
                "count": self.count, "sum": self.sum,
                "buckets": {("+Inf" if math.isinf(b) else b): v
                            for b, v in sorted(self.buckets.items())}}


@dataclass
class MetricFamily:
    """Every sample sharing a base metric name, plus its declared `# TYPE`."""

    name: str
    type: str = "untyped"
    help: str = ""
    samples: list[Sample] = field(default_factory=list)


def parse_text(text: str) -> list[Sample]:
    """Every parseable sample line, in file order. Comments and malformed
    lines are skipped silently -- a truncated scrape is common and should
    cost you the bad line, not the dump."""
    out: list[Sample] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        v = _to_float(m.group(3))
        if v is None:
            continue
        out.append(Sample(m.group(1), _labels(m.group(2)), v))
    return out


def parse_types(text: str) -> dict[str, str]:
    """`# TYPE` declarations: base metric name -> counter|gauge|histogram|summary."""
    types: dict[str, str] = {}
    for raw in text.splitlines():
        m = _TYPE.match(raw.strip())
        if m:
            types[m.group(1)] = m.group(2).lower()
    return types


def parse_help(text: str) -> dict[str, str]:
    """`# HELP` text: base metric name -> description."""
    out: dict[str, str] = {}
    for raw in text.splitlines():
        m = _HELP.match(raw.strip())
        if m:
            out[m.group(1)] = _unescape(m.group(2).strip())
    return out


def parse_families(text: str) -> dict[str, MetricFamily]:
    """Samples grouped by base name, with `# TYPE` / `# HELP` attached.

    A histogram's members (`x_bucket`, `x_sum`, `x_count`) are filed under
    the base name `x`, so `families["vllm:time_to_first_token_seconds"]`
    holds all three.
    """
    types, helps = parse_types(text), parse_help(text)
    fams: dict[str, MetricFamily] = {}
    for s in parse_text(text):
        base = _base_name(s.name, types)
        f = fams.get(base)
        if f is None:
            f = fams[base] = MetricFamily(base, types.get(base, "untyped"),
                                          helps.get(base, ""))
        f.samples.append(s)
    return fams


def _base_name(name: str, types: dict[str, str]) -> str:
    """Strip a histogram/summary member suffix, but ONLY when the resulting
    base was declared a histogram or summary.

    This matters: `vllm:iteration_tokens_total_count` is a histogram member
    whose base is `vllm:iteration_tokens_total`, while a plain counter named
    `..._count` is its own family. Without the `# TYPE` check the two are
    indistinguishable.
    """
    for suffix in ("_bucket", "_sum", "_count"):
        if name.endswith(suffix):
            base = name[: -len(suffix)]
            if types.get(base) in ("histogram", "summary"):
                return base
            # No TYPE line (a `--keep`-filtered scrape drops comments): fall
            # back to the shape, a `_bucket` member is unambiguous.
            if suffix == "_bucket" and base not in types:
                return base
    return name


def group_histograms(samples: Iterable[Sample],
                     types: dict[str, str] | None = None) -> dict[str, list[Histogram]]:
    """Histogram families found among `samples`, split by label set.

    Returns base name -> one `Histogram` per distinct label set (with `le`
    dropped). vLLM labels every series with `model_name` and, under V1, an
    `engine` index, so a data-parallel deployment yields one histogram per
    engine and the caller decides whether summing them is meaningful.
    """
    types = types or {}
    by_family: dict[str, dict[tuple, list[Sample]]] = {}
    for s in samples:
        base = _base_name(s.name, types)
        if base == s.name:
            continue                      # not a histogram member
        key = s.label_key(drop=("le",))
        by_family.setdefault(base, {}).setdefault(key, []).append(s)
    out: dict[str, list[Histogram]] = {}
    for base, per_labels in by_family.items():
        out[base] = [Histogram.from_samples(base, group, dict(key))
                     for key, group in per_labels.items()]
    return out
