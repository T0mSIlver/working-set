"""The hypotheses `ws test` can put to a live endpoint.

Every H-* the explorer's generated harness carried, one class each, plus the
machinery that decides which of them a given run can honestly perform.

  Registry.select(keys)   resolve keys (or --all) to hypothesis objects
  plan(...)               gate them on what the run actually offers
                          (--exclusive, --burst N, a /metrics sampler) and
                          say which probes the survivors need

A requirement a run does not meet is a SKIP with a reason, never a silent
downgrade: a hypothesis that needs to generate its own population must not
quietly report a number measured under someone else's load.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .base import (BOUNDED_BELOW, BURST, BURST_PROBE, EXCLUSIVE, GLYPH,
                   LADDER, METRICS, NOT_ESTABLISHED, REFUTED, REQUIREMENTS,
                   SAMPLE, STATUSES, SUPPORTED, Hypothesis, Measurement,
                   Prediction, Verdict, bracket_verdict, ratio_verdict)
from .burst import HBurst
from .ceilings import HBinding, HCache, HDecode, HLatency, HSaturation
from .context import LadderView, RunContext
from .gaps import HItlMean, HItlSpike, HSteady, HTtftMiss


class Registry:
    """Ordered list of hypotheses, addressable by key."""

    def __init__(self, items):
        self._items = list(items)
        self._by_key = {h.key: h for h in self._items}
        if len(self._by_key) != len(self._items):
            raise ValueError("duplicate hypothesis key in registry")

    def __iter__(self):
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    @property
    def keys(self) -> list[str]:
        return [h.key for h in self._items]

    def get(self, key: str) -> Hypothesis:
        try:
            return self._by_key[key]
        except KeyError:
            raise KeyError(f"unknown hypothesis {key!r}; known: "
                           f"{', '.join(self.keys)}") from None

    def all(self) -> list[Hypothesis]:
        return list(self._items)

    def select(self, keys) -> list[Hypothesis]:
        """Resolve keys to hypotheses, de-duplicated and in registry order.

        Empty (or None) selects everything. Keys are matched exactly; a
        typo raises rather than quietly testing less than the user asked for.
        """
        if not keys:
            return self.all()
        wanted = {self.get(k).key for k in keys}
        return [h for h in self._items if h.key in wanted]


REGISTRY = Registry([
    HCache(), HDecode(), HLatency(), HSaturation(), HBinding(),
    HTtftMiss(), HBurst(), HSteady(), HItlSpike(), HItlMean(),
])


@dataclass
class Plan:
    """What a run will actually do.

    `probes` is the complete set the run will execute — it is what --dry-run
    prints AND what RunContext is handed, so the two cannot disagree. A
    hypothesis asks the context which probes exist; it never decides for
    itself at measure time.
    """
    selected: list = field(default_factory=list)
    skipped: list = field(default_factory=list)   # (hypothesis, reason)
    probes: frozenset = frozenset()               # {"ladder", "sample", "burst"}

    @property
    def run_ladder(self) -> bool:
        return "ladder" in self.probes

    def to_dict(self) -> dict:
        return {"selected": [h.describe(self.probes) for h in self.selected],
                "skipped": [{**h.describe(), "reason": r}
                            for h, r in self.skipped],
                "probes": sorted(self.probes)}


SKIP_REASON = {
    EXCLUSIVE: "needs --exclusive: it generates the population it measures, "
               "so it cannot share an endpoint",
    METRICS: "needs a metrics sampler: pass --metrics-url",
    BURST: "needs the burst probe: pass --burst N",
}


def plan(hypotheses, exclusive: bool = False, metrics: bool = False,
         burst: int = 0) -> Plan:
    """Gate `hypotheses` on the PERMISSIONS this run offers, then resolve the
    PROBES the survivors need.

    A skip names EVERY unmet permission, in a fixed order so the reason is
    stable across runs. Probes resolve in two passes: the mandatory ones
    first, then the conditional ones against that set — so a cheap hypothesis
    reads a ladder that some ceiling is running anyway, and opens a sample of
    its own only when no ladder exists.
    """
    have = {EXCLUSIVE: bool(exclusive), METRICS: bool(metrics),
            BURST: bool(burst)}
    p = Plan()
    for h in hypotheses:
        unmet = [r for r in REQUIREMENTS if r in h.requires and not have[r]]
        if unmet:
            p.skipped.append((h, "; ".join(SKIP_REASON[r] for r in unmet)))
            continue
        p.selected.append(h)
    mandatory = frozenset().union(*(h.probes for h in p.selected)) \
        if p.selected else frozenset()
    conditional = frozenset().union(
        *(h.conditional_probes(mandatory) for h in p.selected)) \
        if p.selected else frozenset()
    p.probes = mandatory | conditional
    return p


__all__ = [
    "BOUNDED_BELOW", "BURST", "BURST_PROBE", "EXCLUSIVE", "GLYPH", "LADDER",
    "METRICS", "SAMPLE",
    "NOT_ESTABLISHED", "REFUTED", "REGISTRY", "REQUIREMENTS", "STATUSES",
    "SUPPORTED", "Hypothesis", "LadderView", "Measurement", "Plan",
    "Prediction", "Registry", "RunContext", "Verdict", "bracket_verdict",
    "plan", "ratio_verdict",
]
