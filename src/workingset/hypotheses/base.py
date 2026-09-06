"""What a hypothesis is: a prediction, a measurement, and a verdict.

One class per hypothesis. `predict` quotes `workingset.predict` /
`workingset.model` and does no modelling of its own; `measure` drives probes
through a `RunContext` (which caches them, so the ladder runs once however
many hypotheses read it); `verdict` is a pure function of the two, so every
verdict in the report can be re-derived from the run record.

STATUSES
  supported        the prediction lands inside what was measured
  refuted          the measurement excludes the prediction
  bounded_below    the run only established a lower bound consistent with (or
                   above) the prediction — it never drove the system to the
                   failure the prediction is about. Warm capacity is the
                   canonical case: unless a rung reaches eviction, the pool
                   was never tested from above.
  not_established  the run does not separate this ceiling from the others, or
                   the measurement's own validity guards did not clear

DEVIATION from scripts/validate_deployment.py: the harness has three glyphs
(✓ ~ ✗) and prints "lower bound only" in the note column. `bounded_below` is
that note promoted to a status, so a machine reading the record cannot mistake
a lower bound for a confirmation. The note TEXT is unchanged.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

SUPPORTED = "supported"
REFUTED = "refuted"
BOUNDED_BELOW = "bounded_below"
NOT_ESTABLISHED = "not_established"
STATUSES = (SUPPORTED, REFUTED, BOUNDED_BELOW, NOT_ESTABLISHED)

# requirement tokens
EXCLUSIVE = "exclusive"   # needs to generate the population itself
METRICS = "metrics"       # needs a /metrics sampler
BURST = "burst"           # needs the burst probe (--burst N)
REQUIREMENTS = (EXCLUSIVE, METRICS, BURST)

GLYPH = {SUPPORTED: "✓", REFUTED: "✗",
         BOUNDED_BELOW: "≥", NOT_ESTABLISHED: "~"}


@dataclass(frozen=True)
class Prediction:
    value: Any = None                 # float, or a str for a categorical one
    lo: float | None = None           # bracket, when the model quotes one
    hi: float | None = None
    unit: str = ""
    text: str | None = None           # rendered "predicted" column

    def render(self) -> str:
        if self.text is not None:
            return self.text
        if self.value is None:
            return "-"
        if isinstance(self.value, str):
            return self.value
        v = f"{self.value:g}{self.unit}"
        if self.lo is not None and self.hi is not None:
            v += f" [{self.lo:g}-{self.hi:g}]"
        return v

    def to_dict(self) -> dict:
        return {"value": _safe(self.value), "lo": _safe(self.lo),
                "hi": _safe(self.hi), "unit": self.unit, "text": self.render()}


@dataclass(frozen=True)
class Measurement:
    value: Any = None
    lo: float | None = None           # measured bracket (largest pass, smallest fail]
    hi: float | None = None
    unit: str = ""
    text: str = "-"                   # rendered "measured" column
    data: dict = field(default_factory=dict)   # supporting numbers, recorded

    def to_dict(self) -> dict:
        return {"value": _safe(self.value), "lo": _safe(self.lo),
                "hi": _safe(self.hi), "unit": self.unit, "text": self.text,
                "data": {k: _safe(v) for k, v in self.data.items()}}


@dataclass(frozen=True)
class Verdict:
    status: str
    text: str

    def __post_init__(self):
        if self.status not in STATUSES:
            raise ValueError(f"unknown verdict status {self.status!r}; "
                             f"known: {STATUSES}")

    @property
    def glyph(self) -> str:
        return GLYPH[self.status]

    def to_dict(self) -> dict:
        return {"status": self.status, "text": self.text, "glyph": self.glyph}


def _safe(x):
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


LADDER, SAMPLE, BURST_PROBE = "ladder", "sample", "burst"


class Hypothesis:
    """Base class. Subclasses set `key`, `title`, `requires` and implement the
    three methods.

    `requires` is a set of PERMISSIONS (what the operator allowed this run to
    do); `probes` and `conditional_probes` are the WORK that follows from
    them. The two were one field, and it produced plans that lied: H-burst
    carries the `exclusive` permission and had a ladder planned for it that it
    never ran, while H-itl-spike planned a `sample` and then fired a burst.
    `--dry-run` prints the probe set, so a plan that disagrees with the run is
    a plan that under-reports what is about to hit the endpoint.
    """
    key: str = ""
    title: str = ""
    requires: frozenset = frozenset()      # permissions
    probes: frozenset = frozenset()        # probes it always needs

    def conditional_probes(self, planned: frozenset) -> frozenset:
        """Probes that depend on what the rest of the run is already doing —
        resolved once, against the mandatory set, never re-decided at
        measure time."""
        return frozenset()

    def statement(self, cfg, predictions) -> str:
        """The quotable sentence, with this config's numbers in it. Overridden
        by every hypothesis; the fallback keeps a custom one printable."""
        return f"{self.key}: {self.title}"

    def predict(self, cfg, predictions) -> Prediction:
        raise NotImplementedError

    async def measure(self, ctx) -> Measurement:
        raise NotImplementedError

    def verdict(self, prediction: Prediction,
                measurement: Measurement) -> Verdict:
        raise NotImplementedError

    # ---- convenience ---------------------------------------------------
    def describe(self, probes: frozenset | None = None) -> dict:
        d = {"key": self.key, "title": self.title,
             "requires": sorted(self.requires)}
        if probes is not None:
            d["probes"] = sorted(self.probes
                                 | self.conditional_probes(probes))
        return d

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.key}>"


# ============================================================================
# the harness's bracket comparison, verbatim in text, refined in status
# ============================================================================
def bracket_verdict(pred: float | None, lo: float | None,
                    hi: float | None) -> Verdict:
    """Compare a predicted ceiling against the measured (largest-pass,
    smallest-fail] population bracket. lo/hi None = side not observed.
    "within 25%" of the nearest observed edge is not_established.

    Ported from `bracket_verdict` in scripts/validate_deployment.py. The only
    change is the status of the "never failed" branch: the harness prints ~
    there, this returns bounded_below, because a run that never failed has
    established a lower bound and nothing else.
    """
    if pred is None:
        return Verdict(NOT_ESTABLISHED, "no prediction")
    if lo is None and hi is None:
        return Verdict(NOT_ESTABLISHED, "no data")
    if hi is None:                       # never failed: lower bound only
        if pred >= lo:
            return Verdict(BOUNDED_BELOW,
                           f"not reached (measured >= {lo:g}, predicted {pred:g})")
        if lo <= 1.25 * pred:
            return Verdict(BOUNDED_BELOW,
                           f"measured >= {lo:g}, within 25% above prediction")
        return Verdict(REFUTED, f"passed at {lo:g} > 1.25x predicted {pred:g} "
                                "(model conservative)")
    if lo is None:                       # every rung failed: upper bound only
        if pred <= hi:
            return Verdict(NOT_ESTABLISHED,
                           f"capacity < {hi:g}; prediction {pred:g} not excluded")
        if pred <= hi / 0.75:
            return Verdict(NOT_ESTABLISHED,
                           f"capacity < {hi:g}, within 25% below prediction")
        return Verdict(REFUTED, f"failed at {hi:g} < 0.75x predicted {pred:g} "
                                "(model optimistic)")
    if lo <= pred <= hi:
        return Verdict(SUPPORTED, f"predicted {pred:g} inside measured ({lo:g}, {hi:g}]")
    edge = lo if pred < lo else hi
    if 0.75 <= pred / edge <= 1.25:
        return Verdict(NOT_ESTABLISHED,
                       f"predicted {pred:g} within 25% of measured ({lo:g}, {hi:g}]")
    return Verdict(REFUTED, f"predicted {pred:g} outside measured ({lo:g}, {hi:g}]")


def ratio_verdict(measured: float | None, predicted: float | None,
                  what: str, tight: tuple = (0.7, 1.3),
                  loose: tuple = (0.5, 2.0)) -> Verdict:
    """The harness's level comparison (miss mean TTFT): supported inside the
    tight band, not_established inside the loose one, refuted outside — the
    model quotes order-of-magnitude bounds, so the bands are wide on purpose."""
    if predicted is None or measured is None or not math.isfinite(measured) \
            or not math.isfinite(predicted) or predicted == 0:
        return Verdict(NOT_ESTABLISHED, f"no {what} samples")
    r = measured / predicted
    if tight[0] <= r <= tight[1]:
        return Verdict(SUPPORTED, f"{r:.2f}x predicted")
    if loose[0] <= r <= loose[1]:
        return Verdict(NOT_ESTABLISHED, f"{r:.2f}x predicted (inside the "
                                        "model's order-of-magnitude bound)")
    return Verdict(REFUTED, f"{r:.2f}x predicted")
