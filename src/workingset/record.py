"""One JSON document per run — everything needed to re-derive the report.

A record carries the config it was run from, the predictions that config
produced, the mode (exclusive or shared), the raw probe output in compact
form, every hypothesis's prediction / measurement / verdict, and the
"what this run does not establish" list. `ws report run.json` re-prints the
report from it; nothing in the report is computed anywhere else.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1


@dataclass
class RunRecord:
    workingset: str = ""
    schema_version: int = SCHEMA_VERSION
    created: str = ""
    mode: str = "shared"                  # "exclusive" | "shared"
    interrupted: bool = False
    config: dict = field(default_factory=dict)
    predictions: dict = field(default_factory=dict)
    options: dict = field(default_factory=dict)
    endpoint: dict = field(default_factory=dict)
    plan: dict = field(default_factory=dict)
    rungs: list = field(default_factory=list)
    sample: dict | None = None
    burst: dict | None = None
    hypotheses: list = field(default_factory=list)
    skipped: list = field(default_factory=list)
    not_established: list = field(default_factory=list)
    measured_capacity_bracket: list = field(default_factory=lambda: [None, None])

    @classmethod
    def new(cls, version: str, **kw) -> "RunRecord":
        return cls(workingset=version,
                   created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   **kw)

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RunRecord":
        d = dict(d)
        v = d.pop("schema_version", SCHEMA_VERSION)
        if v != SCHEMA_VERSION:
            raise ValueError(f"run record schema_version {v} not supported "
                             f"(this workingset reads {SCHEMA_VERSION})")
        known = {f for f in cls.__dataclass_fields__}
        unknown = sorted(set(d) - known)
        if unknown:
            raise ValueError(f"unknown run-record keys: {unknown}")
        return cls(schema_version=v, **d)

    def dumps(self) -> str:
        # nan/inf are not JSON: they round-trip through Python's `NaN` literal
        # but not through any other reader, so they are written as null. Every
        # statistic that can be absent is nan-valued somewhere, so this is the
        # common case, not an edge one.
        return json.dumps(_clean(self.to_dict()), indent=2,
                          default=_json_default) + "\n"

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.write_text(self.dumps(), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "RunRecord":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _clean(o):
    import math
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set, frozenset)):
        return [_clean(v) for v in o]
    return o


def _json_default(o):
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    return str(o)


# ============================================================================
# "what this run does not establish"
# ----------------------------------------------------------------------------
# Ported from the trailer in scripts/validate_deployment.py's print_report.
# It is the part of the output users are most likely to skip and most need,
# so it is built from the run, not hard-coded, and it is stored in the record
# rather than regenerated at print time.
# ============================================================================
def not_established_notes(cfg, opts, plan, ran_ladder: bool,
                          ran_burst: bool, exclusive: bool,
                          metrics: bool) -> list[str]:
    from .probe.session import sub_prefix_floor

    notes: list[str] = []
    if ran_ladder:
        notes.append(
            "Warm capacity is bounded BELOW only: unless a rung shows >5% "
            "effective-cold hit turns, the pool was never driven to eviction, "
            "and the eviction classifier itself is the baseline's 0.4x-cold "
            "TTFT heuristic, not a pool measurement.")
        notes.append(
            "Ceilings the ladder did not fail on are 'not separable': one run "
            "observes one binding constraint, not four.")
    else:
        notes.append(
            "No ladder was run: every capacity ceiling (cache, decode, "
            "latency, saturation, and the binding one) is untested here. This "
            "run establishes levels and gap distributions at the endpoint's "
            "prevailing load, nothing about how many users it serves.")
    if not exclusive:
        notes.append(
            "Shared mode: the endpoint's standing load is unknown and not "
            "controlled. Every level below was measured at whatever the "
            "server was already doing, not at the predicted operating point "
            "the model priced these numbers at.")
    notes.append(
        "Token counts are chars/{:g} approximations; the achieved/intended "
        "ratio above says how far off — re-run calibrated before trusting "
        "capacity numbers to better than ~15%.".format(opts.chars_per_token))
    notes.append(
        "Think time is exponential around a fixed mean; real agentic cadence "
        "is burstier (lognormal sigma 2.43 in the measured trace), which "
        "moves the latency ceiling down, not up.")
    notes.append(
        "One seed, one window per rung: no variance estimate. Steady state is "
        "assumed after the ramp, not verified.")
    notes.append(
        "DP deployments: this drives ONE endpoint; replica-splitting and "
        "session-sticky routing are not exercised.")
    if not metrics:
        notes.append(
            "No /metrics sampler: the concurrent-decode count, the queue "
            "depth and the KV-cache occupancy are inferred from client-side "
            "timing or not at all. Pass --metrics-url for the server's own "
            "view.")
    wl = cfg.workload
    sub_floor = sub_prefix_floor(wl)
    if sub_floor >= wl.subagent_median_tokens:
        notes.append(
            "Subagent leg is DEGENERATE under this config: the prefix floor "
            f"({sub_floor:,} tok) >= the subagent median "
            f"({wl.subagent_median_tokens:,} tok), so most subagent first "
            "turns are byte-identical and vLLM dedups them to one cache "
            "entry — the subagent KV load is not being exercised.")
    if not ran_burst:
        notes.append(
            "Correlated-flush tolerance (B*) needs the separate burst probe "
            "(--burst N --exclusive); independent per-turn misses cannot "
            "see it.")
    for h, reason in plan.skipped:
        notes.append(f"{h.key} was not tested: {reason}.")
    return notes
