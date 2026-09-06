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
    # shared-endpoint mode: the safety rails and what they spent, the
    # covariate fits with their coefficients / n / residuals / condition
    # number, the operating point they were evaluated at, the natural ladder
    # and the server-side cross-check. Present only for a shared run.
    shared: dict | None = None
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
def not_established_notes(cfg, opts, plan, rungs=None, sample=None,
                          burst=None, hypotheses=None, exclusive: bool = False,
                          metrics: bool = False, interrupted: bool = False,
                          shared: dict | None = None) -> list[str]:
    """Built from what the run MEASURED, not from what it planned.

    The first cut of this list read the plan's booleans: a sample whose every
    request failed still produced "this run establishes levels and gap
    distributions", because a sample had been planned. A trailer that
    overstates a failed run is worse than no trailer, so every line below is
    conditioned on a result.
    """
    from .probe.session import sub_prefix_floor

    rungs = rungs or []
    hypotheses = hypotheses or []
    notes: list[str] = []
    full = [r for r in rungs if not r.get("partial")]
    ok_rungs = [r for r in full if (r.get("n_turns") or 0) > 0]
    sample_ok = bool(sample) and (sample.get("n_ok") or 0) > 0

    if interrupted:
        notes.append(
            "INTERRUPTED: the run did not finish. Any rung shown as partial "
            "is orientation only, and every hypothesis below was scored "
            "against the evidence that existed when the run stopped.")

    # --- what the ladder did or did not establish ------------------------
    if ok_rungs:
        notes.append(
            "Ceilings the ladder did not fail on are 'not separable': one run "
            "observes one binding constraint, not four.")
        if _verdict(hypotheses, "H-cache") == "bounded_below":
            notes.append(
                "Warm capacity is bounded BELOW only: the pool was never "
                "driven to eviction, and the eviction classifier itself is "
                "the baseline's 0.4x-cold TTFT heuristic, not a pool "
                "measurement.")
    elif full:
        notes.append(
            f"The ladder ran {len(full)} rung(s) and none produced a measured "
            "turn: no capacity ceiling is established here, and the rung "
            "table below is a record of failures, not of load.")
    else:
        notes.append(
            "No ladder was run: every capacity ceiling (cache, decode, "
            "latency, saturation, and the binding one) is untested here.")

    # --- what the cheap probe did or did not establish -------------------
    if sample is not None and not sample_ok:
        notes.append(
            f"The sample probe answered {sample.get('n_ok', 0)} of "
            f"{sample.get('n', 0)} requests: the level and gap rows below "
            "rest on nothing. Fix the endpoint before reading them.")
    if not exclusive and (sample_ok or sample is None):
        notes.extend(_shared_notes(shared))

    # --- measurements that were unavailable ------------------------------
    if not _token_accounting(rungs, sample):
        notes.append(
            "No `usage` readback from this endpoint: prompt- and "
            "completion-token counts are the client's chars/{:g} estimate "
            "only, so the achieved/intended ratio, freezes per 1k tokens and "
            "tokens per SSE event are all unavailable or unverified."
            .format(opts.chars_per_token))
    else:
        notes.append(
            "Token counts are chars/{:g} approximations; the achieved/intended "
            "ratio above says how far off — re-run calibrated before trusting "
            "capacity numbers to better than ~15%.".format(opts.chars_per_token))
    if not metrics:
        notes.append(
            "No /metrics sampler: the concurrent-decode count, the queue "
            "depth and the KV-cache occupancy are inferred from client-side "
            "timing or not at all. Pass --metrics-url for the server's own "
            "view.")

    # --- hypotheses that reached no conclusion ---------------------------
    open_rows = [h for h in hypotheses
                 if h.get("verdict", {}).get("status") == "not_established"]
    if open_rows:
        notes.append(
            "Not established by this run, each for the reason on its row: "
            + ", ".join(h["key"] for h in open_rows) + ".")

    # --- standing caveats of the workload itself -------------------------
    notes.append(
        "Think time is exponential around a fixed mean; real agentic cadence "
        "is burstier (lognormal sigma 2.43 in the measured trace), which "
        "moves the latency ceiling down, not up.")
    if ok_rungs:
        notes.append(
            "One seed, one window per rung: no variance estimate. Steady "
            "state is assumed after the ramp, not verified.")
    notes.append(
        "DP deployments: this drives ONE endpoint; replica-splitting and "
        "session-sticky routing are not exercised.")
    wl = cfg.workload
    sub_floor = sub_prefix_floor(wl)
    if sub_floor >= wl.subagent_median_tokens:
        notes.append(
            "Subagent leg is DEGENERATE under this config: the prefix floor "
            f"({sub_floor:,} tok) >= the subagent median "
            f"({wl.subagent_median_tokens:,} tok), so most subagent first "
            "turns are byte-identical and vLLM dedups them to one cache "
            "entry — the subagent KV load is not being exercised.")
    if not burst:
        notes.append(
            "Correlated-flush tolerance (B*) needs the separate burst probe "
            "(--burst N --exclusive); independent per-turn misses cannot "
            "see it.")
    for h, reason in plan.skipped:
        notes.append(f"{h.key} was not tested: {reason}.")
    return notes


CAPPED = ("Shared mode: the endpoint's standing load is unknown and not "
          "controlled, so every level here was measured at whatever the "
          "server was already doing rather than at the operating point the "
          "predictions were made for. Those rows are capped at 'not "
          "established' whatever the numbers say; only an exclusive run, or a "
          "shared run whose covariate fit clears its gates, can support or "
          "refute them.")


def _shared_notes(shared: dict | None) -> list[str]:
    """What the shared run's covariate fit did and did not buy.

    The old unconditional cap stays the note for a run that fitted nothing —
    which is every run without a `/metrics` sampler. A run that DID fit says
    which readings cleared the gate and, for the rest, which gate they failed;
    a reader must never have to infer that from the absence of a line.
    """
    if not shared:
        return [CAPPED]
    out: list[str] = []
    if shared.get("aborted"):
        out.append(f"ABORTED by a probe safety rail: {shared['aborted']}. "
                   "Everything below rests on what had been measured when the "
                   "rail tripped, which is less than the plan asked for.")
    readings = shared.get("readings") or {}
    ok = sorted(k for k, r in readings.items() if r.get("available"))
    bad = {k: r.get("reason") for k, r in readings.items()
           if not r.get("available")}
    if ok:
        out.append(
            "Shared mode, CORRECTED: the prevailing load was measured "
            "alongside every probe request and regressed out, so "
            + ", ".join(ok) + " are read AT the configured operating point "
            "rather than at whatever the server was doing. Each such row "
            "prints its extrapolation distance — the number that says how far "
            "the correction reached beyond the load actually observed. A "
            "small distance is an interpolation; a large one would have been "
            "refused.")
    else:
        out.append(CAPPED)
    for k, why in sorted(bad.items()):
        out.append(f"The {k} reading was not corrected to the operating "
                   f"point: {why}.")
    return out


def _verdict(hypotheses, key: str) -> str | None:
    for h in hypotheses:
        if h.get("key") == key:
            return h.get("verdict", {}).get("status")
    return None


def _token_accounting(rungs, sample) -> bool:
    """Did the endpoint return `usage` anywhere? request.py's fallback drops
    stream_options on a rejection, and without it there is no token count to
    check the synthetic-text calibration against."""
    for r in rungs:
        for t in r.get("traces") or []:
            if t.get("ptok_achieved"):
                return True
    for t in (sample or {}).get("traces") or []:
        if t.get("ptok_achieved"):
            return True
    return False
