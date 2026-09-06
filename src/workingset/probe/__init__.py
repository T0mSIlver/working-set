"""Measurement primitives that know nothing about hypotheses.

Each probe drives a live OpenAI-compatible endpoint and returns a result
object; the derived statistics are the ones the retired standalone harness
defines, and the scoring functions (`eval_rung`, `eval_sample`, `eval_burst`)
are pure so they can be exercised on synthetic traces.

  request     one streamed completion -> RequestTrace
  session     byte-stable prefixes + per-session context generation
  population  the closed-loop user loop -> Rung (a ladder rung)
  burst       the simultaneous-miss flush probe -> BurstResult
  ladder      the geometric load ladder bracketing a predicted limit

Every probe takes an optional `metrics` object, duck-typed on
`at(t) -> snapshot-like` and `window(t0, t1) -> delta-like`. When present,
each RequestTrace carries `covariates` (the snapshot at send time) and each
Rung/Sample/BurstResult carries the window delta as an opaque `server` field.
Nothing here imports the sampler.
"""
from .burst import BurstResult, eval_burst, run_burst
from .ladder import build_ladder
from .options import ProbeOptions
from .population import (Rung, Sample, decode_batch, eval_rung, eval_sample,
                         per_user_p50, run_population, run_sample,
                         spike_evidence)
from .request import EndpointSpec, RequestTrace, make_client, send_request
from .session import (Prefixes, Session, build_prefixes, draw_session_tokens,
                      make_session, make_text, sampler_selfcheck,
                      sub_prefix_floor)
from .stats import DEFAULT_RUNGS, FREEZE_LADDER_MS, fmt, pct

__all__ = [
    "BurstResult", "DEFAULT_RUNGS", "EndpointSpec", "FREEZE_LADDER_MS",
    "Prefixes", "ProbeOptions", "RequestTrace", "Rung", "Sample", "Session",
    "build_ladder", "build_prefixes", "decode_batch", "draw_session_tokens",
    "eval_burst", "eval_rung", "eval_sample", "fmt", "make_client",
    "make_session", "make_text", "pct", "per_user_p50", "run_burst",
    "run_population", "run_sample", "sampler_selfcheck", "send_request",
    "spike_evidence", "sub_prefix_floor",
]
