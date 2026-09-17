"""Knobs every probe shares — the harness's `args` namespace, made explicit.

One object so a probe result can record exactly what produced it, and so a
`RunContext` can key its cache on (probe, parameters).
"""
from __future__ import annotations

import secrets
from dataclasses import asdict, dataclass, field

from .stats import DEFAULT_RUNGS

# ONE nonce per process, drawn at import. Every prompt the probe builds is
# otherwise a pure function of `seed`, so a second run at the same seed
# re-sends byte-identical "misses" and a server with prefix caching answers
# them from cache — the run then measures a hit and calls it a miss. Per
# PROCESS rather than per `ProbeOptions` so two option objects built in one
# run (a `replace`, a dry run next to the real thing) agree on the bytes.
_PROCESS_NONCE = secrets.token_hex(8)


def process_nonce() -> str:
    return _PROCESS_NONCE


@dataclass(frozen=True)
class ProbeOptions:
    # --- ladder shape -----------------------------------------------------
    rungs: str = DEFAULT_RUNGS          # multipliers of predicted_limit_users
    max_users: int = 1024               # hard cap on any rung's population
    ramp_s: float = 90.0                # sessions establish; not measured
    measure_s: float = 180.0            # steady-state measure window
    turns_per_user: int = 0             # 0 = unlimited within the window
    # --- burst probe ------------------------------------------------------
    burst: int = 0                      # N simultaneous forced misses
    burst_users: int = 0                # standing load (0 = operating point)
    # --- cheap sample (the non-exclusive probe) ---------------------------
    sample_requests: int = 8            # forced misses fired for the cheap tests
    sample_warm_turns: int = 3          # warm turns per sampled session
    # --- request shaping --------------------------------------------------
    chars_per_token: float = 4.0        # synthetic-text calibration
    tokenizer: str | None = None        # HF slug / tokenizer.json that set it
    context_cap_tokens: int = 180_000   # truncation cap (the model's max_seq_len)
    request_timeout_s: float = 300.0    # keep well above the TTFT budget
    ignore_eos: bool = True             # vLLM extension: fixed-length decode
    api: str = "completions"            # "completions" | "chat"
    # --- gap accounting ---------------------------------------------------
    freeze_threshold_ms: float = 100.0  # a gap >= this counts as a FREEZE
    seed: int = 0
    # what makes THIS run's misses unmatchable by the last run's cache: mixed
    # into every miss salt and every per-session context (never into the
    # shared prefix, which is byte-stable across runs on purpose). `seed`
    # still fixes the prompt LENGTHS, the think times and the miss pattern,
    # so two runs at one seed offer the same load in different bytes; the
    # pair (seed, run_nonce) reproduces a run exactly, which is why the nonce
    # is in the record. "" mixes nothing in (runs at one seed then collide in
    # the prefix cache; it does not reproduce a pre-nonce version's bytes).
    run_nonce: str = field(default_factory=process_nonce)

    def to_dict(self) -> dict:
        return asdict(self)

    def ladder_key(self) -> tuple:
        """Cache key: everything that changes what a ladder rung measures."""
        return (self.rungs, self.max_users, self.ramp_s, self.measure_s,
                self.turns_per_user, self.chars_per_token,
                self.context_cap_tokens, self.request_timeout_s,
                self.ignore_eos, self.api, self.freeze_threshold_ms, self.seed,
                self.run_nonce)
