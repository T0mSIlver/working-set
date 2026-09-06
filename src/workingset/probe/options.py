"""Knobs every probe shares — the harness's `args` namespace, made explicit.

One object so a probe result can record exactly what produced it, and so a
`RunContext` can key its cache on (probe, parameters).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .stats import DEFAULT_RUNGS


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
    context_cap_tokens: int = 180_000   # truncation cap (the model's max_seq_len)
    request_timeout_s: float = 300.0    # keep well above the TTFT budget
    ignore_eos: bool = True             # vLLM extension: fixed-length decode
    api: str = "completions"            # "completions" | "chat"
    # --- gap accounting ---------------------------------------------------
    freeze_threshold_ms: float = 100.0  # a gap >= this counts as a FREEZE
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def ladder_key(self) -> tuple:
        """Cache key: everything that changes what a ladder rung measures."""
        return (self.rungs, self.max_users, self.ramp_s, self.measure_s,
                self.turns_per_user, self.chars_per_token,
                self.context_cap_tokens, self.request_timeout_s,
                self.ignore_eos, self.api, self.freeze_threshold_ms, self.seed)
