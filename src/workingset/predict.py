"""Price a RunConfig: the four ceilings, the binding one, and the operating
point's load figures. This is the `predictions` block the explorer's generated
harness used to carry — produced here from the model instead of stored.

Conventions (match the explorer's planner):
  * open loop by default: rate = users x (1 + r) / think_time_s (assumption 2);
    `closed=True` switches latency/saturation to the closed-loop conversion,
    where `think_time_s` is read as Z, the WAITING time per request (the
    model supplies the service time R itself; the measured Z is 32.5 s).
  * misses are priced WITH the per-pass overhead (`per_pass_overhead=True`),
    the explorer's convention since 2026-08-02; docs/scenarios.md's tables
    keep the roofline convention (False) and differ in the last digits.
  * every ceiling is PER REPLICA GROUP; the `system` block multiplies cache
    and decode by the replica count (balanced routing only).
  * Monte-Carlo ceilings (cache, decode) use a flat `n_iter`, the explorer
    scales its iteration counts per probe: same conventions, different
    sampling, so they agree to sampling noise, not to the last digit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from . import model as M
from .config import RunConfig


@dataclass(frozen=True)
class Predictions:
    # ceilings, users per replica group
    warm_capacity_p5: int           # user-class warm sessions, p5 = the cache ceiling
    cache_ceiling_users: int        # every reusable session (users + subagents), p5
    decode_ceiling_users: int
    latency_ceiling_users: int
    saturation_ceiling_users: int   # 999999 when never binds at this rate
    binding_constraint: str
    predicted_limit_users: int
    # the operating point
    operating_point_users: int
    req_rate_main: float            # main-agent req/s at the operating point
    prefill_duty: float
    ttft_miss_s: float              # mean TTFT of a forced miss (FCFS)
    ttft_hit_s: float
    bstar_misses: float
    replicas: int
    # --- the steady-decode / inter-token-gap block ------------------------
    # Present only where the steady point is REAL (prefill duty < 1 and the
    # decode demand lands on the sampled axis); None otherwise, and every
    # hypothesis that quotes them degrades to "not established". Mirrors the
    # explorer's `harnessPredictions` guard (interactive/src/harness.js).
    steady_decode_seqs: float | None = None    # sequences decoding at the load
    steady_decode_tok_s: float | None = None   # per-user decode at THAT batch
    itl_normal_ms: float | None = None         # gap with no prefill in the pass
    itl_worst_freeze_ms: float | None = None   # last chunk of a cold re-prefill
    itl_freeze_lo_ms: float | None = None      # MFU 55% — the bracket's low edge
    itl_freeze_hi_ms: float | None = None      # MFU 35% — the bracket's high edge

    def to_dict(self) -> dict:
        """JSON-safe: non-finite floats become None (strict JSON has no inf)."""
        from dataclasses import asdict
        return {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                for k, v in asdict(self).items()}

    def system(self) -> dict:
        """Whole-deployment ceilings under balanced DP routing."""
        r = self.replicas
        return {"warm_capacity_p5": self.warm_capacity_p5 * r,
                "cache_ceiling_users": self.cache_ceiling_users * r,
                "decode_ceiling_users": self.decode_ceiling_users * r,
                "latency_ceiling_users": self.latency_ceiling_users * r,
                "saturation_ceiling_users": (self.saturation_ceiling_users * r
                                             if self.saturation_ceiling_users < 999999
                                             else 999999),
                "predicted_limit_users": self.predicted_limit_users * r}


def predict(cfg: RunConfig, closed: bool = False, n_iter: int = 400,
            seed: int = 0) -> Predictions:
    cfg.validate()
    m, t, wl = cfg.to_model(), cfg.to_topology(), cfg.to_workload()
    w, slo, cal, dep = cfg.workload, cfg.slo, cfg.calibration, cfg.deployment
    chunk = dep.max_num_batched_tokens
    users = w.users

    ram = dep.ram_gib
    op = M.operating_point(
        m, t, wl, users, chunk=chunk, turn_tokens=w.warm_turn_tokens,
        sla_seconds=slo.ttft_budget_s, think_time_s=w.think_time_s,
        decode_floor=slo.itl_floor_tok_s, mfu=cal.mfu, ram_gib=ram,
        per_pass_overhead=True, closed=closed, z_think_s=w.think_time_s,
        out_tokens=w.max_output_tokens, n_iter=n_iter, seed=seed)
    # operating_point prices decode at the study default MBU; re-price at the
    # configured one so the calibration block is honoured
    if cal.mbu != M.MBU_DEFAULT:
        dec = M.max_users_decode(m, t, wl, floor=slo.itl_floor_tok_s,
                                 n_iter=n_iter, seed=seed, mbu=cal.mbu)
        op["ceilings"]["decode"] = dec
        op["binding"] = min(op["ceilings"], key=op["ceilings"].get)
        op["limit"] = op["ceilings"][op["binding"]]
        op["headroom"] = users / op["limit"] if op["limit"] > 0 else math.inf
        op["fits"] = users <= op["limit"]

    # op["ceilings"]["cache"] is already the user-class warm p5 (the plan
    # column); the all-classes count is what the pool physically holds
    draw = int(4000 + M.kv_pool_tokens(m, t) / 8000)
    p5_all, _, _ = M.warm_capacity(m, t, wl, ram_gib=ram, n_iter=n_iter,
                                   draw=draw, seed=seed, which="all")

    rate = op["req_rate"]                       # main-agent req/s
    rate_total = rate * (1.0 + wl.sub_ratio)    # what the prefill server sees
    duty = M.prefill_duty(m, t, wl, rate_total, chunk, w.warm_turn_tokens,
                          cal.mfu, per_pass_overhead=True)
    if duty >= 1.0:
        ttft_miss = ttft_hit = math.inf
    else:
        ttft_miss = M.prefill_ttft_seconds(m, t, wl, rate_total, chunk,
                                           w.warm_turn_tokens, cal.mfu, "cold",
                                           per_pass_overhead=True)
        ttft_hit = M.prefill_ttft_seconds(m, t, wl, rate_total, chunk,
                                          w.warm_turn_tokens, cal.mfu, "warm",
                                          per_pass_overhead=True)
    bstar = M.spike_tolerance(m, t, wl, slo.ttft_budget_s, rate_total, chunk,
                              w.warm_turn_tokens, cal.mfu, per_pass_overhead=True)

    def _int(x: float) -> int:
        return 999999 if not math.isfinite(x) else int(round(x))

    steady = _steady_block(m, t, wl, rate_total, w.max_output_tokens,
                           dep.max_model_len, chunk, cal.mfu, duty,
                           n_iter=n_iter, seed=seed)

    return Predictions(
        warm_capacity_p5=_int(op["ceilings"]["cache"]),
        cache_ceiling_users=_int(p5_all),
        decode_ceiling_users=_int(op["ceilings"]["decode"]),
        latency_ceiling_users=_int(op["ceilings"]["latency"]),
        saturation_ceiling_users=_int(op["ceilings"]["saturation"]),
        binding_constraint=op["binding"],
        predicted_limit_users=_int(op["limit"]),
        operating_point_users=int(round(users)),
        req_rate_main=round(rate, 4),
        prefill_duty=round(duty, 4),
        ttft_miss_s=round(ttft_miss, 3) if math.isfinite(ttft_miss) else math.inf,
        ttft_hit_s=round(ttft_hit, 3) if math.isfinite(ttft_hit) else math.inf,
        bstar_misses=round(bstar, 2),
        replicas=t.replicas or 1,
        **steady,
    )


# ============================================================================
# the steady-decode / inter-token-gap block
# ----------------------------------------------------------------------------
# Formulas transcribed from interactive/src/harness.js (`harnessPredictions`
# and `freezeMs`), which is the code that generated every `predictions` block
# a validate_deployment.py ever carried. No new modelling: every term comes
# from workingset.model.
#
#   steady_decode_seqs   steady_decode_point(...)["n"]
#   steady_decode_tok_s  steady_decode_point(...)["per_user_tok_s"]  (= pu)
#   itl_normal_ms        1000 x model.mtp / pu
#                        one decode step, per user, with no prefill in the pass
#   itl_worst_freeze_ms  1000 x (model.mtp / pu
#                                + prefill_seconds(step, mfu, prior=cap - step))
#                        with step = min(chunk, cap). The LAST chunk of a
#                        full-context cold re-prefill joins the decode batch,
#                        so every decoder sees one gap of step-time plus
#                        chunk-time. MARGINAL pricing: the host pass streams
#                        the weights anyway.
#   the [lo, hi] bracket the same at MFU_HIGH / MFU_LOW (the study's 35-55%
#                        band). Higher MFU = shorter freeze, so the HI anchor
#                        is the bracket's LOW edge.
#
# Guard (the explorer's): the block exists only where the steady point is
# real. Here that means prefill duty < 1 (no steady state upstream otherwise),
# the steady point not saturated, and pu > 0.
# ============================================================================
def freeze_ms(model: M.Model, topo: M.Topology, cap: float, chunk: float,
              per_user_tok_s: float, mfu: float = M.MFU_DEFAULT) -> float:
    """Worst inter-token gap behind one chunk of a full-context cold
    re-prefill, ms. Mirrors `freezeMs` in interactive/src/harness.js."""
    if per_user_tok_s <= 0:
        raise ValueError("per_user_tok_s must be > 0")
    # a chunk larger than the whole context is one cap-sized pass with no
    # cache behind it — never a full chunk with a negative prior
    step = min(chunk, cap)
    return 1e3 * (model.mtp / per_user_tok_s
                  + M.prefill_seconds(model, topo, step, mfu, prior=cap - step))


def _steady_block(m, t, wl, rate_total: float, out_tokens: float, cap: float,
                  chunk: float, mfu: float, duty: float,
                  n_iter: int, seed: int) -> dict:
    empty = {"steady_decode_seqs": None, "steady_decode_tok_s": None,
             "itl_normal_ms": None, "itl_worst_freeze_ms": None,
             "itl_freeze_lo_ms": None, "itl_freeze_hi_ms": None}
    if duty >= 1.0:
        return empty
    sp = M.steady_decode_point(m, t, wl, rate_total, out_tokens=out_tokens,
                               n_iter=n_iter, seed=seed)
    pu = sp["per_user_tok_s"]
    if sp["saturated"] or not (pu > 0):
        return empty
    return {
        "steady_decode_seqs": round(sp["n"], 2),
        "steady_decode_tok_s": round(pu),
        "itl_normal_ms": round(1e3 * m.mtp / pu, 1),
        "itl_worst_freeze_ms": round(freeze_ms(m, t, cap, chunk, pu, mfu)),
        # higher MFU = shorter freeze, so the HI anchor is the low edge
        "itl_freeze_lo_ms": round(freeze_ms(m, t, cap, chunk, pu, M.MFU_HIGH)),
        "itl_freeze_hi_ms": round(freeze_ms(m, t, cap, chunk, pu, M.MFU_LOW)),
    }
