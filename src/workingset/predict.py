"""Price a RunConfig: the four ceilings, the binding one, and the operating
point's load figures. This is the `predictions` block the explorer's generated
harness used to carry — produced here from the model instead of stored.

Conventions (match the explorer's planner):
  * open loop by default: rate = users x (1 + r) / think_time_s (assumption 2);
    `closed=True` switches latency/saturation to the closed-loop conversion.
  * every ceiling is PER REPLICA GROUP; the `system` block multiplies cache
    and decode by the replica count (balanced routing only).
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

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)

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

    op = M.operating_point(
        m, t, wl, users, chunk=chunk, turn_tokens=w.warm_turn_tokens,
        sla_seconds=slo.ttft_budget_s, think_time_s=w.think_time_s,
        decode_floor=slo.itl_floor_tok_s, mfu=cal.mfu, closed=closed,
        out_tokens=w.max_output_tokens, n_iter=n_iter, seed=seed)
    # operating_point prices decode at the study default MBU; re-price at the
    # configured one so the calibration block is honoured
    if cal.mbu != M.MBU_DEFAULT:
        dec = M.max_users_decode(m, t, wl, floor=slo.itl_floor_tok_s,
                                 n_iter=n_iter, seed=seed, mbu=cal.mbu)
        op["ceilings"]["decode"] = dec
        op["binding"] = min(op["ceilings"], key=op["ceilings"].get)
        op["limit"] = op["ceilings"][op["binding"]]

    # op["ceilings"]["cache"] is already the user-class warm p5 (the plan
    # column); the all-classes count is what the pool physically holds
    draw = int(4000 + M.kv_pool_tokens(m, t) / 8000)
    p5_all, _, _ = M.warm_capacity(m, t, wl, n_iter=n_iter, draw=draw,
                                   seed=seed, which="all")

    rate = op["req_rate"]                       # main-agent req/s
    rate_total = rate * (1.0 + wl.sub_ratio)    # what the prefill server sees
    duty = M.prefill_duty(m, t, wl, rate_total, chunk, w.warm_turn_tokens, cal.mfu)
    if duty >= 1.0:
        ttft_miss = ttft_hit = math.inf
    else:
        ttft_miss = M.prefill_ttft_seconds(m, t, wl, rate_total, chunk,
                                           w.warm_turn_tokens, cal.mfu, "cold")
        ttft_hit = M.prefill_ttft_seconds(m, t, wl, rate_total, chunk,
                                          w.warm_turn_tokens, cal.mfu, "warm")
    bstar = M.spike_tolerance(m, t, wl, slo.ttft_budget_s, rate_total, chunk,
                              w.warm_turn_tokens, cal.mfu)

    def _int(x: float) -> int:
        return 999999 if not math.isfinite(x) else int(round(x))

    return Predictions(
        warm_capacity_p5=_int(op["ceilings"]["cache"]),
        cache_ceiling_users=_int(p5_all),
        decode_ceiling_users=_int(op["ceilings"]["decode"]),
        latency_ceiling_users=_int(op["ceilings"]["latency"]),
        saturation_ceiling_users=_int(op["ceilings"]["saturation"]),
        binding_constraint=op["binding"],
        predicted_limit_users=_int(op["limit"]),
        operating_point_users=max(1, int(round(users))),
        req_rate_main=round(rate, 4),
        prefill_duty=round(duty, 4),
        ttft_miss_s=round(ttft_miss, 3) if math.isfinite(ttft_miss) else math.inf,
        ttft_hit_s=round(ttft_hit, 3) if math.isfinite(ttft_hit) else math.inf,
        bstar_misses=round(bstar, 2),
        replicas=t.replicas or 1,
    )
