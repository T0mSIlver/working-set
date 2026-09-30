"""The two latency limits: slowed generation and cold-request wait.

Both are ladder hypotheses (they generate the population they are about) and
both read one number per rung, then bracket the ceiling across rungs exactly as
the ceiling hypotheses do: the lower edge is the largest rung inside the limit,
the upper edge the smallest rung past it.

  H-slowed     while a cold prompt is prefilled every step carries one chunk of
               it, so every stream slows to a few tok/s. Scored on the share of
               stream time spent in such steps.
  H-cold-wait  a cold request queues behind the prefills already waiting.
               Scored on the p-th wait before its own prefill starts.

When the run also fired a burst, both read it too: the standing streams give
the slowed share, the fired misses give the wait. The burst is reported next to
the rungs, never mixed into the bracket, because it is a different load.
"""
from __future__ import annotations

import math

from .. import model as M
from .base import (BURST_PROBE, EXCLUSIVE, LADDER, NOT_ESTABLISHED, REFUTED,
                   Hypothesis, Measurement, Prediction, Verdict,
                   bracket_verdict)
from .ceilings import _bracket_text

NEVER = 999999          # Predictions' "this ceiling never binds"


def _fin(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def bracket_over(rungs, value, limit):
    """(lo, hi) of the largest rung inside `limit` and the smallest past it.

    `value(rung)` is the rung's statistic, None/nan when the rung carried none.
    Rungs without it are ignored, not read as passing.
    """
    seen = [(r.pop, value(r)) for r in rungs]
    seen = [(pop, v) for pop, v in seen if _fin(v)]
    over = [pop for pop, v in seen if v > limit]
    if not over:
        return None, None, bool(seen)
    hi = min(over)
    lo = max((pop for pop, v in seen if pop < hi and v <= limit), default=None)
    return lo, hi, True


def ceiling_verdict(pred: Prediction, m: Measurement) -> Verdict:
    """Bracket verdict, plus the case the bracket cannot express: the model
    says the limit never binds and a rung crossed it anyway."""
    if m.lo is None and m.hi is None:
        return Verdict(NOT_ESTABLISHED,
                       m.data.get("reason", "no rung crossed the limit"))
    if pred.value is None:
        if m.hi is not None:
            return Verdict(REFUTED, f"crossed at {m.hi:g} users; the model "
                                    "says this limit never binds")
        return Verdict(NOT_ESTABLISHED, "the model says this limit never binds")
    return bracket_verdict(pred.value, m.lo, m.hi)


def _rate_total(cfg, preds, pop):
    """Total arrival rate at `pop` users: the operating point's, scaled in
    users. None when the operating point is zero."""
    if not preds.operating_point_users:
        return None
    return (preds.req_rate_main * pop / preds.operating_point_users
            * (1.0 + cfg.workload.subagent_ratio))


def predicted_share(cfg, preds, pop):
    rate = _rate_total(cfg, preds, pop)
    if rate is None:
        return None
    return M.slowed_share(cfg.to_model(), cfg.to_topology(), cfg.to_workload(),
                          rate, cfg.deployment.max_num_batched_tokens,
                          cfg.slo.itl_floor_tok_s, cfg.calibration.mfu,
                          per_pass_overhead=True)


def predicted_wait(cfg, preds, pop):
    rate = _rate_total(cfg, preds, pop)
    if rate is None:
        return None
    slo = cfg.slo
    return M.cold_wait_seconds(
        cfg.to_model(), cfg.to_topology(), cfg.to_workload(), rate,
        cfg.deployment.max_num_batched_tokens, cfg.workload.warm_turn_tokens,
        cfg.calibration.mfu, per_pass_overhead=True,
        percentile=None if slo.ttft_statistic == "miss_mean"
        else slo.percentile)


def _ceiling_prediction(users) -> Prediction:
    if users >= NEVER:
        return Prediction(value=None, text="never binds")
    return Prediction(value=users, unit=" users")


class HSlowed(Hypothesis):
    key = "H-slowed"
    title = "streams are slowed no more than the allowed share of the time"
    requires = frozenset({EXCLUSIVE})
    probes = frozenset({LADDER})

    def conditional_probes(self, planned) -> frozenset:
        return frozenset({BURST_PROBE}) if BURST_PROBE in planned \
            else frozenset()

    def statement(self, cfg, p) -> str:
        slo = cfg.slo
        reach = ("never reaches it" if p.slowed_ceiling_users >= NEVER
                 else f"reaches it near ~{p.slowed_ceiling_users:g} users")
        return (
            f"H-slowed: while a cold prompt is prefilled each step carries "
            f"one {cfg.deployment.max_num_batched_tokens:,}-token chunk, so "
            f"every stream slows to ~{p.slowed_speed_tok_s:g} tok/s. At the "
            f"~{p.operating_point_users:g}-user operating point streams are "
            f"slowed {p.slowed_share:.1%} of the time; the slowed share "
            f"{reach} (limit {slo.slowed_share_max:.1%}), and one cold request "
            f"of the p{slo.percentile} prompt length keeps them slowed "
            f"~{p.slowed_stretch_s:g} s. H-itl-spike measures the size of "
            "one spike; this measures how much of the time streams spend "
            "slowed.")

    def predict(self, cfg, p) -> Prediction:
        return _ceiling_prediction(p.slowed_ceiling_users)

    async def measure(self, ctx) -> Measurement:
        v = await ctx.ladder()
        cfg, preds = ctx.cfg, ctx.predictions
        limit = cfg.slo.slowed_share_max
        rungs = [r for r in v.full if r.slowed]
        by_pop = {r.pop: {
            "share": r.slowed["share"],
            "speed_tok_s": r.slowed.get("speed_tok_s"),
            "longest_run_s": r.slowed.get("longest_run_s"),
            "run_pX_s": r.slowed.get("run_pX_s"),
            "n_streams": r.slowed.get("n_streams"),
            "predicted_share": predicted_share(cfg, preds, r.pop)}
            for r in rungs}
        data = {"by_pop": by_pop, "limit": limit,
                "threshold_ms": next((r.slowed["threshold_ms"]
                                      for r in rungs), None),
                "predicted_speed_tok_s": preds.slowed_speed_tok_s,
                "predicted_stretch_s": preds.slowed_stretch_s,
                "percentile": cfg.slo.percentile}
        burst = await ctx.burst() if BURST_PROBE in ctx.probes else None
        if burst is not None and burst.slowed:
            data["burst"] = {"standing_users": burst.standing_users,
                             **burst.slowed}
        lo, hi, any_seen = bracket_over(rungs, lambda r: r.slowed["share"],
                                        limit)
        if hi is None:
            data["reason"] = (
                f"no rung's slowed share exceeded {limit:.1%}" if any_seen
                else "no rung measured a stream with inter-token gaps")
            return Measurement(text="not separable", data=data)
        return Measurement(value=hi, lo=lo, hi=hi, unit=" users",
                           text=_bracket_text(lo, hi), data=data)

    def verdict(self, pred: Prediction, m: Measurement) -> Verdict:
        return ceiling_verdict(pred, m)


class HColdWait(Hypothesis):
    key = "H-cold-wait"
    title = "a cold request waits no longer than the budget before its prefill starts"
    requires = frozenset({EXCLUSIVE})
    probes = frozenset({LADDER})

    def conditional_probes(self, planned) -> frozenset:
        return frozenset({BURST_PROBE}) if BURST_PROBE in planned \
            else frozenset()

    def statement(self, cfg, p) -> str:
        slo = cfg.slo
        which = ("mean" if slo.ttft_statistic == "miss_mean"
                 else f"p{slo.percentile}")
        reach = ("never reaches it" if p.cold_wait_ceiling_users >= NEVER
                 else f"reaches it near ~{p.cold_wait_ceiling_users:g} users")
        wait = ("unbounded (past saturation)" if not _fin(p.cold_wait_s)
                else f"~{p.cold_wait_s:g} s")
        return (
            f"H-cold-wait: the {which} wait before a cold request's own "
            f"prefill starts is {wait} at the "
            f"~{p.operating_point_users:g}-user operating point and {reach} "
            f"(budget {slo.cold_wait_budget_s:g} s). Measured as client TTFT "
            "minus the idle prefill time of that prompt length, from the "
            "model at the configured MFU.")

    def predict(self, cfg, p) -> Prediction:
        return _ceiling_prediction(p.cold_wait_ceiling_users)

    async def measure(self, ctx) -> Measurement:
        v = await ctx.ladder()
        cfg, preds = ctx.cfg, ctx.predictions
        budget = cfg.slo.cold_wait_budget_s
        rungs = [r for r in v.full if r.cold_wait]
        by_pop = {r.pop: {
            "client_wait_s": r.cold_wait["value_s"],
            "mean_s": r.cold_wait.get("mean_s"),
            "n": r.cold_wait.get("n"),
            "server_queue_mean_s": (r.queue_time or {}).get("mean_s"),
            "server_queue_p95_s": (r.queue_time or {}).get("p95_s"),
            "predicted_s": predicted_wait(cfg, preds, r.pop)}
            for r in rungs}
        data = {"by_pop": by_pop, "budget_s": budget,
                "idle_source": next((r.cold_wait["idle_source"]
                                     for r in rungs), None),
                "server_queue_note": (
                    "vLLM counts queue time until the request is first "
                    "scheduled, which is not the same as until its prefill "
                    "starts when prefills share a step; both are reported "
                    "and not forced to agree")}
        burst = await ctx.burst() if BURST_PROBE in ctx.probes else None
        if burst is not None and burst.cold_wait:
            data["burst"] = {"n": burst.n, **burst.cold_wait,
                             "server_queue": burst.queue_time}
        lo, hi, any_seen = bracket_over(rungs, lambda r: r.cold_wait["value_s"],
                                        budget)
        if hi is None:
            data["reason"] = (
                f"no rung's cold wait exceeded {budget:g} s" if any_seen
                else "no rung measured a cold request")
            return Measurement(text="not separable", data=data)
        return Measurement(value=hi, lo=lo, hi=hi, unit=" users",
                           text=_bracket_text(lo, hi), data=data)

    def verdict(self, pred: Prediction, m: Measurement) -> Verdict:
        return ceiling_verdict(pred, m)
