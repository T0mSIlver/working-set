"""The four ceilings and the binding one — the ladder hypotheses.

Every one of these needs to GENERATE the population it is about, so all five
carry `requires = {"exclusive"}`. They share one ladder: the `RunContext`
runs it once and each reads the bracket it cares about.

Verdict logic ported from `print_report`'s PREDICTED vs MEASURED block in
scripts/validate_deployment.py, statement text from `harnessHypotheses` in
interactive/src/harness.js.
"""
from __future__ import annotations

import math

from .base import (BOUNDED_BELOW, EXCLUSIVE, NOT_ESTABLISHED, REFUTED,
                   Hypothesis, Measurement, Prediction, Verdict,
                   bracket_verdict)


def _bracket_text(lo, hi) -> str:
    if lo is not None and hi is not None:
        return f"({lo},{hi}]"
    if lo is not None:
        return f">={lo}"
    if hi is not None:
        return f"<{hi}"
    return "-"


class HCache(Hypothesis):
    key = "H-cache"
    title = "the warm-session pool holds the predicted p5 population"
    requires = frozenset({EXCLUSIVE})

    def statement(self, cfg, p) -> str:
        return (f"H-cache: >= {p.warm_capacity_p5:g} user sessions stay warm "
                "(p5). A run bounds this below unless load reaches eviction "
                "(watch the effective-cold fraction).")

    def predict(self, cfg, p) -> Prediction:
        return Prediction(value=p.warm_capacity_p5, unit=" users")

    async def measure(self, ctx) -> Measurement:
        v = await ctx.ladder()
        held = v.warm_held
        if held:
            bound = max(held)
            return Measurement(value=bound, lo=bound, unit=" users",
                               text=f">= {bound}",
                               data={"held_pops": held,
                                     "classifier": "0.4x-cold TTFT heuristic"})
        ev = v.warm_evicted
        if ev is not None:
            return Measurement(value=ev.pop, hi=ev.pop, unit=" users",
                               text=f"< ~{ev.pop}",
                               data={"evict_frac": ev.evict_frac,
                                     "pop": ev.pop,
                                     "cached_frac": ev.cached_frac})
        return Measurement(text="not separable", data={})

    def verdict(self, pred: Prediction, m: Measurement) -> Verdict:
        wc = pred.value
        if m.lo is not None:
            if wc is not None and m.lo >= wc:
                return Verdict(BOUNDED_BELOW, "held at/above prediction with "
                                              "<5% effective-cold hits (lower bound)")
            return Verdict(BOUNDED_BELOW, "lower bound only — eviction not reached")
        if m.hi is not None:
            ef = m.data.get("evict_frac", float("nan"))
            note = (f"{ef:.0%} of hit turns re-prefilled at {m.hi} users"
                    if isinstance(ef, float) and math.isfinite(ef)
                    else f"eviction observed at {m.hi} users")
            if wc is not None and m.hi < 0.75 * wc:
                return Verdict(REFUTED, note)
            return Verdict(NOT_ESTABLISHED, note)
        return Verdict(NOT_ESTABLISHED, "no data")


class HDecode(Hypothesis):
    key = "H-decode"
    title = "per-user p50 decode holds at the floor up to the decode ceiling"
    requires = frozenset({EXCLUSIVE})

    def statement(self, cfg, p) -> str:
        return (f"H-decode: per-user p50 decode holds >= "
                f"{cfg.slo.itl_floor_tok_s:g} tok/s up to "
                f"~{p.decode_ceiling_users:g} concurrent users.")

    def predict(self, cfg, p) -> Prediction:
        return Prediction(value=p.decode_ceiling_users, unit=" users")

    async def measure(self, ctx) -> Measurement:
        v = await ctx.ladder()
        fails = v.decode_fails
        if not fails:
            return Measurement(text="not separable",
                               data={"reason": "no decode-floor failure observed"})
        lo, hi = v.decode_lo, min(fails)
        return Measurement(value=hi, lo=lo, hi=hi, unit=" users",
                           text=_bracket_text(lo, hi),
                           data={"decode_p50_by_pop":
                                 {r.pop: r.decode_p50 for r in v.full}})

    def verdict(self, pred: Prediction, m: Measurement) -> Verdict:
        if m.lo is None and m.hi is None:
            return Verdict(NOT_ESTABLISHED, "no decode-floor failure observed")
        return bracket_verdict(pred.value, m.lo, m.hi)


class HLatency(Hypothesis):
    key = "H-latency"
    title = "a miss's mean TTFT reaches the budget at the latency ceiling"
    requires = frozenset({EXCLUSIVE})

    def statement(self, cfg, p) -> str:
        return (f"H-latency: a cache miss's mean TTFT reaches the "
                f"{cfg.slo.ttft_budget_s:g} s budget near "
                f"~{p.latency_ceiling_users:g} users.")

    def predict(self, cfg, p) -> Prediction:
        return Prediction(value=p.latency_ceiling_users, unit=" users")

    async def measure(self, ctx) -> Measurement:
        v = await ctx.ladder()
        fails = v.ttft_fails
        if not fails:
            return Measurement(text="not separable",
                               data={"reason": "no TTFT-mode failure observed"})
        lo, hi = v.latency_lo, min(fails)
        return Measurement(value=hi, lo=lo, hi=hi, unit=" users",
                           text=_bracket_text(lo, hi),
                           data={"ttft_all_pX_by_pop":
                                 {r.pop: r.ttft_all_pX for r in v.full}})

    def verdict(self, pred: Prediction, m: Measurement) -> Verdict:
        if m.lo is None and m.hi is None:
            return Verdict(NOT_ESTABLISHED, "no TTFT-mode failure observed")
        return bracket_verdict(pred.value, m.lo, m.hi)


class HSaturation(Hypothesis):
    key = "H-saturation"
    title = "prefill duty reaches 100% at the saturation ceiling"
    requires = frozenset({EXCLUSIVE})

    def statement(self, cfg, p) -> str:
        sat = ("never binds" if p.saturation_ceiling_users >= 999999
               else f"~{p.saturation_ceiling_users:g}")
        return (f"H-saturation: prefill duty reaches 100% near {sat} users; "
                "above it the queue has no steady state.")

    def predict(self, cfg, p) -> Prediction:
        if p.saturation_ceiling_users >= 999999:
            return Prediction(value=None, text="never binds")
        return Prediction(value=p.saturation_ceiling_users, unit=" users")

    async def measure(self, ctx) -> Measurement:
        v = await ctx.ladder()
        r0 = v.saturation_evidence
        if r0 is None:
            return Measurement(
                text="not separable",
                data={"reason": "closed loop throttles before duty=100% "
                                "is visible"})
        return Measurement(value=r0.pop, hi=r0.pop, unit=" users",
                           text=f"<= {r0.pop}",
                           data={"achieved_rps": r0.achieved_rps,
                                 "offered_rps": r0.offered_rps})

    def verdict(self, pred: Prediction, m: Measurement) -> Verdict:
        # Never better than not_established: a closed loop throttles itself
        # before prefill duty hits 100%, so the ladder sees a throughput
        # plateau, not the ceiling. Same conclusion as the harness, which
        # prints ~ on both branches of this row.
        if m.hi is None:
            return Verdict(NOT_ESTABLISHED,
                           "closed loop throttles before duty=100% is visible")
        return Verdict(NOT_ESTABLISHED,
                       f"achieved {m.data['achieved_rps']:.2f} req/s vs offered "
                       f"{m.data['offered_rps']:.2f} — throughput plateau")


class HBinding(Hypothesis):
    key = "H-binding"
    title = "measured SLO capacity lands at the binding ceiling"
    requires = frozenset({EXCLUSIVE})

    def statement(self, cfg, p) -> str:
        return (f"H-binding: the binding constraint is "
                f"'{p.binding_constraint}' — measured SLO capacity should "
                f"land near {p.predicted_limit_users:g} users.")

    def predict(self, cfg, p) -> Prediction:
        return Prediction(value=p.predicted_limit_users, unit=" users",
                          text=f"{p.predicted_limit_users:g} "
                               f"({p.binding_constraint} binds)")

    async def measure(self, ctx) -> Measurement:
        v = await ctx.ladder()
        lo, hi = v.lo, v.hi
        slo = ctx.cfg.slo
        return Measurement(
            value=lo, lo=lo, hi=hi, unit=" users", text=_bracket_text(lo, hi),
            data={"definition": f"p{slo.percentile} TTFT <= "
                                f"{slo.ttft_budget_s:g}s AND per-user p50 "
                                f"decode >= {slo.itl_floor_tok_s:g} tok/s",
                  "passed": [r.pop for r in v.full if r.passed],
                  "failed": [r.pop for r in v.full if not r.passed]})

    def verdict(self, pred: Prediction, m: Measurement) -> Verdict:
        return bracket_verdict(pred.value, m.lo, m.hi)
