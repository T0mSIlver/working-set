"""The four ceilings and the binding one — the ladder hypotheses.

Every one of these needs to GENERATE the population it is about, so all five
carry `requires = {"exclusive"}`. They share one ladder: the `RunContext`
runs it once and each reads the bracket it cares about.

Verdict logic ported from `print_report`'s PREDICTED vs MEASURED block in
the retired standalone harness, statement text from `harnessHypotheses` in
interactive/src/harness.js.
"""
from __future__ import annotations

import math

from .base import (BOUNDED_BELOW, EXCLUSIVE, LADDER, NOT_ESTABLISHED, REFUTED,
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
    probes = frozenset({LADDER})

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
        unwitnessed = v.warm_unwitnessed
        if unwitnessed:
            # rungs ran, but nothing could tell warm from cold on them
            return Measurement(
                text="not separable",
                data={"unwitnessed_pops": unwitnessed,
                      "reason": "no forced miss to calibrate the cold-TTFT "
                                "classifier and no cached_tokens from the "
                                "server — warmth was never observed"})
        return Measurement(text="not separable",
                           data={"reason": "no rung produced hit turns"})

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
        return Verdict(NOT_ESTABLISHED, m.data.get("reason", "no data"))


# THE LADDER IS THE WRONG INSTRUMENT FOR THIS CEILING, and the row has to say
# so rather than leave "no decode-floor failure observed" to be read as good
# news. `max_users_decode` is a count of sequences decoding AT ONCE. The
# ladder is a closed loop with think time: a user spends most of a cycle
# thinking, so N users hold a decode batch far below N
# (`Predictions.steady_decode_seqs` at the operating point), and any ladder
# whose rungs stay near the other ceilings never brings the batch anywhere
# near this one. The instrument that does is a sweep that HOLDS k sequences
# decoding together.
DECODE_INSTRUMENT = ("a held-batch decode sweep (scripts/decode_probe.py in "
                     "the repository, which holds k sequences decoding at "
                     "once) is the instrument for this ceiling")


def _decode_not_reached(v, ceiling) -> dict:
    """What the ladder's decode batch actually was, against the ceiling.

    `Rung.decode_seqs` is the mean `requests_running` inside the measure
    window, so it exists only with a metrics sampler attached.
    """
    seen = [(r.decode_seqs, r.pop) for r in v.full
            if math.isfinite(r.decode_seqs)]
    data = {"decode_ceiling_seqs": ceiling, "largest_decode_batch": None,
            "largest_decode_batch_pop": None}
    why = "no decode-floor failure observed"
    if not seen:
        why += ("; the decode batch the ladder held was not observed (no "
                "--metrics-url), and a closed loop with think time holds far "
                f"fewer sequences decoding at once than it has users — "
                f"{DECODE_INSTRUMENT}")
    else:
        batch, pop = max(seen)
        data.update(largest_decode_batch=batch, largest_decode_batch_pop=pop)
        why += (f"; the largest decode batch this ladder produced was "
                f"{batch:.1f} sequences decoding at once (the {pop}-user "
                f"rung) against a predicted ceiling of ~{ceiling:g}")
        if ceiling is None or batch < ceiling:
            why += (" — a closed loop with think time never holds the batch "
                    f"this ceiling is about, so it was not tested: "
                    f"{DECODE_INSTRUMENT}")
    data["reason"] = why
    return data


class HDecode(Hypothesis):
    key = "H-decode"
    title = "per-user p50 decode holds at the floor up to the decode ceiling"
    requires = frozenset({EXCLUSIVE})
    probes = frozenset({LADDER})

    def statement(self, cfg, p) -> str:
        if getattr(p, "decode_capped_by_max_num_seqs", False):
            # a different claim, not a smaller number: the batch is pinned at
            # the cap, so decode speed never falls to the floor by bandwidth
            return (f"H-decode: the scheduler caps the batch at "
                    f"{p.decode_ceiling_users:g} sequences (max_num_seqs), below "
                    f"the bandwidth ceiling: per-user decode stays above "
                    f"{cfg.slo.itl_floor_tok_s:g} tok/s and requests past the "
                    "cap queue instead — watch TTFT. No decode-floor failure "
                    "is expected, so this row cannot be bracketed.")
        seqs = getattr(p, "steady_decode_seqs", None)
        held = (f"~{seqs:g} at the ~{p.operating_point_users:g}-user "
                "operating point" if seqs is not None
                else "far fewer than it has users")
        return (f"H-decode: per-user p50 decode holds >= "
                f"{cfg.slo.itl_floor_tok_s:g} tok/s up to "
                f"~{p.decode_ceiling_users:g} concurrent users. NOTE the "
                "ceiling counts sequences decoding AT ONCE, and the ladder's "
                f"closed loop with think time holds {held}: unless a rung "
                "fails the decode floor this row reports the batch the "
                f"ladder reached and nothing more — {DECODE_INSTRUMENT}.")

    def predict(self, cfg, p) -> Prediction:
        return Prediction(value=p.decode_ceiling_users, unit=" users")

    async def measure(self, ctx) -> Measurement:
        v = await ctx.ladder()
        fails = v.decode_fails
        if not fails:
            return Measurement(
                text="not separable",
                data=_decode_not_reached(
                    v, ctx.predictions.decode_ceiling_users))
        lo, hi = v.decode_lo, min(fails)
        return Measurement(value=hi, lo=lo, hi=hi, unit=" users",
                           text=_bracket_text(lo, hi),
                           data={"decode_p50_by_pop":
                                 {r.pop: r.decode_p50 for r in v.full}})

    def verdict(self, pred: Prediction, m: Measurement) -> Verdict:
        if m.lo is None and m.hi is None:
            # the measurement's own account of WHY: which batch the ladder
            # reached, and that it is not the instrument for this ceiling
            return Verdict(NOT_ESTABLISHED,
                           m.data.get("reason",
                                      "no decode-floor failure observed"))
        return bracket_verdict(pred.value, m.lo, m.hi)


class HLatency(Hypothesis):
    key = "H-latency"
    title = "the checked TTFT statistic reaches the budget at the latency ceiling"
    requires = frozenset({EXCLUSIVE})
    probes = frozenset({LADDER})

    def statement(self, cfg, p) -> str:
        stat = ("a cache miss's mean TTFT"
                if cfg.slo.ttft_statistic == "miss_mean"
                else f"the p{cfg.slo.percentile} TTFT over all requests "
                     f"(model proxy: mean wait + the p{cfg.slo.percentile} "
                     f"of the hit/miss service mixture)")
        return (f"H-latency: {stat} reaches the {cfg.slo.ttft_budget_s:g} s "
                f"budget near ~{p.latency_ceiling_users:g} users.")

    def predict(self, cfg, p) -> Prediction:
        return Prediction(value=p.latency_ceiling_users, unit=" users")

    async def measure(self, ctx) -> Measurement:
        v = await ctx.ladder()
        slo = ctx.cfg.slo
        if slo.ttft_statistic == "miss_mean":
            # judge what was predicted: each rung's mean TTFT over its forced
            # misses, not the rung's all-request p{percentile} verdict
            budget = slo.ttft_budget_s
            seen = [r for r in v.full if math.isfinite(r.ttft_miss_mean)]
            over = [r.pop for r in seen if r.ttft_miss_mean > budget]
            data = {"ttft_miss_mean_by_pop": {r.pop: r.ttft_miss_mean
                                              for r in seen}}
            if not over:
                return Measurement(text="not separable", data={
                    **data, "reason": "no rung's miss-mean TTFT exceeded the "
                                      "budget" if seen else
                                      "no rung measured a miss"})
            hi = min(over)
            lo = max((r.pop for r in seen if r.pop < hi
                      and r.ttft_miss_mean <= budget), default=None)
            return Measurement(value=hi, lo=lo, hi=hi, unit=" users",
                               text=_bracket_text(lo, hi), data=data)
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
            return Verdict(NOT_ESTABLISHED,
                           m.data.get("reason", "no TTFT-mode failure observed"))
        return bracket_verdict(pred.value, m.lo, m.hi)


class HSaturation(Hypothesis):
    key = "H-saturation"
    title = "prefill duty reaches 100% at the saturation ceiling"
    requires = frozenset({EXCLUSIVE})
    probes = frozenset({LADDER})

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
    probes = frozenset({LADDER})

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
