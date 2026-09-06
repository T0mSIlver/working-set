"""The cheap hypotheses: a level, a batch, and two gap distributions.

None of these needs a ladder. Each runs on a handful of requests
(`ProbeOptions.sample_requests` sessions x a few turns each), so they can be
tested against an endpoint that is already serving other traffic — which is
also their limitation, stated on every row: a shared run measures at the
endpoint's PREVAILING load, not at the predicted operating point. When a
ladder IS being run (an exclusive run that also selected a ceiling
hypothesis), they read the rung nearest the operating point instead, exactly
as scripts/validate_deployment.py does.
"""
from __future__ import annotations

import math

from .base import (NOT_ESTABLISHED, REFUTED, SUPPORTED, Hypothesis,
                   Measurement, Prediction, Verdict, ratio_verdict)

_WEAKEST = {SUPPORTED: 0, NOT_ESTABLISHED: 1, REFUTED: 2}


def _weaker(a: Verdict, b: Verdict) -> Verdict:
    return a if _WEAKEST[a.status] >= _WEAKEST[b.status] else b


def _fin(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


async def _reading(ctx, predicate):
    """(source, obj, at_users). The ladder rung nearest the operating point
    when a ladder exists and carries the statistic; the cheap sample
    otherwise."""
    if ctx.run_ladder:
        view = await ctx.ladder()
        r = view.nearest(ctx.predictions.operating_point_users, predicate)
        if r is not None:
            return "rung", r, r.pop
    return "sample", await ctx.sample(), None


def _decode_seqs(traces) -> float:
    """Mean `requests_running` over the covariates a metrics sampler left on
    the traces — the measured concurrent-decode count H-steady is read
    against. nan when no sampler was attached."""
    vals = [t.covariates.get("requests_running") for t in traces
            if t.covariates and t.covariates.get("requests_running") is not None]
    return sum(vals) / len(vals) if vals else float("nan")


class HTtftMiss(Hypothesis):
    key = "H-ttft-miss"
    title = "a forced miss's mean TTFT at the operating point"
    requires = frozenset()

    def statement(self, cfg, p) -> str:
        return (f"H-ttft-miss: a forced miss's mean TTFT at the "
                f"~{p.operating_point_users:g}-user operating point is "
                f"~{p.ttft_miss_s:g} s (read at the ladder rung nearest that "
                "load).")

    def predict(self, cfg, p) -> Prediction:
        v = p.ttft_miss_s
        return Prediction(value=None if not _fin(v) else v, unit="s")

    async def measure(self, ctx) -> Measurement:
        src, obj, at = await _reading(
            ctx, lambda r: r.n_miss > 0 and math.isfinite(r.ttft_miss_mean))
        v = obj.ttft_miss_mean
        if not _fin(v):
            return Measurement(text="not separable",
                               data={"source": src,
                                     "reason": "no forced-miss samples"})
        where = f" @ {at}u" if at is not None else " @ prevailing load"
        n_miss = getattr(obj, "n_miss", None)
        return Measurement(value=v, unit="s", text=f"{v:.2f}s{where}",
                           data={"source": src, "at_users": at,
                                 "n_miss": n_miss,
                                 "ttft_miss_p50": obj.ttft_miss_p50})

    def verdict(self, pred: Prediction, m: Measurement) -> Verdict:
        v = ratio_verdict(m.value, pred.value, "forced-miss")
        if m.data.get("source") == "sample":
            return Verdict(v.status, v.text + " at the endpoint's prevailing "
                                              "load, not a generated operating point")
        at = m.data.get("at_users")
        return Verdict(v.status, v.text + f" at the ~{at}-user operating point "
                                          "(model quotes order-of-magnitude bounds)")


class HSteady(Hypothesis):
    key = "H-steady"
    title = "the decode batch the load produces, and its per-user speed"
    requires = frozenset()

    def statement(self, cfg, p) -> str:
        if p.steady_decode_tok_s is None:
            return ("H-steady: no steady decode point for this configuration "
                    "(prefill duty >= 100%, or the demand is off the sampled "
                    "axis) — nothing to test.")
        return (f"H-steady: at the ~{p.operating_point_users:g}-user operating "
                f"point the decode batch holds ~{p.steady_decode_seqs:g} "
                f"sequences at ~{p.steady_decode_tok_s:g} tok/s each — NOT the "
                "whole warm pool at the stress figure. Read against the "
                "measured concurrent-decode count, not the population.")

    def predict(self, cfg, p) -> Prediction:
        if p.steady_decode_tok_s is None:
            return Prediction(value=None, text="-")
        return Prediction(value=p.steady_decode_tok_s, unit=" tok/s",
                          text=f"{p.steady_decode_seqs:g} seqs @ "
                               f"{p.steady_decode_tok_s:g} tok/s")

    async def measure(self, ctx) -> Measurement:
        src, obj, at = await _reading(
            ctx, lambda r: math.isfinite(r.decode_p50))
        seqs = _decode_seqs(obj.traces)
        v = obj.decode_p50
        text = "-" if not _fin(v) else f"{v:.0f} tok/s"
        if _fin(seqs):
            text = f"{seqs:.2f} seqs @ {text}"
        if at is not None:
            text += f" @ {at}u"
        return Measurement(value=v if _fin(v) else None, unit=" tok/s",
                           text=text,
                           data={"source": src, "at_users": at,
                                 "decode_p50": v, "seqs": seqs,
                                 "predicted_seqs":
                                     ctx.predictions.steady_decode_seqs,
                                 "server": obj.server})

    def verdict(self, pred: Prediction, m: Measurement) -> Verdict:
        if pred.value is None:
            return Verdict(NOT_ESTABLISHED,
                           "no steady decode point for this configuration")
        v = ratio_verdict(m.value, pred.value, "decode")
        seqs = m.data.get("seqs")
        pred_seqs = m.data.get("predicted_seqs")
        if not _fin(seqs):
            return Verdict(v.status, v.text + " on per-user decode; the "
                                              "concurrent-decode count needs "
                                              "/metrics and is not established")
        if pred_seqs is None:
            return Verdict(v.status, v.text + f" on per-user decode, against a "
                                              f"measured {seqs:.2f}-sequence "
                                              "decode batch")
        vs = ratio_verdict(seqs, pred_seqs, "decode-batch")
        out = _weaker(v, vs)
        return Verdict(out.status,
                       f"decode {v.text}; batch {vs.text} "
                       f"({seqs:.2f} measured sequences)")


class HItlSpike(Hypothesis):
    key = "H-itl-spike"
    title = "the worst inter-token freeze behind one prefill chunk"
    requires = frozenset()

    def statement(self, cfg, p) -> str:
        if p.itl_worst_freeze_ms is None:
            return ("H-itl-spike: no steady decode point for this "
                    "configuration — no freeze prediction to test.")
        cap = cfg.deployment.max_model_len
        return (f"H-itl-spike: the worst freeze behind one chunk of a "
                f"{cap / 1000:.0f}k cold re-prefill is "
                f"~{p.itl_worst_freeze_ms:g} ms "
                f"[{p.itl_freeze_lo_ms:g}-{p.itl_freeze_hi_ms:g}, the MFU "
                f"35-55% bracket] at max_num_batched_tokens="
                f"{cfg.deployment.max_num_batched_tokens:,}. The spike "
                "MAGNITUDE scales ~inversely with the unvalidated MFU, but "
                "the RATIO between two chunk settings does not — quote the "
                "ratio, not the milliseconds. Read it on the INTER-TOKEN GAPS "
                "table, never on decode p50 (a mean over the stream is nearly "
                "blind to a freeze); make sure --freeze-threshold-ms sits "
                "below the predicted freeze.")

    def predict(self, cfg, p) -> Prediction:
        if p.itl_worst_freeze_ms is None:
            return Prediction(value=None, text="-")
        return Prediction(value=p.itl_worst_freeze_ms, lo=p.itl_freeze_lo_ms,
                          hi=p.itl_freeze_hi_ms, unit=" ms")

    async def measure(self, ctx) -> Measurement:
        # the burst probe is the RELIABLE place to read a spike: a controlled
        # cold-prefill event against a small standing load, where client
        # event-loop contention cannot manufacture gaps the way it can under
        # a high ladder rung
        if ctx.burst_n and ctx.exclusive:
            b = await ctx.burst()
            if b.standing_n:
                return Measurement(
                    value=b.standing_worst_max_ms, unit=" ms",
                    text=f"{b.standing_worst_max_ms:.0f} ms worst "
                         f"({b.standing_n} in flight)",
                    data={"source": "burst",
                          "worst_p50_ms": b.standing_worst_p50_ms,
                          "normal_ms": b.standing_itl_p50_ms,
                          "floor_ms": b.standing_floor_ms,
                          "freeze_per_ktok": b.standing_freeze_per_ktok,
                          "freeze_ladder": b.standing_freeze_ladder})
        src, obj, at = await _reading(ctx, lambda r: math.isfinite(r.itl_max_ms))
        worst = getattr(obj, "itl_worst_p95_ms", None)
        if worst is None or not _fin(worst):
            worst = getattr(obj, "itl_worst_max_ms", float("nan"))
        if not _fin(worst):
            return Measurement(text="not separable",
                               data={"source": src,
                                     "reason": "no inter-token gaps measured"})
        where = f" @ {at}u" if at is not None else ""
        return Measurement(
            value=worst, unit=" ms", text=f"{worst:.0f} ms{where}",
            data={"source": src, "at_users": at,
                  "worst_p50_ms": obj.itl_worst_p50_ms,
                  "normal_ms": obj.itl_p50_ms, "floor_ms": obj.itl_floor_ms,
                  "freeze_per_ktok": obj.freeze_per_ktok,
                  "chunk_tok_ratio": obj.chunk_tok_ratio})

    def verdict(self, pred: Prediction, m: Measurement) -> Verdict:
        caveat = ("quote the RATIO between chunk settings, not the "
                  "milliseconds — the magnitude rides on the unvalidated MFU")
        if pred.value is None:
            return Verdict(NOT_ESTABLISHED,
                           "no steady decode point — no freeze prediction")
        if m.value is None or not _fin(m.value):
            return Verdict(NOT_ESTABLISHED, "no inter-token gaps measured")
        floor, normal = m.data.get("floor_ms"), m.data.get("normal_ms")
        if _fin(floor) and _fin(normal) and normal > 0 and floor > 0.10 * normal:
            return Verdict(NOT_ESTABLISHED,
                           f"client floor {floor:.2f} ms is >10% of the normal "
                           f"gap {normal:.2f} ms — the event loop, not the "
                           "server, may be setting these gaps")
        lo = pred.lo if pred.lo is not None else pred.value
        hi = pred.hi if pred.hi is not None else pred.value
        v = m.value
        if lo <= v <= hi:
            return Verdict(SUPPORTED, f"{v:.0f} ms inside the MFU bracket "
                                      f"[{lo:g}-{hi:g}] — {caveat}")
        if 0.75 * lo <= v <= 1.33 * hi:
            return Verdict(NOT_ESTABLISHED,
                           f"{v:.0f} ms within 25-33% of the MFU bracket "
                           f"[{lo:g}-{hi:g}] — {caveat}")
        side = "below" if v < lo else "above"
        return Verdict(REFUTED, f"{v:.0f} ms is {side} the MFU bracket "
                                f"[{lo:g}-{hi:g}] — {caveat}")


class HItlMean(Hypothesis):
    key = "H-itl-mean"
    title = "the normal inter-token gap, between freezes"
    requires = frozenset()

    def statement(self, cfg, p) -> str:
        if p.itl_normal_ms is None:
            return ("H-itl-mean: no steady decode point for this "
                    "configuration — no normal-gap prediction to test.")
        return (f"H-itl-mean: the normal inter-token gap "
                f"(~{p.itl_normal_ms:g} ms) is ~unchanged across chunk "
                "settings — prefill FLOPs telescope; only the per-pass "
                "overhead does not. NOTE a few ms is near the client-side "
                "timing floor — check the 'floor' column before reading "
                "anything into this row.")

    def predict(self, cfg, p) -> Prediction:
        return Prediction(value=p.itl_normal_ms, unit=" ms")

    async def measure(self, ctx) -> Measurement:
        src, obj, at = await _reading(ctx, lambda r: math.isfinite(r.itl_p50_ms))
        v = obj.itl_p50_ms
        if not _fin(v):
            return Measurement(text="not separable",
                               data={"source": src,
                                     "reason": "no inter-token gaps measured"})
        where = f" @ {at}u" if at is not None else ""
        return Measurement(value=v, unit=" ms", text=f"{v:.1f} ms{where}",
                           data={"source": src, "at_users": at,
                                 "floor_ms": obj.itl_floor_ms,
                                 "chunk_tok_ratio": obj.chunk_tok_ratio})

    def verdict(self, pred: Prediction, m: Measurement) -> Verdict:
        if pred.value is None:
            return Verdict(NOT_ESTABLISHED,
                           "no steady decode point — no normal-gap prediction")
        if m.value is None or not _fin(m.value):
            return Verdict(NOT_ESTABLISHED, "no inter-token gaps measured")
        floor = m.data.get("floor_ms")
        if _fin(floor) and m.value > 0 and floor > 0.10 * m.value:
            return Verdict(NOT_ESTABLISHED,
                           f"client floor {floor:.2f} ms is >10% of the "
                           f"measured gap {m.value:.2f} ms — this row is "
                           "client-side scheduling, not server behaviour")
        ratio = m.data.get("chunk_tok_ratio")
        v = ratio_verdict(m.value, pred.value, "inter-token gap")
        if _fin(ratio) and abs(ratio - 1.0) > 0.05:
            return Verdict(NOT_ESTABLISHED,
                           f"{ratio:.2f} tokens per SSE event: the gap is per "
                           "EVENT, so this is not comparable with a per-token "
                           "prediction — compare freezes/ktok instead")
        return v
