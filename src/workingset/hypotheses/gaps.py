"""The cheap hypotheses: a level, a batch, and two gap distributions.

None of these needs a ladder. Each runs on a handful of requests
(`ProbeOptions.sample_requests` sessions x a few turns each), so they can be
put to an endpoint that is already serving other traffic. Which is also the
limit of what they can then conclude, and this module is strict about it:

  A PLAIN SHARED SAMPLE CAPS EVERY VERDICT AT `not_established`.
      A prediction here is made AT the configured operating point. A sample
      taken against unknown background load measures a different experiment:
      a miss TTFT below the prediction is what a quieter server would give,
      one above it is what a busier server would give, and neither outcome
      discriminates. The measurement is kept, the number is printed, the note
      says where it came from — but the run has not confirmed or refuted
      anything, and `bounded_below` would be just as wrong, since the load is
      unknown in BOTH directions.

  A SHARED RUN THAT MEASURED THE LOAD MAY SCORE, THROUGH THE FIT AND ONLY
  THROUGH IT.
      `workingset.shared` stamps every probe request with the server's own
      (running, waiting) at send time and fits the measured quantity on them
      (plus prompt length, for TTFT). Evaluating that fit at the operating
      point's EXPECTED running/waiting is a reading at the point the
      prediction is about — no longer a different experiment, but an
      extrapolation from the probed load to the predicted one, whose distance
      is stated. Four gates, all of which must pass, or the row stays
      `not_established` and says which one failed:

        1. the covariates exist at all (a run without --metrics-url has none,
           so nothing changes for it — the old cap holds exactly)
        2. `n` clears three observations per fitted coefficient
        3. the design matrix is not rank-deficient and its condition number
           is under 1e8 — the load and the prompt length must have varied
           INDEPENDENTLY, or the coefficients are not separable
        4. the operating point sits no more than `--max-extrapolation`
           observed standard deviations outside the probed range of every
           regressor

      The raw prevailing-load number is still recorded; what the verdict is
      computed from is the fitted value at the operating point, and the row
      prints the extrapolation distance next to it.

When a ladder IS being run (an exclusive run that also selected a ceiling
hypothesis), they read the rung nearest the operating point instead, exactly
as scripts/validate_deployment.py does — and if no rung carries the statistic
they report "not separable" rather than opening a live sample against an
endpoint the ladder has just finished draining.

Two scoring rules here are NOT in the harness, which only narrated these
rows; both are deliberately conservative:

  H-itl-mean promotes the harness's two WARNINGS to verdicts. A client floor
      above 10% of the measured gap, or an SSE event carrying materially more
      than one token, means the row is not measuring what the prediction is
      about — client scheduling in the first case, a per-event quantity
      against a per-token prediction in the second. The harness printed a
      warning next to a number; here that is `not_established`.
  H-steady is scored on the FREEZE-EXCLUDED decode rate against a measured
      decode-batch size, because that is what steady_decode_point predicts
      (clean decode between prefill spikes, at a batch). Both halves must be
      available or the verdict caps at `not_established`: without /metrics
      there is no batch size, and a batch size is half the claim.
"""
from __future__ import annotations

import math

from .base import (BURST_PROBE, LADDER, NOT_ESTABLISHED, REFUTED, SAMPLE,
                   SHARED, SUPPORTED, Hypothesis, Measurement, Prediction,
                   Verdict, ratio_verdict)

_WEAKEST = {SUPPORTED: 0, NOT_ESTABLISHED: 1, REFUTED: 2}

# what a shared-mode row says about itself, on every hypothesis in this module
PREVAILING = ("at the endpoint's prevailing load, not a generated operating "
              "point")

# the sources whose readings were taken under load this run did not set
_UNCONTROLLED = ("sample", "shared")


def _weaker(a: Verdict, b: Verdict) -> Verdict:
    return a if _WEAKEST[a.status] >= _WEAKEST[b.status] else b


def _cap(v: Verdict, source: str, note: str = PREVAILING) -> Verdict:
    """Shared-mode cap: keep the text, refuse the claim."""
    if source not in _UNCONTROLLED:
        return v
    if v.status == NOT_ESTABLISHED:
        return Verdict(NOT_ESTABLISHED, f"{v.text} — {note}")
    return Verdict(NOT_ESTABLISHED,
                   f"{v.text}, but measured {note} — not scored")


# ============================================================================
# shared mode: the covariate fit is the ONLY route to a verdict
# ============================================================================
async def _fit(ctx, which: str) -> dict | None:
    """The shared probe's reading of `which` at the configured operating
    point, or None when this run has no shared probe.

    The dict is `workingset.shared.SharedResult.reading`: it always carries
    `available` and, when it is False, the `reason` — which of the four gates
    the run failed and what would fix it.
    """
    if SHARED not in ctx.probes:
        return None
    res = await ctx.shared()
    return None if res is None else res.reading(which)


def _fit_note(r: dict) -> str:
    at = r.get("at") or {}
    fit = r.get("fit") or {}
    return (f"read off the covariate fit at the configured operating point "
            f"(running {_g(at.get('running'))}, waiting "
            f"{_g(at.get('waiting'))}) over n={r.get('n', 0)} stamped "
            f"requests, residual sd {_g(fit.get('residual_std'))}, "
            f"extrapolation {r.get('extrapolation', float('nan')):.2f} sd "
            f"(max {r.get('max_extrapolation', float('nan')):g})")


def _g(x) -> str:
    return "-" if x is None or not _fin(x) else f"{x:.3g}"


def _steady_pair(v: Verdict, pred_seqs) -> tuple:
    """H-steady's shared-mode text. The speed half is the fitted decode rate
    against the prediction; the BATCH half is not a separately measured mean
    here — it is the extrapolation gate, which refuses the whole row unless
    the endpoint was observed at ~the predicted concurrency."""
    seqs = "-" if pred_seqs is None else f"{pred_seqs:g}"
    return (v.status,
            f"decode {v.text}, evaluated at the predicted decode batch "
            f"(~{seqs} sequences); the batch half of the claim is carried by "
            "the extrapolation gate, not by a measured batch mean")


def _fit_verdict(m: Measurement, make) -> Verdict:
    """Shared mode's verdict rule: `make()` is the ordinary comparison, and it
    is allowed to stand ONLY when the fit behind the measurement passed every
    gate. Otherwise the row is `not_established` carrying the gate's own
    reason, which is the same honesty the plain cap gave — with a diagnosis
    attached instead of a shrug."""
    r = m.data.get("fit")
    if not r:
        return Verdict(NOT_ESTABLISHED,
                       f"measured {PREVAILING}, and this run fitted no "
                       "covariates to correct for it — pass --metrics-url")
    if not r.get("available"):
        return Verdict(NOT_ESTABLISHED,
                       f"measured {PREVAILING}; not corrected to it because "
                       f"{r.get('reason', 'the covariate fit was refused')}")
    v = make()
    return Verdict(v.status, f"{v.text} — {_fit_note(r)}")


def _fin(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


async def _reading(ctx, predicate):
    """(source, obj, at_users), driven by the PLAN's probe set.

    With a ladder in the plan: the rung nearest the operating point that
    carries the statistic — the harness's rule, because ttft_miss rises
    steeply with load and sampling it at the predicted LIMIT instead
    systematically failed a correct model. If no rung carries it, that is
    "not separable", the harness's answer; opening a live sample instead
    would measure an endpoint the ladder had just stopped loading and score
    it against an operating-point prediction.
    """
    if LADDER in ctx.probes:
        view = await ctx.ladder()
        r = view.nearest(ctx.predictions.operating_point_users, predicate)
        if r is not None:
            return "rung", r, r.pop
        return "none", None, None
    if SHARED in ctx.probes:
        # the shared probe produces the same Sample the plain one does, so
        # every statistic below reads identically; what it adds is the fit
        return "shared", await ctx.shared_sample(), None
    if SAMPLE in ctx.probes:
        return "sample", await ctx.sample(), None
    return "none", None, None


def _not_separable(src: str, reason: str) -> Measurement:
    return Measurement(text="not separable",
                       data={"source": src, "reason": reason})


class HTtftMiss(Hypothesis):
    key = "H-ttft-miss"
    title = "a forced miss's mean TTFT at the operating point"
    requires = frozenset()

    def conditional_probes(self, planned) -> frozenset:
        return frozenset({LADDER}) if LADDER in planned else frozenset({SAMPLE})

    def statement(self, cfg, p) -> str:
        return (f"H-ttft-miss: a forced miss's mean TTFT at the "
                f"~{p.operating_point_users:g}-user operating point is "
                f"~{p.ttft_miss_s:g} s.")

    def statement_for(self, cfg, p, probes) -> str:
        """The harness reads this at a ladder rung. Say so only when there is
        a ladder to read it at."""
        base = self.statement(cfg, p)
        if LADDER in probes:
            return base[:-1] + " (read at the ladder rung nearest that load)."
        if SHARED in probes:
            return base[:-1] + (" — this run has no ladder, so it is measured "
                                "at the endpoint's prevailing load and "
                                "CORRECTED to that point by the covariate fit "
                                "on (prompt length, running, waiting), with "
                                "the extrapolation distance stated.")
        return base[:-1] + (" — this run has no ladder, so it is sampled at "
                            "the endpoint's prevailing load and cannot be "
                            "scored against that point.")

    def predict(self, cfg, p) -> Prediction:
        v = p.ttft_miss_s
        return Prediction(value=None if not _fin(v) else v, unit="s")

    async def measure(self, ctx) -> Measurement:
        src, obj, at = await _reading(
            ctx, lambda r: r.n_miss > 0 and math.isfinite(r.ttft_miss_mean))
        if obj is None:
            return _not_separable(src, "no forced-miss samples")
        v = obj.ttft_miss_mean
        if not _fin(v):
            return _not_separable(src, "no forced-miss samples")
        data = {"source": src, "at_users": at,
                "n_miss": getattr(obj, "n_miss", None),
                "ttft_miss_p50": obj.ttft_miss_p50,
                "prevailing_mean_s": v}
        if src == "shared":
            # the SCORED value is the fit evaluated at the operating point;
            # the raw prevailing-load mean stays in `data`, never in the
            # comparison, because it answers a different question
            r = await _fit(ctx, "ttft_miss")
            data["fit"] = r
            if r is not None and r.get("available"):
                fv = r["value"]
                return Measurement(value=fv, unit="s",
                                   text=f"{fv:.2f}s @ op (fit)", data=data)
            return Measurement(value=v, unit="s",
                               text=f"{v:.2f}s @ prevailing", data=data)
        where = f" @ {at}u" if at is not None else " @ prevailing load"
        return Measurement(value=v, unit="s", text=f"{v:.2f}s{where}",
                           data=data)

    def verdict(self, pred: Prediction, m: Measurement) -> Verdict:
        src = m.data.get("source")
        if m.value is None:
            return Verdict(NOT_ESTABLISHED,
                           m.data.get("reason", "no forced-miss samples"))
        if src == "shared":
            return _fit_verdict(
                m, lambda: ratio_verdict(m.value, pred.value, "forced-miss"))
        v = ratio_verdict(m.value, pred.value, "forced-miss")
        if src == "sample":
            return _cap(v, src)
        at = m.data.get("at_users")
        return Verdict(v.status, v.text + f" at the ~{at}-user operating point "
                                          "(model quotes order-of-magnitude bounds)")


class HSteady(Hypothesis):
    key = "H-steady"
    title = "the decode batch the load produces, and its per-user speed"
    requires = frozenset()

    def conditional_probes(self, planned) -> frozenset:
        return frozenset({LADDER}) if LADDER in planned else frozenset({SAMPLE})

    def statement(self, cfg, p) -> str:
        if p.steady_decode_tok_s is None:
            return ("H-steady: no steady decode point for this configuration "
                    "(prefill duty >= 100%, or the demand is off the sampled "
                    "axis) — nothing to test.")
        return (f"H-steady: at the ~{p.operating_point_users:g}-user operating "
                f"point the decode batch holds ~{p.steady_decode_seqs:g} "
                f"sequences at ~{p.steady_decode_tok_s:g} tok/s each — NOT the "
                "whole warm pool at the stress figure. Read against the "
                "measured concurrent-decode count and the freeze-excluded "
                "decode rate, not the population and not a mean over stalls.")

    def predict(self, cfg, p) -> Prediction:
        if p.steady_decode_tok_s is None:
            return Prediction(value=None, text="-")
        return Prediction(value=p.steady_decode_tok_s, unit=" tok/s",
                          text=f"{p.steady_decode_seqs:g} seqs @ "
                               f"{p.steady_decode_tok_s:g} tok/s")

    async def measure(self, ctx) -> Measurement:
        src, obj, at = await _reading(
            ctx, lambda r: math.isfinite(r.decode_clean_p50))
        if obj is None:
            return _not_separable(src, "no decode samples")
        # clean_decode_p50, not decode_p50: steady_decode_point prices the
        # decode speed BETWEEN prefill spikes, and a mean over the whole
        # stream charges the freezes to decode
        v = obj.decode_clean_p50
        seqs, n_obs = obj.decode_seqs, obj.n_seqs_obs
        data = {"source": src, "at_users": at, "decode_clean_p50": v,
                "decode_p50_with_freezes": obj.decode_p50,
                "seqs": seqs, "n_seqs_obs": n_obs,
                "predicted_seqs": ctx.predictions.steady_decode_seqs,
                "server": obj.server}
        if src == "shared":
            # BOTH halves of the claim come from the fit here. The speed is
            # the fitted decode rate evaluated at the operating point's
            # expected `running`; the BATCH half is carried by the
            # extrapolation gate, which is what refuses the reading unless
            # the endpoint was actually observed at ~that concurrency. That
            # is a stricter test than the prevailing-load mean the plain
            # sample offers, not a looser one.
            r = await _fit(ctx, "decode")
            data["fit"] = r
            if r is not None and r.get("available"):
                fv = r["value"]
                return Measurement(
                    value=fv, unit=" tok/s",
                    text=f"{fv:.0f} tok/s @ "
                         f"{_g((r.get('at') or {}).get('running'))} running",
                    data=data)
            text = "-" if not _fin(v) else f"{v:.0f} tok/s @ prevailing"
            return Measurement(value=v if _fin(v) else None, unit=" tok/s",
                               text=text, data=data)
        text = "-" if not _fin(v) else f"{v:.0f} tok/s"
        if _fin(seqs):
            text = f"{seqs:.2f} seqs @ {text}"
        if at is not None:
            text += f" @ {at}u"
        return Measurement(value=v if _fin(v) else None, unit=" tok/s",
                           text=text, data=data)

    def verdict(self, pred: Prediction, m: Measurement) -> Verdict:
        if pred.value is None:
            return Verdict(NOT_ESTABLISHED,
                           "no steady decode point for this configuration")
        if m.value is None:
            return Verdict(NOT_ESTABLISHED,
                           m.data.get("reason", "no decode samples"))
        src = m.data.get("source")
        if src == "shared":
            return _fit_verdict(m, lambda: Verdict(
                *_steady_pair(ratio_verdict(m.value, pred.value, "decode"),
                              m.data.get("predicted_seqs"))))
        v = ratio_verdict(m.value, pred.value, "decode")
        seqs, pred_seqs = m.data.get("seqs"), m.data.get("predicted_seqs")
        # a batch size is HALF the claim: without one, the run has tested the
        # speed and not the "NOT the whole warm pool" part, so it caps out
        if not _fin(seqs) or pred_seqs is None:
            return _cap(Verdict(NOT_ESTABLISHED,
                                v.text + " on the freeze-excluded decode rate; "
                                "the concurrent-decode count needs /metrics "
                                "and is not established"), src)
        vs = ratio_verdict(seqs, pred_seqs, "decode-batch")
        out = _weaker(v, vs)
        return _cap(Verdict(out.status,
                            f"decode {v.text}; batch {vs.text} "
                            f"({seqs:.2f} measured sequences over "
                            f"{m.data.get('n_seqs_obs', 0)} snapshots)"), src)


class HItlSpike(Hypothesis):
    key = "H-itl-spike"
    title = "the worst inter-token freeze behind one prefill chunk"
    requires = frozenset()

    def conditional_probes(self, planned) -> frozenset:
        # the burst is the RELIABLE place to read a spike — a controlled cold
        # prefill against a small standing load, where client event-loop
        # contention cannot manufacture gaps the way it can under a high rung.
        # It is used only when the run is ALREADY running one; this hypothesis
        # never adds a 64-user standing load to a plan that printed "sample".
        if BURST_PROBE in planned:
            return frozenset({BURST_PROBE})
        return frozenset({LADDER}) if LADDER in planned else frozenset({SAMPLE})

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
                "ratio, not the milliseconds. Scored only on responses that "
                "were decoding while a cold prefill at least half the "
                "context cap deep was running; make sure "
                "--freeze-threshold-ms sits below the predicted freeze.")

    def predict(self, cfg, p) -> Prediction:
        if p.itl_worst_freeze_ms is None:
            return Prediction(value=None, text="-")
        return Prediction(value=p.itl_worst_freeze_ms, lo=p.itl_freeze_lo_ms,
                          hi=p.itl_freeze_hi_ms, unit=" ms")

    async def measure(self, ctx) -> Measurement:
        if BURST_PROBE in ctx.probes:
            b = await ctx.burst()
            if b is None:
                return _not_separable("burst", "the run ended before the "
                                               "burst probe ran")
            return self._from_spike("burst", b.spike, None)
        src, obj, at = await _reading(ctx, lambda r: bool(r.spike.get("n")))
        if obj is None:
            # nothing witnessed a spike; read the nearest probe that at least
            # looked, so the row can say WHY the event was not observed
            src, obj, at = await _reading(ctx, lambda r: bool(r.spike))
        if obj is None:
            return _not_separable(
                src, "no probe recorded inter-token gaps to look for a spike "
                     "in")
        m = self._from_spike(src, obj.spike, at)
        if src != "shared":
            return m
        # the freeze this predicts is what ONE prefill chunk does to the
        # decode batch AT the operating point's concurrency. On a shared
        # endpoint the load-only fit is what says whether the endpoint was
        # ever carrying that concurrency, so its gate governs this row too —
        # on top of the spike evidence having to exist at all.
        return Measurement(value=m.value, lo=m.lo, hi=m.hi, unit=m.unit,
                           text=m.text,
                           data={**m.data, "fit": await _fit(ctx, "itl")})

    @staticmethod
    def _from_spike(src: str, spike: dict, at) -> Measurement:
        """One statistic across every probe: the p95 of the per-response worst
        gap, over the responses that were streaming while a deep cold prefill
        ran. The max-of-maxes stays in `data` — the ported comment calls it
        sample-size biased and a footnote, so it is not what gets scored."""
        data = {"source": src, "at_users": at, **{f"spike_{k}": v
                                                  for k, v in spike.items()}}
        if not spike or not spike.get("n"):
            deep = spike.get("n_deep", 0) if spike else 0
            deepest = spike.get("deepest_ptok", 0) if spike else 0
            need = spike.get("min_ptok", 0) if spike else 0
            reason = (
                f"{deep} cold prefill(s) at least {need:,.0f} tokens deep ran, "
                "but nothing was decoding through one"
                if deep else
                f"no cold prefill reached {need:,.0f} tokens (deepest seen "
                f"{deepest:,.0f}) — the predicted freeze is the LAST chunk of "
                "a full-context re-prefill, and that event did not occur")
            return Measurement(text="not observed",
                               data={**data, "reason": reason})
        v = spike["worst_p95_ms"]
        where = f" @ {at}u" if at is not None else ""
        return Measurement(value=v, unit=" ms",
                           text=f"{v:.0f} ms p95{where}", data=data)

    def verdict(self, pred: Prediction, m: Measurement) -> Verdict:
        caveat = ("quote the RATIO between chunk settings, not the "
                  "milliseconds — the magnitude rides on the unvalidated MFU")
        if pred.value is None:
            return Verdict(NOT_ESTABLISHED,
                           "no steady decode point — no freeze prediction")
        src = m.data.get("source")
        if m.value is None or not _fin(m.value):
            # NOT a refutation: not encountering the event cannot bound its
            # duration
            return Verdict(NOT_ESTABLISHED,
                           m.data.get("reason", "no inter-token gaps measured"))
        floor = m.data.get("spike_floor_ms")
        normal = m.data.get("spike_normal_ms")
        if _fin(floor) and _fin(normal) and normal > 0 and floor > 0.10 * normal:
            return _cap(Verdict(NOT_ESTABLISHED,
                                f"client floor {floor:.2f} ms is >10% of the "
                                f"normal gap {normal:.2f} ms — the event loop, "
                                "not the server, may be setting these gaps"),
                        src)
        lo = pred.lo if pred.lo is not None else pred.value
        hi = pred.hi if pred.hi is not None else pred.value
        v = m.value
        n = m.data.get("spike_n", 0)
        if lo <= v <= hi:
            out = Verdict(SUPPORTED, f"{v:.0f} ms p95 over {n} witnesses is "
                                     f"inside the MFU bracket [{lo:g}-{hi:g}] "
                                     f"— {caveat}")
        elif 0.75 * lo <= v <= 1.33 * hi:
            out = Verdict(NOT_ESTABLISHED,
                          f"{v:.0f} ms p95 within 25-33% of the MFU bracket "
                          f"[{lo:g}-{hi:g}] — {caveat}")
        else:
            side = "below" if v < lo else "above"
            out = Verdict(REFUTED, f"{v:.0f} ms p95 over {n} witnesses is "
                                   f"{side} the MFU bracket [{lo:g}-{hi:g}] "
                                   f"— {caveat}")
        if src == "shared":
            return _fit_verdict(m, lambda: out)
        return _cap(out, src)


class HItlMean(Hypothesis):
    key = "H-itl-mean"
    title = "the normal inter-token gap, between freezes"
    requires = frozenset()

    def conditional_probes(self, planned) -> frozenset:
        return frozenset({LADDER}) if LADDER in planned else frozenset({SAMPLE})

    def statement(self, cfg, p) -> str:
        if p.itl_normal_ms is None:
            return ("H-itl-mean: no steady decode point for this "
                    "configuration — no normal-gap prediction to test.")
        return (f"H-itl-mean: the normal inter-token gap "
                f"(~{p.itl_normal_ms:g} ms) is ~unchanged across chunk "
                "settings — prefill FLOPs telescope; only the per-pass "
                "overhead does not. NOTE a few ms is near the client-side "
                "timing floor, and the gap is per SSE EVENT — either one "
                "out of range makes this row unreadable, not merely noisy.")

    def predict(self, cfg, p) -> Prediction:
        return Prediction(value=p.itl_normal_ms, unit=" ms")

    async def measure(self, ctx) -> Measurement:
        src, obj, at = await _reading(ctx, lambda r: math.isfinite(r.itl_p50_ms))
        if obj is None:
            return _not_separable(src, "no inter-token gaps measured")
        v = obj.itl_p50_ms
        if not _fin(v):
            return _not_separable(src, "no inter-token gaps measured")
        data = {"source": src, "at_users": at, "floor_ms": obj.itl_floor_ms,
                "chunk_tok_ratio": obj.chunk_tok_ratio,
                "prevailing_p50_ms": v}
        if src == "shared":
            r = await _fit(ctx, "itl")
            data["fit"] = r
            if r is not None and r.get("available"):
                fv = r["value"]
                return Measurement(value=fv, unit=" ms",
                                   text=f"{fv:.1f} ms @ op (fit)", data=data)
            return Measurement(value=v, unit=" ms",
                               text=f"{v:.1f} ms @ prevailing", data=data)
        where = f" @ {at}u" if at is not None else ""
        return Measurement(value=v, unit=" ms", text=f"{v:.1f} ms{where}",
                           data=data)

    def verdict(self, pred: Prediction, m: Measurement) -> Verdict:
        if pred.value is None:
            return Verdict(NOT_ESTABLISHED,
                           "no steady decode point — no normal-gap prediction")
        if m.value is None or not _fin(m.value):
            return Verdict(NOT_ESTABLISHED,
                           m.data.get("reason", "no inter-token gaps measured"))
        src = m.data.get("source")
        floor = m.data.get("floor_ms")
        if _fin(floor) and m.value > 0 and floor > 0.10 * m.value:
            return _cap(Verdict(NOT_ESTABLISHED,
                                f"client floor {floor:.2f} ms is >10% of the "
                                f"measured gap {m.value:.2f} ms — this row is "
                                "client-side scheduling, not server behaviour"),
                        src)
        ratio = m.data.get("chunk_tok_ratio")
        if _fin(ratio) and abs(ratio - 1.0) > 0.05:
            return _cap(Verdict(NOT_ESTABLISHED,
                                f"{ratio:.2f} tokens per SSE event: the gap is "
                                "per EVENT, so this is not comparable with a "
                                "per-token prediction — compare freezes/ktok "
                                "instead"), src)
        if src == "shared":
            return _fit_verdict(m, lambda: ratio_verdict(
                m.value, pred.value, "inter-token gap"))
        return _cap(ratio_verdict(m.value, pred.value, "inter-token gap"), src)
