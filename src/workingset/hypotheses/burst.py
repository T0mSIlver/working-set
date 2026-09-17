"""H-burst — the correlated-flush tolerance, B*.

The ladder's independent per-turn misses cannot see this: B* is about
SIMULTANEOUS misses, and it is measured by firing N of them at once from a
steady standing load and timing the drain.

`requires` is {"burst", "exclusive"}: the probe needs `--burst N` AND it
generates the standing load the prediction was priced at. A burst fired into
an endpoint whose load is unknown drains at an unknown duty cycle, which is a
systematically favourable measurement — so a shared run skips this rather
than reporting a number that cannot be compared with B*.
"""
from __future__ import annotations

from ..probe.burst import burst_prompt_tokens
from .base import (BURST, BURST_PROBE, EXCLUSIVE, NOT_ESTABLISHED, REFUTED,
                   SUPPORTED, Hypothesis, Measurement, Prediction, Verdict)


def predicted_drain_seconds(cfg, preds, prompt_tokens) -> float | None:
    """The model's drain for THESE prompts rather than for N mean ones.

    `model.burst_drain_seconds` is B x E[S | miss] / (1 - rho). The burst's
    lengths are a handful of draws from a heavy-tailed log-normal, so their
    service times sum to something that can sit far from B x the mean, and
    "N against B*" alone is then close to a coin flip. This is the same fluid
    drain with the backlog priced on the lengths actually sent:

        T = sum_i S_miss(L_i) / (1 - rho)

    `S_miss` is `model.miss_context_seconds` at the config's chunk and MFU —
    the per-request form of the per-pass-overhead pricing `predict` uses for
    B* — and rho is `Predictions.prefill_duty`, the rate x E[S] of the
    operating point B* was priced at. Nothing is modelled here.

    None when the model has no steady state there (rho >= 1) or no prompt
    answered.
    """
    from .. import model as M

    tokens = [L for L in prompt_tokens if L and L > 0]
    rho = preds.prefill_duty
    if not tokens or rho is None or not rho < 1.0:
        return None
    m, t = cfg.to_model(), cfg.to_topology()
    chunk, mfu = cfg.deployment.max_num_batched_tokens, cfg.calibration.mfu
    backlog = sum(M.miss_context_seconds(m, t, L, chunk, mfu_anchor=mfu)
                  for L in tokens)
    return backlog / (1.0 - rho)


class HBurst(Hypothesis):
    key = "H-burst"
    title = "a simultaneous flush of <= B* misses drains inside the budget"
    requires = frozenset({BURST, EXCLUSIVE})
    # the burst probe, and ONLY the burst probe: `exclusive` is the permission
    # to generate the standing load, not an instruction to ladder
    probes = frozenset({BURST_PROBE})

    def statement(self, cfg, p) -> str:
        return (f"H-burst (needs --burst N): at the "
                f"~{p.operating_point_users:g}-user standing load, a "
                f"simultaneous flush of <= {int(p.bstar_misses)} misses "
                f"(B* = {p.bstar_misses:g}) drains inside the "
                f"{cfg.slo.ttft_budget_s:g} s TTFT budget; a larger one "
                "does not.")

    def predict(self, cfg, p) -> Prediction:
        return Prediction(value=p.bstar_misses, unit=" misses")

    async def measure(self, ctx) -> Measurement:
        b = await ctx.burst()
        if b is None:
            return Measurement(text="not measured",
                               data={"reason": "the run ended before the "
                                               "burst probe ran"})
        if b.last_ttft_s is None:
            return Measurement(text="not measured",
                               data={"reason": "no burst request answered",
                                     "n": b.n, "n_err": b.n_err})
        # ADDED next to the count-based test, which is unchanged: what the
        # model says the drain of the prompts actually sent should have been.
        # A replayed record without traces has no lengths and prints neither.
        tokens = burst_prompt_tokens(b.traces)
        drain_pred = predicted_drain_seconds(ctx.cfg, ctx.predictions, tokens)
        text = f"N={b.n}: last {b.last_ttft_s:.2f}s"
        if drain_pred is not None and b.drain_s is not None:
            text += (f"; drain {b.drain_s:.2f}s measured / {drain_pred:.2f}s "
                     f"predicted for these {sum(tokens) / 1e3:.0f}k tokens")
        return Measurement(
            value=b.last_ttft_s, unit="s", text=text,
            data={"n": b.n, "standing_users": b.standing_users,
                  # a record replayed without traces has no lengths to sum,
                  # but the result kept the total
                  "prompt_tokens_total": b.ptok_total or sum(tokens),
                  "prompt_tokens_from_usage": b.ptok_from_usage,
                  "drain_predicted_for_these_tokens_s": drain_pred,
                  "drain_measured_over_predicted":
                      (b.drain_s / drain_pred
                       if drain_pred and b.drain_s is not None else None),
                  "n_ok": b.n_ok, "n_err": b.n_err, "drain_s": b.drain_s,
                  "ttft_p50_s": b.ttft_p50_s,
                  # 0 on a clean burst; anything else put an establishing
                  # prefill in the queue ahead of it (see probe.burst)
                  "n_establishing_at_fire": b.n_establishing_at_fire,
                  "establish_wait_s": b.establish_wait_s,
                  "ttft_budget_s": ctx.cfg.slo.ttft_budget_s})

    def verdict(self, pred: Prediction, m: Measurement) -> Verdict:
        """The harness's test: B* is a THRESHOLD, so the falsifiable claim is
        that "burst <= B*" and "drained inside the budget" agree."""
        bstar = pred.value
        if m.value is None:
            return Verdict(NOT_ESTABLISHED,
                           "run with --burst N --exclusive to probe the "
                           "correlated-flush tolerance")
        n, budget = m.data["n"], m.data["ttft_budget_s"]
        # DEVIATION from the retired standalone harness, which scored the max
        # TTFT over the requests that ANSWERED. B* is about a flush of N: with
        # a failure among them, "last first-token" is the drain of a burst of
        # n_ok, and a burst that half failed could report support because the
        # slow half was excluded from the max. A partial flush is no flush.
        if m.data.get("n_err"):
            return Verdict(NOT_ESTABLISHED,
                           f"{m.data['n_err']} of {n} burst requests failed — "
                           f"the drain measured is a burst of {m.data['n_ok']}, "
                           "not of N")
        within = n <= (bstar or 0)
        met = m.value <= budget
        if within == met:
            return Verdict(SUPPORTED,
                           "burst <= B* drained inside budget" if within
                           else "burst > B* breached budget, as predicted")
        if bstar and 0.75 <= n / bstar <= 1.33:
            return Verdict(NOT_ESTABLISHED,
                           f"N={n} sits within 25-33% of B*={bstar:g} — the "
                           "threshold is not resolved at this burst size")
        return Verdict(REFUTED, "outcome contradicts B*")
