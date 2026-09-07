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

from .base import (BURST, BURST_PROBE, EXCLUSIVE, NOT_ESTABLISHED, REFUTED,
                   SUPPORTED, Hypothesis, Measurement, Prediction, Verdict)


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
        return Measurement(
            value=b.last_ttft_s, unit="s",
            text=f"N={b.n}: last {b.last_ttft_s:.2f}s",
            data={"n": b.n, "standing_users": b.standing_users,
                  "n_ok": b.n_ok, "n_err": b.n_err, "drain_s": b.drain_s,
                  "ttft_p50_s": b.ttft_p50_s,
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
