"""Shared-endpoint mode — measuring on an endpoint you do not own.

`ws test` without `--exclusive` puts its probes to a server that is already
serving somebody else. Two things follow, and this module is the answer to
both.

SAFETY. Somebody else's SLO is on the line, so the probe carries rails
(`ProbeBudget`, enforced by `ProbeGovernor`): a cap on how many of OUR
requests may be in flight, an abort on the server's own queue depth and KV
occupancy, a total prompt-token budget for the run, and a periodic 1-token
CANARY whose TTFT drift is the contention signal when no `/metrics` is
reachable. Any rail that trips raises `BudgetAbort`; the run then writes a
record carrying the reason and exits nonzero. Without `--exclusive` the
defaults are the conservative ones in `ProbeBudget.conservative()`, and
`--dry-run` prints them.

HONESTY. The other traffic is a COVARIATE, not noise. Before this module the
cheap hypotheses capped every shared-mode verdict at `not_established`, and
correctly so: a miss TTFT measured under unknown load is not a measurement at
the configured operating point, and it is biased in an unknown DIRECTION, so
it cannot even bound the prediction. What changes that is measuring the load
instead of ignoring it. Every probe request is stamped at SEND time with

    (L, running, waiting, kv_usage)

  L         this request's prompt length, KILOTOKENS (client intent; the
            server's `usage.prompt_tokens` readback is recorded alongside)
  running   the server's `requests_running` gauge, REQUESTS
  waiting   the server's `requests_waiting` gauge, REQUESTS
  kv_usage  KV pool occupancy, FRACTION in [0, 1] (recorded, not a regressor)

and the run then fits, by ordinary least squares,

    TTFT [s] = c0 + c1*L + c2*L^2 + c3*running + c4*waiting

which is the shape `model.prefill_ttft_seconds` predicts: a request's own
prefill is ~quadratic in its length (attention FLOPs against the prior), and
the M/G/1 FCFS wait it queues behind is carried by how much work is already
in the server — `running` and `waiting`. Evaluating the fit at the CONFIGURED
operating point's expected (L, running, waiting) turns a shared sample into a
comparison against the prediction, WITH a stated extrapolation distance:

    extrapolation distance = max over regressors of how far the evaluation
        point lies OUTSIDE the observed range of that regressor, in units of
        that regressor's observed standard deviation. Zero inside the range.

A verdict is allowed through only when the fit is well conditioned, `n` is
large enough, and that distance is at or below `--max-extrapolation`
(default 1.0 sd). Otherwise the row stays `not_established` and says which of
the four gates it failed. A plain sample with no covariates attached — no
`--metrics-url` — cannot fit anything and keeps the old cap exactly.

Units, everywhere in this module: seconds for TTFT and durations,
MILLISECONDS for inter-token gaps, tokens/second for decode rates,
KILOTOKENS for `L`, requests for `running`/`waiting`, fraction for
`kv_usage`. Every instant handed to a metrics sampler comes from
`probe.request.sampler_now`, which reads the SAMPLER's own clock;
`RequestTrace.t_send` stays `time.monotonic()` because the probe layer does
span arithmetic with it, and the two bases differ by the unix epoch.

No modelling happens here. Every predicted quantity is fetched from
`workingset.predict` / `workingset.model`.
"""
from __future__ import annotations

import asyncio
import math
import random
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field

import numpy as np

from .probe.population import Sample, eval_sample
from .probe.request import (RequestTrace, _covariates, sampler_now,
                           send_request, window_dict)
from .probe.session import make_text
from .probe.stats import pct

__all__ = [
    "BudgetAbort", "CovariateFit", "OperatingPoint", "ProbeBudget",
    "ProbeGovernor", "SharedOptions", "SharedResult", "cross_check",
    "build_fits", "covariate_rows", "fit_covariates", "ladder_model_curve",
    "natural_ladder", "operating_point_covariates", "plan_lines", "run_shared",
]


# ============================================================================
# knobs
# ============================================================================
DEFAULT_SHARED_LENGTHS = "0.1,0.25,0.5,0.75,1.0"

# A regressor needs observations before it means anything. Three per fitted
# coefficient is the floor used below: at 5 coefficients that is 15 forced
# misses. The default ladder (5 lengths x 4 rounds) sends 20, so a handful of
# failed requests does not silently turn the whole run into "n too small".
MIN_OBS_PER_COEF = 3
MIN_OBS_FLOOR = 8

# Collinearity gates. BOTH are computed on a design whose non-constant columns
# have been standardised, because the RAW condition number is dominated by
# column SCALING and says almost nothing about collinearity: the default
# ladder (5 lengths, 18k-180k tokens) puts a constant column of 1 next to an
# L^2 column spanning 324-32400, which is a raw cond of ~6e4 on a design whose
# load regressors are independent (VIF 1.4). Gating the raw number at 1e4
# would refuse every legitimate quadratic fit this probe can produce.
#
#   MAX_VIF   per-regressor variance inflation factor, 1/(1 - R^2_j) from
#             regressing column j on the others. The textbook 10 separates
#             this probe's healthy designs (VIF 1.1-1.4) from a genuinely
#             collinear load pair (VIF ~3000) by three orders of magnitude.
#   MAX_SCALED_CONDITION  the whole-design backstop, on the standardised
#             matrix: ~1.9 healthy, ~110 when running and waiting move
#             together.
#
# The quadratic term is CENTRED for the same reason (see `_design`): L and L^2
# are structurally correlated (VIF ~20) whatever the data quality, and that is
# a parameterisation artefact, not a finding about the endpoint.
MAX_VIF = 10.0
MAX_SCALED_CONDITION = 1e4

# regressors measured in REQUESTS, where an extrapolation has an absolute
# meaning as well as a relative one (see `CovariateFit.extrapolation`)
REQUEST_COLUMNS = ("running", "waiting")

# Absolute cap on how far outside the probed load an operating point may sit,
# in REQUESTS, alongside the standard-deviation cap. Two gates because the
# sd one alone is perverse: it is measured in units of the probe's own
# empirical spread, so a NOISIER background buys a WIDER absolute licence to
# extrapolate. The queue is also hyperbolic in utilisation -- the P-K wait is
# lambda E[S^2] / (2(1 - rho)) -- so a straight line fitted at low load and
# extended upward UNDERSTATES the wait, and the error grows without bound as
# rho approaches 1. Two requests is about the most a linear reading of that
# curve survives.
MAX_EXTRAPOLATION_REQUESTS = 2.0

# What the linear form does to a hyperbolic truth, stated wherever an upward
# extrapolation is reported rather than left for the reader to derive.
UPWARD_BIAS = ("a straight line fitted below the operating point UNDERSTATES "
               "a queueing delay that is hyperbolic in utilisation, so a "
               "fitted TTFT read above the probed load is biased LOW and this "
               "row errs toward 'the model is pessimistic'")


@dataclass(frozen=True)
class SharedOptions:
    """Shape of the shared probe and the gate its verdicts must pass.

    lengths            comma-separated FRACTIONS of `context_cap_tokens`; one
                       forced miss is sent at each, per round. Spanning the
                       cap is what makes c1/c2 separable from c3/c4.
    rounds             passes over the length ladder
    warm_turns         warm (prefix-hit) turns per round, for the ITL and
                       decode fits
    ladder             run for `duration_s` cycling the length ladder, so the
                       endpoint's OWN load variation is sampled — the
                       "natural ladder" (`--shared-ladder`)
    duration_s         seconds the natural-ladder run lasts
    max_extrapolation  largest extrapolation distance, in observed standard
                       deviations, at which a fitted verdict is still allowed
    max_extrapolation_requests  the same in REQUESTS, so a noisy background
                       cannot buy a wider absolute licence. Both gates bind.
    verdict_sigmas     how many standard errors of the fitted value a verdict
                       must survive (see `_fit_verdict`)
    seed               RNG seed for the probe's synthetic text
    """
    lengths: str = DEFAULT_SHARED_LENGTHS
    rounds: int = 4
    warm_turns: int = 2
    ladder: bool = False
    duration_s: float = 300.0
    max_extrapolation: float = 1.0
    max_extrapolation_requests: float = MAX_EXTRAPOLATION_REQUESTS
    verdict_sigmas: float = 3.0
    seed: int = 0

    def length_fractions(self) -> list[float]:
        out = []
        for part in self.lengths.split(","):
            part = part.strip()
            if not part:
                continue
            f = float(part)
            if not 0 < f <= 1.0:
                raise ValueError(f"--shared-lengths entries are fractions of "
                                 f"the context cap in (0, 1]; got {f!r}")
            out.append(f)
        if not out:
            raise ValueError("--shared-lengths named no lengths")
        return sorted(set(out))

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ProbeBudget:
    """What this run is allowed to do to somebody else's endpoint.

    max_extra_load     never more than this many of OUR requests in flight,
                       canary included (0 = no cap). ENFORCED AS GIVEN: an
                       operator's explicit cap is never raised, so at 1 the
                       canary simply takes its turn in the single slot
                       (`asyncio.Semaphore` is FIFO, so it cannot starve).
    abort_if_waiting   abort the moment the server's `requests_waiting` gauge
                       EXCEEDS this many requests (None = rail off). Needs
                       `--metrics-url`; without one the gauge is unreadable
                       and the rail cannot fire, which `--dry-run` says.
    abort_if_kv_above  abort when KV occupancy exceeds this FRACTION
                       (None = rail off). Also needs `--metrics-url`.
    max_metrics_gaps   abort after this many CONSECUTIVE failures to read the
                       gauges once a sampler has been working (0 = rail off).
                       Fails CLOSED: a sampler that dies mid-run would
                       otherwise disarm the two gauge rails silently, and
                       look exactly like a run that never had metrics.
    max_probe_tokens   total INTENDED prompt tokens the run may send, summed
                       over every request (0 = no cap). Checked before each
                       send, so the cap is never exceeded, only reached.
    canary             fire a 1-token request every `canary_every_s`. Its
                       TTFT is the client-side contention baseline.
    canary_drift       ABORT RULE, stated exactly: let `base` be the p50
                       canary TTFT over the first `canary_baseline_s` of the
                       run and `recent` the p50 over the last
                       `canary_window_s`. Once both hold at least
                       `canary_min_n` samples AND the recent window no longer
                       overlaps the baseline window, abort when
                       `recent > canary_drift * base`.
    """
    max_extra_load: int = 2
    abort_if_waiting: float | None = 0.0
    abort_if_kv_above: float | None = 0.90
    max_metrics_gaps: int = 3
    max_probe_tokens: int = 4_000_000
    canary: bool = True
    canary_every_s: float = 10.0
    canary_baseline_s: float = 60.0
    canary_window_s: float = 60.0
    canary_drift: float = 3.0
    canary_min_n: int = 5
    gauge_poll_s: float = 1.0
    exclusive: bool = False

    @classmethod
    def conservative(cls, **kw) -> "ProbeBudget":
        """The shared-mode default, whatever the field defaults above say.
        Deliberately timid — the operator can raise every one of these, and
        `--dry-run` prints what they are set to."""
        return cls(**kw)

    @classmethod
    def for_exclusive(cls, **kw) -> "ProbeBudget":
        """`--exclusive` owns the endpoint: the rails come off, because the
        queue the ladder is about to build is the measurement, not a
        trespass. The token budget and the canary go with them."""
        base = dict(max_extra_load=0, abort_if_waiting=None,
                    abort_if_kv_above=None, max_metrics_gaps=0,
                    max_probe_tokens=0, canary=False, exclusive=True)
        base.update(kw)
        return cls(**base)

    def __post_init__(self):
        if self.max_extra_load < 0:
            raise ValueError("--max-extra-load must be >= 0 (0 = no cap)")
        if self.max_probe_tokens < 0:
            raise ValueError("--max-probe-tokens must be >= 0 (0 = no cap)")
        if self.canary_drift <= 1.0:
            raise ValueError("--canary-drift must be > 1.0 (it is a ratio "
                             "against the run's own first-minute p50)")

    def describe(self, metrics=None) -> list[str]:
        """The `--dry-run` block. `metrics` is the sampler (or None / False):
        the gauge rails cannot fire without one, and with one its scrape
        interval sets how stale the gauges they read may be."""
        cap = self.max_extra_load or "uncapped"
        if self.exclusive:
            return [f"in flight      : {cap} (exclusive: this run owns the "
                    "endpoint, so the rails are off)",
                    "queue / KV     : not enforced",
                    "prompt tokens  : uncapped",
                    "canary         : off"]
        gauge = ("" if metrics else "  [NOT ENFORCEABLE: no --metrics-url, so "
                                    "the server's gauges are unreadable]")
        tok = (f"{self.max_probe_tokens:,} intended prompt tokens for the "
               "whole run" if self.max_probe_tokens else "uncapped")
        return [
            f"in flight      : at most {cap} of our requests, canary included",
            f"abort waiting  : requests_waiting > "
            f"{'off' if self.abort_if_waiting is None else f'{self.abort_if_waiting:g}'}"
            f"{gauge}",
            f"abort KV       : kv_cache_usage > "
            f"{'off' if self.abort_if_kv_above is None else f'{self.abort_if_kv_above:.0%}'}"
            f"{gauge}",
            f"detection lag  : {self.detection_lag(metrics)}",
            "metrics loss   : "
            + ("off — a sampler that dies mid-run disarms the two gauge rails "
               "silently" if not self.max_metrics_gaps else
               f"abort after {self.max_metrics_gaps} consecutive failed gauge "
               "reads once the sampler has been working (fails CLOSED)"),
            f"prompt tokens  : {tok}",
            "canary         : "
            + (f"1-token request every {self.canary_every_s:g}s; abort when "
               f"its p50 over the last {self.canary_window_s:g}s exceeds "
               f"{self.canary_drift:g}x the p50 over the first "
               f"{self.canary_baseline_s:g}s (both need "
               f"{self.canary_min_n} samples)" if self.canary else "off"),
        ]

    def to_dict(self) -> dict:
        return asdict(self)

    def detection_lag(self, metrics=None) -> str:
        """How long a rail can take to notice, stated rather than implied.

        A gauge rail does NOT fire "the moment" the server queues: it fires at
        the next observation, and the value it reads is already up to one
        scrape interval old. So the worst-case lag is the poll period plus the
        scrape interval, and the probe may send during it.
        """
        iv = getattr(metrics, "interval", None) if metrics else None
        if not isinstance(iv, (int, float)):
            return (f"gauges are read before each send and every "
                    f"{self.gauge_poll_s:g}s; the reading is up to one scrape "
                    "interval stale on top of that")
        return (f"up to ~{self.gauge_poll_s + float(iv):g}s "
                f"({self.gauge_poll_s:g}s poll + {float(iv):g}s scrape "
                "interval): a rail fires at the next observation of an "
                "already-stale gauge, not at the instant the server queues")


class BudgetAbort(RuntimeError):
    """A safety rail tripped. Carries the reason, the numbers behind it, and
    (once `run_shared` has caught it) the partial result, so the run record
    says exactly what stopped the probe."""

    def __init__(self, reason: str, **detail):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail
        self.result: "SharedResult | None" = None

    def to_dict(self) -> dict:
        return {"reason": self.reason, "detail": dict(self.detail)}


class ProbeGovernor:
    """Enforces a `ProbeBudget` over the life of one shared probe.

    Gauges are read before each send and by a watchdog on its own timer. That
    is NOT continuous coverage: the reading is already up to one scrape
    interval old, so a rail fires at the next observation of a stale gauge
    rather than at the instant the server queues. `ProbeBudget.detection_lag`
    states the bound, and `--dry-run` prints it.

    `metrics_expected` is what makes the gauge rails fail CLOSED. Without it a
    dead sampler and a run that never had one are the same thing — an
    `observe(None)` that quietly does nothing — so a mid-run scrape failure
    would disarm both rails and leave the probe running.
    """

    def __init__(self, budget: ProbeBudget, metrics_expected: bool = False):
        self.budget = budget
        cap = budget.max_extra_load
        self._sem = asyncio.Semaphore(cap) if cap else None
        self.metrics_expected = bool(metrics_expected)
        self.tokens_spent = 0
        self.n_requests = 0
        self.canary_ttft: list[tuple[float, float]] = []   # (t_rel s, ttft s)
        self.t0 = time.monotonic()
        self.aborted: BudgetAbort | None = None
        self.n_gauge_checks = 0
        self.peak_waiting: float | None = None
        self.peak_kv: float | None = None
        # metrics-loss detection. `armed` only after a reading has WORKED:
        # "lost" implies we had it, and a sampler's first scrape has not
        # landed when the probe's first request goes out.
        self.metrics_armed = False
        self.metrics_gaps = 0             # consecutive failed reads
        self.max_metrics_gaps_seen = 0
        # our OWN in-flight requests, by the sampler-base instant each was
        # sent at. What the server's gauge says about them depends on whether
        # the scrape behind it predates them, which is what `own_after` uses
        # this to resolve.
        self._open: dict[int, float] = {}
        self._next_id = 0
        self.peak_in_flight = 0

    # ---- in-flight cap --------------------------------------------------
    @asynccontextmanager
    async def slot(self):
        """Hold one of the `max_extra_load` in-flight slots.

        The cap is enforced AS GIVEN — an operator's explicit
        `--max-extra-load 1` is never raised to make room for the canary.
        `asyncio.Semaphore` hands slots out FIFO, so the canary queues behind
        the current probe request and runs next rather than starving.
        """
        if self._sem is None:
            yield
            return
        await self._sem.acquire()
        try:
            yield
        finally:
            self._sem.release()

    # ---- our own contribution to the server's gauge ----------------------
    @asynccontextmanager
    async def in_flight(self, t_sent: float):
        """Mark one of our requests open, from `t_sent` (SAMPLER base)."""
        rid, self._next_id = self._next_id, self._next_id + 1
        self._open[rid] = t_sent
        self.peak_in_flight = max(self.peak_in_flight, len(self._open))
        try:
            yield
        finally:
            self._open.pop(rid, None)

    def own_after(self, snapshot_t: float | None) -> int:
        """How many of OUR open requests were sent after `snapshot_t`, and so
        are provably NOT in the gauge that snapshot carries.

        This is the resolvable half of the race the stamped `running` has with
        the scrape behind it. A request of ours open since BEFORE the scrape
        was running when the server counted, and is already in the number; one
        sent after it cannot be. Adding the latter turns a regressor that
        meant "background, or background plus up to k of ours, depending on
        timing" into one that consistently means "what the server was carrying
        when we sent". Blind subtraction of our in-flight count would be the
        opposite error — it assumes the gauge always includes us.
        """
        if snapshot_t is None or not _fin(snapshot_t):
            return 0
        return sum(1 for t in self._open.values() if t > snapshot_t)

    # ---- token budget ---------------------------------------------------
    def spend(self, tokens: int) -> None:
        """Charge `tokens` INTENDED prompt tokens against the run's budget.

        Raises before the request is sent, so the cap is a cap and not a
        high-water mark.
        """
        cap = self.budget.max_probe_tokens
        if cap and self.tokens_spent + tokens > cap:
            self._abort("prompt-token budget exhausted",
                        spent_tokens=self.tokens_spent,
                        next_request_tokens=tokens, max_probe_tokens=cap)
        self.tokens_spent += tokens
        self.n_requests += 1

    # ---- the server's own gauges ----------------------------------------
    def observe(self, covariates: dict | None) -> None:
        """One reading of `requests_waiting` / `kv_cache_usage`.

        A falsy reading is NOT a pass. With no sampler configured it means the
        rail cannot fire, which the report says out loud. With one configured
        it means a scrape FAILED, and consecutive failures abort: otherwise a
        sampler dying mid-run silently disarms both gauge rails while the
        probe keeps sending, and the record cannot tell that apart from a run
        that never had metrics at all.
        """
        if not covariates:
            self._note_metrics_gap()
            return
        self.metrics_armed = True
        self.metrics_gaps = 0
        self.n_gauge_checks += 1
        w = covariates.get("requests_waiting")
        if w is not None and math.isfinite(w):
            self.peak_waiting = w if self.peak_waiting is None \
                else max(self.peak_waiting, w)
            lim = self.budget.abort_if_waiting
            if lim is not None and w > lim:
                self._abort(
                    f"the server's queue reached {w:g} waiting requests "
                    f"(--abort-if-waiting {lim:g}): somebody else is already "
                    "queueing behind this endpoint",
                    requests_waiting=w, limit=lim)
        kv = covariates.get("kv_cache_usage")
        if kv is not None and math.isfinite(kv):
            self.peak_kv = kv if self.peak_kv is None else max(self.peak_kv, kv)
            lim = self.budget.abort_if_kv_above
            if lim is not None and kv > lim:
                self._abort(
                    f"KV occupancy reached {kv:.1%} (--abort-if-kv-above "
                    f"{lim:.1%}): the pool is close to evicting somebody "
                    "else's session",
                    kv_cache_usage=kv, limit=lim)

    def _note_metrics_gap(self) -> None:
        """A gauge read that came back with nothing, when one was expected."""
        if not self.metrics_expected or not self.metrics_armed:
            return
        self.metrics_gaps += 1
        self.max_metrics_gaps_seen = max(self.max_metrics_gaps_seen,
                                         self.metrics_gaps)
        limit = self.budget.max_metrics_gaps
        if limit and self.metrics_gaps > limit:
            self._abort(
                f"metrics_lost: {self.metrics_gaps} consecutive gauge reads "
                f"failed after the sampler had been working "
                f"(--max-metrics-gaps {limit}). The queue and KV rails cannot "
                "fire without it, so the probe stops rather than run blind on "
                "somebody else's endpoint",
                consecutive_failures=self.metrics_gaps, limit=limit,
                kind="metrics_lost")

    # ---- the canary ------------------------------------------------------
    def note_canary(self, t_send: float, ttft: float | None) -> None:
        """Record one canary TTFT (seconds) sent at monotonic `t_send`, then
        apply the drift rule."""
        if ttft is None or not math.isfinite(ttft):
            return
        self.canary_ttft.append((t_send - self.t0, ttft))
        drift = self.canary_drift()
        if drift is not None:
            base, recent = drift
            self._abort(
                f"canary TTFT drifted {recent / base:.1f}x: p50 over the last "
                f"{self.budget.canary_window_s:g}s is {recent:.2f}s against "
                f"{base:.2f}s over the first {self.budget.canary_baseline_s:g}s "
                f"(--canary-drift {self.budget.canary_drift:g}). The endpoint "
                "got busier while we were probing it",
                baseline_p50_s=base, recent_p50_s=recent,
                ratio=recent / base)

    def canary_drift(self) -> tuple[float, float] | None:
        """(baseline p50, recent p50) when the drift rule has FIRED, else
        None. Pure, so the rule is testable without a governor's timers.

        The rule refuses to compare a window with itself: the recent window
        must start after the baseline window ended, so a run shorter than
        `canary_baseline_s + canary_window_s` can never trip it.
        """
        b = self.budget
        if not b.canary or not self.canary_ttft:
            return None
        now = self.canary_ttft[-1][0]
        if now < b.canary_baseline_s + b.canary_window_s:
            return None
        base = [v for t, v in self.canary_ttft if t <= b.canary_baseline_s]
        recent = [v for t, v in self.canary_ttft if t > now - b.canary_window_s]
        if len(base) < b.canary_min_n or len(recent) < b.canary_min_n:
            return None
        p_base, p_recent = pct(base, 50), pct(recent, 50)
        if not (p_base > 0) or not math.isfinite(p_recent):
            return None
        if p_recent > b.canary_drift * p_base:
            return p_base, p_recent
        return None

    # ---- abort -----------------------------------------------------------
    def _abort(self, reason: str, **detail) -> None:
        """The FIRST rail to trip owns the reason.

        A later one overwriting it would rewrite history: the run stopped
        because of the first, and everything after it happened on the way out.
        """
        if self.aborted is None:
            self.aborted = BudgetAbort(reason, **detail)
        raise self.aborted

    def raise_if_aborted(self) -> None:
        if self.aborted is not None:
            raise self.aborted

    def to_dict(self) -> dict:
        return {"budget": self.budget.to_dict(),
                "tokens_spent": self.tokens_spent,
                "n_requests": self.n_requests,
                "n_gauge_checks": self.n_gauge_checks,
                "metrics_expected": self.metrics_expected,
                "metrics_armed": self.metrics_armed,
                "max_consecutive_metrics_gaps": self.max_metrics_gaps_seen,
                "peak_requests_waiting": self.peak_waiting,
                "peak_kv_cache_usage": self.peak_kv,
                "peak_probe_in_flight": self.peak_in_flight,
                "n_canary": len(self.canary_ttft),
                "canary_p50_s": pct([v for _, v in self.canary_ttft], 50)
                if self.canary_ttft else None,
                "aborted": None if self.aborted is None
                else self.aborted.to_dict()}


# ============================================================================
# the fit
# ============================================================================
TTFT_COLUMNS = ("const", "L_ktok", "L_ktok2", "running", "waiting")
LOAD_COLUMNS = ("const", "running", "waiting")

_COLUMN_UNITS = {
    "const": "", "L_ktok": "per kilotoken", "L_ktok2": "per kilotoken^2",
    "running": "per running request", "waiting": "per waiting request",
}


@dataclass(frozen=True)
class CovariateFit:
    """One ordinary-least-squares fit of a measured quantity on the load the
    server was carrying when the measurement was taken.

    coefficients   column name -> coefficient, in `unit` per that column's own
                   unit (see `_COLUMN_UNITS`)
    residual_std   sqrt(SSR / (n - k)), in `unit`. The spread the covariates
                   did NOT explain — read it before the coefficients.
    condition_number  2-norm condition number of the design matrix AS SOLVED.
                   Reported, not gated: it is dominated by column scaling.
    scaled_condition_number  the same on a standardised design — this one IS
                   a collinearity statistic, and it is gated.
    vif            column -> variance inflation factor, 1/(1 - R^2_j) from
                   regressing that column on the others. Says WHICH
                   coefficient is unstable, which a whole-design number
                   cannot.
    centre         the L the quadratic was centred on, kilotokens (0 for a
                   fit with no length term). Coefficients are in the CENTRED
                   parameterisation; `coefficients_raw_L` converts back.
    ranges         column -> {min, max, mean, sd} over the observations AS
                   FITTED (so centred, for the length columns), which is what
                   an extrapolation distance is measured against
    refused        why this fit may not be used, or None
    """
    target: str
    unit: str
    columns: tuple = ()
    coefficients: dict = field(default_factory=dict)
    n: int = 0
    dof: int = 0
    residual_std: float = float("nan")
    condition_number: float = float("inf")
    scaled_condition_number: float = float("inf")
    vif: dict = field(default_factory=dict)
    centre: float = 0.0
    r_squared: float = float("nan")
    ranges: dict = field(default_factory=dict)
    refused: str | None = None
    _cov: tuple = field(default=(), repr=False)      # (X'X)^-1, row-major

    # ---- use -------------------------------------------------------------
    @property
    def usable(self) -> bool:
        return self.refused is None

    def _row(self, point: dict) -> np.ndarray:
        """One design row from a point given in RAW coordinates.

        `point` carries raw `L_ktok` and `L_ktok2`; both are moments, so a
        point may describe a DISTRIBUTION rather than one request — which is
        the whole reason the quadratic is worth having. Centring is exact on
        moments:

            E[(L - c)^2] = E[L^2] - 2 c E[L] + c^2

        so the operating point's E[L] and E[L^2] evaluate the centred fit
        with no approximation, and a single-L point (a ladder bin) reduces to
        (L - c)^2 as it should.
        """
        c = self.centre
        out = []
        for name in self.columns:
            if name == "const":
                out.append(1.0)
            elif name == "L_ktok":
                out.append(float(point["L_ktok"]) - c)
            elif name == "L_ktok2":
                out.append(float(point["L_ktok2"])
                           - 2.0 * c * float(point["L_ktok"]) + c * c)
            else:
                out.append(float(point[name]))
        return np.array(out, dtype=float)

    @property
    def coefficients_raw_L(self) -> dict:
        """The same fit in the UNCENTRED parameterisation the module docstring
        advertises, y = c0 + c1 L + c2 L^2 + ...:

            c0 = a0 - a1 c + a2 c^2,   c1 = a1 - 2 a2 c,   c2 = a2

        Identical predictions; reported so the printed form matches the form
        the reader was promised.
        """
        if not self.usable or "L_ktok" not in self.columns:
            return dict(self.coefficients)
        a, c = self.coefficients, self.centre
        out = dict(a)
        a1, a2 = a.get("L_ktok", 0.0), a.get("L_ktok2", 0.0)
        out["const"] = a.get("const", 0.0) - a1 * c + a2 * c * c
        out["L_ktok"] = a1 - 2.0 * a2 * c
        out["L_ktok2"] = a2
        return out

    def predict(self, point: dict) -> float:
        """Evaluate the fit at `point` (column name -> value). Units: `unit`."""
        if not self.usable:
            raise ValueError(f"fit refused: {self.refused}")
        x = self._row(point)
        b = np.array([self.coefficients[c] for c in self.columns], dtype=float)
        return float(x @ b)

    def predict_se(self, point: dict) -> float:
        """Standard error of the FITTED MEAN at `point`, in `unit`:
        residual_std * sqrt(x' (X'X)^-1 x). Grows with distance from the
        centre of the data, which is the arithmetic behind the extrapolation
        gate charging for exactly that."""
        if not self.usable or not self._cov:
            return float("nan")
        k = len(self.columns)
        cov = np.array(self._cov, dtype=float).reshape(k, k)
        x = self._row(point)
        v = float(x @ cov @ x)
        if not math.isfinite(v) or v < 0:
            return float("nan")
        return float(self.residual_std * math.sqrt(v))

    def extrapolation(self, point: dict) -> tuple[float, dict]:
        """(worst distance in sd, per-column distances in sd).

        A column's distance is how far `point` lies OUTSIDE the observed
        [min, max] of that column, divided by the column's observed standard
        deviation; zero inside the range. `const` is skipped. A column that
        never varied (sd = 0) gives inf outside its single observed value —
        a regressor with no spread supports no extrapolation at all.
        """
        per = {c: d["sd"] for c, d in self.offsets(point).items()}
        worst = max(per.values()) if per else 0.0
        return worst, per

    def offsets(self, point: dict) -> dict:
        """Per-column {sd, absolute, above} for a point, in ONE pass.

        `absolute` is in the column's own units — requests for `running` and
        `waiting` — and `above` says whether the point sits above the probed
        range rather than below it. Both matter: the sd distance alone is
        measured in units of the probe's own empirical spread, so a noisier
        background would buy a wider absolute licence, and the direction
        decides whether a linear reading of a hyperbolic queue is biased low
        (above) or high (below).
        """
        x = self._row(point)
        out: dict[str, dict] = {}
        for i, c in enumerate(self.columns):
            if c == "const":
                continue
            r = self.ranges.get(c) or {}
            v = float(x[i])
            lo, hi, sd = r.get("min"), r.get("max"), r.get("sd")
            if lo is None or hi is None:
                out[c] = {"sd": float("inf"), "absolute": float("inf"),
                          "above": True, "value": v}
                continue
            over, under = v - hi, lo - v
            dist = max(0.0, over, under)
            if dist <= 0:
                sd_d = 0.0
            elif sd and sd > 0:
                sd_d = dist / sd
            else:
                sd_d = float("inf")
            out[c] = {"sd": sd_d, "absolute": dist, "above": over > 0,
                      "value": v}
        return out

    def over_absolute(self, point: dict, limit: float) -> dict:
        """Columns measured in REQUESTS whose absolute distance outside the
        probed range exceeds `limit`. Empty when the point is close enough in
        absolute terms, whatever the probe's own noise happened to be."""
        return {c: d for c, d in self.offsets(point).items()
                if c in REQUEST_COLUMNS and d["absolute"] > limit}

    def to_dict(self) -> dict:
        return {"target": self.target, "unit": self.unit,
                "columns": list(self.columns),
                "coefficients": dict(self.coefficients),
                "coefficients_raw_L": self.coefficients_raw_L,
                "centre_ktok": self.centre,
                "coefficient_units": {c: f"{self.unit} {_COLUMN_UNITS[c]}".strip()
                                      for c in self.columns},
                "n": self.n, "dof": self.dof,
                "residual_std": _num(self.residual_std),
                "condition_number": _num(self.condition_number),
                "scaled_condition_number": _num(self.scaled_condition_number),
                "vif": {k: _num(v) for k, v in self.vif.items()},
                "r_squared": _num(self.r_squared),
                "ranges": dict(self.ranges), "refused": self.refused}

    def summary(self) -> str:
        if not self.usable:
            return f"{self.target}: no fit — {self.refused}"
        terms = " ".join(f"{c}={self.coefficients[c]:+.4g}"
                         for c in self.columns)
        return (f"{self.target} [{self.unit}]: {terms} | n={self.n} "
                f"resid sd {self.residual_std:.3g} R2 {self.r_squared:.2f} "
                f"scaled cond {self.scaled_condition_number:.3g}")


def _design(rows: list[dict], columns, centre: float) -> np.ndarray:
    """The design matrix, with the quadratic CENTRED on `centre`.

    L and L^2 are structurally correlated whatever the data quality — on the
    default ladder their VIFs are ~20 — and that is an artefact of writing the
    quadratic about zero, not a finding about the endpoint. Centring on the
    probed mean drops both to ~1.1 and leaves the collinearity statistics free
    to say something about `running` and `waiting`, which is what they are
    for. Predictions are identical either way; `coefficients_raw_L` converts
    back to the parameterisation the docstring advertises.
    """
    out = []
    for r in rows:
        row = []
        for c in columns:
            if c == "const":
                row.append(1.0)
            elif c == "L_ktok":
                row.append(float(r["L_ktok"]) - centre)
            elif c == "L_ktok2":
                row.append((float(r["L_ktok"]) - centre) ** 2)
            else:
                row.append(float(r[c]))
        out.append(row)
    return np.array(out, dtype=float)


def _vifs(X: np.ndarray, columns) -> dict:
    """Variance inflation factor per non-constant column: 1/(1 - R^2_j) from
    regressing column j on every other column. Infinite when a column is an
    exact combination of the others."""
    out: dict[str, float] = {}
    for j, c in enumerate(columns):
        if c == "const":
            continue
        yj = X[:, j]
        others = np.delete(X, j, axis=1)
        ss_tot = float(((yj - yj.mean()) ** 2).sum())
        if ss_tot <= 0:
            out[c] = float("inf")
            continue
        b, *_ = np.linalg.lstsq(others, yj, rcond=None)
        resid = yj - others @ b
        r2 = 1.0 - float(resid @ resid) / ss_tot
        out[c] = float("inf") if r2 >= 1.0 else float(1.0 / (1.0 - r2))
    return out


def _scaled_condition(X: np.ndarray, columns) -> float:
    """Condition number of the design with every non-constant column
    standardised. The RAW number is dominated by column scaling — a constant
    column of 1 beside an L^2 column spanning 324-32400 is a raw cond of ~6e4
    on a design whose regressors are independent — so it is the standardised
    one that means "the regressors moved together"."""
    S = X.copy()
    for j, c in enumerate(columns):
        if c == "const":
            continue
        col = S[:, j]
        sd = col.std(ddof=1)
        S[:, j] = (col - col.mean()) / (sd if sd > 0 else 1.0)
    return float(np.linalg.cond(S))


def fit_covariates(rows: list[dict], columns=TTFT_COLUMNS, target: str = "y",
                   unit: str = "", min_obs_per_coef: int = MIN_OBS_PER_COEF,
                   max_vif: float = MAX_VIF,
                   max_condition: float = MAX_SCALED_CONDITION) -> CovariateFit:
    """OLS of `row["y"]` on `columns`, refusing rather than guessing.

    `rows` is a list of dicts carrying every name in `columns` (bar `const`
    and `L_ktok2`, which is derived from `L_ktok`) plus `"y"`. Rows with a
    missing or non-finite entry are DROPPED — a request sent while no metrics
    snapshot existed carries no load reading, and imputing one would invent
    the covariate the whole design rests on.

    Refuses, with the reason in `refused` and what would fix it:
      * n below `min_obs_per_coef` x (number of coefficients), floor
        `MIN_OBS_FLOOR`
      * a rank-deficient design matrix (naming the columns that did not vary)
      * a variance inflation factor above `max_vif`, NAMING the inflated
        coefficient — a whole-design number cannot say which one is unstable
      * a standardised condition number above `max_condition`
      * a degenerate residual (n == k: no degrees of freedom left)
    """
    k = len(columns)
    need = max(min_obs_per_coef * k, MIN_OBS_FLOOR)
    names = [c for c in columns if c not in ("const", "L_ktok2")]
    kept = []
    for r in rows:
        y = r.get("y")
        vals = [r.get(c) for c in names]
        if y is None or not _fin(y) or any(v is None or not _fin(v)
                                           for v in vals):
            continue
        kept.append((r, float(y)))
    n = len(kept)
    base = dict(target=target, unit=unit, columns=tuple(columns), n=n)
    if n < need:
        return CovariateFit(
            **base, refused=f"n={n} covariate-stamped observations, {need} "
                            f"needed for {k} coefficients ({min_obs_per_coef} "
                            f"per coefficient). Raise --shared-rounds, add "
                            f"--shared-ladder, or attach --metrics-url so the "
                            f"load columns exist at all")
    centre = (float(np.mean([float(r["L_ktok"]) for r, _ in kept]))
              if "L_ktok" in columns else 0.0)
    X = _design([r for r, _ in kept], columns, centre)
    y = np.array([v for _, v in kept], dtype=float)
    base["centre"] = centre
    base["ranges"] = {
        c: {"min": float(X[:, i].min()), "max": float(X[:, i].max()),
            "mean": float(X[:, i].mean()), "sd": float(X[:, i].std(ddof=1))}
        for i, c in enumerate(columns) if c != "const"}

    flat = [c for i, c in enumerate(columns)
            if c != "const" and X[:, i].std() == 0.0]
    rank = int(np.linalg.matrix_rank(X))
    if rank < k:
        why = (f"column(s) {', '.join(flat)} never varied"
               if flat else "the columns are exactly collinear")
        return CovariateFit(
            **base, refused=f"design matrix is rank-deficient (rank {rank} of "
                            f"{k}): {why}. The load and the prompt length must "
                            f"vary INDEPENDENTLY for their coefficients to "
                            f"separate — probe over a longer window, or over "
                            f"more --shared-lengths")
    cond = float(np.linalg.cond(X))
    scond = _scaled_condition(X, columns)
    vif = _vifs(X, columns)
    base.update(condition_number=cond, scaled_condition_number=scond, vif=vif)
    worst = max(vif, key=lambda c: vif[c]) if vif else None
    if worst is not None and vif[worst] > max_vif:
        return CovariateFit(
            **base,
            refused=f"`{worst}` has a variance inflation factor of "
                    f"{vif[worst]:.3g} (limit {max_vif:g}): it moved with the "
                    f"other regressors, so its coefficient is inflated "
                    f"~{math.sqrt(vif[worst]):.1f}x in standard error and the "
                    f"split between them is arbitrary. Probe across a wider "
                    f"range of server load, or over more --shared-lengths")
    if not math.isfinite(scond) or scond > max_condition:
        return CovariateFit(
            **base,
            refused=f"standardised condition number {scond:.3g} exceeds "
                    f"{max_condition:.3g}: the regressors moved together, so "
                    f"the split between their coefficients is arbitrary. "
                    f"Probe across a wider range of server load, or over more "
                    f"--shared-lengths")
    if n <= k:
        return CovariateFit(**base,
                            refused=f"n={n} equals the {k} coefficients: the "
                                    "fit would be exact and its residual "
                                    "undefined")

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = n - k
    resid_std = float(math.sqrt(float(resid @ resid) / dof))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = float(1.0 - float(resid @ resid) / ss_tot) if ss_tot > 0 else float("nan")
    try:
        cov = np.linalg.inv(X.T @ X)
        cov_flat = tuple(float(v) for v in cov.reshape(-1))
    except np.linalg.LinAlgError:
        cov_flat = ()
    return CovariateFit(**base, dof=dof, residual_std=resid_std, r_squared=r2,
                        _cov=cov_flat,
                        coefficients={c: float(b)
                                      for c, b in zip(columns, beta)})


# ============================================================================
# where the predictions live: the configured operating point, in the fit's
# own coordinates
# ============================================================================
@dataclass(frozen=True)
class OperatingPoint:
    """The (L, running, waiting) the CONFIGURED operating point implies —
    the point the fit is evaluated at.

    running    steady_decode_seqs + prefill occupancy. `steady_decode_point`
               gives the expected number of DECODING sequences; a request
               being prefilled is also `running` in vLLM's gauge, and under
               M/G/1 the expected number in prefill service is exactly the
               duty cycle rho = `Predictions.prefill_duty`. Requests, not
               users.
    waiting    Little's law on the prefill QUEUE: arrival rate x mean P-K
               wait (`model.queue_wait_seconds`). Requests.
    L_ktok     E[L] over the workload's context distribution, kilotokens
    L2_ktok2   E[L^2] over the same, kilotokens^2. Both moments are carried
               because the fit is QUADRATIC in L, so E[TTFT] over the
               distribution is exactly c0 + c1 E[L] + c2 E[L^2] + ... — the
               fit can be evaluated at the mean of a distribution without
               approximation, which is what makes it comparable with
               `ttft_miss_s` (itself a mean over that distribution).
    """
    running: float = float("nan")
    waiting: float = float("nan")
    L_ktok: float = float("nan")
    L2_ktok2: float = float("nan")
    steady_decode_seqs: float = float("nan")
    prefill_occupancy: float = float("nan")
    rate_total_req_s: float = float("nan")
    # what the model says the two TTFTs are AT this point, carried so the
    # report can put a fitted reading next to its prediction without
    # recomputing anything — `predicted_hit_ttft_s` in particular has no
    # hypothesis of its own and would otherwise be a number with nothing to
    # compare it to
    predicted_miss_ttft_s: float = float("nan")
    predicted_hit_ttft_s: float = float("nan")
    refused: str | None = None

    def point(self, columns) -> dict:
        return {"const": 1.0, "L_ktok": self.L_ktok, "L_ktok2": self.L2_ktok2,
                "running": self.running, "waiting": self.waiting}

    def to_dict(self) -> dict:
        d = {k: _num(v) if isinstance(v, float) else v
             for k, v in asdict(self).items()}
        return d


def operating_point_covariates(cfg, preds, n_iter: int | None = None,
                               seed: int = 0) -> OperatingPoint:
    """Translate the configured operating point into the fit's coordinates.

    Every number is fetched from `workingset.model` / `workingset.predict`;
    nothing is modelled here. Refuses (with a reason) when the model itself
    has no steady state to quote — prefill duty at or above 100%, or no
    steady decode point for this configuration.
    """
    from . import model as M

    if preds.prefill_duty >= 1.0 or not math.isfinite(preds.ttft_miss_s):
        return OperatingPoint(
            refused=f"prefill duty is {preds.prefill_duty:.0%} at the "
                    "configured operating point: the model has no steady "
                    "state there, so there is no (running, waiting) to "
                    "evaluate a fit at")
    if preds.steady_decode_seqs is None:
        return OperatingPoint(
            refused="no steady decode point for this configuration, so the "
                    "expected decode batch — half of the expected `running` — "
                    "is undefined")
    m, t, wl = cfg.to_model(), cfg.to_topology(), cfg.to_workload()
    w, dep, cal = cfg.workload, cfg.deployment, cfg.calibration
    rate_total = preds.req_rate_main * (1.0 + wl.sub_ratio)
    wait_s = M.queue_wait_seconds(m, t, wl, rate_total,
                                  dep.max_num_batched_tokens,
                                  w.warm_turn_tokens, cal.mfu,
                                  per_pass_overhead=True)
    if not math.isfinite(wait_s):
        return OperatingPoint(
            refused="the M/G/1 queue has no steady state at this load "
                    "(rho >= 1), so the expected queue depth is unbounded")
    # the model's OWN sampling default (200k) unless a caller says otherwise:
    # E[L^2] is a second moment of a heavy-tailed lognormal mixture, which is
    # exactly where a thinner sample is least trustworthy, and this one is
    # computed once per run
    e_l, e_l2 = (M.context_moments(wl, seed=seed) if n_iter is None
                 else M.context_moments(wl, n=n_iter, seed=seed))
    return OperatingPoint(
        running=float(preds.steady_decode_seqs) + float(preds.prefill_duty),
        waiting=float(rate_total * wait_s),
        L_ktok=float(e_l) / 1e3, L2_ktok2=float(e_l2) / 1e6,
        steady_decode_seqs=float(preds.steady_decode_seqs),
        prefill_occupancy=float(preds.prefill_duty),
        rate_total_req_s=float(rate_total),
        predicted_miss_ttft_s=float(preds.ttft_miss_s),
        predicted_hit_ttft_s=float(preds.ttft_hit_s))


# ============================================================================
# the natural ladder: bin a long shared run by the load it happened to see
# ============================================================================
LADDER_EDGES = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)


def natural_ladder(rows: list[dict], edges=LADDER_EDGES,
                   min_n: int = 3) -> list[dict]:
    """Bin covariate-stamped readings by the `running` the SERVER happened to
    be carrying, and report each bin's latency statistics.

    This is the ladder a shared run gets for free: it never sets the load, it
    observes it. `rows` carry `running` plus any of `ttft` (s, with `kind`),
    `itl_ms` and `decode_tok_s`. Bins with fewer than `min_n` readings are
    reported with their count and nothing else — a p50 over two samples is a
    number, not a measurement.

    Bins are [edge, next_edge), the last one [edges[-1], inf).
    """
    edges = tuple(sorted(set(float(e) for e in edges)))
    out = []
    for i, lo in enumerate(edges):
        hi = edges[i + 1] if i + 1 < len(edges) else float("inf")
        inside = [r for r in rows
                  if _fin(r.get("running")) and lo <= r["running"] < hi]
        row = {"running_lo": lo, "running_hi": hi, "n": len(inside),
               "running_mean": pct([r["running"] for r in inside], 50)
               if inside else float("nan"),
               "ttft_miss_p50_s": float("nan"), "ttft_hit_p50_s": float("nan"),
               "itl_p50_ms": float("nan"), "decode_p50_tok_s": float("nan"),
               "enough": len(inside) >= min_n}
        if inside:
            row["running_mean"] = sum(r["running"] for r in inside) / len(inside)
        if row["enough"]:
            row["ttft_miss_p50_s"] = pct(
                [r["ttft"] for r in inside
                 if r.get("kind") == "miss" and _fin(r.get("ttft"))], 50)
            row["ttft_hit_p50_s"] = pct(
                [r["ttft"] for r in inside
                 if r.get("kind") == "hit" and _fin(r.get("ttft"))], 50)
            row["itl_p50_ms"] = pct([r["itl_ms"] for r in inside
                                     if _fin(r.get("itl_ms"))], 50)
            row["decode_p50_tok_s"] = pct([r["decode_tok_s"] for r in inside
                                           if _fin(r.get("decode_tok_s"))], 50)
        out.append(row)
    return [r for r in out if r["n"]]


def ladder_model_curve(cfg, running: float, n_iter: int = 96,
                       seed: int = 0) -> dict:
    """What the MODEL says at a given concurrency — the curve the natural
    ladder's bins are read against.

    `running` is the server's gauge: decoders plus whatever is in prefill
    service. The decode half is read straight off `model.decode_curves` at
    that batch. The TTFT half needs a load, not a batch, so the arrival rate
    that would PRODUCE this `running` is recovered by bisection on

        running(rate) = steady_decode_point(rate)["n"] + prefill_duty(rate)

    (both strictly increasing in rate), and `prefill_ttft_seconds` is then
    quoted at that rate. Every term comes from `workingset.model`.

    Returns {decode_tok_s, itl_ms, rate_req_s, ttft_miss_s, ttft_hit_s}, with
    nan where the inversion found no rate below saturation.
    """
    from . import model as M

    m, t, wl = cfg.to_model(), cfg.to_topology(), cfg.to_workload()
    w, dep, cal = cfg.workload, cfg.deployment, cfg.calibration
    chunk, turn = dep.max_num_batched_tokens, w.warm_turn_tokens
    out = {"running": float(running), "decode_tok_s": float("nan"),
           "itl_ms": float("nan"), "rate_req_s": float("nan"),
           "ttft_miss_s": float("nan"), "ttft_hit_s": float("nan")}
    n = max(1, int(round(running)))
    p5, p50, p95, _ = M.decode_curves(m, t, wl, [n], n_iter=n_iter, seed=seed,
                                      mbu=cal.mbu)
    pu = float(p50[0])
    out["decode_tok_s"] = pu
    if pu > 0:
        out["itl_ms"] = 1e3 * m.mtp / pu

    def running_at(rate: float) -> float:
        sp = M.steady_decode_point(m, t, wl, rate, out_tokens=w.max_output_tokens,
                                   mbu=cal.mbu, n_iter=n_iter, seed=seed)
        duty = M.prefill_duty(m, t, wl, rate, chunk, turn, cal.mfu,
                              per_pass_overhead=True)
        return sp["n"] + min(duty, 1.0)

    lo, hi = 0.0, max(1e-3, out["decode_tok_s"] / max(w.max_output_tokens, 1))
    for _ in range(24):                       # bracket by doubling
        if running_at(hi) >= running:
            break
        hi *= 2.0
    else:
        return out
    for _ in range(28):                       # bisect
        mid = 0.5 * (lo + hi)
        if running_at(mid) < running:
            lo = mid
        else:
            hi = mid
    rate = 0.5 * (lo + hi)
    duty = M.prefill_duty(m, t, wl, rate, chunk, turn, cal.mfu,
                          per_pass_overhead=True)
    if duty >= 1.0:
        return out
    out["rate_req_s"] = rate
    out["ttft_miss_s"] = M.prefill_ttft_seconds(m, t, wl, rate, chunk, turn,
                                                cal.mfu, "cold",
                                                per_pass_overhead=True)
    out["ttft_hit_s"] = M.prefill_ttft_seconds(m, t, wl, rate, chunk, turn,
                                               cal.mfu, "warm",
                                               per_pass_overhead=True)
    return out


# ============================================================================
# the server's own view, alongside the client's
# ============================================================================
async def cross_check(metrics, t0: float, t1: float,
                      traces: list) -> dict | None:
    """The server's TTFT/ITL quantiles over [t0, t1] next to the client's.

    `t0`/`t1` are UNIX seconds. `next_tick()` is awaited first, because the
    enclosing high endpoint of a window does not exist until a scrape STARTS
    after t1 — without it `MetricsSampler.window` raises `WindowNotCovered`
    rather than answering a question about a different stretch of time.

    Returns None with no sampler; a dict with `"error"` when the window could
    not be covered, which is itself the finding.

    PROXY OVERHEAD is `client - server` on the same quantile: the client
    measures TTFT from POST to first byte on the wire, the server's histogram
    from admission to first token. The difference is the proxy, the network
    and the client's own event loop, and it is the number that says whether a
    client-side TTFT can be read as a server-side one at all.

    FORCED MISSES are confirmed two ways: the window's `prefix_hit_rate`
    (over ALL traffic, ours and theirs — a shared endpoint cannot attribute
    it) and the per-request `usage.prompt_tokens_details.cached_tokens`
    readback, which IS per-request and is the one that settles it.
    """
    if metrics is None:
        return None
    tick = getattr(metrics, "next_tick", None)
    if tick is not None:
        try:
            r = tick()
            if asyncio.iscoroutine(r):
                await r
        except Exception:                       # noqa: BLE001 — reported below
            pass
    try:
        w = metrics.window(t0, t1)
    except Exception as e:                      # noqa: BLE001
        return {"t0": t0, "t1": t1,
                "error": f"{type(e).__name__}: {e}"[:300]}

    out: dict = {"t0": t0, "t1": t1, "window": window_dict(w)}
    out.update(_server_quantiles(w))
    ok = [t for t in traces if not t.error and t.ttft is not None]
    miss = [t for t in ok if t.kind in ("miss", "first")]
    out["client_ttft_p50_s"] = pct([t.ttft for t in ok], 50)
    out["client_ttft_p95_s"] = pct([t.ttft for t in ok], 95)
    out["client_itl_p50_ms"] = pct([t.itl_p50 * 1e3 for t in ok
                                    if t.itl_p50 is not None], 50)
    for q in ("p50", "p95"):
        s, c = out.get(f"server_ttft_{q}_s"), out.get(f"client_ttft_{q}_s")
        out[f"proxy_overhead_ttft_{q}_s"] = (c - s if _fin(s) and _fin(c)
                                             else float("nan"))
    s, c = out.get("server_itl_p50_ms"), out.get("client_itl_p50_ms")
    out["proxy_overhead_itl_p50_ms"] = (c - s if _fin(s) and _fin(c)
                                        else float("nan"))
    # did the forced misses actually miss?
    readback = [t for t in miss
                if t.cached_tokens is not None and t.ptok_achieved]
    out["n_miss_with_cached_readback"] = len(readback)
    out["forced_miss_clean_frac"] = (
        sum(1 for t in readback
            if t.cached_tokens <= 0.10 * t.ptok_achieved) / len(readback)
        if readback else float("nan"))
    out["forced_miss_cached_tokens_p50"] = pct(
        [t.cached_tokens for t in readback], 50) if readback else float("nan")
    return out


def _server_quantiles(w) -> dict:
    """TTFT / per-token-latency quantiles off a `WindowDelta`, defensively:
    a delta whose histograms are `invalid` (a counter reset mid-window)
    carries None, not a fabricated number."""
    out = {"server_ttft_p50_s": float("nan"), "server_ttft_p95_s": float("nan"),
           "server_ttft_n": 0, "server_itl_p50_ms": float("nan"),
           "server_itl_n": 0, "prefix_hit_rate": float("nan")}
    h = getattr(w, "ttft", None)
    if h is not None and getattr(h, "observations", 0):
        out["server_ttft_p50_s"] = _f(h.quantile(0.5))
        out["server_ttft_p95_s"] = _f(h.quantile(0.95))
        out["server_ttft_n"] = int(h.observations)
    g = getattr(w, "request_tpot", None) or getattr(w, "tpot", None)
    if g is not None and getattr(g, "observations", 0):
        out["server_itl_p50_ms"] = _f(g.quantile(0.5)) * 1e3
        out["server_itl_n"] = int(g.observations)
    hr = getattr(w, "prefix_hit_rate", None)
    if hr is not None:
        out["prefix_hit_rate"] = _f(hr)
    return out


# ============================================================================
# the shared probe
# ============================================================================
CANARY_PROMPT = "ping"       # byte-stable, so after the first it is a hit


@dataclass
class SharedResult:
    """What a shared run establishes, and what it refuses to.

    `sample` is the same `Sample` the plain cheap probe produces, built from
    these traces, so the report's SAMPLE PROBE block and every existing
    Sample reader keep working unchanged.
    """
    fits: dict = field(default_factory=dict)          # name -> CovariateFit
    op: OperatingPoint = field(default_factory=OperatingPoint)
    max_extrapolation: float = 1.0
    max_extrapolation_requests: float = MAX_EXTRAPOLATION_REQUESTS
    verdict_sigmas: float = 3.0
    probe_in_flight_bound: int = 0
    ladder: list = field(default_factory=list)
    windows: list = field(default_factory=list)
    cross: dict | None = None
    governor: dict = field(default_factory=dict)
    aborted: str | None = None
    n_covariate_rows: int = 0
    lengths_ktok: list = field(default_factory=list)
    sample: Sample | None = None
    options: dict = field(default_factory=dict)

    # ---- the gate --------------------------------------------------------
    def reading(self, which: str) -> dict:
        """A fitted reading at the configured operating point, or the reason
        there is none.

        `which` is a key in `fits`. The returned dict always carries
        `available` and `reason`; when available it also carries `value`,
        `se`, `extrapolation` and the fit's own numbers, which is what a
        hypothesis records and prints.
        """
        fit: CovariateFit | None = self.fits.get(which)
        base = {"available": False, "reason": None, "which": which,
                "value": None, "se": float("nan"), "n": 0,
                "extrapolation": float("inf"), "extrapolation_by": {},
                "max_extrapolation": self.max_extrapolation,
                "max_extrapolation_requests": self.max_extrapolation_requests,
                "verdict_sigmas": self.verdict_sigmas,
                "extrapolating_upward": [], "upward_bias": None,
                "at": self.op.to_dict(), "fit": None}
        if fit is None:
            base["reason"] = f"no {which} fit was attempted"
            return base
        base["fit"] = fit.to_dict()
        base["n"] = fit.n
        if not fit.usable:
            base["reason"] = fit.refused
            return base
        if self.op.refused:
            base["reason"] = self.op.refused
            return base
        point = self.op.point(fit.columns)
        offs = fit.offsets(point)
        per = {c: d["sd"] for c, d in offs.items()}
        base["extrapolation"], base["extrapolation_by"] = (
            max(per.values()) if per else 0.0), per
        up = sorted(c for c, d in offs.items()
                    if d["absolute"] > 0 and d["above"])
        base["extrapolating_upward"] = up
        if up:
            base["upward_bias"] = UPWARD_BIAS
        # GATE 1, relative: how many of the probe's OWN standard deviations
        # outside the probed cloud the operating point sits
        if base["extrapolation"] > self.max_extrapolation:
            worst = max(per, key=lambda c: per[c])
            base["reason"] = self._too_far(
                fit, worst, offs[worst],
                f"{base['extrapolation']:.2f} observed sd",
                f"--max-extrapolation {self.max_extrapolation:g}")
            return base
        # GATE 2, absolute: the sd gate is scaled by the probe's own noise, so
        # a busier background would silently buy a wider licence. This one is
        # in requests and does not move.
        over = fit.over_absolute(point, self.max_extrapolation_requests)
        if over:
            worst = max(over, key=lambda c: over[c]["absolute"])
            base["reason"] = self._too_far(
                fit, worst, over[worst],
                f"{over[worst]['absolute']:.2f} requests",
                f"--max-extrapolation-requests "
                f"{self.max_extrapolation_requests:g}")
            return base
        base["available"] = True
        base["value"] = fit.predict(point)
        base["se"] = fit.predict_se(point)
        return base

    @staticmethod
    def _too_far(fit, worst: str, off: dict, distance: str,
                 flag: str) -> str:
        r = fit.ranges.get(worst) or {}
        why = (f"the operating point is {distance} outside the probed range of "
               f"`{worst}` (probed {r.get('min', float('nan')):.3g}-"
               f"{r.get('max', float('nan')):.3g}, operating point "
               f"{off.get('value', float('nan')):.3g}), above {flag}. The "
               "endpoint was never carrying the load the prediction is about")
        return f"{why}; note that {UPWARD_BIAS}" if off.get("above") else why

    def to_dict(self) -> dict:
        return {"fits": {k: f.to_dict() for k, f in self.fits.items()},
                "operating_point": self.op.to_dict(),
                "max_extrapolation": self.max_extrapolation,
                "max_extrapolation_requests": self.max_extrapolation_requests,
                "probe_in_flight_bound": self.probe_in_flight_bound,
                "natural_ladder": [_clean_row(r) for r in self.ladder],
                "windows": self.windows, "cross_check": self.cross,
                "governor": self.governor, "aborted": self.aborted,
                "n_covariate_rows": self.n_covariate_rows,
                "lengths_ktok": self.lengths_ktok, "options": self.options,
                # `fit` is dropped from each reading: the same dict is already
                # under "fits", and a record that carries it twice invites the
                # two copies to disagree
                "readings": {k: _clean_row({x: y for x, y in
                                            self.reading(k).items()
                                            if x != "fit"})
                             for k in self.fits}}


def covariate_rows(traces: list) -> list[dict]:
    """One row per successful, covariate-stamped request — the fit's input.

    A trace with no `covariates` (no metrics sampler at send time) yields a
    row with `running`/`waiting` absent, which `fit_covariates` DROPS. That
    is the mechanism by which a run without `--metrics-url` fits nothing and
    keeps the old `not_established` cap.
    """
    rows = []
    for t in traces:
        if t.error or t.ttft is None:
            continue
        cov = t.covariates or {}
        ptok = t.ptok_achieved or t.ptok_intended or 0
        l_ktok = ptok / 1e3
        # `running_adjusted` when the probe resolved its own contribution to
        # the gauge (see `_stamp_own_load`); the raw gauge otherwise, which is
        # what a trace replayed from an older record carries
        running = cov.get("running_adjusted")
        if running is None:
            running = cov.get("requests_running")
        rows.append({
            "kind": t.kind, "L_ktok": l_ktok, "L_ktok2": l_ktok * l_ktok,
            "running": running,
            "running_reported": cov.get("requests_running"),
            "probe_open_after_scrape": cov.get("probe_open_after_scrape"),
            "waiting": cov.get("requests_waiting"),
            "kv_usage": cov.get("kv_cache_usage"),
            "ttft": t.ttft,
            "itl_ms": t.itl_p50 * 1e3 if t.itl_p50 is not None else None,
            "decode_tok_s": t.clean_decode_tps,
            "ptok": ptok, "cached_tokens": t.cached_tokens,
        })
    return rows


def build_fits(rows: list[dict]) -> dict:
    """The three fits a shared run attempts.

      ttft_miss   TTFT ~ 1 + L + L^2 + running + waiting, over FORCED MISSES
                  only. Warm turns have a different service time by
                  construction (`prefill_service_moments` splits E[S|miss]
                  from E[S|hit]), so mixing them would fit neither.
      ttft_hit    the same shape over warm turns. NO HYPOTHESIS SCORES THIS
                  ONE — the registry has nothing that claims a warm-hit TTFT
                  — so the report prints it against `Predictions.ttft_hit_s`
                  as an explicitly UNSCORED cross-check rather than leaving a
                  number with nothing to compare it to. It costs nothing: the
                  warm turns are sent anyway for the `itl` and `decode` fits,
                  and the hit/miss split is exactly what the M/G/1 model
                  brackets (`prefill_ttft_seconds` splits E[S|hit] from
                  E[S|miss], and a run where one lands and the other does not
                  is telling you which half of the service-time model is
                  wrong).
      itl         normal inter-token gap [ms] ~ 1 + running + waiting. The
                  decode step's cost does not depend on THIS request's prompt
                  length the way prefill does, so L is not a regressor here;
                  the reduced model is stated rather than fitted-and-dropped.
      decode      freeze-excluded decode rate [tok/s], same reduced shape. It
                  is what H-steady is scored on. LOCALLY LINEAR by
                  construction: the model's own decode curve is ~C/(a + b n),
                  so this is a linearisation valid over the probed range of
                  `running` — which is exactly what the extrapolation gate
                  enforces.
    """
    miss = [r for r in rows if r["kind"] in ("miss", "first")]
    hit = [r for r in rows if r["kind"] == "hit"]
    fits = {}
    fits["ttft_miss"] = fit_covariates(
        [{**r, "y": r["ttft"]} for r in miss], TTFT_COLUMNS,
        target="forced-miss TTFT", unit="s")
    fits["ttft_hit"] = fit_covariates(
        [{**r, "y": r["ttft"]} for r in hit], TTFT_COLUMNS,
        target="warm-hit TTFT", unit="s")
    fits["itl"] = fit_covariates(
        [{**r, "y": r["itl_ms"]} for r in rows], LOAD_COLUMNS,
        target="normal inter-token gap", unit="ms")
    fits["decode"] = fit_covariates(
        [{**r, "y": r["decode_tok_s"]} for r in rows], LOAD_COLUMNS,
        target="freeze-excluded decode rate", unit="tok/s")
    return fits


def _miss_prompt(rng: random.Random, prefix: str, prefix_tokens: int,
                 tokens: int, cpt: float) -> str:
    """A forced miss at ~`tokens` prompt tokens: a random salt AHEAD of the
    byte-stable prefix makes the whole request unmatchable, exactly as
    `Session.next_turn` does for a miss."""
    body = make_text(rng, max(tokens - prefix_tokens, 1), cpt)
    return f"[miss-salt {rng.getrandbits(64):016x}] {prefix}\n{body}"


async def _one(client, ep, opts, gov: ProbeGovernor, metrics, traces: list,
               prompt: str, kind: str, max_tokens: int,
               cpt: float) -> RequestTrace:
    """Send one probe request under the rails: charge the token budget, take
    an in-flight slot, read the server's gauges just before the send, then
    stream it.

    The gauges are read ONCE, just before the send. The post-send read this
    used to do re-examined `trace.covariates`, which was stamped at that same
    instant from the same snapshot — it could not observe anything new, and it
    counted the reading twice. The watchdog covers the window during a
    request; that is what it is for.
    """
    # a rail the canary or the watchdog tripped is picked up HERE, before the
    # next request goes out — the side tasks run on their own timers, so the
    # main loop learns of an abort at its next send rather than at the end
    gov.raise_if_aborted()
    intended = int(len(prompt) / cpt)
    gov.spend(intended)
    async with gov.slot():
        gov.raise_if_aborted()
        t_wall = sampler_now(metrics)
        snap = _covariates(metrics, t_wall)
        gov.observe(snap)
        tr = RequestTrace(uid=900_001, is_sub=False, kind=kind,
                          t_send=time.monotonic(), ptok_intended=intended)
        traces.append(tr)
        async with gov.in_flight(t_wall):
            # AT SEND TIME, not after: how many of OUR requests were already
            # open and sent after the snapshot behind the gauge, so the gauge
            # provably cannot contain them. Counted here because the set
            # changes while the request streams; applied to the trace once
            # `send_request` has stamped its covariates on it.
            own = gov.own_after(snap.get("t") if snap else None)
            await send_request(client, ep, opts, prompt, tr, max_tokens,
                               metrics)
            _stamp_own_load(tr, own)
        return tr


def _stamp_own_load(tr: RequestTrace, own: int) -> None:
    """Record the probe's own contribution to the load this request saw.

    `running` as the server reports it is background traffic plus however many
    of ours the scrape behind it happened to catch. `own` is the part it
    provably missed, and `running_adjusted` is the total the request actually
    faced — a consistently defined regressor, which is what an OLS coefficient
    needs. The remaining ambiguity is bounded by the in-flight cap and is
    reported with the fit.
    """
    cov = tr.covariates
    if not cov:
        return
    cov["probe_open_after_scrape"] = own
    r = cov.get("requests_running")
    if r is not None and _fin(r):
        cov["running_adjusted"] = float(r) + own


async def _canary_loop(client, ep, opts, gov: ProbeGovernor, metrics,
                       traces: list, stop: asyncio.Event) -> None:
    """A 1-token request every `canary_every_s`. Its TTFT is the client-side
    baseline: a tiny, byte-stable prompt has no prefill of its own worth
    speaking of, so what moves it is the queue in front of it."""
    while not stop.is_set():
        tr = await _one(client, ep, opts, gov, metrics, traces, CANARY_PROMPT,
                        "canary", 1, opts.chars_per_token)
        gov.note_canary(tr.t_send, tr.ttft)
        try:
            await asyncio.wait_for(stop.wait(), gov.budget.canary_every_s)
        except asyncio.TimeoutError:
            pass


async def _watchdog(gov: ProbeGovernor, metrics, stop: asyncio.Event) -> None:
    """Read the server's gauges on their own timer, so a rail can fire between
    two of our requests rather than only alongside one.

    NOT continuous coverage, and the docstring used to imply it was. The poll
    period bounds how often a rail is evaluated, and the gauge it evaluates is
    itself up to one scrape interval old, so the worst-case detection lag is
    the sum of the two — which `ProbeBudget.detection_lag` states and
    `--dry-run` prints. It is also what arms the metrics-loss rail: a sampler
    that stops answering shows up here as consecutive empty reads.
    """
    while not stop.is_set():
        gov.observe(_covariates(metrics, sampler_now(metrics)))
        try:
            await asyncio.wait_for(stop.wait(), gov.budget.gauge_poll_s)
        except asyncio.TimeoutError:
            pass


async def run_shared(client, ep, cfg, opts, prefixes, budget: ProbeBudget,
                     sopts: SharedOptions, metrics=None,
                     on_progress=None) -> SharedResult:
    """The shared-endpoint probe: a prompt-length ladder of forced misses and
    warm turns, every request stamped with the load the server was carrying,
    all of it under `budget`.

    Raises `BudgetAbort` when a rail trips; the exception carries the partial
    `SharedResult` on `.result`, so the run record can still say what was
    measured before the stop.
    """
    on_progress = on_progress or (lambda *_a, **_k: None)
    wl = cfg.workload
    cpt = opts.chars_per_token
    gov = ProbeGovernor(budget, metrics_expected=metrics is not None)
    traces: list[RequestTrace] = []
    windows: list[dict] = []
    rng = random.Random((sopts.seed << 21) ^ 0x5EED)
    cap = opts.context_cap_tokens
    floor = wl.system_prefix_tokens
    lengths = sorted({max(floor + 1, int(f * cap))
                      for f in sopts.length_fractions()})
    stop = asyncio.Event()
    side: list[asyncio.Task] = []
    t_start = sampler_now(metrics)

    if budget.canary:
        side.append(asyncio.create_task(
            _canary_loop(client, ep, opts, gov, metrics, traces, stop)))
    if metrics is not None and (budget.abort_if_waiting is not None
                                or budget.abort_if_kv_above is not None
                                or budget.max_metrics_gaps):
        side.append(asyncio.create_task(_watchdog(gov, metrics, stop)))

    abort: BudgetAbort | None = None
    try:
        deadline = (time.monotonic() + sopts.duration_s if sopts.ladder
                    else math.inf)
        rounds = 10**9 if sopts.ladder else max(1, sopts.rounds)
        warm_history = ""
        for r in range(rounds):
            if time.monotonic() >= deadline:
                break
            on_progress("shared-round", (r + 1, len(lengths)))
            t_round, i0 = sampler_now(metrics), len(traces)
            for n_tok in lengths:
                await _one(client, ep, opts, gov, metrics, traces,
                           _miss_prompt(rng, prefixes.user, floor, n_tok, cpt),
                           "miss", wl.max_output_tokens, cpt)
            # warm turns: the SAME byte-stable prefix and a growing history,
            # so the server's prefix cache is what answers them
            for _ in range(max(0, sopts.warm_turns)):
                if len(warm_history) / cpt > 0.5 * cap:
                    # a --shared-ladder run cycles for minutes; an unbounded
                    # history would walk the warm turn up to the context cap
                    # and turn the cheapest probe in the run into its most
                    # expensive. The session restarts instead, which is also
                    # what a real agentic session does at its cap.
                    warm_history = ""
                warm_history += "\n" + make_text(rng, wl.warm_turn_tokens, cpt)
                await _one(client, ep, opts, gov, metrics, traces,
                           prefixes.user + "\n" + warm_history, "hit",
                           wl.max_output_tokens, cpt)
            # ONE window per round, over the round's OWN requests: a window
            # is a comparison between the client's view and the server's over
            # the same stretch of time, so it must not be handed traces from
            # outside it. Canaries are excluded — they are the safety signal,
            # not part of the measurement.
            w = await cross_check(metrics, t_round,
                                  sampler_now(metrics),
                                  [t for t in traces[i0:]
                                   if t.kind != "canary"])
            if w is not None:
                windows.append(w)
            if sopts.ladder and time.monotonic() >= deadline:
                break
    except BudgetAbort as e:
        abort = e
    finally:
        stop.set()
        for task in side:
            task.cancel()
        for got in await asyncio.gather(*side, return_exceptions=True):
            if isinstance(got, BudgetAbort) and abort is None:
                abort = got
        if abort is None and gov.aborted is not None:
            abort = gov.aborted

    # the WHOLE-RUN window first, so the Sample's `server` block is over the
    # same stretch as every other statistic on it
    overall = await cross_check(metrics, t_start, sampler_now(metrics),
                                [t for t in traces if t.kind != "canary"])
    result = _assemble(cfg, opts, sopts, gov, traces, windows, lengths,
                       overall=overall)
    if abort is not None:
        result.aborted = abort.reason
        abort.result = result
        raise abort
    return result


def _assemble(cfg, opts, sopts, gov, traces, windows, lengths, preds=None,
              overall=None) -> SharedResult:
    """Turn the traces into fits, an operating point, a ladder and a Sample.

    Pure apart from `predict`, so a test can hand it synthetic traces.

    `overall` is the WHOLE-RUN metrics window. The Sample's `server` block was
    the LAST ROUND's window, which is a different question from the one the
    rest of the Sample answers — every other statistic on it is over the whole
    run — so the two disagreed for any run of more than one round. The
    per-round windows stay in `windows`, where their scope is in the name.
    """
    from .predict import predict

    probe_traces = [t for t in traces if t.kind != "canary"]
    scored = covariate_rows(probe_traces)
    preds = preds if preds is not None else predict(cfg, n_iter=64, seed=0)
    fits = build_fits(scored)
    op = operating_point_covariates(cfg, preds)
    ladder = natural_ladder(scored)
    if sopts.ladder:
        # the model's own curve at each observed concurrency, which is what
        # the bins are read against. Only under --shared-ladder: each bin
        # costs a bisection over `steady_decode_point`, and a run that did
        # not ask for the ladder should not pay for it.
        for b in ladder:
            if b["enough"]:
                b["model"] = ladder_model_curve(cfg, b["running_mean"])
    sample = eval_sample(probe_traces,
                         server=(overall or {}).get("window"),
                         cap_tokens=opts.context_cap_tokens)
    return SharedResult(
        fits=fits, op=op, max_extrapolation=sopts.max_extrapolation,
        max_extrapolation_requests=sopts.max_extrapolation_requests,
        verdict_sigmas=sopts.verdict_sigmas,
        probe_in_flight_bound=gov.budget.max_extra_load,
        ladder=ladder, windows=windows, governor=gov.to_dict(), cross=overall,
        n_covariate_rows=sum(1 for r in scored
                             if r.get("running") is not None),
        lengths_ktok=[round(n / 1e3, 2) for n in lengths],
        sample=sample, options=sopts.to_dict())


# ============================================================================
# printing
# ============================================================================
def _planned_tokens(cfg, opts, sopts: SharedOptions, lengths,
                    rounds: int) -> tuple[int, int]:
    """(forced-miss tokens, warm-turn tokens) the plan will send.

    The warm half is not a rounding error and used to be left out of the
    figure compared against `--max-probe-tokens`: a warm turn carries the
    shared prefix plus every warm turn before it, so its cost GROWS within a
    round and across rounds until the history resets at half the context cap
    — the same rule `run_shared` applies, mirrored here rather than guessed.
    """
    wl, cap = cfg.workload, opts.context_cap_tokens
    miss = sum(lengths) * rounds
    warm, history = 0, 0
    for _ in range(rounds):
        for _ in range(max(0, sopts.warm_turns)):
            if history > 0.5 * cap:
                history = 0
            history += wl.warm_turn_tokens
            warm += wl.system_prefix_tokens + history
    return int(miss), int(warm)


def plan_lines(cfg, opts, sopts: SharedOptions, budget: ProbeBudget,
               metrics: bool) -> list[str]:
    """The `--dry-run` shared-mode block: what the probe will send, and what
    it will and will not be able to conclude from it."""
    cap = opts.context_cap_tokens
    lengths = sorted({max(cfg.workload.system_prefix_tokens + 1, int(f * cap))
                      for f in sopts.length_fractions()})
    warm = max(0, sopts.warm_turns)
    per_round = len(lengths) + warm
    rounds = 1 if sopts.ladder else max(1, sopts.rounds)
    if sopts.ladder:
        shape = (f"cycling for {sopts.duration_s:g}s (--shared-ladder), "
                 f"{per_round} requests per cycle")
    else:
        shape = (f"{sopts.rounds} round(s) x {per_round} requests = "
                 f"{sopts.rounds * per_round} requests")
    miss_tok, warm_tok = _planned_tokens(cfg, opts, sopts, lengths, rounds)
    tok = miss_tok + warm_tok
    out = [
        f"lengths        : {', '.join(f'{n / 1e3:.1f}k' for n in lengths)} "
        f"prompt tokens (fractions {sopts.lengths} of the {cap:,}-token cap)",
        f"shape          : {shape}",
        # EVERY planned send, not just the forced misses: this number is
        # compared against --max-probe-tokens below, and a warm turn carries
        # the shared prefix plus a growing history, which is not free
        f"planned cost   : ~{tok:,} intended prompt tokens "
        f"({miss_tok:,} forced miss + {warm_tok:,} warm), plus ~1 token per "
        "canary request",
        "fit            : TTFT = c0 + c1 L + c2 L^2 + c3 running "
        "+ c4 waiting  (L in kilotokens, OLS, quadratic centred on the "
        "probed mean L)",
        f"gate           : a verdict needs a usable fit, an extrapolation "
        f"distance <= {sopts.max_extrapolation:g} sd (--max-extrapolation) "
        f"AND <= {sopts.max_extrapolation_requests:g} requests "
        f"(--max-extrapolation-requests), and must survive "
        f"+/-{sopts.verdict_sigmas:g} standard errors of the fitted value "
        f"(--verdict-sigmas)",
        f"own load       : the probe adds at most "
        f"{budget.max_extra_load or 'unbounded'} request(s) of its own; the "
        "part of that the server's gauge already counted is resolved against "
        "the scrape's own timestamp, and the residual ambiguity is bounded by "
        "that cap",
    ]
    if not sopts.ladder and budget.max_probe_tokens and \
            tok > budget.max_probe_tokens:
        out.append(
            f"WARNING        : the planned {tok:,} prompt tokens exceed "
            f"--max-probe-tokens {budget.max_probe_tokens:,}. The run will "
            "abort partway through, leaving a fit with fewer observations "
            "than the plan implies. Raise the budget or drop a round.")
    if len(lengths) < 3:
        out.append(
            f"WARNING        : the ladder collapsed to {len(lengths)} "
            f"distinct length(s) — the requested fractions clip to the "
            f"{cfg.workload.system_prefix_tokens:,}-token prefix floor. c1 "
            "and c2 need at least three distinct lengths to be identified, "
            "so the TTFT fit will refuse. Raise --context-cap-tokens, or "
            "spread --shared-lengths above the floor.")
    if not metrics:
        out.append(
            "NOTE           : no --metrics-url, so `running`/`waiting` are "
            "never stamped, no fit is possible, and every cheap hypothesis "
            "stays not_established exactly as before. The canary still runs.")
    return out



# ============================================================================
# small helpers
# ============================================================================
def _fin(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) \
        and math.isfinite(x)


def _f(x) -> float:
    return float(x) if x is not None else float("nan")


def _num(x):
    return None if isinstance(x, float) and not math.isfinite(x) else x


def _clean_row(d: dict):
    if isinstance(d, dict):
        return {k: _clean_row(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_clean_row(v) for v in d]
    return _num(d) if isinstance(d, float) else d
