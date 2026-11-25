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
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field

import numpy as np

from .probe.population import Sample, eval_sample
from .probe.request import (RequestTrace, _covariates, sampler_now,
                           sampler_ready, send_request, window_dict)
from .probe.session import make_text, nonce_bits
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

# LOCAL SUPPORT. Every gate above is about the design as a whole; none of them
# asks whether there is evidence NEAR the point being evaluated. A two-level
# design -- `running` only ever 0 or 10, 120 observations of a response that is
# hyperbolic in load -- passes all of them (extrapolation 0 sd, max VIF 1.05,
# scaled cond 1.26, se 0.002) and reads 7.54 s where the truth at running=5 is
# 3.00 s. A straight line through the two ends of a convex curve is a chord,
# and nothing global can see that.
#
#   MIN_LOCAL_N / LOCAL_RADIUS_SD  at least this many observations within this
#       standardised radius of the evaluation point, measured over the LOAD
#       columns only. The two-level design has 0 within 0.75 sd; a three-level
#       one has 18 and a continuous one 22, so the gate separates them by more
#       than an order of magnitude.
#
# The length columns are deliberately excluded from that neighbourhood: the
# probe DESIGNS them (a deliberate ladder) and models their nonlinearity
# explicitly with the quadratic, while the load columns are observational and
# their curvature is exactly what is unmodelled.
MIN_LOCAL_N = 3
LOCAL_RADIUS_SD = 0.75

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
    min_local_n: int = MIN_LOCAL_N
    local_radius: float = LOCAL_RADIUS_SD
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
    abort_if_waiting   abort the moment the server's `requests_waiting` gauge,
                       NET of the probe's own requests still waiting for a
                       first token (`ProbeGovernor.own_in_prefill`),
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
                       `recent > canary_drift * base`. Canaries that waited
                       behind the probe's OWN prefill are in neither window.
    """
    max_extra_load: int = 2
    abort_if_waiting: float | None = 0.0
    abort_if_kv_above: float | None = 0.90
    max_metrics_gaps: int = 3
    # None = trust the sampler's own `stale` flag, which it computes from its
    # scrape interval; a number overrides it in seconds
    max_gauge_age_s: float | None = None
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
                    max_gauge_age_s=None, max_probe_tokens=0, canary=False,
                    exclusive=True)
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
        gauge = ("" if _have_metrics(metrics)
                 else "  [NOT ENFORCEABLE: no --metrics-url, so the server's "
                      "gauges are unreadable]")
        tok = (f"{self.max_probe_tokens:,} intended prompt tokens for the "
               "whole run" if self.max_probe_tokens else "uncapped")
        return [
            f"in flight      : at most {cap} of our requests, canary included",
            f"abort waiting  : requests_waiting > "
            f"{'off' if self.abort_if_waiting is None else f'{self.abort_if_waiting:g}'}"
            ", counted NET of the probe's own requests that were sent and "
            "still without a first token when the gauge was read (its own "
            "canary queues behind its own long miss; the rail is blind to a "
            "queue no deeper than that, "
            + (f"at most {cap}" if self.max_extra_load
               else "unbounded without --max-extra-load") + ")"
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
               f"{self.canary_min_n} samples). A canary that waited while a "
               "non-canary request of ours was still in prefill is recorded "
               "but NOT counted: that delay is the probe's own"
               if self.canary else "off"),
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
        iv = (getattr(metrics, "interval", None)
              if _have_metrics(metrics) else None)
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


@dataclass
class _OwnRequest:
    """One of the probe's own requests, as the governor tracks it.

    `t_sent` is in the SAMPLER's base, because it is compared with a
    snapshot's timestamp. `trace` is the live `RequestTrace`: its `ttft` is
    filled in the moment the first token arrives, which is how the governor
    knows whether this request is still in prefill — queued or being
    prefilled, the two states in which it can hold up somebody's first token,
    our own canary's included.
    """
    t_sent: float
    trace: RequestTrace | None = None
    canary: bool = False
    contended: bool = False           # canaries only, see `in_flight`
    t_done: float | None = None       # sampler base; None while open

    def awaiting_first_token(self) -> bool:
        tr = self.trace
        return tr is not None and tr.ttft is None and not tr.error

    def t_first(self) -> float | None:
        """First-token instant, sampler base; None while there is none."""
        tr = self.trace
        if tr is None or tr.ttft is None:
            return None
        return self.t_sent + tr.ttft

    def in_prefill_at(self, t: float) -> bool:
        """Sent by `t` and still without a first token at `t`. A request that
        ended with none (an error) was in prefill until it ended."""
        if self.t_sent > t or self.trace is None:
            return False
        end = self.t_first()
        if end is None:
            end = self.t_done
        return end is None or end > t


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
        self.metrics_gaps = 0             # consecutive unusable reads
        self.max_metrics_gaps_seen = 0
        self.last_metrics_gap = ""
        self.n_engines = 0
        # our OWN in-flight requests, by the sampler-base instant each was
        # sent at. What the server's gauge says about them depends on whether
        # the scrape behind it predates them, which is what `own_after` uses
        # this to resolve.
        self._open: dict[int, _OwnRequest] = {}
        # ...and the ones that recently finished: a gauge is up to a scrape
        # old, so a request of ours it counted may be over by the time the
        # reading is used. Bounded — only the last scrape interval matters.
        self._closed: deque[_OwnRequest] = deque(maxlen=64)
        self._next_id = 0
        self.peak_in_flight = 0
        # the waiting gauge NET of our own requests, which is what the rail
        # is keyed on (see `observe`); `peak_waiting` stays the raw gauge
        self.peak_waiting_net: float | None = None
        self.peak_own_in_prefill = 0
        # canaries that waited behind the probe's OWN prefill: recorded,
        # flagged, and kept out of the drift statistic (see `note_canary`)
        self.canary_contended: list[tuple[float, float]] = []

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
    async def in_flight(self, t_sent: float, trace: RequestTrace | None = None,
                        canary: bool = False, clock=None):
        """Mark one of our requests open, from `t_sent` (SAMPLER base).

        `trace` lets the governor see when this request gets its first token;
        `clock` (the sampler's) stamps when it ended. Yields the record, whose
        `contended` flag answers, for a CANARY, "did this wait behind the
        probe's own prefill?" — true when a non-canary request of ours was
        still without a first token at any point while the canary was
        waiting for its own. A scheduler that hands a running prefill the
        whole token budget holds a 1-token canary back for as long as our own
        long miss prefills, and that delay says nothing about anybody else.
        A request of ours that is already DECODING does not count: it takes
        one token of the budget per step and delays nothing.
        """
        rid, self._next_id = self._next_id, self._next_id + 1
        rec = _OwnRequest(t_sent=t_sent, trace=trace, canary=canary)
        if canary:
            rec.contended = any(o.awaiting_first_token()
                                for o in self._open.values() if not o.canary)
        else:
            for o in self._open.values():
                if o.canary and o.awaiting_first_token():
                    o.contended = True
        self._open[rid] = rec
        self.peak_in_flight = max(self.peak_in_flight, len(self._open))
        try:
            yield rec
        finally:
            self._open.pop(rid, None)
            if clock is not None:
                try:
                    rec.t_done = float(clock())
                except Exception:               # noqa: BLE001 -- a clock
                    rec.t_done = None
            self._closed.append(rec)

    def own_in_prefill(self, snapshot_t: float | None) -> int:
        """How many of OUR requests were sent and still without a first token
        when the gauge was read — the most the server's `requests_waiting`
        can owe to the probe itself.

        With the snapshot's own timestamp this is resolved AT that instant,
        over requests still open and ones that have ended since: a canary the
        scrape caught waiting is usually over by the time the reading is
        used. Without a timestamp it is the requests awaiting a first token
        right now.
        """
        if snapshot_t is None or not _fin(snapshot_t):
            return sum(1 for o in self._open.values()
                       if o.awaiting_first_token())
        return sum(1 for o in (*self._open.values(), *self._closed)
                   if o.in_prefill_at(snapshot_t))

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
        return sum(1 for o in self._open.values() if o.t_sent > snapshot_t)

    def own_finished_since(self, snapshot_t: float | None) -> int:
        """How many of OUR requests were open when `snapshot_t` was read and
        have FINISHED since, and so are in the gauge that snapshot carries
        without being in the server any more.

        The mirror of `own_after`, and the common case rather than a corner:
        the probe sends back to back, and the reading stamped on a request is
        up to one scrape old, so it still shows the probe's own PREVIOUS
        request as running. Left in, every `running` covariate sat about one
        request above what the server was carrying at the send — and by the
        same amount on every row, which an OLS reads as a shifted intercept
        and the extrapolation gate reads as load that was never there.

        Only requests whose end was stamped (`in_flight(clock=...)`) can be
        resolved. Whether the gauge had such a request under `running` or
        under `waiting` at that instant is not knowable from here; it is
        taken off `running`, where a request spends almost all of its life,
        and the adjusted count is clamped at 0.
        """
        if snapshot_t is None or not _fin(snapshot_t):
            return 0
        return sum(1 for o in self._closed
                   if o.t_done is not None
                   and o.t_sent <= snapshot_t < o.t_done)

    # ---- token budget ---------------------------------------------------
    def spend(self, tokens: int) -> None:
        """Charge `tokens` INTENDED prompt tokens against the run's budget.

        Raises before the request is sent, so the cap is a cap and not a
        high-water mark.
