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
`kv_usage`. Wall-clock timestamps handed to a metrics sampler are UNIX
seconds (`time.time()`), which is the clock `MetricsSampler` stamps its
snapshots with; `RequestTrace.t_send` stays `time.monotonic()` because the
probe layer does span arithmetic with it.

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
from .probe.request import RequestTrace, _covariates, _plain, send_request
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

# Above this the design matrix is rank-deficient in practice: the regressors
# did not vary independently, so the coefficients are not separable however
# many digits numpy prints. 1e8 is ~half of float64's significand.
MAX_CONDITION = 1e8


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
    seed               RNG seed for the probe's synthetic text
    """
    lengths: str = DEFAULT_SHARED_LENGTHS
    rounds: int = 4
    warm_turns: int = 2
    ladder: bool = False
    duration_s: float = 300.0
    max_extrapolation: float = 1.0
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
                       canary included (0 = no cap). The canary needs a slot
                       of its own, so a budget with the canary on is clamped
                       to at least 2.
    abort_if_waiting   abort the moment the server's `requests_waiting` gauge
                       EXCEEDS this many requests (None = rail off). Needs
                       `--metrics-url`; without one the gauge is unreadable
                       and the rail cannot fire, which `--dry-run` says.
    abort_if_kv_above  abort when KV occupancy exceeds this FRACTION
                       (None = rail off). Also needs `--metrics-url`.
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
        """The shared-mode default: two requests in flight, abort on ANY
        queue the server reports, abort at 90% KV, 2M prompt tokens, canary
        on. Deliberately timid — the operator can raise every one of these,
        and `--dry-run` prints what they are set to."""
        return cls(**kw)

    @classmethod
    def for_exclusive(cls, **kw) -> "ProbeBudget":
        """`--exclusive` owns the endpoint: the rails come off, because the
        queue the ladder is about to build is the measurement, not a
        trespass. The token budget and the canary go with them."""
        base = dict(max_extra_load=0, abort_if_waiting=None,
                    abort_if_kv_above=None, max_probe_tokens=0, canary=False,
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

    @property
    def effective_max_load(self) -> int:
        """The in-flight cap actually enforced. A canary that can never win a
        slot is not a contention signal, so a capped budget with the canary
        on is raised to 2."""
        if not self.max_extra_load:
            return 0
        return max(2, self.max_extra_load) if self.canary else self.max_extra_load

    def describe(self, metrics: bool) -> list[str]:
        """The `--dry-run` block. `metrics` says whether the gauge rails can
        fire at all."""
        cap = self.effective_max_load or "uncapped"
        if self.exclusive:
            return [f"in flight      : {cap} (exclusive: this run owns the "
                    "endpoint, so the rails are off)",
                    "queue / KV     : not enforced",
                    "prompt tokens  : uncapped",
                    "canary         : off"]
        clamp = ("" if self.effective_max_load == self.max_extra_load
                 else f" (raised from {self.max_extra_load} to leave the "
                      "canary a slot)")
        gauge = ("" if metrics else "  [NOT ENFORCEABLE: no --metrics-url, so "
                                    "the server's gauges are unreadable]")
        tok = (f"{self.max_probe_tokens:,} intended prompt tokens for the "
               "whole run" if self.max_probe_tokens else "uncapped")
        return [
            f"in flight      : at most {cap} of our requests{clamp}",
            f"abort waiting  : requests_waiting > "
            f"{'off' if self.abort_if_waiting is None else f'{self.abort_if_waiting:g}'}"
            f"{gauge}",
            f"abort KV       : kv_cache_usage > "
            f"{'off' if self.abort_if_kv_above is None else f'{self.abort_if_kv_above:.0%}'}"
            f"{gauge}",
            f"prompt tokens  : {tok}",
            "canary         : "
            + (f"1-token request every {self.canary_every_s:g}s; abort when "
               f"its p50 over the last {self.canary_window_s:g}s exceeds "
               f"{self.canary_drift:g}x the p50 over the first "
               f"{self.canary_baseline_s:g}s (both need "
               f"{self.canary_min_n} samples)" if self.canary else "off"),
        ]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["effective_max_load"] = self.effective_max_load
        return d


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

    Every rail is checked on the way IN (before a request is sent) as well as
    from a watchdog reading the gauges on its own timer, so a rail can fire
    between two of our requests rather than only alongside one.
    """

    def __init__(self, budget: ProbeBudget):
        self.budget = budget
        cap = budget.effective_max_load
        self._sem = asyncio.Semaphore(cap) if cap else None
        self.tokens_spent = 0
        self.n_requests = 0
        self.canary_ttft: list[tuple[float, float]] = []   # (t_rel s, ttft s)
        self.t0 = time.monotonic()
        self.aborted: BudgetAbort | None = None
        self.n_gauge_checks = 0
        self.peak_waiting: float | None = None
        self.peak_kv: float | None = None

    # ---- in-flight cap --------------------------------------------------
    @asynccontextmanager
    async def slot(self):
        """Hold one of the `max_extra_load` in-flight slots."""
        if self._sem is None:
            yield
            return
        await self._sem.acquire()
        try:
            yield
        finally:
            self._sem.release()

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
        """One reading of `requests_waiting` / `kv_cache_usage`. `None` (no
        metrics sampler) is not a pass — it means the rail cannot fire, which
        the report says out loud."""
        if not covariates:
            return
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
                "peak_requests_waiting": self.peak_waiting,
                "peak_kv_cache_usage": self.peak_kv,
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
    condition_number  2-norm condition number of the design matrix. Large
                   means the regressors moved together and the split between
                   them is arbitrary.
    ranges         column -> {min, max, mean, sd} over the observations, which
                   is what an extrapolation distance is measured against
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
    r_squared: float = float("nan")
    ranges: dict = field(default_factory=dict)
    refused: str | None = None
    _cov: tuple = field(default=(), repr=False)      # (X'X)^-1, row-major

    # ---- use -------------------------------------------------------------
    @property
    def usable(self) -> bool:
        return self.refused is None

    def _row(self, point: dict) -> np.ndarray:
        return np.array([1.0 if c == "const" else float(point[c])
                         for c in self.columns], dtype=float)

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
        """(worst distance, per-column distances).

        A column's distance is how far `point` lies OUTSIDE the observed
        [min, max] of that column, divided by the column's observed standard
        deviation; zero inside the range. `const` is skipped. A column that
        never varied (sd = 0) gives inf outside its single observed value —
        a regressor with no spread supports no extrapolation at all.
        """
        per: dict[str, float] = {}
        for c in self.columns:
            if c == "const":
                continue
            r = self.ranges.get(c) or {}
            v = float(point[c])
            lo, hi, sd = r.get("min"), r.get("max"), r.get("sd")
            if lo is None or hi is None:
                per[c] = float("inf")
                continue
            out = max(0.0, v - hi, lo - v)
            if out <= 0:
                per[c] = 0.0
            elif sd and sd > 0:
                per[c] = out / sd
            else:
                per[c] = float("inf")
        worst = max(per.values()) if per else 0.0
        return worst, per

    def to_dict(self) -> dict:
        return {"target": self.target, "unit": self.unit,
                "columns": list(self.columns),
                "coefficients": dict(self.coefficients),
                "coefficient_units": {c: f"{self.unit} {_COLUMN_UNITS[c]}".strip()
                                      for c in self.columns},
                "n": self.n, "dof": self.dof,
                "residual_std": _num(self.residual_std),
                "condition_number": _num(self.condition_number),
                "r_squared": _num(self.r_squared),
                "ranges": dict(self.ranges), "refused": self.refused}

    def summary(self) -> str:
        if not self.usable:
            return f"{self.target}: no fit — {self.refused}"
        terms = " ".join(f"{c}={self.coefficients[c]:+.4g}"
                         for c in self.columns)
        return (f"{self.target} [{self.unit}]: {terms} | n={self.n} "
                f"resid sd {self.residual_std:.3g} R2 {self.r_squared:.2f} "
                f"cond {self.condition_number:.3g}")


def fit_covariates(rows: list[dict], columns=TTFT_COLUMNS, target: str = "y",
                   unit: str = "", min_obs_per_coef: int = MIN_OBS_PER_COEF,
                   max_condition: float = MAX_CONDITION) -> CovariateFit:
    """OLS of `row["y"]` on `columns`, refusing rather than guessing.

    `rows` is a list of dicts carrying every name in `columns` (bar `const`)
    plus `"y"`. Rows with a missing or non-finite entry are DROPPED — a
    request sent while no metrics snapshot existed carries no load reading,
    and imputing one would invent the covariate the whole design rests on.

    Refuses, with the reason in `refused` and what would fix it:
      * n below `min_obs_per_coef` x (number of coefficients), floor
        `MIN_OBS_FLOOR`
      * a rank-deficient design matrix (naming the columns that did not vary)
      * a condition number above `max_condition`
      * a degenerate residual (n == k: no degrees of freedom left)
    """
    k = len(columns)
    need = max(min_obs_per_coef * k, MIN_OBS_FLOOR)
    names = [c for c in columns if c != "const"]
    kept = []
    for r in rows:
        y = r.get("y")
        vals = [r.get(c) for c in names]
        if y is None or not _fin(y) or any(v is None or not _fin(v)
                                           for v in vals):
            continue
        kept.append(([1.0 if c == "const" else float(r[c]) for c in columns],
                     float(y)))
    n = len(kept)
    base = dict(target=target, unit=unit, columns=tuple(columns), n=n)
    if n < need:
        return CovariateFit(
            **base, refused=f"n={n} covariate-stamped observations, {need} "
                            f"needed for {k} coefficients ({min_obs_per_coef} "
                            f"per coefficient). Raise --shared-rounds, add "
                            f"--shared-ladder, or attach --metrics-url so the "
                            f"load columns exist at all")
    X = np.array([x for x, _ in kept], dtype=float)
    y = np.array([v for _, v in kept], dtype=float)
    ranges = {c: {"min": float(X[:, i].min()), "max": float(X[:, i].max()),
                  "mean": float(X[:, i].mean()), "sd": float(X[:, i].std(ddof=1))}
              for i, c in enumerate(columns) if c != "const"}
    base["ranges"] = ranges

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
    if not math.isfinite(cond) or cond > max_condition:
        return CovariateFit(
            **base, condition_number=cond,
            refused=f"condition number {cond:.3g} exceeds {max_condition:.3g}: "
                    f"the regressors moved together, so the split between "
                    f"their coefficients is arbitrary. Probe across a wider "
                    f"range of server load, or over more --shared-lengths")
    if n <= k:
        return CovariateFit(**base, condition_number=cond,
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
    return CovariateFit(**base, dof=dof, condition_number=cond,
                        residual_std=resid_std, r_squared=r2, _cov=cov_flat,
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
    refused: str | None = None

    def point(self, columns) -> dict:
        return {"const": 1.0, "L_ktok": self.L_ktok, "L_ktok2": self.L2_ktok2,
                "running": self.running, "waiting": self.waiting}

    def to_dict(self) -> dict:
        d = {k: _num(v) if isinstance(v, float) else v
             for k, v in asdict(self).items()}
        return d


def operating_point_covariates(cfg, preds, n_iter: int = 20_000,
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
    e_l, e_l2 = M.context_moments(wl, n=n_iter, seed=seed)
    return OperatingPoint(
        running=float(preds.steady_decode_seqs) + float(preds.prefill_duty),
        waiting=float(rate_total * wait_s),
        L_ktok=float(e_l) / 1e3, L2_ktok2=float(e_l2) / 1e6,
        steady_decode_seqs=float(preds.steady_decode_seqs),
        prefill_occupancy=float(preds.prefill_duty),
        rate_total_req_s=float(rate_total))


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

    out: dict = {"t0": t0, "t1": t1, "window": _plain_window(w)}
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


def _plain_window(w) -> dict | None:
    d = getattr(w, "to_dict", None)
    if d is not None:
        try:
            return d()
        except Exception:                       # noqa: BLE001
            return None
    return _plain(w)


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
        dist, per = fit.extrapolation(point)
        base["extrapolation"], base["extrapolation_by"] = dist, per
        if dist > self.max_extrapolation:
            worst = max(per, key=lambda c: per[c]) if per else "?"
            r = fit.ranges.get(worst) or {}
            base["reason"] = (
                f"the operating point is {dist:.2f} observed sd outside the "
                f"probed range of `{worst}` (probed "
                f"{r.get('min', float('nan')):.3g}-{r.get('max', float('nan')):.3g}, "
                f"operating point {point.get(worst, float('nan')):.3g}), above "
                f"--max-extrapolation {self.max_extrapolation:g}. The endpoint "
                "was never carrying the load the prediction is about")
            return base
        base["available"] = True
        base["value"] = fit.predict(point)
        base["se"] = fit.predict_se(point)
        return base

    def to_dict(self) -> dict:
        return {"fits": {k: f.to_dict() for k, f in self.fits.items()},
                "operating_point": self.op.to_dict(),
                "max_extrapolation": self.max_extrapolation,
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
        rows.append({
            "kind": t.kind, "L_ktok": l_ktok, "L_ktok2": l_ktok * l_ktok,
            "running": cov.get("requests_running"),
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
      ttft_hit    the same shape over warm turns.
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
    stream it."""
    # a rail the canary or the watchdog tripped is picked up HERE, before the
    # next request goes out — the side tasks run on their own timers, so the
    # main loop learns of an abort at its next send rather than at the end
    gov.raise_if_aborted()
    intended = int(len(prompt) / cpt)
    gov.spend(intended)
    async with gov.slot():
        gov.raise_if_aborted()
        # the gauges as of NOW, before we add to them. Wall clock: that is
        # what a MetricsSampler stamps its snapshots with.
        gov.observe(_covariates(metrics, time.time()))
        tr = RequestTrace(uid=900_001, is_sub=False, kind=kind,
                          t_send=time.monotonic(), ptok_intended=intended)
        traces.append(tr)
        await send_request(client, ep, opts, prompt, tr, max_tokens, metrics)
        gov.observe(tr.covariates)
        return tr


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
    """Read the server's gauges on their own timer, so a rail can fire
    between two of our requests — "at any tick", not "at any send"."""
    while not stop.is_set():
        gov.observe(_covariates(metrics, time.time()))
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
    gov = ProbeGovernor(budget)
    traces: list[RequestTrace] = []
    windows: list[dict] = []
    rng = random.Random((sopts.seed << 21) ^ 0x5EED)
    cap = opts.context_cap_tokens
    floor = wl.system_prefix_tokens
    lengths = sorted({max(floor + 1, int(f * cap))
                      for f in sopts.length_fractions()})
    stop = asyncio.Event()
    side: list[asyncio.Task] = []
    t_start = time.time()

    if budget.canary:
        side.append(asyncio.create_task(
            _canary_loop(client, ep, opts, gov, metrics, traces, stop)))
    if metrics is not None and (budget.abort_if_waiting is not None
                                or budget.abort_if_kv_above is not None):
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
            t_round, i0 = time.time(), len(traces)
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
            w = await cross_check(metrics, t_round, time.time(),
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

    result = _assemble(cfg, opts, sopts, gov, traces, windows, lengths)
    result.cross = await cross_check(
        metrics, t_start, time.time(),
        [t for t in traces if t.kind != "canary"])
    if abort is not None:
        result.aborted = abort.reason
        result.governor = gov.to_dict()
        abort.result = result
        raise abort
    return result


def _assemble(cfg, opts, sopts, gov, traces, windows,
              lengths, preds=None) -> SharedResult:
    """Turn the traces into fits, an operating point, a ladder and a Sample.

    Pure apart from `predict`, so a test can hand it synthetic traces.
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
                         server=(windows[-1].get("window") if windows
                                 else None),
                         cap_tokens=opts.context_cap_tokens)
    return SharedResult(
        fits=fits, op=op, max_extrapolation=sopts.max_extrapolation,
        ladder=ladder, windows=windows, governor=gov.to_dict(),
        n_covariate_rows=sum(1 for r in scored
                             if r.get("running") is not None),
        lengths_ktok=[round(n / 1e3, 2) for n in lengths],
        sample=sample, options=sopts.to_dict())


# ============================================================================
# printing
# ============================================================================
def plan_lines(cfg, opts, sopts: SharedOptions, budget: ProbeBudget,
               metrics: bool) -> list[str]:
    """The `--dry-run` shared-mode block: what the probe will send, and what
    it will and will not be able to conclude from it."""
    cap = opts.context_cap_tokens
    lengths = sorted({max(cfg.workload.system_prefix_tokens + 1, int(f * cap))
                      for f in sopts.length_fractions()})
    per_round = len(lengths) + max(0, sopts.warm_turns)
    if sopts.ladder:
        shape = (f"cycling for {sopts.duration_s:g}s (--shared-ladder), "
                 f"{per_round} requests per cycle")
    else:
        shape = (f"{sopts.rounds} round(s) x {per_round} requests = "
                 f"{sopts.rounds * per_round} requests")
    tok = sum(lengths) * (sopts.rounds if not sopts.ladder else 1)
    out = [
        f"lengths        : {', '.join(f'{n / 1e3:.1f}k' for n in lengths)} "
        f"prompt tokens (fractions {sopts.lengths} of the {cap:,}-token cap)",
        f"shape          : {shape}",
        f"per-round cost : ~{sum(lengths):,} prompt tokens of forced miss "
        f"({tok:,} for the planned rounds)",
        "fit            : TTFT = c0 + c1 L + c2 L^2 + c3 running "
        "+ c4 waiting  (L in kilotokens, OLS)",
        f"gate           : a verdict needs a usable fit AND an extrapolation "
        f"distance <= {sopts.max_extrapolation:g} observed sd "
        f"(--max-extrapolation)",
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
