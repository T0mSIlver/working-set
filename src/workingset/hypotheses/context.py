"""RunContext — the probes a run may perform, each performed at most once.

Several hypotheses read the same ladder: it is run once and cached, keyed by
the parameters that change what it measures. The context also carries what a
hypothesis needs to know about the run itself: whether it is `exclusive`
(allowed to generate the population), whether a metrics sampler is attached,
and whether the ladder is being run at all (a cheap hypothesis reads the
ladder when one exists and falls back to its own handful of requests when it
does not).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..probe.burst import BurstResult, run_burst
from ..probe.ladder import build_ladder
from ..probe.population import Rung, Sample, run_population, run_sample
from ..probe.session import build_prefixes


@dataclass
class LadderView:
    """The bracket arithmetic every ceiling hypothesis reads, computed once.

    Ported from the bracket / failure-mode split in `print_report`
    (scripts/validate_deployment.py). A partial rung is orientation, never
    evidence, and is excluded here.
    """
    rungs: list = field(default_factory=list)

    @property
    def full(self) -> list:
        return [r for r in self.rungs if not r.partial]

    @property
    def lo(self) -> int | None:
        """Largest passing population."""
        p = [r.pop for r in self.full if r.passed]
        return max(p) if p else None

    @property
    def hi(self) -> int | None:
        """Smallest failing population."""
        f = [r.pop for r in self.full if not r.passed]
        return min(f) if f else None

    def _fails_on(self, needle: str) -> list[int]:
        return [r.pop for r in self.full
                if any(needle in s for s in r.reasons)]

    def _lo_for(self, needle: str) -> int | None:
        return max((r.pop for r in self.full
                    if r.passed or not any(needle in s for s in r.reasons)),
                   default=None)

    @property
    def ttft_fails(self) -> list[int]:
        return self._fails_on("TTFT")

    @property
    def decode_fails(self) -> list[int]:
        return self._fails_on("decode")

    @property
    def latency_lo(self) -> int | None:
        return self._lo_for("TTFT")

    @property
    def decode_lo(self) -> int | None:
        return self._lo_for("decode")

    @property
    def saturation_evidence(self):
        """Rungs where achieved req/s fell well under the offered rate — the
        throughput plateau a closed loop shows instead of duty = 100%."""
        ev = [r for r in self.full if not r.passed and r.n_turns > 0
              and math.isfinite(r.achieved_rps)
              and r.achieved_rps < 0.7 * r.offered_rps]
        return min(ev, key=lambda r: r.pop) if ev else None

    @property
    def warm_held(self) -> list[int]:
        """Populations whose hit turns stayed warm (low effective-cold)."""
        return [r.pop for r in self.full if r.n_hit > 0
                and (not math.isfinite(r.evict_frac) or r.evict_frac < 0.05)]

    @property
    def warm_evicted(self):
        ev = [r for r in self.full if math.isfinite(r.evict_frac)
              and r.evict_frac >= 0.05]
        return min(ev, key=lambda r: r.pop) if ev else None

    def nearest(self, pop: float, predicate=None):
        cand = [r for r in self.full if predicate is None or predicate(r)]
        return min(cand, key=lambda r: abs(r.pop - pop)) if cand else None


class RunContext:
    """Probe access + run mode. `client` is an httpx.AsyncClient (or the test
    double); `metrics` is anything with `at(t)` and `window(t0, t1)`."""

    def __init__(self, cfg, predictions, opts, ep, client=None, metrics=None,
                 exclusive: bool = False, burst: int = 0,
                 burst_users: int = 0, run_ladder: bool = False,
                 on_progress=None):
        self.cfg = cfg
        self.predictions = predictions
        self.opts = opts
        self.ep = ep
        self.client = client
        self.metrics = metrics
        self.exclusive = exclusive
        self.burst_n = burst
        self.burst_users = burst_users
        self.run_ladder = run_ladder
        self.on_progress = on_progress or (lambda *_a, **_k: None)
        self.prefixes = build_prefixes(cfg.workload, opts.chars_per_token)
        self._cache: dict = {}
        self.partial = None            # (pop, n_sub, traces, measure_start)

    # ---- cache keys ----------------------------------------------------
    def _ladder_key(self) -> tuple:
        return ("ladder", self.opts.ladder_key(),
                self.predictions.predicted_limit_users,
                self.predictions.operating_point_users)

    def _sample_key(self) -> tuple:
        return ("sample", self.opts.ladder_key(), self.opts.sample_requests,
                self.opts.sample_warm_turns)

    def _burst_pop(self) -> int:
        return self.burst_users or max(1, round(
            self.predictions.operating_point_users
            or 0.5 * self.predictions.predicted_limit_users))

    def _burst_key(self) -> tuple:
        return ("burst", self.opts.ladder_key(), self.burst_n, self._burst_pop())

    def seed(self, rungs=None, sample=None, burst=None) -> "RunContext":
        """Pre-fill the cache with probe results measured elsewhere — a replay
        of a stored run, or a synthetic fixture. A seeded probe is never
        re-run."""
        if rungs is not None:
            self._cache[self._ladder_key()] = LadderView(list(rungs))
        if sample is not None:
            self._cache[self._sample_key()] = sample
        if burst is not None:
            self._cache[self._burst_key()] = burst
        return self

    # ---- probes, cached by parameters ----------------------------------
    def ladder_pops(self) -> list[int]:
        return build_ladder(self.predictions.predicted_limit_users,
                            self.opts.rungs, self.opts.max_users,
                            self.predictions.operating_point_users)

    async def ladder(self) -> LadderView:
        key = self._ladder_key()
        if key in self._cache:
            return self._cache[key]
        rungs: list[Rung] = []
        view = LadderView(rungs)
        self._cache[key] = view
        for pop in self.ladder_pops():
            self.on_progress("rung", pop)
            r = await run_population(self.client, self.ep, self.cfg, self.opts,
                                     pop, self.prefixes, self.metrics,
                                     on_partial=self._note_partial)
            self.partial = None
            rungs.append(r)
            self.on_progress("rung-done", r)
            if r.blown:
                # SLOs clearly blown — higher rungs would only stress the
                # endpoint, and the bracket is already closed above
                self.on_progress("blown", r)
                break
        return view

    def _note_partial(self, pop, n_sub, traces, measure_start):
        self.partial = (pop, n_sub, traces, measure_start)

    async def sample(self) -> Sample:
        key = self._sample_key()
        if key not in self._cache:
            self.on_progress("sample", self.opts.sample_requests)
            self._cache[key] = await run_sample(
                self.client, self.ep, self.cfg, self.opts, self.prefixes,
                self.metrics)
        return self._cache[key]

    async def burst(self) -> BurstResult:
        pop = self._burst_pop()
        key = self._burst_key()
        if key not in self._cache:
            self.on_progress("burst", (self.burst_n, pop))
            self._cache[key] = await run_burst(
                self.client, self.ep, self.cfg, self.opts, self.burst_n, pop,
                self.prefixes, self.metrics)
        return self._cache[key]

    # ---- what the cheap hypotheses read --------------------------------
    async def at_operating_point(self):
        """The best available reading at (or nearest) the operating point.

        With a ladder: the rung nearest the operating point that carries the
        statistic asked for — the harness's rule, because ttft_miss rises
        steeply with load and sampling it at the predicted LIMIT instead
        systematically failed a correct model. Without one: the cheap sample,
        which measures at whatever load the endpoint already carries.
        """
        if self.run_ladder:
            view = await self.ladder()
            return view
        return await self.sample()

    def cached(self) -> dict:
        """Everything measured so far, for the run record."""
        out = {"rungs": [], "sample": None, "burst": None}
        for key, val in self._cache.items():
            if key[0] == "ladder":
                out["rungs"] = [r.to_dict() for r in val.rungs]
            elif key[0] == "sample":
                out["sample"] = val.to_dict()
            elif key[0] == "burst":
                out["burst"] = val.to_dict()
        return out
