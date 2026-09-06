"""The closed-loop population: N sessions thinking, sending and streaming.

Ported from `user_loop`, `run_rung` and `eval_rung` in
scripts/validate_deployment.py. `eval_rung` is a PURE function of the traces
and the measure-window start, so a rung's statistics can be (and are) tested
on synthetic traces without an endpoint.

Every derived statistic keeps the harness's definition, in particular:

  per-user p50 decode   median within each user, THEN p50 across users — the
                        model's "per-user p50" is a population statistic, not
                        a turn statistic.
  gap columns           per SSE EVENT; per-token rates (freezes/ktok, stall
                        ms/ktok) run only over turns that reported `usage`,
                        so numerator and denominator share a set.
  eviction              the baseline's heuristic classifier: a hit-intended
                        turn whose TTFT is >= 0.4x the median forced-miss TTFT
                        re-prefilled.
  SLO verdict           pass = p{percentile} TTFT <= budget AND per-user p50
                        decode >= floor AND error rate <= 5%.

ADDED over the harness: `server`, the metrics-sampler window delta over the
measure window, attached opaquely; and `cached_frac`, the share of hit turns
the SERVER reported a prefix-cache hit for (vLLM's
`usage.prompt_tokens_details.cached_tokens`) — evidence the harness's TTFT
heuristic had to stand in for.
"""
from __future__ import annotations

import asyncio
import math
import random
import time
from dataclasses import asdict, dataclass, field

from .request import EndpointSpec, RequestTrace, _plain, send_request
from .session import Prefixes, make_session
from .stats import FREEZE_LADDER_MS, pct, restore_nans


@dataclass
class Rung:
    pop: int                  # user-class sessions (the population axis)
    n_sub: int = 0
    n_turns: int = 0
    n_hit: int = 0
    n_miss: int = 0
    n_err: int = 0
    ttft_hit_p50: float = float("nan")
    ttft_hit_pX: float = float("nan")
    ttft_miss_p50: float = float("nan")
    ttft_miss_pX: float = float("nan")
    ttft_miss_mean: float = float("nan")
    ttft_all_pX: float = float("nan")
    decode_p50: float = float("nan")
    # --- ITL gap distribution ---
    itl_p50_ms: float = float("nan")        # the normal inter-token gap
    itl_worst_p50_ms: float = float("nan")  # typical worst freeze, per response
    itl_worst_p95_ms: float = float("nan")  # unlucky response's worst freeze
    # max-of-maxes: sample-size biased (the arm with more responses draws more
    # from the same tail), so it is a footnote, never a headline comparison
    itl_max_ms: float = float("nan")
    itl_floor_ms: float = float("nan")      # smallest gap seen — CLIENT FLOOR
    freeze_per_ktok: float = float("nan")   # freezes per 1k DECODED TOKENS
    stall_frac: float = float("nan")        # share of stream wall-time stalled
    stall_ms_per_ktok: float = float("nan")
    freeze_ladder: list = field(default_factory=list)
    chunk_tok_ratio: float = float("nan")   # tokens per SSE event (1.0 = per-token)
    achieved_rps: float = float("nan")
    offered_rps: float = float("nan")
    evict_frac: float = float("nan")   # hit-intended turns that look cold (heuristic)
    cached_frac: float = float("nan")  # hit turns the SERVER reported cached for
    ptok_ratio: float = float("nan")   # achieved / intended prompt tokens, median
    percentile: int = 95
    passed: bool = False
    reasons: list = field(default_factory=list)
    blown: bool = False                # SLOs clearly blown -> ladder stops here
    partial: bool = False              # interrupted mid-window: shown, never counted
    server: dict | None = None         # metrics window delta, opaque
    traces: list = field(default_factory=list, repr=False)

    def to_dict(self, traces: bool = True) -> dict:
        d = {k: v for k, v in asdict(self).items() if k != "traces"}
        d["traces"] = [t.to_dict() for t in self.traces] if traces else []
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Rung":
        d = restore_nans(cls, d)
        raw = d.pop("traces", []) or []
        known = {f for f in cls.__dataclass_fields__}
        r = cls(**{k: v for k, v in d.items() if k in known})
        r.traces = [RequestTrace.from_dict(t) for t in raw]
        return r


# ============================================================================
# rung statistics — pure, testable on synthetic traces
# ============================================================================
def eval_rung(pop: int, n_sub: int, traces: list, measure_start: float,
              cfg, opts, server: dict | None = None) -> Rung:
    slo, wl = cfg.slo, cfg.workload
    p = slo.percentile
    res = Rung(pop=pop, n_sub=n_sub, percentile=p, server=server)

    measured = [t for t in traces
                if t.t_send >= measure_start and t.kind in ("hit", "miss")]
    res.n_err = sum(1 for t in measured if t.error)
    ok = [t for t in measured if not t.error and t.ttft is not None]
    hit = [t for t in ok if t.kind == "hit"]
    miss = [t for t in ok if t.kind == "miss"]
    res.n_turns, res.n_hit, res.n_miss = len(ok), len(hit), len(miss)

    res.ttft_hit_p50 = pct([t.ttft for t in hit], 50)
    res.ttft_hit_pX = pct([t.ttft for t in hit], p)
    res.ttft_miss_p50 = pct([t.ttft for t in miss], 50)
    res.ttft_miss_pX = pct([t.ttft for t in miss], p)
    if miss:
        res.ttft_miss_mean = sum(t.ttft for t in miss) / len(miss)
    res.ttft_all_pX = pct([t.ttft for t in ok], p)

    # per-user decode: median within each user, then p50 across users
    by_user: dict[int, list] = {}
    for t in ok:
        if t.decode_tps is not None:
            by_user.setdefault(t.uid, []).append(t.decode_tps)
    res.decode_p50 = pct([pct(v, 50) for v in by_user.values()], 50)

    # ITL gap distribution. decode_p50 above answers "how fast on average";
    # these answer "what did it feel like" — the chunk-size hypothesis lives
    # entirely here, because chunking moves the tail and leaves the mean flat.
    g = [t for t in ok if t.n_gaps > 0 and t.span_s]
    if g:
        res.itl_p50_ms = pct([t.itl_p50 for t in g], 50) * 1e3
        worst = [t.itl_max for t in g]
        res.itl_worst_p50_ms = pct(worst, 50) * 1e3
        res.itl_worst_p95_ms = pct(worst, 95) * 1e3
        res.itl_max_ms = max(worst) * 1e3
        # the client floor: the smallest gap the loop was able to resolve. If
        # this is a material fraction of itl_p50_ms, the "normal gap" reading
        # is client-side scheduling, not server behaviour.
        res.itl_floor_ms = min(t.itl_min for t in g) * 1e3
        span = sum(t.span_s for t in g)
        if span > 0:
            res.stall_frac = sum(t.stall_s for t in g) / span
        # per-TOKEN rates need a token count, not a gap count: gaps are per
        # SSE event and an event may carry several tokens.
        g_tok = [t for t in g if t.ctok]
        ctoks = sum(t.ctok for t in g_tok)
        if ctoks:
            res.freeze_per_ktok = 1e3 * sum(t.n_freeze for t in g_tok) / ctoks
            res.stall_ms_per_ktok = 1e6 * sum(t.stall_s for t in g_tok) / ctoks
            res.chunk_tok_ratio = ctoks / sum(t.n_chunks for t in g_tok)
            res.freeze_ladder = [
                {"threshold_ms": thr,
                 "per_ktok": 1e3 * sum(t.n_freeze_at[i] for t in g_tok) / ctoks,
                 "stall_ms_per_ktok":
                     1e6 * sum(t.stall_at[i] for t in g_tok) / ctoks}
                for i, thr in enumerate(FREEZE_LADDER_MS)]

    res.achieved_rps = len(measured) / opts.measure_s if opts.measure_s else float("nan")
    res.offered_rps = (pop + n_sub) / wl.think_time_s

    # eviction heuristic (the baseline's warm/cold classifier)
    if hit and miss and math.isfinite(res.ttft_miss_p50):
        thresh = 0.4 * res.ttft_miss_p50
        res.evict_frac = sum(1 for t in hit if t.ttft >= thresh) / len(hit)
    # the server's own answer, where it gives one
    cached = [t for t in hit if t.cached_tokens is not None and t.ptok_achieved]
    if cached:
        res.cached_frac = sum(1 for t in cached
                              if t.cached_tokens > 0.5 * t.ptok_achieved) / len(cached)

    ratios = [t.ptok_achieved / t.ptok_intended for t in ok
              if t.ptok_achieved and t.ptok_intended]
    res.ptok_ratio = pct(ratios, 50)

    # SLO verdict for this rung
    budget, floor = slo.ttft_budget_s, slo.itl_floor_tok_s
    err_frac = res.n_err / max(len(measured), 1)
    if not ok:
        res.reasons.append("no successful turns in the measure window")
        res.blown = True
    else:
        if err_frac > 0.05:
            res.reasons.append(f"error rate {err_frac:.0%}")
        if res.ttft_all_pX > budget:
            res.reasons.append(f"p{p} TTFT {res.ttft_all_pX:.2f}s > {budget:g}s")
        if math.isfinite(res.decode_p50) and res.decode_p50 < floor:
            res.reasons.append(f"decode p50 {res.decode_p50:.1f} < {floor:g} tok/s")
        res.blown = (res.ttft_all_pX > 2 * budget
                     or (math.isfinite(res.decode_p50)
                         and res.decode_p50 < 0.5 * floor)
                     or err_frac > 0.20)
    res.passed = not res.reasons
    res.traces = list(traces)
    return res


# ============================================================================
# the closed loop
# ============================================================================
async def user_loop(client, ep: EndpointSpec, cfg, opts, uid: int, is_sub: bool,
                    prefixes: Prefixes, traces: list, stop: asyncio.Event,
                    stagger_s: float, metrics=None, seed: int | None = None):
    wl = cfg.workload
    session = make_session(wl, opts, prefixes, uid, is_sub, seed=seed)
    cap = opts.turns_per_user or math.inf

    await asyncio.sleep(stagger_s)     # spread session establishment over the ramp
    while not stop.is_set() and session.n_turn < cap:
        if session.n_turn:
            # exponential think time; wake early if the rung ends
            try:
                await asyncio.wait_for(
                    stop.wait(), session.rng.expovariate(1.0 / wl.think_time_s))
                break
            except asyncio.TimeoutError:
                pass
        prompt, kind = session.next_turn()
        tr = RequestTrace(uid=uid, is_sub=is_sub, kind=kind,
                          t_send=time.monotonic(),
                          ptok_intended=session.intended_prompt_tokens(prompt))
        traces.append(tr)
        reply = await send_request(client, ep, opts, prompt, tr,
                                   wl.max_output_tokens, metrics)
        session.commit(reply)


async def run_population(client, ep: EndpointSpec, cfg, opts, pop: int,
                         prefixes: Prefixes, metrics=None,
                         on_partial=None) -> Rung:
    """One ladder rung: `pop` user sessions plus `round(pop * r)` subagents,
    ramped in, then measured for `measure_s`."""
    wl = cfg.workload
    n_sub = round(pop * wl.subagent_ratio)
    traces: list[RequestTrace] = []
    stop = asyncio.Event()
    rng = random.Random(opts.seed ^ pop)

    tasks = [asyncio.create_task(user_loop(
        client, ep, cfg, opts, uid=pop * 100_000 + i, is_sub=(i >= pop),
        prefixes=prefixes, traces=traces, stop=stop,
        stagger_s=rng.uniform(0, max(opts.ramp_s, 1.0)), metrics=metrics))
        for i in range(pop + n_sub)]
    t0 = time.monotonic()
    measure_start = t0 + opts.ramp_s
    if on_partial is not None:
        on_partial(pop, n_sub, traces, measure_start)
    try:
        await asyncio.sleep(opts.ramp_s + opts.measure_s)
    finally:
        stop.set()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    server = _window(metrics, measure_start, time.monotonic())
    return eval_rung(pop, n_sub, traces, measure_start, cfg, opts, server)


def _window(metrics, t0: float, t1: float) -> dict | None:
    """Duck-typed read of a metrics sampler: `window(t0, t1) -> delta-like`."""
    if metrics is None:
        return None
    try:
        return _plain(metrics.window(t0, t1))
    except Exception:
        return None


# ============================================================================
# the cheap sample — a handful of requests, no population to generate
# ============================================================================
@dataclass
class Sample:
    """What a non-exclusive run can establish: a few forced misses and a few
    warm turns from one session, against whatever load the endpoint already
    carries.

    DEVIATION from the harness, stated because it changes what the numbers
    mean: the harness reads miss TTFT and the gap distribution at the ladder
    rung nearest the operating point, i.e. under load IT generated. A shared
    run has no such control — `standing_users` is unknown, and `server` (when
    a metrics sampler is present) is the only witness of what else the
    endpoint was doing. Treat these as measurements at the endpoint's
    prevailing load, not at the predicted operating point.
    """
    n: int = 0
    n_ok: int = 0
    n_err: int = 0
    ttft_miss_mean: float = float("nan")
    ttft_miss_p50: float = float("nan")
    ttft_hit_p50: float = float("nan")
    decode_p50: float = float("nan")
    itl_p50_ms: float = float("nan")
    itl_worst_p50_ms: float = float("nan")
    itl_worst_max_ms: float = float("nan")
    itl_floor_ms: float = float("nan")
    freeze_per_ktok: float = float("nan")
    chunk_tok_ratio: float = float("nan")
    ptok_ratio: float = float("nan")
    cached_frac: float = float("nan")
    server: dict | None = None
    traces: list = field(default_factory=list, repr=False)

    def to_dict(self, traces: bool = True) -> dict:
        d = {k: v for k, v in asdict(self).items() if k != "traces"}
        d["traces"] = [t.to_dict() for t in self.traces] if traces else []
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Sample":
        d = restore_nans(cls, d)
        raw = d.pop("traces", []) or []
        known = {f for f in cls.__dataclass_fields__}
        s = cls(**{k: v for k, v in d.items() if k in known})
        s.traces = [RequestTrace.from_dict(t) for t in raw]
        return s


def eval_sample(traces: list, server: dict | None = None) -> Sample:
    """Pure: the same derived quantities as `eval_rung`, over a handful of
    turns and with no measure window to clip to."""
    s = Sample(n=len(traces), server=server)
    ok = [t for t in traces if not t.error and t.ttft is not None]
    s.n_ok, s.n_err = len(ok), len(traces) - len(ok)
    miss = [t for t in ok if t.kind == "miss"]
    hit = [t for t in ok if t.kind == "hit"]
    if miss:
        s.ttft_miss_mean = sum(t.ttft for t in miss) / len(miss)
        s.ttft_miss_p50 = pct([t.ttft for t in miss], 50)
    s.ttft_hit_p50 = pct([t.ttft for t in hit], 50)
    by_user: dict[int, list] = {}
    for t in ok:
        if t.decode_tps is not None:
            by_user.setdefault(t.uid, []).append(t.decode_tps)
    s.decode_p50 = pct([pct(v, 50) for v in by_user.values()], 50)
    g = [t for t in ok if t.n_gaps > 0 and t.span_s]
    if g:
        s.itl_p50_ms = pct([t.itl_p50 for t in g], 50) * 1e3
        worst = [t.itl_max for t in g]
        s.itl_worst_p50_ms = pct(worst, 50) * 1e3
        s.itl_worst_max_ms = max(worst) * 1e3
        s.itl_floor_ms = min(t.itl_min for t in g) * 1e3
        g_tok = [t for t in g if t.ctok]
        ctoks = sum(t.ctok for t in g_tok)
        if ctoks:
            s.freeze_per_ktok = 1e3 * sum(t.n_freeze for t in g_tok) / ctoks
            s.chunk_tok_ratio = ctoks / sum(t.n_chunks for t in g_tok)
    ratios = [t.ptok_achieved / t.ptok_intended for t in ok
              if t.ptok_achieved and t.ptok_intended]
    s.ptok_ratio = pct(ratios, 50)
    cached = [t for t in hit if t.cached_tokens is not None and t.ptok_achieved]
    if cached:
        s.cached_frac = sum(1 for t in cached
                            if t.cached_tokens > 0.5 * t.ptok_achieved) / len(cached)
    s.traces = list(traces)
    return s


async def run_sample(client, ep: EndpointSpec, cfg, opts, prefixes: Prefixes,
                     metrics=None) -> Sample:
    """`sample_requests` sessions, each: one establishing turn, then
    `sample_warm_turns` warm turns and one FORCED miss.

    Sessions run concurrently (a handful of them — this is the probe that must
    not itself become the load), so the miss TTFTs are drawn under the same
    conditions as each other. Nothing here generates a population: on a shared
    endpoint the standing load is whatever else is running.
    """
    wl = cfg.workload
    traces: list[RequestTrace] = []
    t0 = time.monotonic()

    async def one(i: int):
        session = make_session(wl, opts, prefixes, uid=700_000 + i,
                               is_sub=False)
        plan = [None] + [False] * opts.sample_warm_turns + [True]
        for force in plan:
            prompt, kind = session.next_turn(force_miss=force)
            tr = RequestTrace(uid=session.uid, is_sub=False, kind=kind,
                              t_send=time.monotonic(),
                              ptok_intended=session.intended_prompt_tokens(prompt))
            traces.append(tr)
            reply = await send_request(client, ep, opts, prompt, tr,
                                       wl.max_output_tokens, metrics)
            session.commit(reply)

    await asyncio.gather(*[one(i) for i in range(max(1, opts.sample_requests))])
    return eval_sample(traces, _window(metrics, t0, time.monotonic()))
