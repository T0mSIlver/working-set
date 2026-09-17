"""The simultaneous-miss flush probe — B*, the correlated-flush tolerance.

Ported from `run_burst` in the retired standalone harness. From steady
standing load, fire N simultaneous forced misses and time the drain: B*
predicts the largest N whose LAST first-token still lands inside the TTFT
budget (`model.spike_tolerance`).

The standing load's inter-token gaps during the drain are the chunk-size
hypothesis in its purest form — a controlled cold-prefill event, and the
decoders it lands on. It is also the RELIABLE place to read a spike: the
standing load is small (the steady-state decode batch, not a high ladder
rung), so client event-loop contention cannot manufacture gaps the way it can
under the ladder.
"""
from __future__ import annotations

import asyncio
import random
import time
from dataclasses import asdict, dataclass, field

from .population import spike_evidence, user_loop
from .request import (EndpointSpec, RequestTrace, sampler_now, sampler_ready,
                      sampler_window, send_request)
from .session import Prefixes, draw_session_tokens, make_text, nonce_bits
from .stats import FREEZE_LADDER_MS, pct, restore_nans


# Standing sessions establish over the FIRST part of the ramp, not all of it.
# Staggered over the whole ramp with the fire exactly at its end, a session
# starting late has its cold establishing prefill — a full context, the most
# expensive request it will ever send — in flight at the fire, sitting in the
# server's queue ahead of the burst. The drain then times the burst PLUS
# somebody's establishment, against a B* that priced a steady standing load.
ESTABLISH_FRAC = 0.8
# ...and the last fifth of the ramp is not a guarantee, so the fire also waits
# until every standing session has its first token. BOUNDED: an endpoint that
# never answers an establishing turn must not hang the probe, and a fire that
# went ahead regardless says so in `n_establishing_at_fire`.
ESTABLISH_WAIT_MAX_S = 30.0
_ESTABLISH_POLL_S = 0.02


def establishing_pending(traces: list, n_sessions: int) -> int:
    """Standing sessions whose establishing ("first") turn has NOT yet got its
    first token — in flight, or not even sent. Pure.

    A first turn that errored is over and counts as settled: it holds no
    place in the server's queue, and waiting on it would wait forever.
    """
    settled = sum(1 for t in traces if t.kind == "first"
                  and (t.ttft is not None or t.error))
    return max(0, n_sessions - settled)


@dataclass
class BurstResult:
    n: int = 0
    standing_users: int = 0
    n_ok: int = 0
    n_err: int = 0
    # how long the fire was held past the ramp for standing sessions to finish
    # establishing, and how many still had not when it went. Anything but 0 in
    # the second is a CONTAMINATED burst: a cold establishing prefill was
    # queued in front of it.
    establish_wait_s: float = 0.0
    n_establishing_at_fire: int = 0
    # standing turns of ANY kind still waiting for a first token at the fire:
    # a drawn miss in prefill sits ahead of the burst just as an establishing
    # turn does. Not waited for (under load there is always one), only counted
    n_standing_prefilling_at_fire: int = 0
    # drain: fire -> the LAST request's first token (all fired together, so
    # last first-token = the fluid model's T_drain, whatever the scheduler's
    # discipline — see model.burst_drain_seconds)
    drain_s: float | None = None
    last_ttft_s: float | None = None
    ttft_p50_s: float = float("nan")
    # what was actually flushed: the prompt tokens of the requests the drain
    # is over. The burst's lengths are random draws from a heavy-tailed
    # log-normal, so N alone says little about the work — the same N drains
    # in very different times depending on the draws. `usage` readback where
    # the endpoint gave one, the client's intent otherwise; `ptok_from_usage`
    # counts the former.
    ptok_total: int = 0
    ptok_from_usage: int = 0
    # achieved / intended prompt tokens, median over every request of this
    # probe that returned `usage` — the burst's own misses AND the standing
    # load's completed turns. nan = the endpoint returned no usage at all.
    ptok_ratio: float = float("nan")
    # what the STANDING load felt while the burst was draining
    standing_n: int = 0
    standing_itl_p50_ms: float = float("nan")
    standing_worst_p50_ms: float = float("nan")
    standing_worst_p95_ms: float = float("nan")
    standing_worst_max_ms: float = float("nan")
    standing_floor_ms: float = float("nan")
    standing_freeze_per_ktok: float | None = None
    standing_freeze_ladder: list | None = None
    spike: dict = field(default_factory=dict)   # see probe.spike_evidence
    server: dict | None = None
    traces: list = field(default_factory=list, repr=False)

    def to_dict(self, traces: bool = True) -> dict:
        d = {k: v for k, v in asdict(self).items() if k != "traces"}
        d["traces"] = [t.to_dict() for t in self.traces] if traces else []
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BurstResult":
        d = restore_nans(cls, d)
        raw = d.pop("traces", []) or []
        known = {f for f in cls.__dataclass_fields__}
        b = cls(**{k: v for k, v in d.items() if k in known})
        b.traces = [RequestTrace.from_dict(t) for t in raw]
        return b


def burst_prompt_tokens(traces: list) -> list[int]:
    """Prompt tokens of each burst request that ANSWERED — the same set
    `drain_s` is over. The server's `usage` count where there is one, the
    intended count where there is not."""
    return [int(t.ptok_achieved or t.ptok_intended or 0) for t in traces
            if t.ttft is not None and not t.error]


def eval_burst(n: int, standing_users: int, burst_traces: list,
               standing_traces: list, t_fire: float,
               server: dict | None = None,
               cap_tokens: float = 0.0) -> BurstResult:
    """Pure: the burst's drain and the standing load's gap distribution.

    `last_ttft_s` and `drain_s` are the max over the requests that ANSWERED.
    With a failure among them they are the drain of a smaller burst, which is
    why `n_err` is carried next to them and why H-burst refuses to score a
    partial burst.
    """
    ok = [t for t in burst_traces if t.ttft is not None and not t.error]
    r = BurstResult(n=n, standing_users=standing_users, n_ok=len(ok),
                    n_err=n - len(ok), server=server)
    if ok:
        r.drain_s = max(t.t_send + t.ttft for t in ok) - t_fire
        r.last_ttft_s = max(t.ttft for t in ok)
        r.ptok_total = sum(burst_prompt_tokens(ok))
        r.ptok_from_usage = sum(1 for t in ok if t.ptok_achieved)
    r.ttft_p50_s = pct([t.ttft for t in ok], 50)
    # the standing load's traces are not kept on the result, so whatever
    # `usage` they returned is summarised HERE or lost. (A standing stream
    # cancelled when the probe ends never reaches its usage trailer; the
    # turns that completed before that did.)
    r.ptok_ratio = pct([t.ptok_achieved / t.ptok_intended
                        for t in list(burst_traces) + list(standing_traces)
                        if t.ptok_achieved and t.ptok_intended], 50)
    # the same spike statistic the ladder and the sample report, over both
    # legs: the burst's own misses are the cold prefills, the standing load
    # supplies the decoders
    r.spike = spike_evidence(list(burst_traces) + list(standing_traces),
                             cap_tokens)

    # the window is "request in flight at fire time": t_end is the real
    # end-of-stream (turns that never got a first token have no t_end and are
    # already excluded by the n_gaps > 0 filter)
    victims = [t for t in standing_traces
               if t.n_gaps > 0 and t.t_end and t.t_send <= t_fire <= t.t_end]
    if victims:
        worst = [t.itl_max for t in victims]
        r.standing_n = len(victims)
        r.standing_itl_p50_ms = pct([t.itl_p50 for t in victims], 50) * 1e3
        r.standing_worst_p50_ms = pct(worst, 50) * 1e3
        r.standing_worst_p95_ms = pct(worst, 95) * 1e3
        r.standing_worst_max_ms = max(worst) * 1e3
        r.standing_floor_ms = min(t.itl_min for t in victims) * 1e3
        # per DECODED TOKEN, not per gap — see eval_rung
        v_tok = [t for t in victims if t.ctok]
        ctoks = sum(t.ctok for t in v_tok)
        if ctoks:
            r.standing_freeze_per_ktok = (
                1e3 * sum(t.n_freeze for t in v_tok) / ctoks)
            r.standing_freeze_ladder = [
                {"threshold_ms": thr,
                 "per_ktok": 1e3 * sum(t.n_freeze_at[i] for t in v_tok) / ctoks}
                for i, thr in enumerate(FREEZE_LADDER_MS)]
    r.traces = list(burst_traces)
    return r


async def run_burst(client, ep: EndpointSpec, cfg, opts, n: int,
                    standing_users: int, prefixes: Prefixes,
                    metrics=None) -> BurstResult:
    wl = cfg.workload
    pop = max(0, standing_users)
    n_sub = round(pop * wl.subagent_ratio)
    traces: list[RequestTrace] = []
    stop = asyncio.Event()
    rng = random.Random(opts.seed ^ 0xB0057)
    bits = nonce_bits(opts.run_nonce)
    # the burst's window opens at `w_start` below, and its low endpoint has
    # to be a snapshot that completed before that (see `sampler_ready`)
    await sampler_ready(metrics)
    tasks = [asyncio.create_task(user_loop(
        client, ep, cfg, opts, uid=900_000 + i, is_sub=(i >= pop),
        prefixes=prefixes, traces=traces, stop=stop,
        stagger_s=rng.uniform(0, ESTABLISH_FRAC * max(opts.ramp_s, 1.0)),
        metrics=metrics))
        for i in range(pop + n_sub)]
    # the SAMPLER's base, not monotonic: the traces below keep their own
    # monotonic timestamps for span arithmetic, and the two differ by the
    # unix epoch (see probe.request.sampler_now)
    w_start = sampler_now(metrics)
    wait_s, n_pending, n_prefilling = 0.0, 0, 0
    try:
        await asyncio.sleep(opts.ramp_s)
        # hold the fire until the standing load IS standing (see
        # ESTABLISH_WAIT_MAX_S): no establishing prefill ahead of the burst
        t_hold = time.monotonic()
        while True:
            n_pending = establishing_pending(traces, pop + n_sub)
            wait_s = time.monotonic() - t_hold
            if not n_pending or wait_s >= ESTABLISH_WAIT_MAX_S:
                break
            await asyncio.sleep(_ESTABLISH_POLL_S)

        async def one_miss(i: int) -> RequestTrace:
            r = random.Random((opts.seed << 8) ^ (0xF00D + i))
            full = draw_session_tokens(r, wl.user_prompt_median_tokens,
                                       wl.user_prompt_sigma,
                                       wl.system_prefix_tokens,
                                       opts.context_cap_tokens)
            # the run nonce, as in `Session.next_turn`: `r` is a function of
            # the seed, and a salt the server saw on the last run is a hit
            salt = f"[miss-salt {r.getrandbits(64) ^ bits:016x}] "
            prompt = (salt + prefixes.user + "\n"
                      + make_text(r, max(full - wl.system_prefix_tokens, 0),
                                  opts.chars_per_token))
            t = RequestTrace(uid=990_000 + i, is_sub=False, kind="miss",
                             t_send=time.monotonic(),
                             ptok_intended=int(len(prompt) / opts.chars_per_token))
            await send_request(client, ep, opts, prompt, t,
                               wl.max_output_tokens, metrics)
            return t

        t_fire = time.monotonic()
        n_prefilling = sum(1 for t in traces if t.ttft is None and not t.error)
        burst_traces = list(await asyncio.gather(*[one_miss(i) for i in range(n)]))
    finally:
        stop.set()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    server = await sampler_window(metrics, w_start, sampler_now(metrics))
    res = eval_burst(n, pop, burst_traces, traces, t_fire, server,
                     cap_tokens=opts.context_cap_tokens)
    res.establish_wait_s, res.n_establishing_at_fire = wait_s, n_pending
    res.n_standing_prefilling_at_fire = n_prefilling
    return res
