"""The simultaneous-miss flush probe — B*, the correlated-flush tolerance.

Ported from `run_burst` in scripts/validate_deployment.py. From steady
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

from .population import _window, spike_evidence, user_loop
from .request import EndpointSpec, RequestTrace, send_request
from .session import Prefixes, draw_session_tokens, make_text
from .stats import FREEZE_LADDER_MS, pct, restore_nans


@dataclass
class BurstResult:
    n: int = 0
    standing_users: int = 0
    n_ok: int = 0
    n_err: int = 0
    # drain: fire -> the LAST request's first token (all fired together, so
    # last first-token = the fluid model's T_drain, whatever the scheduler's
    # discipline — see model.burst_drain_seconds)
    drain_s: float | None = None
    last_ttft_s: float | None = None
    ttft_p50_s: float = float("nan")
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
    r.ttft_p50_s = pct([t.ttft for t in ok], 50)
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
    tasks = [asyncio.create_task(user_loop(
        client, ep, cfg, opts, uid=900_000 + i, is_sub=(i >= pop),
        prefixes=prefixes, traces=traces, stop=stop,
        stagger_s=rng.uniform(0, max(opts.ramp_s, 1.0)), metrics=metrics))
        for i in range(pop + n_sub)]
    t_start = time.monotonic()
    try:
        await asyncio.sleep(opts.ramp_s)

        async def one_miss(i: int) -> RequestTrace:
            r = random.Random((opts.seed << 8) ^ (0xF00D + i))
            full = draw_session_tokens(r, wl.user_prompt_median_tokens,
                                       wl.user_prompt_sigma,
                                       wl.system_prefix_tokens,
                                       opts.context_cap_tokens)
            salt = f"[miss-salt {r.getrandbits(64):016x}] "
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
        burst_traces = list(await asyncio.gather(*[one_miss(i) for i in range(n)]))
    finally:
        stop.set()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    server = _window(metrics, t_start, time.monotonic())
    return eval_burst(n, pop, burst_traces, traces, t_fire, server,
                      cap_tokens=opts.context_cap_tokens)
