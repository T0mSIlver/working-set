"""`ws replay`: how many recorded agent sessions an endpoint serves.

The answer comes from the endpoint and the trace alone. Nothing in it is
priced by the model or read from a config:

1. Alone. One replayed session sends its requests back to back, one in
   flight at a time, and the endpoint's own uncontended speed is fitted from
   them (`Speed`): time to first token from the uncached tokens and the
   context, then the time per output token from the context.
2. Load. N sessions replay at once, with their recorded waits, for a ramp and
   a measured window. N doubles from 1 until a step fails, then the bracket
   between the last pass and the first fail is halved down to 10%.
3. Verdict. A turn is what the agent did between two messages from the
   person (`TURN_KINDS`). Its slowdown is the model time it took (sum over
   its requests of send to last token) over the time the fitted speed gives
   for the same requests alone. A step passes when the p90 turn slowdown is
   at most `--max-slowdown` and at most 5% of requests failed.

The model time counted per request is the uncontended estimate for the
tokens the session MEANT to send uncached (its growth since the previous
request, or the whole prompt past the shared prefix on a cold start), so a
prefix the server evicted under load costs slowdown, as it costs the person.

`--config` adds the model's prediction for the trace's averages next to the
measurement; it never feeds the verdict.
"""
from __future__ import annotations

import asyncio
import itertools
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace

from .agent_trace import PARK_S, AgentTrace
from .probe.options import ProbeOptions
from .probe.replay import ReplaySession, load_trace, root_sessions
from .probe.request import (EndpointSpec, RequestTrace, make_client,
                            sampler_now, sampler_ready, sampler_window,
                            send_request)
from .probe.session import build_prefixes

FORMAT = "workingset-replay-run"
VERSION = 1

# A recorded wait on one of these means the person typed: a new turn starts.
# The other kinds (tool, permission, compaction, error) are inside a turn.
TURN_KINDS = frozenset({"start", "think", "ask", "abort", "branch", "resume",
                        "idle"})

PERCENTILE = 90
MAX_ERROR_SHARE = 0.05
DEFAULT_MAX_SLOWDOWN = 1.5
RESOLUTION = 0.10           # stop halving when fail - pass <= 10% of pass
CALIBRATE_REQUESTS = 40
CALIBRATE_MAX_S = 300.0


# ============================================================================
# the endpoint alone
# ============================================================================
def _nnls(rows: list[list[float]], y: list[float]) -> tuple[list[float], float]:
    """Non-negative least squares for a handful of columns: every subset of
    columns is solved by ordinary least squares and the best solution with
    no negative coefficient wins. Returns (coefficients, rms residual)."""
    import numpy as np
    X, Y = np.asarray(rows, float), np.asarray(y, float)
    k = X.shape[1]
    best, best_res = [0.0] * k, float("inf")
    for r in range(1, k + 1):
        for cols in itertools.combinations(range(k), r):
            c, *_ = np.linalg.lstsq(X[:, cols], Y, rcond=None)
            if (c < 0).any():
                continue
            res = float(np.sqrt(np.mean((X[:, cols] @ c - Y) ** 2)))
            if res < best_res - 1e-15:
                coef = [0.0] * k
                for i, v in zip(cols, c):
                    coef[i] = float(v)
                best, best_res = coef, res
    return best, best_res


@dataclass(frozen=True)
class Speed:
    """The endpoint with one request in flight.

    ttft(u, ctx) = a + b u + c u ctx          u uncached tokens, ctx prompt
    per token    = d + e ctx                  ctx mid-reply context
    """
    a: float = 0.0
    b: float = 0.0
    c: float = 0.0
    d: float = 0.0
    e: float = 0.0
    n: int = 0
    ttft_rms_s: float = float("nan")
    decode_rms_s: float = float("nan")
    # a tree's own streams slow each other: one request takes
    # (1 + tree_beta * k) times its single-stream time, k the other requests
    # of its tree in flight (`tree_k`), fitted on whole trees replayed alone
    tree_beta: float = 0.0
    tree_n: int = 0             # requests the fit used

    def ttft(self, u: float, ctx: float) -> float:
        return self.a + self.b * u + self.c * u * ctx

    def per_token(self, ctx: float) -> float:
        return self.d + self.e * ctx

    def service(self, u: float, ctx: float, out: int) -> float:
        """Send to last token, alone."""
        return (self.ttft(u, ctx)
                + max(out - 1, 0) * self.per_token(ctx + out / 2))

    def to_dict(self) -> dict:
        return {k: (None if isinstance(v, float) and not math.isfinite(v)
                    else v) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, d: dict) -> "Speed":
        return cls(**{k: (float("nan") if v is None else v)
                      for k, v in d.items()})


def fit_speed(rows: list["Row"]) -> Speed:
    ok = [r for r in rows if r.ok]
    if not ok:
        raise ValueError("no request completed while measuring the endpoint "
                         "alone")
    (a, b, c), t_rms = _nnls([[1.0, r.uncached, r.uncached * r.prompt]
                              for r in ok], [r.ttft for r in ok])
    dec = [r for r in ok if r.generated > 1]
    if dec:
        (d, e), d_rms = _nnls(
            [[r.generated - 1, (r.generated - 1) * (r.prompt + r.generated / 2)]
             for r in dec], [r.service - r.ttft for r in dec])
    else:
        d = e = 0.0
        d_rms = float("nan")
    return Speed(a, b, c, d, e, len(ok), t_rms, d_rms)


def tree_k(rows: list, speed: Speed) -> dict[int, int]:
    """For each request, how many other requests of its tree overlap it in time,
    with every request lasting its single-stream time
    (`Speed.service`). Counting by the time a request would take alone, not the
    time it took, keeps load from inflating the count: the tree's own
    concurrency is what its recording does on an idle server."""
    by_tree: dict[int, list] = {}
    for r in rows:
        by_tree.setdefault(r.seat, []).append(r)
    k: dict[int, int] = {}
    for rs in by_tree.values():
        spans = [(r.sent, r.sent + speed.service(r.uncached, r.prompt,
                                                 r.generated)) for r in rs]
        for i, r in enumerate(rs):
            a_i, b_i = spans[i]
            k[id(r)] = sum(1 for j, (a, b) in enumerate(spans)
                           if j != i and a < b_i and a_i < b)
    return k


def fit_tree_beta(rows: list, speed: Speed) -> tuple[float, int]:
    """The per-stream slowdown of a tree alone: (took / single-stream time - 1)
    against the other requests of its tree in flight, through the origin."""
    ok = [r for r in rows if r.ok]
    ks = tree_k(ok, speed)
    num = den = 0.0
    n = 0
    for r in ok:
        k = ks[id(r)]
        base = speed.service(r.uncached, r.prompt, r.generated)
        if k and base > 0:
            num += k * (r.service / base - 1.0)
            den += k * k
            n += 1
    return (max(0.0, num / den) if den else 0.0), n


# ============================================================================
# one request, and the session that sends it
# ============================================================================
@dataclass
class Row:
    """One request as the verdict sees it. Times are seconds from the start
    of the step."""
    seat: int
    turn: int
    kind: str               # "first" | "hit" | "miss"
    uncached: int           # tokens the session meant to send uncached
    prompt: int             # tokens it meant to send
    out_tokens: int
    sent: float
    ttft: float = float("nan")
    service: float = float("nan")   # send to last token
    error: str | None = None
    done: bool = False      # False: still running when the step ended
    ptok: int | None = None         # prompt tokens the server counted
    ctok: int | None = None         # tokens it generated
    ended: float = float("nan")     # step seconds when it finished or failed
    child: bool = False             # a subagent's request (see `_tree`)

    @property
    def generated(self) -> int:
        """Tokens generated: the server's count, else the ones asked for
        (they agree under ignore_eos)."""
        return self.ctok if self.ctok else self.out_tokens

    @property
    def ok(self) -> bool:
        return self.done and self.error is None and math.isfinite(self.ttft)

    @property
    def end(self) -> float:
        """Step seconds when it finished (nan: still running)."""
        if not self.done:
            return float("nan")
        if math.isfinite(self.ended):
            return self.ended
        return self.sent + (self.service if math.isfinite(self.service)
                            else 0.0)

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("ttft", "service", "ended"):
            if not math.isfinite(d[k]):
                d[k] = None
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Row":
        d = dict(d)
        for k in ("ttft", "service", "ended"):
            if d.get(k) is None:
                d[k] = float("nan")
        return cls(**d)


class _Seat(ReplaySession):
    """A replayed session that also says, for each request, how many tokens
    it meant to send uncached and whether a new turn starts there."""

    def next_turn(self, force_miss=None):
        r = self._req
        want = min(r.prompt_tokens, self.room())
        prev, new_rec = self._prev, self._fresh or self._pos == 0
        text, kind = super().next_turn(force_miss)
        if kind == "hit":
            uncached = max(want - prev[0], 1)
        else:
            uncached = max(want - min(self.prefix_tokens, want), 1)
        starts = self.n_turn == 0 or new_rec or r.kind in TURN_KINDS
        self.last = (uncached, want, max(1, r.output_tokens), starts)
        return text, kind


def _finish(row: Row, tr: RequestTrace, t0: float) -> None:
    row.error = tr.error
    row.ptok = tr.ptok_achieved
    row.ctok = tr.ctok
    if tr.ttft is not None:
        row.ttft = tr.ttft
        end = tr.t_end if tr.t_end is not None else tr.t_send + tr.ttft
        row.service = end - tr.t_send
    row.ended = time.monotonic() - t0
    row.done = True


async def _seat_loop(client, ep, opts, seat: _Seat, rows: list, t0: float,
                     stop: asyncio.Event, stagger_s: float, gaps: bool,
                     metrics=None, limit: int | None = None, spawn=None,
                     row_seat: int | None = None,
                     fixed_turn: int | None = None) -> None:
    """Replay `seat` request by request. `spawn(recording, request, turn)` is
    called after each request completes (the subagents it launched start
    there); `row_seat` and `fixed_turn` file a subagent's requests under its
    tree's seat and the turn that spawned it."""
    wl = SimpleNamespace(think_time_s=30.0)
    try:
        await asyncio.wait_for(stop.wait(), stagger_s)
        return
    except asyncio.TimeoutError:
        pass
    turn = 0
    while not stop.is_set() and (limit is None or seat.n_turn < limit):
        if seat.n_turn and gaps:
            try:
                await asyncio.wait_for(stop.wait(), seat.think_s(wl))
                return
            except asyncio.TimeoutError:
                pass
        prompt, kind = seat.next_turn()
        uncached, want, out, starts = seat.last
        if starts and seat.n_turn:
            turn += 1
        tr = RequestTrace(uid=seat.uid, kind=kind, t_send=time.monotonic())
        row = Row(seat.uid if row_seat is None else row_seat,
                  turn if fixed_turn is None else fixed_turn, kind, uncached,
                  want, out, sent=time.monotonic() - t0,
                  child=fixed_turn is not None)
        rows.append(row)
        reply = await send_request(client, ep, opts, prompt, tr, out, metrics)
        _finish(row, tr, t0)
        recording, pos = seat._s, seat._pos
        seat.commit(reply)
        if spawn is not None:
            spawn(recording, pos, turn)


# ============================================================================
# a load step and its verdict
# ============================================================================
def _pct(xs: list[float], q: float) -> float:
    xs = sorted(x for x in xs if math.isfinite(x))
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * q / 100
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


@dataclass
class Step:
    sessions: int
    rows: list = field(default_factory=list)
    window: tuple = (0.0, 0.0)          # measured [start, end), step seconds
    drain_s: float = 0.0                # in-flight requests finishing after it
    server: dict | None = None
    # derived by `judge`
    requests: int = 0
    errors: int = 0
    unfinished: int = 0         # still running when the drain ended
    turns: int = 0
    streams_mean: float = float("nan")   # requests in flight, window mean
    ran_out: str = ""
    slowdown_p90: float = float("nan")
    slowdown_mean: float = float("nan")
    raw_p90: float = float("nan")       # vs one stream alone, not the tree
    raw_mean: float = float("nan")
    turn_s_p50: float = float("nan")
    turn_s_p90: float = float("nan")
    passed: bool = False
    reasons: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k != "rows"}
        for k, v in d.items():
            if isinstance(v, float) and not math.isfinite(v):
                d[k] = None
        d["window"] = list(self.window)
        d["rows"] = [r.to_dict() for r in self.rows]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Step":
        d = dict(d)
        rows = [Row.from_dict(r) for r in d.pop("rows", [])]
        d["window"] = tuple(d.get("window") or (0.0, 0.0))
        for k in ("slowdown_p90", "slowdown_mean", "turn_s_p50", "turn_s_p90"):
            if d.get(k) is None:
                d[k] = float("nan")
        return cls(rows=rows, **d)


MIN_TURN_SHARE = 0.5    # a step judged on fewer turns than this share of
                        # its sessions fails: it showed too little to pass


def _in_window(r: Row, w0: float, w1: float, end_all: float) -> bool:
    """A request counts when it was in flight at any time in the window: sent
    during the ramp and finishing in the window, sent in it, or still running
    at its end. Leaving out the ramp's requests would drop exactly the ones
    an overloaded server is still working on."""
    if r.sent >= w1:
        return False
    end = r.end if r.done else end_all
    return not math.isfinite(end) or end >= w0


def judge(step: Step, speed: Speed, max_slowdown: float) -> Step:
    """Score a step's requests against the endpoint alone. Every request in
    flight during the window counts. One still running when the drain ended
    counts with the time it had taken so far, a lower bound, and as a failed
    request, so an overloaded step is never flattered by leaving its slowest
    requests out. A subagent's requests count in the turn that spawned them."""
    w0, w1 = step.window
    end_all = w1 + step.drain_s
    rows = [r for r in step.rows if _in_window(r, w0, w1, end_all)]
    step.requests = len(rows)
    step.errors = sum(1 for r in rows if r.done and r.error)
    step.unfinished = sum(1 for r in rows if not r.done)
    took: dict[tuple, float] = {}
    alone: dict[tuple, float] = {}
    single: dict[tuple, float] = {}
    live = [r for r in rows if not (r.done and r.error)]
    ks = tree_k(live, speed) if speed.tree_beta else {}
    for r in live:
        if r.ok:
            t = r.service if math.isfinite(r.service) else r.ttft
        else:
            t = max(end_all - r.sent, 0.0)
        key = (r.seat, r.turn)
        base = speed.service(r.uncached, r.prompt, r.generated)
        took[key] = took.get(key, 0.0) + t
        single[key] = single.get(key, 0.0) + base
        alone[key] = alone.get(key, 0.0) + base * (
            1.0 + speed.tree_beta * ks.get(id(r), 0))
    keys = [k for k in took if alone[k] > 0]
    step.turns = len(keys)
    slow = [took[k] / alone[k] for k in keys]
    step.slowdown_p90 = _pct(slow, PERCENTILE)
    tot_alone = sum(alone[k] for k in keys)
    step.slowdown_mean = (sum(took[k] for k in keys) / tot_alone
                          if tot_alone else float("nan"))
    raw = [took[k] / single[k] for k in keys if single[k] > 0]
    step.raw_p90 = _pct(raw, PERCENTILE)
    tot_single = sum(single[k] for k in keys)
    step.raw_mean = (sum(took[k] for k in keys) / tot_single
                     if tot_single else float("nan"))
    step.turn_s_p50 = _pct([took[k] for k in keys], 50)
    step.turn_s_p90 = _pct([took[k] for k in keys], PERCENTILE)
    span = max(w1 - w0, 1e-9)
    step.streams_mean = sum(
        max(0.0, min(r.end if r.done else end_all, w1) - max(r.sent, w0))
        for r in rows) / span
    step.ran_out = ran_out(step.server)[0]
    step.reasons = []
    failed = step.errors + step.unfinished
    if failed > MAX_ERROR_SHARE * max(step.requests, 1):
        first = next((r.error for r in rows if r.done and r.error), "")
        why = f"first: {first[:160]}" if first else (
            f"{step.unfinished} still running after the drain")
        step.reasons.append(f"{failed} of {step.requests} requests failed "
                            f"({why})")
    if not step.turns:
        step.reasons.append("no request completed in the window")
    else:
        need = max(1, math.ceil(MIN_TURN_SHARE * step.sessions))
        if step.turns < need:
            step.reasons.append(
                f"only {step.turns} turns judged for {step.sessions} "
                f"session(s), at least {need} needed (lengthen --measure-s)")
        if step.slowdown_p90 > max_slowdown:
            step.reasons.append(f"p{PERCENTILE} turn {step.slowdown_p90:.2f}x "
                                f"alone > {max_slowdown:g}x")
    step.passed = not step.reasons
    return step


KV_FULL = 0.95

RAN_OUT_RULES = (
    f"what ran out, from /metrics over the window: the KV cache when any "
    f"request was preempted or it reached {KV_FULL:.0%}; else the queue when "
    "requests waited (0.5 on average, or a quarter of those running); else "
    "neither, and the model itself got slower under load (compute)")


def ran_out(server: dict | None) -> tuple[str, str]:
    """(key, words): the first of the KV cache, the queue, compute that the
    server's own signals show over a step (RAN_OUT_RULES)."""
    if not server:
        return "unknown", "unknown (no /metrics reading)"
    kv = server.get("kv_max") or 0.0
    pre = server.get("preemptions") or 0.0
    wmean = server.get("waiting_mean") or 0.0
    wmax = server.get("waiting_max") or 0.0
    run = server.get("running_max") or 0.0
    if pre > 0 or kv >= KV_FULL:
        return "kv", (f"the KV cache (at {kv:.0%}, {pre:.0f} request(s) "
                      "preempted)")
    if wmean >= 0.5 or wmax >= max(2.0, 0.25 * run):
        return "queue", (f"the queue (up to {wmax:.0f} waiting, "
                         f"{wmean:.1f} on average)")
    return "compute", ("compute (neither the KV cache nor the queue filled: "
                       "the model got slower under load)")


# ============================================================================
# the search
# ============================================================================
def next_sessions(results: dict[int, bool], max_sessions: int) -> int | None:
    """The next step to run, or None when the answer is known. Doubles from
    1 while every step passes, then halves the bracket between the largest
    pass and the smallest fail until it is within RESOLUTION of the pass."""
    if not results:
        return 1
    passes = [n for n, ok in results.items() if ok]
    fails = [n for n, ok in results.items() if not ok]
    hi = min(fails) if fails else None
    lo = max((n for n in passes if hi is None or n < hi), default=0)
    if hi is None:
        nxt = min(max(lo, 1) * 2, max_sessions)
        return None if nxt in results else nxt
    if lo == 0 or hi - lo <= max(1, math.floor(RESOLUTION * lo)):
        return None
    mid = (lo + hi) // 2
    return None if mid in results else mid


def answer(results: dict[int, bool], max_sessions: int) -> dict:
    fails = [n for n, ok in results.items() if not ok]
    hi = min(fails) if fails else None
    lo = max((n for n, ok in results.items() if ok
              and (hi is None or n < hi)), default=0)
    return {"serves": lo, "fails_at": hi,
            "capped": hi is None and lo >= max_sessions}


# ============================================================================
# the run
# ============================================================================
@dataclass
class Plan:
    trace_path: str
    max_slowdown: float = DEFAULT_MAX_SLOWDOWN
    max_sessions: int = 256
    ramp_s: float = 60.0
    measure_s: float = 300.0
    calibrate_requests: int = CALIBRATE_REQUESTS
    calibrate_max_s: float = CALIBRATE_MAX_S
    shared_prefix_tokens: int = 0
    context_cap_tokens: int = 0
    chars_per_token: float = 4.0
    tokenizer: str | None = None
    seed: int = 0
    request_timeout_s: float = 300.0
    # where chars_per_token came from: "given" (--chars-per-token or
    # --tokenizer), "measured" (the server's prompt token counts), "default"
    cpt_source: str = "given"
    cpt_probes: int = 0

    def options(self) -> ProbeOptions:
        return ProbeOptions(chars_per_token=self.chars_per_token,
                            tokenizer=self.tokenizer,
                            context_cap_tokens=self.context_cap_tokens,
                            replay=self.trace_path, replay_max_gap_s=PARK_S,
                            seed=self.seed,
                            request_timeout_s=self.request_timeout_s)


def _seat(trace: AgentTrace, plan: Plan, prefix_text: str, uid: int,
          run_nonce: str, start: tuple[int, int] | None = None) -> _Seat:
    """A seat plays the trace's own recordings, not a subagent's (`_Forest`
    launches those from their parent)."""
    return _Seat(trace=trace, uid=uid, prefix_text=prefix_text,
                 prefix_tokens=plan.shared_prefix_tokens,
                 cpt=plan.chars_per_token, cap_tokens=plan.context_cap_tokens,
                 max_gap_s=PARK_S,
                 rng=random.Random(f"replay {plan.seed} {uid}"),
                 text_rng=random.Random(f"bench-text {plan.seed} {uid} "
                                        f"{run_nonce}"),
                 roots=root_sessions(trace), start=start)


def _prefix_text(plan: Plan) -> str:
    wl = SimpleNamespace(system_prefix_tokens=plan.shared_prefix_tokens,
                         subagent_prefix_tokens=0)
    return build_prefixes(wl, plan.chars_per_token).user


def spawn_points(trace: AgentTrace) -> dict[int, list[tuple[int, int]]]:
    """recording -> [(subagent recording, the request after which it starts)].
    A subagent starts when the parent request that spawned it completes
    (`SessionLink.spawn_request`); with none recorded, after the last request
    that began no later than `start_offset_s` into the parent."""
    out: dict[int, list[tuple[int, int]]] = {}
    n = len(trace.sessions)
    for c, link in enumerate(trace.links or ()):
        if (not link or not trace.sessions[c] or link.parent == c
                or not 0 <= link.parent < n or not trace.sessions[link.parent]):
            continue
        par = trace.sessions[link.parent]
        if link.spawn_request is not None:
            k = min(max(link.spawn_request, 0), len(par) - 1)
        else:
            k, t = 0, 0.0
            for i, r in enumerate(par):
                t += (r.gap_s or 0.0) if i else 0.0
                if t > link.start_offset_s:
                    break
                k = i
        out.setdefault(link.parent, []).append((c, k))
    return out


class _Forest:
    """The subagents a step's seats have launched. A seat is a session tree:
    its recording's subagents start where the recording spawned them, run
    concurrently with it (and launch their own), and send under the seat's
    uid and the turn that spawned them."""

    def __init__(self, client, ep, plan, trace, prefix_text, nonce, rows, t0,
                 stop, metrics, uid_base):
        self.args = (client, ep, plan, trace, prefix_text, nonce, rows, t0,
                     stop, metrics)
        self.points = spawn_points(trace)
        self.tasks: list = []
        self.uid_base = uid_base
        self.launched = 0

    def spawner(self, root_uid: int):
        def spawn(recording: int, pos: int, turn: int) -> None:
            for c, k in self.points.get(recording, ()):
                if k == pos:
                    self._start(c, root_uid, turn)
        return spawn

    def _start(self, c: int, root_uid: int, turn: int) -> None:
        client, ep, plan, trace, prefix, nonce, rows, t0, stop, metrics = \
            self.args
        if stop.is_set():
            return
        self.launched += 1
        seat = _seat(trace, plan, prefix, self.uid_base + self.launched,
                     nonce, start=(c, 0))
        self.tasks.append(asyncio.create_task(_seat_loop(
            client, ep, plan.options(), seat, rows, t0, stop, 0.0, gaps=True,
            metrics=metrics, limit=len(trace.sessions[c]),
            spawn=self.spawner(root_uid), row_seat=root_uid,
            fixed_turn=turn)))


def calibration_starts(trace: AgentTrace, plan: Plan, seats: int
                       ) -> list[tuple[int, int]]:
    """Where the alone-speed requests start: `seats` (recording, request)
    points at evenly spaced quantiles of the prompt size over all the trace's
    own recordings, so the fit sees small and large prompts, from different
    sessions. Each seat then plays on from there (warm requests)."""
    cands = []
    for i in root_sessions(trace):
        for j, r in enumerate(trace.sessions[i]):
            room = (max(plan.context_cap_tokens - max(1, r.output_tokens), 1)
                    if plan.context_cap_tokens else r.prompt_tokens)
            cands.append((min(r.prompt_tokens, room), i, j))
    cands.sort()
    if not cands:
        return []
    return [cands[min(int((k + 0.5) * len(cands) / seats), len(cands) - 1)][1:]
            for k in range(seats)]


CALIBRATE_PER_SEAT = 5


async def calibrate(client, ep, plan: Plan, trace: AgentTrace,
                    prefix_text: str, run_nonce: str) -> tuple[Speed, list]:
    """The endpoint alone: one request in flight, from several recordings at
    spread prompt sizes (`calibration_starts`), each followed by its next few
    requests, which are warm."""
    rows: list[Row] = []
    stop = asyncio.Event()
    t0 = time.monotonic()
    per = min(CALIBRATE_PER_SEAT, plan.calibrate_requests)
    seats = max(1, -(-plan.calibrate_requests // per))
    deadline = t0 + plan.calibrate_max_s

    async def play():
        for k, start in enumerate(calibration_starts(trace, plan, seats)):
            if time.monotonic() > deadline:
                break
            seat = _seat(trace, plan, prefix_text, uid=k, run_nonce=run_nonce,
                         start=start)
            await _seat_loop(client, ep, plan.options(), seat, rows, t0, stop,
                             0.0, gaps=False, limit=per)
    task = asyncio.create_task(play())
    try:
        await asyncio.wait_for(asyncio.shield(task), plan.calibrate_max_s)
    except asyncio.TimeoutError:
        stop.set()
        await task
    failed = [r for r in rows if r.done and r.error]
    if failed and len(failed) > MAX_ERROR_SHARE * len(rows):
        raise ValueError(
            f"{len(failed)} of {len(rows)} requests failed while measuring "
            f"the endpoint alone; the first: {failed[0].error[:300]}")
    speed = fit_speed(rows)
    if spawn_points(trace):
        tree_rows = await calibrate_trees(client, ep, plan, trace,
                                          prefix_text, run_nonce)
        rows += tree_rows
        beta, n = fit_tree_beta(tree_rows, speed)
        speed = replace(speed, tree_beta=beta, tree_n=n)
    return speed, rows


CALIBRATE_TREES = 3
CALIBRATE_TREE_S = 90.0


async def calibrate_trees(client, ep, plan: Plan, trace: AgentTrace,
                          prefix_text: str, run_nonce: str) -> list:
    """Whole trees replayed alone, one at a time, from their first request,
    for `CALIBRATE_TREE_S` each: the recordings with the most subagents
    spawned early. Their requests show how much a tree's own streams slow
    each other (`fit_tree_beta`)."""
    points = spawn_points(trace)
    roots = [i for i in root_sessions(trace) if i in points]
    roots.sort(key=lambda i: (-len(points[i]), len(trace.sessions[i])))
    rows: list[Row] = []
    for n, root in enumerate(roots[:CALIBRATE_TREES]):
        stop = asyncio.Event()
        t0 = time.monotonic()
        uid = 900_000 + n * 1_000
        forest = _Forest(client, ep, plan, trace, prefix_text, run_nonce,
                         rows, t0, stop, None, uid_base=uid + 500)
        seat = _seat(trace, plan, prefix_text, uid, run_nonce, start=(root, 0))
        task = asyncio.create_task(_seat_loop(
            client, ep, plan.options(), seat, rows, t0, stop, 0.0, gaps=True,
            limit=len(trace.sessions[root]), spawn=forest.spawner(uid)))
        try:
            await asyncio.wait_for(asyncio.shield(task), CALIBRATE_TREE_S)
        except asyncio.TimeoutError:
            stop.set()
        await task
        pending = [t for t in forest.tasks if not t.done()]
        if pending:
            await asyncio.wait(pending, timeout=CALIBRATE_TREE_S)
            for t in pending:
                t.cancel()
        await asyncio.gather(*forest.tasks, return_exceptions=True)
    return rows


CPT_PROBE_TOKENS = (1_500, 3_000, 6_000)


async def measure_chars_per_token(client, ep, plan: Plan,
                                  run_nonce: str) -> float | None:
    """Characters per token as the server's tokenizer counts the synthetic
    text: send a few cold prompts of known length, one output token each, and
    divide by the `usage` prompt tokens. None when the server reports none."""
    from .probe.session import make_text
    opts = plan.options()
    chars = toks = 0
    for k, n in enumerate(CPT_PROBE_TOKENS):
        n = min(n, max(plan.context_cap_tokens // 4, 100)
                if plan.context_cap_tokens else n)
        text = make_text(random.Random(f"cpt {plan.seed} {run_nonce} {k}"),
                         n, plan.chars_per_token)
        tr = RequestTrace(uid=-1 - k, kind="probe", t_send=time.monotonic())
        await send_request(client, ep, opts, text, tr, 1, None)
        if tr.error or not tr.ptok_achieved:
            continue
        chars += len(text)
        toks += tr.ptok_achieved
        plan.cpt_probes += 1
    return chars / toks if toks else None


async def run_step(client, ep, plan: Plan, trace: AgentTrace, n: int,
                   prefix_text: str, run_nonce: str, metrics=None) -> Step:
    rows: list[Row] = []
    stop = asyncio.Event()
    rng = random.Random(f"stagger {plan.seed} {n}")
    await sampler_ready(metrics)
    t0 = time.monotonic()
    opts = plan.options()
    forest = _Forest(client, ep, plan, trace, prefix_text, run_nonce, rows,
                     t0, stop, metrics, uid_base=n * 100_000 + 50_000)
    tasks = [asyncio.create_task(_seat_loop(
        client, ep, opts, _seat(trace, plan, prefix_text, uid=n * 100_000 + i,
                                run_nonce=run_nonce),
        rows, t0, stop, rng.uniform(0, plan.ramp_s), gaps=True,
        metrics=metrics, spawn=forest.spawner(n * 100_000 + i)))
        for i in range(n)]
    s_start = sampler_now(metrics) + plan.ramp_s
    await asyncio.sleep(plan.ramp_s + plan.measure_s)
    s_end = sampler_now(metrics)
    # no new request starts; the ones in flight finish, so the slowest of
    # them are measured rather than dropped. A request still running after
    # the drain counts with the time it has taken so far (`judge`).
    stop.set()
    drain_s = 0.0
    pending = [t for t in tasks + forest.tasks if not t.done()]
    if pending:
        t_d = time.monotonic()
        _, still = await asyncio.wait(pending, timeout=plan.measure_s)
        drain_s = time.monotonic() - t_d
        for t in still:
            t.cancel()
        await asyncio.gather(*still, return_exceptions=True)
    step = Step(sessions=n, rows=rows, drain_s=drain_s,
                window=(plan.ramp_s, plan.ramp_s + plan.measure_s))
    step.server = _server(await sampler_window(metrics, s_start, s_end))
    return step


def _server(w: dict | None) -> dict | None:
    """What the server showed over the window: the few signals that say what
    ran out."""
    if not w:
        return None
    g, c = w.get("gauges") or {}, w.get("counters") or {}
    h = (w.get("histograms") or {}).get("queue_time_hist") or {}

    def gs(k, s):
        return (g.get(k) or {}).get(s)
    return {"running_max": gs("requests_running", "max"),
            "waiting_mean": gs("requests_waiting", "mean"),
            "waiting_max": gs("requests_waiting", "max"),
            "kv_max": gs("kv_cache_usage", "max"),
            "preemptions": c.get("preemptions_total"),
            "queue_p95_s": h.get("p95"),
            "prefix_hit_rate": (w.get("derived") or {}).get("prefix_hit_rate")}


def signals(server: dict | None) -> str:
    """One line: what grew."""
    if not server:
        return ""
    out = []
    if server.get("waiting_max"):
        out.append(f"queue up to {server['waiting_max']:.0f}")
    if server.get("preemptions"):
        out.append(f"{server['preemptions']:.0f} preempted")
    if server.get("kv_max") is not None:
        out.append(f"KV {server['kv_max']:.0%}")
    if server.get("running_max") is not None:
        out.append(f"running {server['running_max']:.0f}")
    if server.get("prefix_hit_rate") is not None:
        out.append(f"prefix hits {server['prefix_hit_rate']:.0%}")
    return ", ".join(out)


def _say(msg: str = "") -> None:
    print(msg, flush=True)      # a run piped to a file or `tee` shows progress


async def run(ep: EndpointSpec, plan: Plan, metrics=None,
              on_step=_say) -> dict:
    trace = load_trace(plan.trace_path)
    from .probe.options import process_nonce
    nonce = process_nonce()
    async with make_client() as client:
        if plan.cpt_source == "default":
            cpt = await measure_chars_per_token(client, ep, plan, nonce)
            if cpt:
                plan.chars_per_token, plan.cpt_source = round(cpt, 3), \
                    "measured"
                on_step(f"chars per token {plan.chars_per_token:g}, from the "
                        f"server's prompt token counts ({plan.cpt_probes} "
                        "probes)")
            else:
                on_step("the server reported no prompt token counts: "
                        f"keeping {plan.chars_per_token:g} chars per token")
        prefix = _prefix_text(plan)
        on_step(f"alone: up to {plan.calibrate_requests} requests back to "
                "back, one in flight")
        speed, cal_rows = await calibrate(client, ep, plan, trace, prefix,
                                          nonce)
        on_step(describe_speed(speed, trace, plan))
        results: dict[int, bool] = {}
        steps: list[Step] = []
        while (n := next_sessions(results, plan.max_sessions)) is not None:
            on_step(f"\n--- {n} session{'s' if n != 1 else ''} ---")
            step = judge(await run_step(client, ep, plan, trace, n, prefix,
                                        nonce, metrics),
                         speed, plan.max_slowdown)
            steps.append(step)
            results[n] = step.passed
            on_step(step_line(step))
    return {"speed": speed, "calibration": cal_rows, "steps": steps,
            "answer": answer(results, plan.max_sessions), "nonce": nonce}


# ============================================================================
# report
# ============================================================================
def _turns_alone(trace: AgentTrace, speed: Speed, plan: Plan) -> float:
    """Model time of the trace's mean turn, alone: what `1.0x` is."""
    total, turns = 0.0, 0
    links = trace.links or (None,) * len(trace.sessions)
    for s, link in zip(trace.sessions, links):
        prev = None
        for i, r in enumerate(s):
            out = max(1, r.output_tokens)
            want = min(r.prompt_tokens, max(plan.context_cap_tokens - out, 1)
                       if plan.context_cap_tokens else r.prompt_tokens)
            if not link and (i == 0 or r.kind in TURN_KINDS):
                turns += 1
            if prev is None or want < prev[0]:
                u = max(want - min(plan.shared_prefix_tokens, want), 1)
            else:
                u = max(want - prev[0], 1)
            total += speed.service(u, want, max(1, r.output_tokens))
            prev = (want, r.output_tokens)
    return total / turns if turns else float("nan")


def _rates(speed: Speed, ctx: int = 40_000, cap: int = 0) -> str:
    ctx = min(ctx, cap) if cap else ctx
    pre = speed.b + speed.c * ctx
    tok = speed.per_token(ctx)
    pf = f"{1 / pre:,.0f}" if pre > 0 else "n/a"
    dec = f"{1 / tok:,.1f}" if tok > 0 else "n/a"
    return (f"first token {speed.a:.2f} s + prefill {pf} tok/s, decode "
            f"{dec} tok/s, at {ctx // 1000}k context")


def describe_speed(speed: Speed, trace: AgentTrace, plan: Plan) -> str:
    return (f"alone ({speed.n} requests): "
            f"{_rates(speed, cap=plan.context_cap_tokens)}"
            + (f"; a tree's own streams add {speed.tree_beta:.2f} per other "
               f"stream in flight ({speed.tree_n} requests)"
               if speed.tree_n else "") + "\n"
            f"  1.0x = a mean turn of {_turns_alone(trace, speed, plan):.0f} s "
            "of model time")


def step_line(s: Step) -> str:
    v = "PASS" if s.passed else "FAIL (" + "; ".join(s.reasons) + ")"
    sig = signals(s.server)
    failed = s.errors + s.unfinished
    return (f"  {s.turns} turns, {s.requests} requests, {failed} failed | "
            f"turn p90 {_x(s.slowdown_p90)} alone, mean {_x(s.slowdown_mean)} "
            f"(vs one stream: p90 {_x(s.raw_p90)}, mean {_x(s.raw_mean)}) "
            f"| turn time p50 {_s(s.turn_s_p50)} p90 {_s(s.turn_s_p90)} | "
            f"{_f1(s.streams_mean)} requests in flight | {v}"
            + (f"\n  server: {sig}" if sig else "")
            + (f"\n  the limit is {ran_out(s.server)[1]}" if not s.passed
               else ""))


def _f1(v: float) -> str:
    return f"{v:.1f}" if math.isfinite(v) else "-"


def _x(v: float) -> str:
    return f"{v:.2f}x" if math.isfinite(v) else "-"


def _s(v: float) -> str:
    return f"{v:.0f} s" if math.isfinite(v) else "-"


def headline(rec: dict) -> str:
    a = rec["answer"]
    ms = rec["plan"]["max_slowdown"]
    if a["fails_at"] is None and a["serves"]:
        return (f"serves at least {a['serves']} sessions at p{PERCENTILE} "
                f"turn <= {ms:g}x alone (the --max-sessions cap; nothing "
                "failed)")
    out = ""
    bad = next((st for st in rec.get("steps", ())
                if st.get("sessions") == a["fails_at"]), None)
    if bad is not None:
        key, words = ran_out(bad.get("server"))
        out = (f"; at {a['fails_at']} the limit is {words}" if key != "unknown"
               else "; the limit is unknown (no /metrics reading)")
    if a["serves"] == 0:
        return (f"serves no session at p{PERCENTILE} turn <= {ms:g}x alone "
                f"(one session already fails: see the step above){out}")
    return (f"serves {a['serves']} sessions at p{PERCENTILE} turn <= {ms:g}x "
            f"alone; fails at {a['fails_at']}{out}")


def print_report(rec: dict, out=None) -> None:
    out = out or sys.stdout
    w = lambda s="": print(s, file=out)          # noqa: E731
    steps = sorted((Step.from_dict(s) for s in rec["steps"]),
                   key=lambda s: s.sessions)
    tr = rec["trace"]
    w()
    w("=" * 78)
    w(f"REPLAY   {tr['path']}: {tr['sessions']} recorded sessions, "
      f"{tr['requests']} requests")
    w(f"         {rec['endpoint']['base_url']}  model="
      f"{rec['endpoint']['model'] or '<server default>'}")
    w("=" * 78)
    sp = Speed.from_dict(rec["speed"])
    pl = rec["plan"]
    w(f"alone ({sp.n} requests): "
      f"{_rates(sp, cap=pl.get('context_cap_tokens') or 0)}; a mean turn "
      f"takes {rec['turn_alone_s']:.0f} s of model time")
    if sp.tree_n:
        w(f"a session tree alone: each other stream of the tree in flight "
          f"adds {sp.tree_beta:.2f} to a request's time (fitted on "
          f"{sp.tree_n} requests of whole trees replayed alone). Turns are "
          "judged against that, so a tree's own concurrency is not held "
          "against the endpoint; the 'vs one stream' column is not corrected")
    cp = rec.get("capped")
    if cp and cp["requests"]:
        w(f"capped: {cp['requests']:,} of {cp['of']:,} recorded requests "
          f"({cp['share']:.0%}) are longer than the {cp['cap']:,}-token "
          "context cap and were cut to it. A capped request is cold at the "
          "cap, so this run understates prefix cache hits")
    src = {"measured": "measured from the server's prompt token counts",
           "default": "the default, never measured",
           "given": "given"}.get(pl.get("cpt_source", "given"), "given")
    w(f"text: {pl['chars_per_token']:.3g} chars per token ({src}); a session "
      "is a tree: a recording's subagents start where it spawned them and "
      "count in the turn that spawned them")
    w()
    w(f"{'sessions':>8} {'turns':>6} {'req':>6} {'fail':>5} "
      f"{'turn p90':>9} {'mean':>6} {'vs 1 stream':>12} "
      f"{'turn s p50/p90':>15} {'in flight':>9}  verdict")
    for s in steps:
        v = "PASS" if s.passed else "FAIL: " + "; ".join(s.reasons)
        w(f"{s.sessions:>8} {s.turns:>6} {s.requests:>6} {s.errors:>5} "
          f"{_x(s.slowdown_p90):>9} {_x(s.slowdown_mean):>6} "
          f"{_x(s.raw_p90):>12} {_s(s.turn_s_p50):>7}/{_s(s.turn_s_p90):<7} "
          f"{_f1(s.streams_mean):>9}  {v}")
    sig = [(s.sessions, signals(s.server)) for s in steps if s.server]
    if sig:
        w("\nwhat the server showed (--metrics-url)")
        for n, line in sig:
            w(f"{n:>8}  {line}")
        w(f"  {RAN_OUT_RULES}")
    w()
    w(headline(rec))
    small = [s.sessions for s in steps if s.turns and s.turns < 10]
    if small:
        w(f"  few turns at {', '.join(map(str, small))} session(s): their "
          f"p{PERCENTILE} is close to the worst turn; --measure-s widens the "
          "window")
    ratio = rec.get("prompt_token_ratio")
    if ratio and abs(ratio - 1) > 0.1:
        w(f"  the server counted {ratio:.2f}x the prompt tokens the trace "
          "recorded: re-run with --tokenizer <model> or --chars-per-token "
          f"{rec['plan']['chars_per_token'] / ratio:.2f}")
    pred = rec.get("prediction")
    if pred and "error" not in pred:
        w(f"  the model, for {pred['config']} at the trace's averages: "
          f"{pred['predicted_limit_users']} users ({pred['binding']} binds); "
          "not used above")


# ============================================================================
# command
# ============================================================================
def _endpoint_info(base_url: str, api_key: str) -> dict:
    """The served model id and its max_model_len, from /models (vLLM lists
    both). Empty when the endpoint does not answer."""
    import httpx
    try:
        r = httpx.get(f"{base_url}/models", timeout=10.0, headers=(
            {"Authorization": f"Bearer {api_key}"} if api_key else {}))
        data = (r.json().get("data") or [{}])[0]
        return {"model": data.get("id", ""),
                "max_model_len": data.get("max_model_len")}
    except Exception:                               # noqa: BLE001
        return {}


def default_shared_prefix(trace: AgentTrace) -> int:
    """The smallest first prompt of any recorded session: every session's
    first request is the system prompt and tools plus the person's first
    message, so the part every session shares is at most this."""
    firsts = [s[0].prompt_tokens for s in trace.sessions if s]
    return min(firsts) if firsts else 0


def _prompt_token_ratio(rows: list[Row]) -> float | None:
    rs = sorted(r.ptok / r.prompt for r in rows if r.ok and r.ptok
                and r.prompt)
    return rs[len(rs) // 2] if rs else None


def _prediction(config: str, trace_path: str) -> dict:
    from .config import load_config
    from .predict import predict
    from .test_cmd import apply_replay
    cfg, err = apply_replay(load_config(config), trace_path, PARK_S)
    if err:
        raise ValueError(err)
    p = predict(cfg)
    return {"config": config,
            "predicted_limit_users": p.system()["predicted_limit_users"],
            "binding": p.binding_constraint}


def metrics_headers(given: list[str], key: str, base_url: str,
                    metrics_url: str) -> dict[str, str]:
    """Headers for /metrics: each `--metrics-header 'Name: value'`, else the
    API key as a bearer token when /metrics is on the API's own host (never
    sent to another host)."""
    from urllib.parse import urlsplit
    h: dict[str, str] = {}
    for item in given or ():
        name, sep, value = item.partition(":")
        if not sep or not name.strip():
            raise SystemExit(f"--metrics-header {item!r}: expected "
                             "'Name: value'")
        h[name.strip()] = value.strip()
    if (key and not any(k.lower() == "authorization" for k in h)
            and urlsplit(base_url).netloc == urlsplit(metrics_url).netloc):
        h["Authorization"] = f"Bearer {key}"
    return h


def capped_requests(trace: AgentTrace, cap: int) -> dict:
    """Recorded requests whose prompt plus reply passes the context cap: the
    replay cuts them to it, and they are cold there."""
    n = sum(1 for s in trace.sessions for r in s
            if cap and r.prompt_tokens > cap - max(1, r.output_tokens))
    total = trace.n_requests
    return {"requests": n, "of": total, "cap": cap,
            "share": n / total if total else 0.0}


def cmd_replay(args) -> int:
    import os
    from . import __version__
    trace = load_trace(args.trace)
    if not trace.sessions:
        print(f"ws replay: {args.trace} holds no session", file=sys.stderr)
        return 2
    key = os.environ.get(args.api_key_env, "") if args.api_key_env else ""
    base = args.base_url.rstrip("/")
    info = _endpoint_info(base, key)       # a dry run reads it too
    cap = args.max_model_len or info.get("max_model_len") or max(
        r.prompt_tokens + r.output_tokens for s in trace.sessions for r in s)
    cpt, cpt_source = args.chars_per_token, "given"
    if args.tokenizer:
        from .test_cmd import _calibrated_cpt
        cpt = _calibrated_cpt(args.tokenizer)
    elif cpt is None:
        cpt, cpt_source = 4.0, "default"
    plan = Plan(trace_path=args.trace, max_slowdown=args.max_slowdown,
                max_sessions=args.max_sessions, ramp_s=args.ramp_s,
                measure_s=args.measure_s,
                shared_prefix_tokens=(args.shared_prefix_tokens
                                      if args.shared_prefix_tokens is not None
                                      else default_shared_prefix(trace)),
                context_cap_tokens=int(cap), chars_per_token=cpt,
                tokenizer=args.tokenizer, seed=args.seed,
                cpt_source=cpt_source,
                request_timeout_s=args.request_timeout_s,
                calibrate_requests=CALIBRATE_REQUESTS,
                calibrate_max_s=CALIBRATE_MAX_S)
    ep = EndpointSpec(base_url=base, model=args.model or info.get("model", ""),
                      api_key=key, api=args.api)
    pred = None
    if args.config:
        try:
            pred = _prediction(args.config, args.trace)
        except (ValueError, KeyError, OSError) as e:
            pred = {"config": args.config, "error": str(e)}

    print(f"replay {args.trace}: {len(trace.sessions)} recorded sessions, "
          f"{trace.n_requests} requests -> {base} model="
          f"{ep.model or '<server default>'}")
    print(f"  context cap {plan.context_cap_tokens:,} tokens"
          + (" (from /models)" if info.get("max_model_len")
             and not args.max_model_len else "")
          + f", shared prefix {plan.shared_prefix_tokens:,} tokens, "
          f"{plan.chars_per_token:.3g} chars per token"
          + (" (until measured from the server's prompt token counts)"
             if cpt_source == "default" else ""))
    cp = capped_requests(trace, plan.context_cap_tokens)
    if cp["requests"]:
        print(f"  {cp['requests']:,} of {cp['of']:,} recorded requests "
              f"({cp['share']:.0%}) are longer than the cap: they are cut "
              "to it and cold there, so the run understates cache hits")
    kids = sum(1 for l in trace.links if l)
    if kids:
        print(f"  {len(root_sessions(trace))} sessions start a seat; the "
              f"other {kids} are subagents, launched from their parent where "
              "it spawned them")
    print(f"  target: p{PERCENTILE} turn <= {plan.max_slowdown:g}x the "
          f"endpoint alone, <= {MAX_ERROR_SHARE:.0%} failed requests")
    print(f"  steps: 1, 2, 4, ... sessions until one fails, then halved to "
          f"{RESOLUTION:.0%}; each {plan.ramp_s:g} s ramp + "
          f"{plan.measure_s:g} s measured (cap {plan.max_sessions})")
    if pred and "error" in pred:
        print(f"  the model ({args.config}) could not price this trace: "
              f"{pred['error']}")
    elif pred:
        print(f"  the model ({args.config}): {pred['predicted_limit_users']} "
              f"users, {pred['binding']} binds; printed, not used")
    if args.dry_run:
        print("dry run: nothing sent")
        return 0

    metrics = None
    if args.metrics_url:
        from .test_cmd import open_metrics, _maybe
        metrics = open_metrics(args.metrics_url, args.engine,
                               metrics_headers(args.metrics_header, key,
                                               base, args.metrics_url))

    async def go():
        if metrics is not None:
            from .test_cmd import _maybe
            await _maybe(metrics, "start", "open")
        try:
            return await run(ep, plan, metrics)
        finally:
            if metrics is not None:
                from .test_cmd import _maybe
                await _maybe(metrics, "stop", "aclose", "close")

    try:
        res = asyncio.run(go())
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except ValueError as e:
        print(f"ws replay: {e}", file=sys.stderr)
        return 3
    rec = {"format": FORMAT, "version": VERSION, "workingset": __version__,
           "trace": {"path": args.trace, "sessions": len(trace.sessions),
                     "requests": trace.n_requests},
           "endpoint": ep.redacted(), "plan": asdict(plan),
           "run_nonce": res["nonce"],
           "speed": res["speed"].to_dict(),
           "turn_alone_s": _turns_alone(trace, res["speed"], plan),
           "capped": capped_requests(trace, plan.context_cap_tokens),
           "calibration": [r.to_dict() for r in res["calibration"]],
           "steps": [s.to_dict() for s in res["steps"]],
           "answer": res["answer"],
           "prompt_token_ratio": _prompt_token_ratio(
               res["calibration"] + [r for s in res["steps"] for r in s.rows]),
           "prediction": pred}
    if args.out:
        Path(args.out).write_text(json.dumps(rec, indent=1) + "\n",
                                  encoding="utf-8")
    print_report(rec)
    if args.metrics_url and not any(s.server for s in res["steps"]):
        print("\nwarning: --metrics-url gave no reading (unreachable, or "
              "refused without credentials): the verdict above stands, but it "
              "cannot say what ran out", file=sys.stderr)
    if args.out:
        print(f"\nrun record written to {args.out}")
    return 0


def is_record(d: dict) -> bool:
    return isinstance(d, dict) and d.get("format") == FORMAT


def add_subparser(sub) -> None:
    import argparse
    p = sub.add_parser(
        "replay", help="how many recorded agent sessions an endpoint serves",
        description="Replay recorded agent sessions (`ws trace pi`) against "
                    "an OpenAI-compatible endpoint, at 1, 2, 4, ... sessions "
                    "at once, and report the most it serves with the p90 "
                    "turn at most --max-slowdown times as long as the "
                    "endpoint alone. Needs no config: nothing in the answer "
                    "comes from the model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  ws replay trace.json --base-url http://gpu:8000/v1 "
               "--out run.json\n"
               "  ws replay trace.json --base-url http://gpu:8000/v1 "
               "--metrics-url http://gpu:8000/metrics --max-slowdown 1.3\n"
               "  ws report run.json\n")
    p.add_argument("trace", help="agent trace from `ws trace pi`")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--model", help="served model id (default: the first one "
                                   "/models lists)")
    p.add_argument("--api-key-env", default="VLLM_API_KEY",
                   help="environment variable holding the API key")
    p.add_argument("--api", choices=("completions", "chat"),
                   default="completions")
    p.add_argument("--max-slowdown", type=float, default=DEFAULT_MAX_SLOWDOWN,
                   help=f"the target: p{PERCENTILE} turn at most this many "
                        f"times as long as alone (default "
                        f"{DEFAULT_MAX_SLOWDOWN:g})")
    p.add_argument("--max-sessions", type=int, default=256)
    p.add_argument("--ramp-s", type=float, default=60.0,
                   help="seconds sessions start over, not measured")
    p.add_argument("--measure-s", type=float, default=300.0,
                   help="seconds measured per step")
    p.add_argument("--metrics-url", help="the server's Prometheus /metrics, "
                                         "to say what ran out")
    p.add_argument("--engine", help="one engine of a multi-engine /metrics")
    p.add_argument("--tokenizer", help="size the synthetic text with this "
                                       "model's tokenizer (needs toklen)")
    p.add_argument("--chars-per-token", type=float, default=None,
                   help="characters the synthetic text spends per token "
                        "(default: measured from the server's prompt token "
                        "counts at the start of the run)")
    p.add_argument("--metrics-header", action="append", default=[],
                   metavar="'Name: value'",
                   help="a header to send with every /metrics request "
                        "(repeatable). Without it the API key goes as a "
                        "bearer token to /metrics on the same host as "
                        "--base-url")
    p.add_argument("--max-model-len", type=int,
                   help="context cap (default: /models, else the trace's "
                        "largest request)")
    p.add_argument("--shared-prefix-tokens", type=int,
                   help="tokens every session shares (default: the smallest "
                        "first prompt in the trace)")
    p.add_argument("--config", help="also print the model's prediction for "
                                    "this config at the trace's averages")
    p.add_argument("--request-timeout-s", type=float, default=300.0,
                   help="seconds to wait for the next byte of a response "
                        "before the request counts as failed (default 300)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan, send nothing")
    p.add_argument("-o", "--out", help="write the run record here (JSON)")
    p.set_defaults(fn=cmd_replay)
