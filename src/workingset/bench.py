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
3. Verdict. A step of N sessions passes when all of these hold over its
   window (`Limits`, one flag each):
   - mid-session evictions are at most 1% of warm requests. A request is
     warm when the session's previous request exists, no compaction or reset
     lies between, and it adds at most `--warm-new-tokens` tokens. The replay
     knows what it sent, so the tokens the request shares with the session's
     previous prompt are what the server should have cached; an eviction is
     `usage.prompt_tokens_details.cached_tokens` short of that by more than a
     block. A server that never reports cached_tokens leaves the check
     "unknown": it does not gate;
   - the p90 time to first token of warm requests is at most 3 s;
   - the p10 decode speed per request, (output - 1) / (last - first token),
     over requests of at least 32 output tokens, is at least 50 tok/s;
   - at most 1% of requests failed or were left unfinished.
   A turn is what the agent did between two messages from the person
   (`TURN_KINDS`); its slowdown (model time over the time the fitted speed
   gives alone) is reported, and gates only under an explicit
   `--max-slowdown`. Cold requests (a session's first, after a compaction)
   report their time to first token less the idle-speed estimate, which is
   queueing; it is not gated.

Several `--base-url`s bench a data-parallel deployment: `--routing sticky`
pins a session tree to one URL by a stable hash of its seat id (what a
session-affine router does), `round-robin` sends each request to the next.
`--output-scale` multiplies every recorded reply length and the context it
adds to later prompts. `--chat-rate` adds background chat requests, Poisson,
reported apart and never judged.

The time counted per request for the turn slowdown is the uncontended
estimate for the tokens the session MEANT to send uncached (its growth since
the previous request, or the whole prompt past the shared prefix on a cold
start), so a prefix the server evicted under load costs slowdown, as it
costs the person.

`--config` adds the model's prediction for the trace's averages next to the
measurement; it never feeds the verdict.
"""
from __future__ import annotations

import asyncio
import hashlib
import itertools
import json
import math
import random
import sys
import time
from dataclasses import MISSING, asdict, dataclass, field, fields, replace
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
MAX_ERROR_SHARE = 0.01
MAX_EVICTION_SHARE = 0.01
MAX_WARM_TTFT_S = 3.0
MIN_DECODE_TOK_S = 50.0
WARM_NEW_TOKENS = 8192
BLOCK_FROM_REPLIES_MIN = 8  # distinct cached_tokens values before their gcd counts
KV_BLOCK_TOKENS = 16        # a Limits default for tests and old records only
MIN_SAMPLES = 10            # a check with fewer requests than this is not judged
MIN_DECODE_TOKENS = 32      # requests with fewer output tokens have no speed
ROUTINGS = ("sticky", "round-robin")
CHAT_SIGMA = 0.8
CALIBRATION_ERROR_SHARE = 0.05      # the alone fit tolerates a few failures
RESOLUTION = 0.10           # stop halving when fail - pass <= 10% of pass
DEFAULT_BASE_URL = "http://localhost:8000/v1"
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


MIN_BETA_REQUESTS = 20  # fewer overlapping requests: no correction (stricter)
MAX_TREE_BETA = 1.0     # a tree's own streams never excuse more than this


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
    if n < MIN_BETA_REQUESTS or not den:
        return 0.0, n
    return min(max(0.0, num / den), MAX_TREE_BETA), n


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
    phase: str = "single"           # calibration: "single" stream alone | "tree" concurrent
    # --- what the verdict's checks read (old records lack them: defaults) ---
    warm: bool = False      # the session's previous request succeeded, no reset
    cold: bool = False      # a session's first request, or after a compaction
    expected: int = 0       # prompt tokens it shares with the previous request
    cached: int | None = None       # tokens the server says it served cached
    span: float = float("nan")      # first to last token, seconds
    url: int = 0            # which --base-url it went to

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
        for k in ("ttft", "service", "ended", "span"):
            if not math.isfinite(d[k]):
                d[k] = None
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Row":
        d = dict(d)
        for k in ("ttft", "service", "ended", "span"):
            if d.get(k) is None:
                d[k] = float("nan")
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


class _Seat(ReplaySession):
    """A replayed session that also says, for each request, how many tokens
    it meant to send uncached and whether a new turn starts there."""

    def next_turn(self, force_miss=None):
        r = self._req
        prev, new_rec = self._prev, self._fresh or self._pos == 0
        text, kind = super().next_turn(force_miss)
        want = self.want
        if kind == "hit":
            uncached = max(want - prev[0], 1)
        else:
            uncached = max(want - min(self.prefix_tokens, want), 1)
        starts = self.n_turn == 0 or new_rec or r.kind in TURN_KINDS
        self.last = (uncached, want, self.out_tokens(), starts)
        # a hit that follows a compaction is cold: the summary replaced the
        # history, whatever the prompt sizes say
        self.cold = kind != "hit" or r.kind == "compaction"
        self.shared = min(prev[0], want) if kind == "hit" else 0
        return text, kind


def _finish(row: Row, tr: RequestTrace, t0: float) -> None:
    row.error = tr.error
    row.ptok = tr.ptok_achieved
    row.ctok = tr.ctok
    row.cached = tr.cached_tokens
    if tr.span_s is not None:
        row.span = tr.span_s
    if tr.ttft is not None:
        row.ttft = tr.ttft
        end = tr.t_end if tr.t_end is not None else tr.t_send + tr.ttft
        row.service = end - tr.t_send
    row.ended = time.monotonic() - t0
    row.done = True


def sticky_index(key: int, n: int) -> int:
    """The URL a session tree is pinned to: a stable hash of its seat id
    (never Python's per-process `hash`), as a session-affine router does."""
    if n <= 1:
        return 0
    h = hashlib.blake2b(str(key).encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") % n


class Router:
    """Picks the endpoint of each request when `--base-url` is repeated.
    `sticky`: by the seat id, so a session tree stays on one URL.
    `round-robin`: the next URL for every request."""

    def __init__(self, eps: list, mode: str = "sticky"):
        if mode not in ROUTINGS:
            raise ValueError(f"routing {mode!r}: expected one of {ROUTINGS}")
        self.eps, self.mode = list(eps), mode
        self._next = itertools.count()

    def __len__(self) -> int:
        return len(self.eps)

    def pick(self, key: int) -> tuple[int, EndpointSpec]:
        n = len(self.eps)
        i = (sticky_index(key, n) if self.mode == "sticky"
             else next(self._next) % n)
        return i, self.eps[i]


def _target(ep, key: int) -> tuple[int, EndpointSpec]:
    return ep.pick(key) if isinstance(ep, Router) else (0, ep)


def _sampler(metrics, i: int):
    """The sampler of URL `i` (`metrics` is one sampler or a list)."""
    if isinstance(metrics, (list, tuple)):
        return metrics[i] if i < len(metrics) else None
    return metrics if i == 0 else None


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
    prev_ok, prev_ptok, prev_want = False, None, 0
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
        i, spec = _target(ep, seat.uid if row_seat is None else row_seat)
        row.url = i
        row.cold = seat.cold
        reply = await send_request(client, spec, opts, prompt, tr, out,
                                   _sampler(metrics, i))
        _finish(row, tr, t0)
        # warm: the session's previous request succeeded, so the server holds
        # its prompt; what this request shares with it is what should hit
        if kind == "hit" and not seat.cold and prev_ok:
            row.warm = True
            shared = prev_ptok or prev_want
            now = tr.ptok_achieved or want
            row.expected = min(shared, now)
        prev_ok = tr.error is None and tr.ttft is not None
        prev_ptok, prev_want = tr.ptok_achieved, want
        recording, pos = seat._s, seat._pos
        seat.commit(reply)
        if spawn is not None:
            spawn(recording, pos, turn if fixed_turn is None else fixed_turn)


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


def block_from_replies(rows) -> int:
    """The block the server caches in, read off its replies: vLLM counts
    cached tokens in whole KV blocks, so the gcd of enough distinct
    cached_tokens values is the block (1,600 on a hybrid model whose
    attention block grows to the recurrent state's page). 0 below
    BLOCK_FROM_REPLIES_MIN distinct values: too few to tell the block from
    chance."""
    vals = {r.cached for r in rows if r.cached}
    if len(vals) < BLOCK_FROM_REPLIES_MIN:
        return 0
    return math.gcd(*vals)


def block_from_metrics(text: str) -> int | None:
    """block_size from vllm:cache_config_info, None when absent."""
    import re
    for line in text.splitlines():
        if line.startswith("vllm:cache_config_info"):
            m = re.search(r'block_size="(\d+)"', line)
            return int(m.group(1)) if m else None
    return None


def prefix_caching_from_metrics(text: str) -> bool | None:
    """enable_prefix_caching from vllm:cache_config_info, None when absent."""
    import re
    for line in text.splitlines():
        if line.startswith("vllm:cache_config_info"):
            m = re.search(r'enable_prefix_caching="(\w+)"', line)
            return (m.group(1) == "True") if m else None
    return None


def read_metrics(url: str, headers: dict[str, str]) -> str:
    """The text of a vLLM /metrics. Raises ValueError, with what to pass
    instead, when it does not answer or is not vLLM's."""
    import httpx
    try:
        r = httpx.get(url, headers=headers, timeout=10.0)
        r.raise_for_status()
    except Exception as e:                          # noqa: BLE001
        raise ValueError(
            f"GET {url} failed ({e}). Pass --metrics-url (and "
            "--metrics-header for its auth), or --no-metrics to run without "
            "it: the report then cannot say what ran out, and "
            "--kv-block-tokens is required") from None
    if "vllm:" not in r.text:
        raise ValueError(f"{url} answers but holds no vllm: metric (a router "
                         "or gateway?). Pass each engine's own /metrics with "
                         "--metrics-url, once per replica behind a router, "
                         "or --no-metrics")
    return r.text


@dataclass
class Limits:
    """What a step must hold to pass (`judge`). One flag each."""
    max_eviction_share: float = MAX_EVICTION_SHARE
    max_warm_ttft_s: float = MAX_WARM_TTFT_S
    min_decode_tok_s: float = MIN_DECODE_TOK_S
    max_failed_share: float = MAX_ERROR_SHARE
    warm_new_tokens: int = WARM_NEW_TOKENS
    kv_block_tokens: int = KV_BLOCK_TOKENS
    max_slowdown: float | None = None   # gates only when given
    min_samples: int = MIN_SAMPLES
    eviction_check: bool = True         # False: --skip-eviction-check


# the checks, in the order they are reported
CHECKS = ("evictions", "warm time to first token", "decode speed",
          "failed requests", "turn slowdown")


def _check(status: str, value: float = float("nan"), limit=None,
           n: int = 0, **extra) -> dict:
    """status: pass | fail | few (fewer samples than --min-samples: not
    judged) | unknown (not reported by the server) | n/a (no request to
    judge) | info (reported, not gating)."""
    return {"status": status, "n": n, "limit": limit,
            "value": value if math.isfinite(value) else None, **extra}


def merge_servers(servers: list) -> dict | None:
    """One reading for the whole deployment from one per URL: the signals
    that say what ran out, summed or maxed the way a pool of replicas
    shows them."""
    got = [s for s in servers if s]
    if not got:
        return None
    if len(got) == 1:
        return got[0]

    def tot(k):
        v = [s[k] for s in got if s.get(k) is not None]
        return sum(v) if v else None

    def top(k):
        v = [s[k] for s in got if s.get(k) is not None]
        return max(v) if v else None
    hit = [s["prefix_hit_rate"] for s in got
           if s.get("prefix_hit_rate") is not None]
    return {"running_max": tot("running_max"),
            "waiting_mean": tot("waiting_mean"),
            "waiting_max": top("waiting_max"), "kv_max": top("kv_max"),
            "preemptions": tot("preemptions"), "queue_p95_s": top("queue_p95_s"),
            "prefix_hit_rate": sum(hit) / len(hit) if hit else None}


@dataclass
class Step:
    sessions: int
    rows: list = field(default_factory=list)
    chat_rows: list = field(default_factory=list)   # background chat requests
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
    inconclusive: bool = False  # too few turns to say, even after measuring on
    p90_lo: float = float("nan")        # bootstrap 90% interval on the p90
    p90_hi: float = float("nan")
    slowdown_p90: float = float("nan")
    slowdown_mean: float = float("nan")
    raw_p90: float = float("nan")       # vs one stream alone, not the tree
    raw_mean: float = float("nan")
    turn_s_p50: float = float("nan")
    turn_s_p90: float = float("nan")
    passed: bool = False
    # too few requests to judge and nothing queued or failed: the sampled
    # sessions sat idle. Neither a pass nor a fail; the search moves on
    no_evidence: bool = False
    reasons: list = field(default_factory=list)
    # --- the checks (CHECKS -> {status, value, limit, n}); empty in records
    # from before them ---
    checks: dict = field(default_factory=dict)
    warm_requests: int = 0
    evictions: int = 0
    # the block the eviction check rounded to: the plan's, or larger when the
    # server's cached_tokens all came in whole blocks of a larger size
    kv_block_tokens: int = 0
    cold_requests: int = 0
    cold_queue_p50: float = float("nan")    # cold time to first token less
    cold_queue_p90: float = float("nan")    # the idle estimate: queueing
    # --- per URL (--base-url repeated) ---
    n_urls: int = 1
    url_requests: list = field(default_factory=list)
    servers: list = field(default_factory=list)
    # --- background chat (--chat-rate): reported, never judged ---
    chat: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {k: v for k, v in asdict(self).items()
             if k not in ("rows", "chat_rows")}
        for k, v in d.items():
            if isinstance(v, float) and not math.isfinite(v):
                d[k] = None
        d["window"] = list(self.window)
        d["rows"] = [r.to_dict() for r in self.rows]
        d["chat_rows"] = [r.to_dict() for r in self.chat_rows]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Step":
        d = dict(d)
        rows = [Row.from_dict(r) for r in d.pop("rows", [])]
        chat_rows = [Row.from_dict(r) for r in d.pop("chat_rows", [])]
        d["window"] = tuple(d.get("window") or (0.0, 0.0))
        for f in fields(cls):
            if f.default is not MISSING and isinstance(f.default, float) \
                    and d.get(f.name, 0.0) is None:
                d[f.name] = float("nan")
        return cls(rows=rows, chat_rows=chat_rows, **d)


def bootstrap_p90(xs: list[float], n: int = 500, seed: int = 0
                  ) -> tuple[float, float]:
    """A 90% interval on the p90 of `xs`, resampling the turns."""
    xs = [x for x in xs if math.isfinite(x)]
    if len(xs) < 2:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    ps = sorted(_pct([rng.choice(xs) for _ in xs], PERCENTILE)
                for _ in range(n))
    return ps[int(0.05 * n)], ps[min(int(0.95 * n), n - 1)]


def min_turns(plan, sessions: int) -> int:
    """Turns a step needs to be judged: --min-turns, but at most 5 per
    session, so a step of 1 or 2 sessions is not inconclusive by design."""
    return min(plan.min_turns, 5 * sessions)


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


def count_samples(step: Step, lim: Limits) -> tuple[int, int]:
    """(warm requests, decode-eligible requests) finished in the window: what
    the checks need `--min-samples` of. Cheap, like `count_turns`."""
    w0, w1 = step.window
    end_all = w1 + step.drain_s
    rows = [r for r in step.rows if _in_window(r, w0, w1, end_all)]
    warm = sum(1 for r in rows if r.warm and r.ok
               and r.uncached <= lim.warm_new_tokens)
    return warm, len(_decode_rates(rows, MIN_DECODE_TOKENS))


def count_turns(step: Step) -> int:
    """The turns `judge` would score, without scoring them: cheap enough to
    run on the load generator's event loop while a step measures on."""
    w0, w1 = step.window
    end_all = w1 + step.drain_s
    return len({(r.seat, r.turn) for r in step.rows
                if _in_window(r, w0, w1, end_all)
                and not (r.done and r.error)})


def _as_limits(limits) -> Limits:
    """`judge` takes Limits; a bare number is a `--max-slowdown`, which gates
    on top of the defaults."""
    if limits is None:
        return Limits()
    if isinstance(limits, Limits):
        return limits
    return Limits(max_slowdown=float(limits))


def _decode_rates(rows: list, min_tokens: int) -> list[float]:
    return [(r.generated - 1) / r.span for r in rows
            if r.ok and r.generated >= min_tokens
            and math.isfinite(r.span) and r.span > 0]


def judge(step: Step, speed: Speed, limits=None, min_turns: int = 0) -> Step:
    """Score a step's requests against the pass rule (`Limits`; the module
    docstring says it) and, as information, against the endpoint alone. Every
    request in flight during the window counts. One still running when the
    drain ended counts with the time it had taken so far, a lower bound, and
    as a failed request, so an overloaded step is never flattered by leaving
    its slowest requests out. A subagent's requests count in the turn that
    spawned them."""
    lim = _as_limits(limits)
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
    step.p90_lo, step.p90_hi = bootstrap_p90(slow)
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
    step.checks = checks = {}

    # 1. mid-session evictions, exact per request
    warm = [r for r in rows if r.warm and r.uncached <= lim.warm_new_tokens]
    step.warm_requests = len(warm)
    known = [r for r in warm if r.ok and r.cached is not None]
    block = max(lim.kv_block_tokens, 1)
    step.kv_block_tokens = block
    evicted = [r for r in known
               if r.cached < r.expected // block * block - block]
    step.evictions = len(evicted)
    ms = lim.min_samples
    if not lim.eviction_check:
        checks["evictions"] = _check("skipped", limit=lim.max_eviction_share,
                                     n=len(warm))
    elif warm and not known:
        # the run checked the server reports cached tokens before the load:
        # a step where it stopped is a fail, not a pass
        checks["evictions"] = _check("unknown", limit=lim.max_eviction_share,
                                     n=len(warm))
        if any(r.ok for r in warm):
            step.reasons.append(
                f"evictions: the server reported no cached tokens on "
                f"{sum(r.ok for r in warm)} warm requests")
    elif not known and ms <= 0:
        checks["evictions"] = _check("n/a", limit=lim.max_eviction_share)
    elif len(known) < ms or not known:
        share = len(evicted) / len(known) if known else float("nan")
        checks["evictions"] = _check("few", share, lim.max_eviction_share,
                                     len(known), count=len(evicted), min=ms)
    else:
        share = len(evicted) / len(known)
        checks["evictions"] = _check(
            "pass" if share <= lim.max_eviction_share else "fail", share,
            lim.max_eviction_share, len(known), count=len(evicted), min=ms)
        if share > lim.max_eviction_share:
            step.reasons.append(
                f"evictions: {len(evicted)} of {len(known)} warm requests "
                f"lost their cached prefix ({share:.1%} > "
                f"{lim.max_eviction_share:.1%})")

    # 2. warm time to first token (a request still running counts with the
    # wait so far)
    waits = [r.ttft if r.ok else max(end_all - r.sent, 0.0)
             for r in warm if r.ok or not r.done]
    ttft_p90 = _pct(waits, PERCENTILE)
    if not waits and ms <= 0:
        checks["warm time to first token"] = _check(
            "n/a", limit=lim.max_warm_ttft_s)
    elif len(waits) < ms or not waits:
        checks["warm time to first token"] = _check(
            "few", ttft_p90, lim.max_warm_ttft_s, len(waits), min=ms)
    else:
        bad = ttft_p90 > lim.max_warm_ttft_s
        checks["warm time to first token"] = _check(
            "fail" if bad else "pass", ttft_p90, lim.max_warm_ttft_s,
            len(waits), min=ms)
        if bad:
            step.reasons.append(
                f"warm time to first token: p{PERCENTILE} {ttft_p90:.2f} s "
                f"> {lim.max_warm_ttft_s:g} s")

    # 3. decode speed per request
    rates = _decode_rates(rows, MIN_DECODE_TOKENS)
    p10 = _pct(rates, 10)
    if not rates:
        checks["decode speed"] = _check("n/a", limit=lim.min_decode_tok_s)
    elif len(rates) < ms:
        checks["decode speed"] = _check(
            "few", p10, lim.min_decode_tok_s, len(rates), min=ms)
    else:
        bad = p10 < lim.min_decode_tok_s
        checks["decode speed"] = _check(
            "fail" if bad else "pass", p10, lim.min_decode_tok_s, len(rates),
            min=ms)
        if bad:
            step.reasons.append(f"decode speed: p10 {p10:.1f} tok/s < "
                                f"{lim.min_decode_tok_s:g} tok/s")

    # 4. failed or unfinished requests
    failed = step.errors + step.unfinished
    fshare = failed / max(step.requests, 1)
    checks["failed requests"] = _check(
        "fail" if fshare > lim.max_failed_share else "pass", fshare,
        lim.max_failed_share, step.requests, count=failed)
    if fshare > lim.max_failed_share:
        first = next((r.error for r in rows if r.done and r.error), "")
        why = f"first: {first[:160]}" if first else (
            f"{step.unfinished} still running after the drain")
        step.reasons.append(f"failed requests: {failed} of {step.requests} "
                            f"failed ({why})")

    # cold requests: time to first token less the idle estimate, queueing
    cold = [r for r in rows if r.cold and r.ok]
    step.cold_requests = len(cold)
    queue = [r.ttft - speed.ttft(r.uncached, r.prompt) for r in cold]
    step.cold_queue_p50 = _pct(queue, 50)
    step.cold_queue_p90 = _pct(queue, PERCENTILE)

    # per-turn slowdown: information, a gate only when asked for
    if lim.max_slowdown is None:
        checks["turn slowdown"] = _check("info", step.slowdown_p90, None,
                                         step.turns)
    elif step.turns:
        bad = step.slowdown_p90 > lim.max_slowdown
        checks["turn slowdown"] = _check(
            "fail" if bad else "pass", step.slowdown_p90, lim.max_slowdown,
            step.turns)
        if bad:
            step.reasons.append(
                f"turn slowdown: p{PERCENTILE} turn "
                f"{step.slowdown_p90:.2f}x alone > {lim.max_slowdown:g}x")

    # per URL, and the background chat
    n_url = max([len(step.servers), step.n_urls] + [r.url + 1 for r in step.rows]
                + [r.url + 1 for r in step.chat_rows])
    step.url_requests = [sum(1 for r in rows if r.url == i)
                         for i in range(n_url)]
    step.chat = _chat_summary(step, w0, w1, end_all, n_url)

    step.inconclusive = step.no_evidence = False
    need = max(1, math.ceil(MIN_TURN_SHARE * step.sessions), min_turns)
    short = [f"{c['n']} {label} of {c['min']}" for name, label in (
        ("evictions", "warm requests"),
        ("warm time to first token", "warm waits"),
        ("decode speed", "decode-eligible requests"))
        if (c := checks[name])["status"] == "few"]
    if step.turns < need:
        short.insert(0, f"{step.turns} turns of {need}")
    if not step.reasons and short:
        # the server's own signals (`ran_out`): a queue that stayed or grew,
        # preemptions, a full KV cache. One request waiting for a tick is not
        queued = ran_out(step.server)[0] in ("kv", "queue")
        if queued:
            # requests waited on the server: the shortage is overload, and
            # counts as a fail
            step.inconclusive = True
            step.reasons.append(
                "inconclusive: only " + ", ".join(short) + f" judged in "
                f"{w1 - w0:.0f} s while requests waited on the server "
                "(counted as a fail; --max-measure-s measures longer)")
        else:
            # nothing failed, unfinished or queued: the sessions were idle
            step.no_evidence = True
            step.reasons.append(
                "no evidence: only " + ", ".join(short) + f" in "
                f"{w1 - w0:.0f} s, and nothing failed or queued, so idle, "
                "not overloaded (the search goes on; --max-measure-s "
                "measures longer)")
    step.passed = not step.reasons
    return step


def _num(v: float) -> float | None:
    return v if math.isfinite(v) else None


def _chat_summary(step: Step, w0: float, w1: float, end_all: float,
                  n_url: int) -> dict:
    """The background chat requests of the window, apart from the sessions:
    how many, their p90 time to first token, their p10 decode speed."""
    rows = [r for r in step.chat_rows if _in_window(r, w0, w1, end_all)]
    if not rows:
        return {}
    ok = [r for r in rows if r.ok]
    rates = _decode_rates(rows, MIN_DECODE_TOKENS)
    return {"requests": len(rows), "failed": sum(
                1 for r in rows if r.done and r.error)
            + sum(1 for r in rows if not r.done),
            "ttft_p90": _num(_pct([r.ttft for r in ok], PERCENTILE)),
            "decode_p10": _num(_pct(rates, 10)), "decode_n": len(rates),
            "url_requests": [sum(1 for r in rows if r.url == i)
                             for i in range(n_url)]}


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
def next_sessions(results: dict, max_sessions: int,
                  start: int = 1) -> int | None:
    """The next step to run, or None when the answer is known. Starts at
    `start`, doubles while every step passes (or shows no evidence: too few
    requests, nothing failed or queued), then halves the bracket between the
    largest pass and the smallest fail until it is within RESOLUTION of the
    pass. A result is True (pass), False (fail) or None (no evidence)."""
    if not results:
        return max(1, min(start, max_sessions))
    passes = [n for n, ok in results.items() if ok is True]
    fails = [n for n, ok in results.items() if ok is False]
    hi = min(fails) if fails else None
    lo = max((n for n in passes if hi is None or n < hi), default=0)
    if hi is None:
        seen = max(results)
        nxt = min(max(seen, 1) * 2, max_sessions)
        return None if nxt in results else nxt
    if lo == 0 or hi - lo <= max(1, math.floor(RESOLUTION * lo)):
        return None
    mid = (lo + hi) // 2
    return None if mid in results else mid


def answer(results: dict, max_sessions: int) -> dict:
    fails = [n for n, ok in results.items() if ok is False]
    hi = min(fails) if fails else None
    lo = max((n for n, ok in results.items() if ok is True
              and (hi is None or n < hi)), default=0)
    out = {"serves": lo, "fails_at": hi,
           "capped": hi is None and lo >= max_sessions}
    idle = sorted(n for n, ok in results.items() if ok is None)
    if idle:
        out["no_evidence"] = idle
    return out


# ============================================================================
# the run
# ============================================================================
@dataclass
class Plan:
    trace_path: str
    max_slowdown: float | None = None   # gates only when given
    min_samples: int = MIN_SAMPLES
    start_sessions: int = 1
    max_eviction_share: float = MAX_EVICTION_SHARE
    max_warm_ttft_s: float = MAX_WARM_TTFT_S
    min_decode_tok_s: float = MIN_DECODE_TOK_S
    max_failed_share: float = MAX_ERROR_SHARE
    warm_new_tokens: int = WARM_NEW_TOKENS
    kv_block_tokens: int = KV_BLOCK_TOKENS
    # where kv_block_tokens came from: "given" (--kv-block-tokens),
    # "/metrics" (vllm:cache_config_info) or "default"; the eviction check
    # also raises it to the block the replies show (block_from_replies)
    kv_block_source: str = "given"
    eviction_check: bool = True         # False: --skip-eviction-check
    routing: str = "sticky"             # with several URLs
    output_scale: float = 1.0
    chat_rate: float = 0.0              # background chat requests per second
    chat_prompt_tokens: int = 2000
    chat_output_tokens: int = 300
    chat_sigma: float = CHAT_SIGMA
    max_sessions: int = 256
    ramp_s: float = 60.0
    measure_s: float = 300.0
    min_turns: int = 30                 # a step measures on until it has these
    max_measure_s: float = 0.0          # ... up to this (0: 3 x measure_s)
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

    def limits(self) -> Limits:
        return Limits(self.max_eviction_share, self.max_warm_ttft_s,
                      self.min_decode_tok_s, self.max_failed_share,
                      self.warm_new_tokens, self.kv_block_tokens,
                      self.max_slowdown, self.min_samples,
                      eviction_check=self.eviction_check)

    @property
    def measure_cap_s(self) -> float:
        return max(self.max_measure_s or 3 * self.measure_s, self.measure_s)

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
                 roots=root_sessions(trace), start=start,
                 output_scale=plan.output_scale)


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
    if failed and len(failed) > CALIBRATION_ERROR_SHARE * len(rows):
        raise ValueError(
            f"{len(failed)} of {len(rows)} requests failed while measuring "
            f"the endpoint alone; the first: {failed[0].error[:300]}")
    speed = fit_speed(rows)
    if spawn_points(trace):
        tree_rows = await calibrate_trees(client, ep, plan, trace,
                                          prefix_text, run_nonce)
        for r in tree_rows:
            r.phase = "tree"
        rows += tree_rows
        beta, n = fit_tree_beta(tree_rows, speed)
        speed = replace(speed, tree_beta=beta, tree_n=n)
    return speed, rows


CALIBRATE_TREES = 3
CALIBRATE_TREE_S = 120.0


async def calibrate_trees(client, ep, plan: Plan, trace: AgentTrace,
                          prefix_text: str, run_nonce: str) -> list:
    """Whole trees replayed alone, one at a time, from the request that first spawns subagents,
    for `CALIBRATE_TREE_S` each: the recordings with the most subagents
    spawned early. Their requests show how much a tree's own streams slow
    each other (`fit_tree_beta`)."""
    points = spawn_points(trace)
    # start each tree at the request that first spawns subagents (what comes
    # before it is a lone stream); take those spawning the most at once
    first = {i: min(k for _, k in points[i]) for i in root_sessions(trace)
             if i in points}
    wide = {i: sum(1 for _, k in points[i] if k == first[i]) for i in first}
    roots = sorted(first, key=lambda i: (
        -wide[i], trace.sessions[i][first[i]].prompt_tokens))
    rows: list[Row] = []
    for n, root in enumerate(roots[:CALIBRATE_TREES]):
        stop = asyncio.Event()
        t0 = time.monotonic()
        uid = 900_000 + n * 1_000
        forest = _Forest(client, ep, plan, trace, prefix_text, run_nonce,
                         rows, t0, stop, None, uid_base=uid + 500)
        seat = _seat(trace, plan, prefix_text, uid, run_nonce, start=(root, first[root]))
        task = asyncio.create_task(_seat_loop(
            client, ep, plan.options(), seat, rows, t0, stop, 0.0, gaps=True,
            limit=len(trace.sessions[root]) - first[root],
            spawn=forest.spawner(uid)))
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


def lognormal_tokens(rng: random.Random, mean: float, sigma: float,
                     lo: int = 1) -> int:
    """A length with this mean, drawn lognormal: mu = ln(mean) - sigma^2 / 2
    puts the mean of the distribution (not its median) at `mean`."""
    if sigma <= 0:
        return max(lo, round(mean))
    mu = math.log(max(mean, 1e-9)) - sigma * sigma / 2
    return max(lo, round(rng.lognormvariate(mu, sigma)))


async def _chat_one(client, ep, plan: Plan, opts, k: int, nonce: str,
                    rows: list, t0: float, rng: random.Random) -> None:
    """One background chat request: its own prompt (a number and the run
    nonce first, so no two share a prefix), its own reply length."""
    from .probe.session import make_text
    cap = plan.context_cap_tokens
    out = lognormal_tokens(rng, plan.chat_output_tokens, plan.chat_sigma)
    n_in = lognormal_tokens(rng, plan.chat_prompt_tokens, plan.chat_sigma)
    if cap:
        n_in = min(n_in, max(cap - out, 1))
    text = (f"chat {k} {nonce}\n"
            + make_text(random.Random(f"chat-text {plan.seed} {k} {nonce}"),
                        n_in, plan.chars_per_token))
    i, spec = _target(ep, 2_000_000_000 + k)
    tr = RequestTrace(uid=-1000 - k, kind="chat", t_send=time.monotonic())
    row = Row(-1000 - k, 0, "chat", n_in, n_in, out,
              sent=time.monotonic() - t0, cold=True, url=i)
    rows.append(row)
    await send_request(client, spec, opts, text, tr, out, None)
    _finish(row, tr, t0)


async def _chat_load(client, ep, plan: Plan, nonce: str, rows: list,
                     t0: float, stop: asyncio.Event, tasks: list,
                     seed_tag: str) -> None:
    """Poisson arrivals at `plan.chat_rate` per second until `stop`."""
    rng = random.Random(f"chat {plan.seed} {seed_tag}")
    opts = plan.options()
    k = 0
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(),
                                   rng.expovariate(plan.chat_rate))
            return
        except asyncio.TimeoutError:
            pass
        k += 1
        tasks.append(asyncio.create_task(_chat_one(
            client, ep, plan, opts, k, nonce, rows, t0, rng)))


async def run_step(client, ep, plan: Plan, trace: AgentTrace, n: int,
                   prefix_text: str, run_nonce: str, metrics=None,
                   speed: Speed | None = None) -> Step:
    """`ep` is an EndpointSpec or a Router; `metrics` one sampler or a list,
    one per URL."""
    rows: list[Row] = []
    chat_rows: list[Row] = []
    stop = asyncio.Event()
    rng = random.Random(f"stagger {plan.seed} {n}")
    samplers = (list(metrics) if isinstance(metrics, (list, tuple))
                else [metrics])
    for m in samplers:
        await sampler_ready(m)
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
    chat_tasks: list = []
    if plan.chat_rate > 0:
        tasks.append(asyncio.create_task(_chat_load(
            client, ep, plan, run_nonce, chat_rows, t0, stop, chat_tasks,
            str(n))))
    s_start = [sampler_now(m) + plan.ramp_s for m in samplers]
    await asyncio.sleep(plan.ramp_s + plan.measure_s)
    # keep measuring until the window holds enough turns to say anything
    measured = plan.measure_s
    lim = plan.limits()
    decodes = any(r.output_tokens * plan.output_scale >= MIN_DECODE_TOKENS
                  for sess in trace.sessions for r in sess)
    while (speed is not None and measured < plan.measure_cap_s
           and (plan.min_turns or plan.min_samples)):
        now = Step(sessions=n, rows=list(rows), window=(
            plan.ramp_s, plan.ramp_s + measured))
        warm, dec = count_samples(now, lim)
        if (count_turns(now) >= min_turns(plan, n)
                and warm >= plan.min_samples
                and (dec >= plan.min_samples or not decodes)):
            break
        more = min(plan.measure_s / 6, plan.measure_cap_s - measured, 30.0)
        await asyncio.sleep(max(more, 0.01))
        measured += max(more, 0.01)
    s_end = [sampler_now(m) for m in samplers]
    # no new request starts; the ones in flight finish, so the slowest of
    # them are measured rather than dropped. A request still running after
    # the drain counts with the time it has taken so far (`judge`).
    stop.set()
    drain_s = 0.0
    await asyncio.sleep(0)      # the chat generator sees `stop` first
    pending = [t for t in tasks + forest.tasks + chat_tasks if not t.done()]
    if pending:
        t_d = time.monotonic()
        _, still = await asyncio.wait(pending, timeout=plan.measure_s)
        drain_s = time.monotonic() - t_d
        for t in still:
            t.cancel()
        await asyncio.gather(*still, return_exceptions=True)
    step = Step(sessions=n, rows=rows, chat_rows=chat_rows, drain_s=drain_s,
                window=(plan.ramp_s, plan.ramp_s + measured),
                n_urls=len(ep) if isinstance(ep, Router) else 1)
    if any(m is not None for m in samplers):
        step.servers = [_server(await sampler_window(m, a, b))
                        for m, a, b in zip(samplers, s_start, s_end)]
        step.server = merge_servers(step.servers)
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


def check_cache_reporting(rows, plan: Plan) -> None:
    """Before any load: the server must report cached tokens, in blocks no
    larger than the one the eviction check rounds to. Raises ValueError with
    the flag that overrides it."""
    if not plan.eviction_check:
        return
    ok = [r for r in rows if r.ok]
    if ok and all(r.cached is None for r in ok):
        raise ValueError(
            "the server reports no cached tokens (vLLM needs "
            "--enable-prompt-tokens-details), so evictions cannot be counted. "
            "Pass --skip-eviction-check to run without that check")
    g = block_from_replies(ok)
    if g > plan.kv_block_tokens and plan.kv_block_source != "given":
        raise ValueError(
            f"the server's cached tokens come in whole blocks of {g:,}, but "
            f"the block from {plan.kv_block_source} is "
            f"{plan.kv_block_tokens:,}: every warm request would read as an "
            f"eviction. Pass --kv-block-tokens {g} if that is the block")


async def run(ep, plan: Plan, metrics=None, on_step=_say) -> dict:
    """`ep`: an EndpointSpec, or a list of them (a data-parallel deployment,
    routed by `plan.routing`); `metrics`: a sampler, or a list paired with
    them. The endpoint alone is fitted on the first."""
    eps = list(ep) if isinstance(ep, (list, tuple)) else [ep]
    ep0 = eps[0]
    target = Router(eps, plan.routing) if len(eps) > 1 else ep0
    trace = load_trace(plan.trace_path)
    from .probe.options import process_nonce
    nonce = process_nonce()
    async with make_client() as client:
        if plan.cpt_source == "default":
            cpt = await measure_chars_per_token(client, ep0, plan, nonce)
            if cpt:
                plan.chars_per_token, plan.cpt_source = round(cpt, 3), \
                    "measured"
                on_step(f"chars per token {plan.chars_per_token:g}, from the "
                        f"server's prompt token counts ({plan.cpt_probes} "
                        "probes)")
            else:
                raise ValueError(
                    "the server reported no prompt token counts, so chars "
                    "per token cannot be measured: pass --chars-per-token or "
                    "--tokenizer")
        prefix = _prefix_text(plan)
        on_step(f"alone: up to {plan.calibrate_requests} requests back to "
                "back, one in flight")
        speed, cal_rows = await calibrate(client, ep0, plan, trace, prefix,
                                          nonce)
        check_cache_reporting(cal_rows, plan)
        on_step(describe_speed(speed, trace, plan))
        results: dict[int, bool | None] = {}
        steps: list[Step] = []
        while (n := next_sessions(results, plan.max_sessions,
                                  plan.start_sessions)) is not None:
            on_step(f"\n--- {n} session{'s' if n != 1 else ''} ---")
            step = judge(await run_step(client, target, plan, trace, n, prefix,
                                        nonce, metrics, speed),
                         speed, plan.limits(), min_turns(plan, n))
            steps.append(step)
            results[n] = None if step.no_evidence else step.passed
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


UNKNOWN_CACHED = ("the server reported no cached_tokens in this step (vLLM "
                  "needs --enable-prompt-tokens-details): evictions cannot be "
                  "counted, so the step fails")


def _word(s: Step) -> str:
    return ("NO EVIDENCE" if s.no_evidence else
            "INCONCLUSIVE" if s.inconclusive else "FAIL")


def check_cells(s: Step) -> dict[str, str]:
    """Each check as a short cell: its value against its limit, or why it has
    none."""
    out = {}
    for name in CHECKS:
        c = s.checks.get(name)
        if c is None:
            continue
        st, v, lim = c["status"], c["value"], c["limit"]
        if v is None and st not in ("n/a", "unknown", "skipped"):
            out[name] = (f"too few samples ({c['n']} < {c.get('min')}), "
                         "not judged" if st == "few" else "-")
            continue
        if name == "evictions":
            cell = {"unknown": "unknown", "n/a": "no warm request",
                    "skipped": "skipped (--skip-eviction-check)"}.get(st) or (
                f"{c.get('count', 0)}/{c['n']} = {v:.1%} (<= {lim:.1%})")
        elif name == "warm time to first token":
            cell = ("no warm request" if st == "n/a" else
                    f"p{PERCENTILE} {v:.2f} s (<= {lim:g} s)")
        elif name == "decode speed":
            cell = ("no request of {0}+ tokens".format(MIN_DECODE_TOKENS)
                    if st == "n/a" else f"p10 {v:.1f} tok/s (>= {lim:g})")
        elif name == "failed requests":
            cell = f"{c.get('count', 0)}/{c['n']} = {v:.1%} (<= {lim:.1%})"
        else:
            cell = (f"p{PERCENTILE} {v:.2f}x alone" + (
                f" (<= {lim:g}x)" if lim is not None else ", not gating")
                    if v is not None else "-")
        if st == "few":
            cell = (f"too few samples ({c['n']} < {c['min']}), not judged"
                    + (f": {cell}" if v is not None else ""))
        out[name] = f"{cell}{' FAIL' if st == 'fail' else ''}"
    return out


def step_line(s: Step) -> str:
    v = ("PASS" if s.passed else _word(s) + " ("
         + "; ".join(s.reasons) + ")")
    sig = signals(s.server)
    failed = s.errors + s.unfinished
    cells = check_cells(s)
    lines = [f"  {s.turns} turns, {s.requests} requests, {failed} failed, "
             f"{_f1(s.streams_mean)} requests in flight | {v}"]
    lines += [f"    {name}: {cell}" for name, cell in cells.items()]
    if s.checks.get("evictions", {}).get("status") == "unknown":
        lines.append("    " + UNKNOWN_CACHED)
    if s.cold_requests:
        lines.append(f"    cold requests: {s.cold_requests}, queueing p50 "
                     f"{_sec(s.cold_queue_p50)} p{PERCENTILE} "
                     f"{_sec(s.cold_queue_p90)} (time to first token less "
                     "the idle estimate; not gated)")
    lines.append(f"    information: turn p90 {_x(s.slowdown_p90)}{_ci(s)} "
                 f"alone, mean {_x(s.slowdown_mean)} (vs one stream: p90 "
                 f"{_x(s.raw_p90)}, mean {_x(s.raw_mean)}); turn time p50 "
                 f"{_s(s.turn_s_p50)} p90 {_s(s.turn_s_p90)}")
    if len(s.url_requests) > 1:
        lines.append("    requests per URL: " + ", ".join(
            f"#{i} {n}" for i, n in enumerate(s.url_requests)))
    if s.chat:
        lines.append("    " + chat_line(s.chat))
    if sig:
        lines.append(f"    server: {sig}")
    for i, srv in enumerate(s.servers if len(s.servers) > 1 else ()):
        lines.append(f"    server #{i}: {signals(srv) or 'no reading'}")
    if not s.passed:
        lines.append(f"    the limit is {ran_out(s.server)[1]}")
    return "\n".join(lines)


def chat_line(c: dict) -> str:
    d = c.get("decode_p10")
    t = c.get("ttft_p90")
    return (f"background chat: {c['requests']} requests, {c.get('failed', 0)} "
            f"failed, p{PERCENTILE} time to first token "
            f"{_sec(t) if t is not None else '-'}, p10 decode speed "
            + (f"{d:.1f} tok/s" if d is not None else "-")
            + " (not judged)")


def _sec(v: float) -> str:
    return f"{v:.2f} s" if v is not None and math.isfinite(v) else "-"


def _ci(s: Step) -> str:
    return (f" [{s.p90_lo:.2f}-{s.p90_hi:.2f}]"
            if math.isfinite(s.p90_lo) else "")


def _f1(v: float) -> str:
    return f"{v:.1f}" if math.isfinite(v) else "-"


def _x(v: float) -> str:
    return f"{v:.2f}x" if math.isfinite(v) else "-"


def _s(v: float) -> str:
    return f"{v:.0f} s" if math.isfinite(v) else "-"


def failed_checks(step: dict) -> list[str]:
    """The names of the checks a step (as in a record) failed."""
    return [n for n in CHECKS if (step.get("checks") or {}).get(n, {}
                                                                ).get("status") == "fail"]


def rule_words(plan: dict) -> str:
    """The pass rule of a record's plan, in words. A record from before the
    checks held the turn slowdown and 5% failed requests."""
    if "max_warm_ttft_s" not in plan:
        return (f"p{PERCENTILE} turn <= {plan['max_slowdown']:g}x alone")
    words = (f"evictions <= {plan['max_eviction_share']:.1%} of warm "
             f"requests, warm time to first token p{PERCENTILE} <= "
             f"{plan['max_warm_ttft_s']:g} s, decode speed p10 >= "
             f"{plan['min_decode_tok_s']:g} tok/s, failed requests <= "
             f"{plan['max_failed_share']:.1%}")
    if plan.get("max_slowdown") is not None:
        words += f", p{PERCENTILE} turn <= {plan['max_slowdown']:g}x alone"
    return words


def headline(rec: dict) -> str:
    a = rec["answer"]
    rule = rule_words(rec["plan"])
    idle = a.get("no_evidence") or []
    note = (f"; no evidence at {', '.join(map(str, idle))} (too few requests "
            "in the window, nothing failed or queued: the sessions sat idle)"
            if idle else "")
    if a["fails_at"] is None and not a["serves"]:
        return ("no verdict: every step had too few requests to judge"
                + note + " (lengthen --measure-s and --max-measure-s, or "
                "start higher with --start-sessions)")
    if a["fails_at"] is None:
        cap = ("the --max-sessions cap; nothing failed" if a["capped"]
               else "nothing failed")
        return (f"serves at least {a['serves']} sessions within {rule} "
                f"({cap}){note}")
    out = ""
    bad = next((st for st in rec.get("steps", ())
                if st.get("sessions") == a["fails_at"]), None)
    if bad is not None:
        key, words = ran_out(bad.get("server"))
        out = (f"; at {a['fails_at']} the limit is {words}" if key != "unknown"
               else "; the limit is unknown (no /metrics reading)")
        names = failed_checks(bad)
        if names:
            out = f"; at {a['fails_at']} failed: {', '.join(names)}" + out
        if bad.get("inconclusive"):
            out = (f"; the step at {a['fails_at']} is INCONCLUSIVE ("
                   f"{bad.get('turns')} turns judged, too few to call it a "
                   "fail while requests waited; counted as a fail)")
    out += note
    if a["serves"] == 0:
        return (f"serves no session within {rule} (the first step that "
                f"had evidence already fails: see the step above){out}")
    return f"serves {a['serves']} sessions within {rule}; fails at " \
           f"{a['fails_at']}{out}"


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
    eps = rec.get("endpoints") or []
    if len(eps) > 1:
        w(f"         {len(eps)} URLs, routing {rec['plan'].get('routing')}: "
          + ", ".join(e["base_url"] for e in eps))
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
      f"{'turn p90 [90% CI]':>21} {'mean':>6} {'vs 1 stream':>12} "
      f"{'turn s p50/p90':>15} {'in flight':>9}  verdict")
    for s in steps:
        v = ("PASS" if s.passed else _word(s) + ": " + "; ".join(s.reasons))
        w(f"{s.sessions:>8} {s.turns:>6} {s.requests:>6} {s.errors:>5} "
          f"{_x(s.slowdown_p90) + _ci(s):>21} {_x(s.slowdown_mean):>6} "
          f"{_x(s.raw_p90):>12} {_s(s.turn_s_p50):>7}/{_s(s.turn_s_p90):<7} "
          f"{_f1(s.streams_mean):>9}  {v}")
    if any(s.checks for s in steps):
        w(f"\nthe checks per step (pass rule: {rule_words(pl)})")
        for s in steps:
            for name, cell in check_cells(s).items():
                w(f"{s.sessions:>8}  {name}: {cell}")
            if s.cold_requests:
                w(f"{s.sessions:>8}  cold requests: {s.cold_requests}, "
                  f"queueing p50 {_sec(s.cold_queue_p50)} p{PERCENTILE} "
                  f"{_sec(s.cold_queue_p90)} (not gated)")
            if len(s.url_requests) > 1:
                w(f"{s.sessions:>8}  requests per URL: " + ", ".join(
                    f"#{i} {n}" for i, n in enumerate(s.url_requests)))
            if s.chat:
                w(f"{s.sessions:>8}  {chat_line(s.chat)}")
        if any(c["status"] == "unknown" for s in steps
               for c in [s.checks.get("evictions", {})]):
            w(f"\n{UNKNOWN_CACHED}")
    sig = [(s.sessions, signals(s.server)) for s in steps if s.server]
    if sig:
        w("\nwhat the server showed (--metrics-url)")
        for n, line in sig:
            w(f"{n:>8}  {line}")
            if len(s_ := next(x for x in steps if x.sessions == n
                              ).servers) > 1:
                for i, srv in enumerate(s_):
                    w(f"{'':>8}    URL #{i}: {signals(srv) or 'no reading'}")
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
def _endpoint_info(base_url: str, api_key: str,
                   model: str | None = None) -> dict:
    """The served model id and its max_model_len, from /models (vLLM lists
    both): the entry for `model`, or the only one served. {"error": ...}
    when the endpoint does not answer, does not serve `model`, or serves
    several and none was named."""
    import httpx
    try:
        r = httpx.get(f"{base_url}/models", timeout=10.0, headers=(
            {"Authorization": f"Bearer {api_key}"} if api_key else {}))
        r.raise_for_status()
        data = r.json().get("data") or []
    except Exception as e:                          # noqa: BLE001
        return {"error": str(e)}
    ids = [d.get("id", "") for d in data]
    if model:
        pick = [d for d in data if d.get("id") == model]
        if not pick:
            return {"error": f"it does not serve {model!r} (served: "
                             f"{', '.join(ids) or 'none'})"}
    elif len(data) == 1:
        pick = data
    else:
        return {"error": f"it serves {len(data)} models "
                         f"({', '.join(ids) or 'none'}): pass --model"}
    return {"model": pick[0].get("id", ""),
            "max_model_len": pick[0].get("max_model_len")}


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
    bases = [b.rstrip("/") for b in (args.base_url or [DEFAULT_BASE_URL])]
    base = bases[0]
    murls = list(args.metrics_url or [])
    if args.no_metrics and murls:
        print("ws replay: --no-metrics and --metrics-url contradict each "
              "other", file=sys.stderr)
        return 2
    if not murls and not args.no_metrics:
        from urllib.parse import urlsplit
        murls = ["{0.scheme}://{0.netloc}/metrics".format(urlsplit(b))
                 for b in bases]
        print("  --metrics-url not given: reading " + ", ".join(murls))
    if murls and len(bases) > 1 and len(murls) != len(bases):
        print(f"ws replay: {len(murls)} --metrics-url for {len(bases)} "
              "--base-url: give one per base URL, in the same order (one "
              "--base-url with several --metrics-url is a router in front of "
              "replicas)", file=sys.stderr)
        return 2
    for flag, v, lo in (("--output-scale", args.output_scale, 0.0),
                        ("--chat-rate", args.chat_rate, -1e-9)):
        if not v > lo:
            print(f"ws replay: {flag} must be positive" if lo == 0.0 else
                  f"ws replay: {flag} cannot be negative", file=sys.stderr)
            return 2
    def fail(msg: str) -> int:
        print(f"ws replay: {msg}", file=sys.stderr)
        return 2

    info = _endpoint_info(base, key, args.model)   # a dry run reads it too
    if "error" in info:
        return fail(f"GET {base}/models failed ({info['error']}): check "
                    "--base-url and --api-key-env")
    cap = args.max_model_len or info.get("max_model_len")
    if not cap:
        return fail(f"{base}/models gives no max_model_len: pass "
                    "--max-model-len")
    texts = []
    for i, u in enumerate(murls):
        b = bases[i] if len(bases) > 1 else base
        try:
            texts.append(read_metrics(u, metrics_headers(
                args.metrics_header, key, b, u)))
        except ValueError as e:
            return fail(str(e))
    blocks = {u: block_from_metrics(t) for u, t in zip(murls, texts)}
    if len(set(blocks.values())) > 1:
        return fail("the replicas disagree on the KV block: " + ", ".join(
            f"{u} {v}" for u, v in blocks.items()))
    block, block_source = args.kv_block_tokens, "given"
    seen = block_from_metrics(texts[0]) if texts else None
    if block is None:
        if seen is None:
            return fail("the KV block size is unknown (" + (
                "--no-metrics" if not texts else
                f"{murls[0]} has no vllm:cache_config_info block_size")
                + "): pass --kv-block-tokens")
        block, block_source = seen, "/metrics"
    caching = prefix_caching_from_metrics(texts[0]) if texts else None
    if caching is False and not args.skip_eviction_check:
        return fail("the server runs without prefix caching, so there is no "
                    "cache to evict from: pass --skip-eviction-check")
    cpt, cpt_source = args.chars_per_token, "given"
    if args.tokenizer:
        from .test_cmd import _calibrated_cpt
        cpt = _calibrated_cpt(args.tokenizer)
    elif cpt is None:
        cpt, cpt_source = 4.0, "default"
    plan = Plan(trace_path=args.trace, max_slowdown=args.max_slowdown,
                min_samples=args.min_samples,
                start_sessions=args.start_sessions,
                max_eviction_share=args.max_eviction_share,
                max_warm_ttft_s=args.max_warm_ttft_s,
                min_decode_tok_s=args.min_decode_tok_s,
                max_failed_share=args.max_failed_share,
                warm_new_tokens=args.warm_new_tokens,
                kv_block_tokens=block, kv_block_source=block_source,
                eviction_check=not args.skip_eviction_check,
                routing=args.routing or "sticky",
                output_scale=args.output_scale, chat_rate=args.chat_rate,
                chat_prompt_tokens=args.chat_prompt_tokens,
                chat_output_tokens=args.chat_output_tokens,
                chat_sigma=args.chat_sigma,
                max_sessions=args.max_sessions, ramp_s=args.ramp_s,
                measure_s=args.measure_s, min_turns=args.min_turns,
                max_measure_s=args.max_measure_s,
                shared_prefix_tokens=(args.shared_prefix_tokens
                                      if args.shared_prefix_tokens is not None
                                      else default_shared_prefix(trace)),
                context_cap_tokens=int(cap), chars_per_token=cpt,
                tokenizer=args.tokenizer, seed=args.seed,
                cpt_source=cpt_source,
                request_timeout_s=args.request_timeout_s,
                calibrate_requests=CALIBRATE_REQUESTS,
                calibrate_max_s=CALIBRATE_MAX_S)
    model = args.model or info.get("model", "")
    eps = [EndpointSpec(base_url=b, model=model, api_key=key, api=args.api)
           for b in bases]
    ep = eps[0]
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
    print(f"  KV block {plan.kv_block_tokens:,} tokens ("
          + ("from /metrics" if plan.kv_block_source == "/metrics" else
             "--kv-block-tokens" + (f"; /metrics says {seen:,}"
                                    if seen and seen != block else ""))
          + ")" + ("; eviction check skipped (--skip-eviction-check)"
                   if not plan.eviction_check else ""))
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
    print(f"  a step passes when: "
          f"{rule_words(asdict(plan))}")
    if len(eps) > 1:
        print(f"  {len(eps)} URLs, routing {plan.routing}: "
              + ", ".join(e.base_url for e in eps)
              + ("; a session tree stays on the URL its seat id hashes to"
                 if plan.routing == "sticky" else
                 "; every request goes to the next URL"))
    if plan.output_scale != 1.0:
        print(f"  replies and the context they add scaled by "
              f"{plan.output_scale:g}")
    if plan.chat_rate > 0:
        print(f"  background chat: {plan.chat_rate:g} requests/s, mean "
              f"{plan.chat_prompt_tokens:,} prompt and "
              f"{plan.chat_output_tokens:,} output tokens (lognormal, sigma "
              f"{plan.chat_sigma:g}); reported apart, not judged")
    print(f"  steps: 1, 2, 4, ... sessions until one fails, then halved to "
          f"{RESOLUTION:.0%}; each {plan.ramp_s:g} s ramp + "
          f"{plan.measure_s:g} s measured, longer (up to "
          f"{plan.measure_cap_s:g} s) until {plan.min_turns} turns (5 per session at most) are judged "
          f"(cap {plan.max_sessions})")
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
    if murls:
        from .test_cmd import open_metrics
        metrics = [open_metrics(u, args.engine,
                                metrics_headers(
                                    args.metrics_header, key,
                                    bases[i] if len(bases) > 1 else base, u))
                   for i, u in enumerate(murls)]

    async def go():
        from .test_cmd import _maybe
        for m in metrics or ():
            await _maybe(m, "start", "open")
        try:
            return await run(eps, plan, metrics)
        finally:
            for m in metrics or ():
                await _maybe(m, "stop", "aclose", "close")

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
           "endpoints": [e.redacted() for e in eps],
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
    if args.out:
        print(f"\nrun record written to {args.out}")
    if (any(m is not None for m in metrics or ()) and res["steps"]
            and not any(s.server for s in res["steps"])):
        print("\nws replay: /metrics answered before the run but gave no "
              "reading during it: the verdict above cannot say what ran out",
              file=sys.stderr)
        return 4
    return 0


def is_record(d: dict) -> bool:
    return isinstance(d, dict) and d.get("format") == FORMAT


def add_subparser(sub) -> None:
    import argparse
    p = sub.add_parser(
        "replay", help="how many recorded agent sessions an endpoint serves",
        description="Replay recorded agent sessions (`ws trace pi`) against "
                    "an OpenAI-compatible endpoint, at 1, 2, 4, ... sessions "
                    "at once, and report the most it serves while every "
                    "check of the pass rule holds (mid-session evictions, "
                    "warm time to first token, decode speed, failed "
                    "requests: see the flags below). Needs no config: "
                    "nothing in the answer comes from the model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  ws replay trace.json --base-url http://gpu:8000/v1 "
               "--out run.json\n"
               "  ws replay trace.json --base-url http://gpu:8000/v1 "
               "--metrics-url http://gpu:8000/metrics --max-warm-ttft-s 2\n"
               "  ws replay trace.json --base-url http://a:8000/v1 "
               "--base-url http://b:8000/v1 --routing round-robin\n"
               "  ws report run.json\n")
    p.add_argument("trace", help="agent trace from `ws trace pi`")
    p.add_argument("--base-url", action="append", default=None,
                   help="the endpoint (default http://localhost:8000/v1). "
                        "Repeat it to bench a data-parallel deployment: "
                        "--routing says how requests are spread")
    p.add_argument("--routing", choices=ROUTINGS, default=None,
                   help="with several --base-url: sticky (default) pins a "
                        "session tree to one URL by a stable hash of its "
                        "seat id, as a session-affine router does; "
                        "round-robin sends each request to the next URL, "
                        "as a router without affinity does")
    p.add_argument("--model", help="served model id (default: the first one "
                                   "/models lists)")
    p.add_argument("--api-key-env", default="VLLM_API_KEY",
                   help="environment variable holding the API key")
    p.add_argument("--api", choices=("completions", "chat"),
                   default="completions")
    g = p.add_argument_group(
        "pass rule", "A step of N sessions passes when every check holds "
                     "over its window. Each threshold is a flag.")
    g.add_argument("--min-samples", type=int, default=MIN_SAMPLES,
                   help="a check judges only with this many requests (warm "
                        "ones for evictions and time to first token, "
                        "decode-eligible ones for decode speed); a step "
                        "measures on, up to --max-measure-s, to reach them. "
                        "A step that ends short with nothing failed or "
                        "queued shows NO EVIDENCE: not a fail, the search "
                        "goes on (default %(default)s)")
    p.add_argument("--start-sessions", type=int, default=1,
                   help="sessions of the first step (default 1); doubling "
                        "goes on from there")
    g.add_argument("--max-eviction-share", type=float,
                   default=MAX_EVICTION_SHARE,
                   help="mid-session evictions at most this share of warm "
                        "requests (default %(default)s). An eviction: the "
                        "server's usage.prompt_tokens_details.cached_tokens "
                        "is more than one block under the tokens the "
                        "request shares with its session's previous "
                        "request. Needs vLLM --enable-prompt-tokens-details: "
                        "the run stops before any load if the server reports "
                        "no cached tokens")
    g.add_argument("--skip-eviction-check", action="store_true",
                   help="run without the eviction check (a server that "
                        "reports no cached tokens, or runs without prefix "
                        "caching)")
    g.add_argument("--max-warm-ttft-s", type=float, default=MAX_WARM_TTFT_S,
                   help="p90 time to first token of warm requests, seconds "
                        "(default %(default)s)")
    g.add_argument("--warm-new-tokens", type=int, default=WARM_NEW_TOKENS,
                   help="a request is warm when its session's previous "
                        "request exists, no compaction or reset lies "
                        "between, and it adds at most this many new tokens "
                        "(default %(default)s)")
    g.add_argument("--kv-block-tokens", type=int, default=None,
                   help="the server's KV cache block size: the shared prefix "
                        "rounds down to it and the eviction test tolerates "
                        "one block (default: block_size from /metrics; the "
                        "run stops when neither says, or when the replies' "
                        "cached tokens come in larger blocks)")
    g.add_argument("--min-decode-tok-s", type=float, default=MIN_DECODE_TOK_S,
                   help="p10 decode speed per request, (output - 1) / (last "
                        "- first token time), over requests of at least "
                        f"{MIN_DECODE_TOKENS} output tokens (default "
                        "%(default)s tok/s)")
    g.add_argument("--max-failed-share", type=float, default=MAX_ERROR_SHARE,
                   help="failed or unfinished requests at most this share "
                        "(default %(default)s)")
    g.add_argument("--max-slowdown", type=float, default=None,
                   help=f"also require the p{PERCENTILE} turn to take at "
                        "most this many times as long as the endpoint alone. "
                        "Without it the slowdown is reported, not gated")
    p.add_argument("--output-scale", type=float, default=1.0,
                   help="multiply every recorded reply length, and the "
                        "context those replies add to the session's later "
                        "prompts, by this (default 1): replays a model that "
                        "writes more or less per task")
    p.add_argument("--chat-rate", type=float, default=0.0,
                   help="background chat requests per second, Poisson, each "
                        "with its own prompt (default 0: none). Reported "
                        "apart (count, p90 time to first token, p10 decode "
                        "speed), never judged")
    p.add_argument("--chat-prompt-tokens", type=int, default=2000,
                   help="mean prompt tokens of a chat request (lognormal)")
    p.add_argument("--chat-output-tokens", type=int, default=300,
                   help="mean output tokens of a chat request (lognormal)")
    p.add_argument("--chat-sigma", type=float, default=CHAT_SIGMA,
                   help="sigma of the lognormal chat lengths "
                        "(default %(default)s)")
    p.add_argument("--max-sessions", type=int, default=256)
    p.add_argument("--min-turns", type=int, default=30,
                   help="a step measures past --measure-s until it has this "
                        "many judged turns (default 30); with fewer by "
                        "--max-measure-s it shows no evidence (idle: the "
                        "search goes on), or is inconclusive (a fail) if "
                        "requests waited on the server")
    p.add_argument("--max-measure-s", type=float, default=0.0,
                   help="longest a step measures (default 3 x --measure-s)")
    p.add_argument("--ramp-s", type=float, default=60.0,
                   help="seconds sessions start over, not measured")
    p.add_argument("--measure-s", type=float, default=300.0,
                   help="seconds measured per step")
    p.add_argument("--metrics-url", action="append", default=None,
                   help="the server's Prometheus /metrics, to say what ran "
                        "out. Repeat it once per --base-url, in the same "
                        "order")
    p.add_argument("--no-metrics", action="store_true",
                   help="run without /metrics (by default each base URL's "
                        "own /metrics is read, and the run stops if it does "
                        "not answer); --kv-block-tokens is then required")
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
