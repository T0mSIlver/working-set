"""Public prefix-cache traces: read them, replay them through an exact LRU,
and check the capacity model against what the replay and the papers say.

Each reader turns a published trace into `Request`s: arrival time, session
(when the trace has one), prompt and output tokens, and the request's prompt
as a list of block ids. Blocks are turned into *prefix pages*
(`prefix_pages`): page k's key stands for the whole prompt up to the end of
page k, as a radix tree sees it, so a page can only hit where every earlier
page matched too. That holds whether or not a trace's ids are chained (the
Mooncake and Bailian coder ids are; ~5% of the Bailian to-C repeats are not).

Three ways to price a capacity C (tokens), from most to least information:

  exact     `lru_curve`: one global LRU over prefix pages, a request's pages
            touched tail first (vLLM frees a request's blocks tail first, so
            its prefix is the last to go). A page hits at C if its stack
            distance is under C/page and so is every earlier page's. The
            distance is taken when the page is touched, so it counts the
            request's own tail pages touched just before: at a capacity
            near one request's length a long prompt can push out its own
            prefix (on the real traces, under 0.1 point). One pass
            gives the whole curve (Mattson), as KVSET does.
  denning   `denning_curve`: Denning's working set on the same pages. A page
            reused within T hits; T is where the mean number of distinct
            tokens touched in a window of T reaches C. Needs the block ids
            and the times, no simulation.
  planner   `planner_curve`: the same characteristic-time rule fed only what
            a capacity planner knows about a workload: the request rate, the
            prompt-size distribution, each request's gap since its session's
            previous request and how much of the prompt that request could
            reuse. No block ids. This is the form the model can use.

`fill_users` is the model's original rule (model.warm_capacity's fill): a
cache holds as many sessions as their unique contexts fill; with more live
sessions than that, LRU over a cyclic access order collapses to ~0
(docs/writeup.md). research/public_traces.md records where each method lands
against the exact replay and the published numbers.
"""
from __future__ import annotations

import bisect
import gzip
import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Request:
    t: float                    # arrival, seconds from the trace's start
    session: str | None         # None: the trace has no session ids
    tokens_in: int
    tokens_out: int
    blocks: list[int]           # the prompt's block ids, in order
    bs: int                     # tokens per block id
    cached: int | None = None   # tokens a real server reported as cached


def _open(path: str | Path):
    p = str(path)
    return gzip.open(p, "rt") if p.endswith(".gz") else open(p)


def read_mooncake(path: str | Path) -> list[Request]:
    """Mooncake FAST'25 traces: timestamp (ms), 512-token global hash ids, no
    sessions."""
    out = []
    with _open(path) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                out.append(Request(d["timestamp"] / 1000.0, None, d["input_length"],
                                   d["output_length"], d["hash_ids"], 512))
    out.sort(key=lambda r: r.t)
    return out


def read_bailian(path: str | Path) -> list[Request]:
    """Alibaba Bailian traces: timestamp (s), 16-token salted block ids, chat
    turns linked by parent_chat_id. A session is the root of the chain."""
    rows = []
    with _open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    root: dict[int, int] = {}
    for d in sorted(rows, key=lambda d: d["timestamp"]):
        p = d["parent_chat_id"]
        root[d["chat_id"]] = root.get(p, p) if p != -1 else d["chat_id"]
    out = [Request(d["timestamp"], str(root[d["chat_id"]]), d["input_length"],
                   d["output_length"], d["hash_ids"], 16) for d in rows]
    out.sort(key=lambda r: r.t)
    return out


def read_tracelab(path: str | Path) -> tuple[dict, dict]:
    """TraceLab (uw-syfi/TraceLab, CC BY 4.0): 357k rounds of Claude Code and
    Codex sessions with the provider's own prompt-cache counters, no block
    ids. A round reuses its session's previous prompt when it is no shorter
    (append-only; a shorter one is a compaction and reuses nothing). Gaps run
    from one round's first timing event to the next's. Returns the session
    statistics per provider ("claude", "codex", "all") and, per provider,
    the share of prompt tokens the provider reported as cached."""
    from datetime import datetime

    def ts(x: str) -> float:
        return datetime.fromisoformat(x).timestamp()

    rows = []
    with _open(path) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                ev = d.get("timing_events") or []
                if not ev:
                    continue
                rows.append((d["provider"], d["session_id"], d["round_index"], ts(ev[0]["timestamp"]),
                             d["input_tokens_total"] or 0, d["prefix_tokens"] or 0))
    rows.sort(key=lambda r: (r[1], r[2]))
    out, measured = {}, {}
    for prov in ("claude", "codex", "all"):
        sel = [r for r in rows if prov == "all" or r[0] == prov]
        prompt, reusable, gap_prev, gap_next = [], [], [], []
        t_min, t_max = math.inf, -math.inf
        for i, r in enumerate(sel):
            t_min, t_max = min(t_min, r[3]), max(t_max, r[3])
            prev = sel[i - 1] if i and sel[i - 1][1] == r[1] else None
            prompt.append(r[4])
            if prev is None:
                reusable.append(0)
                gap_prev.append(math.inf)
            else:
                reusable.append(prev[4] if r[4] >= prev[4] else 0)
                gap_prev.append(r[3] - prev[3])
            nxt = sel[i + 1] if i + 1 < len(sel) and sel[i + 1][1] == r[1] else None
            gap_next.append(nxt[3] - r[3] if nxt else math.inf)
        out[prov] = SessionStats(t_max - t_min, prompt, reusable, gap_prev, gap_next, list(reusable))
        measured[prov] = sum(r[5] for r in sel) / max(sum(prompt), 1)
    return out, measured


READERS = {"mooncake": read_mooncake, "bailian": read_bailian}


# --- prefix pages -----------------------------------------------------------

def prefix_pages(reqs: list[Request], page_tokens: int) -> list[list[int]]:
    """Each request's prompt as prefix-page keys (ints). Page k covers blocks
    [k*g, (k+1)*g) with g = page_tokens // bs; a partial last page is dropped,
    as a block-granular cache would. The page must be a whole number of
    blocks: a smaller one would price a block-sized page at page_tokens."""
    ids: dict[tuple, int] = {}
    out = []
    for r in reqs:
        if page_tokens % r.bs:
            raise ValueError(f"page of {page_tokens} tokens is not a whole number "
                             f"of {r.bs}-token blocks")
        g = page_tokens // r.bs
        keys, prev = [], -1
        for k in range(len(r.blocks) // g):
            key = (prev, *r.blocks[k * g:(k + 1) * g])
            prev = ids.setdefault(key, len(ids))
            keys.append(prev)
        out.append(keys)
    return out


# --- exact LRU (Mattson stack distances) --------------------------------------

class _Fenwick:
    def __init__(self, n: int):
        self.n = n
        self.t = [0] * (n + 1)

    def add(self, i: int, v: int) -> None:
        i += 1
        while i <= self.n:
            self.t[i] += v
            i += i & -i

    def prefix(self, i: int) -> int:     # sum of [0, i)
        s = 0
        while i > 0:
            s += self.t[i]
            i -= i & -i
        return s


@dataclass
class Curve:
    """Hit rate as a function of capacity, token-weighted over prompt tokens."""
    page_tokens: int
    prompt_tokens: int
    depth: list[int]            # per page access: pages needed to hit it (inf = cold)
    ceiling: float

    def hit(self, capacity_tokens: float) -> float:
        pages = capacity_tokens / self.page_tokens
        n = bisect.bisect_right(self._sorted, pages - 1) if pages >= 1 else 0
        return n * self.page_tokens / self.prompt_tokens

    def capacity_for(self, frac_of_ceiling: float) -> float:
        """Smallest capacity (tokens) reaching `frac_of_ceiling` x the ceiling."""
        hits = len(self._sorted)
        k = max(0, math.ceil(frac_of_ceiling * hits) - 1)
        return (self._sorted[k] + 1) * self.page_tokens if hits else 0.0

    def __post_init__(self):
        self._sorted = sorted(d for d in self.depth if d != math.inf)


def lru_curve(reqs: list[Request], page_tokens: int = 256,
              pages: list[list[int]] | None = None) -> Curve:
    """Exact LRU hit-rate curve over every capacity at once.

    Each request touches its pages last to first, so its first page is the
    most recently used. A page's stack distance is the number of distinct
    pages touched since its last use; it hits at capacity c pages if that
    distance is < c. A prefix hit needs every earlier page to hit, so a
    page's depth is the running max of the distances from the start of the
    request.
    """
    pages = pages if pages is not None else prefix_pages(reqs, page_tokens)
    total = sum(len(p) for p in pages)
    fw = _Fenwick(total)
    last: dict[int, int] = {}
    clock = 0
    depth: list = []
    for keys in pages:
        dist = [0] * len(keys)
        for j in range(len(keys) - 1, -1, -1):
            k = keys[j]
            prev = last.get(k)
            if prev is None:
                dist[j] = math.inf
            else:
                # distinct pages touched after `prev`: live markers in (prev, clock)
                dist[j] = fw.prefix(clock) - fw.prefix(prev + 1)
                fw.add(prev, -1)
            fw.add(clock, 1)
            last[k] = clock
            clock += 1
        run = 0
        for d in dist:
            run = max(run, d)
            depth.append(run)
    prompt = sum(r.tokens_in for r in reqs)
    hits_inf = sum(1 for d in depth if d != math.inf)
    return Curve(page_tokens, prompt, depth, hits_inf * page_tokens / prompt)


# --- Denning's working set on the same pages ---------------------------------

def denning_curve(reqs: list[Request], page_tokens: int = 256,
                  pages: list[list[int]] | None = None):
    """Hit rate at capacity C from reuse times and the mean working-set size.

    s(T) = (1/D) * sum over page uses of min(time to the page's next use, T)
    * page_tokens is the mean number of tokens touched in the last T seconds.
    A page use hits if the page was used within T* before, s(T*) = C, and
    every earlier page of the request did too. Returns hit(C).
    """
    pages = pages if pages is not None else prefix_pages(reqs, page_tokens)
    t_end = reqs[-1].t
    span = max(t_end - reqs[0].t, 1e-9)
    last: dict[int, float] = {}
    fwd: list[float] = []          # time to next use (or to the trace's end)
    idx: dict[int, int] = {}       # page -> index of its last use in fwd
    reuse: list[float] = []        # per page use: running max of the reuse time
    for r, keys in zip(reqs, pages):
        run = 0.0
        for k in keys:
            p = last.get(k)
            g = math.inf if p is None else r.t - p
            run = max(run, g)
            reuse.append(run)
            if k in idx:
                fwd[idx[k]] = r.t - p
            idx[k] = len(fwd)
            fwd.append(t_end - r.t)
            last[k] = r.t
    fwd.sort()
    cum = [0.0]
    for x in fwd:
        cum.append(cum[-1] + x)
    n = len(fwd)
    prompt = sum(r.tokens_in for r in reqs)
    hits_sorted = sorted(x for x in reuse if x != math.inf)

    def size(T: float) -> float:
        i = bisect.bisect_left(fwd, T)
        return (cum[i] + (n - i) * T) * page_tokens / span

    def hit(C: float) -> float:
        lo, hi = 0.0, span
        if size(hi) <= C:
            T = math.inf
        else:
            for _ in range(60):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if size(mid) < C else (lo, mid)
            T = lo
        return bisect.bisect_left(hits_sorted, T) * page_tokens / prompt

    return hit


# --- the planner's view: no block ids ----------------------------------------

@dataclass
class SessionStats:
    """What a planner can know about a workload without its block ids."""
    span_s: float
    prompt_tokens: list[int]    # per request
    reusable: list[int]         # per request: tokens shared with the session's previous request
    gap_prev: list[float]       # per request: seconds since that previous request (inf if none)
    gap_next: list[float]       # per request: seconds until the session's next request (inf if none)
    shared_any: list[int]       # per request: tokens shared with ANY earlier request (any session)


def session_stats(reqs: list[Request], page_tokens: int | None = None) -> SessionStats:
    """Per-request reuse of the session's previous prompt, by block prefix,
    and of anything earlier (a trie over block-id prefixes). With
    `page_tokens`, both are rounded down to whole pages, to compare with an
    exact replay at that page size."""
    def pg(x: int) -> int:
        return x if not page_tokens else (x // page_tokens) * page_tokens

    prev: dict[str, int] = {}
    seen: dict[tuple[int, int], int] = {}
    shared_any = [0] * len(reqs)
    for i, r in enumerate(reqs):
        node, k = -1, 0
        while k < len(r.blocks) and (node, r.blocks[k]) in seen:
            node = seen[(node, r.blocks[k])]
            k += 1
        shared_any[i] = pg(min(k * r.bs, r.tokens_in))
        for b in r.blocks[k:]:
            node = seen.setdefault((node, b), len(seen))
    reusable, gap_prev = [0] * len(reqs), [math.inf] * len(reqs)
    gap_next = [math.inf] * len(reqs)
    for i, r in enumerate(reqs):
        if r.session is None:
            continue
        j = prev.get(r.session)
        if j is not None:
            a, b = reqs[j].blocks, r.blocks
            k = 0
            while k < min(len(a), len(b)) and a[k] == b[k]:
                k += 1
            reusable[i] = pg(min(k * r.bs, r.tokens_in))
            gap_prev[i] = r.t - reqs[j].t
            gap_next[j] = r.t - reqs[j].t
        prev[r.session] = i
    return SessionStats(max(reqs[-1].t - reqs[0].t, 1e-9),
                        [r.tokens_in for r in reqs], reusable, gap_prev, gap_next,
                        shared_any)


def planner_curve(st: SessionStats, horizon_s: float | None = None,
                  ttl_s: float | None = None):
    """Characteristic-time LRU estimate from session statistics alone.

    Every request's prompt stays resident until its session's next request
    or until T, whichever is first: s(T) = (1/D) sum_r P_r min(g_next_r, T).
    A request reuses its session's previous prompt if that came less than T*
    ago, s(T*) = C. Tokens a request shares with other sessions beyond its
    own session's previous prompt (system prompts, common tool definitions)
    are counted as hits at any capacity: the few hot prefixes stay resident.
    `horizon_s` caps g_next for the session's last request (default: the
    trace span, which counts a finished session as live to the end).
    `ttl_s` adds an expiry (a provider prompt cache): T* is never above it.
    With every gap equal (sessions taking turns in a fixed order) the rule
    gives the fill rule's cliff: all hits while the sessions fit, none after.
    """
    H = horizon_s if horizon_s is not None else st.span_s
    items = sorted((min(g, H), p) for g, p in zip(st.gap_next, st.prompt_tokens))
    gs = [g for g, _ in items]
    cum_gp, cum_p = [0.0], [0.0]
    for g, p in items:
        cum_gp.append(cum_gp[-1] + g * p)
        cum_p.append(cum_p[-1] + p)
    total_p = cum_p[-1]
    prompt = sum(st.prompt_tokens)
    reuse = sorted((g, u) for g, u in zip(st.gap_prev, st.reusable) if u > 0)
    rg = [g for g, _ in reuse]
    hot = sum(max(a - u, 0) for a, u in zip(st.shared_any, st.reusable))
    cum_u = [0]
    for _, u in reuse:
        cum_u.append(cum_u[-1] + u)

    def size(T: float) -> float:
        i = bisect.bisect_left(gs, T)
        return (cum_gp[i] + (total_p - cum_p[i]) * T) / st.span_s

    def hit(C: float) -> float:
        lo, hi = 0.0, H
        if size(hi) <= C:
            T = math.inf
        else:
            for _ in range(60):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if size(mid) < C else (lo, mid)
            T = lo
        if ttl_s is not None:
            T = min(T, ttl_s)
        return (cum_u[bisect.bisect_left(rg, T)] + hot) / prompt

    return hit


def fill_users(st: SessionStats, capacity_tokens: float) -> tuple[float, float]:
    """The model's original rule, on the trace's own numbers: how many
    sessions a cache of `capacity_tokens` keeps warm (one mean prompt each),
    against how many sessions sit between two requests on average (Little's
    law over the gaps). Below `live`, the rule predicts the cyclic collapse.
    Returns (warm, live)."""
    live = sum(g for g in st.gap_next if g != math.inf) / st.span_s
    mean_prompt = sum(st.prompt_tokens) / len(st.prompt_tokens)
    return capacity_tokens / mean_prompt, live


def fill_hit(st: SessionStats, capacity_tokens: float) -> float:
    """The fill rule as a hit rate: the same-session ceiling (plus the hot
    shared prefixes) while the live sessions fit, the hot prefixes alone once
    they don't."""
    warm, live = fill_users(st, capacity_tokens)
    prompt = sum(st.prompt_tokens)
    hot = sum(max(a - u, 0) for a, u in zip(st.shared_any, st.reusable))
    return (hot + (sum(st.reusable) if warm >= live else 0)) / prompt


def proportional_hit(st: SessionStats, capacity_tokens: float) -> float:
    """The fill rule with random returns: when sessions come back after
    exponentially distributed gaps, LRU keeps a returning session warm with
    probability warm/live, not 0 (characteristic time with an exponential
    gap: s(T) = live * prompt * (1 - exp(-T/m)) = C gives
    P(gap < T*) = C / (live * prompt)). Same-session reuse is scaled by
    that, the hot shared prefixes always hit."""
    warm, live = fill_users(st, capacity_tokens)
    prompt = sum(st.prompt_tokens)
    hot = sum(max(a - u, 0) for a, u in zip(st.shared_any, st.reusable))
    return (hot + sum(st.reusable) * min(1.0, warm / live if live else 1.0)) / prompt


# --- `ws traces` ---------------------------------------------------------------

def cmd_traces(args) -> int:
    if args.traces_cmd == "tracelab":
        stats, measured = read_tracelab(args.path)
        ttls = [None if x == "inf" else float(x) for x in args.ttl.split(",")]
        print("provider  rounds   measured  " + "  ".join(
            f"ttl {'inf' if t is None else f'{t:g}s':>6}" for t in ttls))
        for prov in ("claude", "codex", "all"):
            st = stats[prov]
            row = [planner_curve(st, ttl_s=t)(math.inf) for t in ttls]
            print(f"{prov:<8} {len(st.prompt_tokens):>7}   {measured[prov]:.4f}    "
                  + "  ".join(f"{h:>10.4f}" for h in row))
        return 0
    reqs = READERS[args.traces_cmd](args.path)
    page = args.page or max(256, reqs[0].bs)
    pg = prefix_pages(reqs, page)
    curve = lru_curve(reqs, page, pg)
    den = denning_curve(reqs, page, pg)
    has_sessions = any(r.session is not None for r in reqs)
    st = session_stats(reqs, page)
    plan = planner_curve(st)
    caps = [float(x) * 1e6 for x in args.capacity.split(",")]
    print(f"{len(reqs)} requests, {sum(r.tokens_in for r in reqs):,} prompt tokens, "
          f"ceiling {curve.ceiling:.3f} at {page}-token pages")
    head = "capacity   exact  denning"
    if has_sessions:
        head += "  planner  fill  proportional"
    print(head)
    for C in caps:
        row = f"{C / 1e6:>7g}M  {curve.hit(C):.3f}  {den(C):.3f}"
        if has_sessions:
            row += f"    {plan(C):.3f}  {fill_hit(st, C):.3f}  {proportional_hit(st, C):.3f}"
        print(row)
    for f in (0.5, 0.9, 0.95):
        print(f"capacity for {f:.0%} of the ceiling: {curve.capacity_for(f) / 1e6:.2f}M tokens")
    return 0


def add_subparser(sub) -> None:
    p = sub.add_parser("traces", help="public prefix-cache traces: exact LRU "
                                      "against the planner and the fill rule")
    s = p.add_subparsers(dest="traces_cmd", required=True)
    for name in READERS:
        q = s.add_parser(name, help=f"replay a {name} trace (.jsonl or .jsonl.gz)")
        q.add_argument("path")
        q.add_argument("--capacity", default="1,3,10,30,100",
                       help="capacities in millions of tokens, comma-separated")
        q.add_argument("--page", type=int,
                       help="tokens per page, a whole number of blocks "
                            "(default: 256, or one block if larger)")
        q.set_defaults(fn=cmd_traces)
    q = s.add_parser("tracelab", help="TraceLab rounds: provider-measured cache "
                                      "hits against the planner with a TTL")
    q.add_argument("path")
    q.add_argument("--ttl", default="300,3600,inf",
                   help="cache TTLs in seconds, comma-separated (inf = none)")
    q.set_defaults(fn=cmd_traces)
