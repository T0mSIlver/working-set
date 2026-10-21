"""Replay AgentX traces against a finite prefix cache: predict the hit rates.

SemiAnalysis' AgentX replays Claude Code sessions (`agentx.py`) at a fixed
concurrency against a live server and reads the engine's prefix-cache
counters. This module replays the same sessions against a model of the
server's caches instead, so the hit rate becomes a function of HBM capacity,
DRAM capacity and concurrency that can be computed before renting the GPUs.

What is modelled, each taken from the AgentX harness (agentx-harness 1.0.12)
or the InferenceX recipe it names:

- Lanes. `concurrency` lanes each replay one root session and its sub-agents.
  A lane starts mid-session at a random instant t* (25-75% of the session's
  duration, the `agentx` scenario), primes the cache with each live stream's
  last request before t*, then replays from t*. A lane whose session tree has
  drained starts the next session from turn 0. Gaps between a request's end
  and the next start are the recorded ones, capped at `idle_gap_cap_s`.
- Routing. DP attention with vllm-router `consistent_hash` on the session id:
  every stream (root or sub-agent, each its own correlation id) is pinned to
  one rank, and each rank has its own cache. Ranks share nothing.
- HBM. Per rank, `hbm_tokens` of KV in `block` token pages. A running
  request pins its whole prompt plus output; when it finishes its prompt
  stays cached (the output is not: the next prompt carries the recorded
  reply, not the live one). Eviction is vLLM's: least recently freed first,
  a request's blocks tail-first, so an idle stream loses its prefix from the
  end. A request that cannot be fitted after evicting every idle stream waits.
- DRAM. vLLM's SimpleCPUOffloadConnector with eager (write-through) offload:
  every prompt block is also written to a per-rank CPU cache of
  `dram_tokens`, LRU. A prefix the HBM lost but the CPU kept is a DRAM hit.
- Service time. Prefill of the missed tokens on a per-rank FIFO at
  `prefill_tps`, DRAM hits loaded at `dram_load_tps`, decode at
  `decode_tps` per sequence (MTP folded in). These only move the hit rate by
  changing how long a context stays pinned; `sweep` reports how much.

Hit rates are token-weighted over the requests that start in the measured
window, the way the engine's `prompt_tokens_by_source` counters are.
"""
from __future__ import annotations

import hashlib
import heapq
import random
from dataclasses import dataclass, field

from .agentx import Stream, Trace


@dataclass(frozen=True)
class Server:
    """One serving deployment as the cache sees it."""
    ranks: int = 8                  # DP attention ranks (DEP8)
    hbm_tokens: float = 43e6        # KV capacity, all ranks (vLLM "GPU KV cache size")
    dram_tokens: float = 0.0        # CPU offload capacity, all ranks, in tokens
    block: int = 256                # vLLM --block-size
    prefill_tps: float = 15_000.0   # missed tokens prefilled per second, per rank
    decode_tps: float = 50.0        # output tokens per second per sequence
    dram_load_tps: float = 2e6      # DRAM-hit tokens loaded per second, per rank
    max_seqs: int | None = None     # --max-num-seqs per rank (None = unlimited)


@dataclass(frozen=True)
class Replay:
    """The AgentX client side (`agentx` scenario defaults)."""
    concurrency: int = 384
    duration_s: float = 3600.0
    start_min: float = 0.25
    start_max: float = 0.75
    idle_gap_cap_s: float = 300.0
    seed: int = 42


@dataclass
class Result:
    prompt_tokens: int = 0
    hbm_hit: int = 0
    dram_hit: int = 0
    ceiling_hit: int = 0            # infinite cache, same stream (what a cache could reuse)
    requests: int = 0
    waits: int = 0                  # requests that queued for KV space
    wait_s: float = 0.0
    peak_pinned: float = 0.0        # most KV pinned by running requests on one rank
    lanes_recycled: int = 0

    def rate(self, part: int) -> float:
        return part / self.prompt_tokens if self.prompt_tokens else float("nan")

    def to_dict(self) -> dict:
        return {"hbm_hit_rate": self.rate(self.hbm_hit),
                "dram_hit_rate": self.rate(self.dram_hit),
                "overall_hit_rate": self.rate(self.hbm_hit + self.dram_hit),
                "ceiling_hit_rate": self.rate(self.ceiling_hit),
                "requests": self.requests, "prompt_tokens": self.prompt_tokens,
                "requests_waited": self.waits, "mean_wait_s":
                self.wait_s / self.requests if self.requests else 0.0,
                "peak_pinned_tokens_per_rank": self.peak_pinned,
                "lanes_recycled": self.lanes_recycled}


def _rank_of(key: str, ranks: int) -> int:
    h = hashlib.blake2b(key.encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") % ranks


class _Cache:
    """One rank's tier: per-stream resident prefix, LRU over idle streams.

    Pinned (running) streams are not in the LRU. `used` counts every token
    held, pinned or idle.
    """

    def __init__(self, capacity: float):
        self.cap = capacity
        self.used = 0.0
        self.resident: dict[str, float] = {}     # stream -> cached prefix tokens
        self.stamp: dict[str, float] = {}         # stream -> last free time (idle only)
        self.heap: list[tuple[float, str]] = []   # (stamp, stream), lazy deletion
        self.pinned: dict[str, float] = {}        # stream -> tokens pinned while running

    def idle_tokens(self) -> float:
        return self.used - sum(self.pinned.values())

    def _evict(self, need: float, keep: str | None) -> float:
        """Free up to `need` tokens from idle streams, oldest first, tail first."""
        freed = 0.0
        skipped = []
        while freed < need and self.heap:
            st, s = heapq.heappop(self.heap)
            if self.stamp.get(s) != st:
                continue
            if s == keep:
                skipped.append((st, s))
                continue
            take = min(self.resident[s], need - freed)
            self.resident[s] -= take
            self.used -= take
            freed += take
            if self.resident[s] <= 0:
                del self.resident[s], self.stamp[s]
            else:
                skipped.append((st, s))   # still the oldest: stays at the front
                break
        for item in skipped:
            heapq.heappush(self.heap, item)
        return freed

    def lookup(self, s: str) -> float:
        return self.resident.get(s, 0.0) if s not in self.pinned else 0.0

    def can_fit(self, s: str, keep_tokens: float, total: float) -> bool:
        """Could `s` run with `keep_tokens` of its prefix reused and `total` held?"""
        own = self.resident.get(s, 0.0)
        other_idle = self.idle_tokens() - own
        return (self.cap - self.used) + other_idle + (own - keep_tokens) >= total - keep_tokens

    def pin(self, s: str, keep_tokens: float, total: float) -> None:
        """Run `s`: keep `keep_tokens` of its cached prefix, hold `total`."""
        own = self.resident.pop(s, 0.0)
        self.stamp.pop(s, None)
        self.used -= own - keep_tokens            # the orphaned tail is freed
        need = total - keep_tokens
        free = self.cap - self.used
        if need > free:
            self._evict(need - free, keep=s)
        self.used += need
        self.pinned[s] = total

    def unpin(self, s: str, keep: float, now: float) -> None:
        """`s` finished: keep `keep` tokens cached, free the rest."""
        total = self.pinned.pop(s)
        self.used -= total - keep
        if keep > 0:
            self.resident[s] = keep
            self.stamp[s] = now
            heapq.heappush(self.heap, (now, s))

    def store(self, s: str, tokens: float, now: float) -> None:
        """Write-through: the CPU tier keeps `tokens` of `s`'s prefix (LRU)."""
        own = self.resident.pop(s, 0.0)
        self.used -= own
        tokens = min(tokens, self.cap)
        free = self.cap - self.used
        if tokens > free:
            self._evict(tokens - free, keep=s)
        self.used += tokens
        self.resident[s] = tokens
        self.stamp[s] = now
        heapq.heappush(self.heap, (now, s))


@dataclass
class _StreamState:
    key: str                        # unique per lane instance
    stream: Stream
    rank: int
    idx: int                        # next request index
    done: bool = False
    running: bool = False
    waiting_spawn: bool = False     # sub-agent not yet spawned by its parent
    join_wait: bool = False         # root turn waiting for a sub-agent to end


@dataclass(order=True)
class _Ev:
    t: float
    seq: int
    kind: str = field(compare=False)
    data: object = field(compare=False, default=None)


def _floor(x: float, b: int) -> float:
    return (int(x) // b) * b


def _stream_end(s: Stream) -> float:
    last = s.requests[-1]
    return last.t + (last.api_time or 0.0)


def _duration(tr: Trace) -> float:
    return max(_stream_end(s) for s in tr.streams if s.requests)


def simulate(traces: list[Trace], server: Server, replay: Replay = Replay()) -> Result:
    rng = random.Random(replay.seed)
    order = list(range(len(traces)))
    rng.shuffle(order)
    next_trace = [0]
    b = server.block
    hbm = [_Cache(server.hbm_tokens / server.ranks) for _ in range(server.ranks)]
    dram = [_Cache(server.dram_tokens / server.ranks) for _ in range(server.ranks)]
    prefill_free_at = [0.0] * server.ranks
    running_count = [0] * server.ranks
    waiting: list[list[tuple[_StreamState, float]]] = [[] for _ in range(server.ranks)]
    res = Result()
    events: list[_Ev] = []
    seq = [0]
    T_END = replay.duration_s
    instance = [0]

    def push(t: float, kind: str, data=None) -> None:
        seq[0] += 1
        heapq.heappush(events, _Ev(t, seq[0], kind, data))

    # --- lanes ---------------------------------------------------------------
    @dataclass
    class Lane:
        n: int
        trace: Trace | None = None
        streams: list = field(default_factory=list)
        live: int = 0

    def gap(prev, nxt) -> float:
        g = nxt.t - (prev.t + (prev.api_time or 0.0))
        return min(max(g, 0.0), replay.idle_gap_cap_s)

    def start_lane(lane: Lane, now: float, mid_session: bool) -> None:
        tr = traces[order[next_trace[0] % len(order)]]
        next_trace[0] += 1
        instance[0] += 1
        lane.trace = tr
        lane.streams = []
        lane.live = 0
        tstar = (rng.uniform(replay.start_min, replay.start_max) * _duration(tr)
                 if mid_session else -1.0)
        for s in tr.streams:
            if not s.requests:
                continue
            key = f"{instance[0]}:{tr.id}:{s.key}"
            st = _StreamState(key, s, _rank_of(key, server.ranks), 0)
            lane.streams.append(st)
            # first request at or after t*
            i = next((k for k, r in enumerate(s.requests) if r.t >= tstar), None)
            if i is None:
                st.done = True
                continue
            st.idx = i
            lane.live += 1
            if mid_session and i > 0:
                # warmup: the last request before t* primed the cache
                prev = s.requests[i - 1]
                keep = _floor(prev.tokens_in, b)
                rk = st.rank
                hbm[rk].pin(st.key, 0.0, keep)
                hbm[rk].unpin(st.key, keep, now)
                if server.dram_tokens:
                    dram[rk].store(st.key, keep, now)
                push(now + gap(prev, s.requests[i]), "arrive", (lane, st))
            elif s.key == "root":
                first = s.requests[i]
                push(now + (0.0 if not mid_session else
                            min(max(first.t - tstar, 0.0), replay.idle_gap_cap_s)),
                     "arrive", (lane, st))
            else:
                st.waiting_spawn = True   # started by its parent turn
        # sub-agents whose spawn turn is already behind t*: start them now
        root = lane.streams[0] if lane.streams and lane.streams[0].stream.key == "root" else None
        for st in lane.streams:
            if st.waiting_spawn and (root is None or root.done or (
                    st.stream.parent_turn is not None and st.stream.parent_turn <= root.idx)):
                st.waiting_spawn = False
                first = st.stream.requests[st.idx]
                push(now + min(max(first.t - max(tstar, 0.0), 0.0), replay.idle_gap_cap_s),
                     "arrive", (lane, st))
        if lane.live == 0:
            push(now, "lane_done", lane)

    def try_admit(rk: int, now: float) -> None:
        q = waiting[rk]
        while q:
            st, t_arr = q[0]
            r = st.stream.requests[st.idx]
            if server.max_seqs is not None and running_count[rk] >= server.max_seqs:
                return
            lcp = _floor(min(r.lcp_prev * 64, max(r.tokens_in - 1, 0)), b)
            hit = min(lcp, _floor(hbm[rk].lookup(st.key), b))
            total = r.tokens_in + r.tokens_out
            if not hbm[rk].can_fit(st.key, hit, total):
                return
            q.pop(0)
            if now > t_arr and 0 <= now < T_END:
                res.waits += 1
                res.wait_s += now - t_arr
            run(st, r, rk, hit, lcp, now)

    def run(st: _StreamState, r, rk: int, hit: float, lcp: float, now: float) -> None:
        dhit = 0.0
        if server.dram_tokens:
            dhit = max(min(lcp, _floor(dram[rk].lookup(st.key), b)) - hit, 0.0)
        hbm[rk].pin(st.key, hit, r.tokens_in + r.tokens_out)
        if server.dram_tokens:
            dram[rk].store(st.key, _floor(r.tokens_in, b), now)
        res.peak_pinned = max(res.peak_pinned, sum(hbm[rk].pinned.values()))
        if 0 <= now < T_END:
            res.requests += 1
            res.prompt_tokens += r.tokens_in
            res.hbm_hit += int(hit)
            res.dram_hit += int(dhit)
            res.ceiling_hit += int(lcp)
        miss = r.tokens_in - hit - dhit
        start = max(now, prefill_free_at[rk])
        done_prefill = start + miss / server.prefill_tps + dhit / server.dram_load_tps
        prefill_free_at[rk] = done_prefill
        end = done_prefill + r.tokens_out / server.decode_tps
        st.running = True
        running_count[rk] += 1
        push(end, "finish", (st, rk, r))

    lanes = [Lane(n) for n in range(replay.concurrency)]
    for lane in lanes:
        start_lane(lane, 0.0, mid_session=True)

    lane_of: dict[str, Lane] = {}
    for lane in lanes:
        for st in lane.streams:
            lane_of[st.key] = lane

    while events:
        ev = heapq.heappop(events)
        now = ev.t
        if now >= T_END:
            break
        if ev.kind == "arrive":
            lane, st = ev.data
            lane_of[st.key] = lane
            waiting[st.rank].append((st, now))
            try_admit(st.rank, now)
        elif ev.kind == "finish":
            st, rk, r = ev.data
            lane = lane_of[st.key]
            st.running = False
            running_count[rk] -= 1
            hbm[rk].unpin(st.key, _floor(r.tokens_in, b), now)
            reqs = st.stream.requests
            st.idx += 1
            if st.stream.key == "root":
                # spawn sub-agents whose parent turn has now been sent
                for c in lane.streams:
                    if c.waiting_spawn and c.stream.parent_turn is not None \
                            and c.stream.parent_turn <= st.idx:
                        c.waiting_spawn = False
                        first = c.stream.requests[c.idx]
                        push(now + min(max(first.t - (r.t + (r.api_time or 0.0)), 0.0),
                                       replay.idle_gap_cap_s), "arrive", (lane, c))
            if st.idx >= len(reqs):
                st.done = True
                lane.live -= 1
                # a parent blocked on this child may go now
                for c in lane.streams:
                    if c.join_wait and not _blocked_on_child(lane, c.stream.requests[c.idx].t):
                        c.join_wait = False
                        push(now, "arrive", (lane, c))
                if lane.live <= 0:
                    push(now, "lane_done", lane)
                try_admit(rk, now)
                continue
            nxt = reqs[st.idx]
            if st.stream.key == "root" and _blocked_on_child(lane, nxt.t):
                st.join_wait = True
            else:
                push(now + gap(r, nxt), "arrive", (lane, st))
            try_admit(rk, now)
        elif ev.kind == "lane_done":
            lane = ev.data
            if lane.live > 0:
                continue
            res.lanes_recycled += 1
            start_lane(lane, now, mid_session=False)
            for st in lane.streams:
                lane_of[st.key] = lane
    return res


def _blocked_on_child(lane, t_next: float) -> bool:
    """A root turn recorded after a sub-agent ended waits for that sub-agent."""
    for c in lane.streams:
        if c.stream.key == "root" or c.done:
            continue
        if c.waiting_spawn:
            continue
        if _stream_end(c.stream) <= t_next:
            return True
    return False


# --- DeepSeek V4 Pro on the two published AgentX points ---------------------
# research/agentx_kv_hits.md has every constant's source.

DSV4_PRO = {
    "kv_bpt": 4_939.5,               # 30 x 576/4 + 31 x 576/128 + 30 x 64/4 (fp8 latent, fp4 indexer)
    "w_per_gpu_dep8": 133.6e9,       # 29.2 GB non-expert on every rank + 835.5 GB experts / 8
}

# Measured by SemiAnalysis (blog of 2026-08-25). `hbm_tokens` is their
# "HBM KV cache working set", taken as the capacity input (see the log's
# capacity puzzle). DRAM: CPU bytes per rank over GPU KV bytes per rank
# (VRAM x utilization - weights - a 10 GB reserve), times the HBM tokens:
# the offload stores the same pages. B200's node DRAM is assumed (DGX B200,
# 2 TB). Service rates are guesses, so `sweep` varies them.
POINTS = {
    "b300-384": {
        "server": Server(ranks=8, hbm_tokens=43e6,
                         dram_tokens=43e6 * 356e9 / (288.4e9 * 0.92 - 133.6e9 - 10e9),
                         prefill_tps=20_000.0, decode_tps=60.0, max_seqs=96),
        "replay": Replay(concurrency=384),
        "measured": {"hbm_hit_rate": 0.91, "dram_hit_rate": 0.0136},
    },
    "b200-196": {
        "server": Server(ranks=8, hbm_tokens=22e6,
                         dram_tokens=22e6 * 237.5e9 / (192.3e9 * 0.90 - 133.6e9 - 10e9),
                         prefill_tps=12_000.0, decode_tps=45.0, max_seqs=49),
        "replay": Replay(concurrency=196),
        "measured": {"hbm_hit_rate": 0.73, "dram_hit_rate": 0.20},
    },
}


def _fmt_row(label: str, d: dict) -> str:
    return (f"{label:<34} HBM {d['hbm_hit_rate']:6.1%}  DRAM {d['dram_hit_rate']:6.1%}  "
            f"ceiling {d['ceiling_hit_rate']:6.1%}  requests {d['requests']:>6,}  "
            f"waited {d['requests_waited']:>5,}")


def cmd_agentx(args) -> int:
    import json
    from dataclasses import replace

    from . import agentx as A
    if args.agentx_cmd == "compact":
        print(json.dumps(A.compact_jsonl(args.src, args.out)))
        return 0
    traces = A.load(args.traces)
    out = {}
    for name in args.point or list(POINTS):
        pt = POINTS[name]
        srv, rp = pt["server"], replace(pt["replay"], seed=args.seed)
        if args.duration:
            rp = replace(rp, duration_s=args.duration)
        rows = {"central": simulate(traces, srv, rp).to_dict()}
        if args.sweep:
            for f in (0.5, 2.0):
                rows[f"service x{f:g}"] = simulate(traces, replace(
                    srv, prefill_tps=srv.prefill_tps * f, decode_tps=srv.decode_tps * f), rp).to_dict()
            for f in (0.75, 1.25):
                rows[f"hbm x{f:g}"] = simulate(traces, replace(
                    srv, hbm_tokens=srv.hbm_tokens * f, dram_tokens=srv.dram_tokens), rp).to_dict()
            rows["dram 1.5x hbm"] = simulate(traces, replace(
                srv, dram_tokens=srv.hbm_tokens * 1.5), rp).to_dict()
            rows["no dram"] = simulate(traces, replace(srv, dram_tokens=0.0), rp).to_dict()
            for s in (1, 2, 3):
                rows[f"seed {args.seed + s}"] = simulate(
                    traces, srv, replace(rp, seed=args.seed + s)).to_dict()
        out[name] = {"measured": pt["measured"], "predicted": rows}
        if not args.json:
            m = pt["measured"]
            print(f"{name}: measured HBM {m['hbm_hit_rate']:.1%}, DRAM {m['dram_hit_rate']:.1%}")
            for label, d in rows.items():
                print("  " + _fmt_row(label, d))
    if args.json:
        print(json.dumps(out, indent=2))
    return 0


def add_subparser(sub) -> None:
    p = sub.add_parser("agentx", help="SemiAnalysis AgentX traces: compact them, "
                                      "predict the published hit rates")
    s = p.add_subparsers(dest="agentx_cmd", required=True)
    c = s.add_parser("compact", help="traces.jsonl -> the compact prefix structure")
    c.add_argument("src")
    c.add_argument("out")
    c.set_defaults(fn=cmd_agentx)
    q = s.add_parser("predict", help="replay the traces against each published point")
    q.add_argument("traces", help="compact file from `ws agentx compact` (or a raw .jsonl)")
    q.add_argument("--point", action="append", choices=list(POINTS))
    q.add_argument("--seed", type=int, default=42)
    q.add_argument("--duration", type=float, help="seconds of replay (default 3600)")
    q.add_argument("--sweep", action="store_true",
                   help="also vary service rates, capacity, DRAM and the seed")
    q.add_argument("--json", action="store_true")
    q.set_defaults(fn=cmd_agentx)
