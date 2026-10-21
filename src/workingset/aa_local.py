"""Artificial Analysis' AgentPerf-Local replay, run by several agents at once.

`agentperf-local` (github.com/ArtificialAnalysis/aa-agentperf-local,
Apache-2.0) replays eight recorded agent tasks, 168 model turns, against one
OpenAI-compatible server, strictly one request at a time. Its article says
multi-agent coverage is planned. This module predicts what happens when N
copies of that replay share one local server: N `agentperf-local run`
processes started together against a llama-server with `--parallel N`.

The workload comes straight from the bundled traces: per turn the recorded
prompt and output token counts (the default `exact` policy generates exactly
the recorded output length), and the tokens it shares with the request
before it in the same lane. Each process prefixes every conversation with
its own random cache namespace, so lanes share nothing; within a lane a turn
reuses its predecessor's prompt up to the first character that differs (the
cached reply is the server's own, never the recorded one, so it never hits).

The server is a llama.cpp-style engine stepped one batch at a time:

- A batch carries one token for every decoding sequence, then prompt tokens
  up to `n_batch`. Its time is `fixed + max(memory, compute)`: memory reads
  the weights once per ubatch plus every running sequence's KV and recurrent
  state; compute prices each token at the single-stream prefill rate,
  stretched by its attention context.
- Prefix cache: each lane's slot keeps its last prompt. A unified pool of
  `pool_tokens` holds every slot's cache; when a request does not fit, idle
  slots are cleared least recently used first, and the cleared lane
  re-prefills its next prompt from zero (or from the host-RAM prompt cache,
  when `ram_tokens` > 0).
- Hybrid (recurrent) models can only resume at a saved checkpoint, so with
  `restore="checkpoint"` a hit ends at the latest earlier prompt end that
  lies inside the common prefix, not at the common prefix itself.

Two speeds come from outside the model: single-stream decode and prefill on
this box. `prior` takes them from public llama.cpp numbers; `--anchor`
reads them from a measured N=1 run, and everything at N>1 is then predicted.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

SOURCE = "ArtificialAnalysis/aa-agentperf-local agentperf-default-v1"
MiB = 1024 ** 2
GiB = 1024 ** 3


@dataclass(frozen=True)
class Turn:
    task: str
    prompt: int          # recorded prompt tokens
    out: int             # target output tokens (exact policy)
    tool_s: float        # recorded tool delay after the turn
    lcp: int             # tokens shared with the lane's previous request


def _ser(messages) -> str:
    return json.dumps(messages, ensure_ascii=False, separators=(",", ":"))


def _lcp_chars(a: str, b: str) -> int:
    n = min(len(a), len(b))
    if a[:n] == b[:n]:
        return n
    lo, hi = 0, n
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid
    return lo


def load(replay_dir: str | Path, manifest: str = "manifest-x86_64.json",
         order: list[str] | None = None) -> list[Turn]:
    """One lane's turns, in manifest order (or `order`, a list of task ids).

    `lcp` is the previous request's recorded prompt tokens scaled by the
    share of its serialized messages the current request repeats. Across a
    task boundary that share is the common system prompt and tool list."""
    root = Path(replay_dir)
    man = json.loads((root / manifest).read_text())
    tasks = {t["task_id"]: t for t in man["tasks"]}
    ids = order or [t["task_id"] for t in man["tasks"]]
    turns: list[Turn] = []
    prev_ser, prev_prompt = None, 0
    for tid in ids:
        for line in (root / tasks[tid]["trace"]).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            s = _ser({"tools": r.get("tools", []), "messages": r["messages"]})
            lcp = 0
            if prev_ser is not None:
                lcp = int(prev_prompt * _lcp_chars(prev_ser, s) / max(1, len(prev_ser)))
            turns.append(Turn(tid, int(r["recorded_prompt_tokens"]),
                              int(r.get("target_output_tokens") or r["recorded_completion_tokens"]),
                              float(r.get("simulated_tool_delay_ms_after", 0.0)) / 1000, lcp))
            prev_ser, prev_prompt = s, int(r["recorded_prompt_tokens"])
    return turns


def task_order(replay_dir: str | Path, manifest: str = "manifest-x86_64.json") -> list[str]:
    man = json.loads((Path(replay_dir) / manifest).read_text())
    return [t["task_id"] for t in man["tasks"]]


def rotated(order: list[str], lane: int, lanes: int) -> list[str]:
    """Lane k starts k/lanes of the way through the task list, so N agents
    are doing different work at any moment instead of marching in step."""
    k = (lane * len(order)) // lanes
    return order[k:] + order[:k]


def stats(turns: list[Turn]) -> dict:
    prompt = sum(t.prompt for t in turns)
    reuse = sum(min(t.lcp, t.prompt) for t in turns)
    return {"turns": len(turns), "tasks": len({t.task for t in turns}),
            "prompt_tokens": prompt, "output_tokens": sum(t.out for t in turns),
            "max_prompt": max(t.prompt for t in turns),
            "reusable_prompt_tokens": reuse, "best_hit_rate": reuse / prompt,
            "tool_delay_s": sum(t.tool_s for t in turns)}


@dataclass(frozen=True)
class Box:
    """One model on one GPU under llama-server, as the scheduler sees it."""
    name: str
    weights_bytes: float      # read once per ubatch
    kv_bpt: float             # KV bytes per context token, read every decode step
    state_bytes: float        # recurrent state per sequence, read+written per step
    bw: float                 # memory bandwidth, bytes/s
    decode_tps1: float        # single-stream decode at short context (anchor)
    prefill_tps1: float       # single-stream prefill at short context (anchor)
    attn_ratio: float         # extra compute per token per context token, relative
    pool_tokens: int          # llama-server -c (unified KV)
    ram_tokens: int = 0       # --cache-ram, in tokens of this model's KV
    n_batch: int = 2048
    n_ubatch: int = 512
    fixed_s: float = 0.0      # per-batch launch overhead
    restore: str = "lcp"      # "lcp" or "checkpoint" (hybrid models)
    checkpoints: int = 8      # llama-server --ctx-checkpoints
    combine: str = "max"      # roofline "max", or "sum" (no overlap)

    def bw_eff(self) -> float:
        """Bandwidth efficiency implied by the single-stream decode anchor."""
        t = 1.0 / self.decode_tps1 - self.fixed_s
        return (self.weights_bytes + self.state_bytes + 2048 * self.kv_bpt) / (t * self.bw)

    def tok_compute_s(self, ctx: float) -> float:
        return (1.0 / self.prefill_tps1) * (1.0 + self.attn_ratio * ctx)


# Qwen3.8-27B (same config as Qwen3.6-27B: 64 layers, 16 full-attention with
# 4 KV heads x 256, 48 Gated DeltaNet), Q4_K_M GGUF, f16 KV, on one RTX 3090.
# weights: 16.5e9 B is the Q4_K_M file size class for 27B (to be replaced by
#   the measured file). kv: 16 x 4 x 256 x 2 (K,V) x 2 B = 64 KiB/token.
# state: 48 DN layers x 48 v-heads x 128 x 128 x 4 B (f32 in llama.cpp) + conv.
# bw: 936 GB/s (RTX 3090 spec). attn_ratio: 4 x 16 layers x 24 x 256 / (2 x 24.5e9).
# decode/prefill priors: llama.cpp RTX 3090 tg128/pp512 for ~27-32B dense Q4_K_M
#   are ~35 and ~1,100 tok/s; UNMEASURED here, replaced by --anchor.
BOXES = {
    "qwen38-27b-q4km-3090": Box(
        name="Qwen3.8-27B Q4_K_M, f16 KV, RTX 3090",
        weights_bytes=16.5e9, kv_bpt=16 * 4 * 256 * 2 * 2,
        state_bytes=48 * 48 * 128 * 128 * 4 + 48 * 10_240 * 4 * 4,
        bw=936e9, decode_tps1=35.0, prefill_tps1=1_100.0,
        attn_ratio=4 * 16 * 24 * 256 / (2 * 24.5e9),
        pool_tokens=131_072, restore="checkpoint"),
}


@dataclass
class _Lane:
    turns: list[Turn]
    i: int = 0
    phase: str = "idle"       # idle | prefill | decode | done
    ready_at: float = 0.0
    left: int = 0             # prompt tokens left to prefill, or output tokens left
    ctx: int = 0              # tokens in the slot right now
    cached: int = 0           # tokens kept from the last prompt (0 if cleared)
    marks: list[int] = field(default_factory=list)   # checkpoint positions
    last_used: float = 0.0
    t_first: float = 0.0
    rows: list[dict] = field(default_factory=list)


def _hit(lane: _Lane, t: Turn, box: Box) -> int:
    lcp = min(t.lcp, lane.cached, t.prompt - 1)
    if lcp <= 0:
        return 0
    if box.restore == "lcp":
        return lcp
    ok = [m for m in lane.marks if m <= lcp]
    return max(ok) if ok else 0


def simulate(box: Box, lanes: list[list[Turn]], max_time: float = 36_000.0,
             tool_delays: bool = False) -> dict:
    L = [_Lane(turns=ts) for ts in lanes]
    now = 0.0
    ram: dict[int, int] = {}          # lane -> tokens kept in host RAM
    evictions = overflows = 0
    peak = 0

    def used() -> int:
        return sum(l.ctx if l.phase in ("prefill", "decode") else l.cached for l in L)

    def make_room(need: int, me: int) -> None:
        nonlocal evictions
        while used() + need > box.pool_tokens:
            idle = [(l.last_used, k) for k, l in enumerate(L)
                    if k != me and l.phase in ("idle", "done") and l.cached > 0]
            if not idle:
                return
            _, k = min(idle)
            if box.ram_tokens:
                ram[k] = L[k].cached
            L[k].cached, L[k].marks = 0, []
            evictions += 1

    while now < max_time:
        # admit lanes whose next request is due
        for k, l in enumerate(L):
            if l.phase == "idle" and l.ready_at <= now:
                if l.i >= len(l.turns):
                    l.phase = "done"
                    continue
                t = l.turns[l.i]
                if l.cached == 0 and k in ram and box.ram_tokens:
                    l.cached = ram.pop(k)
                    l.marks = [l.cached]
                hit = _hit(l, t, box)
                make_room(t.prompt + t.out - l.cached, k)
                if used() + t.prompt + t.out - l.cached > box.pool_tokens:
                    overflows += 1
                l.ctx, l.left, l.phase = hit, t.prompt - hit, "prefill"
                l.rows.append({"task": t.task, "prompt": t.prompt, "out": t.out,
                               "cached": hit, "t0": now})
        active = [l for l in L if l.phase in ("prefill", "decode")]
        if not active:
            pending = [l.ready_at for l in L if l.phase == "idle"]
            if not pending:
                break
            now = max(now, min(pending))
            continue
        peak = max(peak, used())
        # one batch: decode tokens first, then prompt tokens up to n_batch
        dec = [l for l in active if l.phase == "decode"]
        budget = box.n_batch - len(dec)
        fills: list[tuple[_Lane, int]] = []
        for l in active:
            if l.phase == "prefill" and budget > 0:
                n = min(l.left, budget)
                fills.append((l, n))
                budget -= n
        ntok = len(dec) + sum(n for _, n in fills)
        mem = (math.ceil(ntok / box.n_ubatch) * box.weights_bytes
               + sum(l.ctx for l in active) * box.kv_bpt
               + len(active) * box.state_bytes) / (box.bw * box.bw_eff())
        comp = sum(box.tok_compute_s(l.ctx) for l in dec) + \
            sum(n * box.tok_compute_s(l.ctx + n / 2) for l, n in fills)
        dt = box.fixed_s + (max(mem, comp) if box.combine == "max" else mem + comp)
        now += dt
        for l in dec:
            l.ctx += 1
            l.left -= 1
            if l.left <= 0:
                t = l.turns[l.i]
                r = l.rows[-1]
                r.update(t1=now, ttft=r["t_first"] - r["t0"],
                         gen_tps=(t.out - 1) / max(1e-9, now - r["t_first"]))
                # the slot keeps the prompt; the generated tail never matches
                l.cached = t.prompt
                if box.restore == "checkpoint":
                    l.marks = (l.marks + [t.prompt])[-box.checkpoints:]
                l.ctx, l.phase, l.last_used = 0, "idle", now
                l.ready_at = now + (t.tool_s if tool_delays else 0.0)
                l.i += 1
        for l, n in fills:
            l.ctx += n
            l.left -= n
            if l.left <= 0:
                t = l.turns[l.i]
                l.phase, l.left = "decode", t.out
                l.rows[-1]["t_first"] = now
                # the first output token comes out of the prompt's last batch
                l.ctx += 1
                l.left -= 1
                if l.left <= 0:
                    l.left = 1
    rows = [r for l in L for r in l.rows if "t1" in r]
    return summarize(rows, [l.rows for l in L], extra={"evictions": evictions,
                                                       "overflows": overflows,
                                                       "peak_kv_tokens": peak})


def summarize(rows: list[dict], per_lane: list[list[dict]], extra: dict | None = None) -> dict:
    """The table's columns, computed the same way for simulated and measured runs."""
    def q(xs, p):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))] if xs else float("nan")
    walls = [max(r["t1"] for r in lr) - min(r["t0"] for r in lr) for lr in per_lane if lr]
    span = max(r["t1"] for r in rows) - min(r["t0"] for r in rows)
    out_tok = sum(r["out"] for r in rows)
    prompt = sum(r["prompt"] for r in rows)
    d = {"lanes": len(per_lane), "turns": len(rows),
         "wall_s": span, "lane_wall_s_mean": sum(walls) / len(walls),
         "agg_out_tps": out_tok / span,
         "gen_tps_mean": sum(r["gen_tps"] for r in rows) / len(rows),
         "ttft_p50_s": q([r["ttft"] for r in rows], 0.5),
         "ttft_p95_s": q([r["ttft"] for r in rows], 0.95),
         "hit_rate": sum(r["cached"] for r in rows) / prompt}
    d.update(extra or {})
    return d


def lanes_for(replay_dir, n: int, rotate: bool = True,
              manifest: str = "manifest-x86_64.json") -> list[list[Turn]]:
    order = task_order(replay_dir, manifest)
    return [load(replay_dir, manifest, rotated(order, k, n) if rotate else order)
            for k in range(n)]


def read_measured(run_dirs: list[str | Path]) -> dict:
    """Same columns from agentperf-local result directories, one per lane.

    `turns.jsonl` has no wall-clock stamps, only per-turn durations, so each
    lane's timeline is rebuilt as its turns back to back (plus the replayed
    tool delay) from a common start: the driver starts every lane at once."""
    per_lane = []
    for d in run_dirs:
        lr, t = [], 0.0
        for line in (Path(d) / "turns.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            j = json.loads(line)
            tm, tk = j.get("timing") or {}, j.get("tokens") or {}
            if not j.get("success") or tm.get("e2e_latency_ms") is None:
                continue
            e2e = tm["e2e_latency_ms"] / 1000
            gen = (tm.get("generation_ms") or 0) / 1000
            out = tk.get("server_output_tokens") or tk.get("local_output_tokens") or 0
            lr.append({"task": j.get("task_id"),
                       "prompt": tk.get("server_prompt_tokens") or tk.get("recorded_prompt_tokens") or 0,
                       "out": out, "cached": tk.get("server_cached_prompt_tokens") or 0,
                       "t0": t, "t1": t + e2e,
                       "ttft": (tm.get("time_to_first_token_ms") or 0) / 1000,
                       "gen_tps": (out - 1) / gen if gen > 0 else 0.0})
            t += e2e + (j.get("replayed_tool_delay_ms") or 0) / 1000
        per_lane.append(lr)
    return summarize([r for lr in per_lane for r in lr], per_lane)


def anchor_from(box: Box, run_dir: str | Path) -> Box:
    """Box with decode and prefill speeds read from a measured N=1 run:
    decode from short-context turns, prefill from cold turns' TTFT."""
    rows = [json.loads(x) for x in (Path(run_dir) / "turns.jsonl").read_text().splitlines() if x.strip()]
    dec, pre = [], []
    for j in rows:
        tm, tk = j.get("timing") or {}, j.get("tokens") or {}
        out = tk.get("server_output_tokens") or 0
        if tm.get("generation_ms") and out > 32 and (tk.get("server_prompt_tokens") or 0) < 12_000:
            dec.append((out - 1) / (tm["generation_ms"] / 1000))
        unc = tk.get("server_uncached_prompt_tokens")
        if unc and unc > 2_000 and tm.get("time_to_first_token_ms"):
            pre.append(unc / (tm["time_to_first_token_ms"] / 1000))
    if not dec or not pre:
        raise ValueError(f"{run_dir}: no turns to anchor on (decode {len(dec)}, prefill {len(pre)})")
    dec.sort(), pre.sort()
    return replace(box, decode_tps1=dec[len(dec) // 2], prefill_tps1=pre[len(pre) // 2])


COLS = ("wall_s", "agg_out_tps", "gen_tps_mean", "ttft_p50_s", "ttft_p95_s", "hit_rate")


def _fmt(d: dict) -> str:
    return (f"wall {d['wall_s']:7.0f} s  agg {d['agg_out_tps']:6.1f} tok/s  "
            f"per-stream {d['gen_tps_mean']:5.1f} tok/s  TTFT p50 {d['ttft_p50_s']:5.2f} s "
            f"p95 {d['ttft_p95_s']:6.2f} s  hits {d['hit_rate']:6.1%}"
            + (f"  peak KV {d['peak_kv_tokens']:,}  evictions {d['evictions']}  overflows {d['overflows']}"
               if "peak_kv_tokens" in d else ""))


def cmd_aa(args) -> int:
    if args.aa_cmd == "stats":
        print(json.dumps(stats(load(args.replay)), indent=2))
        return 0
    if args.aa_cmd == "measured":
        root = Path(args.runs)
        out = {}
        for nd in sorted(root.glob("n*"), key=lambda p: int(p.name[1:])):
            lanes = sorted(nd.glob("lane*/"), key=lambda p: int(p.name[4:]))
            lanes = [x for x in lanes if (x / "turns.jsonl").exists()]
            if lanes:
                out[int(nd.name[1:])] = d = read_measured(lanes)
                if not args.json:
                    print(f"N={nd.name[1:]}  {_fmt(d)}")
        if args.json:
            print(json.dumps(out, indent=2))
        return 0
    box = BOXES[args.box]
    over = {k: v for k, v in (("pool_tokens", args.pool), ("weights_bytes", args.weights_bytes),
                              ("restore", args.restore), ("ram_tokens", args.ram_tokens),
                              ("combine", args.combine)) if v is not None}
    box = replace(box, **over)
    if args.anchor:
        box = anchor_from(box, args.anchor)
    out = {"box": asdict(box), "bw_eff": box.bw_eff(), "source": SOURCE, "points": {}}
    for n in (int(x) for x in args.conc.split(",")):
        d = simulate(box, lanes_for(args.replay, n, rotate=not args.no_rotate),
                     tool_delays=args.tool_delays)
        out["points"][n] = d
        if not args.json:
            print(f"N={n}  {_fmt(d)}")
    if args.json:
        print(json.dumps(out, indent=2))
    return 0


def add_subparser(sub) -> None:
    p = sub.add_parser("aa", help="AA AgentPerf-Local traces at N concurrent agents")
    s = p.add_subparsers(dest="aa_cmd", required=True)
    st = s.add_parser("stats", help="the replay's token totals and best possible hit rate")
    st.add_argument("replay", help="agentperf_local/data/replays/agentperf-default-v1")
    st.set_defaults(fn=cmd_aa)
    q = s.add_parser("predict", help="predict N lanes sharing one llama-server")
    q.add_argument("replay")
    q.add_argument("--box", default="qwen38-27b-q4km-3090", choices=list(BOXES))
    q.add_argument("--conc", default="1,2,4,8")
    q.add_argument("--anchor", help="measured N=1 result dir: take its decode and prefill speeds")
    q.add_argument("--pool", type=int, help="llama-server -c, tokens")
    q.add_argument("--weights-bytes", type=float)
    q.add_argument("--ram-tokens", type=int, help="host-RAM prompt cache, in tokens")
    q.add_argument("--restore", choices=("lcp", "checkpoint"))
    q.add_argument("--combine", choices=("max", "sum"))
    q.add_argument("--no-rotate", action="store_true", help="every lane runs tasks in manifest order")
    q.add_argument("--tool-delays", action="store_true", help="wait the recorded tool delays")
    q.add_argument("--json", action="store_true")
    q.set_defaults(fn=cmd_aa)
    m = s.add_parser("measured", help="the same columns from runs/n<N>/lane<k>/ result dirs")
    m.add_argument("runs")
    m.add_argument("--json", action="store_true")
    m.set_defaults(fn=cmd_aa)
