#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx>=0.27",
#     "numpy>=1.26",   # only for --dry-run pricing via scenario_model
# ]
# ///
"""Active decode plateaus: the step-byte leverage a shared instance won't give.

decode_mbu.py fits t_pass = t0 + bytes/(eta x bw). Both terms are identified
only if step bytes VARY: at one operating point every (t0, eta) on a line
through the data fits equally well. Passive production traffic sits at n ~
1-10 and, on a dense model whose weights are 94% of the step, that is a
byte range of about 1.2x. This script buys the range, then gets out.

MOVING STEP BYTES CHEAPLY. On the 27B the weight term is constant, so the
only lever is sum_L = the total context the batch attends over:

    step_bytes = w_decode  +  kv_bpt x sum_L  +  2 n x state
                 30.9 GB      32 KiB/token      75 MiB/seq

k sequences x L tokens each is k x L of read, but only L of STORAGE and ONE
prefill if the k prompts share a byte-identical prefix -- vLLM dedups the
blocks, while every sequence still gathers all of them each step. So arm A
reaches 100+ GB steps for the price of one 64k prefill and 64k of pool.

    arm A  shared 64k prefix, k = 1,2,4,8,16(,32)      the leverage engine
    arm B  UNIQUE prompts,    k = 4 x 8k/32k/128k      kv_bpt slope, clean
    arm C  k=8 shared vs k=8 unique at equal sum_L     cascade-attention test

Arm C is the control arm A needs: if vLLM is serving the shared prefix with
cascade attention it reads those blocks ONCE for the whole batch, arm A's
sum_L is a fiction, and -- more importantly -- so is the study's per-sequence
KV term at large n. Equal step times across C is that finding.

WHAT THIS COSTS THE OTHER TENANTS. Decode bandwidth is shared: while a
plateau runs, everyone else's inter-token latency rises by roughly
step_bytes(with you) / step_bytes(without you). --dry-run prints that factor
per rung -- read it before running arm A's top rungs. Bounded and abortable:
  * plateaus last --hold seconds (default 40), one at a time;
  * a watchdog polls /metrics and aborts mid-plateau if anything queues
    (--max-waiting) or the KV pool passes --max-kv;
  * arm A costs one prefill and --prefix-tokens of pool, total;
  * Ctrl-C cancels every stream and still writes the arms file.

  uv run scripts/decode_probe.py --dry-run
  uv run scripts/decode_probe.py --url https://host/v1 --api-key $KEY \\
      --metrics http://backend:8000/metrics --arm A --arm C
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import string
import sys
import time
from pathlib import Path

import httpx

CHARS_PER_TOKEN = 3.6      # random lowercase text; refined from usage at runtime


def rungs_for(args):
    """(arm, k, prompt_tokens, shared) plateaus, in run order."""
    P = args.prefix_tokens
    plan = {
        "A": [("A", k, P, True) for k in (1, 2, 4, 8, 16, 32)],
        "B": [("B", 4, c, False) for c in (8_000, 32_000, 128_000)],
        "C": [("C", 8, P, True), ("C", 8, P, False)],
    }
    return [r for a in args.arm for r in plan[a] if r[1] <= args.max_k]


def filler(tokens, rnd):
    n = max(16, int(tokens * CHARS_PER_TOKEN))
    return "".join(rnd.choice(string.ascii_lowercase + "  ") for _ in range(n))


async def stream_one(client, args, prompt, stop, stats):
    """One long decode. Retries once without min_tokens/ignore_eos: a proxy in
    front of vLLM often strips or rejects them, and a plateau that ends when
    the model decides to stop is not a plateau."""
    body = {"model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": args.out_tokens, "temperature": 0.0, "stream": True,
            "min_tokens": args.out_tokens, "ignore_eos": True,
            "stream_options": {"include_usage": True}}
    for attempt in (0, 1):
        try:
            async with client.stream("POST", "/chat/completions", json=body,
                                     timeout=httpx.Timeout(1800.0, connect=30.0)) as r:
                if r.status_code != 200:
                    txt = (await r.aread())[:200].decode(errors="replace")
                    if attempt == 0:
                        body.pop("min_tokens", None)
                        body.pop("ignore_eos", None)
                        continue
                    stats.setdefault("errors", []).append(f"HTTP {r.status_code}: {txt}")
                    return
                async for line in r.aiter_lines():
                    if stop.is_set():
                        return
                    if '"prompt_tokens"' in line:
                        m = re.search(r'"prompt_tokens"\s*:\s*(\d+)', line)
                        if m:
                            stats.setdefault("ptok", []).append(int(m.group(1)))
                return
        except (httpx.ReadError, httpx.RemoteProtocolError, httpx.ReadTimeout):
            return
        except asyncio.CancelledError:
            raise


async def read_metrics(client, url, key):
    if not url:
        return {}
    hdr = {"Authorization": f"Bearer {key}"} if key else {}
    try:
        r = await client.get(url, headers=hdr, timeout=5.0)
    except Exception:
        return {}
    out = {}
    for line in r.text.splitlines():
        for name in ("vllm:num_requests_waiting", "vllm:num_requests_running",
                     "vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"):
            if line.startswith(name):
                try:
                    out[name] = out.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
                except (ValueError, IndexError):
                    pass
    return out


async def watchdog(client, args, stop, events, armed_in=0.0):
    """Politeness: three consecutive bad samples (~3 s) end the plateau. One
    bad sample is a scheduling blip; three is somebody waiting on us.

    Not armed until the ramp is over. Launching k streams at once queues our
    OWN requests by construction -- at k=8 that reads as "somebody is
    waiting" and aborts the plateau before it starts, which is what killed
    arm A's k=8 rung and both arm C rungs on the first production run. After
    the ramp, a queue means a co-tenant and the abort is real."""
    if not args.metrics:
        return
    if armed_in:
        await asyncio.sleep(armed_in)
    strikes = 0
    while not stop.is_set():
        m = await read_metrics(client, args.metrics, args.metrics_key)
        waiting = m.get("vllm:num_requests_waiting", 0.0)
        kv = m.get("vllm:kv_cache_usage_perc",
                   m.get("vllm:gpu_cache_usage_perc", 0.0))
        if waiting > args.max_waiting or kv > args.max_kv:
            strikes += 1
            if strikes >= 3:
                events.append({"event": "abort", "t": time.time(),
                               "waiting": waiting, "kv_usage": kv})
                print(f"\n  !! ABORT: waiting={waiting:.0f} kv={kv:.0%}",
                      file=sys.stderr)
                stop.set()
                return
        else:
            strikes = 0
        await asyncio.sleep(1.0)


async def plateau(client, args, arm, k, ctx, shared, out, rnd):
    """One held batch. Shared rungs send byte-identical prompts (prefix cache
    dedups the storage, every sequence still reads it); unique rungs send k
    independent prompts so read bytes and stored bytes are one number."""
    tag = "shared" if shared else "unique"
    print(f"  arm {arm}  k={k:<3} ctx~{ctx:>7,} {tag:<6} ", end="", flush=True)
    base = filler(ctx, rnd)
    prompts = [("Continue this text.\n\n" + base) if shared
               else ("Continue this text.\n\n" + filler(ctx, rnd)) for _ in range(k)]

    stop, stats, events = asyncio.Event(), {}, []
    ramp = args.settle + (args.lead if (shared and k > 1) else 0.0)
    watch = asyncio.create_task(watchdog(client, args, stop, events, ramp))
    # the shared rung's first stream pays the prefill alone, so the rest hit a
    # warm prefix instead of k cold prefills racing each other
    lead = None
    if shared and k > 1:
        lead = asyncio.create_task(stream_one(client, args, prompts[0], stop, stats))
        await asyncio.sleep(args.lead)
    streams = [asyncio.create_task(stream_one(client, args, p, stop, stats))
               for p in prompts[(1 if lead else 0):]]
    if lead:
        streams.insert(0, lead)

    await asyncio.sleep(args.settle)
    start = time.time()
    while time.time() - start < args.hold and not stop.is_set():
        await asyncio.sleep(0.5)
    end = time.time()
    aborted = stop.is_set()
    stop.set()
    for s in streams:
        s.cancel()
    watch.cancel()
    await asyncio.gather(*streams, watch, return_exceptions=True)

    pt = stats.get("ptok") or []
    rec = {"arm": arm, "k": k, "target_prompt_tokens": ctx, "shared_prefix": shared,
           # the analyser needs this: storage dedups a shared prefix, reads do
           # not, so sum_L from the KV gauge is short by (n-1) copies of it
           "shared_prefix_tokens": (sum(pt) / len(pt) if pt else ctx) if shared else 0,
           "measured_prompt_tokens": (sum(pt) / len(pt)) if pt else None,
           "start": start, "end": end, "held_s": end - start,
           "aborted": aborted, "events": events,
           "errors": stats.get("errors", [])[:3]}
    out.append(rec)
    print(f"held {end-start:4.0f}s"
          + (f"  ptok~{sum(pt)/len(pt):,.0f}" if pt else "")
          + ("  [ABORTED]" if aborted else "")
          + (f"  [{len(stats.get('errors', []))} errors]" if stats.get("errors") else ""))


async def run(args):
    out, rnd = [], random.Random(args.seed)
    hdr = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    limits = httpx.Limits(max_connections=args.max_k + 8)
    # one client serves both the endpoint and the watchdog's absolute
    # /metrics URL, so the TLS setting covers both
    async with httpx.AsyncClient(base_url=args.url.rstrip("/"), headers=hdr,
                                 limits=limits, verify=args.verify) as c:
        for arm, k, ctx, shared in rungs_for(args):
            m = await read_metrics(c, args.metrics, args.metrics_key)
            w = m.get("vllm:num_requests_waiting")
            if w is not None and w > args.max_waiting:
                print(f"  arm {arm}  k={k:<3} SKIPPED, instance busy (waiting={w:.0f})")
                continue
            await plateau(c, args, arm, k, ctx, shared, out, rnd)
            await asyncio.sleep(args.settle)
    return out


def plan_table(args):
    """Predicted step bytes, leverage and neighbour cost per rung. Uses the
    study's own byte ledger so the plan is priced by the model it calibrates."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import scenario_model as S
        m = S.MODELS[args.model_key]
        base = m.w_decode(1) + 2 * m.deltanet_state
    except Exception as e:                       # planning aid, never a blocker
        print(f"(byte estimates unavailable: {e})")
        return
    rows = []
    for arm, k, ctx, shared in rungs_for(args):
        sum_L = k * ctx                          # reads, shared or not
        b = m.w_decode(k) + sum_L * m.kv_bpt + 2 * k * m.deltanet_state
        pool = ctx if shared else k * ctx        # what it occupies
        rows.append((arm, k, ctx, shared, b, pool, b / base))
    lo = min(r[4] for r in rows)
    hi = max(r[4] for r in rows)
    print(f"{'arm':>4} {'k':>3} {'ctx':>8} {'prompts':>7} {'step bytes':>11} "
          f"{'pool@t0':>9}  neighbour ITL")
    for arm, k, ctx, shared, b, pool, slow in rows:
        print(f"{arm:>4} {k:>3} {ctx:>8,} {'shared' if shared else 'unique':>7} "
              f"{b/1e9:>8.1f} GB {pool/1e3:>7.0f}k  x{slow:.1f}")
    gen = args.max_k * 120 * args.hold / 1e3
    print(f"\npool@t0 is the PROMPT footprint; the streams then add up to "
          f"~{gen:.0f}k more tokens over a {args.hold:.0f}s plateau.")
    print(f"leverage across the plan: {hi/lo:.2f}x  "
          f"({'OK, >=2x identifies both terms' if hi/lo >= 2 else 'TOO LOW -- raise --max-k or --prefix-tokens'})")
    print(f"wall clock: ~{len(rows)*(args.hold+2*args.settle)/60:.1f} min")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", help="base such that {url}/chat/completions exists")
    ap.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY"))
    ap.add_argument("--model", default="", help="model name in the JSON body")
    ap.add_argument("--model-key", default="27B", help="scenario_model key, for --dry-run pricing")
    ap.add_argument("--arm", action="append", choices=["A", "B", "C"],
                    help="repeatable; default A then C")
    ap.add_argument("--metrics", help="/metrics URL for the watchdog "
                                      "(strongly recommended on a shared instance)")
    ap.add_argument("--metrics-key")
    ap.add_argument("--prefix-tokens", type=int, default=64_000,
                    help="arm A/C shared prefix length -- the leverage knob")
    ap.add_argument("--hold", type=float, default=40, help="plateau seconds")
    ap.add_argument("--settle", type=float, default=8,
                    help="ramp before a plateau, and idle after it")
    ap.add_argument("--lead", type=float, default=6,
                    help="head start for the prefix-warming stream on shared rungs")
    ap.add_argument("--out-tokens", type=int, default=16_000)
    ap.add_argument("--max-k", type=int, default=16, help="hard cap on streams")
    ap.add_argument("--max-waiting", type=float, default=0.0,
                    help="abort if this many requests queue")
    ap.add_argument("--max-kv", type=float, default=0.80,
                    help="abort if KV pool usage passes this fraction")
    ap.add_argument("--ca-bundle", help="CA bundle to verify against")
    ap.add_argument("--insecure", action="store_true",
                    help="skip TLS verification (internal endpoints only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms-file", default="arms.jsonl")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and what it costs, contact nothing")
    args = ap.parse_args()
    args.arm = args.arm or ["A", "C"]
    args.verify = args.ca_bundle or (False if args.insecure else True)
    if args.insecure:
        print("TLS verification DISABLED")
    if not args.model and args.url:
        args.model = args.url.rstrip("/").rsplit("/", 1)[-1]

    plan_table(args)
    if args.dry_run:
        print("\n--dry-run: nothing contacted.  Pair the real run with:")
        print("  uv run scripts/scrape_metrics.py <metrics-url> --interval 0.5")
        return 0
    if not args.url:
        ap.error("--url is required (or use --dry-run)")
    if not args.metrics:
        print("\nWARNING: no --metrics, so the watchdog is OFF and a plateau "
              "cannot abort on a queue. Other tenants share this bandwidth.\n",
              file=sys.stderr)

    out = []
    try:
        out = asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        if out:
            Path(args.arms_file).write_text("\n".join(json.dumps(o) for o in out) + "\n")
            print(f"\nwrote {args.arms_file} ({len(out)} plateaus, "
                  f"{sum(o['aborted'] for o in out)} aborted)")
            print("analyse:  uv run scripts/decode_mbu.py <scrape.log> "
                  f"--arms {args.arms_file} --pool-tokens <N>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
