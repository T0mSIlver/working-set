"""`ws decode-sweep`: decode step time against the number of sequences decoding.

For each batch size n it holds n sequences decoding at once, all at the same
context, and measures the decode step from two sides over one window:

  server  deltas of the engine's own `/metrics` counters and histograms
          (forward passes, generated tokens, inter-token latency, speculative
          decoding acceptance, the engine's estimated bytes read);
  client  the tokens and events each stream received (vLLM's
          `continuous_usage_stats` puts the running token count in every
          chunk).

The two must agree; `step_ratio` in each rung is client step time over server
step time. The server figure is the measurement, the client one is the check.

How a rung runs:

1. The context is a prompt of `--context` tokens. By default the n sequences
   share it (one prefill, then prefix-cache hits) and differ by a short
   random suffix. They differ on purpose: sequences with identical text route
   to the same experts, which would hide the MoE expert union. `--distinct`
   gives every sequence its own context instead (n prefills, n x context of KV).
2. The prompts are sent with `max_tokens=1` first, so prefill is done and the
   prefix cache holds the context.
3. n streams start together with `ignore_eos`. Once all n have produced a
   token and `--settle` seconds have passed, one `--hold` second window is
   measured. Then the streams are cancelled and the rung waits for the server
   to go idle.

A rung is `batch_ok` only when the server was seen running at least 95% of n
sequences for the whole window, nothing waited, no stream failed and none
finished early. A rung that fails any of these is still reported, with the
reason, and must not be fitted.

Rungs whose KV need exceeds the pool (read from `vllm:cache_config_info`, or
`--kv-tokens`) are skipped with a reason rather than left to preempt.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import re
import sys
import time
from pathlib import Path

from .metrics.parse import HistogramMismatch, HistogramReset, parse_text
from .metrics.vllm import detect_adapter
from .sweep_mixed import (classify, depth_profile, format_row, judge_mixed,
                          stream_gaps, summarize, summary_rows, SUMMARY_HEADER)

FORMAT = "workingset-decode-sweep"
VERSION = 1   # additive fields (mixed_windows, pure_step_s) keep v1 records readable
DEFAULT_BATCHES = "1,8,32,64,128,256,512"
MIN_RUNNING_SHARE = 0.95
_COMPLETION_TOKENS = re.compile(r'"completion_tokens"\s*:\s*(\d+)')
_PROMPT_TOKENS = re.compile(r'"prompt_tokens"\s*:\s*(\d+)')
_ALPHABET = "abcdefghijklmnopqrstuvwxyz"


# ---------------------------------------------------------------------------
# pure: arguments, text, the planning arithmetic
# ---------------------------------------------------------------------------
def parse_batches(s: str) -> list[int]:
    ns = [int(x) for x in s.split(",") if x.strip()]
    if not ns or min(ns) < 1:
        raise ValueError("--batch needs a comma list of batch sizes >= 1")
    return sorted(set(ns))


def parse_sizes(s: str | None, flag: str) -> list[int]:
    """A comma list of token counts, or nothing."""
    if not s:
        return []
    ns = [int(x) for x in s.split(",") if x.strip()]
    if not ns or min(ns) < 1:
        raise ValueError(f"{flag} needs a comma list of token counts >= 1")
    return sorted(set(ns))


def random_words(rnd: random.Random, chars: int) -> str:
    """Lowercase pseudo-words, about `chars` characters long."""
    out, n = [], 0
    while n < chars:
        w = "".join(rnd.choices(_ALPHABET, k=rnd.randint(3, 9)))
        out.append(w)
        n += len(w) + 1
    return " ".join(out)


def tile_text(text: str, chars: int) -> str:
    if not text.strip():
        raise ValueError("--prompt-file is empty")
    return (text.rstrip() + "\n\n") * (chars // (len(text) + 2) + 1)


def output_budget(rate_bound: float, span_s: float, margin: float = 1.25) -> int:
    """Tokens each stream may generate: enough that it outlasts the rung at
    `rate_bound` tok/s per sequence, so a stream never finishes in the window."""
    return max(64, int(rate_bound * span_s * margin))


def kv_need(n: int, context: int, suffix: int, out_tokens: int, shared: bool) -> int:
    """Tokens of KV the rung occupies at its end. A shared context is stored
    once; every sequence still reads all of it."""
    per_seq = suffix + out_tokens
    return context + n * per_seq if shared else n * (context + per_seq)


# ---------------------------------------------------------------------------
# pure: the two windows
# ---------------------------------------------------------------------------
def _delta(ad, a, b, key):
    x, y = ad.counter(a, key), ad.counter(b, key)
    return None if x is None or y is None else y - x


def _hist_delta(ad, a, b, key, notes):
    ha, hb = ad.histogram(a, key), ad.histogram(b, key)
    if ha is None or hb is None:
        return None
    try:
        return hb - ha
    except (HistogramMismatch, HistogramReset) as e:
        notes.append(str(e))
        return None


def server_window(ad, a, b, dt: float) -> dict:
    """The engine's view of one window, from two parsed `/metrics` dumps.

    `step_time_s` is wall time over forward passes. Under speculative decoding
    one pass emits `tokens_per_step / running` tokens per sequence, so
    `tok_s_seq` is generated tokens per second per running sequence, not
    1 / step_time.
    """
    notes: list[str] = []
    gen = _delta(ad, a, b, "generation_tokens_total")
    it = _hist_delta(ad, a, b, "iteration_tokens_hist", notes)
    steps = it.observations if it is not None else None
    itl = _hist_delta(ad, a, b, "tpot_hist", notes)
    out = {"dt_s": dt, "generation_tokens": gen, "steps": steps,
           "step_time_s": dt / steps if steps else None,
           "tokens_per_step": gen / steps if gen is not None and steps else None,
           "tok_s_total": gen / dt if gen is not None and dt > 0 else None,
           "itl_mean_s": itl.mean() if itl is not None else None,
           "itl_p50_s": itl.quantile(0.5) if itl is not None else None,
           "itl_p99_s": itl.quantile(0.99) if itl is not None else None,
           "itl_observations": itl.observations if itl is not None else None}
    rd = _delta(ad, a, b, "estimated_read_bytes_total")
    out["est_read_bytes_per_step"] = rd / steps if rd and steps else None
    drafts = _delta(ad, a, b, "spec_decode_num_drafts_total")
    drafted = _delta(ad, a, b, "spec_decode_num_draft_tokens_total")
    accepted = _delta(ad, a, b, "spec_decode_num_accepted_tokens_total")
    if drafts is not None and accepted is not None and drafts > 0:
        spec = {"drafts": drafts, "draft_tokens": drafted, "accepted_tokens": accepted,
                "acceptance_rate": accepted / drafted if drafted else None,
                "mean_accepted_len": 1.0 + accepted / drafts}
        pa, pb = (ad.by_position(a, "spec_decode_accepted_per_pos"),
                  ad.by_position(b, "spec_decode_accepted_per_pos"))
        if pa is not None and pb is not None:
            spec["accepted_per_position"] = {
                str(k): (pb[k] - pa.get(k, 0.0)) / drafts for k in sorted(pb)}
        out["spec_decode"] = spec
    else:
        out["spec_decode"] = None
    out["notes"] = notes
    return out


def client_window(tok0, tok1, ev0, ev1, dt: float, started: list[bool]) -> dict:
    """The clients' view: per-sequence tokens/s and events/s over the window,
    from each stream's running counts at the two ends."""
    idx = [i for i, s in enumerate(started) if s]
    if not idx or dt <= 0:
        return {"streams": 0}
    dtok = [tok1[i] - tok0[i] for i in idx]
    dev = [ev1[i] - ev0[i] for i in idx]
    have_tok = sum(dtok) > 0
    # no continuous usage on this server: an event is one token at best
    per = sorted((t if have_tok else e) / dt for t, e in zip(dtok, dev))
    ev_s = sum(dev) / len(dev) / dt
    return {"streams": len(idx), "tokens_counted": have_tok,
            "tok_s_seq_mean": sum(per) / len(per),
            "tok_s_seq_p10": per[int(0.1 * (len(per) - 1))],
            "tok_s_seq_p50": per[len(per) // 2],
            "events_s_seq": ev_s,
            "step_time_s": 1.0 / ev_s if ev_s > 0 else None}


def judge(n: int, running: list[float], waiting: list[float], errors: int,
          drained: int) -> tuple[bool, list[str]]:
    why = []
    if running and min(running) < MIN_RUNNING_SHARE * n:
        why.append(f"server ran only {min(running):.0f} of {n} sequences at its low point")
    if not running:
        why.append("no running-sequence gauge was read")
    if waiting and max(waiting) > 0:
        why.append(f"{max(waiting):.0f} requests waited")
    if errors:
        why.append(f"{errors} streams failed")
    if drained:
        why.append(f"{drained} streams finished inside the window")
    return not why, why


# ---------------------------------------------------------------------------
# the endpoint
# ---------------------------------------------------------------------------
class Endpoint:
    def __init__(self, client, args, headers_metrics: dict, metrics_client):
        # `metrics_client` has no default headers: the API client's bearer
        # token must not ride along to a /metrics on another host.
        self.c, self.a, self.mh = client, args, headers_metrics
        self.mc = metrics_client
        self.ad = None

    async def scrape(self):
        t0 = time.monotonic()
        r = await self.mc.get(self.a.metrics_url, headers=self.mh, timeout=self.a.timeout)
        r.raise_for_status()
        t1 = time.monotonic()
        s = parse_text(r.text)
        if self.ad is None:
            self.ad = detect_adapter(s, engine=self.a.engine)
        return (t0 + t1) / 2, s, r.text

    async def complete(self, prompt: str, max_tokens: int) -> int:
        r = await self.c.post("/completions", timeout=3600.0, json={
            "model": self.a.model, "prompt": prompt, "max_tokens": max_tokens,
            "temperature": 0.0})
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        return r.json()["usage"]["prompt_tokens"]


class Streams:
    """Running counts per stream, written by the stream tasks."""

    def __init__(self, n: int, record: bool = False):
        # (monotonic time, running token count) per event, for mixed-step gaps
        self.ts: list[list[tuple[float, int]]] | None = [[] for _ in range(n)] if record else None
        self.tok = [0] * n
        self.ev = [0] * n
        self.first = [False] * n
        self.done = [False] * n
        self.err: list[str] = []


async def _stream(ep: Endpoint, body: dict, i: int, st: Streams) -> None:
    import httpx
    try:
        async with ep.c.stream("POST", "/completions", json=body,
                               timeout=httpx.Timeout(None, connect=60.0)) as r:
            if r.status_code != 200:
                st.err.append(f"HTTP {r.status_code}: {(await r.aread())[:300].decode(errors='replace')}")
                return
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                if line[5:].strip() == "[DONE]":
                    st.done[i] = True
                    return
                st.ev[i] += 1
                st.first[i] = True
                m = _COMPLETION_TOKENS.search(line)
                if m:
                    st.tok[i] = int(m.group(1))
                if st.ts is not None:
                    st.ts[i].append((time.monotonic(), st.tok[i]))
    except httpx.HTTPError as e:
        st.err.append(f"{type(e).__name__}: {e}")


async def _gauges(ep: Endpoint, stop: asyncio.Event, out: dict) -> None:
    while not stop.is_set():
        try:
            _, s, _ = await ep.scrape()
            for key, dst in (("requests_running", "running"),
                             ("requests_waiting", "waiting"),
                             ("kv_cache_usage", "kv")):
                v = ep.ad.gauge(s, key)
                if v is not None:
                    out[dst].append(v)
        except Exception:
            pass
        try:
            await asyncio.wait_for(stop.wait(), 1.0)
        except asyncio.TimeoutError:
            pass


async def wait_idle(ep: Endpoint, timeout: float) -> bool | None:
    """True when the server was seen running nothing; None when the gauge is
    unreadable (do not sleep out the timeout on a wrong URL)."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        try:
            _, s, _ = await ep.scrape()
        except Exception:
            return None
        busy = (ep.ad.gauge(s, "requests_running") or 0) + (ep.ad.gauge(s, "requests_waiting") or 0)
        if busy == 0:
            return True
        await asyncio.sleep(1.0)
    return False


def pool_tokens(text: str, ep: Endpoint) -> tuple[int | None, bool | None]:
    """(KV pool size in tokens, prefix caching on) from vllm:cache_config_info."""
    for line in text.splitlines():
        if line.startswith("vllm:cache_config_info"):
            lab = dict(re.findall(r'(\w+)="([^"]*)"', line))
            try:
                blocks, bs = int(lab["num_gpu_blocks"]), int(lab["block_size"])
            except (KeyError, ValueError):
                return None, None
            return blocks * bs, lab.get("enable_prefix_caching") == "True"
    return None, None


# ---------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------
class Prompts:
    """Context text at a measured chars-per-token, and each sequence's prompt."""

    def __init__(self, args, chars_per_token: float):
        self.a, self.cpt = args, chars_per_token
        self.file = Path(args.prompt_file).read_text() if args.prompt_file else None

    def context(self, tokens: int, seq: int) -> str:
        chars = int(tokens * self.cpt)
        if self.file is not None:
            t = tile_text(self.file, chars)
            return t[:chars]
        return random_words(random.Random(f"{self.a.seed}/ctx/{seq}"), chars)

    def suffix(self, seq: int) -> str:
        return "\n\n" + random_words(random.Random(f"{self.a.seed}/sfx/{seq}"),
                                     int(self.a.suffix_tokens * self.cpt))


async def calibrate(ep: Endpoint, args) -> float:
    """Characters per token of this generator's text on this tokenizer."""
    probe = random_words(random.Random(f"{args.seed}/cal"), 8000)
    if args.prompt_file:
        probe = tile_text(Path(args.prompt_file).read_text(), 8000)[:8000]
    return len(probe) / await ep.complete(probe, 1)


async def cold_request(ep: Endpoint, args, prompt: str, timeout: float) -> dict:
    """One cold, one-token streamed request: when it was sent and when its first
    token came back (the interval whose steps carry its prefill)."""
    import httpx
    t_send, first, ptoks, err = time.monotonic(), None, None, None
    body = {"model": args.model, "prompt": prompt, "max_tokens": 1, "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True, "continuous_usage_stats": True}}

    async def go():
        nonlocal first, ptoks
        async with ep.c.stream("POST", "/completions", json=body,
                               timeout=httpx.Timeout(None, connect=60.0)) as r:
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: "
                                   f"{(await r.aread())[:200].decode(errors='replace')}")
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                if line[5:].strip() == "[DONE]":
                    return
                if first is None:
                    first = time.monotonic()
                m = _PROMPT_TOKENS.search(line)
                if m:
                    ptoks = int(m.group(1))
    try:
        await asyncio.wait_for(go(), timeout)
    except (httpx.HTTPError, RuntimeError, asyncio.TimeoutError) as e:
        err = f"{type(e).__name__}: {e}"
    return {"send": t_send, "first": first, "prompt_tokens": ptoks, "error": err}


async def mixed_window(ep: Endpoint, args, pr: Prompts, n: int, st: Streams,
                       kind: str, value: int, pure_s: float | None, tag: int) -> dict:
    """One window with the n held streams decoding and cold prefill requests in
    flight: `kind` "tokens" keeps a stream of `value`-token requests, one at a
    time, for --hold seconds; "depth" sends one `value`-token request, which
    the server chunks."""
    gauges = {"running": [], "waiting": [], "kv": []}
    stop_g = asyncio.Event()
    g = asyncio.create_task(_gauges(ep, stop_g, gauges))
    err0, done0 = len(st.err), sum(st.done)
    ta, sa, _ = await ep.scrape()
    reqs: list[dict] = []

    def prompt(i: int) -> str:
        rnd = random.Random(f"{args.seed}/cold/{n}/{kind}/{value}/{tag}/{i}")
        return random_words(rnd, int(value * pr.cpt))

    if kind == "tokens":
        while time.monotonic() < ta + args.hold:
            reqs.append(await cold_request(ep, args, prompt(len(reqs)), args.prefill_timeout))
            if reqs[-1]["error"]:
                break
    else:
        reqs.append(await cold_request(ep, args, prompt(0), args.prefill_timeout))
        await asyncio.sleep(1.0)   # a pure tail after the last chunk
    tb, sb, _ = await ep.scrape()
    stop_g.set()
    await g
    dt = tb - ta
    good = [r for r in reqs if r["first"] is not None and not r["error"]]
    intervals = [(r["send"], r["first"]) for r in good]
    have_tok = any(t[-1][1] for t in st.ts if t)
    per = [classify(stream_gaps(ev, ta, tb, have_tok), intervals) for ev in st.ts]
    pt = [r["prompt_tokens"] for r in good if r["prompt_tokens"]]
    w = {"kind": kind, "window_s": dt, "requests": len(good),
         "request_errors": [r["error"] for r in reqs if r["error"]][:3],
         "request_s_mean": (sum(r["first"] - r["send"] for r in good) / len(good)
                            if good else None),
         "prompt_tokens_mean": sum(pt) / len(pt) if pt else None,
         "server": server_window(ep.ad, sa, sb, dt),
         "mixed": summarize(per),
         "running": {k: (min(v), sum(v) / len(v), max(v)) for k, v in gauges.items() if v}}
    if kind == "tokens":
        w["prefill_tokens"] = value
    else:
        w["prefill_depth"] = value
        w["chunk_tokens"] = args.chunk_tokens
        w["depth_profile"] = depth_profile(per, value, args.chunk_tokens)
    # the server's mean step blends mixed and pure steps; with the mixed share
    # known, what the mixed ones cost
    s_step, share = w["server"]["step_time_s"], w["mixed"]["share_mixed"]
    w["server_mixed_step_est_s"] = (
        (s_step - (1 - share) * pure_s) / share
        if s_step and pure_s and share and share > 0.05 else None)
    w["ok"], w["why_not"] = judge_mixed(
        n, gauges["running"], len(st.err) - err0, sum(st.done) - done0, len(good))
    w["waiting_max"] = max(gauges["waiting"]) if gauges["waiting"] else None
    return w


async def run_rung(ep: Endpoint, args, pr: Prompts, n: int, out_tokens: int,
                   pool: int | None) -> dict:
    shared = not args.distinct
    ctx = args.context
    rec: dict = {"n": n, "context_target": ctx, "shared": shared,
                 "out_tokens": out_tokens}
    need = kv_need(n, ctx, args.suffix_tokens, out_tokens, shared)
    rec["kv_need_tokens"] = need
    if pool and need > args.kv_fill * pool:
        rec["skipped"] = (f"needs ~{need:,} KV tokens, over {args.kv_fill:.0%} of the "
                          f"{pool:,}-token pool")
        return rec
    base = pr.context(ctx, 0) if shared else None
    prompts = [(base if shared else pr.context(ctx, i)) + pr.suffix(i) for i in range(n)]

    # 1. warm the prefix cache; the prompt length it reports is the context
    t0 = time.monotonic()
    sem = asyncio.Semaphore(args.warm_concurrency)

    async def warm(p):
        async with sem:
            return await ep.complete(p, 1)
    if shared:
        ptoks = [await warm(prompts[0])] + list(await asyncio.gather(
            *[warm(p) for p in prompts[1:]]))
    else:
        ptoks = list(await asyncio.gather(*[warm(p) for p in prompts]))
    rec["warm_s"] = round(time.monotonic() - t0, 1)
    rec["prompt_tokens_mean"] = sum(ptoks) / len(ptoks)
    if await wait_idle(ep, args.idle_timeout) is False:
        rec["server_idle"] = False

    # 2. start the batch
    st = Streams(n, record=bool(args.prefill_tokens or args.prefill_depth))
    gauges = {"running": [], "waiting": [], "kv": []}
    stop_g = asyncio.Event()
    tasks = [asyncio.create_task(_stream(ep, {
        "model": args.model, "prompt": p, "max_tokens": out_tokens,
        "temperature": args.temperature, "seed": args.seed * 100003 + i,
        "ignore_eos": True, "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True}},
        i, st)) for i, p in enumerate(prompts)]
    t_ramp = time.monotonic()
    while not all(st.first) and time.monotonic() - t_ramp < args.ramp_timeout and not st.err:
        await asyncio.sleep(0.25)
    rec["ramp_s"] = round(time.monotonic() - t_ramp, 1)
    await asyncio.sleep(args.settle)

    # 3. the window
    g = asyncio.create_task(_gauges(ep, stop_g, gauges))
    ta, sa, _ = await ep.scrape()
    c0 = (list(st.tok), list(st.ev), list(st.done))
    await asyncio.sleep(args.hold)
    tb, sb, _ = await ep.scrape()
    c1 = (list(st.tok), list(st.ev), list(st.done))
    stop_g.set()
    await g

    dt = tb - ta
    started = list(st.first)
    drained = sum(1 for x in c1[2] if x)
    rec["server"] = server_window(ep.ad, sa, sb, dt)
    rec["pure_step_s"] = rec["server"]["step_time_s"]
    # 4. the same streams again, with cold prefill in the batch
    rec["mixed_windows"] = []
    loads = ([("tokens", p) for p in args.prefill_tokens]
             + [("depth", d) for d in args.prefill_depth])
    for tag, (kind, v) in enumerate(loads):
        if st.err or drained:
            break
        w = await mixed_window(ep, args, pr, n, st, kind, v, rec["pure_step_s"], tag)
        rec["mixed_windows"].append(w)
        print(f"  n={n}: {w['kind']} {v:,} done, {w['requests']} cold requests", flush=True)
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    rec["client"] = client_window(c0[0], c1[0], c0[1], c1[1], dt, started)
    rec["running"] = {k: (min(v), sum(v) / len(v), max(v)) for k, v in gauges.items() if v}
    rec["batch_ok"], rec["why_not"] = judge(
        n, gauges["running"], gauges["waiting"], len(st.err), drained)
    rec["errors"] = st.err[:3]
    s, c = rec["server"], rec["client"]
    if s["step_time_s"] and c.get("step_time_s"):
        rec["step_ratio"] = c["step_time_s"] / s["step_time_s"]
    run_mean = rec["running"].get("running", (0, n, 0))[1] or n
    rec["tok_s_seq_server"] = s["tok_s_total"] / run_mean if s["tok_s_total"] else None
    return rec


def _row(r: dict) -> str:
    if r.get("skipped"):
        return f"{r['n']:>5}  skipped: {r['skipped']}"
    s, c = r["server"], r["client"]
    f = lambda x, spec: format(x, spec) if x is not None else "-"
    sp = s["spec_decode"]
    return (f"{r['n']:>5} {f(r['prompt_tokens_mean'], ',.0f'):>9} "
            f"{f(s['step_time_s'] and s['step_time_s'] * 1e3, '8.2f'):>9} "
            f"{f(c.get('step_time_s') and c['step_time_s'] * 1e3, '8.2f'):>9} "
            f"{f(r.get('step_ratio'), '.2f'):>6} "
            f"{f(r['tok_s_seq_server'], '8.1f'):>9} {f(c.get('tok_s_seq_mean'), '8.1f'):>9} "
            f"{f(s['tok_s_total'], '9.0f'):>9} "
            f"{f(sp and sp['mean_accepted_len'], '.2f'):>6}"
            + ("" if r["batch_ok"] else "  NOT OK: " + "; ".join(r["why_not"])))


HEADER = (f"{'n':>5} {'ctx tok':>9} {'step ms':>9} {'cli ms':>9} {'ratio':>6} "
          f"{'tok/s/seq':>9} {'cli t/s':>9} {'tok/s tot':>9} {'acc len':>6}")


async def sweep(args) -> dict:
    import httpx
    from .metrics.cli import _headers, _verify
    hdr = _headers(args)
    base = args.base_url.rstrip("/")
    origin = re.match(r"^\w+://[^/]+", base).group(0)
    if not args.metrics_url:
        args.metrics_url = origin + "/metrics"
    same = args.metrics_url.startswith(origin + "/")
    mh = dict(hdr) if same else {}
    for h in args.metrics_header or []:
        k, _, v = h.partition(":")
        mh[k.strip()] = v.strip()
    limits = httpx.Limits(max_connections=max(args.batch) + 32,
                          max_keepalive_connections=max(args.batch) + 32)
    async with httpx.AsyncClient(base_url=base, headers=hdr, limits=limits,
                                 verify=_verify(args)) as c, \
            httpx.AsyncClient(limits=limits, verify=_verify(args)) as mc:
        ep = Endpoint(c, args, mh, mc)
        _, s0, text0 = await ep.scrape()
        pool, prefix_on = pool_tokens(text0, ep)
        pool = args.kv_tokens or pool
        if not args.distinct and prefix_on is False and not args.force:
            raise ValueError("the server runs without prefix caching: a shared context "
                             "would be prefilled once per sequence. Use --distinct "
                             "or --force.")
        max_len = None
        try:
            r = await c.get("/models", timeout=args.timeout)
            for m in r.json().get("data", []):
                if m.get("id") == args.model:
                    max_len = m.get("max_model_len")
        except Exception:
            pass
        if max_len and args.context + args.suffix_tokens + 64 > max_len:
            raise ValueError(f"--context {args.context:,} does not fit max_model_len {max_len:,}")
        print(f"decode sweep: {args.model} at {base}, context {args.context:,} "
              f"({'distinct' if args.distinct else 'shared'}), hold {args.hold:.0f}s, "
              f"KV pool {pool:,} tokens" if pool else
              f"decode sweep: {args.model} at {base}, context {args.context:,}, KV pool unknown")
        if ep.ad.has("spec_decode_num_drafts_total"):
            print("speculative decoding counters present: acceptance is recorded")
        cpt = await calibrate(ep, args)
        pr = Prompts(args, cpt)
        print(f"{cpt:.2f} characters per token\n{HEADER}")
        rungs, rate = [], args.max_tok_s
        # the streams outlast the pure window, every prefill-token window and
        # an allowance per depth request (a stream that ends early voids its window)
        span = (args.settle + args.hold + args.ramp_allowance
                + len(args.prefill_tokens) * args.hold
                + len(args.prefill_depth) * args.ramp_allowance)
        for n in args.batch:
            out = output_budget(rate, span)
            if max_len:
                out = min(out, max_len - args.context - args.suffix_tokens - 64)
            rec = await run_rung(ep, args, pr, n, out, pool)
            rungs.append(rec)
            print(_row(rec), flush=True)
            if rec.get("batch_ok") and rec["tok_s_seq_server"]:
                # per-sequence speed only falls as n grows: the last clean
                # rung bounds the next one's output budget from above
                rate = min(rate, rec["tok_s_seq_server"])
            if await wait_idle(ep, args.idle_timeout) is False:
                print("  server still busy after the idle timeout; next rung may be contaminated")
        return {"format": FORMAT, "version": VERSION,
                "prefill_tokens": args.prefill_tokens, "prefill_depth": args.prefill_depth,
                "chunk_tokens": args.chunk_tokens, "min_tok_s": args.min_tok_s, "time": time.time(),
                "model": args.model, "base_url": base, "metrics_url": args.metrics_url,
                "context": args.context, "shared_context": not args.distinct,
                "suffix_tokens": args.suffix_tokens, "temperature": args.temperature,
                "hold_s": args.hold, "settle_s": args.settle, "seed": args.seed,
                "chars_per_token": cpt, "kv_pool_tokens": pool, "max_model_len": max_len,
                "engine": args.engine, "adapter": ep.ad.resolution().version_hint,
                "tensor_parallel": args.tensor_parallel,
                "rungs": rungs}


def cmd_decode_sweep(args) -> int:
    import os
    args.batch = parse_batches(args.batch)
    try:
        args.prefill_tokens = parse_sizes(args.prefill_tokens, "--prefill-tokens")
        args.prefill_depth = parse_sizes(args.prefill_depth, "--prefill-depth")
    except ValueError as e:
        print(f"ws: {e}", file=sys.stderr)
        return 2
    if not args.api_key and args.api_key_env and not os.environ.get(args.api_key_env):
        print(f"ws: --api-key-env {args.api_key_env} is not set", file=sys.stderr)
        return 2
    res = asyncio.run(sweep(args))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))
        print(f"\nwrote {args.out}")
    rows = summary_rows(res, args.min_tok_s)
    if rows:
        print(f"\nmixed steps (cold prefill in the batch; pure ms = decode alone), "
              f"floor {args.min_tok_s:g} tok/s per sequence\n{SUMMARY_HEADER}")
        for r in rows:
            print(format_row(r))
    bad = [r["n"] for r in res["rungs"] if not r.get("batch_ok") and not r.get("skipped")]
    if bad:
        print(f"rungs not OK (do not fit them): n = {bad}", file=sys.stderr)
    return 0


def add_subparser(sub) -> None:
    from .metrics.cli import _add_common
    p = sub.add_parser(
        "decode-sweep", help="decode step time at n sequences held decoding",
        description=__doc__.split("\n\n")[0] +
        "\n\nHold n sequences decoding at a fixed context and measure the decode "
        "step from the server's /metrics and from the clients. See the module "
        "docstring and docs/decode-sweep.md.")
    p.add_argument("--base-url", required=True, help="OpenAI base, e.g. http://host:8000/v1")
    p.add_argument("--model", required=True)
    _add_common(p)
    p.add_argument("--metrics-url", help="default: <origin of --base-url>/metrics")
    p.add_argument("--metrics-header", action="append", metavar="'Name: value'",
                   help="extra header for the /metrics request (repeatable). "
                        "The API key is sent to /metrics only when it is on the "
                        "same origin as --base-url")
    p.add_argument("--batch", default=DEFAULT_BATCHES, metavar="N,N,...",
                   help=f"sequences held decoding (default {DEFAULT_BATCHES})")
    p.add_argument("--context", type=int, default=100_000, metavar="TOKENS",
                   help="prompt tokens every sequence starts decoding from (default 100000)")
    p.add_argument("--distinct", action="store_true",
                   help="give every sequence its own context (n x context of KV) "
                        "instead of one shared context")
    p.add_argument("--prompt-file", help="tile this text to the context instead of random "
                                         "words (speculative-decoding acceptance on random "
                                         "words is not representative)")
    p.add_argument("--suffix-tokens", type=int, default=128,
                   help="random tokens after the context that make sequences differ (default 128)")
    p.add_argument("--temperature", type=float, default=0.6,
                   help="sampling temperature; at 0 identical prompts decode identical "
                        "tokens (default 0.6)")
    p.add_argument("--prefill-tokens", metavar="P,P,...",
                   help="for each n and each P, also measure with cold requests of P tokens "
                        "(max_tokens=1) kept in flight one at a time while the n sequences "
                        "decode: the mixed step. Needs max_num_seqs > n")
    p.add_argument("--prefill-depth", metavar="D,D,...",
                   help="for each n and each D, also measure one cold request of D tokens, "
                        "which the server chunks: mixed step time against chunk position")
    p.add_argument("--chunk-tokens", type=int, metavar="TOKENS",
                   help="the server's max_num_batched_tokens, to place --prefill-depth steps "
                        "(default: estimated as D over the mixed steps seen)")
    p.add_argument("--prefill-timeout", type=float, default=300,
                   help="longest a cold request may take, seconds (default 300)")
    p.add_argument("--min-tok-s", type=float, default=50,
                   help="per-sequence tok/s floor; mixed rows below it are marked (default 50)")
    p.add_argument("--hold", type=float, default=30, help="measured window, seconds (default 30)")
    p.add_argument("--settle", type=float, default=10,
                   help="seconds after every sequence has started before the window (default 10)")
    p.add_argument("--ramp-timeout", type=float, default=900,
                   help="give up waiting for every sequence's first token after this (default 900)")
    p.add_argument("--ramp-allowance", type=float, default=60,
                   help="seconds of decoding assumed before the window, for sizing "
                        "max_tokens (default 60)")
    p.add_argument("--max-tok-s", type=float, default=300,
                   help="per-sequence tok/s ceiling used to size max_tokens for the first "
                        "rung; later rungs use the last clean rung's measured rate (default 300)")
    p.add_argument("--kv-tokens", type=int, help="KV pool size in tokens, when /metrics lacks it")
    p.add_argument("--tensor-parallel", type=int, metavar="TP",
                   help="the server's --tensor-parallel-size, recorded so `ws calibrate` "
                        "can tell runs at two widths apart (default: the config's)")
    p.add_argument("--kv-fill", type=float, default=0.9,
                   help="skip a rung needing more than this share of the pool (default 0.9)")
    p.add_argument("--warm-concurrency", type=int, default=8,
                   help="prefill requests in flight while warming (default 8)")
    p.add_argument("--idle-timeout", type=float, default=300,
                   help="seconds to wait for the server to go idle between rungs (default 300)")
    p.add_argument("--force", action="store_true", help="run a shared context without prefix caching")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", metavar="FILE.json", help="write the full result here")
    p.set_defaults(fn=cmd_decode_sweep)
