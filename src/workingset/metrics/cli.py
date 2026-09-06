"""`ws metrics` — the sampler's command line.

    ws metrics probe  URL                     what can this server measure?
    ws metrics tail   URL [--out FILE.jsonl]  live line + raw JSONL log
    ws metrics window FILE.jsonl --from T --to T   delta a recorded window

`probe` is the one to run first: it prints which semantic keys this
deployment exports, so you learn BEFORE a run whether the question you want
to ask is answerable here (no prefix-cache counters -> no measured miss
rate; no spec-decode counters -> no measured acceptance).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import signal
import sys
import time

from .adapter import KEY_KIND, KEY_UNITS, SEMANTIC_KEYS
from .sampler import MetricsSampler, keep_filter, load_jsonl, window_from_snapshots

__all__ = ["add_subparser", "cmd_probe", "cmd_tail", "cmd_window"]


# ---------------------------------------------------------------------------
# shared option handling
# ---------------------------------------------------------------------------
def _verify(args) -> bool | str:
    """--ca-bundle / --insecure -> httpx's `verify`.

    A corporate interception proxy presents its own certificate, so the
    system trust store is not enough. `--ca-bundle` keeps verification and is
    the right fix; `--insecure` turns it off, for an internal endpoint only.
    """
    if getattr(args, "ca_bundle", None):
        return args.ca_bundle
    return not getattr(args, "insecure", False)


def _headers(args) -> dict[str, str]:
    """`--api-key` or `--api-key-env` -> an Authorization header. The env-var
    form is preferred: a key on the command line lands in the shell history
    and in `ps`."""
    key = getattr(args, "api_key", None)
    env = getattr(args, "api_key_env", None)
    if not key and env:
        key = os.environ.get(env)
    return {"Authorization": f"Bearer {key}"} if key else {}


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--api-key", help="bearer token (prefer --api-key-env)")
    p.add_argument("--api-key-env", metavar="VAR",
                   help="env var holding the bearer token")
    p.add_argument("--ca-bundle", help="CA bundle to verify TLS against")
    p.add_argument("--insecure", action="store_true",
                   help="skip TLS verification (internal endpoints only)")
    p.add_argument("--timeout", type=float, default=5.0, help="seconds")
    p.add_argument("--engine", help="engine index to select under DP>1 "
                                    "(default: sum every engine)")


def _fmt(x, nd: int = 2) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        if math.isnan(x):
            return "-"
        if math.isinf(x):
            return "inf"
        return f"{x:,.{nd}f}"
    return f"{x:,}" if isinstance(x, int) else str(x)


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------
def cmd_probe(args) -> int:
    async def run():
        s = MetricsSampler(args.url, headers=_headers(args), verify=_verify(args),
                           timeout=args.timeout, engine=args.engine)
        return await s.probe()

    snap, res = asyncio.run(run())
    if not snap.ok:
        print(f"scrape failed: {snap.error}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps({"url": args.url, "rtt_s": snap.rtt,
                          "n_series": len(snap.samples), **res.to_dict()}, indent=2))
        return 0
    print(f"{args.url}  rtt {snap.rtt * 1e3:.0f} ms  {len(snap.samples)} series  "
          f"engine={res.engine} ({res.version_hint})")
    print()
    width = max(len(k) for k in SEMANTIC_KEYS)
    for key in SEMANTIC_KEYS:
        name = res.resolved.get(key)
        unit = KEY_UNITS.get(key, "")
        if name:
            print(f"  ok      {key:<{width}}  {name}   [{unit}]")
        else:
            print(f"  MISSING {key:<{width}}  -")
    print()
    print(f"  {len(res.found)}/{len(SEMANTIC_KEYS)} semantic keys resolved")
    if res.missing:
        print(f"  a run against this server cannot measure: {', '.join(res.missing)}")
    return 0


# ---------------------------------------------------------------------------
# tail
# ---------------------------------------------------------------------------
_TAIL_HEAD = (f"{'elapsed':>8} {'run':>5} {'wait':>5} {'kv%':>6} "
              f"{'hit%':>6} {'tok/s':>9}")


def cmd_tail(args) -> int:
    async def run() -> int:
        started = time.time()
        n = 0
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, stop.set)
            except (NotImplementedError, ValueError):
                pass
        async with MetricsSampler(args.url, interval=args.interval,
                                  headers=_headers(args), verify=_verify(args),
                                  out=args.out, keep=keep_filter(args.keep),
                                  timeout=args.timeout, engine=args.engine) as s:
            if args.out:
                print(f"appending raw snapshots to {args.out}", file=sys.stderr)
            print(_TAIL_HEAD, file=sys.stderr)
            seen = 0
            while not stop.is_set():
                try:
                    await asyncio.wait_for(stop.wait(), timeout=args.interval / 4)
                except asyncio.TimeoutError:
                    pass
                if len(s) == seen:
                    continue
                seen = len(s)
                snap = s.snapshots[-1]
                v = s.live() if snap.ok else {}
                if not v:
                    # a failed scrape, or a --keep that matched nothing
                    why = snap.error or "no series survived --keep"
                    print(f"{snap.t_sent - started:8.1f} {why}", file=sys.stderr)
                else:
                    kv = v.get("kv")
                    hit = v.get("hit_rate")
                    print(f"{v['t'] - started:8.1f} {_fmt(v.get('running'), 0):>5} "
                          f"{_fmt(v.get('waiting'), 0):>5} "
                          f"{'-' if kv is None else f'{kv * 100:6.1f}'} "
                          f"{'-' if hit is None else f'{hit * 100:6.1f}'} "
                          f"{_fmt(v.get('tok_s'), 1):>9}", flush=True)
                n = seen
                if args.count and n >= args.count:
                    break
            print(f"{n} snapshots, {s.n_failed} failed", file=sys.stderr)
        return 0

    return asyncio.run(run())


# ---------------------------------------------------------------------------
# window
# ---------------------------------------------------------------------------
def _parse_t(spec: str, snaps) -> float:
    """A window bound: an absolute unix time, or `+SECONDS` / `-SECONDS`
    relative to the log's first / last snapshot."""
    spec = spec.strip()
    if spec.startswith("+"):
        return snaps[0].t + float(spec[1:])
    if spec.startswith("-"):
        return snaps[-1].t - float(spec[1:])
    return float(spec)


def cmd_window(args) -> int:
    snaps = load_jsonl(args.log)
    if len(snaps) < 2:
        print(f"{args.log}: {len(snaps)} snapshots, need >= 2", file=sys.stderr)
        return 1
    t0 = _parse_t(args.frm, snaps) if args.frm else snaps[0].t
    t1 = _parse_t(args.to, snaps) if args.to else snaps[-1].t
    w = window_from_snapshots(snaps, t0, t1)
    if args.json:
        print(json.dumps(w.to_dict(), indent=2, default=str))
        return 0

    print(f"{args.log}: window [{t0:.3f}, {t1:.3f}] -> endpoints "
          f"[{w.lo.t:.3f}, {w.hi.t:.3f}]  (nearest snapshots OUTSIDE the ask)")
    print(f"  dt {w.dt:.3f} s  +/- {w.dt_uncertainty * 1e3:.0f} ms from scrape "
          f"round trips  |  {w.n_snapshots} snapshots, {w.n_failed} failed  "
          f"|  {w.version_hint}")
    print()
    print("  counters (delta over the window)")
    for k, kind in KEY_KIND.items():
        if kind != "counter":
            continue
        v = w.counters.get(k)
        if v is None:
            continue
        print(f"    {k:<38} {_fmt(v):>16}   {KEY_UNITS.get(k, '')}")
    print()
    print("  gauges (over the window)")
    for k, g in w.gauges.items():
        if not g.n:
            continue
        print(f"    {k:<38} mean {_fmt(g.mean, 3):>10}  max {_fmt(g.max, 3):>10}"
              f"  (n={g.n})")
    live = [(k, h) for k, h in w.histograms.items()
            if h is not None and h.observations]
    if live:
        print()
        print(f"  histograms{'':<28}{'count':>8}{'mean':>10}{'p50':>10}"
              f"{'p95':>10}{'p99':>10}")
        for k, h in live:
            print(f"    {k:<36}{h.observations:>8.0f}{_fmt(h.mean(), 4):>10}"
                  f"{_fmt(h.quantile(0.5), 4):>10}{_fmt(h.quantile(0.95), 4):>10}"
                  f"{_fmt(h.quantile(0.99), 4):>10}")
    print()
    print("  derived")
    for label, val, nd in (("output tok/s", w.output_tok_s, 1),
                           ("prefix hit rate", w.prefix_hit_rate, 4),
                           ("miss rate", w.miss_rate, 4),
                           ("alpha (per draft token)", w.alpha, 4),
                           ("mean accepted len", w.mean_accepted_len, 4),
                           ("draft width k", w.draft_width, 2),
                           ("forward passes", w.steps or None, 0),
                           ("tokens / pass", w.tokens_per_step, 1),
                           ("wall s / pass", w.step_time_s, 5)):
        if val is not None:
            print(f"    {label:<38} {_fmt(val, nd):>16}")
    if w.per_position and any(w.per_position.values()):
        pos = "  ".join(f"{p}:{v:,.0f}" for p, v in sorted(w.per_position.items()))
        print(f"    {'accepted by draft position':<38} {pos}")
    return 0


# ---------------------------------------------------------------------------
# wiring
# ---------------------------------------------------------------------------
def add_subparser(sub) -> None:
    """Attach `ws metrics ...` to the top-level subparsers object."""
    p = sub.add_parser("metrics", help="sample and delta a vLLM /metrics endpoint",
                       description=__doc__,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    msub = p.add_subparsers(dest="metrics_cmd", required=True)

    q = msub.add_parser("probe", help="which semantic keys does this server export?")
    q.add_argument("url", help="the /metrics URL")
    q.add_argument("--json", action="store_true")
    _add_common(q)
    q.set_defaults(fn=cmd_probe)

    q = msub.add_parser("tail", help="live line + raw JSONL log")
    q.add_argument("url", help="the /metrics URL")
    q.add_argument("--interval", type=float, default=1.0, help="seconds")
    q.add_argument("--out", metavar="FILE.jsonl",
                   help="append every raw snapshot here")
    q.add_argument("--keep", default="all",
                   help="'all', 'decode' (8x smaller, but no prefix-cache "
                        "counters and no latency histograms), or "
                        "comma-separated name substrings")
    q.add_argument("--count", type=int, default=0,
                   help="stop after N snapshots (0 = until Ctrl-C)")
    _add_common(q)
    q.set_defaults(fn=cmd_tail)

    q = msub.add_parser("window", help="delta a window out of a recorded log")
    q.add_argument("log", metavar="FILE.jsonl")
    q.add_argument("--from", dest="frm", help="unix time, or +S from log start")
    q.add_argument("--to", help="unix time, or -S from log end")
    q.add_argument("--json", action="store_true")
    q.set_defaults(fn=cmd_window)
