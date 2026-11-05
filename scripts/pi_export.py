#!/usr/bin/env python3
"""Export pi sessions for a load test: numbers and timings only, no text.

    python3 pi_export.py                  # sessions of models whose id contains "glm"
    python3 pi_export.py --model qwen -o qwen

Writes two files, both safe to carry off the machine (no prompt or reply text):

  <out>-trace.json   the selected sessions as a `ws trace pi` trace, for
                     `ws replay` (needs workingset >= 0.4.1 on PATH as `ws`)
  <out>-cache.json   per model, how much of each request's prompt the server
                     served from its prefix cache, against the previous
                     request's prompt (what an append-only history could reuse)

A request's cache result, when nothing reset the history since the previous
request (compaction, branch, model change):

  hit    cached >= 95% of the previous prompt
  tail   50-95%: the start was kept, the end re-prefilled (a template that
         rewrites recent turns, e.g. dropping reasoning)
  lost   < 50%: the prefix was gone (evicted, or sent to another replica)

`lost` split by the idle gap before the request tells the two causes apart:
eviction grows with the gap, a replica switch does not.
"""
import argparse
import collections
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

GAPS = ((60, "<1 min"), (600, "1-10 min"), (float("inf"), ">10 min"))


def ts(d):
    t = d.get("timestamp")
    try:
        return datetime.fromisoformat(str(t).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def requests(path):
    """(model, prompt, cached, gap_s, reset) per assistant request, file order."""
    reset, last = True, None
    for line in open(path, encoding="utf-8", errors="replace"):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        t = d.get("type")
        if t in ("compaction", "branch_summary", "model_change"):
            reset = True
            continue
        m = d.get("message") if t == "message" else None
        if not isinstance(m, dict):
            continue
        now = ts(d)
        if m.get("role") != "assistant":
            last = now if now is not None else last
            continue
        u = m.get("usage") or {}
        prompt = (u.get("input", 0) + u.get("cacheRead", 0)
                  + u.get("cacheWrite", 0))
        if prompt <= 0:
            continue
        gap = now - last if now is not None and last is not None else None
        yield m.get("model") or "?", prompt, u.get("cacheRead", 0), gap, reset
        reset, last = False, now


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sessions", type=Path,
                    default=Path.home() / ".pi/agent/sessions")
    ap.add_argument("--model", default="glm",
                    help="keep sessions whose requests' model id contains "
                         "this (case-insensitive; default glm)")
    ap.add_argument("-o", "--out", default="pi-glm")
    a = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    files = [f for f in sorted(a.sessions.glob("*/*.jsonl"))
             if not f.parent.name.startswith("--tmp")]
    per_model = collections.Counter()
    keep = []
    stats = collections.defaultdict(lambda: collections.Counter())
    for f in files:
        reqs = list(requests(f))
        if not reqs:
            continue
        mine = sum(1 for r in reqs if a.model.lower() in r[0].lower())
        for r in reqs:
            per_model[r[0]] += 1
        if mine >= 0.9 * len(reqs):
            keep.append(f)
        prev = None
        for model, prompt, cached, gap, reset in reqs:
            s = stats[model]
            s["requests"] += 1
            s["prompt_tokens"] += prompt
            s["cached_tokens"] += cached
            if prev is not None and not reset and prev > 0:
                r = cached / prev
                kind = "hit" if r >= 0.95 else "tail" if r >= 0.5 else "lost"
                s[kind] += 1
                if kind == "lost" and gap is not None:
                    s["lost " + next(n for lim, n in GAPS if gap < lim)] += 1
                if gap is not None:
                    s["judged " + next(n for lim, n in GAPS if gap < lim)] += 1
            prev = prompt

    print(f"{len(files)} session files; requests per model:")
    for m, n in per_model.most_common():
        print(f"  {n:8d}  {m}")
    print(f"\ncache, per model (requests judged against the previous one):")
    for m, s in sorted(stats.items(), key=lambda kv: -kv[1]["requests"]):
        j = s["hit"] + s["tail"] + s["lost"]
        if not j:
            continue
        share = s["cached_tokens"] / max(s["prompt_tokens"], 1)
        line = (f"  {m}: {s['requests']} requests, {share:.1%} of prompt "
                f"tokens cached; hit {s['hit'] / j:.1%}, tail "
                f"{s['tail'] / j:.1%}, lost {s['lost'] / j:.1%}")
        if s["cached_tokens"] == 0:
            line += ("  <- no cached tokens reported: is vLLM's "
                     "--enable-prompt-tokens-details on?")
        print(line)
        for _, n in GAPS:
            if s["judged " + n]:
                print(f"      lost after a {n:9s} gap: "
                      f"{s['lost ' + n] / s['judged ' + n]:.1%} "
                      f"of {s['judged ' + n]}")
    Path(a.out + "-cache.json").write_text(json.dumps(
        {m: dict(s) for m, s in stats.items()}, indent=1))
    print(f"\nwrote {a.out}-cache.json")

    print(f"{len(keep)} sessions with >= 90% of requests on '{a.model}'")
    if not keep:
        return
    ws = shutil.which("ws")
    if not ws:
        sys.exit("`ws` not found: uv tool install workingset==0.4.1, "
                 "then run this again for the trace")
    with tempfile.TemporaryDirectory() as tmp:
        for f in keep:
            d = Path(tmp) / f.parent.name
            d.mkdir(exist_ok=True)
            shutil.copy2(f, d / f.name)
        subprocess.run([ws, "trace", "pi", tmp, "-o", a.out + "-trace.json"],
                       check=True)


if __name__ == "__main__":
    main()
