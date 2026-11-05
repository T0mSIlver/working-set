"""Mixed steps for `ws decode-sweep`: forward passes that carry prefill tokens
next to the held decode batch.

The held streams each log (time, running token count) per event. A cold
request's life is an interval [send, first token]. Every step of that interval
carries some of its prefill, so a stream gap is

  mixed      starts at or after the request was sent and ends no later than its
             first token (+ `tol`): the whole step ran with the request in the
             batch;
  pure       clear of every interval: a plain decode step;
  ambiguous  everything else, chiefly the step that was already running when
             the request arrived. Dropped from both sides.

A one-chunk request therefore yields one mixed gap per stream, the gap that
closes at its first token (`last`). Gaps are client-side, so `tol` absorbs the
network jitter between a step's events.
"""
from __future__ import annotations

import statistics

DEFAULT_TOL_S = 0.02


def stream_gaps(events: list[tuple[float, int]], t0: float, t1: float,
                have_tok: bool = True) -> list[tuple[float, float, int]]:
    """(start, end, tokens) for consecutive events lying inside [t0, t1]."""
    out = []
    for (ta, ka), (tb, kb) in zip(events, events[1:]):
        if ta >= t0 and tb <= t1 and tb > ta:
            out.append((ta, tb, max(kb - ka, 0) if have_tok else 1))
    return out


def classify(gaps: list[tuple[float, float, int]],
             intervals: list[tuple[float, float]], tol: float = DEFAULT_TOL_S) -> dict:
    """Split one stream's gaps by the prefill intervals (send, first token)."""
    mixed, pure, amb = [], [], 0
    last_end: dict[int, tuple] = {}
    for g in gaps:
        s, e, _ = g
        hit = None
        for j, (snd, first) in enumerate(intervals):
            if s >= snd and e <= first + tol:
                hit = j
                break
        if hit is not None:
            mixed.append(g)
            if e >= intervals[hit][1] - tol and (hit not in last_end or e > last_end[hit][1]):
                last_end[hit] = g
        elif all(e < snd or s >= first + tol for snd, first in intervals):
            pure.append(g)
        else:
            amb += 1
    return {"mixed": mixed, "pure": pure, "ambiguous": amb,
            "last": [last_end[j] for j in sorted(last_end)]}


def _pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[int(q * (len(xs) - 1))]


def _rate(gs) -> float | None:
    d = sum(e - s for s, e, _ in gs)
    return sum(k for _, _, k in gs) / d if d > 0 else None


def summarize(per_stream: list[dict]) -> dict:
    """Step times and per-sequence speed from the classified gaps of every stream.

    tok_s_seq*: tokens over time across a stream's mixed gaps (delivered tokens,
    so accepted ones under speculation); the mean and the slowest stream.
    tok_s_worst_step: mean tokens per mixed gap over the 99th-percentile mixed gap.
    """
    mixed = [g for p in per_stream for g in p["mixed"]]
    pure = [g for p in per_stream for g in p["pure"]]
    last = [g for p in per_stream for g in p["last"]]
    amb = sum(p["ambiguous"] for p in per_stream)
    total = len(mixed) + len(pure) + amb
    out = {"streams": len(per_stream), "mixed_gaps": len(mixed), "pure_gaps": len(pure),
           "ambiguous_gaps": amb,
           "share_mixed": len(mixed) / total if total else None,
           "mixed_step_s": None, "mixed_step_p99_s": None, "mixed_step_max_s": None,
           "last_step_s": None, "pure_step_s": None,
           "tok_s_seq_mean": None, "tok_s_seq_min": None, "tok_s_worst_step": None}
    if mixed:
        d = [e - s for s, e, _ in mixed]
        out["mixed_step_s"] = statistics.fmean(d)
        out["mixed_step_p99_s"] = _pct(d, 0.99)
        out["mixed_step_max_s"] = max(d)
        rates = [r for r in (_rate(p["mixed"]) for p in per_stream if p["mixed"]) if r]
        if rates:
            out["tok_s_seq_mean"] = statistics.fmean(rates)
            out["tok_s_seq_min"] = min(rates)
        tpg = statistics.fmean(k for _, _, k in mixed)
        out["tok_s_worst_step"] = tpg / out["mixed_step_p99_s"]
    if last:
        out["last_step_s"] = statistics.fmean(e - s for s, e, _ in last)
    if pure:
        out["pure_step_s"] = statistics.fmean(e - s for s, e, _ in pure)
    return out


def depth_profile(per_stream: list[dict], depth: int,
                  chunk: int | None = None) -> list[dict]:
    """Mixed step time against chunk position for ONE long cold request.

    Step j of each stream is the same forward pass, so the profile is the
    median over streams of their j-th mixed gap. With `chunk` unknown the
    chunk is depth over the number of mixed steps (an estimate of the server's
    budget). The depth of step j is the prompt tokens done at its end.
    """
    cols = [[e - s for s, e, _ in p["mixed"]] for p in per_stream if p["mixed"]]
    if not cols:
        return []
    m = int(statistics.median(len(c) for c in cols))
    cols = [c for c in cols if len(c) >= m]
    step = chunk or depth / m
    return [{"depth": min(int(round((j + 1) * step)), depth),
             "step_s": statistics.median(c[j] for c in cols)} for j in range(m)]


def judge_mixed(n: int, running: list[float], errors: int, drained: int,
                requests: int) -> tuple[bool, list[str]]:
    """The batch was still n sequences plus the prefill request, nothing failed."""
    why = []
    if running and min(running) < 0.95 * n:
        why.append(f"server ran only {min(running):.0f} of {n} sequences at its low point")
    if not running:
        why.append("no running-sequence gauge was read")
    if errors:
        why.append(f"{errors} streams failed")
    if drained:
        why.append(f"{drained} streams finished inside the window")
    if not requests:
        why.append("no prefill request completed")
    return not why, why


def window_row(rung: dict, w: dict, min_tok_s: float) -> dict:
    """One summary row for a mixed window; `flags` names the floors it misses."""
    m = w["mixed"]
    load = (f"P={w['prefill_tokens']}" if w["kind"] == "tokens" else f"D={w['prefill_depth']}")
    prof = w.get("depth_profile")
    worst = max(p["step_s"] for p in prof) if prof else m.get("mixed_step_p99_s")
    flags = []
    if m.get("tok_s_seq_mean") is not None and m["tok_s_seq_mean"] < min_tok_s:
        flags.append(f"mean < {min_tok_s:g} tok/s")
    if m.get("tok_s_worst_step") is not None and m["tok_s_worst_step"] < min_tok_s:
        flags.append(f"worst step < {min_tok_s:g} tok/s")
    return {"n": rung["n"],
            "context": rung.get("prompt_tokens_mean") or rung.get("context_target"),
            "load": load, "pure_step_s": rung.get("pure_step_s"),
            "mixed_step_s": m.get("mixed_step_s"), "worst_step_s": worst,
            "tok_s_seq": m.get("tok_s_seq_mean"), "tok_s_worst_step": m.get("tok_s_worst_step"),
            "share_mixed": m.get("share_mixed"), "ok": w.get("ok", True), "flags": flags}


def summary_rows(result: dict, min_tok_s: float = 50.0) -> list[dict]:
    return [window_row(r, w, min_tok_s) for r in result.get("rungs", [])
            for w in r.get("mixed_windows", [])]


SUMMARY_HEADER = (f"{'n':>5} {'ctx tok':>9} {'load':>10} {'pure ms':>8} {'mixed ms':>9} "
                  f"{'worst ms':>9} {'tok/s/seq':>9} {'worst t/s':>9} {'mixed%':>6}")


def format_row(r: dict) -> str:
    def f(x, spec, k=1.0):
        return format(x * k, spec) if x is not None else "-"
    s = (f"{r['n']:>5} {f(r['context'], ',.0f'):>9} {r['load']:>10} "
         f"{f(r['pure_step_s'], '.1f', 1e3):>8} {f(r['mixed_step_s'], '.1f', 1e3):>9} "
         f"{f(r['worst_step_s'], '.1f', 1e3):>9} {f(r['tok_s_seq'], '.1f'):>9} "
         f"{f(r['tok_s_worst_step'], '.1f'):>9} {f(r['share_mixed'], '.0f', 100):>6}")
    if r["flags"]:
        s += "  BELOW: " + "; ".join(r["flags"])
    if not r["ok"]:
        s += "  NOT OK"
    return s
