"""Mixed steps for `ws decode-sweep`: forward passes that carry prefill tokens
next to the held decode batch.

The held streams each log (time, running token count) per event. A cold
request's life is an interval [send, first token]. Tokenization, HTTP and
scheduling latency put pure decode steps inside that interval too, so the
interval is not the measurement. The server fixes how many steps carry the
prefill: k = ceil(P / (chunk - n)), because each step spends n tokens on the
decodes and the rest of the budget on the chunk. Per stream and interval, the
mixed steps are the k longest gaps that start after the send and end by the
first token (+ `tol`); the other gaps in the interval are pure.

Headlines are medians over intervals (and the 90th percentile), not means over
gaps, so they do not depend on how the client paces its requests.
"""
from __future__ import annotations

import math
import statistics

DEFAULT_TOL_S = 0.02
CHECK_TOL = 0.30


def chunk_steps(prefill: int, chunk: int | None, n: int) -> int | None:
    """Steps a P-token prefill spans when each step has `chunk` tokens of budget
    and n of them go to the decodes; None when the chunk is unknown."""
    if not chunk:
        return None
    return max(1, math.ceil(prefill / max(chunk - n, 1)))


def stream_gaps(events: list[tuple[float, int]], t0: float, t1: float,
                have_tok: bool = True) -> list[tuple[float, float, int]]:
    """(start, end, tokens) for consecutive events lying inside [t0, t1]."""
    out = []
    for (ta, ka), (tb, kb) in zip(events, events[1:]):
        if ta >= t0 and tb <= t1 and tb > ta:
            out.append((ta, tb, max(kb - ka, 0) if have_tok else 1))
    return out


def classify(gaps: list[tuple[float, float, int]],
             intervals: list[tuple[float, float]], k: int | None = None,
             tol: float = DEFAULT_TOL_S) -> dict:
    """Split one stream's gaps by the prefill intervals (send, first token).

    `by_interval[j]` holds interval j's k longest candidate gaps (all of them
    when k is None); the rest of the candidates join `pure` with the gaps clear
    of every interval. A gap straddling a send or a first token is ambiguous.
    """
    cand: dict[int, list] = {}
    pure, amb = [], 0
    for g in gaps:
        s, e, _ = g
        hit = next((j for j, (snd, first) in enumerate(intervals)
                    if s >= snd and e <= first + tol), None)
        if hit is not None:
            cand.setdefault(hit, []).append(g)
        elif all(e < snd or s >= first + tol for snd, first in intervals):
            pure.append(g)
        else:
            amb += 1
    by = {}
    for j, gs in cand.items():
        if k is None or len(gs) <= k:
            by[j] = gs
        else:
            keep = sorted(gs, key=lambda g: g[1] - g[0], reverse=True)[:k]
            by[j] = sorted(keep)
            pure += [g for g in gs if g not in keep]
    return {"by_interval": by, "mixed": [g for j in sorted(by) for g in by[j]],
            "pure": pure, "ambiguous": amb}


def _pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[int(q * (len(xs) - 1))]


def summarize(per_stream: list[dict]) -> dict:
    """The mixed step of a window: per interval, the median across streams of
    each stream's mean mixed gap; then the median and p90 over intervals.

    tok_s_seq is tokens per mixed step (delivered tokens, so accepted ones under
    speculation) over the median mixed step: the speed of a sequence on a fully
    mixed step. tok_s_p90_step uses the p90 step.
    """
    mixed = [g for p in per_stream for g in p["mixed"]]
    pure = [g for p in per_stream for g in p["pure"]]
    out = {"streams": len(per_stream), "intervals": 0, "mixed_gaps": len(mixed),
           "pure_gaps": len(pure), "ambiguous_gaps": sum(p["ambiguous"] for p in per_stream),
           "mixed_step_s": None, "mixed_step_p90_s": None, "tokens_per_step": None,
           "tok_s_seq": None, "tok_s_p90_step": None, "pure_in_window_s": None}
    if pure:
        out["pure_in_window_s"] = statistics.fmean(e - s for s, e, _ in pure)
    js = sorted({j for p in per_stream for j in p["by_interval"]})
    per_iv = []
    for j in js:
        vals = [statistics.fmean(e - s for s, e, _ in p["by_interval"][j])
                for p in per_stream if p["by_interval"].get(j)]
        if vals:
            per_iv.append(statistics.median(vals))
    if not per_iv:
        return out
    out["intervals"] = len(per_iv)
    out["mixed_step_s"] = statistics.median(per_iv)
    out["mixed_step_p90_s"] = _pct(per_iv, 0.9)
    out["tokens_per_step"] = statistics.fmean(k for _, _, k in mixed)
    out["tok_s_seq"] = out["tokens_per_step"] / out["mixed_step_s"]
    out["tok_s_p90_step"] = out["tokens_per_step"] / out["mixed_step_p90_s"]
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


def self_check(mixed_s: float | None, pure_s: float | None, alone_s: float | None,
               steps: int | None) -> dict:
    """Expected mixed step = pure step + the prefill's time alone over the steps
    it spans. `ok` is False when the measurement is more than CHECK_TOL off."""
    if not (mixed_s and pure_s and alone_s):
        return {"expected_s": None, "ratio": None, "ok": None}
    exp = pure_s + alone_s / (steps or 1)
    return {"expected_s": exp, "ratio": mixed_s / exp,
            "ok": abs(mixed_s / exp - 1) <= CHECK_TOL}


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
    p90 = m.get("mixed_step_p90_s")
    step = m.get("mixed_step_s")
    if prof:
        ss = [p["step_s"] for p in prof]
        step, p90 = statistics.median(ss), _pct(ss, 0.9)
        tps = m.get("tokens_per_step")
        m = dict(m, tok_s_seq=tps / step if tps else None,
                 tok_s_p90_step=tps / p90 if tps else None)
    flags = []
    if m.get("tok_s_seq") is not None and m["tok_s_seq"] < min_tok_s:
        flags.append(f"< {min_tok_s:g} tok/s")
    if m.get("tok_s_p90_step") is not None and m["tok_s_p90_step"] < min_tok_s:
        flags.append(f"p90 step < {min_tok_s:g} tok/s")
    chk = w.get("check") or {}
    if chk.get("ok") is False:
        flags.append(f"self-check: mixed {chk['ratio']:.2f}x the expected step")
    return {"n": rung["n"],
            "context": rung.get("prompt_tokens_mean") or rung.get("context_target"),
            "load": load, "pure_step_s": rung.get("pure_step_s"),
            "mixed_step_s": step, "p90_step_s": p90,
            "tok_s_seq": m.get("tok_s_seq"), "tok_s_p90_step": m.get("tok_s_p90_step"),
            "alone_s": w.get("alone_s"), "check_ratio": chk.get("ratio"),
            "ok": w.get("ok", True), "flags": flags}


def summary_rows(result: dict, min_tok_s: float = 50.0) -> list[dict]:
    return [window_row(r, w, min_tok_s) for r in result.get("rungs", [])
            for w in r.get("mixed_windows", [])]


SUMMARY_HEADER = (f"{'n':>5} {'ctx tok':>9} {'load':>10} {'pure ms':>8} {'mixed ms':>9} "
                  f"{'p90 ms':>8} {'tok/s/seq':>9} {'p90 t/s':>8} {'alone ms':>8} {'chk':>5}")


def format_row(r: dict) -> str:
    def f(x, spec, k=1.0):
        return format(x * k, spec) if x is not None else "-"
    s = (f"{r['n']:>5} {f(r['context'], ',.0f'):>9} {r['load']:>10} "
         f"{f(r['pure_step_s'], '.1f', 1e3):>8} {f(r['mixed_step_s'], '.1f', 1e3):>9} "
         f"{f(r['p90_step_s'], '.1f', 1e3):>8} {f(r['tok_s_seq'], '.1f'):>9} "
         f"{f(r['tok_s_p90_step'], '.1f'):>8} {f(r['alone_s'], '.1f', 1e3):>8} "
         f"{f(r['check_ratio'], '.2f'):>5}")
    if r["flags"]:
        s += "  FLAG: " + "; ".join(r["flags"])
    if not r["ok"]:
        s += "  NOT OK"
    return s
