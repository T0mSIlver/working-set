"""The run report — the text users quote.

Ported from `print_report` in scripts/validate_deployment.py. Same tables,
same warnings, same trailer; the PREDICTED vs MEASURED block is now the
hypothesis table, one row per H-*, and it prints the four verdict statuses
instead of three glyphs.

Everything here reads a `RunRecord` and nothing else, so `ws report run.json`
and the tail of `ws test` produce identical output.
"""
from __future__ import annotations

import math
import sys

from .hypotheses.base import GLYPH
from .probe.population import Rung, Sample
from .probe.stats import FREEZE_LADDER_MS, fmt, pct


def print_report(rec, out=None) -> None:
    out = out or sys.stdout
    w = lambda s="": print(s, file=out)          # noqa: E731

    rungs = [Rung.from_dict(r) for r in rec.rungs]
    sample = Sample.from_dict(rec.sample) if rec.sample else None
    opts = rec.options
    pred = rec.predictions
    slo = (rec.config.get("slo") or {})
    p = slo.get("percentile", 95)
    cpt = opts.get("chars_per_token", 4.0)
    thr = opts.get("freeze_threshold_ms", 100.0)

    w()
    w("=" * 78)
    title = "VALIDATION REPORT"
    if rec.interrupted:
        title += "  [INTERRUPTED — partial]"
    w(f"{title}   mode={rec.mode}   workingset {rec.workingset}")
    w("=" * 78)

    if rungs:
        _rung_table(w, rungs, p)
        _gap_table(w, rungs, thr, pred)
        ratios = [r.ptok_ratio for r in rungs if math.isfinite(r.ptok_ratio)]
        if ratios:
            med = pct(ratios, 50)
            w(f"\nachieved/intended prompt tokens (median): {med:.2f} "
              f"(tokens ~ chars/{cpt:g}; if far from 1.0, re-run with "
              f"--chars-per-token {cpt / med:.1f})")
    if sample is not None:
        _sample_block(w, sample, cpt)
    if getattr(rec, "shared", None):
        _shared_block(w, rec.shared)
    if rec.burst:
        _burst_block(w, rec.burst)

    _capacity(w, rec, slo, p)
    _hypothesis_table(w, rec)

    w("\nWHAT THIS RUN DOES NOT ESTABLISH")
    for s in rec.not_established:
        w(f"  - {s}")


# ============================================================================
def _rung_table(w, rungs, p) -> None:
    w(f"\n{'users':>6} {'turns':>6} {'hit':>5} {'miss':>5} {'err':>4} "
      f"{'TTFT hit p50/p' + str(p):>17} {'TTFT miss p50/p' + str(p):>18} "
      f"{'dec p50':>8} {'req/s':>6} {'verdict'}")
    for r in rungs:
        verdict = "PASS" if r.passed else "FAIL: " + "; ".join(r.reasons)
        if r.partial:
            verdict += " [partial — not counted]"
        w(f"{r.pop:>6} {r.n_turns:>6} {r.n_hit:>5} {r.n_miss:>5} {r.n_err:>4} "
          f"{fmt(r.ttft_hit_p50, '', 2):>8}/{fmt(r.ttft_hit_pX, 's', 2):<8} "
          f"{fmt(r.ttft_miss_p50, '', 2):>8}/{fmt(r.ttft_miss_pX, 's', 2):<9} "
          f"{fmt(r.decode_p50, '', 1):>8} {fmt(r.achieved_rps, '', 2):>6} "
          f"{verdict}")


def _gap_table(w, rungs, thr, pred) -> None:
    if not any(math.isfinite(r.itl_p50_ms) for r in rungs):
        return
    w(f"\nINTER-TOKEN GAPS (freeze = a gap >= {thr:g} ms)")
    w(f"{'users':>6} {'normal p50':>11} {'worst/resp p50':>15} "
      f"{'worst/resp p95':>15} {'freezes/ktok':>13} {'stall ms/ktok':>14} "
      f"{'stalled':>8} {'tok/evt':>8} {'floor':>8}")
    for r in rungs:
        stalled = (r.stall_frac * 100 if math.isfinite(r.stall_frac)
                   else float("nan"))
        w(f"{r.pop:>6} {fmt(r.itl_p50_ms, ' ms', 1):>11} "
          f"{fmt(r.itl_worst_p50_ms, ' ms', 0):>15} "
          f"{fmt(r.itl_worst_p95_ms, ' ms', 0):>15} "
          f"{fmt(r.freeze_per_ktok, '', 1):>13} "
          f"{fmt(r.stall_ms_per_ktok, '', 0):>14} "
          f"{fmt(stalled, '%', 1):>8} {fmt(r.chunk_tok_ratio, '', 2):>8} "
          f"{fmt(r.itl_floor_ms, ' ms', 2):>8}")
    w("  worst-seen (max of maxes, sample-size biased — footnote only): "
      + ", ".join(f"{r.pop}u {fmt(r.itl_max_ms, ' ms', 0)}" for r in rungs))

    for r in rungs:
        if math.isfinite(r.chunk_tok_ratio) and abs(r.chunk_tok_ratio - 1.0) > 0.05:
            w(f"  WARNING [{r.pop}u]: {r.chunk_tok_ratio:.2f} tokens per SSE "
              "event. Gap columns are per EVENT: this rung is NOT directly "
              "comparable with a run whose ratio differs. Compare "
              "freezes/ktok and stall ms/ktok (per-token, ratio-free) instead.")
        if (math.isfinite(r.itl_floor_ms) and math.isfinite(r.itl_p50_ms)
                and r.itl_p50_ms > 0 and r.itl_floor_ms > 0.10 * r.itl_p50_ms):
            w(f"  WARNING [{r.pop}u]: client floor {r.itl_floor_ms:.2f} ms is "
              f">10% of the normal gap {r.itl_p50_ms:.2f} ms — the event "
              "loop, not the server, may be setting these gaps. Prefer the "
              "burst probe at the operating point for spike claims.")

    # the headline threshold must sit BELOW the freeze it is meant to catch,
    # or the arm being advocated for reads as 'zero freezes'
    pw = pred.get("itl_worst_freeze_ms")
    if pw and thr > pw:
        w(f"  WARNING: --freeze-threshold-ms {thr:g} exceeds this config's "
          f"PREDICTED worst freeze ({pw:g} ms). freezes/ktok and stalled% "
          "will read ~0 whatever the truth. Use the ladder below, or lower "
          "the threshold.")

    if any(r.freeze_ladder for r in rungs):
        w("\n  FREEZE LADDER — freezes per 1k tokens at each threshold "
          "(no single threshold is load-bearing)")
        head = "".join(f"{t:g} ms".rjust(11) for t in FREEZE_LADDER_MS)
        w(f"  {'users':>6}{head}")
        for r in rungs:
            if not r.freeze_ladder:
                continue
            row = "".join(fmt(e["per_ktok"], '', 2).rjust(11)
                          for e in r.freeze_ladder)
            w(f"  {r.pop:>6}{row}")


def _sample_block(w, s: Sample, cpt) -> None:
    w(f"\nSAMPLE PROBE ({s.n} requests, {s.n_err} failed) — measured at the "
      "endpoint's PREVAILING load, which this run neither set nor observed")
    w(f"  TTFT miss mean {fmt(s.ttft_miss_mean, 's')} | miss p50 "
      f"{fmt(s.ttft_miss_p50, 's')} | hit p50 {fmt(s.ttft_hit_p50, 's')} | "
      f"decode p50 {fmt(s.decode_p50, ' tok/s', 1)}")
    w(f"  gaps: normal p50 {fmt(s.itl_p50_ms, ' ms', 1)} | worst/resp p50 "
      f"{fmt(s.itl_worst_p50_ms, ' ms', 0)} | worst seen "
      f"{fmt(s.itl_worst_max_ms, ' ms', 0)} | client floor "
      f"{fmt(s.itl_floor_ms, ' ms', 2)} | tok/evt "
      f"{fmt(s.chunk_tok_ratio, '', 2)}")
    if math.isfinite(s.ptok_ratio):
        w(f"  achieved/intended prompt tokens (median): {s.ptok_ratio:.2f} "
          f"(tokens ~ chars/{cpt:g})")
    if math.isfinite(s.cached_frac):
        w(f"  server-reported prefix-cache hits on warm turns: "
          f"{s.cached_frac:.0%}")


def _shared_block(w, sh: dict) -> None:
    """The shared-endpoint fit, printed from the RECORD — so `ws report
    run.json` reproduces it byte for byte without re-running anything."""
    g = sh.get("governor") or {}
    op = sh.get("operating_point") or {}
    w(f"\nSHARED-ENDPOINT FIT — the prevailing load as a COVARIATE "
      f"({sh.get('n_covariate_rows', 0)} covariate-stamped requests)")
    if sh.get("aborted"):
        w(f"  ABORTED BY A SAFETY RAIL: {sh['aborted']}")
    w(f"  budget spent: {g.get('n_requests', 0)} requests, "
      f"{g.get('tokens_spent', 0):,} intended prompt tokens | peak waiting "
      f"{fmt(g.get('peak_requests_waiting'), '', 1)} | peak KV "
      f"{_pct(g.get('peak_kv_cache_usage'))} | canary n="
      f"{g.get('n_canary', 0)} p50 {fmt(g.get('canary_p50_s'), 's')}")
    if op.get("refused"):
        w(f"  operating point: NOT AVAILABLE — {op['refused']}")
    else:
        w(f"  operating point: running {fmt(op.get('running'))} "
          f"(= {fmt(op.get('steady_decode_seqs'))} decoding + "
          f"{fmt(op.get('prefill_occupancy'))} prefilling) | waiting "
          f"{fmt(op.get('waiting'))} | E[L] {fmt(op.get('L_ktok'), 'k tok', 1)}"
          f" | E[L^2] {fmt(op.get('L2_ktok2'), '', 1)} (ktok^2)")
    w(f"  a verdict needs an extrapolation distance <= "
      f"{sh.get('max_extrapolation', 1.0):g} observed sd")
    for name, f in (sh.get("fits") or {}).items():
        if f.get("refused"):
            w(f"  {name:<10} no fit — {f['refused']}")
            continue
        terms = " ".join(f"{c}={f['coefficients'][c]:+.4g}"
                         for c in f["columns"])
        w(f"  {name:<10} [{f['unit']}] {terms}")
        w(f"  {'':<10} n={f['n']} dof={f['dof']} residual sd "
          f"{fmt(f.get('residual_std'), '', 4)} R2 {fmt(f.get('r_squared'))} "
          f"cond {fmt(f.get('condition_number'), '', 1)}")
        r = (sh.get("readings") or {}).get(name) or {}
        if r.get("available"):
            w(f"  {'':<10} -> {fmt(r.get('value'), ' ' + f['unit'], 4)} at the "
              f"operating point (se {fmt(r.get('se'), '', 4)}, extrapolation "
              f"{fmt(r.get('extrapolation'))} sd) — SCORED")
        else:
            w(f"  {'':<10} -> not scored: {r.get('reason', 'no reading')}")
    lad = sh.get("natural_ladder") or []
    if lad:
        modelled = any(b.get("model") for b in lad)
        w("\n  NATURAL LADDER — binned by the `running` the server happened "
          "to be carrying (this run set none of it)")
        head = (f"  {'running':>14} {'n':>5} {'TTFT miss p50':>14} "
                f"{'TTFT hit p50':>13} {'ITL p50':>10} {'decode p50':>12}")
        if modelled:
            head += (f" | {'model TTFT':>11} {'model ITL':>10} "
                     f"{'model decode':>13} {'implied req/s':>14}")
        w(head)
        for b in lad:
            hi = b.get("running_hi")
            span = ("[" + format(b["running_lo"], "g") + ", "
                    + ("inf" if hi is None else format(hi, "g")) + ")")
            row = (f"  {span:>14} {b['n']:>5} "
                   f"{fmt(b.get('ttft_miss_p50_s'), 's'):>14} "
                   f"{fmt(b.get('ttft_hit_p50_s'), 's'):>13} "
                   f"{fmt(b.get('itl_p50_ms'), ' ms', 1):>10} "
                   f"{fmt(b.get('decode_p50_tok_s'), ' tok/s', 0):>12}")
            m = b.get("model") or {}
            if modelled:
                row += (f" | {fmt(m.get('ttft_miss_s'), 's'):>11} "
                        f"{fmt(m.get('itl_ms'), ' ms', 1):>10} "
                        f"{fmt(m.get('decode_tok_s'), ' tok/s', 0):>13} "
                        f"{fmt(m.get('rate_req_s'), '', 2):>14}")
            w(row)
        if modelled:
            w("    model columns: the decode curve read at this bin's batch, "
              "and the mean cold TTFT at the arrival rate that would PRODUCE "
              "that batch (bisected on the model's own steady decode point). "
              "The bins are observations, not a load the run set.")
    _cross_block(w, sh.get("cross_check"))


def _cross_block(w, c: dict | None) -> None:
    """Client minus server, on the same quantile — the proxy tax, and whether
    the forced misses actually missed."""
    if not c:
        return
    if c.get("error"):
        w(f"\n  SERVER CROSS-CHECK unavailable: {c['error']}")
        return
    w("\n  SERVER CROSS-CHECK over the probe window (client - server = the "
      "proxy, the network and our own event loop)")
    w(f"    TTFT p50: client {fmt(c.get('client_ttft_p50_s'), 's')} - server "
      f"{fmt(c.get('server_ttft_p50_s'), 's')} = "
      f"{fmt(c.get('proxy_overhead_ttft_p50_s'), 's')}  "
      f"(server n={c.get('server_ttft_n', 0)})")
    w(f"    TTFT p95: client {fmt(c.get('client_ttft_p95_s'), 's')} - server "
      f"{fmt(c.get('server_ttft_p95_s'), 's')} = "
      f"{fmt(c.get('proxy_overhead_ttft_p95_s'), 's')}")
    w(f"    ITL  p50: client {fmt(c.get('client_itl_p50_ms'), ' ms', 1)} - "
      f"server {fmt(c.get('server_itl_p50_ms'), ' ms', 1)} = "
      f"{fmt(c.get('proxy_overhead_itl_p50_ms'), ' ms', 1)}")
    w(f"    forced misses confirmed cold: "
      f"{_pct(c.get('forced_miss_clean_frac'))} of "
      f"{c.get('n_miss_with_cached_readback', 0)} with a cached_tokens "
      f"readback (median cached "
      f"{fmt(c.get('forced_miss_cached_tokens_p50'), '', 0)} tok); window "
      f"prefix hit rate {_pct(c.get('prefix_hit_rate'))} — over ALL traffic, "
      "ours and theirs, so it cannot attribute a hit to anybody")


def _pct(x) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "-"
    return f"{x:.0%}"


def _burst_block(w, b: dict) -> None:
    w(f"\nBURST PROBE: N={b['n']} simultaneous forced misses at a "
      f"{b['standing_users']}-user standing load")
    w(f"  {b['n_ok']}/{b['n']} answered | TTFT p50 "
      f"{fmt(b.get('ttft_p50_s'), 's')} | last first-token "
      f"{fmt(b.get('last_ttft_s'), 's')} | drain {fmt(b.get('drain_s'), 's')}")
    if b.get("standing_n"):
        w(f"  standing load hit by it: {b['standing_n']} responses in flight "
          f"| normal gap {fmt(b.get('standing_itl_p50_ms'), ' ms', 1)} | "
          f"worst gap p50 {fmt(b.get('standing_worst_p50_ms'), ' ms', 0)} | "
          f"worst seen {fmt(b.get('standing_worst_max_ms'), ' ms', 0)} | "
          f"client floor {fmt(b.get('standing_floor_ms'), ' ms', 2)}")
        lad = b.get("standing_freeze_ladder")
        if lad:
            w("  freezes per 1k tokens by threshold: "
              + " ".join(f"{e['threshold_ms']:g}ms:{e['per_ktok']:.2f}"
                         for e in lad))


def _capacity(w, rec, slo, p) -> None:
    lo, hi = (rec.measured_capacity_bracket + [None, None])[:2]
    w("\nMEASURED SLO CAPACITY "
      f"(p{p} TTFT <= {slo.get('ttft_budget_s', 10):g}s AND per-user p50 "
      f"decode >= {slo.get('itl_floor_tok_s', 40):g} tok/s):")
    if lo is not None and hi is not None:
        w(f"  in ({lo}, {hi}] — largest passing population {lo} users")
    elif lo is not None:
        w(f"  >= {lo} users (no failing rung observed — a lower bound)")
    elif hi is not None:
        w(f"  < {hi} users (no passing rung observed)")
    else:
        w("  not measured")


def _hypothesis_table(w, rec) -> None:
    w("\nPREDICTED vs MEASURED")
    w(f"{'hypothesis':<14} {'predicted':>22} {'measured':>24}  verdict")
    for h in rec.hypotheses:
        v = h["verdict"]
        w(f"{h['key']:<14} {h['prediction']['text']:>22} "
          f"{h['measurement']['text']:>24}  {GLYPH[v['status']]} "
          f"{v['status']}: {v['text']}")
    for s in rec.skipped:
        w(f"{s['key']:<14} {'-':>22} {'skipped':>24}  · {s['reason']}")
    w("\n  ✓ supported   ✗ refuted   ≥ bounded below (a lower bound only)"
      "   ~ not established")
