#!/usr/bin/env python3
"""Golden vectors: Python computes, the JS mirror must agree.

`workingset.model` is the source of truth; `interactive/src/*.js` is a
hand-maintained JS mirror of it. Nothing enforced that. This script samples a
deterministic set of explorer states across every axis the page exposes,
prices each one with the Python model, and writes the inputs + outputs to
`tests/golden/vectors.json`. `tests/js/golden.test.mjs` replays the same states
through the JS modules and asserts agreement.

    uv run scripts/golden.py            # regenerate tests/golden/vectors.json
    uv run scripts/golden.py --check    # fail if the committed file is stale

The file is byte-reproducible: every draw is seeded, every float is rounded to
12 significant digits, and the JSON is written with fixed separators.

WHAT THE STATES CARRY. Each vector's `state` block is exactly the explorer's
`state` object fields that move a number (interactive/src/state.js). The JS
test assigns them onto `state` and calls `activeModel()` / `currentTopo()` /
`currentWL()`, i.e. it enters the mirror through the same door the page does.

CONVENTIONS the vectors are computed under, because the explorer uses them and
the Python defaults do not:

  per_pass_overhead=True   the explorer has priced misses with the per-pass
                           weight-stream overhead since 2026-08-02; the
                           published Python tables keep the flat-MFU default.
  rate_group               every queue/power figure is priced at the TOTAL
                           per-replica-group arrival rate,
                           users x (1 + sub_ratio) / think / replicas.
  per group                latency/saturation ceilings are per replica group
                           (Python's convention). The explorer's
                           operatingPoint() multiplies both by `replicas`
                           before it labels the binding constraint; that is a
                           documented convention difference, not a
                           disagreement, so the vectors compare the per-group
                           closed forms both sides actually compute.

TOLERANCE CLASSES.
  exact  closed-form, no sampling anywhere in the chain. rel 1e-6.
  mc     Monte-Carlo on both sides, with DIFFERENT samplers (numpy PCG64 +
         numpy lognormal at n = 200,000 in Python; mulberry32 + Box-Muller at
         n = 20,000 in the explorer). The band per quantity is derived from
         the seed-to-seed spread measured below and stored in `mc_spread`.
"""
from __future__ import annotations

import argparse
import contextlib
import functools
import json
import math
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

from workingset import model as M

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "golden" / "vectors.json"

SCHEMA = 1

# ---------------------------------------------------------------------------
# Monte-Carlo sizes. The vectors are generated at the PYTHON model's own
# defaults wherever it has one; only the two whose defaults are too slow to
# regenerate a few hundred states with are cut down, and both are recorded here
# so the tolerance bands can be read against them.
# ---------------------------------------------------------------------------
WARM_ITER = 200          # model default 400
DECODE_ITER = 200        # model default 400
CTX_N = 200_000          # model default for context_moments / service moments

# The two DIAGNOSTIC-only samples (the `cond` block's mirror-axis figures) run
# far lighter than the compared quantities: they size an axis and a ratio that
# gate allowlist entries, and a few percent of noise on either moves no
# verdict. Keeping them at the full count roughly doubled regeneration time.
COND_WARM_ITER = 40
COND_DECODE_ITER = 60

# The mirror's own context-draw size. contextStats() defaults to 20,000 samples
# and every prefill readout on the page shares that one draw, against the
# 200,000 workingset.model takes — a 10x asymmetry, and the only one that
# matters: the explorer's warm fill (CONFIG.WARM_ITER = 700, capped by a
# 600,000-draw budget, so a few hundred fills in practice) and its decode
# sweeps run at counts comparable to the ones the vectors are generated at.
# The comparison's noise floor is set by whichever side samples less, so the
# bands are derived with the context draw cut to the mirror's size.
MIRROR_CTX_N = 20_000

GPU_KEYS = ("H200", "B300")
# every GPU count the explorer's slider reaches (min 1, max 8, step 1) — NOT
# just the powers of two. The odd counts are where tp_efficiency's
# 0.90^log2(tp) is evaluated off a power of two and where the DP/TP grid has
# splits that no power-of-two count produces (DP3xTP2 on six GPUs).
NGPUS = tuple(range(1, 9))
CHUNKS = (2048, 4096, 8192, 16384, 32768, 65536)

# state.js defaults, verbatim — the published reference configuration
DEFAULT_STATE = {
    "model": "27B", "gpu": "H200", "wdt": "fp8", "ngpu": 1, "tp": 1, "ram": 0,
    "kv": "fp8", "cap": 180, "mtp": 2.94,
    "mbu": M.MBU_DEFAULT, "mfu": M.MFU_DEFAULT, "chunk": "32768",
    "user_median": 31, "user_sigma": 0.81, "sub_median": 8, "sub_sigma": 0.90,
    "sub_ratio": 0.10, "sub_shares_prefix": False, "sys": 15, "inval": 1.0,
    "users": 64, "think": 30, "sla": 10, "turn": 2000, "burst": 32,
    "out": M.AVG_OUT_TOK, "decode_floor": M.DECODE_FLOOR_TOKS,
    "ekwh": 0.19, "pue": "1.5", "gpuh": M.GPUS["H200"].eur_gpu_h,
}

# The reference burst prefill.js's spikeMetrics() hard-wires as SPIKE_BURST.
# state.burst reaches the tile only through inline arithmetic in render.js, so
# the comparable drain is the one at this value on both sides; see the
# `burst_drain_seconds_b32` mapping row.
SPIKE_BURST = 32.0

# The prefill lengths every state is priced at: a chunk-sized pass, a short
# pass well under any chunk, and a long one at the reference cap.
PREFILL_LENGTHS = (2048, 32768, 180000)
CONTEXT_LEN = 120_000     # the fixed context prefill_context_seconds is timed on


# ---------------------------------------------------------------------------
# state -> model objects.  Mirrors state.js currentWL()/currentTopo() and
# render.js modelFor(), which is how the explorer turns its state into the
# arguments workingset.model takes.
# ---------------------------------------------------------------------------
def divisors(n: int):
    return [d for d in range(1, n + 1) if n % d == 0]


def cap_slider_max(model_key: str) -> int:
    return round(M.MODELS[model_key].max_ctx / 1000)


def state_model(st: dict) -> M.Model:
    m = M.MODELS[st["model"]]
    if st["wdt"] != "fp8":
        m = M.with_weight_dtype(m, st["wdt"])
    m = M.with_kv_dtype(m, st["kv"])
    if st["mtp"] is not None:
        m = replace(m, mtp=float(st["mtp"]))
    return m


def state_topo(st: dict) -> M.Topology:
    return M.topology_grid(st["ngpu"] // st["tp"], st["tp"], st["gpu"])


def state_wl(st: dict) -> M.Workload:
    cap = (M.MODELS[st["model"]].max_ctx if st["cap"] >= cap_slider_max(st["model"])
           else st["cap"] * 1000)
    return M.Workload(
        user_median=st["user_median"] * 1000, user_sigma=st["user_sigma"],
        sub_median=st["sub_median"] * 1000, sub_sigma=st["sub_sigma"],
        sub_ratio=st["sub_ratio"], sys_user=st["sys"] * 1000, sys_sub=3000,
        sub_shares_prefix=st["sub_shares_prefix"],
        invalidation=st["inval"] / 100.0, cap=cap, min_tokens=1000)


def ram_per_cache(st: dict, topo: M.Topology) -> float:
    return st["ram"] / topo.replicas if topo.replicas > 1 else st["ram"]


def rate_group(st: dict, wl: M.Workload, topo: M.Topology) -> float:
    """serverRate(users, think, sub_ratio) / replicas — the TOTAL req/s one
    replica group sees, which is the unit every queue and power figure on the
    page is priced in (render.js renderPlanner).

    model.request_rate is the conversion; only the /replicas split is the
    explorer's, so only that is written here."""
    return M.request_rate(st["users"], st["think"], wl.sub_ratio) / topo.replicas


# ---------------------------------------------------------------------------
# the quantities
# ---------------------------------------------------------------------------
def compute(st: dict, seed: int = 0) -> tuple[dict, dict]:
    """(out, cond) for one state.

    `out` is the quantities the JS mirror must reproduce — every key of it MUST
    have a counterpart in tests/js/drive.mjs, which the test asserts.
    `cond` is conditioning diagnostics: numbers that say where an estimator is
    ill-behaved, so a known_disagreements.json entry can be gated on the CAUSE
    rather than on a model name. They are never compared.
    """
    m, topo, wl = state_model(st), state_topo(st), state_wl(st)
    chunk = float(st["chunk"])
    mfu, turn = st["mfu"], float(st["turn"])
    ram = ram_per_cache(st, topo)
    rate = rate_group(st, wl, topo)
    o: dict[str, float] = {}

    # ---- exact: no sampling anywhere in the chain -------------------------
    o["kv_pool_tokens"] = M.kv_pool_tokens(m, topo)
    o["effective_bw"] = M.effective_bw(topo)
    o["peak_flops"] = M.peak_flops(topo)
    o["tp_efficiency"] = M.tp_efficiency(topo.tp, topo.gpu.nvlink_domain)
    o["w_decode_n64"] = m.w_decode(64)
    o["prefill_overhead_seconds"] = M.prefill_overhead_seconds(m, topo)
    o["mfu_ceiling"] = M.mfu_ceiling(m, topo, mfu_anchor=mfu)
    o["mfu_effective_at_chunk"] = M.mfu_effective(m, topo, chunk, mfu_anchor=mfu)
    for L in PREFILL_LENGTHS:
        o[f"prefill_flops_{L}"] = M.prefill_flops(m, L)[2]
        o[f"prefill_seconds_{L}"] = M.prefill_seconds(m, topo, L, mfu)
    o["prefill_context_seconds_120k"] = M.prefill_context_seconds(
        m, topo, CONTEXT_LEN, chunk, mfu)
    o["miss_context_seconds_120k"] = M.miss_context_seconds(
        m, topo, CONTEXT_LEN, chunk, mfu_anchor=mfu)
    # a warm hit's own pass, priced with an EMPTY cache: the sampled version
    # (prior = E[L]) is warm_request_seconds below
    o["warm_pass_seconds_turn"] = M.prefill_context_seconds(
        m, topo, turn, chunk, mfu, prior=0.0)

    # ---- Monte-Carlo: context-length distribution -------------------------
    el, el2 = M.context_moments(wl)
    o["ctx_mean"], o["ctx_mean_sq"] = el, el2
    o["mean_passes"] = M.mean_passes(wl, chunk)
    o["cold_request_seconds"] = M.cold_request_seconds(
        m, topo, wl, chunk, mfu, per_pass_overhead=True)
    o["warm_request_seconds"] = M.warm_request_seconds(
        m, topo, turn, chunk, mfu, prior=el)

    # the four PER-CLASS moments prefillServiceMoments returns, which is what
    # every queue figure downstream is built from. prefill_service_moments
    # returns the f-MIXED pair plus the two class means, so the two class
    # SECOND moments come from the arrays it is itself built on.
    cold_arr, warm_arr = M._prefill_service_arrays(
        m, topo, wl, chunk, turn, mfu, per_pass_overhead=True)
    o["moments_miss"] = float(cold_arr.mean())
    o["moments_miss_sq"] = float((cold_arr ** 2).mean())
    o["moments_hit"] = float(warm_arr.mean())
    o["moments_hit_sq"] = float((warm_arr ** 2).mean())

    o["prefill_duty"] = M.prefill_duty(m, topo, wl, rate, chunk, turn, mfu,
                                       per_pass_overhead=True)
    o["queue_wait_seconds"] = M.queue_wait_seconds(
        m, topo, wl, rate, chunk, turn, mfu, per_pass_overhead=True)
    for req, disc, key in (("cold", "fcfs", "ttft_miss_fcfs"),
                           ("warm", "fcfs", "ttft_hit_fcfs"),
                           ("warm", "ps", "ttft_hit_ps")):
        o[key] = M.prefill_ttft_seconds(m, topo, wl, rate, chunk, turn, mfu,
                                        request=req, discipline=disc,
                                        per_pass_overhead=True)
    o["breakeven_miss_rate"] = M.breakeven_miss_rate(
        m, topo, wl, rate, chunk, turn, mfu, per_pass_overhead=True)
    o["spike_tolerance"] = M.spike_tolerance(
        m, topo, wl, st["sla"], rate, chunk, turn, mfu, per_pass_overhead=True)
    # ...and at the hard-wired 10 s budget prefill.js's spikeMetrics() uses,
    # so the test can see the two JS entry points apart
    o["spike_tolerance_sla10"] = M.spike_tolerance(
        m, topo, wl, 10.0, rate, chunk, turn, mfu, per_pass_overhead=True)
    o["max_users_latency"] = M.max_users_latency(
        m, topo, wl, chunk, st["sla"], turn, st["think"], mfu, "fcfs",
        per_pass_overhead=True)
    o["max_users_saturation"] = M.max_users_saturation(
        m, topo, wl, chunk, turn, st["think"], mfu, per_pass_overhead=True)
    # at 10 s, not state.sla: spikeMetrics computes fsla against the same
    # hard-wired SPIKE_SLA_S its bstar uses, so 10 is the only budget at which
    # the two sides are answering the same question
    o["sla_miss_rate_sla10"] = M.sla_miss_rate(
        m, topo, wl, rate, chunk, 10.0, turn, mfu, discipline="fcfs",
        request="cold", per_pass_overhead=True)
    # spikeMetrics() hard-wires SPIKE_BURST = 32, so 32 is the burst the two
    # sides can be compared at; state.burst reaches the readout only through
    # inline render.js arithmetic (see the README's coverage limitations)
    o["burst_drain_seconds_b32"] = M.burst_drain_seconds(
        m, topo, wl, SPIKE_BURST, rate, chunk, turn, mfu,
        per_pass_overhead=True)

    # ---- Monte-Carlo: warm fill and decode curves -------------------------
    draw = M.warm_draw(m, topo)
    o["max_users_cache"] = _warm_user(m, topo, wl, ram, draw, seed)
    o["warm_p5_all"] = _warm_all(m, topo, wl, ram, draw, seed, "all")
    warm_gpu95 = _warm_all(m, topo, wl, ram, draw, seed, "gpu", pct=2,
                           n_iter=COND_WARM_ITER)
    dec, cens = _decode_ceiling(m, topo, wl, st, seed)
    o["max_users_decode"] = dec
    o["max_users_decode_censored"] = cens
    curve = M.decode_curves(m, topo, wl, [1, 8, 64], n_iter=DECODE_ITER,
                            seed=seed, mbu=st["mbu"])
    for i, n in enumerate((1, 8, 64)):
        o[f"decode_p50_n{n}"] = float(curve[1][i])
    sd = M.steady_decode_point(m, topo, wl, rate, out_tokens=st["out"],
                               n_iter=DECODE_ITER, seed=seed, mbu=st["mbu"])
    o["steady_n"] = sd["n"]
    o["steady_per_user_tok_s"] = sd["per_user_tok_s"]
    o["steady_saturated"] = bool(sd["saturated"])

    # ---- power and the bill ----------------------------------------------
    pue = float(st["pue"])
    e = M.energy_cost(m, topo, wl, rate, dec, st["users"], chunk,
                      turn_tokens=turn, pue=pue, eur_kwh=st["ekwh"], mfu=mfu,
                      out_tokens=st["out"], per_pass_overhead=True,
                      eur_gpu_h=st["gpuh"])
    o["power_d_p"] = e["d_p"]
    o["power_d_d"] = e["d_d"]
    o["power_per_gpu_w"] = e["per_gpu_w"]
    o["power_kw"] = e["kw"]
    o["energy_eur_month"] = e["eur_month"]
    o["energy_hw_month"] = e["hw_month"]
    o["energy_total_month"] = e["total_month"]
    o["energy_eur_user"] = e["eur_user"]
    o["energy_eur_mtok"] = e["eur_mtok"]

    # ---- conditioning diagnostics (never compared) ------------------------
    cond = {
        # 1/(1 - rho) amplifies every queue figure as this approaches 1
        "duty": o["prefill_duty"],
        # k = 2(SLA - E[S|miss]) in max_users_latency vanishes as this hits 0,
        # and goes NEGATIVE where the miss alone already breaches the budget
        "sla_headroom": 1.0 - o["moments_miss"] / st["sla"],
        # ...and the same against the 10 s budget spikeMetrics hard-wires, for
        # the quantities compared at that budget rather than at state.sla
        "sla10_headroom": 1.0 - o["moments_miss"] / 10.0,
        # counts: one session either way is 33% of three
        "warm_p5": o["warm_p5_all"],
        "decode_ceiling": dec,
        # 1 when sla_miss_rate returned its `hi` clamp, i.e. the SLA survives
        # an all-cold stream at this load and there is no root inside [0, 1].
        # Python reports the clamp; the explorer's closed form keeps going and
        # returns an "f*" of 40 — the same verdict, and not a miss rate.
        "sla_f_unreachable": float(o["sla_miss_rate_sla10"] >= 1.0),
        # squared coefficient of variation of the context length: E[S^2|miss]
        # runs on L^4, so its sampling variance scales with this
        "ctx_cv2": (o["ctx_mean_sq"] / o["ctx_mean"] ** 2 - 1.0
                    if o["ctx_mean"] > 0 else 0.0),
        # HOW FAR PAST THE MIRROR'S OWN AXIS the load sits. steadyDecodePoint
        # inverts the sweep the page already drew, and decodePlan sizes that
        # sweep from the GPU-resident warm band:
        #     nMax = max(DECODE_NMIN, ceil(warm_gpu_p95 x DECODE_HEADROOM))
        # At a ratio >= 1 the demand runs off the end of that axis and the
        # mirror reports `saturated` while Python, searching to n = 4,096,
        # resolves a point. This is the CAUSE of the steady_* disagreement, so
        # the allowlist gates on it. The two constants below are the
        # explorer's (CONFIG.DECODE_NMIN / DECODE_HEADROOM), restated here
        # ONLY to compute this diagnostic — nothing compared depends on them.
        "steady_nmax_mirror": float(_mirror_nmax(warm_gpu95)),
        "steady_nmax_ratio": _steady_nmax_ratio(m, topo, wl, st, rate,
                                                warm_gpu95, seed),
    }
    return o, cond


MIRROR_DECODE_NMIN = 120        # CONFIG.DECODE_NMIN
MIRROR_DECODE_HEADROOM = 1.15   # CONFIG.DECODE_HEADROOM


def _mirror_nmax(warm_gpu_p95: float) -> int:
    return max(MIRROR_DECODE_NMIN,
               int(math.ceil(warm_gpu_p95 * MIRROR_DECODE_HEADROOM)))


def _steady_nmax_ratio(m, topo, wl, st, rate, warm_gpu_p95, seed) -> float:
    """demand / (aggregate decode throughput at the mirror's widest n).

    >= 1 means the load asks for more output than the explorer's own decode
    axis can retire, which is exactly when steadyDecodePoint stops inverting
    and starts reporting `saturated`.
    """
    demand = rate * st["out"]
    if demand <= 0:
        return 0.0
    n = _mirror_nmax(warm_gpu_p95)
    p50 = float(M.decode_curves(m, topo, wl, [n], n_iter=COND_DECODE_ITER,
                                seed=seed, mbu=st["mbu"])[1][0])
    agg = n * p50
    return demand / agg if agg > 0 else float("inf")


def _grow_draw(fn, draw: int):
    """Call fn(draw), doubling until warm_capacity stops calling it censored.

    warm_capacity REFUSES a censored draw rather than returning a silent
    under-count. model.warm_draw is sized for the study's workload; a big pool
    full of tiny sessions (a 5k median under a 40k shared prefix) needs more.
    Doubling keeps every legal explorer state in the fixture instead of quietly
    dropping the awkward ones. Raises if eight doublings do not suffice — a
    silent NaN here would become a silently skipped comparison.
    """
    for _ in range(8):
        try:
            return fn(draw)
        except ValueError as err:
            if "censored" not in str(err):
                raise
            draw *= 2
    raise RuntimeError(f"warm fill still censored at draw={draw}")


def _warm_user(m, topo, wl, ram, draw, seed) -> float:
    """THE model's own user-class ceiling — max_users_cache, not a
    reimplementation of it. It owns the p5 / which='user' choice; passing the
    draw only replaces the default it would have computed itself."""
    return _grow_draw(
        lambda d: M.max_users_cache(m, topo, wl, ram_gib=ram, n_iter=WARM_ITER,
                                    draw=d, seed=seed), draw)


def _warm_all(m, topo, wl, ram, draw, seed, which, pct: int = 0,
              n_iter: int = None) -> float:
    """warm_capacity's other arms, which have no max_users_* wrapper:
    which='all' (the storage view the explorer's tile quotes) and which='gpu'
    (the HBM-resident band decodePlan sizes its axis from). pct indexes the
    (p5, p50, p95) triple the function returns."""
    it = WARM_ITER if n_iter is None else n_iter
    return _grow_draw(
        lambda d: float(M.warm_capacity(m, topo, wl, ram_gib=ram,
                                        n_iter=it, draw=d, seed=seed,
                                        which=which)[pct]), draw)


def _decode_ceiling(m, topo, wl, st, seed) -> tuple:
    """(ceiling, censored), searched to the SAME cap the explorer uses.

    prefill.js scales its search cap with the floor — HI = max(4096, 4096 x 40
    / floor) — because per-user speed is aggregate/n, so a chat-shaped 5 tok/s
    floor puts the crossing ~8x higher. max_users_decode's `hi` default is a
    flat 4096 and it RAISES past it, so the two disagree on every low-floor
    state unless the cap is passed in. Passing it keeps the vectors comparing
    the model, not the default. A crossing genuinely beyond the shared cap is
    recorded as (cap, censored=True), which is what the explorer returns.
    """
    hi = max(4096, round(4096 * M.DECODE_FLOOR_TOKS / st["decode_floor"]))
    try:
        return M.max_users_decode(m, topo, wl, floor=st["decode_floor"],
                                  n_iter=DECODE_ITER, seed=seed, hi=hi,
                                  mbu=st["mbu"]), False
    except ValueError:
        return float(hi), True


# ---------------------------------------------------------------------------
# the state set
# ---------------------------------------------------------------------------
def legal_deployments():
    """Every (model, gpu, wdt, kv, ngpu, tp) the explorer can reach with a
    non-empty KV pool — the enumeration enforceConstraints() walks a user
    through one control at a time."""
    for mk in M.MODELS:
        base0 = M.MODELS[mk]
        for gpu in GPU_KEYS:
            g = M.GPUS[gpu]
            wdts = ["fp8"]
            if g.supports_nvfp4 and base0.nvfp4_w is not None:
                wdts.append("nvfp4")
            for wdt in wdts:
                base = base0 if wdt == "fp8" else M.with_weight_dtype(base0, wdt)
                for kv in ("fp8", "fp16"):
                    if kv == "fp16" and not base.kv_fp16_ok:
                        continue
                    if (kv == "fp8" and base.kv_fp8_blackwell_only
                            and not g.supports_nvfp4):
                        continue
                    m = M.with_kv_dtype(base, kv)
                    need = M.min_tp_for(m, gpu)
                    for ngpu in NGPUS:
                        for tp in divisors(ngpu):
                            if tp < need:
                                continue
                            topo = M.topology_grid(ngpu // tp, tp, gpu)
                            if M.kv_pool_tokens(m, topo) <= 0:
                                continue
                            yield dict(model=mk, gpu=gpu, wdt=wdt, kv=kv,
                                       ngpu=ngpu, tp=tp)


def base_state(dep: dict) -> dict:
    st = dict(DEFAULT_STATE)
    st.update(dep)
    st["mtp"] = M.MODELS[dep["model"]].mtp
    st["gpuh"] = M.GPUS[dep["gpu"]].eur_gpu_h
    st["cap"] = min(st["cap"], cap_slider_max(dep["model"]))
    return st


# One knob at a time, on top of a reference deployment. These isolate which
# CONTROL a disagreement follows, which is what makes the allowlist readable.
# Values are inside each control's real range (interactive/index.html): users
# 4-1024, think 5-180, turn 250-16000, out 100-8000, sla 1-60, decode_floor
# 5-100, inval 0-100, sub_ratio 0-1, sigmas 0.30-1.40, sys 1-40, ram 0-1024,
# burst 1-512, mbu/mfu 0.10-1.00, ekwh 0.05-0.60, gpuh 0.5-20, mtp 1.0-3.0.
# The medians take applyURLState's x4 scaling (a share link reaches 4x the
# slider's max), and `cap` takes capSliderMax() per model. A value outside a
# control's range would be a state no user can produce, i.e. a test of nothing.
KNOB_SWEEP = [
    ("chunk", ["2048", "4096", "8192", "16384", "65536"]),
    ("users", [8, 200, 1000]),
    ("think", [10, 60, 120]),
    ("inval", [0.0, 5.0, 25.0, 60.0]),
    ("sla", [2, 30, 60]),
    ("decode_floor", [5, 20, 80]),
    ("turn", [500, 8000, 16000]),
    ("out", [100, 1200, 4000]),
    ("burst", [1, 128, 512]),
    ("mbu", [0.15, 0.30, 1.0]),
    ("mfu", [0.35, 0.55]),
    ("cap", [32, 64, 262, "max"]),
    ("sys", [3, 30, 40]),
    ("ram", [128, 1024]),
    ("sub_ratio", [0.0, 0.5, 1.0]),
    ("sub_shares_prefix", [True]),
    ("user_median", [5, 90]),
    ("user_sigma", [0.3, 1.4]),
    ("sub_median", [2, 40]),
    ("sub_sigma", [0.3, 1.4]),
    ("mtp", [1.0, 1.7]),
    ("ekwh", [0.08, 0.26]),
    ("pue", ["1.2", "2.0"]),
    ("gpuh", [1.0, 12.0]),
]

# Deployments the knob sweep is run on: a dense single GPU, an MoE on TP2, a
# huge MoE that only exists at TP8, and a DP grid (replicas > 1, the axis where
# the two operating-point conventions part company).
SWEEP_ANCHORS = [
    dict(model="27B", gpu="H200", wdt="fp8", kv="fp8", ngpu=1, tp=1),
    dict(model="35BA3B", gpu="H200", wdt="fp8", kv="fp8", ngpu=2, tp=2),
    dict(model="GLM52", gpu="H200", wdt="fp8", kv="fp8", ngpu=8, tp=8),
    dict(model="MM35", gpu="H200", wdt="fp8", kv="fp8", ngpu=8, tp=2),
]
# how many knobs of the sweep each anchor gets (the first anchor gets all of
# them; the others cover the knobs most likely to interact with topology)
ANCHOR_KNOBS = {
    1: {"chunk", "inval", "sla", "decode_floor", "mbu", "users", "out", "ram"},
    2: {"chunk", "inval", "sla", "decode_floor", "users", "mbu"},
    3: {"chunk", "inval", "sla", "decode_floor", "users", "mbu", "out"},
}


def build_states() -> list[dict]:
    states, seen = [], set()

    def add(st):
        key = json.dumps(st, sort_keys=True)
        if key in seen:
            return
        seen.add(key)
        states.append(st)

    # 1. every model x GPU at its reference deployment, all defaults
    for dep in legal_deployments():
        if dep["wdt"] == "fp8" and dep["kv"] == "fp8" and dep["ngpu"] == dep["tp"]:
            add(base_state(dep))

    # 2. one knob at a time on the sweep anchors
    for idx, anchor in enumerate(SWEEP_ANCHORS):
        allowed = ANCHOR_KNOBS.get(idx)
        for knob, values in KNOB_SWEEP:
            if allowed is not None and knob not in allowed:
                continue
            for v in values:
                st = base_state(anchor)
                if knob == "cap":
                    v = (cap_slider_max(st["model"]) if v == "max"
                         else min(v, cap_slider_max(st["model"])))
                st[knob] = v
                add(st)

    # 3. EVERY legal deployment, each with a seeded knob draw. Not subsampled:
    #    a deployment the fixture never prices is a deployment the mirror is
    #    free to get wrong, and the enumeration is the one axis where that is
    #    cheap to rule out. The knobs are stratified rather than crossed - the
    #    full product is astronomically large, and the one-at-a-time sweep
    #    above already isolates which control a disagreement follows.
    rng = random.Random(20260906)
    for dep in legal_deployments():
        st = base_state(dep)
        st["chunk"] = rng.choice([str(c) for c in CHUNKS])
        st["users"] = rng.choice([16, 32, 64, 128, 400])
        st["think"] = rng.choice([15, 30, 45, 90])
        st["inval"] = rng.choice([0.5, 1.0, 3.0, 10.0, 35.0])
        st["sla"] = rng.choice([5, 10, 20])
        st["decode_floor"] = rng.choice([10, 40, 60])
        st["turn"] = rng.choice([1000, 2000, 6000])
        st["out"] = rng.choice([200, 400, 900])
        st["burst"] = rng.choice([8, 32, 200])
        st["mbu"] = rng.choice([0.15, 0.22, 0.30])
        st["mfu"] = rng.choice([0.35, 0.45, 0.55])
        st["sys"] = rng.choice([3, 15, 30])
        st["sub_ratio"] = rng.choice([0.0, 0.1, 0.4])
        st["user_median"] = rng.choice([12, 31, 60])
        st["user_sigma"] = rng.choice([0.5, 0.81, 1.1])
        st["ram"] = rng.choice([0, 0, 256])
        st["cap"] = min(rng.choice([64, 180, 262]), cap_slider_max(st["model"]))
        add(st)

    states.sort(key=lambda s: json.dumps(s, sort_keys=True))
    return states


# ---------------------------------------------------------------------------
# seed-to-seed spread — where the `mc` bands come from
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def sampling(n: int, seed: int):
    """Re-point the model's three sampling entry points at (n, seed).

    They take n/seed keyword arguments already; the functions built on top of
    them (cold_request_seconds, the service moments, every queue metric) call
    them with defaults and expose no seed of their own. Binding the defaults is
    the only way to ask "what would this have returned on another draw?"
    without restating a formula that lives in model.py. Used ONLY by the spread
    probe — the emitted vectors always run the unpatched module.
    """
    names = ("context_moments", "mean_passes", "mean_context",
             "_prefill_service_arrays")
    saved = {k: getattr(M, k) for k in names}
    try:
        for k in names:
            setattr(M, k, functools.partial(saved[k], n=n, seed=seed))
        yield
    finally:
        for k, v in saved.items():
            setattr(M, k, v)


# quantities whose value is a sampled statistic (everything else is closed form
# from constants only). Anything not listed here is class `exact`.
MC_QUANTITIES = [
    "ctx_mean", "ctx_mean_sq", "mean_passes",
    "cold_request_seconds", "warm_request_seconds",
    "moments_miss", "moments_miss_sq", "moments_hit", "moments_hit_sq",
    "prefill_duty", "queue_wait_seconds",
    "ttft_miss_fcfs", "ttft_hit_fcfs", "ttft_hit_ps",
    "breakeven_miss_rate", "spike_tolerance", "spike_tolerance_sla10",
    "sla_miss_rate_sla10", "burst_drain_seconds_b32",
    "max_users_latency", "max_users_saturation",
    "max_users_cache", "warm_p5_all", "max_users_decode",
    "decode_p50_n1", "decode_p50_n8", "decode_p50_n64",
    "steady_n", "steady_per_user_tok_s",
    "power_d_p", "power_d_d", "power_per_gpu_w", "power_kw",
    "energy_eur_month", "energy_total_month", "energy_eur_user",
    "energy_eur_mtok",
]

# booleans: compared for equality, never for a relative error
FLAG_QUANTITIES = ["max_users_decode_censored", "steady_saturated"]

SPREAD_SEEDS = (0, 1, 2)
SPREAD_PROBE_STATES = 24   # states sampled for the spread, evenly strided


def _spread_one(st: dict, n: int, warm_iter: int, dec_iter: int) -> tuple:
    """(spread-per-quantity, cond) for one state: compute() at three seeds.

    `cond` rides along so the band derivation can EXCLUDE probe states from a
    quantity whose noise is dominated by a regime the allowlist already owns —
    the tiny-warm-count tail, where a count of three moves 33% on one session
    and would otherwise set the band for every configuration.
    """
    global WARM_ITER, DECODE_ITER
    runs = []
    saved = (WARM_ITER, DECODE_ITER)
    try:
        WARM_ITER, DECODE_ITER = warm_iter, dec_iter
        for seed in SPREAD_SEEDS:
            with sampling(n, seed):
                runs.append(compute(st, seed=seed))
    finally:
        WARM_ITER, DECODE_ITER = saved
    out = {}
    for q in MC_QUANTITIES:
        vals = [r[0].get(q) for r in runs]
        if any(v is None or not math.isfinite(v) for v in vals):
            continue
        lo, hi = min(vals), max(vals)
        mid = 0.5 * (lo + hi)
        out[q] = 0.0 if mid == 0 else abs(hi - lo) / abs(mid)
    return out, runs[0][1]


# A probe state is dropped from these quantities' band when its warm count is
# in the handful-of-sessions regime: the allowlist already carries that tail
# (gated on _warm_p5), and letting it set the band would license the same
# slack on the configurations that hold thousands.
SMALL_COUNT_QUANTITIES = ("max_users_cache", "warm_p5_all")
SMALL_COUNT_THRESHOLD = 12.0


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    r = (p / 100.0) * (len(s) - 1)
    lo = int(math.floor(r))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (r - lo) * (s[hi] - s[lo])


def _spread_task(args):
    """Picklable entry point for the process pool: (state, n, warm_iter,
    decode_iter) -> (spread, cond)."""
    return _spread_one(*args)


def measure_spread(states: list[dict], stride: int, jobs: int = 1) -> dict:
    """Seed-to-seed spread per quantity, at two sampling scales.

    `python` is the scale the committed vectors are generated at.
    `mirror` cuts the context draw to the explorer's own 20,000 — the one
    place the two sides sample materially differently, and therefore what sets
    the comparison's noise floor. A band read off the Python scale alone would
    be far too tight to survive a correct mirror.

    p50 / p90 / max over the probe states, not just the max: a couple of the
    sampled states put an estimator somewhere it is ill-conditioned (the
    latency ceiling near c -> SLA, the breakeven rate near cold -> warm) and
    their spread runs an order of magnitude above every other state's. A band
    set on the max would then license real drift everywhere; the band is set
    on p90 and the ill-conditioned states are named in the allowlist instead.
    """
    probe = states[::stride]
    out = {"seeds": list(SPREAD_SEEDS), "n_states": len(probe),
           "statistic": "relative full spread (max-min)/mid across the seeds, "
                        "summarised over the probe states",
           "python": {}, "mirror": {}}
    for label, n, wi, di in (("python", CTX_N, WARM_ITER, DECODE_ITER),
                             ("mirror", MIRROR_CTX_N, WARM_ITER, DECODE_ITER)):
        agg: dict[str, list] = {}
        for spr, cond in _map(_spread_task,
                              [(st, n, wi, di) for st in probe], jobs):
            small = cond["warm_p5"] < SMALL_COUNT_THRESHOLD
            for q, v in spr.items():
                if small and q in SMALL_COUNT_QUANTITIES:
                    continue
                agg.setdefault(q, []).append(v)
        out[label] = {q: {"p50": r12(_pct(v, 50)), "p90": r12(_pct(v, 90)),
                          "max": r12(max(v)), "n": len(v)}
                      for q, v in sorted(agg.items())}
    out["excluded_from_bands"] = {
        q: f"probe states with warm_p5 < {SMALL_COUNT_THRESHOLD:g} "
           "(the allowlist owns that tail)"
        for q in SMALL_COUNT_QUANTITIES}
    return out


# how much of the band each part of the derivation contributes
SPREAD_MULT = 3.0     # two independent samplers, both with tails
BAND_FLOOR = 0.02     # below this is noise on any sampled statistic
BAND_CAP = 0.25       # above this, name the states instead of widening for all

# Quantities that ARE the same statistic take the widest of the group's bands.
# power_draw's d_p is prefill_duty clamped at 1; below the clamp they are the
# same number, but the clamp shrinks d_p's measured spread on the probe, so an
# independent derivation hands the identical figure a tighter band and it trips
# on noise its twin absorbs.
BAND_GROUPS = [("prefill_duty", "power_d_p")]


def bands_from_spread(spread: dict) -> dict:
    """The `mc` tolerance band per quantity.

    3x the p90 seed-to-seed spread measured at the MIRROR's sampling scale,
    floored at 2%, capped at 25%, rounded up to two significant digits so the
    file reads as a decision rather than a fitted constant.

    3x because the measured spread is the range of three seeds of ONE sampler,
    while the comparison runs two independent samplers (numpy PCG64 against
    mulberry32 + Box-Muller) and the band has to survive both tails at once.
    The cap is the important half: a quantity whose noise genuinely exceeds 25%
    is not being pinned by a tolerance at all, and saying so in
    known_disagreements.json — with the states it happens on — is worth more
    than a band nothing could fail.
    """
    bands = {}
    for q in MC_QUANTITIES:
        s = spread["mirror"].get(q, {}).get("p90", 0.0)
        b = min(BAND_CAP, max(BAND_FLOOR, SPREAD_MULT * s))
        mag = 10 ** math.floor(math.log10(b))
        bands[q] = r12(math.ceil(b / mag * 10) * mag / 10)
    for group in BAND_GROUPS:
        widest = max(bands[q] for q in group if q in bands)
        for q in group:
            if q in bands:
                bands[q] = widest
    return bands


# ---------------------------------------------------------------------------
# Python -> JS map. Every quantity above names the function on each side.
# ---------------------------------------------------------------------------
MAPPING = [
    # quantity, python, js, class, note
    ("kv_pool_tokens", "model.kv_pool_tokens", "config.js kv_pool_tokens", "exact", ""),
    ("effective_bw", "model.effective_bw", "config.js effective_bw", "exact", ""),
    ("peak_flops", "model.peak_flops", "prefill.js peakFlops", "exact", ""),
    ("tp_efficiency", "model.tp_efficiency", "config.js tpEff", "exact", ""),
    ("w_decode_n64", "model.Model.w_decode", "config.js w_decode", "exact",
     "linear (no-overlap) expert union; JS has no `coverage` arm"),
    ("prefill_overhead_seconds", "model.prefill_overhead_seconds",
     "prefill.js prefillOverheadSeconds", "exact", ""),
    ("mfu_ceiling", "model.mfu_ceiling", "prefill.js mfuCeil", "exact",
     "both solve the anchor at CHUNK_DEFAULT/PREFILL_CHUNK = 32,768"),
    ("mfu_effective_at_chunk", "model.mfu_effective", "prefill.js mfuEff", "exact", ""),
    ("prefill_flops_*", "model.prefill_flops (total)", "prefill.js prefillFlops", "exact", ""),
    ("prefill_seconds_*", "model.prefill_seconds", "prefill.js prefillSeconds", "exact", ""),
    ("prefill_context_seconds_120k", "model.prefill_context_seconds",
     "prefill.js prefillContextSeconds", "exact", ""),
    ("miss_context_seconds_120k", "model.miss_context_seconds",
     "prefill.js missContextSeconds", "exact", ""),
    ("warm_pass_seconds_turn", "model.prefill_context_seconds(prior=0)",
     "prefill.js prefillContextSeconds(turn, C, 0)", "exact", ""),
    ("ctx_mean / ctx_mean_sq", "model.context_moments",
     "prefill.js contextStats -> {mean, meanSq}", "mc",
     "Python draws 200,000 with numpy PCG64; the explorer draws 20,000 with mulberry32"),
    ("mean_passes", "model.mean_passes",
     "prefill.js meanPasses (module-private) — DUPLICATED in drive.mjs", "mc",
     "NOT CALLED: meanPasses is not exported, so the test recomputes "
     "E[ceil(L/C)] over the same contextStats draw"),
    ("cold_request_seconds", "model.cold_request_seconds(per_pass_overhead=True)",
     "prefill.js coldRequestSeconds", "mc",
     "the explorer has no flat-MFU arm: coldRequestSeconds is always the overhead pricing"),
    ("warm_request_seconds", "model.warm_request_seconds(prior=mean_context)",
     "prefill.js prefillContextSeconds(turn, C, cs.mean)", "mc", ""),
    ("moments_*", "model.prefill_service_moments / _prefill_service_arrays",
     "prefill.js prefillServiceMoments -> {miss, missSq, hit, hitSq}", "mc", ""),
    ("prefill_duty", "model.prefill_duty", "prefill.js spikeMetrics(...).rho", "mc", ""),
    ("queue_wait_seconds", "model.queue_wait_seconds",
     "prefill.js spikeMetrics(...).wait", "mc", ""),
    ("ttft_miss_fcfs", "model.prefill_ttft_seconds(request='cold')",
     "render.js op.ttftMiss (inline) — DUPLICATED in drive.mjs", "mc",
     "NOT CALLED: render.js stamps this expression onto the operating point "
     "inline, so the test restates it. The arithmetic in render.js itself is "
     "unpinned until it moves into prefill.js — see the README's coverage "
     "limitations."),
    ("ttft_hit_fcfs", "model.prefill_ttft_seconds(request='warm')",
     "render.js op.ttftHitFcfs (inline) — DUPLICATED in drive.mjs", "mc",
     "NOT CALLED; same inline block as ttft_miss_fcfs"),
    ("ttft_hit_ps", "model.prefill_ttft_seconds(request='warm', discipline='ps')",
     "render.js op.ttftHitPs (inline) — DUPLICATED in drive.mjs", "mc",
     "NOT CALLED; same inline block as ttft_miss_fcfs"),
    ("sla_miss_rate_sla10", "model.sla_miss_rate(sla_seconds=10, request='cold')",
     "prefill.js spikeMetrics(...).fsla", "mc",
     "spikeMetrics hard-wires SPIKE_SLA_S = 10 here too, so 10 is the only "
     "budget the two answer the same question at. Python bisects f over "
     "[0, 1] and CLAMPS to that interval; the explorer solves the same "
     "equation in closed form (E[S] and E[S^2] are both linear in f) and "
     "clamps only at 0, and their 'no load meets it' tests differ: Python "
     "compares the full TTFT at f = 0 (queue wait included), the explorer "
     "compares the miss's own prefill alone (k = 2(SLA - E[S|miss]) <= 0)"),
    ("burst_drain_seconds_b32", "model.burst_drain_seconds(burst=32)",
     "prefill.js spikeMetrics(...).drain", "mc",
     "spikeMetrics hard-wires SPIKE_BURST = 32; state.burst reaches the tile "
     "only through inline render.js arithmetic, so 32 is the burst the two "
     "sides can be compared at"),
    ("breakeven_miss_rate", "model.breakeven_miss_rate",
     "prefill.js breakevenMissRate", "mc", ""),
    ("spike_tolerance", "model.spike_tolerance(sla_seconds=state.sla)",
     "planner.js bStar(mo, f, state.sla, rate)", "mc", ""),
    ("spike_tolerance_sla10", "model.spike_tolerance(sla_seconds=10)",
     "prefill.js spikeMetrics(...).bstar", "mc",
     "spikeMetrics hard-wires SPIKE_SLA_S = 10 instead of state.sla"),
    ("max_users_latency", "model.max_users_latency", "prefill.js maxUsersLatency", "mc",
     "per replica GROUP on both sides; operatingPoint() then scales by replicas"),
    ("max_users_saturation", "model.max_users_saturation",
     "prefill.js maxUsersSaturation", "mc", "per replica GROUP on both sides"),
    ("max_users_cache", "model.max_users_cache (warm_capacity which='user')",
     "render.js warmUsersNow = warmCapacity(...).all[0] * (1 - p_sub) "
     "(inline) — DUPLICATED in drive.mjs", "mc",
     "NOT CALLED, and a STANDING APPROXIMATION either way: the explorer scales "
     "the whole warm p5 by the non-subagent share where Python counts "
     "user-class sessions inside each fill"),
    ("warm_p5_all", "model.warm_capacity(which='all')[0]",
     "capacity.js warmCapacity(...).all[0]", "mc", ""),
    ("max_users_decode", "model.max_users_decode", "prefill.js maxUsersDecode(...).n", "mc", ""),
    ("decode_p50_n*", "model.decode_curves (p50)", "capacity.js decodeCurves (p50)", "mc", ""),
    ("steady_n / steady_per_user_tok_s", "model.steady_decode_point",
     "prefill.js steadyDecodePoint", "mc",
     "Python bisects integer n re-sampling each probe; the explorer inverts the "
     "linearly-interpolated aggregate of one pre-sampled sweep"),
    ("power_*", "model.power_draw", "cost.js powerDraw (inside energyCost)", "mc", ""),
    ("energy_*", "model.energy_cost", "cost.js energyCost", "mc", ""),
    ("(state -> model)", "golden.py state_model / state_topo / state_wl",
     "render.js modelFor + state.js currentTopo/currentWL", "n/a",
     "the explorer's dtype/mtp switches; state_dt and wover are explorer-only "
     "and are NOT sampled (Python has no counterpart knob)"),
    ("(NOT COMPARED) itl_spike / spike_token_debt",
     "model.itl_spike, model.spike_token_debt",
     "render.js itlSpikeRatio (module-private) -> op.tokensLost", "n/a",
     "itlSpikeRatio is private to render.js AND prices the spike differently "
     "on purpose: step = min(C, E[L]) with prior = 0 when the chunk exceeds "
     "the mean context, against Python's always-C at prior = E[L]/2; its "
     "decode leg also reads an interpolated stress point rather than a "
     "decode_curves probe. Comparable only once it is exported and the two "
     "pricings are reconciled — a Python-first change, not a test change."),
]


# ---------------------------------------------------------------------------
# serialisation
# ---------------------------------------------------------------------------
def r12(v):
    """Round to 12 significant digits so the file is byte-reproducible across
    platforms without pinning anyone to the last ulp."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return v
    if isinstance(v, int):
        return v
    if math.isinf(v):
        return "Infinity" if v > 0 else "-Infinity"
    if math.isnan(v):
        return "NaN"
    return float(f"{v:.12g}")


def _map(fn, items, jobs: int):
    """fn over items, in order, across `jobs` processes.

    Every state is priced from fixed seeds and touches no shared state, so the
    result does not depend on how the work is split — `--jobs 1` and
    `--jobs 16` produce the same bytes. The fixture prices ~800 states with two
    Monte-Carlo warm fills each; serially that is ten minutes of CI on every
    pull request, which is the kind of cost that gets a check deleted.
    """
    if jobs <= 1:
        return [fn(x) for x in items]
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        return list(ex.map(fn, items, chunksize=4))


def render(states: list[dict], spread: dict, bands: dict, jobs: int) -> str:
    vectors = []
    priced = _map(compute, states, jobs)
    for i, (st, (out, cond)) in enumerate(zip(states, priced)):
        topo = state_topo(st)
        vectors.append({
            "id": i,
            "label": f"{st['model']}/{topo.name}/{st['wdt']}/{st['kv']}KV",
            "state": st,
            "out": {k: r12(v) for k, v in out.items()},
            "cond": {k: r12(v) for k, v in cond.items()},
        })
    doc = {
        "schema": SCHEMA,
        "generated_by": "scripts/golden.py",
        "source_of_truth": "src/workingset/model.py",
        "mirror": "interactive/src/*.js",
        "rule": ("a modelling change lands in Python first; the JS golden test "
                 "failing is the intended signal, not the bug"),
        "settings": {
            "warm_n_iter": WARM_ITER, "decode_n_iter": DECODE_ITER,
            "context_n": CTX_N, "mirror_context_n": MIRROR_CTX_N,
            "per_pass_overhead": True,
            "discipline": "fcfs",
            "union": "linear",
            "rate_group": "users * (1 + sub_ratio) / think / replicas",
            "ceiling_scope": "per replica group",
            "prefill_lengths": list(PREFILL_LENGTHS),
            "context_len": CONTEXT_LEN,
        },
        "tolerances": {
            "exact": {"rel": 1e-06},
            "mc": bands,
            "mc_default": 0.1,
            "mc_derivation": (
                f"{SPREAD_MULT:g}x the p90 seed-to-seed spread under mc_spread"
                f".mirror, floored at {BAND_FLOOR:g}, capped at {BAND_CAP:g}, "
                "rounded up to two significant digits. See "
                "scripts/golden.py bands_from_spread and tests/golden/README.md"),
            "storage_precision": (
                "every value here is rounded to 12 significant digits so the "
                "file regenerates byte-identically. The exact class is "
                "asserted at 1e-6, six orders coarser than that quantisation, "
                "so the residuals it reports bottom out around 1e-12 — read a "
                "reported 5e-12 as 'nothing above 1e-11', not as a measured "
                "agreement to 5e-12."),
        },
        "mc_spread": spread,
        "mc_quantities": MC_QUANTITIES,
        "flag_quantities": FLAG_QUANTITIES,
        "mapping": [dict(zip(("quantity", "python", "js", "class", "note"), r))
                    for r in MAPPING],
        "vectors": vectors,
    }
    return json.dumps(doc, indent=1, sort_keys=False,
                      separators=(",", ": "), allow_nan=False) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="verify the committed vectors are current; write nothing")
    ap.add_argument("--spread-probe", type=int, default=SPREAD_PROBE_STATES,
                    help="how many states to probe for the seed-to-seed "
                         "spread (evenly strided through the fixture)")
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 1,
                    help="worker processes; the output is identical at any "
                         "value (every state is independently seeded)")
    args = ap.parse_args()

    states = build_states()
    print(f"states: {len(states)} (jobs: {args.jobs})", file=sys.stderr)
    # a fixed PROBE COUNT, not a fixed stride: the state set grows whenever the
    # study gains a model or a GPU part, and a fixed stride would silently make
    # regeneration slower every time
    stride = max(1, len(states) // max(1, args.spread_probe))
    spread = measure_spread(states, stride, args.jobs)
    bands = bands_from_spread(spread)
    text = render(states, spread, bands, args.jobs)

    if args.check:
        if not OUT.exists():
            print(f"MISSING {OUT.relative_to(ROOT)} — run: uv run scripts/golden.py",
                  file=sys.stderr)
            return 1
        if OUT.read_text(encoding="utf-8") != text:
            print(f"STALE {OUT.relative_to(ROOT)} — the model has moved since it "
                  "was generated. Run: uv run scripts/golden.py", file=sys.stderr)
            return 1
        print(f"OK {OUT.relative_to(ROOT)} is current ({len(states)} vectors)")
        return 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(states)} vectors, "
          f"{len(text)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
