"""Regenerate every number quoted in docs/scenarios.md.

Run:  python scripts/tables.py
All tables print in the order they appear in the doc, so the doc can be
diffed against this output after any model change.
"""
import dataclasses

import numpy as np

import scenario_model as M
from scenario_model import Workload, MODELS, TOPOLOGIES

MODELS_K = ["27B", "35BA3B"]
TOPOS_K = ["1xH200", "2xH200-TP2", "2xH200-DP2"]


def wl(**kw):
    return Workload(**kw)


def mean_context(w: Workload, n=400_000, seed=1):
    full, _, _, _ = w.sample(np.random.default_rng(seed), n)
    return full.mean()


def max_mns_at_floor(model, topo, w, floor, union="linear", hi=1500):
    """Largest max_num_seqs with per-user p50 >= floor tok/s.

    Raises if the crossing lies beyond `hi` rather than returning a
    silently-censored value.
    """
    best = 0
    for n in range(1, hi + 1):
        _, p50, _, _ = M.decode_curves(model, topo, w, [n], n_iter=400, union=union)
        if p50[0] >= floor:
            best = n
        else:
            return best
    raise ValueError(f"floor crossing beyond hi={hi}: censored result")


def main():
    w0 = wl()

    print("== KV pools (M tokens) ==")
    for mk in MODELS_K:
        for tk in TOPOS_K:
            print(f"  {mk:7} {tk:12} {M.kv_pool_tokens(MODELS[mk], TOPOLOGIES[tk]) / 1e6:6.2f}")

    print("\n== Warm reusable sessions p5/p50/p95 (reference workload, 0 GB offload) ==")
    for mk in MODELS_K:
        for tk in TOPOS_K:
            p5, p50, p95 = M.warm_capacity(MODELS[mk], TOPOLOGIES[tk], w0, n_iter=1500)
            print(f"  {mk:7} {tk:12} {p5:5.0f} / {p50:5.0f} / {p95:5.0f}")

    print("\n== Warm with 600 GiB CPU offload (35B-A3B) ==")
    # draw must exceed the expected count or warm_capacity raises (censoring)
    for tk in ["1xH200", "2xH200-TP2"]:
        p5, p50, p95 = M.warm_capacity(MODELS["35BA3B"], TOPOLOGIES[tk], w0,
                                       ram_gib=600, n_iter=400, draw=10_000)
        print(f"  {tk:12} {p5:5.0f} / {p50:5.0f} / {p95:5.0f}")

    print("\n== System-prompt sweep, warm p50 (35B-A3B) ==")
    for tk in ["1xH200", "2xH200-TP2"]:
        row = [f"{M.warm_capacity(MODELS['35BA3B'], TOPOLOGIES[tk], wl(sys_user=s), n_iter=1500)[1]:.0f}"
               for s in (3_000, 15_000, 30_000)]
        print(f"  {tk:12} 3k/15k/30k = {' / '.join(row)}")

    print("\n== Per-user decode p50 tok/s (35B-A3B; linear|coverage union) ==")
    for tk in TOPOS_K[:2]:
        for n in (16, 64, 120):
            _, a, _, _ = M.decode_curves(MODELS["35BA3B"], TOPOLOGIES[tk], w0, [n], n_iter=2000)
            _, b, _, _ = M.decode_curves(MODELS["35BA3B"], TOPOLOGIES[tk], w0, [n], n_iter=2000,
                                         union="coverage")
            print(f"  {tk:12} mns={n:3d}  {a[0]:6.0f} | {b[0]:6.0f}")

    print("\n== Aggregate p50 ktok/s at mns=64 (system total) ==")
    for mk in MODELS_K:
        for tk in TOPOS_K:
            _, _, _, agg = M.decode_curves(MODELS[mk], TOPOLOGIES[tk], w0, [64], n_iter=2000)
            print(f"  {mk:7} {tk:12} {agg[0] / 1000:6.1f}")

    print("\n== Subagent ratio sweep, warm p50 (35B-A3B, TP2, f=1%) ==")
    for r in (0.0, 0.1, 0.5, 1.0):
        p50 = M.warm_capacity(MODELS["35BA3B"], TOPOLOGIES["2xH200-TP2"],
                              wl(sub_ratio=r), n_iter=1200)[1]
        print(f"  r={r:.1f}  {p50:5.0f}")

    print("\n== Invalidation sweep, warm p50 (35B-A3B, TP2) ==")
    for f in (0.0, 0.01, 0.05, 0.10):
        p50 = M.warm_capacity(MODELS["35BA3B"], TOPOLOGIES["2xH200-TP2"],
                              wl(invalidation=f), n_iter=1200)[1]
        print(f"  f={f * 100:4.1f}%  {p50:5.0f}")

    print("\n== Serving-capacity planning table (conservative linear union) ==")
    print("  warm p5   = THE planning number: sessions kept warm in >=95% of draws")
    print("  warm p50  = median; warm_user = user-class sessions only (distinct users)")
    print("  v@warm    = per-user p50 tok/s if ALL GPU-resident warm sessions decode")
    print("              at once (100% duty); offloaded sessions cannot decode, so the")
    print("              concurrency comes from which='gpu' (identical here: 0 offload)")
    print("  mns@40    = max concurrent decoders at >=40 tok/s p50 (speed bound alone)")
    print("  -> the binding constraint is min(warm, mns@40); duty<100% relaxes only mns@40")
    mc = mean_context(w0)
    print(f"  mean sampled context = {mc / 1000:.1f}k tokens")
    for mk in MODELS_K:
        for tk in TOPOS_K:
            model, topo = MODELS[mk], TOPOLOGIES[tk]
            # n_iter matches the warm-capacity table so both quote the same p50
            p5, warm, _ = M.warm_capacity(model, topo, w0, n_iter=1500)
            u5, u50, _ = M.warm_capacity(model, topo, w0, n_iter=1500, which="user")
            # decode concurrency is ALWAYS the GPU-resident count, by
            # construction rather than by "this table happens to run at
            # ram_gib=0" (where which="gpu" and which="all" coincide exactly)
            _, warm_gpu, _ = M.warm_capacity(model, topo, w0, n_iter=1500, which="gpu")
            _, v_at_warm, _, _ = M.decode_curves(model, topo, w0, [int(warm_gpu)],
                                                 n_iter=1000)
            m40 = max_mns_at_floor(model, topo, w0, 40)
            per_cache = " (per replica)" if topo.kind == "dp" else ""
            print(f"  {mk:7} {tk:12} warm p5={p5:4.0f} p50={warm:4.0f} "
                  f"(user p5={u5:4.0f} p50={u50:4.0f}){per_cache}  "
                  f"v@warm={v_at_warm[0]:5.0f} tok/s  mns@40={m40:3d}")

    print("\n== Binding order WITHOUT MTP (speculative decoding off, mtp=1.0) ==")
    print("  the decision table's 'cache binds first' is CONDITIONAL on the 1.7x MTP")
    print("  speedup. MTP scales SPEED, not memory: at a FIXED concurrency, turning it")
    print("  off divides per-user tok/s by exactly 1.7 and leaves capacity untouched.")
    print("  The 40 tok/s CROSSING moves further than 1.7x (1.8-2.0x below) because it")
    print("  lands at a lower n, where the fixed per-step weight read is a bigger share")
    print("  of the bytes moved, so each seq removed buys back less speed.")
    print("  mns@40 = max decoders at >=40 tok/s p50; binds = min(warm, mns@40)")
    for mk in MODELS_K:
        for tk in TOPOS_K:
            model, topo = MODELS[mk], TOPOLOGIES[tk]
            no_mtp = dataclasses.replace(model, mtp=1.0)
            _, warm, _ = M.warm_capacity(model, topo, w0, n_iter=1500)
            _, u50, _ = M.warm_capacity(model, topo, w0, n_iter=1500, which="user")
            _, warm_gpu, _ = M.warm_capacity(model, topo, w0, n_iter=1500, which="gpu")
            m40_on = max_mns_at_floor(model, topo, w0, 40)
            m40_off = max_mns_at_floor(no_mtp, topo, w0, 40)
            _, v_off, _, _ = M.decode_curves(no_mtp, topo, w0, [int(warm_gpu)],
                                             n_iter=1000)
            binds = "cache" if m40_off >= warm else "BANDWIDTH"
            print(f"  {mk:7} {tk:12} warm p50={warm:4.0f} (users {u50:4.0f})  "
                  f"mns@40 {m40_on:4d} -> {m40_off:4d} no-MTP  "
                  f"v@warm {v_off[0]:5.1f} tok/s  binds: {binds}")

    print("\n== Scaling to N x H200 (35B-A3B, FP8) ==")
    print("  TP: one engine, ONE shared cache | DP: N replicas, cache splits (sticky routing)")
    print("  warm p5 = the planning number (95% of draws hold at least this many)")
    for n in (1, 2, 4, 8):
        row = []
        for kind in ("tp", "dp"):
            t = M.topology(kind, n)
            p5, p50, _ = M.warm_capacity(MODELS["35BA3B"], t, w0, n_iter=max(200, 1500 // n), draw=2000 + 1500 * n)
            sys_p5 = p5 * t.replicas
            row.append(f"{kind.upper()}: cache p5={p5:5.0f} system p5={sys_p5:5.0f} (p50 {p50:.0f})")
        print(f"  N={n}  " + "  |  ".join(row))

    print("\n== max_seq_len cap sweep (35B-A3B, TP2, warm p5/p50) ==")
    print("  lower cap truncates the log-normal tail -> smaller worst-case sessions")
    for cap in (60_000, 120_000, 180_000, 262_144):
        p5, p50, _ = M.warm_capacity(MODELS["35BA3B"], TOPOLOGIES["2xH200-TP2"],
                                     wl(cap=cap), n_iter=1200)
        print(f"  cap={cap // 1000:3d}k  {p5:5.0f} / {p50:5.0f}")

    print("\n== CPU offload sweep (35B-A3B, 1xH200, warm p50) ==")
    print("  offload is STORAGE ONLY: it adds warm sessions in host RAM, but the")
    print("  GPU-resident count -- the only sessions that can decode without a")
    print("  PCIe restore, and therefore the basis of every decode figure -- is flat")
    for gib in (0, 64, 128, 256, 512, 1024):
        kw = dict(ram_gib=gib, n_iter=300, draw=16_000)
        p50 = M.warm_capacity(MODELS["35BA3B"], TOPOLOGIES["1xH200"], w0, **kw)[1]
        g50 = M.warm_capacity(MODELS["35BA3B"], TOPOLOGIES["1xH200"], w0,
                              which="gpu", **kw)[1]
        print(f"  {gib:4d} GiB  warm(storage) {p50:5.0f}   GPU-resident {g50:5.0f}")

    print("\n== Median-context sweep (27B dense, FP8 KV, 1xH200) ==")
    print("  sweeps the USER prompt median (subagents stay at their 8k median);")
    print("  'warm users' = which='user', the distinct-user planning count")
    med_ks = (31, 45, 60, 80, 100, 140)
    rows = []
    for k in med_ks:
        p5, p50, _ = M.warm_capacity(MODELS["27B"], TOPOLOGIES["1xH200"],
                                     wl(user_median=k * 1000), n_iter=1500,
                                     which="user")
        rows.append((k, p5, p50))
    print("  median context |" + "".join(f" {k:4d}k |" for k, _, _ in rows))
    print("  warm users p50 |" + "".join(f" {p50:5.0f} |" for _, _, p50 in rows))
    print("  warm users p5  |" + "".join(f" {p5:5.0f} |" for _, p5, _ in rows))
    print("  p5 / p50       |" + "".join(f" {p5 / p50:5.2f} |" for _, p5, p50 in rows))

    print("\n== KV dtype switch: FP16 KV cache (default everywhere else: FP8) ==")
    print("  pool halves; warm capacity falls slightly less (state charge is dtype-independent)")
    for mk in MODELS_K:
        for tk in ["1xH200", "2xH200-TP2"]:
            m16 = M.with_kv_dtype(MODELS[mk], "fp16")
            pool = M.kv_pool_tokens(m16, TOPOLOGIES[tk])
            warm = M.warm_capacity(m16, TOPOLOGIES[tk], w0, n_iter=1500)[1]
            warm_u = M.warm_capacity(m16, TOPOLOGIES[tk], w0, n_iter=1500, which="user")[1]
            _, v64, _, _ = M.decode_curves(m16, TOPOLOGIES[tk], w0, [64], n_iter=2000)
            print(f"  {mk:7} {tk:12} pool={pool / 1e6:5.2f}M  warm={warm:4.0f} "
                  f"(user {warm_u:4.0f})  v@mns64={v64[0]:4.0f} tok/s")

    print("\n== Sensitivity: fp32 DeltaNet state (35B-A3B, warm p50) ==")
    m = MODELS["35BA3B"]
    m_fp32 = dataclasses.replace(m, deltanet_state=m.deltanet_state * 2)
    for tk in ["1xH200", "2xH200-TP2"]:
        a = M.warm_capacity(m, TOPOLOGIES[tk], w0, n_iter=800)[1]
        b = M.warm_capacity(m_fp32, TOPOLOGIES[tk], w0, n_iter=800)[1]
        print(f"  {tk:12} bf16={a:.0f}  fp32={b:.0f}  ({(b / a - 1) * 100:+.1f}%)")

    print("\n== Sensitivity: +15% deployed-weight overhead (35B-A3B, pool M tokens) ==")
    m_ovh = dataclasses.replace(m, w_resident=m.w_resident * 1.15)
    for tk in ["1xH200", "2xH200-TP2"]:
        a = M.kv_pool_tokens(m, TOPOLOGIES[tk])
        b = M.kv_pool_tokens(m_ovh, TOPOLOGIES[tk])
        print(f"  {tk:12} raw={a / 1e6:.2f}M  +15%={b / 1e6:.2f}M  ({(b / a - 1) * 100:+.1f}%)")

    print("\n== Structural-uncertainty stack (35B-A3B, warm p5) ==")
    print("  MC sampling spread (p50->p5 ~ 46 sessions) is SMALL next to structural")
    print("  unknowns. Stacking plausible adverse assumptions bounds the downside:")
    print("  anchor at 2x the measured FP16 LOWER bound (2.278M), fp32 recurrent")
    print("  state, +15% deployed-weight overhead, 10% invalidation.")
    base_p5 = M.warm_capacity(m, TOPOLOGIES["2xH200-TP2"], w0, n_iter=1500)[0]
    saved_anchor, saved_reserve = M.BASELINE_POOL_TOKENS_27B_1GPU, M.ACT_RESERVE
    try:
        M.BASELINE_POOL_TOKENS_27B_1GPU = 2 * 1.139e6
        M.ACT_RESERVE = M._act_reserve()
        m_stack = dataclasses.replace(m, deltanet_state=m.deltanet_state * 2,
                                      w_resident=m.w_resident * 1.15)
        w_stack = wl(invalidation=0.10)
        m_nomtp = dataclasses.replace(m_stack, mtp=1.0)   # MTP = immature path, excluded
        for tk in ["1xH200", "2xH200-TP2"]:
            s5, s50, _ = M.warm_capacity(m_stack, TOPOLOGIES[tk], w_stack, n_iter=1500)
            s5u = M.warm_capacity(m_stack, TOPOLOGIES[tk], w_stack, n_iter=1500,
                                  which="user")[0]
            _, v, _, _ = M.decode_curves(m_nomtp, TOPOLOGIES[tk], w_stack, [int(s5)],
                                         n_iter=800)
            print(f"  {tk:12} stacked p5={s5:4.0f} p50={s50:4.0f} (user p5 {s5u:4.0f})  "
                  f"v@p5warm MTP-off={v[0]:3.0f} tok/s")
    finally:
        M.BASELINE_POOL_TOKENS_27B_1GPU, M.ACT_RESERVE = saved_anchor, saved_reserve
    print(f"  (central TP2 p5 = {base_p5:.0f}; stacked downside ~{base_p5 - 403:.0f} sessions)")

    # ------------------------------------------------------------------
    # 2026-07 extension: B300 GPUs, NVFP4 weights, Mistral-Medium-3.5, GLM-5.2
    # ------------------------------------------------------------------
    print("\n== B300 x weight dtype: pool + warm p5/p50 (fp8 KV, reference workload) ==")
    print("  NVFP4 is B300-only (native FP4; the Hopper fallback is not modelled).")
    print("  Configs chosen so FP8 weights actually fit; '-' = no NVFP4 constants.")
    ext_configs = [
        ("27B",    [("tp", 1, "B300"), ("tp", 2, "B300")]),
        ("35BA3B", [("tp", 1, "B300"), ("tp", 2, "B300")]),
        ("MM35",   [("tp", 1, "B300"), ("tp", 2, "B300")]),
        ("GLM52",  [("tp", 4, "B300"), ("tp", 8, "B300")]),
    ]
    for mk, topos in ext_configs:
        for kind, n, gpu in topos:
            t = M.topology(kind, n, gpu)
            row = []
            for wd in M.WEIGHT_DTYPES:
                mdl = M.with_weight_dtype(MODELS[mk], wd)
                pool = M.kv_pool_tokens(mdl, t)
                draw = int(4000 + pool / 8_000)   # big pools hold thousands of sessions
                p5, p50, _ = M.warm_capacity(mdl, t, w0, n_iter=400, draw=draw)
                row.append(f"{wd:5}: pool={pool / 1e6:6.2f}M warm p5={p5:5.0f} p50={p50:5.0f}")
            print(f"  {mk:7} {t.name:16} " + " | ".join(row))

    print("\n== New models on H200 (where FP8 weights fit) ==")
    for mk, kind, n in [("MM35", "tp", 2), ("MM35", "tp", 4), ("GLM52", "tp", 8)]:
        t = M.topology(kind, n)
        mdl = MODELS[mk]
        pool = M.kv_pool_tokens(mdl, t)
        p5, p50, _ = M.warm_capacity(mdl, t, w0, n_iter=600, draw=6000)
        _, g50, _ = M.warm_capacity(mdl, t, w0, n_iter=600, draw=6000, which="gpu")
        _, v, _, _ = M.decode_curves(mdl, t, w0, [max(int(g50), 1)], n_iter=800)
        print(f"  {mk:7} {t.name:16} pool={pool / 1e6:6.2f}M  warm p5={p5:4.0f} p50={p50:4.0f}  "
              f"v@warm={v[0]:6.0f} tok/s")

    print("\n== NVFP4 decode effect on MoE (35B-A3B, 1xB300, per-user p50 tok/s) ==")
    print("  NVFP4 shrinks expert reads 1.78x but the BF16-kept blocks (DeltaNet,")
    print("  lm_head, router) make the FIXED per-step read 1.7x heavier -> slower at")
    print("  low concurrency, faster once expert reads dominate (research/nvfp4.md 6.1)")
    t_b1 = M.topology("tp", 1, "B300")
    for n in (1, 4, 16, 64):
        _, a, _, _ = M.decode_curves(MODELS["35BA3B"], t_b1, w0, [n], n_iter=2000)
        m4 = M.with_weight_dtype(MODELS["35BA3B"], "nvfp4")
        _, b, _, _ = M.decode_curves(m4, t_b1, w0, [n], n_iter=2000)
        print(f"  mns={n:3d}  fp8={a[0]:6.0f}  nvfp4={b[0]:6.0f}  ({(b[0] / a[0] - 1) * 100:+.0f}%)")

    print("\n== GLM-5.2 sparse-attention decode: DSA pricing vs dense-read pricing ==")
    print("  DSA reads top-2048 tokens/layer + an indexer scan instead of the full")
    print("  cache; the dense-read row shows what the same bytes would cost if")
    print("  decode streamed the whole cache (the study's default pricing).")
    t_h8 = M.topology("tp", 8)
    glm_dense = dataclasses.replace(MODELS["GLM52"], kv_decode_bpt=None, kv_decode_const=0.0)
    for n in (16, 64, 120):
        _, a, _, _ = M.decode_curves(MODELS["GLM52"], t_h8, w0, [n], n_iter=1500)
        _, b, _, _ = M.decode_curves(glm_dense, t_h8, w0, [n], n_iter=1500)
        print(f"  8xH200 mns={n:3d}  DSA={a[0]:5.0f} tok/s  dense-read={b[0]:5.0f} tok/s")

    print("\n== B300 sensitivity: Hopper unit-convention reserve transfer ==")
    print("  If the H200's vendor '141 GB' understates usable bytes ~7% (documented")
    print("  for the H100), the transferred reserve under-counts true overhead by")
    print("  ~9.8e9 B/GPU and B300 pools here are optimistic by that much:")
    for mk, n in [("35BA3B", 1), ("GLM52", 4)]:
        t = M.topology("tp", n, "B300")
        mdl = MODELS[mk]
        pool = M.kv_pool_tokens(mdl, t)
        pool_adj = pool - n * 9.8e9 / mdl.kv_bpt
        print(f"  {mk:7} {t.name:16} central={pool / 1e6:6.2f}M  "
              f"adjusted={pool_adj / 1e6:6.2f}M  ({(pool_adj / pool - 1) * 100:+.1f}%)")


if __name__ == "__main__":
    main()
