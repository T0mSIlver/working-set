"""Regenerate every number quoted in docs/scenarios.md.

Run:  python scripts/tables.py
All tables print in the order they appear in the doc, so the doc can be
diffed against this output after any model change.
"""
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
    print("  warm      = reusable sessions cached between turns (p50, one cache)")
    print("  warm_user = user-class sessions only (the 'distinct users kept warm' number)")
    print("  v@warm    = per-user p50 tok/s if ALL warm sessions decode at once (100% duty)")
    print("  mns@40    = max concurrent decoders at >=40 tok/s p50 (speed bound alone)")
    print("  -> the binding constraint is min(warm, mns@40); duty<100% relaxes only mns@40")
    mc = mean_context(w0)
    print(f"  mean sampled context = {mc / 1000:.1f}k tokens")
    for mk in MODELS_K:
        for tk in TOPOS_K:
            model, topo = MODELS[mk], TOPOLOGIES[tk]
            # n_iter matches the warm-capacity table so both quote the same p50
            warm = M.warm_capacity(model, topo, w0, n_iter=1500)[1]
            warm_u = M.warm_capacity(model, topo, w0, n_iter=1500, which="user")[1]
            _, v_at_warm, _, _ = M.decode_curves(model, topo, w0, [int(warm)], n_iter=1000)
            m40 = max_mns_at_floor(model, topo, w0, 40)
            per_cache = " (per replica)" if topo.kind == "dp" else ""
            print(f"  {mk:7} {tk:12} warm={warm:4.0f} (user {warm_u:4.0f}){per_cache}  "
                  f"v@warm={v_at_warm[0]:5.0f} tok/s  mns@40={m40:3d}")

    print("\n== Sensitivity: fp32 DeltaNet state (35B-A3B, warm p50) ==")
    m = MODELS["35BA3B"]
    import dataclasses
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


if __name__ == "__main__":
    main()
