"""
Generalized capacity + decode-speed model for the GPU-scaling scenarios.

This is the single source of truth for the extended study. It keeps the SAME
methodology as the baseline scripts (warm_whisker.py / real_mns.py) and
generalizes it along the axes we now want to explore:

  * model         : 27B (dense sibling) vs 35B-A3B (MoE, ~3B active)
  * topology      : 1xH200, 2xH200 tensor-parallel (TP2), 2xH200 data-parallel (DP2)
  * system prompt : shared-prefix size (e.g. 3k vs 15k vs 30k tokens)
  * subagents     : a second request class with its own length distribution,
                    mixed in at a ratio r (subagent requests per user request)
  * invalidation  : a fraction f of requests that can't match ANY cached KV
                    (always cold, no prefix reuse) -> a hit-rate ceiling + churn

The two quantities we care about (same as the baseline) are:

  CAPACITY  (memory)      how many *reusable* sessions stay warm in the KV pool
  CONCURRENCY (decode)    per-user / aggregate tok/s vs max_num_seqs

Everything is derived from a transparent memory model that is *calibrated to
reproduce the baseline's measured 1xH200 + 27B pool of 2.77e6 KV tokens*, so
the 27B/1-GPU path matches the original study by construction and the other
configs fall out of the same arithmetic.

The interactive viz (interactive/index.html) mirrors this math in JS; keep the
two in sync.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field

KIB, MIB, GIB = 1024, 1024**2, 1024**3

# ============================================================================
# HARDWARE
# ============================================================================
VRAM_PER_GPU   = 141e9      # H200 HBM3e, bytes (vendor "141 GB")
HBM_BW         = 4.8e12     # per-GPU HBM bandwidth, bytes/s
TP_EFFICIENCY  = 0.90       # tensor-parallel comm/overhead haircut on aggregate BW

# The baseline measured this for 1xH200 + 27B (FP8 KV). We anchor the memory
# model to it so ACT_RESERVE is *solved*, not guessed.
BASELINE_POOL_TOKENS_27B_1GPU = 2.77e6

# ============================================================================
# MODELS
# ----------------------------------------------------------------------------
# kv_bpt          : KV-cache bytes per token (FP8) from the full-attention layers
# deltanet_state  : per-session constant recurrent-state bytes (Gated DeltaNet)
# w_resident      : FP8 weight bytes that must stay resident in VRAM (all experts)
# w_decode_shared : weight bytes read PER DECODE STEP that are always active
#                   (attention/deltanet + shared expert). Read once per step,
#                   amortized across the whole decode batch.
# w_route_pertok  : routed-expert weight bytes activated by ONE decoding token
# w_route_total   : total routed-expert weight bytes (the saturation ceiling as
#                   the batch's union of experts approaches "all of them")
# mtp             : effective decode speedup from Multi-Token Prediction
#
# Dense model  => w_decode_shared = w_resident, w_route_* = 0 (reads all weights).
# MoE model    => decode reads w_decode_shared + routed-expert union bytes:
#   union="linear"   : min(n*w_route_pertok, w_route_total)   -- no-overlap upper
#                      bound on bytes (CONSERVATIVE: slow-side; default)
#   union="coverage" : w_route_total * (1-(1-k/E)^n)          -- expected union
#                      under uniform independent routing (OPTIMISTIC: real
#                      routing is correlated, so coverage grows slower)
# The two bracket reality; figures show both.
#
# All 35B-A3B constants come from the PUBLISHED Qwen/Qwen3.6-35B-A3B-FP8
# config.json (40 layers = 30 DeltaNet + 10 full-attn, 2 KV heads x 256,
# 256 experts / 8 routed + 1 shared, vocab 248,320). Full derivations and
# the assumption ledger: research/model_35ba3b.md.
# ============================================================================
@dataclass
class Model:
    name: str
    kv_bpt: float
    deltanet_state: float
    w_resident: float
    w_decode_shared: float
    w_route_pertok: float
    w_route_total: float
    mtp: float = 1.7

    @property
    def is_moe(self) -> bool:
        return self.w_route_total > 0

    def w_decode(self, n: int, union: str = "linear") -> float:
        """Weight bytes read per decode step with n concurrent decoders."""
        if not self.is_moe:
            return self.w_decode_shared
        if union == "coverage":
            frac_new = self.w_route_pertok / self.w_route_total   # = k/E
            routed = self.w_route_total * (1.0 - (1.0 - frac_new) ** n)
        else:
            routed = min(n * self.w_route_pertok, self.w_route_total)
        return self.w_decode_shared + routed


MODELS = {
    # Baseline dense sibling — numbers straight from the original study. The
    # published Qwen3.6-27B config (64 layers, interval 4 -> 16 full-attn x
    # 4 KV heads x 256) reproduces the baseline's 32 KiB/token exactly.
    "27B": Model(
        name="Qwen3.6-27B (dense)",
        kv_bpt=32 * KIB,                 # 16 attn layers x 4 KV heads x 256 x 2(K,V) x 1B
        deltanet_state=75 * MIB,         # 48 DN layers x 48 vheads x 128x128 bf16 (+conv) = 75.7 MiB
        w_resident=28.8 * GIB,           # baseline's stated as-deployed FP8 footprint
        w_decode_shared=28.8 * GIB,      # dense: every step reads all weights
        w_route_pertok=0.0,
        w_route_total=0.0,
        mtp=1.7,
    ),
    # MoE model — published Qwen3.6-35B-A3B config (see research/model_35ba3b.md).
    "35BA3B": Model(
        name="Qwen3.6-35B-A3B (MoE, ~3B active)",
        kv_bpt=10_240,                   # 10 full-attn layers x 2 KV heads x 256 x 2(K,V) x 1B
        deltanet_state=33_423_360,       # 30 DN layers x 32 vheads x 128x128 bf16 (+conv) = 31.9 MiB
        w_resident=35_500_000_000,       # ~35.5B params x 1B FP8 (all experts + MTP module)
        w_decode_shared=1_940_000_000,   # attn + deltanet + shared expert + router + lm_head / step
        w_route_pertok=1_006_632_960,    # 8 routed experts x 3.146M x 40 layers, FP8
        w_route_total=32_212_254_720,    # 256 routed experts (saturates at exactly n=32 linear)
        mtp=1.7,                         # MTP module, speedup kept equal to baseline's fit
    ),
}

# ============================================================================
# TOPOLOGY
# ============================================================================
@dataclass
class Topology:
    name: str
    n_gpu: int
    kind: str            # "single" | "tp" | "dp"
    replicas: int        # independent KV caches (1 for single/tp, 2 for dp2)

TOPOLOGIES = {
    "1xH200":     Topology("1xH200",              1, "single", 1),
    "2xH200-TP2": Topology("2xH200 tensor-par",   2, "tp",     1),
    "2xH200-DP2": Topology("2xH200 data-par",     2, "dp",     2),
}


# ---- solve the activation reserve from the baseline anchor -------------------
def _act_reserve() -> float:
    m = MODELS["27B"]
    pool_bytes = BASELINE_POOL_TOKENS_27B_1GPU * m.kv_bpt
    return VRAM_PER_GPU - m.w_resident - pool_bytes

ACT_RESERVE = _act_reserve()   # per-GPU activation/workspace/fragmentation reserve


def kv_pool_tokens(model: Model, topo: Topology) -> float:
    """KV-cache capacity of ONE cache (tokens), from the transparent memory model.

    single : VRAM - weights - reserve
    tp     : both GPUs act as one engine; weights sharded (counted once),
             reserve per GPU -> pool roughly (2*VRAM - weights - 2*reserve)
    dp     : each replica is a 1xH200 engine; this returns the PER-REPLICA pool
    """
    if topo.kind == "tp":
        pool_bytes = topo.n_gpu * VRAM_PER_GPU - model.w_resident - topo.n_gpu * ACT_RESERVE
    else:  # single or per-replica of dp
        pool_bytes = VRAM_PER_GPU - model.w_resident - ACT_RESERVE
    return max(pool_bytes, 0.0) / model.kv_bpt


def effective_bw(topo: Topology) -> float:
    """Effective decode bandwidth seen by ONE engine/replica."""
    if topo.kind == "tp":
        return topo.n_gpu * HBM_BW * TP_EFFICIENCY
    return HBM_BW  # single, or per-replica of dp


# ============================================================================
# WORKLOAD  (user + subagent mixture, with cache-invalidation churn)
# ============================================================================
@dataclass
class Workload:
    # log-normal prompt-length params, expressed as (median tokens, sigma)
    user_median: float = 31_000
    user_sigma: float = 0.81
    sub_median: float = 8_000
    sub_sigma: float = 0.9
    sub_ratio: float = 0.10           # subagent requests per user request (1/10)
    sys_user: float = 15_000          # main-user shared system-prompt prefix (tokens)
    sys_sub: float = 3_000            # subagent shared prefix (tokens); own, leaner block
    sub_shares_prefix: bool = False   # if True, subagents reuse the user prefix
    invalidation: float = 0.01        # fraction of requests that can't match any KV
    cap: float = 180_000              # max_seq_len truncation cap (tokens)
    min_tokens: float = 1_000

    def p_sub(self) -> float:
        return self.sub_ratio / (1.0 + self.sub_ratio)

    def sample(self, rng: np.random.Generator, size: int):
        """Sample `size` requests. Returns (full_len, prefix, is_cold) arrays.

        full_len : clipped prompt length (tokens)
        prefix   : shared-prefix tokens this request can dedup against (0 if cold)
        is_cold  : True for invalidating/unmatchable requests (no reuse at all)
        """
        is_sub = rng.random(size) < self.p_sub()
        # log-normal: sigma is the shape param; scale = median
        lu = rng.lognormal(np.log(self.user_median), self.user_sigma, size)
        ls = rng.lognormal(np.log(self.sub_median), self.sub_sigma, size)
        full = np.where(is_sub, ls, lu)

        sub_prefix = self.sys_user if self.sub_shares_prefix else self.sys_sub
        prefix = np.where(is_sub, sub_prefix, self.sys_user).astype(float)

        # a prompt always contains at least its shared system prefix, never > cap
        lower = np.minimum(prefix, self.cap)
        full = np.clip(full, lower, self.cap)

        is_cold = rng.random(size) < self.invalidation
        prefix = np.where(is_cold, 0.0, prefix)   # cold: matches nothing (occupies full length)
        return full, prefix, is_cold


# ============================================================================
# CAPACITY  (warm, *reusable* sessions kept in one KV cache)
# ============================================================================
def _warm_once(full, prefix, is_cold, pool_tokens, ram_gib, model: Model, wl: Workload):
    """Fill one KV cache (+ optional CPU offload) in arrival order; count how many
    resident sessions are *reusable* (i.e. not cold/unmatchable).

    Shared-prefix blocks are reserved once each (user + subagent) up front.
    A cold request occupies its FULL length (no dedup) and never counts as warm.

    Every resident session additionally holds its constant Gated-DeltaNet
    recurrent state (a warm hit needs the state, not just the attention KV), so
    each session is charged `deltanet_state` in KV-token equivalents. The
    baseline scripts omitted this charge; for the reference workload it costs
    ~10-20% of warm capacity (state/kv_bpt = 2.4k tok-equiv for 27B, 3.3k for
    35B-A3B, vs a median unique length of ~16k).
    """
    unique = np.where(is_cold, full, np.maximum(full - prefix, 0.0))
    state_tok = model.deltanet_state / model.kv_bpt
    gpu_cost = unique + state_tok

    # reserve the distinct prefix blocks once (only those actually in this draw)
    reserved = 0.0
    reserved += wl.sys_user
    if not wl.sub_shares_prefix:
        reserved += wl.sys_sub
    gpu_budget = pool_tokens - reserved

    cs = np.cumsum(gpu_cost)
    n_gpu = int(np.searchsorted(cs, gpu_budget))
    n_gpu = min(n_gpu, len(unique))

    if ram_gib > 0:
        rem = unique[n_gpu:]
        rem_cold = is_cold[n_gpu:]
        # CPU cost per offloaded session: KV bytes + DeltaNet state
        cost = rem * model.kv_bpt + model.deltanet_state
        cb = np.cumsum(cost)
        ram_budget = ram_gib * GIB - reserved * model.kv_bpt
        n_cpu = int(np.searchsorted(cb, ram_budget))
        n_cpu = min(n_cpu, len(rem))
        resident_cold = int(is_cold[:n_gpu + n_cpu].sum())
        return (n_gpu + n_cpu) - resident_cold
    return n_gpu - int(is_cold[:n_gpu].sum())


def warm_capacity(model: Model, topo: Topology, wl: Workload, ram_gib=0,
                  n_iter=4000, draw=2000, seed=0):
    """Monte-Carlo warm *reusable* capacity for ONE cache. Returns (p5,p50,p95)."""
    rng = np.random.default_rng(seed)
    pool = kv_pool_tokens(model, topo)
    counts = np.empty(n_iter)
    for i in range(n_iter):
        full, prefix, cold = wl.sample(rng, draw)
        counts[i] = _warm_once(full, prefix, cold, pool, ram_gib, model, wl)
    return np.percentile(counts, [5, 50, 95])


# ============================================================================
# CONCURRENCY  (per-user / aggregate decode tok/s vs max_num_seqs)
# ============================================================================
def decode_curves(model: Model, topo: Topology, wl: Workload, mns_range,
                  n_iter=3000, seed=0, union="linear"):
    """Per-user tok/s percentiles (p5,p50,p95) and aggregate p50 vs max_num_seqs.

    step_bytes(n) = weights (MoE: shared + expert union, see Model.w_decode)
                  + KV read of all active contexts
                  + DeltaNet recurrent-state read+write for every active
                    sequence (2 x state x n; each decode step updates S).
    The state term was omitted by the baseline decode model; at its mns=6 it is
    <2% of step bytes, but at mns 64+ it is no longer negligible.
    """
    rng = np.random.default_rng(seed)
    bw = effective_bw(topo)
    p5, p50, p95, agg = [], [], [], []
    for n in mns_range:
        full, _, _ = wl.sample(rng, (n_iter, n))
        kv_bytes = full.sum(axis=1) * model.kv_bpt
        state_bytes = 2.0 * n * model.deltanet_state
        step_bytes = model.w_decode(n, union) + kv_bytes + state_bytes
        pu = model.mtp * bw / step_bytes
        a, b, c = np.percentile(pu, [5, 50, 95])
        p5.append(a); p50.append(b); p95.append(c)
        # aggregate tok/s of one engine; for DP the *system* is replicas x this
        agg.append(n * b)
    scale = topo.replicas
    return (np.array(p5), np.array(p50), np.array(p95),
            np.array(agg) * scale)


# ============================================================================
# SELF-CHECKS  (python scripts/scenario_model.py)
# ============================================================================
def _selfcheck():
    m27, m35 = MODELS["27B"], MODELS["35BA3B"]
    t1, tp2 = TOPOLOGIES["1xH200"], TOPOLOGIES["2xH200-TP2"]

    # calibration: 27B/1xH200 reproduces the measured pool by construction
    assert abs(kv_pool_tokens(m27, t1) - BASELINE_POOL_TOKENS_27B_1GPU) < 1, \
        "27B/1xH200 pool must equal the measured 2.77M-token anchor"
    assert ACT_RESERVE > 0, "activation reserve must be positive"

    # 35B-A3B constants: internal identities from the published config
    assert m35.kv_bpt == 10 * 2 * 256 * 2 * 1               # 10 KiB/token FP8
    assert m35.w_route_pertok == 8 * (3 * 2048 * 512) * 40  # 8 routed experts
    assert m35.w_route_total == 256 * (3 * 2048 * 512) * 40 # 256 experts
    assert abs(m35.w_route_total / m35.w_route_pertok - 32) < 1e-9  # kink at n=32
    active = m35.w_decode_shared + m35.w_route_pertok
    assert 2.8e9 < active < 3.1e9, "active bytes/token should be ~3B (published)"
    assert m35.w_resident > m35.w_route_total + m35.w_decode_shared - 509e6, \
        "resident must cover all experts + shared (lm_head aside, embed/MTP extra)"

    # decode monotonicity + union bracketing
    mns = np.arange(1, 129)
    wl = Workload()
    for mdl in (m27, m35):
        _, p50, _, agg = decode_curves(mdl, t1, wl, [1, 8, 64], n_iter=300)
        assert p50[0] > p50[1] > p50[2] > 0
        assert agg[2] > agg[0]
    for n in (1, 8, 32, 64):
        assert m35.w_decode(n, "coverage") <= m35.w_decode(n, "linear") + 1e-6

    # warm capacity sanity: pool grows => capacity grows; state charge shrinks it
    lo = warm_capacity(m35, t1, wl, n_iter=120)[1]
    hi = warm_capacity(m35, tp2, wl, n_iter=120)[1]
    assert 0 < lo < hi

    print("selfcheck OK")
    print(f"  ACT_RESERVE          = {ACT_RESERVE / GIB:6.2f} GiB")
    for mk in MODELS:
        for tk in TOPOLOGIES:
            p = kv_pool_tokens(MODELS[mk], TOPOLOGIES[tk])
            print(f"  pool {mk:7} {tk:12} = {p / 1e6:6.2f} M tokens")


if __name__ == "__main__":
    _selfcheck()
