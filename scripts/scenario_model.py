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
# MoE model    => decode reads w_decode_shared + min(n*w_route_pertok, w_route_total).
#
# NOTE: the 35B-A3B numbers are provisional pending architecture research and
# are overwritten from research/model_35ba3b.json when that file exists.
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
    provisional: bool = False

    @property
    def is_moe(self) -> bool:
        return self.w_route_total > 0

    def w_decode(self, n: int) -> float:
        """Weight bytes read per decode step with n concurrent decoders."""
        return self.w_decode_shared + min(n * self.w_route_pertok, self.w_route_total)


MODELS = {
    # Baseline dense sibling — numbers straight from the original study.
    "27B": Model(
        name="Qwen 3.6 27B (dense)",
        kv_bpt=32 * KIB,                 # 16 attn layers x 4 KV heads x 256 x 2(K,V) x 1B
        deltanet_state=75 * MIB,
        w_resident=28.8 * GIB,
        w_decode_shared=28.8 * GIB,      # dense: every step reads all weights
        w_route_pertok=0.0,
        w_route_total=0.0,
        mtp=1.7,
    ),
    # MoE model — PROVISIONAL placeholders, refined from architecture research.
    "35BA3B": Model(
        name="Qwen 3.6 35B-A3B (MoE, ~3B active)",
        kv_bpt=16 * KIB,                 # provisional: hybrid deltanet => fewer attn layers
        deltanet_state=90 * MIB,         # provisional
        w_resident=36.0 * GIB,           # provisional: 35B FP8, all experts resident
        w_decode_shared=1.6 * GIB,       # provisional: attn/deltanet + shared expert
        w_route_pertok=1.4 * GIB,        # provisional: routed active experts / token
        w_route_total=34.0 * GIB,        # provisional: all routed experts
        mtp=1.7,
        provisional=True,
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
    sys_sub: float = 15_000           # subagent shared prefix (tokens); own block
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
        full = np.clip(full, self.min_tokens, self.cap)

        sub_prefix = self.sys_user if self.sub_shares_prefix else self.sys_sub
        prefix = np.where(is_sub, sub_prefix, self.sys_user).astype(float)

        is_cold = rng.random(size) < self.invalidation
        prefix = np.where(is_cold, 0.0, prefix)   # cold: matches nothing
        return full, prefix, is_cold


# ============================================================================
# CAPACITY  (warm, *reusable* sessions kept in one KV cache)
# ============================================================================
def _warm_once(full, prefix, is_cold, pool_tokens, ram_gib, model: Model, wl: Workload):
    """Fill one KV cache (+ optional CPU offload) in arrival order; count how many
    resident sessions are *reusable* (i.e. not cold/unmatchable).

    Shared-prefix blocks are reserved once each (user + subagent) up front.
    A cold request occupies its FULL length (no dedup) and never counts as warm.
    """
    unique = np.where(is_cold, full, np.maximum(full - prefix, 0.0))

    # reserve the distinct prefix blocks once (only those actually in this draw)
    reserved = 0.0
    reserved += wl.sys_user
    if not wl.sub_shares_prefix:
        reserved += wl.sys_sub
    gpu_budget = pool_tokens - reserved

    cs = np.cumsum(unique)
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
                  n_iter=3000, seed=0):
    """Per-user tok/s percentiles (p5,p50,p95) and aggregate p50 vs max_num_seqs.

    For MoE, weight bytes read per step grow with the batch's expert union
    (min(n*w_route_pertok, w_route_total)); KV bytes = sum of active contexts.
    """
    rng = np.random.default_rng(seed)
    bw = effective_bw(topo)
    p5, p50, p95, agg = [], [], [], []
    for n in mns_range:
        full, _, _ = wl.sample(rng, (n_iter, n))
        kv_bytes = full.sum(axis=1) * model.kv_bpt
        step_bytes = model.w_decode(n) + kv_bytes
        pu = model.mtp * bw / step_bytes
        a, b, c = np.percentile(pu, [5, 50, 95])
        p5.append(a); p50.append(b); p95.append(c)
        # aggregate tok/s of one engine; for DP the *system* is replicas x this
        agg.append(n * b)
    scale = topo.replicas
    return (np.array(p5), np.array(p50), np.array(p95),
            np.array(agg) * scale)
