"""
Generalized capacity + decode-speed model for the GPU-scaling scenarios.

This is the single source of truth for the extended study. It keeps the SAME
methodology as the baseline scripts (warm_whisker.py / real_mns.py) and
generalizes it along the axes we now want to explore:

  * model         : 27B (dense sibling) vs 35B-A3B (MoE, ~3B active)
  * topology      : a DP x TP grid of H200s. A replica is a *group* of TP GPUs,
                    not a single GPU, so configurations whose weights exceed one
                    GPU are expressible (and infeasible ones raise rather than
                    silently reporting an empty pool)
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
from dataclasses import dataclass, field, replace

KIB, MIB, GIB = 1024, 1024**2, 1024**3

# ============================================================================
# HARDWARE
# ============================================================================
VRAM_PER_GPU   = 141e9      # H200 HBM3e, bytes (vendor "141 GB")
HBM_BW         = 4.8e12     # per-GPU HBM bandwidth, bytes/s
TP_EFFICIENCY  = 0.90       # tensor-parallel comm/overhead haircut on aggregate BW

# TP all-reduces fire twice per layer and are latency-critical, so they are only
# cheap inside an NVLink/NVSwitch domain. On an HGX H200 baseboard that domain is
# 8 GPUs; a TP group wider than this crosses the node boundary onto
# InfiniBand/Ethernet, an order of magnitude less bandwidth at much worse latency.
# ASSUMPTION: CROSS_DOMAIN_EFFICIENCY is an *unmeasured* penalty applied per
# doubling beyond the domain. Nothing in this study exercises TP > 8; the constant
# exists so the model cliffs instead of silently extrapolating the in-domain
# haircut forever. DP replicas are unaffected — they do no collective work at
# inference and may span nodes freely.
NVLINK_DOMAIN            = 8
CROSS_DOMAIN_EFFICIENCY  = 0.65

# Calibration anchor for 1xH200 + 27B with FP8 KV. NOTE: this is a PROJECTED
# figure, not a direct measurement — the baseline measured the FP16 pool
# (P in [1139k, 1399k] tokens, best estimate ~1337k) and projected FP8 as ~2x
# plus freed activation memory (see scripts/warm_capacity.py). We anchor the
# memory model to it so ACT_RESERVE is *solved*, not guessed; if the FP8 pool
# is ever measured directly, update this one constant.
BASELINE_POOL_TOKENS_27B_1GPU = 2.77e6

# Measured cross-check (2026-07-22): 27B, FP8 weights, FP16 KV, TP2,
# max_seq_len 184,320. vLLM startup log reported "GPU KV cache size:
# 3,233,564 tokens" (110.59 GiB available per worker; 17.54x concurrency
# = 3,233,564 / 184,320 exactly). Nothing about TP2 or FP16 was fitted, so
# this is an independent check on the reserve + TP arithmetic + FP16
# doubling; _selfcheck asserts the model reproduces it within 1%.
MEASURED_POOL_TOKENS_27B_TP2_FP16 = 3_233_564

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
        deltanet_state=75 * MIB,         # baseline's 75 MiB; bf16 arithmetic (48 DN layers x
                                         # 48 vheads x 128x128 + conv) gives 75.7 MiB

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
        w_resident=35_500_000_000,       # ~35.5B params x 1B FP8: full model (embeddings,
                                         # lm_head, attn/DN, all experts, MTP module)

        w_decode_shared=1_940_000_000,   # attn + deltanet + shared expert + router + lm_head / step
        w_route_pertok=1_006_632_960,    # 8 routed experts x 3.146M x 40 layers, FP8
        w_route_total=32_212_254_720,    # 256 routed experts (saturates at exactly n=32 linear)
        mtp=1.7,                         # MTP module, speedup kept equal to baseline's fit
    ),
}

# ---- MTP speedup <-> per-draft acceptance ------------------------------------
# MTP-2 decodes with k=2 draft tokens, accepted i.i.d. until the first
# rejection: expected tokens/forward = 1 + a + a^2. The study's 1.7x default
# is the baseline's measured fit and corresponds to a per-draft acceptance
# a ~ 47%; a measured 87% acceptance would give ~2.6x. The acceptance is the
# base quantity — the speedup is derived from it.
def mtp_speedup(acceptance: float, k: int = 2) -> float:
    """Expected decode speedup from MTP with k draft tokens at per-draft
    acceptance `acceptance` (accept-until-first-rejection)."""
    if not 0.0 <= acceptance <= 1.0:
        raise ValueError(f"acceptance must be in [0, 1], got {acceptance!r}")
    return float(sum(acceptance ** i for i in range(k + 1)))


# ---- KV-cache dtype switch --------------------------------------------------
# All MODELS constants assume the FP8 KV cache (`--kv-cache-dtype fp8_e4m3`),
# as in the baseline study. FP16 KV doubles the bytes/token, which both halves
# the pool (capacity) and doubles the per-step KV read (decode) — weights and
# the DeltaNet recurrent state are unaffected. The activation-reserve
# calibration always uses the FP8 27B constants (the anchor's definition), so
# switching dtype never re-calibrates the reserve. Sanity: the FP16 27B pool
# comes out at 1.385M tokens, inside the baseline's measured FP16 interval
# [1.139M, 1.399M].
KV_DTYPES = ("fp8", "fp16")

def with_kv_dtype(model: Model, kv_dtype: str) -> Model:
    """Return `model` configured for the given KV-cache dtype."""
    if kv_dtype not in KV_DTYPES:
        raise ValueError(f"kv_dtype must be one of {KV_DTYPES}, got {kv_dtype!r}")
    if kv_dtype == "fp8":
        return model
    return replace(model, kv_bpt=model.kv_bpt * 2, name=model.name + " [FP16 KV]")


# ============================================================================
# TOPOLOGY  — a DP x TP grid of H200s
# ----------------------------------------------------------------------------
# The unit that DP replicates is a *group of TP GPUs*, not a GPU. That
# distinction is invisible while the weights fit in one GPU (the whole baseline
# study), and load-bearing the moment they do not: a model too large for a
# single H200 has no valid `tp=1` configuration at all, and DP must replicate
# whole TP groups. Total GPUs = dp * tp.
#
#   weights : sharded across the tp GPUs of a group -> stored ONCE per group
#   reserve : activation/workspace scratch, charged PER GPU
#   cache   : one KV pool per group; dp groups means dp independent pools
#   bandwidth: a group has tp GPUs' worth, haircut by tp_efficiency(tp)
# ============================================================================
class InfeasibleTopology(ValueError):
    """Raised when a model's weights + reserve do not fit in a replica group.

    Subclasses ValueError so existing `except ValueError` call sites still work.
    """


@dataclass
class Topology:
    name: str
    dp: int                          # replica groups; each owns an independent KV cache
    tp: int                          # GPUs per group (weights sharded across these)
    nvlink_domain: int = NVLINK_DOMAIN

    @property
    def n_gpu(self) -> int:
        """Total GPUs in the deployment."""
        return self.dp * self.tp

    @property
    def replicas(self) -> int:
        """Independent KV caches (== dp)."""
        return self.dp

    @property
    def kind(self) -> str:
        """Legacy label: "single" | "tp" | "dp" | "hybrid"."""
        if self.n_gpu == 1:
            return "single"
        if self.dp == 1:
            return "tp"
        if self.tp == 1:
            return "dp"
        return "hybrid"


def topology_grid(dp: int, tp: int, nvlink_domain: int = NVLINK_DOMAIN) -> Topology:
    """Build an arbitrary DP x TP topology of H200s.

    dp : independent replica groups. They exchange nothing at inference (there
         are no gradients to all-reduce), so they scale throughput linearly but
         SPLIT the prefix cache — system warm capacity is dp x per-group, and
         only with session-sticky routing.
    tp : GPUs per group. Weights are sharded across them and stored once, so
         raising tp is what makes a model fit AND frees VRAM for cache.

    topology_grid(1, n) is pure TP; topology_grid(n, 1) is pure DP; the general
    case (e.g. dp=4, tp=4 on 16 GPUs) is what a model too large for one GPU
    actually requires.
    """
    for label, v in (("dp", dp), ("tp", tp)):
        if v < 1 or int(v) != v:
            raise ValueError(f"{label} must be a positive integer, got {v!r}")
    if nvlink_domain < 1 or int(nvlink_domain) != nvlink_domain:
        raise ValueError(f"nvlink_domain must be a positive integer, got {nvlink_domain!r}")
    dp, tp, nvlink_domain = int(dp), int(tp), int(nvlink_domain)
    if dp * tp == 1:
        name = "1xH200"
    elif dp == 1:
        name = f"{tp}xH200 tensor-par"
    elif tp == 1:
        name = f"{dp}xH200 data-par"
    else:
        name = f"{dp * tp}xH200 DP{dp}xTP{tp}"
    return Topology(name, dp, tp, nvlink_domain)


def topology(kind: str, n_gpu: int, nvlink_domain: int = NVLINK_DOMAIN) -> Topology:
    """Build a single-axis topology for `n_gpu` H200s (the study's original API).

    kind="tp" : ONE engine — weights sharded (stored once), ONE shared prefix
                cache, bandwidth = n x HBM x tp_efficiency(n).
    kind="dp" : n independent 1-GPU replicas — aggregate serving scales by n,
                but the prefix cache SPLITS.
    n_gpu=1 collapses both to the single-GPU baseline.

    For a model that does not fit in one GPU, kind="dp" is not expressible here
    (its replica would be a single GPU); use topology_grid(dp, tp) instead.
    """
    if n_gpu < 1 or int(n_gpu) != n_gpu:
        raise ValueError(f"n_gpu must be a positive integer, got {n_gpu!r}")
    n_gpu = int(n_gpu)
    if kind == "tp":
        return topology_grid(1, n_gpu, nvlink_domain)
    if kind == "dp":
        return topology_grid(n_gpu, 1, nvlink_domain)
    raise ValueError(f"kind must be 'tp' or 'dp', got {kind!r}")


def tp_efficiency(tp: int, nvlink_domain: int = NVLINK_DOMAIN) -> float:
    """Aggregate-bandwidth efficiency of a TP group of `tp` GPUs.

    ASSUMPTION: the baseline's 0.90 haircut is applied PER DOUBLING
    (0.90^log2(tp)): TP2 -> 0.90 (the original assumption, unchanged),
    TP4 -> 0.81, TP8 -> 0.73. Real efficiency depends on interconnect and
    kernel overlap; treat >2 GPUs as a projection needing measurement.

    Beyond `nvlink_domain` GPUs the group leaves the NVSwitch domain and each
    further doubling takes an additional CROSS_DOMAIN_EFFICIENCY penalty. That
    second regime is unmeasured and deliberately pessimistic — its purpose is to
    stop the in-domain haircut from being extrapolated to rack-scale TP, where
    it would badly overstate throughput.
    """
    if tp <= 1:
        return 1.0
    eff = TP_EFFICIENCY ** np.log2(min(tp, nvlink_domain))
    if tp > nvlink_domain:
        eff *= CROSS_DOMAIN_EFFICIENCY ** np.log2(tp / nvlink_domain)
    return float(eff)


def min_tp_for(model: Model) -> int:
    """Smallest TP group size whose VRAM holds `model`'s weights + reserve.

    Each GPU added to a group contributes (VRAM_PER_GPU - ACT_RESERVE) of usable
    space, so this is the point at which a KV pool of non-zero size first exists.
    A deployment wants meaningfully more than this — at exactly min_tp the pool
    is ~empty and no session can be held warm.
    """
    usable = VRAM_PER_GPU - ACT_RESERVE
    if usable <= 0:
        raise InfeasibleTopology(
            f"activation reserve ({ACT_RESERVE / GIB:.1f} GiB) exceeds per-GPU "
            f"VRAM ({VRAM_PER_GPU / GIB:.1f} GiB); no topology can fit any model")
    return int(np.floor(model.w_resident / usable)) + 1


# legacy named topologies (the baseline study's three configurations)
TOPOLOGIES = {
    "1xH200":     topology("tp", 1),
    "2xH200-TP2": topology("tp", 2),
    "2xH200-DP2": topology("dp", 2),
}


# ---- solve the activation reserve from the baseline anchor -------------------
def _act_reserve() -> float:
    m = MODELS["27B"]
    pool_bytes = BASELINE_POOL_TOKENS_27B_1GPU * m.kv_bpt
    return VRAM_PER_GPU - m.w_resident - pool_bytes

ACT_RESERVE = _act_reserve()   # per-GPU activation/workspace/fragmentation reserve


def kv_pool_tokens(model: Model, topo: Topology) -> float:
    """KV-cache capacity of ONE replica group (tokens).

    A group owns `topo.tp` GPUs. The weights are sharded across them and so are
    counted ONCE; the activation reserve is scratch space and is charged per GPU:

        pool = tp*VRAM - weights - tp*reserve

    This is per-group, i.e. per-cache. A dp>1 deployment has `topo.dp` of these,
    each independent — multiply by topo.replicas for the system total (and only
    with session-sticky routing, since a returning user is warm on one group).

    Raises InfeasibleTopology if the group cannot hold the weights + reserve.
    Returning 0 here would be worse than useless: it reads as a valid deployment
    that simply holds no sessions, when in fact the engine would not start.
    """
    pool_bytes = topo.tp * (VRAM_PER_GPU - ACT_RESERVE) - model.w_resident
    if pool_bytes <= 0:
        need = min_tp_for(model)
        raise InfeasibleTopology(
            f"{model.name} does not fit a TP group of {topo.tp} GPU(s): weights "
            f"{model.w_resident / GIB:.1f} GiB + reserve "
            f"{topo.tp * ACT_RESERVE / GIB:.1f} GiB exceed "
            f"{topo.tp * VRAM_PER_GPU / GIB:.1f} GiB of group VRAM, leaving no "
            f"room for KV. Needs tp >= {need} to fit at all (and more than that "
            f"to hold any session warm); try topology_grid(dp, tp) with a larger "
            f"tp rather than replicating single GPUs.")
    return pool_bytes / model.kv_bpt


def fits(model: Model, topo: Topology) -> bool:
    """True if `model` has a non-empty KV pool on one replica group of `topo`."""
    return topo.tp * (VRAM_PER_GPU - ACT_RESERVE) - model.w_resident > 0


def effective_bw(topo: Topology) -> float:
    """Effective decode bandwidth seen by ONE replica group.

    A group is `topo.tp` GPUs, so it sees tp x HBM_BW less the TP haircut. DP
    replicas do not add bandwidth to a group — they add more groups, which is
    accounted for by topo.replicas at the aggregate level (see decode_curves).
    """
    return topo.tp * HBM_BW * tp_efficiency(topo.tp, topo.nvlink_domain)


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
        """Sample `size` requests. Returns (full_len, prefix, is_cold, is_sub).

        full_len : clipped prompt length (tokens)
        prefix   : shared-prefix tokens this request can dedup against (0 if cold)
        is_cold  : True for invalidating/unmatchable requests (no reuse at all)
        is_sub   : True for subagent-class requests
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
        return full, prefix, is_cold, is_sub


# ============================================================================
# CAPACITY  (warm, *reusable* sessions kept in one KV cache)
# ============================================================================
def _warm_once(full, prefix, is_cold, is_sub, pool_tokens, ram_gib,
               model: Model, wl: Workload):
    """Fill one KV cache (+ optional CPU offload) in arrival order; count the
    resident sessions that are *reusable* (i.e. not cold/unmatchable), total,
    user-class only, and GPU-resident only.
    Returns (n_warm_all, n_warm_user, n_warm_gpu, censored).

    CPU offload is *storage*: an offloaded session's KV lives in host RAM and
    cannot decode until it is restored over PCIe (see docs/scenarios.md
    limitations). So `n_warm_gpu` — HBM-resident sessions only — is the count
    that any decode-concurrency figure must be built on; `n_warm_all` is the
    capacity/storage view and is unaffected by that distinction.

    Shared-prefix blocks are reserved once per request class present in the
    mixture. A cold request occupies its FULL length (no dedup) and never
    counts as warm.

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

    # reserve one block per distinct prefix a request class can actually use
    reserved = wl.sys_user
    if not wl.sub_shares_prefix and wl.sub_ratio > 0:
        reserved += wl.sys_sub
    gpu_budget = pool_tokens - reserved

    # 'right': a session whose cumulative cost exactly equals the budget fits
    cs = np.cumsum(gpu_cost)
    n_gpu = int(np.searchsorted(cs, gpu_budget, side="right"))
    n_gpu = min(n_gpu, len(unique))

    n_res = n_gpu
    if ram_gib > 0:
        rem = unique[n_gpu:]
        # CPU cost per offloaded session: KV bytes + DeltaNet state
        cost = rem * model.kv_bpt + model.deltanet_state
        cb = np.cumsum(cost)
        ram_budget = ram_gib * GIB - reserved * model.kv_bpt
        n_cpu = int(np.searchsorted(cb, ram_budget, side="right"))
        n_res = n_gpu + min(n_cpu, len(rem))
    warm = ~is_cold[:n_res]
    censored = n_res >= len(unique)   # budget not exhausted by this draw
    return (int(warm.sum()), int((warm & ~is_sub[:n_res]).sum()),
            int((~is_cold[:n_gpu]).sum()), censored)


def warm_capacity(model: Model, topo: Topology, wl: Workload, ram_gib=0,
                  n_iter=4000, draw=2000, seed=0, which="all"):
    """Monte-Carlo warm *reusable* capacity for ONE cache. Returns (p5,p50,p95).

    which="all"  counts every reusable session, GPU-resident + CPU-offloaded
                 (the capacity / storage view);
    which="user" counts only user-class sessions (the 'distinct users kept
                 warm' planning number);
    which="offload" counts the CPU-offloaded reusable sessions, taken PER DRAW
                 (all - gpu inside the same fill) and percentiled afterwards.
                 Percentile the difference, never difference the percentiles:
                 p5(all) - p5(gpu) mixes draws and counts nothing. 0 at
                 ram_gib=0;
    which="gpu"  counts only sessions whose KV is resident in GPU HBM — the
                 DECODE view. Offloaded sessions are storage: they must be
                 restored over PCIe before they can decode, so any per-user or
                 aggregate decode figure must be computed at a concurrency
                 taken from which="gpu", never which="all". With ram_gib=0 the
                 two coincide.

    ram_gib is the CPU-offload buffer available to THIS cache. A DP deployment
    shares the host buffer across its replicas, so DP callers must pass
    system_ram / topo.replicas (the explorer does this via ramPerCache()).
    """
    valid = ("all", "user", "gpu", "offload")
    if which not in valid:
        raise ValueError(f"which must be one of {valid}, got {which!r}")
    rng = np.random.default_rng(seed)
    pool = kv_pool_tokens(model, topo)
    counts = np.empty(n_iter)
    for i in range(n_iter):
        full, prefix, cold, sub = wl.sample(rng, draw)
        n_all, n_user, n_gpu, censored = _warm_once(full, prefix, cold, sub,
                                                    pool, ram_gib, model, wl)
        if censored:
            raise ValueError(f"draw={draw} too small: budget not exhausted "
                             "(censored result); re-run with a larger draw")
        counts[i] = {"all": n_all, "user": n_user, "gpu": n_gpu,
                     "offload": n_all - n_gpu}[which]
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
        full, _, _, _ = wl.sample(rng, (n_iter, n))
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
    assert m35.w_resident > m35.w_route_total + m35.w_decode_shared, \
        "resident weights must exceed the bytes decode can touch in one step"

    # decode monotonicity + union bracketing
    mns = np.arange(1, 129)
    wl = Workload()
    for mdl in (m27, m35):
        _, p50, _, agg = decode_curves(mdl, t1, wl, [1, 8, 64], n_iter=300)
        assert p50[0] > p50[1] > p50[2] > 0
        assert agg[2] > agg[0]
    for n in (1, 8, 32, 64):
        assert m35.w_decode(n, "coverage") <= m35.w_decode(n, "linear") + 1e-6

    # KV dtype switch: FP16 doubles bytes/token -> pool halves exactly; the
    # FP16 27B pool must land inside the baseline's MEASURED FP16 interval
    m27_16 = with_kv_dtype(m27, "fp16")
    assert m27_16.kv_bpt == 2 * m27.kv_bpt
    assert abs(kv_pool_tokens(m27_16, t1) - kv_pool_tokens(m27, t1) / 2) < 1
    assert 1.139e6 <= kv_pool_tokens(m27_16, t1) <= 1.399e6, \
        "FP16 27B pool should fall in the measured [1139k, 1399k] interval"
    # non-circular cross-check: the 27B/TP2/FP16 pool must reproduce the
    # vLLM-reported 3,233,564 tokens (2xH200 startup log) within 1%
    pool_tp2_16 = kv_pool_tokens(m27_16, tp2)
    assert abs(pool_tp2_16 / MEASURED_POOL_TOKENS_27B_TP2_FP16 - 1) < 0.01, \
        f"TP2 FP16 pool {pool_tp2_16:.0f} should match the measured 3,233,564"
    # MTP acceptance <-> speedup: the 1.7x default corresponds to a ~ 47%
    assert abs(mtp_speedup(0.4747) - 1.7) < 1e-3
    assert mtp_speedup(0.0) == 1.0 and abs(mtp_speedup(0.87) - 2.627) < 1e-3
    assert with_kv_dtype(m27, "fp8") is m27          # fp8 = identity
    _, p16, _, _ = decode_curves(with_kv_dtype(m35, "fp16"), t1, wl, [64], n_iter=300)
    _, p8, _, _ = decode_curves(m35, t1, wl, [64], n_iter=300)
    assert p16[0] < p8[0], "FP16 KV must decode slower (double KV read)"

    # arbitrary-N topologies: TP2 matches the legacy constants; pools grow
    # monotonically with n; DP keeps the per-replica pool at the 1-GPU value
    assert abs(tp_efficiency(2) - TP_EFFICIENCY) < 1e-12
    assert tp_efficiency(1) == 1.0 and tp_efficiency(8) < tp_efficiency(4) < tp_efficiency(2)
    assert topology("tp", 2) == TOPOLOGIES["2xH200-TP2"]
    pools = [kv_pool_tokens(m35, topology("tp", n)) for n in (1, 2, 4, 8)]
    assert pools == sorted(pools) and pools[0] < pools[-1]
    for n in (2, 4, 8):
        dp = topology("dp", n)
        assert dp.replicas == n
        assert abs(kv_pool_tokens(m35, dp) - kv_pool_tokens(m35, t1)) < 1
    for bad in (0, -1, 1.5):
        try:
            topology("tp", bad); raise AssertionError("expected ValueError")
        except ValueError:
            pass

    # ---- DP x TP grid: a replica is a GROUP, not a GPU ----------------------
    # the single-axis API must be exactly the two edges of the grid
    assert topology("tp", 4) == topology_grid(1, 4)
    assert topology("dp", 4) == topology_grid(4, 1)
    assert topology_grid(1, 1) == TOPOLOGIES["1xH200"]
    g = topology_grid(4, 4)
    assert (g.dp, g.tp, g.n_gpu, g.replicas, g.kind) == (4, 4, 16, 4, "hybrid")
    assert topology_grid(1, 2).kind == "tp" and topology_grid(2, 1).kind == "dp"
    # a group's pool depends ONLY on tp; dp replicates that pool untouched
    for tp in (1, 2, 4):
        base = kv_pool_tokens(m35, topology_grid(1, tp))
        for dp in (1, 3, 8):
            assert abs(kv_pool_tokens(m35, topology_grid(dp, tp)) - base) < 1, \
                "per-group pool must be independent of dp"
    # ...and so does its bandwidth: DP adds groups, never widens one
    for dp in (1, 4):
        assert effective_bw(topology_grid(dp, 4)) == effective_bw(topology_grid(1, 4))
    assert effective_bw(topology_grid(1, 4)) > effective_bw(topology_grid(1, 1))
    # DP4xTP4 must hold 4x the system capacity of a single TP4 group
    t_tp4, t_grid = topology_grid(1, 4), topology_grid(4, 4)
    w_tp4 = warm_capacity(m35, t_tp4, wl, n_iter=120, draw=8000)[1]
    w_grid = warm_capacity(m35, t_grid, wl, n_iter=120, draw=8000)[1]
    assert abs(w_grid - w_tp4) < 1e-9, "per-group warm count must match TP4"
    assert abs(t_grid.replicas * w_grid - 4 * w_tp4) < 1e-9
    for bad in ((0, 4), (4, 0), (1.5, 4), (4, -1)):
        try:
            topology_grid(*bad); raise AssertionError("expected ValueError")
        except ValueError:
            pass

    # ---- infeasible topologies RAISE, they do not report an empty pool ------
    # a model whose weights exceed one GPU has no valid tp=1 configuration
    import dataclasses as _dc
    huge = _dc.replace(m35, name="oversize", w_resident=3.0 * VRAM_PER_GPU)
    need = min_tp_for(huge)
    assert need > 3, f"oversize model should need tp>3, got {need}"
    assert not fits(huge, topology_grid(8, 1)), "8 single-GPU replicas must not fit it"
    for tp in range(1, need):
        try:
            kv_pool_tokens(huge, topology_grid(1, tp))
            raise AssertionError(f"expected InfeasibleTopology at tp={tp}")
        except InfeasibleTopology:
            pass
    assert isinstance(InfeasibleTopology("x"), ValueError)  # legacy handlers still catch
    # at min_tp the pool exists but is ~empty; a real deployment needs more
    assert fits(huge, topology_grid(1, need))
    assert kv_pool_tokens(huge, topology_grid(1, need)) < \
           kv_pool_tokens(huge, topology_grid(1, need + 2))
    # every model in the study still fits a single GPU (the baseline regime)
    for mdl in MODELS.values():
        assert min_tp_for(mdl) == 1 and fits(mdl, TOPOLOGIES["1xH200"])

    # ---- TP past the NVLink domain must cliff, not extrapolate --------------
    assert tp_efficiency(8) == tp_efficiency(8, nvlink_domain=8)   # at the edge
    in_dom = tp_efficiency(8) / tp_efficiency(4)                   # a doubling inside
    out_dom = tp_efficiency(16) / tp_efficiency(8)                 # a doubling outside
    assert out_dom < in_dom, "crossing the NVLink domain must cost more than a doubling inside"
    assert abs(out_dom - CROSS_DOMAIN_EFFICIENCY) < 1e-9
    # the in-domain regime is UNCHANGED from the original study
    for n in (1, 2, 4, 8):
        assert abs(tp_efficiency(n) - TP_EFFICIENCY ** np.log2(n)) < 1e-12
    # a smaller domain moves the cliff earlier
    assert tp_efficiency(8, nvlink_domain=4) < tp_efficiency(8, nvlink_domain=8)
    # bandwidth still rises with tp (the cliff is a haircut, not an inversion)
    bws = [effective_bw(topology_grid(1, n)) for n in (1, 2, 4, 8, 16)]
    assert bws == sorted(bws)

    # warm capacity sanity: pool grows => capacity grows
    lo = warm_capacity(m35, t1, wl, n_iter=120)[1]
    hi = warm_capacity(m35, tp2, wl, n_iter=120)[1]
    assert 0 < lo < hi
    # charging the per-session recurrent state must shrink capacity
    import dataclasses
    m35_nostate = dataclasses.replace(m35, deltanet_state=0.0)
    assert warm_capacity(m35_nostate, t1, wl, n_iter=120)[1] > lo
    # user-class count must be below the all-sessions count
    assert warm_capacity(m35, t1, wl, n_iter=120, which="user")[1] < lo
    # CPU offload is STORAGE ONLY: it adds warm sessions in host RAM but must
    # leave the GPU-resident count (the decode-concurrency basis) untouched.
    kw = dict(n_iter=60, draw=16_000)
    g_off = warm_capacity(m35, t1, wl, ram_gib=600, which="gpu", **kw)
    g_non = warm_capacity(m35, t1, wl, ram_gib=0, which="gpu", **kw)
    a_off = warm_capacity(m35, t1, wl, ram_gib=600, which="all", **kw)
    assert np.array_equal(g_off, g_non), \
        "CPU offload must not change the GPU-resident warm count"
    assert a_off[1] > g_off[1], "offload must add warm STORAGE beyond HBM"
    # the offloaded tier is a per-draw statistic, zero without a RAM buffer
    assert np.all(warm_capacity(m35, t1, wl, ram_gib=0, which="offload", **kw) == 0)
    o_off = warm_capacity(m35, t1, wl, ram_gib=600, which="offload", **kw)
    assert 0 < o_off[0] <= o_off[1] <= o_off[2]
    for bad in ("ALL", "cpu", ""):
        try:
            warm_capacity(m35, t1, wl, n_iter=1, which=bad)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass

    print("selfcheck OK")
    print(f"  ACT_RESERVE          = {ACT_RESERVE / GIB:6.2f} GiB")
    for dtype in KV_DTYPES:
        for mk in MODELS:
            for tk in TOPOLOGIES:
                p = kv_pool_tokens(with_kv_dtype(MODELS[mk], dtype), TOPOLOGIES[tk])
                print(f"  pool {mk:7} {tk:12} {dtype:5} = {p / 1e6:6.2f} M tokens")


if __name__ == "__main__":
    _selfcheck()
