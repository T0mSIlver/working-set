"""
Generalized capacity + decode-speed model for the GPU-scaling scenarios.

This is the single source of truth for the extended study. It keeps the SAME
methodology as the baseline scripts (warm_whisker.py / real_mns.py) and
generalizes it along the axes we now want to explore:

  * model         : 27B (dense sibling), 35B-A3B (MoE, ~3B active),
                    Mistral-Medium-3.5-128B (dense GQA), GLM-5.2 (744B-A40B
                    MoE, MLA + DeepSeek Sparse Attention)
  * gpu           : H200 (calibrated baseline) or B300 (Blackwell Ultra)
  * weight dtype  : FP8 (baseline) or NVFP4 (B300-only; weights, never KV)
  * topology      : N GPUs, tensor-parallel (one shared cache) or data-parallel
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
# ----------------------------------------------------------------------------
# The study started H200-only; the GPU is now a selectable axis. Constants and
# provenance for the B300 (Blackwell Ultra): research/gpu_b300.md. The
# activation/workspace reserve is calibrated on the H200 anchor and applied
# PER GPU unchanged on other parts (see ACT_RESERVE below) — an assumption,
# flagged in docs/scenarios.md, until a B300 measurement exists.
#
# supports_nvfp4: whether the part has native FP4 tensor cores. NVFP4 weights
# are gated to GPUs with this flag (a deliberate serving-policy choice: vLLM
# can emulate FP4 on Hopper via Marlin, but we model native-only deployments).
# ============================================================================
@dataclass(frozen=True)
class GPU:
    name: str
    vram: float            # HBM bytes (vendor decimal convention)
    hbm_bw: float          # HBM bandwidth, bytes/s
    supports_nvfp4: bool   # native FP4 tensor cores (Blackwell+)


GPUS = {
    "H200": GPU("H200", 141e9, 4.8e12, supports_nvfp4=False),
    # Blackwell Ultra: 288 GB HBM3e, 8 TB/s, native FP4 tensor cores. The
    # H200-calibrated reserve is applied per B300 unchanged (flagged
    # assumption — see research/gpu_b300.md #3: if Hopper's vendor "141 GB"
    # understates usable bytes by ~7% as documented for the H100, B300 pools
    # here are ~10 GB/GPU optimistic; a real B300 startup log would pin it).
    "B300": GPU("B300", 288e9, 8.0e12, supports_nvfp4=True),
}

VRAM_PER_GPU   = GPUS["H200"].vram    # calibration anchor GPU (baseline study)
HBM_BW         = GPUS["H200"].hbm_bw
TP_EFFICIENCY  = 0.90       # tensor-parallel comm/overhead haircut on aggregate BW

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
    weight_dtype: str = "fp8"   # set via with_weight_dtype(); gates GPU choice
    # NVFP4-checkpoint weight bytes (w_resident, w_decode_shared,
    # w_route_pertok, w_route_total), or None if no NVFP4 variant exists.
    # Derivations + provenance: research/nvfp4.md. NVFP4 changes WEIGHT bytes
    # only — kv_bpt and the DeltaNet state are untouched (no 4-bit KV: vLLM's
    # nvfp4 KV cache is not yet stable, see the research note).
    nvfp4_w: tuple | None = None
    # Sparse-attention decode pricing (GLM-5.2 / DSA). By default decode reads
    # the FULL cached context (kv_decode_bpt=None -> kv_bpt). A DSA model
    # instead reads kv_decode_bpt bytes per CONTEXT token (the indexer scan)
    # plus kv_decode_const bytes per ACTIVE SEQUENCE (the top-k sparse read).
    # Storage (kv_bpt) is unaffected. research/model_glm52.md #3.
    kv_decode_bpt: float | None = None
    kv_decode_const: float = 0.0
    # False for models vLLM can only serve with a quantized KV cache
    # (GLM-5.2's DSA path asserts fp8) — with_kv_dtype("fp16") then raises.
    kv_fp16_ok: bool = True
    # Largest max_seq_len the study allows for this model (tokens). The
    # workload cap (Workload.cap) may not exceed it — warm_capacity and
    # decode_curves raise otherwise. Owner decision 2026-07: 1,048,576 for
    # the Qwens (262,144 native, 1M via YaRN rope scaling) and GLM-5.2
    # (1M native); Mistral-Medium-3.5 stays at its 262,144 model max.
    max_ctx: float = 262_144

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
        # nvidia/Qwen3.6-27B-NVFP4 (~22 GB checkpoint vs 27.8 GB FP8-raw),
        # scaled by the baseline's as-deployed convention: 28.8 GiB x 22/27.8
        # = 22.8 GiB. Dense: decode reads everything. See research/nvfp4.md 6.2.
        nvfp4_w=(24.47e9, 24.47e9, 0.0, 0.0),
        max_ctx=1_048_576,               # 262,144 native; 1M via YaRN (owner decision)
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
        # RedHatAI/Qwen3.6-35B-A3B-NVFP4 recipe: experts + full-attn NVFP4
        # (0.5625 B/param), but embed/lm_head/DeltaNet/router stay BF16
        # (2 B/param) — so the SHARED per-step read is 1.7x HEAVIER than FP8
        # while the routed-expert read is 1.78x lighter. research/nvfp4.md 6.1.
        nvfp4_w=(22.92e9,                # 3.10x vs BF16 (reported ~3.06x)
                 3.308e9,                # attn+shared-exp NVFP4; DN+lm_head+router BF16
                 566_231_040,            # 1_006_632_960 x 0.5625
                 18_119_393_280),        # 32_212_254_720 x 0.5625 (kink stays n=32)
        max_ctx=1_048_576,               # 262,144 native; 1M via YaRN (owner decision)
    ),
    # Dense 128B, open weights (2026-04), 88 uniform full-attention GQA layers
    # (8 KV heads x 128) — the study's KV-hungriest model: 176 KiB/token FP8,
    # no recurrent state, no MoE. No MTP module -> speculative default 1.0x
    # (an external EAGLE-v1 draft exists; its speedup is unmeasured here).
    # Constants: research/model_mistral_medium35.md.
    "MM35": Model(
        name="Mistral-Medium-3.5-128B (dense)",
        kv_bpt=180_224,                  # 88 layers x 8 KV heads x 128 x 2(K,V) x 1B
        deltanet_state=0.0,              # pure attention, no recurrent state
        w_resident=133.6e9,              # 124.43 GiB as-shipped FP8 checkpoint
        w_decode_shared=125.0e9,         # 88 FP8 layers + BF16 lm_head; vision tower not read
        w_route_pertok=0.0,
        w_route_total=0.0,
        mtp=1.0,                         # no MTP module (EAGLE draft is external)
        # nvidia/Mistral-Medium-3.5-128B-NVFP4 mixed recipe: MLP 4-86 NVFP4,
        # edge MLP + all attention FP8, lm_head BF16 (research note #3)
        nvfp4_w=(92.7e9, 86.6e9, 0.0, 0.0),
        max_ctx=262_144,                 # hard model max (YaRN x64 over a 4k base)
    ),
    # MoE 744B-A40B (753B incl. MTP), MLA + DeepSeek Sparse Attention, open
    # weights (2026-06). Cached: 576-B MLA latent/layer + 132-B indexer keys
    # on 21+1 layers = 47.3 KiB/token fp8; vLLM REQUIRES fp8 KV on this path
    # (kv_fp16_ok=False). Decode is sparse: indexer scans the context at
    # 2,772 B/token while attention reads only top-2048 tokens/layer.
    # Constants: research/model_glm52.md.
    "GLM52": Model(
        name="GLM-5.2 (MoE 744B-A40B, MLA+DSA)",
        kv_bpt=48_408,                   # 79 x 576 (MLA latent) + 22 x 132 (indexer)
        deltanet_state=0.0,              # MLA is cached attention, no recurrent state
        w_resident=755.5e9,              # official FP8 ckpt: 753.3e9 params + BF16 excess
        w_decode_shared=18.92e9,         # MLA + indexers + dense MLP + shared exp + lm_head
        w_route_pertok=22_649_241_600,   # 8 experts x (3x6144x2048) x 75 MoE layers, FP8
        w_route_total=724_775_731_200,   # 256 experts (saturates at n=32, like 35BA3B)
        mtp=1.7,                         # MTP module (5 drafts); transplanted fit, unmeasured
        # nvidia/GLM-5.2-NVFP4: ONLY routed experts NVFP4; attn/shared/dense/
        # embeddings/lm_head/MTP stay BF16. Derived 464.8e9 B matches the vLLM
        # recipe's "~465 GB" within 0.05% (research/model_glm52.md #4).
        nvfp4_w=(464.8e9, 35.30e9,
                 12_740_198_400,          # 22_649_241_600 x 0.5625
                 407_686_348_800),        # 724_775_731_200 x 0.5625 (kink n=32)
        kv_decode_bpt=2_772,             # 21 indexer layers x 132 B per context token
        kv_decode_const=92.0e6,          # 78 layers x top-2048 x 576 B per active seq
        kv_fp16_ok=False,                # vLLM DSA path asserts a quantized KV cache
        max_ctx=1_048_576,               # native 1M context (theta 8e6)
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
    if not model.kv_fp16_ok:
        raise ValueError(
            f"{model.name}: FP16 KV is not servable (vLLM's sparse-MLA/DSA "
            "path requires a quantized KV cache; see research/model_glm52.md)")
    return replace(model, kv_bpt=model.kv_bpt * 2, name=model.name + " [FP16 KV]")


# ---- weight-dtype switch ----------------------------------------------------
# All MODELS constants are the FP8 serving checkpoints (the study baseline).
# "nvfp4" swaps in the per-model NVFP4-checkpoint weight bytes (derived in
# research/nvfp4.md and the per-model notes). Weights ONLY: kv_bpt and the
# recurrent state are untouched (4-bit KV is deliberately not modelled), and
# the FP8-anchored reserve calibration never re-runs. The resulting model is
# only usable on GPUs with native FP4 (check_dtype_supported gates it).
WEIGHT_DTYPES = ("fp8", "nvfp4")

def with_weight_dtype(model: Model, weight_dtype: str) -> Model:
    """Return `model` configured for the given weight quantization."""
    if weight_dtype not in WEIGHT_DTYPES:
        raise ValueError(
            f"weight_dtype must be one of {WEIGHT_DTYPES}, got {weight_dtype!r}")
    if weight_dtype == "fp8":
        # no fp8_w constants are kept, so an NVFP4 model cannot be converted
        # back — fail loudly instead of silently returning NVFP4 bytes
        if model.weight_dtype != "fp8":
            raise ValueError(f"{model.name}: cannot convert back to FP8; "
                             "start from the MODELS[...] base entry")
        return model
    if model.nvfp4_w is None:
        raise ValueError(f"{model.name}: no NVFP4 checkpoint constants "
                         "(see research/nvfp4.md)")
    wr, wds, wpt, wtot = model.nvfp4_w
    return replace(model, w_resident=wr, w_decode_shared=wds,
                   w_route_pertok=wpt, w_route_total=wtot,
                   weight_dtype="nvfp4", name=model.name + " [NVFP4]")


# ---- weight-dtype x GPU gating ----------------------------------------------
# NVFP4 weights are only allowed on GPUs with native FP4 tensor cores (B300);
# H-generation GPUs are deliberately excluded even where emulation kernels
# exist (see research/nvfp4.md). Every capacity/decode entry point calls this.
def check_cap_allowed(model: Model, wl) -> None:
    """The workload's max_seq_len cap may not exceed the model's context."""
    if wl.cap > model.max_ctx:
        raise ValueError(
            f"cap={wl.cap:.0f} exceeds {model.name}'s max context "
            f"({model.max_ctx:.0f} tokens)")


def check_dtype_supported(model: Model, topo: Topology) -> None:
    if model.weight_dtype == "nvfp4" and not topo.gpu.supports_nvfp4:
        raise ValueError(
            f"NVFP4 weights ({model.name}) require a GPU with native FP4 "
            f"tensor cores; {topo.gpu.name} is not one (this study gates "
            "NVFP4 to the B300 — no Hopper emulation path is modelled)")


# ============================================================================
# TOPOLOGY  — arbitrary numbers of GPUs, tensor- or data-parallel
# ============================================================================
@dataclass
class Topology:
    name: str
    n_gpu: int
    kind: str            # "single" | "tp" | "dp"
    replicas: int        # independent KV caches (1 for single/tp, n for dp)
    gpu: GPU = GPUS["H200"]


def topology(kind: str, n_gpu: int, gpu: str = "H200") -> Topology:
    """Build a topology for `n_gpu` GPUs of the given part (default H200).

    kind="tp" : ONE engine — weights sharded (stored once), ONE shared prefix
                cache, bandwidth = n x HBM x tp_efficiency(n).
    kind="dp" : n independent 1-GPU replicas — aggregate serving scales by n,
                but the prefix cache SPLITS (system-wide warm capacity is
                n x per-replica, and only with session-sticky routing).
    n_gpu=1 collapses both to the single-GPU baseline.
    """
    if n_gpu < 1 or int(n_gpu) != n_gpu:
        raise ValueError(f"n_gpu must be a positive integer, got {n_gpu!r}")
    if gpu not in GPUS:
        raise ValueError(f"gpu must be one of {tuple(GPUS)}, got {gpu!r}")
    g = GPUS[gpu]
    n_gpu = int(n_gpu)
    if n_gpu == 1:
        return Topology(f"1x{g.name}", 1, "single", 1, g)
    if kind == "tp":
        return Topology(f"{n_gpu}x{g.name} tensor-par", n_gpu, "tp", 1, g)
    if kind == "dp":
        return Topology(f"{n_gpu}x{g.name} data-par", n_gpu, "dp", n_gpu, g)
    raise ValueError(f"kind must be 'tp' or 'dp', got {kind!r}")


def tp_efficiency(n_gpu: int) -> float:
    """Aggregate-bandwidth efficiency of TP over n GPUs.

    ASSUMPTION: the baseline's 0.90 haircut is applied PER DOUBLING
    (0.90^log2(n)): TP2 -> 0.90 (the original assumption, unchanged),
    TP4 -> 0.81, TP8 -> 0.73. Real efficiency depends on interconnect and
    kernel overlap; treat >2 GPUs as a projection needing measurement.
    """
    if n_gpu <= 1:
        return 1.0
    return TP_EFFICIENCY ** np.log2(n_gpu)


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
    """KV-cache capacity of ONE cache (tokens), from the transparent memory model.

    single : VRAM - weights - reserve
    tp     : n GPUs act as one engine; weights sharded (counted once),
             reserve per GPU -> pool = (n*VRAM - weights - n*reserve)
    dp     : each replica is a single-GPU engine; this returns the PER-REPLICA pool

    The activation reserve is the H200-calibrated ACT_RESERVE applied per GPU
    on every part (see the reserve-transfer assumption in docs/scenarios.md).
    """
    check_dtype_supported(model, topo)
    vram = topo.gpu.vram
    if topo.kind == "tp":
        pool_bytes = topo.n_gpu * vram - model.w_resident - topo.n_gpu * ACT_RESERVE
    else:  # single or per-replica of dp
        pool_bytes = vram - model.w_resident - ACT_RESERVE
    return max(pool_bytes, 0.0) / model.kv_bpt


def effective_bw(topo: Topology) -> float:
    """Effective decode bandwidth seen by ONE engine/replica."""
    if topo.kind == "tp":
        return topo.n_gpu * topo.gpu.hbm_bw * tp_efficiency(topo.n_gpu)
    return topo.gpu.hbm_bw  # single, or per-replica of dp


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
    check_cap_allowed(model, wl)
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
    check_dtype_supported(model, topo)
    check_cap_allowed(model, wl)
    rng = np.random.default_rng(seed)
    bw = effective_bw(topo)
    # dense-attention models read every cached token per step (kv_bpt); a
    # sparse-attention model (GLM-5.2/DSA) reads kv_decode_bpt per context
    # token (indexer scan) plus a constant top-k read per active sequence
    kv_read_bpt = model.kv_bpt if model.kv_decode_bpt is None else model.kv_decode_bpt
    p5, p50, p95, agg = [], [], [], []
    for n in mns_range:
        full, _, _, _ = wl.sample(rng, (n_iter, n))
        kv_bytes = full.sum(axis=1) * kv_read_bpt + n * model.kv_decode_const
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

    # ---- B300 GPU + NVFP4 weight dtype + new models ------------------------
    b1, b4 = topology("tp", 1, "B300"), topology("tp", 4, "B300")
    mm, glm = MODELS["MM35"], MODELS["GLM52"]

    # B300: 288 GB / 8 TB/s; pools grow vs H200 at equal weights; NVFP4 flag
    assert GPUS["B300"].supports_nvfp4 and not GPUS["H200"].supports_nvfp4
    assert kv_pool_tokens(m27, b1) > kv_pool_tokens(m27, t1)
    assert effective_bw(b1) / effective_bw(t1) == 8.0 / 4.8

    # NVFP4 gate: B300-only — H-generation GPUs must raise
    m27_4 = with_weight_dtype(m27, "nvfp4")
    for topo_bad in (t1, tp2):
        try:
            kv_pool_tokens(m27_4, topo_bad); raise AssertionError("expected ValueError")
        except ValueError:
            pass
    assert kv_pool_tokens(m27_4, b1) > kv_pool_tokens(m27, b1)  # smaller weights -> more KV
    assert with_weight_dtype(m27, "fp8") is m27                 # fp8 = identity
    for bad_call in (lambda: with_weight_dtype(replace(m27, nvfp4_w=None), "nvfp4"),
                     lambda: with_weight_dtype(m27_4, "fp8")):  # no way back to fp8
        try:
            bad_call(); raise AssertionError("expected ValueError")
        except ValueError:
            pass

    # NVFP4 identities (research/nvfp4.md, research/model_glm52.md):
    m35_4 = with_weight_dtype(m35, "nvfp4")
    assert m35_4.w_route_total == 32_212_254_720 * 0.5625       # exact 0.5625 B/param
    assert abs(m35_4.w_route_total / m35_4.w_route_pertok - 32) < 1e-9  # kink stays n=32
    assert m35_4.w_resident < m35.w_resident                    # checkpoint shrinks...
    assert m35_4.w_decode_shared > m35.w_decode_shared          # ...but BF16 exclusions
    glm_4 = with_weight_dtype(glm, "nvfp4")                     #    weigh on shared reads
    assert abs(glm_4.w_resident / 465e9 - 1) < 0.005, \
        "GLM-5.2 NVFP4 resident must match the vLLM recipe's ~465 GB"
    assert abs(glm_4.w_route_total / glm_4.w_route_pertok - 32) < 1e-9
    # both Qwen models must be NVFP4-selectable (checkpoints exist for both)
    for mk in ("27B", "35BA3B"):
        assert MODELS[mk].nvfp4_w is not None

    # published-config KV identities for the new models
    assert mm.kv_bpt == 88 * 8 * 128 * 2 * 1                    # 176 KiB/token FP8
    assert glm.kv_bpt == 79 * 576 + 22 * 132                    # MLA latent + indexer
    assert mm.mtp == 1.0, "no MTP module on Mistral Medium 3.5"
    assert mm.deltanet_state == 0.0 and glm.deltanet_state == 0.0

    # KV dtype: Mistral doubles like the Qwens; GLM's DSA path must refuse FP16
    assert with_kv_dtype(mm, "fp16").kv_bpt == 2 * mm.kv_bpt
    try:
        with_kv_dtype(glm, "fp16"); raise AssertionError("expected ValueError")
    except ValueError:
        pass

    # context caps (owner decision 2026-07): Qwens + GLM allow up to 1M
    # (Qwen native 262k, 1M via YaRN); Mistral's hard model max is 262,144
    assert m27.max_ctx == m35.max_ctx == glm.max_ctx == 1_048_576
    assert mm.max_ctx == 262_144
    wl_1m = replace(wl, cap=1_048_576)
    assert warm_capacity(m35, t1, wl_1m, n_iter=40)[1] > 0     # 1M cap runs
    for fn in (lambda: warm_capacity(mm, tp2, wl_1m, n_iter=1),
               lambda: decode_curves(mm, tp2, wl_1m, [1], n_iter=10)):
        try:
            fn(); raise AssertionError("expected ValueError")
        except ValueError:
            pass

    # GLM-5.2 sizing: FP8 weights (755.5e9 B) fit no single GPU — pool clamps
    # to 0 — but 8xH200 TP and 4xB300 hold a real pool, NVFP4 a bigger one
    assert kv_pool_tokens(glm, t1) == 0 and kv_pool_tokens(glm, b1) == 0
    assert kv_pool_tokens(glm, topology("tp", 8)) > 0
    assert kv_pool_tokens(with_weight_dtype(glm, "nvfp4"), b4) > kv_pool_tokens(glm, b4) > 0

    # sparse-attention decode: GLM's DSA pricing must beat the dense-read
    # pricing of the same bytes at long context (that is DSA's entire point)
    glm_dense_read = replace(glm, kv_decode_bpt=None, kv_decode_const=0.0)
    t8 = topology("tp", 8)
    _, p_dsa, _, _ = decode_curves(glm, t8, wl, [64], n_iter=300)
    _, p_dense, _, _ = decode_curves(glm_dense_read, t8, wl, [64], n_iter=300)
    assert p_dsa[0] > p_dense[0], "DSA decode must out-speed full-cache reads"
    # decode monotonicity holds for the new models on hardware they fit
    for mdl, topo_fit in ((mm, tp2), (glm, t8), (with_weight_dtype(glm, "nvfp4"), b4)):
        _, p50n, _, aggn = decode_curves(mdl, topo_fit, wl, [1, 8, 64], n_iter=300)
        assert p50n[0] > p50n[1] > p50n[2] > 0
        assert aggn[2] > aggn[0]

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
            if dtype == "fp16" and not MODELS[mk].kv_fp16_ok:
                continue   # GLM-5.2: FP16 KV not servable on the DSA path
            for tk in TOPOLOGIES:
                p = kv_pool_tokens(with_kv_dtype(MODELS[mk], dtype), TOPOLOGIES[tk])
                print(f"  pool {mk:7} {tk:12} {dtype:5} = {p / 1e6:6.2f} M tokens")
    print("  -- B300, fp8 KV, weight dtype fp8 | nvfp4 --")
    for mk in MODELS:
        for n in (1, 4):
            t = topology("tp", n, "B300")
            pools = []
            for wd in WEIGHT_DTYPES:
                mdl = with_weight_dtype(MODELS[mk], wd)
                pools.append(f"{kv_pool_tokens(mdl, t) / 1e6:6.2f}")
            print(f"  pool {mk:7} {t.name:19} = {' | '.join(pools)} M tokens")


if __name__ == "__main__":
    _selfcheck()
