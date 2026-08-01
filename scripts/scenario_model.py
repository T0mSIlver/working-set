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
    vram: float            # usable HBM bytes (see per-part comments)
    hbm_bw: float          # HBM bandwidth, bytes/s
    supports_nvfp4: bool   # native FP4 tensor cores (Blackwell+)
    # Extra per-GPU reserve applied when the H200-SOLVED reserve transfers to
    # this part. Hopper over-provisions HBM: the H200 actually delivers
    # ~150.75e9 usable bytes against the 141e9 vendor figure the calibration
    # uses (thundergolfer.com, confirmed 2026-07-27). The anchor absorbs that
    # hidden ~9.75e9 into the solved reserve implicitly, so H200 results are
    # exact by construction — but a part WITHOUT the over-provision must add
    # it back explicitly or its pools inherit ~9.75e9/GPU of phantom HBM.
    reserve_extra: float = 0.0
    # GPUs reachable over NVLink/NVSwitch without leaving the node. TP
    # all-reduces fire twice per layer and are latency-critical, so a TP group
    # wider than this crosses onto IB/Ethernet and pays the extra
    # CROSS_DOMAIN_EFFICIENCY penalty (see tp_efficiency). DP replicas are
    # unaffected: they run no collectives at inference and may span nodes.
    # 8 for both parts here — HGX H200 and the 8-GPU HGX B300 baseboard we
    # actually deploy. A GB300 NVL72 rack would be 72; we do not have one.
    nvlink_domain: int = 8
    # DENSE FP8 tensor-core throughput, FLOP/s. Used ONLY by the prefill model
    # (research/prefill.md) — every capacity and decode figure in this study is
    # an HBM roofline and never reads this. "Dense" matters: NVIDIA heads its
    # spec sheets with the 2x structured-sparsity number, which no LLM serving
    # path achieves. See research/prefill.md #1 for the per-part derivation.
    peak_flops_fp8: float = 0.0


GPUS = {
    # peak_flops_fp8: 1,979 TFLOPS DENSE. NVIDIA's H200 datasheet leads with
    # "3,958 TFLOPS FP8" — that figure is WITH 2:4 structured sparsity, which
    # no dense LLM GEMM reaches. Halve it. research/prefill.md #1.
    "H200": GPU("H200", 141e9, 4.8e12, supports_nvfp4=False,
                peak_flops_fp8=1.979e15),
    # Blackwell Ultra: 288 GB HBM3e, 8 TB/s, native FP4 tensor cores.
    # vram: MEASURED — a real BM.GPU.B300.8 nvidia-smi dump (Oracle OCI
    # quickstart, driver 590.48.01) shows 275,040 MiB = 288.4e9 B per GPU:
    # the B300 delivers nominal decimal bytes with NO Hopper-style ~7%
    # over-provision. reserve_extra: the measured H200 hidden margin that the
    # transferred reserve must therefore add back (research/gpu_b300.md #3 —
    # formerly a flagged sensitivity, now the measured central case).
    # peak_flops_fp8: 4.5 PFLOPS dense. The DGX B300 datasheet quotes
    # "72 PFLOPS FP8" for 8 GPUs WITH 2:4 sparsity -> 9 PFLOPS sparse/GPU ->
    # 4.5 dense (corroborated by third-party spec tables listing HGX B300
    # FP8 dense at 4,500 TFLOPS). NOTE the trap this replaces: Blackwell
    # Ultra's 1.5x FP4 uplift (13.5 PFLOPS dense, research/gpu_b300.md) did
    # NOT carry to FP8 — "FP8 = half of FP4", true on Hopper, over-credits
    # the B300 by 1.5x. Never measured here. research/prefill.md #1.
    "B300": GPU("B300", 288.4e9, 8.0e12, supports_nvfp4=True,
                reserve_extra=9.75e9, peak_flops_fp8=4.5e15),
}

VRAM_PER_GPU   = GPUS["H200"].vram    # calibration anchor GPU (baseline study)
HBM_BW         = GPUS["H200"].hbm_bw
TP_EFFICIENCY  = 0.90       # tensor-parallel comm/overhead haircut on aggregate BW
# Extra haircut per GPU-count doubling once a TP group is wider than its part's
# nvlink_domain (i.e. spans nodes). UNMEASURED and deliberately pessimistic —
# see tp_efficiency(). No configuration in this study crosses the boundary.
CROSS_DOMAIN_EFFICIENCY = 0.65

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
    # only — kv_bpt and the DeltaNet state are untouched. The KV cache stays
    # FP8/FP16 by OWNER POLICY: vLLM's nvfp4 KV shipped 2026-05 (Blackwell-
    # datacenter-only, values dequantized to FP8 before attention) but this
    # study does not model it — see research/nvfp4.md #3.
    nvfp4_w: tuple | None = None
    # Sparse-attention decode pricing (GLM-5.2 / DSA). By default decode reads
    # the FULL cached context (kv_decode_bpt=None -> kv_bpt). A DSA model
    # instead reads kv_decode_bpt bytes per CONTEXT token (the indexer scan)
    # plus kv_decode_const bytes per ACTIVE SEQUENCE (the top-k sparse read,
    # scaled by min(len, kv_decode_topk)/kv_decode_topk — a sequence shorter
    # than the top-k window only reads its own tokens).
    # Storage (kv_bpt) is unaffected. research/model_glm52.md #3.
    kv_decode_bpt: float | None = None
    kv_decode_const: float = 0.0
    kv_decode_topk: float | None = None
    # False for models vLLM can only serve with a quantized KV cache
    # (GLM-5.2's DSA path asserts fp8) — with_kv_dtype("fp16") then raises.
    kv_fp16_ok: bool = True
    # Largest max_seq_len the study allows for this model (tokens). The
    # workload cap (Workload.cap) may not exceed it — warm_capacity and
    # decode_curves raise otherwise. Owner decision 2026-07: 1,048,576 for
    # the Qwens (262,144 native, 1M via YaRN rope scaling) and GLM-5.2
    # (1M native); Mistral-Medium-3.5 stays at its 262,144 model max.
    max_ctx: float = 262_144

    # ---- PREFILL constants (research/prefill.md) -----------------------
    # Prefill is COMPUTE-bound, so it is priced in FLOPs, not bytes — the only
    # part of this study that is. These three fields exist solely for that.
    #
    # params_prefill: parameters doing a GEMM per token. Dense: total minus
    #   embeddings (a lookup, no matmul) and lm_head (fires on the LAST token
    #   of a prefill only). MoE: ACTIVE params per token — a routed token
    #   touches k of E experts however long the chunk is, which is why an MoE
    #   prefills far cheaper than its parameter count suggests.
    # attn_layers / attn_d: the quadratic term. Only layers with real
    #   attention pay O(L^2); linear-attention (DeltaNet) layers do not.
    #   attn_d = n_query_heads x head_dim, i.e. the width QK^T and AV run at.
    # Deliberately biased AGAINST the thrash hypothesis: excluding lm_head and
    # embeddings makes prefill look cheaper, so a cold request's cost is a
    # slight UNDER-estimate rather than a convenient over-estimate.
    params_prefill: float = 0.0
    attn_layers: int = 0
    attn_d: float = 0.0

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
        # nvidia/Qwen3.6-27B-NVFP4: MEASURED safetensors total 21,921,428,072 B
        # (2026-07-27 re-verification). Same convention as the FP8 28.8 GiB,
        # which the measured Qwen/Qwen3.6-27B-FP8 checkpoint (30.87e9 B =
        # 28.75 GiB) matches within 0.2% — the old x22/27.8 "as-deployed"
        # scaling was a mis-derivation. Dense: decode reads everything.
        nvfp4_w=(21.92e9, 21.92e9, 0.0, 0.0),
        max_ctx=1_048_576,               # 262,144 native; 1M via YaRN (owner decision)
        # prefill (research/prefill.md #2): 27B dense, less ~2.5e9 of
        # embed + untied lm_head (vocab 248,320 — the family figure the
        # 35B-A3B ledger reads from config.json — x hidden 5,120, x2).
        # 64 layers, interval 4 -> 16 full-attention; 24 Q heads x 256.
        params_prefill=24.5e9, attn_layers=16, attn_d=24 * 256,
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
        # (0.5625 B/param), but embed/lm_head/DeltaNet/router — AND the MTP
        # module (`re:^mtp.*` is in the measured ignore list) — stay BF16,
        # so the SHARED per-step read is 1.7x HEAVIER than FP8 while the
        # routed-expert read is 1.78x lighter. research/nvfp4.md 6.1.
        nvfp4_w=(24.13e9,                # MEASURED LM bytes (2.94x vs BF16 base)
                 3.308e9,                # measured 3.3065e9: attn+shared NVFP4; DN+lm_head BF16
                 566_231_040,            # measured 566,238,720 (+1.4e-5)
                 18_119_393_280),        # measured 18,119,639,040 (kink stays n=32)
        max_ctx=1_048_576,               # 262,144 native; 1M via YaRN (owner decision)
        # prefill: MoE — a token routes to 8 of 256 experts no matter how
        # long the chunk is, so the ACTIVE parameters prefill, not the 35B.
        # From this model's own component ledger (model_35ba3b.md): per-step
        # shared read 1.940e9 MINUS lm_head 0.509e9 (fires once per prefill,
        # not per token) PLUS the 8 routed experts 1.007e9 = 2.44e9. (The
        # published "~3B active" includes embed + lm_head.)
        # 40 layers, 10 full-attention; 16 Q heads x 256.
        params_prefill=2.44e9, attn_layers=10, attn_d=16 * 256,
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
        # edge MLP + all attention FP8, lm_head BF16 (research note #3).
        # Resident = the MEASURED safetensors total (95,224,812,960 B,
        # 2026-07-27 re-verification); decode read measured 86.643e9.
        nvfp4_w=(95.2e9, 86.6e9, 0.0, 0.0),
        max_ctx=262_144,                 # hard model max (YaRN x64 over a 4k base)
        # prefill: the TEXT DECODER only — model_mistral_medium35.md's shard
        # ledger gives 121.8e9 decoder-layer params exactly; embeddings and
        # lm_head (2 x 131,072 x 12,288 = 3.22e9) are outside it, and the
        # ~2.7e9-param vision tower is an ENCODER, never executed when
        # re-prefilling these text contexts (subtracting embed+lm_head from
        # the 128B multimodal total, as this constant first did, left the
        # tower inside the GEMM). ALL 88 layers are full GQA -> the study's
        # heaviest quadratic term by far; 96 Q heads x 128.
        params_prefill=121.8e9, attn_layers=88, attn_d=96 * 128,
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
        kv_decode_topk=2_048,            # ...scaled down for sequences < 2,048 tokens
        kv_fp16_ok=False,                # vLLM DSA path asserts a quantized KV cache
        max_ctx=1_048_576,               # native 1M context (theta 8e6)
        # prefill: MoE — model_glm52.md's derivation gives 39.3e9 active
        # EXCLUDING embeddings (vLLM "39B"); less lm_head 1.903e9 (fires
        # once per prefill) = 37.4e9. (The round "~40B active less embed +
        # lm_head" this constant first used double-trusted the marketing
        # figure over the ledger's own sum.)
        # attn_layers/attn_d price MLA as DENSE attention over all 78 layers:
        # an UPPER BOUND. DSA's top-2048 sparsity is established for decode
        # (kv_decode_*) but its prefill behaviour is not characterised in
        # research/model_glm52.md, so the quadratic term here is pessimistic
        # and flagged. 64 heads x 256 (qk_nope 192 + rope 64 = v_head_dim).
        params_prefill=37.4e9, attn_layers=78, attn_d=64 * 256,
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
# TOPOLOGY  — a DP x TP grid of GPUs
# ----------------------------------------------------------------------------
# The unit DP replicates is a GROUP of `tp` GPUs, not a GPU. That distinction is
# invisible while a model fits in one GPU (the 27B and 35B-A3B on either part)
# and load-bearing wherever one does not:
#
#     min TP to fit FP8 weights      H200    B300
#       Mistral-Medium-3.5-128B         2       1     <- fits a single B300
#       GLM-5.2 (744B-A40B)             7       3
#
# In every cell above where min TP > 1, `topology("dp", n)` — n INDEPENDENT
# SINGLE GPUs — is not a deployment that exists: no count of single GPUs ever
# holds the weights, so the whole DP axis reported a 0 pool. (MM35 on B300 is
# the exception: min TP 1, so pure DP is real there.) Data parallelism in the
# other cells means replicating whole TP groups: on one 8-GPU node GLM-5.2 runs
# DP2xTP4 on B300, and MM35 runs DP4xTP2 on H200. Total GPUs = dp * tp.
#
#   weights  : sharded across a group's tp GPUs -> stored ONCE per group
#   reserve  : activation/workspace scratch, charged PER GPU
#   cache    : one KV pool per group; dp groups means dp independent pools
#   bandwidth: a group has tp GPUs' worth, less tp_efficiency(tp)
# ============================================================================
@dataclass
class Topology:
    name: str
    dp: int                          # replica groups; each owns one KV cache
    tp: int                          # GPUs per group (weights sharded across these)
    gpu: GPU = GPUS["H200"]

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

    @property
    def crosses_node(self) -> bool:
        """True if a single TP group is wider than the part's NVLink domain."""
        return self.tp > self.gpu.nvlink_domain


def topology_grid(dp: int, tp: int, gpu: str = "H200") -> Topology:
    """Build an arbitrary DP x TP topology of `gpu` parts.

    dp : independent replica groups. They exchange NOTHING at inference (there
         are no gradients to all-reduce), so they scale aggregate throughput
         linearly and may span nodes — but the prefix cache SPLITS, so system
         warm capacity is dp x per-group and only with session-sticky routing.
    tp : GPUs per group. Weights are sharded across them and stored once, so
         raising tp is what makes a model FIT and what frees VRAM for cache.

    topology_grid(1, n) is pure TP, topology_grid(n, 1) is pure DP, and the
    general case is what the models too large for one GPU actually require.
    """
    for label, v in (("dp", dp), ("tp", tp)):
        if v < 1 or int(v) != v:
            raise ValueError(f"{label} must be a positive integer, got {v!r}")
    if gpu not in GPUS:
        raise ValueError(f"gpu must be one of {tuple(GPUS)}, got {gpu!r}")
    g = GPUS[gpu]
    dp, tp = int(dp), int(tp)
    if dp * tp == 1:
        name = f"1x{g.name}"
    elif dp == 1:
        name = f"{tp}x{g.name} tensor-par"
    elif tp == 1:
        name = f"{dp}x{g.name} data-par"
    else:
        name = f"{dp * tp}x{g.name} DP{dp}xTP{tp}"
    return Topology(name, dp, tp, g)


def topology(kind: str, n_gpu: int, gpu: str = "H200") -> Topology:
    """Build a single-axis topology for `n_gpu` GPUs of the given part.

    kind="tp" : ONE engine — weights sharded (stored once), ONE shared prefix
                cache, bandwidth = n x HBM x tp_efficiency(n).
    kind="dp" : n independent 1-GPU replicas — aggregate serving scales by n,
                but the prefix cache SPLITS.
    n_gpu=1 collapses both to the single-GPU baseline.

    For a model that does not fit one GPU, kind="dp" is not a real deployment
    (its replica would be a single GPU); use topology_grid(dp, tp) instead.
    """
    if n_gpu < 1 or int(n_gpu) != n_gpu:
        raise ValueError(f"n_gpu must be a positive integer, got {n_gpu!r}")
    n_gpu = int(n_gpu)
    if kind == "tp":
        return topology_grid(1, n_gpu, gpu)
    if kind == "dp":
        return topology_grid(n_gpu, 1, gpu)
    raise ValueError(f"kind must be 'tp' or 'dp', got {kind!r}")


def tp_efficiency(tp: int, nvlink_domain: int = 8) -> float:
    """Aggregate-bandwidth efficiency of a TP group of `tp` GPUs.

    ASSUMPTION: the baseline's 0.90 haircut is applied PER DOUBLING
    (0.90^log2(tp)): TP2 -> 0.90 (the original assumption, unchanged),
    TP4 -> 0.81, TP8 -> 0.73. Real efficiency depends on interconnect and
    kernel overlap; treat >2 GPUs as a projection needing measurement.

    Past `nvlink_domain` GPUs the group leaves the node's NVSwitch fabric and
    each further doubling takes an additional CROSS_DOMAIN_EFFICIENCY penalty.
    That second regime is UNMEASURED and deliberately pessimistic: its purpose
    is to stop the in-node haircut being extrapolated to multi-node TP, where it
    would badly overstate throughput. Both parts here are 8-GPU nodes, so every
    configuration the study reports stays in the first regime.
    """
    if nvlink_domain < 1 or int(nvlink_domain) != nvlink_domain:
        raise ValueError(
            f"nvlink_domain must be a positive integer, got {nvlink_domain!r}")
    if tp <= 1:
        return 1.0
    eff = TP_EFFICIENCY ** np.log2(min(tp, nvlink_domain))
    if tp > nvlink_domain:
        eff *= CROSS_DOMAIN_EFFICIENCY ** np.log2(tp / nvlink_domain)
    return float(eff)


def min_tp_for(model: Model, gpu: str = "H200") -> int:
    """Smallest TP group size whose VRAM holds `model`'s weights + reserve.

    Each GPU in a group contributes (vram - reserve) of usable space, so this is
    where a non-empty KV pool first exists. A real deployment wants meaningfully
    more than this: at exactly min_tp the pool is ~empty and holds no session.
    """
    if gpu not in GPUS:
        raise ValueError(f"gpu must be one of {tuple(GPUS)}, got {gpu!r}")
    g = GPUS[gpu]
    usable = g.vram - (ACT_RESERVE + g.reserve_extra)
    if usable <= 0:
        raise ValueError(f"{g.name}: reserve exceeds VRAM; no topology can fit")
    return int(np.floor(model.w_resident / usable)) + 1


def node_splits(model: Model, gpu: str = "H200", node: int = 8):
    """Every (dp, tp) split of ONE `node`-GPU node that actually fits `model`.

    Returns topologies with dp*tp == node and tp >= min_tp_for(model, gpu),
    widest-DP first — the real menu for a model that needs more than one GPU.
    Empty if the model does not fit the whole node.
    """
    need = min_tp_for(model, gpu)
    return [topology_grid(node // tp, tp, gpu)
            for tp in sorted(d for d in range(1, node + 1) if node % d == 0)
            if tp >= need]


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

    A replica group owns `topo.tp` GPUs. The weights are sharded across them and
    so are counted ONCE; the reserve is scratch space and is charged per GPU:

        pool = tp*vram - weights - tp*reserve

    This is per-GROUP, i.e. per-cache. A dp>1 deployment has topo.replicas of
    them, each independent — multiply for the system total, and only with
    session-sticky routing. Substituting tp=1 recovers the old single/DP-replica
    branch and tp=n the old TP branch, so every published number is unchanged.

    A model too large for its group still clamps to 0 (the study's existing
    "does not fit" sentinel, asserted for GLM-5.2 on one GPU in _selfcheck).
    min_tp_for() reports the group size that would fit.

    The activation reserve is the H200-calibrated ACT_RESERVE applied per GPU
    on every part, plus the part's reserve_extra — the measured H200 HBM
    over-provision the solved reserve implicitly absorbed, added back on parts
    that do not share it (see GPU.reserve_extra and docs/scenarios.md).
    """
    check_dtype_supported(model, topo)
    reserve = ACT_RESERVE + topo.gpu.reserve_extra
    # NOT factored as tp*(vram - reserve) - w: keeping the three terms in the
    # original order makes every pre-existing configuration bit-identical to the
    # pre-grid code (tp=1 reproduces the old single/DP branch, tp=n the old TP
    # branch) rather than merely equal after formatting.
    pool_bytes = topo.tp * topo.gpu.vram - model.w_resident - topo.tp * reserve
    return max(pool_bytes, 0.0) / model.kv_bpt


def effective_bw(topo: Topology) -> float:
    """Effective decode bandwidth seen by ONE replica group.

    A group is `topo.tp` GPUs, so it sees tp x HBM less the TP haircut. DP adds
    groups, it never widens one — topo.replicas enters only in the aggregate
    (see decode_curves).
    """
    return (topo.tp * topo.gpu.hbm_bw
            * tp_efficiency(topo.tp, topo.gpu.nvlink_domain))


# ============================================================================
# PREFILL  — the cost of a cache MISS, and what it does to everyone else
# ----------------------------------------------------------------------------
# Everything above this line prices HBM bandwidth, because decode is memory-
# bound. Prefill is not: a 32k-token chunk reads the weights ONCE and does
# ~2 x params x tokens FLOPs on them, which puts it four orders of magnitude
# above the roofline ridge point. Bytes are free there; FLOPs are the budget.
#
# This section exists to put a number on the study's founding hypothesis —
# "a cold request re-prefills its whole context and briefly thrashes the GPU
# for every active user". Until now that was an assertion. The quantities that
# make it checkable:
#
#   prefill_seconds()      how long one forward pass over T tokens takes
#   thrash_ratio()         cold request cost / warm request cost
#   itl_spike()            what a decoding user sees when a chunk lands in
#                          their batch — the "for every active user" part
#   max_cold_rate()        cold requests/s before prefill alone saturates
#   prefill_duty()         share of the machine spent re-prefilling, vs miss
#                          rate — the curve that decides whether KV capacity
#                          or prefill throughput is the binding constraint
#
# STATUS: analytic, UNVALIDATED. The baseline collected prefill speeds but
# recorded only the ttft < 0.4 x cold warm/cold heuristic, so there is no
# measured prefill number in this repo to check against. MFU is the soft spot
# (see MFU_* below): the plausible range moves every figure here by 2x. Treat
# these as order-of-magnitude bounds that rank configurations, exactly as the
# rest of the study does — not as latency commitments.
# ============================================================================

# Model FLOP Utilisation: achieved FLOPs / dense peak on a large prefill GEMM.
# Not measured here. 45% is a mid-range figure for FP8 attention+MLP prefill on
# Hopper-class parts with TP2 collectives in the loop; the bracket is what
# published vLLM/TensorRT-LLM prefill benchmarks generally span. Every prefill
# number in this study should be read with the bracket, not the point.
MFU_LOW, MFU_DEFAULT, MFU_HIGH = 0.30, 0.45, 0.60


def prefill_flops(model: Model, tokens: float, prior: float = 0.0) -> tuple:
    """(gemm, attention, total) FLOPs to prefill `tokens` in ONE forward pass,
    with `prior` tokens already sitting in the KV cache.

    gemm      = 2 x params_prefill x tokens   (multiply-accumulate = 2 FLOPs)
    attention = 2 x tokens^2 x attn_d x attn_layers        (intra-chunk, causal)
              + 4 x tokens x prior x attn_d x attn_layers  (queries vs cache)
                QK^T and AV are 2 x pairs x d each; causal masking halves the
                intra-chunk pairs, but every new query attends to ALL `prior`
                cached tokens — the KV cache saves recomputing their keys and
                values, not attending over them. Linear-attention (DeltaNet)
                layers contribute nothing quadratic and are already inside
                params_prefill.

    Because pair-counting telescopes (sum of Ti x Pi + Ti^2/2 over any
    partition = L^2/2), the TOTAL attention work of a context is independent
    of how it is chunked. Chunking bounds the per-forward-pass latency (the
    ITL spike a decode batch sees), NOT the total machine time — that is
    exactly what max_num_batched_tokens trades.
    """
    if tokens < 0:
        raise ValueError(f"tokens must be >= 0, got {tokens!r}")
    if prior < 0:
        raise ValueError(f"prior must be >= 0, got {prior!r}")
    if model.params_prefill <= 0:
        raise ValueError(f"{model.name}: no prefill constants "
                         "(params_prefill unset — see research/prefill.md)")
    gemm = 2.0 * model.params_prefill * tokens
    attn = (2.0 * tokens ** 2 * model.attn_d * model.attn_layers
            + 4.0 * tokens * prior * model.attn_d * model.attn_layers)
    return gemm, attn, gemm + attn


def peak_flops(topo: Topology) -> float:
    """Dense FP8 FLOP/s of one replica GROUP, with the same TP haircut the
    bandwidth model uses.

    Reusing tp_efficiency() here is an ASSUMPTION: it was fitted (loosely) to
    bandwidth scaling, and prefill collectives have a different shape —
    prefill all-reduces move more bytes but amortise over far more compute, so
    the real prefill TP haircut is probably GENTLER than 0.90/doubling. Erring
    that way makes prefill look slower and the thrash claim stronger, so the
    conservative choice would be no haircut at all; the study keeps the haircut
    for consistency with every other topology figure and flags it here.

    WEIGHT DTYPE IS NOT PRICED: every prefill figure reads peak_flops_fp8,
    NVFP4 checkpoints included. The NVFP4 recipes are model-specific mixtures
    of W4A4 layers (faster than FP8), FP8 layers, and BF16 layers (SLOWER
    than FP8, ~half rate), so the mixture's true rate can land on either
    side of the FP8 line — unknowable without per-layer benchmarks. For
    NVFP4 configurations these figures are a modeling CHOICE, not a bound;
    the section's "every bias points against the hypothesis" bookkeeping
    cannot be claimed there. Flagged in research/prefill.md #1 and on the
    explorer tile.
    """
    if topo.gpu.peak_flops_fp8 <= 0:
        raise ValueError(f"{topo.gpu.name}: peak_flops_fp8 unset")
    return (topo.tp * topo.gpu.peak_flops_fp8
            * tp_efficiency(topo.tp, topo.gpu.nvlink_domain))


def prefill_seconds(model: Model, topo: Topology, tokens: float,
                    mfu: float = MFU_DEFAULT, prior: float = 0.0) -> float:
    """Wall-clock of ONE prefill forward pass over `tokens`, with `prior`
    tokens already cached (0 = the first chunk of a cold context)."""
    if not 0 < mfu <= 1:
        raise ValueError(f"mfu must be in (0, 1], got {mfu!r}")
    check_dtype_supported(model, topo)
    return prefill_flops(model, tokens, prior)[2] / (peak_flops(topo) * mfu)


def prefill_tokens_per_s(model: Model, topo: Topology, chunk: float,
                         mfu: float = MFU_DEFAULT) -> float:
    """Sustained prefill throughput at a given chunk size (tokens/s).

    Chunk-size dependent: the quadratic term means a bigger chunk prefills a
    token more slowly. vLLM's max_num_batched_tokens sets this.
    """
    return chunk / prefill_seconds(model, topo, chunk, mfu)


def prefill_context_seconds(model: Model, topo: Topology, context: float,
                            chunk: float, mfu: float = MFU_DEFAULT,
                            prior: float = 0.0) -> float:
    """Machine time to prefill a whole `context`, chunked at `chunk` tokens,
    on top of `prior` already-cached tokens.

    Each chunk is priced with the cache it actually attends over, so later
    chunks cost more than earlier ones. The pair-count telescopes: the total
    equals a single unchunked pass exactly (see prefill_flops), making this
    chunk-size INVARIANT — the slider trades spike size, not total work.
    """
    if chunk <= 0:
        raise ValueError(f"chunk must be > 0, got {chunk!r}")
    total, done = 0.0, 0.0
    context = float(context)
    while done < context:
        step = min(chunk, context - done)
        total += prefill_seconds(model, topo, step, mfu, prior=prior + done)
        done += step
    return total


def context_moments(wl, n: int = 200_000, seed: int = 0) -> tuple:
    """(E[L], E[L^2]) of the workload's context length.

    Prefill cost is quadratic in the context, so its EXPECTATION needs the
    second moment — pricing the mean length would underprice the heavy tail
    (E[L^2] >= E[L]^2, strictly so for these lognormal mixtures).
    """
    full, _, _, _ = wl.sample(np.random.default_rng(seed), n)
    return float(full.mean()), float((full.astype(float) ** 2).mean())


def arithmetic_intensity(model: Model, tokens: float) -> float:
    """FLOPs per byte of weight traffic for a prefill of `tokens`.

    Compare against gpu.peak_flops_fp8 / gpu.hbm_bw (the roofline ridge, ~412
    on the H200): far above it means compute-bound, far below means
    memory-bound. Prefill and decode land on opposite sides by ~3 orders of
    magnitude, which is exactly why they interfere so badly when batched
    together — and why the bandwidth-only decode model cannot see it.
    """
    return prefill_flops(model, tokens)[2] / model.w_resident


def ridge_point(gpu: GPU) -> float:
    """FLOP/byte at which a kernel stops being memory-bound on this part."""
    return gpu.peak_flops_fp8 / gpu.hbm_bw


def mean_context(wl, n: int = 200_000, seed: int = 0) -> float:
    """Mean sampled context length of a workload (tokens).

    The MEAN, not the median, is what prefill cost scales with — same reason
    it drives warm capacity (docs/scenarios.md, "What sigma means here").
    """
    full, _, _, _ = wl.sample(np.random.default_rng(seed), n)
    return float(full.mean())


def cold_request_seconds(model: Model, topo: Topology, wl, chunk: float,
                         mfu: float = MFU_DEFAULT) -> float:
    """EXPECTED machine time to serve one CACHE MISS: re-prefill the whole
    context, averaged over the workload's context-length distribution.

    Uses the closed form the telescoping pair-count allows (total FLOPs of a
    chunked context L = gemm(L) + 2 L^2 d n_layers, independent of chunking):
        E[cost] = (2 params E[L] + 2 attn_d attn_layers E[L^2]) / (peak x mfu)
    E[L^2], not E[L]^2 — the quadratic term must be priced on the heavy tail,
    not on the mean draw.
    """
    check_dtype_supported(model, topo)
    if not 0 < mfu <= 1:
        raise ValueError(f"mfu must be in (0, 1], got {mfu!r}")
    el, el2 = context_moments(wl)
    flops = (2.0 * model.params_prefill * el
             + 2.0 * model.attn_d * model.attn_layers * el2)
    return flops / (peak_flops(topo) * mfu)


def warm_request_seconds(model: Model, topo: Topology, turn_tokens: float,
                         chunk: float, mfu: float = MFU_DEFAULT,
                         prior: float = 0.0) -> float:
    """Machine time to serve one WARM HIT: prefill the new turn's suffix on
    top of the `prior` cached context.

    A warm hit is not free — the study is explicit that it still prefills the
    new turn (docs/scenarios.md limitation 9, "Warm != SLA"). Nor is the
    cached context free: the new turn's queries attend over ALL of it (the
    cache spares recomputing its keys/values, not attending over them), so a
    warm hit carries a term LINEAR in the cached length. Callers price it at
    prior = E[L]; the earlier accounting (prior = 0) inflated the thrash
    ratio by making hits look cheaper than the machine serves them.
    """
    return prefill_context_seconds(model, topo, turn_tokens, chunk, mfu,
                                   prior=prior)


def thrash_ratio(model: Model, topo: Topology, wl, turn_tokens: float,
                 chunk: float, mfu: float = MFU_DEFAULT) -> float:
    """How many times more machine time a miss costs than a hit.

    THE number behind the study's hypothesis. Independent of MFU and of the
    GPU part (both cancel) — a property of the workload's context-length
    distribution against the turn length. It is NOT independent of the
    attention model itself: on attention-heavy rows (GLM-5.2's dense-MLA
    upper bound above all) the quadratic term drives both numerator and the
    warm hit's cross term, so those rows inherit research/prefill.md
    weakness #2 rather than escaping it.
    """
    return (cold_request_seconds(model, topo, wl, chunk, mfu)
            / warm_request_seconds(model, topo, turn_tokens, chunk, mfu,
                                   prior=mean_context(wl)))


def itl_spike(model: Model, topo: Topology, wl, n_decode: int, chunk: float,
              mfu: float = MFU_DEFAULT, n_iter: int = 2000, seed: int = 0):
    """What a prefill chunk does to the OTHER users in the batch.

    Returns (decode_ms, mixed_ms, ratio). With chunked prefill, vLLM batches
    the chunk together with the running decodes, so nobody is starved — the
    forward pass containing the chunk simply takes prefill-time instead of
    decode-time, and every one of the `n_decode` users waiting on it sees a
    single inter-token gap that long.

    That is the precise sense in which one cold request "thrashes the GPU for
    every active user": not a stall, a synchronised latency spike whose size
    is the ratio returned here.
    """
    _, p50, _, _ = decode_curves(model, topo, wl, [n_decode], n_iter=n_iter,
                                 seed=seed)
    decode_s = model.mtp / p50[0]          # seconds per token, per user
    # the chunk is priced mid-re-prefill (prior = E[L]/2, the average cache
    # it attends over during a full cold re-prefill); the LAST chunk of a
    # mean-length context runs ~prior = E[L], about twice this cross term
    mixed_s = decode_s + prefill_seconds(model, topo, chunk, mfu,
                                         prior=mean_context(wl) / 2)
    return decode_s * 1e3, mixed_s * 1e3, mixed_s / decode_s


def max_cold_rate(model: Model, topo: Topology, wl, chunk: float,
                  mfu: float = MFU_DEFAULT) -> float:
    """Cold requests/second at which prefill alone consumes the whole machine.

    A hard ceiling the capacity model cannot see: it is set by FLOPs, so no
    amount of KV pool, CPU offload or warm-session headroom raises it. Above
    this rate the deployment is prefill-bound and warm capacity has stopped
    being the binding constraint.
    """
    return 1.0 / cold_request_seconds(model, topo, wl, chunk, mfu)


def prefill_duty(model: Model, topo: Topology, wl, req_rate: float,
                 chunk: float, turn_tokens: float = 0.0,
                 mfu: float = MFU_DEFAULT) -> float:
    """Fraction of the replica group spent prefilling at `req_rate` req/s.

    Counts the cold re-prefills (wl.invalidation of the traffic) plus, if
    turn_tokens > 0, the new-turn prefill every WARM hit still pays. Values
    above 1.0 mean prefill alone oversubscribes the group: the queue grows
    without bound and TTFT diverges, whatever the warm-session count says.
    """
    if req_rate < 0:
        raise ValueError(f"req_rate must be >= 0, got {req_rate!r}")
    f = wl.invalidation
    per_s = f * cold_request_seconds(model, topo, wl, chunk, mfu)
    if turn_tokens > 0:
        per_s += (1 - f) * warm_request_seconds(model, topo, turn_tokens,
                                                chunk, mfu,
                                                prior=mean_context(wl))
    return req_rate * per_s


def breakeven_miss_rate(model: Model, topo: Topology, wl, req_rate: float,
                        chunk: float, turn_tokens: float = 0.0,
                        mfu: float = MFU_DEFAULT) -> float:
    """Miss rate at which prefill duty hits 1.0 for this request rate.

    Returns >1 (i.e. unreachable) when even an all-cold workload fits, and can
    return <=0 when the warm-turn prefill alone already saturates the group.
    Solving duty(f) = 1 for f, with duty linear in f:
        f x cold + (1-f) x warm = 1 / rate
    """
    cold = cold_request_seconds(model, topo, wl, chunk, mfu)
    warm = (warm_request_seconds(model, topo, turn_tokens, chunk, mfu,
                                 prior=mean_context(wl))
            if turn_tokens > 0 else 0.0)
    if cold == warm:
        return float("inf")
    return (1.0 / req_rate - warm) / (cold - warm)


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
    # A zero pool means the weights (+ reserve) don't leave any KV space — no
    # engine can serve here, so capacity is zero REGARDLESS of CPU offload:
    # host RAM is restore-storage for an engine's sessions, and there is no
    # engine to restore into. Without this, a big offload buffer would report
    # hundreds of phantom "warm" sessions on a config that cannot even load.
    if pool <= 0:
        return np.zeros(3)
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
    if kv_pool_tokens(model, topo) <= 0:
        raise ValueError(f"{model.name}: weights do not fit {topo.name} — "
                         "no context can be resident, decode is undefined")
    rng = np.random.default_rng(seed)
    bw = effective_bw(topo)
    # dense-attention models read every cached token per step (kv_bpt); a
    # sparse-attention model (GLM-5.2/DSA) reads kv_decode_bpt per context
    # token (indexer scan) plus a top-k read per active sequence, capped at
    # the sequence's own length when it is shorter than the top-k window
    kv_read_bpt = model.kv_bpt if model.kv_decode_bpt is None else model.kv_decode_bpt
    topk = model.kv_decode_topk
    p5, p50, p95, agg = [], [], [], []
    for n in mns_range:
        full, _, _, _ = wl.sample(rng, (n_iter, n))
        if model.kv_decode_const and topk:
            # DSA reads min(len, top-k) tokens per layer, not a flat top-k
            topk_bytes = (model.kv_decode_const / topk) * \
                np.minimum(full, topk).sum(axis=1)
        else:
            topk_bytes = n * model.kv_decode_const
        kv_bytes = full.sum(axis=1) * kv_read_bpt + topk_bytes
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
    # the single-axis API is exactly the two edges of the grid
    assert topology("tp", 4) == topology_grid(1, 4)
    assert topology("dp", 4) == topology_grid(4, 1)
    assert topology_grid(1, 1) == TOPOLOGIES["1xH200"]
    g = topology_grid(2, 4, "B300")
    assert (g.dp, g.tp, g.n_gpu, g.replicas, g.kind) == (2, 4, 8, 2, "hybrid")
    assert g.gpu is GPUS["B300"] and g.name == "8xB300 DP2xTP4"
    assert topology_grid(1, 2).kind == "tp" and topology_grid(2, 1).kind == "dp"
    # a group's pool and bandwidth depend ONLY on tp; dp replicates them untouched
    for tp in (1, 2, 4):
        base_pool = kv_pool_tokens(m35, topology_grid(1, tp))
        base_bw = effective_bw(topology_grid(1, tp))
        for dp in (1, 3, 8):
            t = topology_grid(dp, tp)
            assert abs(kv_pool_tokens(m35, t) - base_pool) < 1, \
                "per-group pool must be independent of dp"
            assert effective_bw(t) == base_bw, "DP adds groups, never widens one"
    for bad in ((0, 4), (4, 0), (1.5, 4), (4, -1)):
        try:
            topology_grid(*bad); raise AssertionError("expected ValueError")
        except ValueError:
            pass
    try:
        topology_grid(1, 2, "H100"); raise AssertionError("expected ValueError")
    except ValueError:
        pass

    # ---- the grid is what makes DP expressible for the 2026-07 models -------
    # MM35 and GLM-5.2 fit no single H200, so pure DP is a 0 pool at every N --
    # that is the study's existing "does not fit" sentinel, and it stands.
    for mdl in (MODELS["MM35"], MODELS["GLM52"]):
        for n in (1, 2, 4, 8):
            assert kv_pool_tokens(mdl, topology("dp", n)) == 0
    # ...but replicating GROUPS does hold a real pool on one 8-GPU node.
    assert min_tp_for(MODELS["MM35"], "H200") == 2
    assert min_tp_for(MODELS["MM35"], "B300") == 1
    assert min_tp_for(MODELS["GLM52"], "H200") == 7
    assert min_tp_for(MODELS["GLM52"], "B300") == 3
    assert kv_pool_tokens(MODELS["MM35"], topology_grid(4, 2)) > 0    # DP4xTP2
    assert kv_pool_tokens(MODELS["GLM52"], topology_grid(2, 4, "B300")) > 0
    # min_tp is exactly the boundary: one GPU less holds nothing
    for mk, gk in (("MM35", "H200"), ("MM35", "B300"),
                   ("GLM52", "H200"), ("GLM52", "B300")):
        need = min_tp_for(MODELS[mk], gk)
        assert kv_pool_tokens(MODELS[mk], topology_grid(1, need, gk)) > 0
        if need > 1:
            assert kv_pool_tokens(MODELS[mk], topology_grid(1, need - 1, gk)) == 0
    # node_splits offers exactly the fitting divisors of the node, widest DP first
    for mk, gk, want in (("MM35",  "H200", [(4, 2), (2, 4), (1, 8)]),
                         ("MM35",  "B300", [(8, 1), (4, 2), (2, 4), (1, 8)]),
                         ("GLM52", "H200", [(1, 8)]),
                         ("GLM52", "B300", [(2, 4), (1, 8)])):
        got = [(t.dp, t.tp) for t in node_splits(MODELS[mk], gk, node=8)]
        assert got == want, f"node_splits({mk}, {gk}) = {got}, want {want}"
        for t in node_splits(MODELS[mk], gk, node=8):
            assert t.n_gpu == 8 and kv_pool_tokens(MODELS[mk], t) > 0
    # Within a FIXED node of N GPUs the system total has a closed form:
    #
    #     system(tp) = (N/tp) * (tp*(V-R) - W)  =  N*(V-R) - N*W/tp
    #
    # so it depends on tp ONLY through -N*W/tp: monotone non-decreasing always,
    # and strictly increasing exactly when the weight charge W is positive and
    # numerically material (a weightless model is flat, and a tiny-W one can tie
    # under float rounding). Every real model here has material weights, so the
    # totals must rise along node_splits, which is ordered widest-DP first. The
    # margin scales with W: on 8 GPUs, TP8 beats the widest fitting DP split by
    # 1.36x for the 35B-A3B (35.5 GB), 1.91x for MM35 (133.6 GB) and 2.34x for
    # GLM-5.2 (755.5 GB). DP buys cache isolation and routing headroom, never
    # capacity.
    for mk, gk in (("GLM52", "B300"), ("MM35", "H200"),
                   ("MM35", "B300"), ("35BA3B", "H200")):
        sys_tot = [t.replicas * kv_pool_tokens(MODELS[mk], t)
                   for t in node_splits(MODELS[mk], gk, node=8)]
        assert sys_tot == sorted(sys_tot), \
            f"{mk}/{gk}: system total must rise as TP widens, got {sys_tot}"
        if len(sys_tot) > 1:
            assert sys_tot[-1] > sys_tot[0], "TP8 must beat the widest DP split"
        # the closed form must reproduce it exactly
        g_ = GPUS[gk]
        V, R = g_.vram, ACT_RESERVE + g_.reserve_extra
        for t in node_splits(MODELS[mk], gk, node=8):
            closed = (8 * (V - R) - 8 * MODELS[mk].w_resident / t.tp) / MODELS[mk].kv_bpt
            actual = t.replicas * kv_pool_tokens(MODELS[mk], t)
            assert abs(actual - closed) < 1e-6, f"closed form mismatch {mk}/{gk}"
    # ...and the strictness is conditional on a material weight charge: a
    # weightless model is FLAT across every split, not rising. The docs say
    # "strictly raises" for real models only.
    weightless = replace(m35, w_resident=0.0)
    flat = [t.replicas * kv_pool_tokens(weightless, t)
            for t in node_splits(weightless, "H200", node=8)]
    assert len(set(flat)) == 1, f"weightless model must be flat, got {flat}"

    # ---- TP past the node's NVLink domain must cliff, not extrapolate ------
    assert all(g.nvlink_domain == 8 for g in GPUS.values())   # 8-GPU nodes
    assert not topology_grid(1, 8).crosses_node and topology_grid(1, 16).crosses_node
    in_node = tp_efficiency(8) / tp_efficiency(4)             # a doubling inside
    cross = tp_efficiency(16) / tp_efficiency(8)              # a doubling outside
    assert cross < in_node, "crossing nodes must cost more than an in-node doubling"
    assert abs(cross - CROSS_DOMAIN_EFFICIENCY) < 1e-9
    # the in-node regime is UNCHANGED from the pre-grid study
    for n in (1, 2, 4, 8):
        assert abs(tp_efficiency(n) - TP_EFFICIENCY ** np.log2(n)) < 1e-12
    assert tp_efficiency(8, nvlink_domain=4) < tp_efficiency(8, nvlink_domain=8)
    # a non-positive domain must RAISE, not divide by zero or silently mean 8
    for bad in (0, -1, 2.5):
        try:
            tp_efficiency(4, nvlink_domain=bad)
            raise AssertionError(f"expected ValueError for domain={bad}")
        except ValueError:
            pass
    # bandwidth still rises with tp (the cliff is a haircut, not an inversion)
    bws = [effective_bw(topology_grid(1, n)) for n in (1, 2, 4, 8, 16)]
    assert bws == sorted(bws)

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

    # ---- PREFILL (research/prefill.md) ------------------------------------
    CH = 32_768                              # vLLM max_num_batched_tokens
    # roofline sides: prefill is compute-bound by ~2 orders over the ridge,
    # decode is memory-bound by ~1.5 — the whole reason they interfere
    ridge = ridge_point(GPUS["H200"])
    assert arithmetic_intensity(m27, CH) > 100 * ridge, "prefill must be compute-bound"
    dec_bytes = m27.w_decode(64) + 64 * 40_000 * m27.kv_bpt + 2 * 64 * m27.deltanet_state
    assert (2 * m27.params_prefill * 64) / dec_bytes < ridge / 10, \
        "decode must be memory-bound"
    # FLOP split: gemm dominates a 32k chunk, but attention is not negligible
    g, a, tot = prefill_flops(m27, CH)
    assert abs(g - 2 * m27.params_prefill * CH) < 1, "gemm = 2 x params x tokens"
    assert 0.05 < a / tot < 0.20, f"27B attention share at 32k ~12%, got {a / tot:.1%}"
    # ...and it is genuinely quadratic: 2x the chunk, >2x the attention share
    assert prefill_flops(m27, 2 * CH)[1] / prefill_flops(m27, CH)[1] == 4.0
    # a chunk costs ~1 s on TP2 — the headline order of magnitude
    assert 0.5 < prefill_seconds(m27, tp2, CH) < 2.0, "27B TP2 32k chunk ~1 s"
    # MFU bracket brackets it monotonically
    assert (prefill_seconds(m27, tp2, CH, MFU_HIGH)
            < prefill_seconds(m27, tp2, CH, MFU_DEFAULT)
            < prefill_seconds(m27, tp2, CH, MFU_LOW))
    # THE hypothesis: a miss costs an order of magnitude more than a hit, and
    # the ratio is invariant to MFU and to the GPU part (both cancel)
    th = thrash_ratio(m27, tp2, wl, 2_000, CH)
    assert 10 < th < 100, f"cold/warm cost ratio should be ~19x, got {th:.0f}"
    for other in (topology("tp", 1), topology("tp", 1, "B300")):
        assert abs(thrash_ratio(m27, other, wl, 2_000, CH, MFU_LOW) - th) < 0.5, \
            "thrash ratio must not depend on MFU or GPU part"
    # MoE prefills far cheaper than a SMALLER dense model: a token routes to
    # 8 of 256 experts however long the chunk is. This is the extension's most
    # load-bearing new claim, so it is asserted, not just printed.
    assert (prefill_tokens_per_s(m35, tp2, CH)
            > 5 * prefill_tokens_per_s(m27, tp2, CH)), \
        "35B-A3B must prefill >5x faster than the dense 27B"
    # the pair-count telescopes: a chunked context costs exactly one big pass
    # (chunking bounds the spike, not the total), and the second chunk pays
    # its cross-attention over the first
    assert abs(prefill_context_seconds(m27, tp2, 2 * CH, CH)
               - prefill_seconds(m27, tp2, 2 * CH)) < 1e-9, \
        "chunked total must equal the unchunked pass (telescoping pairs)"
    assert (prefill_seconds(m27, tp2, CH, prior=CH)
            > prefill_seconds(m27, tp2, CH)), "later chunks cost more"
    assert abs(prefill_flops(m27, CH, prior=CH)[1]
               - 3 * prefill_flops(m27, CH)[1]) < 1, \
        "chunk 2's attention = intra + 2x cross = 3x chunk 1's"
    # a warm hit is charged its attention over the cached context: linear in
    # the cache, so it cannot be cheaper than the cold turn-only pass
    assert (warm_request_seconds(m27, tp2, 2_000, CH, prior=100_000)
            > warm_request_seconds(m27, tp2, 2_000, CH)), \
        "cached context is not free to attend over"
    # E[L^2] > E[L]^2 must make the expected miss dearer than the mean-length miss
    _el, _el2 = context_moments(wl)
    assert _el2 > _el ** 2, "second moment sanity"
    assert (cold_request_seconds(m27, tp2, wl, CH)
            > prefill_context_seconds(m27, tp2, _el, CH)), \
        "expected miss cost must price the heavy tail, not the mean draw"
    # duty cycle is linear in miss rate and crosses 1.0 at breakeven_miss_rate
    fstar = breakeven_miss_rate(m27, tp2, wl, 2.13, CH, 2_000)
    assert 0 < fstar < 1, f"27B TP2 breakeven miss rate should be ~26%, got {fstar:.0%}"
    assert abs(prefill_duty(m27, tp2, replace(wl, invalidation=fstar), 2.13,
                            CH, 2_000) - 1.0) < 1e-9, "breakeven must give duty 1.0"
    # max_cold_rate is the all-cold case of the same arithmetic
    assert abs(prefill_duty(m27, tp2, replace(wl, invalidation=1.0),
                            max_cold_rate(m27, tp2, wl, CH), CH) - 1.0) < 1e-9
    # guards
    for bad_mfu in (0.0, -0.1, 1.5):
        try:
            prefill_seconds(m27, tp2, CH, bad_mfu)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
    try:
        prefill_flops(replace(m27, params_prefill=0.0), CH)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass

    print("selfcheck OK")
    print(f"  ACT_RESERVE          = {ACT_RESERVE / GIB:6.2f} GiB")
    print(f"  prefill 27B TP2 32k  = {prefill_seconds(m27, tp2, CH) * 1e3:6.0f} ms "
          f"({prefill_flops(m27, CH)[2] / 1e12:.0f} TFLOP, "
          f"{prefill_flops(m27, CH)[1] / prefill_flops(m27, CH)[2]:.0%} attention)")
    print(f"  cold/warm cost ratio = {th:6.0f}x   breakeven miss rate "
          f"{fstar:.0%} @ 2.13 req/s")
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
