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

The interactive viz (interactive/src/*.js) mirrors this math in JS; keep the
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
    # ---- POWER constants (research/power.md) — used ONLY by power_draw() ----
    # tdp_w: the vendor spec plate. SPEC (HIGH). H200 SXM "Up to 700W
    #   (configurable)"; B300 "up to 1,400 W" per NVIDIA's "Inside Blackwell
    #   Ultra". The air-cooled B300 SXM6 AC variant caps at 1,100 W — deploying
    #   that part scales all three B300 GPU wattages by ~0.79 (power.md #3);
    #   not modelled here.
    # idle_w: warm idle. MEASURED H100 proxy — 72.5 W ± 0.1 per GPU at 0.1 s
    #   resolution (NLR facility-planning study, 4xH100 nodes), +10% for the
    #   H200's larger HBM3e array -> 80 W. B300: NO measurement exists
    #   anywhere; the same ~10.5%-of-TDP fraction transferred -> 150 W.
    # p_decode_w: bandwidth-bound token-phase draw, 0.55 x TDP central (band
    #   0.45-0.75 — the note's softest constant). MEASURED band on Hopper:
    #   Splitwise (ISCA'24) finds token-phase power flat in batch size and
    #   tolerant of a 700->350 W cap; mixed vLLM serving measured 229-477 W
    #   on H200-class parts. B300 transfers the FRACTION, not the watts.
    # p_prefill_w: compute-bound prefill is POWER-CAP-limited, 0.90 x TDP
    #   central (band 0.80-1.00) — FLAT across the study's MFU 35-55% bracket,
    #   because the cap binds before the FLOP peak does. MEASURED anchor:
    #   saturated vLLM inference at ~0.85-0.89 of summed GPU TDP (NLR).
    # host_w: flat per-GPU chassis adder (CPUs/NVSwitch/NICs/fans/PSU loss),
    #   DERIVED from system spec maxima: DGX H200 10.2 kW − 8 x 700 W ->
    #   575 W/GPU; the DGX B300 arithmetic brackets 410-710 W/GPU and 500 sits
    #   below the 562 mid, weighted toward the 1,400 W-TDP scenario. A spec
    #   ceiling used FLAT, not duty-cycled — over-charges a lightly-loaded
    #   chassis by up to ~2x, the same conservative direction as the study's
    #   other bookkeeping (power.md #4).
    # B300 rows are ENTIRELY EXTRAPOLATED (Hopper fractions of TDP moved onto
    # the 1,400 W plate): no published Blackwell Ultra power-state measurement
    # exists at all (power.md #3). The single measurement that would firm the
    # softest rows up: one nvidia-smi power trace beside a vllm bench run on
    # the 2xH200 hardware the study already has.
    tdp_w: float = 0.0
    idle_w: float = 0.0
    p_decode_w: float = 0.0
    p_prefill_w: float = 0.0
    host_w: float = 0.0
    # ---- HARDWARE price, EUR per GPU-hour -- used ONLY by energy_cost() ----
    # On-demand rental LIST price: cross-provider medians (getdeploying.com,
    # read 2026-09-03: H200 $4.40, B300 $7.89) at EUR/USD 1.16, rounded to the
    # explorer slider's 0.10 step. A market snapshot, not a measurement and not
    # a cost of ownership. NVIDIA publishes no list price for either part.
    # Mirrors CONFIG.GPUS[].eur_gpu_h in the explorer, where a slider overrides
    # it. Convention: a rental rate is all-in (power included), so for a rented
    # GPU the electricity line double-counts; for owned hardware substitute an
    # amortised figure and both lines apply.
    eur_gpu_h: float = 0.0


GPUS = {
    # peak_flops_fp8: 1,979 TFLOPS DENSE. NVIDIA's H200 datasheet leads with
    # "3,958 TFLOPS FP8" — that figure is WITH 2:4 structured sparsity, which
    # no dense LLM GEMM reaches. Halve it. research/prefill.md #1.
    "H200": GPU("H200", 141e9, 4.8e12, supports_nvfp4=False,
                peak_flops_fp8=1.979e15,
                # power (research/power.md #1): spec plate 700 W SXM; idle a
                # measured H100 proxy +10%; decode/prefill the measured Hopper
                # fractions 0.55 / 0.90 of TDP; host the DGX H200 spec adder
                tdp_w=700.0, idle_w=80.0, p_decode_w=385.0, p_prefill_w=630.0,
                host_w=575.0, eur_gpu_h=3.80),
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
                reserve_extra=9.75e9, peak_flops_fp8=4.5e15,
                # power: ENTIRELY EXTRAPOLATED — no Blackwell Ultra power-state
                # measurement exists. Rule (power.md #3): transfer Hopper's
                # fractions of TDP (idle ~10.5%, decode 0.55, prefill 0.90)
                # onto the 1,400 W plate; host_w is the mid of the DGX B300
                # spec bracket (410-710 W/GPU). If the target is the 1,100 W
                # air-cooled SXM6 AC part, scale the three GPU wattages ~0.79.
                tdp_w=1400.0, idle_w=150.0, p_decode_w=770.0,
                p_prefill_w=1260.0, host_w=500.0, eur_gpu_h=6.80),
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
# ---- DECODE efficiency: MFU's counterpart, and the study's newest constant ----
# Decode was priced as a PURE roofline until 2026-08-28: bytes / effective_bw,
# with no efficiency term at all, while prefill had carried an MFU anchor for
# weeks. research/decode_mbu.md measures the gap on the reference row
# (27B / 4xH200 TP4, production, five sessions): the model ran **4.1x
# optimistic**, and ONE multiplicative constant brings it to a median 13%
# (worst 20%) across batch sizes 1-25 and step bytes 35-90 GB.
#
# CONVENTION, as in prefill.md: this is the MODEL-convention figure, i.e.
# already divided by tp_efficiency, so it multiplies effective_bw() directly.
# The measurement reads 0.179 of the RAW advertised aggregate; 0.179/0.81 =
# 0.221 here. Quote the convention whenever quoting the number.
#
# WHAT IT ABSORBS, and why that matters: the constant is fitted on a HYBRID
# model (48 Gated-DeltaNet layers of 64) running speculative decoding at
# num_speculative_tokens=2. It therefore absorbs both the streaming
# inefficiency and whatever the speculative verify costs on a sequential
# recurrence -- decode_mbu.md could not separate those from outside the pod.
# Two consequences:
#   * MBU and the model's `mtp` TRAVEL TOGETHER. The 27B's mtp is the measured
#     accepted length at k=2; changing one without the other breaks the fit.
#   * It is NOT a hardware constant. A dense model has no recurrence to
#     serialise and should sit higher; nothing in this study measures one.
# ONE deployment, ONE model, ONE spec config -- an anchor, not a bracket. The
# LOW/HIGH pair spans the residual spread, not independent measurements.
#
# EXTRAPOLATION BIAS -- the sharpest limitation, and it bites where the study
# cares most. The constant was fitted over n = 1-25, where WEIGHTS are 80-95%
# of the step. Folding a weight-side inefficiency into a multiplier on the
# WHOLE ledger then over-charges the KV term, and the decode ceiling lives at
# n ~ 100-250 where KV dominates instead. Against the two candidate mechanisms
# in decode_mbu.md sec 4.3, both anchored to the same n=1 measurement, the
# 27B / 4xH200 decode ceiling reads:
#     this single-constant fold      108 users
#     reading A (weights x(1+k))     238 users
#     reading B (serial latency)     154 users
# against a cache ceiling of 249. So the CEILING IS NOT IDENTIFIED by this
# measurement, and neither is the binding order: A leaves H7 standing, B and
# the fold overturn it. What IS identified is the per-user speed near the
# fitted mix (n = 1-25), where all three agree to ~10%.
# Quote decode ceilings from this model as a range, not a point, until a
# spec-off / k-sweep A/B settles which mechanism holds.
# ONE value for EVERY model. A dense or MLA row has no recurrence to serialise
# and should sit higher, but nothing in this study measures one, and a
# per-model guess would rank configurations on the guess. Mirrors DECODE_MBU
# in the explorer, where the slider moves every row at once.
MBU_LOW, MBU_DEFAULT, MBU_HIGH = 0.15, 0.22, 0.30


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
    # The KV dtype these constants currently price ("fp8" = the study
    # baseline; with_kv_dtype("fp16") stamps "fp16"). Exists so
    # check_dtype_supported can see which arm it is being asked to price.
    kv_dtype: str = "fp8"
    # True for models whose FP8 KV cache is BLACKWELL-ONLY (GLM-5.3-Flash:
    # "Hopper ... must run BF16 KV" — vLLM recipe). check_dtype_supported
    # then REFUSES the fp8-KV arm on non-Blackwell parts, so every H200
    # figure must be produced from the with_kv_dtype("fp16") arm — silent
    # mis-pricing becomes a crash, not a wrong number.
    kv_fp8_blackwell_only: bool = False
    # False when deltanet_state is NOT a bf16 recurrent state the fp32 toggle
    # can meaningfully double (DSv4-Flash reuses the field for its fixed
    # per-session window + fp32 compressor buffers, already mixed-precision).
    # Python charges deltanet_state as-is either way; the flag exists for
    # mirror parity with the explorer, which gates its fp32-state control on it.
    state_fp32_ok: bool = True
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
    # Named Qwen3.8-27B since 2026-09-05: its config and its FP8 checkpoint
    # (1,606 tensors, dtypes and shapes) are identical to Qwen3.6-27B's, so
    # every architecture-determined constant carries over; mtp is unmeasured
    # on it and nvfp4_w is a projection (see the notes on each).
    "27B": Model(
        name="Qwen3.8-27B (dense)",
        kv_bpt=32 * KIB,                 # 16 attn layers x 4 KV heads x 256 x 2(K,V) x 1B
        deltanet_state=75 * MIB,         # baseline's 75 MiB; bf16 arithmetic (48 DN layers x
                                         # 48 vheads x 128x128 + conv) gives 75.7 MiB

        w_resident=28.8 * GIB,           # baseline's stated as-deployed FP8 footprint
        w_decode_shared=28.8 * GIB,      # dense: every step reads all weights
        w_route_pertok=0.0,
        w_route_total=0.0,
        # MEASURED 2026-08-28 on the production 27B deployment recorded as
        # Qwen3.6-27B (research/decode_mbu.md): accepted length 2.94 at
        # num_speculative_tokens=2, per-draft acceptance 0.971 -- and the
        # 1+a+a^2 acceptance model confirmed to three decimals against vLLM's
        # per-position counters. Was 1.7, the baseline's fit. NOT re-measured
        # on Qwen3.8-27B: acceptance is a property of the draft head's
        # weights, not the architecture. PAIRED WITH MBU_DEFAULT: both were
        # calibrated against the same passes, so moving one without the
        # other breaks the fit (see MBU_DEFAULT).
        mtp=2.94,
        # PROJECTION onto Qwen3.8-27B's tensor-identical weights of
        # nvidia/Qwen3.6-27B-NVFP4's MEASURED safetensors total 21,921,428,072 B
        # (2026-07-27 re-verification); no NVIDIA NVFP4 of 3.8 exists. Same
        # convention as the FP8 28.8 GiB, which the measured
        # Qwen/Qwen3.6-27B-FP8 checkpoint (30.87e9 B = 28.75 GiB) matches
        # within 0.2% — the old x22/27.8 "as-deployed" scaling was a
        # mis-derivation. Dense: decode reads everything.
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
    # MoE 284B-A13B (304B on disk incl. 3 DSpark stages), MQA over a 512-dim
    # latent + per-layer-class compression (2 SWA / 21 CSA ratio-4 / 20 HCA
    # ratio-128), open weights (2026-07-31). Only COMPRESSED caches grow:
    # 3,450 B/token — ~10x below V3-class MLA — plus a fixed ~14.9 MiB/session
    # (128-entry windows on 46 layers + fp32 compressor state), carried in
    # deltanet_state. Decode reads the FP4 indexer scan + dense-HCA compressed
    # axis (426 B/ctx-token) and top-512 latents + windows (9.36 MB/seq).
    # Native checkpoint is already mixed FP8/FP4 (experts FP4 + E8M0 scales,
    # servable on H200 per the vLLM recipe) -> no NVFP4 variant exists or
    # helps (the community conversion is LARGER). research/model_dsv4flash.md.
    "DSV4F": Model(
        name="DeepSeek-V4-Flash-0731 (MoE 284B-A13B, CSA)",
        kv_bpt=3_450,                    # 21 x 576/4 + 20 x 576/128 + 21 x 64/4 (fp8 latent+fp4 idx)
        deltanet_state=15_597_568,       # 46 x 128 x 576 windows + 12,206,080 fp32 compressor state
        state_fp32_ok=False,             # already fp32/fp8-mixed; doubling models nothing
        w_resident=166.88e9,             # measured safetensors total 166,878,536,440 B
        w_decode_shared=7.66e9,          # attn 4.60 + shared exp 1.08 + compressors/indexers/
                                         # gates/mHC 0.92 + lm_head 1.06 (research note 4)
        w_route_pertok=3_449_290_752,    # 6 experts x 13,369,344 B (FP4 packed + scales) x 43
        w_route_total=147_169_738_752,   # 256 experts (kink at n = 256/6 ~ 42.7 — non-integer)
        mtp=1.7,                         # DSpark drafts 7 tokens; transplanted fit, unmeasured
        nvfp4_w=None,                    # experts already FP4 natively; no official 0731 NVFP4
        kv_decode_bpt=426,               # 21 x 64/4 fp4 indexer scan + 20 x 576/128 dense HCA
        kv_decode_const=9_363_456,       # 21 x 512 x 576 top-k reads + 43 x 128 x 576 windows
        kv_decode_topk=2_048,            # 512 compressed entries x ratio 4, in token space
        kv_fp16_ok=False,                # vLLM's V4 path asserts fp8 main KV; SGLang's bf16
                                         # KV-decode is unfinished (research note 6)
        max_ctx=1_048_576,               # native 1M (YaRN x16 over 65,536 baked into the config)
        # prefill: MoE active GEMM params excl embed/lm_head (12.703e9 from the
        # param ledger). Quadratic term: the indexer scores the full compressed
        # axis (equiv. attn_d 1024 x 21 CSA layers) and HCA attends it densely
        # (equiv. 256 x 20); the CSA top-512 and window reads are LINEAR per
        # token and deliberately left out — prefill priced cheaper, biased
        # AGAINST the thrash hypothesis (research/model_dsv4flash.md #6).
        params_prefill=12.70e9, attn_layers=41, attn_d=26_624 / 41,
    ),
    # MoE 125B-A6B (180B on disk incl. the 51B FP8 n-gram table, 2.7B MTP and
    # a never-executed vision tower), open weights (2026-08). Qwen3.6-style
    # hybrid: 36 DeltaNet + 12 QSA sparse-attention layers (GQA 24/2 x 256
    # with a ratio-4 compressed fp8 indexer, budget 2048). Only the routed
    # experts and the n-gram table are FP8 in the serving checkpoint — the
    # always-active blocks stay BF16, so the shared per-step read is heavy
    # (8.6 GB) while the cache is light (12.4 KiB/token). Deepest expert-union
    # kink in the study: n = 512/10 = 51.2. research/model_qwen38flashnext.md.
    "Q38FN": Model(
        name="Qwen3.8-Flash-Next (MoE 125B-A6B, QSA+n-gram)",
        kv_bpt=12_672,                   # 12 attn x 2 KV x 256 x 2(K,V) x 1B + 12 x 128/4 indexer
        deltanet_state=59_572_224,       # 36 DN layers x 48 vheads x 128x128 bf16 + conv (10,240 x 4)
        w_resident=185_502_232_570,      # FP8 ckpt metadata.total_size, shard-header-verified
        w_decode_shared=8_623_999_000,   # exact per-tensor ledger sum — BF16 always-active:
                                         # attn 1.23 + DN 4.17 + shared exp 0.47 + routers 0.13
                                         # + hyper-conns 1.28 + PLE 0.07 + lm_head 1.27
                                         # (n-gram/embed lookups excluded)
        w_route_pertok=2_359_584_000,    # 10 experts x 4,915,800 B (FP8 + block scales) x 48
        w_route_total=120_810_700_800,   # 512 experts (kink at n = 512/10 = 51.2 — the deepest)
        mtp=1.7,                         # MTP module (1 hybrid layer, 3 drafts per the vLLM
                                         # recipe); transplanted fit, unmeasured
        nvfp4_w=None,                    # no official NVFP4 (community only); experts already FP8
        kv_decode_bpt=384,               # QSA indexer scan: 12 layers x 128 B / ratio 4 per ctx tok
        kv_decode_const=25_165_824,      # 12 layers x top-2048 x 1,024 B full-KV reads per seq
        kv_decode_topk=2_048,            # indexer_budget, read in token space (research note #6)
        max_ctx=1_048_576,               # 262,144 native; 1M via YaRN (owner decision, as the Qwens)
        # prefill: MoE active GEMM params excl embed/lm_head/n-gram lookups
        # (6.04e9 from the shard-header ledger — the published "6B activated").
        # Quadratic term priced as DENSE attention on the 12 QSA layers: an
        # upper bound, QSA prefill sparsity uncharacterised (note #6).
        params_prefill=6.04e9, attn_layers=12, attn_d=24 * 256,
    ),
    # MoE 320B-A18B (321B on disk incl. a 7.4B MTP draft layer and a vision
    # tower), open weights (2026-08-25, MIT). GLM-5.2's DSA married to a
    # Qwen-style linear backbone: 34 KDA linear-attention + 11 NoPE
    # sparse-MLA layers (512-B latents, NO rope bytes) with a kpool-4
    # COMPRESSED indexer cache — 6.39 KiB/token, 7.4x below GLM-5.2 —
    # plus the study's second-heaviest recurrent state (74.4 MiB bf16).
    # 288 experts / 8 routed (kink n = 36). KV dtype is GPU-COUPLED: the
    # fp8-KV base arm is Blackwell-only — "Hopper ... must run BF16 KV"
    # (vLLM recipe), i.e. on H200 the FP16-KV toggle is the only servable
    # arm. research/model_glm53flash.md.
    "GLM53F": Model(
        name="GLM-5.3-Flash (MoE 320B-A18B, KDA+NoPE-MLA)",
        kv_bpt=6_540,                    # 12 x 512 nope-only MLA latent + 12 x 132/4 indexer keys
                                         # (12 = 11 main + the MTP draft layer's DSA stack — the
                                         # GLM-5.2 convention: storage incl. MTP, decode excl.)
        deltanet_state=77_987_840,       # 34 KDA layers x 64 heads x 128x128 bf16 + q/k/v conv
        w_resident=328_326_771_576,      # FP8 ckpt metadata.total_size, shard-header-verified
        w_decode_shared=13_957_216_504,  # exact ledger sum — KDA 9.37 (BF16!) + DSA attn 1.64
                                         # + shared exp 1.06 + dense MLPs 0.45 + routers 0.10
                                         # + hyper-conns 0.07 + lm_head 1.27 + norms
        w_route_pertok=8_457_781_248,    # 8 experts x 25,171,968 B (FP8 + F32 scales) x 42
        w_route_total=304_480_124_928,   # 288 experts (kink at n = 288/8 = 36)
        mtp=1.7,                         # MTP draft layer (recipe runs 5 drafts); transplanted fit
        nvfp4_w=None,                    # no official NVFP4 (community only)
        kv_decode_bpt=363,               # compressed indexer scan: 11 x 132 B / kpool 4 per ctx tok
        kv_decode_const=11_534_336,      # 11 layers x top-2048 x 512-B latent reads per seq
        kv_decode_topk=2_048,            # index_topk, read in token space (research note #6)
        kv_fp8_blackwell_only=True,      # "Hopper ... must run BF16 KV" (vLLM recipe): the fp8-KV
                                         # arm raises on H200 — price with_kv_dtype("fp16") there
        max_ctx=1_048_576,               # native 1M context
        # prefill: MoE active GEMM params excl embed/lm_head (16.11e9 ledger;
        # card "18B active" incl. embed+lm_head = 17.38e9 ✓). Quadratic term
        # priced as DENSE attention on the 11 DSA layers: an upper bound,
        # DSA prefill sparsity uncharacterised (note #6).
        params_prefill=16.11e9, attn_layers=11, attn_d=64 * 256,
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
    # one-way door, like with_weight_dtype: an fp16 arm cannot be converted
    # again (double-doubling) or "back" (no fp8 constants are kept) — and a
    # round-trip would un-stamp kv_dtype and slip past check_dtype_supported
    if model.kv_dtype != "fp8":
        raise ValueError(f"{model.name}: with_kv_dtype expects the base "
                         "(fp8) model; start from the MODELS[...] entry")
    if kv_dtype == "fp8":
        return model
    if not model.kv_fp16_ok:
        raise ValueError(
            f"{model.name}: FP16 KV is not servable (vLLM requires a quantized "
            "KV cache on this model's sparse-attention path — GLM-5.2's DSA, "
            "DSv4-Flash's CSA; see the model's research note)")
    # On a sparse-decode model the top-k gathers read MAIN-KV bytes and double
    # with the cache dtype; the indexer scan (kv_decode_bpt) keeps its own
    # quantized width. kv_bpt doubles wholesale — the indexer share inside it
    # (≤3% on Q38FN, ≤6% on GLM-5.3-Flash) over-doubles, a conservative
    # slack flagged in each model's note.
    return replace(model, kv_bpt=model.kv_bpt * 2,
                   kv_decode_const=model.kv_decode_const * 2,
                   kv_dtype="fp16", name=model.name + " [FP16 KV]")


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
    # GPU-coupled KV dtype (GLM-5.3-Flash): the fp8 KV cache is servable on
    # Blackwell only — "Hopper does not support FP8 KV cache for this model
    # and must run BF16 KV" (vLLM recipe). supports_nvfp4 doubles as the
    # study's Blackwell marker (true exactly for the B300).
    if (model.kv_fp8_blackwell_only and model.kv_dtype == "fp8"
            and not topo.gpu.supports_nvfp4):
        raise ValueError(
            f"{model.name}: the FP8 KV cache is Blackwell-only — "
            f"{topo.gpu.name} must price the BF16-KV arm "
            "(with_kv_dtype(model, \"fp16\"); vLLM recipe, "
            "research/model_glm53flash.md #2)")


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
# STATUS: analytic, MFU CALIBRATED 2026-08-27 (was: unvalidated). Two
# production measurements anchor MFU (research/prefill.md #1), agreeing at
# ~40% of the raw ADVERTISED peak on both BF16-Ampere (39.6% ± 0.4%, n=9,
# engine-side counter deltas) and Hopper FP8 (40.0% implied from a 7-day
# production mean prefill time, research/workload_agentic_poc.md). In THIS
# module's convention — mfu divides peak_flops(topo), which already carries
# the tp_efficiency haircut — those same points read 44.4% (TP2) and 49.4%
# (TP4) against the 45% central. State the convention when quoting. The
# bracket tightened [0.30,0.60] → [0.35,0.55]: the plausible range now moves
# every figure here by ~1.6x (was 2x). Still bounds that rank
# configurations, not latency commitments.
# ============================================================================

# Model FLOP Utilisation: achieved FLOPs / dense peak on a large prefill GEMM.
# 45% is a mid-range figure for FP8 attention+MLP prefill on Hopper-class
# parts with TP collectives in the loop, now bracketed (in the model
# convention only: 44.4% / 49.4%) by the two calibration points above rather
# than by published-benchmark spread alone. Every prefill
# number in this study should be read with the bracket, not the point.
MFU_LOW, MFU_DEFAULT, MFU_HIGH = 0.35, 0.45, 0.55



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
    ITL spike a decode batch sees), not the total FLOP count. The total
    machine TIME is another matter: per-pass overheads do not telescope, so
    a miss's time IS chunk-dependent — see miss_context_seconds.
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
    chunk-size INVARIANT — but only under this MARGINAL flat-MFU pricing.
    The miss-side cost with per-pass overhead is chunk-dependent; that
    version is miss_context_seconds.
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


# ---------------------------------------------------------------------------
# MFU as a function of chunk size (mirrors the explorer's chart E; added
# 2026-08-02, research/prefill.md #3). The FLOPs of a context telescope
# across chunks, but each forward pass also pays costs that do not: it
# streams the full resident weights once (a chunk of any practical size
# touches every routed expert on a MoE, so the whole expert bank streams —
# the MoE curves degrade hardest), plus kernel launches and collectives.
# Model: one MISS-side pass costs
#     flops / (peak x mfu_ceiling)  +  w_resident / effective_bw(topo)
# an additive no-overlap roofline, conservative exactly at small chunks
# where there is too little compute to hide the stream. mfu_ceiling is
# solved per (model, topo) so the EFFECTIVE MFU of a first chunk at the
# 32,768 default equals the calibrated MFU_DEFAULT — the anchor absorbs
# whatever overlap the real machine achieves there, and every table built
# on the flat-MFU functions above stays put (those functions keep their
# published behaviour; these are OPT-IN and price the chunk-size trade).
# Kernel-launch/scheduler costs remain unpriced: the small-chunk end still
# reads BETTER than a real machine, the honest direction for a family of
# functions whose message is "small chunks are not free".
# ---------------------------------------------------------------------------

CHUNK_DEFAULT = 32_768   # vLLM max_num_batched_tokens, the study's default


def prefill_overhead_seconds(model: Model, topo: Topology) -> float:
    """Per-forward-pass cost that does NOT telescope: the resident-weight
    stream. w_resident, not w_decode_shared — prefill activates every routed
    expert at any practical chunk size."""
    return model.w_resident / effective_bw(topo)


def mfu_ceiling(model: Model, topo: Topology,
                chunk: float = CHUNK_DEFAULT,
                mfu_anchor: float = MFU_DEFAULT) -> float:
    """Compute-leg MFU solved so that a first (cache-empty) chunk of `chunk`
    tokens nets exactly `mfu_anchor` effective MFU once the per-pass
    overhead is added. Slightly above the anchor (0.4514 for the 27B on one
    H200, 0.4623 for the 35B-A3B — the more overhead, the more headroom the
    compute leg must carry)."""
    ref = prefill_flops(model, chunk)[2]
    inv = (1.0 / mfu_anchor
           - peak_flops(topo) * prefill_overhead_seconds(model, topo) / ref)
    if inv <= 0:
        raise ValueError(f"{model.name}: per-pass overhead exceeds the whole "
                         f"MFU budget at chunk={chunk} — anchor unsolvable")
    return 1.0 / inv


def prefill_pass_seconds(model: Model, topo: Topology, tokens: float,
                         prior: float = 0.0,
                         mfu_anchor: float = MFU_DEFAULT) -> float:
    """One MISS-side forward pass: FLOPs at the solved ceiling + overhead.
    (A warm hit's small pass and the ITL-spike chunk stay on the MARGINAL
    prefill_seconds pricing: they join passes whose weight stream is already
    paid — by the decode batch — so charging them the stream twice would be
    wrong. A miss at the duty ceiling has no host pass; every one of its
    chunks pays.)"""
    return (prefill_flops(model, tokens, prior)[2]
            / (peak_flops(topo) * mfu_ceiling(model, topo,
                                              mfu_anchor=mfu_anchor))
            + prefill_overhead_seconds(model, topo))


def mfu_effective(model: Model, topo: Topology, chunk: float,
                  mfu_anchor: float = MFU_DEFAULT) -> float:
    """Achieved fraction of dense peak for a first chunk of `chunk` tokens —
    the curve that answers why max_num_batched_tokens sits high: at the
    32,768 anchor this is mfu_anchor by construction; at 2,048 the 27B still
    holds ~43% but the 35B-A3B falls to ~28% and GLM-5.2/TP8 to ~25% (too
    little active-param compute to hide the full expert-bank stream)."""
    return (prefill_flops(model, chunk)[2]
            / (peak_flops(topo)
               * prefill_pass_seconds(model, topo, chunk,
                                      mfu_anchor=mfu_anchor)))


def miss_context_seconds(model: Model, topo: Topology, context: float,
                         chunk: float, prior: float = 0.0,
                         mfu_anchor: float = MFU_DEFAULT) -> float:
    """A miss's whole context, chunked at `chunk`: the FLOPs telescope as
    ever, the overhead multiplies by the pass count — so unlike
    prefill_context_seconds this is NOT chunk-size invariant; smaller chunks
    cost more total machine time. At chunk=32,768 it reproduces the flat-MFU
    prefill_context_seconds within <0.2% on the dense models and within ~2%
    (UNDER, i.e. the flat tables err conservative) on the MoEs at long
    contexts — the anchor pins a first cache-empty pass exactly, and the
    MoEs' later passes run at a visibly higher mfu_ceiling than 45%, which
    their small overhead does not fully pay back. Contexts SHORTER than the
    chunk are the deliberate exception: they are one small pass whose
    effective MFU sits below the anchor, so they cost genuinely more than
    the flat model said (~+10% at the reference workload's p5 length for
    the 35B-A3B, ~+13% for GLM-5.2) — that increase is the model's point
    (short passes amortise the weight stream worse), not an artefact."""
    if chunk <= 0:
        raise ValueError(f"chunk must be > 0, got {chunk!r}")
    ceil = mfu_ceiling(model, topo, mfu_anchor=mfu_anchor)
    over = prefill_overhead_seconds(model, topo)
    total, done = 0.0, 0.0
    context = float(context)
    while done < context:
        step = min(chunk, context - done)
        total += (prefill_flops(model, step, prior + done)[2]
                  / (peak_flops(topo) * ceil) + over)
        done += step
    return total


def mean_passes(wl, chunk: float, n: int = 200_000, seed: int = 0) -> float:
    """E[ceil(L / chunk)] over the workload's context lengths — the expected
    pass count a miss's per-pass overhead multiplies by. Sampled, not
    E[L]/chunk + 0.5: truncation at the cap makes the fractional parts
    anything but uniform."""
    full, _, _, _ = wl.sample(np.random.default_rng(seed), n)
    return float(np.ceil(full.astype(float) / chunk).mean())


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
                         mfu: float = MFU_DEFAULT,
                         per_pass_overhead: bool = False) -> float:
    """EXPECTED machine time to serve one CACHE MISS: re-prefill the whole
    context, averaged over the workload's context-length distribution.

    Uses the closed form the telescoping pair-count allows (total FLOPs of a
    chunked context L = gemm(L) + 2 L^2 d n_layers, independent of chunking):
        E[cost] = (2 params E[L] + 2 attn_d attn_layers E[L^2]) / (peak x mfu)
    E[L^2], not E[L]^2 — the quadratic term must be priced on the heavy tail,
    not on the mean draw.

    per_pass_overhead=False keeps the published flat-MFU pricing every table
    in docs/ was built on (chunk is then unused — the FLOPs telescope).
    per_pass_overhead=True is what the explorer prices since 2026-08-02: the
    FLOPs leg runs at mfu_ceiling and each of the E[ceil(L/chunk)] passes
    adds the weight-stream overhead, making the cost chunk-dependent (`mfu`
    is then the anchor, see mfu_ceiling). At chunk=32,768 the two agree
    within <1%.
    """
    check_dtype_supported(model, topo)
    if not 0 < mfu <= 1:
        raise ValueError(f"mfu must be in (0, 1], got {mfu!r}")
    el, el2 = context_moments(wl)
    flops = (2.0 * model.params_prefill * el
             + 2.0 * model.attn_d * model.attn_layers * el2)
    if not per_pass_overhead:
        return flops / (peak_flops(topo) * mfu)
    if chunk <= 0:
        raise ValueError(f"chunk must be > 0, got {chunk!r}")
    return (flops / (peak_flops(topo)
                     * mfu_ceiling(model, topo, mfu_anchor=mfu))
            + mean_passes(wl, chunk) * prefill_overhead_seconds(model, topo))


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
                 chunk: float, mfu: float = MFU_DEFAULT,
                 per_pass_overhead: bool = False) -> float:
    """How many times more machine time a miss costs than a hit.

    THE number behind the study's hypothesis. With per_pass_overhead=False
    (the published default) it is independent of MFU and of the GPU part
    (both cancel) — a property of the workload's context-length distribution
    against the turn length; with the opt-in overhead pricing the miss side
    gains a weight-stream term and the exact cancellation no longer holds.
    It is NOT independent of the attention model itself: on attention-heavy
    rows (GLM-5.2's dense-MLA upper bound above all) the quadratic term
    drives both numerator and the warm hit's cross term, so those rows
    inherit research/prefill.md weakness #2 rather than escaping it.
    """
    return (cold_request_seconds(model, topo, wl, chunk, mfu,
                                 per_pass_overhead=per_pass_overhead)
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
                  mfu: float = MFU_DEFAULT,
                  per_pass_overhead: bool = False) -> float:
    """Cold requests/second at which prefill alone consumes the whole machine.

    A hard ceiling the capacity model cannot see: it is set by FLOPs, so no
    amount of KV pool, CPU offload or warm-session headroom raises it. Above
    this rate the deployment is prefill-bound and warm capacity has stopped
    being the binding constraint.

    per_pass_overhead: the explorer prices this with True (chunk-dependent,
    see cold_request_seconds); the published tables use the False default.
    """
    return 1.0 / cold_request_seconds(model, topo, wl, chunk, mfu,
                                      per_pass_overhead=per_pass_overhead)


def prefill_duty(model: Model, topo: Topology, wl, req_rate: float,
                 chunk: float, turn_tokens: float = 0.0,
                 mfu: float = MFU_DEFAULT,
                 per_pass_overhead: bool = False) -> float:
    """Fraction of the replica group spent prefilling at `req_rate` req/s.

    Counts the cold re-prefills (wl.invalidation of the traffic) plus, if
    turn_tokens > 0, the new-turn prefill every WARM hit still pays. Values
    above 1.0 mean prefill alone oversubscribes the group: the queue grows
    without bound and TTFT diverges, whatever the warm-session count says.

    per_pass_overhead prices the MISS side only (warm turns stay marginal —
    see prefill_pass_seconds for the boundary); the explorer uses True.
    """
    if req_rate < 0:
        raise ValueError(f"req_rate must be >= 0, got {req_rate!r}")
    f = wl.invalidation
    per_s = f * cold_request_seconds(model, topo, wl, chunk, mfu,
                                     per_pass_overhead=per_pass_overhead)
    if turn_tokens > 0:
        per_s += (1 - f) * warm_request_seconds(model, topo, turn_tokens,
                                                chunk, mfu,
                                                prior=mean_context(wl))
    return req_rate * per_s


def breakeven_miss_rate(model: Model, topo: Topology, wl, req_rate: float,
                        chunk: float, turn_tokens: float = 0.0,
                        mfu: float = MFU_DEFAULT,
                        per_pass_overhead: bool = False) -> float:
    """Miss rate at which prefill duty hits 1.0 for this request rate.

    Returns >1 (i.e. unreachable) when even an all-cold workload fits, and can
    return <=0 when the warm-turn prefill alone already saturates the group.
    Solving duty(f) = 1 for f, with duty linear in f:
        f x cold + (1-f) x warm = 1 / rate
    per_pass_overhead prices the miss side only, as in prefill_duty; the
    explorer's f* uses True.
    """
    cold = cold_request_seconds(model, topo, wl, chunk, mfu,
                                per_pass_overhead=per_pass_overhead)
    warm = (warm_request_seconds(model, topo, turn_tokens, chunk, mfu,
                                 prior=mean_context(wl))
            if turn_tokens > 0 else 0.0)
    if cold == warm:
        return float("inf")
    return (1.0 / req_rate - warm) / (cold - warm)


# ============================================================================
# COLD SPIKES  (research/spike.md)
# ----------------------------------------------------------------------------
# Section 8 prices prefill as a DUTY CYCLE: a mean arrival rate against a mean
# service time. Two things that model cannot see, and limitations 2 and 8 name
# both of them:
#
#   variance   even smooth (Poisson) arrivals QUEUE. A miss's service time is
#              driven by L and L^2 on a lognormal context distribution, so its
#              second moment is enormous and the Pollaczek-Khinchine wait
#              diverges long before duty reaches 100%.
#   bursts     agentic traffic invalidates in CLUMPS, not one request at a
#              time: a prompt-template deploy, a cache flush, a restarted
#              worker fleet — limitation 8's "correlated invalidation". B
#              misses land together and drain at whatever rate the standing
#              load leaves free.
#
# So f* is not a planning number. It is the miss rate at which burst tolerance
# reaches ZERO — the point where the queue has already diverged, not the point
# where trouble starts. What a deployment can actually absorb is what this
# section computes, and it is where the MoE's active-parameter prefill
# advantage compounds (or, on a global flush, partly cancels against its own
# larger warm population).
#
# The server here is ONE REPLICA GROUP (peak_flops is a group's FLOP/s), so a
# DP deployment has `topo.replicas` such queues; a burst spreads across them
# only as well as the router balances it. Every figure below is per group.
# ============================================================================

def _prefill_service_arrays(model: Model, topo: Topology, wl, chunk: float,
                            turn_tokens: float = 0.0,
                            mfu: float = MFU_DEFAULT,
                            per_pass_overhead: bool = False,
                            n: int = 200_000, seed: int = 0):
    """Per-draw prefill service time (seconds) if the request were a MISS, and
    if it were a HIT — one pair per sampled context length.

    Both legs are returned over ALL draws rather than over the cold/warm
    subsamples: `is_cold` is drawn independently of the length in
    Workload.sample, so mixing the two legs analytically at f = invalidation
    is exact and spares the miss leg's heavy quadratic tail the sampling noise
    of a 1%-of-n subsample. Defaults n/seed match context_moments, which makes
    the means reproduce cold_request_seconds / warm_request_seconds exactly
    rather than approximately.
    """
    check_dtype_supported(model, topo)
    if not 0 < mfu <= 1:
        raise ValueError(f"mfu must be in (0, 1], got {mfu!r}")
    if chunk <= 0:
        raise ValueError(f"chunk must be > 0, got {chunk!r}")
    full, _, _, _ = wl.sample(np.random.default_rng(seed), n)
    length = full.astype(float)

    flops = (2.0 * model.params_prefill * length
             + 2.0 * model.attn_d * model.attn_layers * length ** 2)
    if per_pass_overhead:
        cold = (flops / (peak_flops(topo)
                         * mfu_ceiling(model, topo, mfu_anchor=mfu))
                + np.ceil(length / chunk)
                * prefill_overhead_seconds(model, topo))
    else:
        cold = flops / (peak_flops(topo) * mfu)

    if turn_tokens > 0:
        # a hit's cost is AFFINE in the cached length it attends over (the
        # cross term is linear in `prior` chunk by chunk), so two evaluations
        # pin the whole line; the 1e6-token lever keeps the slope out of the
        # cancellation noise of two nearly-equal big numbers
        w0 = prefill_context_seconds(model, topo, turn_tokens, chunk, mfu,
                                     prior=0.0)
        w1 = prefill_context_seconds(model, topo, turn_tokens, chunk, mfu,
                                     prior=1e6)
        warm = w0 + (w1 - w0) / 1e6 * length
    else:
        warm = np.zeros_like(length)
    return cold, warm


def prefill_service_moments(model: Model, topo: Topology, wl, chunk: float,
                            turn_tokens: float = 0.0,
                            mfu: float = MFU_DEFAULT,
                            per_pass_overhead: bool = False) -> tuple:
    """(E[S], E[S^2], E[S | miss], E[S | hit]) for the prefill server, in
    seconds, over the mixed cold/warm request stream.

    E[S] is what the duty cycle already used. E[S^2] is the new quantity: the
    Pollaczek-Khinchine wait is proportional to it, and on this workload it is
    dominated by the rare long miss (service ~ L^2 on a lognormal L), so the
    queue is far more sensitive to the tail than the duty cycle is to the
    mean. That gap is the whole reason a deployment sized at f* queues.
    """
    cold, warm = _prefill_service_arrays(model, topo, wl, chunk, turn_tokens,
                                         mfu, per_pass_overhead)
    f = wl.invalidation
    e_cold, e_warm = float(cold.mean()), float(warm.mean())
    e_s = f * e_cold + (1 - f) * e_warm
    e_s2 = f * float((cold ** 2).mean()) + (1 - f) * float((warm ** 2).mean())
    return e_s, e_s2, e_cold, e_warm


def queue_wait_seconds(model: Model, topo: Topology, wl, req_rate: float,
                       chunk: float, turn_tokens: float = 0.0,
                       mfu: float = MFU_DEFAULT,
                       per_pass_overhead: bool = False) -> float:
    """Mean time a request waits BEFORE its own prefill starts (M/G/1, FCFS).

    Pollaczek-Khinchine:  E[W] = lambda x E[S^2] / (2 (1 - rho)).

    Poisson arrivals are an assumption, and a mild one relative to real
    agentic traffic (limitation 8) — which is why the burst functions below
    exist alongside this. Returns inf at rho >= 1: the queue has no steady
    state there, which is precisely what duty > 100% means.
    """
    if req_rate < 0:
        raise ValueError(f"req_rate must be >= 0, got {req_rate!r}")
    e_s, e_s2, _, _ = prefill_service_moments(model, topo, wl, chunk,
                                              turn_tokens, mfu,
                                              per_pass_overhead)
    rho = req_rate * e_s
    if rho >= 1:
        return float("inf")
    return req_rate * e_s2 / (2 * (1 - rho))


def prefill_ttft_seconds(model: Model, topo: Topology, wl, req_rate: float,
                         chunk: float, turn_tokens: float = 0.0,
                         mfu: float = MFU_DEFAULT,
                         request: str = "cold", discipline: str = "fcfs",
                         per_pass_overhead: bool = False) -> float:
    """Mean time-to-first-token: queueing delay + this request's own prefill.

    `discipline` brackets what vLLM actually does, because vLLM is neither:

      "fcfs"  M/G/1 first-come-first-served. Every request waits the SAME
              P-K delay whatever its own size, so a 2k-token warm turn queues
              behind a 180k-token re-prefill — the convoy effect. Sensitive to
              E[S^2], i.e. to the miss tail.
      "ps"    M/G/1 processor sharing: E[T | S=s] = s / (1 - rho). Chunked
              prefill time-shares admitted requests, so a hit interleaves with
              a running miss instead of queueing behind it. Distribution-
              INsensitive — it never sees E[S^2] at all.

    Neither is uniformly the optimistic end, and which is which flips by
    class: PS charges every request in proportion to its own size, so it is
    dearer for MISSES (the long jobs) and far cheaper for HITS; FCFS charges
    one shared wait, which the short jobs cannot amortise. Read the pair as a
    two-sided bracket per class, not as a best/worst case.

    The truth is in between: vLLM admits in order but runs several admitted
    prefills concurrently, bounded by max_num_batched_tokens. Reporting the
    pair is the honest move; picking one would be a claim the study cannot
    support.
    """
    if request not in ("cold", "warm"):
        raise ValueError(f"request must be 'cold' or 'warm', got {request!r}")
    if discipline not in ("fcfs", "ps"):
        raise ValueError(f"discipline must be 'fcfs' or 'ps', got {discipline!r}")
    e_s, e_s2, e_cold, e_warm = prefill_service_moments(
        model, topo, wl, chunk, turn_tokens, mfu, per_pass_overhead)
    own = e_cold if request == "cold" else e_warm
    rho = req_rate * e_s
    if rho >= 1:
        return float("inf")
    if discipline == "ps":
        return own / (1 - rho)
    return req_rate * e_s2 / (2 * (1 - rho)) + own


def sla_miss_rate(model: Model, topo: Topology, wl, req_rate: float,
                  chunk: float, sla_seconds: float,
                  turn_tokens: float = 0.0, mfu: float = MFU_DEFAULT,
                  discipline: str = "fcfs", request: str = "cold",
                  per_pass_overhead: bool = False, hi: float = 1.0) -> float:
    """Largest miss rate whose mean TTFT still meets `sla_seconds`.

    The planning counterpart to breakeven_miss_rate: f* asks when the server
    saturates, this asks when it stops being fast enough, and the second
    always binds first. Returns 0.0 when even an all-warm stream breaches the
    SLA, and `hi` when the SLA survives all the way to f = hi (i.e. the
    constraint is not reached inside the range asked about).
    """
    if not sla_seconds > 0:
        raise ValueError(f"sla_seconds must be > 0, got {sla_seconds!r}")
    if not 0 < hi <= 1:
        raise ValueError(f"hi must be in (0, 1], got {hi!r}")

    def ttft(f):
        return prefill_ttft_seconds(model, topo, replace(wl, invalidation=f),
                                    req_rate, chunk, turn_tokens, mfu,
                                    request=request, discipline=discipline,
                                    per_pass_overhead=per_pass_overhead)

    if ttft(0.0) > sla_seconds:
        return 0.0
    if ttft(hi) <= sla_seconds:
        return hi
    lo, up = 0.0, hi
    for _ in range(60):                     # TTFT rises monotonically in f
        mid = 0.5 * (lo + up)
        if ttft(mid) <= sla_seconds:
            lo = mid
        else:
            up = mid
    return lo


def burst_drain_seconds(model: Model, topo: Topology, wl, burst: float,
                        req_rate: float, chunk: float,
                        turn_tokens: float = 0.0, mfu: float = MFU_DEFAULT,
                        per_pass_overhead: bool = False) -> float:
    """Wall-clock to clear `burst` SIMULTANEOUS cache misses, on top of the
    standing load — the deterministic-fluid drain of a correlated
    invalidation event (limitation 8's template deploy or cache flush).

    Backlog B x E[S | miss] drains at (1 - rho) seconds of work per second,
    because the standing traffic keeps arriving throughout:

        T_drain = B x E[S | miss] / (1 - rho)

    and the burst's LAST request gets its first token at T_drain whichever
    discipline runs (FCFS serves it last; processor sharing finishes the whole
    burst together). Its MEAN TTFT does differ: ~T_drain / 2 under FCFS,
    ~T_drain under PS.

    Two omissions, both making the real machine worse than this: the decode
    batch sharing each forward pass stretches the drain by t_decode / t_chunk
    (1-3% at the reference settings), and preemption/recompute under a full
    KV pool is unpriced entirely.
    """
    if burst < 0:
        raise ValueError(f"burst must be >= 0, got {burst!r}")
    e_s, _, e_cold, _ = prefill_service_moments(model, topo, wl, chunk,
                                                turn_tokens, mfu,
                                                per_pass_overhead)
    rho = req_rate * e_s
    if rho >= 1:
        return float("inf")
    return burst * e_cold / (1 - rho)


def spike_tolerance(model: Model, topo: Topology, wl, sla_seconds: float,
                    req_rate: float, chunk: float, turn_tokens: float = 0.0,
                    mfu: float = MFU_DEFAULT,
                    per_pass_overhead: bool = False) -> float:
    """COLD-SPIKE TOLERANCE: the largest simultaneous burst of misses whose
    last request still gets a first token inside `sla_seconds`.

        B* = sla x (1 - rho) / E[S | miss]

    Inverting burst_drain_seconds. Linear in the SLA, so another latency
    budget is a multiplication — the ranking between configurations does not
    move. Two factors set it, and on a MoE they pull the SAME way: a small
    active-parameter count shrinks E[S | miss], and the cheap warm turns that
    follow from it leave rho low, widening the headroom the burst drains
    into. That product is why the MoE's spike tolerance beats the dense 27B by
    MORE than its prefill-speed ratio. B* falls to zero exactly at f* — the
    duty ceiling is the point where a deployment can absorb no burst at all,
    not a point it can sit at.

    Fractional by construction (the largest integer burst is its floor).
    """
    e_s, _, e_cold, _ = prefill_service_moments(model, topo, wl, chunk,
                                                turn_tokens, mfu,
                                                per_pass_overhead)
    rho = req_rate * e_s
    if rho >= 1:
        return 0.0
    return max(0.0, sla_seconds * (1 - rho) / e_cold)


def spike_token_debt(model: Model, topo: Topology, wl, burst: float,
                     n_decode: int, req_rate: float, chunk: float,
                     turn_tokens: float = 0.0, mfu: float = MFU_DEFAULT,
                     per_pass_overhead: bool = False,
                     n_iter: int = 800, seed: int = 0) -> tuple:
    """What a burst costs the users who did NOTHING wrong.

    Returns (drain_s, itl_ratio, tokens_lost_per_user, tokens_lost_total).

    Section 8's ITL spike is one forward pass; during a drain the scheduler
    has a chunk to place in EVERY pass, so the spike is not a blip but the
    steady state for `drain_s` seconds. Each of the `n_decode` warm users
    generates at 1/ITL_mixed instead of 1/ITL_normal throughout, and the
    difference is output tokens that never arrive. This is the metric that
    makes a cold spike legible to someone reading a latency dashboard rather
    than a duty cycle.
    """
    drain = burst_drain_seconds(model, topo, wl, burst, req_rate, chunk,
                                turn_tokens, mfu, per_pass_overhead)
    d_ms, x_ms, ratio = itl_spike(model, topo, wl, n_decode, chunk, mfu,
                                  n_iter=n_iter, seed=seed)
    if not np.isfinite(drain):
        return drain, ratio, float("inf"), float("inf")
    per_user = drain * (1e3 / d_ms - 1e3 / x_ms)
    return drain, ratio, per_user, per_user * n_decode


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
                  n_iter=3000, seed=0, union="linear", mbu: float = None):
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
    # MEASURED decode efficiency (research/decode_mbu.md). Before 2026-08-28
    # this line read `bw = effective_bw(topo)` -- a pure roofline, 4.1x
    # optimistic against production. Pass mbu=1.0 to recover that reading.
    bw = effective_bw(topo) * (MBU_DEFAULT if mbu is None else mbu)
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
# THE OPERATING POINT  (research/spike.md — the two-axis planner)
# ----------------------------------------------------------------------------
# The study has always reported its constraints in DIFFERENT UNITS and refused
# to combine them. Section 8 says so outright: the prefill band is "a
# sensitivity band for the prefill axis alone, not a two-axis planner — KV
# capacity is a separate constraint in different units (sessions held vs work
# rate)". That refusal was correct at the time and is the reason a reader must
# hold two numbers at once to size a deployment.
#
# There is a way to make them commensurable, and it costs exactly two
# assumptions, both stated rather than hidden:
#
#   1. ONE USER HOLDS ONE SESSION. So a session count converts to a user count.
#   2. A USER'S MAIN-AGENT STREAM ISSUES A REQUEST EVERY `think_time_s`
#      SECONDS — the full turn-to-turn interval, open loop (the previous
#      response's service time is inside it, not on top of it). So a user
#      count converts to a request rate, and a work rate converts back to
#      users. Each main request additionally tows `wl.sub_ratio` subagent
#      requests through the prefill server, so the total arrival rate is
#      (1 + sub_ratio) x users / think_time — the same (1 + r) that the
#      service moments already carry in their class mixture.
#
# Under those, all four of the study's constraints become the SAME quantity —
# **the maximum concurrent users this configuration serves** — and the binding
# constraint is simply the smallest of them:
#
#   cache        the warm p5 user-class population that fits in the KV pool
#   decode       the concurrency at which per-user tok/s falls to the floor
#   latency      the load at which mean TTFT reaches the budget (research/
#                spike.md § 2 — this is the one the duty cycle cannot see)
#   saturation   the load at which prefill duty reaches 100% (section 8's f*,
#                re-expressed in users)
#
# Two of these depend on the miss rate f strongly and two barely at all, so
# WHICH ONE BINDS CHANGES with f — the crossover is the planner's whole point,
# and neither axis alone can show it.
#
# Both assumptions are load-bearing, so both are limitations, not conveniences:
# see docs/scenarios.md § 9 "Reading the two-axis planner". A user with several
# concurrent sessions, or bursty think time, moves the cache and latency
# frontiers in opposite directions.
#
# Assumption 2 now has a measured anchor (the MEASURED_* block below): a
# role-tagged pi-agent trace puts the open-loop interval at 43 s — waiting
# Z = 32.5 s (47% tool execution, 53% human) plus being-served R = 10.8 s on
# the traced API backend. R does NOT transfer to an on-prem deployment, which
# is exactly why the CLOSED variants of the latency and saturation ceilings
# exist: they take Z as the knob and let the model supply its own R
# (queue wait + prefill + decode), so a slower deployment stretches the
# cycle — and lightens its own arrival rate — automatically. The open
# conversion with the 30 s reference stays the default and is the
# conservative side of the measurement (30 s < 43 s measured).
# ============================================================================

DECODE_FLOOR_TOKS = 40.0    # the study's hard per-user floor (50 is comfortable)
THINK_TIME_S = 30.0         # reference: one main-agent request every 30 s
REF_USERS = 64              # reference population -> 64/30 = 2.13 req/s (main
                            # requests; the prefill server sees 1 + sub_ratio x)
OUT_TOKENS_DEFAULT = 400    # MEASURED 2026-08-27: 404.1 mean output tokens
                            # per request over ~39k requests / 7 days on a
                            # production agentic deployment (research/
                            # workload_agentic_poc.md). Was ASSUMED 1,000
                            # ("decoded share of the 2,000-token turn").

# Measured think-time anchors — role-tagged pi-agent trace, 2026-08-04:
# 8 sessions, 306 main-agent requests, 39 human turns. Regenerate with
# scripts/think_time_trace.py from an inter-event-gap CSV (the trace itself is
# not committed). The one gap > 30 min is excluded as a parked session.
MEASURED_REQ_PER_TURN = 7.8   # main-agent requests per human turn
MEASURED_T_TOOL_S = 18.3      # mean tool wait; median 0.61 s (lognormal sigma 2.43
                              # — the mean is build-dominated, 30x the median)
MEASURED_T_HUMAN_S = 275.0    # mean human wait; n = 19, tail-dominated (median 58 s)
MEASURED_THINK_Z_S = 32.5     # waiting per request (tool 47% + human 53%)
MEASURED_SERVICE_R_S = 10.8   # being-served per request ON THE TRACED API BACKEND
MEASURED_CYCLE_S = 43.3       # Z + R: the open-loop inter-request interval


def think_z(req_per_turn: float = MEASURED_REQ_PER_TURN,
            t_tool_s: float = MEASURED_T_TOOL_S,
            t_human_s: float = MEASURED_T_HUMAN_S) -> float:
    """Steady-state waiting per request: a turn is `req_per_turn` requests, the
    first n-1 each followed by a tool wait and the last by a human wait.

    With the raw measured means this returns 51 s against the directly
    measured 32.5 s (MEASURED_THINK_Z_S). The gap is censoring, not error:
    session-final turns never show their human gap (only 19 of the 39 traced
    turns have one), so the formula is the upper edge and the direct anchor
    the lower edge of an honest [32, 51] s band for Z. The knobs exist for
    sensitivity; they do not outrank the direct measurement.
    """
    if req_per_turn < 1:
        raise ValueError(f"req_per_turn must be >= 1, got {req_per_turn!r}")
    if t_tool_s < 0 or t_human_s < 0:
        raise ValueError("tool/human waits must be >= 0")
    return ((req_per_turn - 1) * t_tool_s + t_human_s) / req_per_turn


def request_rate(users: float, think_time_s: float = THINK_TIME_S,
                 sub_ratio: float = 0.0) -> float:
    """Requests/s a population of `users` generates — MAIN-agent requests by
    default (the published 2.13 req/s reference). The conversion the whole
    planner rests on (assumption 2 above). Pass `sub_ratio` to get the total
    arrival rate the prefill server actually sees: each main request tows
    that many subagent requests, and the service moments already mix the two
    classes at the same ratio."""
    if users < 0:
        raise ValueError(f"users must be >= 0, got {users!r}")
    if think_time_s <= 0:
        raise ValueError(f"think_time_s must be > 0, got {think_time_s!r}")
    if sub_ratio < 0:
        raise ValueError(f"sub_ratio must be >= 0, got {sub_ratio!r}")
    return users * (1.0 + sub_ratio) / think_time_s


def max_users_cache(model: Model, topo: Topology, wl: Workload, ram_gib=0,
                    n_iter: int = 400, draw: int = None, seed: int = 0) -> float:
    """Users whose sessions fit warm in the pool — warm p5, USER-class only.

    p5, not p50: the study plans on the conservative tail. User-class, because
    subagent sessions occupy the pool but are not users (which="user" is the
    exact count the explorer approximates as warm x (1 - p_sub)).
    """
    if draw is None:
        draw = int(4000 + kv_pool_tokens(model, topo) / 8000)
    return float(warm_capacity(model, topo, wl, ram_gib=ram_gib, n_iter=n_iter,
                               draw=draw, seed=seed, which="user")[0])


def max_users_decode(model: Model, topo: Topology, wl: Workload,
                     floor: float = DECODE_FLOOR_TOKS, union: str = "linear",
                     n_iter: int = 400, seed: int = 0, hi: int = 4096,
                     mbu: float = None) -> float:
    """Concurrent decoders at which per-user p50 tok/s falls to `floor`.

    Bisection, not the linear scan tables.py uses: per-user speed is monotone
    decreasing in concurrency (every extra sequence adds KV bytes to the same
    step), so ~12 evaluations replace up to 1,500. Returns 0 when even a single
    decoder misses the floor, and raises if the crossing lies beyond `hi`
    rather than returning a silently censored value.
    """
    def p50(n):
        return decode_curves(model, topo, wl, [n], n_iter=n_iter, seed=seed,
                             union=union, mbu=mbu)[1][0]
    if p50(1) < floor:
        return 0.0
    if p50(hi) >= floor:
        raise ValueError(f"decode floor crossing beyond hi={hi}: censored")
    lo, up = 1, hi
    while up - lo > 1:
        mid = (lo + up) // 2
        if p50(mid) >= floor:
            lo = mid
        else:
            up = mid
    return float(lo)


def steady_decode_point(model: Model, topo: Topology, wl: Workload,
                        rate_group: float,
                        out_tokens: float = OUT_TOKENS_DEFAULT,
                        union: str = "linear", n_iter: int = 400,
                        seed: int = 0, hi: int = 4096) -> dict:
    """The decode batch a given LOAD actually produces, and its per-user speed.

    Every other decode figure in this model is a stress test: max_users_decode
    and the explorer's v@warm both ask what one user gets when the whole warm
    population decodes at once. That is the right worst case and the wrong
    expectation. Arrivals are open-loop — a user fires once every think-time
    seconds and spends most of that interval waiting on a tool or a human — so
    the number of sequences in the decode batch at any instant is far below the
    warm population, and each of them runs far faster.

    Little's law on the DECODE phase alone (a request queueing for prefill, or
    being prefilled, is not yet decoding):

        E[n] = rate_group x E[seconds spent decoding]
             = rate_group x out_tokens / v(n)

    Multiply through and the fixed point is a flow balance needing no inversion:

        n x v(n)   =   rate_group x out_tokens
        (delivered output tok/s)  (demanded output tok/s)

    n x v(n) is the aggregate decode curve, strictly increasing in n, so the
    crossing is unique. Bracketed by doubling and bisected on the integers the
    curve is actually sampled at, then interpolated across the final unit
    interval; `rate_group` is the TOTAL req/s ONE replica group sees (main +
    subagent), the same unit every queue metric here uses.

    Returns {"n", "per_user_tok_s", "demand_tok_s", "delivered_tok_s",
    "demanded_tok_s", "saturated"}, with n and per_user_tok_s per replica group
    and both aggregates system-wide. delivered and demanded are equal at a real
    fixed point and DIFFER when saturated — the batch cannot retire what the
    load asks for, which is what "no steady state" means. They were one field;
    the explorer's chart D read it as demand and so mislabelled the saturated
    marker by up to 24x.
    `saturated` is True when the demand exceeds what `hi` concurrent decoders
    could retire: no steady state exists, so n is a floor, not an estimate.

    Approximations, both material enough to state wherever this is quoted:
      MEAN FIELD  v is evaluated at the MEAN batch, not averaged over the
          batch-size distribution. v is convex in n, so E[v(N)] >= v(E[N]) by
          Jensen: this returns the conservative side.
      DECODE ONLY  prefill chunks sharing a forward pass are priced separately
          (itl_spike); this is the clean-decode speed BETWEEN those spikes.
    It also says nothing about whether the requests get served at all — if
    prefill_duty() is at 1.0 the queue is unbounded and no steady state exists
    upstream of this one. Check that first; the explorer does.

    Mirrors steadyDecodePoint() in interactive/src/prefill.js.
    """
    if rate_group < 0:
        raise ValueError(f"rate_group must be >= 0, got {rate_group!r}")
    if out_tokens < 0:
        raise ValueError(f"out_tokens must be >= 0, got {out_tokens!r}")
    if hi < 1:
        raise ValueError(f"hi must be >= 1, got {hi!r}")
    reps = topo.replicas
    demand = rate_group * out_tokens

    def v(n: int) -> float:
        return float(decode_curves(model, topo, wl, [n], n_iter=n_iter,
                                   seed=seed, union=union)[1][0])

    out = {"demand_tok_s": demand, "saturated": False}
    if demand == 0:                        # no load: nothing is decoding
        return {**out, "n": 0.0, "per_user_tok_s": v(1),
                "delivered_tok_s": 0.0, "demanded_tok_s": 0.0}
    v1 = v(1)
    if demand <= v1:
        # below one decoder the batch holds a single sequence whenever anyone
        # is decoding at all, so v stays v(1) and the fixed point is exact
        return {**out, "n": demand / v1, "per_user_tok_s": v1,
                "delivered_tok_s": demand * reps,
                "demanded_tok_s": demand * reps}
    lo, up = 1, 2
    while up <= hi and up * v(up) < demand:
        lo, up = up, up * 2
    if up > hi:
        # the study does not do silent caps: say the demand ran off the end
        # delivered != demanded here, which is exactly what saturated means:
        # one field for both silently understated the demand by up to 24x
        return {**out, "n": float(hi), "per_user_tok_s": v(hi),
                "delivered_tok_s": hi * v(hi) * reps,
                "demanded_tok_s": demand * reps, "saturated": True}
    while up - lo > 1:
        mid = (lo + up) // 2
        if mid * v(mid) < demand:
            lo = mid
        else:
            up = mid
    a_lo, a_up = lo * v(lo), up * v(up)
    # linear across the final unit interval — the aggregate curve is smooth
    # and near-linear over one sequence, so this is the last digit, not a model
    frac = 0.0 if a_up <= a_lo else (demand - a_lo) / (a_up - a_lo)
    n = lo + min(max(frac, 0.0), 1.0)
    # per-user speed is the demand shared out: at the fixed point the batch
    # delivers exactly what the load asks for, by construction
    return {**out, "n": n, "per_user_tok_s": demand / n,
            "delivered_tok_s": demand * reps,
            "demanded_tok_s": demand * reps}


def _check_closed_args(closed_z_s: float, out_tokens: float,
                       decode_toks: float) -> None:
    if closed_z_s < 0:
        raise ValueError(f"closed_z_s must be >= 0, got {closed_z_s!r}")
    if out_tokens < 0:
        raise ValueError(f"out_tokens must be >= 0, got {out_tokens!r}")
    if decode_toks <= 0:
        raise ValueError(f"decode_toks must be > 0, got {decode_toks!r}")


def closed_request_rate(model: Model, topo: Topology, wl, users: float,
                        chunk: float, turn_tokens: float = 0.0,
                        z_think_s: float = MEASURED_THINK_Z_S,
                        out_tokens: float = OUT_TOKENS_DEFAULT,
                        decode_toks: float = DECODE_FLOOR_TOKS,
                        mfu: float = MFU_DEFAULT, discipline: str = "fcfs",
                        per_pass_overhead: bool = False) -> float:
    """MAIN-agent req/s a CLOSED population of `users` sustains: the fixed
    point of  users = lam_total (Z + R(lam_total)) / (1 + r)  with R the
    prefill sojourn plus decode time. The right side is strictly increasing
    in lam_total on [0, 1/E[S]) and diverges at the wall, so every finite
    population has exactly one rate — found by bisection, returned divided
    by (1 + r) to match request_rate()'s main-agent unit.
    """
    if users < 0:
        raise ValueError(f"users must be >= 0, got {users!r}")
    _check_closed_args(z_think_s, out_tokens, decode_toks)
    if users == 0:
        return 0.0
    e_s, e_s2, _, _ = prefill_service_moments(model, topo, wl, chunk,
                                              turn_tokens, mfu,
                                              per_pass_overhead)
    if e_s <= 0:
        return float("inf")
    dec = out_tokens / decode_toks
    one_r = 1.0 + wl.sub_ratio

    def pop(lam):
        if discipline == "ps":
            resp = e_s / (1 - lam * e_s)
        else:
            resp = lam * e_s2 / (2 * (1 - lam * e_s)) + e_s
        return lam * (z_think_s + resp + dec) / one_r

    lo, hi = 0.0, (1.0 / e_s) * (1 - 1e-12)
    for _ in range(200):
        mid = (lo + hi) / 2
        if pop(mid) < users:
            lo = mid
        else:
            hi = mid
    return lo / one_r


def max_users_saturation(model: Model, topo: Topology, wl, chunk: float,
                         turn_tokens: float = 0.0,
                         think_time_s: float = THINK_TIME_S,
                         mfu: float = MFU_DEFAULT,
                         per_pass_overhead: bool = False,
                         closed_z_s: float = None,
                         out_tokens: float = OUT_TOKENS_DEFAULT,
                         decode_toks: float = DECODE_FLOOR_TOKS) -> float:
    """Users at which prefill duty reaches 100% — section 8's f*, in users.

    OPEN (default): rho = lambda E[S] = 1 at lambda = 1/E[S] and each user
    contributes (1 + sub_ratio) requests per think interval, so
    users = think_time / ((1 + r) E[S]). Above this the queue has no steady
    state at all.

    CLOSED (`closed_z_s` set): a session cannot fire while it is being
    served, so the queue never diverges — throughput saturates instead. The
    reported ceiling is the balanced-bounds knee of the interactive
    (machine-repairman) model, N* = (Z + R0) / D: below it users add
    throughput, above it they only add latency. R0 is the zero-load response
    (own prefill + decoding `out_tokens` at `decode_toks` — the study's
    floor, i.e. the slowest acceptable decode) and D = (1 + r) E[S] is the
    prefill demand one user-cycle places on the bottleneck. Subagent requests
    are priced as demand, not as cycle time: they overlap the main turn.
    """
    e_s, _, _, _ = prefill_service_moments(model, topo, wl, chunk, turn_tokens,
                                           mfu, per_pass_overhead)
    demand = (1.0 + wl.sub_ratio) * e_s
    if demand <= 0:
        return float("inf")
    if closed_z_s is None:
        return think_time_s / demand
    _check_closed_args(closed_z_s, out_tokens, decode_toks)
    r0 = e_s + out_tokens / decode_toks
    return (closed_z_s + r0) / demand


def max_users_latency(model: Model, topo: Topology, wl, chunk: float,
                      sla_seconds: float, turn_tokens: float = 0.0,
                      think_time_s: float = THINK_TIME_S,
                      mfu: float = MFU_DEFAULT, discipline: str = "fcfs",
                      per_pass_overhead: bool = False,
                      closed_z_s: float = None,
                      out_tokens: float = OUT_TOKENS_DEFAULT,
                      decode_toks: float = DECODE_FLOOR_TOKS) -> float:
    """Users at which a MISS's mean TTFT reaches `sla_seconds`.

    Closed form in both disciplines, because E[S] and E[S^2] do not depend on
    the arrival rate:
        FCFS  lam a / (2(1 - lam b)) + c = SLA  ->  lam = k / (a + k b)
        PS    c / (1 - lam b) = SLA            ->  lam = (1 - c/SLA) / b
    with a = E[S^2], b = E[S], c = E[S | miss], k = 2(SLA - c).

    `lam` is the TOTAL arrival rate (the moments mix both request classes),
    and each user contributes (1 + sub_ratio) requests per interval, so the
    OPEN conversion is users = lam think_time / (1 + r).

    CLOSED (`closed_z_s` set): the population that sustains `lam` also spends
    each cycle being served, so users = lam (Z + R(lam)) / (1 + r), with
    R = mean prefill sojourn at lam (wait + own prefill; PS: b/(1 - lam b))
    plus decoding `out_tokens` at `decode_toks`. No fixed point is needed:
    lam_sla does not depend on the population, only the conversion back to
    users does. The closed count is ALWAYS >= the open count at the same
    interval parameter — response feedback stretches the cycle.

    ALWAYS strictly inside max_users_saturation (open vs open): k/(a + k b)
    < 1/b for any a > 0, which is the algebraic statement of "the queue
    diverges before the server does" — the section's headline, and asserted
    in the self-checks. Returns 0 when the request's own prefill already
    exceeds the budget, i.e. when no load at all can meet it.
    """
    if discipline not in ("fcfs", "ps"):
        raise ValueError(f"discipline must be 'fcfs' or 'ps', got {discipline!r}")
    if not sla_seconds > 0:
        raise ValueError(f"sla_seconds must be > 0, got {sla_seconds!r}")
    a, b, c = (lambda m: (m[1], m[0], m[2]))(
        prefill_service_moments(model, topo, wl, chunk, turn_tokens, mfu,
                                per_pass_overhead))
    if c >= sla_seconds or b <= 0:
        return 0.0
    if discipline == "ps":
        lam = (1 - c / sla_seconds) / b
    else:
        k = 2 * (sla_seconds - c)
        lam = k / (a + k * b)
    if closed_z_s is None:
        return max(0.0, lam * think_time_s / (1.0 + wl.sub_ratio))
    _check_closed_args(closed_z_s, out_tokens, decode_toks)
    if discipline == "ps":
        resp = b / (1 - lam * b)
    else:
        resp = lam * a / (2 * (1 - lam * b)) + b
    cycle = closed_z_s + resp + out_tokens / decode_toks
    return max(0.0, lam * cycle / (1.0 + wl.sub_ratio))


def operating_point(model: Model, topo: Topology, wl: Workload, users: float,
                    chunk: float = CHUNK_DEFAULT, turn_tokens: float = 2_000,
                    sla_seconds: float = 10.0,
                    think_time_s: float = THINK_TIME_S,
                    decode_floor: float = DECODE_FLOOR_TOKS,
                    mfu: float = MFU_DEFAULT, discipline: str = "fcfs",
                    ram_gib=0, union: str = "linear",
                    per_pass_overhead: bool = False,
                    closed: bool = False,
                    z_think_s: float = MEASURED_THINK_Z_S,
                    out_tokens: float = OUT_TOKENS_DEFAULT,
                    n_iter: int = 400, seed: int = 0) -> dict:
    """All four ceilings in ONE unit — max concurrent users — plus which binds.

    THE two-axis planner. `binding` is the argmin: whichever ceiling is lowest
    is the one that actually limits this deployment, and `headroom` is how much
    of it the requested population uses. Everything is per replica GROUP; a DP
    deployment multiplies the cache and decode ceilings by `topo.replicas` only
    under balanced routing, which sticky routing works against (spike.md #3).

    `closed=True` switches the latency and saturation columns to the
    closed-loop conversion: `z_think_s` (waiting only — measured 32.5 s)
    replaces `think_time_s` (the full interval — measured 43 s, reference
    30 s), and the model supplies its own service time. Cache is unchanged
    on purpose (held sessions occupy KV whether or not the user is active),
    and decode keeps its published worst-case reading (every user decoding
    at once) rather than the closed steady-state duty — both are stated in
    docs/scenarios.md § 9.
    """
    if users < 0:
        raise ValueError(f"users must be >= 0, got {users!r}")
    z = z_think_s if closed else None
    ceilings = {
        "cache": max_users_cache(model, topo, wl, ram_gib=ram_gib,
                                 n_iter=n_iter, seed=seed),
        "decode": max_users_decode(model, topo, wl, floor=decode_floor,
                                   union=union, n_iter=n_iter, seed=seed),
        "latency": max_users_latency(model, topo, wl, chunk, sla_seconds,
                                     turn_tokens, think_time_s, mfu,
                                     discipline, per_pass_overhead,
                                     closed_z_s=z, out_tokens=out_tokens,
                                     decode_toks=decode_floor),
        "saturation": max_users_saturation(model, topo, wl, chunk, turn_tokens,
                                           think_time_s, mfu,
                                           per_pass_overhead,
                                           closed_z_s=z, out_tokens=out_tokens,
                                           decode_toks=decode_floor),
    }
    binding = min(ceilings, key=ceilings.get)
    limit = ceilings[binding]
    return {
        "users": users,
        # MAIN-agent req/s in BOTH modes (the prefill server sees (1 + r) x
        # this). Open: the assumption-2 conversion. Closed: solved from the
        # cycle fixed point — response feedback slows the cadence, so this
        # is below users / z_think_s by construction.
        "req_rate": (closed_request_rate(model, topo, wl, users, chunk,
                                         turn_tokens, z_think_s, out_tokens,
                                         decode_floor, mfu, discipline,
                                         per_pass_overhead)
                     if closed else request_rate(users, think_time_s)),
        "ceilings": ceilings,
        "binding": binding,
        "limit": limit,
        "headroom": (users / limit) if limit > 0 else float("inf"),
        "fits": users <= limit,
    }


# ============================================================================
# THE ELECTRICITY BILL  (research/power.md)
# ----------------------------------------------------------------------------
# Wall power from the duty cycle the model already computes. Three GPU states,
# priced per research/power.md's measured anchors:
#
#   prefill   d_p = the prefill duty rho — compute-bound, power-cap-limited
#             (~0.90 x TDP, FLAT across the MFU band: the cap binds first)
#   decode    d_d — bandwidth-bound, well under TDP (~0.55 x, the softest
#             constant, band 0.45-0.75)
#   idle      the remainder, at warm-idle watts
#
# d_p and d_d PARTITION time (no double counting — power.md #4's integrator
# trap): d_d is the decode demand the load implies against the decode capacity
# at the 40 tok/s floor, capped at whatever prefill leaves.
# P_total = n_gpu x (P_gpu + host) x PUE. Mirrors the explorer's powerDraw /
# energyCost (interactive/src/cost.js, "THE ELECTRICITY BILL") — same
# arithmetic, same clamps, so the two cannot disagree.
#
# STATUS: mixed provenance, weaker than the capacity notes. TDPs and system
# maxima are vendor SPEC; idle and the 0.55/0.90 phase split are MEASURED on
# Hopper (none on this exact H200-SXM/vLLM operating point); every B300 duty
# figure is an extrapolation rule. GPU term carries ±20-25%; the host adder is
# a spec ceiling used flat; pue and eur_kwh are exact user-chosen multipliers,
# not model error.
# ============================================================================

PUE_DEFAULT = 1.5            # colo — Uptime Institute 2024 survey: mean 1.56,
                             # capacity-weighted 1.47. Presets the explorer
                             # offers: 2.0 (server room, LBNL small-DC), 1.2
                             # (hyperscale). SURVEY, user-chosen multiplier.
EUR_PER_KWH_DEFAULT = 0.19   # Eurostat non-household EU average €0.1902/kWh
                             # H1-2025; country spread €0.08 (FI) - €0.26 (CY).
                             # STATISTICAL (HIGH), user-chosen multiplier.
HOURS_PER_MONTH = 720.0      # the explorer's flat billing month (30 x 24 h)

# Output tokens ONE REQUEST decodes (applied to every request, subagents
# included) — MEASURED 2026-08-27 (research/workload_agentic_poc.md): 404.1
# mean over ~39k production requests; was assumed 1,000. At the observed
# 50-90 tok/s this decodes in ~5-8 s of the measured 10.8 s served per turn
# (MEASURED_SERVICE_R_S) — the remainder is prefill + queue. Scales d_d linearly;
# the €/1M-token figure moves hyperbolically (fixed idle/prefill watts
# amortise as outputs lengthen); the €/month bill moves least (decode is one
# term of three). Mirrors the explorer's AVG_OUT_TOK.
#
# ALIAS, not a second constant: the closed-loop conversion and the steady-state
# decode point price exactly the same quantity, and OUT_TOKENS_DEFAULT got there
# first. Two literals that must stay equal is a drift bug waiting to happen.
AVG_OUT_TOK = OUT_TOKENS_DEFAULT


def power_draw(model: Model, topo: Topology, wl: Workload, rate_group: float,
               decode_users_group: float, chunk: float = CHUNK_DEFAULT,
               # 2_000 = the study's reference warm turn (operating_point's own
               # default): 0.0 would price every warm hit at zero machine time
               # and silently under-bill relative to the explorer
               turn_tokens: float = 2_000, pue: float = PUE_DEFAULT,
               mfu: float = MFU_DEFAULT, out_tokens: float = AVG_OUT_TOK,
               per_pass_overhead: bool = False) -> dict:
    """Average draw of one GPU and the whole system, at the current load.

    `rate_group` is the TOTAL req/s ONE replica group sees (main + subagent —
    the serverRate/replicas figure, same unit the queue metrics price);
    `decode_users_group` is the per-group decode ceiling at the 40 tok/s floor
    (its capacity proxy, max_users_decode). Returns a dict:

      d_p        prefill duty = min(1, rate x E[S]) — E[S] is the mixed
                 cold/warm prefill service time the spike model already
                 computes, so d_p IS the section-8 duty cycle, clamped
      d_d        decode-active fraction: the output-token demand
                 (rate x out_tokens) against the floor capacity
                 (decode_users_group x 40 tok/s), capped at whatever prefill
                 leaves (partition, no overlap); a non-positive capacity
                 falls back to d_d = 1 - d_p (the explorer's guard)
      per_gpu_w  d_p x p_prefill + d_d x p_decode + remainder x idle
      kw         n_gpu x (per_gpu_w + host_w) x pue / 1000 — at the meter
      pue        echoed multiplier

    Provenance: the phase watts are MEASURED on Hopper (H100 proxies; B300
    rows entirely extrapolated — see the GPU dataclass), the duty split is
    this study's model, and out_tokens is ASSUMED. Mirrors the explorer's
    powerDraw() exactly.
    """
    if rate_group < 0:
        raise ValueError(f"rate_group must be >= 0, got {rate_group!r}")
    if decode_users_group < 0:
        raise ValueError(
            f"decode_users_group must be >= 0, got {decode_users_group!r}")
    if pue <= 0:
        raise ValueError(f"pue must be > 0, got {pue!r}")
    g = topo.gpu
    if g.tdp_w <= 0:
        raise ValueError(f"{g.name}: power constants unset (research/power.md)")
    # E[S] mixes the classes at f = wl.invalidation — identical to the
    # explorer's f*mo.miss + (1-f)*mo.hit (prefill_service_moments returns
    # exactly that mixture as its first element)
    e_s = prefill_service_moments(model, topo, wl, chunk, turn_tokens, mfu,
                                  per_pass_overhead)[0]
    d_p = min(1.0, rate_group * e_s)
    if out_tokens < 0:
        raise ValueError(f"out_tokens must be >= 0, got {out_tokens!r}")
    demand = rate_group * out_tokens                     # output tok/s asked
    cap = decode_users_group * DECODE_FLOOR_TOKS         # output tok/s at floor
    d_d = min(max(0.0, 1.0 - d_p), demand / cap if cap > 0 else 1.0)
    per_gpu_w = (d_p * g.p_prefill_w + d_d * g.p_decode_w
                 + max(0.0, 1.0 - d_p - d_d) * g.idle_w)
    kw = topo.n_gpu * (per_gpu_w + g.host_w) * pue / 1000.0
    return {"d_p": d_p, "d_d": d_d, "per_gpu_w": per_gpu_w, "kw": kw,
            "pue": pue}


def energy_cost(model: Model, topo: Topology, wl: Workload, rate_group: float,
                decode_users_group: float, users: float,
                chunk: float = CHUNK_DEFAULT, turn_tokens: float = 2_000,
                pue: float = PUE_DEFAULT,
                eur_kwh: float = EUR_PER_KWH_DEFAULT,
                mfu: float = MFU_DEFAULT, out_tokens: float = AVG_OUT_TOK,
                per_pass_overhead: bool = False,
                eur_gpu_h: float = None) -> dict:
    """€ figures on top of power_draw() — the explorer's energyCost().

    720 h/month flat. Two lines: eur_month is the ELECTRICITY (the power
    model's own output); hw_month is GPU-hours at the rental rate
    (eur_gpu_h, default the part's list price; the explorer's slider);
    total_month their sum, and eur_user / eur_mtok divide the TOTAL — a
    €/Mtok that priced the watts but not the silicon read an order of
    magnitude too cheap. The €/1M-output-tokens figure divides the whole
    system's cost rate by the output-token rate the load implies
    (rate_group x replicas x out_tokens — every group assumed equally
    loaded, the same symmetry the rest of the study uses), and is Infinity
    at zero output rather than a silent zero. eur_user divides the monthly
    total across `users` (floored at 1, matching the explorer). pue and
    eur_kwh are exact user-chosen multipliers on the ELECTRICITY line —
    linear in each, asserted in _selfcheck — so tariff/facility scenarios
    are one multiply, never a re-model. The €-per-token figure inherits
    out_tokens' ASSUMED status linearly.
    """
    if users < 0:
        raise ValueError(f"users must be >= 0, got {users!r}")
    if eur_kwh < 0:
        raise ValueError(f"eur_kwh must be >= 0, got {eur_kwh!r}")
    if eur_gpu_h is None:
        eur_gpu_h = topo.gpu.eur_gpu_h
    if eur_gpu_h < 0:
        raise ValueError(f"eur_gpu_h must be >= 0, got {eur_gpu_h!r}")
    # keywords past `chunk`: power_draw grew an out_tokens parameter, and a
    # positional tail would silently feed per_pass_overhead into it
    p = power_draw(model, topo, wl, rate_group, decode_users_group, chunk,
                   turn_tokens=turn_tokens, pue=pue, mfu=mfu,
                   out_tokens=out_tokens, per_pass_overhead=per_pass_overhead)
    eur_month = p["kw"] * HOURS_PER_MONTH * eur_kwh
    hw_month = topo.n_gpu * eur_gpu_h * HOURS_PER_MONTH
    total_month = eur_month + hw_month
    out_tok_s = rate_group * topo.replicas * out_tokens
    eur_mtok = ((total_month / HOURS_PER_MONTH / 3600.0) / out_tok_s * 1e6
                if out_tok_s > 0 else float("inf"))
    return {**p, "eur_month": eur_month, "hw_month": hw_month,
            "total_month": total_month,
            "eur_user": total_month / max(1.0, users),
            "eur_mtok": eur_mtok}


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

    # ---- the grid is what makes DP expressible for the 2026-07+ models ------
    # MM35, GLM-5.2, DSv4-Flash, Qwen3.8-Flash-Next and GLM-5.3-Flash fit no
    # single H200, so pure DP is a 0 pool at every N -- the study's existing
    # "does not fit" sentinel, and it stands.
    # (GLM-5.3-Flash appears as its BF16-KV arm wherever an H200 topology is
    # priced: the fp8-KV arm is Blackwell-only and check_dtype_supported
    # refuses it there — asserted below)
    def _arm(mk, gk):
        m_ = MODELS[mk]
        return (with_kv_dtype(m_, "fp16")
                if m_.kv_fp8_blackwell_only and gk == "H200" else m_)
    for mdl in (MODELS["MM35"], MODELS["GLM52"], MODELS["DSV4F"],
                MODELS["Q38FN"], _arm("GLM53F", "H200")):
        for n in (1, 2, 4, 8):
            assert kv_pool_tokens(mdl, topology("dp", n)) == 0
    # ...but replicating GROUPS does hold a real pool on one 8-GPU node.
    assert min_tp_for(MODELS["MM35"], "H200") == 2
    assert min_tp_for(MODELS["MM35"], "B300") == 1
    assert min_tp_for(MODELS["GLM52"], "H200") == 7
    assert min_tp_for(MODELS["GLM52"], "B300") == 3
    assert min_tp_for(MODELS["DSV4F"], "H200") == 2
    assert min_tp_for(MODELS["DSV4F"], "B300") == 1
    assert min_tp_for(MODELS["Q38FN"], "H200") == 2
    assert min_tp_for(MODELS["Q38FN"], "B300") == 1
    assert min_tp_for(_arm("GLM53F", "H200"), "H200") == 3
    assert min_tp_for(MODELS["GLM53F"], "B300") == 2
    assert kv_pool_tokens(MODELS["MM35"], topology_grid(4, 2)) > 0    # DP4xTP2
    assert kv_pool_tokens(MODELS["GLM52"], topology_grid(2, 4, "B300")) > 0
    # min_tp is exactly the boundary: one GPU less holds nothing
    for mk, gk in (("MM35", "H200"), ("MM35", "B300"),
                   ("GLM52", "H200"), ("GLM52", "B300"),
                   ("DSV4F", "H200"), ("DSV4F", "B300"),
                   ("Q38FN", "H200"), ("Q38FN", "B300"),
                   ("GLM53F", "H200"), ("GLM53F", "B300")):
        need = min_tp_for(_arm(mk, gk), gk)
        assert kv_pool_tokens(_arm(mk, gk), topology_grid(1, need, gk)) > 0
        if need > 1:
            assert kv_pool_tokens(_arm(mk, gk), topology_grid(1, need - 1, gk)) == 0
    # node_splits offers exactly the fitting divisors of the node, widest DP first
    for mk, gk, want in (("MM35",  "H200", [(4, 2), (2, 4), (1, 8)]),
                         ("MM35",  "B300", [(8, 1), (4, 2), (2, 4), (1, 8)]),
                         ("GLM52", "H200", [(1, 8)]),
                         ("GLM52", "B300", [(2, 4), (1, 8)]),
                         ("DSV4F", "H200", [(4, 2), (2, 4), (1, 8)]),
                         ("DSV4F", "B300", [(8, 1), (4, 2), (2, 4), (1, 8)]),
                         ("Q38FN", "H200", [(4, 2), (2, 4), (1, 8)]),
                         ("Q38FN", "B300", [(8, 1), (4, 2), (2, 4), (1, 8)]),
                         ("GLM53F", "H200", [(2, 4), (1, 8)]),
                         ("GLM53F", "B300", [(4, 2), (2, 4), (1, 8)])):
        got = [(t.dp, t.tp) for t in node_splits(_arm(mk, gk), gk, node=8)]
        assert got == want, f"node_splits({mk}, {gk}) = {got}, want {want}"
        for t in node_splits(_arm(mk, gk), gk, node=8):
            assert t.n_gpu == 8 and kv_pool_tokens(_arm(mk, gk), t) > 0
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

    # DSv4-Flash identities (research/model_dsv4flash.md): compressed caches
    dsf = MODELS["DSV4F"]
    assert dsf.kv_bpt == 21 * 576 / 4 + 20 * 576 / 128 + 21 * 64 / 4   # 3,450 B
    assert dsf.deltanet_state == 46 * 128 * 576 + 12_206_080    # windows + fp32 state
    assert not dsf.state_fp32_ok and all(
        MODELS[k].state_fp32_ok for k in MODELS if k != "DSV4F")
    assert dsf.w_route_pertok == 6 * 13_369_344 * 43            # FP4 experts + E8M0 scales
    assert dsf.w_route_total == 256 * 13_369_344 * 43
    assert abs(dsf.w_route_total / dsf.w_route_pertok - 256 / 6) < 1e-9  # kink ~42.7
    assert dsf.kv_decode_bpt == 21 * 64 / 4 + 20 * 576 / 128    # scan + dense HCA
    assert dsf.kv_decode_const == 21 * 512 * 576 + 43 * 128 * 576
    assert dsf.kv_decode_topk == 2_048
    assert abs(dsf.attn_layers * dsf.attn_d - (21 * 1024 + 20 * 256)) < 1e-9
    assert dsf.nvfp4_w is None                                  # no NVFP4 variant modelled
    try:
        with_weight_dtype(dsf, "nvfp4"); raise AssertionError("expected ValueError")
    except ValueError:
        pass

    # Qwen3.8-Flash-Next identities (research/model_qwen38flashnext.md)
    q38 = MODELS["Q38FN"]
    assert q38.kv_bpt == 12 * 2 * 256 * 2 + 12 * 128 / 4        # KV + compressed indexer
    assert q38.deltanet_state == 36 * 48 * 128 * 128 * 2 + 36 * 10_240 * 4 * 2
    assert q38.w_route_pertok == 10 * 4_915_800 * 48            # FP8 experts + block scales
    assert q38.w_route_total == 512 * 4_915_800 * 48
    assert abs(q38.w_route_total / q38.w_route_pertok - 51.2) < 1e-9  # deepest kink
    assert q38.kv_decode_bpt == 12 * 128 / 4                    # indexer scan
    assert q38.kv_decode_const == 12 * 2_048 * 1_024            # top-budget full-KV reads
    assert q38.kv_decode_topk == 2_048
    # the FP8 ckpt quantizes ONLY experts + n-gram table: the always-active
    # read is BF16-heavy (~2 B/param on 4.3e9 params) and the resident total
    # (n-gram table, embed, vision, MTP) far exceeds what a step can touch
    assert q38.w_decode_shared == 8_623_999_000                 # exact ledger sum (note #4)
    assert q38.w_resident > q38.w_route_total + q38.w_decode_shared + 50e9
    assert q38.nvfp4_w is None                                  # no official NVFP4 (note #4)
    try:
        with_weight_dtype(q38, "nvfp4"); raise AssertionError("expected ValueError")
    except ValueError:
        pass
    assert q38.state_fp32_ok and q38.kv_fp16_ok                 # bf16 DN state; no fp8-KV assert
    # FP16 KV on the sparse path: pool bytes double AND the top-k main-KV
    # gathers double; the fp8 indexer scan does not
    q38_16 = with_kv_dtype(q38, "fp16")
    assert q38_16.kv_bpt == 2 * q38.kv_bpt
    assert q38_16.kv_decode_const == 2 * q38.kv_decode_const
    assert q38_16.kv_decode_bpt == q38.kv_decode_bpt

    # GLM-5.3-Flash identities (research/model_glm53flash.md)
    g53 = MODELS["GLM53F"]
    # STORAGE charges 12 DSA stacks (11 main + the MTP draft layer's — the
    # GLM-5.2 convention, "incl. MTP layer"); DECODE charges the 11 main
    # layers only, also the GLM-5.2 convention (78-of-79 there)
    assert g53.kv_bpt == 12 * 512 + 12 * 132 / 4                # nope MLA + compressed indexer
    assert g53.deltanet_state == 34 * 64 * 128 * 128 * 2 + 34 * 3 * 8_192 * 4 * 2
    assert g53.w_route_pertok == 8 * 25_171_968 * 42            # FP8 experts + F32 block scales
    assert g53.w_route_total == 288 * 25_171_968 * 42
    assert abs(g53.w_route_total / g53.w_route_pertok - 36) < 1e-9  # kink at 288/8
    assert g53.kv_decode_bpt == 11 * 132 / 4                    # compressed indexer scan
    assert g53.kv_decode_const == 11 * 2_048 * 512              # top-2048 latent reads
    assert g53.kv_decode_topk == 2_048
    assert g53.w_decode_shared == 13_957_216_504                # exact ledger sum (note #4)
    # the note's closing identity, to the byte: shared per-step read + all
    # routed experts + embed + MTP layer + vision tower = the checkpoint
    assert (g53.w_decode_shared + g53.w_route_total
            + 1_268_776_960 + 7_493_399_168 + 1_127_254_016) == g53.w_resident
    assert g53.nvfp4_w is None                                  # no official NVFP4 (note #4)
    try:
        with_weight_dtype(g53, "nvfp4"); raise AssertionError("expected ValueError")
    except ValueError:
        pass
    # bf16 KDA state; BF16 KV is not merely allowed but REQUIRED on Hopper
    # (fp8 KV is Blackwell-only — vLLM recipe, note #2), so the FP16 toggle
    # must stay live and scale the sparse constants like Q38FN's
    assert g53.state_fp32_ok and g53.kv_fp16_ok
    g53_16 = with_kv_dtype(g53, "fp16")
    assert g53_16.kv_bpt == 2 * g53.kv_bpt
    assert g53_16.kv_decode_const == 2 * g53.kv_decode_const
    assert g53_16.kv_decode_bpt == g53.kv_decode_bpt
    # ...and the GPU-coupled gate: the fp8-KV arm must RAISE on H200 while
    # the BF16 arm prices there and both arms price on Blackwell. Only
    # GLM-5.3-Flash carries the flag.
    assert g53.kv_fp8_blackwell_only and all(
        not MODELS[k].kv_fp8_blackwell_only for k in MODELS if k != "GLM53F")
    assert g53.kv_dtype == "fp8" and g53_16.kv_dtype == "fp16"
    b2 = topology("tp", 2, "B300")
    try:
        kv_pool_tokens(g53, topology("tp", 4))
        raise AssertionError("expected ValueError: fp8 KV on H200")
    except ValueError:
        pass
    assert kv_pool_tokens(g53_16, topology("tp", 4)) > 0
    assert kv_pool_tokens(g53, b2) > 0 and kv_pool_tokens(g53_16, b2) > 0

    # KV dtype: Mistral doubles like the Qwens; GLM's DSA path and DSv4-Flash's
    # CSA path must refuse FP16 (both serve only with a quantized main KV)
    assert with_kv_dtype(mm, "fp16").kv_bpt == 2 * mm.kv_bpt
    for quant_only in (glm, dsf):
        try:
            with_kv_dtype(quant_only, "fp16")
            raise AssertionError("expected ValueError")
        except ValueError:
            pass

    # context caps (owner decision 2026-07): Qwens + GLM allow up to 1M
    # (Qwen native 262k, 1M via YaRN); Mistral's hard model max is 262,144;
    # DSv4-Flash is natively 1M (YaRN x16 baked into its config)
    assert (m27.max_ctx == m35.max_ctx == glm.max_ctx == dsf.max_ctx
            == q38.max_ctx == g53.max_ctx == 1_048_576)
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
    # pricing of the same bytes at long context (that is DSA's entire point);
    # same check for DSv4-Flash's compressed-sparse reads
    glm_dense_read = replace(glm, kv_decode_bpt=None, kv_decode_const=0.0)
    t8 = topology("tp", 8)
    _, p_dsa, _, _ = decode_curves(glm, t8, wl, [64], n_iter=300)
    _, p_dense, _, _ = decode_curves(glm_dense_read, t8, wl, [64], n_iter=300)
    assert p_dsa[0] > p_dense[0], "DSA decode must out-speed full-cache reads"
    dsf_dense_read = replace(dsf, kv_decode_bpt=None, kv_decode_const=0.0)
    _, p_csa, _, _ = decode_curves(dsf, tp2, wl, [64], n_iter=300)
    _, p_full, _, _ = decode_curves(dsf_dense_read, tp2, wl, [64], n_iter=300)
    assert p_csa[0] > p_full[0], "CSA decode must out-speed full-cache reads"
    q38_dense_read = replace(q38, kv_decode_bpt=None, kv_decode_const=0.0)
    _, p_qsa, _, _ = decode_curves(q38, tp2, wl, [64], n_iter=300)
    _, p_qfull, _, _ = decode_curves(q38_dense_read, tp2, wl, [64], n_iter=300)
    assert p_qsa[0] > p_qfull[0], "QSA decode must out-speed full-cache reads"
    # ...and the FP16-KV arm must actually decode slower (the doubled top-k
    # read) — the property with_kv_dtype's kv_decode_const scaling exists for
    _, p_q16, _, _ = decode_curves(with_kv_dtype(q38, "fp16"), tp2, wl, [64],
                                   n_iter=300)
    assert p_q16[0] < p_qsa[0], "FP16 KV must decode slower on the QSA path too"
    # GLM-5.3-Flash sparse decode: DSA-beats-dense on H200 runs the BF16 arm
    # (the only servable one there); the fp8-vs-BF16 comparison runs on B300,
    # where both arms are legal
    t4 = topology("tp", 4)
    g53_h200 = with_kv_dtype(g53, "fp16")
    g53_dense_read = replace(g53_h200, kv_decode_bpt=None, kv_decode_const=0.0)
    _, p_g53, _, _ = decode_curves(g53_h200, t4, wl, [64], n_iter=300)
    _, p_g53d, _, _ = decode_curves(g53_dense_read, t4, wl, [64], n_iter=300)
    assert p_g53[0] > p_g53d[0], "GLM-5.3-Flash DSA must out-speed full-cache reads"
    _, p_g53_b8, _, _ = decode_curves(g53, b2, wl, [64], n_iter=300)
    _, p_g53_b16, _, _ = decode_curves(with_kv_dtype(g53, "fp16"), b2, wl, [64],
                                       n_iter=300)
    assert p_g53_b16[0] < p_g53_b8[0], \
        "BF16 KV (the Hopper-required arm) must decode slower than fp8 KV"
    # decode monotonicity holds for the new models on hardware they fit
    for mdl, topo_fit in ((mm, tp2), (glm, t8), (dsf, tp2), (q38, tp2),
                          (g53_h200, t4), (with_weight_dtype(glm, "nvfp4"), b4)):
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
    # ---- MFU(chunk): anchor, monotonicity, and who pays most (chart E) -----
    for mdl in (m27, m35):
        assert abs(mfu_effective(mdl, tp2, CHUNK_DEFAULT) - MFU_DEFAULT) < 1e-12, \
            "effective MFU at the 32,768 anchor must equal the calibrated default"
    assert (mfu_effective(m27, tp2, 2_048)
            < mfu_effective(m27, tp2, 8_192)
            < mfu_effective(m27, tp2, CHUNK_DEFAULT)), \
        "effective MFU must rise with chunk size (overhead amortises)"
    assert mfu_effective(m35, tp2, 2_048) < mfu_effective(m27, tp2, 2_048), \
        "the MoE must degrade harder at small chunks (full expert-bank stream)"
    assert (miss_context_seconds(m27, tp2, 180_000, 2_048)
            > miss_context_seconds(m27, tp2, 180_000, 16_384)
            > miss_context_seconds(m27, tp2, 180_000, 65_536)), \
        "miss cost must fall as the chunk grows (fewer passes pay overhead)"
    for mdl, tol in ((m27, 0.01), (m35, 0.02)):
        assert abs(miss_context_seconds(mdl, tp2, 180_000, CHUNK_DEFAULT)
                   / prefill_context_seconds(mdl, tp2, 180_000, CHUNK_DEFAULT)
                   - 1) < tol, \
            "at the default chunk the miss cost reproduces the flat-MFU " \
            f"model within {tol:.0%} ({mdl.name})"
    assert abs(cold_request_seconds(m27, tp2, wl, CHUNK_DEFAULT,
                                    per_pass_overhead=True)
               / cold_request_seconds(m27, tp2, wl, CHUNK_DEFAULT)
               - 1) < 0.01, \
        "opt-in overhead pricing agrees with the published tables at 32,768"
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

    # ---- COLD SPIKES (research/spike.md) -----------------------------------
    RATE, TURN, SLA = 2.13, 2_000, 10.0
    # the mixture reproduces section 8's means EXACTLY (same draws, same seed):
    # the spike model must extend the duty model, not quietly re-derive it
    e_s, e_s2, e_cold, e_warm = prefill_service_moments(m27, tp2, wl, CH, TURN)
    assert abs(e_cold / cold_request_seconds(m27, tp2, wl, CH) - 1) < 1e-12, \
        "E[S | miss] must equal cold_request_seconds"
    assert abs(e_warm / warm_request_seconds(m27, tp2, TURN, CH,
                                             prior=mean_context(wl)) - 1) < 1e-9, \
        "E[S | hit] must equal warm_request_seconds at prior = E[L]"
    assert abs(RATE * e_s / prefill_duty(m27, tp2, wl, RATE, CH, TURN) - 1) < 1e-12, \
        "rho must equal the published duty cycle"
    assert e_s2 > e_s ** 2, "second moment sanity"
    # the miss tail dominates the queue: squared CV of ~5.5 (27B) to ~8.3
    # (GLM-5.2) — an M/M/1 would sit at 1
    assert e_s2 / e_s ** 2 - 1 > 5, "service-time variance must be large (L^2 tail)"
    # queueing rises with load and diverges at the duty ceiling
    assert (queue_wait_seconds(m27, tp2, wl, RATE, CH, TURN)
            > queue_wait_seconds(m27, tp2, wl, RATE / 2, CH, TURN) > 0)
    assert queue_wait_seconds(m27, tp2, replace(wl, invalidation=fstar), RATE,
                              CH, TURN) == float("inf"), "no steady state at f*"
    # The convoy effect: under FCFS a HIT waits behind misses, under PS it
    # does not. This is the section's sharpest claim (the miss tax is paid by
    # users who HIT the cache), so it is asserted, not merely printed.
    w10 = replace(wl, invalidation=0.10)
    hit_fcfs = prefill_ttft_seconds(m27, tp2, w10, RATE, CH, TURN,
                                    request="warm", discipline="fcfs")
    hit_ps = prefill_ttft_seconds(m27, tp2, w10, RATE, CH, TURN,
                                  request="warm", discipline="ps")
    assert hit_fcfs > 5 * hit_ps, "FCFS must convoy hits behind misses"
    assert hit_fcfs > e_warm and hit_ps > e_warm, "TTFT includes own service"
    # ...and the bracket flips by class: PS bills each request for its own
    # size, which is dearer for the long jobs and cheaper for the short ones
    assert (prefill_ttft_seconds(m27, tp2, w10, RATE, CH, TURN,
                                 request="cold", discipline="ps")
            > prefill_ttft_seconds(m27, tp2, w10, RATE, CH, TURN,
                                   request="cold", discipline="fcfs")), \
        "PS must be the dearer end for MISSES even as it is cheaper for hits"
    # THE planning claim: the SLA binds before the duty ceiling does
    f_sla = sla_miss_rate(m27, tp2, wl, RATE, CH, SLA, TURN)
    assert 0 < f_sla < fstar, \
        f"SLA-limited miss rate {f_sla:.1%} must sit below f* {fstar:.1%}"
    assert sla_miss_rate(m27, tp2, wl, RATE, CH, SLA, TURN,
                         discipline="ps") < f_sla, \
        "the miss-side SLA must bind sooner under PS"
    # burst tolerance: linear in the SLA, zero at the duty ceiling
    b_dense = spike_tolerance(m27, tp2, wl, SLA, RATE, CH, TURN)
    assert abs(spike_tolerance(m27, tp2, wl, 2 * SLA, RATE, CH, TURN)
               / b_dense - 2) < 1e-9, "B* is linear in the SLA"
    assert spike_tolerance(m27, tp2, replace(wl, invalidation=fstar), SLA,
                           RATE, CH, TURN) < 1e-9, "B* must vanish at f*"
    assert abs(burst_drain_seconds(m27, tp2, wl, b_dense, RATE, CH, TURN)
               - SLA) < 1e-9, "drain(B*) must equal the SLA by construction"
    # ...and the branch's thesis: the MoE's spike tolerance beats the dense
    # 27B by MORE than its prefill-speed ratio, because a cheap miss and a
    # cheap warm turn are the same property seen twice
    b_moe = spike_tolerance(m35, tp2, wl, SLA, RATE, CH, TURN)
    speed_x = (cold_request_seconds(m27, tp2, wl, CH)
               / cold_request_seconds(m35, tp2, wl, CH))
    assert b_moe / b_dense > speed_x > 1, \
        "MoE spike tolerance must beat dense by more than the prefill ratio"
    # the drain is not free for anyone: warm decoders lose real output tokens
    drain, ratio, per_user, total = spike_token_debt(m27, tp2, wl, 32, 64,
                                                    RATE, CH, TURN, n_iter=200)
    assert drain > 0 and ratio > 1 and 0 < per_user < drain * 1e3
    assert abs(total / (64 * per_user) - 1) < 1e-12
    for bad in ("FCFS", "lifo", ""):
        try:
            prefill_ttft_seconds(m27, tp2, wl, RATE, CH, TURN, discipline=bad)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass

    # ---- THE OPERATING POINT: four ceilings in one unit --------------------
    # the reference load is exactly the study's 64 users / 30 s = 2.13 req/s
    assert abs(request_rate(REF_USERS) - RATE) < 0.01, \
        "64 users at one turn per 30 s must reproduce the 2.13 req/s reference"
    op = operating_point(m27, tp2, wl, REF_USERS, CH, TURN, SLA)
    assert set(op["ceilings"]) == {"cache", "decode", "latency", "saturation"}
    assert op["binding"] == min(op["ceilings"], key=op["ceilings"].get)
    # the algebraic heart of section 9: the queue diverges before the server
    # does, so the latency ceiling is ALWAYS strictly inside saturation
    for mdl, tp in ((m27, tp2), (m27, t1), (m35, tp2)):
        lat = max_users_latency(mdl, tp, wl, CH, SLA, TURN)
        sat = max_users_saturation(mdl, tp, wl, CH, TURN)
        assert 0 < lat < sat, \
            f"{mdl.name}: latency ceiling {lat:.0f} must sit inside saturation {sat:.0f}"
        # ...and processor sharing bounds it from the other side (dearer for
        # the long misses, so it admits fewer users than FCFS)
        assert max_users_latency(mdl, tp, wl, CH, SLA, TURN,
                                 discipline="ps") < lat
    # a budget below one miss's own prefill cannot be met at ANY load
    assert max_users_latency(m27, tp2, wl, CH, 0.5, TURN) == 0.0
    # ceilings move the right way: a bigger budget and a longer think time both
    # admit more users; a fatter miss rate admits fewer
    assert (max_users_latency(m27, tp2, wl, CH, 2 * SLA, TURN)
            > max_users_latency(m27, tp2, wl, CH, SLA, TURN))
    assert abs(max_users_latency(m27, tp2, wl, CH, SLA, TURN, think_time_s=60)
               / max_users_latency(m27, tp2, wl, CH, SLA, TURN) - 2) < 1e-9, \
        "think time scales the user ceilings linearly"
    assert (max_users_latency(m27, tp2, replace(wl, invalidation=0.10), CH,
                              SLA, TURN)
            < max_users_latency(m27, tp2, wl, CH, SLA, TURN))
    # ---- think time: measured anchors and the closed-loop variants ---------
    # the anchors must be one consistent measurement: cycle = Z + R, the
    # published 30 s reference sits on the conservative side of the measured
    # 43 s interval, and the decomposition formula sits ABOVE the direct Z
    # (session-final turns censor their human gap, so raw means overshoot)
    assert abs(MEASURED_CYCLE_S
               - (MEASURED_THINK_Z_S + MEASURED_SERVICE_R_S)) < 0.2, \
        "cycle anchor must equal Z + R (they are one measurement, split)"
    assert THINK_TIME_S < MEASURED_CYCLE_S, \
        "the 30 s reference must remain the conservative side of the anchor"
    assert MEASURED_THINK_Z_S < think_z() < 2 * MEASURED_THINK_Z_S, \
        "censoring pushes the formula above the direct Z, but not absurdly so"
    # each main request tows sub_ratio subagent requests: the total arrival
    # rate carries (1 + r), and the user ceilings shrink by the same factor
    assert abs(request_rate(REF_USERS, sub_ratio=wl.sub_ratio)
               - request_rate(REF_USERS) * (1 + wl.sub_ratio)) < 1e-9, \
        "total request rate must be (1 + sub_ratio) x the main-agent rate"
    # ...and the PLACEMENT is pinned two ways: against the published § 9
    # numbers (like the cache/decode pins below — these are the regenerated
    # 2026-08-04 values), and against the closed-form identity, so a
    # multiply-where-divide regression cannot slip past a lucky ordering
    sat_ref = max_users_saturation(m27, tp2, wl, CH, TURN)
    assert abs(max_users_latency(m27, tp2, wl, CH, SLA, TURN) - 273) <= 2, \
        "latency ceiling must reproduce the published 273 (27B/TP2, f=1%)"
    assert abs(sat_ref - 283) <= 2, \
        "saturation ceiling must reproduce the published 283 (27B/TP2, f=1%)"
    e_s_mix = prefill_service_moments(m27, tp2, wl, CH, TURN)[0]
    assert abs(sat_ref * (1 + wl.sub_ratio) * e_s_mix - THINK_TIME_S) < 1e-6, \
        "saturation must be think / ((1 + r) E[S]) exactly"
    # closed loop: at the SAME interval parameter the closed count is larger
    # (response time stretches the cycle), latency stays inside the knee at
    # the reference configuration, and a slower decode admits MORE users
    # (each session hammers the prefill server less often)
    lat_o = max_users_latency(m27, tp2, wl, CH, SLA, TURN)
    lat_c = max_users_latency(m27, tp2, wl, CH, SLA, TURN,
                              closed_z_s=THINK_TIME_S)
    sat_c = max_users_saturation(m27, tp2, wl, CH, TURN,
                                 closed_z_s=THINK_TIME_S)
    assert lat_c > lat_o, "closed conversion must admit more users at equal Z"
    # NOTE: open mode's strict ordering lat < sat is NOT a closed-mode
    # theorem. The knee converts users with the ZERO-LOAD response while the
    # latency count uses the congested cycle (W at lam_sla), so the knee can
    # precede SLA exhaustion — beyond it users buy latency, not throughput,
    # and the planner's argmin reports whichever gives out first.
    assert 0 < sat_c < float("inf") and 0 < lat_c < float("inf")
    assert sat_c > max_users_saturation(m27, tp2, wl, CH, TURN), \
        "at the reference params the closed knee exceeds the open ceiling"
    assert (max_users_latency(m27, tp2, wl, CH, SLA, TURN,
                              closed_z_s=THINK_TIME_S, decode_toks=20.0)
            > lat_c), "a slower decode must lengthen the cycle -> more users"
    op_c = operating_point(m27, tp2, wl, REF_USERS, CH, TURN, SLA, closed=True)
    op_o = operating_point(m27, tp2, wl, REF_USERS, CH, TURN, SLA)
    assert op_c["ceilings"]["cache"] == op_o["ceilings"]["cache"] and \
        op_c["ceilings"]["decode"] == op_o["ceilings"]["decode"], \
        "closed mode must not touch the capacity columns"
    # closed req_rate: the fixed point must invert back to the population
    # (roundtrip), and response feedback must slow the cadence below the
    # no-feedback bound users / (Z + decode)
    lam_main = op_c["req_rate"]
    lam_tot = lam_main * (1 + wl.sub_ratio)
    a_, b_, c_ = (lambda m: (m[1], m[0], m[2]))(
        prefill_service_moments(m27, tp2, wl, CH, TURN))
    resp_ = lam_tot * a_ / (2 * (1 - lam_tot * b_)) + b_
    cyc_ = MEASURED_THINK_Z_S + resp_ + OUT_TOKENS_DEFAULT / DECODE_FLOOR_TOKS
    assert abs(lam_tot * cyc_ / (1 + wl.sub_ratio) - REF_USERS) < 1e-6, \
        "closed req_rate must satisfy users = lam (Z + R(lam)) / (1 + r)"
    assert lam_main < REF_USERS / (MEASURED_THINK_Z_S
                                   + OUT_TOKENS_DEFAULT / DECODE_FLOOR_TOKS), \
        "response feedback must slow the closed cadence below the Z+dec bound"
    for bad in (dict(decode_toks=0.0), dict(out_tokens=-1.0)):
        try:
            max_users_saturation(m27, tp2, wl, CH, TURN,
                                 closed_z_s=THINK_TIME_S, **bad)
            raise AssertionError(f"expected ValueError for {bad}")
        except ValueError:
            pass
    # The planner must REPRODUCE the study's published decision table, not
    # restate it differently: two of its four ceilings are already in § 7's
    # table (warm users p5 = 69 / 177, mns@40 = 118 / 228), and only the
    # latency and saturation columns are new. If these drift, the planner has
    # silently forked from the numbers the rest of the study plans on.
    assert abs(max_users_cache(m27, t1, wl, n_iter=800) - 69) <= 3, \
        "cache ceiling must reproduce the published warm-users p5 of 69"
    assert abs(max_users_cache(m27, tp2, wl, n_iter=800) - 177) <= 6, \
        "cache ceiling must reproduce the published warm-users p5 of 177"
    # PUBLISHED-TABLE CONVENTION. These two pin figures that appear in
    # docs/scenarios.md, and those tables were generated before decode had any
    # efficiency term at all (MBU_DEFAULT, 2026-08-28). They are pinned at
    # mbu=1.0 so the published numbers stay reproducible from this file; the
    # DEFAULT model is now the calibrated one and disagrees with them by ~2.6x.
    # Regenerating the tables is an owner decision, not a mechanical one --
    # until it happens, every decode figure in the docs is roofline-convention.
    # ...and at the pre-calibration mtp too: the 27B's speedup moved 1.7 -> 2.94
    # on the same measurement, so BOTH knobs must be rewound to reproduce a
    # published figure. That they travel together is the point, not an
    # inconvenience -- see MBU_DEFAULT.
    m27_pub = dataclasses.replace(m27, mtp=1.7)
    assert abs(max_users_decode(m27_pub, t1, wl, n_iter=800, mbu=1.0) - 118) <= 4, \
        "decode ceiling must reproduce the published mns@40 of 118"
    assert abs(max_users_decode(m27_pub, tp2, wl, n_iter=800, mbu=1.0) - 228) <= 8, \
        "decode ceiling must reproduce the published mns@40 of 228"
    # ---- H7, and what the decode measurement did and did not settle --------
    # H7 is the study's thesis: the CACHE binds before decode bandwidth at the
    # reference workload. It held for every configuration under the roofline
    # convention, and the first assertion still guards that published claim.
    #
    # What the measurement (research/decode_mbu.md, 2026-08-28) DID identify
    # is per-user decode speed at the batch sizes it ran: 250-330 tok/s on the
    # 27B / 4xH200 TP4 across n = 1-4, reproduced in four sessions. That is
    # the pin: the model must land inside the measured band there (it reads
    # 314 / 302 / 276 at n = 1 / 2 / 4). What it did NOT identify is the
    # decode CEILING, and so not the binding order either: the constant is
    # fitted where weights dominate the step (n = 1-25) and applied where KV
    # does (n ~ 100-250), and the note's two candidate mechanisms put the
    # 4xH200 ceiling at 238 (A, H7 stands) or 154 (B, H7 inverts) against a
    # cache ceiling of 249, with the single-constant fold used here at ~150.
    # On the 1xH200 row these assertions run on, the fold reads decode ~32
    # against cache ~70 (published: 118 against 69) -- but that ordering is a
    # projection of the fold, not a measurement, so it is deliberately NOT
    # asserted in either direction. A spec-off / k-sweep A/B decides it. Do
    # not add that assertion back without the measurement behind it.
    cache = max_users_cache(m27, t1, wl, n_iter=200)
    dec_pub = max_users_decode(m27_pub, t1, wl, n_iter=200, mbu=1.0)
    assert dec_pub > cache, \
        "H7 as published: under the roofline convention the cache binds first"
    tp4 = topology("tp", 4, "H200")
    _, p50_tp4, _, _ = decode_curves(m27, tp4, wl, [1, 2, 4], n_iter=400)
    assert all(250 <= v <= 330 for v in p50_tp4), \
        f"27B/TP4 per-user decode at n=1,2,4 must sit in the measured 250-330 " \
        f"tok/s band, got {[round(float(v)) for v in p50_tp4]}"
    # wiring guard only (same mtp on both sides): the efficiency term must
    # actually reach the ceiling search
    dec = max_users_decode(m27, t1, wl, n_iter=200)
    assert dec < max_users_decode(m27, t1, wl, n_iter=200, mbu=1.0), \
        "MBU_DEFAULT must lower the decode ceiling against the same-mtp roofline"
    assert max_users_decode(m27, t1, wl, floor=20.0, n_iter=200) > dec, \
        "a lower floor must admit more concurrent decoders"
    # an unreachable floor is a censored result, not a silent cap
    try:
        max_users_decode(m27, t1, wl, floor=1e-6, n_iter=200)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    # WHICH constraint binds must change with the miss rate — the planner's
    # whole reason to exist (cache at f=1%, latency once misses get common)
    b_lo = operating_point(m27, tp2, wl, REF_USERS, CH, TURN, SLA,
                           n_iter=200)["binding"]
    b_hi = operating_point(m27, tp2, replace(wl, invalidation=0.25),
                           REF_USERS, CH, TURN, SLA, n_iter=200)["binding"]
    assert b_lo != b_hi, \
        f"the binding constraint must switch with f (got {b_lo} at both ends)"
    assert b_hi == "latency", f"latency must bind at f=25%, got {b_hi}"
    for bad_users in (-1,):
        try:
            operating_point(m27, tp2, wl, bad_users, n_iter=50)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass

    # ---- THE STEADY-STATE DECODE POINT -------------------------------------
    # The counterpart to every stress figure above: what the load ACTUALLY
    # produces. Open-loop arrivals mean the decode batch holds a handful of
    # sequences, not the warm population, and each runs multiples faster.
    rate_g = request_rate(REF_USERS, THINK_TIME_S, wl.sub_ratio)
    sdp = steady_decode_point(m27, tp2, wl, rate_g, n_iter=400)
    # 1. THE identity — n x per-user speed = arrival rate x output tokens.
    #    Everything the explorer's act-2 tiles print is a reading of this line.
    assert abs(sdp["n"] * sdp["per_user_tok_s"] - rate_g * OUT_TOKENS_DEFAULT) \
        < 1e-6 * rate_g * OUT_TOKENS_DEFAULT, \
        "steady point must balance delivered against demanded output tok/s"
    # 2. ...which is Little's law read the other way round
    assert abs(sdp["n"] - rate_g * (OUT_TOKENS_DEFAULT
                                    / sdp["per_user_tok_s"])) < 1e-9, \
        "steady n must equal arrival rate x seconds spent decoding"
    # 3. the headline, PINNED: the 27B on TP2 at the published reference load
    #    decodes ~2 sequences at a time at ~430 tok/s, against the 228-decoder
    #    / 40 tok/s ceiling the planner's decode column reports. The gap IS the
    #    finding; if it closes, the load conversion has broken somewhere.
    #    (Re-pinned 2026-08-27 when OUT_TOKENS_DEFAULT moved 1,000 -> 400,
    #    the measured value: fewer decode-seconds per request means a smaller
    #    steady batch running faster. Old pin: ~6.4 at ~370 tok/s.
    #    Re-pinned again 2026-08-28 for MBU_DEFAULT + the 27B's measured mtp:
    #    slower decode keeps each request in the batch longer, so the steady
    #    batch GROWS and each member runs slower. 2.2 at 430 -> 6.8 at 138.
    #    The band is tight: cross-seed spread is ~0.1 in n and ~2 in tok/s,
    #    so an earlier +/-1.0 / +/-25 pin -- justified by hand-waving about a
    #    steeper curve -- would have let real drift through. Measured, not
    #    reasoned.)
    assert abs(sdp["n"] - 6.8) <= 0.4, \
        f"27B/TP2 reference load: ~6.8 decoders expected, got {sdp['n']:.2f}"
    assert abs(sdp["per_user_tok_s"] - 138) <= 10, \
        f"27B/TP2 reference load: ~138 tok/s expected, " \
        f"got {sdp['per_user_tok_s']:.0f}"
    #    The stress-vs-steady gap SURVIVES calibration but narrows sharply:
    #    the ceiling fell 228 -> 74 while the steady batch rose 2.2 -> 6.8, so
    #    the ratio went ~100x -> ~11x. Still a real distinction, no longer a
    #    spectacular one -- and the reason the decode ceiling now binds.
    assert sdp["n"] < max_users_decode(m27, tp2, wl, n_iter=200) / 5, \
        "the steady batch must still sit inside the decode ceiling at this load"
    # 4. exactly linear in the PRODUCT rate x output tokens, and in nothing else
    assert (steady_decode_point(m27, tp2, wl, 2 * rate_g, n_iter=200)["n"]
            > sdp["n"]), "more arrivals must mean more concurrent decoders"
    a = steady_decode_point(m27, tp2, wl, 2 * rate_g, n_iter=200)
    b = steady_decode_point(m27, tp2, wl, rate_g,
                            out_tokens=2 * OUT_TOKENS_DEFAULT, n_iter=200)
    assert abs(a["n"] - b["n"]) < 1e-9 and abs(a["demand_tok_s"]
                                               - b["demand_tok_s"]) < 1e-9, \
        "only the product rate x output tokens can move the steady point"
    # 5. no load, no decoders — and no division by zero on the way there
    z = steady_decode_point(m27, tp2, wl, 0.0, n_iter=100)
    assert z["n"] == 0.0 and z["delivered_tok_s"] == 0.0 \
        and z["demanded_tok_s"] == 0.0 and z["per_user_tok_s"] > 0
    # 6. a demand past what `hi` decoders could retire is CENSORED, not clamped
    hot = steady_decode_point(m27, tp2, wl, 1e6, n_iter=100, hi=64)
    assert hot["saturated"] and hot["n"] == 64.0, \
        "demand beyond the decode curve must flag saturated, not invent a batch"
    # 7. per-group vs system: n is what ONE cache decodes, agg is the estate
    dpg = topology_grid(dp=2, tp=1)
    sd_dp = steady_decode_point(m27, dpg, wl, rate_g, n_iter=200)
    assert abs(sd_dp["demanded_tok_s"] - 2 * rate_g * OUT_TOKENS_DEFAULT) < 1e-6, \
        "the aggregate, and only the aggregate, carries the replica count"
    for bad in (dict(rate_group=-1.0), dict(out_tokens=-1.0), dict(hi=0)):
        try:
            steady_decode_point(m27, tp2, wl,
                                **{**dict(rate_group=rate_g, n_iter=50), **bad})
            raise AssertionError(f"expected ValueError for {bad}")
        except ValueError:
            pass

    # ---- THE ELECTRICITY BILL (research/power.md) --------------------------
    # constants are the note's exact fractions of the spec plate — decode 0.55,
    # prefill 0.90 — on BOTH parts (the B300 rows transfer the fraction, they
    # are entirely extrapolated), and the states order idle < decode < prefill
    for g_ in GPUS.values():
        assert abs(g_.p_decode_w - 0.55 * g_.tdp_w) < 1e-9, \
            f"{g_.name}: p_decode_w must be the note's 0.55 x TDP central"
        assert abs(g_.p_prefill_w - 0.90 * g_.tdp_w) < 1e-9, \
            f"{g_.name}: p_prefill_w must be the note's 0.90 x TDP central"
        assert 0 < g_.idle_w < g_.p_decode_w < g_.p_prefill_w <= g_.tdp_w
        assert g_.host_w > 0
        # power.md's honesty cross-check: a 50/50 prefill/decode split lands
        # at 0.725 x TDP, and the split's band [0.5(0.80+0.45), 0.5(1.00+0.75)]
        # must CONTAIN the measured 0.85-of-TDP saturated-node anchor (NLR) —
        # mixed continuous batching runs near, not at, the cap, and the model's
        # phase constants reproduce that without tuning
        mix = 0.5 * (g_.p_prefill_w + g_.p_decode_w) / g_.tdp_w
        assert abs(mix - 0.725) < 1e-9, f"{g_.name}: 50/50 mix must be 0.725 x TDP"
        assert 0.5 * (0.80 + 0.45) < 0.85 < 0.5 * (1.00 + 0.75), \
            "the phase bands must keep the measured 0.85 saturation anchor plausible"
    # duty split: bounded, partitioned, and the draw pinned between the idle
    # floor and the prefill plateau at EVERY load (a convex mix of the three
    # states can never leave [idle_w, p_prefill_w])
    du = max_users_decode(m27, tp2, wl, n_iter=200)
    rate_ref = request_rate(REF_USERS, sub_ratio=wl.sub_ratio) / tp2.replicas
    for r_ in (0.0, 0.1, rate_ref, 5 * rate_ref, 1e4):
        pd_ = power_draw(m27, tp2, wl, r_, du, turn_tokens=TURN)
        assert 0.0 <= pd_["d_p"] <= 1.0 and 0.0 <= pd_["d_d"] <= 1.0
        assert pd_["d_p"] + pd_["d_d"] <= 1.0 + 1e-12, "d_p, d_d must partition time"
        assert (GPUS["H200"].idle_w - 1e-9 <= pd_["per_gpu_w"]
                <= GPUS["H200"].p_prefill_w + 1e-9), \
            "per-GPU draw must sit between the idle floor and the prefill plateau"
    # zero rate is EXACTLY the warm-idle floor...
    p0 = power_draw(m27, tp2, wl, 0.0, du, turn_tokens=TURN)
    assert p0["d_p"] == 0.0 and p0["d_d"] == 0.0
    assert p0["per_gpu_w"] == GPUS["H200"].idle_w, \
        "an unloaded GPU must draw exactly idle_w"
    # ...and saturation (d_p -> 1) exactly the prefill plateau: the cap binds
    assert power_draw(m27, tp2, wl, 1e4, du, turn_tokens=TURN)["d_p"] == 1.0
    assert (power_draw(m27, tp2, wl, 1e4, du, turn_tokens=TURN)["per_gpu_w"]
            == GPUS["H200"].p_prefill_w), \
        "at saturation the draw must be the prefill plateau, not TDP"
    # a non-positive decode capacity falls back to d_d = 1 - d_p (the
    # explorer's cap > 0 guard, mirrored)
    pz = power_draw(m27, tp2, wl, 0.1, 0.0, turn_tokens=TURN)
    assert abs(pz["d_d"] - (1.0 - pz["d_p"])) < 1e-12, \
        "cap <= 0 must give d_d = 1 - d_p, matching the explorer's guard"
    # the bill is LINEAR in the tariff and in PUE — both are user-chosen
    # multipliers, not model error, and must behave like it
    cost_ref = energy_cost(m27, tp2, wl, rate_ref, du, REF_USERS,
                           turn_tokens=TURN)
    cost_2e = energy_cost(m27, tp2, wl, rate_ref, du, REF_USERS,
                          turn_tokens=TURN, eur_kwh=2 * EUR_PER_KWH_DEFAULT)
    cost_2p = energy_cost(m27, tp2, wl, rate_ref, du, REF_USERS,
                          turn_tokens=TURN, pue=2 * PUE_DEFAULT)
    assert abs(cost_2e["eur_month"] / cost_ref["eur_month"] - 2) < 1e-12, \
        "eur_month must be linear in eur_kwh"
    assert abs(cost_2p["eur_month"] / cost_ref["eur_month"] - 2) < 1e-12 \
        and abs(cost_2p["kw"] / cost_ref["kw"] - 2) < 1e-12, \
        "eur_month (via kw) must be linear in pue"
    # the hardware line is the part's list price x GPUs x hours, untouched by
    # the tariff and PUE; the per-user figure divides the TOTAL
    assert abs(cost_ref["hw_month"]
               - tp2.n_gpu * GPUS["H200"].eur_gpu_h * HOURS_PER_MONTH) < 1e-6
    assert cost_2e["hw_month"] == cost_ref["hw_month"] \
        and cost_2p["hw_month"] == cost_ref["hw_month"]
    assert cost_ref["total_month"] == cost_ref["eur_month"] + cost_ref["hw_month"]
    # every part must carry a price: the dataclass default is 0.0 so that an
    # unpriced GPU cannot pass as free by accident
    for _gk, _g in GPUS.items():
        assert _g.eur_gpu_h > 0, f"GPUS[{_gk!r}] has no eur_gpu_h"

    assert cost_ref["eur_user"] == cost_ref["total_month"] / REF_USERS
    assert energy_cost(m27, tp2, wl, rate_ref, du, REF_USERS, turn_tokens=TURN,
                       eur_gpu_h=0.0)["total_month"] == cost_ref["eur_month"], \
        "eur_gpu_h = 0 must recover the electricity-only bill"
    assert 0 < cost_ref["eur_mtok"] < float("inf")
    # zero output rate: €/1M tokens is undefined (inf), never a silent zero
    assert energy_cost(m27, tp2, wl, 0.0, du, REF_USERS,
                       turn_tokens=TURN)["eur_mtok"] == float("inf")
    # guards
    for bad_kw in (dict(rate_group=-1.0), dict(decode_users_group=-1.0),
                   dict(pue=0.0)):
        try:
            power_draw(m27, tp2, wl, **{**dict(rate_group=rate_ref,
                                               decode_users_group=du),
                                        **bad_kw})
            raise AssertionError(f"expected ValueError for {bad_kw}")
        except ValueError:
            pass
    try:
        energy_cost(m27, tp2, wl, rate_ref, du, REF_USERS, eur_kwh=-0.01)
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
    print(f"  power 27B TP2 @64u   = {cost_ref['kw']:6.2f} kW "
          f"({cost_ref['per_gpu_w']:.0f} W/GPU; dP {cost_ref['d_p']:.0%} "
          f"dD {cost_ref['d_d']:.0%})   EUR {cost_ref['eur_month']:.0f}/mo "
          f"@ {EUR_PER_KWH_DEFAULT} EUR/kWh, PUE {PUE_DEFAULT}")
    for dtype in KV_DTYPES:
        for mk in MODELS:
            if dtype == "fp16" and not MODELS[mk].kv_fp16_ok:
                continue   # GLM-5.2 (DSA) / DSv4-Flash (CSA): FP16 KV not servable
            if dtype == "fp8" and MODELS[mk].kv_fp8_blackwell_only:
                continue   # GLM-5.3-Flash: fp8 KV is Blackwell-only and these
                           # legacy topologies are all H200 — its H200 arm is
                           # the fp16 row below
            for tk in TOPOLOGIES:
                p = kv_pool_tokens(with_kv_dtype(MODELS[mk], dtype), TOPOLOGIES[tk])
                print(f"  pool {mk:7} {tk:12} {dtype:5} = {p / 1e6:6.2f} M tokens")
    print("  -- B300, fp8 KV, weight dtype fp8 | nvfp4 --")
    for mk in MODELS:
        for n in (1, 4):
            t = topology("tp", n, "B300")
            pools = []
            for wd in WEIGHT_DTYPES:
                if wd == "nvfp4" and MODELS[mk].nvfp4_w is None:
                    pools.append("   n/a")   # DSv4-Flash: no NVFP4 variant exists
                    continue
                mdl = with_weight_dtype(MODELS[mk], wd)
                pools.append(f"{kv_pool_tokens(mdl, t) / 1e6:6.2f}")
            print(f"  pool {mk:7} {t.name:19} = {' | '.join(pools)} M tokens")


if __name__ == "__main__":
    _selfcheck()
