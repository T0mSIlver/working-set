"""
Generalized capacity + decode-speed model for the GPU-scaling scenarios.

This is the single source of truth for the extended study (the `workingset`
package; `scripts/scenario_model.py` is a compatibility alias of this module). It keeps the SAME
methodology as the baseline scripts (warm_whisker.py / real_mns.py) and
generalizes it along the axes we now want to explore:

  * model         : 27B (dense sibling), 35B-A3B (MoE, ~3B active),
                    Mistral-Medium-3.5-128B (dense GQA), GLM-5.3 (744B-A40B
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
import math
import numpy as np
import tomllib
from dataclasses import dataclass, field, fields, replace
from importlib import resources

KIB, MIB, GIB = 1024, 1024**2, 1024**3


def load_catalog() -> dict:
    """The model and GPU catalogue (data/catalog.toml), parsed."""
    with resources.files("workingset").joinpath("data/catalog.toml").open("rb") as f:
        return tomllib.load(f)


CATALOG = load_catalog()
# catalog keys that are not Model fields (tables only the explorer reads)
CATALOG_EXTRA_KEYS = frozenset({"explorer", "quality"})

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
    #   central (band 0.80-1.00) — FLAT across the study's MFU 30-55% bracket,
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
    # it. A rental rate is all-in: power, cooling and the host are in the rent,
    # so a rented GPU carries no electricity line (energy_cost, pay="rent").
    eur_gpu_h: float = 0.0
    # ---- PURCHASE price, EUR per GPU -- used ONLY by energy_cost(pay="own") ----
    # A complete 8-GPU server's price over 8 (the host, CPUs, memory and NICs
    # included), at the ECB rate of the day it was read. NVIDIA publishes none;
    # research/gpu_costs in catalog.toml's comments has the sources.
    eur_capex_gpu: float = 0.0


GPUS = {k: GPU(**row) for k, row in CATALOG["gpus"].items()}

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
# spec_tokens/acceptance: draft depth k and per-draft acceptance; the
#                   speedup is tokens_per_step(acceptance, k), never a constant
#
# Dense model  => w_decode_shared = w_resident, w_route_* = 0 (reads all weights).
# MoE model    => decode reads w_decode_shared + the routed-expert union over
# P = n x (1 + k) positions (every verify position routes, accepted or not):
#   union="coverage" : total x (1-(1-k_e/E)^P)   -- expectation under
#                      independent routing; the DEFAULT. One sequence's
#                      positions route with correlation, so the real union
#                      sits below it
#   union="linear"   : min(P x per-token, total) -- no-overlap upper bound
# A spec-on/spec-off pair at one n measures where reality sits.
#
# All 35B-A3B constants come from the PUBLISHED Qwen/Qwen3.6-35B-A3B-FP8
# config.json (40 layers = 30 DeltaNet + 10 full-attn, 2 KV heads x 256,
# 256 experts / 8 routed + 1 shared, vocab 248,320). Full derivations and
# the assumption ledger: research/model_35ba3b.md.
# ============================================================================
# ---- DECODE: one step-time equation ----------------------------------------
# A decode step of one replica group (tp GPUs) holding n sequences at draft
# depth k (the model's spec_tokens) costs
#
#   t_step = B / (eta x tp x hbm_bw)                      streamed bytes
#          + n (1 + k) x c_tok                             verify GEMMs
#          + (tau_0 + [tp > 1] tau_ar) x (L + p L_d)       per-layer fixed cost
#          + tau_rec x L_rec x (1 + k)                     serial recurrent updates
#
#   B = w_decode_shared + routed-expert union over P = n (1 + k) positions
#     + p x draft-pass bytes (draft layers, the lm_head they re-read, their
#       routed experts over n positions); p = Model.passes(k), the draft
#       passes per step: k, or 1 for a block drafter that writes all k at once
#     + KV reads + recurrent/window state traffic (both independent of k: the
#       1 + k positions of a sequence share one KV read and one state pass)
#   c_tok = 2 x params_prefill / (tp x peak_flops_fp8 x MFU)
#
# and yields tokens_per_step(acceptance, k) = 1 + a + ... + a^k tokens per
# sequence. Spec: research/decode-model.md; decisions on the priors below:
# research/decode-model.md "Priors".
#
# The constants are kernel and interconnect properties, shared by every model:
#   DECODE_ETA   achieved fraction of RAW HBM bandwidth (no tp_efficiency
#                haircut: the all-reduce cost is tau_ar). 0.29 = the 0.36 the
#                held-batch sweep read in the old model convention x 0.81, the
#                TP4 haircut it was divided by.
#   TAU_0        per-layer launch cost, 5 us: a prior.
#   TAU_AR       per-layer cost of tp > 1: two one-shot NVLink all-reduces at
#                ~5 us each (research/gpu_catalogue.md); a prior until the link
#                model (WS-7) derives it per part.
#   tau_rec      SOLVED so the anchor (27B, 4xH200 TP4, k = 3) reproduces the
#                held-batch sweep's measured 2.7 ms intercept once its draft
#                passes are priced as bytes and the two priors above are
#                spent: what the intercept leaves, per recurrent update.
# `ws calibrate` fits all four from a decode sweep (research/calibrate.md).
DECODE_ETA = 0.29
TAU_0 = 5e-6
TAU_AR = 10e-6
_ANCHOR_FIXED_S = 2.7e-3                # measured intercept: 27B, 4xH200 TP4, k = 3
_ANCHOR_SPEC_TOKENS = 3


@dataclass
class Model:
    name: str
    kv_bpt: float
    deltanet_state: float
    w_resident: float
    w_decode_shared: float
    w_route_pertok: float
    w_route_total: float
    # Draft tokens per step the model's serving recipe runs
    # (num_speculative_tokens; 0 = no draft model ships with it) and how they
    # are drafted: "mtp" (a built-in MTP module), "dspark" (DeepSeek's DSpark
    # stages) or "eagle" (no module; an external EAGLE-style draft).
    spec_tokens: int = 0
    draft_method: str = "mtp"
    # Per-draft acceptance (accept until the first rejection): a scalar, or
    # one figure per draft position. 0.971 is Qwen3.6-27B's measured value at
    # k = 2 (geometric to three decimals); every other model carries it until
    # `ws decode-sweep` measures its own (acceptance_measured False).
    acceptance: float | tuple = 0.971
    acceptance_measured: bool = False
    # Layer counts the step equation's fixed terms read: target layers run per
    # step, recurrent (DeltaNet / KDA) layers among them, and draft-model
    # layers run per draft pass (MTP head 1, DSpark drafter 3).
    layers: int = 0
    rec_layers: int = 0
    draft_layers: int = 0
    # Draft passes per decode step. None: one per draft token (an MTP head or
    # an autoregressive drafter runs k times). A block drafter writes all k
    # draft tokens in one forward pass and sets 1 (Kimi-K3.1-Pro's DSpark
    # draft: "7 tokens per forward pass").
    draft_passes: int | None = None
    # Bytes ONE draft pass streams beyond w_decode_shared: the draft layers'
    # always-active weights plus the lm_head they re-read for the draft
    # logits (w_draft), and their routed experts per position
    # (w_draft_route_pertok; union over the batch with the target's k/E).
    w_draft: float = 0.0
    w_draft_route_pertok: float = 0.0
    # True when w_resident is already the measured AS-DEPLOYED footprint, so
    # the +15% weight-overhead allowance (calibrated on exactly that figure)
    # must not be added on top.
    weights_as_deployed: bool = False
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
    # False for models vLLM can only serve with a quantized KV cache (the
    # DeepSeek Flash models' fp8/fp4 caches; Mistral-Large-4, documented on
    # fp8_ds_mla only) — with_kv_dtype("fp16") then raises.
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
    # (kv_bpt, kv_decode_const) of the FP8 arm on a HOPPER part, for a model
    # whose FP8 cache row depends on the GPU; None = one row on every part.
    # GLM-5.3: Blackwell serves DSA through FlashInfer's sparse MLA with plain
    # fp8_e4m3 rows (576 B per latent, the base constants), Hopper through
    # FlashMLA's fp8_ds_mla rows (656 B: 512 fp8 + 16 B of fp32 scales + 64
    # bf16 rope values). with_gpu() swaps them in and stamps kv_gpu;
    # check_dtype_supported REFUSES a Hopper topology without the swap.
    # research/model_glm52.md #2.
    kv_hopper_fp8: tuple | None = None
    # The GPU part with_gpu() resolved the FP8 row for (None = not resolved;
    # only meaningful on models with kv_hopper_fp8).
    kv_gpu: str | None = None
    # False when deltanet_state is NOT a bf16 recurrent state the fp32 toggle
    # can meaningfully double (the DeepSeek Flash models reuse the field for
    # their fixed per-session window + fp32 compressor buffers, already
    # mixed-precision).
    # Python charges deltanet_state as-is either way; the flag exists for
    # mirror parity with the explorer, which gates its fp32-state control on it.
    state_fp32_ok: bool = True
    # Bytes the recurrent / fixed per-session state moves PER ACTIVE SEQUENCE
    # PER DECODE STEP. None (the DeltaNet models) means read + write of the
    # whole state, 2 x deltanet_state. A model whose deltanet_state is fixed
    # STORAGE that decode does not re-stream — DSv4.1-Flash's windows are
    # already read inside kv_decode_const, and a step writes one 528-B ring
    # slot per layer — sets the true per-step figure here instead
    # (codex review of the V4.1 swap, 2026-09-10: 2 x 2.9 MB was charged on
    # top of the windows' own read; V4-Flash-0731 shares the layout and the
    # correction).
    state_step_bytes: float | None = None
    # KV heads the attention cache can be SHARDED across under tensor
    # parallelism. A GQA model splits its KV heads over the TP ranks; once tp
    # exceeds them (or always, for the single-latent MQA / MLA caches: 1) the
    # cache is REPLICATED on every rank instead — vLLM's plain --tensor-
    # parallel-size keeps a full copy per rank, and only decode context
    # parallelism (--decode-context-parallel-size, sharding along the
    # sequence) or attention-DP + expert-EP removes the copies.
    # kv_replication() turns this into the factor the pool and the per-step
    # KV reads are multiplied by. research/kv_tp_sharding.md.
    kv_heads: int = 1
    # Same for the per-session recurrent / fixed state: None = it shards
    # across the TP ranks at every tp the study prices (DeltaNet / KDA value
    # heads, 32-64 of them); the DeepSeek Flash models' windows are the single
    # latent and replicate with the cache (1).
    state_heads: int | None = None
    # Largest max_seq_len the study allows for this model (tokens). The
    # workload cap (Workload.cap) may not exceed it — warm_capacity and
    # decode_curves raise otherwise. Owner decision 2026-07: 1,048,576 for
    # the Qwens (262,144 native, 1M via YaRN rope scaling) and GLM-5.3
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

    @property
    def tokens_per_step(self) -> float:
        """Tokens one sequence advances per decode step: m(acceptance, k)."""
        return tokens_per_step(self.acceptance, self.spec_tokens)

    @property
    def state_traffic(self) -> float:
        """Recurrent-state bytes moved per active sequence per decode step."""
        return (2.0 * self.deltanet_state if self.state_step_bytes is None
                else self.state_step_bytes)

    def _union(self, positions: float, per_tok: float, union: str) -> float:
        """Routed-expert bytes read by `positions` routed tokens."""
        if per_tok <= 0 or not self.is_moe:
            return 0.0
        total = per_tok * self.w_route_total / self.w_route_pertok   # E/k_e x per token
        if union == "coverage":
            # expectation under independent routing: overstates the union a
            # little, since one sequence's positions route with correlation
            return total * (1.0 - (1.0 - per_tok / total) ** positions)
        if union == "linear":
            return min(positions * per_tok, total)            # no-overlap bound
        raise ValueError(f"union must be 'coverage' or 'linear', got {union!r}")

    def w_decode(self, positions: float, union: str = "coverage") -> float:
        """Target-model weight bytes one decode step reads when `positions`
        tokens route through it (n sequences x (1 + k) verify positions)."""
        return self.w_decode_shared + self._union(positions, self.w_route_pertok, union)

    def passes(self, k: int) -> int:
        """Draft passes one step runs at draft depth k."""
        if k <= 0:
            return 0
        return k if self.draft_passes is None else min(self.draft_passes, k)

    def draft_pass_bytes(self, n: float, union: str = "coverage") -> float:
        """Bytes one draft pass streams for n sequences: the draft layers, the
        lm_head they re-read, and their routed experts over n positions."""
        return self.w_draft + self._union(n, self.w_draft_route_pertok, union)


# The constants themselves live in data/catalog.toml, shared with the
# explorer (scripts/catalog.py writes interactive/src/catalog.js from it).
_CATALOG_TUPLES = ("nvfp4_w", "kv_hopper_fp8")


def _model_from_catalog(key: str, row: dict) -> Model:
    known = {f.name for f in fields(Model)}
    unknown = set(row) - known - CATALOG_EXTRA_KEYS
    if unknown:
        raise ValueError(f"catalog model {key!r}: unknown fields {sorted(unknown)}")
    return Model(**{k: (tuple(v) if k in _CATALOG_TUPLES else v)
                    for k, v in row.items() if k in known})


MODELS = {k: _model_from_catalog(k, row) for k, row in CATALOG["models"].items()}

# ---- Draft speedup: tokens per step ------------------------------------------
# With per-position acceptance a_1..a_k (a_j = P(draft j accepted | drafts
# 1..j-1 were)) a sequence advances m = 1 + sum_j prod_{i<=j} a_i tokens per
# step; one scalar a gives the geometric (1 - a^(k+1)) / (1 - a). This is the
# only draft speedup in the model: per-user tok/s = m / t_step.
def tokens_per_step(acceptance, k: int) -> float:
    """Expected tokens one sequence advances per decode step at draft depth k."""
    if k < 0:
        raise ValueError(f"draft depth must be >= 0, got {k!r}")
    per_pos = (list(acceptance) if isinstance(acceptance, (list, tuple))
               else [acceptance] * k)
    if k > len(per_pos):
        per_pos += [per_pos[-1] if per_pos else 0.0] * (k - len(per_pos))
    m, run = 1.0, 1.0
    for a in per_pos[:k]:
        if not 0.0 <= a <= 1.0:
            raise ValueError(f"acceptance must be in [0, 1], got {a!r}")
        run *= a
        m += run
    return m


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
    if model.kv_gpu is not None:
        raise ValueError(f"{model.name}: with_kv_dtype must run before "
                         "with_gpu (it would double the GPU's FP8 row)")
    if kv_dtype == "fp8":
        return model
    if not model.kv_fp16_ok:
        raise ValueError(
            f"{model.name}: FP16 KV is not servable (vLLM requires a quantized "
            "KV cache on this model's sparse-attention path — Mistral-Large-4's "
            "DSA, DSv4-Flash's CSA, DSv4.1-Flash's CSA2; see the model's research "
            "note)")
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


# ---- GPU-coupled FP8 cache row -------------------------------------------------
def with_gpu(model: Model, gpu: str) -> Model:
    """Return `model` with the FP8 cache row `gpu` serves it with.

    Identity for every model whose row is the same on both parts. A model
    with kv_hopper_fp8 (GLM-5.3) takes those constants on a Hopper part when
    it prices the FP8 arm; every arm is stamped with the part, so
    check_dtype_supported can refuse a model resolved for the other part.
    Apply it last, after with_weight_dtype and with_kv_dtype."""
    if gpu not in GPUS:
        raise ValueError(f"gpu must be one of {tuple(GPUS)}, got {gpu!r}")
    if model.kv_hopper_fp8 is None:
        return model
    if model.kv_gpu is not None:
        raise ValueError(f"{model.name}: already resolved for {model.kv_gpu}")
    if model.kv_dtype == "fp8" and not GPUS[gpu].supports_nvfp4:
        kv_bpt, kv_decode_const = model.kv_hopper_fp8
        return replace(model, kv_bpt=kv_bpt, kv_decode_const=kv_decode_const,
                       kv_gpu=gpu)
    return replace(model, kv_gpu=gpu)


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
    # GPU-coupled FP8 row (GLM-5.3): the base constants are the Blackwell
    # row, so a Hopper topology needs the with_gpu() swap and a swapped model
    # may only price Hopper — otherwise the pool and the decode reads would
    # price the other part's row. The FP16 row is the same on both parts.
    if model.kv_hopper_fp8 is not None and model.kv_dtype == "fp8":
        hopper_row = (model.kv_gpu is not None
                      and not GPUS[model.kv_gpu].supports_nvfp4)
        if hopper_row != (not topo.gpu.supports_nvfp4):
            raise ValueError(
                f"{model.name}: its FP8 cache row depends on the GPU — price "
                f"with_gpu(model, \"{topo.gpu.name}\") on this topology "
                "(research/model_glm52.md #2)")


# ============================================================================
# TOPOLOGY  — a DP x TP grid of GPUs
# ----------------------------------------------------------------------------
# The unit DP replicates is a GROUP of `tp` GPUs, not a GPU. That distinction is
# invisible while a model fits in one GPU (the 27B and 35B-A3B on either part)
# and load-bearing wherever one does not:
#
#     min TP to fit FP8 weights      H200    B300
#       Mistral-Medium-3.5-128B         2       1     <- fits a single B300
#       GLM-5.3 (744B-A40B)             7       3
#
# In every cell above where min TP > 1, `topology("dp", n)` — n INDEPENDENT
# SINGLE GPUs — is not a deployment that exists: no count of single GPUs ever
# holds the weights, so the whole DP axis reported a 0 pool. (MM35 on B300 is
# the exception: min TP 1, so pure DP is real there.) Data parallelism in the
# other cells means replicating whole TP groups: on one 8-GPU node GLM-5.3 runs
# DP2xTP4 on B300, and MM35 runs DP4xTP2 on H200. Total GPUs = dp * tp.
#
#   weights  : sharded across a group's tp GPUs -> stored ONCE per group
#   reserve  : activation/workspace scratch, charged PER GPU
#   cache    : one KV pool per group; dp groups means dp independent pools
#   bandwidth: a group has tp GPUs' worth, less tp_efficiency(tp)
# ============================================================================
KV_SHARDS = ("dcp", "replicate")


@dataclass
class Topology:
    name: str
    dp: int                          # replica groups; each owns one KV cache
    tp: int                          # GPUs per group (weights sharded across these)
    gpu: GPU = GPUS["H200"]
    # How the KV cache is laid out across the tp ranks of a group.
    #   "dcp"       decode context parallelism: ranks beyond the model's KV
    #               heads split the cache along the SEQUENCE, so the group
    #               stores one copy (--decode-context-parallel-size
    #               tp // kv_heads). The study's default — the layout its
    #               capacity numbers have always assumed; the deploy recipe
    #               now emits the flag that achieves it.
    #   "replicate" plain --tensor-parallel-size: every rank past the KV
    #               heads keeps a full copy. The pool divides by tp/kv_heads
    #               and every rank re-reads the whole cache each step.
    # research/kv_tp_sharding.md (codex finding F5 on the V4.1-Flash swap).
    kv_shard: str = "dcp"
    # Per-GPU bytes held back from the KV pool (activations, workspace,
    # fragmentation). None = ACT_RESERVE + gpu.reserve_extra, the H200-solved
    # reserve; a deployment whose vLLM start-up line reported its pool sets
    # the reserve that pool implies (`ws calibrate`, [calibration] reserve_gib).
    reserve: float | None = None

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


def topology_grid(dp: int, tp: int, gpu: str = "H200",
                  kv_shard: str = "dcp") -> Topology:
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
    if kv_shard not in KV_SHARDS:
        raise ValueError(f"kv_shard must be one of {KV_SHARDS}, got {kv_shard!r}")
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
    if kv_shard == "replicate" and tp > 1:
        name += " [KV replicated]"
    return Topology(name, dp, tp, g, kv_shard)


def kv_replication(model: Model, topo: Topology) -> tuple:
    """(cache factor, state factor): how many copies of each token's cache
    and of each session's state the TP group stores and re-reads per step.

    A cache with h shardable heads splits across min(tp, h) ranks and is
    replicated on the tp / h rank groups beyond that — 1.0 while tp <= h,
    tp / h past it (continuous for the odd TP widths the study prices; vLLM
    itself only accepts tp a multiple of h there). Decode context
    parallelism splits the replicated dimension along the sequence, so the
    "dcp" layout stores ONE copy: 1.0 at every tp. (At an odd width the
    heads do not divide, dcp_size() rounds down and a real engine would
    refuse the width; the study extrapolates it as sharded, exactly as it
    did before this layout existed — no published number moves.)
    """
    def factor(heads):
        if topo.kv_shard == "dcp" or heads is None or topo.tp <= heads:
            return 1.0
        return topo.tp / heads
    return factor(model.kv_heads), factor(model.state_heads)


def dcp_size(model: Model, topo: Topology) -> int:
    """The --decode-context-parallel-size a "dcp" topology needs: the rank
    groups beyond the KV heads, 1 (no flag) when the cache already shards."""
    if topo.kv_shard != "dcp":
        return 1
    return max(1, topo.tp // max(1, model.kv_heads))


def _replicated(model: Model, topo: Topology) -> Model:
    """The model with its cache and state bytes multiplied by the copies the
    group holds — what the pool and the decode step actually pay. Internal:
    kv_pool_tokens, warm_capacity and decode_curves apply it to the CALLER'S
    model, and nothing returns it, so it cannot be applied twice."""
    r_kv, r_st = kv_replication(model, topo)
    if r_kv == 1.0 and r_st == 1.0:
        return model
    return replace(
        model,
        kv_bpt=model.kv_bpt * r_kv,
        kv_decode_bpt=(None if model.kv_decode_bpt is None
                       else model.kv_decode_bpt * r_kv),
        kv_decode_const=model.kv_decode_const * r_kv,
        deltanet_state=model.deltanet_state * r_st,
        state_step_bytes=(None if model.state_step_bytes is None
                          else model.state_step_bytes * r_st))


def topology(kind: str, n_gpu: int, gpu: str = "H200",
             kv_shard: str = "dcp") -> Topology:
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
        return topology_grid(1, n_gpu, gpu, kv_shard)
    if kind == "dp":
        return topology_grid(n_gpu, 1, gpu, kv_shard)
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
    "does not fit" sentinel, asserted for GLM-5.3 on one GPU in _selfcheck).
    min_tp_for() reports the group size that would fit.

    The activation reserve is the H200-calibrated ACT_RESERVE applied per GPU
    on every part, plus the part's reserve_extra — the measured H200 HBM
    over-provision the solved reserve implicitly absorbed, added back on parts
    that do not share it (see GPU.reserve_extra and docs/scenarios.md).
    """
    check_dtype_supported(model, topo)
    reserve = (ACT_RESERVE + topo.gpu.reserve_extra if topo.reserve is None
               else topo.reserve)
    # NOT factored as tp*(vram - reserve) - w: keeping the three terms in the
    # original order makes every pre-existing configuration bit-identical to the
    # pre-grid code (tp=1 reproduces the old single/DP branch, tp=n the old TP
    # branch) rather than merely equal after formatting.
    pool_bytes = topo.tp * topo.gpu.vram - model.w_resident - topo.tp * reserve
    # ...in tokens of the cache the group STORES: a replicated layout pays
    # kv_bpt x copies per token (kv_replication), so the same bytes hold
    # fewer tokens. 1.0 on every default ("dcp") topology at a TP the KV
    # heads divide, i.e. every published number.
    return max(pool_bytes, 0.0) / _replicated(model, topo).kv_bpt


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
# STATUS: analytic. The MFU bracket rests on measurements not published in
# this repo (research/prefill.md §1). In THIS module's convention, mfu
# divides peak_flops(topo), which already carries the tp_efficiency haircut;
# state the convention when quoting. The bracket [0.30, 0.55] moves every
# figure here by ~1.8x. Still bounds that rank configurations, not latency
# commitments.
# ============================================================================

# Model FLOP Utilisation: achieved FLOPs / dense peak on a large prefill GEMM.
# 45% is a mid-range figure for FP8 attention+MLP prefill on Hopper-class
# parts with TP collectives in the loop. Every prefill number in this study
# should be read with the bracket, not the point.
MFU_LOW, MFU_DEFAULT, MFU_HIGH = 0.30, 0.45, 0.55



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
    # one decode STEP: a forward pass that hands each user `mtp` tokens
    # (p50 is per-user tokens/s, speculation included), and the gap a
    # stream sees between its server-sent events
    decode_s = model.tokens_per_step / p50[0]
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
    return _ServiceMix(model, topo, wl, chunk, turn_tokens, mfu,
                       per_pass_overhead).moments(wl.invalidation)


class _ServiceMix:
    """The two per-class service distributions, drawn once, mixed at any miss
    share f. The limits that price capacity misses (_capacity_limit) evaluate
    the same configuration at many shares; the classes do not depend on f
    (Workload.sample draws the lengths before the cold flags), so only the
    mixing is redone."""

    def __init__(self, model, topo, wl, chunk, turn_tokens, mfu,
                 per_pass_overhead):
        self.cold, self.warm = _prefill_service_arrays(
            model, topo, wl, chunk, turn_tokens, mfu, per_pass_overhead)
        self.e_cold, self.e_warm = float(self.cold.mean()), float(self.warm.mean())
        self.e_cold2 = float((self.cold ** 2).mean())
        self.e_warm2 = float((self.warm ** 2).mean())
        self._quantile = None

    def moments(self, f: float) -> tuple:
        """(E[S], E[S^2], E[S | miss], E[S | hit]) at miss share f."""
        e_s = f * self.e_cold + (1 - f) * self.e_warm
        e_s2 = f * self.e_cold2 + (1 - f) * self.e_warm2
        return e_s, e_s2, self.e_cold, self.e_warm

    def quantile(self, f: float, p: float) -> float:
        """ttft_service_quantile's mixture quantile at miss share f, p in
        (0, 1); the sort is done once per configuration."""
        if self._quantile is None:
            self._quantile = _mixture_quantiler(self.cold, self.warm)
        return self._quantile(f, p)


def miss_service_quantile(model: Model, topo: Topology, wl, chunk: float,
                          percentile: float, turn_tokens: float = 0.0,
                          mfu: float = MFU_DEFAULT,
                          per_pass_overhead: bool = False) -> float:
    """The `percentile`-th percentile of a MISS's own prefill time (seconds).

    A miss re-prefills its whole context and that cost is monotone in the
    context length, so this is the service time of a miss at the
    `percentile`-th percentile of the context-length distribution (the
    user/subagent lognormal mixture, truncated at the cap). Same draws as
    prefill_service_moments, so percentile -> E[S | miss] ordering is exact
    rather than a sampling accident.
    """
    if not 0 < percentile < 100:
        raise ValueError(f"percentile must be in (0, 100), got {percentile!r}")
    cold, _ = _prefill_service_arrays(model, topo, wl, chunk, turn_tokens,
                                      mfu, per_pass_overhead)
    return float(np.percentile(cold, percentile))


def ttft_service_quantile(model: Model, topo: Topology, wl, chunk: float,
                          percentile: float, turn_tokens: float = 0.0,
                          mfu: float = MFU_DEFAULT,
                          per_pass_overhead: bool = False) -> float:
    """The own-prefill term of the `percentile`-th TTFT over ALL requests,
    the population the probe scores (probe/population.py, ttft_all_pX) and
    the one an SLO is written against, in seconds.

    It is the `percentile`-th quantile of the MIXTURE of the two service
    distributions, misses at weight m = wl.invalidation and hits at 1 - m:
    the smallest sampled service time s with
        m F_miss(s) + (1 - m) F_hit(s) >= P.
    A mixture rather than a hit/miss split because the two overlap: a hit's
    long new turn over a long cached context can outlast a short miss, so no
    rank of one class above the other holds.

    Monotone in m only when the miss service distribution dominates the
    hit's (F_miss <= F_hit everywhere, e.g. every miss costs at least what a
    hit does). That fails with short prompts and long warm turns: at a
    1,000-token prompt median and an 8,000-token turn a hit re-prefills more
    than a miss does, and more misses LOWER c_p and raise the ceiling.

    Both per-draw costs are monotone in the context length, so each class's
    samples sort by sorting the lengths once; the explorer's
    ttftServiceQuantile walks the same two sorted sequences as a merge.
    """
    if not 0 < percentile < 100:
        raise ValueError(f"percentile must be in (0, 100), got {percentile!r}")
    cold, warm = _prefill_service_arrays(model, topo, wl, chunk, turn_tokens,
                                         mfu, per_pass_overhead)
    return _mixture_quantile(cold, warm, wl.invalidation, percentile / 100.0)


def _mixture_quantile(cold, warm, m: float, p: float) -> float:
    """Smallest value v of cold U warm with weighted CDF >= p, cold draws
    weighing m / n each and warm draws (1 - m) / n. Ties rank a warm draw
    first; the accumulation runs in sorted order, as the mirror's merge does.
    """
    return _mixture_quantiler(cold, warm)(m, p)


def _mixture_quantiler(cold, warm):
    """_mixture_quantile with the sort hoisted: the order of cold U warm does
    not depend on the weights, so a caller reading many miss shares sorts
    once. Returns q(m, p)."""
    n = len(cold)
    vals = np.concatenate([np.sort(warm), np.sort(cold)])
    order = np.argsort(vals, kind="stable")
    sorted_vals = vals[order]
    is_cold = order >= n

    def q(m: float, p: float) -> float:
        w = np.where(is_cold, m / n, (1.0 - m) / n)
        cum = np.cumsum(w)
        i = int(np.searchsorted(cum, p * (1.0 - 1e-12), side="left"))
        return float(sorted_vals[min(i, 2 * n - 1)])

    return q


def _ttft_own(model, topo, wl, chunk, turn_tokens, mfu, per_pass_overhead,
              e_cold: float, percentile) -> float:
    """E[S | miss] when `percentile` is None (a miss's mean TTFT), else the
    own-prefill term of the all-request percentile (ttft_service_quantile)
    — the one term the TTFT statistic changes."""
    if percentile is None:
        return e_cold
    return ttft_service_quantile(model, topo, wl, chunk, percentile,
                                 turn_tokens, mfu, per_pass_overhead)


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
                         per_pass_overhead: bool = False,
                         percentile: float | None = None) -> float:
    """Mean time-to-first-token: queueing delay + this request's own prefill.

    `percentile` (None = a miss's mean TTFT): return a PROXY for the
    `percentile`-th TTFT over ALL requests instead — mean wait plus the
    own-prefill term of ttft_service_quantile. `request` must be left at
    "cold" then; the statistic covers both classes. See max_users_latency.

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
    if percentile is not None and request != "cold":
        raise ValueError("percentile is an all-request statistic; "
                         "leave request='cold'")
    own = (_ttft_own(model, topo, wl, chunk, turn_tokens, mfu,
                     per_pass_overhead, e_cold, percentile)
           if request == "cold" else e_warm)
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
                  per_pass_overhead: bool = False, hi: float = 1.0,
                  percentile: float | None = None) -> float:
    """Largest miss rate whose mean TTFT still meets `sla_seconds`
    (`percentile` set: whose all-request p-th TTFT proxy does, as in
    max_users_latency).

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
                                    per_pass_overhead=per_pass_overhead,
                                    percentile=percentile)

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
    gets a step every t_mixed instead of every t_decode throughout, and
    every step carries `mtp` tokens, so the difference,
    drain * mtp * (1/t_decode - 1/t_mixed), is output tokens that never
    arrive. Without the `mtp` factor it counted steps, which undercounts by
    the accepted length (2.94x on the 27B). This is the metric that makes a
    cold spike legible to someone reading a latency dashboard rather than a
    duty cycle.
    """
    drain = burst_drain_seconds(model, topo, wl, burst, req_rate, chunk,
                                turn_tokens, mfu, per_pass_overhead)
    d_ms, x_ms, ratio = itl_spike(model, topo, wl, n_decode, chunk, mfu,
                                  n_iter=n_iter, seed=seed)
    if not np.isfinite(drain):
        return drain, ratio, float("inf"), float("inf")
    per_user = drain * model.tokens_per_step * (1e3 / d_ms - 1e3 / x_ms)
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
        # CPU cost per offloaded session: KV bytes + DeltaNet state (per
        # stored copy — the caller passes the replicated model)
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
    # Both budgets pay per stored copy: vLLM's native offload
    # (--kv-offloading-size) is RANK-LOCAL, each rank spilling its own blocks,
    # so a replicated cache is replicated in host memory too. A deduplicating
    # store (LMCache-style) that kept one copy is not modelled (codex F1).
    model = _replicated(model, topo)
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
def decode_token_seconds(model: Model, topo: Topology, mfu: float = None) -> float:
    """c_tok: GEMM time of ONE token position on one replica group, at
    prefill's kernel MFU and the group's RAW peak (no TP haircut: decode's
    all-reduces are tau_ar). No attention term: a decode position's attention
    is the KV READ, which the byte ledger carries."""
    if model.params_prefill <= 0:
        raise ValueError(f"{model.name}: no prefill constants "
                         "(params_prefill unset — see research/prefill.md)")
    mfu = MFU_DEFAULT if mfu is None else mfu
    if not 0 < mfu <= 1:
        raise ValueError(f"mfu must be in (0, 1], got {mfu!r}")
    if topo.gpu.peak_flops_fp8 <= 0:
        raise ValueError(f"{topo.gpu.name}: peak_flops_fp8 unset")
    return 2.0 * model.params_prefill / (topo.tp * topo.gpu.peak_flops_fp8 * mfu)


@dataclass(frozen=True)
class DecodeConstants:
    """The four constants of the decode step equation (see DECODE_ETA), plus
    the union arm. `ws calibrate` fits them; [calibration] overrides them.
    `compute=False` drops the verify-GEMM leg: with eta = 1 and every tau 0
    that is the pure byte roofline the baseline study priced."""
    eta: float = DECODE_ETA
    tau_0: float = None              # None = the solved prior (decode_priors)
    tau_ar: float = None
    tau_rec: float = None
    mfu: float = None                # None = MFU_DEFAULT
    union: str = "coverage"          # or "linear", the no-overlap upper bound
    compute: bool = True

    def __post_init__(self):
        if not 0 < self.eta <= 1:
            raise ValueError(f"eta must be in (0, 1], got {self.eta!r}")
        for f in ("tau_0", "tau_ar", "tau_rec"):
            v = getattr(self, f)
            if v is not None and v < 0:
                raise ValueError(f"{f} must be >= 0, got {v!r}")
        if self.union not in ("coverage", "linear"):
            raise ValueError(f"union must be 'coverage' or 'linear', got {self.union!r}")

    def taus(self) -> tuple:
        p0, par, prec = decode_priors()
        return (p0 if self.tau_0 is None else self.tau_0,
                par if self.tau_ar is None else self.tau_ar,
                prec if self.tau_rec is None else self.tau_rec)

    @classmethod
    def roofline(cls) -> "DecodeConstants":
        return cls(eta=1.0, tau_0=0.0, tau_ar=0.0, tau_rec=0.0, compute=False)


def _fixed_seconds(model: Model, tp: int, k: int, taus: tuple) -> float:
    """Per-layer fixed cost plus serial recurrent updates of one step."""
    tau_0, tau_ar, tau_rec = taus
    tau_l = tau_0 + (tau_ar if tp > 1 else 0.0)
    return (tau_l * (model.layers + model.passes(k) * model.draft_layers)
            + tau_rec * model.rec_layers * (1 + k))


_PRIORS = None


def decode_priors() -> tuple:
    """(tau_0, tau_ar, tau_rec) in seconds: the two priors and tau_rec solved
    from the anchor's measured intercept (see DECODE_ETA)."""
    global _PRIORS
    if _PRIORS is None:
        m = MODELS["27B"]
        k = _ANCHOR_SPEC_TOKENS
        draft_s = m.passes(k) * m.draft_pass_bytes(0) / (4 * GPUS["H200"].hbm_bw * DECODE_ETA)
        left = _ANCHOR_FIXED_S - draft_s - _fixed_seconds(m, 4, k, (TAU_0, TAU_AR, 0.0))
        if left <= 0:
            raise ValueError("anchor: draft bytes and the tau priors exceed the "
                             "measured 2.7 ms intercept")
        _PRIORS = (TAU_0, TAU_AR, left / (m.rec_layers * (1 + k)))
    return _PRIORS


def decode_step_seconds(model: Model, topo: Topology, n: int, kv_bytes,
                        decode: DecodeConstants = None, k: int = None):
    """Seconds of one decode step for n sequences whose KV and state reads
    total `kv_bytes` (scalar or array: already replicated, see decode_curves).
    k defaults to the model's spec_tokens."""
    dc = decode or DecodeConstants()
    k = model.spec_tokens if k is None else k
    positions = n * (1 + k)
    step_bytes = (model.w_decode(positions, dc.union)
                  + model.passes(k) * model.draft_pass_bytes(n, dc.union) + kv_bytes)
    t = step_bytes / (dc.eta * topo.tp * topo.gpu.hbm_bw)
    if dc.compute:
        t = t + positions * decode_token_seconds(model, topo, dc.mfu)
    return t + _fixed_seconds(model, topo.tp, k, dc.taus())


def decode_curves(model: Model, topo: Topology, wl: Workload, mns_range,
                  n_iter=3000, seed=0, decode: DecodeConstants = None):
    """Per-user tok/s percentiles (p5, p50, p95) and aggregate p50 vs the
    number of sequences decoding at once (max_num_seqs).

    per-user tok/s = tokens_per_step(acceptance, k) / t_step, with t_step from
    decode_step_seconds. The KV and state reads are sampled per sequence from
    the workload's context distribution; the weights, draft passes, verify
    compute and fixed terms do not depend on the contexts.
    """
    check_dtype_supported(model, topo)
    check_cap_allowed(model, wl)
    if kv_pool_tokens(model, topo) <= 0:
        raise ValueError(f"{model.name}: weights do not fit {topo.name} — "
                         "no context can be resident, decode is undefined")
    # a replicated cache is re-read by every rank that holds a copy: the
    # per-step KV and state bytes scale by the same factor as the storage
    model = _replicated(model, topo)
    rng = np.random.default_rng(seed)
    m_tok = tokens_per_step(model.acceptance, model.spec_tokens)
    # dense-attention models read every cached token per step (kv_bpt); a
    # sparse-attention model (DSA) reads kv_decode_bpt per context token
    # (indexer scan) plus a top-k read per active sequence, capped at the
    # sequence's own length when it is shorter than the top-k window
    kv_read_bpt = model.kv_bpt if model.kv_decode_bpt is None else model.kv_decode_bpt
    topk = model.kv_decode_topk
    p5, p50, p95, agg = [], [], [], []
    for n in mns_range:
        full, _, _, _ = wl.sample(rng, (n_iter, n))
        if model.kv_decode_const and topk:
            topk_bytes = (model.kv_decode_const / topk) * \
                np.minimum(full, topk).sum(axis=1)
        else:
            topk_bytes = n * model.kv_decode_const
        kv_bytes = full.sum(axis=1) * kv_read_bpt + topk_bytes + n * model.state_traffic
        pu = m_tok / decode_step_seconds(model, topo, n, kv_bytes, decode)
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
# There is a way to make them commensurable. A population conversion sits
# above the existing `users` load:
#
#   concurrent sessions = headcount x peak active share
#                       x sessions per active user
#
# `users` remains concurrent sessions per replica group. The population
# conversion therefore divides the system-wide product by the number of DP
# replica groups before any ceiling or arrival-rate formula sees it.
#
# Making the four ceilings commensurable then costs two assumptions, both
# stated rather than hidden:
#
#   1. ONE ACTIVE PERSON RUNS ONE OR MORE FULL REQUEST STREAMS. The conservative
#      reading prices each concurrent session as its own think-time loop. A
#      person who merely holds k sessions but attends them one at a time is the
#      lower-load end of the range: cache occupancy grows by k, request rate
#      does not.
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
# These assumptions are load-bearing, so they are limitations, not conveniences:
# see docs/scenarios.md § 9 "Reading the two-axis planner". Multiple independent
# sessions per person lower every ceiling expressed in people. Multiple held
# sessions attended one at a time lower only the cache ceiling. Bursty think
# time changes the load-derived frontiers again.
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
OUT_TOKENS_DEFAULT = 400    # mean output tokens per agentic request, from a
                            # trace not published in this repo. Was ASSUMED
                            # 1,000 ("decoded share of the 2,000-token turn").

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


def sessions_from_headcount(headcount: float, peak_active_share: float = 1.0,
                            sessions_per_active_user: float = 1.0) -> float:
    """Convert people with access into system-wide concurrent sessions.

    Each session is conservatively treated as a full request stream with its
    own think-time loop. Replica-group division belongs to the configuration
    layer because this function describes the whole population.
    """
    if headcount < 0:
        raise ValueError("workload.headcount must be >= 0")
    if not 0 < peak_active_share <= 1:
        raise ValueError("workload.peak_active_share must be in (0, 1]")
    if sessions_per_active_user < 1:
        raise ValueError("workload.sessions_per_active_user must be >= 1")
    return headcount * peak_active_share * sessions_per_active_user


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


def warm_draw(model: Model, topo: Topology) -> int:
    """Requests per warm fill: enough to exhaust this pool's budget.

    warm_capacity REFUSES a censored draw rather than under-counting, so the
    caller has to size it. One session costs at least a shared-prefix-sized
    remainder plus its recurrent state, so a pool of P tokens holds O(P/8000)
    of them on the study's workload; the flat 4,000 covers the small-pool end.
    Extracted from max_users_cache so a caller that ALSO wants the which="all"
    or which="gpu" arm of the same fill can ask for the same draw instead of
    restating the rule — two draw rules that must stay equal is a drift bug
    waiting to happen (the same reason AVG_OUT_TOK is an alias).
    """
    return int(4000 + kv_pool_tokens(model, topo) / 8000)


def max_users_cache(model: Model, topo: Topology, wl: Workload, ram_gib=0,
                    n_iter: int = 400, draw: int = None, seed: int = 0) -> float:
    """Users whose sessions fit warm in the pool — warm p5, USER-class only.

    p5, not p50: the study plans on the conservative tail. User-class, because
    subagent sessions occupy the pool but are not users (which="user" is the
    exact count the explorer approximates as warm x (1 - p_sub)).
    """
    if draw is None:
        draw = warm_draw(model, topo)
    return float(warm_capacity(model, topo, wl, ram_gib=ram_gib, n_iter=n_iter,
                               draw=draw, seed=seed, which="user")[0])


# ============================================================================
# THE HIT RATE AT THE CONFIGURED USERS  (issue #117, research/public_traces.md)
# ----------------------------------------------------------------------------
# max_users_cache is W, the user sessions one cache keeps warm. The planner
# used to read W as a cliff: past it the deployment failed on cache, which is
# what LRU does when prompts come round in a fixed order (docs/writeup.md:
# ten 140k-token prompts sent round-robin, each evicted just before its turn).
# No live trace does that. Sessions come back after gaps spread from seconds
# to hours, LRU loses the ones away longest, and past W the hit rate falls
# gradually (research/public_traces.md: Bailian and AgentX).
#
# The rule is LRU's characteristic time T*: a context stays resident until it
# has been idle T*, and T* is where the contexts so held fill the pool. A
# planner that knows only the think time takes the gaps as exponential, so a
# session is resident a share 1 - exp(-T*/Z) of its cycle and comes back warm
# with the same probability; U live sessions in W slots give
#     P(resident) = min(1, W' / U),   W' the sessions the pool holds.
# W counts REUSABLE sessions (warm_capacity does not count a forced miss), so
# W' = W / (1 - f0) and the share of requests that miss at U users is
#     f(U) = max(f0, 1 - W / U),      f0 the configured miss rate.
# Below the knee W / (1 - f0) nothing changes. Past it the extra misses are
# cold requests like any other: they feed the prefill queue, the cold wait and
# the slowed-generation share, so overload shows up as a latency limit. There
# is no cache ceiling in the verdict any more.
#
# The cyclic cliff stays what it is: the worst case of a benchmark that sends
# prompts round-robin, where every request past W users misses. It is W
# itself, reported beside the verdict.
# ============================================================================
def miss_share_at(users: float, warm_users: float | None,
                  invalidation: float) -> float:
    """Share of requests that miss the cache with `users` live sessions on a
    cache that keeps `warm_users` warm: max(invalidation, 1 - W / users).
    `warm_users` None (or inf) prices an unbounded cache: the configured
    `invalidation` alone."""
    if users < 0:
        raise ValueError(f"users must be >= 0, got {users!r}")
    if not 0 <= invalidation <= 1:
        raise ValueError(f"invalidation must be in [0, 1], got {invalidation!r}")
    if warm_users is None or users <= 0 or not math.isfinite(warm_users):
        return invalidation
    return max(invalidation, 1.0 - max(warm_users, 0.0) / users)


def cache_hit_rate(users: float, warm_users: float | None,
                   invalidation: float) -> float:
    """Share of requests served from cache at `users`: 1 - miss_share_at."""
    return 1.0 - miss_share_at(users, warm_users, invalidation)


def workload_at(wl: "Workload", users: float,
                warm_users: float | None) -> "Workload":
    """`wl` with its miss rate replaced by the miss share at `users`, which
    is what every at-load prefill figure (duty, queue, TTFT, B*, slowed share,
    power) is priced on."""
    return replace(wl, invalidation=miss_share_at(users, warm_users,
                                                  wl.invalidation))


def _capacity_limit(at_share, warm_users: float | None,
                    invalidation: float) -> float:
    """Users where a limit binds once capacity misses count.

    `at_share(f)` is the limit, in users, at a FIXED miss share f: each
    max_users_* closed form. Below the knee W / (1 - f0) the share is f0 and
    that is the answer. Past it the share at U users is 1 - W / U, and the
    limit is the root of U = at_share(1 - W / U). At a fixed share every
    limit's metric rises with the request rate, so the metric at U users is
    inside its budget exactly while U < at_share(f(U)): bisection on that
    sign. The mirror's capacityLimit (interactive/src/prefill.js) runs the
    same search.
    """
    u0 = at_share(invalidation)
    if warm_users is None or not math.isfinite(warm_users) or invalidation >= 1:
        return u0
    w = max(warm_users, 0.0)
    if w <= 0:
        return at_share(1.0)            # nothing stays warm: every request misses
    knee = w / (1.0 - invalidation)
    if u0 <= knee:
        return u0

    def g(u):
        return at_share(1.0 - w / u)

    lo = knee
    hi = u0 if math.isfinite(u0) else 2.0 * knee
    for _ in range(64):                 # past the knee the limit is finite
        if g(hi) <= hi:                 # unless no share at all reaches it
            break
        lo, hi = hi, 2.0 * hi
    else:
        return float("inf")
    # 100 halvings: past float resolution for any bracket the doubling can
    # leave (the mirror runs the same 100)
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if g(mid) > mid:
            lo = mid
        else:
            hi = mid
    return lo


def max_users_decode(model: Model, topo: Topology, wl: Workload,
                     floor: float = DECODE_FLOOR_TOKS,
                     n_iter: int = 400, seed: int = 0, hi: int = 4096,
                     decode: "DecodeConstants" = None) -> float:
    """Concurrent decoders at which per-user p50 tok/s falls to `floor`.

    Bisection, not the linear scan tables.py uses: per-user speed is monotone
    decreasing in concurrency (every extra sequence adds KV bytes to the same
    step), so ~12 evaluations replace up to 1,500. Returns 0 when even a single
    decoder misses the floor, and raises if the crossing lies beyond `hi`
    rather than returning a silently censored value.
    """
    def p50(n):
        return decode_curves(model, topo, wl, [n], n_iter=n_iter, seed=seed,
                             decode=decode)[1][0]
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


def decode_search_cap(floor: float = DECODE_FLOOR_TOKS) -> int:
    """The largest concurrency the decode-ceiling search probes.

    Per-user speed is aggregate/n, so the crossing sits at ~1/floor: a 5 tok/s
    floor lands ~8x higher than the study's 40. The cap scales the same way and
    stays at exactly 4,096 for floors of 40 and above. Mirrors HI in
    prefill.js maxUsersDecode."""
    return max(4096, round(4096 * DECODE_FLOOR_TOKS / floor))


def max_users_decode_bounded(model: Model, topo: Topology, wl: Workload,
                             floor: float = DECODE_FLOOR_TOKS, **kw) -> tuple:
    """(ceiling, censored): max_users_decode searched to decode_search_cap.

    A crossing beyond the cap is a LOWER BOUND, returned as (cap, True) -- what
    the explorer prints as ">= cap". max_users_decode itself keeps raising so
    that a caller who cannot carry the flag never takes the cap for a value.
    """
    cap = decode_search_cap(floor)
    try:
        return max_users_decode(model, topo, wl, floor=floor, hi=cap, **kw), False
    except ValueError as e:
        if "censored" not in str(e):
            raise
        return float(cap), True


def steady_decode_point(model: Model, topo: Topology, wl: Workload,
                        rate_group: float,
                        out_tokens: float = OUT_TOKENS_DEFAULT,
                        n_iter: int = 400,
                        seed: int = 0, hi: int = 4096,
                        resident: float = None,
                        decode: "DecodeConstants" = None) -> dict:
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
    THE BATCH CANNOT OUTGROW THE POOL. A batch of n sequences needs n
    contexts resident in HBM, and decode_curves prices exactly that read
    without asking whether they fit. So the search stops at `resident`, the
    GPU-resident warm population of one cache (warm_capacity which="gpu",
    p95: the most sessions any fill packs). Past it the engine is evicting
    and every returning session is a cold prefill, which is the miss-rate
    axis's business and is not priced here; a point found out there would be
    a number about a machine that does not exist. `saturated` is therefore
    True when the demand exceeds what the resident population retires (or,
    as a hard bracket, what `hi` decoders retire): no steady state exists and
    n is a floor, not an estimate. Callers that already hold the warm figure
    pass `resident`; None computes it at the same seed.

    `decode` is passed straight to decode_curves (None = the priors), so a
    caller honouring a calibration block prices this point and
    max_users_decode with the SAME constants. Without it the two
    disagreed silently: at mbu 0.1 vs 0.8 the decode ceiling moved 3 -> 171
    users while this point stayed at 59 tok/s either way.

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
    if resident is None:
        resident = float(warm_capacity(model, topo, wl, ram_gib=0,
                                       n_iter=max(40, n_iter // 4), seed=seed,
                                       which="gpu")[2])
    # at least one sequence decodes if anything does; below one resident
    # context decode_curves itself refuses (no pool), so the floor is safe
    cap = max(1, min(hi, int(math.floor(resident))))

    def v(n: int) -> float:
        return float(decode_curves(model, topo, wl, [n], n_iter=n_iter,
                                   seed=seed, decode=decode)[1][0])

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
    a_cap = cap * v(cap)
    if demand >= a_cap:
        # the resident population, all decoding at once, does not retire what
        # the load asks for: no steady state. delivered != demanded here,
        # which is exactly what saturated means (one field for both once
        # understated the demand by up to 24x)
        return {**out, "n": float(cap), "per_user_tok_s": a_cap / cap,
                "delivered_tok_s": a_cap * reps,
                "demanded_tok_s": demand * reps, "saturated": True}
    lo, up = 1, 2
    while up < cap and up * v(up) < demand:
        lo, up = up, min(cap, up * 2)
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


DECODE_SLOT_QUANTILE = 0.99   # the decode batch percentile a cap must hold


def poisson_cdf(k: int, mean: float) -> float:
    """P(N <= k) for N ~ Poisson(mean), summed term by term in log space so a
    mean in the thousands does not underflow. Mirrors poissonCdf in
    interactive/src/prefill.js, operation for operation."""
    if mean <= 0:
        return 1.0
    lm, lp, s = math.log(mean), -mean, math.exp(-mean)
    for i in range(1, int(k) + 1):
        lp += lm - math.log(i)
        s += math.exp(lp)
    return min(1.0, s)


def decode_slot_mean(max_num_seqs: int,
                     quantile: float = DECODE_SLOT_QUANTILE) -> float:
    """The largest MEAN decode batch whose `quantile` stays within
    max_num_seqs.

    Requests enter the batch as a random stream and each decodes for its own
    time, so the number decoding at once is an infinite-server queue's
    occupancy: Poisson with the steady_decode_point mean (M/G/inf; exact for
    Poisson arrivals, whatever the decode-time distribution). Binding where
    the MEAN reaches the cap would leave the batch over it about half the
    time, with a wait for a slot that never drains. Arrivals burstier than
    Poisson (an agent turn's back-to-back requests) widen the batch further,
    so this is the optimistic side of the percentile."""
    if max_num_seqs < 1:
        raise ValueError(f"max_num_seqs must be >= 1, got {max_num_seqs!r}")
    lo, hi = 0.0, float(max_num_seqs)
    for _ in range(60):
        mid = (lo + hi) / 2
        if poisson_cdf(max_num_seqs, mid) >= quantile:
            lo = mid
        else:
            hi = mid
    return lo


def recommended_max_num_seqs(bandwidth: float, resident: float) -> int:
    """The max_num_seqs working-set recommends, per replica group: the largest
    batch whose per-user p50 still meets the floor (max_users_decode, read as
    sequences), within the GPU-resident warm p5 (beyond it nobody warm is
    left to admit). Past the floor crossing a bigger batch only slows every
    sequence below the floor, so queueing is the better answer. Mirrors
    recommendedMns in interactive/src/prefill.js."""
    return max(1, int(math.floor(min(bandwidth, resident))))


def max_users_decode_slots(model: Model, topo: Topology, wl: Workload,
                           max_num_seqs: int,
                           think_time_s: float = THINK_TIME_S,
                           out_tokens: float = OUT_TOKENS_DEFAULT,
                           n_iter: int = 400,
                           seed: int = 0,
                           decode: "DecodeConstants" = None,
                           per_user_tok_s: float = None,
                           quantile: float = DECODE_SLOT_QUANTILE) -> float:
    """Users per replica group at which the steady decode batch's `quantile`
    reaches max_num_seqs.

    The scheduler caps the sequences decoding at once, and at a given load the
    batch that decodes is steady_decode_point's, not the warm population: an
    open-loop user spends most of each interval waiting on a tool or a human.
    The batch fluctuates around that mean (decode_slot_mean), so the cap
    binds where the mean reaches m = decode_slot_mean(max_num_seqs). By the
    steady point's flow balance,

        rate_group x out_tokens = m x v(m)

    converted to users the way max_users_saturation converts its rate:
    users = rate x think / (1 + r). v is read at ceil(m), the conservative
    side of the integer the decode curve is sampled at. Under a closed loop
    `think_time_s` is the waiting time Z alone, so this open conversion
    undercounts: the conservative side again.

    `per_user_tok_s` is v(ceil(m)) when the caller already drew it.
    """
    m = decode_slot_mean(max_num_seqs, quantile)
    if out_tokens <= 0:
        return math.inf
    v = per_user_tok_s
    if v is None:
        v = float(decode_curves(model, topo, wl, [max(1, math.ceil(m))],
                                n_iter=n_iter, seed=seed, decode=decode)[1][0])
    rate = m * v / out_tokens
    return rate * think_time_s / (1.0 + wl.sub_ratio)


def cap_decode_ceiling(bandwidth: float, max_num_seqs: int | None,
                       slots: float) -> tuple:
    """(decode ceiling, capped) once max_num_seqs is set; users per group.

    `bandwidth` is max_users_decode: every warm user decoding at once, per-user
    p50 at the floor. The engine never runs more than max_num_seqs at once, so
    a cap at or below that crossing keeps every batch at or above the floor
    and the bandwidth ceiling is never reached. Above it, the crossing stands.
    `slots` is max_users_decode_slots, where the steady batch fills the cap.
    The ceiling is the lower of the two that apply, and `capped` says the
    slots term is the one binding.
    """
    if max_num_seqs is None:
        return bandwidth, False
    bw = math.inf if max_num_seqs <= bandwidth else bandwidth
    return (slots, True) if slots < bw else (bw, False)


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
                         decode_toks: float = DECODE_FLOOR_TOKS,
                         warm_users: float | None = None) -> float:
    """Users at which prefill duty reaches 100% — section 8's f*, in users.

    `warm_users` (the cache's W, max_users_cache): price the capacity misses
    past it (miss_share_at, _capacity_limit). None keeps the configured miss
    rate at every load, the published tables' reading.

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
    mix = _ServiceMix(model, topo, wl, chunk, turn_tokens, mfu,
                      per_pass_overhead)

    def at(f):
        e_s = mix.moments(f)[0]
        demand = (1.0 + wl.sub_ratio) * e_s
        if demand <= 0:
            return float("inf")
        if closed_z_s is None:
            return think_time_s / demand
        _check_closed_args(closed_z_s, out_tokens, decode_toks)
        r0 = e_s + out_tokens / decode_toks
        return (closed_z_s + r0) / demand

    return _capacity_limit(at, warm_users, wl.invalidation)


def max_users_latency(model: Model, topo: Topology, wl, chunk: float,
                      sla_seconds: float, turn_tokens: float = 0.0,
                      think_time_s: float = THINK_TIME_S,
                      mfu: float = MFU_DEFAULT, discipline: str = "fcfs",
                      per_pass_overhead: bool = False,
                      closed_z_s: float = None,
                      out_tokens: float = OUT_TOKENS_DEFAULT,
                      decode_toks: float = DECODE_FLOOR_TOKS,
                      percentile: float | None = None,
                      warm_users: float | None = None) -> float:
    """Users at which the TTFT statistic reaches `sla_seconds`: a MISS's
    mean TTFT when `percentile` is None, else a proxy for the
    `percentile`-th TTFT over ALL requests. `warm_users`: price the capacity
    misses past the cache's W (_capacity_limit); None keeps the configured
    miss rate at every load.

    THE PERCENTILE PROXY. The probe scores p-th TTFT over every request, so
    that is the population here. The proxy is
        TTFT_p ~= E[W] + c_p
    with c_p = ttft_service_quantile, the P-th quantile of the hit/miss
    service mixture at the workload's miss share. It is c -> c_p in both
    closed forms below. It is NOT Q_p(W + S): the wait enters at its P-K mean
    because the model has no distribution for it, and a percentile of a sum
    is not the sum of a mean and a percentile. How far the proxy sits from
    the true percentile, and in which direction, is unmeasured.

    Default None (the mean) keeps every published table reproducible until
    the docs re-issue; the service-level path (predict, via
    [slo] percentile, default 95) and the explorer pass the percentile.

    Closed form in both disciplines, because E[S] and E[S^2] do not depend on
    the arrival rate:
        FCFS  lam a / (2(1 - lam b)) + c = SLA  ->  lam = k / (a + k b)
        PS    c / (1 - lam b) = SLA            ->  lam = (1 - c/SLA) / b
    with a = E[S^2], b = E[S], c = E[S | miss] (or c_p), k = 2(SLA - c).

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
    if percentile is not None and not 0 < percentile < 100:
        raise ValueError(f"percentile must be in (0, 100), got {percentile!r}")
    mix = _ServiceMix(model, topo, wl, chunk, turn_tokens, mfu,
                      per_pass_overhead)

    def at(f):
        b, a, c, _ = mix.moments(f)
        # the own-prefill term (_ttft_own): a miss's mean, or the all-request
        # percentile of the service mixture at this share
        if percentile is not None:
            c = mix.quantile(f, percentile / 100.0)
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

    return _capacity_limit(at, warm_users, wl.invalidation)


# ============================================================================
# THE LATENCY LIMITS USERS FEEL  (issue #94)
# ----------------------------------------------------------------------------
# The p-th TTFT over all requests sits in the miss tail, so it mostly reports
# prompt length: a long cold prompt misses a 10 s budget on an idle server
# and no user count fixes that. Two latencies grow with load instead, and
# they are the limits the verdict uses:
#
#   slowed generation   while a cold prefill runs, every step that carries
#                       one of its chunks takes decode step + chunk pass, and
#                       every decoding stream gets one step's tokens in that
#                       time. The limit is the share of wall time spent in
#                       such steps.
#   cold-request wait   the time before a cold request's own prefill starts:
#                       FCFS M/G/1 over the hit/miss service mixture.
#
# A cold request's own prefill on an idle server (idle_cold_ttft) is a
# property of the deployment and the prompt length, not a limit.
#
# Both are priced per replica group.
# ============================================================================

IDLE_TTFT_PROMPTS = (10_000, 50_000, 100_000, 200_000)


def context_quantile(wl, percentile: float, n: int = 200_000,
                     seed: int = 0) -> float:
    """The `percentile`-th context length (tokens), numpy's linear
    interpolation over the same draw context_moments takes."""
    if not 0 < percentile < 100:
        raise ValueError(f"percentile must be in (0, 100), got {percentile!r}")
    full, _, _, _ = wl.sample(np.random.default_rng(seed), n)
    return float(np.percentile(full.astype(float), percentile))


def idle_cold_ttft(model: Model, topo: Topology, prompt_tokens: float,
                   chunk: float, mfu: float = MFU_DEFAULT) -> float:
    """Seconds to the first token of a cold request of `prompt_tokens` on an
    idle replica group: its whole prompt re-prefilled, every chunk paying the
    weight stream (miss_context_seconds). No queue, no decode batch."""
    return miss_context_seconds(model, topo, prompt_tokens, chunk,
                                mfu_anchor=mfu)


def idle_cold_ttft_table(model: Model, topo: Topology, chunk: float,
                         max_model_len: float,
                         mfu: float = MFU_DEFAULT) -> tuple:
    """((prompt_tokens, seconds), ...) at IDLE_TTFT_PROMPTS up to
    `max_model_len`, and at max_model_len itself."""
    lengths = [L for L in IDLE_TTFT_PROMPTS if L < max_model_len]
    lengths.append(int(max_model_len))
    return tuple((L, idle_cold_ttft(model, topo, L, chunk, mfu))
                 for L in lengths)


def slowed_steps(model: Model, topo: Topology, wl, chunk: float,
                 decode_toks: float = DECODE_FLOOR_TOKS,
                 mfu: float = MFU_DEFAULT) -> dict:
    """The step lengths slowed generation is built from, in seconds.

    decode_s   one decode step: model.tokens_per_step accepted tokens per step at
               `decode_toks` per stream, so mtp / decode_toks. Priced at the
               decode floor rather than at the load's own batch: the floor is
               the slowest step the decode SLO admits, and a fixed step keeps
               the slowed share linear in the request rate. The chunk pass
               dominates the mixed step, so the choice moves the share little.
    chunk_s    one full-chunk prefill pass, priced as itl_spike prices it:
               marginal (the decode batch already streams the weights), at
               prior = E[L] / 2, the mean cache a chunk of a cold re-prefill
               attends over.
    mixed_s    decode_s + chunk_s, a step that carries the chunk.
    speed      model.tokens_per_step / mixed_s, tokens per second a stream gets while a
               cold prefill runs.
    """
    if decode_toks <= 0:
        raise ValueError(f"decode_toks must be > 0, got {decode_toks!r}")
    decode_s = model.tokens_per_step / decode_toks
    chunk_s = prefill_seconds(model, topo, chunk, mfu,
                              prior=mean_context(wl) / 2)
    mixed_s = decode_s + chunk_s
    return {"decode_s": decode_s, "chunk_s": chunk_s, "mixed_s": mixed_s,
            "speed": model.tokens_per_step / mixed_s}


def slowed_share(model: Model, topo: Topology, wl, req_rate: float,
                 chunk: float, decode_toks: float = DECODE_FLOOR_TOKS,
                 mfu: float = MFU_DEFAULT,
                 per_pass_overhead: bool = False) -> float:
    """Fraction of wall time spent in steps that carry a cold-prefill chunk.

        share = min(1, cold duty x mixed_s / chunk_s)
        cold duty = req_rate x f x E[S | miss]

    Cold duty is the machine time cold prefills need per second; each second
    of it runs in steps that also carry the decode batch, which stretches it
    by mixed_s / chunk_s. `req_rate` is the TOTAL arrival rate at the group.
    Warm-turn prefills are left out: a few thousand tokens slow a step far
    less than a full chunk does.
    """
    if req_rate < 0:
        raise ValueError(f"req_rate must be >= 0, got {req_rate!r}")
    st = slowed_steps(model, topo, wl, chunk, decode_toks, mfu)
    e_cold = cold_request_seconds(model, topo, wl, chunk, mfu,
                                  per_pass_overhead=per_pass_overhead)
    cold_duty = req_rate * wl.invalidation * e_cold
    return min(1.0, cold_duty * st["mixed_s"] / st["chunk_s"])


def slowed_stretch(model: Model, topo: Topology, wl, chunk: float,
                   percentile: float = 95.0,
                   decode_toks: float = DECODE_FLOOR_TOKS,
                   mfu: float = MFU_DEFAULT) -> float:
    """Seconds one cold request of the `percentile`-th prompt length keeps
    every stream slowed: its chunks, each run as a mixed step.

        stretch = ceil(L_p / C) x decode_s + prefill_context_seconds(L_p, C)

    Each chunk is priced with the cache it actually attends over (marginal
    pricing, as the chunk leg of the mixed step), so a long prompt's later
    chunks cost more than the E[L] / 2 chunk the slowed speed quotes.
    """
    length = context_quantile(wl, percentile)
    st = slowed_steps(model, topo, wl, chunk, decode_toks, mfu)
    return (math.ceil(length / chunk) * st["decode_s"]
            + prefill_context_seconds(model, topo, length, chunk, mfu))


def wait_quantile(mean_wait: float, rho: float, p: float) -> float:
    """The p-th quantile (p in (0, 1)) of an M/G/1 FCFS wait, from its mean.

    An approximation: P(W > t) ~= rho exp(-t rho / E[W]). It keeps the exact
    atom P(W = 0) = 1 - rho and the exact mean, and is exact for exponential
    service; this workload's heavy-tailed miss service makes the true tail
    heavier. So
        W_p = (E[W] / rho) ln(rho / (1 - p))   when p > 1 - rho, else 0.
    vLLM is neither FCFS nor processor sharing (prefill_ttft_seconds), so read
    this as the FCFS end of that bracket.
    """
    if not 0 < p < 1:
        raise ValueError(f"p must be in (0, 1), got {p!r}")
    if rho >= 1:
        return float("inf")
    if rho <= 0 or p <= 1 - rho:
        return 0.0
    return mean_wait / rho * math.log(rho / (1 - p))


def cold_wait_seconds(model: Model, topo: Topology, wl, req_rate: float,
                      chunk: float, turn_tokens: float = 0.0,
                      mfu: float = MFU_DEFAULT,
                      per_pass_overhead: bool = False,
                      percentile: float | None = None) -> float:
    """Time before a cold request's prefill starts. FCFS makes every request
    wait the same, so this is the queue wait: its mean (queue_wait_seconds)
    when `percentile` is None, else its `percentile`-th (wait_quantile).
    `req_rate` is the TOTAL arrival rate at the group."""
    if req_rate < 0:
        raise ValueError(f"req_rate must be >= 0, got {req_rate!r}")
    e_s, e_s2, _, _ = prefill_service_moments(model, topo, wl, chunk,
                                              turn_tokens, mfu,
                                              per_pass_overhead)
    rho = req_rate * e_s
    if rho >= 1:
        return float("inf")
    mean = req_rate * e_s2 / (2 * (1 - rho))
    if percentile is None:
        return mean
    return wait_quantile(mean, rho, percentile / 100.0)


def _users_at_rate(lam: float, wl, think_time_s: float, e_s: float,
                   e_s2: float, discipline: str, closed_z_s,
                   out_tokens: float, decode_toks: float) -> float:
    """Users whose TOTAL arrival rate is `lam`: lam x think / (1 + r) open,
    lam x (Z + R(lam) + decode) / (1 + r) closed (max_users_latency's
    conversion). inf when lam is at or past the prefill wall, where no
    population reaches it."""
    if not math.isfinite(lam) or lam * e_s >= 1:
        return float("inf")
    if closed_z_s is None:
        return max(0.0, lam * think_time_s / (1.0 + wl.sub_ratio))
    _check_closed_args(closed_z_s, out_tokens, decode_toks)
    if discipline == "ps":
        resp = e_s / (1 - lam * e_s)
    else:
        resp = lam * e_s2 / (2 * (1 - lam * e_s)) + e_s
    cycle = closed_z_s + resp + out_tokens / decode_toks
    return max(0.0, lam * cycle / (1.0 + wl.sub_ratio))


def max_users_slowed(model: Model, topo: Topology, wl, chunk: float,
                     share_max: float = 0.05, turn_tokens: float = 0.0,
                     think_time_s: float = THINK_TIME_S,
                     mfu: float = MFU_DEFAULT, discipline: str = "fcfs",
                     per_pass_overhead: bool = False,
                     closed_z_s: float = None,
                     out_tokens: float = OUT_TOKENS_DEFAULT,
                     decode_toks: float = DECODE_FLOOR_TOKS,
                     warm_users: float | None = None) -> float:
    """Users at which the slowed share reaches `share_max`.

    The share is linear in the request rate below its cap at 1, so
        lam = share_max x chunk_s / (f x E[S | miss] x mixed_s)
    and the users conversion is max_users_latency's. inf when nothing misses,
    when share_max >= 1, or when lam lies past the prefill wall (saturation
    binds first there).

    `warm_users`: price the capacity misses past the cache's W
    (_capacity_limit). Open loop, the cold stream past the knee is then the
    users the cache cannot hold, (1 + r)(U - W) / think, so the limit is
    min(the configured-share limit, W + K) with K the cold users that fill
    `share_max` on their own; None keeps the configured miss rate.
    """
    if not share_max > 0:
        raise ValueError(f"share_max must be > 0, got {share_max!r}")
    mix = _ServiceMix(model, topo, wl, chunk, turn_tokens, mfu,
                      per_pass_overhead)
    st = slowed_steps(model, topo, wl, chunk, decode_toks, mfu)

    def at(f):
        e_s, e_s2, e_cold, _ = mix.moments(f)
        per_rate = f * e_cold * st["mixed_s"] / st["chunk_s"]
        if share_max >= 1 or per_rate <= 0:
            return float("inf")
        return _users_at_rate(share_max / per_rate, wl, think_time_s, e_s,
                              e_s2, discipline, closed_z_s, out_tokens,
                              decode_toks)

    return _capacity_limit(at, warm_users, wl.invalidation)


def max_users_cold_wait(model: Model, topo: Topology, wl, chunk: float,
                        budget_s: float = 5.0, turn_tokens: float = 0.0,
                        think_time_s: float = THINK_TIME_S,
                        mfu: float = MFU_DEFAULT, discipline: str = "fcfs",
                        per_pass_overhead: bool = False,
                        closed_z_s: float = None,
                        out_tokens: float = OUT_TOKENS_DEFAULT,
                        decode_toks: float = DECODE_FLOOR_TOKS,
                        percentile: float | None = None,
                        warm_users: float | None = None) -> float:
    """Users at which the cold-request wait reaches `budget_s`.

    Mean (`percentile` None): lam a / (2(1 - lam b)) = B, so
        lam = 2B / (a + 2B b),   a = E[S^2], b = E[S].
    p-th: W_p(rho) = a / (2 b (1 - rho)) x ln(rho / (1 - p)) rises from 0 at
    rho = 1 - p to inf at rho = 1, so rho is found by bisection and
    lam = rho / b. The users conversion is max_users_latency's.
    `warm_users`: price the capacity misses past the cache's W
    (_capacity_limit); None keeps the configured miss rate at every load.
    """
    if not budget_s > 0:
        raise ValueError(f"budget_s must be > 0, got {budget_s!r}")
    if percentile is not None and not 0 < percentile < 100:
        raise ValueError(f"percentile must be in (0, 100), got {percentile!r}")
    mix = _ServiceMix(model, topo, wl, chunk, turn_tokens, mfu,
                      per_pass_overhead)

    def at(f):
        e_s, e_s2, _, _ = mix.moments(f)
        if e_s <= 0:
            return float("inf")
        if percentile is None:
            lam = 2 * budget_s / (e_s2 + 2 * budget_s * e_s)
        else:
            p = percentile / 100.0
            resid = e_s2 / (2 * e_s)            # E[W] (1 - rho) / rho
            lo, hi = 1 - p, 1.0
            for _ in range(100):
                mid = 0.5 * (lo + hi)
                if resid / (1 - mid) * math.log(mid / (1 - p)) < budget_s:
                    lo = mid
                else:
                    hi = mid
            lam = lo / e_s
        return _users_at_rate(lam, wl, think_time_s, e_s, e_s2, discipline,
                              closed_z_s, out_tokens, decode_toks)

    return _capacity_limit(at, warm_users, wl.invalidation)


def operating_point(model: Model, topo: Topology, wl: Workload, users: float,
                    chunk: float = CHUNK_DEFAULT, turn_tokens: float = 2_000,
                    sla_seconds: float = 10.0,
                    think_time_s: float = THINK_TIME_S,
                    decode_floor: float = DECODE_FLOOR_TOKS,
                    mfu: float = MFU_DEFAULT, discipline: str = "fcfs",
                    ram_gib=0, decode: "DecodeConstants" = None,
                    per_pass_overhead: bool = False,
                    closed: bool = False,
                    z_think_s: float = MEASURED_THINK_Z_S,
                    out_tokens: float = OUT_TOKENS_DEFAULT,
                    n_iter: int = 400, seed: int = 0,
                    ttft_percentile: float | None = None,
                    slowed_share_max: float = 0.15,
                    cold_wait_budget_s: float = 5.0) -> dict:
    """The ceilings in ONE unit — max concurrent users — plus which binds.

    `ceilings` holds the four that decide: decode, cold_wait
    (max_users_cold_wait at `cold_wait_budget_s`), slowed (max_users_slowed
    at `slowed_share_max`) and saturation. The TTFT ceiling
    (max_users_latency at `sla_seconds`) is reported as `ttft_ceiling` and
    no longer decides: at a realistic miss share the TTFT percentile sits in
    the miss tail and reports prompt length more than load.

    THE CACHE IS NOT A CEILING (issue #117). `warm_users` is W, the user
    sessions the pool keeps warm (max_users_cache, p5). Past W / (1 - f0)
    users the extra requests miss (miss_share_at) and every latency limit
    above prices them as cold requests, so a pool too small for the load shows
    up as cold wait, slowed generation or saturation. `hit_rate` and
    `miss_share` are the cache's figures at `users`. W is also the stated
    worst case of a round-robin benchmark, whose hit rate falls to ~0 past it
    (docs/writeup.md); it is reported, not decided on.

    THE planner. `binding` is the argmin: whichever ceiling is lowest
    is the one that actually limits this deployment, and `headroom` is how much
    of it the requested population uses. Everything is per replica GROUP; a DP
    deployment multiplies the decode ceiling and W by `topo.replicas` only
    under balanced routing, which sticky routing works against (spike.md #3).

    `closed=True` switches the latency and saturation columns to the
    closed-loop conversion: `z_think_s` (waiting only — measured 32.5 s)
    replaces `think_time_s` (the full interval — measured 43 s, reference
    30 s), and the model supplies its own service time. W is unchanged
    on purpose (held sessions occupy KV whether or not the user is active),
    and decode keeps its published worst-case reading (every user decoding
    at once) rather than the closed steady-state duty — both are stated in
    docs/scenarios.md § 9.

    `ttft_percentile` is the statistic of the TTFT ceiling and of the cold
    wait: None reads a miss's mean TTFT and the mean wait, a number that
    percentile of each.
    """
    if users < 0:
        raise ValueError(f"users must be >= 0, got {users!r}")
    z = z_think_s if closed else None
    warm = max_users_cache(model, topo, wl, ram_gib=ram_gib, n_iter=n_iter,
                           seed=seed)
    decode_n, decode_censored = max_users_decode_bounded(
        model, topo, wl, floor=decode_floor, decode=decode, n_iter=n_iter,
        seed=seed)
    ceilings = {
        "decode": decode_n,
        "cold_wait": max_users_cold_wait(model, topo, wl, chunk,
                                         cold_wait_budget_s, turn_tokens,
                                         think_time_s, mfu, discipline,
                                         per_pass_overhead, closed_z_s=z,
                                         out_tokens=out_tokens,
                                         decode_toks=decode_floor,
                                         percentile=ttft_percentile,
                                         warm_users=warm),
        "slowed": max_users_slowed(model, topo, wl, chunk, slowed_share_max,
                                   turn_tokens, think_time_s, mfu, discipline,
                                   per_pass_overhead, closed_z_s=z,
                                   out_tokens=out_tokens,
                                   decode_toks=decode_floor,
                                   warm_users=warm),
        "saturation": max_users_saturation(model, topo, wl, chunk, turn_tokens,
                                           think_time_s, mfu,
                                           per_pass_overhead,
                                           closed_z_s=z, out_tokens=out_tokens,
                                           decode_toks=decode_floor,
                                           warm_users=warm),
    }
    ttft_ceiling = max_users_latency(model, topo, wl, chunk, sla_seconds,
                                     turn_tokens, think_time_s, mfu,
                                     discipline, per_pass_overhead,
                                     closed_z_s=z, out_tokens=out_tokens,
                                     decode_toks=decode_floor,
                                     percentile=ttft_percentile,
                                     warm_users=warm)
    binding = min(ceilings, key=ceilings.get)
    limit = ceilings[binding]
    miss = miss_share_at(users, warm, wl.invalidation)
    return {
        "users": users,
        # MAIN-agent req/s in BOTH modes (the prefill server sees (1 + r) x
        # this). Open: the assumption-2 conversion. Closed: solved from the
        # cycle fixed point — response feedback slows the cadence, so this
        # is below users / z_think_s by construction. The closed cycle's
        # prefill is priced at the miss share at `users`.
        "req_rate": (closed_request_rate(model, topo,
                                         replace(wl, invalidation=miss),
                                         users, chunk, turn_tokens, z_think_s,
                                         out_tokens, decode_floor, mfu,
                                         discipline, per_pass_overhead)
                     if closed else request_rate(users, think_time_s)),
        "ceilings": ceilings,
        # True when the decode crossing lies beyond the search cap: the decode
        # ceiling is then a lower bound (">= N"), not a value
        "decode_censored": decode_censored,
        "ttft_ceiling": ttft_ceiling,
        "warm_users": warm,
        "miss_share": miss,
        "hit_rate": 1.0 - miss,
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
# The hours a month the team works, and so the hours the GPUs carry load:
# 10 h a day, 21 working days. Rented GPUs are up (and billed) only then;
# owned GPUs idle the rest of the month. A monthly reservation is rent at 720.
WORK_HOURS_DEFAULT = 210.0
# Months an owned server is written off over. Hyperscalers book 5 to 6 years
# (Amazon 5, Meta 5.5, CoreWeave 6, press reports); a GPU generation lasts
# about 2, so the planner takes 3 years, the cautious end.
AMORT_MONTHS_DEFAULT = 36.0
PAY_MODES = ("rent", "own")

# Output tokens ONE REQUEST decodes (applied to every request, subagents
# included) — ~400 mean, from a trace not published in this repo; was
# assumed 1,000. At 50-90 tok/s this decodes in ~5-8 s of the measured
# 10.8 s served per turn
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
               per_pass_overhead: bool = False,
               decode_floor: float = DECODE_FLOOR_TOKS) -> dict:
    """Average draw of one GPU and the whole system, at the current load.

    `rate_group` is the TOTAL req/s ONE replica group sees (main + subagent —
    the serverRate/replicas figure, same unit the queue metrics price);
    `decode_users_group` is the per-group decode ceiling at `decode_floor`
    tok/s (its capacity proxy, max_users_decode — pass the SAME floor that
    sized it, or the capacity and the price of that capacity disagree).
    Returns a dict:

      d_p        prefill duty = min(1, rate x E[S]) — E[S] is the mixed
                 cold/warm prefill service time the spike model already
                 computes, so d_p IS the section-8 duty cycle, clamped
      d_d        decode-active fraction: the output-token demand
                 (rate x out_tokens) against the floor capacity
                 (decode_users_group x decode_floor), capped at whatever
                 prefill leaves (partition, no overlap); a non-positive
                 capacity falls back to d_d = 1 - d_p (the explorer's guard)
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
    if decode_floor <= 0:
        raise ValueError(f"decode_floor must be > 0, got {decode_floor!r}")
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
    cap = decode_users_group * decode_floor              # output tok/s at floor
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
                eur_gpu_h: float = None,
                decode_floor: float = DECODE_FLOOR_TOKS,
                pay: str = "rent", work_hours: float = WORK_HOURS_DEFAULT,
                eur_capex_gpu: float = None,
                amort_months: float = AMORT_MONTHS_DEFAULT) -> dict:
    """The monthly bill of a deployment, power counted once — the
    explorer's energyCost().

    Two ways to pay for the GPUs (`pay`):

      rent   eur_gpu_h x n_gpu x work_hours: the GPUs are up, and billed,
             only the hours the team works. The rent includes power,
             cooling and the host, so there is no electricity line
             (eur_month 0). A monthly reservation is work_hours = 720 at
             its own lower eur_gpu_h.
      own    eur_capex_gpu x n_gpu / amort_months, plus the electricity at
             the meter: the load's draw (power_draw) for work_hours and the
             idle draw for the rest of the 720-hour month. pue and eur_kwh
             are exact multipliers on that line.

    hw_month is the rent or the write-off, eur_month the electricity,
    total_month their sum; eur_user divides the total across `users`
    (floored at 1, matching the explorer). eur_mtok divides it by the output
    tokens the load produces in work_hours (rate_group x replicas x
    out_tokens, every group equally loaded), Infinity at zero output.
    """
    if users < 0:
        raise ValueError(f"users must be >= 0, got {users!r}")
    if eur_kwh < 0:
        raise ValueError(f"eur_kwh must be >= 0, got {eur_kwh!r}")
    if eur_gpu_h is None:
        eur_gpu_h = topo.gpu.eur_gpu_h
    if eur_gpu_h < 0:
        raise ValueError(f"eur_gpu_h must be >= 0, got {eur_gpu_h!r}")
    if pay not in PAY_MODES:
        raise ValueError(f"pay must be one of {PAY_MODES}, got {pay!r}")
    if not 0 < work_hours <= HOURS_PER_MONTH:
        raise ValueError(f"work_hours must be in (0, {HOURS_PER_MONTH:g}], got {work_hours!r}")
    if eur_capex_gpu is None:
        eur_capex_gpu = topo.gpu.eur_capex_gpu
    if eur_capex_gpu < 0 or amort_months <= 0:
        raise ValueError("eur_capex_gpu must be >= 0 and amort_months > 0")
    # keywords past `chunk`: power_draw grew an out_tokens parameter, and a
    # positional tail would silently feed per_pass_overhead into it
    p = power_draw(model, topo, wl, rate_group, decode_users_group, chunk,
                   turn_tokens=turn_tokens, pue=pue, mfu=mfu,
                   out_tokens=out_tokens, per_pass_overhead=per_pass_overhead,
                   decode_floor=decode_floor)
    if pay == "rent":
        eur_month = 0.0
        hw_month = topo.n_gpu * eur_gpu_h * work_hours
    else:
        g = topo.gpu
        kw_idle = topo.n_gpu * (g.idle_w + g.host_w) * pue / 1000.0
        eur_month = (p["kw"] * work_hours
                     + kw_idle * (HOURS_PER_MONTH - work_hours)) * eur_kwh
        hw_month = topo.n_gpu * eur_capex_gpu / amort_months
    total_month = eur_month + hw_month
    out_tok_s = rate_group * topo.replicas * out_tokens
    eur_mtok = ((total_month / work_hours / 3600.0) / out_tok_s * 1e6
                if out_tok_s > 0 else float("inf"))
    return {**p, "pay": pay, "eur_month": eur_month, "hw_month": hw_month,
            "total_month": total_month,
            "eur_user": total_month / max(1.0, users),
            "eur_mtok": eur_mtok}


# ============================================================================
# SELF-CHECKS  (uv run python -m workingset.model, or `ws selfcheck`)
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
    # tokens per step: geometric closed form, per-position list, k = 0
    assert abs(tokens_per_step(0.4747, 2) - 1.7) < 1e-3
    assert tokens_per_step(0.0, 3) == 1.0 and tokens_per_step(0.9, 0) == 1.0
    assert abs(tokens_per_step(0.971, 3) - (1 - 0.971 ** 4) / (1 - 0.971)) < 1e-12
    assert abs(tokens_per_step((0.9, 0.8), 2) - (1 + 0.9 + 0.72)) < 1e-12
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
    # MM35, GLM-5.3, the DeepSeek Flash models, Qwen3.8-Flash-Next and GLM-5.3-Flash fit no
    # single H200, so pure DP is a 0 pool at every N -- the study's existing
    # "does not fit" sentinel, and it stands.
    # (GLM-5.3-Flash appears as its BF16-KV arm wherever an H200 topology is
    # priced: the fp8-KV arm is Blackwell-only and check_dtype_supported
    # refuses it there — asserted below)
    # (and GLM-5.3 as its Hopper FP8 row on H200: with_gpu, identity elsewhere)
    def _arm(mk, gk):
        m_ = MODELS[mk]
        m_ = (with_kv_dtype(m_, "fp16")
              if m_.kv_fp8_blackwell_only and gk == "H200" else m_)
        return with_gpu(m_, gk)
    for mdl in (MODELS["MM35"], _arm("GLM52", "H200"), MODELS["DSV4F"], MODELS["DSV4FV"],
                MODELS["DSV41F"], MODELS["Q38FN"], _arm("GLM53F", "H200"),
                MODELS["ML4"], _arm("K31P", "H200")):
        for n in (1, 2, 4, 8):
            assert kv_pool_tokens(mdl, topology("dp", n)) == 0
    # ...but replicating GROUPS does hold a real pool on one 8-GPU node.
    assert min_tp_for(MODELS["MM35"], "H200") == 2
    assert min_tp_for(MODELS["MM35"], "B300") == 1
    assert min_tp_for(MODELS["GLM52"], "H200") == 7
    assert min_tp_for(MODELS["GLM52"], "B300") == 3
    assert min_tp_for(MODELS["DSV4F"], "H200") == 2
    assert min_tp_for(MODELS["DSV4F"], "B300") == 1
    # the vision tower's 0.93 GB moves neither boundary
    assert min_tp_for(MODELS["DSV4FV"], "H200") == 2
    assert min_tp_for(MODELS["DSV4FV"], "B300") == 1
    # DSv4.1-Flash: 510.3 GB resident (the 203 GB Engram tables live in HBM
    # under vLLM) — 5 x 121.7 GB usable H200, 2 x 259.3 GB usable B300
    assert min_tp_for(MODELS["DSV41F"], "H200") == 5
    assert min_tp_for(MODELS["DSV41F"], "B300") == 2
    assert min_tp_for(MODELS["Q38FN"], "H200") == 2
    assert min_tp_for(MODELS["Q38FN"], "B300") == 1
    assert min_tp_for(_arm("GLM53F", "H200"), "H200") == 3
    assert min_tp_for(MODELS["GLM53F"], "B300") == 2
    # Mistral-Large-4: 1,060.5 GB in FP8 needs 9 x 121.7 GB usable H200, so
    # no 8-GPU node holds it (the vLLM recipe's floor is 2 x 8 H200); 5 x
    # 259.3 GB B300, or 3 with the NVFP4 checkpoint's 610.0 GB
    assert min_tp_for(MODELS["ML4"], "H200") == 9
    assert min_tp_for(MODELS["ML4"], "B300") == 5
    assert min_tp_for(with_weight_dtype(MODELS["ML4"], "nvfp4"), "B300") == 3
    # Kimi-K3.1-Pro: 1,566.6 GB with its drafter needs 13 usable H200 and
    # 7 B300, so TP8 on one 8xB300 node is its only single-node fit
    assert min_tp_for(_arm("K31P", "H200"), "H200") == 13
    assert min_tp_for(MODELS["K31P"], "B300") == 7
    assert kv_pool_tokens(MODELS["MM35"], topology_grid(4, 2)) > 0    # DP4xTP2
    assert kv_pool_tokens(MODELS["GLM52"], topology_grid(2, 4, "B300")) > 0
    # min_tp is exactly the boundary: one GPU less holds nothing
    for mk, gk in (("MM35", "H200"), ("MM35", "B300"),
                   ("GLM52", "H200"), ("GLM52", "B300"),
                   ("DSV4F", "H200"), ("DSV4F", "B300"),
                   ("DSV4FV", "H200"), ("DSV4FV", "B300"),
                   ("DSV41F", "H200"), ("DSV41F", "B300"),
                   ("Q38FN", "H200"), ("Q38FN", "B300"),
                   ("GLM53F", "H200"), ("GLM53F", "B300"),
                   ("ML4", "H200"), ("ML4", "B300"),
                   ("K31P", "H200"), ("K31P", "B300")):
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
                         ("DSV4FV", "H200", [(4, 2), (2, 4), (1, 8)]),
                         ("DSV4FV", "B300", [(8, 1), (4, 2), (2, 4), (1, 8)]),
                         ("DSV41F", "H200", [(1, 8)]),
                         ("DSV41F", "B300", [(4, 2), (2, 4), (1, 8)]),
                         ("Q38FN", "H200", [(4, 2), (2, 4), (1, 8)]),
                         ("Q38FN", "B300", [(8, 1), (4, 2), (2, 4), (1, 8)]),
                         ("GLM53F", "H200", [(2, 4), (1, 8)]),
                         ("GLM53F", "B300", [(4, 2), (2, 4), (1, 8)]),
                         ("ML4", "H200", []),
                         ("ML4", "B300", [(1, 8)]),
                         ("K31P", "H200", []),
                         ("K31P", "B300", [(1, 8)])):
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
    # GLM-5.3 (755.5 GB). DP buys cache isolation and routing headroom, never
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
        "GLM-5.3 NVFP4 resident must stay near the vLLM recipe's ~465 GB"
    assert glm_4.w_route_total == 256 * 21_233_688 * 75
    assert abs(glm_4.w_route_total / glm_4.w_route_pertok - 32) < 1e-9
    assert glm_4.w_decode_shared == glm.w_decode_shared + 15_891_309_576
    # every model but DSv4-Flash-Vision and Kimi-K3.1-Pro has an official
    # NVFP4 checkpoint (research/nvfp4_2026-10.md): DSv4-Flash-Vision has
    # community repacks only (2026-10-02), Kimi-K3.1-Pro none (2026-11-05;
    # its experts already ship MXFP4)
    for mk in MODELS:
        assert (MODELS[mk].nvfp4_w is None) == (mk in ("DSV4FV", "K31P")), mk

    # published-config KV identities for the new models
    assert mm.kv_bpt == 88 * 8 * 128 * 2 * 1                    # 176 KiB/token FP8
    assert glm.kv_bpt == 79 * 576 + 22 * 132                    # B300: fp8_e4m3 latent + indexer
    # H200: FlashMLA's fp8_ds_mla row, 512 fp8 + 4 fp32 scales + 64 bf16 rope
    assert glm.kv_hopper_fp8 == (79 * (512 + 16 + 128) + 22 * 132, 78 * 2048 * 656)
    glm_h = with_gpu(glm, "H200")
    assert (glm_h.kv_bpt, glm_h.kv_decode_const) == glm.kv_hopper_fp8
    assert with_gpu(glm, "B300").kv_bpt == glm.kv_bpt          # the base row is Blackwell's
    assert with_gpu(with_kv_dtype(glm, "fp16"), "H200").kv_bpt == 2 * glm.kv_bpt
    for bad_call in (lambda: kv_pool_tokens(glm, topology("tp", 8)),          # unresolved on H200
                     lambda: kv_pool_tokens(glm_h, topology_grid(1, 4, "B300")),  # H200 row on B300
                     lambda: with_kv_dtype(glm_h, "fp16")):                    # fp16 after the swap
        try:
            bad_call(); raise AssertionError("expected ValueError")
        except ValueError:
            pass
    assert all(MODELS[k].kv_hopper_fp8 is None for k in MODELS if k not in ("GLM52", "K31P"))
    assert mm.spec_tokens == 0 and mm.tokens_per_step == 1.0, "no MTP module on Mistral Medium 3.5"
    assert mm.deltanet_state == 0.0 and glm.deltanet_state == 0.0

    # DSv4-Flash-0731 identities (research/model_dsv4flash.md): compressed caches
    ds4 = MODELS["DSV4F"]
    assert ds4.kv_bpt == 21 * 576 / 4 + 20 * 576 / 128 + 21 * 64 / 4   # 3,450 B
    assert ds4.deltanet_state == 46 * 128 * 576 + 12_206_080    # windows + fp32 state
    # the fp32 compressor state (inference/model.py Compressor): kv_state and
    # score_state, each (1 + overlap) x ratio slots of (1 + overlap) x d
    # elements, overlap = (ratio == 4). CSA main d=512, CSA indexer d=128
    # (ratio 4, overlapping), HCA d=512 (ratio 128)
    def _buf(ratio, d):
        c = 2 if ratio == 4 else 1
        return 2 * (c * ratio) * (c * d) * 4
    assert 12_206_080 == 21 * _buf(4, 512) + 21 * _buf(4, 128) + 20 * _buf(128, 512)

    # per decode step, per sequence (the operations of Compressor.forward's
    # decode branch, each group's work amortized over `ratio` steps):
    #   write  one slot of c x d into both buffers, every step
    #   pool   read, when a group completes: both halves-selected slices
    #          (2 x ratio slots x d with overlap, ratio x d without)
    #   roll   overlap only: state[:, :ratio] = state[:, ratio:] in both
    #          buffers — ratio x 2d read AND ratio x 2d written
    def _compressor_step(ratio, d):
        c = 2 if ratio == 4 else 1
        write = 2 * (c * d) * 4
        pool = 2 * (c * ratio) * d * 4 / ratio
        roll = (2 * 2 * ratio * (2 * d) * 4 / ratio) if c == 2 else 0
        return write + pool + roll
    assert _compressor_step(4, 512) == 32_768       # 8,192 write + 8,192 pool + 16,384 roll
    assert _compressor_step(4, 128) == 8_192
    assert _compressor_step(128, 512) == 8_192      # no overlap: write + whole-buffer pool
    assert ds4.state_step_bytes == (43 * 576          # one ring-buffer window slot per main layer
                                    + 21 * _compressor_step(4, 512)
                                    + 21 * _compressor_step(4, 128)
                                    + 20 * _compressor_step(128, 512))
    assert ds4.w_route_pertok == 6 * 13_369_344 * 43            # FP4 experts + E8M0 scales
    assert ds4.w_route_total == 256 * 13_369_344 * 43
    assert abs(ds4.w_route_total / ds4.w_route_pertok - 256 / 6) < 1e-9  # kink ~42.7
    assert ds4.kv_decode_bpt == 21 * 64 / 4 + 20 * 576 / 128    # scan + dense HCA
    assert ds4.kv_decode_const == 21 * 512 * 576 + 43 * 128 * 576
    assert ds4.kv_decode_topk == 2_048
    # 21 QK-only indexers at 64 x 128 / 4 (halved under 2 T^2 d) + 20 dense HCA
    assert abs(ds4.attn_layers * ds4.attn_d
               - (21 * 64 * 128 / 4 / 2 + 20 * 64 * 512 / 128)) < 1e-9
    assert not ds4.state_fp32_ok and not ds4.kv_fp16_ok
    # NVIDIA's 0731 NVFP4 repacks the natively-4-bit experts with denser
    # scales: the arm is HEAVIER than FP8 everywhere but the fixed read
    ds4_4 = with_weight_dtype(ds4, "nvfp4")
    assert ds4_4.w_route_total == 256 * 14_155_800 * 43
    assert ds4_4.w_route_pertok == 6 * 14_155_800 * 43
    assert ds4_4.w_route_total > ds4.w_route_total and ds4_4.w_resident > ds4.w_resident
    assert ds4_4.w_decode_shared == ds4.w_decode_shared
    # only the routed experts moved: the resident delta IS the expert delta
    # (w_resident's FP8 figure is rounded to 166.88e9, hence the tolerance)
    assert abs((ds4_4.w_resident - ds4.w_resident)
               - (155_827_046_400 - 147_169_738_752)) < 1e7

    # DSv4-Flash-Vision-Exp (research/model_dsv4flash_vision.md): the 0731
    # text backbone + a vision tower. Every field but the name, the resident
    # bytes, the (absent) NVFP4 arm and the draft count (each model card
    # serves its own depth: 7 vs 3) IS DSV4F's — the text config and the
    # cache code are unchanged, so a drift here is a copy error
    ds4v = MODELS["DSV4FV"]
    _same = ("name", "w_resident", "nvfp4_w", "spec_tokens")
    assert all(getattr(ds4v, f.name) == getattr(ds4, f.name)
               for f in fields(Model) if f.name not in _same)
    # resident delta = ViT 823,685,120 + aligner 109,068,288 + 4 image
    # embeddings 32,768 (BF16) + fp32 gate biases: 46 bias_vl x 256 x 4 and
    # the 3 hash layers' own bias 3 x 256 x 4 (shard headers, 2026-10-02)
    assert ds4v.w_resident == 166_878_536_440 + (823_685_120 + 109_068_288 + 32_768
                                                 + 46 * 256 * 4 + 3 * 256 * 4)
    assert ds4v.nvfp4_w is None

    # DSv4.1-Flash identities (research/model_dsv41flash.md): four shared
    # compressed caches (CSA2), FP4 main KV, CED prefill
    DEEPSEEK_FLASH = ("DSV4F", "DSV4FV", "DSV41F")   # fixed window + compressor state
    # every model whose deltanet_state is a FIXED latent window, not a bf16
    # recurrent state: the DeepSeek Flash models and Mistral-Large-4's DSpark
    # drafter windows
    FIXED_STATE = DEEPSEEK_FLASH + ("ML4",)
    dsf = MODELS["DSV41F"]
    assert dsf.kv_bpt == 3 * (288 + 68) / 2 + (288 + 68)           # 890 B, the paper's figure
    assert dsf.deltanet_state == 43 * 128 * 528 + 3 * 2 * 2 * 512 * 4  # windows + fp32 state
    assert dsf.state_step_bytes == 40 * 528 + 3 * 2 * 2 * 512 * 4        # slot writes + state r/w
    assert dsf.state_traffic == dsf.state_step_bytes and all(
        MODELS[k].state_step_bytes is None and
        MODELS[k].state_traffic == 2 * MODELS[k].deltanet_state
        for k in MODELS if k not in FIXED_STATE)
    assert not dsf.state_fp32_ok and all(
        MODELS[k].state_fp32_ok for k in MODELS if k not in FIXED_STATE)
    assert dsf.w_route_pertok == 6 * 18_800_640 * 40            # MXFP4 experts + E8M0 scales
    assert dsf.w_route_total == 384 * 18_800_640 * 40
    assert abs(dsf.w_route_total / dsf.w_route_pertok - 64) < 1e-9  # kink at n = 64
    assert dsf.kv_decode_bpt == 3 * 68 / 2 + 68                 # four full-axis scans
    assert dsf.kv_decode_const == (38 * 512 * 288 + 4 * 16_384 * 68
                                   + 40 * 128 * 528)
    assert dsf.kv_decode_topk == 1_024
    assert abs(dsf.attn_layers * dsf.attn_d - 3 * (32 * 128 / 2 / 2)) < 1e-9  # 3 QK-only indexers
    # CED: prefill activates ~half of decode's 16B (8B), and the Engram
    # tables are 40% of the resident bytes — the fit is set by them
    assert 7.5e9 < dsf.params_prefill < 8.5e9
    assert dsf.w_resident > 2 * (dsf.w_route_total + dsf.w_decode_shared) * 0.8
    # NVIDIA's NVFP4 repacks the natively-4-bit experts with denser scales,
    # as on 0731: heavier everywhere but the fixed read, and only experts move
    dsf_4 = with_weight_dtype(dsf, "nvfp4")
    assert dsf_4.w_route_total == 384 * 19_906_584 * 40
    assert dsf_4.w_route_pertok == 6 * 19_906_584 * 40
    assert dsf_4.w_decode_shared == dsf.w_decode_shared
    assert (dsf_4.w_resident - dsf.w_resident
            == dsf_4.w_route_total - dsf.w_route_total)

    # Mistral-Large-4 identities (research/model_mistral_large4.md): MLA +
    # DSA on 72 layers; 63 store latents (9 NoPE layers attend to layer 35's)
    # and 44 run an indexer (28 reuse the top-k of the indexer before them)
    ml4 = MODELS["ML4"]
    # (rows are vLLM's fp8_ds_mla on both parts: 512 fp8 + 16 B scales + 128 B rope)
    assert ml4.kv_bpt == 63 * 656 + 44 * 132                    # 47,136 B
    assert ml4.kv_decode_bpt == 44 * 132                        # every indexer scans
    assert ml4.kv_decode_const == 72 * 2048 * 656               # every layer gathers top-k
    assert ml4.deltanet_state == 3 * 2048 * 656                 # DSpark windows, storage only
    assert ml4.state_traffic == 0.0 and not ml4.state_fp32_ok
    assert not ml4.kv_fp16_ok and not ml4.kv_fp8_blackwell_only   # fp8_ds_mla on both parts
    # an expert is 3 x 6144^2 FP8 plus 3 x 48^2 F32 block-128 scales, and the
    # ModelOpt NVFP4 one 3 x 6144^2 / 2 packed + 3 x 6144^2 / 16 E4M3 scales +
    # 6 F32 globals (shard headers, 2026-11-04)
    assert ml4.w_route_pertok == 4 * (3 * 6144**2 + 3 * 48**2 * 4) * 71
    assert ml4.w_route_total == 128 * (3 * 6144**2 + 3 * 48**2 * 4) * 71
    assert ml4.w_route_total / ml4.w_route_pertok == 32         # kink at n = 32
    ml4_4 = with_weight_dtype(ml4, "nvfp4")
    _e4 = 3 * 6144**2 // 2 + 3 * 6144**2 // 16 + 6 * 4
    assert ml4_4.w_route_pertok == 4 * _e4 * 71 and ml4_4.w_route_total == 128 * _e4 * 71
    assert ml4_4.w_decode_shared == ml4.w_decode_shared
    assert (ml4_4.w_resident - ml4.w_resident
            == ml4_4.w_route_total - ml4.w_route_total)
    # active GEMM params: card "49B" (52B with embeddings and lm_head)
    assert 49e9 < ml4.params_prefill < 50e9

    # Kimi-K3.1-Pro identities (research/model_kimi_k31pro.md): 69 KDA + 24
    # DSA layers; the 5-layer dense DSpark drafter's latents share the row
    k31 = MODELS["K31P"]
    assert k31.kv_bpt == 24 * 576 + 24 * 132 + 5 * 576          # B300: fp8_e4m3 rows
    assert k31.kv_hopper_fp8 == (24 * 656 + 24 * 132 + 5 * 656, 24 * 2048 * 656)
    assert k31.kv_decode_bpt == 24 * 132 + 5 * 576              # scans + the drafter's dense read
    assert k31.kv_decode_const == 24 * 2048 * 576
    assert k31.deltanet_state == 69 * 96 * 128 * 128 * 2 + 69 * 3 * 12_288 * 4 * 2
    assert k31.rec_layers == 69 and k31.layers == 93 and k31.kv_heads == 1
    # an expert is 3 x (3072 x 3584) MXFP4: values packed two per byte plus one
    # UE8M0 scale per 32 (shard headers, 2026-11-05)
    _ek = 3 * 3072 * 3584 // 2 + 3 * 3072 * 3584 // 32
    assert k31.w_route_pertok == 16 * _ek * 92 and k31.w_route_total == 896 * _ek * 92
    assert k31.w_route_total / k31.w_route_pertok == 56         # kink at n = 56
    # closing identity: shared read + experts + embeddings + vision tower
    # = the index's total_size, and the drafter's checkpoint on top
    assert (k31.w_decode_shared + k31.w_route_total + 2_348_810_240 + 894_717_952
            == 1_561_217_377_280 == k31.w_resident - 5_373_504_514)
    # a block drafter: one pass per step whatever the depth
    assert k31.passes(5) == 1 and k31.passes(0) == 0 and ml4.passes(3) == 3
    assert 103e9 < k31.params_prefill < 104e9                    # card "104B activated"

    # KV sharding under TP (research/kv_tp_sharding.md): every default
    # ("dcp") topology at a TP the KV heads divide stores ONE copy, so every
    # published number is untouched; the "replicate" arm pays tp/kv_heads
    # copies past the heads, and the single-latent models pay it at every tp > 1
    for mk, heads in (("27B", 4), ("35BA3B", 2), ("MM35", 8), ("GLM52", 1),
                      ("DSV4F", 1), ("DSV4FV", 1), ("DSV41F", 1), ("Q38FN", 2),
                      ("GLM53F", 1), ("ML4", 1), ("K31P", 1)):
        assert MODELS[mk].kv_heads == heads, mk
        for tp_ in (1, 2, 4, 8):
            assert kv_replication(MODELS[mk], topology_grid(1, tp_, "B300")) == (1.0, 1.0)
            rep = topology_grid(1, tp_, "B300", "replicate")
            r_kv, r_st = kv_replication(MODELS[mk], rep)
            assert r_kv == max(1.0, tp_ / heads), (mk, tp_, r_kv)
            assert r_st == (r_kv if mk in FIXED_STATE else 1.0), (mk, tp_, r_st)
            assert dcp_size(MODELS[mk], topology_grid(1, tp_, "B300")) == max(1, tp_ // heads)
            assert dcp_size(MODELS[mk], rep) == 1
    assert all(MODELS[k].state_heads == 1 for k in FIXED_STATE) and all(
        MODELS[k].state_heads is None for k in MODELS if k not in FIXED_STATE)
    # odd TP: the default layout is one copy at every width (an extrapolation
    # where the heads do not divide); "replicate" pays the continuous ratio
    assert kv_replication(m27, topology_grid(1, 6, "B300")) == (1.0, 1.0)
    assert kv_replication(m27, topology_grid(1, 6, "B300", "replicate"))[0] == 1.5
    assert kv_replication(glm, topology_grid(1, 3, "B300", "replicate"))[0] == 3.0
    assert dcp_size(m27, topology_grid(1, 6, "B300")) == 1
    # the pool of a replicated group divides by the factor, the host offload
    # buffer does not (one copy), and the per-step KV read grows by it
    b8r = topology_grid(1, 8, "B300", "replicate")
    b8 = topology_grid(1, 8, "B300")
    assert abs(kv_pool_tokens(glm, b8r) * 8 - kv_pool_tokens(glm, b8)) < 1e-6
    assert kv_pool_tokens(mm, b8r) == kv_pool_tokens(mm, b8)          # 8 heads: no copies
    assert abs(kv_pool_tokens(m35, b8r) * 4 - kv_pool_tokens(m35, b8)) < 1e-6
    assert "[KV replicated]" in b8r.name and "[KV replicated]" not in b8.name
    assert topology_grid(1, 1, "B300", "replicate").name == "1xB300"
    p_r = warm_capacity(glm, b8r, wl, n_iter=60)[1]
    p_d = warm_capacity(glm, b8, wl, n_iter=60)[1]
    assert 0 < p_r < p_d / 4, (p_r, p_d)
    off_r = warm_capacity(glm, b8r, wl, ram_gib=512, n_iter=60, which="offload")[1]
    off_d = warm_capacity(glm, b8, wl, ram_gib=512, n_iter=60, which="offload")[1]
    assert 0 < off_r < off_d / 4, "rank-local host offload replicates with the cache"
    _, v_r, _, _ = decode_curves(glm, b8r, wl, [64], n_iter=200)
    _, v_d, _, _ = decode_curves(glm, b8, wl, [64], n_iter=200)
    assert v_r[0] < v_d[0], "eight ranks re-reading the whole MLA cache must decode slower"
    try:
        topology_grid(1, 2, "B300", "sharded"); raise AssertionError("expected ValueError")
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
    # NVIDIA's NVFP4: routed experts x0.5625 (4 bits + E4M3 block-16 scales),
    # everything else — n-gram table included — byte-identical
    q384 = with_weight_dtype(q38, "nvfp4")
    assert q384.w_route_total == 512 * 2_764_824 * 48
    assert q384.w_route_pertok == 10 * 2_764_824 * 48
    assert q384.w_decode_shared == q38.w_decode_shared
    assert abs(q384.w_resident - (q38.w_resident - q38.w_route_total + q384.w_route_total)) < 1e6
    assert abs(2_764_824 / 4_915_800 - 0.5625) < 2e-4
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
    # RedHatAI's NVFP4: 42 decode layers' experts x0.5625, MTP layer's FP8
    # experts kept, the rest upcast to BF16 (+2.75e9 on the fixed read)
    g534 = with_weight_dtype(g53, "nvfp4")
    assert g534.w_route_total == 288 * 14_155_800 * 42
    assert g534.w_route_pertok == 8 * 14_155_800 * 42
    assert g534.w_decode_shared - g53.w_decode_shared == 2_748_499_332
    assert g534.w_resident < g53.w_resident * 0.61        # 197.8e9 vs 328.3e9
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

    # KV dtype: Mistral-Medium and GLM-5.3 (BF16 on both parts) double like the
    # Qwens; the DeepSeek Flash models' CSA / CSA2 paths must refuse FP16 (they
    # serve only with a quantized main KV)
    assert with_kv_dtype(mm, "fp16").kv_bpt == 2 * mm.kv_bpt
    assert with_kv_dtype(glm, "fp16").kv_bpt == 2 * glm.kv_bpt
    for quant_only in (ds4, ds4v, dsf):
        try:
            with_kv_dtype(quant_only, "fp16")
            raise AssertionError("expected ValueError")
        except ValueError:
            pass

    # context caps (owner decision 2026-07): Qwens + GLM allow up to 1M
    # (Qwen native 262k, 1M via YaRN); Mistral's hard model max is 262,144;
    # the DeepSeek Flash models are natively 1M (YaRN x16 baked into the config)
    assert (m27.max_ctx == m35.max_ctx == glm.max_ctx == ds4.max_ctx == ds4v.max_ctx
            == dsf.max_ctx
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

    # GLM-5.3 sizing: FP8 weights (755.5e9 B) fit no single GPU — pool clamps
    # to 0 — but 8xH200 TP and 4xB300 hold a real pool, NVFP4 a bigger one
    assert kv_pool_tokens(glm_h, t1) == 0 and kv_pool_tokens(glm, b1) == 0
    assert kv_pool_tokens(glm_h, topology("tp", 8)) > 0
    assert kv_pool_tokens(with_weight_dtype(glm, "nvfp4"), b4) > kv_pool_tokens(glm, b4) > 0

    # sparse-attention decode: GLM's DSA pricing must beat the dense-read
    # pricing of the same bytes at long context (that is DSA's entire point);
    # same check for DSv4.1-Flash's compressed-sparse reads (8xH200: the
    # smallest H200 node split that holds its 510 GB)
    glm_dense_read = replace(glm_h, kv_decode_bpt=None, kv_decode_const=0.0)
    t8 = topology("tp", 8)
    _, p_dsa, _, _ = decode_curves(glm_h, t8, wl, [64], n_iter=300)
    _, p_dense, _, _ = decode_curves(glm_dense_read, t8, wl, [64], n_iter=300)
    assert p_dsa[0] > p_dense[0], "DSA decode must out-speed full-cache reads"
    dsf_dense_read = replace(dsf, kv_decode_bpt=None, kv_decode_const=0.0)
    _, p_csa, _, _ = decode_curves(dsf, t8, wl, [64], n_iter=300)
    _, p_full, _, _ = decode_curves(dsf_dense_read, t8, wl, [64], n_iter=300)
    assert p_csa[0] > p_full[0], "CSA2 decode must out-speed full-cache reads"
    ds4_dense_read = replace(ds4, kv_decode_bpt=None, kv_decode_const=0.0)
    _, p_csa4, _, _ = decode_curves(ds4, tp2, wl, [64], n_iter=300)
    _, p_full4, _, _ = decode_curves(ds4_dense_read, tp2, wl, [64], n_iter=300)
    assert p_csa4[0] > p_full4[0], "CSA decode must out-speed full-cache reads"
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
    for mdl, topo_fit in ((mm, tp2), (glm_h, t8), (ds4, tp2), (dsf, t8), (q38, tp2),
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

    # ---- THE OPERATING POINT: the ceilings in one unit ---------------------
    # the reference load is exactly the study's 64 users / 30 s = 2.13 req/s
    assert abs(request_rate(REF_USERS) - RATE) < 0.01, \
        "64 users at one turn per 30 s must reproduce the 2.13 req/s reference"
    op = operating_point(m27, tp2, wl, REF_USERS, CH, TURN, SLA)
    # the TTFT ceiling left the verdict (issue #94): it is reported beside it;
    # the cache left it too (issue #117): its misses feed the other four
    assert set(op["ceilings"]) == {"decode", "cold_wait", "slowed",
                                   "saturation"}
    w_ref = op["warm_users"]
    assert w_ref == max_users_cache(m27, tp2, wl)
    assert op["ttft_ceiling"] == max_users_latency(m27, tp2, wl, CH, SLA, TURN,
                                                   warm_users=w_ref)
    assert op["binding"] == min(op["ceilings"], key=op["ceilings"].get)
    # the reference load sits inside W, so nothing misses beyond f0
    assert op["miss_share"] == wl.invalidation and \
        abs(op["hit_rate"] - (1 - wl.invalidation)) < 1e-15

    # ---- THE HIT RATE AT THE CONFIGURED USERS (issue #117) -----------------
    # f(U) = max(f0, 1 - W/U): f0 up to the knee W/(1 - f0), then the users
    # the cache cannot hold; W = None is the unbounded cache
    f0 = wl.invalidation
    knee = w_ref / (1 - f0)
    assert miss_share_at(0.9 * knee, w_ref, f0) == f0
    assert abs(miss_share_at(2 * w_ref, w_ref, f0) - 0.5) < 1e-12
    assert miss_share_at(10 * w_ref, None, f0) == f0
    assert miss_share_at(5, 0.0, f0) == 1.0, "an empty pool misses everything"
    assert workload_at(wl, 2 * w_ref, w_ref).invalidation == \
        miss_share_at(2 * w_ref, w_ref, f0)
    # each limit is unchanged where it sits inside the knee, and never higher
    # than the unbounded-cache reading (a miss costs more than a hit here)
    for fn, args in ((max_users_slowed, (0.15, TURN)),
                     (max_users_cold_wait, (5.0, TURN)),
                     (max_users_saturation, (TURN,)),
                     (max_users_latency, (SLA, TURN))):
        u_inf = fn(m27, tp2, wl, CH, *args)
        u_w = fn(m27, tp2, wl, CH, *args, warm_users=w_ref)
        assert u_w <= u_inf, f"{fn.__name__}: capacity misses raised the limit"
        assert (u_w == u_inf) == (u_inf <= knee), \
            f"{fn.__name__}: must move exactly when it sat past the knee"
        assert fn(m27, tp2, wl, CH, *args, warm_users=float("inf")) == u_inf
        # at the limit the metric meets its budget at the shifted share: the
        # root of U = at_share(f(U)), not a point on the old curve
        if u_w < u_inf:
            wl_u = workload_at(wl, u_w, w_ref)
            assert abs(fn(m27, tp2, wl_u, CH, *args) / u_w - 1) < 1e-9, \
                f"{fn.__name__}: the limit must be the fixed point"
    # open loop, slowed generation has a closed form past the knee: the cold
    # stream is the (U - W) users the cache cannot hold, so the limit is
    # min(K / f0, W + K) with K / f0 the configured-share limit
    k_cold = max_users_slowed(m27, tp2, wl, CH, 0.15, TURN) * f0
    u_sl = max_users_slowed(m27, tp2, wl, CH, 0.15, TURN, warm_users=w_ref)
    assert abs(u_sl - min(k_cold / f0, w_ref + k_cold)) < 1e-6 * u_sl, \
        "slowed limit past the knee must be W + K"
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
    assert op_c["warm_users"] == op_o["warm_users"] and \
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
    # The published decode tables (docs/scenarios.md, mns@40 = 118 / 228)
    # were priced on the retired byte roofline at mtp 1.7; they are not
    # reproduced any more. What is pinned instead is the construction of the
    # step equation's priors: at the anchor (27B, 4xH200 TP4, k = 3) the
    # draft passes plus the fixed terms make exactly the held-batch sweep's
    # measured 2.7 ms intercept.
    tp4 = topology("tp", 4, "H200")
    k_a = _ANCHOR_SPEC_TOKENS
    intercept = (m27.passes(k_a) * m27.draft_pass_bytes(0)
                 / (DECODE_ETA * tp4.tp * tp4.gpu.hbm_bw)
                 + _fixed_seconds(m27, 4, k_a, decode_priors()))
    assert abs(intercept - _ANCHOR_FIXED_S) < 1e-12, "the tau priors must close on the anchor"
    assert decode_priors()[2] > 0, "the anchor must leave a positive recurrent cost"
    # KNOWN DISAGREEMENT, recorded in research/decode-model.md: the k = 2
    # measurement the retired MBU fold was fitted on put the 27B/TP4 n = 1
    # step at ~9.3 ms (314 tok/s); this form, which follows the k = 3 sweep,
    # reads 8.0 ms (366 tok/s) there. A k-sweep on one server settles it, so
    # no per-user band is asserted until then.
    _, p50_tp4, _, _ = decode_curves(m27, tp4, wl, [1, 2, 4], n_iter=400)
    assert p50_tp4[0] > p50_tp4[1] > p50_tp4[2], "per-user speed falls with the batch"
    # wiring guard: the constants reach the ceiling search (the pure byte
    # roofline is faster than the full equation)
    dec = max_users_decode(m27, t1, wl, n_iter=200)
    assert dec < max_users_decode(m27, t1, wl, n_iter=200,
                                  decode=DecodeConstants.roofline()), \
        "the step equation must lower the decode ceiling against the byte roofline"
    assert max_users_decode(m27, t1, wl, floor=20.0, n_iter=200) > dec, \
        "a lower floor must admit more concurrent decoders"
    # an unreachable floor is a censored result, not a silent cap
    try:
        max_users_decode(m27, t1, wl, floor=1e-6, n_iter=200)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    # WHICH constraint binds must change with the miss rate — the planner's
    # whole reason to exist (decode at f=1%, slowed generation once misses
    # get common: at f=25% cold chunks fill steps long before any queue grows)
    b_lo = operating_point(m27, tp2, wl, REF_USERS, CH, TURN, SLA,
                           n_iter=200)["binding"]
    b_hi = operating_point(m27, tp2, replace(wl, invalidation=0.25),
                           REF_USERS, CH, TURN, SLA, n_iter=200)["binding"]
    assert b_lo != b_hi, \
        f"the binding constraint must switch with f (got {b_lo} at both ends)"
    assert b_hi == "slowed", f"slowed generation must bind at f=25%, got {b_hi}"
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
    #    (Re-pinned when OUT_TOKENS_DEFAULT moved 1,000 -> 400: fewer
    #    decode-seconds per request means a smaller steady batch running
    #    faster. Old pin: ~6.4 at ~370 tok/s.
    #    Re-pinned again for MBU_DEFAULT + the 27B's calibrated mtp:
    #    slower decode keeps each request in the batch longer, so the steady
    #    batch GROWS and each member runs slower. 2.2 at 430 -> 6.8 at 138.
    #    The band is tight: cross-seed spread is ~0.1 in n and ~2 in tok/s,
    #    so an earlier +/-1.0 / +/-25 pin -- justified by hand-waving about a
    #    steeper curve -- would have let real drift through. Measured, not
    #    reasoned.)
    #    Re-pinned for the decode step equation (k = 3 at acceptance 0.971,
    #    3.83 tokens per step, against the fold's 2.94): 6.8 at 138 -> 4.2 at
    #    221; cross-seed spread 0.03 in n and 2 tok/s.
    assert abs(sdp["n"] - 4.2) <= 0.3, \
        f"27B/TP2 reference load: ~4.2 decoders expected, got {sdp['n']:.2f}"
    assert abs(sdp["per_user_tok_s"] - 221) <= 10, \
        f"27B/TP2 reference load: ~221 tok/s expected, " \
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
    # 6b. and so is a demand past the GPU-RESIDENT population: a batch of n
    #     needs n contexts in HBM, and past that every return is a cold
    #     prefill this point does not price. 27B/TP2 holds ~200 resident; a
    #     demand that 4,096 decoders would retire is still no steady state.
    res = float(warm_capacity(m27, tp2, wl, n_iter=100, which="gpu")[2])
    v_res = float(decode_curves(m27, tp2, wl, [int(res)], n_iter=100)[1][0])
    past = steady_decode_point(m27, tp2, wl, 1.5 * int(res) * v_res / OUT_TOKENS_DEFAULT,
                               n_iter=100, resident=res)
    assert past["saturated"] and past["n"] == float(int(res)), \
        f"a batch past the resident population ({res:.0f}) must saturate, " \
        f"got n={past['n']} saturated={past['saturated']}"
    assert past["delivered_tok_s"] < past["demanded_tok_s"]
    #     and the explicit `resident` matches the computed one in kind
    assert steady_decode_point(m27, tp2, wl, rate_g, n_iter=100, resident=res)["n"] \
        == steady_decode_point(m27, tp2, wl, rate_g, n_iter=100, resident=None)["n"], \
        "inside the population the cap must not move the point"
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
    # the decode floor is a PARAMETER, not a constant: cap = ceiling x floor,
    # so halving it doubles the decode duty wherever the 1 - d_p clip is slack
    p_f40 = power_draw(m27, tp2, wl, 0.1, du, turn_tokens=TURN)
    p_f20 = power_draw(m27, tp2, wl, 0.1, du, turn_tokens=TURN,
                       decode_floor=DECODE_FLOOR_TOKS / 2)
    assert p_f40 == power_draw(m27, tp2, wl, 0.1, du, turn_tokens=TURN,
                               decode_floor=DECODE_FLOOR_TOKS), \
        "decode_floor must default to DECODE_FLOOR_TOKS"
    assert abs(p_f20["d_d"] - 2 * p_f40["d_d"]) < 1e-12, \
        "d_d must scale as 1/decode_floor below the 1 - d_p clip"
    # rent: the rent alone, power inside it, for the hours the team works
    cost_ref = energy_cost(m27, tp2, wl, rate_ref, du, REF_USERS,
                           turn_tokens=TURN)
    assert cost_ref["eur_month"] == 0.0 and cost_ref["total_month"] == cost_ref["hw_month"]
    assert abs(cost_ref["hw_month"]
               - tp2.n_gpu * GPUS["H200"].eur_gpu_h * WORK_HOURS_DEFAULT) < 1e-6
    assert energy_cost(m27, tp2, wl, rate_ref, du, REF_USERS, turn_tokens=TURN,
                       eur_kwh=2 * EUR_PER_KWH_DEFAULT)["total_month"] == cost_ref["total_month"], \
        "a rented GPU's bill must not move with the tariff"
    # own: the write-off plus electricity, the electricity LINEAR in the
    # tariff and in PUE (user-chosen multipliers, not model error)
    own = dict(turn_tokens=TURN, pay="own")
    own_ref = energy_cost(m27, tp2, wl, rate_ref, du, REF_USERS, **own)
    own_2e = energy_cost(m27, tp2, wl, rate_ref, du, REF_USERS, **own,
                         eur_kwh=2 * EUR_PER_KWH_DEFAULT)
    own_2p = energy_cost(m27, tp2, wl, rate_ref, du, REF_USERS, **own, pue=2 * PUE_DEFAULT)
    assert abs(own_2e["eur_month"] / own_ref["eur_month"] - 2) < 1e-12, \
        "eur_month must be linear in eur_kwh"
    assert abs(own_2p["eur_month"] / own_ref["eur_month"] - 2) < 1e-12, \
        "eur_month must be linear in pue"
    assert abs(own_ref["hw_month"]
               - tp2.n_gpu * GPUS["H200"].eur_capex_gpu / AMORT_MONTHS_DEFAULT) < 1e-6
    assert own_2e["hw_month"] == own_ref["hw_month"] == own_2p["hw_month"]
    assert own_ref["total_month"] == own_ref["eur_month"] + own_ref["hw_month"]
    # a month of work at full load draws more than a month of mostly idle
    assert energy_cost(m27, tp2, wl, rate_ref, du, REF_USERS, **{**own, "work_hours": 720.0}
                       )["eur_month"] > own_ref["eur_month"]
    # every part must carry both prices: the dataclass defaults are 0.0 so
    # that an unpriced GPU cannot pass as free by accident
    for _gk, _g in GPUS.items():
        assert _g.eur_gpu_h > 0, f"GPUS[{_gk!r}] has no eur_gpu_h"
        assert _g.eur_capex_gpu > 0, f"GPUS[{_gk!r}] has no eur_capex_gpu"

    assert cost_ref["eur_user"] == cost_ref["total_month"] / REF_USERS
    assert 0 < cost_ref["eur_mtok"] < float("inf")
    # zero output rate: €/1M tokens is undefined (inf), never a silent zero
    assert energy_cost(m27, tp2, wl, 0.0, du, REF_USERS,
                       turn_tokens=TURN)["eur_mtok"] == float("inf")
    # guards
    for bad_kw in (dict(rate_group=-1.0), dict(decode_users_group=-1.0),
                   dict(pue=0.0), dict(decode_floor=0.0)):
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
                continue   # Mistral-Large-4 / DSv4-Flash (CSA) / DSv4.1-Flash (CSA2): FP16 KV not servable
            if dtype == "fp8" and MODELS[mk].kv_fp8_blackwell_only:
                continue   # GLM-5.3-Flash: fp8 KV is Blackwell-only and these
                           # legacy topologies are all H200 — its H200 arm is
                           # the fp16 row below
            for tk in TOPOLOGIES:
                p = kv_pool_tokens(with_gpu(with_kv_dtype(MODELS[mk], dtype), "H200"),
                                   TOPOLOGIES[tk])
                print(f"  pool {mk:7} {tk:12} {dtype:5} = {p / 1e6:6.2f} M tokens")
    print("  -- B300, fp8 KV, weight dtype fp8 | nvfp4 --")
    for mk in MODELS:
        for n in (1, 4):
            t = topology("tp", n, "B300")
            pools = []
            for wd in WEIGHT_DTYPES:
                if wd == "nvfp4" and MODELS[mk].nvfp4_w is None:
                    pools.append("   n/a")
                    continue
                mdl = with_weight_dtype(MODELS[mk], wd)
                pools.append(f"{kv_pool_tokens(mdl, t) / 1e6:6.2f}")
            print(f"  pool {mk:7} {t.name:19} = {' | '.join(pools)} M tokens")


if __name__ == "__main__":
    _selfcheck()
