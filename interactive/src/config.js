/* ============================================================================
   CONFIG — ALL physical constants in one place (mirrors scenario_model.py).
   Update MoE / provisional numbers here; everything downstream re-derives.
   ========================================================================== */
export const KIB = 1024, MIB = 1024**2, GIB = 1024**3;

// DECODE efficiency -- MFU's counterpart, and the newest constant here.
// Decode was a PURE roofline until 2026-08-28 (bytes / effective_bw, no
// efficiency term at all) while prefill had carried an anchor for weeks.
// research/decode_mbu.md measures the gap on the reference row -- 27B /
// 4xH200 TP4, production, five sessions -- at 4.1x optimistic, and ONE
// constant brings it to a median 13% error across n = 1-25.
// MODEL CONVENTION (already divided by tp_efficiency), so it multiplies
// effective_bw directly; the raw advertised reading is 0.179.
// It is fitted on a HYBRID model running speculative decode at k=2, so it
// absorbs both streaming inefficiency and whatever the speculative verify
// costs on a sequential recurrence. It therefore travels WITH the model's
// mtp: change one and the fit breaks. Mirrors MBU_DEFAULT in scenario_model.py.
// EXTRAPOLATION BIAS: fitted over n = 1-25, where weights are 80-95% of the
// step. The decode ceiling lives at n ~ 100-250, where KV dominates and a
// whole-ledger multiplier over-charges. The 27B/TP4 ceiling reads 108 here,
// 238 under decode_mbu.md's reading A and 154 under reading B -- against a
// cache ceiling of 249, so the BINDING ORDER is not identified. Per-user
// speed near the fitted mix is; the ceiling is not. See MBU_DEFAULT.
// ONE value for EVERY row, from the slider. A dense or MLA model has no
// recurrence to serialise and should sit higher, but nothing in this study
// measures one, and a per-model guess would rank the frontier on the guess.
// So the slider is a GLOBAL sensitivity knob: drag it and watch which
// configurations survive. (Not the MTP rule -- own capability per row --
// because MTP is a shipped feature and MBU is an unmeasured efficiency.)
// Per-row values come back the day a second architecture is measured.
export const DECODE_MBU    = 0.22;

export const CONFIG = {
  // ---- HARDWARE ----
  // The GPU is a selectable part. H200 is the CALIBRATED baseline (the
  // activation reserve is solved from its 27B anchor and applied per GPU on
  // every part — a flagged assumption for the B300, see research/gpu_b300.md).
  // supports_nvfp4: native FP4 tensor cores — NVFP4 weights are gated to
  // these parts (Hopper's Marlin fallback is deliberately not modelled).
  // reserve_extra: the measured H200 HBM over-provision (~9.75e9 B — vendor
  // "141 GB" actually delivers ~150.75e9 usable) that the SOLVED reserve
  // absorbs implicitly. A part without that over-provision (the B300 delivers
  // its nominal bytes: a real nvidia-smi dump shows 275,040 MiB = 288.4e9 B)
  // must add it back, or its pools inherit ~9.75 GB/GPU of phantom HBM.
  // nvlink_domain: GPUs reachable over NVLink without leaving the node. A TP
  // group wider than this crosses onto IB/Ethernet and pays the extra
  // CROSS_DOMAIN_EFFICIENCY penalty. 8 for both parts — HGX H200 and the 8-GPU
  // HGX B300 baseboard we deploy (a GB300 NVL72 rack would be 72; we have none).
  // peak_flops_fp8: DENSE FP8 tensor-core FLOP/s, used ONLY by the prefill
  // ceiling (research/prefill.md). NVIDIA's headline "3,958 TFLOPS" is WITH
  // 2:4 sparsity, which no dense LLM GEMM reaches — halve it. B300: the DGX
  // B300 datasheet's 72 PFLOPS FP8 (8 GPUs, sparse) -> 4.5 dense/GPU; the
  // Ultra FP4 uplift did NOT carry to FP8, so "FP8 = FP4/2" is wrong here.
  // Power constants (research/power.md): tdp_w is the spec plate; idle_w a
  // measured H100 warm-idle proxy +10% (B300 extrapolated at the same ~10.5%
  // of TDP); p_decode_w the measured bandwidth-bound token-phase draw (0.55 x
  // TDP central, band 0.45-0.75); p_prefill_w the power-cap-limited prefill
  // draw (0.90 x TDP — the cap binds before the FLOP peak, so it is FLAT
  // across the MFU 35-55% band); host_w the DGX chassis adder per GPU (spec
  // ceiling, conservative). B300 rows are ENTIRELY extrapolated — no
  // published Blackwell Ultra power-state measurements exist.
  // eur_gpu_h: what one GPU-hour RENTS for, on demand, no commitment — the
  // hardware line of the bill (research/power.md prices only the electricity).
  // NVIDIA publishes no list price for either part and owned hardware
  // amortises to whatever the buyer's depreciation schedule says, so the
  // defensible public number is the cross-provider on-demand MEDIAN:
  // getdeploying.com/gpus/nvidia-h200 $4.40 and /nvidia-b300 $7.89 per
  // GPU-hour (both read 2026-09-03; H200 spans $3.59 RunPod – $10.00 OCI,
  // B300 $6.50 – $17.80 AWS), converted at EUR/USD 1.16 (market quote, same day)
  // and rounded to a slider step. The ratio (~1.8x) is what the frontier
  // actually leans on: the GPU-price slider scales BOTH parts by one factor,
  // so a reader on a hyperscaler or an owned rack moves the level, not the
  // relative ranking. Not a measurement; revisit when the market moves.
  GPUS: {
    "H200": { name: "H200", vram: 141e9,   hbm_bw: 4.8e12, supports_nvfp4: false, reserve_extra: 0,      nvlink_domain: 8, peak_flops_fp8: 1.979e15,
              tdp_w: 700,  idle_w: 80,  p_decode_w: 385, p_prefill_w: 630,  host_w: 575, eur_gpu_h: 3.8 },
    "B300": { name: "B300", vram: 288.4e9, hbm_bw: 8.0e12, supports_nvfp4: true,  reserve_extra: 9.75e9, nvlink_domain: 8, peak_flops_fp8: 4.5e15,
              tdp_w: 1400, idle_w: 150, p_decode_w: 770, p_prefill_w: 1260, host_w: 500, eur_gpu_h: 6.8 },
  },
  VRAM_PER_GPU: 141e9,     // calibration-anchor GPU (H200) HBM bytes
  HBM_BW:       4.8e12,    // calibration-anchor GPU bandwidth, bytes/s
  TP_EFFICIENCY: 0.90,     // tensor-parallel aggregate-BW haircut
  // extra haircut per doubling once a TP group spans nodes. UNMEASURED and
  // deliberately pessimistic; nothing in the study crosses the boundary.
  CROSS_DOMAIN_EFFICIENCY: 0.65,
  // Calibration anchor: 1xH200 + 27B FP8 KV pool (tokens). PROJECTED from the
  // baseline's measured FP16 pool (~1.337M x2 + freed activations), not measured.
  BASELINE_POOL_TOKENS_27B_1GPU: 2.77e6,

  // ---- MODELS ----
  // kv_bpt: KV bytes/token (FP8) | deltanet_state: recurrent state bytes/session
  // w_resident: FP8 weights resident | w_decode_shared: always-active weight bytes/step
  // w_route_pertok: routed-expert bytes per decoding token | w_route_total: all routed experts
  // mtp: Multi-Token-Prediction decode speedup (1.0 = model has no MTP module)
  // nvfp4_w: [w_resident, w_decode_shared, w_route_pertok, w_route_total] of the
  //          NVFP4 checkpoint (weights ONLY — KV and recurrent state unchanged;
  //          4-bit KV is deliberately not modelled). See research/nvfp4.md.
  // kv_decode_bpt/_const: sparse-attention decode pricing (GLM-5.2/DSA): bytes
  //          read per CONTEXT token / per ACTIVE SEQUENCE per step. Dense-
  //          attention models omit them (decode reads the full cache at kv_bpt).
  // kv_fp16_ok: false = vLLM can only serve this model with a quantized KV
  //          cache (GLM-5.3's DSA path) -> the FP16 toggle is disabled.
  // params_prefill/attn_layers/attn_d: prefill-ceiling constants (research/
  //          prefill.md). params_prefill = parameters doing a GEMM per token
  //          (dense: total minus embeddings+lm_head; MoE: ACTIVE params — a
  //          token routes to 8 experts however long the chunk is). attn_d x
  //          attn_layers price the quadratic QK^T/AV term of full-attention
  //          layers only (DeltaNet layers contribute nothing quadratic).
  MODELS: {
    "27B": {                       // Qwen3.8-27B since 2026-09-05; research/model_qwen38_27b.md
      // Every constant below was MEASURED or derived on Qwen3.6-27B. The
      // swap to Qwen3.8-27B keeps them all: the two config.json files agree
      // on all 34 text fields and the vision block, and the two official FP8
      // checkpoints hold the same 1,606 tensors with identical dtypes and
      // shapes (30,866,866,928 B each) — only the values differ. The one
      // measurement that is a property of the VALUES is mtp (draft
      // acceptance); it is carried over unmeasured, see below.
      name: "Qwen3.8-27B (dense)",
      kv_bpt: 32 * KIB,            // 16 attn x 4 KV heads x 256 x 2 x 1B (published config)
      deltanet_state: 75 * MIB,    // 48 DN layers x 48 vheads x 128x128 bf16 (+conv) = 75.7 MiB
      w_resident: 28.8 * GIB,      // baseline's stated as-deployed FP8 footprint
      w_decode_shared: 28.8 * GIB, // dense: every step reads all weights
      w_route_pertok: 0.0,
      w_route_total: 0.0,
      // MEASURED 2026-08-28 on the production Qwen3.6-27B deployment
      // (decode_mbu.md): accepted length 2.94 at num_speculative_tokens=2,
      // per-draft acceptance 0.971, and 1+a+a^2 confirmed against vLLM's
      // per-position counters. Was 1.7. PAIRED WITH DECODE_MBU -- see its
      // comment. NOT re-measured on Qwen3.8-27B: acceptance depends on the
      // draft head's weights, not the architecture. Re-measure when it ships.
      mtp: 2.94,
      // RedHatAI/Qwen3.8-27B-NVFP4 (2026-08-17), MEASURED from every shard
      // header (research/nvfp4_2026-09.md): 23,417,339,744 B. The recipe
      // quantizes only the dense MLPs of 56 of the 64 layers; attention,
      // DeltaNet and lm_head stay FP8, embed/MTP/vision BF16 — hence
      // +6.8% over NVIDIA's 3.6 recipe (21.92e9, which no longer applies).
      // Same whole-checkpoint convention as the FP8 arm: dense, every step
      // reads everything.
      nvfp4_w: [23417339744, 23417339744, 0.0, 0.0],
      params_prefill: 24.5e9,      // 27B dense less ~2.5e9 embed + lm_head (vocab 248,320)
      attn_layers: 16, attn_d: 24*256,
      max_ctx: 1048576,            // 262,144 native; 1M via YaRN (owner decision)
    },
    "35BA3B": {                    // published Qwen3.6-35B-A3B config; see research/model_35ba3b.md
      name: "Qwen3.6-35B-A3B (MoE, ~3B active)",
      kv_bpt: 10240,               // 10 full-attn layers x 2 KV heads x 256 x 2(K,V) x 1B
      deltanet_state: 33423360,    // 30 DN layers x 32 vheads x 128x128 bf16 (+conv) = 31.9 MiB
      w_resident: 35500000000,     // ~35.5B params x 1B FP8 (all experts + MTP module)
      w_decode_shared: 1940000000, // attn + deltanet + shared expert + router + lm_head / step
      w_route_pertok: 1006632960,  // 8 routed experts x 3.146M x 40 layers, FP8
      w_route_total: 32212254720,  // 256 routed experts (saturates at exactly n=32 linear)
      mtp: 1.7,                    // MTP module, speedup kept equal to baseline's fit
      // RedHatAI NVFP4 recipe: experts+attn NVFP4; DeltaNet/lm_head/router/MTP BF16
      // -> shared per-step read GROWS 1.7x while expert reads shrink 1.78x
      nvfp4_w: [24.13e9, 3.308e9, 566231040, 18119393280],
      params_prefill: 2.44e9,      // MoE: ledger active GEMM params (shared - lm_head + routed)
      attn_layers: 10, attn_d: 16*256,
      max_ctx: 1048576,            // 262,144 native; 1M via YaRN (owner decision)
    },
    "MM35": {                      // research/model_mistral_medium35.md
      name: "Mistral-Medium-3.5-128B (dense)",
      kv_bpt: 180224,              // 88 layers x 8 KV heads x 128 x 2(K,V) x 1B = 176 KiB
      deltanet_state: 0,           // pure GQA attention, no recurrent state
      w_resident: 133.6e9,         // 124.43 GiB as-shipped FP8 checkpoint
      w_decode_shared: 125.0e9,    // 88 FP8 layers + BF16 lm_head; vision tower not read
      w_route_pertok: 0.0,
      w_route_total: 0.0,
      mtp: 1.0,                    // NO MTP module (external EAGLE draft exists)
      nvfp4_w: [95.2e9, 86.6e9, 0.0, 0.0],  // nvidia mixed recipe, MEASURED safetensors total
      params_prefill: 121.8e9,     // TEXT decoder only (ledger); vision tower excluded
      attn_layers: 88, attn_d: 96*128,
      max_ctx: 262144,             // hard model max (YaRN x64 over a 4k base)
    },
    "GLM52": {                     // GLM-5.3 since 2026-09-06; research/model_glm53.md
      // Every constant below was derived for GLM-5.2 (research/model_glm52.md).
      // GLM-5.3 keeps them all: config.json agrees on 55 of 56 keys (the
      // 56th is transformers_version) and the two official FP8 checkpoints
      // hold the same 118,629 tensors with identical dtypes, shapes and
      // bytes (755,632,050,320 B). Only the values differ. Two things are
      // not carried by architecture and are flagged below: the MTP fit and
      // the NVFP4 bytes. License changed: MIT -> "GLM-5.3 License" (MIT plus
      // a security-review clause for >$10B-revenue Model-as-a-Service).
      name: "GLM-5.3 (MoE 744B-A40B, MLA+DSA)",
      kv_bpt: 48408,               // 79 x 576 MLA latent + 22 x 132 indexer keys (fp8)
      deltanet_state: 0,           // MLA is cached attention, no recurrent state
      w_resident: 755.5e9,         // official FP8 ckpt: 753.3e9 params + BF16 excess
      w_decode_shared: 18.92e9,    // MLA + indexers + dense MLP + shared exp + lm_head
      w_route_pertok: 22649241600, // 8 experts x (3x6144x2048) x 75 MoE layers, FP8
      w_route_total: 724775731200, // 256 experts (saturates at n=32)
      mtp: 1.7,                    // MTP module (5 drafts, same in the 5.3 recipe); transplanted fit, unmeasured
      // PROJECTION: nvidia/GLM-5.2-NVFP4 (ONLY routed experts NVFP4, ~465 GB
      // recipe) applied to tensor-identical weights. No NVIDIA NVFP4 of
      // GLM-5.3 exists (2026-09-06).
      nvfp4_w: [464.8e9, 35.30e9, 12740198400, 407686348800],
      kv_decode_bpt: 2772,         // DSA: 21 indexer layers x 132 B per context token
      kv_decode_const: 92.0e6,     // DSA: 78 layers x top-2048 x 576 B per active seq
      kv_decode_topk: 2048,        // ...scaled by min(len, topk)/topk per sequence
      kv_fp16_ok: false,           // vLLM DSA path asserts a quantized KV cache
      params_prefill: 37.4e9,      // MoE: ledger 39.3B active (excl embed) less lm_head 1.9e9
      attn_layers: 78, attn_d: 64*256,
      max_ctx: 1048576,            // native 1M context
    },
    "DSV4F": {                     // research/model_dsv4flash.md
      name: "DeepSeek-V4-Flash-0731 (MoE 284B-A13B, CSA)",
      kv_bpt: 3450,                // 21 x 576/4 CSA + 20 x 576/128 HCA + 21 x 64/4 fp4 indexer
      deltanet_state: 15597568,    // NOT DeltaNet: 46 x 128 x 576 windows + 12.2e6 fp32 compressor state
      state_fp32_ok: false,        // fixed mixed-precision state — the fp32 toggle models nothing
      w_resident: 166.88e9,        // measured safetensors total (native mixed FP8/FP4 checkpoint)
      w_decode_shared: 7.66e9,     // attn 4.60 + shared exp 1.08 + comp/idx/gates/mHC 0.92 + lm_head 1.06
      w_route_pertok: 3449290752,  // 6 experts x 13,369,344 B (FP4 packed + E8M0 scales) x 43
      w_route_total: 147169738752, // 256 experts (kink at n = 256/6 ~ 42.7 — non-integer)
      mtp: 1.7,                    // DSpark drafts 7 tokens; transplanted fit, unmeasured
      // nvidia/DeepSeek-V4-Flash-0731-NVFP4 (2026-08-19), MEASURED from every
      // shard header (research/nvfp4_2026-09.md). Only the 43 main layers'
      // routed experts change: the native checkpoint already packs them 4-bit
      // with E8M0 block-32 scales (13,369,344 B/expert), NVFP4 repacks them
      // with E4M3 block-16 scales (14,155,800 B/expert, +5.9%). Everything
      // else byte-identical, MTP experts stay native. So this arm is 5.2%
      // HEAVIER than FP8 — it exists for the NVFP4 kernel path, not for
      // bytes; the explorer prices what the checkpoint weighs.
      nvfp4_w: [175535844088, 7.66e9, 3652196400, 155827046400],
      kv_decode_bpt: 426,          // 21 x 64/4 fp4 indexer scan + 20 x 576/128 dense HCA read
      kv_decode_const: 9363456,    // 21 x 512 x 576 top-k reads + 43 x 128 x 576 windows
      kv_decode_topk: 2048,        // 512 compressed entries x ratio 4, in token space
      kv_fp16_ok: false,           // vLLM V4 path asserts fp8 main KV; SGLang bf16 decode unfinished
      params_prefill: 12.70e9,     // MoE: active GEMM params excl embed + lm_head
      attn_layers: 41, attn_d: 26624/41,  // 21 indexer layers @1024-equiv + 20 HCA @256-equiv
      max_ctx: 1048576,            // native 1M (YaRN x16 baked into the config)
    },
    "Q38FN": {                     // research/model_qwen38flashnext.md
      name: "Qwen3.8-Flash-Next (MoE 125B-A6B, QSA+n-gram)",
      kv_bpt: 12672,               // 12 attn x 2 KV x 256 x 2(K,V) x 1B + 12 x 128/4 indexer keys
      deltanet_state: 59572224,    // 36 DN layers x 48 vheads x 128x128 bf16 + conv (10,240 x 4)
      w_resident: 185502232570,    // FP8 ckpt metadata.total_size (incl. 51.2e9 FP8 n-gram table)
      w_decode_shared: 8623999000, // exact ledger sum — BF16 always-active: attn+DN+shared exp+routers+hyper-conns+PLE+lm_head
      w_route_pertok: 2359584000,  // 10 experts x 4,915,800 B (FP8 + block scales) x 48
      w_route_total: 120810700800, // 512 experts (kink at n = 512/10 = 51.2 — the study's deepest)
      mtp: 1.7,                    // MTP module (1 hybrid layer, 3 drafts per the vLLM recipe); transplanted fit, unmeasured
      // nvidia/Qwen3.8-Flash-Next-NVFP4 (2026-09-02), MEASURED from every
      // shard header (research/nvfp4_2026-09.md): only the 48 main layers'
      // routed experts are NVFP4 (2,764,824 B/expert vs 4,915,800 FP8, x0.5625
      // = 4 bits + 1/16 scales); the n-gram table, MTP experts, attention,
      // DeltaNet and heads are byte-identical, so w_decode_shared is the
      // FP8 ledger's. 132.64e9 B resident vs 185.50e9.
      nvfp4_w: [132639846394, 8623999000, 1327115520, 67948314624],
      kv_decode_bpt: 384,          // QSA indexer scan: 12 layers x 128 B / ratio 4 per ctx token
      kv_decode_const: 25165824,   // 12 layers x top-2048 x 1,024 B full-KV reads per active seq
      kv_decode_topk: 2048,        // indexer_budget, read in token space (research note #6)
      kv_fp16_ok: true,            // no documented fp8-KV assert on the QSA path (note #6)
      state_fp32_ok: true,         // bf16-priced (inferred) DeltaNet state — the fp32 knob applies
      params_prefill: 6.04e9,      // MoE: active GEMM params excl embed/lm_head/n-gram lookups
      attn_layers: 12, attn_d: 24*256,  // dense upper bound; QSA prefill sparsity uncharacterised
      max_ctx: 1048576,            // 262,144 native; 1M via YaRN (owner decision, as the Qwens)
    },
    "GLM53F": {                    // research/model_glm53flash.md
      name: "GLM-5.3-Flash (MoE 320B-A18B, KDA+NoPE-MLA)",
      kv_bpt: 6540,                // 12 x 512 nope-only MLA latent + 12 x 132/4 compressed indexer
                                   // (12 = 11 main + the MTP draft layer — GLM-5.2's convention:
                                   // storage incl. MTP, decode excl.)
      deltanet_state: 77987840,    // 34 KDA layers x 64 heads x 128x128 bf16 + q/k/v conv state
      w_resident: 328326771576,    // FP8 ckpt metadata.total_size, shard-header-verified
      w_decode_shared: 13957216504,// exact ledger sum — KDA 9.37 (BF16) + DSA 1.64 + shared exp
                                   // 1.06 + dense 0.45 + routers 0.10 + hc 0.07 + lm_head 1.27
      w_route_pertok: 8457781248,  // 8 experts x 25,171,968 B (FP8 + F32 scales) x 42
      w_route_total: 304480124928, // 288 experts (kink at n = 288/8 = 36)
      mtp: 1.7,                    // MTP draft layer (recipe runs 5 drafts); transplanted fit
      // RedHatAI/GLM-5.3-Flash-NVFP4 (2026-08-27; the vLLM recipe's named
      // NVFP4 checkpoint), MEASURED from every shard header (research/
      // nvfp4_2026-09.md). Routed experts of the 42 decode MoE layers NVFP4
      // (14,155,800 B/expert vs 25,171,968 FP8, x0.5625); the MTP draft layer
      // (index 45) keeps its FP8 experts. The recipe UPCASTS the rest to
      // BF16 — shared experts, o_proj, dense MLP, DSA projections — so the
      // always-active read grows 13.96e9 -> 16.71e9 (+2.75e9, the FP8->BF16
      // delta measured on the same tensors) while expert reads shrink 1.78x.
      // 197.82e9 B resident vs 328.33e9.
      nvfp4_w: [197822818044, 16705715836, 4756348800, 171228556800],
      kv_decode_bpt: 363,          // compressed indexer scan: 11 x 132 B / kpool 4 per ctx token
      kv_decode_const: 11534336,   // 11 layers x top-2048 x 512-B latent reads per active seq
      kv_decode_topk: 2048,        // index_topk, read in token space (research note #6)
      kv_fp16_ok: true,            // BF16 KV is REQUIRED on Hopper — fp8 KV is Blackwell-only
      kv_fp8_blackwell_only: true, // ...so modelFor auto-prices BF16 KV on H200 and the fp8
                                   // toggle locks (servableKv; mirrors check_dtype_supported)
      state_fp32_ok: true,         // bf16-priced (inferred) KDA state — the fp32 knob applies
      params_prefill: 16.11e9,     // MoE: active GEMM params excl embed/lm_head (card "18B" incl.)
      attn_layers: 11, attn_d: 64*256,  // dense upper bound; DSA prefill sparsity uncharacterised
      max_ctx: 1048576,            // native 1M context
    },
  },

  // ---- QUALITY ----
  // One coding-agent score per model, so the frontier ranks on what a
  // configuration delivers and not only on what it costs. Terminal-Bench
  // 2.1 pass@1 as MEASURED BY ARTIFICIAL ANALYSIS (one lab, one harness:
  // Terminus 2, 89 tasks x 3 repeats), read 2026-09-05 — vendor cards
  // quote other harnesses and other versions and are not comparable, so
  // they are deliberately not used. Reasoning variant at the effort AA ran
  // for its index. The FP16-KV / NVFP4 arms inherit the base score: the
  // quantisation loss is unmeasured, and a per-arm guess would rank the
  // frontier on the guess. Ledger and protocol: research/terminal_bench.md.
  QUALITY: {
    "27B":    { tb21: 213/267, aa: "qwen3-8-27b" },   // xhigh effort (the index run); 3.6 was 162/267
    "35BA3B": { tb21: 120/267, aa: "qwen3-6-35b-a3b" },
    "MM35":   { tb21: 135/267, aa: "mistral-medium-3-5" },
    "GLM52":  { tb21: 224/267, aa: "glm-5-3" },   // max effort (the index run); 5.2 was 208/267
    "DSV4F":  { tb21: 210/267, aa: "deepseek-v4-flash" },
    "Q38FN":  { tb21: 230/267, aa: "qwen3-8-flash-next" },
    "GLM53F": { tb21: 225/267, aa: "glm-5-3-flash" },
  },

  // ---- TOPOLOGY ----
  // A DP x TP grid: the unit DP replicates is a GROUP of `tp` GPUs, not a GPU.
  // n_gpu/kind/replicas are derived (dp*tp / label / dp) and spelled out here
  // only because these literals are built before the helpers exist.
  // `gpu` is filled in below (the GPUS literal is not in scope yet), so these
  // are structurally identical to makeGrid(dp, tp) — asserted in unitChecks.
  TOPOLOGIES: {
    "1xH200":     { name: "1×H200",            dp: 1, tp: 1, n_gpu: 1, kind: "single", replicas: 1 },
    "2xH200-TP2": { name: "2×H200 tensor-par", dp: 1, tp: 2, n_gpu: 2, kind: "tp",     replicas: 1 },
    "2xH200-DP2": { name: "2×H200 data-par",   dp: 2, tp: 1, n_gpu: 2, kind: "dp",     replicas: 2 },
  },

  // ---- MONTE-CARLO sizes (full quality) ----
  WARM_ITER: 700,
  // The decode x-axis is DYNAMIC: it must always reach the stress point this
  // study cares about — every GPU-resident warm session decoding at once — which
  // is ~94 seqs for the 27B on one H200 but ~680 for the 35B-A3B on TP2. NMIN
  // keeps small configs on a familiar axis; HEADROOM leaves room past warm p95.
  DECODE_ITER: 600, DECODE_NMIN: 120, DECODE_HEADROOM: 1.15, DECODE_SAMPLES: 60,
  // Sweep cost is O(iter x nMax) (see decodeCurves' shared draw), so iterations
  // are capped by a WORK budget rather than a flat floor: every realistic config
  // keeps the full 600, and only a pathologically wide axis tapers toward the
  // minimum. A fixed sample COUNT keeps the curve smooth at any range.
  DECODE_BUDGET: 1_400_000, DECODE_ITER_MIN: 40,
  // Warm-fill work budget (draws per warmCapacity call). A fill samples one
  // request per resident session, so pool size — not iteration count — sets the
  // cost; these caps keep the widest legal config interactive. SCAN is chart B's
  // 12-point survey, which is coarse by design.
  WARM_BUDGET: 600_000, WARM_BUDGET_SCAN: 120_000, WARM_ITER_MIN: 8,
  // draft (while dragging) — lighter for responsiveness
  DRAFT: { WARM_ITER: 220, DECODE_ITER: 200, DECODE_SAMPLES: 30,
           DECODE_BUDGET: 400_000, DECODE_ITER_MIN: 30,
           WARM_BUDGET: 200_000, WARM_BUDGET_SCAN: 35_000 },
};

// Backfill the GPU object onto the legacy topology literals — GPUS is not in
// scope inside the CONFIG object literal, and without this `topo.gpu` is
// undefined there and only survives via `|| GPUS["H200"]` fallbacks. Python's
// Topology carries the GPU directly; this makes the mirror structurally exact.
for (const t of Object.values(CONFIG.TOPOLOGIES)) t.gpu = CONFIG.GPUS["H200"];

/* ---- model helpers ---- */
export function is_moe(m){ return m.w_route_total > 0; }
// concurrency at which the conservative expert union saturates (every routed
// expert read each step): n_sat = total/pertok = E/k = 32 for both MoE models
export function unionKink(m){ return is_moe(m) ? Math.round(m.w_route_total / m.w_route_pertok) : null; }
// Conservative (no-overlap) expert-union bound, the study's planning default;
// the expected-union "coverage" model is the optimistic bracket (see docs).
export function w_decode(m, n){ return m.w_decode_shared + Math.min(n * m.w_route_pertok, m.w_route_total); }
// FP16-KV transform (mirrors Python's with_kv_dtype): doubles kv_bpt AND the
// top-k main-KV gathers (kv_decode_const) of a sparse-decode model; the
// quantized indexer scan (kv_decode_bpt) keeps its own width. Identity for
// fp8 and for models whose FP16 arm is not servable (kv_fp16_ok false).
// ONE implementation, called by modelFor and by the load-time unit checks.
export function withKvDtype(m, kv){
  // one-way door (mirrors with_kv_dtype): an fp16 arm must not be converted
  // again or "back" — a round-trip would double-double and un-stamp kv_dtype
  if (m.kv_dtype === "fp16")
    throw new Error("withKvDtype expects a base (fp8) model");
  if (kv !== "fp16" || m.kv_fp16_ok === false) return m;
  return { ...m, kv_bpt: m.kv_bpt*2, kv_decode_const: (m.kv_decode_const||0)*2,
           kv_dtype: "fp16", name: m.name + " [FP16 KV]" };
}
// The KV dtype actually SERVABLE for this model on this GPU (mirrors
// check_dtype_supported): GLM-5.3-Flash's fp8 KV cache is Blackwell-only —
// "Hopper ... must run BF16 KV" (vLLM recipe) — so on H200 the fp8 request
// resolves to fp16. Keeps the frontier table and tiles from ever pricing an
// unservable arm, whatever the global toggle says.
export function servableKv(m, kv, gpuKey){
  return (kv === "fp8" && m.kv_fp8_blackwell_only === true
          && !CONFIG.GPUS[gpuKey].supports_nvfp4) ? "fp16" : kv;
}

/* ---- derived: activation reserve solved from the anchor ---- */
function actReserve(){
  const m = CONFIG.MODELS["27B"];
  const poolBytes = CONFIG.BASELINE_POOL_TOKENS_27B_1GPU * m.kv_bpt;
  return CONFIG.VRAM_PER_GPU - m.w_resident - poolBytes;
}
// 17.98 GiB. Independently corroborated: the 2xH200 TP2 FP16 startup log
// implies 18.24 GiB per GPU (+1.4%), with nothing about TP2 or FP16 fitted.
// There is deliberately no second, "conservative" reserve any more — see
// activeModel() for why the low anchor was retired.
export const ACT_RESERVE = actReserve();

/* ---- study constants: the reference values every published figure is quoted
   at (mirrors scenario_model.py). They sit here rather than beside the prefill
   and decode code that uses them because the state object's DEFAULTS read
   them at load, and the module that holds them must not depend on anything
   that depends on state. ---- */
export const PREFILL_MFU   = 0.45;    // mid of the plausible [0.35, 0.55] FP8 bracket

                               // (tightened from [0.30, 0.60] 2026-08-27 on two
                               // production calibration points — prefill.md #1)
export const PREFILL_MFU_LO = 0.35, PREFILL_MFU_HI = 0.55;  // the bracket itself:
// an ERROR BAR on every prefill-derived figure, not a knob anyone chooses
export const DECODE_FLOOR_TOKS = 40;      // the study's hard per-user floor — now the
                                   // DEFAULT of state.decode_floor, kept as a
                                   // named constant because every published
                                   // figure is quoted at it
// the comfortable mark has always been 1.25x the hard floor (50 against 40);
// keeping the ratio means the chart-C guide lines and the tile's green/amber
// split still mean the same thing at any floor the slider is set to
export const DECODE_COMFORT_RATIO = 1.25;
// Output tokens ONE REQUEST decodes (every request, subagents included) —
// MEASURED 2026-08-27: ~400 mean over a 7-day production agentic trace
// (research/workload_agentic_poc.md; was assumed 1,000). At the observed
// 50-90 tok/s this decodes in ~5-8 s of the measured 10.8 s served per turn.
// The reference value; the `out` slider starts here and STATE_DEFAULTS pins it,
// so a default page still reproduces every published number. Drives the
// steady-state decode point below and the power model's decode duty.
export const AVG_OUT_TOK = 400;


// NVFP4 weights require native FP4 tensor cores (B300) — the UI greys the
// option out on the H200; this guard backstops any programmatic path.
function checkDtypeSupported(m, topo){
  const gpu = topo.gpu || CONFIG.GPUS["H200"];
  if (m.weight_dtype === "nvfp4" && !gpu.supports_nvfp4)
    throw new Error(`NVFP4 weights require native FP4; ${gpu.name} has none`);
  // GPU-coupled KV dtype (GLM-5.3-Flash): fp8 KV is Blackwell-only — the
  // BF16-KV arm (withKvDtype "fp16", which stamps kv_dtype) must be priced
  // on Hopper instead (mirrors Python's check_dtype_supported)
  if (m.kv_fp8_blackwell_only === true && m.kv_dtype !== "fp16"
      && !gpu.supports_nvfp4)
    throw new Error(`${m.name}: FP8 KV is Blackwell-only — price the BF16-KV arm on ${gpu.name}`);
}

// KV pool of ONE replica group. A group owns topo.tp GPUs: the weights are
// sharded across them and counted once, the reserve is scratch and charged per
// GPU -> pool = tp*(vram - reserve) - weights. Per-GROUP, i.e. per-cache; a
// dp>1 deployment has topo.replicas of them. A model too large for its group
// still clamps to 0 (the study's "does not fit" sentinel); minTpFor() reports
// the group size that would fit.
export function kv_pool_tokens(m, topo){
  checkDtypeSupported(m, topo);
  const gpu = topo.gpu || CONFIG.GPUS["H200"];
  const R = ACT_RESERVE + (gpu.reserve_extra || 0);
  // NOT factored as tp*(vram - R) - w: the three terms in this order make every
  // pre-existing configuration bit-identical to the pre-grid code (and to
  // Python's kv_pool_tokens), rather than merely equal after formatting.
  const poolBytes = topo.tp * gpu.vram - m.w_resident - topo.tp * R;
  return Math.max(poolBytes, 0) / m.kv_bpt;
}

// Smallest TP group size with a non-empty pool on this part. Shares the single
// ACT_RESERVE with the kv_pool_tokens sitting next to it, so the two always
// agree — and, now that the low anchor is gone, so does Python's min_tp_for
// (the reserve no longer depends on any UI state, so JS and Python cannot
// diverge here at all). Pass the model from activeModel() so a +15% weight
// bump is reflected in the threshold the Split control reports.
export function minTpFor(m, gpuKey){
  const gpu = CONFIG.GPUS[gpuKey || "H200"];
  return Math.floor(m.w_resident / (gpu.vram - (ACT_RESERVE + (gpu.reserve_extra || 0)))) + 1;
}

// topology for arbitrary N GPUs of a part (mirrors topology()/tp_efficiency()
// in scenario_model.py): TP = one engine, weights sharded, one shared cache,
// 0.90 bandwidth haircut per GPU-count doubling inside the node, then a steeper
// CROSS_DOMAIN_EFFICIENCY penalty per doubling once a TP group spans nodes.
export function tpEff(n, domain){
  // an explicit 0/negative domain is an ERROR, not a request for the default —
  // `domain || 8` would silently mask a non-NVLink part as an 8-GPU node while
  // Python's tp_efficiency raises. Only undefined/null falls back.
  const d = (domain === undefined || domain === null) ? 8 : domain;
  if (!(d >= 1) || Math.floor(d) !== d)
    throw new Error(`nvlink_domain must be a positive integer, got ${domain}`);
  if (n<=1) return 1.0;
  let eff = Math.pow(CONFIG.TP_EFFICIENCY, Math.log2(Math.min(n, d)));
  if (n > d) eff *= Math.pow(CONFIG.CROSS_DOMAIN_EFFICIENCY, Math.log2(n/d));
  return eff;
}
// DP x TP grid (mirrors topology_grid() in scenario_model.py). DP replicates
// whole GROUPS of `tp` GPUs — the only way the models that fit no single GPU
// (Mistral-Medium-3.5, GLM-5.3) can be data-parallel at all.
export function makeGrid(dp, tp, gpuKey){
  // mirror topology_grid()'s validation: a silently-accepted dp=0 would build a
  // zero-GPU topology whose pool still reads non-empty (the pool depends on tp)
  for (const [label, v] of [["dp", dp], ["tp", tp]])
    if (!(v >= 1) || Math.floor(v) !== v)
      throw new Error(`${label} must be a positive integer, got ${v}`);
  const key = gpuKey || "H200";
  if (!CONFIG.GPUS[key]) throw new Error(`unknown gpu ${key}`);
  const g = CONFIG.GPUS[key], n = dp*tp;
  // names match Python's topology_grid() exactly (× for x)
  const name = n===1 ? `1×${g.name}`
             : dp===1 ? `${tp}×${g.name} tensor-par`
             : tp===1 ? `${dp}×${g.name} data-par`
             : `${n}×${g.name} DP${dp}×TP${tp}`;
  const kind = n===1 ? "single" : dp===1 ? "tp" : tp===1 ? "dp" : "hybrid";
  return {name, dp, tp, n_gpu:n, kind, replicas:dp, gpu:g};
}
// single-axis helper for the explorer's TP/DP toggle: the two grid edges
export function makeTopo(kind, n, gpuKey){
  if (kind!=="tp" && kind!=="dp") throw new Error(`kind must be 'tp' or 'dp', got ${kind}`);
  return kind==="tp" ? makeGrid(1, n, gpuKey) : makeGrid(n, 1, gpuKey);
}
// divisors of n, ascending — the TP widths that evenly split n GPUs
export function divisors(n){ const d=[]; for (let i=1;i<=n;i++) if (n%i===0) d.push(i); return d; }
// Snap a TP width to a legal one for `ngpu`: the largest divisor <= tp, so
// shrinking the GPU count keeps as much TP as still divides evenly rather than
// resetting the user to pure TP or pure DP. Always returns a divisor of ngpu.
export function clampTp(tp, ngpu){
  const ds = divisors(ngpu);
  return ds.includes(tp) ? tp : (ds.filter(d => d <= tp).pop() ?? 1);
}
// Bandwidth seen by ONE replica group. DP adds groups, it never widens one.
export function effective_bw(topo){
  const gpu = topo.gpu || CONFIG.GPUS["H200"];
  return topo.tp * gpu.hbm_bw * tpEff(topo.tp, gpu.nvlink_domain);
}
