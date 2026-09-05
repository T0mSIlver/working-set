# DeepSeek-V4-Flash-0731 (284B-A13B, CSA/HCA compressed sparse attention) — parameterization note

**Purpose:** defensible KV-cache / decode-bandwidth constants for
**DeepSeek-V4-Flash-0731** (`deepseek-ai/DeepSeek-V4-Flash-0731`, MIT weights,
released 2026-07-31) as used by `scripts/scenario_model.py` and the explorer.

> **Provenance (2026-08-03):** all primary artifacts were read directly from
> huggingface.co (no proxy block this time): `config.json` (fetched twice,
> byte-identical), `inference/config.json`, `inference/model.py` (DeepSeek's
> reference implementation — authoritative for cache semantics),
> `model.safetensors.index.json`, the HF model API dtype histogram, and raw
> safetensors headers via HTTP range requests (exact dtypes/shapes). The
> reconstruction of `total_size` from the dtype histogram matches
> `metadata.total_size` **exactly** (§ 4), which validates the byte accounting.
> Two independent research passes (Claude Opus, GPT-5.6 Sol with web search)
> were run against the same source set and cross-checked; every config field,
> the layer-class split, the expert geometry and the NVFP4 situation agreed.
> The three disagreements (cached-entry bytes 576 vs 512, FP16-KV
> servability, VRAM floors) are resolved and flagged inline (§ 2, § 5, § 6).

## 1. Architecture table (config.json + inference/model.py)

| Field | Value |
|---|---|
| `architectures` / `model_type` | `DeepseekV4ForCausalLM` / `deepseek_v4` |
| `num_hidden_layers` | **43** (+3 DSpark/MTP layers `mtp.0–2`; `inference/config.json` `n_mtp_layers: 3` — the top-level `num_nextn_predict_layers: 1` is contradicted by the weight map, which carries 3 MTP stages) |
| `hidden_size` / `vocab_size` | 4096 / 129,280 |
| Attention | **MQA over one 512-dim latent** (single KV head, K = V = the latent) + per-layer-class compression: 2 pure sliding-window layers (0, 1), **21 CSA** layers (compress_ratio 4, top-512 sparse selection via an indexer), **20 HCA** layers (compress_ratio 128, dense over the compressed axis) |
| `head_dim` / `qk_rope_head_dim` / heads | **512** (448 nope + 64 rope) / 64 / 64 Q heads, `num_key_value_heads` **1** |
| `q_lora_rank` / `o_lora_rank` / `o_groups` | 1024 / 1024 / 8 (the output projection is low-rank and grouped) |
| Indexer (`index_topk` / `index_head_dim` / `index_n_heads`) | **512** (compressed entries = 2,048 original tokens) / 128 / 64 — on the 21 CSA layers only, with its own ratio-4 compressed **FP4** key cache |
| `sliding_window` | **128** — a per-layer ring buffer on all 43 (+3 MTP) layers; O(1), never grows |
| `n_routed_experts` / `num_experts_per_tok` / shared | **256 / 6 / 1** |
| `moe_intermediate_size` | 2048; **no dense FFN layers at all** (`first_k_dense_replace` absent; all 43 layers are MoE) |
| Hash routing | layers 0–2 route by a fixed 129,280×6 token-ID→expert lookup (`num_hash_layers: 3`) — zero gate FLOPs |
| `scoring_func` / `topk_method` | `sqrtsoftplus` / `noaux_tc` (`routed_scaling_factor` 1.5) |
| `max_position_embeddings` | **1,048,576** (native 1M: YaRN ×16 over `original_max_position_embeddings` 65,536; `compress_rope_theta` 160,000; the 2 SWA layers disable YaRN) |
| MTP / speculative | **DSpark** in-checkpoint draft (7 tokens, `dspark_target_layer_ids [40,41,42]`) or classic MTP mode (3 tokens) |
| Total / active params | **284B / 13B active** (paper, arXiv 2606.19348); 304B on disk incl. the ~20B DSpark stages — derived § 4 |
| Checkpoint dtype | native mixed: routed experts **FP4** (I8-packed + E8M0 scales), attention/shared-expert **FP8** (block 128×128, ue8m0), gates/norms/compressors/embed/lm_head BF16, mHC/sinks FP32 |

Param arithmetic (config-derived; reconciles the published counts): routed
experts 46 × 256 × 3 × 4096 × 2048 = 296.35e9 logical params — exactly the HF
I8 histogram entry ✓; main 43 layers + embed + head = 284.3e9 ✓ "284B"; active
12.703e9 excl. embed/head (13.76e9 incl.) ✓ "13B".

## 2. KV bytes per token — compressed, sub-linear-then-tiny

**This model breaks the study's usual "cache grows by kv_bpt per token"
assumption twice**, and the constants below split the true cost into the
study's two existing fields:

1. A **fixed per-session part** (`deltanet_state`): every layer's 128-entry
   sliding-window ring buffer plus the reference implementation's FP32
   compressor state — allocated once per sequence, never growing.
2. A **per-token part** (`kv_bpt`): only the *compressed* caches grow — one
   576-B latent entry per `compress_ratio` tokens, plus the indexer's FP4 keys.

Cached-entry layout (from `inference/model.py`): 512-elem latent = 448 nope
dims FP8-quantized + 64 rope dims kept BF16 "for positional precision" →
**576 B/entry**; indexer entries 128 dims at FP4 (`fp4_act_quant` after a
Hadamard rotation) → **64 B/entry**. The vLLM README command serves exactly
this scheme (`--kv-cache-dtype fp8`, `--attention-config
'{"use_fp4_indexer_cache": true}'`).

```
kv_bpt (fp8 latent + fp4 indexer, per ORIGINAL token):
  CSA latent    21 layers x 576 B / 4    = 3,024.0
  HCA latent    20 layers x 576 B / 128  =    90.0
  indexer keys  21 layers x  64 B / 4    =   336.0
  --------------------------------------------------
  TOTAL                                  = 3,450 B/token   (3.37 KiB)

deltanet_state (fixed per session, reused field — not DeltaNet here):
  windows       46 layers x 128 x 576 B  =  3,391,488   (43 main + 3 MTP)
  fp32 compressor state (model.py:309)   = 12,206,080
  --------------------------------------------------
  TOTAL                                  = 15,597,568 B ≈ 14.9 MiB/session
```

A 262k-token session holds **0.84 GiB** (vs GLM-5.2's 11.8 GiB, MM35's
44 GiB); the full 1M context holds 3.37 GiB (comparison: DeepSeek-V3-class
MLA at 35.1 KiB/token would hold 33.5 GiB — the compression is ~10×, matching
the paper's "10% of V3.2's KV" claim). Quantization-scale bytes on the cached
latents (ue8m0, block 64: ~7 B/entry) are **not** modelled: +~1.2% on kv_bpt,
noted in § 6.

(Alternative all-FP8 layout: if serving stores the full 512-dim entry at
1 B/elem — the vLLM blog's description — entries are 512 B and kv_bpt falls
to 3,104 (−10%). The reference implementation's mixed 448-FP8 + 64-BF16
layout is modelled because it is the literal `model.py` behaviour; the
alternative is the same treatment the GLM-5.2 note gives `fp8_ds_mla`.)

**FP16-KV toggle disabled** (`kv_fp16_ok=False`): vLLM's V4 path asserts a
quantized (FP8) main KV cache (vllm#42876) and SGLang's roadmap lists BF16
V4 KV-decode as unfinished — BF16 KV is not a servable configuration today.
The vLLM *recipe* phrases fp8 KV as "recommended", but the launch commands
all pass `--kv-cache-dtype fp8`; the assertion is trusted over the phrasing
(discrepancy logged in § 6).

## 3. Decode-bandwidth model — three cache geometries, none read in full

Per decode step, a query reads: (a) the indexer's FP4 keys over the whole
*compressed* axis on CSA layers (the scan), (b) the top-512 selected latent
entries on CSA layers plus the dense compressed axis on HCA layers, and
(c) every layer's 128-entry window.

```
kv_decode_bpt   = 21 x 64/4  +  20 x 576/128 = 426 B per CONTEXT token per step
                  (indexer scan 336 + dense-HCA compressed read 90)
kv_decode_const = 21 x 512 x 576   (CSA top-512 latent reads)  =  6,193,152
                + 43 x 128 x 576   (window reads, main layers)  =  3,170,304
                                                               =  9.36e6 B per ACTIVE SEQ per step
kv_decode_topk  = 2,048 original tokens (512 compressed entries x ratio 4);
                  sequences shorter than 2,048 tokens scale the constant by
                  min(len, 2048)/2048 (study convention, research/model_glm52.md #3)
```

At the reference 31k-median workload the context scan is ~13 MB/seq and the
constant reads 9.4 MB/seq — versus 107 MB/seq if decode streamed the full
cache at kv_bpt, and ~1.1 GB/seq for a dense-attention model with V3-class
MLA. The MTP layers' window reads and the bursty compressor writes (one
compression every 4/128 tokens) are folded into the MTP speedup / ignored,
as MTP-module costs are everywhere in this study.

## 4. Weight bytes

### Native checkpoint (`deepseek-ai/DeepSeek-V4-Flash-0731`) — the study's "FP8" arm

The base checkpoint is already mixed FP8/FP4 (`quant_method: fp8`, block
128×128, ue8m0 scales; experts stored as I8-packed FP4 pairs + E8M0 block
scales — confirmed from safetensors headers). Byte reconstruction from the HF
dtype histogram matches `metadata.total_size` **exactly**:

```
BF16 1,483,567,488 x2 + I64 2,327,040 x8 + F32 37,741,630 x4
  + F8_E4M3 6,304,038,912 x1 + FP4-packed 148,176,371,712 x1
  + E8M0 expert scales 9,261,408,000 x1
= 166,878,536,440 B  =  w_resident  (155.4 GiB)
```

Per-expert bytes (3 × 4096×2048 packed to 12,582,912 B + 786,432 B scales =
**13,369,344 B**; the 6.25% scale overhead is real and charged):

```
w_route_pertok  = 6   x 13,369,344 x 43 =   3,449,290,752 B
w_route_total   = 256 x 13,369,344 x 43 = 147,169,738,752 B
```

Expert-union saturation at n = 256/6 ≈ **42.7** — the first non-integer kink
in the study (both other MoEs sit at exactly 32).

Shared per-step read — derived from the active-param ledger (12.703e9 active
excl. embed/head, minus routed 6.493e9 = 6.210e9 shared params) with the
measured per-tensor dtypes:

```
attn (43 x 106,954,752, FP8)                     = 4,599.1e6
shared experts (43 x 25,165,824, FP8)            = 1,082.1e6
compressors + indexers + gates + mHC (529.1e6
  params: FP8 indexer wq_b 176.2e6 x1, FP32 hc
  ~17e6 x4, BF16 remainder ~335.9e6 x2)          =   915.6e6
lm_head (129,280 x 4096, BF16)                   = 1,059.1e6
-----------------------------------------------------------------
w_decode_shared                                  = 7.66e9 B
```

Active check (params, not bytes): 6.21e9 shared params excl. lm_head +
6.49e9 routed-expert params/token = 12.70e9 active params/step ✓ "A13B"
(= `params_prefill`). The 916e6 mixed-dtype line carries the note's largest
uncertainty (±0.2e9, ~±3% of the shared read — the exact dtype split of the
529e6 non-attention shared params was sampled, not exhaustively summed).

### NVFP4 — **deliberately not modelled** (`nvfp4_w = None`)

> **Superseded 2026-09-06:** `nvidia/DeepSeek-V4-Flash-0731-NVFP4` (2026-08-19) is
> priced, and it is 5.2% heavier than the native checkpoint (E8M0 block-32 →
> E4M3 block-16 scales). `research/nvfp4_2026-09.md`.

There is **no official NVFP4 checkpoint of the 0731 release**:
`nvidia/DeepSeek-V4-Flash-NVFP4` (created 2026-05-18) targets the April base
model, and the only 0731 conversion is community-made
(`MJPansa/DeepSeek-V4-Flash-0731-NVFP4`). More fundamentally, the base
checkpoint's routed experts are **already FP4** — the community NVFP4
checkpoint is *larger* than the original (175.5e9 B vs 166.9e9 B: NVFP4's
group-16 scales cost more than the native E8M0 block scales). NVFP4 has
nothing to offer this model; the explorer greys the option out.

Note the native-FP4 experts do **not** restrict the model to B300-class GPUs
the way NVFP4 checkpoints do: the official vLLM recipe serves it on 8×H200
(`--moe-backend deep_gemm_mega_moe`), so both study GPUs run the native
checkpoint.

## 5. Serving notes

- **vLLM ≥ 0.25.0** for the 0731/DSpark checkpoint on NVIDIA (≥ 0.26.0 on
  ROCm; `--trust-remote-code`, `--tokenizer-mode deepseek_v4`,
  `--block-size 256`). SGLang ≥ 0.5.12 for the base V4 path (the 0731-specific
  minimum is unstated). FP8 main KV required in practice (§ 2); FP4 indexer
  cache behind `--attention-config`.
- Official recipe VRAM floors: **0731 fused checkpoint 200 GB** · preview
  mixed FP4+FP8 170 GB · NVIDIA NVFP4 (preview) 170 GB. Recipe hardware:
  single DGX Station, 1×MI325X, 4×MI355X, 8×H200 (TP4 + DP, disaggregated
  prefill/decode), GB200 NVL4, B300. The 200 GB floor agrees with the pool
  arithmetic's fit boundary: min TP2 on H200 (2×141 GB), a single B300
  (288 GB).
- **Speculative decoding: DSpark** (7 draft tokens from in-checkpoint draft
  stages; a 3-token classic MTP mode also exists). The study keeps its 1.7×
  **transplanted** fit (same treatment as GLM-5.2: module present, acceptance
  unmeasured on this workload; a 7-draft greedy scheme could be well above —
  the slider covers it).
- Positioning: "Flash" is the fast sibling of DeepSeek-V4-Pro (1.6T-A49B);
  the 0731 refresh (released 2026-07-31, MIT) is the agentic/tool-use update
  (Terminal Bench 2.1: 82.7 vs GLM-5.2's 81.0; DeepSWE 54.4 vs the preview's
  7.3). API (`deepseek-v4-flash` = this checkpoint, official pricing page):
  $0.14/M input (miss), $0.0028/M cached, $0.28/M output; 1M context, 384K
  max output.

## 6. Remaining assumptions / re-verification ledger

- Latent/indexer quantization-scale bytes (~+1.2% kv_bpt) not charged.
- Cached-entry bytes: the reference implementation's mixed layout (576 B)
  modelled; an all-FP8 512-B layout (vLLM blog wording) would be −10% on
  kv_bpt and proportionally on the window/decode constants (§ 2).
- FP16-KV: modelled as unservable on the strength of the vLLM assertion
  (vllm#42876) + SGLang roadmap; the vLLM recipe's "recommended" phrasing
  disagrees. If BF16 KV ships, kv_fp16_ok flips to True and ×2 overstates
  the true BF16 layout by only 0.3%.
- The FP32 compressor state (11.65 MiB/seq) is charged per *resident* session;
  a serving stack could plausibly keep it only for *active* sequences. Biases
  capacity DOWN (conservative).
- The per-session fixed cost reuses the `deltanet_state` field; the fp32-state
  toggle is disabled for this model (`state_fp32_ok=False`) — the buffers are
  already fp32/fp8-mixed, so doubling them models nothing.
- `kv_decode_topk` scaling maps the top-512-of-compressed selection onto the
  study's token-space rule; sequences between 128 and 2,048 tokens slightly
  under-charge the window read (< 3 MB/seq).
- Prefill quadratic term (§ scenario_model.py constants): the indexer scores
  the full compressed axis (equivalent attn_d 1024/layer × 21) and HCA attends
  it densely (256/layer × 20) → `attn_layers=41, attn_d=26,624/41`. The CSA
  top-512 and window attention are *linear* per token and left out of both the
  quadratic term and `params_prefill` — prefill priced cheaper, i.e. biased
  AGAINST the thrash hypothesis, per research/prefill.md convention.
- `w_decode_shared`'s mixed-dtype residual (±0.2e9 B) — see § 4.
- MTP-stage weights (~20B on disk) charged in w_resident, never in per-step
  reads (folded into the MTP speedup, as everywhere in this study).
- `num_nextn_predict_layers` 1-vs-3 config discrepancy unresolved upstream;
  the weight map's 3 stages are trusted. No effect on any constant (MTP reads
  are not priced).

## Sources

Primary (read directly, exact bytes):
- https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/raw/main/config.json
  (fetched twice, identical) ·
  https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/raw/main/inference/config.json
- https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/raw/main/inference/model.py
  — cache semantics: `kv_cache_size = window + max_seq_len/ratio` (l.479),
  ring-buffer window (l.535), fp8 nope + bf16 rope split (l.511), FP4 indexer
  cache (l.374, 420), fp32 compressor buffers (l.309)
- https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/raw/main/model.safetensors.index.json
  (`total_size` 166,878,536,440) + safetensors shard headers via HTTP range
  requests (dtypes/shapes) + https://huggingface.co/api/models/deepseek-ai/DeepSeek-V4-Flash-0731
- https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/raw/main/README.md
  (serve command, DSpark config, benchmarks, MIT license)
- NVFP4 variants: https://huggingface.co/nvidia/DeepSeek-V4-Flash-NVFP4 (April
  base; ignore list `["*.attn.*", "*.ffn.shared_experts.*", "head", "mtp.*"]`) ·
  https://huggingface.co/MJPansa/DeepSeek-V4-Flash-0731-NVFP4 (community,
  175,535,844,088 B)
- Siblings: deepseek-ai/DeepSeek-V4-Flash (158.07e9 B) ·
  DeepSeek-V4-Flash-DSpark (165.27e9 B) · DeepSeek-V4-Pro (1,598.8e9 B)

Secondary:
- DeepSeek-V4 technical report (official 284B/13B, 1.6T/49B):
  https://arxiv.org/abs/2606.19348 · release announcement:
  https://api-docs.deepseek.com/news/news260424/ · official pricing:
  https://api-docs.deepseek.com/quick_start/pricing/
- vLLM recipe (min versions, VRAM floors, DSpark 7 drafts):
  https://github.com/vllm-project/recipes/blob/main/models/deepseek-ai/DeepSeek-V4-Flash.yaml ·
  vLLM V4 implementation post (five-way hybrid cache, page buckets):
  https://github.com/vllm-project/vllm-project.github.io/blob/main/_posts/2026-04-24-deepseek-v4.md ·
  fp8-KV assertion: https://github.com/vllm-project/vllm/issues/42876
- transformers deepseek_v4 doc (shared-K=V MQA, hash-MoE, layer classes):
  https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/deepseek_v4.md
- LMCache (multi-group cache geometry):
  https://docs.lmcache.ai/recipes/deepseek_v4_flash.html ·
  SGLang cookbook (c4/c128 fp32 state pools, fp4-indexer flag):
  https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4 ·
  SGLang bf16-KV-decode roadmap gap: https://github.com/sgl-project/sglang/issues/23602
