# Qwen3.8-Flash-Next (125B-A6B, QSA + n-gram embeddings) — parameterization note

**Purpose:** defensible KV-cache / decode-bandwidth constants for
**Qwen3.8-Flash-Next** (`Qwen/Qwen3.8-Flash-Next`, Apache-2.0 weights,
released 2026-08) as used by `scripts/scenario_model.py` and the explorer.

> **Provenance (2026-08-26):** all primary artifacts were read directly from
> huggingface.co: `config.json` of the base repo, the **FP8 serving
> checkpoint's** `model.safetensors.index.json`, and the raw safetensors
> headers of **all 131 FP8 shards via HTTP range requests** (exact
> dtypes/shapes for every one of the 152,089 tensors). The per-module byte
> sums reconstructed from those headers total **185,502,232,570 B — equal to
> the index's `metadata.total_size` exactly**, which validates the byte
> accounting end-to-end. One research pass; the cross-check this time is the
> merge-request review (three independent reviewers against this note).

## 1. Architecture table (config.json, FP8 shard headers)

| Field | Value |
|---|---|
| `architectures` / `model_type` | `Qwen4ExpForConditionalGeneration` / `qwen4_exp` (multimodal wrapper; text config `qwen4_exp_text`) |
| `num_hidden_layers` | **48** = 36 `linear_attention` (Gated DeltaNet) + 12 `full_attention` (`full_attention_interval: 4`) |
| `hidden_size` / `vocab_size` | 2,560 / 248,320 (the Qwen3.6-family figure; embed and lm_head untied) |
| Full attention (QSA) | GQA **24 Q heads / 2 KV heads × head_dim 256**, partial rotary 0.25 (rope dim 64, mrope-interleaved) + a **sparse indexer**: 4 heads × 128 (q) over a ratio-4 **compressed** key axis (1 kv head × 128), selection budget **2,048** |
| Gated DeltaNet | 16 QK heads × 128, **48 V heads × 128**, conv kernel 4 (conv dim 10,240) |
| MoE | **512 routed experts / 10 per token + 1 shared**, `moe_intermediate_size` 640 — every one of the 48 layers is MoE |
| Hyper-connections | gated residual, `hc_count` 4 branches × low-rank 320, on both the attn and mlp sub-blocks of every layer (+ a top-level mixer) |
| n-gram embeddings (PLE) | layer 2 only: 20M-base bigram/trigram table (`ngram_size` 3, 8 heads/ngram, 128 shards) = **51.2e9 params**, FP8 in the serving checkpoint |
| MTP | in-checkpoint module, **1 hybrid full-attention layer** + its own 512-expert MoE (~2.6e9 params on disk) |
| `max_position_embeddings` | **262,144** native; 1M via YaRN rope-scaling per the model card |
| Total / active params | **125B with 6B activated** (card) **+ 51B n-gram embedding + 4B MTP**; the FP8 checkpoint's param total is 180.0e9 (HF API histogram) |
| Vision tower | 27-block ViT, ~0.45e9 params — an encoder, never executed on these text workloads |
| Checkpoint dtype (FP8 repo) | routed experts + n-gram table **FP8** (fine-grained block 128); **everything else BF16** (attention, DeltaNet, shared experts, routers, hyper-connections, embed, lm_head, MTP) |

Param arithmetic (shard-header sums; reconciles the published counts): routed
experts 48 × 512 × 3 × 2560 × 640 = 120.80e9; active per step excl. embed and
lm_head = attn 0.617 + DeltaNet 2.087 + shared expert 0.236 + router 0.063 +
hyper-conn 0.641 + PLE projections 0.033 + routed 10-expert read 2.359 =
**6.04e9 ✓ "6B activated"**. 125B total = 180.0e9 on disk − 51.2e9 n-gram −
2.6e9 MTP − 0.45e9 vision − (FP8-vs-param rounding) ✓.

## 2. KV bytes per token — 12 QSA layers + a compressed indexer axis

Only the 12 full-attention layers cache K/V; the 36 DeltaNet layers hold a
fixed recurrent state (§ 3). The QSA indexer additionally caches its selection
keys on a **ratio-4 compressed axis** (`indexer_compress_ratio: 4`, 1 kv head
× 128 dims), the same construction as DSv4-Flash's CSA indexer:

```
kv_bpt (fp8 KV + fp8 compressed indexer keys, per token):
  full-attn KV   12 layers x 2 KV heads x 256 x 2 (K,V) x 1 B = 12,288
  indexer keys   12 layers x 128 B / 4                         =    384
  ------------------------------------------------------------------------
  TOTAL                                                        = 12,672 B/token (12.4 KiB)

deltanet_state (fixed per session, bf16 recurrent state):
  SSM state  36 layers x 48 vheads x 128 x 128 x 2 B = 56,623,104
  conv state 36 layers x 10,240 x kernel 4 x 2 B     =  2,949,120
  ------------------------------------------------------------------------
  TOTAL                                              = 59,572,224 B ≈ 56.8 MiB/session
```

The conv-dim arithmetic (10,240 = 2×16×128 QK + 48×128 V) follows the exact
convention that reproduces the 35B-A3B's 33,423,360 B and the 27B baseline's
measured 75 MiB. The state is a genuine bf16 DeltaNet state, so the fp32-state
toggle stays enabled (`state_fp32_ok=True`) — doubling is exactly what fp32
would do.

A 262k-token session holds 3.09 GiB — between the 35B-A3B (2.5 GiB) and the
27B (8 GiB), ~15× below GLM-5.2. **FP16-KV toggle enabled**
(`kv_fp16_ok=True`): no vLLM assertion requiring a quantized KV cache on the
QSA path was found (unlike GLM-5.2's DSA and DSv4-Flash's V4 paths, which
carry documented asserts); the base checkpoint itself ships BF16. Flagged in
§ 6 — if vLLM's QSA path turns out to assert fp8 KV, flip the flag.

## 3. Decode-bandwidth model — QSA sparse reads

Per decode step a query (a) scans the indexer's compressed fp8 keys over the
whole context, then (b) reads full K/V for only the top-`indexer_budget`
selected tokens; (c) the DeltaNet layers read their fixed state (charged
separately by the study's decode model):

```
kv_decode_bpt   = 12 x 128/4                       =    384 B per CONTEXT token per step
kv_decode_const = 12 x 2,048 x (2 x 256 x 2 x 1 B) = 25,165,824 B per ACTIVE SEQ per step
kv_decode_topk  = 2,048 tokens (indexer_budget); sequences shorter than the
                  budget scale the constant by min(len, 2048)/2048
                  (study convention, research/model_glm52.md #3)
```

At the reference 31k-median workload the scan is ~12 MB/seq and the top-k
read 25 MB/seq — versus ~397 MB/seq if decode streamed the full cache at
`kv_bpt`. The n-gram embedding lookups (a handful of rows per token, ~5 KB)
are excluded like all embedding lookups in this study.

## 4. Weight bytes

### FP8 checkpoint (`Qwen/Qwen3.8-Flash-Next-FP8`) — the study's FP8 arm

Reconstructed per-module bytes from all 131 shard headers; the sum equals
`metadata.total_size` **exactly**:

```
routed experts  48 x 512 x 3 x (640x2560 FP8 + 200 B BF16 block scales)
                                                 = 120,810,700,800
n-gram table    51.2e9 params, FP8 (+1 B scale)  =  51,200,245,762
DeltaNet        36 layers, BF16                  =   4,173,020,928
MTP module      1 layer + 512 experts, mixed     =   2,698,026,496
hyper-conns     48 x 2 + mixer, BF16             =   1,281,249,280
embed_tokens    248,320 x 2,560 BF16             =   1,271,398,400
lm_head         248,320 x 2,560 BF16             =   1,271,398,400
attention (QSA) 12 layers incl. indexer, BF16    =   1,234,716,672
vision tower    27 ViT blocks, BF16              =     897,862,112
shared experts  48 layers, BF16                  =     472,104,960
routers         48 x 512x2560 BF16               =     125,829,120
PLE projections layer 2 key/value/conv, BF16     =      65,679,640
--------------------------------------------------------------------
w_resident      (= metadata.total_size)          = 185,502,232,570 B  (172.8 GiB)
```

Per-expert bytes: 3 × (1,638,400 FP8 + 200 BF16 scale) = **4,915,800 B**
(the 0.012% scale overhead is real and charged):

```
w_route_pertok  = 10  x 4,915,800 x 48 =   2,359,584,000 B
w_route_total   = 512 x 4,915,800 x 48 = 120,810,700,800 B
```

Expert-union saturation at n = 512/10 = **51.2** — the deepest kink in the
study (35B-A3B and GLM-5.2 sit at 32, DSv4-Flash at 42.7): with 10-of-512
routing, per-token expert reads keep growing to ~51 concurrent decoders
before the union saturates.

Shared per-step read — everything always-active is **BF16 in this "FP8"
checkpoint** (only experts and the n-gram table are quantized), so the fixed
read weighs 2 B/param:

```
attention 1.235 + DeltaNet 4.173 + shared experts 0.472 + routers 0.126
  + hyper-conns 1.281 + PLE projections 0.066 + lm_head 1.271
w_decode_shared = 8.624e9 B
```

Exclusions from the per-step read: embed_tokens and the n-gram table
(lookups), the vision tower (encoder, never run), the MTP module (folded into
the MTP speedup, as everywhere in this study). Resident-vs-touchable check:
185.5e9 resident ≫ 120.8 + 8.6e9 touchable ✓ (the n-gram table, embed,
vision and MTP make the difference).

### NVFP4 — **not modelled** (`nvfp4_w = None`)

No official NVFP4 checkpoint exists (`nvidia/Qwen3.8-Flash-Next-NVFP4` is
absent; RadixArk/Inferact/community conversions only, 2026-08-26). The upside
would also be modest: the two dominant blocks (experts 120.8e9, n-gram table
51.2e9) are already FP8, so NVFP4 could at best shave ~44% off those while
the BF16 shared read stays. Same treatment as DSv4-Flash: the explorer greys
the option out. Revisit if NVIDIA ships an official recipe.

## 5. Serving notes

- **Frameworks:** vLLM, SGLang, TokenSpeed per the model card (`vllm serve
  Qwen/Qwen3.8-Flash-Next`); no published flag-level recipe yet. Contexts
  past 262,144 need YaRN `rope_parameters` + `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`.
  The card publishes no VRAM floor; the pool arithmetic gives **min TP 2 on
  H200** (2×141 GB) and a **single B300** (288 GB) for the 172.8-GiB FP8
  checkpoint.
- **n-gram table residency:** charged fully GPU-resident (it is part of the
  serving checkpoint and its rows are latency-critical lookups at layer 2).
  If a serving stack offloads it to host RAM, `w_resident` drops by 51.2e9 B
  (≈ 28%) and the H200 min-TP story does not change (134.3e9 still exceeds
  one GPU's 121.7e9 budget); a B300 gains ~6.4M pool tokens. Conservative
  direction: resident (biases capacity DOWN). § 6.
- **Speculative decoding:** in-checkpoint MTP (1 hybrid layer, "trained with
  multi-steps"; draft count unpublished). The study keeps its 1.7×
  **transplanted** fit and the family's MTP-2 (2-draft) recipe line — same
  treatment as GLM-5.2/DSv4-Flash: module present, acceptance unmeasured on
  this workload; the slider covers the range.
- **Thinking mode** is on by default (`enable_thinking: False` to disable) —
  affects output length, not any constant here.
- Positioning: the "Flash" sibling of the Qwen3.8 family — the n-gram
  parameter-scaling experiment (51B of lookup parameters that never enter a
  GEMM) on top of the Qwen3.6-style DeltaNet hybrid, with the family's first
  sparse full-attention layers (QSA).

## 6. Remaining assumptions / re-verification ledger

- **Indexer cache layout** (`kv_bpt`'s 384 B/token line and the decode scan):
  modelled as fp8 keys on a ratio-4 compressed axis — 1 entry of 128 B per 4
  tokens per QSA layer — by direct analogy with DSv4-Flash's CSA indexer
  (`indexer_compress_ratio: 4` in this config). An uncompressed indexer cache
  would make it 1,536 B/token stored and 4× the scan. Both are < 12% of
  `kv_bpt`; capacity moves < 9%.
- **`indexer_budget` = 2,048 read in TOKEN space** (like GLM-5.2's top-2048).
  If the budget counts *compressed* entries (DSv4-Flash's convention:
  top-512 compressed = 2,048 tokens), the top-k read is 4× bigger
  (100.7 MB/seq) — decode slows at long context but stays far below the
  397 MB/seq dense read. Optimistic-side choice; flagged.
- **FP16 KV modelled as servable** (`kv_fp16_ok=True`): no documented vLLM
  assert on the QSA path was found (the GLM/DSv4 flags each rest on a cited
  issue). If one exists, flip to False; nothing else changes.
- **n-gram table charged GPU-resident** (§ 5); its per-token lookup rows
  (~5 KB) excluded from the decode read like all embedding lookups.
- **MTP draft count assumed 2** (family convention; unpublished for this
  model) — affects only the recipe line and the α readout, not any constant.
- Prefill quadratic term priced as **dense** attention on the 12 QSA layers
  (`attn_layers=12, attn_d=24×256`): an upper bound — QSA's prefill-side
  sparsity is uncharacterised, same flagged treatment as GLM-5.2's MLA.
  `params_prefill=6.04e9` = the active-param ledger of § 1 (embed, lm_head
  and n-gram lookups excluded; a cheap-side bias, per research/prefill.md).
- `w_decode_shared` (8.624e9) sums exact per-tensor bytes; no residual term —
  the FP8 checkpoint's dtype split is fully enumerated, unlike DSv4-Flash's
  sampled ±0.2e9.
- MTP-module weights (2.7e9 B) and vision tower (0.9e9 B) charged in
  `w_resident`, never in per-step reads.
- The base (BF16) repo would roughly double every weight constant; the study
  models the FP8 serving checkpoint only, as for every other model.

## Sources

Primary (read directly, exact bytes):
- https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/main/config.json
  (layer types, QSA/indexer/DeltaNet/MoE/PLE/MTP geometry)
- https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/resolve/main/model.safetensors.index.json
  (`total_size` 185,502,232,570; 152,089-tensor weight map) + **all 131
  safetensors shard headers via HTTP range requests** (per-tensor
  dtypes/shapes; module sums reproduce total_size exactly)
- https://huggingface.co/api/models/Qwen/Qwen3.8-Flash-Next-FP8 (dtype
  histogram: BF16 5.487e9 / F8_E4M3 174.513e9 params — matches the
  experts + n-gram + MTP-expert split)
- Model cards: https://huggingface.co/Qwen/Qwen3.8-Flash-Next ·
  https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8 (125B-A6B + 51B n-gram
  + 4B MTP; layer layout; YaRN-to-1M; fine-grained fp8 block 128; vLLM/
  SGLang/TokenSpeed support)
- NVFP4 absence: https://huggingface.co/api/models?search=Qwen3.8-Flash-Next
  (community conversions only, no `nvidia/` or `Qwen/` NVFP4 repo, 2026-08-26)

Secondary:
- research/model_dsv4flash.md (compressed-indexer precedent, top-k scaling
  convention) · research/model_35ba3b.md (DeltaNet state + conv arithmetic
  convention) · research/model_glm52.md (sparse-decode pricing convention)
