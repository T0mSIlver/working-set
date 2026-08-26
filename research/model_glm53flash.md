# GLM-5.3-Flash (320B-A18B, KDA + NoPE sparse MLA) — parameterization note

**Purpose:** defensible KV-cache / decode-bandwidth constants for
**GLM-5.3-Flash** (`zai-org/GLM-5.3-Flash`, MIT weights, released 2026-08-25)
as used by `scripts/scenario_model.py` and the explorer.

> **Provenance (2026-08-26):** all primary artifacts read directly from
> huggingface.co: `config.json`, the model cards (main + BF16, byte-identical
> bodies), and **all 62 safetensors shard headers via HTTP range requests**
> (exact dtypes/shapes for every one of the 76,108 tensors). The per-module
> byte sums reconstructed from those headers total **328,326,771,576 B —
> equal to the index's `metadata.total_size` exactly**, and the closing
> identity `shared-read + routed experts + embed + MTP + vision =
> total_size` also holds to the byte (§ 4). The main repo **is** the FP8
> serving checkpoint (tagged `fp8`); the BF16 base lives separately at
> `zai-org/GLM-5.3-Flash-BF16`. One research pass; the cross-check is the
> merge-request review (three independent reviewers against this note).

## 1. Architecture table (config.json + shard headers + vLLM recipe)

| Field | Value |
|---|---|
| `architectures` / `model_type` | `Glm5NextForConditionalGeneration` / `glm5_next` (multimodal wrapper; text config `glm5_next_text`) |
| `num_hidden_layers` | **45** = 34 `linear_attention` (KDA) + 11 `deepseek_sparse_attention` (layers 3, 7, …, 43; interval 4) — plus **1 MTP draft layer** on disk (layer 45: its own DSA stack + its own 288 experts, 7.43e9 params) |
| `hidden_size` / `vocab_size` | 4,096 / 154,880 (embed and lm_head untied) |
| Sparse attention (DSA) | **NoPE MLA**: `kv_lora_rank` 512, `qk_rope_head_dim` **0** (`mla_use_nope`; position lives in the indexer, `indexer_rope_interleave`) — 64 Q heads × 256 nope + 256 v; `q_lora_rank` 1536 |
| Indexer | 32 heads × 128 (`index_n_heads`/`index_head_dim`), **top-2048** selection, and a **kpool-4 compressed key cache** (`index_kpool` 4, `index_kpool_compress`, learned pooling gate + ape; `index_kpool_always_select_tail`) |
| Linear attention (KDA) | **64 heads × 128** (`linear_attn_config`), separate q/k/v kernel-4 short convs (`[8192,1,4]` each) |
| MoE | layers 0–2 dense MLP (`first_k_dense_replace: 3`, intermediate 12,288); layers 3–44: **288 routed experts / 8 per token + 1 shared**, `moe_intermediate_size` 2048, sigmoid routing (`noaux_tc`, scaling 2.5, fp32 router) |
| Hyper-connections | **mHC** (Manifold-Constrained Hyper-Connections, card): `hc_mult` 4, Sinkhorn mixer (20 iters), 35.4e6 params over 45 layers |
| `max_position_embeddings` | **1,048,576** (native 1M) |
| Total / active params | card: "**320B total … 18B active**"; disk 321.34e9 (incl. 7.43e9 MTP + 0.56e9 vision); active ledger 16.11e9 excl. embed/lm_head, 17.38e9 incl. ✓ |
| Vision tower | 24-block ViT, 0.564e9 params — an encoder, never executed on these text workloads |
| Checkpoint dtype (main repo) | MLA q/kv/o projections, all experts and dense MLPs **FP8** (block scales F32); `kv_b_proj`, the indexer, all of KDA, embed, lm_head, hc **BF16** |

## 2. KV bytes per token — 11 NoPE-MLA latents + a compressed indexer axis

Only the 11 DSA layers cache anything per token; the 34 KDA layers hold a
fixed recurrent state. The NoPE design removes GLM-5.2's per-entry rope
bytes: the cached entry is the bare 512-dim latent. The indexer caches its
128-dim keys on a **kpool-4 compressed axis** (one pooled entry per 4
tokens — the same construction as DSv4-Flash's CSA and Qwen3.8's QSA
indexers; GLM-5.2's own indexer was uncompressed):

```
kv_bpt (fp8 KV arm, per token):
  MLA latent    11 layers x 512 B (nope-only, fp8)     = 5,632
  indexer keys  11 layers x 132 B (128 fp8 + 4 scale,
                GLM-5.2 convention) / kpool 4          =   363
  ----------------------------------------------------------------
  TOTAL                                                = 5,995 B/token (5.85 KiB)

deltanet_state (fixed per session, bf16 KDA state):
  SSM state   34 layers x 64 heads x 128 x 128 x 2 B   = 71,303,168
  conv state  34 layers x 3 (q,k,v) x 8,192 x 4 x 2 B  =  6,684,672
  ----------------------------------------------------------------
  TOTAL                                                = 77,987,840 B ≈ 74.4 MiB/session
```

A 262k-token session holds **1.46 GiB** — 8.1× below GLM-5.2's 11.8 GiB
(48,408 / 5,995), still 1.7× above DSv4-Flash. The per-session state is the
study's second-heaviest, a hair under the 27B's 75 MiB; it is a genuine
recurrent state priced bf16 by the repo's calibrated convention
(`state_fp32_ok=True`; no `mamba_ssm_dtype` field in this config).

**KV dtype is GPU-coupled — the study's first such model.** The vLLM recipe:
"On Blackwell you can add `--kv-cache-dtype fp8` to both pools; **Hopper
does not support FP8 KV cache for this model and must run BF16 KV**." So
the FP8-KV base constants above are the *Blackwell* serving arm; on H200
the FP16-KV toggle (×2 on `kv_bpt` and the top-k reads) is the **only**
servable configuration (`kv_fp16_ok=True`, obviously). The explorer's
tooltip and deploy recipe both say so; the pool/decode math itself has no
GPU×dtype coupling mechanism, so an H200 + fp8-KV selection is a
modelled-but-not-servable arm, flagged in the recipe comment (§ 6).

## 3. Decode-bandwidth model — compressed scan + top-2048 latent gathers

Per decode step a query (a) scans the indexer's compressed keys over the
whole context, (b) reads the top-2048 selected latents per DSA layer, and
(c) the KDA layers read their fixed state (charged separately):

```
kv_decode_bpt   = 11 x 132/4                = 363 B per CONTEXT token per step
kv_decode_const = 11 x 2,048 x 512          = 11,534,336 B per ACTIVE SEQ per step
kv_decode_topk  = 2,048 tokens (index_topk); sequences shorter than the
                  budget scale the constant by min(len, 2048)/2048
                  (study convention, research/model_glm52.md #3)
```

At the reference 31k-median workload the scan is ~11 MB/seq and the top-k
read 11.5 MB/seq — versus ~188 MB/seq streaming the full cache at `kv_bpt`.
`index_kpool_always_select_tail` (the recent-token tail is always selected)
is inside the top-2048 budget, not additional. Under the FP16/BF16-KV arm
`kv_decode_const` doubles with `kv_bpt` (main-KV bytes); the compressed
indexer scan keeps its own width — the exact machinery added for Qwen3.8.

## 4. Weight bytes

### Native FP8 checkpoint (`zai-org/GLM-5.3-Flash`) — the study's FP8 arm

Reconstructed per-module bytes from all 62 shard headers; the sum equals
`metadata.total_size` **exactly**, and the vLLM recipe's "about 306 GiB"
matches (305.79 GiB):

```
routed experts  42 layers x 288 x (3 x 8,388,608 FP8
                + 6,144 F32 block scales)            = 304,480,124,928
KDA             34 layers, BF16                      =   9,366,356,992
MTP layer       layer 45: DSA + 288 experts + heads  =   7,493,399,168
attention (DSA) 11 layers, FP8 MLA + BF16 kv_b_proj
                and indexer                          =   1,641,091,584
embed_tokens    154,880 x 4,096 BF16                 =   1,268,776,960
lm_head         154,880 x 4,096 BF16                 =   1,268,776,960
vision tower    24 ViT blocks, BF16                  =   1,127,254,016
shared experts  42 layers, FP8 + scales              =   1,057,222,656
dense MLPs      layers 0-2, FP8 + scales             =     453,095,424
routers         42 x (288x4096 + bias)               =      99,138,816
hyper-conns     45 layers (mHC)                      =      70,788,600
norms           input/post-attn/final                =         745,472
------------------------------------------------------------------------
w_resident      (= metadata.total_size)              = 328,326,771,576 B  (305.79 GiB)
```

Per-routed-expert bytes: 3 × 8,388,608 FP8 + 6,144 F32 scales =
**25,171,968 B** (0.024% scale overhead, charged):

```
w_route_pertok  = 8   x 25,171,968 x 42 =   8,457,781,248 B
w_route_total   = 288 x 25,171,968 x 42 = 304,480,124,928 B
```

Expert-union saturation at n = 288/8 = **36** — between the 256/8 models'
32 and Qwen3.8's 51.2.

Shared per-step read — every always-active bucket, summed exactly:

```
KDA 9.366 + attention/DSA 1.641 + shared experts 1.057 + dense MLPs 0.453
  + routers 0.099 + hyper-conns 0.071 + lm_head 1.269 + norms 0.001
w_decode_shared = 13,957,216,504 B   (exact per-tensor sum, charged as-is)
```

Closing identity (all to the byte): `w_decode_shared + w_route_total +
embed 1,268,776,960 + MTP 7,493,399,168 + vision 1,127,254,016 =
328,326,771,576 = w_resident` ✓. Active check (params): 16.11e9 excl.
embed/lm_head (attn 1.374 + KDA 4.683 + shared 1.057 + dense 0.453 +
routers 0.050 + hc 0.035 + routed 8.456), 17.38e9 incl. ✓ card "18B
active" (= `params_prefill` basis).

### NVFP4 — **not modelled** (`nvfp4_w = None`)

No official NVFP4 checkpoint exists (no `nvidia/GLM-5.3-Flash-NVFP4`; the
only conversions are community — `LibertAIDAI/…-NVFP4` at 181.3 GiB and an
empty `vcruz305` repo, 2026-08-26). Same treatment as DSv4-Flash and
Qwen3.8: the explorer greys the option out; revisit if NVIDIA ships an
official recipe.

## 5. Serving notes

- **vLLM ≥ 0.27.0, Hopper and newer only**; FlashInfer ≥ 0.6.18 for the
  NoPE sparse-MLA path. Recipe launch (GB200 TP4): `--tensor-parallel-size
  4 --kv-cache-dtype fp8 --speculative-config '{"method":"mtp",
  "num_speculative_tokens":5}' --tool-call-parser glm47 --reasoning-parser
  glm45 --enable-auto-tool-choice`. PD-disaggregation notes:
  `VLLM_SSM_CONV_STATE_LAYOUT=DS`, `VLLM_KV_CACHE_LAYOUT=HND`;
  `num_speculative_tokens` must match across pools.
- **KV dtype:** fp8 KV **Blackwell-only**; "Hopper … must run BF16 KV"
  (recipe, quoted § 2). The recipe states no GPU-count floor and no OOM
  notes; the pool arithmetic gives **min TP 3 on H200** (3×141 GB) and
  **min TP 2 on B300** for the 305.79-GiB checkpoint — both are this
  study's arithmetic only, no recipe validation either way.
- **Speculative decoding:** one in-checkpoint MTP draft layer
  (`num_nextn_predict_layers: 1`); the recipe runs **5 draft tokens** — the
  2-draft α inversion is indicative only on this model. The study keeps its
  1.7× **transplanted** fit (module present, acceptance unmeasured).
- Positioning (card): "320B total parameters and just 18B active …
  outperforms GLM-5.2 … at one-tenth the price"; the KDA+NoPE-MLA hybrid
  is GLM-5.2's DSA married to a Qwen-style linear-attention backbone, with
  mHC hyper-connections (arXiv 2602.15763). MIT license.

## 6. Remaining assumptions / re-verification ledger

- **MLA latent entry = 512 B fp8** (nope-only, no rope bytes — the design's
  point) with quantization-scale bytes on cached latents **not** charged
  (same treatment as GLM-5.2/DSv4-Flash: ~+1% on `kv_bpt` if ue8m0-style
  block scales are stored).
- **Indexer entry = 132 B (128 fp8 + 4 scale) per kpool-4 pooled position**,
  by the GLM-5.2 convention + the DSv4/Q38FN compressed-axis precedent. An
  uncompressed cache would be 5,632 + 1,452 = 7,084 B/token (+18%); a
  BF16 indexer cache would add ~+5%. Neither the config nor the recipe
  states the dtype; re-verify when the serving path is documented.
- **`index_topk` = 2,048 read in TOKEN space** (GLM-5.2's own convention;
  DSv4-Flash counts compressed entries). If it counts pooled entries, the
  top-k read is 4× (46.1 MB/seq) — still 4× below the dense stream.
- **KV dtype × GPU coupling is not enforced in the math** (§ 2): the
  fp8-KV constants are servable on Blackwell only; on H200 use the FP16-KV
  toggle. The deploy recipe emits a warning comment on the unservable
  combination; the pool model itself will happily price it.
- **KDA state priced bf16** (71.3 MB SSM + 6.7 MB conv): the config carries
  no state-dtype field; bf16 is the repo's calibrated convention (the 27B's
  measured 75 MiB) and the fp32 toggle covers the alternative. The
  recipe's `VLLM_SSM_CONV_STATE_LAYOUT=DS` concerns layout, not dtype.
- Prefill quadratic term priced as **dense** attention on the 11 DSA layers
  (`attn_layers=11, attn_d=64×256`): an upper bound, DSA prefill sparsity
  uncharacterised — identical flagged treatment to GLM-5.2 and Qwen3.8
  (research/prefill.md weaknesses 2–3). `params_prefill=16.11e9` = the
  active-GEMM ledger excl. embed/lm_head (cheap-side bias, per convention).
- MTP layer (7.49e9 B) and vision tower (1.13e9 B) charged in `w_resident`,
  never in per-step reads.
- The BF16 sibling checkpoint (`-BF16`, ~2× weight bytes) is not modelled;
  the study models the FP8 serving checkpoint, as for every other model.

## Sources

Primary (read directly, exact bytes):
- https://huggingface.co/zai-org/GLM-5.3-Flash/resolve/main/config.json
  (layer map, KDA/DSA/indexer/MoE/mHC geometry, 1M context)
- https://huggingface.co/zai-org/GLM-5.3-Flash/resolve/main/model.safetensors.index.json
  (`total_size` 328,326,771,576; 76,108-tensor weight map) + **all 62
  safetensors shard headers via HTTP range requests** (per-tensor
  dtypes/shapes; module sums reproduce total_size exactly)
- https://huggingface.co/api/models/zai-org/GLM-5.3-Flash (license MIT,
  `fp8` tag, dtype histogram BF16 6.926e9 / F8_E4M3 314.397e9 / F32 0.3e6
  — matches the ledger with `*_scale_inv` excluded, a known HF-counter
  convention) · …/GLM-5.3-Flash-BF16 (the bf16 base)
- Model card: https://huggingface.co/zai-org/GLM-5.3-Flash ("320B total
  parameters and just 18B active"; mHC; arXiv 2602.15763; no serving
  commands or cache-layout details — the recipe carries those)
- Official vLLM recipe (45-layer hybrid, ~306 GiB FP8, Hopper-needs-BF16-KV,
  MTP num_speculative_tokens 5, FlashInfer/PD-disagg notes):
  https://recipes.vllm.ai/zai-org/GLM-5.3-Flash
- NVFP4 absence: https://huggingface.co/api/models?search=GLM-5.3-Flash
  (community conversions only, 2026-08-26)

Secondary:
- research/model_glm52.md (DSA decode-pricing + indexer-entry conventions) ·
  research/model_dsv4flash.md and research/model_qwen38flashnext.md
  (compressed-indexer precedent, top-k scaling, FP16-KV sparse handling) ·
  research/model_35ba3b.md (recurrent-state + conv arithmetic convention)
