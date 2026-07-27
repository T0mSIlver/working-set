# GLM-5.2 (744B-A40B, MLA + DeepSeek Sparse Attention) — parameterization note

**Purpose:** defensible KV-cache / decode-bandwidth constants for **GLM-5.2**
(Z.ai / Zhipu, `zai-org/GLM-5.2`, MIT weights, released 2026-06) as used by
`scripts/scenario_model.py` and the explorer.

> **Egress note (2026-07-27):** `huggingface.co` and `z.ai` were blocked at
> this environment's proxy when this note was first written; the `config.json`
> was read from three agreeing GitHub mirrors instead.
>
> **Re-verified same day after the block lifted — every value below is now
> confirmed against the literal HF files:** base + FP8 + NVFP4 `config.json`
> (the FP8 `modules_to_not_convert` has exactly 541 entries; the NVFP4
> `ignore` list matches § 4 pattern-for-pattern), the `indexer_types` array
> (literally 21 "full" indexers at layers 0,1,2,6,10,…,74 + 
> `index_share_for_mtp_iteration: true`), shard-header dtype sampling (routed
> experts are U8-packed FP4 + E4M3 block scales = exactly 0.5625 B/param),
> and both `total_size` figures: **FP8 755,617,140,416 B** (note's 755.5e9,
> +0.016%) and **NVFP4 464,795,267,072 B** (note's 464.8e9, −0.001%). No
> constant changed. MIT license, 2026-06-17 release and the 1M context are
> confirmed on the HF card and docs.z.ai (the z.ai blog page itself still
> returned empty — the sole remaining secondary-only citation).

## 1. Architecture table (config.json, mirrored)

| Field | Value |
|---|---|
| `architectures` / `model_type` | `GlmMoeDsaForCausalLM` / `glm_moe_dsa` |
| `num_hidden_layers` | **78** (+1 MTP layer, index 78) |
| `hidden_size` | 6144 |
| Attention | **MLA + DeepSeek Sparse Attention (DSA)** on all 78 layers — no GQA cache, no recurrent layers |
| `kv_lora_rank` / `qk_rope_head_dim` | **512 / 64** → cached latent = **576/token/layer** |
| `qk_nope_head_dim` / `v_head_dim` / heads | 192 / 256 / 64 |
| DSA `index_topk` | **2048** (tokens attended per query) |
| DSA indexer | `index_head_dim` 128, key cache **132 B/token/indexer-layer** (fp8 keys + group scales) |
| **IndexShare** (`index_topk_freq=4`) | only **21 of 78** layers carry a real indexer (layers 0,1,2,6,10,…,74); the rest reuse |
| `n_routed_experts` / `num_experts_per_tok` / shared | **256 / 8 / 1** |
| `moe_intermediate_size` / dense `intermediate_size` | 2048 / 12288 |
| `first_k_dense_replace` | 3 (layers 0–2 dense, 75 MoE) |
| `vocab_size` | 154880 |
| `max_position_embeddings` | **1,048,576** (1M; θ = 8e6) |
| MTP | **1 module** (`num_nextn_predict_layers: 1`), 5 draft tokens in vLLM |
| Total / active params | **753.3B incl. MTP (743.4B backbone) / 39.3B active** — derived below; official "744B-A40B", NVIDIA card "753B" |

Param arithmetic (config-derived; reconciles all three published counts):
MLA 165.0M/layer; indexer 9.37M × 21; dense MLP 226.5M × 3; MoE layer
9.703B × 75 (256 experts × 37.75M + shared + router); embeddings + lm_head
2 × 951.6M; MTP block 9.95B. Totals: **743.4B** (vLLM "~743B"), **753.3B**
with MTP (NVIDIA "753B"), active 39.3B excl. embeddings (vLLM "39B") ✓✓✓.

## 2. KV bytes per token — MLA latent + DSA indexer cache

Only a **compressed latent** is cached per layer (`kv_lora_rank +
qk_rope_head_dim = 576` elements, one "head"), confirmed in vLLM
(`MLAAttentionSpec(num_kv_heads=1, head_size=576)`). The DSA **indexer key
cache** (132 B/token) exists on the 21 indexer layers + the MTP layer.
**vLLM requires a quantized (FP8) KV cache on this path** — BF16 KV is not
servable, so the study's FP16-KV toggle is disabled for this model.

```
kv_bpt (fp8, incl. MTP layer):
  MLA latent  79 layers x 576 B      = 45,504
  indexer     22 layers x 132 B      =  2,904
  ----------------------------------------------
  TOTAL                              = 48,408 B = 47.3 KiB/token
(alternative fp8_ds_mla FlashMLA layout: 656 B/layer -> 54,728 B = +13%,
 not modelled; BF16 theoretical 91.7 KiB — unservable)
```

A 262k-token session holds 11.8 GiB; the model's full 1M context holds
47.3 GiB. The study's *reference* cap (180k) stays far below GLM-5.2's
native window; the allowed range extends to the full 1M
(`max_ctx = 1_048_576`, owner decision 2026-07).

## 3. Decode-bandwidth model — DSA reads are NOT the full cache

Sparse attention breaks the study's default "decode reads every cached
token" pricing, so GLM-5.2 carries two decode-specific KV fields:

```
kv_decode_bpt   = 21 x 132        = 2,772 B per CONTEXT token per step
                  (the indexer must score every token, on indexer layers only)
kv_decode_const = 78 x 2048 x 576 = 92.0e6 B per ACTIVE SEQUENCE per step
                  (sparse MLA reads only the top-2048 tokens per layer)
```

At the reference 31k-median workload the indexer scan (~86 MB/seq) and the
sparse read (92 MB/seq) are comparable; both are far below the dense
equivalent (31k × 47.3 KiB ≈ 1.5 GB/seq) — DSA is why a 750B-class model
can decode long contexts at all.

## 4. Weight bytes

### FP8 — official `zai-org/GLM-5.2-FP8`

`quant_method: fp8`, block 128×128; `modules_to_not_convert` (541 entries):
norms, `mlp.gate` (+bias), indexer projections, `lm_head`, `embed_tokens`,
MTP extras — ≈ 2.2e9 params kept BF16.

```
w_resident      = 753.3e9 x 1 B + 2.2e9 (BF16 excess) = 755.5e9 B  (703.6 GiB)
w_decode_shared = MLA 12.87 + indexers 0.394(BF16) + dense MLP 0.680
                + shared experts 2.831 + gates 0.243(BF16) + lm_head 1.903(BF16)
                                                       = 18.92e9 B
w_route_pertok  = 8 x 37.75e6 x 75                     = 22.65e9 B
w_route_total   = 256 x 37.75e6 x 75                   = 724.8e9 B
```

Active check: 18.92 (less lm_head read ≈ 17.0 in params) + 22.65 ≈ 39.3e9
params/step ✓ "A40B". Expert-union saturation at n = 256/8 = **32**, same as
the 35B-A3B. MTP module weight reads ignored (folded into the speedup),
as everywhere in this study.

### NVFP4 — official `nvidia/GLM-5.2-NVFP4`

ModelOpt `quant_algo: NVFP4`, `group_size: 16`, **FP8 KV** (`kv_cache_scheme:
8-bit float`). The `ignore` list keeps dense layers 0–2, ALL `self_attn*`,
ALL `mlp.shared_experts*`, `lm_head`, `embed_tokens` in BF16 — **only the
256-routed-expert linears are NVFP4**:

```
routed experts  724.8e9 x 0.5625                       = 407.7e9 B
BF16 everything else: attn 25.74 + shared-exp 5.66 + dense MLP 1.36
  + indexers 0.394 + gates 0.243 + lm_head 1.903 + embed 1.903 + MTP 19.9
                                                       =  57.1e9 B
--------------------------------------------------------------------
w_resident_nvfp4                                       = 464.8e9 B (432.9 GiB)
```

**Cross-check (non-circular): the official vLLM recipe reports the NVFP4
checkpoint as "~465 GB"** — the config-derived total lands within 0.05%.
(NVIDIA card snippets also show 410/459 GB variants; 465 is the recipe's.)

```
w_decode_shared_nvfp4 = 25.74+0.394+1.36+5.66+0.243+1.903 = 35.30e9 B
w_route_pertok_nvfp4  = 22.65e9 x 0.5625                  = 12.74e9 B
w_route_total_nvfp4   = 407.7e9
```

Same phenomenon as the 35B-A3B: NVFP4-from-BF16 makes the *shared* per-step
read ~1.9× heavier than the FP8 checkpoint's, while the routed-expert read
falls 1.78×.

## 5. Serving notes

- **vLLM ≥ 0.23.0** (day-0 support via the DeepSeek-V3.2 DSA path;
  `--kv-cache-dtype fp8` mandatory). MTP speculative decoding supported
  (5 draft tokens); expert parallel recommended for NVFP4.
- Recipe VRAM floors: BF16 1786 GB · FP8 893 GB · NVFP4 558 GB — GLM-5.2
  does not fit ≤ 6×H200 at FP8; the explorer will show a zero pool there
  rather than hiding the config.
- MTP speedup default kept at the study's 1.7× **transplanted** fit (same
  treatment as the 35B-A3B: module present, acceptance unmeasured on this
  workload; GLM-5.2's 5-draft MTP could be higher — knob covers it).

## 6. Remaining assumptions / re-verification ledger

- fp8 (576 B) KV layout modelled; `fp8_ds_mla` (+13%) not.
- The DSA decode pricing (§3) is a byte model of the vLLM kernels' *reads*;
  indexer top-k compute and gather costs are not priced (roofline limitation,
  as everywhere).
- MTP-layer indexer shares the main top-k (`index_share_for_mtp_iteration`),
  so no extra decode scan is charged for it.
- ~~The top-2048 sparse read (`kv_decode_const`) is charged in full for every
  active sequence, including ones shorter than 2,048 tokens.~~ **Fixed
  2026-07-27** (review finding): the per-sequence read is now scaled by
  `min(len, kv_decode_topk) / kv_decode_topk` in both `decode_curves` and the
  explorer, so a sub-2k context only pays for its own tokens. Invisible at
  the reference workload (every sampled length is floored at its ≥3k prefix).
- ~~Literal HF bytes unread (blocked); mirrors are consistent three-way.~~
  **Resolved 2026-07-27:** literal HF configs, quant configs, ignore lists,
  shard-header dtypes and both checkpoint `total_size` figures all read and
  matched (see the egress note at the top).

## Sources

- Z.ai official README (744B-A40B, repos, vLLM 0.23+, licenses):
  https://raw.githubusercontent.com/zai-org/GLM-5/main/README.md
- config.json mirrors (3-way agreement):
  https://raw.githubusercontent.com/ai-dynamo/aiconfigurator/e95ebf3c501848013c57eaab77c6abf4583f0927/aic-core/src/aiconfigurator_core/model_configs/zai-org--GLM-5.2_config.json ·
  https://raw.githubusercontent.com/KinChow/kinchow.github.io/3e56e71302641ade4dfb5c5d479d80dcb4e63227/plugins/model-structure-viewer/models/zai-org/GLM-5.2/config.json ·
  https://raw.githubusercontent.com/zhao9797/ai-research/9da7a092fdbe8c1c2d1071d3c990d2befb081e64/sources/llm/2026/glm-5.2-config.json
- FP8 + NVFP4 quantization configs (same mirror set):
  `zai-org--GLM-5.2-FP8_config.json`, `nvidia--GLM-5.2-NVFP4_config.json`
- transformers `GlmMoeDsaConfig` (defaults cross-check):
  https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm_moe_dsa/configuration_glm_moe_dsa.py
- vLLM DSA implementation (576-latent, 132 B indexer cache, fp8-KV
  requirement, IndexShare guard):
  https://github.com/vllm-project/vllm/blob/main/vllm/models/deepseek_v32/nvidia/attention.py ·
  https://github.com/vllm-project/vllm/blob/main/vllm/v1/kv_cache_interface.py ·
  https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/deepseek_v2.py
- vLLM official recipe (min version, VRAM floors, "~465 GB" NVFP4, MTP/EP):
  https://raw.githubusercontent.com/vllm-project/recipes/main/models/zai-org/GLM-5.2.yaml
- IndexShare paper: https://arxiv.org/abs/2603.12201 · GLM-5 report:
  https://arxiv.org/abs/2602.15763
- Secondary (dates/license corroboration): NIST CAISI assessment —
  https://www.nist.gov/news-events/news/2026/07/caisi-assessment-zais-glm-52 ·
  https://datanorth.ai/news/zhipu-ai-releases-glm-5-2
