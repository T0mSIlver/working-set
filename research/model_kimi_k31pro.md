# Kimi K3.1 Pro (2.8T-A104B, KDA + gated DSA) — architecture parameterization note

**Purpose:** the constants behind the `K31P` row of
`src/workingset/data/catalog.toml` (loaded by `src/workingset/model.py`,
mirrored in `interactive/src/catalog.js`).

**Status (2026-11-05):** open weights since 2026-11-04 (first Hub commit
2026-11-02), integrated as `K31P`. Tom asked for "Kimi support"; K3.1 Pro is
the newest open-weight Kimi and the best-scoring model the study prices on
Terminal-Bench 4.0.

Sources, all read 2026-11-05:

| Source | Revision | Read |
|---|---|---|
| [`moonshotai/Kimi-K3.1-Pro`](https://huggingface.co/moonshotai/Kimi-K3.1-Pro) | `6d0a0202` | `config.json`, `configuration_kimi_k3.py`, the safetensors index, **all 96 shard headers**, README, LICENSE |
| [`Inferact/Kimi-K3.1-Pro-DSpark-preview`](https://huggingface.co/Inferact/Kimi-K3.1-Pro-DSpark-preview) | `5e654fbe` | `config.json`, the shard header, README (acceptance tables) |
| vLLM recipe `recipes/models/moonshotai/Kimi-K3.1-Pro.yaml` (vLLM repo) | `main` at `56dc7b70`, `date_updated: 2026-11-05` | flags, hardware floors, KV and DCP notes, prefill timing |
| vLLM [#65684](https://github.com/vllm-project/vllm/pull/65684) recipe, [#65685](https://github.com/vllm-project/vllm/pull/65685) "[Model] Kimi K3.1 Pro" (open) | head `9d6ac73f` | `vllm/models/kimi_k3/common/dsa_cache.py`, `nvidia/dsa.py`, `nvidia/kda.py`, `model_executor/models/config.py` |
| InferenceX `golden_al_distribution/kimi-k3.1-pro__dspark__kimi-k3.1-pro-dspark-preview__probabilistic-block.yaml` | `115b00b8` (added in `2f6289dd`, 2026-11-04) | accepted length per draft depth |
| SGLang cookbook `docs.sglang.io/cookbook/autoregressive/Moonshotai/Kimi-K3.1-Pro` | — | serving layouts (§ 7) |
| Artificial Analysis model page | — | embedded dataset (§ 9) |
| `T0mSIlver/website`, branch `research/open-weights-architecture`, `research/open-weights-architecture/moonshot.md` | — | cross-check of the architecture fields; it agrees with every field below |

## 1. Architecture

| Field | Value | Source |
|---|---|---|
| `architectures` / `model_type` | `KimiK31ForConditionalGeneration` / `kimi_k31`; text config `kimi_linear` (MoonViT-V2 vision tower + text decoder) | config |
| Layers / hidden | **93** / 7168 | config |
| Layer map | **69 KDA** (Kimi Delta Attention, linear, recurrent state) + **24 gated DSA**. The config's lists are 1-indexed: DSA at 4, 8, …, 92 **and 93**, so the last two layers are both full attention; KDA everywhere else. In the 0-indexed weight map the DSA layers are 3, 7, …, 91, 92 | `linear_attn_config`, `dsa_config`; shard headers |
| Gated DSA | MLA (96 heads, `q_lora_rank` 1536, `kv_lora_rank` 512, `qk_nope_head_dim` 128, `qk_rope_head_dim` 64, `v_head_dim` 128, `mla_use_nope` true) with an output gate (`mla_use_output_gate`, a 12,288 × 7168 `g_proj`) and a DeepSeek-style indexer: 32 heads × 128, `index_topk` **2048**, no indexer RoPE, layernorm on the keys. The card: "Compared with Kimi K3, we replace the full-attention Gated MLA layers with Gated DeepSeek Sparse Attention (Gated DSA) layers" | config, card |
| Index sharing | **none**: every DSA layer runs its own indexer (24 sets of indexer weights in the headers). vLLM refuses `index_topk_freq != 1`: "Kimi-K3.1 has no index-share mechanism" | headers; vLLM `dsa.py` |
| KDA | 96 heads × 128, separate q/k/v short convolutions of kernel 4, full-rank gate (`f_a_proj`/`f_b_proj`) | config, headers |
| MoE | layer 0 dense (`intermediate_size` 33,792); layers 1–92: **896 routed experts, 16 per token**, 2 shared (6,144 wide together), `moe_intermediate_size` 3072. Routed experts run in a **3,584-wide latent space**: a BF16 down-projection 7168 → 3584 and up-projection back, per layer, always active (`routed_expert_hidden_size`, `latent_moe_use_norm`) | config, headers |
| Attention Residuals | block AttnRes, `attn_res_block_size` 12: each layer mixes earlier block outputs of the same token through a learned pseudo-query (`*_res_proj`, 1 × 7168). Depth-wise, inside one forward pass, so it caches nothing across tokens | config, headers |
| Vocab / embeddings | 163,840, untied | config |
| Context | `max_position_embeddings` **1,048,576**; the recipe serves `--max-model-len 1048576` | config, recipe |
| Draft model | none in the checkpoint (`num_nextn_predict_layers` 0); a separate **DSpark** drafter (§ 5) | config; recipe |
| Quantization | "MXFP4 weights / MXFP8 activations (quantization-aware training)": compressed-tensors `mxfp4-pack-quantized`, group 32, UE8M0 scales, on the routed experts only; everything else BF16 | card, `quantization_config` |
| License | **Kimi K3.1 Pro License**: MIT terms, plus a separate agreement with Moonshot for a Model-as-a-Service business above $20M revenue over 12 months, and the model's name displayed above 100M monthly users or $20M monthly revenue. Internal use is exempt from both | `LICENSE` |

**Parameters (shard-header ledger, MXFP4 values counted two per byte):**
2,780.1B total, the HF count to the parameter. Active per token: **103.19B**
excluding embeddings and lm_head, 104.37B with lm_head, 105.54B with both;
the card's "104B". Vision tower and projector 0.447B (card: 401M for the
encoder).

## 2. KV bytes per token

vLLM's K3.1 verifier (`KimiK3ForConditionalGenerationConfig` in
`model_executor/models/config.py`, PR #65685) picks the cache format: on SM10x
with no backend named, `auto` becomes **per-tensor `fp8_e4m3`** for FlashInfer's
sparse MLA ("FlashInfer reads per-tensor fp8 MLA rows, not the ds_mla
formats"); anywhere else an fp8 request becomes `fp8_ds_mla`. So the B300 row
is 576 B per latent and the Hopper row 656 B, GLM-5.3's split (issue #122).
The recipe confirms the B300 figure: its NVFP4-KV option stores "352 bytes per
token per layer instead of 576".

The indexer keys are FP8 plus one fp32 scale, 132 B (`indexer_cache_entry_bytes`;
an opt-in MXFP4 format is 68 B). The DSpark drafter's five dense MLA layers
store their latents in the same block row (`dense_mla_draft_bytes_per_token`),
so they are a per-token cost, not a window:

```
kv_bpt (fp8, B300):
  MLA latent       24 layers x 576 B   = 13,824
  indexer keys     24 layers x 132 B   =  3,168
  DSpark drafter    5 layers x 576 B   =  2,880
  ------------------------------------------------
  TOTAL                                = 19,872 B = 19.4 KiB/token
H200 row (kv_hopper_fp8): 24 x 656 + 24 x 132 + 5 x 656 = 22,192 B
```

59% below GLM-5.3 on B300 (48,408 B) and 58% below Mistral-Large-4 (47,136 B).
A 262k session holds 4.85 GiB, the full 1M 19.4 GiB. "Gated" adds weights
(`g_proj`), not cache bytes; NoPE keeps the 64 rope slots in the row.

**Recurrent state.** Each KDA layer holds a 128 × 128 state per head; the
short convolutions hold their window:

```
deltanet_state (bf16):
  KDA state   69 layers x 96 heads x 128 x 128 x 2 B   = 217,055,232
  conv state  69 layers x 3 (q,k,v) x 12,288 x 4 x 2 B =  20,348,928
  ---------------------------------------------------------------------
  TOTAL                                                = 237,404,160 B = 226.4 MiB/session
```

The study's heaviest state, 3× GLM-5.3-Flash's 74.4 MiB. vLLM keys its dtype
on `--mamba-ssm-cache-dtype` (`get_state_dtype` in `nvidia/kda.py`), whose
default is the model dtype, BF16; the fp32 toggle stays live. The conv state
follows the repo's kernel-width convention (4); vLLM stores the kernel width
less one plus a row per draft token.

**KV dtype.** FP8 is the default on B300 (vLLM turns `auto` into it), so the
page's FP8 arm emits `fp8_e4m3`. An explicit `bfloat16` passes the verifier
untouched, and the FP16 arm prices it (`kv_fp16_ok` true), but the recipe
documents only FP8 and NVFP4. Under that arm `kv_bpt` doubles wholesale, which
over-doubles the FP8 indexer keys (3,168 B, 9% of the BF16 row) and under-counts
the drafter's read by 2,880 B per context token (§ 3).

## 3. Decode reads

```
kv_decode_bpt   = 24 x 132 + 5 x 576 = 6,048 B per CONTEXT token per step
                  (24 indexer scans, plus the drafter's dense read: its 5 layers
                   attend the full context once per step, in one pass, § 5)
kv_decode_const = 24 x 2048 x 576    = 28,311,552 B per ACTIVE SEQUENCE per step
kv_decode_topk  = 2048
state traffic   = 2 x 237,404,160    = 474.8 MB per active sequence per step
```

At the 31k median context a sequence reads 98 MB of indexer keys, 89 MB of
drafter latents and 28 MB of top-k latents, and moves 475 MB of KDA state.
The other drafters' caches go unread in the step equation: each is one MTP
layer or, for Mistral-Large-4, three 2,048-token windows. Kimi's drafter reads
five full-context layers every step, 91% of the indexer scan, so it is charged.

## 4. Weight bytes

`quant_method: compressed-tensors`, `mxfp4-pack-quantized`, group 32, UE8M0
scales (`scale_dtype: torch.uint8`). The ignore list keeps BF16: every
attention tensor, the shared experts, the dense MLP, the router, the latent
MoE projections, lm_head, the vision tower and projector. vLLM rewrites the
method to `mxfp4` to reach its full MXFP4 MoE backend set.

| Component | Bytes | Note |
|---|---:|---|
| Routed experts | 1,446,456,066,048 | 82,432 experts × 17,547,264 B (3 × 3072 × 3584 MXFP4 packed + 3 × 3072 × 3584 / 32 UE8M0 scales) |
| KDA (69 layers) | 61,258,257,408 | BF16: q/k/v/g/o projections of 12,288 × 7168 each, gates, convs |
| Shared experts | 24,310,185,984 | 92 × 3 × 6144 × 7168 BF16 |
| MLA (24 layers) | 11,145,412,608 | BF16, `g_proj` included |
| Latent MoE projections | 9,454,620,672 | 92 × 2 × 3584 × 7168 BF16 + norms |
| lm_head / embeddings | 2,348,810,240 each | BF16 |
| Dense MLP (layer 0) | 1,453,326,336 | BF16 |
| Routers (92) | 1,182,074,880 | BF16 + F32 bias |
| Vision tower + projector | 894,717,952 | BF16 |
| Indexers (24) | 357,052,416 | BF16 |
| Norms and AttnRes | 8,042,496 | |
| **Total** | **1,561,217,377,280** | = the index's `total_size`, to the byte |

```
w_resident      = 1,566,590,881,794 B   total_size + the BF16 DSpark drafter (5,373,504,514)
w_decode_shared =   111,517,783,040 B   KDA + shared experts + MLA + latent projections
                                        + lm_head + dense MLP + routers + indexers + norms
w_route_pertok  =    25,829,572,608 B   16 x 17,547,264 x 92
w_route_total   = 1,446,456,066,048 B   896 experts: the union saturates at n = 56
```

The always-active read is the study's heaviest, 5.3× Mistral-Large-4's
21.0 GB, because nothing outside the experts is quantized; 61.3 GB of it is
KDA projections. The routed read per token (25.8 GB) sits between GLM-5.3's
22.6 and Mistral-Large-4's 32.2. With 6 verify positions per sequence (5
drafts), the expert union covers 82% of all experts at 16 sequences and 97% at
32, so a step reads 1.3 TB and 1.5 TB of weights there: decode, not the cache,
limits the node.

**NVFP4.** No NVFP4 checkpoint exists (Hub search, 2026-11-05: Moonshot's
repo, mirrors and two drafters only); `nvfp4_w = None`. The experts are
already 4-bit; an NVFP4 repack would mainly matter if it quantized the BF16
remainder. The base arm is labelled MXFP4 in the deploy card
(`explorer.weights`).

**Not modelled:** weights vLLM keeps on every TP rank rather than splitting.
§ 8's capacity check puts vLLM's resident footprint about 24 GB per GPU above
1/8 of the checkpoint.

## 5. DSpark drafter

`Inferact/Kimi-K3.1-Pro-DSpark-preview` (marked preview: "we are working on a
stronger version"): 5 dense MLA decoder layers in BF16 (2.687B parameters,
5,373,504,514 B, its Markov and confidence heads included), full causal
attention with no sliding window, reading hidden states of target layers 75,
79, 83, 87 and 91 (0-indexed), sharing the target's embedding and lm_head. Its
README: "Draft block: 7 tokens per forward pass". The recipe runs 5.

```
w_draft         = 5,373,504,514 + 2,348,810,240 (lm_head) = 7.722e9 B per pass
draft_passes    = 1     one forward pass drafts the whole block
draft_layers    = 5
spec_tokens     = 5
acceptance      = 0.721 (geometric, from 3.08 tokens per step at 5 drafts)
```

`draft_passes` is new with this model: the step equation charged one draft
pass per draft token (`k · W_draft`), right for MTP heads, five times too many
here. `Model.passes(k)` returns `k` unless the catalogue sets `draft_passes`;
only `K31P` does (`research/decode-model.md`, Open). At 8×B300 TP8 the one-pass
pricing adds 11% per-user speed at one sequence and 2–4% from 8 on.

**Acceptance**, measured outside the repo, three figures at 5 drafts:

| Source | Workload | Tokens per step | a |
|---|---|---:|---:|
| InferenceX golden AL, B300, vLLM, probabilistic drafting + block verification | SPEED-Bench coding, thinking on | **3.08** | **0.721** |
| vLLM recipe, MI355X AgentX preset | InferenceX AgentX (synthetic acceptance setting) | 3.62 | 0.792 |
| drafter README, SWE-bench Pro, cumulative rates of positions 1–5 | 642 prompts, 2,048 tokens | 2.63 | 0.648 |

The study takes InferenceX's 3.08, the same source and protocol as GLM-5.3's
0.797. The README's 2.63 truncates a 7-draft run at 5, an approximation.

## 6. Prefill constants

`params_prefill = 103.19e9`, the active GEMM parameters excluding embeddings
and lm_head. `attn_layers = 24`, `attn_d = 15,360` = 96 heads × 160, the mean of
the QK width (192) and the AV width (128), so the causal FLOP count is exact
for dense attention. As for GLM-5.3, dense attention over the DSA layers is an
upper bound. The KDA layers' chunked state updates are not counted, as for the
other linear-attention models.

## 7. Serving

| | Weights | Min TP (study arithmetic) | Recipe |
|---|---:|---:|---|
| H200 | 1,566.6 GB | **13** | not supported (the K3.1 recipe lists B300, GB300, VR200, MI355X and Zhenwu PPUs; the K3 recipe ran Hopper) |
| B300 | 1,566.6 GB | **7** → TP8 | one B300 node at TP8, or TEP8; DCP8 for capacity |

The page prices it on 8×B300 at TP8 only. Its command follows the recipe:
the `vllm/vllm-openai:kimi-k31` image ("K3.1 support isn't in a wheel yet",
vLLM #65685), `--trust-remote-code`, FP8 KV (`fp8_e4m3`), and the DSpark
config `{"model":"Inferact/Kimi-K3.1-Pro-DSpark-preview","num_speculative_tokens":5,"method":"dspark","draft_sample_method":"probabilistic"}`.
DSpark with DCP8 is validated in the recipe's MI355X AgentX preset (TP8 DCP8,
this draft), so the page emits both flags (`spec_with_dcp`). The recipe adds
`--load-format fastsafetensors` on Blackwell to load the 1.56 TB faster.

The recipe says DCP8 decode steps run "7–25% slower than TP8 at the same
batch"; the page prices DCP's single cache copy at TP8 speed, as for every
model.

SGLang also serves it (cookbook, 2026-11-05): B300 TP8 for latency, DCP8 for
concurrency, `fp8_e4m3` KV, a DFlash drafter (`RadixArk/Kimi-K3.1-Dflash2-Preview`),
H200 only across two nodes (TP16 · EP16).

## 8. Measured anchors against the model

| | Measured | Model | Ratio |
|---|---:|---:|---:|
| Prefill, one 340k-token prompt, 8×B300 TP8 (recipe, pre-release checkpoint) | ~15 s | 12.8–13.9 s (chunk 32k–8k) | model 7–15% fast |
| KV pool, 8×B300 DCP8, FP8 KV, no drafter (recipe, "about") | 18.5M tokens | 29.9M as published, 16.1M at +15% weights | between the two weight arms |

The pool check: with the drafter's row the page holds 25.6M tokens at
as-published weights, 63.5 GB of cache per GPU. vLLM's 18.5M at the
no-drafter row is 39.3 GB per GPU: 24 GB less, about 80% of what the explorer's
+15% weight arm takes (29 GB per GPU). So vLLM resides about 24 GB per GPU
beyond 1/8 of the checkpoint (replicated projections, MXFP4 kernel layouts,
the hybrid block-row packing; none separated). The recipe's TP8 figure (2.6M
tokens, one full copy per GPU) is 44.2 GB per GPU, within 5 GB of the DCP8
figure.
Decode is the binding limit (§ 10), so the pool's slack does not move the
verdict. No measured decode step time for K3.1 on B300 was found: the
recipe's low-latency figures are relative (1.6–1.7× at batch 1).

## 9. Quality

The study takes Terminal-Bench from Artificial Analysis only
(`terminal_bench.md`). Read 2026-11-05 from the AA page's dataset
(`kimi-k3-1-pro`, released 2026-10-23 per AA):

| Version | Score | Runs | Note |
|---|---:|---:|---|
| 2.1 | **no run** | — | `terminalBench21: null` |
| 4.0 | **67.2%** | 133 / 198 | AA labels no effort variant |

Moonshot's card reports 65.7% on 4.0 (max effort, Claude Code harness). AA's
mini-swe-agent run lands 1.5 points above it: the score holds across two
harnesses, unlike Mistral-Large-4's (+16.7 in its vendor's agent). It is the
best 4.0 score the study prices, 25.3 points above GLM-5.3 (41.9).

## 10. Results on 8×B300 (default workload, 64 users, 2026-11-05)

| Row | Max users | Binds on | €/seat | Per-user tok/s, n = 1 / 16 / 64 |
|---|---:|---|---:|---|
| Kimi-K3.1-Pro TP8 | 32 | decode | 352 | 175 / 41 / 33 |
| Mistral-Large-4 TP8 (FP8) | 587 | slowed generation | 19 | 283 / 55 / 46 |
| GLM-5.3 TP8 | 431 | slowed generation | 26 | 375 / 88 / 79 |

Per-user speed falls to the 40 tok/s floor at about 17 sequences decoding at
once: from there every step reads the 111.5 GB always-active set and most of
the 1.45 TB of experts. A seat costs 13.5× GLM-5.3's and 18.5×
Mistral-Large-4's, for 25.3 and 42.5 more points on Terminal-Bench 4.0.

## 11. What the brief got right

The brief had 2.8T / ~104B active (2,780.1B; 103.2B excluding embeddings and
lm_head), 93 layers of 69 KDA + 24 full attention (right, with the full
attention gated DSA, not gated MLA; that was K3), 896 / 16 experts, 1M
context, native multimodality, ~1.56 TB on disk (1,561.2 GB), and the fit:
8×B300, no 8×H200 node. The experts are MXFP4 with UE8M0 group-32 scales,
not INT4.

## 12. Unverified

- vLLM's per-GPU resident weights at TP8 (§ 8): at as-published weights the
  page's pool is 1.6× the recipe's figure; the +15% arm is closer.
- KDA state in BF16 (vLLM's default); the K3 recipe's low-latency path wanted
  FP32. The conv state uses the repo's kernel-width convention.
- The BF16 KV arm: accepted by vLLM's verifier, undocumented by the recipe.
- `index_skip_topk_offset: 1`: semantics not read.
- DSpark acceptance in this study's workload (pi / opencode sessions): 3.08
  is coding prompts at 4,096 output tokens; AgentX's 3.62 and SWE-bench Pro's
  2.63 bracket it.
- Decode step time: no measurement; DCP8's 7–25% step penalty is not priced.
- The Hopper row (656 B): from the verifier's code, unexercised; no 8-GPU
  H200 node holds the weights.
- Image input is not priced: the 0.45B vision tower is resident only.
- vLLM support is an open PR and a docker image; the drafter is a preview.
