# Mistral Large 4 Preview — what is known, and why it is not in the model yet

**Status (2026-10-06, release day):** API-only public preview. Weights are
promised for the end of October (VentureBeat gives 2026-10-27); the license
is "coming soon". No `config.json`, no checkpoint, no vLLM or Transformers
code is public, so none of the constants `model.py` needs can be read.
**Not integrated.** Revisit when the weights land.

## 1. Published facts

| Field | Value | Source |
|---|---|---|
| Parameters | 1.05T total, 49B active, plus a 1.6B vision encoder | [docs model card](https://docs.mistral.ai/models/mistral-large-4-0) |
| Architecture | "granular Mixture-of-Experts", natively multimodal input, text output, hybrid instruct/reasoning | model card; [launch post](https://mistral.ai/news/mistral-large-4/) |
| Context | 1M (Mistral); 524K at Artificial Analysis and Vercel, 512K at Vals — unresolved | model card; [kingy.ai](https://kingy.ai/blog/mistral-large-4-specs-benchmarks-pricing/) |
| Training | from scratch, 3,800 Grace Blackwell GPUs, ~2 months, 160+ languages | launch post |
| API price | $0.68 in / $0.07 cached / $2.09 out per M tokens (half the launch rate of $1.36 / $0.14 / $4.18) | model card |

Not published: expert count and top-k, shared experts, layer count, hidden
size, attention type (MLA, GQA, sparse, sliding window), KV bytes per token,
weight dtype of the release checkpoint, MTP or draft model.

## 2. What `model.py` needs and what is missing

Every field of `Model` is unknown: `kv_bpt` and `kv_decode_*` (attention
design), `w_resident` (checkpoint dtype), `w_decode_shared`,
`w_route_pertok` and `w_route_total` (expert count, size and top-k),
`params_prefill`, `attn_layers`, `attn_d`, `kv_heads`. The study's rule is
that weight and cache constants come from the checkpoint's own headers, not
from marketing totals, so a placeholder row would rank the frontier on a
guess.

Mistral Large 3 is a weak prior. Its NVFP4 `params.json` is a DeepSeek-V3
shape: MLA (`kv_lora_rank` 512 + rope 64 = 576 B/layer, 61 layers, so
35,136 B/token in fp8), 128 experts with 4 routed and 1 shared. "Granular"
MoE in the Large 4 card suggests many smaller experts instead, so the
expert-union kink and the per-token routed read cannot be carried over.

## 3. Bounds from the totals alone

- **Resident weights.** FP8 ≈ 1.05 TB. That leaves ~75 GB of 8×H200's
  1,128 GB for KV and activations, so 8×H200 is a marginal fit at best;
  8×B300 (2.3 TB) fits. Mistral shipped NVFP4 checkpoints for Large 3 and
  Small 4; an experts-only NVFP4 repack would land near 0.6 TB, i.e. 3–4×B300.
- **Decode weight read.** 49B active vs GLM-5.3's 40B: about 1.2× the
  bytes per step at small batch on the same hardware, before attention.
- **Size vs GLM-5.3.** 1.4× the total parameters (1.05T vs 744B).

## 4. Quality vs the study's models

The study scores quality with Artificial Analysis runs only
(`research/terminal_bench.md`). AA lists the preview (Intelligence Index 38,
116 tok/s) but shows no Terminal-Bench 4.0 score yet.

| Terminal-Bench 4.0 | Score | Source |
|---|---|---|
| GLM-5.3 | 41.9% (83/198) | AA |
| GLM-5.3-Flash | 32.8% (65/198) | AA |
| DeepSeek-V4.1-Flash | 31.2% | vendor card |
| **Mistral Large 4** | **28.3%** | Mistral launch claim |
| Qwen3.8-Flash-Next | 25.3% (50/198) | AA |
| **Mistral Large 4** | **22.7%** | Vals (different harness) |
| DeepSeek-V4-Flash-0731 | 12.1% (24/198) | AA |

On Mistral's own figure, Large 4 trails GLM-5.3 by 13.6 points while
needing ~1.4× its weights. Other launch claims: DeepSWE v1.1 61.7% (GLM-5.3
61%, per Mistral's chart), SWE-Atlas-QnA 59.4%; its claimed lead is in
cybersecurity (CyberGym-E2E 82%) and legal/finance agents, outside what the
study measures.

## 5. Integration checklist (when weights land)

1. Read `config.json`, `params.json`, the safetensors index and every shard
   header; build the per-tensor ledger as in `model_glm53flash.md`.
2. Note the vLLM recipe's KV dtype constraints and any NVFP4 checkpoint.
3. Add the row to `model.py`, mirror it in `interactive/src`, regenerate
   `tests/golden/`.
4. Take the Terminal-Bench scores from AA once it publishes them.
