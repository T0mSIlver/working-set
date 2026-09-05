# Qwen3.8-27B — the baseline's successor, checked tensor by tensor

**Purpose:** the study's calibrated baseline was Qwen3.6-27B. Qwen3.8-27B
(released 2026-08-14) scores 79.8% on Terminal-Bench 2.1 against 60.7%, and
the owner asked whether it can take the baseline's place without touching a
constant. This note is the check. Verdict: **yes for everything the
architecture determines; one measured value (MTP acceptance) is carried
over unmeasured.**

## 1. What was compared (read 2026-09-05)

| Artifact | Qwen3.6-27B | Qwen3.8-27B | Result |
|---|---|---|---|
| `config.json` (`Qwen/*-27B-FP8`) `text_config` | 34 fields | 34 fields | **identical, 34/34** |
| `config.json` `vision_config`, top level | — | — | identical except `transformers_version` (4.57.1 → 5.8.0.dev0) |
| `architectures` / `model_type` | `Qwen3_5ForConditionalGeneration` / `qwen3_5` | same | same vLLM path |
| `model.safetensors.index.json` tensor names | 1,606 | 1,606 | identical sets |
| safetensors shard headers (all 66 shards, HTTP range reads) | dtype + shape per tensor | same | **0 differences**; parameter sum 27,782,935,472 both |
| FP8 checkpoint bytes | 30,866,866,928 (66 shards) | 30,866,866,928 (66 shards) | identical |
| `generation_config.json` | sampling, T=1.0, top-k 20, top-p 0.95 | same | identical |
| License | Apache-2.0 | Apache-2.0 | same |

The HF model API reports a 1,507,520-parameter difference in its
BF16/F8 breakdown (3,083,727,792 vs 3,082,220,272 BF16). The shard headers
say otherwise — every tensor's dtype and shape matches and the totals agree
exactly — so that figure is an indexing artefact, not a checkpoint
difference.

Sources: `https://huggingface.co/Qwen/Qwen3.6-27B-FP8` and
`https://huggingface.co/Qwen/Qwen3.8-27B-FP8` (`raw/main/config.json`,
`raw/main/generation_config.json`, `resolve/main/model.safetensors.index.json`,
each `model-000NN-of-00066.safetensors` header via HTTP range reads);
`https://huggingface.co/api/models/<repo>?blobs=true` for sizes, dates and
the parameter breakdown this note disbelieves; `https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4`
for the community NVFP4 size; `https://artificialanalysis.ai/models/qwen3-8-27b`
(embedded dataset, `terminalbenchV21`, effort variants) for the scores.

Architecture-determined constants that therefore carry over untouched:
`kv_bpt` (32 KiB), `deltanet_state` (75 MiB), `w_resident` /
`w_decode_shared` (28.8 GiB FP8), `params_prefill`, `attn_layers`, `attn_d`,
`max_ctx`, and every calibration anchored on them (activation reserve, pool
tokens, the MFU(chunk) anchor, decode MBU).

## 2. What is not the same

- **Weights.** Different values, hence the score. Nothing in the memory,
  prefill or decode model reads a weight value.
- **MTP acceptance (`mtp: 2.94`).** Measured 2026-08-28 on the production
  27B deployment that `decode_mbu.md` records as Qwen3.6-27B. Draft
  acceptance is a property of the MTP head's weights against the base
  model's, so it does not carry over by architecture. Kept at 2.94 and
  marked unmeasured; re-measure on a confirmed 3.8 deployment (the
  harness's per-position counters do this).
- **Which checkpoint the production deployment ran — owner to confirm.**
  `prefill.md` § "First measured calibration point (2026-08-27)" records
  the MFU anchor on "a Qwen3.8-27B FP8-weight checkpoint"; `decode_mbu.md`,
  one day later on what reads as the same 4×H200 TP4 deployment, records
  Qwen3.6-27B. One of the two labels is likely wrong. If the deployment was
  3.8, the MTP/MBU pair is a 3.8 measurement and the caveat above lifts; if
  3.6, the MFU anchor's label in `prefill.md` needs correcting. Nothing
  numeric depends on the answer (the two checkpoints are tensor-identical
  in shape), only the provenance wording.
- **Reasoning effort ladder.** Qwen3.8-27B exposes effort levels; AA ran
  low (67.4%), medium (65.2%), xhigh (79.8%, the index run) and
  non-reasoning (49.1%). The ledger takes xhigh per the variant rule. The
  study's tokens-per-request (400, a production trace on 3.6) were not
  measured at that effort; a higher effort emits more tokens per turn.
- **NVFP4 checkpoint.** `nvidia/Qwen3.6-27B-NVFP4` was measured (21.92e9 B).
  No NVIDIA NVFP4 of Qwen3.8-27B exists as of 2026-09-05; the community
  `unsloth/Qwen3.8-27B-NVFP4` keeps more tensors in FP8 (23.42e9 B, a
  different recipe). The explorer's NVFP4 arm kept the NVIDIA recipe's
  bytes as a projection until 2026-09-06, when `RedHatAI/Qwen3.8-27B-NVFP4`
  (23,417,339,744 B, measured) replaced it — `research/nvfp4_2026-09.md`.
- **Release.** 2026-08-14 vs 2026-04-22; `transformers` 5.8 at export.

## 3. What changed in the repo

`CONFIG.MODELS["27B"]` is named Qwen3.8-27B with the comments above;
`CONFIG.QUALITY["27B"]` is 213/267; `scenario_model.py`'s mirror, the
model button, the method page, the README and the harness's default
endpoint follow. Measurements in `docs/scenarios.md`, `decode_mbu.md`,
`prefill.md` and `workload_agentic_poc.md` keep saying Qwen3.6-27B: that is
the model they were taken on.
