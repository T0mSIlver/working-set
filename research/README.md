# research/

The sourced derivations behind the model's constants, and the dated studies
that tested them. Numbers that code or the explorer read live in data files
(`src/workingset/model.py` for model and GPU constants today, `data/external/`
for numbers from outside); a note derives or tests a number, and cites the
data file rather than restating it elsewhere.

## Constants: one note per thing the model prices

| note | owns |
|---|---|
| [model_qwen38_27b.md](model_qwen38_27b.md) | `MODELS["27B"]`, the calibrated baseline |
| [model_35ba3b.md](model_35ba3b.md) | `MODELS["35BA3B"]` |
| [model_mistral_medium35.md](model_mistral_medium35.md) | `MODELS["MM35"]` |
| [model_glm52.md](model_glm52.md), [model_glm53.md](model_glm53.md) | `MODELS["GLM52"]` (GLM-5.3; the derivation is in the GLM-5.2 note, the 5.3 note checks it still holds) |
| [model_glm53flash.md](model_glm53flash.md) | `MODELS["GLM53F"]` |
| [model_dsv4flash.md](model_dsv4flash.md) | `MODELS["DSV4F"]` |
| [model_dsv4flash_vision.md](model_dsv4flash_vision.md) | `MODELS["DSV4FV"]` |
| [model_dsv41flash.md](model_dsv41flash.md) | `MODELS["DSV41F"]` |
| [model_qwen38flashnext.md](model_qwen38flashnext.md) | `MODELS["Q38FN"]` |
| [model_mistral_large4.md](model_mistral_large4.md) | `MODELS["ML4"]` |
| [model_kimi_k31pro.md](model_kimi_k31pro.md) | `MODELS["K31P"]` |
| [gpu_catalogue.md](gpu_catalogue.md) | Nvidia's datacenter lineup: specs, rental bands, interconnect, the source for new `GPUS` rows |
| [gpu_b300.md](gpu_b300.md) | `GPUS["B300"]` (the H200 row is the calibration anchor, see [docs/writeup.md](../docs/writeup.md)) |
| [prefill.md](prefill.md) | MFU, the prefill FLOP model, TTFT |
| [spike.md](spike.md) | the cold-spike and queueing model |
| [kv_tp_sharding.md](kv_tp_sharding.md) | KV layout across tensor-parallel ranks |
| [nvfp4.md](nvfp4.md) | NVFP4 weight bytes and the Blackwell gate |
| [power.md](power.md) | power draw and energy cost |
| [terminal_bench.md](terminal_bench.md) | the quality axis (Terminal-Bench scores) |

## Studies and checks: dated, appended to, never rewritten

| note | what it tested |
|---|---|
| [public_traces.md](public_traces.md) | the cache rule against public prefix-cache traces |
| [agentx_kv_hits.md](agentx_kv_hits.md) | SemiAnalysis AgentX hit rates, predicted from their traces |
| [aa_concurrency.md](aa_concurrency.md) | AA AgentPerf-Local at 1–8 agents on one RTX 3090 |
| [replay_gaps.md](replay_gaps.md) | the waits inside recorded agent sessions |
| [nvfp4_2026-09.md](nvfp4_2026-09.md), [nvfp4_2026-10.md](nvfp4_2026-10.md) | NVFP4 checkpoint sweeps |

## Open questions

| note | question |
|---|---|
| [nvfp4_prefill_dsv41f.md](nvfp4_prefill_dsv41f.md) | does NVFP4 speed up DeepSeek-V4.1-Flash prefill? |

## Outside sources

One note per external source under [sources/](sources/): what it measures,
how its data file under `data/external/` was read, and how to refresh it.

## Writing a note

- A note that sets a constant says which one in its first lines, and the
  code comment beside the constant links the note instead of repeating its
  argument.
- A number read from outside gets a record in `data/external/` (URL, read
  date, kind); the note cites the record id.
- Studies get a date in their first section and are appended to, never
  rewritten. When a study changes a constant, the constant's note says so
  and links the study.
- Work-deployment material goes to the private repo, never here.
