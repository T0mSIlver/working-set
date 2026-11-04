# Outside sources: what each one can tell the model

Every number from outside lands in `data/external/` with its URL, read date,
kind and use (format: [data/external/README.md](../../data/external/README.md)).
This note says which use each source supports. Read 2026-11-04.

## Three uses

- **calibrate**: the record states model, GPU count, engine, weight dtype,
  draft depth and context, and the workload is known. A fit over such records
  may set a constant (decode bandwidth efficiency and fixed step time from a
  per-user speed curve; MFU from TTFT). A vendor-tuned recipe is an upper
  bound for a default vLLM, never a centre. Same model and GPU as one of ours:
  the fit can replace that row's constant. Same GPU, other model: a prior
  centre with a ×/÷1.5 band. Other GPU: no update.
- **check**: the model predicts the same quantity under the same conditions.
  The page reports reported ÷ predicted with the model's own band. A check
  never moves a constant; when one GPU's rows all miss by more than ×1.5, it
  opens an issue naming the constant the miss loads on.
- **context**: shown with its provenance, never fitted or checked. Press
  figures about deployments, prices, calculator assumptions.

A record whose workload is unknown never calibrates. A record can carry
metrics of different uses (`use` per metric).

## Sources

| source | what it measures | use | public file |
|---|---|---|---|
| [InferenceX](inferencex.md) | per-user vs per-GPU tok/s curves per model × GPU × engine × concurrency; AgentX trace replay with prefix cache | calibrate (curve points), check (derived MBU) | yes, cited |
| [MLPerf Inference v6.0](mlperf.md) | peak tok/s under fixed TTFT/TPOT bounds, single-turn | context: a bound on decode throughput per GPU, never plotted against model output (MLCommons rules) | yes, cited |
| [Artificial Analysis AgentPerf](aa-agentperf.md) | max concurrent coding agents per system at speed/TTFT tiers | check, run locally only | **no**: terms forbid redistribution and comparison products |
| [Deployments](deployments.md) | press and operator figures (Revolut, DeepSeek) | context | yes, cited |
| [Calculators](calculators.md) | self-host vs API break-even tools | context: GPU price and overhead defaults | yes, cited |

## Checks run so far

**Single-stream decode, InferenceX** (records in `inferencex.json`, model
defaults, context matched). Reported ÷ predicted, MBU fold / latency form:
GLM-5.3 8×B300 NVFP4 0.95 / 1.13; DeepSeek-V4.1-Flash 4×B300 1.17 / 2.06,
8×H200 0.59 / 0.92; Qwen3.8-Flash-Next 4×B300 0.41 / 0.88, 4×H200 0.49 / 0.70.
The MBU fold over-predicts small-active MoEs by ~2× because it has no fixed
step time; the latency form's flat 2.7 ms fixed term under-predicts
DeepSeek-V4.1-Flash on B300.

**Revolut** (`deployments.json`, context). 56 B200, GLM-5.3 NVFP4, ~2,000
engineers, ~200 tok/s per stream at peak, under 3 s TTFT at ~256K. With a
B300 row cut to 180 GB as the B200 stand-in, the model carries ~79 users
per 8-GPU box at the 40 tok/s floor, and its load-per-box at the reported
">7 req/s" (secondary source) is ~1 req/s, where it predicts ~325 tok/s per
stream. The numbers fit any peak active share from ~7% to ~50% of headcount,
so they pin neither the workload nor the hardware. The one sharp gap is
prefill: the model's cold 262K TTFT is 16 s because it prices GLM-5.3
attention as dense; Revolut and the vLLM blog's GLM-5.3 sparse-prefill
numbers say it is several times faster.
