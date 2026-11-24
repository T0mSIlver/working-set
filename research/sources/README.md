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

**Revolut** (`deployments.json`, context; transcript read 2026-11-24). 56 B200, GLM-5.3 NVFP4, ~2,000 engineers, 3 then over 7 req/s on the same GPUs, ~200 tok/s per stream at peak, $0.30 per million "blended". The B200 stand-in is the B300 row with 192.3e9 B per GPU and `reserve_extra` from the same formula (13.07e9 B), unmeasured. Workload: 120K-token prompts, 2K fresh, 300 out, 60 s think, TP8, `max_model_len` 262,144.

- Capacity fits only with one cache copy per group. With DCP an 8×B200 box holds a 16.8M-token pool, 123 warm sessions, and its capacity is 95 sessions (cold request waits 5 s). With plain TP8 every rank keeps a copy: 2.1M tokens, 13 sessions. At this workload 7 req/s is about 420 sessions, 60 per box. The April point (3 req/s, 14 s TTFT) reads as a cache that evicts, and the fixes the talk credits (FP4, TP8 over TP4, a context cap) all enlarge the pool.
- The cost fits only if cache reads count. At InferenceX's B200 prices ($1.73 owned, $3.70 rented per GPU-hour), $0.30 per million buys 7.8 to 16.6 billion tokens a day: 7 req/s of 120K prompts for 3 to 5 hours, 22 to 48% of InferenceX's tuned GLM-5.3 B200 rate. Fresh input and output are 2.3K of a request's 120K tokens, so without cache reads the price would be ~50 times higher.
- The one-tenth invoice does not follow from per-token prices. `ws api` (prices of 2026-11-04) blends the same workload at $0.20 per million on GPT-6.1 Sol, $0.24 on Claude Sonnet 5.5, $0.35 on the GLM-5.3 API, $0.48 on Claude Opus 5.5 and $0.95 on Claude Fable 5.1.
- Decode reads 1.4 to 1.7× above the model: 470 tok/s for a lone stream and 110 to 145 at 13 to 95 streams, against 200 under load and 800 idle. eta was never measured on Blackwell; a held-batch sweep there would settle it.
- Prefill stays the gap noted on 2026-11-04: the model prices GLM-5.3 attention as dense, so its cold 120K TTFT (5.8 s) is an upper bound.

**Baseten's DeepSeek engine** (https://github.com/basetenlabs/deepseek-ultrafast, read 2026-11-24; context, no record). A hand-written Rust/CUDA engine for DeepSeek-V4.1-Flash, measured on one stream of a short coding prompt with thinking off: 2.71 ms per step on 8×B300 (TP8) and 3.30 ms on 4×GB300 (TP4), 5.6 tokens accepted per step, so 2,060 and 1,740 tok/s; 1M cold tokens in 11.7 and 13.2 s. The README gives no batched figure, so it says nothing about capacity under load.

- Decode: the model's step on 4×B300 is 4.28 ms at 3 drafts and 7.91 ms at 8 (Baseten's draft count is not stated; 5.6 accepted needs at least 6). The engine's step is 1.3 to 2.4× shorter than the model's, and its acceptance on Tetris is far above the catalogue's 0.777 (3.5 tokens per step at 5 drafts). The model's single-stream rate, 608 tok/s at 5 drafts, sits below this engine; the eta it uses was measured on vLLM.
- Prefill: the model gives 1M tokens in 3.3 s on 4×B300 and 1.9 s on 8×B300, 4 to 6× faster than this engine. Since a hand-tuned engine is an upper bound for vLLM, the model's DeepSeek-V4.1-Flash prefill is too fast (WS-50).
