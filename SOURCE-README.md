---
license: apache-2.0
pretty_name: CC Traces — Weka, With Subagents, v7 only (Jun 21 2026)
task_categories:
  - text-generation
tags:
  - llm
  - inference
  - benchmarking
  - kv-cache
  - agentic
  - multi-turn
  - claude
  - subagents
size_categories:
  - n<1K
configs:
  - config_name: default
    data_files:
      - split: train
        path: traces.jsonl
---

# semianalysisai/cc-traces-weka-062126

WekaTrace corpus derived from SemiAnalysis Claude Code proxy traces. Built 2026-06-21 17:48:24 UTC via `utils/agentic/build_weka_hf_dataset.py`.

## Filters

- Trace version: exactly v7
- min Anthropic requests per session: 20
- Claude Code CLI ≥ 2.1.139 (every row)
- peak concurrent sub-agent groups ≤ 10
- Non-image rows only (image content excluded at source)
- Classifier calls excluded (`max_tokens<=64 AND no tools` → SUGGESTION MODE, title-gen, Security Monitor)
- Exact-duplicate proxy rows deduped by `(timestamp, model, in, out, dur_ms, agent_id)`
- Dynamic-workflow-bug sessions excluded: Claude Code CLI < 2.1.174 emitted dynamic-workflow subagents without a subagent-label header, so they appear as many interleaved unlabeled trajectories in one session. Dropped when peak concurrent unlabeled multi-turn trajectories ≥ 3 (changelog 2.1.174).
- Per-request ISL cap: input ≤ 990,016 tokens (weka `in` = hash-block count × 64). Drops KV-cache-block overcount artifacts that read above ~1M while preserving the surviving timeline.

## ISL cap — KV-cache-block overcount note

Requests whose input length (`in`) exceeded **990,016 tokens** (the closest multiple of 64 to 990k) were dropped from this build, while preserving the surviving timeline's relative timestamps.

**Why.** The weka `in` field is derived from the proxy's `hash_token_count`, which is recorded as *(number of 64-token KV-cache prefix blocks) × 64* — it is always an exact multiple of 64, not a true tokenizer count. For very large agentic contexts with heavy prompt caching this block count drifts **above** the real prompt size: cross-checking against the billed token columns (`input + cache_read + cache_write`) shows it tracks ~1.00× on average but overcounts by as much as ~260k tokens in the heavy-cache-write tail. That tail pushed a small number of requests past 1M `in` even though **no request's real prompt actually exceeds 1M**. These inflated rows are the artifacts removed here.

Applied at request granularity (main-agent turns and sub-agent inner requests evaluated independently); a sub-agent group is dropped only if *every* inner request was over the cap.

## Stats

```
traces:             393
main_turns:          56,798
subagent_groups:           1,697
subagent_inner_requests:          42,029
total_model_requests:          98,827
total_input_tokens:  21,635,381,376
total_output_tokens:     106,474,498
```

## Distribution plots

### Main-agent stream

![Main-stream distributions — log x](plots/distributions_log.png)

![Main-stream distributions — linear x](plots/distributions_linear.png)

### Sub-agent fan-out

![Sub-agent distributions — log x](plots/subagent_distributions_log.png)

![Sub-agent distributions — linear x](plots/subagent_distributions_linear.png)

## Source script

```
python utils/agentic/sample_proxy_traces.py --out '<workdir>/proxy' --sampling top --min-trace-version 7 --max-trace-version 7 --min-requests 20 --require-cli-min 2.1.139 --max-parallel-subagents 10 --exclude-dynamic-workflow-bug
```

## Loader plugin

Load in aiperf via:

```
--public-dataset semianalysis_cc_traces_weka_with_subagents
```
