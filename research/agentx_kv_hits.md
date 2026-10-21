# Predicting AgentX's KV hit rates from the public traces — running log

Goal: predict the two hit rates SemiAnalysis measured for DeepSeek V4 Pro on
AgentX, from their public traces and a model of the server's caches, before
looking at anything else they measured. Blog of 2026-08-25
(`InferenceX-app/packages/app/content/blog/deepseek-v4-pro-agentx-b200-vs-b300-kv-working-set.mdx`):

| point | concurrency | HBM hit | DRAM hit | "HBM KV working set" |
|---|---|---|---|---|
| B300 vLLM DEP8, 3 TB DRAM, simple offload | 384 | 91% | 1.36% | ~43M tokens |
| B200 vLLM DEP8, same parameters | 196 | 73% | ~20% | 22M tokens |

InferenceX #1632 (closed 2026-06-15) computed only the unlimited-cache
ceiling. Nobody has published the finite-capacity curve. That curve is the
point of this work; matching two numbers is the test of it.

Public data only. Tom's own agent traces are not used anywhere here.

## 2026-10-21 — what the runs were, exactly

Read from the InferenceX repo at `49ac5c0` (2026-08-21, the run date in the
blog's dashboard links), `configs/nvidia-master.yaml` and
`benchmarks/single_node/agentic/dsv4_fp4_b{2,3}00_vllm_mtp.sh`:

- Model `deepseek-ai/DeepSeek-V4-Pro`, FP4 experts, `--kv-cache-dtype fp8`,
  `--block-size 256`, hybrid KV cache manager on, `--max-model-len 1048576`,
  MTP with 3 speculative tokens at a synthetic acceptance length of 2.49.
- DEP8: `--data-parallel-size 8`, expert parallel, `vllm-router 0.1.14`
  with `--policy consistent_hash` on `X-Session-ID` = AIPerf's correlation
  id. Every turn of a stream lands on one rank; each rank has its own cache.
- `--max-num-seqs 2*CONC/8` per rank (96 on B300 at 384, 49 on B200 at 196),
  `--max-num-batched-tokens` 16384 (B300) / 8192 (B200), long-prefill
  threshold 512, `--gpu-memory-utilization` 0.92 (B300 DEP8) / 0.90 (B200).
- DRAM: `SimpleCPUOffloadConnector`, eager (write-through) offload,
  `cpu_bytes_to_use` = TOTAL_CPU_DRAM_GB / 8 per rank. B300 (cluster:b300-nv,
  dram-utilization 0.95): 356 GB per rank, 2,849 GB total, per the script's
  own comment. B200 (cluster:b200-dgxc): node DRAM not stated in the repo;
  a DGX B200 has 2 TB, which would give ~237 GB per rank (assumption).
- `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=32768`: sliding-window prefix tails
  kept sparsely (vllm #43447). Not modelled yet; see open questions.

Client (agentx-harness 1.0.12 on PyPI, scenario `agentx`): one hour, each
lane starts mid-session at 25-75% of the session, primes the cache with each
live stream's last request before that instant, then replays; idle gaps
capped at 300 s; first-turn prefix cache-busted (no sharing across sessions);
a lane whose session tree drained starts a fresh session. Concurrency counts
session trees, so sub-agents add requests above it. Hit rates come from
vLLM's `prompt_tokens_by_source` (`local_cache_hit`, `external_kv_transfer`)
over the profiling window.

## 2026-10-21 — the traces

`semianalysisai/cc-traces-weka-062126`: one 1.85 GB `traces.jsonl`, 393
sessions, 98,827 requests, 21.6B input tokens (219k per request on average),
106M output. Block hashes per 64 tokens, per-session scope. huggingface.co
is blocked from the cloud box this work started on, so the sandbox
downloads it and `python -m workingset.agentx` compacts it to the prefix
structure (blocks shared with the stream's previous request, blocks shared
with any earlier request, block count, timings).

## 2026-10-21 — the simulator

`src/workingset/agentx_sim.py`, mechanisms and sources in its docstring.
Per rank: an HBM tier where running requests pin prompt + output and idle
streams lose their prefix tail-first, least recently freed first (vLLM's free
queue order); a write-through DRAM tier, LRU; a FIFO prefill server per rank.
Admission is first come first served: a request that cannot fit waits, and
its own idle prefix can be evicted while it waits.

Smoke test on synthetic sessions (not the real traces): the HBM hit rate
collapses from ~93% to ~3% between 43M and 20M tokens with no DRAM, because
missed prefills lengthen service, more contexts are pinned at once, and
eviction feeds itself. With DRAM the overall rate stays at the ceiling and
only the HBM/DRAM split moves. The real traces decide where the knee is.

## 2026-10-21 — DeepSeek V4 Pro, and a capacity puzzle

`config.json` and the safetensors dtype histogram (Hugging Face API, read
2026-10-21): 61 layers plus one MTP layer; `compress_ratios` gives 30 CSA
layers (ratio 4), 31 HCA (ratio 128), MTP layer window-only; 384 routed
experts, 6 active, hidden 7168, `index_topk` 1024, window 128.

- KV per token by the DSV4-Flash method (`model_dsv4flash.md` §2):
  30 x 576/4 + 31 x 576/128 + 30 x 64/4 = **4,939.5 B**.
- Weights: routed experts 1,572,763,336,704 logical FP4 params = 786.4 GB
  packed + 49.1 GB E8M0 scales; everything else 29.2 GB (FP8 23.2,
  BF16 5.6, FP32 0.35). Total 864.7 GB. Under DEP8 each GPU holds all the
  non-expert weights and 1/8 of the experts: **133.6 GB**.
- KV room per GPU, before activations: B300 288.4 x 0.92 - 133.6 =
  131.7 GB; B200 (~192.3 GB visible, an assumption) x 0.90 - 133.6 =
  39.4 GB.

The published working sets do not follow from these bytes. 43M/8 tokens at
4.94 kB is 26.6 GB per GPU, a fifth of the B300's room. And no single
bytes-per-token figure fits both GPUs: 43M/22M is 1.95, while the room ratio
is at least 3.3 for any shared activation reserve. So either "working set"
is not vLLM's "GPU KV cache size" sum, or vLLM's DSV4 accounting (hybrid
groups, page padding, sliding-window pages) costs far more per token than the
cache layout, and differently per GPU. Until a server log settles it, the
predictions below take the published 43M and 22M as the capacity input and
say so. This is a question for the InferenceX note.

Not added to `MODELS`: an explorer model needs its JS mirror, which is a
deploy (AGENTS.md). The constants live in `agentx_sim.DSV4_PRO` for now.

## Open questions

1. What exactly is the 43M / 22M figure (sum of `GPU KV cache size` lines?).
2. The sliding-window retention interval: does a prefix hit need a retained
   window tail, so hits round down to 32k-token points except at a request's
   own end?
3. Server tokens vs trace tokens: AIPerf re-tokenizes synthetic text with the
   DeepSeek tokenizer; assumed 1:1 here.
4. Live replies are not reused (the next prompt carries the recorded reply);
   assumed from `weka_synth_buf.advance_turn`.
