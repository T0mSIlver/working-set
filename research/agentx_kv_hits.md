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
  kept sparsely (vllm #43447). Modelled since the afternoon; see
  "sliding-window retention" below.

Client (agentx-harness 1.0.12 on PyPI, scenario `agentx`; but see the
evening's correction below: the recipe runs `inferencex-agentx-mvp`):
one hour, each lane starts mid-session at 25-75% of the session, primes the cache with each
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
say so. This is a question for the InferenceX note. (Later the same day: that
first prediction failed, and capacity now comes from bytes; see below.)

Not added to `MODELS`: an explorer model needs its JS mirror, which is a
deploy (AGENTS.md). The constants live in `agentx_sim.DSV4_PRO` for now.

## 2026-10-21 — sliding-window retention, now modelled

vllm #43447 (closed, merged into the image these runs used, per the B200
script's comment): DeepSeek V4's sliding-window blocks, freed as each window
moves, used to flush other requests' prefixes out of the free queue. The fix
puts uncached blocks at the front of the queue, and
`VLLM_PREFIX_CACHE_RETENTION_INTERVAL` keeps window tails only at the
latest prompt boundary and every N tokens. With N = 32768 a request resumes
at the previous prompt's end if it is still whole, else at the last 32k
multiple below the shared prefix. The simulator applies that rule to both
tiers and reports a second ceiling (`ceiling_retained_rate`): what an
infinite cache could reuse under the rule.

Two numbers from that PR worth keeping: on TP8 8xB300, the reported KV
capacity was ~14.5x concurrency for 1M-token requests (~14.5M tokens), and
an AgentX-style replay at concurrency 16 went from 0% to 74.3% hits with the
fix. Related, open: vllm #47922 (a native CPU-only KV-cache simulator inside
vLLM, no MLA/DSV4 support yet) and #55403 (opportunistic hits on masked
blocks).

## Open questions

1. What exactly is the 43M / 22M figure (sum of `GPU KV cache size` lines?).
2. Retention interval: does SimpleCPUOffload store only reachable blocks, as
   Mooncake does after vllm #44774? Assumed yes.
3. Server tokens vs trace tokens: AIPerf re-tokenizes synthetic text with the
   DeepSeek tokenizer; assumed 1:1 here.
4. Live replies are not reused (the next prompt carries the recorded reply);
   assumed from `weka_synth_buf.advance_turn`.

## 2026-10-21 — the traces, read

Compact file from the sandbox (`data/agentx-062126` branch, SHA 3e48d96,
2.2 MB; source sha256 matches Hugging Face's etag): 393 traces, 98,827
requests, 1,697 sub-agents, 21.6B input tokens. Median prompt 142k, p90 550k;
mean output 1,077. Recorded gaps between a root request's end and the next
start: median 4.9 s, p90 132 s, p99 54 min (capped at 300 s by the replay).
No request carries `stop`, so that column is empty; `think_time` is missing
on the first request of each trace (unused here).

Infinite-cache ceilings over all requests: 95.7% if a request can only reuse
its own stream's previous prompt, 98.3% if it can reuse anything earlier in
the trace (sub-agents forking the parent's prefix). Inside the replay window
and under the 32k retention rule the same-stream ceiling is 92-93%. The
measured overall rates (HBM + DRAM) are 92.4% on B300 and ~93% on B200: both
runs sit on that ceiling, which says the replay reaches little cross-stream
reuse. Retention itself costs little: 93.1% without it against 92.8% with it
(B300 window). The rest of the gap to 95.7% is the replay window: recycled
lanes start sessions from turn 0, cold.

Where the cross-stream reuse is: root requests could reuse 97.5% of their
prompt tokens from their own previous request and 98.8% from anything
earlier; sub-agent requests 87.5% against 96.1%. The sub-agent gap is the
parent's prefix, and consistent-hash routing on the sub-agent's own session
id sends it to another rank 7 times in 8. The simulator counts no
cross-stream hits at all, so it under-counts by the 1-in-8 case.

## 2026-10-21 — first prediction: 43M as capacity fails (superseded)

Taking the published 43M / 22M as HBM capacity predicts 43% and 39% HBM
hits, with thousands of requests queueing for KV space. The contexts are too
large: 48 sessions per rank at a 142k median prompt pin more than 5.4M
tokens. So the working set is not the capacity the cache runs on.

## 2026-10-21 — capacity from bytes, one public anchor

vllm #43447 states ~14.5M tokens of KV on TP8 8xB300 at 0.90 utilization,
where every rank holds the full cache. With the weights known (108.1 GB per
GPU under TP8) that fixes vLLM's cost per token: (288.4 x 0.90 - 108.1 - R)
/ 14.5M = 9.8 kB at a reserve R of 10 GB, about twice the 4.94 kB of the
layout. That bytes-per-token figure is itself fitted to this one anchor, and
the reserve R stays a free input (swept below). With it fixed, the same
bytes give the DEP8 capacities:
B300 99.8M tokens (12.5M per rank), B200 24.2M. DRAM at the same bytes per
token: 292M and 195M.

`ws agentx predict --sweep`, one hour, seed 42, `agentx` scenario
(superseded by the evening section, which runs the recipe's scenario over
ten seeds):

| point | measured HBM / DRAM | predicted HBM / DRAM | range over the sweep |
|---|---|---|---|
| B300, 384 | 91.0% / 1.4% | **90.7% / 2.1%** | 89.7-91.3% (service x0.5-x2, R 5-15 GB, 4 seeds; not the published-working-set-as-HBM row, 43%) |
| B200, 196 | 73.0% / 20.0% | **47.8% / 44.2%** | 31-69% (superseded below: seed 42 was the luckiest of ten) |

B300 is predicted to within half a point and the answer barely moves with
any assumption: the cache holds about twice what the lanes pin, so hits sit
on the ceiling and DRAM hardly matters (91.9% with no DRAM at all).

B200 misses by 25 points and the sweep says why: it sits on the knee. Its
KV room is ~30 GB per GPU after 133.6 GB of weights, so 5 GB of reserve
moves the prediction from 59% to 34%, and which 196 sessions land in the
lanes moves it from 31% to 48%. The capacity sweep puts 73% at about 30M
tokens; the bytes give 21-27M. Something worth ~6 GB per GPU is wrong in the
B200 inputs (visible VRAM, the reserve, or the weights a DEP rank holds), or
vLLM preempts and recomputes where this model queues. Either way, a 196-lane
point on a 24M-30M-token cache is one where small changes swing the hit rate
by tens of points, and the published 73% is one draw from that.

What the 43M and 22M are remains open: neither matches these capacities
(B300 43/99.8 = 0.43, B200 22/24.2 = 0.91).

## 2026-10-21 evening — the refuter's corrections, and the recipe's scenario

A fresh reviewer at e126cb6 returned NOT VERIFIED. What it found, and what
changed:

- **The recipe runs `inferencex-agentx-mvp`, not `agentx`.**
  `benchmark_lib.sh` (49ac5c0) calls `aiperf profile --scenario
  inferencex-agentx-mvp` with 10 warmup requests per lane: t* anywhere in
  the session (0-100%), the primers, then 10 more requests per lane with
  idle gaps removed, and only then the measured hour, each held-back turn
  keeping its recorded delay. The scenario's 10 s idle cap is whole-system
  (it fires only when nothing at all is in flight), so it never fires at
  these concurrencies. `Replay` now defaults to the recipe; `AGENTX` is the
  harness's own scenario, kept as a sweep row.
- **The B200 number was a lucky seed.** Seed 42 gave 47.8%; ten seeds give
  a mean of 41%. The miss is about 32 points, not 25.
- **73% needs ~34M tokens, not ~30M.** Capacity curve, B200, recipe
  scenario, mean of five seeds: 24.2M 41%, 28M 55%, 32M 67%, 36M 77%,
  40M 83%, 48M 90% (`ws agentx predict --point b200-196 --curve
  24.2,28,32,34,36,40,48`; 34M gives 72.6%). 73% sits at ~34M, 1.4 times the 24.2M the bytes give:
  about 12 GB per GPU of room this model does not find.
- **"Holds under every sweep" was too broad for B300.** It holds under
  every input that comes from the recipe or the bytes. Feeding the
  published 43M working set in as capacity gives 36%; that row is the
  failed hypothesis of the afternoon, not a sensitivity.
- **Queueing versus preemption.** vLLM admits a request chunk by chunk and
  preempts the newest when a running one cannot grow; this model waits
  until the whole request fits. `greedy_admission` makes the head of the
  queue evict every idle prefix it would need at once, the most destructive
  reading of vLLM's order. It changes nothing (42.1% either way): by the
  time a request waits, the idle prefixes are
  already gone and running contexts fill the cache. So admission order is
  not where the 32 points are. Recompute after preemption could count a
  request's own blocks as hits again if vLLM's `prompt_tokens_by_source`
  counts each scheduling; not checked.
- Also fixed: the block size is read from the trace, not assumed to be 64;
  idle streams freed at the same instant (all the primers) are evicted in
  insertion order, not by key. Tests now cover warmup, primers, recycling,
  spawn and join, the eviction tie-break, the capacity bound and block
  size; a second reviewer's mutations (join removed, spawn gate removed,
  primers dropped, tie-break reversed) each fail one. The module docstring now lists the
  simplifications with the direction each errs in: all three under-count
  hits.

The warmup is long: with idle gaps removed, 10 requests per lane plus the
drain take 21 minutes of simulated time on B300 and 29 on B200 before the
measured hour opens (`window opens` in the output), mostly cold recycled
sessions and decode at 45-60 tokens/s. The real run's warmup length is not
published, so this is unchecked.

Results, `ws agentx predict --sweep`, recipe scenario, one hour:

| point | measured HBM / DRAM | predicted HBM / DRAM (seed 42) | HBM over seeds 42-51 |
|---|---|---|---|
| B300, 384 | 91.0% / 1.36% | **91.7% / 1.4%** | mean 91.4%, 90.5-92.2% |
| B200, 196 | 73.0% / ~20% | **42.1% / 50.1%** | mean 41.2%, 27.7-48.4% |

Inputs from the recipe and the bytes keep B300 in 90.5-92.2% (service
x0.5-x2, R 5-15 GB, DRAM 1.5x, no DRAM, either admission, either
scenario). It is a weak test, though: the cache holds about twice what the
lanes pin, so almost any model that reuses a stream's own prefix lands on
the ceiling.

B200 overall (HBM + DRAM) is 92% predicted against ~93% measured; the split
is wrong, with three times too many prefixes served from DRAM. Inputs that
would close it, any one of them worth ~12 GB per GPU: the B200's visible
VRAM (192.3e9 assumed), a reserve under 5 GB, the weights a DEP rank
actually holds, or a vLLM cost per token on B200 below what the TP8 B300
anchor implies. The B200 run's server log (its `GPU KV cache size` line)
would settle it. That is the question for InferenceX.
