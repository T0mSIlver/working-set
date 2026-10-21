# AA AgentPerf-Local at 1, 2, 4 and 8 concurrent agents on one RTX 3090

Job 38. Artificial Analysis' `agentperf-local` (v0.3.4, 1fe8056, Apache-2.0)
replays eight agent tasks, 168 turns, one request at a time; its article says
multi-agent coverage is planned. Here N copies run at once against one
llama-server (`scripts/aa_concurrency.py`), lane k starting k/N of the way
through the task list, each with its own cache namespace. `ws aa predict`
prices the same lanes on a model of the server (`src/workingset/aa_local.py`).

## The workload (`ws aa stats`)

168 turns, 8 tasks, 2,499,441 prompt tokens, 30,883 output tokens, largest
prompt 56,000. 93.3% of prompt tokens repeat the lane's previous request; the
SWE-bench tasks rewrite a trailing "Tool-call turn N of 100" message each
turn, so their reuse ends at the start of the last user message, not at the
previous prompt's end.

## The box (sandbox, 21 Oct 2026, phase 1)

- RTX 3090, 24,576 MiB; at 19:00Z 9,015 MiB held outside the sandbox
  container at 0% use (holder not visible from inside).
- `/models/gguf-manual/Qwen3.8-27B-UD-Q4_K_M.gguf`, 16,464,440,224 B (Unsloth
  dynamic Q4_K_M; AA's recipe pins the plain Q4_K_M, 17,106,775,008 B, so the
  run is an attached-server `run`, not `managed-run`).
- llama-server master 0a1ef309c (build 11796, CUDA). Read from its source:
  `--kv-unified` is on by default only with `-np` auto; `--cache-idle-slots`
  (on) saves every idle slot to the 8,192 MiB `--cache-ram` and clears it
  from VRAM when a request starts; KV full clears the first idle slot, then
  halves the batch, then fails every active slot with "Context size has been
  exceeded"; hybrid models checkpoint at the last user message's start and at
  4 + n_ubatch and 4 tokens before the prompt's end (32 per slot).

## Predictions, pinned before any load ran

Priors for single-stream speed (no measurement on this box yet): decode
35 tok/s, prefill 1,100 tok/s, the llama.cpp RTX 3090 class for 27-32B dense
Q4_K_M. Everything else is the model's: weights read per ubatch, every
sequence's KV and 151 MB recurrent state read per step, compute stretched by
attention context, the checkpoint rule above, the pool.

Server for the run: `-np 8 -kvu -c 180224 -fa on -ctk q8_0 -ctv q8_0`, the
same for every N. Raw: `aa_concurrency/prior-*.json`.

q8_0 KV, pool 180,224 tokens (`--box qwen38-27b-q4km-3090-q8kv`):

| N | wall s | output tok/s, all lanes | tok/s per stream | TTFT p50 s | TTFT p95 s | prefix hits | failed turns |
|---|---|---|---|---|---|---|---|
| 1 | 1,119 | 27.6 | 34.1 | 0.46 | 4.46 | 92.5% | 0 |
| 2 | 1,372 | 45.0 | 28.7 | 0.46 | 4.47 | 92.3% | 0 |
| 4 | 1,869 | 66.1 | 21.9 | 0.46 | 5.48 | 92.4% | 0 |
| 8 | 5,824 | 22.2 | 12.9 | 0.51 | 34.84 | 83.2% | 176 of 1,344 |

The claim to test: four agents is this box's ceiling. Four lanes peak at
165k live tokens, inside the pool; eight need ~304k (the same run with an
unbounded pool), so the server runs out of KV, fails 13% of turns and does
less useful work than two lanes. With an unbounded pool N=8 would reach
86 tok/s; the gap is the price of 24 GB.

f16 KV, pool 110,000 tokens, for contrast (`--box qwen38-27b-q4km-3090`):
N=1 26.6, N=2 42.5, N=4 29.3 tok/s with 48 failed turns, N=8 6.2 tok/s with
712 failed. The KV dtype moves the ceiling from two agents to four.

Sensitivity: no roofline overlap (`--combine sum`) costs 4-9%; lanes in
lockstep (`--no-rotate`) fail 16 turns at N=4 and 256 at N=8, because every
lane hits the 44k-56k meeting tasks at once.

## What the run records

Per lane, `turns.jsonl` (TTFT, generation time, server prompt, cached and
uncached tokens). `ws aa measured runs/` gives the same columns; then
`ws aa predict --anchor runs/n1/lane0` re-prices N=2-8 on the measured
single-stream speeds.
