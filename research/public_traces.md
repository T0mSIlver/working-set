# The capacity model against every public prefix-cache trace

Started 2026-10-21. Append dated sections; never rewrite.

The question: does working-set's cache rule hold on agent and chat traces
that other people published, and where it misses, what fixes it. Tom's own
traces are work data and are not used here.

Code: `src/workingset/traces.py` (readers, exact LRU, Denning, the planner,
the fill rules), `ws traces mooncake|bailian|tracelab PATH`, and
`agentx_sim.planner` for the SemiAnalysis AgentX points. Tests:
`tests/test_traces.py`, `tests/test_agentx.py`.

## The datasets

| dataset | what it holds | source |
|---|---|---|
| Mooncake FAST'25 | conversation, tool-agent and synthetic traces, 512-token hash ids, no sessions; plus the earlier arxiv trace | kvcache-ai/Mooncake, `FAST25-release/` |
| Alibaba Bailian | to-C traces A and B, Qwen coder and thinking traces, 16-token salted block ids, turns chained by `parent_chat_id` | alibaba-edu/qwen-bailian-usagetraces-anon |
| TraceLab | 357k rounds of Claude Code and Codex sessions with the provider's own prompt-cache counters, no block ids | uw-syfi/TraceLab, CC BY 4.0 |
| SemiAnalysis AgentX | Claude Code sessions with sub-agents, 64-token ids, and two published hit rates on DeepSeek V4 Pro | `semianalysisai/cc-traces-weka-062126` (compact copy on branch `data/agentx-062126`) |
| KVSET | a Mattson replayer over the Mooncake traces; used as a cross-check of the exact LRU | its repo's `test_data` |

Not yet run: the Hugging Face sets (FreeInference, the LMCache agent traces)
and InferenceX's server logs. A sandbox session was fetching them to branch
`data/public-traces`; it had not pushed when this was written.

## Three ways to price a capacity

- **exact**: one LRU over prefix pages, a request's pages touched tail
  first, a page hitting only if every earlier page does (Mattson, one pass).
- **planner**: the characteristic-time rule fed only what a planner knows
  (prompt sizes, each request's gap since its session's previous request and
  how much of the prompt it could reuse). No block ids.
- **fill**: the model's rule today (`model.warm_capacity`): the cache keeps
  `C / mean prompt` sessions warm; with more live sessions than that, LRU
  over a cyclic order collapses to ~0 (docs/writeup.md, measured on one
  H200 with prompts sent in a fixed order).
- **proportional**: the fill rule with random returns. If sessions come back
  after exponential gaps, LRU keeps a returning session warm with
  probability `warm / live`, not 0 (with an exponential gap of mean m,
  s(T) = live · P · (1 − e^(−T/m)) = C gives P(gap < T*) = C / (live · P)).

## Mooncake: the exact replay reproduces the paper

Page 512 tokens (one Mooncake block). Fig 9's "% of the maximum hit rate at
3M tokens":

| trace | paper | exact LRU here |
|---|---|---|
| tool agent | 75 | 74 |
| synthetic | 48 | 48 |
| all three merged | 46 | 45 |
| conversation | 41 | 37 |

The ceilings land 1-3 points under the paper's Table 2 cache ratios. On the
older arxiv trace the exact replay sits 5-7 points above the paper's LRU
curve (0.30, 0.40, 0.48, 0.50, 0.51), the same gap an outside replay
(gauravapiscean) reported. Mooncake has no session ids, so only exact and
Denning apply; Denning lands within a point of exact (tool agent: 0.359 vs
0.359 at 1M, 0.421 vs 0.420 at 3M).

**Hit**: the replayer itself. Three of four traces within a point.
**Miss**: conversation, 4 points under. Not chased.

## Bailian: the fill rule misses, the planner does not

Page 256 tokens. Hit rate (share of prompt tokens served from cache):

| trace | C | exact | planner | fill (today) | proportional |
|---|---|---|---|---|---|
| A (to-C) | 1M | 0.236 | 0.202 | **0.065** | 0.326 |
| A | 3M | 0.412 | 0.398 | 0.530 | 0.530 |
| A | 10M | 0.512 | 0.510 | 0.530 | 0.530 |
| coder | 1M | 0.405 | 0.440 | **0.342** | 0.407 |
| coder | 3M | 0.509 | 0.505 | **0.342** | 0.535 |
| coder | 10M | 0.599 | 0.585 | 0.647 | 0.647 |
| thinking | 1M | 0.404 | 0.416 | 0.442 | 0.442 |
| B (to-C) | 1M | 0.410 | 0.451 | 0.451 | 0.451 |

Capacity to reach 95% of the ceiling: A 8.1M, coder 14.0M, thinking 2.1M,
B 2.3M tokens.

**Miss**: the fill rule predicts the collapse (A at 1M, coder at 1M and
3M) where exact LRU keeps 45-79% of the ceiling. Its other error runs the
other way: at A 3M it says "all warm" (0.530) where exact gives 0.412.
Real sessions do not take turns in a fixed order; their gaps are spread
over seconds to hours, so LRU loses the long-gap sessions first and keeps
the rest. The cyclic cliff of docs/writeup.md is real, but it is the worst
case for a benchmark that sends prompts round-robin, not what a live
workload sees.

**Fix**: the planner (characteristic time over the real gap distribution)
stays within 4 points of exact at every capacity on all four traces. The
proportional rule, which needs only the mean gap, fixes the collapse
(coder 3M: 0.535 against 0.509) but overshoots by up to 9 points (A at 1M)
because it treats every gap as exponential.

## TraceLab: provider-measured hits, with a TTL

No block ids, so the planner alone, with reuse = the previous round's
prompt when the new one is no shorter (a shorter one is a compaction).
Capacity is not the limit at a provider; the cache TTL is:

| provider | rounds | measured | planner, 5 min TTL | 1 h TTL | no TTL |
|---|---|---|---|---|---|
| Claude Code | 140,338 | 95.82% | 92.76% | 98.03% | 98.72% |
| Codex | 216,823 | 95.67% | 92.99% | 95.23% | 95.46% |

**Codex: hit.** No TTL predicts 95.46% against 95.67% measured. Of the
rounds after a gap over an hour, 29% of the reusable tokens still came back
as cached.

**Claude Code: between the two TTLs.** 5 min under-predicts by 3 points,
1 h over-predicts by 2. Rounds after a 5 min to 1 h gap got 60% of their
reusable tokens from cache, consistent with a mix of 5-minute and 1-hour
cache writes. The planner brackets it; it cannot say which writes used
which TTL without the request headers, which TraceLab does not carry.

## AgentX: the planner hits both published points

`agentx_sim.planner`: per rank, `concurrency / ranks` sessions cycle through
request (the trace's recorded `api_time`) and gap; running requests pin
prompt + output, idle prefixes hold the rest until T*.

| point | measured HBM | planner HBM | T* | event simulator (10 seeds) |
|---|---|---|---|---|
| B300, conc 384 | 91.0% | **91.1%** | 230 s | 91.4% (90.5-92.2) |
| B200, conc 196 | 73.0% | **72.9%** | 15 s | 41.2% (27.7-48.4) |

The planner and the event simulator differ in one input: the planner
pins tokens for the trace's recorded service times; the simulator pins them
for the service times it prices for DeepSeek V4 Pro on the B200. Speeding
the simulator's service up moves B200 towards the measured point (three
seeds each: x1 0.42/0.28/0.43, x2 0.57/0.50/0.63, x3 0.67/0.64/0.73, x4
0.73/0.72/0.76). So the 32-point miss recorded in agentx_kv_hits.md has two
candidate causes, capacity about 12 GB per GPU larger than the bytes give,
or requests that finish 3-4 times faster than priced, and the B200 server
log would separate them. The planner is sensitive to the reserve on B200
(5 GB: 76.9%, 15 GB: 41.8%) and not on B300 (91.1%, 91.0%).

The fill rule on AgentX: B300 keeps ~57 sessions warm per rank against 48
live (ceiling, right). B200 keeps ~14 against 24.5 live and predicts the
collapse, against 73% measured.

## What changes in the model

The fill rule's cliff is wrong for every live trace here; the
characteristic-time rule is right within a few points on all of them. The
model's `cache_ceiling_users` (warm p5) is used as a hard limit: past it the
explorer reports the deployment as failing on cache. On these traces, past
that point the hit rate degrades in proportion, it does not fall off. The
change is to report a hit rate at the configured users (planner rule with
the workload's gap distribution, exponential when only the think time is
known) and let the miss cost feed the prefill queue, instead of a ceiling.
That changes what the explorer shows and its JS mirror under
`interactive/`, so it goes as a separate PR with a card.

## Reproduce

```
ws traces mooncake FAST25-release/traces/toolagent_trace.jsonl --capacity 1,3
ws traces bailian qwen_coder.jsonl --capacity 1,3,10,30,100
ws traces tracelab syfi_coding_trace.jsonl.gz
ws agentx predict agentx-062126.json.gz        # planner line under each point
```
