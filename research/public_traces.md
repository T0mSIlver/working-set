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
3M tokens" (only the tool-agent row was rerun with `ws traces` at review;
the others and the arxiv comparison come from the same replay code run
from a scratch script before the CLI existed):

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
stays within about 4 points of exact from 1M tokens up on all four traces
(worst: B at 1M, 0.451 against 0.410). It is weaker at small capacity on a
trace whose same-session reuse is small: on B at 0.3M it still reads the
ceiling (0.451) where exact gives 0.358, because B's hits are mostly
cross-session prefixes, which the planner counts as always resident. The
proportional rule, which needs only the mean gap, fixes the collapse
(coder 3M: 0.535 against 0.509) but overshoots by up to 9 points (A at 1M)
because it treats every gap as exponential.

## TraceLab: provider-measured hits, with a TTL

No block ids, so the planner alone, with reuse = the previous round's
prompt when the new one is no shorter (a shorter one is a compaction).
Capacity is not the limit at a provider; the cache TTL is. 16 of 4,265
sessions repeat a round index and 12 rounds start before the round
before them; their negative gaps count as hits (negligible here):

| provider | rounds | measured | planner, 5 min TTL | 1 h TTL | no TTL |
|---|---|---|---|---|---|
| Claude Code | 140,338 | 95.82% | 92.76% | 98.03% | 98.72% |
| Codex | 216,823 | 95.67% | 92.99% | 95.23% | 95.46% |

**Codex: hit.** No TTL predicts 95.46% against 95.67% measured. Of the
reusable tokens in rounds after a gap over an hour (gap between rounds'
first timing events, compactions excluded), 17% still came back as cached.

**Claude Code: between the two TTLs.** 5 min under-predicts by 3 points,
1 h over-predicts by 2. Rounds after a 5 min to 1 h gap got 60% of their
reusable tokens from cache, consistent with a mix of 5-minute and 1-hour
cache writes. The planner brackets it; it cannot say which writes used
which TTL without the request headers, which TraceLab does not carry.

## AgentX: the planner lands on both published points, at the chosen reserve

`agentx_sim.planner`: per rank, `concurrency / ranks` sessions cycle through
request (the trace's recorded `api_time`) and gap; running requests pin
prompt + output, idle prefixes hold the rest until T*.

| point | measured HBM | planner HBM | T* | event simulator (10 seeds) |
|---|---|---|---|---|
| B300, conc 384 | 91.0% | **91.1%** | 230 s | 91.4% (90.5-92.2) |
| B200, conc 196 | 73.0% | **72.9%** | 15 s | 41.2% (27.7-48.4) |

The B200 match is at the 10 GB reserve per GPU used throughout; the
reserve is not independently known and the planner moves with it (5 GB:
76.9%, 15 GB: 41.8%), so B200 is one matching point on a knife edge, not a
confirmation. B300 does not move (91.1%, 91.0%).

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

## 2026-11-02: the hit rate at the configured users replaces the cliff (issue #117)

The change the section above called for, made. Code: `model.miss_share_at`,
`model._capacity_limit`, `model.operating_point`, `predict.predict`; the
explorer's `missShareAt` / `capacityLimit` (`interactive/src/prefill.js`).

**The rule.** W is the cache's warm user p5 (`max_users_cache`, unchanged).
With U live users per replica group, LRU's characteristic time T* is where
the contexts held for T* after each request fill the pool. The model knows a
think time and nothing else about the gaps, so it takes them as exponential:
a session is then resident a share 1 − e^(−T*/Z) of its cycle and comes back
warm with that probability, which gives min(1, W'/U) for W' sessions held.
W counts reusable sessions (a forced miss never counts as warm), so the
share of requests that miss is

    f(U) = max(f0, 1 − W/U)

with f0 the configured `miss_rate`: f0 up to the knee W/(1 − f0), the users
the pool cannot hold past it. `ws predict` reports 1 − f(U) as
`cache_hit_rate`.

**Where the misses go.** Every at-load figure (prefill duty, queue wait,
TTFT, B*, burst drain, slowed share, cold wait, power and the bill) is priced
at f(U). Each of the cold-wait, slowed-generation, saturation and TTFT
ceilings is the root of U = ceiling(f(U)), the user count where its figure,
priced at the miss share at that count, meets its budget; a ceiling whose
configured-share value sits inside the knee does not move. There is no cache
ceiling in the verdict any more: four ceilings, decode, cold wait, slowed
generation and saturation. No new parameter. Open loop, slowed generation
has a closed form past the knee, W + K, with K the all-cold users that fill
the slowed-share limit alone: past W every added user is a cold stream.

**The cyclic cliff** stays as the stated worst case. `ws predict` prints
"a benchmark that sends prompts round-robin misses every request past W
users" beside the verdict; the explorer draws W on chart G as a dotted line
labelled that way, and names it in the hit-rate tile.

### The proof: the rule on the Bailian and AgentX points

Hit rate as each source defines it (Bailian: share of prompt tokens served
from cache at 256-token pages; AgentX: HBM hit rate over every request).
Bailian figures are the 2026-10-21 table above (exact LRU, planner, fill,
proportional), not rerun: the traces are not in this session. AgentX is
rerun here from `data/agentx-062126` (the planner reproduces 91.1% and
72.9%). Points in brackets are the error against exact LRU (Bailian) or the
published measurement (AgentX).

| point | exact / measured | rule, the trace's own gaps | rule, exponential gaps (the model) | cliff (before) |
|---|---|---|---|---|
| Bailian A, 1M | 0.236 | 0.202 (−3.4) | 0.326 (+9.0) | 0.065 (−17.1) |
| Bailian A, 3M | 0.412 | 0.398 (−1.4) | 0.530 (+11.8) | 0.530 (+11.8) |
| Bailian A, 10M | 0.512 | 0.510 (−0.2) | 0.530 (+1.8) | 0.530 (+1.8) |
| Bailian coder, 1M | 0.405 | 0.440 (+3.5) | 0.407 (+0.2) | 0.342 (−6.3) |
| Bailian coder, 3M | 0.509 | 0.505 (−0.4) | 0.535 (+2.6) | 0.342 (−16.7) |
| Bailian coder, 10M | 0.599 | 0.585 (−1.4) | 0.647 (+4.8) | 0.647 (+4.8) |
| Bailian thinking, 1M | 0.404 | 0.416 (+1.2) | 0.442 (+3.8) | 0.442 (+3.8) |
| Bailian B, 1M | 0.410 | 0.451 (+4.1) | 0.451 (+4.1) | 0.451 (+4.1) |
| AgentX B300, conc 384 | 0.910 | 0.911 (+0.1) | 0.957 (+4.7) | 0.957 (+4.7) |
| AgentX B200, conc 196 | 0.730 | 0.729 (−0.1) | 0.539 (−19.1) | 0.000 (−73.0) |

"Rule, the trace's own gaps" is the characteristic-time rule fed each
trace's gap distribution (`traces.planner_curve`, `agentx_sim.planner`).
"Exponential gaps" is the same rule given only a mean gap, which is what the
model computes from a think time: on Bailian the proportional column above;
on AgentX the same-stream ceiling (0.957) times min(1, warm/live) with warm =
HBM tokens per rank over the mean prompt and live = concurrency per rank
(57.0/48.0 on B300, 13.8/24.5 on B200, the session counts the fill rule
used). Priced in token-time instead (the room left after running requests
pin their prompts, over what holding every idle prefix until its session
returns needs) it reads 0.852 and 0.197; the session count is the model's
own accounting, so the table uses it.

**Hit: the rule.** Fed the trace's own gaps it stays within 5 points of
exact LRU or of the measured value on all ten points (worst: Bailian B at
1M, +4.1).

**Partly: the model's form of it.** With only a mean gap the rule is within
5 points on seven of the ten. It misses three:

- Bailian A at 1M and 3M, 9 and 12 points high. Its sessions come back after
  gaps far wider-spread than exponential, and the long-gap sessions hold the
  most tokens, so LRU loses more than a mean gap says. At 3M the exponential
  form still says everyone fits.
- AgentX B200, 19 points low. Most returns there come within seconds
  (T* = 15 s), so a pool that holds 14 of 24.5 sessions keeps far more than
  14/24.5 of the returns.

It replaces a cliff that is within 5 points on five of the ten and off by
up to 73 (B200, where it predicts the collapse against 73% measured). The
error that remains belongs to the gap distribution, which the model does not
take as an input; closing it means a gap-shape parameter, and none is added
here.

### `ws predict`, 27B on 4xH200 TP4 (`ws init --model 27B --gpu H200 --tp 4`)

Before:

```
  cache (warm p5, users)                          403
  decode (max_num_seqs at p99)                    399  <- binds
  cold request waits 5 s (p95)                    459
  generation slowed 15% of the time               425
  saturation (prefill duty 100%)                  510
  not in the verdict: p95 TTFT reaches 10 s at 501 users
```

After:

```
cache: 403 users stay warm (p5); at 64 users 99.0% of requests hit (past 407 users the extra requests miss)

  decode (max_num_seqs at p99)                    399  <- binds
  cold request waits 5 s (p95)                    409
  generation slowed 15% of the time               407
  saturation (prefill duty 100%)                  413
  not in the verdict: p95 TTFT reaches 10 s at 412 users
  not in the verdict: a benchmark that sends prompts round-robin misses every request past 403 users (the cyclic worst case, docs/writeup.md)
```

The at-load line is unchanged (64 users sit inside the pool). Three latency
ceilings that sat past the knee (407) now land just past it: a miss costs
about twenty hits of prefill on this deployment, so a few users past W of
cold traffic exhaust each budget. The verdict does not move (decode, 399).
In the explorer, the 35B-A3B on 2xH200 TP2 at a 3% miss rate keeps 555 users
warm; its cold wait and saturation ceilings, which sat at 1,121 and 1,227,
past the pool, now read 611 and 626, and slowed generation (425, inside the
pool) does not move. At 1,000 users the page reads a 55% hit rate where it
used to read "cache, over".

### The golden vectors that moved

`tests/golden/vectors.json` was regenerated (`uv run scripts/golden.py`)
and diffed against the previous fixture, value by value. 627 of the 955
vectors changed; `cache_hit_rate` is a new quantity on all 955. Every
changed value falls in one of two classes, and none outside them moved
(checked mechanically: the classifier finds 0 unexplained changes, and no
ceiling that sat past the knee was left unmoved):

- **load**: a figure priced at the load (prefill duty, the queue and TTFT
  figures, B*, the drain, slowed share, cold wait, power and the bill) on a
  state whose users per group sit past the knee W / (1 - f0). It is now
  priced at f(U) = 1 - W/U instead of f0, so duty, waits, slowed share and
  watts rise and B* falls. 116 vectors; a figure already infinite or clamped
  (duty past 1, slowed share at 1) stays where it was, which is why most
  keys show 68.
- **ceiling**: a cold-wait, slowed-generation, saturation or TTFT ceiling
  whose configured-rate value sat past the knee, which is exactly where the
  cache ceiling used to cut it off. It is now the fixed point a few users
  past the knee, and always lower than before. Slowed-generation ceilings
  that read "never" (inf: the slowed point lay past the prefill wall) become
  finite where the capacity misses bring it inside. 611 vectors.

The verdict (the lowest ceiling) can change only where the cache was the
lowest before: every moved ceiling lands at or past the knee, above W. The
`mc` bands of the four ceilings widened (3-6% to 11-13%),
because the probe states whose ceilings now land past the knee carry W's
seed spread; `tests/golden/README.md` records why, and the 23 new allowlist
entries, each gated on `_miss_share_swing` or `_warm_p5`.

| quantity | moved: users past the knee | moved: ceiling sat past the knee |
|---|---|---|
| `burst_drain_seconds_b32` | 68 |  |
| `cold_wait_seconds` | 68 |  |
| `energy_eur_month` | 68 |  |
| `energy_eur_mtok` | 68 |  |
| `energy_eur_user` | 68 |  |
| `energy_total_month` | 68 |  |
| `max_users_cold_wait` |  | 445 |
| `max_users_latency` |  | 525 |
| `max_users_saturation` |  | 552 |
| `max_users_slowed` |  | 411 |
| `power_d_d` | 68 |  |
| `power_d_p` | 68 |  |
| `power_kw` | 68 |  |
| `power_per_gpu_w` | 68 |  |
| `prefill_duty` | 116 |  |
| `queue_wait_seconds` | 68 |  |
| `slowed_share` | 91 |  |
| `spike_tolerance` | 68 |  |
| `spike_tolerance_sla10` | 68 |  |
| `ttft_hit_fcfs` | 68 |  |
| `ttft_hit_ps` | 68 |  |
| `ttft_miss_fcfs` | 68 |  |

**ceiling** (511 vectors): 0, 1, 2, 3, 5, 8, 9, 12, 13, 20, 28, 33, 34, 35, 36, 37, 39, 40, 41, 42, 43, 47, 48, 49, 52, 53, 54, 57, 62, 63, 64, 65, 66, 70, 71, 74, 77, 79, 80, 81, 85, 87, 88, 92, 93, 94, 99, 100, 103, 104, 108, 110, 111, 112, 113, 114, 118, 119, 122, 123, 126, 129, 131, 132, 133, 134, 136, 138, 140, 143, 144, 150, 153, 154, 155, 156, 157, 158, 159, 160, 164, 170, 171, 172, 173, 175, 177, 178, 182, 184, 185, 186, 187, 188, 192, 193, 200, 201, 202, 203, 204, 205, 206, 214, 217, 218, 220, 221, 223, 224, 225, 226, 227, 228, 229, 232, 233, 234, 235, 236, 238, 245, 249, 252, 253, 254, 256, 260, 261, 262, 264, 265, 267, 268, 269, 271, 272, 275, 278, 279, 280, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 312, 321, 330, 335, 344, 345, 346, 347, 348, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 370, 371, 372, 374, 375, 376, 378, 379, 381, 382, 384, 385, 386, 387, 388, 389, 390, 392, 393, 395, 396, 397, 398, 401, 402, 404, 406, 407, 408, 409, 410, 412, 413, 414, 415, 418, 419, 420, 421, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 437, 438, 439, 440, 441, 442, 444, 445, 446, 447, 448, 449, 450, 451, 456, 463, 473, 474, 475, 476, 477, 481, 482, 483, 488, 489, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 506, 507, 508, 510, 516, 517, 518, 524, 526, 527, 528, 531, 534, 535, 536, 538, 539, 541, 542, 546, 547, 548, 550, 551, 554, 556, 557, 558, 559, 562, 563, 564, 566, 572, 573, 575, 577, 578, 579, 580, 582, 586, 587, 588, 589, 591, 592, 593, 594, 596, 597, 601, 602, 604, 605, 608, 609, 610, 612, 613, 614, 616, 617, 621, 622, 625, 626, 627, 633, 634, 635, 636, 639, 640, 643, 644, 645, 646, 649, 650, 656, 657, 660, 661, 664, 667, 668, 670, 671, 673, 675, 676, 678, 683, 687, 688, 689, 690, 693, 695, 697, 698, 699, 702, 703, 706, 710, 715, 717, 719, 720, 722, 726, 727, 728, 730, 732, 733, 735, 737, 739, 741, 742, 743, 744, 745, 746, 752, 753, 754, 756, 757, 758, 761, 762, 763, 764, 765, 768, 769, 770, 771, 773, 775, 777, 778, 779, 780, 781, 782, 783, 789, 790, 791, 795, 796, 798, 802, 803, 804, 810, 811, 812, 814, 819, 823, 825, 826, 830, 831, 832, 833, 835, 837, 838, 839, 842, 843, 844, 846, 847, 848, 849, 850, 851, 853, 856, 860, 861, 862, 866, 868, 870, 871, 872, 874, 875, 876, 878, 879, 880, 881, 883, 890, 893, 894, 895, 901, 902, 903, 904, 912, 913, 915, 917, 922, 924, 926, 927, 928, 929, 931, 933, 934, 937, 939, 941, 942, 943, 949, 950, 951, 952, 953, 954

**ceiling+load** (100 vectors): 4, 11, 15, 19, 44, 51, 59, 90, 95, 116, 130, 147, 148, 151, 169, 174, 176, 191, 208, 230, 237, 242, 248, 255, 257, 263, 270, 273, 276, 277, 281, 329, 342, 343, 369, 373, 377, 380, 383, 391, 399, 400, 403, 405, 411, 416, 417, 436, 470, 471, 472, 478, 479, 480, 484, 485, 486, 487, 490, 491, 492, 493, 505, 509, 511, 515, 519, 529, 537, 545, 549, 555, 565, 576, 581, 595, 599, 603, 607, 631, 638, 654, 662, 665, 700, 701, 731, 766, 794, 797, 806, 808, 822, 829, 834, 884, 886, 889, 899, 920

**load** (16 vectors): 22, 73, 76, 128, 216, 512, 520, 523, 553, 685, 705, 793, 852, 865, 932, 948

<details><summary>Every changed vector</summary>

| id | state | users / group | W | knee | f0 | moved | ceilings before -> after |
|---|---|---|---|---|---|---|---|
| 0 | 27B/1xH200/fp8/fp8KV | 64 | 70.0 | 70.7 | 0.01 | ceiling | ttft 149 -> 74; sat 157 -> 76; slowed 140 -> 71; wait 111 -> 72 |
| 1 | 27B/1xH200/fp8/fp8KV | 64 | 70.0 | 70.7 | 0.01 | ceiling | ttft 149 -> 74; sat 157 -> 76; slowed 140 -> 71; wait 111 -> 72 |
| 2 | 27B/8xB300 DP4xTP2/nvfp4/fp16KV | 100 | 138.0 | 139.4 | 0.01 | ceiling | ttft 501 -> 149; sat 539 -> 151; wait 427 -> 145 |
| 3 | MM35/8xB300 tensor-par/nvfp4/fp8KV | 128 | 697.9 | 775.4 | 0.1 | ceiling | ttft 805 -> 780; sat 830 -> 785 |
| 4 | MM35/6xH200 DP3xTP2/fp8/fp16KV | 42.6667 | 8.9 | 9.0 | 0.01 | ceiling+load (18 at-load keys) | ttft 35 -> 10; sat 50 -> 13; slowed inf -> 10; wait 17 -> 9 |
| 5 | 27B/6xB300 DP3xTP2/fp8/fp16KV | 21.3333 | 63.0 | 96.8 | 0.35 | ceiling | sat 131 -> 109 |
| 8 | 27B/8xB300 data-par/nvfp4/fp8KV | 2 | 172.9 | 192.2 | 0.1 | ceiling | ttft 244 -> 199; sat 265 -> 201 |
| 9 | 35BA3B/6xB300 tensor-par/fp8/fp8KV | 400 | 2,043.0 | 2,106.1 | 0.03 | ceiling | ttft 4,512 -> 2,212; sat 4,552 -> 2,214; wait 4,315 -> 2,201 |
| 11 | MM35/4xH200 DP2xTP2/fp8/fp16KV | 8 | 5.0 | 5.2 | 0.03 | ceiling+load (18 at-load keys) | ttft 53 -> 6; sat 77 -> 10; slowed 19 -> 6; wait 7 -> 5 |
| 12 | 27B/6xH200 DP2xTP3/fp8/fp16KV | 32 | 75.0 | 115.4 | 0.35 | ceiling | sat 120 -> 117 |
| 13 | DSV4F/1xB300/fp8/fp8KV | 64 | 574.0 | 576.9 | 0.005 | ceiling | ttft 3,887 -> 618; sat 4,142 -> 628; slowed 647 -> 577; wait 3,385 -> 605 |
| 15 | MM35/2xB300 data-par/fp8/fp8KV | 16 | 13.0 | 14.4 | 0.1 | ceiling+load (2 at-load keys) | sat 16 -> 15 |
| 19 | MM35/2xH200 tensor-par/fp8/fp16KV | 128 | 11.0 | 11.1 | 0.01 | ceiling+load (2 at-load keys) | ttft 66 -> 12; sat 92 -> 14; slowed 40 -> 11; wait 14 -> 11 |
| 20 | 35BA3B/6xH200 DP2xTP3/fp8/fp8KV | 32 | 3,327.9 | 3,697.7 | 0.1 | ceiling | ttft 5,273 -> 3,914; sat 5,319 -> 3,921; wait 5,044 -> 3,880 |
| 22 | MM35/8xB300 DP4xTP2/fp8/fp8KV | 32 | 23.0 | 25.6 | 0.1 | load (1 at-load keys) |  |
| 28 | 27B/3xB300 tensor-par/nvfp4/fp8KV | 128 | 525.9 | 528.5 | 0.005 | ceiling | ttft 1,787 -> 561; sat 1,865 -> 571; slowed 1,198 -> 532; wait 1,633 -> 548 |
| 33 | Q38FN/6xH200 DP3xTP2/fp8/fp16KV | 10.6667 | 84.0 | 84.4 | 0.005 | ceiling | ttft 2,044 -> 243; sat 2,060 -> 260; slowed inf -> 108; wait 2,013 -> 214 |
| 34 | 27B/2xH200 tensor-par/fp8/fp8KV | 64 | 256.0 | 258.6 | 0.01 | ceiling | slowed inf -> 268 |
| 35 | 35BA3B/8xB300 data-par/fp8/fp16KV | 8 | 655.0 | 658.3 | 0.005 | ceiling | ttft 2,950 -> 755; sat 2,956 -> 756; slowed inf -> 672; wait 2,919 -> 747 |
| 36 | DSV4F/5xB300 tensor-par/nvfp4/fp8KV | 128 | 13,004.9 | 13,136.3 | 0.01 | ceiling | slowed inf -> 13,257 |
| 37 | Q38FN/4xB300 tensor-par/fp8/fp16KV | 400 | 1,028.8 | 1,060.6 | 0.03 | ceiling | ttft 4,086 -> 1,209; sat 4,214 -> 1,218; wait 3,833 -> 1,191 |
| 39 | 35BA3B/4xH200 tensor-par/fp8/fp16KV | 400 | 1,065.0 | 1,070.3 | 0.005 | ceiling | ttft 9,404 -> 1,635; sat 9,406 -> 1,637; slowed inf -> 1,136; wait 9,377 -> 1,618 |
| 40 | 35BA3B/6xH200 DP2xTP3/fp8/fp8KV | 8 | 1,190.0 | 1,196.0 | 0.005 | ceiling | ttft 7,219 -> 1,401; sat 7,243 -> 1,406; slowed 6,136 -> 1,221; wait 7,099 -> 1,380 |
| 41 | 35BA3B/7xB300 data-par/fp8/fp16KV | 57.1429 | 176.0 | 176.9 | 0.005 | ceiling | ttft 5,377 -> 269; sat 5,494 -> 279; slowed 473 -> 178; wait 4,811 -> 238 |
| 42 | 35BA3B/6xB300 tensor-par/nvfp4/fp8KV | 400 | 6,353.7 | 6,385.6 | 0.005 | ceiling | ttft 13,861 -> 6,770; sat 13,883 -> 6,775; wait 13,817 -> 6,758 |
| 43 | 27B/7xB300 tensor-par/nvfp4/fp8KV | 128 | 690.0 | 696.9 | 0.01 | ceiling | ttft 1,773 -> 724; sat 1,793 -> 725; wait 1,569 -> 715 |
| 44 | MM35/3xB300 tensor-par/nvfp4/fp8KV | 400 | 156.0 | 157.6 | 0.01 | ceiling+load (18 at-load keys) | ttft 619 -> 174; sat 654 -> 179; slowed 330 -> 159; wait 460 -> 163 |
| 47 | 35BA3B/3xH200 data-par/fp8/fp16KV | 10.6667 | 106.0 | 106.5 | 0.005 | ceiling | ttft 1,875 -> 149; sat 1,913 -> 153; slowed 287 -> 107; wait 1,688 -> 131 |
| 48 | 27B/7xH200 data-par/fp8/fp16KV | 2.28571 | 138.9 | 140.4 | 0.01 | ceiling | ttft 341 -> 153; sat 352 -> 156; slowed 203 -> 141; wait 321 -> 149 |
| 49 | DSV4F/3xH200 tensor-par/fp8/fp8KV | 400 | 801.0 | 809.1 | 0.01 | ceiling | ttft 1,101 -> 832; sat 1,124 -> 836; wait 1,056 -> 825 |
| 51 | MM35/7xB300 tensor-par/fp8/fp16KV | 400 | 214.9 | 216.0 | 0.005 | ceiling+load (2 at-load keys) | slowed inf -> 223 |
| 52 | MM35/6xB300 tensor-par/fp8/fp16KV | 16 | 44.0 | 48.9 | 0.1 | ceiling | ttft 55 -> 49; sat 108 -> 56 |
| 53 | 27B/6xB300 tensor-par/nvfp4/fp8KV | 32 | 549.0 | 566.0 | 0.03 | ceiling | ttft 3,083 -> 684; sat 3,129 -> 687; wait 2,600 -> 652 |
| 54 | 27B/4xB300 data-par/fp8/fp16KV | 8 | 65.0 | 100.0 | 0.35 | ceiling | sat 103 -> 101 |
| 57 | DSV4FV/6xB300 tensor-par/fp8/fp8KV | 16 | 9,777.0 | 9,826.1 | 0.005 | ceiling | ttft 10,097 -> 9,831; sat 10,210 -> 9,834; wait 9,872 -> 9,827 |
| 59 | GLM53F/2xB300 tensor-par/nvfp4/fp16KV | 400 | 274.0 | 276.8 | 0.01 | ceiling+load (18 at-load keys) | ttft 569 -> 305; sat 579 -> 309; slowed inf -> 280; wait 521 -> 291 |
| 62 | 27B/8xB300 DP4xTP2/nvfp4/fp8KV | 16 | 1,691.8 | 1,744.2 | 0.03 | ceiling | ttft 2,329 -> 1,785; sat 2,353 -> 1,787; wait 2,209 -> 1,775 |
| 63 | 35BA3B/8xH200 data-par/fp8/fp8KV | 8 | 461.9 | 513.3 | 0.1 | ceiling | ttft 2,091 -> 764; sat 2,104 -> 767; wait 2,026 -> 749 |
| 64 | Q38FN/4xB300 DP2xTP2/nvfp4/fp16KV | 8 | 1,052.8 | 1,085.4 | 0.03 | ceiling | ttft 2,158 -> 1,232; sat 2,167 -> 1,235; wait 2,141 -> 1,227 |
| 65 | 35BA3B/5xH200 data-par/fp8/fp16KV | 25.6 | 41.0 | 63.0 | 0.35 | ceiling | ttft 109 -> 80; sat 124 -> 86 |
| 66 | Q38FN/6xB300 DP2xTP3/fp8/fp16KV | 8 | 249.0 | 276.7 | 0.1 | ceiling | ttft 388 -> 291; sat 406 -> 293; wait 306 -> 280 |
| 70 | 27B/3xB300 data-par/fp8/fp16KV | 21.3333 | 84.0 | 84.4 | 0.005 | ceiling | ttft 268 -> 125; sat 270 -> 127; slowed inf -> 93; wait 249 -> 109 |
| 71 | 35BA3B/2xB300 tensor-par/fp8/fp16KV | 128 | 6,879.4 | 6,948.8 | 0.01 | ceiling | ttft 17,032 -> 7,962; sat 17,048 -> 7,968; slowed inf -> 7,103; wait 17,001 -> 7,950 |
| 73 | MM35/5xB300 data-par/nvfp4/fp8KV | 12.8 | 4.0 | 6.2 | 0.35 | load (1 at-load keys) |  |
| 74 | 35BA3B/7xB300 tensor-par/fp8/fp16KV | 400 | 1,239.9 | 1,246.1 | 0.005 | ceiling | ttft 4,583 -> 1,358; sat 4,591 -> 1,359; wait 4,546 -> 1,352 |
| 76 | MM35/8xB300 tensor-par/fp8/fp16KV | 400 | 42.0 | 64.6 | 0.35 | load (1 at-load keys) |  |
| 77 | 27B/8xH200 data-par/fp8/fp16KV | 8 | 14.0 | 14.1 | 0.005 | ceiling | ttft 51 -> 15; sat 57 -> 16; slowed 43 -> 14; wait 29 -> 14 |
| 79 | 35BA3B/6xB300 DP2xTP3/nvfp4/fp16KV | 200 | 1,324.0 | 1,337.3 | 0.01 | ceiling | ttft 4,526 -> 1,487; sat 4,531 -> 1,487; wait 4,500 -> 1,483 |
| 80 | 35BA3B/3xB300 tensor-par/nvfp4/fp16KV | 128 | 288.0 | 443.1 | 0.35 | ceiling | ttft 758 -> 560; sat 770 -> 564; wait 644 -> 517 |
| 81 | Q38FN/6xB300 data-par/fp8/fp16KV | 2.66667 | 30.0 | 46.2 | 0.35 | ceiling | ttft 86 -> 61; sat 93 -> 63 |
| 85 | 35BA3B/8xH200 data-par/fp8/fp16KV | 8 | 65.0 | 72.2 | 0.1 | ceiling | ttft 224 -> 85; sat 281 -> 110; wait 144 -> 81 |
| 87 | Q38FN/3xB300 tensor-par/nvfp4/fp16KV | 64 | 243.9 | 251.5 | 0.03 | ceiling | ttft 931 -> 317; sat 987 -> 334; wait 823 -> 294 |
| 88 | DSV4F/3xB300 data-par/fp8/fp8KV | 133.333 | 320.9 | 330.8 | 0.03 | ceiling | ttft 363 -> 334; sat 383 -> 337 |
| 90 | GLM52/2xB300 tensor-par/nvfp4/fp8KV | 64 | 7.0 | 7.8 | 0.1 | ceiling+load (1 at-load keys) | sat 16 -> 9 |
| 92 | 35BA3B/8xB300 DP4xTP2/nvfp4/fp8KV | 32 | 4,756.6 | 4,903.7 | 0.03 | ceiling | ttft 5,766 -> 4,988; sat 5,772 -> 4,989; wait 5,734 -> 4,984 |
| 93 | MM35/6xB300 DP3xTP2/fp8/fp16KV | 10.6667 | 59.0 | 59.2 | 0.005 | ceiling | ttft 70 -> 62; sat 75 -> 65; slowed inf -> 63; wait 63 -> 60 |
| 94 | Q38FN/2xB300 data-par/fp8/fp8KV | 200 | 601.0 | 667.8 | 0.1 | ceiling | ttft 2,425 -> 903; sat 2,451 -> 907; wait 2,302 -> 883 |
| 95 | 27B/4xB300 DP2xTP2/nvfp4/fp8KV | 200 | 157.0 | 161.9 | 0.03 | ceiling+load (18 at-load keys) | ttft 937 -> 194; sat 1,005 -> 199; slowed 189 -> 163; wait 398 -> 169 |
| 99 | 35BA3B/8xB300 data-par/fp8/fp8KV | 8 | 1,117.0 | 1,122.6 | 0.005 | ceiling | ttft 5,554 -> 1,361; sat 5,556 -> 1,362; slowed 1,144 -> 1,123; wait 5,531 -> 1,352 |
| 100 | Q38FN/8xB300 data-par/nvfp4/fp8KV | 8 | 185.0 | 185.9 | 0.005 | ceiling | ttft 258 -> 188; sat 260 -> 188; wait 232 -> 187 |
| 103 | 27B/6xB300 DP3xTP2/nvfp4/fp16KV | 133.333 | 247.0 | 274.4 | 0.1 | ceiling | ttft 404 -> 298; sat 414 -> 300; wait 304 -> 279 |
| 104 | GLM53F/5xB300 data-par/nvfp4/fp8KV | 25.6 | 141.0 | 156.7 | 0.1 | ceiling | ttft 263 -> 175; sat 293 -> 184 |
| 108 | 35BA3B/3xB300 tensor-par/fp8/fp16KV | 400 | 823.9 | 832.2 | 0.01 | ceiling | ttft 3,265 -> 890; sat 3,289 -> 891; wait 3,012 -> 878 |
| 110 | 35BA3B/1xB300/fp8/fp16KV | 64 | 120.0 | 123.7 | 0.03 | ceiling | ttft 679 -> 150; sat 739 -> 155; wait 420 -> 134 |
| 111 | GLM53F/3xB300 data-par/nvfp4/fp8KV | 10.6667 | 485.0 | 500.0 | 0.03 | ceiling | sat 613 -> 505 |
| 112 | Q38FN/1xB300/nvfp4/fp16KV | 32 | 25.9 | 39.9 | 0.35 | ceiling | sat 92 -> 59 |
| 113 | 27B/4xH200 DP2xTP2/fp8/fp8KV | 8 | 1,184.0 | 1,189.9 | 0.005 | ceiling | ttft 1,559 -> 1,203; sat 1,580 -> 1,205; wait 1,452 -> 1,197 |
| 114 | Q38FN/4xH200 DP2xTP2/fp8/fp16KV | 8 | 84.0 | 86.5 | 0.03 | ceiling | ttft 933 -> 151; sat 958 -> 156; slowed 181 -> 89; wait 684 -> 114 |
| 116 | MM35/2xB300 tensor-par/nvfp4/fp8KV | 128 | 47.0 | 48.5 | 0.03 | ceiling+load (1 at-load keys) | sat 70 -> 50 |
| 118 | GLM53F/8xH200 tensor-par/fp8/fp16KV | 32 | 842.9 | 847.1 | 0.005 | ceiling | slowed inf -> 854 |
| 119 | 35BA3B/8xB300 tensor-par/nvfp4/fp16KV | 128 | 2,590.9 | 2,878.8 | 0.1 | ceiling | ttft 13,012 -> 4,539; sat 13,041 -> 4,546; wait 12,954 -> 4,525 |
| 122 | 27B/7xB300 tensor-par/nvfp4/fp16KV | 128 | 446.0 | 448.2 | 0.005 | ceiling | ttft 4,534 -> 559; sat 4,566 -> 563; slowed 1,011 -> 451; wait 4,188 -> 520 |
| 123 | 27B/6xB300 data-par/fp8/fp16KV | 66.6667 | 170.9 | 171.8 | 0.005 | ceiling | slowed inf -> 173 |
| 126 | Q38FN/4xH200 DP2xTP2/fp8/fp16KV | 32 | 70.0 | 70.3 | 0.005 | ceiling | ttft 226 -> 89; sat 228 -> 91; slowed inf -> 72; wait 216 -> 82 |
| 128 | MM35/7xH200 tensor-par/fp8/fp8KV | 400 | 113.0 | 125.5 | 0.1 | load (1 at-load keys) |  |
| 129 | MM35/5xB300 tensor-par/fp8/fp8KV | 32 | 78.0 | 78.3 | 0.005 | ceiling | sat 103 -> 79; slowed 100 -> 78 |
| 130 | MM35/7xB300 data-par/nvfp4/fp8KV | 9.14286 | 7.0 | 7.1 | 0.01 | ceiling+load (18 at-load keys) | sat 20 -> 8; slowed inf -> 7 |
| 131 | 35BA3B/8xH200 DP4xTP2/fp8/fp8KV | 8 | 388.0 | 389.9 | 0.005 | ceiling | ttft 2,363 -> 480; sat 2,374 -> 484; slowed inf -> 400; wait 2,247 -> 448 |
| 132 | 35BA3B/6xH200 tensor-par/fp8/fp16KV | 16 | 9,994.8 | 10,095.7 | 0.01 | ceiling | slowed inf -> 10,248 |
| 133 | 27B/2xH200 data-par/fp8/fp16KV | 8 | 11.0 | 16.9 | 0.35 | ceiling | sat 19 -> 18 |
| 134 | Q38FN/7xB300 data-par/fp8/fp8KV | 4.57143 | 112.0 | 113.1 | 0.01 | ceiling | ttft 582 -> 161; sat 598 -> 174; slowed inf -> 123; wait 552 -> 144 |
| 136 | 27B/7xB300 tensor-par/fp8/fp8KV | 64 | 5,078.9 | 5,236.0 | 0.03 | ceiling | ttft 8,584 -> 5,435; sat 8,619 -> 5,438; wait 8,199 -> 5,405 |
| 138 | DSV4FV/4xB300 DP2xTP2/fp8/fp8KV | 200 | 2,193.7 | 2,215.9 | 0.01 | ceiling | ttft 2,743 -> 2,229; sat 2,791 -> 2,230; wait 2,509 -> 2,222 |
| 140 | 27B/7xH200 tensor-par/fp8/fp16KV | 16 | 142.0 | 142.7 | 0.005 | ceiling | ttft 152 -> 143; sat 155 -> 144; wait 146 -> 143 |
| 143 | GLM53F/7xB300 data-par/nvfp4/fp16KV | 18.2857 | 209.9 | 211.0 | 0.005 | ceiling | slowed inf -> 216 |
| 144 | Q38FN/3xB300 tensor-par/fp8/fp8KV | 400 | 933.0 | 937.6 | 0.005 | ceiling | ttft 1,762 -> 956; sat 1,768 -> 956; wait 1,692 -> 952 |
| 147 | MM35/6xH200 DP2xTP3/fp8/fp16KV | 64 | 11.9 | 12.3 | 0.03 | ceiling+load (18 at-load keys) | ttft 99 -> 15; sat 140 -> 21; slowed 46 -> 13; wait 12 -> 12 |
| 148 | 27B/1xB300/nvfp4/fp8KV | 400 | 89.0 | 89.4 | 0.005 | ceiling+load (18 at-load keys) | ttft 633 -> 98; sat 677 -> 100; slowed 307 -> 91; wait 440 -> 92 |
| 150 | MM35/3xB300 data-par/fp8/fp16KV | 10.6667 | 131.8 | 133.2 | 0.01 | ceiling | ttft 148 -> 134; sat 156 -> 135; slowed inf -> 134 |
| 151 | MM35/8xB300 DP2xTP4/nvfp4/fp16KV | 200 | 62.0 | 62.6 | 0.01 | ceiling+load (2 at-load keys) | ttft 96 -> 69; sat 99 -> 70; slowed inf -> 65; wait 86 -> 65 |
| 153 | DSV41F/8xB300 DP4xTP2/fp8/fp8KV | 16 | 159.9 | 177.7 | 0.1 | ceiling | ttft 750 -> 244; sat 757 -> 244; wait 684 -> 235 |
| 154 | 35BA3B/6xB300 DP3xTP2/fp8/fp8KV | 5.33333 | 1,235.0 | 1,273.1 | 0.03 | ceiling | ttft 1,726 -> 1,311; sat 1,728 -> 1,311; wait 1,703 -> 1,308 |
| 155 | 35BA3B/2xH200 data-par/fp8/fp8KV | 8 | 156.0 | 156.8 | 0.005 | ceiling | ttft 691 -> 189; sat 695 -> 190; slowed inf -> 160; wait 674 -> 183 |
| 156 | GLM53F/5xH200 tensor-par/fp8/fp16KV | 400 | 1,276.0 | 1,288.9 | 0.01 | ceiling | ttft 1,893 -> 1,323; sat 1,899 -> 1,324; wait 1,827 -> 1,315 |
| 157 | MM35/2xH200 tensor-par/fp8/fp8KV | 32 | 51.0 | 56.7 | 0.1 | ceiling | sat 72 -> 59 |
| 158 | GLM53F/4xB300 tensor-par/fp8/fp16KV | 16 | 3,599.9 | 3,618.0 | 0.005 | ceiling | slowed inf -> 3,640 |
| 159 | Q38FN/5xB300 tensor-par/nvfp4/fp16KV | 16 | 817.0 | 842.3 | 0.03 | ceiling | ttft 4,334 -> 1,043; sat 4,362 -> 1,046; wait 4,278 -> 1,038 |
| 160 | GLM52/4xB300 DP2xTP2/nvfp4/fp8KV | 16 | 18.0 | 18.6 | 0.03 | ceiling | ttft 47 -> 23; sat 50 -> 25; slowed inf -> 20; wait 33 -> 20 |
| 164 | 35BA3B/4xB300 data-par/nvfp4/fp8KV | 8 | 1,276.9 | 1,418.8 | 0.1 | ceiling | ttft 1,652 -> 1,460; sat 1,656 -> 1,461; wait 1,611 -> 1,452 |
| 169 | MM35/4xB300 data-par/fp8/fp16KV | 16 | 7.0 | 7.1 | 0.01 | ceiling+load (18 at-load keys) | ttft 85 -> 8; sat 93 -> 13; slowed inf -> 8; wait 55 -> 8 |
| 170 | 27B/4xH200 DP2xTP2/fp8/fp16KV | 32 | 208.9 | 210.0 | 0.005 | ceiling | ttft 953 -> 229; sat 970 -> 231; slowed 365 -> 211; wait 871 -> 222 |
| 171 | 35BA3B/6xH200 DP3xTP2/fp8/fp16KV | 10.6667 | 605.9 | 624.6 | 0.03 | ceiling | ttft 1,038 -> 648; sat 1,055 -> 649; wait 1,004 -> 645 |
| 172 | 27B/3xB300 data-par/nvfp4/fp8KV | 10.6667 | 256.9 | 258.2 | 0.005 | ceiling | ttft 420 -> 280; sat 423 -> 281; slowed inf -> 262; wait 390 -> 270 |
| 173 | MM35/3xB300 tensor-par/nvfp4/fp16KV | 16 | 61.0 | 67.8 | 0.1 | ceiling | ttft 219 -> 90; sat 233 -> 94; wait 114 -> 73 |
| 174 | GLM52/6xB300 DP2xTP3/fp8/fp8KV | 200 | 5.0 | 7.7 | 0.35 | ceiling+load (1 at-load keys) | ttft 25 -> 14; sat 33 -> 18; wait 12 -> 9 |
| 175 | 35BA3B/1xH200/fp8/fp8KV | 128 | 206.0 | 208.1 | 0.01 | ceiling | ttft 305 -> 221; sat 310 -> 222; slowed 276 -> 209; wait 296 -> 218 |
| 176 | MM35/5xB300 tensor-par/nvfp4/fp16KV | 400 | 166.9 | 168.6 | 0.01 | ceiling+load (2 at-load keys) | ttft 245 -> 173; sat 255 -> 175; slowed inf -> 170; wait 225 -> 171 |
| 177 | MM35/4xB300 tensor-par/fp8/fp16KV | 16 | 99.0 | 99.4 | 0.005 | ceiling | slowed inf -> 100 |
| 178 | 35BA3B/3xB300 data-par/nvfp4/fp16KV | 21.3333 | 552.9 | 558.4 | 0.01 | ceiling | ttft 847 -> 632; sat 848 -> 632; slowed inf -> 580; wait 840 -> 627 |
| 182 | Q38FN/8xB300 DP4xTP2/fp8/fp16KV | 4 | 836.9 | 862.8 | 0.03 | ceiling | ttft 870 -> 865; sat 872 -> 865 |
| 184 | 27B/6xH200 data-par/fp8/fp8KV | 21.3333 | 494.0 | 496.5 | 0.005 | ceiling | slowed inf -> 497 |
| 185 | 35BA3B/8xH200 DP2xTP4/fp8/fp16KV | 16 | 502.9 | 518.5 | 0.03 | ceiling | ttft 1,043 -> 549; sat 1,047 -> 549; wait 1,004 -> 545 |
| 186 | 35BA3B/8xH200 DP4xTP2/fp8/fp16KV | 16 | 286.9 | 295.8 | 0.03 | ceiling | ttft 1,568 -> 412; sat 1,583 -> 415; wait 1,538 -> 405 |
| 187 | 27B/3xB300 data-par/nvfp4/fp16KV | 21.3333 | 316.0 | 317.6 | 0.005 | ceiling | ttft 2,358 -> 447; sat 2,371 -> 454; slowed inf -> 331; wait 2,293 -> 417 |
| 188 | 35BA3B/8xB300 DP2xTP4/nvfp4/fp16KV | 200 | 5,043.8 | 5,069.1 | 0.005 | ceiling | ttft 7,650 -> 5,244; sat 7,656 -> 5,246; wait 7,638 -> 5,242 |
| 191 | GLM52/7xH200 tensor-par/fp8/fp8KV | 400 | 150.0 | 151.5 | 0.01 | ceiling+load (2 at-load keys) | ttft 159 -> 152; sat 164 -> 152; slowed 154 -> 152 |
| 192 | Q38FN/3xB300 data-par/nvfp4/fp8KV | 133.333 | 229.9 | 231.1 | 0.005 | ceiling | ttft 2,293 -> 368; sat 2,297 -> 370; slowed inf -> 244; wait 2,248 -> 346 |
| 193 | GLM53F/6xB300 DP3xTP2/fp8/fp8KV | 10.6667 | 1,205.0 | 1,217.2 | 0.01 | ceiling | slowed inf -> 1,223 |
| 200 | DSV4FV/8xB300 tensor-par/fp8/fp8KV | 400 | 19,985.5 | 20,086.0 | 0.005 | ceiling | ttft 26,387 -> 20,284; sat 26,410 -> 20,285; wait 26,270 -> 20,276 |
| 201 | GLM53F/1xB300/nvfp4/fp8KV | 128 | 265.9 | 268.6 | 0.01 | ceiling | ttft 483 -> 301; sat 495 -> 308; slowed inf -> 278; wait 461 -> 294 |
| 202 | Q38FN/5xB300 data-par/nvfp4/fp8KV | 80 | 247.9 | 255.6 | 0.03 | ceiling | ttft 5,343 -> 576; sat 5,392 -> 582; slowed 1,619 -> 297; wait 5,105 -> 547 |
| 203 | DSV4FV/4xH200 DP2xTP2/fp8/fp8KV | 200 | 1,648.9 | 1,657.2 | 0.005 | ceiling | slowed inf -> 1,666 |
| 204 | MM35/8xB300 DP2xTP4/fp8/fp16KV | 16 | 523.8 | 526.4 | 0.005 | ceiling | slowed inf -> 550 |
| 205 | 35BA3B/7xB300 tensor-par/nvfp4/fp8KV | 400 | 8,073.9 | 8,970.9 | 0.1 | ceiling | ttft 18,447 -> 10,238; sat 18,501 -> 10,247; wait 18,339 -> 10,222 |
| 206 | GLM53F/3xB300 tensor-par/fp8/fp16KV | 128 | 568.0 | 585.5 | 0.03 | ceiling | ttft 819 -> 605; sat 827 -> 606; wait 783 -> 601 |
| 208 | Q38FN/4xB300 data-par/fp8/fp16KV | 100 | 43.0 | 66.2 | 0.35 | ceiling+load (18 at-load keys) | ttft 275 -> 149; sat 280 -> 151; wait 254 -> 140 |
| 214 | 35BA3B/8xB300 DP4xTP2/nvfp4/fp8KV | 16 | 1,865.0 | 1,883.8 | 0.01 | ceiling | ttft 2,256 -> 1,898; sat 2,269 -> 1,898; wait 2,231 -> 1,896 |
| 216 | MM35/6xH200 tensor-par/fp8/fp8KV | 400 | 93.0 | 95.8 | 0.03 | load (2 at-load keys) |  |
| 217 | 27B/7xB300 data-par/nvfp4/fp16KV | 57.1429 | 120.0 | 121.2 | 0.01 | ceiling | ttft 1,410 -> 160; sat 1,475 -> 171; slowed 675 -> 127; wait 1,282 -> 151 |
| 218 | MM35/5xB300 data-par/fp8/fp16KV | 3.2 | 12.0 | 12.4 | 0.03 | ceiling | ttft 64 -> 16; sat 78 -> 36; slowed inf -> 16; wait 48 -> 16 |
| 220 | Q38FN/7xB300 data-par/nvfp4/fp16KV | 2.28571 | 63.0 | 96.9 | 0.35 | ceiling | ttft 135 -> 114; sat 138 -> 115; wait 112 -> 103 |
| 221 | 35BA3B/6xH200 DP2xTP3/fp8/fp16KV | 8 | 881.0 | 889.9 | 0.01 | ceiling | ttft 2,527 -> 1,013; sat 2,529 -> 1,014; slowed 1,483 -> 896; wait 2,500 -> 1,005 |
| 223 | 27B/1xH200/fp8/fp8KV | 64 | 66.0 | 66.6 | 0.01 | ceiling | ttft 142 -> 70; sat 155 -> 72; slowed 133 -> 67; wait 88 -> 67 |
| 224 | 27B/4xB300 tensor-par/fp8/fp8KV | 64 | 2,135.8 | 2,157.4 | 0.01 | ceiling | slowed inf -> 2,179 |
| 225 | GLM52/5xB300 tensor-par/fp8/fp8KV | 128 | 317.9 | 321.2 | 0.01 | ceiling | ttft 553 -> 327; sat 583 -> 328 |
| 226 | Q38FN/4xB300 data-par/nvfp4/fp8KV | 4 | 191.9 | 193.8 | 0.01 | ceiling | ttft 1,674 -> 262; sat 1,747 -> 282; slowed 823 -> 200; wait 1,532 -> 237 |
| 227 | 35BA3B/4xB300 tensor-par/nvfp4/fp16KV | 16 | 1,122.0 | 1,156.6 | 0.03 | ceiling | ttft 2,732 -> 1,239; sat 2,776 -> 1,242; wait 2,643 -> 1,232 |
| 228 | GLM52/8xB300 DP2xTP4/fp8/fp8KV | 8 | 118.0 | 121.6 | 0.03 | ceiling | ttft 158 -> 124; sat 166 -> 125 |
| 229 | 35BA3B/8xB300 data-par/nvfp4/fp8KV | 16 | 300.9 | 463.0 | 0.35 | ceiling | ttft 493 -> 476; sat 506 -> 482 |
| 230 | MM35/3xB300 data-par/nvfp4/fp16KV | 21.3333 | 3.0 | 3.3 | 0.1 | ceiling+load (18 at-load keys) | sat 27 -> 8; slowed 8 -> 4 |
| 232 | GLM53F/4xB300 data-par/nvfp4/fp8KV | 4 | 145.0 | 149.5 | 0.03 | ceiling | ttft 460 -> 169; sat 514 -> 177; slowed 182 -> 150; wait 247 -> 153 |
| 233 | MM35/7xH200 tensor-par/fp8/fp16KV | 32 | 34.0 | 35.1 | 0.03 | ceiling | ttft 191 -> 38; sat 267 -> 50; slowed 87 -> 37; wait 85 -> 37 |
| 234 | 27B/1xH200/fp8/fp8KV | 64 | 70.0 | 70.7 | 0.01 | ceiling | ttft 149 -> 74; sat 157 -> 76; slowed 135 -> 71; wait 110 -> 72 |
| 235 | 35BA3B/2xH200 tensor-par/fp8/fp8KV | 64 | 576.0 | 581.8 | 0.01 | ceiling | ttft 1,588 -> 643; sat 1,599 -> 646; slowed 1,054 -> 586; wait 1,533 -> 631 |
| 236 | GLM52/8xH200 tensor-par/fp8/fp8KV | 64 | 129.0 | 130.3 | 0.01 | ceiling | ttft 200 -> 134; sat 212 -> 136; slowed 195 -> 131; wait 146 -> 131 |
| 237 | MM35/8xH200 DP4xTP2/fp8/fp8KV | 16 | 12.0 | 12.1 | 0.01 | ceiling+load (18 at-load keys) | ttft 35 -> 13; sat 44 -> 14; slowed 41 -> 12 |
| 238 | MM35/1xB300/nvfp4/fp8KV | 400 | 927.9 | 932.6 | 0.005 | ceiling | slowed inf -> 938 |
| 242 | Q38FN/2xB300 data-par/fp8/fp16KV | 64 | 35.0 | 36.1 | 0.03 | ceiling+load (18 at-load keys) | ttft 985 -> 78; sat 1,067 -> 86; slowed 226 -> 42; wait 827 -> 69 |
| 245 | 27B/2xB300 data-par/fp8/fp8KV | 8 | 327.0 | 363.3 | 0.1 | ceiling | ttft 500 -> 390; sat 512 -> 394; wait 444 -> 378 |
| 248 | 27B/3xH200 data-par/fp8/fp16KV | 133.333 | 18.0 | 18.6 | 0.03 | ceiling+load (18 at-load keys) | ttft 265 -> 23; sat 401 -> 39; slowed 33 -> 19; wait 93 -> 21 |
| 249 | 35BA3B/6xB300 data-par/nvfp4/fp16KV | 2.66667 | 486.9 | 491.9 | 0.01 | ceiling | ttft 1,010 -> 601; sat 1,013 -> 604; wait 996 -> 585 |
| 252 | 27B/1xH200/fp8/fp8KV | 64 | 70.0 | 70.7 | 0.01 | ceiling | ttft 147 -> 74; sat 156 -> 76; slowed 89 -> 71; wait 108 -> 71 |
| 253 | 35BA3B/2xH200 tensor-par/fp8/fp8KV | 64 | 576.0 | 581.8 | 0.01 | ceiling | ttft 1,521 -> 629; sat 1,536 -> 631; wait 1,451 -> 619 |
| 254 | GLM52/8xH200 tensor-par/fp8/fp8KV | 64 | 129.0 | 130.3 | 0.01 | ceiling | ttft 190 -> 133; sat 205 -> 134 |
| 255 | MM35/8xH200 DP4xTP2/fp8/fp8KV | 16 | 12.0 | 12.1 | 0.01 | ceiling+load (18 at-load keys) | ttft 35 -> 13; sat 44 -> 14; slowed 38 -> 12 |
| 256 | GLM53F/6xB300 DP3xTP2/fp8/fp16KV | 133.333 | 771.0 | 778.7 | 0.01 | ceiling | slowed inf -> 788 |
| 257 | MM35/6xB300 DP2xTP3/nvfp4/fp8KV | 64 | 54.0 | 54.5 | 0.01 | ceiling+load (18 at-load keys) | ttft 70 -> 56; sat 75 -> 57; slowed inf -> 55 |
| 260 | 27B/1xH200/fp8/fp8KV | 64 | 70.0 | 70.7 | 0.01 | ceiling | ttft 149 -> 74; sat 157 -> 76; slowed 140 -> 71 |
| 261 | 27B/1xH200/fp8/fp8KV | 64 | 70.0 | 70.7 | 0.01 | ceiling | ttft 149 -> 74; sat 157 -> 76; slowed 140 -> 71; wait 149 -> 74 |
| 262 | 35BA3B/8xB300 DP4xTP2/fp8/fp8KV | 4 | 2,407.9 | 2,432.3 | 0.01 | ceiling | ttft 11,960 -> 3,248; sat 11,962 -> 3,249; slowed 4,186 -> 2,450; wait 11,940 -> 3,239 |
| 263 | MM35/6xB300 DP3xTP2/nvfp4/fp16KV | 42.6667 | 14.0 | 15.6 | 0.1 | ceiling+load (18 at-load keys) | ttft 52 -> 18; sat 75 -> 23 |
| 264 | GLM53F/4xB300 data-par/nvfp4/fp16KV | 8 | 93.0 | 95.8 | 0.03 | ceiling | ttft 139 -> 107; sat 140 -> 107; slowed inf -> 98; wait 125 -> 101 |
| 265 | Q38FN/6xH200 DP3xTP2/fp8/fp8KV | 42.6667 | 693.0 | 699.9 | 0.01 | ceiling | ttft 1,454 -> 731; sat 1,465 -> 733; wait 1,397 -> 726 |
| 267 | 27B/1xH200/fp8/fp8KV | 64 | 70.0 | 70.7 | 0.01 | ceiling | ttft 149 -> 74; sat 157 -> 76; slowed 136 -> 71; wait 111 -> 72 |
| 268 | 35BA3B/2xH200 tensor-par/fp8/fp8KV | 64 | 576.0 | 581.8 | 0.01 | ceiling | ttft 1,593 -> 645; sat 1,604 -> 647; slowed 1,116 -> 587; wait 1,539 -> 632 |
| 269 | GLM52/8xH200 tensor-par/fp8/fp8KV | 64 | 129.0 | 130.3 | 0.01 | ceiling | ttft 201 -> 134; sat 213 -> 136; slowed 200 -> 131; wait 147 -> 131 |
| 270 | MM35/8xH200 DP4xTP2/fp8/fp8KV | 16 | 12.0 | 12.1 | 0.01 | ceiling+load (18 at-load keys) | ttft 35 -> 13; sat 44 -> 14; slowed 41 -> 12 |
| 271 | 35BA3B/5xB300 tensor-par/fp8/fp16KV | 64 | 5,762.9 | 6,403.2 | 0.1 | ceiling | ttft 14,834 -> 7,617; sat 14,844 -> 7,619; wait 14,722 -> 7,597 |
| 272 | Q38FN/1xB300/nvfp4/fp8KV | 16 | 965.7 | 995.6 | 0.03 | ceiling | ttft 1,164 -> 1,013; sat 1,175 -> 1,015; wait 1,142 -> 1,010 |
| 273 | 35BA3B/2xH200 tensor-par/fp8/fp16KV | 128 | 118.0 | 118.6 | 0.005 | ceiling+load (18 at-load keys) | ttft 1,585 -> 180; sat 1,592 -> 182; slowed inf -> 127; wait 1,513 -> 161 |
| 275 | Q38FN/8xH200 tensor-par/fp8/fp16KV | 128 | 865.0 | 1,330.7 | 0.35 | ceiling | ttft 1,689 -> 1,484; sat 1,693 -> 1,486; wait 1,644 -> 1,464 |
| 276 | 27B/1xH200/fp8/fp8KV | 200 | 70.0 | 70.7 | 0.01 | ceiling+load (2 at-load keys) | ttft 97 -> 72; sat 105 -> 73; slowed 94 -> 71 |
| 277 | MM35/8xH200 DP4xTP2/fp8/fp8KV | 100 | 12.0 | 12.1 | 0.01 | ceiling+load (2 at-load keys) | ttft 35 -> 13; sat 44 -> 14; slowed 41 -> 12 |
| 278 | 27B/1xH200/fp8/fp8KV | 64 | 70.0 | 70.7 | 0.01 | ceiling | ttft 149 -> 74; sat 157 -> 76; slowed 140 -> 71; wait 111 -> 72 |
| 279 | 35BA3B/2xH200 tensor-par/fp8/fp8KV | 64 | 576.0 | 581.8 | 0.01 | ceiling | ttft 1,593 -> 645; sat 1,604 -> 647; slowed 1,295 -> 589; wait 1,539 -> 632 |
| 280 | GLM52/8xH200 tensor-par/fp8/fp8KV | 64 | 129.0 | 130.3 | 0.01 | ceiling | ttft 201 -> 134; sat 213 -> 136; slowed 205 -> 131; wait 147 -> 131 |
| 281 | MM35/8xH200 DP4xTP2/fp8/fp8KV | 16 | 12.0 | 12.1 | 0.01 | ceiling+load (18 at-load keys) | ttft 35 -> 13; sat 44 -> 14; slowed 41 -> 12 |
| 282 | 35BA3B/2xH200 tensor-par/fp8/fp8KV | 200 | 576.0 | 581.8 | 0.01 | ceiling | ttft 1,593 -> 645; sat 1,604 -> 647; slowed 1,295 -> 589; wait 1,539 -> 632 |
| 283 | 27B/1xH200/fp8/fp8KV | 64 | 70.0 | 70.7 | 0.01 | ceiling | ttft 149 -> 74; sat 157 -> 76; slowed 140 -> 71; wait 111 -> 72 |
| 284 | 27B/1xB300/fp8/fp8KV | 64 | 189.9 | 191.9 | 0.01 | ceiling | ttft 349 -> 201; sat 358 -> 202; slowed 307 -> 193; wait 308 -> 196 |
| 285 | 27B/2xB300 tensor-par/fp8/fp8KV | 64 | 424.9 | 429.2 | 0.01 | ceiling | ttft 636 -> 441; sat 644 -> 442; slowed 524 -> 430; wait 593 -> 436 |
| 286 | 27B/3xB300 tensor-par/fp8/fp8KV | 64 | 666.0 | 672.7 | 0.01 | ceiling | ttft 900 -> 686; sat 909 -> 687; slowed 706 -> 673; wait 857 -> 682 |
| 287 | 27B/4xB300 tensor-par/fp8/fp8KV | 64 | 903.0 | 912.1 | 0.01 | ceiling | ttft 1,151 -> 926; sat 1,160 -> 927; wait 1,108 -> 922 |
| 288 | 27B/5xB300 tensor-par/fp8/fp8KV | 64 | 1,149.0 | 1,160.6 | 0.01 | ceiling | ttft 1,393 -> 1,174; sat 1,402 -> 1,175; wait 1,349 -> 1,170 |
| 289 | 27B/6xB300 tensor-par/fp8/fp8KV | 64 | 1,389.9 | 1,403.9 | 0.01 | ceiling | ttft 1,627 -> 1,417; sat 1,636 -> 1,418; wait 1,584 -> 1,414 |
| 290 | 27B/7xB300 tensor-par/fp8/fp8KV | 64 | 1,637.0 | 1,653.5 | 0.01 | ceiling | ttft 1,856 -> 1,666; sat 1,865 -> 1,666; wait 1,812 -> 1,662 |
| 291 | 27B/8xB300 tensor-par/fp8/fp8KV | 64 | 1,894.9 | 1,914.0 | 0.01 | ceiling | ttft 2,079 -> 1,924; sat 2,088 -> 1,925; wait 2,036 -> 1,921 |
| 292 | 35BA3B/1xB300/fp8/fp8KV | 64 | 624.0 | 630.3 | 0.01 | ceiling | ttft 2,015 -> 717; sat 2,026 -> 720; slowed 1,569 -> 640; wait 1,961 -> 704 |
| 293 | 35BA3B/2xB300 tensor-par/fp8/fp8KV | 64 | 1,366.0 | 1,379.7 | 0.01 | ceiling | ttft 3,636 -> 1,522; sat 3,647 -> 1,525; slowed 2,444 -> 1,390; wait 3,581 -> 1,510 |
| 294 | 35BA3B/3xB300 tensor-par/fp8/fp8KV | 64 | 2,111.9 | 2,133.2 | 0.01 | ceiling | ttft 5,132 -> 2,324; sat 5,143 -> 2,326; slowed 3,065 -> 2,143; wait 5,078 -> 2,311 |
| 295 | 35BA3B/4xB300 tensor-par/fp8/fp8KV | 64 | 2,872.8 | 2,901.9 | 0.01 | ceiling | ttft 6,553 -> 3,134; sat 6,564 -> 3,137; slowed 3,540 -> 2,908; wait 6,499 -> 3,122 |
| 296 | 35BA3B/5xB300 tensor-par/fp8/fp8KV | 64 | 3,635.9 | 3,672.6 | 0.01 | ceiling | ttft 7,921 -> 3,943; sat 7,932 -> 3,946; slowed 3,918 -> 3,675; wait 7,866 -> 3,931 |
| 297 | 35BA3B/6xB300 tensor-par/fp8/fp8KV | 64 | 4,370.9 | 4,415.1 | 0.01 | ceiling | ttft 9,247 -> 4,723; sat 9,258 -> 4,726; wait 9,192 -> 4,711 |
| 298 | 35BA3B/7xB300 tensor-par/fp8/fp8KV | 64 | 5,152.7 | 5,204.7 | 0.01 | ceiling | ttft 10,540 -> 5,545; sat 10,551 -> 5,548; wait 10,485 -> 5,534 |
| 299 | 35BA3B/8xB300 tensor-par/fp8/fp8KV | 64 | 5,915.9 | 5,975.7 | 0.01 | ceiling | ttft 11,805 -> 6,348; sat 11,816 -> 6,350; wait 11,750 -> 6,337 |
| 300 | DSV41F/2xB300 tensor-par/fp8/fp8KV | 64 | 256.9 | 259.5 | 0.01 | ceiling | ttft 2,514 -> 382; sat 2,522 -> 385; slowed 1,619 -> 273; wait 2,473 -> 373 |
| 312 | DSV4F/1xB300/fp8/fp8KV | 64 | 731.0 | 738.3 | 0.01 | ceiling | ttft 772 -> 740; sat 780 -> 741 |
| 321 | DSV4FV/1xB300/fp8/fp8KV | 64 | 723.0 | 730.3 | 0.01 | ceiling | ttft 771 -> 733; sat 780 -> 733; wait 731 -> 730 |
| 329 | GLM52/3xB300 tensor-par/fp8/fp8KV | 64 | 8.9 | 9.0 | 0.01 | ceiling+load (18 at-load keys) | ttft 199 -> 17; sat 211 -> 22; slowed 203 -> 11; wait 146 -> 11 |
| 330 | GLM52/4xB300 tensor-par/fp8/fp8KV | 64 | 171.0 | 172.7 | 0.01 | ceiling | ttft 257 -> 177; sat 269 -> 179; slowed 257 -> 174; wait 202 -> 174 |
| 335 | GLM53F/2xB300 tensor-par/fp8/fp8KV | 64 | 645.9 | 652.4 | 0.01 | ceiling | ttft 673 -> 654; sat 683 -> 654 |
| 342 | MM35/1xB300/fp8/fp8KV | 64 | 16.0 | 16.2 | 0.01 | ceiling+load (2 at-load keys) | ttft 47 -> 17; sat 56 -> 19; slowed 52 -> 17; wait 18 -> 16 |
| 343 | MM35/2xB300 tensor-par/fp8/fp8KV | 64 | 55.0 | 55.6 | 0.01 | ceiling+load (18 at-load keys) | ttft 91 -> 57; sat 100 -> 58; slowed 94 -> 56 |
| 344 | MM35/3xB300 tensor-par/fp8/fp8KV | 64 | 102.0 | 103.0 | 0.01 | ceiling | ttft 132 -> 104; sat 141 -> 105; slowed 132 -> 103 |
| 345 | MM35/4xB300 tensor-par/fp8/fp8KV | 64 | 147.9 | 149.4 | 0.01 | ceiling | ttft 171 -> 151; sat 180 -> 151; slowed 167 -> 150 |
| 346 | MM35/5xB300 tensor-par/fp8/fp8KV | 64 | 191.9 | 193.8 | 0.01 | ceiling | ttft 208 -> 195; sat 218 -> 195; slowed 202 -> 194 |
| 347 | MM35/6xB300 tensor-par/fp8/fp8KV | 64 | 245.9 | 248.4 | 0.01 | ceiling | sat 254 -> 249 |
| 348 | MM35/7xB300 tensor-par/fp8/fp8KV | 64 | 281.9 | 284.8 | 0.01 | ceiling | sat 290 -> 285 |
| 350 | Q38FN/1xB300/fp8/fp8KV | 64 | 147.9 | 149.4 | 0.01 | ceiling | ttft 960 -> 198; sat 970 -> 201; slowed 824 -> 156; wait 910 -> 185 |
| 351 | Q38FN/2xB300 tensor-par/fp8/fp8KV | 64 | 709.0 | 716.1 | 0.01 | ceiling | ttft 1,735 -> 779; sat 1,745 -> 781; slowed 1,369 -> 723; wait 1,685 -> 768 |
| 352 | Q38FN/3xB300 tensor-par/fp8/fp8KV | 64 | 1,300.7 | 1,313.8 | 0.01 | ceiling | ttft 2,451 -> 1,384; sat 2,461 -> 1,386; slowed 1,804 -> 1,319; wait 2,401 -> 1,375 |
| 353 | Q38FN/4xB300 tensor-par/fp8/fp8KV | 64 | 1,884.8 | 1,903.9 | 0.01 | ceiling | ttft 3,131 -> 1,980; sat 3,141 -> 1,982; slowed 2,168 -> 1,907; wait 3,081 -> 1,972 |
| 354 | Q38FN/5xB300 tensor-par/fp8/fp8KV | 64 | 2,441.9 | 2,466.6 | 0.01 | ceiling | ttft 3,786 -> 2,549; sat 3,796 -> 2,551; slowed 2,479 -> 2,467; wait 3,736 -> 2,541 |
| 355 | Q38FN/6xB300 tensor-par/fp8/fp8KV | 64 | 3,046.9 | 3,077.7 | 0.01 | ceiling | ttft 4,420 -> 3,162; sat 4,430 -> 3,163; wait 4,370 -> 3,154 |
| 356 | Q38FN/7xB300 tensor-par/fp8/fp8KV | 64 | 3,637.9 | 3,674.6 | 0.01 | ceiling | ttft 5,039 -> 3,760; sat 5,049 -> 3,761; wait 4,989 -> 3,753 |
| 357 | Q38FN/8xB300 tensor-par/fp8/fp8KV | 64 | 4,221.9 | 4,264.5 | 0.01 | ceiling | ttft 5,644 -> 4,351; sat 5,654 -> 4,352; wait 5,594 -> 4,345 |
| 358 | 35BA3B/8xB300 tensor-par [KV replicated]/fp8/fp8KV | 64 | 1,570.0 | 1,585.8 | 0.01 | ceiling | ttft 11,805 -> 2,238; sat 11,816 -> 2,242; slowed 4,714 -> 1,617; wait 11,750 -> 2,221 |
| 359 | 35BA3B/8xB300 tensor-par [KV replicated]/fp8/fp8KV | 64 | 1,790.0 | 1,808.1 | 0.01 | ceiling | ttft 11,805 -> 2,446; sat 11,816 -> 2,450; slowed 4,714 -> 1,837; wait 11,750 -> 2,429 |
| 360 | DSV41F/8xB300 tensor-par [KV replicated]/fp8/fp8KV | 64 | 6,550.8 | 6,617.0 | 0.01 | ceiling | ttft 8,163 -> 6,702; sat 8,171 -> 6,703; wait 8,122 -> 6,698 |
| 361 | DSV41F/8xB300 tensor-par [KV replicated]/fp8/fp8KV | 64 | 7,712.0 | 7,789.9 | 0.01 | ceiling | ttft 8,163 -> 7,810; sat 8,171 -> 7,811; wait 8,122 -> 7,808 |
| 362 | Q38FN/8xB300 tensor-par [KV replicated]/fp8/fp8KV | 64 | 1,151.9 | 1,163.5 | 0.01 | ceiling | ttft 5,644 -> 1,444; sat 5,654 -> 1,447; slowed 3,205 -> 1,184; wait 5,594 -> 1,430 |
| 363 | Q38FN/8xB300 tensor-par [KV replicated]/fp8/fp8KV | 64 | 1,322.9 | 1,336.3 | 0.01 | ceiling | ttft 5,644 -> 1,606; sat 5,654 -> 1,609; slowed 3,205 -> 1,355; wait 5,594 -> 1,592 |
| 364 | 27B/1xH200/fp8/fp8KV | 64 | 70.0 | 70.7 | 0.01 | ceiling | ttft 149 -> 74; sat 157 -> 76; slowed 140 -> 71; wait 111 -> 72 |
| 365 | 27B/1xH200/fp8/fp8KV | 64 | 70.0 | 70.7 | 0.01 | ceiling | ttft 149 -> 74; sat 157 -> 76; slowed 140 -> 71; wait 111 -> 72 |
| 366 | 27B/1xH200/fp8/fp8KV | 64 | 72.0 | 72.0 | 0 | ceiling | ttft 185 -> 76; sat 186 -> 78; slowed inf -> 73; wait 178 -> 73 |
| 367 | 35BA3B/2xH200 tensor-par/fp8/fp8KV | 64 | 585.0 | 585.0 | 0 | ceiling | ttft 1,879 -> 653; sat 1,881 -> 656; slowed inf -> 598; wait 1,871 -> 641 |
| 368 | GLM52/8xH200 tensor-par/fp8/fp8KV | 64 | 131.0 | 131.0 | 0 | ceiling | ttft 246 -> 136; sat 248 -> 138; slowed inf -> 133; wait 237 -> 133 |
| 369 | MM35/8xH200 DP4xTP2/fp8/fp8KV | 16 | 12.9 | 12.9 | 0 | ceiling+load (18 at-load keys) | ttft 50 -> 14; sat 52 -> 15; slowed inf -> 13; wait 43 -> 13 |
| 370 | 27B/1xH200/fp8/fp8KV | 64 | 70.0 | 70.7 | 0.01 | ceiling | ttft 149 -> 74; sat 157 -> 76; slowed 140 -> 71; wait 111 -> 72 |
| 371 | 35BA3B/2xH200 tensor-par/fp8/fp8KV | 64 | 576.0 | 581.8 | 0.01 | ceiling | ttft 1,593 -> 645; sat 1,604 -> 647; slowed 1,295 -> 589; wait 1,539 -> 632 |
| 372 | GLM52/8xH200 tensor-par/fp8/fp8KV | 64 | 129.0 | 130.3 | 0.01 | ceiling | ttft 201 -> 134; sat 213 -> 136; slowed 205 -> 131; wait 147 -> 131 |
| 373 | MM35/8xH200 DP4xTP2/fp8/fp8KV | 16 | 12.0 | 12.1 | 0.01 | ceiling+load (18 at-load keys) | ttft 35 -> 13; sat 44 -> 14; slowed 41 -> 12 |
| 374 | 27B/1xH200/fp8/fp8KV | 64 | 70.0 | 70.7 | 0.01 | ceiling | ttft 114 -> 73; sat 122 -> 74; slowed 110 -> 71; wait 77 -> 71 |
| 375 | 27B/1xH200/fp8/fp8KV | 64 | 70.0 | 70.7 | 0.01 | ceiling | ttft 149 -> 74; sat 157 -> 76; slowed 140 -> 71; wait 111 -> 72 |
| 376 | 35BA3B/2xH200 tensor-par/fp8/fp8KV | 64 | 576.0 | 581.8 | 0.01 | ceiling | ttft 1,593 -> 645; sat 1,604 -> 647; slowed 1,295 -> 589; wait 1,539 -> 632 |
| 377 | MM35/8xH200 DP4xTP2/fp8/fp8KV | 16 | 12.0 | 12.1 | 0.01 | ceiling+load (18 at-load keys) | ttft 35 -> 13; sat 44 -> 14; slowed 41 -> 12 |
| 378 | 27B/1xH200/fp8/fp8KV | 64 | 70.0 | 70.7 | 0.01 | ceiling | ttft 149 -> 74; sat 157 -> 76; slowed 140 -> 71; wait 111 -> 72 |
| 379 | 35BA3B/2xH200 tensor-par/fp8/fp8KV | 64 | 576.0 | 581.8 | 0.01 | ceiling | ttft 1,593 -> 645; sat 1,604 -> 647; slowed 1,295 -> 589; wait 1,539 -> 632 |
| 380 | MM35/8xH200 DP4xTP2/fp8/fp8KV | 16 | 12.0 | 12.1 | 0.01 | ceiling+load (18 at-load keys) | ttft 35 -> 13; sat 44 -> 14; slowed 41 -> 12 |
| 381 | 27B/1xH200/fp8/fp8KV | 64 | 70.0 | 70.7 | 0.01 | ceiling | ttft 149 -> 74; sat 157 -> 76; slowed 140 -> 71; wait 111 -> 72 |
| 382 | 35BA3B/2xH200 tensor-par/fp8/fp8KV | 64 | 576.0 | 581.8 | 0.01 | ceiling | ttft 1,593 -> 645; sat 1,604 -> 647; slowed 1,295 -> 589; wait 1,539 -> 632 |
| 383 | MM35/8xH200 DP4xTP2/fp8/fp8KV | 16 | 12.0 | 12.1 | 0.01 | ceiling+load (18 at-load keys) | ttft 35 -> 13; sat 44 -> 14; slowed 41 -> 12 |
| 384 | 27B/1xH200/fp8/fp8KV | 64 | 70.0 | 70.7 | 0.01 | ceiling | ttft 149 -> 74; sat 157 -> 76; slowed 143 -> 71; wait 111 -> 72 |
| 385 | 27B/1xH200/fp8/fp8KV | 64 | 70.0 | 70.7 | 0.01 | ceiling | ttft 149 -> 74; sat 157 -> 76; slowed 142 -> 71; wait 111 -> 72 |
| 386 | 27B/1xH200/fp8/fp8KV | 64 | 70.0 | 70.7 | 0.01 | ceiling | ttft 149 -> 74; sat 157 -> 76; slowed 140 -> 71; wait 111 -> 72 |
| 387 | 27B/1xH200/fp8/fp8KV | 64 | 70.0 | 70.7 | 0.01 | ceiling | ttft 149 -> 74; sat 157 -> 76; slowed 140 -> 71; wait 111 -> 72 |
| 388 | 27B/1xH200/fp8/fp8KV | 64 | 70.0 | 70.7 | 0.01 | ceiling | ttft 149 -> 74; sat 157 -> 76; slowed 140 -> 71; wait 111 -> 72 |
| 389 | 27B/1xH200/fp8/fp8KV | 64 | 70.0 | 70.7 | 0.01 | ceiling | ttft 149 -> 74; sat 157 -> 76; wait 111 -> 72 |
| 390 | 27B/1xH200/fp8/fp8KV | 64 | 73.0 | 73.7 | 0.01 | ceiling | ttft 150 -> 77; sat 158 -> 79; slowed 143 -> 74; wait 111 -> 75 |
| 391 | 27B/1xH200/fp8/fp8KV | 64 | 60.0 | 60.6 | 0.01 | ceiling+load (18 at-load keys) | ttft 142 -> 64; sat 152 -> 66; slowed 126 -> 61; wait 102 -> 61 |
| 392 | 27B/1xH200/fp8/fp8KV | 64 | 73.0 | 73.7 | 0.01 | ceiling | ttft 159 -> 78; sat 169 -> 79; slowed 144 -> 74; wait 116 -> 75 |
| 393 | 27B/1xH200/fp8/fp8KV | 64 | 72.9 | 73.6 | 0.01 | ceiling | ttft 149 -> 77; sat 158 -> 79; slowed 141 -> 74; wait 111 -> 74 |
| 395 | 27B/1xH200/fp8/fp8KV | 64 | 70.0 | 70.7 | 0.01 | ceiling | ttft 596 -> 93; sat 630 -> 105; slowed 561 -> 76; wait 443 -> 77 |
| 396 | 27B/1xH200/fp8/fp8KV | 64 | 70.0 | 70.7 | 0.01 | ceiling | ttft 149 -> 75; sat 157 -> 76; slowed 140 -> 71; wait 121 -> 72 |
| 397 | 27B/1xH200/fp8/fp8KV | 64 | 70.0 | 70.7 | 0.01 | ceiling | slowed inf -> 75 |
| 398 | 27B/1xH200/fp8/fp8KV | 64 | 117.0 | 118.1 | 0.01 | ceiling | ttft 172 -> 122; sat 175 -> 123; slowed inf -> 119; wait 155 -> 120 |
| 399 | 27B/1xH200/fp8/fp8KV | 1000 | 70.0 | 70.7 | 0.01 | ceiling+load (1 at-load keys) | ttft 149 -> 74; sat 157 -> 76; slowed 140 -> 71; wait 111 -> 72 |
| 400 | 27B/1xH200/fp8/fp8KV | 200 | 70.0 | 70.7 | 0.01 | ceiling+load (2 at-load keys) | ttft 149 -> 74; sat 157 -> 76; slowed 140 -> 71; wait 111 -> 72 |
| 401 | 27B/1xH200/fp8/fp8KV | 64 | 70.0 | 70.7 | 0.01 | ceiling | ttft 149 -> 74; sat 157 -> 76; slowed 140 -> 71; wait 111 -> 72 |
| 402 | 27B/1xH200/fp8/fp8KV | 8 | 70.0 | 70.7 | 0.01 | ceiling | ttft 149 -> 74; sat 157 -> 76; slowed 140 -> 71; wait 111 -> 72 |
| 403 | 27B/1xH200/fp8/fp8KV | 64 | 45.0 | 45.5 | 0.01 | ceiling+load (18 at-load keys) | ttft 122 -> 48; sat 137 -> 50; slowed 93 -> 46; wait 63 -> 46 |
| 404 | 27B/1xH200/fp8/fp8KV | 64 | 540.0 | 545.5 | 0.01 | ceiling | slowed inf -> 553 |
| 405 | 27B/1xH200/fp8/fp8KV | 64 | 24.0 | 24.2 | 0.01 | ceiling+load (18 at-load keys) | ttft 86 -> 26; sat 102 -> 27; slowed 52 -> 25; wait 33 -> 24 |
| 406 | 27B/1xH200/fp8/fp8KV | 64 | 70.0 | 70.7 | 0.01 | ceiling | ttft 373 -> 76; sat 424 -> 79; slowed 140 -> 71; wait 179 -> 72 |
| 407 | 27B/1xH200/fp8/fp8KV | 64 | 70.0 | 70.7 | 0.01 | ceiling | slowed inf -> 73 |
| 408 | 27B/1xH200/fp8/fp8KV | 64 | 70.0 | 70.7 | 0.01 | ceiling | ttft 149 -> 74; sat 157 -> 76; slowed 140 -> 71; wait 86 -> 71 |
| 409 | 27B/1xH200/fp8/fp8KV | 64 | 70.0 | 70.7 | 0.01 | ceiling | ttft 146 -> 74; sat 157 -> 76; slowed 140 -> 71; wait 142 -> 74 |
| 410 | 27B/1xH200/fp8/fp8KV | 64 | 70.0 | 70.7 | 0.01 | ceiling | ttft 298 -> 81; sat 315 -> 86; slowed 280 -> 73; wait 221 -> 74 |
| 411 | 27B/1xH200/fp8/fp8KV | 64 | 53.0 | 53.5 | 0.01 | ceiling+load (18 at-load keys) | ttft 150 -> 58; sat 159 -> 60; slowed 143 -> 54; wait 111 -> 55 |
| 412 | 27B/1xH200/fp8/fp8KV | 64 | 99.0 | 99.9 | 0.01 | ceiling | ttft 143 -> 102; sat 151 -> 103; slowed 127 -> 100; wait 106 -> 100 |
| 413 | 27B/1xH200/fp8/fp8KV | 64 | 120.0 | 121.2 | 0.01 | ceiling | ttft 137 -> 122; sat 145 -> 122 |
| 414 | 27B/1xH200/fp8/fp8KV | 64 | 66.9 | 67.6 | 0.01 | ceiling | ttft 148 -> 71; sat 156 -> 73; slowed 137 -> 68; wait 109 -> 68 |
| 415 | 27B/1xH200/fp8/fp8KV | 64 | 73.0 | 73.7 | 0.01 | ceiling | ttft 148 -> 77; sat 157 -> 79; slowed 139 -> 74; wait 110 -> 74 |
| 416 | 27B/1xH200/fp8/fp8KV | 64 | 61.0 | 61.6 | 0.01 | ceiling+load (18 at-load keys) | ttft 119 -> 65; sat 124 -> 66; slowed inf -> 62; wait 93 -> 62 |
| 417 | 27B/1xH200/fp8/fp8KV | 64 | 51.0 | 51.5 | 0.01 | ceiling+load (18 at-load keys) | ttft 94 -> 54; sat 98 -> 55; slowed inf -> 52; wait 77 -> 52 |
| 418 | 27B/1xH200/fp8/fp8KV | 64 | 70.0 | 70.7 | 0.01 | ceiling | ttft 149 -> 74; sat 157 -> 76; slowed inf -> 72; wait 111 -> 72 |
| 419 | 27B/1xH200/fp8/fp8KV | 64 | 70.0 | 70.7 | 0.01 | ceiling | ttft 120 -> 72; sat 157 -> 76; slowed 140 -> 71; wait 111 -> 72 |
| 420 | 27B/1xH200/fp8/fp8KV | 64 | 70.0 | 70.7 | 0.01 | ceiling | ttft 155 -> 75; sat 157 -> 76; slowed 140 -> 71; wait 111 -> 72 |
| 421 | 27B/1xH200/fp8/fp8KV | 64 | 70.0 | 70.7 | 0.01 | ceiling | ttft 156 -> 76; sat 157 -> 76; slowed 140 -> 71; wait 111 -> 72 |
| 424 | 27B/1xH200/fp8/fp8KV | 64 | 70.0 | 70.7 | 0.01 | ceiling | ttft 149 -> 74; sat 157 -> 76; slowed 140 -> 71; wait 111 -> 72 |
| 425 | 27B/1xH200/fp8/fp8KV | 64 | 70.0 | 70.7 | 0.01 | ceiling | ttft 149 -> 74; sat 157 -> 76; slowed 140 -> 71; wait 111 -> 72 |
| 426 | 27B/2xH200 tensor-par/fp8/fp8KV | 64 | 177.9 | 179.7 | 0.01 | ceiling | ttft 275 -> 185; sat 283 -> 186; slowed 246 -> 180; wait 234 -> 181 |
| 427 | 27B/3xH200 tensor-par/fp8/fp8KV | 64 | 286.9 | 289.8 | 0.01 | ceiling | ttft 391 -> 295; sat 400 -> 297; slowed 340 -> 290; wait 349 -> 292 |
| 428 | 27B/4xH200 tensor-par/fp8/fp8KV | 64 | 406.9 | 411.0 | 0.01 | ceiling | ttft 501 -> 416; sat 510 -> 417; slowed 425 -> 411; wait 459 -> 413 |
| 429 | 27B/5xH200 tensor-par/fp8/fp8KV | 64 | 510.9 | 516.1 | 0.01 | ceiling | ttft 608 -> 521; sat 616 -> 522; wait 565 -> 518 |
| 430 | 27B/6xH200 tensor-par/fp8/fp8KV | 64 | 627.0 | 633.3 | 0.01 | ceiling | ttft 711 -> 638; sat 720 -> 639; wait 668 -> 635 |
| 431 | 27B/7xH200 tensor-par/fp8/fp8KV | 64 | 737.0 | 744.4 | 0.01 | ceiling | ttft 811 -> 748; sat 820 -> 749; wait 768 -> 746 |
| 432 | 27B/8xH200 tensor-par/fp8/fp8KV | 64 | 845.9 | 854.4 | 0.01 | ceiling | ttft 910 -> 858; sat 918 -> 858; wait 866 -> 855 |
| 433 | 35BA3B/1xH200/fp8/fp8KV | 64 | 226.9 | 229.2 | 0.01 | ceiling | ttft 880 -> 269; sat 891 -> 272; slowed 775 -> 235; wait 826 -> 256 |
| 434 | 35BA3B/2xH200 tensor-par/fp8/fp8KV | 64 | 576.0 | 581.8 | 0.01 | ceiling | ttft 1,593 -> 645; sat 1,604 -> 647; slowed 1,295 -> 589; wait 1,539 -> 632 |
| 435 | 35BA3B/2xH200 tensor-par/fp8/fp8KV | 64 | 576.0 | 581.8 | 0.01 | ceiling | ttft 1,593 -> 645; sat 1,604 -> 647; slowed 1,295 -> 589; wait 1,539 -> 632 |
| 436 | 35BA3B/2xH200 tensor-par/fp8/fp8KV | 1000 | 576.0 | 581.8 | 0.01 | ceiling+load (18 at-load keys) | ttft 1,593 -> 645; sat 1,604 -> 647; slowed 1,295 -> 589; wait 1,539 -> 632 |
| 437 | 35BA3B/2xH200 tensor-par/fp8/fp8KV | 200 | 576.0 | 581.8 | 0.01 | ceiling | ttft 1,593 -> 645; sat 1,604 -> 647; slowed 1,295 -> 589; wait 1,539 -> 632 |
| 438 | 35BA3B/2xH200 tensor-par/fp8/fp8KV | 64 | 576.0 | 581.8 | 0.01 | ceiling | ttft 1,593 -> 645; sat 1,604 -> 647; slowed 1,295 -> 589; wait 1,539 -> 632 |
| 439 | 35BA3B/2xH200 tensor-par/fp8/fp8KV | 8 | 576.0 | 581.8 | 0.01 | ceiling | ttft 1,593 -> 645; sat 1,604 -> 647; slowed 1,295 -> 589; wait 1,539 -> 632 |
| 440 | 35BA3B/2xH200 tensor-par/fp8/fp8KV | 64 | 576.0 | 581.8 | 0.01 | ceiling | ttft 1,550 -> 635; sat 1,604 -> 647; slowed 1,295 -> 589; wait 1,539 -> 632 |
| 441 | 35BA3B/2xH200 tensor-par/fp8/fp8KV | 64 | 576.0 | 581.8 | 0.01 | ceiling | ttft 1,600 -> 646; sat 1,604 -> 647; slowed 1,295 -> 589; wait 1,539 -> 632 |
| 442 | 35BA3B/2xH200 tensor-par/fp8/fp8KV | 64 | 576.0 | 581.8 | 0.01 | ceiling | ttft 1,602 -> 647; sat 1,604 -> 647; slowed 1,295 -> 589; wait 1,539 -> 632 |
| 444 | 35BA3B/2xH200 tensor-par/fp8/fp8KV | 64 | 968.0 | 977.8 | 0.01 | ceiling | ttft 1,593 -> 1,016; sat 1,604 -> 1,018; slowed 1,295 -> 981; wait 1,539 -> 1,008 |
| 445 | 35BA3B/2xH200 tensor-par/fp8/fp8KV | 64 | 576.0 | 581.8 | 0.01 | ceiling | ttft 1,593 -> 645; sat 1,604 -> 647; slowed 1,295 -> 589; wait 1,539 -> 632 |
| 446 | 35BA3B/3xH200 tensor-par/fp8/fp8KV | 64 | 918.9 | 928.2 | 0.01 | ceiling | ttft 2,251 -> 1,011; sat 2,262 -> 1,014; slowed 1,713 -> 936; wait 2,196 -> 999 |
| 447 | 35BA3B/4xH200 tensor-par/fp8/fp8KV | 64 | 1,275.9 | 1,288.8 | 0.01 | ceiling | ttft 2,876 -> 1,389; sat 2,887 -> 1,391; slowed 2,065 -> 1,297; wait 2,821 -> 1,377 |
| 448 | 35BA3B/5xH200 tensor-par/fp8/fp8KV | 64 | 1,634.8 | 1,651.4 | 0.01 | ceiling | ttft 3,477 -> 1,767; sat 3,488 -> 1,769; slowed 2,369 -> 1,659; wait 3,422 -> 1,755 |
| 449 | 35BA3B/6xH200 tensor-par/fp8/fp8KV | 64 | 1,984.0 | 2,004.0 | 0.01 | ceiling | ttft 4,060 -> 2,134; sat 4,071 -> 2,137; slowed 2,635 -> 2,010; wait 4,006 -> 2,123 |
| 450 | 35BA3B/7xH200 tensor-par/fp8/fp8KV | 64 | 2,341.0 | 2,364.6 | 0.01 | ceiling | ttft 4,629 -> 2,508; sat 4,640 -> 2,511; slowed 2,872 -> 2,370; wait 4,574 -> 2,497 |
| 451 | 35BA3B/8xH200 tensor-par/fp8/fp8KV | 64 | 2,692.9 | 2,720.1 | 0.01 | ceiling | ttft 5,185 -> 2,877; sat 5,196 -> 2,879; slowed 3,085 -> 2,724; wait 5,130 -> 2,866 |
| 456 | DSV4F/2xH200 tensor-par/fp8/fp8KV | 64 | 598.9 | 604.9 | 0.01 | ceiling | ttft 609 -> 605; sat 618 -> 606 |
| 463 | DSV4FV/2xH200 tensor-par/fp8/fp8KV | 64 | 597.9 | 603.9 | 0.01 | ceiling | ttft 609 -> 604; sat 618 -> 605 |
| 470 | GLM52/7xH200 tensor-par/fp8/fp8KV | 64 | 52.0 | 52.5 | 0.01 | ceiling+load (18 at-load keys) | ttft 178 -> 59; sat 190 -> 62; slowed 183 -> 54; wait 126 -> 54 |
| 471 | GLM52/8xH200 tensor-par/fp8/fp8KV | 1000 | 129.0 | 130.3 | 0.01 | ceiling+load (2 at-load keys) | ttft 201 -> 134; sat 213 -> 136; slowed 205 -> 131; wait 147 -> 131 |
| 472 | GLM52/8xH200 tensor-par/fp8/fp8KV | 200 | 129.0 | 130.3 | 0.01 | ceiling+load (18 at-load keys) | ttft 201 -> 134; sat 213 -> 136; slowed 205 -> 131; wait 147 -> 131 |
| 473 | GLM52/8xH200 tensor-par/fp8/fp8KV | 64 | 129.0 | 130.3 | 0.01 | ceiling | ttft 201 -> 134; sat 213 -> 136; slowed 205 -> 131; wait 147 -> 131 |
| 474 | GLM52/8xH200 tensor-par/fp8/fp8KV | 8 | 129.0 | 130.3 | 0.01 | ceiling | ttft 201 -> 134; sat 213 -> 136; slowed 205 -> 131; wait 147 -> 131 |
| 475 | GLM52/8xH200 tensor-par/fp8/fp8KV | 64 | 129.0 | 130.3 | 0.01 | ceiling | ttft 159 -> 131; sat 213 -> 136; slowed 205 -> 131; wait 147 -> 131 |
| 476 | GLM52/8xH200 tensor-par/fp8/fp8KV | 64 | 129.0 | 130.3 | 0.01 | ceiling | ttft 209 -> 135; sat 213 -> 136; slowed 205 -> 131; wait 147 -> 131 |
| 477 | GLM52/8xH200 tensor-par/fp8/fp8KV | 64 | 129.0 | 130.3 | 0.01 | ceiling | ttft 211 -> 135; sat 213 -> 136; slowed 205 -> 131; wait 147 -> 131 |
| 478 | MM35/2xH200 tensor-par/fp8/fp8KV | 64 | 12.0 | 12.1 | 0.01 | ceiling+load (2 at-load keys) | ttft 35 -> 13; sat 44 -> 14; slowed 41 -> 12 |
| 479 | MM35/3xH200 tensor-par/fp8/fp8KV | 64 | 30.0 | 30.3 | 0.01 | ceiling+load (2 at-load keys) | ttft 53 -> 31; sat 62 -> 32; slowed 58 -> 31 |
| 480 | MM35/4xH200 tensor-par/fp8/fp8KV | 64 | 50.0 | 50.5 | 0.01 | ceiling+load (18 at-load keys) | ttft 70 -> 51; sat 79 -> 52; slowed 74 -> 51 |
| 481 | MM35/5xH200 tensor-par/fp8/fp8KV | 64 | 73.0 | 73.7 | 0.01 | ceiling | ttft 86 -> 74; sat 96 -> 75; slowed 90 -> 74 |
| 482 | MM35/6xH200 tensor-par/fp8/fp8KV | 64 | 89.0 | 89.9 | 0.01 | ceiling | ttft 102 -> 90; sat 112 -> 91; slowed 104 -> 90 |
| 483 | MM35/7xH200 tensor-par/fp8/fp8KV | 64 | 109.0 | 110.1 | 0.01 | ceiling | ttft 118 -> 110; sat 127 -> 111; slowed 119 -> 110 |
| 484 | MM35/8xH200 DP4xTP2/fp8/fp8KV | 16 | 12.0 | 12.1 | 0.01 | ceiling+load (18 at-load keys) | ttft 35 -> 13; sat 44 -> 14; slowed 41 -> 12 |
| 485 | MM35/8xH200 DP4xTP2/fp8/fp8KV | 16 | 12.0 | 12.1 | 0.01 | ceiling+load (18 at-load keys) | ttft 35 -> 13; sat 44 -> 14; slowed 41 -> 12 |
| 486 | MM35/8xH200 DP4xTP2/fp8/fp8KV | 250 | 12.0 | 12.1 | 0.01 | ceiling+load (2 at-load keys) | ttft 35 -> 13; sat 44 -> 14; slowed 41 -> 12 |
| 487 | MM35/8xH200 DP4xTP2/fp8/fp8KV | 50 | 12.0 | 12.1 | 0.01 | ceiling+load (2 at-load keys) | ttft 35 -> 13; sat 44 -> 14; slowed 41 -> 12 |
| 488 | MM35/8xH200 DP4xTP2/fp8/fp8KV | 2 | 12.0 | 12.1 | 0.01 | ceiling | ttft 35 -> 13; sat 44 -> 14; slowed 41 -> 12 |
| 489 | MM35/8xH200 tensor-par/fp8/fp8KV | 64 | 136.9 | 138.3 | 0.01 | ceiling | sat 143 -> 139 |
| 490 | MM35/8xH200 DP4xTP2/fp8/fp8KV | 16 | 12.0 | 12.1 | 0.01 | ceiling+load (18 at-load keys) | ttft 14 -> 12; sat 44 -> 14; slowed 41 -> 12 |
| 491 | MM35/8xH200 DP4xTP2/fp8/fp8KV | 16 | 12.0 | 12.1 | 0.01 | ceiling+load (18 at-load keys) | ttft 41 -> 13; sat 44 -> 14; slowed 41 -> 12 |
| 492 | MM35/8xH200 DP4xTP2/fp8/fp8KV | 16 | 12.0 | 12.1 | 0.01 | ceiling+load (18 at-load keys) | ttft 42 -> 14; sat 44 -> 14; slowed 41 -> 12 |
| 493 | MM35/8xH200 DP4xTP2/fp8/fp8KV | 16 | 12.0 | 12.1 | 0.01 | ceiling+load (18 at-load keys) | ttft 35 -> 13; sat 44 -> 14; slowed 41 -> 12 |
| 494 | Q38FN/2xH200 tensor-par/fp8/fp8KV | 64 | 111.0 | 112.1 | 0.01 | ceiling | ttft 758 -> 150; sat 768 -> 154; slowed 667 -> 118; wait 708 -> 138 |
| 495 | Q38FN/3xH200 tensor-par/fp8/fp8KV | 64 | 377.9 | 381.8 | 0.01 | ceiling | ttft 1,072 -> 424; sat 1,083 -> 426; slowed 910 -> 387; wait 1,023 -> 413 |
| 496 | Q38FN/4xH200 tensor-par/fp8/fp8KV | 64 | 648.8 | 655.4 | 0.01 | ceiling | ttft 1,372 -> 699; sat 1,382 -> 701; slowed 1,126 -> 660; wait 1,321 -> 690 |
| 497 | Q38FN/5xH200 tensor-par/fp8/fp8KV | 64 | 916.0 | 925.3 | 0.01 | ceiling | ttft 1,659 -> 970; sat 1,670 -> 972; slowed 1,322 -> 929; wait 1,609 -> 962 |
| 498 | Q38FN/6xH200 tensor-par/fp8/fp8KV | 64 | 1,194.0 | 1,206.0 | 0.01 | ceiling | ttft 1,939 -> 1,251; sat 1,949 -> 1,253; slowed 1,501 -> 1,209; wait 1,888 -> 1,243 |
| 499 | Q38FN/7xH200 tensor-par/fp8/fp8KV | 64 | 1,457.0 | 1,471.7 | 0.01 | ceiling | ttft 2,211 -> 1,517; sat 2,221 -> 1,519; slowed 1,667 -> 1,474; wait 2,160 -> 1,510 |
| 500 | Q38FN/8xH200 tensor-par/fp8/fp8KV | 64 | 1,727.8 | 1,745.3 | 0.01 | ceiling | ttft 2,477 -> 1,791; sat 2,487 -> 1,792; slowed 1,821 -> 1,746; wait 2,426 -> 1,784 |
| 501 | 27B/1xH200/fp8/fp8KV | 64 | 70.0 | 70.7 | 0.01 | ceiling | ttft 184 -> 76; sat 192 -> 78; slowed 170 -> 72; wait 144 -> 72 |
| 502 | 27B/1xH200/fp8/fp8KV | 64 | 70.0 | 70.7 | 0.01 | ceiling | ttft 149 -> 74; sat 157 -> 76; slowed 140 -> 71; wait 111 -> 72 |
| 503 | 35BA3B/2xH200 tensor-par/fp8/fp8KV | 64 | 576.0 | 581.8 | 0.01 | ceiling | ttft 1,593 -> 645; sat 1,604 -> 647; slowed 1,295 -> 589; wait 1,539 -> 632 |
| 504 | GLM52/8xH200 tensor-par/fp8/fp8KV | 64 | 129.0 | 130.3 | 0.01 | ceiling | ttft 201 -> 134; sat 213 -> 136; slowed 205 -> 131; wait 147 -> 131 |
| 505 | MM35/8xH200 DP4xTP2/fp8/fp8KV | 16 | 12.0 | 12.1 | 0.01 | ceiling+load (18 at-load keys) | ttft 35 -> 13; sat 44 -> 14; slowed 41 -> 12 |
| 506 | 27B/1xH200/fp8/fp8KV | 64 | 70.0 | 70.7 | 0.01 | ceiling | ttft 149 -> 74; sat 157 -> 76; slowed 140 -> 71; wait 111 -> 72 |
| 507 | 35BA3B/2xH200 tensor-par/fp8/fp8KV | 64 | 576.0 | 581.8 | 0.01 | ceiling | ttft 1,593 -> 645; sat 1,604 -> 647; slowed 1,295 -> 589; wait 1,539 -> 632 |
| 508 | GLM52/8xH200 tensor-par/fp8/fp8KV | 64 | 129.0 | 130.3 | 0.01 | ceiling | ttft 201 -> 134; sat 213 -> 136; slowed 205 -> 131; wait 147 -> 131 |
| 509 | MM35/8xH200 DP4xTP2/fp8/fp8KV | 16 | 12.0 | 12.1 | 0.01 | ceiling+load (18 at-load keys) | ttft 35 -> 13; sat 44 -> 14; slowed 41 -> 12 |
| 510 | 27B/1xH200/fp8/fp8KV | 64 | 70.0 | 70.7 | 0.01 | ceiling | ttft 149 -> 74; sat 157 -> 76; slowed 140 -> 71; wait 111 -> 72 |
| 511 | GLM52/8xH200 tensor-par [KV replicated]/fp8/fp8KV | 64 | 11.9 | 12.1 | 0.01 | ceiling+load (18 at-load keys) | ttft 201 -> 20; sat 213 -> 25; slowed 205 -> 14; wait 147 -> 14 |
| 512 | 27B/1xH200/fp8/fp8KV | 64 | 46.0 | 61.3 | 0.25 | load (1 at-load keys) |  |
| 515 | MM35/8xH200 DP4xTP2/fp8/fp8KV | 16 | 7.0 | 9.3 | 0.25 | ceiling+load (1 at-load keys) | sat 10 -> 9 |
| 516 | 27B/1xH200/fp8/fp8KV | 64 | 67.0 | 70.5 | 0.05 | ceiling | ttft 84 -> 72; sat 97 -> 73 |
| 517 | 35BA3B/2xH200 tensor-par/fp8/fp8KV | 64 | 539.0 | 567.4 | 0.05 | ceiling | ttft 989 -> 610; sat 1,009 -> 612; wait 894 -> 597 |
| 518 | GLM52/8xH200 tensor-par/fp8/fp8KV | 64 | 121.0 | 127.4 | 0.05 | ceiling | sat 135 -> 128 |
| 519 | MM35/8xH200 DP4xTP2/fp8/fp8KV | 16 | 11.9 | 12.6 | 0.05 | ceiling+load (18 at-load keys) | ttft 16 -> 13; sat 27 -> 14 |
| 520 | 27B/1xH200/fp8/fp8KV | 64 | 18.9 | 47.4 | 0.6 | load (1 at-load keys) |  |
| 523 | MM35/8xH200 DP4xTP2/fp8/fp8KV | 16 | 2.0 | 5.0 | 0.6 | load (1 at-load keys) |  |
| 524 | 27B/1xH200/fp8/fp8KV | 64 | 70.0 | 70.7 | 0.01 | ceiling | ttft 149 -> 74; sat 157 -> 76; slowed 140 -> 71; wait 111 -> 72 |
| 526 | 27B/1xH200/fp8/fp8KV | 64 | 70.0 | 70.7 | 0.01 | ceiling | ttft 149 -> 74; sat 157 -> 76; slowed 116 -> 71; wait 111 -> 72 |
| 527 | 35BA3B/2xH200 tensor-par/fp8/fp8KV | 64 | 576.0 | 581.8 | 0.01 | ceiling | ttft 1,593 -> 645; sat 1,604 -> 647; slowed 609 -> 582; wait 1,539 -> 632 |
| 528 | GLM52/8xH200 tensor-par/fp8/fp8KV | 64 | 129.0 | 130.3 | 0.01 | ceiling | ttft 201 -> 134; sat 213 -> 136; slowed 174 -> 131; wait 147 -> 131 |
| 529 | MM35/8xH200 DP4xTP2/fp8/fp8KV | 16 | 12.0 | 12.1 | 0.01 | ceiling+load (18 at-load keys) | ttft 35 -> 13; sat 44 -> 14; slowed 40 -> 12 |
| 531 | 27B/3xB300 tensor-par/nvfp4/fp16KV | 16 | 543.0 | 835.3 | 0.35 | ceiling | ttft 866 -> 849; sat 877 -> 854; wait 845 -> 840 |
| 534 | 27B/1xH200/fp8/fp8KV | 64 | 70.0 | 70.7 | 0.01 | ceiling | ttft 149 -> 74; sat 157 -> 76; slowed 142 -> 71; wait 111 -> 72 |
| 535 | 35BA3B/2xH200 tensor-par/fp8/fp8KV | 64 | 576.0 | 581.8 | 0.01 | ceiling | ttft 1,593 -> 645; sat 1,604 -> 647; slowed 1,408 -> 590; wait 1,539 -> 632 |
| 536 | GLM52/8xH200 tensor-par/fp8/fp8KV | 64 | 129.0 | 130.3 | 0.01 | ceiling | ttft 201 -> 134; sat 213 -> 136; slowed 207 -> 131; wait 147 -> 131 |
| 537 | MM35/8xH200 DP4xTP2/fp8/fp8KV | 16 | 12.0 | 12.1 | 0.01 | ceiling+load (18 at-load keys) | ttft 35 -> 13; sat 44 -> 14; slowed 41 -> 12 |
| 538 | 27B/1xH200/fp8/fp8KV | 64 | 70.0 | 70.7 | 0.01 | ceiling | ttft 149 -> 74; sat 157 -> 76; slowed 140 -> 71; wait 111 -> 72 |
| 539 | MM35/6xH200 DP2xTP3/fp8/fp8KV | 32 | 70.0 | 70.4 | 0.005 | ceiling | slowed inf -> 72 |
| 541 | 27B/4xH200 tensor-par/fp8/fp8KV | 400 | 372.9 | 414.4 | 0.1 | ceiling | ttft 556 -> 432; sat 658 -> 448 |
| 542 | Q38FN/7xB300 tensor-par/nvfp4/fp8KV | 128 | 2,226.8 | 2,238.0 | 0.005 | ceiling | ttft 13,019 -> 2,538; sat 13,072 -> 2,545; wait 12,753 -> 2,506 |
| 545 | MM35/3xH200 tensor-par/fp8/fp16KV | 32 | 12.0 | 12.1 | 0.005 | ceiling+load (18 at-load keys) | ttft 86 -> 13; sat 157 -> 15; slowed 87 -> 12; wait 24 -> 12 |
| 546 | 27B/1xH200/fp8/fp8KV | 64 | 70.0 | 70.7 | 0.01 | ceiling | ttft 148 -> 74; sat 157 -> 76; slowed 111 -> 71; wait 109 -> 72 |
| 547 | 35BA3B/2xH200 tensor-par/fp8/fp8KV | 64 | 576.0 | 581.8 | 0.01 | ceiling | ttft 1,559 -> 636; sat 1,571 -> 639; wait 1,498 -> 625 |
| 548 | GLM52/8xH200 tensor-par/fp8/fp8KV | 64 | 129.0 | 130.3 | 0.01 | ceiling | ttft 196 -> 133; sat 209 -> 135; slowed 146 -> 130; wait 138 -> 130 |
| 549 | MM35/8xH200 DP4xTP2/fp8/fp8KV | 16 | 12.0 | 12.1 | 0.01 | ceiling+load (18 at-load keys) | ttft 35 -> 13; sat 44 -> 14; slowed 40 -> 12 |
| 550 | Q38FN/6xB300 DP3xTP2/fp8/fp8KV | 21.3333 | 639.0 | 645.4 | 0.01 | ceiling | ttft 8,377 -> 863; sat 8,476 -> 871; slowed 1,337 -> 652; wait 7,894 -> 828 |
| 551 | 35BA3B/8xB300 DP2xTP4/fp8/fp8KV | 32 | 2,968.0 | 2,982.9 | 0.005 | ceiling | ttft 9,837 -> 3,522; sat 9,843 -> 3,526; slowed 8,966 -> 3,013; wait 9,824 -> 3,515 |
| 553 | MM35/1xB300/fp8/fp16KV | 400 | 27.9 | 43.0 | 0.35 | load (1 at-load keys) |  |
| 554 | Q38FN/8xH200 DP4xTP2/fp8/fp8KV | 16 | 76.0 | 78.4 | 0.03 | ceiling | ttft 205 -> 85; sat 211 -> 85; wait 146 -> 81 |
| 555 | MM35/6xB300 tensor-par/nvfp4/fp16KV | 400 | 364.9 | 366.8 | 0.005 | ceiling+load (18 at-load keys) | ttft 530 -> 385; sat 535 -> 387; slowed inf -> 375; wait 519 -> 381 |
| 556 | 35BA3B/7xB300 tensor-par/fp8/fp8KV | 32 | 3,021.8 | 3,052.3 | 0.01 | ceiling | ttft 11,922 -> 3,498; sat 11,954 -> 3,504; slowed 6,354 -> 3,085; wait 11,857 -> 3,487 |
| 557 | DSV4F/8xB300 tensor-par/fp8/fp8KV | 128 | 21,761.8 | 21,981.7 | 0.01 | ceiling | slowed inf -> 22,609 |
| 558 | 35BA3B/6xH200 data-par/fp8/fp16KV | 21.3333 | 81.0 | 81.8 | 0.01 | ceiling | ttft 5,036 -> 275; sat 5,072 -> 281; slowed 2,883 -> 110; wait 4,650 -> 223 |
| 559 | 27B/2xH200 data-par/fp8/fp8KV | 16 | 48.0 | 53.3 | 0.1 | ceiling | ttft 105 -> 59; sat 129 -> 64 |
| 562 | 27B/1xH200/fp8/fp8KV | 64 | 70.0 | 70.7 | 0.01 | ceiling | ttft 149 -> 74; sat 158 -> 76; slowed 143 -> 71; wait 111 -> 72 |
| 563 | 35BA3B/2xH200 tensor-par/fp8/fp8KV | 64 | 576.0 | 581.8 | 0.01 | ceiling | ttft 1,595 -> 645; sat 1,606 -> 648; slowed 1,445 -> 590; wait 1,541 -> 633 |
| 564 | GLM52/8xH200 tensor-par/fp8/fp8KV | 64 | 129.0 | 130.3 | 0.01 | ceiling | ttft 201 -> 134; sat 213 -> 136; slowed 210 -> 131; wait 148 -> 131 |
| 565 | MM35/8xH200 DP4xTP2/fp8/fp8KV | 16 | 12.0 | 12.1 | 0.01 | ceiling+load (18 at-load keys) | ttft 35 -> 13; sat 44 -> 14; slowed 41 -> 12 |
| 566 | Q38FN/2xB300 data-par/nvfp4/fp8KV | 200 | 446.0 | 450.5 | 0.01 | ceiling | ttft 663 -> 461; sat 671 -> 462; slowed 527 -> 451; wait 583 -> 455 |
| 572 | GLM53F/7xB300 tensor-par/nvfp4/fp16KV | 16 | 6,478.9 | 6,511.4 | 0.005 | ceiling | slowed inf -> 6,605 |
| 573 | GLM52/4xB300 tensor-par/fp8/fp8KV | 16 | 87.0 | 87.4 | 0.005 | ceiling | slowed inf -> 88 |
| 575 | 27B/6xB300 tensor-par/fp8/fp16KV | 400 | 1,051.0 | 1,616.9 | 0.35 | ceiling | ttft 1,971 -> 1,764; sat 1,985 -> 1,770; wait 1,901 -> 1,734 |
| 576 | MM35/8xH200 DP4xTP2/fp8/fp16KV | 8 | 4.0 | 4.1 | 0.03 | ceiling+load (18 at-load keys) | ttft 29 -> 4; sat 57 -> 8; slowed 19 -> 5; wait 5 -> 4 |
| 577 | 27B/8xH200 DP4xTP2/fp8/fp8KV | 8 | 153.9 | 158.7 | 0.03 | ceiling | ttft 200 -> 161; sat 238 -> 165 |
| 578 | 27B/1xH200/fp8/fp8KV | 64 | 70.0 | 70.7 | 0.01 | ceiling | ttft 149 -> 74; sat 157 -> 76; slowed 126 -> 71; wait 110 -> 72 |
| 579 | 35BA3B/2xH200 tensor-par/fp8/fp8KV | 64 | 576.0 | 581.8 | 0.01 | ceiling | ttft 1,578 -> 641; sat 1,590 -> 644; slowed 753 -> 583; wait 1,521 -> 629 |
| 580 | GLM52/8xH200 tensor-par/fp8/fp8KV | 64 | 129.0 | 130.3 | 0.01 | ceiling | ttft 199 -> 134; sat 211 -> 135; slowed 176 -> 131; wait 144 -> 131 |
| 581 | MM35/8xH200 DP4xTP2/fp8/fp8KV | 16 | 12.0 | 12.1 | 0.01 | ceiling+load (18 at-load keys) | ttft 35 -> 13; sat 44 -> 14; slowed 41 -> 12 |
| 582 | Q38FN/8xB300 DP2xTP4/nvfp4/fp16KV | 8 | 1,999.8 | 2,009.8 | 0.005 | ceiling | ttft 4,972 -> 2,942; sat 4,982 -> 2,957; slowed inf -> 2,097; wait 4,953 -> 2,915 |
| 586 | Q38FN/6xH200 tensor-par/fp8/fp8KV | 16 | 615.9 | 634.9 | 0.03 | ceiling | ttft 2,098 -> 698; sat 2,310 -> 711; wait 1,686 -> 676 |
| 587 | Q38FN/8xB300 DP4xTP2/nvfp4/fp8KV | 100 | 1,149.0 | 1,154.7 | 0.005 | ceiling | ttft 2,951 -> 1,303; sat 2,956 -> 1,307; slowed 2,801 -> 1,163; wait 2,891 -> 1,267 |
| 588 | GLM53F/8xB300 DP2xTP4/nvfp4/fp8KV | 200 | 4,219.0 | 4,261.6 | 0.01 | ceiling | ttft 5,716 -> 4,324; sat 5,754 -> 4,327; wait 5,530 -> 4,311 |
| 589 | 35BA3B/7xH200 data-par/fp8/fp8KV | 18.2857 | 306.9 | 308.5 | 0.005 | ceiling | ttft 2,438 -> 370; sat 2,457 -> 373; slowed 1,139 -> 313; wait 2,234 -> 345 |
| 591 | Q38FN/2xH200 tensor-par/fp8/fp16KV | 64 | 115.0 | 115.6 | 0.005 | ceiling | ttft 917 -> 279; sat 925 -> 289; slowed inf -> 138; wait 902 -> 262 |
| 592 | 27B/4xH200 tensor-par/fp8/fp8KV | 48 | 166.9 | 171.2 | 0.025 | ceiling | ttft 263 -> 176; sat 297 -> 178 |
| 593 | GLM52/8xB300 DP4xTP2/nvfp4/fp8KV | 8 | 10.0 | 10.1 | 0.005 | ceiling | ttft 33 -> 10; sat 61 -> 11; slowed 39 -> 10 |
| 594 | DSV4F/4xB300 DP2xTP2/fp8/fp8KV | 32 | 14,832.5 | 14,907.0 | 0.005 | ceiling | slowed inf -> 14,941 |
| 595 | Q38FN/2xB300 data-par/nvfp4/fp16KV | 200 | 72.0 | 72.4 | 0.005 | ceiling+load (18 at-load keys) | ttft 752 -> 84; sat 807 -> 89; wait 645 -> 79 |
| 596 | GLM52/6xB300 DP2xTP3/nvfp4/fp8KV | 8 | 186.9 | 187.9 | 0.005 | ceiling | ttft 271 -> 193; sat 276 -> 194; wait 249 -> 190 |
| 597 | MM35/4xB300 DP2xTP2/fp8/fp8KV | 8 | 21.0 | 21.2 | 0.01 | ceiling | ttft 27 -> 22; sat 31 -> 22; slowed inf -> 21 |
| 599 | MM35/2xB300 data-par/nvfp4/fp16KV | 64 | 21.0 | 21.1 | 0.005 | ceiling+load (2 at-load keys) | slowed inf -> 21 |
| 601 | 35BA3B/4xB300 data-par/fp8/fp16KV | 4 | 203.8 | 210.2 | 0.03 | ceiling | ttft 682 -> 266; sat 697 -> 272; wait 610 -> 245 |
| 602 | 35BA3B/4xB300 tensor-par/fp8/fp8KV | 64 | 2,133.9 | 2,199.9 | 0.03 | ceiling | ttft 3,687 -> 2,269; sat 3,717 -> 2,271; wait 3,540 -> 2,261 |
| 603 | MM35/4xB300 DP2xTP2/nvfp4/fp16KV | 200 | 54.0 | 83.1 | 0.35 | ceiling+load (1 at-load keys) | ttft 116 -> 95; sat 172 -> 121 |
| 604 | GLM53F/6xB300 DP3xTP2/nvfp4/fp8KV | 133.333 | 944.0 | 948.7 | 0.005 | ceiling | ttft 1,403 -> 959; sat 1,428 -> 960; wait 1,146 -> 951 |
| 605 | Q38FN/4xB300 DP2xTP2/fp8/fp16KV | 8 | 167.0 | 168.7 | 0.01 | ceiling | ttft 3,526 -> 270; sat 3,595 -> 277; slowed 883 -> 176; wait 3,389 -> 259 |
| 607 | MM35/6xB300 data-par/fp8/fp8KV | 21.3333 | 8.0 | 8.2 | 0.03 | ceiling+load (18 at-load keys) | ttft 17 -> 8; sat 43 -> 13; slowed 30 -> 9 |
| 608 | 35BA3B/1xH200/fp8/fp16KV | 64 | 915.9 | 920.5 | 0.005 | ceiling | slowed inf -> 941 |
| 609 | DSV4F/8xH200 DP4xTP2/fp8/fp8KV | 32 | 455.7 | 458.0 | 0.005 | ceiling | sat 473 -> 458 |
| 610 | DSV4FV/4xH200 tensor-par/fp8/fp8KV | 64 | 4,378.9 | 4,400.9 | 0.005 | ceiling | slowed inf -> 4,442 |
| 612 | 27B/5xH200 data-par/fp8/fp8KV | 25.6 | 38.0 | 39.1 | 0.03 | ceiling | ttft 46 -> 40; sat 61 -> 43; slowed 44 -> 39 |
| 613 | 27B/4xB300 data-par/nvfp4/fp16KV | 8 | 36.0 | 36.1 | 0.005 | ceiling | ttft 85 -> 40; sat 86 -> 41; slowed inf -> 37; wait 69 -> 37 |
| 614 | 35BA3B/3xB300 data-par/nvfp4/fp8KV | 42.6667 | 308.0 | 309.5 | 0.005 | ceiling | ttft 2,950 -> 430; sat 2,955 -> 431; slowed inf -> 324; wait 2,887 -> 411 |
| 616 | Q38FN/5xB300 tensor-par/fp8/fp8KV | 400 | 988.0 | 1,097.7 | 0.1 | ceiling | ttft 2,355 -> 1,241; sat 2,504 -> 1,260; wait 2,068 -> 1,206 |
| 617 | Q38FN/8xB300 DP4xTP2/fp8/fp8KV | 4 | 1,029.0 | 1,143.3 | 0.1 | ceiling | ttft 1,501 -> 1,201; sat 1,508 -> 1,202; wait 1,426 -> 1,187 |
| 621 | Q38FN/5xB300 data-par/fp8/fp16KV | 12.8 | 163.9 | 169.0 | 0.03 | ceiling | ttft 456 -> 201; sat 460 -> 203; slowed 234 -> 171; wait 417 -> 191 |
| 622 | Q38FN/6xB300 DP2xTP3/fp8/fp8KV | 16 | 2,383.7 | 2,457.4 | 0.03 | ceiling | ttft 5,378 -> 2,857; sat 5,400 -> 2,865; slowed 2,781 -> 2,467; wait 5,334 -> 2,841 |
| 625 | 27B/1xH200/fp8/fp8KV | 64 | 69.0 | 69.7 | 0.01 | ceiling | ttft 146 -> 73; sat 156 -> 75; slowed 136 -> 70; wait 102 -> 70 |
| 626 | 35BA3B/4xB300 tensor-par/fp8/fp16KV | 400 | 2,267.9 | 2,290.8 | 0.01 | ceiling | ttft 13,657 -> 2,850; sat 13,664 -> 2,851; slowed 6,993 -> 2,338; wait 13,577 -> 2,835 |
| 627 | 27B/2xB300 tensor-par/fp8/fp16KV | 128 | 269.9 | 272.7 | 0.01 | ceiling | ttft 625 -> 309; sat 630 -> 311; slowed inf -> 279; wait 601 -> 298 |
| 631 | MM35/1xB300/nvfp4/fp16KV | 400 | 3.0 | 4.6 | 0.35 | ceiling+load (1 at-load keys) | sat 14 -> 8 |
| 633 | 35BA3B/6xH200 DP3xTP2/fp8/fp8KV | 42.6667 | 2,638.8 | 2,665.5 | 0.01 | ceiling | ttft 2,940 -> 2,691; sat 2,945 -> 2,692; slowed inf -> 2,675; wait 2,914 -> 2,687 |
| 634 | 35BA3B/2xB300 data-par/nvfp4/fp16KV | 8 | 485.6 | 488.1 | 0.005 | ceiling | ttft 1,406 -> 575; sat 1,429 -> 594; slowed 626 -> 489; wait 1,362 -> 547 |
| 635 | 27B/8xB300 tensor-par/fp8/fp16KV | 32 | 2,281.6 | 2,304.6 | 0.01 | ceiling | ttft 3,320 -> 2,381; sat 3,334 -> 2,384; wait 3,290 -> 2,375 |
| 636 | Q38FN/3xB300 tensor-par/fp8/fp16KV | 400 | 260.9 | 401.3 | 0.35 | ceiling | ttft 554 -> 455; sat 637 -> 486; wait 424 -> 409 |
| 638 | MM35/2xB300 data-par/nvfp4/fp8KV | 200 | 91.0 | 91.5 | 0.005 | ceiling+load (18 at-load keys) | ttft 420 -> 101; sat 449 -> 106; slowed inf -> 94; wait 193 -> 92 |
| 639 | GLM52/7xB300 tensor-par/fp8/fp8KV | 64 | 1,013.0 | 1,023.2 | 0.01 | ceiling | ttft 2,180 -> 1,053; sat 2,446 -> 1,068; wait 1,673 -> 1,034 |
| 640 | Q38FN/3xB300 data-par/fp8/fp8KV | 10.6667 | 126.0 | 126.6 | 0.005 | ceiling | ttft 199 -> 137; sat 201 -> 138; slowed inf -> 128; wait 196 -> 135 |
| 643 | 27B/3xH200 tensor-par/fp8/fp8KV | 64 | 412.0 | 416.2 | 0.01 | ceiling | slowed inf -> 419 |
| 644 | Q38FN/6xH200 DP2xTP3/fp8/fp8KV | 32 | 360.0 | 363.6 | 0.01 | ceiling | ttft 1,007 -> 387; sat 1,023 -> 388; wait 844 -> 375 |
| 645 | 35BA3B/3xH200 tensor-par/fp8/fp16KV | 32 | 270.0 | 415.4 | 0.35 | ceiling | ttft 500 -> 447; sat 503 -> 448; wait 468 -> 435 |
| 646 | 35BA3B/6xB300 tensor-par/nvfp4/fp16KV | 16 | 1,322.9 | 1,336.3 | 0.01 | ceiling | ttft 3,536 -> 1,390; sat 3,568 -> 1,392; wait 3,378 -> 1,383 |
| 649 | Q38FN/6xB300 tensor-par/nvfp4/fp16KV | 16 | 1,392.0 | 1,398.9 | 0.005 | ceiling | slowed inf -> 1,437 |
| 650 | 27B/8xB300 data-par/fp8/fp8KV | 2 | 1,039.0 | 1,044.2 | 0.005 | ceiling | slowed inf -> 1,077 |
| 654 | MM35/4xB300 data-par/fp8/fp8KV | 16 | 7.0 | 7.2 | 0.03 | ceiling+load (18 at-load keys) | ttft 15 -> 7; sat 28 -> 8; slowed 8 -> 7 |
| 656 | Q38FN/6xB300 data-par/nvfp4/fp16KV | 66.6667 | 729.9 | 737.2 | 0.01 | ceiling | slowed inf -> 748 |
| 657 | 27B/6xB300 data-par/nvfp4/fp8KV | 10.6667 | 96.0 | 96.9 | 0.01 | ceiling | ttft 711 -> 111; sat 757 -> 114; slowed 235 -> 98; wait 508 -> 103 |
| 660 | Q38FN/4xB300 tensor-par/fp8/fp8KV | 32 | 863.0 | 867.3 | 0.005 | ceiling | ttft 3,367 -> 972; sat 3,373 -> 973; slowed 2,057 -> 873; wait 3,309 -> 959 |
| 661 | Q38FN/6xB300 tensor-par/nvfp4/fp8KV | 32 | 1,896.0 | 1,915.1 | 0.01 | ceiling | ttft 10,220 -> 2,106; sat 10,409 -> 2,115; wait 9,296 -> 2,064 |
| 662 | MM35/4xB300 DP2xTP2/nvfp4/fp8KV | 32 | 27.9 | 31.1 | 0.1 | ceiling+load (18 at-load keys) | sat 39 -> 32 |
| 664 | 27B/1xH200/fp8/fp8KV | 16 | 145.0 | 145.7 | 0.005 | ceiling | ttft 377 -> 148; sat 511 -> 152; slowed 239 -> 146 |
| 665 | GLM53F/6xH200 DP2xTP3/fp8/fp16KV | 200 | 63.0 | 63.6 | 0.01 | ceiling+load (2 at-load keys) | ttft 102 -> 70; sat 103 -> 71; slowed inf -> 66; wait 98 -> 69 |
| 667 | 27B/1xH200/fp8/fp8KV | 64 | 175.0 | 176.8 | 0.01 | ceiling | ttft 184 -> 177; sat 186 -> 178; slowed inf -> 178 |
| 668 | 35BA3B/2xB300 tensor-par/nvfp4/fp16KV | 128 | 600.0 | 603.0 | 0.005 | ceiling | ttft 1,665 -> 815; sat 1,667 -> 817; slowed inf -> 610; wait 1,655 -> 807 |
| 670 | MM35/6xB300 data-par/nvfp4/fp16KV | 2.66667 | 61.0 | 67.7 | 0.1 | ceiling | ttft 179 -> 91; sat 192 -> 95; wait 132 -> 78 |
| 671 | Q38FN/5xB300 data-par/fp8/fp8KV | 12.8 | 190.0 | 211.1 | 0.1 | ceiling | ttft 683 -> 272; sat 689 -> 273; wait 623 -> 263 |
| 673 | GLM52/5xB300 tensor-par/nvfp4/fp8KV | 32 | 768.8 | 772.7 | 0.005 | ceiling | ttft 866 -> 776; sat 889 -> 777; slowed 888 -> 773; wait 821 -> 774 |
| 675 | MM35/6xB300 DP2xTP3/fp8/fp16KV | 200 | 418.9 | 423.1 | 0.01 | ceiling | ttft 669 -> 432; sat 722 -> 437; slowed 483 -> 424; wait 568 -> 427 |
| 676 | 27B/4xB300 data-par/fp8/fp8KV | 4 | 808.0 | 816.2 | 0.01 | ceiling | slowed inf -> 846 |
| 678 | 27B/8xH200 DP2xTP4/fp8/fp16KV | 64 | 240.0 | 242.4 | 0.01 | ceiling | ttft 524 -> 252; sat 546 -> 254; slowed 260 -> 243; wait 482 -> 249 |
| 683 | 35BA3B/4xH200 data-par/fp8/fp16KV | 8 | 146.9 | 163.3 | 0.1 | ceiling | ttft 285 -> 201; sat 288 -> 202; wait 272 -> 195 |
| 685 | MM35/4xB300 tensor-par/nvfp4/fp16KV | 400 | 80.0 | 82.5 | 0.03 | load (1 at-load keys) |  |
| 687 | 35BA3B/6xB300 DP2xTP3/fp8/fp8KV | 8 | 5,146.9 | 5,172.8 | 0.005 | ceiling | ttft 6,119 -> 5,444; sat 6,124 -> 5,447; wait 6,109 -> 5,440 |
| 688 | 27B/7xB300 data-par/fp8/fp8KV | 2.28571 | 270.9 | 273.7 | 0.01 | ceiling | ttft 2,579 -> 357; sat 2,622 -> 363; slowed 582 -> 277; wait 2,370 -> 335 |
| 689 | 35BA3B/7xB300 data-par/nvfp4/fp16KV | 9.14286 | 363.9 | 559.8 | 0.35 | ceiling | ttft 732 -> 640; sat 741 -> 644; wait 691 -> 620 |
| 690 | GLM52/8xH200 tensor-par/fp8/fp8KV | 32 | 594.0 | 599.9 | 0.01 | ceiling | ttft 703 -> 608; sat 723 -> 610; wait 666 -> 604 |
| 693 | Q38FN/5xB300 tensor-par/fp8/fp16KV | 32 | 4,830.8 | 4,855.1 | 0.005 | ceiling | ttft 6,614 -> 4,922; sat 6,631 -> 4,923; wait 6,581 -> 4,919 |
| 695 | MM35/7xB300 data-par/nvfp4/fp16KV | 4.57143 | 13.0 | 13.4 | 0.03 | ceiling | slowed inf -> 13 |
| 697 | DSV4F/2xB300 data-par/fp8/fp8KV | 32 | 1,594.0 | 1,602.0 | 0.005 | ceiling | slowed inf -> 1,637 |
| 698 | 35BA3B/6xB300 DP2xTP3/fp8/fp16KV | 32 | 575.0 | 884.5 | 0.35 | ceiling | ttft 1,821 -> 1,247; sat 1,827 -> 1,249; wait 1,759 -> 1,221 |
| 699 | DSV41F/2xB300 tensor-par/fp8/fp8KV | 32 | 597.0 | 599.9 | 0.005 | ceiling | ttft 3,642 -> 925; sat 3,643 -> 926; slowed inf -> 640; wait 3,631 -> 918 |
| 700 | MM35/4xB300 data-par/nvfp4/fp8KV | 16 | 15.0 | 15.5 | 0.03 | ceiling+load (18 at-load keys) | ttft 147 -> 20; sat 169 -> 24; slowed 46 -> 16; wait 34 -> 16 |
| 701 | MM35/3xH200 tensor-par/fp8/fp16KV | 64 | 30.0 | 30.3 | 0.01 | ceiling+load (18 at-load keys) | ttft 137 -> 33; sat 146 -> 34; slowed 77 -> 31; wait 61 -> 31 |
| 702 | 27B/1xH200/fp8/fp8KV | 64 | 96.0 | 97.0 | 0.01 | ceiling | ttft 165 -> 101; sat 169 -> 102; slowed inf -> 98; wait 144 -> 99 |
| 703 | 35BA3B/2xB300 data-par/nvfp4/fp8KV | 64 | 1,247.9 | 1,260.5 | 0.01 | ceiling | ttft 1,555 -> 1,271; sat 1,567 -> 1,272; wait 1,531 -> 1,270 |
| 705 | GLM52/6xB300 DP3xTP2/nvfp4/fp8KV | 133.333 | 21.0 | 23.3 | 0.1 | load (1 at-load keys) |  |
| 706 | GLM53F/8xB300 tensor-par/nvfp4/fp8KV | 64 | 8,325.9 | 8,583.4 | 0.03 | ceiling | slowed inf -> 8,587 |
| 710 | DSV41F/8xH200 tensor-par/fp8/fp8KV | 32 | 70,293.9 | 72,468.0 | 0.03 | ceiling | slowed inf -> 72,910 |
| 715 | 27B/6xB300 tensor-par/fp8/fp8KV | 16 | 1,167.9 | 1,297.7 | 0.1 | ceiling | ttft 1,817 -> 1,433; sat 1,822 -> 1,435; wait 1,754 -> 1,412 |
| 717 | DSV4FV/7xB300 data-par/fp8/fp8KV | 57.1429 | 922.0 | 950.5 | 0.03 | ceiling | ttft 1,591 -> 981; sat 1,622 -> 983; wait 1,273 -> 964 |
| 719 | 35BA3B/4xH200 tensor-par/fp8/fp8KV | 16 | 2,087.9 | 2,098.4 | 0.005 | ceiling | ttft 26,938 -> 3,367; sat 26,981 -> 3,386; slowed 2,220 -> 2,099; wait 26,851 -> 3,333 |
| 720 | 27B/7xH200 data-par/fp8/fp8KV | 4.57143 | 87.0 | 87.4 | 0.005 | ceiling | slowed inf -> 88 |
| 722 | Q38FN/1xB300/fp8/fp8KV | 64 | 407.9 | 412.0 | 0.01 | ceiling | ttft 8,530 -> 707; sat 8,582 -> 714; slowed 1,833 -> 426; wait 8,270 -> 676 |
| 726 | 35BA3B/3xB300 data-par/fp8/fp8KV | 5.33333 | 974.0 | 983.8 | 0.01 | ceiling | slowed inf -> 990 |
| 727 | GLM53F/5xB300 data-par/nvfp4/fp16KV | 80 | 183.9 | 184.8 | 0.005 | ceiling | slowed inf -> 195 |
| 728 | GLM53F/3xB300 tensor-par/nvfp4/fp8KV | 400 | 6,121.0 | 6,182.8 | 0.01 | ceiling | slowed inf -> 6,253 |
| 730 | GLM53F/6xB300 DP2xTP3/fp8/fp8KV | 8 | 1,009.0 | 1,014.1 | 0.005 | ceiling | slowed inf -> 1,014 |
| 731 | MM35/4xB300 DP2xTP2/fp8/fp16KV | 64 | 42.0 | 42.4 | 0.01 | ceiling+load (18 at-load keys) | ttft 116 -> 46; sat 121 -> 47; slowed 110 -> 43; wait 90 -> 44 |
| 732 | 35BA3B/5xB300 tensor-par/nvfp4/fp16KV | 32 | 5,342.9 | 5,936.6 | 0.1 | ceiling | ttft 18,055 -> 8,030; sat 18,066 -> 8,033; wait 17,999 -> 8,016 |
| 733 | 35BA3B/8xB300 DP2xTP4/fp8/fp16KV | 64 | 886.0 | 913.4 | 0.03 | ceiling | ttft 4,155 -> 1,193; sat 4,169 -> 1,196; wait 4,128 -> 1,188 |
| 735 | 27B/7xH200 tensor-par/fp8/fp8KV | 128 | 443.0 | 492.2 | 0.1 | ceiling | ttft 698 -> 517; sat 707 -> 518; wait 606 -> 505 |
| 737 | MM35/7xB300 data-par/fp8/fp8KV | 4.57143 | 15.0 | 16.7 | 0.1 | ceiling | ttft 38 -> 21; sat 43 -> 24; slowed 17 -> 17 |
| 739 | MM35/6xB300 DP3xTP2/nvfp4/fp16KV | 42.6667 | 56.0 | 56.2 | 0.005 | ceiling | ttft 65 -> 57; sat 68 -> 57; slowed inf -> 56 |
| 741 | DSV41F/4xB300 DP2xTP2/fp8/fp8KV | 64 | 783.8 | 808.0 | 0.03 | ceiling | ttft 2,174 -> 1,114; sat 2,177 -> 1,116; slowed 1,367 -> 825; wait 2,148 -> 1,099 |
| 742 | 27B/6xB300 DP2xTP3/nvfp4/fp8KV | 200 | 539.0 | 555.6 | 0.03 | ceiling | ttft 2,440 -> 702; sat 2,467 -> 706; slowed 595 -> 557; wait 2,308 -> 681 |
| 743 | 35BA3B/7xH200 tensor-par/fp8/fp8KV | 32 | 1,845.0 | 1,854.2 | 0.005 | ceiling | ttft 5,174 -> 2,047; sat 5,179 -> 2,049; slowed 3,951 -> 1,865; wait 5,147 -> 2,039 |
| 744 | 27B/1xH200/fp8/fp8KV | 64 | 70.0 | 70.7 | 0.01 | ceiling | ttft 149 -> 74; sat 157 -> 76; slowed 140 -> 71; wait 111 -> 72 |
| 745 | 27B/4xB300 tensor-par/nvfp4/fp16KV | 128 | 864.0 | 959.9 | 0.1 | ceiling | ttft 1,134 -> 1,033; sat 1,151 -> 1,042; wait 1,101 -> 1,016 |
| 746 | 27B/5xH200 tensor-par/fp8/fp16KV | 32 | 225.0 | 232.0 | 0.03 | ceiling | ttft 908 -> 287; sat 940 -> 294; wait 760 -> 264 |
| 752 | Q38FN/7xB300 data-par/fp8/fp16KV | 2.28571 | 52.0 | 52.3 | 0.005 | ceiling | ttft 1,003 -> 80; sat 1,019 -> 83; slowed 809 -> 56; wait 972 -> 76 |
| 753 | 35BA3B/2xB300 tensor-par/nvfp4/fp8KV | 32 | 1,286.9 | 1,299.9 | 0.01 | ceiling | ttft 2,967 -> 1,504; sat 2,971 -> 1,506; slowed inf -> 1,319; wait 2,917 -> 1,478 |
| 754 | GLM53F/2xB300 data-par/nvfp4/fp8KV | 16 | 537.0 | 542.4 | 0.01 | ceiling | slowed inf -> 551 |
| 756 | GLM53F/6xB300 data-par/nvfp4/fp16KV | 5.33333 | 96.0 | 99.0 | 0.03 | ceiling | ttft 478 -> 113; sat 601 -> 129; slowed 167 -> 101; wait 270 -> 105 |
| 757 | 35BA3B/8xB300 tensor-par/fp8/fp16KV | 400 | 2,318.8 | 2,390.6 | 0.03 | ceiling | ttft 13,254 -> 2,924; sat 13,278 -> 2,926; wait 13,002 -> 2,906 |
| 758 | 27B/2xB300 data-par/fp8/fp16KV | 8 | 48.0 | 73.8 | 0.35 | ceiling | ttft 102 -> 84; sat 113 -> 88 |
| 761 | GLM53F/7xH200 tensor-par/fp8/fp16KV | 64 | 1,506.0 | 1,513.5 | 0.005 | ceiling | ttft 4,529 -> 1,623; sat 4,580 -> 1,632; slowed inf -> 1,530; wait 4,428 -> 1,607 |
| 762 | Q38FN/4xH200 DP2xTP2/fp8/fp8KV | 64 | 511.0 | 513.6 | 0.005 | ceiling | ttft 1,827 -> 681; sat 1,830 -> 683; slowed inf -> 541; wait 1,814 -> 672 |
| 763 | 35BA3B/5xB300 data-par/fp8/fp8KV | 12.8 | 1,228.8 | 1,235.0 | 0.005 | ceiling | ttft 4,252 -> 1,358; sat 4,255 -> 1,358; wait 4,217 -> 1,350 |
| 764 | Q38FN/1xB300/fp8/fp16KV | 16 | 45.0 | 46.4 | 0.03 | ceiling | ttft 362 -> 69; sat 396 -> 81; wait 298 -> 60 |
| 765 | Q38FN/8xB300 data-par/fp8/fp16KV | 4 | 19.0 | 29.2 | 0.35 | ceiling | sat 32 -> 30 |
| 766 | MM35/6xB300 data-par/fp8/fp16KV | 66.6667 | 1.0 | 1.5 | 0.35 | ceiling+load (1 at-load keys) | sat 7 -> 4 |
| 768 | GLM53F/6xH200 tensor-par/fp8/fp16KV | 32 | 1,996.6 | 2,016.8 | 0.01 | ceiling | ttft 3,832 -> 2,069; sat 3,987 -> 2,078; wait 3,526 -> 2,054 |
| 769 | 35BA3B/3xB300 data-par/fp8/fp16KV | 21.3333 | 765.0 | 772.7 | 0.01 | ceiling | ttft 1,169 -> 845; sat 1,174 -> 849; wait 1,147 -> 832 |
| 770 | Q38FN/3xB300 data-par/nvfp4/fp16KV | 10.6667 | 73.0 | 81.1 | 0.1 | ceiling | ttft 289 -> 104; sat 352 -> 115; wait 186 -> 92 |
| 771 | Q38FN/2xB300 tensor-par/nvfp4/fp8KV | 128 | 424.0 | 437.1 | 0.03 | ceiling | ttft 2,649 -> 519; sat 2,917 -> 535; wait 2,129 -> 498 |
| 773 | MM35/3xH200 tensor-par/fp8/fp8KV | 32 | 49.0 | 49.2 | 0.005 | ceiling | slowed inf -> 51 |
| 775 | Q38FN/6xB300 tensor-par/nvfp4/fp8KV | 400 | 2,633.4 | 2,660.1 | 0.01 | ceiling | ttft 3,495 -> 2,697; sat 3,502 -> 2,698; wait 3,416 -> 2,692 |
| 777 | MM35/6xB300 DP2xTP3/nvfp4/fp16KV | 8 | 34.0 | 34.3 | 0.01 | ceiling | slowed inf -> 35 |
| 778 | MM35/2xB300 tensor-par/nvfp4/fp16KV | 16 | 47.0 | 47.2 | 0.005 | ceiling | ttft 592 -> 56; sat 679 -> 77; slowed inf -> 52; wait 436 -> 50 |
| 779 | Q38FN/5xB300 tensor-par/nvfp4/fp8KV | 64 | 1,157.0 | 1,162.8 | 0.005 | ceiling | ttft 9,793 -> 1,347; sat 9,817 -> 1,349; slowed 4,374 -> 1,179; wait 9,531 -> 1,327 |
| 780 | 27B/4xB300 DP2xTP2/fp8/fp16KV | 8 | 210.0 | 211.1 | 0.005 | ceiling | ttft 278 -> 224; sat 279 -> 224; slowed inf -> 217; wait 269 -> 220 |
| 781 | 27B/4xB300 data-par/nvfp4/fp8KV | 8 | 431.9 | 479.9 | 0.1 | ceiling | ttft 924 -> 584; sat 945 -> 592; wait 884 -> 570 |
| 782 | 35BA3B/6xB300 DP3xTP2/nvfp4/fp8KV | 42.6667 | 815.0 | 819.0 | 0.005 | ceiling | ttft 1,951 -> 856; sat 1,958 -> 857; wait 1,873 -> 847 |
| 783 | MM35/8xB300 DP4xTP2/fp8/fp16KV | 4 | 12.0 | 13.3 | 0.1 | ceiling | ttft 19 -> 14; sat 37 -> 17 |
| 789 | Q38FN/4xH200 tensor-par/fp8/fp16KV | 128 | 126.0 | 129.8 | 0.03 | ceiling | ttft 921 -> 163; sat 981 -> 167; wait 651 -> 149 |
| 790 | Q38FN/6xB300 DP3xTP2/fp8/fp16KV | 42.6667 | 234.0 | 236.4 | 0.01 | ceiling | ttft 4,556 -> 334; sat 4,770 -> 348; slowed 946 -> 243; wait 4,133 -> 314 |
| 791 | 35BA3B/6xB300 DP3xTP2/fp8/fp16KV | 5.33333 | 2,013.9 | 2,034.2 | 0.01 | ceiling | ttft 2,684 -> 2,082; sat 2,687 -> 2,083; wait 2,651 -> 2,077 |
| 793 | MM35/6xH200 tensor-par/fp8/fp16KV | 128 | 16.0 | 24.6 | 0.35 | load (1 at-load keys) |  |
| 794 | MM35/4xH200 DP2xTP2/fp8/fp8KV | 16 | 10.0 | 10.3 | 0.03 | ceiling+load (2 at-load keys) | sat 15 -> 11; slowed 11 -> 10 |
| 795 | 27B/4xB300 DP2xTP2/nvfp4/fp16KV | 64 | 267.9 | 297.7 | 0.1 | ceiling | ttft 509 -> 330; sat 515 -> 332; wait 440 -> 318 |
| 796 | GLM53F/5xB300 tensor-par/fp8/fp16KV | 32 | 975.0 | 979.9 | 0.005 | ceiling | ttft 1,856 -> 996; sat 1,924 -> 999; slowed 1,148 -> 981; wait 1,722 -> 991 |
| 797 | MM35/8xB300 data-par/fp8/fp16KV | 16 | 15.0 | 15.2 | 0.01 | ceiling+load (18 at-load keys) | ttft 28 -> 16; sat 32 -> 17; slowed inf -> 16; wait 23 -> 15 |
| 798 | 27B/2xB300 tensor-par/nvfp4/fp16KV | 64 | 209.0 | 211.1 | 0.01 | ceiling | ttft 272 -> 223; sat 273 -> 223; slowed inf -> 217; wait 262 -> 219 |
| 802 | 27B/4xB300 tensor-par/fp8/fp16KV | 64 | 243.9 | 245.2 | 0.005 | ceiling | ttft 6,436 -> 366; sat 6,526 -> 374; slowed 731 -> 248; wait 5,994 -> 336 |
| 803 | Q38FN/2xB300 tensor-par/fp8/fp16KV | 64 | 240.9 | 242.2 | 0.005 | ceiling | ttft 1,039 -> 269; sat 1,045 -> 270; slowed 319 -> 243; wait 971 -> 259 |
| 804 | MM35/5xB300 tensor-par/fp8/fp16KV | 32 | 241.9 | 244.3 | 0.01 | ceiling | ttft 530 -> 257; sat 539 -> 258; slowed 288 -> 245; wait 434 -> 249 |
| 806 | MM35/1xB300/fp8/fp8KV | 32 | 13.0 | 13.1 | 0.01 | ceiling+load (2 at-load keys) | slowed inf -> 13 |
| 808 | MM35/2xB300 data-par/fp8/fp16KV | 200 | 5.0 | 5.2 | 0.03 | ceiling+load (1 at-load keys) | ttft 41 -> 6; sat 65 -> 8; slowed 17 -> 6; wait 7 -> 5 |
| 810 | Q38FN/4xB300 data-par/nvfp4/fp16KV | 4 | 139.0 | 213.8 | 0.35 | ceiling | ttft 435 -> 294; sat 482 -> 313; wait 355 -> 265 |
| 811 | 35BA3B/8xB300 data-par/nvfp4/fp16KV | 8 | 684.0 | 687.4 | 0.005 | ceiling | ttft 1,451 -> 754; sat 1,453 -> 755; slowed inf -> 693; wait 1,441 -> 747 |
| 812 | 27B/2xB300 data-par/nvfp4/fp16KV | 16 | 146.9 | 148.4 | 0.01 | ceiling | ttft 2,346 -> 259; sat 2,395 -> 273; slowed 1,634 -> 163; wait 2,110 -> 211 |
| 814 | GLM53F/4xB300 DP2xTP2/fp8/fp16KV | 64 | 1,989.8 | 2,009.9 | 0.01 | ceiling | slowed inf -> 2,038 |
| 819 | 27B/8xB300 DP2xTP4/fp8/fp8KV | 8 | 1,274.9 | 1,287.8 | 0.01 | ceiling | slowed inf -> 1,299 |
| 822 | MM35/2xB300 tensor-par/fp8/fp8KV | 64 | 23.0 | 23.2 | 0.01 | ceiling+load (2 at-load keys) | ttft 38 -> 24; sat 58 -> 26; slowed inf -> 24 |
| 823 | GLM52/8xB300 DP2xTP4/nvfp4/fp8KV | 8 | 235.0 | 237.4 | 0.01 | ceiling | slowed inf -> 241 |
| 825 | 35BA3B/7xH200 data-par/fp8/fp16KV | 18.2857 | 225.0 | 232.0 | 0.03 | ceiling | ttft 318 -> 276; sat 319 -> 277; slowed inf -> 246; wait 311 -> 270 |
| 826 | 35BA3B/4xB300 data-par/nvfp4/fp16KV | 32 | 110.0 | 110.6 | 0.005 | ceiling | ttft 4,377 -> 213; sat 4,460 -> 226; slowed 192 -> 111; wait 3,970 -> 173 |
| 829 | MM35/3xB300 tensor-par/fp8/fp16KV | 64 | 24.0 | 24.1 | 0.005 | ceiling+load (2 at-load keys) | ttft 55 -> 26; sat 61 -> 27; slowed inf -> 25; wait 33 -> 24 |
| 830 | Q38FN/4xB300 DP2xTP2/nvfp4/fp8KV | 16 | 446.9 | 451.5 | 0.01 | ceiling | ttft 1,971 -> 482; sat 2,068 -> 485; wait 1,523 -> 468 |
| 831 | GLM53F/6xB300 DP3xTP2/nvfp4/fp16KV | 5.33333 | 453.0 | 503.3 | 0.1 | ceiling | ttft 534 -> 509; sat 551 -> 512 |
| 832 | Q38FN/5xB300 data-par/fp8/fp8KV | 3.2 | 97.0 | 97.4 | 0.005 | ceiling | ttft 606 -> 126; sat 631 -> 144; slowed inf -> 101; wait 561 -> 111 |
| 833 | 27B/3xH200 data-par/fp8/fp8KV | 133.333 | 386.8 | 388.7 | 0.005 | ceiling | slowed inf -> 390 |
| 834 | GLM53F/1xB300/nvfp4/fp16KV | 128 | 41.0 | 41.2 | 0.005 | ceiling+load (18 at-load keys) | ttft 362 -> 49; sat 378 -> 51; slowed 303 -> 42; wait 217 -> 42 |
| 835 | Q38FN/2xB300 tensor-par/fp8/fp8KV | 400 | 404.0 | 406.0 | 0.005 | ceiling | slowed inf -> 409 |
| 837 | Q38FN/7xB300 tensor-par/fp8/fp16KV | 400 | 6,416.9 | 6,481.8 | 0.01 | ceiling | ttft 11,799 -> 6,991; sat 11,803 -> 6,992; wait 11,754 -> 6,976 |
| 838 | 35BA3B/6xB300 DP2xTP3/nvfp4/fp8KV | 64 | 1,021.9 | 1,135.4 | 0.1 | ceiling | ttft 1,483 -> 1,177; sat 1,519 -> 1,181; wait 1,414 -> 1,168 |
| 839 | 35BA3B/4xB300 DP2xTP2/nvfp4/fp8KV | 200 | 465.9 | 480.3 | 0.03 | ceiling | ttft 694 -> 504; sat 702 -> 505; wait 614 -> 491 |
| 842 | 27B/8xB300 DP2xTP4/fp8/fp16KV | 16 | 175.9 | 177.7 | 0.01 | ceiling | ttft 1,448 -> 204; sat 1,590 -> 212; slowed 498 -> 181; wait 1,171 -> 193 |
| 843 | GLM53F/6xB300 DP2xTP3/nvfp4/fp16KV | 200 | 885.9 | 894.8 | 0.01 | ceiling | ttft 2,862 -> 950; sat 2,909 -> 953; slowed 1,168 -> 898; wait 2,375 -> 925 |
| 844 | 35BA3B/5xB300 data-par/nvfp4/fp16KV | 12.8 | 730.0 | 811.1 | 0.1 | ceiling | ttft 1,678 -> 958; sat 1,692 -> 961; wait 1,528 -> 925 |
| 846 | 27B/6xB300 DP2xTP3/nvfp4/fp16KV | 32 | 2,842.9 | 2,857.2 | 0.005 | ceiling | slowed inf -> 2,873 |
| 847 | 35BA3B/4xB300 DP2xTP2/fp8/fp8KV | 8 | 559.0 | 561.8 | 0.005 | ceiling | ttft 1,415 -> 643; sat 1,418 -> 644; slowed inf -> 576; wait 1,383 -> 625 |
| 848 | 35BA3B/3xB300 tensor-par/nvfp4/fp8KV | 16 | 1,437.0 | 1,596.6 | 0.1 | ceiling | ttft 5,557 -> 2,121; sat 5,611 -> 2,130; wait 4,990 -> 2,031 |
| 849 | 35BA3B/7xB300 data-par/fp8/fp8KV | 9.14286 | 912.9 | 1,404.4 | 0.35 | ceiling | ttft 2,178 -> 1,754; sat 2,189 -> 1,760; wait 2,156 -> 1,744 |
| 850 | 35BA3B/1xB300/fp8/fp8KV | 64 | 226.9 | 234.0 | 0.03 | ceiling | ttft 559 -> 264; sat 591 -> 271; wait 497 -> 253 |
| 851 | MM35/8xB300 data-par/nvfp4/fp8KV | 2 | 9.9 | 10.1 | 0.01 | ceiling | slowed inf -> 10 |
| 852 | MM35/6xB300 data-par/nvfp4/fp8KV | 66.6667 | 19.9 | 30.7 | 0.35 | load (1 at-load keys) |  |
| 853 | 35BA3B/3xH200 tensor-par/fp8/fp8KV | 128 | 292.0 | 324.4 | 0.1 | ceiling | sat 327 -> 325 |
| 856 | 35BA3B/6xB300 data-par/fp8/fp8KV | 21.3333 | 355.0 | 356.8 | 0.005 | ceiling | ttft 471 -> 360; sat 477 -> 361; slowed 359 -> 357; wait 414 -> 358 |
| 860 | 27B/6xH200 DP3xTP2/fp8/fp16KV | 42.6667 | 48.0 | 49.5 | 0.03 | ceiling | ttft 130 -> 52; sat 221 -> 57 |
| 861 | MM35/4xH200 tensor-par/fp8/fp8KV | 32 | 77.0 | 79.3 | 0.03 | ceiling | slowed inf -> 80 |
| 862 | 35BA3B/4xH200 data-par/fp8/fp8KV | 8 | 90.0 | 99.9 | 0.1 | ceiling | ttft 183 -> 109; sat 224 -> 114 |
| 865 | 27B/2xB300 tensor-par/nvfp4/fp8KV | 400 | 163.9 | 252.2 | 0.35 | load (1 at-load keys) |  |
| 866 | DSV4FV/6xH200 DP3xTP2/fp8/fp8KV | 5.33333 | 361.9 | 363.8 | 0.005 | ceiling | ttft 1,315 -> 386; sat 1,338 -> 388; slowed 986 -> 367; wait 1,072 -> 372 |
| 868 | 27B/7xB300 tensor-par/fp8/fp16KV | 32 | 374.0 | 375.9 | 0.005 | ceiling | ttft 1,719 -> 492; sat 1,728 -> 499; slowed 695 -> 377; wait 1,672 -> 463 |
| 870 | Q38FN/4xB300 data-par/fp8/fp8KV | 100 | 173.0 | 173.9 | 0.005 | ceiling | ttft 2,895 -> 351; sat 2,904 -> 357; slowed inf -> 191; wait 2,849 -> 327 |
| 871 | MM35/5xB300 data-par/fp8/fp8KV | 25.6 | 49.0 | 49.5 | 0.01 | ceiling | ttft 154 -> 53; sat 172 -> 58; slowed 168 -> 51; wait 121 -> 51 |
| 872 | 27B/4xH200 data-par/fp8/fp16KV | 8 | 46.0 | 46.5 | 0.01 | ceiling | ttft 269 -> 50; sat 355 -> 56; slowed 148 -> 47; wait 133 -> 47 |
| 874 | Q38FN/8xH200 DP4xTP2/fp8/fp16KV | 16 | 27.0 | 41.5 | 0.35 | ceiling | ttft 304 -> 162; sat 308 -> 164; wait 266 -> 141 |
| 875 | 35BA3B/4xH200 DP2xTP2/fp8/fp16KV | 200 | 153.0 | 235.4 | 0.35 | ceiling | ttft 410 -> 324; sat 412 -> 325; wait 387 -> 311 |
| 876 | 27B/2xB300 tensor-par/fp8/fp8KV | 400 | 576.0 | 581.8 | 0.01 | ceiling | ttft 1,348 -> 633; sat 1,356 -> 635; slowed 1,045 -> 586; wait 1,307 -> 624 |
| 878 | Q38FN/8xH200 DP2xTP4/fp8/fp16KV | 200 | 516.0 | 793.8 | 0.35 | ceiling | ttft 1,509 -> 1,108; sat 1,515 -> 1,111; wait 1,481 -> 1,094 |
| 879 | Q38FN/8xB300 data-par/nvfp4/fp16KV | 2 | 169.0 | 169.8 | 0.005 | ceiling | ttft 966 -> 222; sat 973 -> 226; slowed inf -> 178; wait 952 -> 216 |
| 880 | GLM53F/8xB300 DP4xTP2/fp8/fp16KV | 8 | 773.0 | 780.8 | 0.01 | ceiling | ttft 1,099 -> 829; sat 1,100 -> 829; slowed inf -> 795; wait 1,087 -> 825 |
| 881 | Q38FN/3xB300 data-par/fp8/fp16KV | 42.6667 | 122.0 | 125.7 | 0.03 | ceiling | ttft 3,014 -> 297; sat 3,088 -> 307; slowed 801 -> 146; wait 2,866 -> 281 |
| 883 | 35BA3B/6xB300 tensor-par/fp8/fp16KV | 64 | 3,419.8 | 3,437.0 | 0.005 | ceiling | ttft 3,690 -> 3,469; sat 3,696 -> 3,471; wait 3,679 -> 3,468 |
| 884 | GLM52/3xB300 tensor-par/fp8/fp8KV | 400 | 10.0 | 10.1 | 0.005 | ceiling+load (18 at-load keys) | ttft 573 -> 22; sat 593 -> 26; slowed 154 -> 11; wait 478 -> 15 |
| 886 | MM35/5xB300 data-par/nvfp4/fp16KV | 25.6 | 9.0 | 9.3 | 0.03 | ceiling+load (18 at-load keys) | ttft 235 -> 22; sat 250 -> 26; slowed 61 -> 11; wait 107 -> 12 |
| 889 | 27B/1xB300/fp8/fp16KV | 400 | 145.9 | 146.6 | 0.005 | ceiling+load (18 at-load keys) | ttft 867 -> 186; sat 875 -> 190; slowed 741 -> 150; wait 851 -> 180 |
| 890 | 27B/7xB300 data-par/fp8/fp16KV | 4.57143 | 183.9 | 184.9 | 0.005 | ceiling | ttft 229 -> 192; sat 234 -> 194; slowed inf -> 188; wait 219 -> 190 |
| 893 | 35BA3B/5xB300 data-par/fp8/fp16KV | 80 | 238.0 | 366.2 | 0.35 | ceiling | ttft 743 -> 539; sat 763 -> 550; wait 707 -> 521 |
| 894 | MM35/8xH200 tensor-par/fp8/fp8KV | 128 | 1,933.9 | 1,953.4 | 0.01 | ceiling | slowed inf -> 2,004 |
| 895 | 35BA3B/7xH200 tensor-par/fp8/fp16KV | 64 | 3,370.8 | 3,475.1 | 0.03 | ceiling | ttft 39,948 -> 6,072; sat 39,974 -> 6,076; wait 39,654 -> 6,026 |
| 899 | MM35/7xB300 data-par/fp8/fp16KV | 18.2857 | 8.0 | 8.8 | 0.1 | ceiling+load (2 at-load keys) | sat 14 -> 10 |
| 901 | Q38FN/6xB300 DP2xTP3/nvfp4/fp8KV | 200 | 3,681.8 | 3,795.7 | 0.03 | ceiling | slowed inf -> 3,821 |
| 902 | Q38FN/5xB300 data-par/nvfp4/fp16KV | 12.8 | 55.0 | 84.6 | 0.35 | ceiling | ttft 257 -> 148; sat 260 -> 149; wait 228 -> 137 |
| 903 | DSV41F/4xB300 tensor-par/fp8/fp8KV | 64 | 22,292.5 | 22,404.5 | 0.005 | ceiling | slowed inf -> 22,442 |
| 904 | MM35/8xB300 DP4xTP2/nvfp4/fp8KV | 16 | 44.0 | 44.2 | 0.005 | ceiling | ttft 294 -> 51; sat 305 -> 54; slowed inf -> 46; wait 244 -> 47 |
| 912 | 35BA3B/8xH200 DP2xTP4/fp8/fp8KV | 64 | 1,931.0 | 1,950.5 | 0.01 | ceiling | slowed inf -> 1,986 |
| 913 | 27B/8xB300 DP2xTP4/nvfp4/fp8KV | 16 | 1,844.8 | 1,854.1 | 0.005 | ceiling | ttft 3,800 -> 1,969; sat 3,805 -> 1,971; wait 3,742 -> 1,955 |
| 915 | 35BA3B/8xB300 tensor-par/nvfp4/fp8KV | 32 | 19,960.8 | 22,178.6 | 0.1 | ceiling | ttft 39,430 -> 25,361; sat 39,451 -> 25,366; wait 39,325 -> 25,335 |
| 917 | 35BA3B/6xB300 data-par/fp8/fp16KV | 66.6667 | 355.0 | 358.6 | 0.01 | ceiling | ttft 497 -> 384; sat 498 -> 384; wait 492 -> 382 |
| 920 | MM35/8xB300 data-par/nvfp4/fp16KV | 50 | 6.0 | 9.2 | 0.35 | ceiling+load (1 at-load keys) | sat 32 -> 18 |
| 922 | GLM53F/6xB300 data-par/nvfp4/fp8KV | 21.3333 | 275.9 | 277.3 | 0.005 | ceiling | ttft 1,357 -> 310; sat 1,372 -> 312; slowed 1,202 -> 282; wait 1,287 -> 301 |
| 924 | 35BA3B/8xB300 DP4xTP2/nvfp4/fp16KV | 16 | 1,512.9 | 1,559.7 | 0.03 | ceiling | ttft 9,984 -> 2,336; sat 9,996 -> 2,339; slowed 2,542 -> 1,589; wait 9,925 -> 2,323 |
| 926 | 35BA3B/4xB300 data-par/fp8/fp8KV | 32 | 239.0 | 367.7 | 0.35 | ceiling | ttft 598 -> 474; sat 607 -> 479; wait 558 -> 454 |
| 927 | 27B/3xH200 tensor-par/fp8/fp16KV | 16 | 3,004.0 | 3,019.1 | 0.005 | ceiling | slowed inf -> 3,037 |
| 928 | Q38FN/8xB300 tensor-par/fp8/fp16KV | 64 | 4,660.0 | 4,683.4 | 0.005 | ceiling | ttft 36,493 -> 8,568; sat 36,500 -> 8,575; slowed inf -> 5,142; wait 36,458 -> 8,534 |
| 929 | Q38FN/7xB300 data-par/nvfp4/fp8KV | 18.2857 | 159.0 | 159.8 | 0.005 | ceiling | ttft 2,997 -> 339; sat 3,013 -> 348; slowed inf -> 187; wait 2,966 -> 324 |
| 931 | 35BA3B/6xH200 data-par/fp8/fp8KV | 66.6667 | 846.0 | 850.2 | 0.005 | ceiling | ttft 2,439 -> 910; sat 2,447 -> 911; slowed inf -> 859; wait 2,402 -> 904 |
| 932 | MM35/4xH200 tensor-par/fp8/fp16KV | 400 | 207.7 | 230.8 | 0.1 | load (1 at-load keys) |  |
| 933 | GLM53F/2xB300 data-par/nvfp4/fp16KV | 200 | 895.0 | 904.0 | 0.01 | ceiling | ttft 935 -> 906; sat 950 -> 907; wait 905 -> 904 |
| 934 | 35BA3B/4xB300 tensor-par/nvfp4/fp8KV | 400 | 2,545.0 | 2,570.7 | 0.01 | ceiling | ttft 33,625 -> 3,751; sat 33,645 -> 3,753; slowed 16,030 -> 2,705; wait 33,408 -> 3,723 |
| 937 | Q38FN/6xB300 data-par/nvfp4/fp8KV | 21.3333 | 602.0 | 620.6 | 0.03 | ceiling | ttft 2,087 -> 803; sat 2,091 -> 804; slowed 1,250 -> 640; wait 2,045 -> 789 |
| 939 | 27B/5xB300 data-par/fp8/fp16KV | 12.8 | 59.0 | 60.8 | 0.03 | ceiling | ttft 472 -> 83; sat 492 -> 85; slowed 62 -> 61; wait 376 -> 75 |
| 941 | 35BA3B/6xH200 tensor-par/fp8/fp8KV | 32 | 1,718.0 | 1,771.1 | 0.03 | ceiling | ttft 2,995 -> 1,874; sat 2,999 -> 1,875; wait 2,950 -> 1,868 |
| 942 | DSV41F/8xB300 DP4xTP2/fp8/fp8KV | 4 | 1,355.8 | 1,362.7 | 0.005 | ceiling | ttft 4,908 -> 1,496; sat 4,917 -> 1,498; slowed 2,304 -> 1,367; wait 4,863 -> 1,488 |
| 943 | Q38FN/2xB300 tensor-par/nvfp4/fp16KV | 400 | 720.9 | 728.2 | 0.01 | ceiling | ttft 9,411 -> 1,310; sat 9,419 -> 1,313; slowed 4,879 -> 770; wait 9,319 -> 1,278 |
| 948 | MM35/8xB300 data-par/fp8/fp8KV | 16 | 8.0 | 12.3 | 0.35 | load (1 at-load keys) |  |
| 949 | Q38FN/7xH200 tensor-par/fp8/fp16KV | 32 | 1,407.4 | 1,414.5 | 0.005 | ceiling | ttft 4,761 -> 1,528; sat 4,776 -> 1,530; slowed 1,876 -> 1,417; wait 4,733 -> 1,523 |
| 950 | MM35/8xH200 tensor-par/fp8/fp16KV | 16 | 153.9 | 154.7 | 0.005 | ceiling | slowed inf -> 164 |
| 951 | MM35/1xB300/nvfp4/fp8KV | 32 | 53.0 | 53.2 | 0.005 | ceiling | ttft 75 -> 60; sat 76 -> 61; slowed inf -> 57; wait 57 -> 54 |
| 952 | MM35/8xB300 DP4xTP2/nvfp4/fp16KV | 4 | 45.0 | 45.4 | 0.01 | ceiling | ttft 61 -> 48; sat 64 -> 49; slowed inf -> 47; wait 48 -> 46 |
| 953 | 27B/2xB300 tensor-par/nvfp4/fp8KV | 128 | 755.0 | 778.3 | 0.03 | ceiling | ttft 2,289 -> 975; sat 2,308 -> 982; slowed 1,098 -> 788; wait 2,199 -> 944 |
| 954 | 27B/5xH200 data-par/fp8/fp16KV | 12.8 | 91.0 | 93.8 | 0.03 | ceiling | ttft 209 -> 148; sat 212 -> 150; slowed inf -> 105; wait 186 -> 130 |

</details>
