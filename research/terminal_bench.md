# Terminal-Bench — the quality axis of the frontier

**Purpose:** one comparable coding-agent score per study model, so the
frontier can rank configurations on what they deliver and not only on what
they cost. Consumed by `CONFIG.QUALITY` in the explorer (chart H's x-axis
and the frontier table's two score columns).

Two versions are carried, **2.1** and **4.0**, because Artificial Analysis
publishes both and they do not rank this field the same way (§ 4). Chart H's
axis toggles between them; the table prints both columns at once.

## 1. Why one lab, one harness

Vendor model cards report Terminal-Bench on whichever version and agent
harness flattered the launch (DeepSeek's card gives V4-Flash-0731 82.7 on
"2.1"; Z.ai's GLM-5.2 launch post 81.0). Different harnesses, different
scaffolds, different effort settings: not comparable across vendors. The
study therefore takes every score from **one independent lab under one
protocol per version** — Artificial Analysis — and ignores the vendor
figures. The cost of that choice is a lower absolute level (AA's numbers run
3–9 points under the cards on 2.1); the frontier only needs the ordering and
the gaps.

**One exception, by owner decision (2026-09-10):** DeepSeek-V4.1-Flash was
released that day with **no AA run of any Terminal-Bench version** (there is
no `artificialanalysis.ai/models/deepseek-v4-1-flash` page — checked
2026-09-10, HTTP 404 — and no DeepSeek V4.1 entry in the model list embedded
in AA's other pages), and the owner chose to carry the vendor card's own
**90.6 (2.1)** and **31.2 (4.0)** rather than leave the row unscored. Both
are DeepSeek Harness, Minimal mode, `reasoning_effort=100`, 1M-token
context, N=3 samples per task, no network on 2.1. It is the only row not
measured under AA's protocol, and the card itself shows why the one-harness
rule exists: the same model at the same effort scores 84.1 (Codex scaffold)
to 90.6 (DSH Minimal) on 2.1, a 6.5-point spread from the scaffold alone
(`research/model_dsv41flash.md` § 5); the card publishes no per-scaffold
split for 4.0, so that version's caveat is the same in kind and unknown in
size.
Going by the other six models, AA's numbers will likely land lower (an
expectation, not a measurement) — and the same card's 2.1 figures for the
two models AA has also run sit 4.3 and 4.0 points above AA's (GLM-5.3 88.2
vs 83.9; V4-Flash-0731 82.7 vs 78.7). On 4.0 the gap runs the *other* way for both models AA has
also run — AA 12.1% vs the card's 7.0 for V4-Flash-0731, AA 41.9% vs the
card's 37.9 for GLM-5.3 — so on 4.0 not even the sign of the vendor bias is
settled, and "expect the AA number to be lower" does not transfer. The vendor figures are to be **replaced by the AA numbers the day
they publish**, and the ledger rows and `CONFIG.QUALITY.DSV41F.source` carry
the provenance until then.

**Protocols (AA methodology page, read 2026-09-10):**

| | Terminal-Bench 2.1 | Terminal-Bench 4.0 |
|---|---|---|
| status at AA | legacy eval | Intelligence Index v4.3 |
| tasks | 89 | 66 |
| harness | Terminus 2 (E2B sandbox) | mini-SWE-agent v2.4.6 (per-task verifier container) |
| repeats | 3 | 3 |
| scoring | pass@1 averaged over repeats | pass@1 averaged over repeats; a task passes only if every test passes, verifier timeout counts as failure |
| denominator | 267 runs | 198 runs |

Every fraction below is an exact multiple of 1/267 or 1/198, which is how the
numbers were checked against the pages.

**Variant rule:** the *reasoning* variant at the effort AA ran for its
Intelligence Index (the agentic-coding workload the study models runs the
model thinking). Where AA lists several effort levels, the one carrying
the index score was taken. The explorer's FP16-KV and NVFP4 arms inherit
their base model's score — quantisation loss on this benchmark is not
measured, and a per-arm guess would rank the frontier on the guess.

## 2. Ledger

Read from the per-model pages' embedded dataset (`terminalbenchV21` and
`terminalbenchV40` fields), 2026-09-10. Percent = runs passed / 267 (2.1) or
/ 198 (4.0).

| Explorer key | AA slug | AA name | TB 2.1 | runs | TB 4.0 | runs | AA release date |
|---|---|---|---|---|---|---|---|
| `27B` | `qwen3-8-27b` | Qwen3.8 27B (xhigh) | **79.8%** | 213 | **5.6%** | 11 | 2026-08-14 |
| — | `qwen3-6-27b` | Qwen3.6 27B (Reasoning) — the model `27B` was until 2026-09-05 | 60.7% | 162 | — | — | 2026-04-22 |
| `35BA3B` | `qwen3-6-35b-a3b` | Qwen3.6 35B A3B (Reasoning) | **44.9%** | 120 | **0.0%** | 0 | 2026-04-16 |
| `MM35` | `mistral-medium-3-5` | Mistral Medium 3.5 (high) | **50.6%** | 135 | **0.0%** | 0 | 2026-04-29 |
| `GLM52` | `glm-5-3` | GLM-5.3 (max) | **83.9%** | 224 | **41.9%** | 83 | 2026-08-18 |
| — | `glm-5-2` | GLM-5.2 (max) — the model `GLM52` was until 2026-09-06 | 77.9% | 208 | — | — | 2026-06-16 |
| — | `deepseek-v4-flash` | DeepSeek V4 Flash 0731 (Reasoning, Max Effort) — the model `DSV4F` was until 2026-09-10 | 78.7% | 210 | 12.1% | 24 | 2026-07-31 |
| `DSV41F` | — | DeepSeek V4.1 Flash — **vendor figures, not AA runs** (§ 1 exception): the model card's pass@1 on the **DeepSeek Harness, Minimal mode, max reasoning effort, 1M context**. Source: `https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash` (README, read 2026-09-10). **No AA run of either version as of 2026-09-10**; replace when it publishes. | **90.6%** (vendor) | — | **31.2%** (vendor) | — | 2026-09-10 |
| `Q38FN` | `qwen3-8-flash-next` | Qwen3.8-Flash-Next | **86.1%** | 230 | **25.3%** | 50 | 2026-08-26 |
| `GLM53F` | `glm-5-3-flash` | GLM-5.3-Flash (max) | **84.3%** | 225 | **32.8%** | 65 | 2026-08-26 |

A `—` in a 4.0 column is **no run**; a **0.0%** is a *measured* zero (0 of
198 runs passed) and is plotted. The two are different facts and the
explorer keeps them apart: `frontierScore` tests `Number.isFinite`, not
truthiness, so a measured zero is a dot at the left edge and a missing run
is an absence.

Not taken, for the record: Qwen3.8 27B's other effort levels (2.1 low 67.4%,
medium 65.2%, non-reasoning 49.1%; 4.0 low 2.5%, medium 5.1%, non-reasoning
no run — xhigh is the level AA's index runs), the non-reasoning variants
(Qwen3.6 27B 51.3%, 35B-A3B 41.6%, GLM-5.2 51.7% on 2.1, none of which has a
4.0 run) and AA's *Terminal-Bench Hard* (only four of the seven models have a
run, so it cannot be an axis).

Source pages: `https://artificialanalysis.ai/models/<AA slug>`; protocols
`https://artificialanalysis.ai/methodology/intelligence-benchmarking`;
leaderboard `https://artificialanalysis.ai/evaluations/terminalbench-v4-0`.
AA's data API (`/api/v2/data/llms/models`) serves the same fields behind a
free key; the pages were used because no key was on hand.

## 3. What the score does and does not say

- It is a **model** property. Every row of a model — TP2, DP4, FP16-KV —
  gets the same x; the vertical spread at one x is what the topology costs.
- It is **not** the study's workload. AA's runs are at the vendor's max
  effort; the explorer's tokens-per-request come from a production trace
  (`research/workload_agentic_poc.md`). A cheaper effort setting would
  score lower and generate fewer tokens; neither side of that trade is in
  the model yet.
- It moves. Six AA-scored models released between April and August 2026
  span 45 to 86 points on 2.1 and 0 to 42 on 4.0; the two newest lead 2.1 but
  on 4.0 they trail GLM-5.3, released eight days earlier. Re-read the ledger when a model is added, and record the date.
  The default for a model without a run on a version is `null` (left off that
  axis, `—` in that column); the one departure is DeepSeek V4.1 Flash's
  vendor pair (§ 1) — read its 2.1 lead over Qwen3.8-Flash-Next (86.1, AA)
  as *unknown*, not as 4.5 points, and its 4.0 position against GLM-5.3
  (41.9, AA) the same way, until AA publishes.

## 4. Why both versions, and not one

The two versions rank the top of this field differently, and the difference
is not noise:

| | 2.1 | 4.0 |
|---|---|---|
| spread over the seven study models | 45.7 pts (44.9–90.6) | 41.9 pts (0.0–41.9) |
| spread over the **top four** | 6.7 pts (83.9–90.6) | 16.7 pts (25.3–41.9) |
| best AA-measured model | Qwen3.8-Flash-Next 86.1 | GLM-5.3 41.9 |
| GLM-5.3 vs Qwen3.8-Flash-Next | −2.2 | **+16.7** |
| models scoring 0 | none | two |

On 2.1 the four strongest models are inside 7 points, which is roughly the
spread one scaffold change produces on the same model (§ 1) — the version
has saturated for this field. On 4.0 the same four are inside 17. How many
steps chart H's staircase then has depends on the explorer state, not on the
benchmark alone: the efficient set is recomputed per GPU, load and price, so
at the 64-user H200 default both versions collapse to a single step
(Qwen3.8-Flash-Next TP8), while at 8 users on 8×H200 the 2.1 staircase has
two steps and the 4.0 staircase four — the same rows, reordered by the axis.
Reading 4.0 as "the harder, therefore better" axis is tempting and wrong in
one direction: it is also a *different harness* (mini-SWE-agent, not
Terminus 2) and a *smaller* task set (66, not 89), so a 0.0% is 0 of 198
runs and carries a wide binomial interval at the bottom of the scale. What
the explorer claims is only that the choice of version changes which
configurations are Pareto-efficient, so the reader should see both — hence
the toggle rather than a pick.

The default axis is **2.1**: it is the version every model in the study has
a run on, and the one the write-up's prose was built against. That is a
presentation default, not a judgement that 2.1 is the better benchmark.
