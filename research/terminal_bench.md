# Terminal-Bench 2.1 — the quality axis of the frontier

**Purpose:** one comparable coding-agent score per study model, so the
frontier can rank configurations on what they deliver and not only on what
they cost. Consumed by `CONFIG.QUALITY` in the explorer (chart H's x-axis
and the frontier table's score column).

## 1. Why one lab, one harness

Vendor model cards report Terminal-Bench on whichever version and agent
harness flattered the launch (DeepSeek's card gives V4-Flash-0731 82.7 on
"2.1"; Z.ai's GLM-5.2 launch post 81.0). Different harnesses, different
scaffolds, different effort settings: not comparable across vendors. The
study therefore takes every score from **one independent lab under one
protocol** — Artificial Analysis — and ignores the vendor figures. The
cost of that choice is a lower absolute level (AA's numbers run 3–9 points
under the cards); the frontier only needs the ordering and the gaps.

**One exception, by owner decision (2026-09-10):** DeepSeek-V4.1-Flash was
released that day with no AA run, and the owner chose to carry the vendor
card's own **90.6** (DeepSeek Harness, Minimal mode, `reasoning_effort=100`,
1M-token context, N=3 samples per task, no network) rather than leave the
row unscored. It is the only row not measured under AA's protocol, and the
card itself shows why the one-harness rule exists: the same model at the
same effort scores 84.1 (Codex scaffold) to 90.6 (DSH Minimal) on the same
benchmark, a 6.5-point spread from the scaffold alone
(`research/model_dsv41flash.md` § 5). Going by the other six models, AA's
number will likely land 3–9 points lower (an expectation, not a
measurement) — and the same card's figures for the two models AA has also
run sit 4.3 and 4.0 points above AA's (GLM-5.3 88.2 vs 83.9; V4-Flash-0731
82.7 vs 78.7). The vendor figure is to be
**replaced by the AA number the day it publishes**, and the ledger row and
`CONFIG.QUALITY.DSV41F.source` carry the provenance until then.

**Protocol (AA methodology page, read 2026-09-05):** Terminal-Bench v2.1,
89 tasks, Terminus 2 agent harness in an E2B sandbox, pass@1 averaged over
3 repeats per task. Every fraction below is an exact multiple of 1/267
(= 89 × 3 runs), which is how the numbers were checked against the page.

**Variant rule:** the *reasoning* variant at the effort AA ran for its
Intelligence Index (the agentic-coding workload the study models runs the
model thinking). Where AA lists several effort levels, the one carrying
the index score was taken. The explorer's FP16-KV and NVFP4 arms inherit
their base model's score — quantisation loss on this benchmark is not
measured, and a per-arm guess would rank the frontier on the guess.

## 2. Ledger

Read from the per-model pages' embedded dataset (`terminalbenchV21` field),
2026-09-05 (the `glm-5-3` row 2026-09-06). Percent = runs passed / 267.

| Explorer key | AA slug | AA name | TB 2.1 | runs | AA release date |
|---|---|---|---|---|---|
| `27B` | `qwen3-8-27b` | Qwen3.8 27B (xhigh) | **79.8%** | 213 | 2026-08-14 |
| — | `qwen3-6-27b` | Qwen3.6 27B (Reasoning) — the model `27B` was until 2026-09-05 | 60.7% | 162 | 2026-04-22 |
| `35BA3B` | `qwen3-6-35b-a3b` | Qwen3.6 35B A3B (Reasoning) | **44.9%** | 120 | 2026-04-16 |
| `MM35` | `mistral-medium-3-5` | Mistral Medium 3.5 (high) | **50.6%** | 135 | 2026-04-29 |
| `GLM52` | `glm-5-3` | GLM-5.3 (max) | **83.9%** | 224 | 2026-08-18 |
| — | `glm-5-2` | GLM-5.2 (max) — the model `GLM52` was until 2026-09-06 | 77.9% | 208 | 2026-06-16 |
| — | `deepseek-v4-flash` | DeepSeek V4 Flash 0731 (Reasoning, Max Effort) — the model `DSV4F` was until 2026-09-10 | 78.7% | 210 | 2026-07-31 |
| `DSV41F` | — | DeepSeek V4.1 Flash — **vendor figure, not an AA run** (§ 1 exception): the model card's Terminal-Bench 2.1 pass@1 on the **DeepSeek Harness, Minimal mode, max reasoning effort, 1M context**. Source: `https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash` (README, read 2026-09-10). **No AA run as of 2026-09-10**; replace when it publishes. | **90.6%** (vendor) | — | 2026-09-10 |
| `Q38FN` | `qwen3-8-flash-next` | Qwen3.8-Flash-Next | **86.1%** | 230 | 2026-08-26 |
| `GLM53F` | `glm-5-3-flash` | GLM-5.3-Flash (max) | **84.3%** | 225 | 2026-08-26 |

Not taken, for the record: Qwen3.8 27B's other effort levels (low 67.4%,
medium 65.2%, non-reasoning 49.1% — xhigh is the level AA's index runs),
the non-reasoning variants (Qwen3.6 27B 51.3%, 35B-A3B 41.6%, GLM-5.2
51.7%; DeepSeek V4 Flash non-reasoning has no v2.1 run) and
AA's *Terminal-Bench Hard* (only four of the seven models have a run, so it
cannot be the axis).

Source pages: `https://artificialanalysis.ai/models/<AA slug>`; protocol
`https://artificialanalysis.ai/methodology/intelligence-benchmarking`. AA's
data API (`/api/v2/data/llms/models`) serves the same fields behind a free
key; the pages were used because no key was on hand.

## 3. What the score does and does not say

- It is a **model** property. Every row of a model — TP2, DP4, FP16-KV —
  gets the same x; the vertical spread at one x is what the topology costs.
- It is **not** the study's workload. AA's runs are at the vendor's max
  effort; the explorer's tokens-per-request come from a production trace
  (`research/workload_agentic_poc.md`). A cheaper effort setting would
  score lower and generate fewer tokens; neither side of that trade is in
  the model yet.
- It moves. Six AA-scored models released between April and August 2026
  span 45 to 86 points; the two newest sit at the top. Re-read the ledger
  when a model is added, and record the date. The default for a model
  without an AA run is `null` (left off the chart, `—` in the table); the
  one departure from that default is DeepSeek V4.1 Flash's vendor 90.6
  (§ 1), which puts it at the top of the axis on a number that is not
  measured like the other six — read its lead over Qwen3.8-Flash-Next
  (86.1, AA) as *unknown*, not as 4.5 points, until AA publishes.
