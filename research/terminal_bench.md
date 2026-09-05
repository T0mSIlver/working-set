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
2026-09-05. Percent = runs passed / 267.

| Explorer key | AA slug | AA name | TB 2.1 | runs | AA release date |
|---|---|---|---|---|---|
| `27B` | `qwen3-8-27b` | Qwen3.8 27B (xhigh) | **79.8%** | 213 | 2026-08-14 |
| — | `qwen3-6-27b` | Qwen3.6 27B (Reasoning) — the model `27B` was until 2026-09-05 | 60.7% | 162 | 2026-04-22 |
| `35BA3B` | `qwen3-6-35b-a3b` | Qwen3.6 35B A3B (Reasoning) | **44.9%** | 120 | 2026-04-16 |
| `MM35` | `mistral-medium-3-5` | Mistral Medium 3.5 (high) | **50.6%** | 135 | 2026-04-29 |
| `GLM52` | `glm-5-3` | GLM-5.3 (max) | **83.9%** | 224 | 2026-08-18 |
| — | `glm-5-2` | GLM-5.2 (max) — the model `GLM52` was until 2026-09-06 | 77.9% | 208 | 2026-06-16 |
| `DSV4F` | `deepseek-v4-flash` | DeepSeek V4 Flash 0731 (Reasoning, Max Effort) | **78.7%** | 210 | 2026-07-31 |
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
- It moves. Seven models released between April and August 2026 span 45 to
  86 points; the two newest sit at the top. Re-read the ledger when a model
  is added, and record the date.
