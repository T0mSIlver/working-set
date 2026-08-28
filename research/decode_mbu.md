# Decode calibration: the roofline is 3.3x optimistic on a production 27B

*Added 2026-08-28. Source: black-box `/metrics` scraping and load probes against
a production 27B deployment (Qwen3.6-27B FP8, 4xH200, vLLM), 2026-08-28,
four sessions. Employer infrastructure — do not cite the instance's identity,
its hostnames or raw dashboards in public-facing material; this note carries
the anonymised numbers, as `workload_agentic_poc.md` does.*

The study prices decode as a pure HBM roofline (`docs/scenarios.md`
limitation 11): per-user tok/s = `mtp * effective_bw / step_bytes`, with no
efficiency term and no fixed cost. Prefill got its MFU anchor
(`research/prefill.md`); decode never got its counterpart, so every tok/s
figure in the study has been an upper bound of unknown tightness. This note
measures the tightness on the study's own reference row.

**Headline: the explorer's per-user decode figure is 3.3-4x high on this
deployment, and the error is entirely in the pass time.** The byte ledger —
weights, KV per token, pool size, prefix handling — is confirmed correct.
Speculative decoding is confirmed to work *better* than the study assumes.
The root cause of the slow pass is NOT established; § 4 lays out what
survives elimination.

## 1. Method

`scripts/decode_mbu.py` harvests decode-only windows from `/metrics` counter
deltas and fits

    t_pass = t0 + step_bytes / (eta x bw_advertised)

`t0` is a fixed per-pass cost (collectives, launches, recurrent kernels);
`eta` is the achieved fraction of advertised aggregate HBM bandwidth — MBU,
the decode twin of prefill's MFU. It absorbs `tp_efficiency`, which a
single-TP-width run cannot separate from it, so it is reported in both
conventions exactly as prefill.md's MFU points are.

A window is used only if it is pure decode and structurally stable: no
prefill anywhere in it (`d(prompt_tokens_total) == 0`), batch size equal at
both ends, no request finished mid-window, and delivered tokens reconciling
against `steps x n x accepted_len`. `scripts/decode_probe.py` supplies the
batch-size and context-length variation production traffic does not.

Two properties of the estimator earn their keep. Windows are scrape PAIRS,
not consecutive samples, so window length adapts to how quiet the hour is;
and the bootstrap resamples PLATEAUS, not windows, because overlapping pairs
from one plateau are nowhere near independent (window-level resampling
reported intervals an order of magnitude too tight).

`decode_mbu.py --self-test` recovers known parameters from a synthetic log
(t0 2.40 -> 2.34 ms, eta 0.470 -> 0.463, accepted_len 1.853 -> 1.853).

## 2. What is measured

| Quantity | Measured | Study assumed | Status |
|---|---|---|---|
| Forward pass @ n=1 | **10-11.3 ms** | 2.11 ms | 4.75-5.0x slower |
| Per-user decode, n=1..4 | **250-330 tok/s** | 812 | reproduced in 4 sessions |
| Effective aggregate bandwidth | **4.71 TB/s** | 15.55 TB/s | **3.3x optimistic** |
| Weights read per pass | **~37 GB** | 30.9 GB (FP8) | FP8 confirmed; BF16 (62 GB) excluded |
| KV bytes per context token | **32 KiB** | 32 KiB | confirmed |
| Shared prefix read | **per sequence** | per sequence | confirmed; no cascade attention |
| KV pool | **13,161,600 tokens** | 13,911,155 | `kv_pool_tokens()` 0.95x — validated |
| `max_model_len` | 184,320 | cap default 180,000 | consistent |
| Speculative `accepted_len` | **2.94** (alpha 0.97) | `mtp` 1.7 | study is CONSERVATIVE |
| Acceptance model `1+a+a^2` | pos0 0.971, pos1 0.944 | predicts 0.943 | **holds** |
| Recurrent state dtype | **fp32** (`mamba_ssm_cache_dtype`) | bf16 (inferred) | study default wrong |

The bandwidth figure is the load-bearing one and it comes from the cleanest
measurement in the set: arm B held the batch at n=4 and varied only context
length, so weights, collectives, launches and draft overhead all cancel in
the difference. 174k extra context tokens cost 1.21 ms, i.e. 5.7 GB of KV at
**4.71 TB/s** — which is 98% of ONE H200 and 25% of the four the deployment
runs on.

## 3. The decomposition that failed, and why

`t0` and `eta` are **not** separately identified by this data. Three attempts:

1. **Passive traffic** (n 1-3): byte range 1.27x. The tool refused the split,
   as designed.
2. **Arm A** (batch sweep, shared prefix, n 1-25): byte range 4.03x, but the
   fit returned `t0 = -2.28 ms`, CI [-8.62, +3.67]. A negative fixed cost is
   unphysical; the affine model misfits.
3. **Pooled, with arm B's slope held fixed**: `t0 = 7.88 ms`, `558 us` per
   sequence. **Withdrawn** — re-fitting the same constrained model on a later
   session gives `-106 us` per sequence. The sign flips, so the per-sequence
   term is a collinearity artifact: in production traffic `n` and total
   context move together, and only a fixed-`n` context sweep separates them.

There is also a **structural degeneracy no timing experiment can break**:
`bytes x 3 at bandwidth x 3` is the same line as `bytes x 1 at bandwidth x 1`.
Only the ratio is identified. `vllm:estimated_read_bytes_per_gpu_total` would
settle it outright — the series exists on this deployment but reads 0.0 in
every one of 433 scrapes, so it is not populated.

What the degeneracy does NOT threaten: the *ratio* is what every study figure
depends on. Per-user tok/s, the decode ceiling and the binding-constraint
verdict are all functions of `bytes/bandwidth`, so they stand regardless of
how the factor divides.

## 4. Root cause: eliminated, and still open

| Hypothesis | Verdict | Evidence |
|---|---|---|
| Weights BF16 not FP8 | dead | arm B intercept ~37 GB; BF16 is ~62 GB |
| Different model than the study's 27B | dead | confirmed by the owner |
| KV > 32 KiB/token | dead | 13.16M tokens x 2x needs 862 GB; 519 GB exists |
| Spec decode runs k+1 FULL passes | dead | would make MTP a 1.0x speedup; and a KV-only inflation would have put the intercept at W/3 ~ 10 GB, not 37 |
| Read amplification (1600-token blocks) | dead | a 57k context is 36 whole blocks; rounding is negligible |
| Not really TP4 (PP, or placement) | dead | prefill runs at **157% of one H200's peak** — impossible on one GPU, and a 7.2k-token warm pass is a single chunk with nothing to pipeline |
| GPUs time-shared or power-capped | doubtful | same prefill argument; compute is healthy |
| Collective latency per layer | cannot explain it | cancels in the arm B difference |
| Partial CUDA-graph capture | explains the n=1 residual only | cancels in the slope |
| **Per-GPU MBU ~25% from the decode kernel path** | **leading** | see below |

Three external reference points, all pointing the same way:

- A single **RTX 5090** running Qwen3.6-27B NVFP4 with MTP reports 80-160
  tok/s decode. Normalised: ~48% MBU. This deployment is at ~16%. A
  single-GPU build of the same model family, with the same speculative
  decoding, is **3x more efficient per byte moved**.
- vLLM's own recipe for this model recommends **TP1** for FP8 on one
  H100/H200-class GPU, and `num_speculative_tokens: 1`. This deployment runs
  TP4 with k=2 on a model whose weights are 31 GB on a 141 GB card.
- Published TP sweeps for a 27B on H100 put TP4 at ~4,565 tok/s aggregate
  decode. This deployment reaches ~2,856 tok/s at n=42 — the same order,
  1.6x off. **Aggregate throughput is mediocre; single-stream is terrible.**

The consistent reading: TP4 is being used on a model small enough to need
TP1, decode GEMMs at 3 token positions are far too skinny to use four
GPUs' bandwidth, and 48 of the 64 layers are Gated DeltaNet, whose recurrent
decode step is occupancy-bound rather than bandwidth-bound. vLLM's own
Qwen3-Next notes list GatedDeltaNet kernel optimisation as roadmap work, and
report that a GDN verify pass "cannot parallelise across the proposed
positions the way attention layers can".

**The decisive experiment is one this study has not been able to run: the
same model, same vLLM, on ONE GPU.** If TP1 single-stream lands near
250-330 tok/s, TP4 is contributing nothing to decode and four DP replicas
would deliver ~3x the aggregate throughput at the same per-user speed. If
TP1 lands near 80 tok/s, TP4 is working and the architecture is simply
expensive. Nothing else separates those.

## 5. What this changes in the model

Proposed, not yet applied:

- **A decode MBU constant**, the counterpart to `MFU_DEFAULT`. On this
  deployment `eta = 0.25` of advertised (0.30 in the model convention, i.e.
  divided by `tp_efficiency(4)`). It is ONE deployment on a hybrid model and
  must not be transplanted to the dense rows without a second point — the
  same discipline prefill.md applied to its own bracket.
- **`mtp` for the 27B: 1.7 -> 2.94** on measured acceptance (alpha 0.97).
  Note this makes the ordering finding WORSE, not better: the study was
  under-crediting speculation and still over-predicting speed.
- **`state_dt` default fp32 for the 27B**, from `mamba_ssm_cache_dtype`.
  Cache ceiling 249 -> 238 users.
- **`kv_pool_tokens()` validated** at 0.95x against a real pool. The
  `BASELINE_POOL_TOKENS_27B_1GPU` projection stands.

Consequence for the study's central claim, at the measured workload:

| Ceiling | Study | Calibrated |
|---|---|---|
| Cache (binding, p5 warm) | 249 | 234 |
| Decode (40 tok/s floor) | 309 | **~150** |

**Decode binds before cache.** The thesis "cache binds before bandwidth
before compute" is inverted on this row. Every hybrid model in the study —
`35B-A3B` (30 DeltaNet layers), `Q38FN` (36), `GLM53F` (34 KDA), `DSV4F` —
is priced on the same dense-transformer roofline and carries the same
unpriced assumption.

## 6. What this did NOT change

- **The byte ledger.** Weights, KV per token, pool arithmetic and per-sequence
  prefix reads all survived measurement. The model's memory side is sound;
  only its time side is wrong.
- **The prefill MFU anchor.** Untouched, and independently corroborated here
  (157% of a single GPU's peak).
- **`kv_bpt`, `w_decode`, `ACT_RESERVE`.** All confirmed.

## 7. Caveats

- `eta` and `tp_efficiency` are not separable at one TP width.
- `t0` is unresolved. Every attempt to fit it returned either an unphysical
  value or one that did not replicate across sessions.
- All figures are one deployment, one model, one vLLM version, one week.
- Acceptance (2.94) is a property of THIS traffic — agentic coding, highly
  predictable output. It does not transplant to another prompt mix.
- The probe's synthetic prompts are random text; they are used only for
  `t_pass` (a hardware property). Acceptance is read passively from real
  traffic, never from the probe.

## Appendix: run log

| Session | Arms | Outcome |
|---|---|---|
| 09:07 | passive only | n 1-3, leverage 1.27x — split refused, as designed |
| 09:41 | A (k=1..16), C | k=8 and both C rungs aborted: the watchdog counted our OWN ramp queue as a co-tenant. Fixed in `6d63560` |
| 10:02 | B (8k/128k), C | **the load-bearing run** — clean fixed-n KV slope |
| 11:32 | D grid | drained: `--out-tokens 4000` at ~300 tok/s outlasted only 13 s of a 40 s hold. Fixed in `5c347a5` |
| 11:51 | D grid, retry | 158k rungs all HTTP 400 — random-text filler tokenises worse than the assumed 3.6 chars/token, overshooting `max_model_len`. Fixed in `eee43ae`. Short rungs ran but the instance was doing 4-5M prefill tokens per plateau, leaving almost no decode-only windows |

Four of five sessions were degraded by harness bugs or co-tenant load; the
measurement rests on the third. That is worth stating plainly: the numbers in
§ 2 are reproduced across sessions, but the bandwidth figure specifically
comes from one clean 80-second pair of plateaus.
