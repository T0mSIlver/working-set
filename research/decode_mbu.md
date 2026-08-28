# Decode calibration: speculation is not free on a hybrid, and the roofline says it is

*Added 2026-08-28. Source: black-box `/metrics` scraping and load probes against
a production 27B deployment (Qwen3.6-27B FP8, 4xH200, vLLM 0.25.1), 2026-08-28,
five sessions, plus the engine's startup log. Employer infrastructure — do not cite the instance's identity,
its hostnames or raw dashboards in public-facing material; this note carries
the anonymised numbers, as `workload_agentic_poc.md` does.*

The study prices decode as a pure HBM roofline (`docs/scenarios.md`
limitation 11): per-user tok/s = `mtp * effective_bw / step_bytes`, with no
efficiency term and no fixed cost. Prefill got its MFU anchor
(`research/prefill.md`); decode never got its counterpart, so every tok/s
figure in the study has been an upper bound of unknown tightness. This note
measures the tightness on the study's own reference row.

**Headline: the explorer's per-user decode figure is 3.3-4x high on this
deployment, and the cause is that the study prices speculative decoding as
free.** It is not free on a hybrid. A Gated DeltaNet state update is a
sequential recurrence, so verifying `1 + k` speculated positions costs
`1 + k` passes over the DeltaNet layers where attention verifies all of them
in one kernel. On this model — 48 GDN layers, 16 attention — that is a
**2.5x multiplier on the weight bytes of every decode step**, which the model
does not carry. Priced correctly the deployment runs at **41-48% MBU**: a
normal, healthy decode efficiency on correctly configured hardware.

The byte ledger the study already had is confirmed exactly right (§ 2). The
error is that the study counts the weights once per step and credits `mtp`
tokens for free. The engine reads them 2.5 times.

Net effect: **MTP is worth ~16% on this architecture, not 2.9x** (§ 4.3).

An earlier draft of this note (and four days of measurement) treated the
gap as a deployment defect. It is not: the startup log shows a correctly
configured engine (§ 4.1). The defect is in the model.

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
| Effective aggregate bandwidth | **4.71 TB/s** | 15.55 TB/s | 3.3x — but see § 4: this is at the study's byte ledger. Price the verify cost and it is 7.8-9.2 TB/s, i.e. **41-48% MBU** |
| Weights read per pass | **~37 GB** at the study's ledger | 30.9 GB (FP8) | FP8 confirmed by the log: `Checkpoint size: 28.75 GiB`, sharded 7.3 GiB/rank |
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

## 4. Root cause: resolved by the startup log

### 4.1 Every deployment hypothesis is dead

Four sessions of black-box probing narrowed this to "the decode path is
slow" and stalled. The engine's startup log closed it in one reading, and
every configuration hypothesis the probing had left alive died at once:

| Suspected | Log says |
|---|---|
| Pipeline parallel, or placement | `tensor_parallel_size=4, pipeline_parallel_size=1` |
| Weights replicated, not sharded | `Model loading took 7.3 GiB` per rank — 28.75/4, sharded |
| Fallback attention backend | `Using FLASH_ATTN`, FlashAttention **version 3** |
| Slow FP8 path | `FlashInferFp8DeepGEMMDynamicBlockScaledKernel` |
| Eager mode / no CUDA graphs | `enforce_eager=False`, `FULL_AND_PIECEWISE`, FULL decode graphs to batch 288 |
| PCIe, custom all-reduce disabled | `flashinfer allreduce backend: mnnvl` — NVLink, fast path |
| Stale vLLM predating GDN kernels | 0.25.1 |

**There is no configuration change behind this.** The deployment is set up
correctly, which is worth recording because it was the working hypothesis
for four days.

### 4.2 What the log actually revealed

    WARNING vllm.config.speculative : Enabling num_speculative_tokens > 1
    will run multiple times of forward on same MTP layer

A GDN state update is `S_new = alpha * S_old + beta * (k (x) v)` — sequential
in position. The layer cannot verify `1 + k` proposed tokens in one batched
pass the way an attention layer can, so the verify cost grows close to
linearly in the number of proposed positions on the GDN portion of the
network. With `num_speculative_tokens = 2`:

    weight multiplier = (48 layers x 3 + 16 layers x 1) / 64 = 2.50x
    step bytes        = 30.9 GB x 2.50 + ~3 GB (two MTP draft forwards) + KV
                      ~ 82 GB, against the 33 GB the study counts

At that ledger the measured passes price out at **7.8 TB/s (41% of
advertised) at n=1 and 9.2 TB/s (48%) in the arm B plateau** — squarely in
the range a healthy decode should hit, and the same range prefill's measured
MFU sits in on this hardware.

This also retires the structural degeneracy of § 3 from the outside: the
"bytes x 2.5" branch is the true one, and it was never resolvable from
timing alone.

### 4.3 Speculation is nearly a wash on this architecture

The study models `mtp` as a free multiplier on decode speed. Pricing the
verify cost against the acceptance actually measured (alpha = 0.971):

| `num_speculative_tokens` | byte multiplier | tokens/pass | tok/s |
|---|---|---|---|
| 0 (off) | 1.00x | 1.00 | 238 |
| 1 (what the vLLM recipe recommends) | 1.75x | 1.97 | 268 |
| **2 (as deployed)** | **2.50x** | **2.91** | **277** |
| 3 | 3.25x | 3.83 | 280 |

**MTP is worth ~16% here, not the 2.9x its acceptance rate suggests.** The
curve is nearly flat in `k`: each extra draft buys another accepted token
and costs another pass over three quarters of the network. The study's
`mtp: 1.7` — credited to every DeltaNet row as a free speedup — is the
single largest error this exercise found, and it is an error of KIND, not
of calibration. A model with no linear-attention layers does not have it.

## 5. What this changes in the model

Proposed, not yet applied:

- **Price the speculative verify cost.** `mtp` must stop being a free
  multiplier on hybrid rows. The correct form multiplies the WEIGHT bytes of
  the linear-attention layers by `1 + k` while dividing the step count by the
  accepted length:

      weight_mult = (n_linear x (1 + k) + n_attn) / (n_linear + n_attn)

  For the 27B at k=2 that is 2.50x. Every model in the study with DeltaNet /
  KDA / linear-attention layers needs it: 35B-A3B (30 of 40), Q38FN (36 of
  48), GLM53F (34 of 45), DSV4F. A dense row is unaffected.
- **A decode MBU constant**, the counterpart to `MFU_DEFAULT`. With the
  verify cost priced, this deployment sits at **eta = 0.41-0.48** of
  advertised — close enough to prefill's 0.40 that one shared efficiency
  constant may serve both. That is a far more comfortable place to be than
  the 0.25 the uncorrected ledger implied.
- **`mtp` for the 27B: measured accepted_len 2.94** (alpha 0.971), but see
  above — the speedup NET of the verify cost is ~1.16x.
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
before compute" is inverted on this row. Note the mechanism: not because
bandwidth is scarcer than the study thought, but because each decode step
moves 2.5x the bytes the study counts. Every hybrid model in the study —
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

| A100 box | B | inconclusive: one plateau completed, model identity unknown, no slope |
| startup log | — | **resolved it in one reading** |

Four of five H200 sessions were degraded by harness bugs or co-tenant load;
the measurement rests on the third. The numbers in § 2 are reproduced across
sessions, but the bandwidth figure specifically comes from one clean
80-second pair of plateaus.

The methodological lesson is worth more than the schedule: **four days of
black-box probing established WHAT (a 3.3x gap, byte ledger correct, time
side wrong) and could not establish WHY.** The startup log answered it in
one line. Ask for the log first. The probing was not wasted — it is what
made the log line legible, and what produced the per-token, per-sequence and
pool validations in § 2 — but the ordering should have been reversed.
