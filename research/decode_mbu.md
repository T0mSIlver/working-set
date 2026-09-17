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
sequential recurrence, so verifying `1 + k` speculated positions cannot be
batched across positions in the DeltaNet layers the way attention batches
them. On this model -- 48 GDN layers, 16 attention -- that serialisation is
the leading explanation for a pass that takes 10.5 ms where the byte
ledger says 2.1.

**Whether it costs BYTES or LATENCY is not established** (§ 4.3). Both
readings fit the timing identically and neither is reachable from outside
the pod. The correction in § 5 does not depend on which is true, which is
the point of an efficiency constant.

The byte ledger the study already had is confirmed exactly right (§ 2) --
weights, KV per token, pool size, per-sequence prefix reads. The error is
entirely on the time side.

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
| Effective aggregate bandwidth | **4.92 TB/s** | 15.55 TB/s | 3.3x — but see § 4: this is at the study's byte ledger. Price the verify cost and it is 7.8-9.2 TB/s, i.e. **41-48% MBU** |
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

What the degeneracy threatens, and what it does not -- **corrected on
cross-review (opencode glm-5.3); the first draft of this paragraph was
wrong.** The ratio argument holds only if the missing factor multiplies the
WHOLE step ledger. Reading A multiplies the linear-attention WEIGHT bytes and
leaves KV alone, so folding it into a single efficiency mis-prices any
weights:KV mix away from the fitted one -- and the fit region (n = 1-25,
weights 80-95% of the step) is the opposite regime from where the decode
ceiling lives (n ~ 100-250, KV-dominated).

Anchoring all three to the same n=1 measurement, the 27B / 4xH200 decode
ceiling at the 40 tok/s floor reads:

| priced as | decode ceiling | vs cache 249 |
|---|---|---|
| a single whole-ledger constant (what this PR ships) | **108** | inverts |
| reading A, weights x(1+k) | **238** | H7 SURVIVES |
| reading B, serial latency | **154** | inverts |

So **the decode ceiling is not identified by this measurement, and neither is
the binding order.** What IS identified is per-user speed near the fitted mix,
where all three agree within ~10%. Every ceiling quoted in this note is
therefore a range, and the H7 result below is conditional on the mechanism.

This raises the value of the k-sweep in sec 4.3 considerably: it settles A
versus B, and A versus B settles H7.

## 4. Root cause: localised by the startup log, not fully resolved

### 4.1 Every deployment hypothesis is dead

Four sessions of black-box probing narrowed this to "the decode path is
slow" and stalled. The engine's startup log killed every configuration
hypothesis the probing had left alive, in one reading:

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

A GDN state update is `S_new = alpha * S_old + beta * (k (x) v)` -- sequential
in position. The layer cannot verify `1 + k` proposed tokens in one batched
pass the way an attention layer can. Three quarters of this model's layers
are GDN, and decode is the regime where that bites: PREFILL uses the chunked
form of the same recurrence (thousands of tokens become matmuls, fully
parallel), which is why prefill measures a healthy 40% MFU on the same GPUs
while decode does not.

### 4.3 Two readings of the same 10.5 ms, and no way to separate them

    A  bytes:   the GDN weights are re-read once per verified position,
                (48x3 + 16x1)/64 = 2.50x, so a step moves ~82 GB not ~33 GB,
                and the machine streams at ~41-48% of advertised
    B  latency: the weights are read ONCE (the q/k/v/gate and output
                projections are GEMMs over all 3 positions at once), the
                machine streams at ~25%, and the residual ~3.5 ms is
                144 dependent state updates that move almost no data

**B is the more parsimonious**, and the direct measurement leans that way:
the arm B slope prices the KV read path, where the count is unambiguous
(FlashAttention reads the KV once for all 3 query positions), at 4.9 TB/s.
Reading A needs the weights to stream at 7.8 TB/s while the KV streams at
4.9 simultaneously -- possible (contiguous versus gathered) but it needs two
efficiencies where B needs one.

An earlier draft of this note asserted A as settled. It is not: it rests on
a mechanism inferred from a log warning about the DRAFT head, and § 3's
structural degeneracy says plainly that timing alone cannot choose. Settling
it needs a profiler on the pod, or a spec-off / k-sweep A/B -- and those
predict OPPOSITE outcomes, which makes the k-sweep the cheap experiment:
under B, raising `num_speculative_tokens` from 2 to 5 gains ~33%; under A it
LOSES ~6%.

Neither reading changes § 5. That is what an efficiency constant is for.

### 4.4 Speculation is nearly a wash under reading A -- and a big win under B

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

  For the 27B at k=2 that is 2.50x -- IF reading A holds (§ 4.3). Under
  reading B the same serialisation costs latency instead, and the right form
  is a per-position term rather than a byte multiplier. Until an A/B settles
  it, `MBU_DEFAULT` absorbs whichever it is, which is why that constant is
  documented as travelling with `mtp`. Every model in the study with
  DeltaNet / KDA / linear-attention layers is affected either way: 35B-A3B
  (30 of 40), Q38FN (36 of 48), GLM53F (34 of 45), DSV4F, DSV41F. A dense row is not.
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
| Decode (40 tok/s floor) | 309 | **108-238**, mechanism-dependent |

**Decode PROBABLY binds before cache, and the study should stop claiming
otherwise -- but "overturned" is stronger than this measurement supports.**
Two of the three pricings invert the order and one does not, and they differ
only in a mechanism no black-box measurement here could resolve (sec 3). What
the measurement does establish unconditionally is that the decode ceiling
falls a long way from 309 and that per-user speed near production batch sizes
is a third of what the study predicted. Every hybrid model in the study —
`35B-A3B` (30 DeltaNet layers), `Q38FN` (36), `GLM53F` (34 KDA), `DSV4F`, `DSV41F` —
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

## 8. Held batches to n = 96: the latency reading holds (added 2026-09-18)

*Same class of deployment as § 1 (27B FP8, 4×H200 TP4, vLLM 0.25.1), now at
`num_speculative_tokens = 3`, with the endpoint to itself for the run. The
anonymity rule at the top of this note applies.*

§ 3 left the decode ceiling unidentified because the sweep stopped at n = 25,
inside the regime where weights are most of the step. § 4.3 asked for a
spec-off / k-sweep to choose between readings A and B. A second route needs no
restart: hold batches large enough that KV dominates, at more than one context
length, so per-sequence and per-token costs separate.

### 8.1 Hypotheses, stated before the run

At n = 90 sequences of ~100k context (≈ 295 GB of KV per step) the three
pricings of § 3 predict, all anchored to the same n = 1 step:

| pricing | step at n = 90, 100k | decode ceiling (§ 3) |
|---|---|---|
| single whole-ledger constant (shipped) | ~97 ms | 108 |
| reading B, fixed latency + KV at its measured rate | ~71 ms | 154 |
| reading A, weights ×(1+k), one higher efficiency | ~50 ms | 238 |

### 8.2 Method

`scripts/decode_probe.py --arm A --ks 1,8,32,64,90` at a 32k and a 100k
shared prefix, then `--ks 72,80,96` at 64k: 13 plateaus of 40 s. Two
independent clocks per plateau: step counts from `/metrics`
(`iteration_tokens_total`), and SSE events per stream per second at the client
(one event is one scheduler step for that stream). They agree to 0.1 ms on all
13, and `tokens per step / (n × (1 + k))` reads 0.99: a step is a forward pass.

One harness lesson, because it cost the first run: **a cancelled stream is not
a cancelled request.** Behind a proxy that does not forward the disconnect, the
engine generates every remaining token; the "k = 64" plateau of the first
attempt ran at 96 sequences and k = 90 could only queue. The probe now waits
for an idle server before each plateau and sizes outputs so streams end by
themselves.

### 8.3 Results

| n | step, ~37k ctx | step, ~66k ctx | step, ~105k ctx |
|---|---|---|---|
| 1 | 8.6 ms | | 10.2 ms |
| 8 | 11.1 | | 14.4 |
| 32 | 19.7 | | 31.7 |
| 64 | 31.3 | | 56.0 |
| 72 | | 70.8 | |
| 80 | | 73.0 | |
| 90 | 48.3 | | 117 |
| 96 | | 71.6 | |

Up to n = 64 an ordinary least-squares plane fits the eight plateaus to
**0.6 ms rms**:

    t_step = 8.3 ms + 0.14 ms × n + 5.9 ms × ΣL [Mtok]

i.e. KV read at **5.6 TB/s** (0.29 of the advertised aggregate; § 2's fixed-n
slope read 4.7–4.9 at n = 4), a per-sequence cost the study has no term for,
and a fixed cost. The single-constant fit of § 1, run on the same log, returns
`t0 = −5 ms` again. That form is refuted, twice, by its own intercept.

### 8.4 The three legs are physical, and two of them were already measured

    t_step = bytes / (η × BW)  +  n × (1 + k) × t_token  +  t_fixed

- **Per-sequence leg = the speculative verify, priced as compute.** Each
  sequence has `1 + k` positions verified per step: `1 + k` tokens of GEMM
  arithmetic the byte ledger never charged. Priced at the per-token time
  PREFILL achieves on the same machine (`2 × params / (peak × MFU)`, ~24 µs at
  the MFU measured the same night, `research/prefill.md` #1), four positions
  cost ~0.10 ms; the recurrent-state read adds ~0.03–0.05 ms. Predicted
  **0.13–0.15 ms, fitted 0.14**, from a measurement the fit never saw.
- **One bandwidth efficiency for every byte.** With the compute leg present,
  weights and KV stream at the same η = 0.29 (0.36 in the model convention):
  30.9 GB of weights cost 5.5 ms of the fixed term. This is § 4.3's parsimony
  argument for reading B, now with the second efficiency gone rather than
  merely unneeded.
- **Fixed leg = what bytes and compute leave over**, 2.7 ms: a TP4 group pays
  two all-reduces per layer whatever the batch (128 × ~20 µs is ~2.6 ms), and
  the draft head runs k extra passes. This split is a budget, not a
  measurement; ±2 ms can move between "weights" and "collectives + draft".

Evaluated from those physical terms — not from the fit — the form reproduces
**seven of the eight** n ≤ 64 plateaus within **5%**. The eighth, n = 1 at
~110k context, comes out ~12% low (9.0 ms against 10.2): a single long sequence
reads its cache less efficiently than a batch does, and the form has no term
for that.

### 8.5 What is NOT explained

Between 64 and 72 sequences (256 → 288 tokens per pass at k = 3) the step
jumps 1.2–1.6× above the line, by ~0.4 ms per ktok of per-stream context, and
is then **flat in batch size to 96**. CUDA graphs cover these sizes. A
bandwidth limit would bend smoothly and keep rising with n; a step at a
power-of-two token count that then stops depending on n looks like a kernel or
configuration boundary. Not identified from outside the pod, and **not
modelled**: one deployment, mechanism unknown. A depth-2 run would show whether
it moves to 85 sequences (256 / 3).

### 8.6 Outcomes

- § 4.3 is settled for **reading B** in kind: fixed latency, one efficiency.
  Reading A and the fold both misfit.
- The decode ceiling at the measured workload (61k mean context, 40 tok/s,
  2.6–2.9 accepted tokens per step): **115–130 in the linear regime**; on this
  deployment the unexplained step puts 72–96 sequences at 37–41 tok/s, so 64 is
  safe and the engine's own `max_num_seqs = 96` is the hard cap
  (`deployment.max_num_seqs`). The shipped 100 is about right here for the
  wrong reason.
- Acceptance during the sweep read 3.8–3.9 tokens per step: that is the
  probe's repetitive filler, not a workload figure. § 2's 2.94 stands.
- KV pool: 12.86 M tokens, **0.92×** `kv_pool_tokens()` (0.95× in § 2).

### 8.7 What changed in the model, and what did not

**Opt-in, not the default.** `model.DecodeLatency` and
`[calibration] decode_pricing = "latency"` price decode with the three legs
(`decode_bw_eff` 0.36 model-convention, `decode_fixed_ms` 2.7, `spec_tokens` 3,
the compute leg at the configured `mfu`). `MBU_DEFAULT` and every published
decode figure are untouched: the constants come from ONE model on ONE machine
at ONE speculative depth, and the explorer does not carry the form yet.

Three predictions that would falsify it, none needing new hardware:

1. **Speculation off**: the per-sequence leg falls to a quarter, the fixed leg
   loses the draft's share.
2. **TP2 instead of TP4**: the collective share of the fixed leg halves while
   the weight read doubles.
3. **Any deployment's prefill rate predicts its decode per-sequence cost**
   with no decode measurement.

### 8.8 Limitations

- One deployment, one model, one vLLM version, one evening.
- The fixed leg's split is a budget (§ 8.4).
- The probe's prompts are random text and its outputs forced (`ignore_eos`):
  only step TIME is read from them, never acceptance.
- The shared-prefix trick stores one prefix and reads it n times; § 2's arm C
  showed no cascade attention at n = 8, and nothing re-checked it at n = 90.
- Followers of a shared-prefix rung are not free on a hybrid: each recomputed
  ~1–2k tokens on admission (cache alignment), so the ramp before a large
  plateau is a real prefill load.

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
