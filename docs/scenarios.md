# Extended scenarios: 2×H200, Qwen3.6-35B-A3B, subagents, prompt size, cache invalidation

**Decision question.** How many agentic-coding users can we serve *comfortably* per
hardware / vLLM configuration — where "comfortably" means a returning user's next
request hits the warm prefix cache (TTFT of seconds, not a full re-prefill) **and**
their decode speed stays above a 40 tok/s comfort floor (20 tok/s as the hard floor)?

**Primary metric and working hypothesis.** The study's central simplification: **a
request served from a warm session is a request served well and comfortably**, while
a cold request must re-prefill its entire context — a burst of prefill work that
briefly steals GPU bandwidth from *every* active user (thrash). We therefore size
deployments by **how many sessions stay warm**, and we plan on the **p5** of that
count (the conservative tail: 95% of Monte-Carlo draws hold at least that many),
not the median. Decode-speed results exist mainly to *verify* that speed is not the
binding constraint — they are not the planning number.

This extends the [baseline study](writeup.md) along five axes, keeping the **same
methodology** (transparent memory model + Monte-Carlo pool fill + a bandwidth-bound
decode model). All numbers come from one shared model,
[`scripts/scenario_model.py`](../scripts/scenario_model.py); every table below
regenerates via [`scripts/tables.py`](../scripts/tables.py), and the interactive
explorer ([`interactive/index.html`](../interactive/index.html)) mirrors the same
math with live sliders.

## What changed vs the baseline

| Axis | Baseline | Extended |
| --- | --- | --- |
| GPUs | 1×H200 | + 2×H200 **tensor-parallel (TP2)** and **data-parallel (DP2)** |
| Model | Qwen3.6-27B (dense) | + **Qwen3.6-35B-A3B** (MoE, ~3B active) — published config |
| Workload | single user class | + **subagent** class (own log-normal, mixed at ratio *r*) |
| System prompt | 15k shared prefix | swept: **3k / 15k / 30k** |
| Caching | full hit / full miss | + **invalidation rate *f*** (requests that match no KV) |

Model refinements over the first draft of this study (each moves numbers by 10–20%):

1. **Real Qwen3.6-35B-A3B constants.** The first draft proxied the model with a scaled
   Qwen3-Next-80B-A3B config; the published 35B-A3B config differs materially
   (40 layers not 48, 10 full-attention layers not 12, 256 experts not 512, 8 routed
   not 10, vocab 248,320 not 151,936). See
   [`research/model_35ba3b.md`](../research/model_35ba3b.md) for the derivation —
   notably, the published config lands on ~35B total with **no interpolation**.
2. **Recurrent state is charged to the pool.** A warm hit on a hybrid model needs the
   constant Gated-DeltaNet state, not just the attention KV, so every resident session
   is charged `state / kv_bpt` token-equivalents (2.4k for 27B, 3.3k for 35B-A3B).
   The baseline scripts omitted this; it costs ~11% of warm capacity.
3. **Recurrent-state bandwidth in decode.** Each decode step reads *and writes* every
   active sequence's DeltaNet state (2 × state × n bytes). Negligible at the
   baseline's `max_num_seqs = 6` (<2% of step bytes), no longer negligible at 64+.
4. **Expert-union bracketing.** MoE weight-read growth is reported under two models:
   a **linear no-overlap bound** `min(n·w_pertok, w_total)` (a true upper bound on
   bytes — the conservative planning default) and the **expected union under
   i.i.d. uniform routing** `w_total·(1−(1−8/256)^n)` (a reference curve, not a
   bound: routing correlation typically shrinks the union further, but load
   balancing could widen it).

## Hypotheses

Stated up front, with predicted direction; outcomes are in [§ Outcomes](#outcomes).

- **H1 — MoE memory.** The hybrid DeltaNet/attention layout gives the 35B-A3B
  10 KiB/token of KV (vs 32 KiB for the 27B), so at equal hardware it keeps
  **≥ 2.5× more sessions warm** despite ~4 GiB more resident weights.
- **H2 — TP2.** Sharding weights across two GPUs frees a weight-copy's worth of HBM
  for KV: the pool grows **more than 2×**, and per-user decode speeds up ~1.8×
  (2 × bandwidth × 0.90 comm haircut).
- **H3 — DP2.** Two replicas double aggregate throughput but split the prefix cache;
  per-cache warm capacity equals 1×H200 and returning users are warm **only with
  sticky routing**.
- **H4 — System prompt.** A genuinely shared prefix is stored once and deduped by
  every session, so a *bigger* prefix **raises** warm capacity; its real cost is the
  miss path (every unmatchable request re-prefills `f × prefix` tokens).
- **H5 — Subagents.** A short-prompt request class (median 8k vs 31k) raises the warm
  session *count* at equal pool, at the cost of one extra shared prefix block.
- **H6 — Invalidation.** A fraction *f* of requests matching no cached KV reduces warm
  capacity near-linearly (≈ 1.5 × *f*) and caps the warm-hit rate at **1 − f**; at the
  assumed 1% the capacity cost is ~1.5%.
- **H7 — Binding constraint.** For this workload, **warm-cache capacity binds before
  decode bandwidth**: the number of users whose sessions fit warm in the pool is
  smaller than the number of concurrent decoders the bandwidth could carry at
  40 tok/s.

## Model

### Memory / capacity

The KV pool is derived from a transparent budget, **calibrated** so that
1×H200 + 27B reproduces the baseline's 2.77M-token FP8 pool. That anchor is itself a
**projection**, not a direct measurement: the baseline measured the FP16 pool
(P ∈ [1139k, 1399k], best estimate ~1337k tokens) and projected FP8 as ~2× plus freed
activation memory. Calibration fixes the per-GPU activation/workspace reserve at
≈ 18.0 GiB:

```
ACT_RESERVE = VRAM_per_GPU − W_resident(27B) − 2.77e6 · KV_bpt(27B)

pool_tokens(single | dp-replica) = (VRAM − W_resident − ACT_RESERVE) / KV_bpt
pool_tokens(tp2)                 = (2·VRAM − W_resident − 2·ACT_RESERVE) / KV_bpt
```

Warm-capacity Monte Carlo (one cache): reserve the distinct shared-prefix blocks once
(user prefix, and the subagent prefix unless shared); fill the remaining budget in
arrival order with each session's **unique** tokens `max(clip(L, cap) − prefix, 0)`
**plus its recurrent-state charge** `state / KV_bpt`; a **cold** session contributes
its *full* length and is subtracted from the warm count. CPU offload adds sessions at
`unique·KV_bpt + state` bytes each until the RAM buffer is full. We report p5/p50/p95
over the draws.

### Decode speed (bandwidth-bound, MoE-aware)

```
step_bytes(n) = w_decode(n) + (Σ active-context tokens)·KV_bpt + 2·n·state
w_decode(n)   = w_shared + min(n·w_pertok, w_total)      # conservative default
              | w_shared + w_total·(1−(1−8/256)^n)       # expected-union bracket
per_user      = MTP · BW_eff / step_bytes(n)      aggregate = n · per_user · replicas
```

`BW_eff` is `HBM_BW` for single / DP-replica and `2·HBM_BW·0.90` for TP2. Dense:
`w_decode = W_resident` (every step reads all weights).

### Model constants

| Constant (FP8) | Qwen3.6-27B (dense) | Qwen3.6-35B-A3B (MoE) |
| --- | --- | --- |
| KV bytes / token | 32 KiB (16 attn × 4 KV × 256 × 2) | **10 KiB** (10 attn × 2 KV × 256 × 2) |
| DeltaNet state / session (bf16) | 75 MiB | **31.9 MiB** |
| Resident weights | 28.8 GiB (as deployed) | 33.1 GiB (~35.5B params) |
| Weight bytes read / decode step | 28.8 GiB (all) | 1.81 GiB shared + up to 30.0 GiB routed |
| Active weights / token | — (dense) | ~2.95 GB (published "~3B") |
| Expert-union saturation (linear) | — | n = 32 (= 256 experts / 8 per token) |
| KV pool, 1×H200 | 2.77M tok (measured anchor) | **8.42M tok** |
| KV pool, 2×H200 TP2 | 6.48M tok | **20.30M tok** |
| MTP decode speedup | 1.7× | 1.7× |

Both columns now come from **published configs** (the 27B's 64-layer / 16-full-attn /
4-KV-head config reproduces the baseline's 32 KiB/token exactly). The 35B-A3B's
KV/token is **3.2× smaller** than the 27B's because only 10 of its 40 layers hold a
growing KV cache. Provenance and full arithmetic:
[`research/model_35ba3b.md`](../research/model_35ba3b.md).

### KV-cache dtype switch (FP8 default / FP16)

Every number in this study assumes the **FP8 KV cache** (`--kv-cache-dtype
fp8_e4m3`), as tested in the baseline. The model exposes a switch
(`with_kv_dtype(model, "fp16")` in Python; a Model-panel toggle in the explorer)
that doubles KV bytes/token — halving the pool and adding decode read cost —
while weights, the DeltaNet state, and the FP8-anchored reserve calibration stay
fixed. Consistency check: the FP16 27B/1×H200 pool comes out at **1.39M tokens**,
inside the baseline's *measured* FP16 interval [1.14M, 1.40M]. Note this is
**partially circular** (the FP8 anchor was built as ~2× the FP16 estimate, so
halving it lands near the measurement largely by construction) — it confirms
internal consistency, not the model. Reference-workload numbers under FP16
(from `tables.py`):

| FP16 KV | pool | warm p50 (user) | per-user p50 @ mns 64 |
| --- | --- | --- | --- |
| 27B, 1×H200 | 1.39M | 49 (45) | 39 tok/s |
| 27B, TP2 | 3.24M | 116 (105) | 71 tok/s |
| 35B-A3B, 1×H200 | 4.21M | 148 (134) | 90 tok/s |
| 35B-A3B, TP2 | 10.15M | 358 (326) | 162 tok/s |

Warm capacity retains slightly *more* than half its FP8 value (the per-session
state charge is dtype-independent, so it shrinks in token-equivalents as KV bytes
grow). Note the 27B on one GPU drops below the 40 tok/s comfort floor at mns 64
under FP16 — the quantitative version of the baseline's "FP8 KV doubles P"
recommendation.

Reference workload for all static figures: users ~ log-normal(median 31k, σ 0.81)
behind a **15k** system prompt; subagents ~ log-normal(median 8k, σ 0.9) behind a
leaner **3k** separate prefix; **1 subagent per 10 requests** (r = 0.1); **f = 1%**
invalidation; 180k `max_seq_len` cap; lengths floored at their own prefix.

## Scenario results

### 1. Warm capacity by model × topology (H1, H2, H3)

![Warm capacity by model and topology](../figures/scenario_capacity.png)

Warm **reusable** sessions in one KV cache (p5 / **p50** / p95), reference workload,
0 GB offload:

| | 1×H200 | 2×H200 TP2 | 2×H200 DP2 (per cache) |
| --- | --- | --- | --- |
| 27B | 76 / **94** / 115 | 195 / **222** / 252 | 76 / **94** / 115 |
| 35B-A3B | 250 / **280** / 312 | 632 / **678** / 726 | 250 / **280** / 312 |

(1) The MoE keeps **~3.0× more** sessions warm than the 27B at equal hardware
(280 vs 94; 678 vs 222) — driven by 10 vs 32 KiB/token. (2) **TP2** gives **2.3–2.4×**
the single-GPU pool (20.30 vs 8.42M tokens): the second weight copy it avoids becomes
KV. (3) **DP2's per-cache capacity is unchanged**; system-wide it holds 2 × 280 = 560
(p50 — the p5 planning view in §5 reads 502 vs TP2's 633)
warm sessions across two caches — **less than TP2's 678 in one cache** — and needs
sticky routing. With 600 GiB of CPU offload the 35B-A3B reaches **~2,400** warm
sessions on 1×H200 (~2,780 on TP2) — but see the offload limitation: this is
*storage* capacity; restore latency over PCIe is not modelled.

### 2. System-prompt size — a two-sided tradeoff (H4)

![System-prompt sweep](../figures/scenario_sysprompt.png)

Warm p50 at a 3k → 15k → 30k *user* prefix (35B-A3B; the subagent prefix stays at
its lean 3k throughout the sweep): **209 → 280 → 399** (1×H200), **506 → 678 → 964**
(TP2). The naïve "a 30k system prompt wastes cache" is **backwards for warm
capacity** — the prefix is stored once and every session dedups against it.

The real cost is the **miss path**: every request that can't match the prefix (the
invalidation fraction *f*, plus true cold starts) re-prefills the whole thing — an
expected `f × prefix` tokens per request — and every *active* decode still reads the
full context including the prefix. A lean 3k prompt buys a 10× cheaper miss, lower
cold TTFT, and robustness when sharing is imperfect: **any per-user drift in a 30k
prefix turns the dedup win into 30k × N of duplicated KV**. Keep the prefix lean and
byte-stable; the capacity upside of a big prefix is real but fragile.

### 3. `max_num_seqs` decode tradeoff, 35B-A3B (H2, H7)

![max_num_seqs tradeoff](../figures/scenario_mns.png)

Per-user p50 tok/s (conservative linear union | expected-union bracket):

| 35B-A3B | mns 16 | mns 64 | mns 120 |
| --- | --- | --- | --- |
| 1×H200 / DP2 per replica | 319 \| 366 | 126 \| 135 | 89 \| 90 |
| 2×H200 TP2 | 574 \| 659 | 227 \| 243 | 161 \| 162 |

The gap between the two union models peaks around the linear-saturation kink
(~30% faster under the coverage model at n ≈ 32) and closes above it (+7% at
mns 64, +1% at 120), so the conservative bound is tight in the high-concurrency
region where the capacity decisions are made. Because decode reads only ~3B active
parameters, per-user speed stays above the 40 tok/s comfort floor to **mns ≈ 355**
(1×H200) and **≈ 695** (TP2) — beyond what the pool can hold warm in every config,
though the TP2 margin is thin (678 warm vs 695; a few-percent shift in either number
could flip its binding constraint, unlike the ~27% margin elsewhere) (H7).
**TP2 ≈ 1.8× per-user speed** at equal mns; **DP2 has the highest system aggregate**
(16.2 ktok/s at mns 64, vs TP2's 14.6 and 1×H200's 8.1 — i.e. 2× its own replica)
but leaves per-user speed at the 1×H200 curve.

### 4. Subagents and cache invalidation (H5, H6)

![Subagent ratio and invalidation](../figures/scenario_subagent_invalidation.png)

**Subagents.** Warm p50 (35B-A3B, TP2) rises with the subagent ratio: **640 (r=0) →
678 (r=0.1) → 802 (r=0.5) → 918 (r=1)** — shorter prompts pack more sessions, at the
cost of one extra shared prefix block (toggleable off if subagents share the user
prefix). Note these are *more, smaller* sessions: per-session value is lower.

**Invalidation.** Warm p50 falls near-linearly: **688 (f=0) → 678 (1%) → 638 (5%) →
591 (10%)** — a 1.5% haircut at the assumed 1%, ~14% at 10% — and the achievable
warm-hit rate is capped at **1 − f**. Modelled as always-cold requests, *not* as a
global flush (which would also evict other sessions — a harsher variant we did not
adopt).

### 5. Scaling to N × H200

![Warm-session scaling with hardware](../figures/scenario_scaling.png)

The model now takes an **arbitrary GPU count** with a TP/DP switch
(`topology(kind, n)`). Warm p5, 35B-A3B, reference workload:

| N × H200 | TP — one shared cache (p5) | DP — system total (p5, sticky) |
| --- | --- | --- |
| 1 | 249 | 249 |
| 2 | **633** | 502 |
| 4 | **1,400** | 1,004 |
| 8 | **2,974** | 2,016 |

TP scales **superlinearly per cache** (each added GPU contributes its full VRAM
while the single weight copy amortizes) and beats DP's sticky-routed system total
at every N. The TP bandwidth haircut is assumed **0.90 per GPU-count doubling**
(0.81 at N=4, 0.73 at N=8) — beyond 2 GPUs this is a projection, and NVLink domain
size / interconnect will decide whether large-N TP is realistic (see limitations).

### 6. The remaining knobs: max_seq_len cap and CPU offload

Both are now continuous knobs in the model and the explorer. `max_seq_len` sweep
(35B-A3B, TP2, warm p5/p50):

| cap | 60k | 120k | 180k | 262k (model max) |
| --- | --- | --- | --- | --- |
| warm p5 / p50 | 860 / 898 | 672 / 714 | 632 / 678 | 617 / 664 |

Lowering the cap **truncates the log-normal tail**: the few huge sessions that hog
the pool get clipped, so capacity *rises* as the cap falls — slowly at first
(262k → 180k: +2%) then steeply (120k → 60k: +28%), because tail mass grows fast
as the cap approaches the median. The cost is real truncation of long agentic
sessions; 180k remains the recommended balance.

CPU offload (35B-A3B, 1×H200, warm p50): **280 (0) → 504 (64 GiB) → 728 (128) →
1,178 (256) → 2,081 (512) → 3,873 (1,024 GiB)** — near-linear at ~1.75 sessions per
GiB (one mean session ≈ unique-KV + state ≈ 0.57 GiB). Offload is *storage*:
restore latency over PCIe is not modelled, so treat offloaded warmth as a weaker
tier than GPU-resident warmth.

## Why some knobs act non-linearly (or non-monotonically)

Two separate causes; distinguishing them matters when reading sweeps.

**Inherent to the metric** (real effects, reproducible):

- **Piecewise-linear kinks.** The conservative expert-union model is
  `min(n·w_pertok, w_total)` — decode cost has a hard kink at n = 32 where all 256
  experts are active; beyond it, extra concurrency adds only KV/state bytes.
- **Tail truncation (the cap).** Warm capacity responds to `max_seq_len` through
  the log-normal *tail mass* above the cap, which shrinks super-exponentially as
  the cap rises: big gains from 120k → 60k, almost nothing from 262k → 180k.
- **Floors and reservations (the prefix).** Prompts are floored at their prefix and
  dedup against it, so a bigger shared prefix *reduces* per-session unique cost
  (convex gain in capacity) while linearly growing the one-off reserved block and
  the miss tax — opposing terms with different shapes.
- **Harmonic responses.** Capacity ≈ pool / (E[unique] + state/kv_bpt): doubling KV
  bytes/token (FP16) does *not* halve capacity, because the state charge shrinks in
  token-equivalents at the same time (148 vs "half of 280" = 140).
- **Amortized fixed costs (TP scaling).** pool(N) = N·VRAM − W − N·reserve: the
  −W intercept makes per-cache capacity superlinear in N.
- **Cold sessions cost more than warm ones.** An invalidating request occupies its
  *full* length (no dedup), so capacity falls ~1.5× faster than f itself.

**Monte-Carlo artifacts** (sampling noise, not real):

- The warm count is an **order statistic of a discrete count** over heavy-tailed
  log-normal draws: finite `n_iter` leaves ±1–3 sessions of jitter on p5/p50, so a
  truly monotonic knob can look locally non-monotonic between adjacent sweep
  points (we saw 279 vs 280 across runs at different `n_iter`).
- The Python sweeps in `tables.py` call `warm_capacity(seed=0)` at every sweep
  point — **common random numbers**: neighbouring points share the same underlying
  draws, so sweep curves are smooth and differences between points are real. The
  **explorer uses an unseeded RNG**, so repeated renders of the same settings
  wobble by a few sessions; that wobble is noise, not signal.
- The flip side of a fixed seed: each Python table is **one random realization** —
  its point values carry the same ±1–3 session uncertainty even though the sweep
  *shape* is trustworthy. The p5–p95 whiskers, not the point estimates, carry the
  spread.

## Serving capacity — the decision table (H7)

The two candidate constraints per configuration, reference workload, conservative
union model. **Warm p5 is the planning column.** "Warm sessions" counts every
reusable cached session; **"warm users"** counts only user-class sessions (subagent
sessions, ~9% of the mix at r = 0.1, are excluded — a *user* corresponds to one
user-class session in this model):

| Config | **Warm p5 (plan on this)** | Warm p50 | warm **users** p5 / p50 | v@warm: p50 tok/s if *all* warm sessions decode at once | mns@40 (speed bound alone) |
| --- | --- | --- | --- | --- | --- |
| 27B, 1×H200 | **76** | 94 | **69** / 86 | 48 | 118 |
| 27B, TP2 | **195** | 222 | **177** / 202 | 41 | 228 |
| 27B, DP2 | **2 × 76** (sticky) | 2 × 94 | **2 × 69** / 2 × 86 | 48 | 118 / replica |
| 35B-A3B, 1×H200 | **250** | 280 | **228** / 255 | 49 | 355 |
| 35B-A3B, TP2 | **632** | 678 | **574** / 616 | 41 | 695 |
| 35B-A3B, DP2 | **2 × 250** (sticky) | 2 × 280 | **2 × 228** / 2 × 255 | 49 | 355 / replica |

**In every configuration the cache binds first** (warm sessions < mns@40), and even
in the worst case — every warm session decoding simultaneously — per-user p50 stays
≥ 41 tok/s. Note the TP2 margin is thin (632–678 warm vs 695), and the roofline is
uncalibrated — a few-percent modelling error flips that binding constraint. So for
agentic coding on this hardware:

- **Comfortable concurrent-user count ≈ warm *user* capacity at p5** (same
  percentile as the planning column): ~69 (27B, 1×H200) up to ~574 (35B-A3B, TP2)
  per node-pair, before CPU offload.
- Real duty cycles < 100% don't raise these numbers — idle users still occupy cache.
  What raises them: CPU offload (600 GiB ≈ **2,400–2,800** warm 35B-A3B sessions —
  as *storage*; restore latency unmodelled), a bigger truly-shared prefix, or more
  subagent-like (short) traffic.
- Oversubscribing beyond warm capacity turns the excess into cold-TTFT churn (LRU
  eviction). We have NOT verified that this is graceful under load: cold prefills
  interfere with active decodes, and the baseline's N=10 cyclic-eviction collapse
  shows the cliff-shaped failure mode. Treat the warm count as the population to
  stay under, not a soft target.

**Suggested vLLM setup** (per the model above): `--kv-cache-dtype fp8_e4m3`;
`--enable-prefix-caching`; `max_seq_len` 180k (caps worst-case cold prefill and pool
hogging); size `max_num_seqs` from the modelled speed ceilings above (mns@40) while
keeping expected active KV within the pool — the study does not model preemption, so
treat large values as needing a measurement pass; TP2 (`--tensor-parallel-size 2`)
preferred over two DP replicas for this workload; if DP, make routing session-sticky.

## Statistical honesty: Monte-Carlo spread vs structural uncertainty

The p5 framing is conservative **only within the model's fixed assumptions**: it
says that under this workload distribution, this allocator arithmetic and these
constants, 95% of random packing draws hold at least that many sessions. It is NOT
a 95%-confidence forecast of production capacity — parameter and structural
uncertainty are far larger than sampling spread. For the headline 35B-A3B TP2 case
(p5 632, p50 678; MC estimator jitter ~1–3 sessions):

| Structural assumption flipped (one at a time) | Δ warm sessions |
| --- | --- |
| +15% deployed-weight overhead | ~ −17 |
| fp32 (not bf16) DeltaNet state | ~ −68 |
| 10% (not 1%) invalidation | ~ −87 |
| anchor at 2× the measured FP16 *lower* bound (2.278M, not 2.77M) | ~ −105 |
| loss of global prefix sharing (per-tenant prefixes) | ~ −200 (proxy) |
| FP16 KV instead of FP8 | ~ −320 |

**Stacking** the first four plausible-adverse assumptions (low anchor + fp32 state
+ weight overhead + 10% invalidation) moves TP2 warm p5 from **632 to 403** (user
class ~367) — a ~230-session structural downside against a 46-session p5-vs-p50
cushion (regenerate: the "Structural-uncertainty stack" section of `tables.py`).
The whiskers in the figures show *packing variability*, not forecast confidence.
Until one exact-configuration measurement re-anchors the model, purchasing-grade
planning should either use a stacked-conservative scenario like the 403 figure or
apply an explicit structural haircut to the central p5.

Related honesty note: the FP16-path "cross-check" (1.39M inside the measured
FP16 interval) is **partially circular** — the FP8 anchor was constructed as
roughly 2× the FP16 estimate, so halving it lands near the measurement largely by
construction. It validates internal consistency, not the model.

## Outcomes

| Hypothesis | Outcome |
| --- | --- |
| H1 MoE ≥ 2.5× warm | **Supported, ~3.0×** (280 vs 94; 678 vs 222) |
| H2 TP2 pool > 2×, speed ~1.8× | **Supported** (2.34–2.41× pool; 227 vs 126 tok/s at mns 64) |
| H3 DP2 splits the cache | **Supported**; moreover TP2 ≥ DP2 even system-wide (678 vs 560 warm) |
| H4 bigger shared prefix ⇒ more warm | **Supported** (506 → 964 at 3k → 30k, TP2) — but fragile to prefix drift |
| H5 subagents raise warm count | **Supported** (640 → 918 across r = 0 → 1) |
| H6 invalidation ≈ linear, ceiling 1 − f | **Supported** (−1.5% at f = 1%, −14% at 10%) |
| H7 cache binds before bandwidth | **Supported in all 6 configs** (warm < mns@40; v@warm ≥ 41 tok/s) |

## Limitations

Ordered roughly by how much each could move the numbers:

1. **The 2.77M-token anchor carries everything — and it is a projection.** Every
   pool derives from one 1×H200 + 27B point that was itself *projected* from the
   measured FP16 pool ([1139k, 1399k], best estimate 1337k) as ~2× plus freed
   activation memory; the FP8 pool was never measured directly. Both the FP16
   error bars (±10%) and the FP8-projection assumptions propagate to every
   capacity number here. A second, smaller inconsistency: the baseline defined
   its measured P as "KV + DeltaNet states", while this study treats the anchor
   as pure KV and charges state separately — double-counting the ~5 measured
   sessions' states (~12k token-equivalents, ~0.4% of the pool; negligible but
   worth removing when the anchor is re-measured).
2. **No prefill/decode interference.** The decode model prices bandwidth only; chunked
   prefill of cold 100k+ prompts steals decode bandwidth and adds TTFT queueing not
   modelled here. At high invalidation or cold-start rates this is first-order.
3. **35B-A3B is modelled, not measured.** Constants come from the published config,
   but FP8-KV support, hybrid-model prefix caching, and MTP acceptance (~1.7× is the
   *27B's* fitted speedup) on vLLM for this exact model are unverified. The whole
   study should be re-anchored with one measurement run once the model is deployed.
4. **DeltaNet state dtype assumed bf16** (31.9 MiB/session), inferred from the
   baseline's 75 MiB fitting the 27B's bf16 arithmetic. If vLLM keeps fp32 state,
   warm capacity drops ~10% (sensitivity in `tables.py`).
5. **Deployed-weight overhead.** 35B-A3B resident bytes are raw param bytes; the 27B's
   stated footprint runs ~15% above raw. Applying +15% costs 6.2% of the 1×H200 pool
   (2.6% on TP2).
6. **Expert-union models, not measurement.** The linear bound is a true byte upper
   bound; the coverage curve is an i.i.d.-routing reference, not a lower bound. Their
   gap peaks ~30% at n ≈ 32 and closes above saturation.
7. **TP haircut fixed at 0.90.** Real TP2 efficiency depends on kernel overlap and
   interconnect; 0.85–0.95 moves TP2 speeds ±5%.
8. **Arrival order is an i.i.d. draw** — no burstiness, no correlated invalidation
   (e.g. a prompt-template deploy that colds *every* session at once), no diurnal
   pattern. The invalidation model is per-request, deliberately milder than a global
   flush.
9. **Warm ≠ SLA.** "Warm capacity" counts sessions resident in cache, not a latency
   guarantee; a warm hit still pays prefill for the new turn's suffix.
10. **CPU offload is priced as storage only.** Offloaded sessions count as warm, but
    the PCIe transfer to restore them on the next turn (≈ 0.1–0.3 s per 100k-token
    session at 32–64 GB/s, plus contention) is not modelled, so offload-inflated
    counts are weaker "comfortable" claims than GPU-resident ones.
11. **Decode is a pure HBM roofline.** Expert-dispatch overhead, attention/DeltaNet
    compute, TP collectives, and scheduler overhead are not priced, so all tok/s
    figures are upper bounds pending calibration against the real 35B-A3B.
12. **N > 2 GPUs is a projection.** The TP haircut (0.90 per GPU-count doubling) is
    an extrapolated assumption — real large-N TP depends on NVLink domain size,
    interconnect, and kernel overlap, and pipeline parallelism is not modelled at
    all. For DP, the shared CPU-offload buffer is split evenly across replicas — a
    simplification of real host-memory contention. Treat every N > 2 number as a
    shape, not a quote, until one multi-GPU measurement anchors it.
13. **vLLM's hybrid-model allocator may not match the byte model.** We price
    attention KV continuously plus one recurrent state per session; vLLM's hybrid
    (attention + GDN) cache unifies page sizes, may pad recurrent state to
    attention-page granularity, uses larger blocks (and only caches *full* blocks
    for prefix reuse), and hybrid-model prefix caching + MTP is still maturing.
    Allocator granularity could move 35B-A3B capacity more than the ±10% dtype
    sensitivity — this is the single biggest reason to re-anchor on the real model.
14. **Warm capacity is a packing limit, not a sustainable population.** The fill
    leaves no headroom for the *next* turn: sessions grow turn over turn, active
    decodes append KV, and a full pool answers growth with eviction or preemption.
    A sustainable population sits meaningfully below the packing numbers quoted
    here; a session-growth trace replay would bound the gap.
15. **Global prefix sharing is an optimistic default.** The model stores one user
    prefix for the whole cache; per-tenant prompts, template versioning, or cache
    salts (deliberate isolation) split it into many equivalence classes. A rough
    no-global-dedup proxy costs ~200 sessions on TP2. The sharing/isolation domain
    is a policy decision, not a modelling detail.

## Reproducibility

```
uv run scripts/scenario_model.py   # self-checks: calibration + config identities
uv run scripts/scenarios.py        # regenerates the four figures
uv run scripts/tables.py           # regenerates every number quoted above
```

The interactive explorer runs the calibration and published-config identity checks
in the browser console on load (a subset of the Python self-checks; the Monte-Carlo
assertions run only in Python).
