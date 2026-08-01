# Extended scenarios: 2×H200, Qwen3.6-35B-A3B, subagents, prompt size, cache invalidation

**Decision question.** How many agentic-coding users can we serve *comfortably* per
hardware / vLLM configuration — where "comfortably" means a returning user's next
request hits the warm prefix cache (TTFT of seconds, not a full re-prefill) **and**
their decode speed stays above a 40 tok/s hard floor (50 tok/s is comfortable)?

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
| GPU part *(2026-07)* | H200 only | + **B300** (Blackwell Ultra, 288 GB / 8 TB/s) — [§ Extension](#extension-2026-07-b300-gpus-nvfp4-weights-mistral-medium-35-glm-52) |
| Weight dtype *(2026-07)* | FP8 | + **NVFP4** (B300-only; weights, never KV) |
| Models *(2026-07)* | — | + **Mistral-Medium-3.5-128B** (dense GQA) and **GLM-5.2** (744B-A40B, MLA+DSA) |

Model refinements over the first draft of this study (each moves numbers by 10–20%):

1. **Real Qwen3.6-35B-A3B constants.** The first draft used provisional proxy
   constants; the published 35B-A3B config differs materially (40 layers, 10
   full-attention layers, 256 experts with 8 routed, vocab 248,320). See
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

pool_tokens(per replica group)   = (tp·VRAM − W_resident − tp·ACT_RESERVE) / KV_bpt
```

Weights are sharded across the `tp` GPUs of a replica group and so are counted
**once**; the activation reserve is scratch and charged **per GPU**. Substituting
`tp=1` recovers the single-GPU and DP-replica cases and `tp=2` the TP2 case, so
every number in this study is unchanged. See [§ Topology](#topology-a-dp--tp-grid).

Warm-capacity Monte Carlo (one cache): reserve the distinct shared-prefix blocks once
(user prefix, and the subagent prefix unless shared); fill the remaining budget in
arrival order with each session's **unique** tokens `max(clip(L, cap) − prefix, 0)`
**plus its recurrent-state charge** `state / KV_bpt`; a **cold** session contributes
its *full* length and is subtracted from the warm count. CPU offload adds sessions at
`unique·KV_bpt + state` bytes each until the RAM buffer is full. We report p5/p50/p95
over the draws for three counts: every warm session (`which="all"`, the storage view),
user-class sessions only (`which="user"`), and HBM-resident sessions only
(`which="gpu"` — the count decode figures are built on, since an offloaded session
cannot decode until it is restored).

### Decode speed (bandwidth-bound, MoE-aware)

```
step_bytes(n) = w_decode(n) + (Σ active-context tokens)·KV_bpt + 2·n·state
w_decode(n)   = w_shared + min(n·w_pertok, w_total)      # conservative default
              | w_shared + w_total·(1−(1−8/256)^n)       # expected-union bracket
per_user      = MTP · BW_eff / step_bytes(n)      aggregate = n · per_user · replicas
```

`BW_eff = tp · HBM_BW · tp_efficiency(tp)` — the bandwidth of **one replica
group**, i.e. `HBM_BW` at `tp=1` (single / DP replica) and `2·HBM_BW·0.90` at
`tp=2`. DP adds groups; it never widens one, so `replicas` enters only in the
aggregate. Dense: `w_decode = W_resident` (every step reads all weights).

### Topology: a DP × TP grid

The unit DP replicates is a **group of `tp` GPUs, not a GPU**. Total GPUs =
`dp · tp`; each group shards one copy of the weights, owns one independent KV
cache, and has its own bandwidth.

That distinction is invisible for the 27B and 35B-A3B, which fit in one GPU on
either part, and **load-bearing wherever a model does not** — which for the two
models added in 2026-07 is every cell below except one:

| min TP to fit FP8 weights | H200 | B300 |
| --- | --- | --- |
| Mistral-Medium-3.5-128B (133.6 GB) | **2** | 1 — *fits a single B300* |
| GLM-5.2 (755.5 GB) | **7** | **3** |

In every bolded cell, `topology("dp", n)` — *n* independent **single** GPUs — is
not a deployment that exists: no number of single GPUs ever holds the weights,
so the entire DP axis reported a 0-token pool at every *N*. (Mistral-Medium-3.5
on B300 is the exception — min TP 1, so pure DP is real there, and it appears as
the `DP8×TP1` column below.) Data parallelism in the other cells means
replicating whole **TP groups**, which `topology_grid(dp, tp)` now expresses.

These min-TP figures use the calibrated `ACT_RESERVE ≈ 18.0 GiB`, and there is
now only one such figure per configuration: the explorer's low-anchor case —
which raised the reserve to 33.0 GiB and pushed GLM-5.2 to 8 on H200 / 4 on
B300 — was retired 2026-07-29 (§ Measured cross-check). The explorer's
`minTpFor` and Python's `min_tp_for` therefore always agree.

The splits of one 8-GPU node that actually fit (FP8 weights, FP8 KV;
`system = replicas × per-group`, and only with session-sticky routing):

| Model | Part | DP8×TP1 | DP4×TP2 | DP2×TP4 | TP8 |
| --- | --- | --- | --- | --- | --- |
| Mistral-Medium-3.5 | H200 | — *(no fit)* | 0.61M ×4 = **2.44M** | 1.96M ×2 = **3.92M** | 4.66M = **4.66M** |
| Mistral-Medium-3.5 | B300 | 0.70M ×8 = **5.58M** | 2.14M ×4 = **8.55M** | 5.01M ×2 = **10.03M** | 10.77M = **10.77M** |
| GLM-5.2 | H200 | — | — | — *(needs TP7)* | 4.50M = **4.50M** |
| GLM-5.2 | B300 | — | — | 5.82M ×2 = **11.65M** | 27.25M = **27.25M** |

**Widening TP raises the system total**, because every DP group re-pays for its
own full copy of the weights. On a fixed node of *N* GPUs the total has a closed
form that makes this exact:

```
system(tp) = [ N·(V − R) − N·W / tp ] / KV_bpt        V = vram, R = reserve, W = weights
```

It depends on `tp` **only** through the `− N·W/tp` term, so it is monotone
non-decreasing always, and *strictly* increasing exactly when the weight charge
`W` is positive and numerically material. A weightless model is flat; every real
model here has material weights, so the margin scales with `W` — on 8 GPUs, TP8
beats the widest fitting DP split by **1.36×** for the 35B-A3B (35.5 GB),
**1.91×** for Mistral-Medium-3.5 (133.6 GB) and **2.34×** for GLM-5.2 (755.5 GB).

DP buys cache isolation and routing headroom, never capacity — the same
conclusion [§ 5](#5-scaling-to-n--h200) reached for the 35B-A3B, but far sharper
the heavier the weights.

**TP is bounded by the node.** TP all-reduces fire twice per layer and are
latency-critical, so they are only cheap inside the node's NVSwitch fabric.
`GPU.nvlink_domain` is **8 for both parts** — HGX H200 and the 8-GPU HGX B300
baseboard we deploy. `tp_efficiency` keeps the measured-regime `0.90` haircut per
doubling up to that boundary, then applies an additional
`CROSS_DOMAIN_EFFICIENCY = 0.65` per doubling beyond it.

> **Unmeasured.** The cross-node penalty is a deliberately pessimistic guess
> whose purpose is to stop the in-node haircut being extrapolated to multi-node
> TP, where it would badly overstate throughput. **No configuration in this study
> crosses the boundary**, so it bounds the extrapolation rather than predicting
> it. Treat any `tp > 2` figure as a projection needing measurement.

**In the explorer.** The **Split (DP × TP)** control offers exactly the legal
splits of the chosen GPU count — the divisors of *N*, so `N=8` gives
`DP8 / DP4×TP2 / DP2×TP4 / TP8` and `N=6` gives `DP6 / DP3×TP2 / DP2×TP3 / TP6`.
Every row of the table above is therefore reachable interactively. Splits whose
TP group is too small to hold the weights are struck through, and selecting one
reports the threshold ("needs TP ≥ 7 on H200") rather than a bare "does not
fit". Changing the GPU count snaps the TP width to the largest divisor that does
not exceed the current one, so shrinking *N* keeps as much TP as still divides
evenly instead of resetting to an end of the range.

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
| MTP decode speedup | 1.7× (α ≈ 47% per-draft acceptance) | 1.7× (transplanted fit) |

Both columns now come from **published configs** (the 27B's 64-layer / 16-full-attn /
4-KV-head config reproduces the baseline's 32 KiB/token exactly). The 35B-A3B's
KV/token is **3.2× smaller** than the 27B's because only 10 of its 40 layers hold a
growing KV cache. Provenance and full arithmetic:
[`research/model_35ba3b.md`](../research/model_35ba3b.md).

**MTP speedup ⇔ acceptance.** The 1.7× figure is the base speedup derived from the
baseline's measured fit. Under the MTP-2 accept-until-reject model (two draft
tokens per forward pass), `speedup = 1 + α + α²` where **α is the per-draft
acceptance rate**; 1.7× ⇔ **α ≈ 47%**. That α is the underlying quantity — the
speedup is computed from it, and a measured 87% acceptance would give ≈ 2.6×.
The explorer exposes the speedup as a continuous knob (1.0× = MTP off) and
displays the implied α next to it; in Python, `mtp_speedup(alpha)` in
`scenario_model.py` maps acceptance to speedup.

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
internal consistency, not the model. The TP2 measured cross-check below is
*not* circular: nothing about TP2 was fitted. Reference-workload numbers under
FP16 (from `tables.py`):

| FP16 KV | pool | warm p50 (user) | per-user p50 @ mns 64 |
| --- | --- | --- | --- |
| 27B, 1×H200 | 1.39M | 49 (45) | 39 tok/s |
| 27B, TP2 | 3.24M | 116 (105) | 71 tok/s |
| 35B-A3B, 1×H200 | 4.21M | 148 (134) | 90 tok/s |
| 35B-A3B, TP2 | 10.15M | 358 (326) | 162 tok/s |

Warm capacity retains slightly *more* than half its FP8 value (the per-session
state charge is dtype-independent, so it shrinks in token-equivalents as KV bytes
grow). Note the 27B on one GPU drops below the 40 tok/s hard floor at mns 64
under FP16 — the quantitative version of the baseline's "FP8 KV doubles P"
recommendation.

### Measured cross-check: 27B on 2×H200 TP2, FP16 KV (2026-07-22)

The exact configuration in the second row of the table above was brought up on
real hardware (27B, FP8 weights, FP16 KV cache, `--tensor-parallel-size 2`,
`max_seq_len` 184,320 = 180×1024). vLLM's startup log:

```
INFO vllm.v1.worker.gpu_worker    : Available KV cache memory: 110.59 GiB
INFO vllm.v1.core.kv_cache_utils  : GPU KV cache size: 3,233,564 tokens
INFO vllm.v1.core.kv_cache_utils  : Maximum concurrency for 184,320 tokens per request: 17.54x
```

Three consistency checks, all passing:

1. **Pool size.** The model predicts a TP2 FP16 pool of **3.242M tokens**
   (table above, rounded to 3.24M); vLLM reports **3,233,564** — a **−0.26%**
   difference. Equivalently, run backwards, the log implies a per-GPU
   non-KV overhead of ~18.2 GiB versus the calibrated 18.0 GiB reserve —
   the anchor-derived calibration now has an *independent* measured
   counterpart (the reserve was solved from the 1-GPU FP8 anchor; nothing
   about TP2 or FP16 was fitted).
2. **Internal arithmetic.** 3,233,564 / 184,320 = **17.54** exactly, matching
   the printed concurrency — unlike the baseline's vLLM 0.19.0 log, whose
   token count contradicted its own concurrency figure (the known
   mis-reporting bug), this log is self-consistent.
3. **Hybrid-allocator overhead.** The workers report 2 × 110.59 = 221.2 GiB
   available, but 3,233,564 tokens × 64 KiB/token accounts for only
   197.4 GiB of attention KV. The ~11% gap (23.8 GiB) is consistent with the
   hybrid allocator carving Gated-DeltaNet state pages from the same pool —
   directionally matching this study's separate per-session state charge —
   but its sizing is internal to the allocator: at 75 MiB/session it would
   cover ~325 sessions' states, far more than the 17 concurrent max-length
   sequences, suggesting a pre-allocation (e.g. sized to `max_num_seqs` and
   page granularity) rather than per-live-session accounting. The gap's
   *existence* supports charging state separately; its *magnitude* is not
   something this model predicts.

Caveats: this is a *startup-log* figure, not a fill-probe measurement like the
baseline's (no warm/cold sweep was run at this config), and it exercises the
27B path — the 35B-A3B remains unmeasured. With those caveats it independently
validates the ingredients the TP2 projections stack on the anchor: the
weights-counted-once TP arithmetic, the FP16 KV doubling, and the reserve's
*transferability* to a second GPU (the reserve's absolute magnitude is defined
by the anchor, so a globally-wrong anchor would shift both numbers together —
this check cannot catch that; only a direct FP8 re-measurement can).

#### Consequence: the low-anchor assumption is retired (2026-07-29)

Until this log existed, the anchor was the study's single largest structural
unknown, and both `tables.py`'s stacked case and the explorer's *Conservative*
toggle hedged it by re-solving the reserve from **2 × the measured FP16 lower
bound** (2.278M FP8 tokens ⇒ a 33.0 GiB per-GPU reserve). Run through the same
pool arithmetic as the log (regenerate: the "Retired assumption" section of
`tables.py`):

| 1×H200 FP8 anchor | per-GPU reserve | predicted 27B FP16 TP2 pool | vs the 3,233,564 measured |
| --- | --- | --- | --- |
| 2.770M *(in use)* | 17.98 GiB | 3.242M | **+0.26%** |
| 2.278M *(the hedge)* | 33.00 GiB | 2.750M | **−14.96%** |
| 2.762M *(implied by the log)* | 18.24 GiB | 3.234M | — |

A plausible-adverse assumption has to be one the hardware has not already ruled
out, and this one is off by fifteen percent in the direction the measurement
closes. The refutation is internal to the model's own reserve arithmetic — it
assumes the non-KV reserve transfers across KV dtype and topology, the same
assumption every projection in this study makes (see the transferability caveat
above). A reserve that were strongly dtype- or topology-dependent could in
principle resurrect a low FP8 anchor, but it would also make the central
anchor's +0.26% match a coincidence; only a direct FP8 re-measurement settles
that residual. It is gone from both the explorer and the stacked case; `tables.py`
keeps the three-way comparison above as a regression check so the refutation
stays reproducible rather than becoming folklore.

Two things this deliberately does **not** do. It does not re-anchor the model
to the measured 18.24 GiB: the 1.4% reserve gap is worth under 0.5% on every
published figure, which is not worth invalidating every number in this document
for. And it does not retire the *other* structural knobs — the recurrent-state
dtype and the deployed-weight overhead are untouched by this log (see
§ Statistical honesty).

Reference workload for all static figures: users ~ log-normal(median 31k, σ 0.81)
behind a **15k** system prompt; subagents ~ log-normal(median 8k, σ 0.9) behind a
leaner **3k** separate prefix; subagent ratio **r = 0.1**; **f = 1%**
invalidation; 180k `max_seq_len` cap; lengths floored at their own prefix.

**What σ means here.** σ is the log-normal *shape* parameter — the standard
deviation of ln(length), not of the length itself. The user-class values
(median 31k, σ 0.81) were obtained by a maximum-likelihood `lognorm.fit` on the
baseline's cleaned empirical prompt lengths (1,850 real requests; see
`scripts/real_capacity.py` and the fit overlay in the baseline write-up), so σ
is measured, not assumed (the subagent σ 0.9 *is* assumed — no subagent trace
exists yet). Read it as a multiplicative spread: ~68% of prompts land within
×/÷ e^σ ≈ 2.25 of the median (roughly 14k–70k), the p95 prompt is
e^(1.645·σ) ≈ 3.8× the median (~118k), and the *mean* sits e^(σ²/2) ≈ 1.39×
above the median (~43k). That last ratio is why σ matters for capacity: warm
capacity ≈ pool / E[unique tokens], and the mean — not the median — sets
E[unique]. Raising σ at a fixed median fattens the tail of huge, pool-hogging
sessions (partly clipped by the `max_seq_len` cap) and lowers the warm count
even though the "typical" prompt is unchanged.

**What the subagent ratio means.** `r` counts **subagent requests per user
request**, so the subagent share of the sampled mixture is r/(1+r) — r = 0.1
means 1 subagent request for every 10 user requests, i.e. ~9% of sessions;
r = 1 means half the mix. Subagents draw from their own, much shorter
log-normal and dedup against their own lean 3k prefix (unless the share-prefix
toggle points them at the user prefix), which is why raising r *increases* the
warm-session count: more, smaller sessions pack into the same pool — but each
warm subagent session is worth less than a warm user session, which is why the
decision table also reports the user-class-only count.

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
parameters, per-user speed stays above the 40 tok/s hard floor to **mns ≈ 355**
(1×H200) and **≈ 695** (TP2) — beyond what the pool can hold warm in every config,
though the TP2 margin is thin (678 warm vs 695; a few-percent shift in either number
could flip its binding constraint, unlike the ~27% margin elsewhere) (H7).
**TP2 ≈ 1.8× per-user speed** at equal mns; **DP2 has the highest system aggregate**
(16.2 ktok/s at mns 64, vs TP2's 14.6 and 1×H200's 8.1 — i.e. 2× its own replica)
but leaves per-user speed at the 1×H200 curve.

### 4. Subagents and cache invalidation (H5, H6)

![Subagent ratio and invalidation](../figures/scenario_subagent_invalidation.png)

**Subagents.** Warm p50 (35B-A3B, TP2) rises with the subagent ratio (r =
subagent requests per user request; mixture share r/(1+r)): **640 (r=0) →
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

The model takes an **arbitrary DP × TP grid** (`topology_grid(dp, tp)`, with
`topology(kind, n)` as the two single-axis edges). Warm p5, 35B-A3B, reference
workload — the DP column here is DP of **single** GPUs, which is a meaningful
comparison only because the 35B-A3B fits in one H200:

| N × H200 | TP — one shared cache (p5) | DP — system total (p5, sticky) |
| --- | --- | --- |
| 1 | 249 | 249 |
| 2 | **633** | 502 |
| 4 | **1,400** | 1,004 |
| 8 | **2,974** | 2,016 |

TP scales **superlinearly per cache** (each added GPU contributes its full VRAM
while the single weight copy amortizes) and beats DP's sticky-routed system total
at every N. The TP bandwidth haircut is assumed **0.90 per GPU-count doubling**
(0.81 at N=4, 0.73 at N=8) — beyond 2 GPUs this is a projection (see limitations);
past the node's 8 GPUs a steeper cross-node penalty applies
([§ Topology](#topology-a-dp--tp-grid)). For a model that does **not** fit one
GPU there is no DP column at these N at all, only DP × TP grids — see the
node-split table in that section.

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
1,178 (256) → 2,081 (512) → 3,873 (1,024 GiB)** — near-linear at ~3.5 sessions per
GiB (one mean session ≈ unique-KV + state ≈ 0.28 GiB). Offload is *storage*:
restore latency over PCIe is not modelled, so treat offloaded warmth as a weaker
tier than GPU-resident warmth.

Because it is storage, **offload changes no decode number**. Only sessions whose
KV is resident in HBM can decode; an offloaded one has to be restored first. The
GPU-resident count stays flat across the whole sweep (**280** at every buffer
size above), and every decode-side figure — the `v@warm` column of the decision
table, the explorer's per-user/aggregate stress tiles, chart C's capacity zone —
is computed at a concurrency taken from `warm_capacity(..., which="gpu")`, never
from the offload-inflated storage count.

### 7. Median context per request

The single workload knob with the steepest effect on capacity. Reference config
(27B dense, FP8 KV, 1×H200), sweeping the *user* prompt median while subagents
stay at their 8k median and **`max_seq_len` stays fixed at the reference 180k**;
**warm users** = the distinct-user count (`which="user"`):

| median context per request | 31k | 45k | 60k | 80k | 100k | 140k |
| --- | --- | --- | --- | --- | --- | --- |
| warm users (p50) | 86 | 56 | 42 | 33 | 28 | 23 |
| warm users (p5 — plan on this) | 69 | 45 | 34 | 27 | 23 | 19 |

Warm capacity ≈ pool / E[unique tokens per session], and across this sweep it
falls **more slowly than 1/median**: 4.5× the context (31k → 140k) costs only
3.7× the users. **That flattening is mostly the fixed 180k cap, not an economy
of scale.** The share of *user* draws truncated at the cap climbs from 1.5% at a
31k median to **37.8% at 140k**, which holds mean unique tokens per session to
104k instead of the 164k an uncapped log-normal would produce. Re-running the
sweep with the cap removed (everything else identical) makes the decline
*steeper* than 1/median instead — 5.5× fewer users for 4.5× the context, with
p5/p50 at 140k dropping from 19/23 to **9/15** —
because each session's dedup'd prefix (15k shared for user sessions, 3k for
subagents) is a fixed subtraction, so unique tokens
grow faster than the median does (6.0× for a 4.5× median). Read this row as "the
answer at a 180k cap", not as a property of the context length alone; §6's cap
sweep is the other half of the same effect.

The same cause drives the p5 column. It runs **17–20% below p50** here (ratio
0.80 at 31k drifting to 0.83 at 140k), i.e. the count distribution *tightens* as
contexts grow — but that is cap clipping removing the heavy-tail draws that
generate count variance, not a stability that survives to production. Uncapped,
the ratio moves the other way (0.78 at 31k → **0.60** at 140k): the spread
widens with the median, exactly as a log-normal should. A deployment that raises
`max_seq_len` along with its context lengths inherits the uncapped behaviour.

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
≥ 41 tok/s. Note the TP2 margin is thin (632–678 warm vs 695 — a ~2% gap at p50,
~9% at the p5 planning column), and the roofline is uncalibrated — a modelling
error of that order flips the binding constraint.

**That ordering is conditional on MTP.** Speculative decoding multiplies speed,
not memory: at a *fixed* concurrency, switching MTP off divides per-user tok/s by
exactly 1.7 and leaves warm capacity untouched. The 40 tok/s **crossing** moves
further than that — ÷1.8 to ÷2.0 across these configs — because it lands at a
lower concurrency, where the fixed per-step weight read is a larger share of the
bytes moved, so each sequence removed from the batch buys back less speed (the
dense 27B, which reads all its weights every step, moves most: ÷1.97). The
binding constraint flips in *every* configuration, not just the marginal ones:

| no MTP (mtp = 1.0) | warm p50 (users) | mns@40 with MTP → without | v@warm | binds |
| --- | --- | --- | --- | --- |
| 27B, 1×H200 | 94 (86) | 118 → **60** | 28.4 tok/s | bandwidth |
| 27B, TP2 | 222 (202) | 228 → **126** | 24.2 tok/s | bandwidth |
| 35B-A3B, 1×H200 | 280 (255) | 355 → **179** | 28.6 tok/s | bandwidth |
| 35B-A3B, TP2 | 678 (616) | 695 → **380** | 24.2 tok/s | bandwidth |

So "the cache binds before bandwidth" is a claim about a serving stack **with
working MTP**, and MTP + hybrid-model prefix caching is exactly the immature path
the conservative purchasing view excludes (§ Limitations). Without it the 27B on
one H200 supports 60 concurrent decoders at the 40 tok/s hard floor against 86
warm users — bandwidth binds, and the surplus warm capacity buys queueing headroom
rather than concurrency. Regenerate with the "Binding order WITHOUT MTP" table in
`tables.py`.

So for agentic coding on this hardware:

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

| Structural assumption flipped (one at a time) | Δ warm sessions (p50) | still live? |
| --- | --- | --- |
| +15% deployed-weight overhead | ~ −17 | yes — unmeasured on 3 of 4 models |
| fp32 (not bf16) DeltaNet state | ~ −68 | yes — dtype never observed |
| 10% (not 1%) invalidation | ~ −87 | yes — a workload input, not a constant |
| ~~anchor at 2× the measured FP16 *lower* bound (2.278M, not 2.77M)~~ | ~~−105~~ | **no — refuted 2026-07-29** |
| loss of global prefix sharing (per-tenant prefixes) | ~ −200 (proxy) | yes — a policy risk |
| FP16 KV instead of FP8 | ~ −320 | n/a — a configuration choice, not an unknown |

**Stacking** the three plausible-adverse assumptions that survive (fp32 state +
weight overhead + 10% invalidation) moves TP2 warm p5 from **632 to 483** (user
class ~439) — a ~149-session structural downside against a 46-session p5-vs-p50
cushion (regenerate: the "Structural-uncertainty stack" section of `tables.py`).
Note the p5 stack delta (−149) is smaller even than the −172 sum of the
surviving one-at-a-time rows — and those are p50 deltas, so the comparison
crosses percentiles; the like-for-like point is that at either percentile
the stack undershoots the sum: the effects interact (each assumption shrinks the pool the next one acts on, so
marginal costs shrink as the stack deepens).

The stack read **403** before the 2×H200 log; the 80-session difference is
entirely the retired low anchor (§ Measured cross-check). That is the shape of
the trust upgrade a single measurement bought: not a better central estimate —
the central case moved by nothing — but a smaller *downside*, because the worst
case the numbers had to defend against turned out to be excluded by hardware.
The two assumptions still in the stack are exactly the two the log cannot
speak to, which is why the explorer now exposes them as two separately-named
controls (**Recurrent state dtype**, **Deployed-weight overhead**) instead of
one word.
The whiskers in the figures show *packing variability*, not forecast confidence.
Until an exact-configuration **FP8** measurement re-anchors the model — the TP2
log settled the reserve, not the FP8 path or the 35B-A3B — purchasing-grade
planning should either use a stacked-conservative scenario like the 483 figure
or apply an explicit structural haircut to the central p5.

Related honesty note: the FP16-path "cross-check" (1.39M inside the measured
FP16 interval) is **partially circular** — the FP8 anchor was constructed as
roughly 2× the FP16 estimate, so halving it lands near the measurement largely by
construction. It validates internal consistency, not the model. The **TP2 FP16
startup-log cross-check** (3.23M measured vs 3.24M predicted, § Measured
cross-check) is the first *non-circular* validation: it tests the reserve, the
TP weight-sharding arithmetic and the FP16 doubling against a configuration
nothing was fitted to — though it is a log-reported figure on the 27B, not a
fill probe, and it says nothing about the 35B-A3B path.

## Planning stance (owner decisions, 2026-07-21)

Recorded so the numbers below are read the way they were decided:

1. **Measurement: deferred.** No re-anchoring measurement is scheduled yet; the
   study stays a projection-anchored **hypothesis generator and configuration
   ranker**. The single biggest trust upgrade remains one exact-configuration
   35B-A3B measurement (limitation 13).
2. **"Comfortable capacity" is defined as measured SLO capacity** — the largest
   replayed population meeting explicit p95 TTFT and inter-token-latency targets.
   This model **cannot produce that number**; until the replay exists, every
   figure here ranks configurations and bounds scenarios, and none is a
   user-count commitment.
3. **Prefix domain: global sharing** is the modelled policy (one shared user
   prefix per cache). Prerequisites this creates: byte-stable prompt versioning
   across all users, and security acceptance of cross-user prefix-cache sharing
   (no per-tenant cache salts). If either fails, subtract the ~200-session
   tenant-split proxy (limitation 15).
4. **Reporting: conservative base, visible headroom.** The purchasing-facing
   number excludes the immature serving paths — **no CPU offload, no MTP speedup,
   no N > 2 TP** — on top of the stacked-adverse structural case. The excluded
   upside stays fully explorable: the interactive explorer has an MTP-speedup
   knob (1.0× = off, up to 3.0×, with the implied per-draft acceptance shown),
   offload (0–1024 GiB), GPU count, and the two surviving structural
   assumptions as separate named controls — **Recurrent state dtype**
   (bf16/fp32) and **Deployed-weight overhead** (as-published/+15%), each
   disabled on the models where it would be a no-op. So every tradeoff is
   playable rather than hidden, and each one says what it is.

**Conservative purchasing view** (stacked structural case + immature paths off;
regenerate via the "Structural-uncertainty stack" section of `tables.py`):

| Config | Warm p5 (stacked) | user-class p5 | v@p5-warm, MTP off |
| --- | --- | --- | --- |
| 35B-A3B, 1×H200 | **184** | **167** | 36 tok/s |
| 35B-A3B, TP2 | **483** | **439** | 29 tok/s |

These are up from 144 / 403 in the pre-measurement stance, purely because the
low anchor left the stack (§ Measured cross-check) — the constants behind the
central case did not move.

Note the MTP-off stress speeds (36 and 29 tok/s) sit below the 40 tok/s hard
floor, and *lower* than the 43 / 34 tok/s quoted before: the stress point is
"every p5-warm session decoding at once", so a larger warm population is read
at a higher concurrency. Without the speculative-decoding path, holding the
floor at full warm load depends on duty cycle < 100% — a little more so now,
not less. This is the one place where retiring a pessimistic assumption makes a
readout look worse, and it is not an artefact: the extra sessions are real, and
so is the bandwidth they would share.

## Outcomes

| Hypothesis | Outcome |
| --- | --- |
| H1 MoE ≥ 2.5× warm | **Supported, ~3.0×** (280 vs 94; 678 vs 222) |
| H2 TP2 pool > 2×, speed ~1.8× | **Supported** (2.34–2.41× pool; 227 vs 126 tok/s at mns 64) |
| H3 DP2 splits the cache | **Supported**; moreover TP2 ≥ DP2 even system-wide (678 vs 560 warm) |
| H4 bigger shared prefix ⇒ more warm | **Supported** (506 → 964 at 3k → 30k, TP2) — but fragile to prefix drift |
| H5 subagents raise warm count | **Supported** (640 → 918 across r = 0 → 1) |
| H6 invalidation ≈ linear, ceiling 1 − f | **Supported** (−1.5% at f = 1%, −14% at 10%) |
| H7 cache binds before bandwidth | **Supported in all 6 configs — with MTP** (warm < mns@40; v@warm ≥ 41 tok/s). **Reversed in all 6 without it** (mns@40 falls 1.8–2.0×, e.g. 118 → 60 on the 27B / 1×H200) |

## Extension (2026-07): B300 GPUs, NVFP4 weights, Mistral-Medium-3.5, GLM-5.2

Added 2026-07-27; regenerate every number via the extension sections of
`tables.py`. Four research notes carry the provenance:
[`research/gpu_b300.md`](../research/gpu_b300.md),
[`research/nvfp4.md`](../research/nvfp4.md),
[`research/model_mistral_medium35.md`](../research/model_mistral_medium35.md),
[`research/model_glm52.md`](../research/model_glm52.md).

> **Provenance.** These notes were first researched with HuggingFace and all
> NVIDIA domains blocked at this environment's proxy, via first-party GitHub
> repos, config mirrors and cross-checked snippets. **Re-verified 2026-07-27
> against the primary sources after the block lifted**: literal `config.json`
> / quantization configs for all four models, measured per-shard safetensors
> dtype splits and `total_size` indexes for every modelled checkpoint, the
> NVIDIA product pages, a **real B300 nvidia-smi dump** (Oracle OCI), and the
> live vLLM issue/docs state. Outcome: GLM-5.2 and Mistral architecture
> constants confirmed exactly (GLM FP8/NVFP4 totals within 0.02%); three
> weight constants were corrected to measured values (27B NVFP4 24.47 →
> **21.92e9**; 35B-A3B NVFP4 22.92 → **24.13e9** — the MTP module is
> BF16-excluded; Mistral NVFP4 92.7 → **95.2e9** — the vision tower is 2.68B
> params); and the B300 reserve-transfer sensitivity was **promoted to the
> measured central case** (limitation 16). Full resolution ledgers live in
> each research note.

### Owner decisions (2026-07-27)

1. **NVFP4 is weights-only and B300-only.** vLLM removed FP4 emulation on
   pre-Blackwell parts; the remaining Hopper path is weight-only Marlin with
   a correctness bug still open as of 2026-07-27 (fix PR unmerged) — so
   H-generation NVFP4 is not modelled at all (`check_dtype_supported`
   raises). Both Qwen models, Mistral-Medium-3.5 and GLM-5.2 all have real
   NVFP4 checkpoints and are selectable.
2. **No 4-bit KV cache — an owner policy.** vLLM's `--kv-cache-dtype nvfp4`
   shipped 2026-05 (Blackwell-datacenter-only; values dequantize to FP8
   before attention) and its early crash bug is fixed, so this is no longer
   a stability constraint — the study still keeps the KV axis at FP8
   (default) / FP16 as a conservatism choice on a young path. Even NVIDIA's
   own NVFP4 weight checkpoints declare FP8 KV.
3. **GLM-5.2 refuses FP16 KV** (`kv_fp16_ok=False`): vLLM's sparse-MLA path
   asserts a quantized cache, so an FP16-KV GLM run is not a servable config.
4. **Per-model `max_seq_len` ranges.** The allowed workload cap now extends to
   **1,048,576** for the Qwens (262,144 native; 1M via YaRN rope scaling) and
   GLM-5.2 (1M native); **Mistral-Medium-3.5 hard-caps at 262,144** — the
   model (`check_cap_allowed`) and the explorer's slider both enforce it.
   Raising the cap keeps *lowering* warm capacity (the log-normal tail keeps
   gaining mass), but the effect saturates — almost all tail mass sits below
   512k: 35B-A3B TP2 warm p5/p50 goes 632/678 (180k) → 616/664 (262k) →
   607/659 (512k) → 605/658 (1M) — regenerate via the 1M cap sweep in
   `tables.py`.

### New model constants (FP8 serving checkpoints)

| Constant | Mistral-Medium-3.5-128B | GLM-5.2 |
| --- | --- | --- |
| Architecture | dense, 88 uniform full-attn GQA layers (8 KV × 128) | MoE 744B-A40B, 78 MLA+DSA layers, 256 experts / 8 routed |
| KV bytes / token (FP8) | **176 KiB** (17.6× the 35B-A3B) | **47.3 KiB** stored (MLA latent 576 B/layer + indexer keys) |
| Recurrent state | none | none |
| Resident weights | 133.6e9 B (124.4 GiB, FP8 ckpt) | 755.5e9 B (703.6 GiB, FP8 ckpt) |
| Decode read / step | 125.0e9 B (dense read) | 18.92e9 shared + up to 724.8e9 routed (kink n = 32) |
| Sparse decode | — (reads full cache) | indexer scan 2,772 B/context-token + 92 MB/seq top-2048 read |
| Speculative default | **1.0×** (no MTP module; external EAGLE unmeasured) | 1.7× (MTP module, transplanted fit) |
| Fits (FP8 weights) | TP2+ on H200; 1×B300 | 7×H200 / 4×B300 (3×B300 passes the pool arithmetic but sits under the vLLM recipe's 893 GB floor); NVFP4 from 2×B300 |

### Results — warm capacity by GPU part and weight dtype

Reference workload, FP8 KV, 0 offload (`tables.py` "B300 × weight dtype"):

| Config | FP8: pool / warm p5 / p50 | NVFP4: pool / warm p5 / p50 |
| --- | --- | --- |
| 27B, 1×B300 | 6.97M / **208** / 239 | 7.25M / **216** / 248 |
| 27B, 2×B300 TP | 14.89M / **470** / 512 | 15.16M / **480** / 524 |
| 35B-A3B, 1×B300 | 21.86M / **688** / 731 | 22.97M / **718** / 770 |
| 35B-A3B, 2×B300 TP | 47.19M / **1,509** / 1,582 | 48.30M / **1,544** / 1,613 |
| MM-3.5, 1×B300 | 0.70M / **16** / 26 | 0.91M / **24** / 33 |
| MM-3.5, 2×B300 TP | 2.14M / **61** / 79 | 2.35M / **70** / 88 |
| GLM-5.2, 4×B300 TP | 5.82M / **187** / 217 | 11.83M / **403** / 442 |
| GLM-5.2, 8×B300 TP | 27.25M / **957** / 1,023 | 33.26M / **1,177** / 1,247 |

And on H200s where the new models fit at FP8:

| Config | pool | warm p5 / p50 | v@warm-p5 (p50 tok/s, all p5-warm decoding) |
| --- | --- | --- | --- |
| MM-3.5, 2×H200 TP | 0.61M | 13 / 22 | **40 tok/s** |
| MM-3.5, 4×H200 TP | 1.96M | 54 / 73 | **30 tok/s** |
| GLM-5.2, 8×H200 TP | 4.50M | 141 / 169 | 62 tok/s |

(v@warm-p5 evaluates the decode curve at the **p5** GPU-resident warm count —
the explorer's stress point and the planning percentile; at the p50 counts
the Mistral rows read 31 / 24 tok/s.)

Observations:

1. **A 1×B300 roughly matches a 2×H200 TP2 pair** for the Qwen models (27B:
   208 vs 195 warm p5; 35B-A3B: 688 vs 632) — one part, no TP haircut, in a
   single NVLink domain. (These are the reserve-corrected B300 numbers;
   see limitation 16.)
2. **Mistral-Medium-3.5 reverses H7: bandwidth binds, not cache.** Even the
   few sessions that fit warm decode at 30–40 tok/s p50 when all active —
   **at or below the 40 tok/s hard floor** (24–31 tok/s at the p50 warm
   counts). Note this is the **no-MTP effect** the "Binding order WITHOUT
   MTP" table documents for *every* model, not a KV-bytes effect: Mistral
   simply has no MTP module, so its honest default is the mtp=1.0 column. At
   a hypothetical 1.7× EAGLE speedup the same configs hit **68 / 52 tok/s**
   and H7 holds again. What *is* Mistral-specific is the warm count — 176
   KiB/token leaves only **13–61 sessions p5** across the 2–4×H200 and
   1–2×B300 configs above, an order of magnitude under the Qwens — so it
   wants NVL-class pools or a measured EAGLE speedup before it serves this
   workload comfortably.
3. **NVFP4's capacity upside scales with how much of the checkpoint the
   experts are.** GLM-5.2's pool more than doubles on 4×B300 (5.82M →
   11.83M, warm p5 187 → 403) because routed experts are **725 GB of its
   756 GB** FP8 footprint and are the *only* tensors `nvidia/GLM-5.2-NVFP4`
   quantizes (725 → 408 GB; whole checkpoint 756 → 465 GB); the Qwen models
   gain only ~4% in warm capacity (their checkpoints keep large BF16
   shares under NVFP4).
4. **NVFP4 decode crosses over on MoE models** (35B-A3B, 1×B300 p50):
   −22% at mns 1, +5% at 4, **+29% at 16**, +25% at 64. The BF16-kept blocks
   (DeltaNet, lm_head, router) make the fixed per-step read 1.7× heavier
   while expert reads shrink 1.78× — so NVFP4 is a *throughput* upgrade, not
   a latency one. This is a **hybrid-architecture × quantizer-recipe**
   phenomenon, not an MoE law: it follows from RedHatAI's recipe keeping the
   DeltaNet blocks BF16, while NVIDIA's 27B recipe holds them at FP8
   (verified from the literal quant config) — which is why the dense hybrid
   27B shows a uniform speedup instead (limitation 17).
5. **GLM-5.2's DSA decode pricing matters and is honest**: at 8×H200,
   DSA-priced decode beats full-cache-read pricing 124 vs 116 tok/s (mns 16)
   and 62 vs 49 (mns 120) — and per-user speed is nearly *flat* beyond the
   expert-saturation kink (63 → 62 tok/s from mns 64 → 120) because the
   saturated 725 GB expert read dominates every step. The linear/coverage
   union bracket is **widest exactly at the kink** (mns 32: 63 vs 98 tok/s,
   +54%) and **narrows as the batch saturates under either model** (+14% at
   mns 64, +2% at 120) — so the flat tail is robust to the union
   assumption even though the kink region itself is not.

### New limitations 16–19 (extending the general list 1–15, which follows in § Limitations below)

16. **The B300 reserve is transferred, then corrected by measurement — but
    still not observed end-to-end.** The hardware constants themselves are
    now primary-verified (288.4e9 usable B/GPU from a real `nvidia-smi`
    dump; 8.0 TB/s from NVIDIA's own aggregates), and the cross-generation
    unit-convention mismatch is **measured, not hypothesized**: the H200
    delivers ~150.75e9 usable bytes against the 141e9 the calibration uses,
    the B300 delivers nominal bytes, so the model adds the hidden 9.75e9
    B/GPU back on the B300 (`GPU.reserve_extra`; −4.0% pool for the
    35B-A3B on 1×B300, −11.7% for GLM-5.2 FP8 on 4×B300 vs the uncorrected
    transfer — `tables.py` prints both). What remains an assumption is the
    activation/workspace reserve itself transferring across generations —
    one B300 vLLM startup log would close it.
17. **NVFP4 constants inherit checkpoint-recipe choices.** Which layers stay
    BF16 differs per quantizer (all recipes now read from the literal quant
    configs: RedHatAI keeps the 35B-A3B's DeltaNet blocks + MTP module BF16;
    NVIDIA's 27B recipe holds attention/DeltaNet at FP8 and even quantizes
    lm_head; NVIDIA's own 35B-A3B recipe differs from RedHatAI's enough to
    *flip* the shared-read-heavier result — the study models the named
    checkpoints only). The RedHatAI 35B-A3B checkpoint self-describes as an
    *early release*. Accuracy is not modelled at all — NVFP4 is treated as
    serving-viable per published ~1% deltas, unverified on this workload.
18. **GLM-5.2's warm-hit story on vLLM is less mature than the Qwens'.**
    Prefix caching over MLA latents + DSA indexer pages, IndexShare, and
    5-draft MTP are all recent paths; the study's warm/cold dichotomy
    assumes they compose. The DSA decode pricing also ignores top-k
    selection compute and gather inefficiency (roofline limitation), and
    the fp8 (576 B) KV layout is modelled — `fp8_ds_mla` would be +13%
    KV/token.
19. **Mistral-Medium-3.5 numbers assume the text-only serving path** (vision
    tower resident but never read during decode; image tokens would consume
    decoder KV like text). EAGLE-draft KV overhead (+2.3%) is ignored.

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
   worth removing when the anchor is re-measured). *Partial mitigation:* the
   TP2 FP16 startup log (§ Measured cross-check) independently lands within
   0.3% of the model's prediction, implying the anchor-derived reserve is
   about right — enough to retire the low-anchor hedge outright, but it shares
   the anchor's 27B lineage, so a direct FP8 and 35B-A3B measurement would
   still move the most.
2. **No prefill/decode interference.** The decode model prices bandwidth only; chunked
   prefill of cold 100k+ prompts steals decode bandwidth and adds TTFT queueing not
   modelled here. At high invalidation or cold-start rates this is first-order.
3. **35B-A3B is modelled, not measured.** Constants come from the published config,
   but FP8-KV support, hybrid-model prefix caching, and MTP acceptance (~1.7× is the
   *27B's* fitted speedup) on vLLM for this exact model are unverified. The whole
   study should be re-anchored with one measurement run once the model is deployed.
4. **DeltaNet state dtype assumed bf16** (31.9 MiB/session), inferred from the
   baseline's 75 MiB fitting the 27B's bf16 arithmetic. If vLLM keeps fp32 state,
   warm capacity drops ~10% (sensitivity in `tables.py`; explorer control
   **Recurrent state dtype**, disabled on Mistral-Medium-3.5 and GLM-5.2, which
   hold no recurrent state). The 2×H200 log does **not** settle this: its 23.8 GiB
   of allocator state pages would be ~325 sessions at bf16 or ~162 at fp32, and
   neither matches the 17 concurrent max-length sequences — that pool reads as a
   `max_num_seqs`-sized pre-allocation, so it cannot discriminate the dtype. With
   the anchor now measured, this is the largest remaining structural unknown.
5. **Deployed-weight overhead.** 35B-A3B resident bytes are raw param bytes; the 27B's
   stated footprint runs ~15% above raw. Applying +15% costs 6.2% of the 1×H200 pool
   (2.6% on TP2). Explorer control **Deployed-weight overhead**, offered on every
   model whose resident bytes are raw or on-disk-checkpoint figures — all but the
   27B (Mistral-Medium-3.5 and GLM-5.2 included), and disabled *on* the 27B, since
   that is where the 15% was derived from and applying it there would double-count.
   The `tables.py` stacked case stays 35B-A3B-scoped by definition. Note the log
   corroborates the 27B's as-deployed 28.8 GiB — the *basis* of the 15% — without
   saying anything about whether the ratio transfers to another architecture.
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
    counts are weaker "comfortable" claims than GPU-resident ones. They are also
    excluded from every *decode* figure — decode concurrency comes from
    `which="gpu"` (HBM-resident) counts alone — so raising the offload buffer
    never moves per-user tok/s. The corollary the model does not price: a
    workload that keeps hitting the offloaded tier pays restore bandwidth that
    competes with decode, so real per-user speed under heavy offload would be
    *worse* than the flat line here, never better.
11. **Decode is a pure HBM roofline.** Expert-dispatch overhead, attention/DeltaNet
    compute, TP collectives, and scheduler overhead are not priced, so all tok/s
    figures are upper bounds pending calibration against the real 35B-A3B.
12. **N > 2 GPUs is a projection.** The TP haircut (0.90 per GPU-count doubling) is
    an extrapolated assumption — real large-N TP depends on interconnect and
    kernel overlap, and pipeline parallelism is not modelled at all. The node
    boundary *is* now modelled (`GPU.nvlink_domain = 8` on both parts, plus a
    0.65-per-doubling cross-node penalty), but that penalty is itself an
    unmeasured pessimistic guess and nothing in the study crosses it, so it
    bounds the extrapolation rather than predicting it. **Expert parallelism is
    not modelled at all** — for GLM-5.2's 256 experts that is the axis a real
    large deployment would shard the FFN on, and the study instead prices all
    experts as one resident blob (correct for the reported configurations, and
    not correct for a rack-scale one). Likewise no attention-DP: a TP group is
    assumed to shard KV heads cleanly, which stops being true once `tp` exceeds
    the KV-head count. For DP, the shared CPU-offload buffer is split evenly
    across replicas — a simplification of real host-memory contention. Treat
    every N > 2 number as a shape, not a quote, until one multi-GPU measurement
    anchors it.
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
