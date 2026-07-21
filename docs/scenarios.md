# Extended scenarios: 2×H200, Qwen3.6-35B-A3B, subagents, prompt size, cache invalidation

**Decision question.** How many agentic-coding users can we serve *comfortably* per
hardware / vLLM configuration — where "comfortably" means a returning user's next
request hits the warm prefix cache (TTFT of seconds, not a full re-prefill) **and**
their decode speed stays above a 40 tok/s comfort floor (20 tok/s as the hard floor)?

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
   The baseline scripts omitted this; it costs ~10–20% of warm capacity.
3. **Recurrent-state bandwidth in decode.** Each decode step reads *and writes* every
   active sequence's DeltaNet state (2 × state × n bytes). Negligible at the
   baseline's `max_num_seqs = 6` (<2% of step bytes), no longer negligible at 64+.
4. **Expert-union bracketing.** MoE weight-read growth is reported under two models
   that bracket reality: a **linear no-overlap bound** `min(n·w_pertok, w_total)`
   (conservative, the planning default) and the **expected union under uniform
   routing** `w_total·(1−(1−8/256)^n)` (optimistic — real routing is correlated).

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
  capacity roughly linearly and caps the warm-hit rate at **1 − f**; below ~5% it is
  second-order.
- **H7 — Binding constraint.** For this workload, **warm-cache capacity binds before
  decode bandwidth**: the number of users whose sessions fit warm in the pool is
  smaller than the number of concurrent decoders the bandwidth could carry at
  40 tok/s.

## Model

### Memory / capacity

The KV pool is derived from a transparent budget, **calibrated** so that
1×H200 + 27B reproduces the baseline's *measured* 2.77M-token pool (this fixes the
per-GPU activation/workspace reserve at ≈ 18.0 GiB):

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
warm sessions across two caches — **less than TP2's 678 in one cache** — and needs
sticky routing. With 600 GB of CPU offload the 35B-A3B reaches **~2,400** warm
sessions on 1×H200 (~2,780 on TP2).

### 2. System-prompt size — a two-sided tradeoff (H4)

![System-prompt sweep](../figures/scenario_sysprompt.png)

Warm p50 at 3k → 15k → 30k shared prefix (35B-A3B): **209 → 280 → 399** (1×H200),
**506 → 678 → 964** (TP2). The naïve "a 30k system prompt wastes cache" is
**backwards for warm capacity** — the prefix is stored once and every session dedups
against it.

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

The union-model bracket is ≤ 15% and closes past the linear-saturation kink at
n = 32 (all 256 experts active), so the conservative bound is tight where decisions
are made. Because decode reads only ~3B active parameters, per-user speed stays above
the 40 tok/s comfort floor to **mns ≈ 355** (1×H200) and **≥ 600** (TP2) — far beyond
what the pool can hold warm (H7). **TP2 ≈ 1.8× per-user speed** at equal mns; **DP2
doubles aggregate** (16.2 vs 14.6 ktok/s at mns 64, system-wide) but leaves per-user
speed at the 1×H200 curve.

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

## Serving capacity — the decision table (H7)

The two candidate constraints per configuration, reference workload, conservative
union model:

| Config | Warm users (p50, cache bound) | v@warm: p50 tok/s if *all* warm users decode at once | mns@40 (speed bound alone) |
| --- | --- | --- | --- |
| 27B, 1×H200 | **94** | 48 | 118 |
| 27B, TP2 | **222** | 41 | 228 |
| 27B, DP2 | **2 × 94** (sticky) | 48 | 118 / replica |
| 35B-A3B, 1×H200 | **279** | 49 | 355 |
| 35B-A3B, TP2 | **677** | 41 | ≥ 600 |
| 35B-A3B, DP2 | **2 × 279** (sticky) | 49 | 355 / replica |

**In every configuration the cache binds first** (warm < mns@40), and even in the
worst case — every warm user decoding simultaneously — per-user p50 stays ≥ 41 tok/s.
So for agentic coding on this hardware:

- **Comfortable concurrent-user count ≈ warm capacity**: ~95 (27B, 1×H200) up to
  ~680 (35B-A3B, TP2) per node-pair, before CPU offload.
- Real duty cycles < 100% don't raise these numbers — idle users still occupy cache.
  What raises them: CPU offload (600 GB ≈ **2,400–2,800** warm 35B-A3B sessions),
  a bigger truly-shared prefix, or more subagent-like (short) traffic.
- Oversubscribing beyond warm capacity degrades gracefully into cold-TTFT churn (LRU
  eviction), not into slow decode — the baseline's N=10 cyclic-eviction collapse is
  the failure mode to watch.

**Suggested vLLM setup** (per the model above): `--kv-cache-dtype fp8_e4m3`;
`--enable-prefix-caching`; `max_seq_len` 180k (caps worst-case cold prefill and pool
hogging); `max_num_seqs` can safely sit at 2–4× the expected concurrent decoders
(speed headroom is large — the risk is preemption/thrash if active KV outgrows the
pool, not slowness); TP2 (`--tensor-parallel-size 2`) preferred over two DP replicas
for this workload; if DP, make routing session-sticky.

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

1. **The 2.77M-token anchor carries everything.** Every pool is derived from one
   measured point (1×H200 + 27B). If that measurement's error bars ([1139k, 1399k]
   at FP16, ×2 for FP8) shift, all capacities shift proportionally.
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
6. **Expert-union bracket, not measurement.** Linear vs coverage differ ≤ 15% (mostly
   below n = 32); real routing correlation lands between them.
7. **TP haircut fixed at 0.90.** Real TP2 efficiency depends on kernel overlap and
   interconnect; 0.85–0.95 moves TP2 speeds ±5%.
8. **Arrival order is an i.i.d. draw** — no burstiness, no correlated invalidation
   (e.g. a prompt-template deploy that colds *every* session at once), no diurnal
   pattern. The invalidation model is per-request, deliberately milder than a global
   flush.
9. **Warm ≠ SLA.** "Warm capacity" counts sessions resident in cache, not a latency
   guarantee; a warm hit still pays prefill for the new turn's suffix.

## Reproducibility

```
python scripts/scenario_model.py   # self-checks: calibration + config identities
python scripts/scenarios.py        # regenerates the four figures
python scripts/tables.py           # regenerates every number quoted above
```

The interactive explorer runs the same unit checks in the browser console on load.
