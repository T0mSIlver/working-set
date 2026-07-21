# Extended scenarios: 2×H200, MoE, subagents, prompt size, cache invalidation

This extends the [baseline study](writeup.md) along five axes, keeping the **same
methodology** (transparent memory model + Monte-Carlo pool fill + a
bandwidth-bound decode model). All figures come from a single shared model,
[`scripts/scenario_model.py`](../scripts/scenario_model.py); the interactive
explorer ([`interactive/index.html`](../interactive/index.html)) mirrors the same
math so you can sweep the whole decision space live.

## What changed vs the baseline

| Axis | Baseline | Extended |
| --- | --- | --- |
| GPUs | 1×H200 | + 2×H200 **tensor-parallel (TP2)** and **data-parallel (DP2)** |
| Model | Qwen 3.6 27B (dense) | + Qwen 3.6 **35B-A3B** (MoE, ~3B active) |
| Workload | single user class | + **subagent** class (own log-normal, mixed at ratio *r*) |
| System prompt | 15k shared prefix | swept: **3k / 15k / 30k** |
| Caching | full hit / full miss | + **invalidation rate *f*** (requests that match no KV) |

## Hypotheses (the decision space)

We deliberately fix a lot of parameters so the decision space is well-defined.
The load-bearing modelling choices — flagged because they change the numbers
materially:

1. **2×H200 topology.** We model **both**:
   - **TP2** — the two GPUs act as one engine. Weights are *sharded* (counted
     once), so the KV pool grows by **more than 2×**; both weights and KV are
     read in parallel, so effective decode bandwidth ≈ `2 × HBM_BW × 0.90`
     (10% tensor-parallel comm haircut). **One shared prefix cache.**
   - **DP2** — two independent replicas. Aggregate throughput ≈ 2×, but the
     prompt cache is **per-replica**: a returning session is only warm if it is
     routed back to the replica that served it (sticky routing). Per-cache warm
     capacity stays at the 1×H200 value.
2. **35B-A3B is MoE.** Decode is bandwidth-bound on the **active** weights only
   (~3B), not all 35B — but every expert must stay **resident** in VRAM. As the
   decode batch grows, the union of activated experts grows too, so weight bytes
   read per step scale as `w_decode_shared + min(n·w_route_pertok, w_route_total)`
   — cheap at low concurrency, saturating at "all experts" at high concurrency.
3. **Subagents have their own prefix** by default (a distinct system prompt, so a
   second shared block sits in cache). Toggleable to "shares the user prefix".
4. **Cache invalidation = an always-cold fraction.** A fraction *f* of requests
   match **no** cached KV (not even the shared prefix): they pay full cold
   prefill, occupy their full length in the pool transiently, and **never** count
   as warm/reusable. This both churns the pool (fewer warm sessions) and caps the
   achievable warm-hit rate at `1 − f`. (It is *not* modelled as a global flush
   that evicts other sessions — that is a harsher variant we did not adopt.)

Reference workload used for the static figures: users ~ log-normal(median 31k,
σ 0.81) behind a **15k** system prompt; subagents ~ log-normal(median 8k, σ 0.9)
behind a leaner **3k** separate prefix; **1 subagent per 10 requests** (`r = 0.1`);
`f = 1%` invalidation; 180k `max_seq_len` cap. A prompt always contains at least
its own prefix (lengths are floored there).

## Model

### Memory / capacity

The KV pool is derived from a transparent budget, **calibrated** so that
1×H200 + 27B reproduces the baseline's *measured* 2.77M-token pool (this fixes the
per-GPU activation/workspace reserve at ≈ 18 GiB):

```
ACT_RESERVE = VRAM_per_GPU − W_resident(27B) − 2.77e6 · KV_bpt(27B)

pool_tokens(single | dp-replica) = (VRAM − W_resident − ACT_RESERVE) / KV_bpt
pool_tokens(tp2)                 = (2·VRAM − W_resident − 2·ACT_RESERVE) / KV_bpt
```

Warm-capacity Monte Carlo (one cache): reserve the distinct shared-prefix blocks
once (user prefix, and the subagent prefix unless shared); fill the remaining
budget with each session's **unique** tokens `max(clip(L, cap) − prefix, 0)` in
arrival order; a **cold** session contributes its *full* length and is subtracted
from the warm count. CPU offload adds sessions at `unique·KV_bpt + DeltaNet_state`
bytes each until the RAM buffer is full. We report p5/p50/p95 over the draws.

### Decode speed (bandwidth-bound, MoE-aware)

```
step_bytes(n) = w_decode(n) + (Σ active-context tokens)·KV_bpt
w_decode(n)   = w_decode_shared + min(n·w_route_pertok, w_route_total)   # dense: = W_resident
per_user_tok/s = MTP · BW_eff / step_bytes(n)
aggregate      = n · per_user · replicas                                 # DP2: replicas = 2
```

`BW_eff` is `HBM_BW` for single / DP-replica and `2·HBM_BW·0.90` for TP2.

### Model constants

| Constant (FP8) | 27B (dense) | 35B-A3B (MoE) |
| --- | --- | --- |
| KV bytes / token | 32 KiB (16 attn × 4 KV × 256 × 2) | **12 KiB** (12 attn × 2 KV × 256 × 2) |
| DeltaNet state / session | 75 MiB | 72 MiB |
| Resident weights | 28.8 GiB | 32.6 GiB (35 GB) |
| Weight bytes read / decode step | 28.8 GiB (all) | 1.49 GiB shared + up to 30.7 GiB routed |
| Active weights / token | — (dense) | ~3.0 GB (1.49 + 1.51) |
| KV pool, 1×H200 | 2.77M tok | **7.05M tok** |
| KV pool, 2×H200 TP2 | 6.48M tok | **16.96M tok** |
| MTP decode speedup | 1.7× | 1.7× |

The 27B row is the baseline study's; the 35B-A3B row comes from the real
**Qwen3-Next-80B-A3B** published config, scaled (expert count only) to 35B /
~3B active — full provenance and arithmetic in
[`research/model_35ba3b.md`](../research/model_35ba3b.md). The headline: the
baseline's KV/token was **~2.7× too high** for a hybrid MoE (only 12 of 48
layers hold a growing KV cache; the 36 DeltaNet layers hold a constant state),
so the MoE's KV pool per GPU is far larger than a naïve scaling would suggest.

## Scenario results

### 1. Warm capacity by model × topology

![Warm capacity by model and topology](../figures/scenario_capacity.png)

Warm **reusable** sessions in one KV cache (p5 / **p50** / p95), reference
workload, 0 GB offload:

| | 1×H200 | 2×H200 TP2 | 2×H200 DP2 (per cache) |
| --- | --- | --- | --- |
| 27B | 82 / **103** / 126 | 212 / **242** / 275 | 82 / **103** / 126 |
| 35B-A3B | 232 / **264** / 299 | 585 / **636** / 689 | 232 / **264** / 299 |

Takeaways: (1) the MoE keeps **~2.5× more** sessions warm than 27B at equal
hardware, almost entirely because its KV/token is 12 KiB vs 32 KiB. (2) **TP2**
roughly **2.4×** the single-GPU pool — weights are sharded so they aren't
duplicated, and the freed memory becomes KV. (3) **DP2's per-cache capacity is
unchanged** from 1×H200: it doubles *aggregate serving*, but a returning session
is only warm on the replica that served it. With 600 GB CPU offload the 35B-A3B
pool reaches ~1,900 warm sessions on a single H200 (and TP2 saturates the 600 GB
buffer).

### 2. System-prompt size — a two-sided tradeoff

![System-prompt sweep](../figures/scenario_sysprompt.png)

The naïve intuition ("a 30k system prompt wastes cache") is **backwards for
warm capacity**: because the prefix is stored *once* and every session dedups
against it, a bigger *shared* prefix leaves less unique KV per session and packs
**more** sessions warm (35B-A3B, TP2: **461 → 636 → 957** warm at 3k → 15k → 30k).

The real cost of a large system prompt is on the **miss path**: every request
that can't reuse the prefix (the invalidation fraction *f*, and every cold start)
must re-prefill the whole thing. That miss tax is `f × prefix` tokens per request
(right panel) and the per-user decode context is longer too. A lean **3k** prompt
trades some steady-state dedup for a **10× cheaper miss**, lower TTFT, and
robustness when the prefix isn't perfectly shared — which is exactly when it
matters. This holds only under near-perfect sharing; any per-user drift in that
30k erases the dedup win and turns it into 30k × N of wasted KV.

### 3. `max_num_seqs` decode tradeoff (35B-A3B)

![max_num_seqs tradeoff](../figures/scenario_mns.png)

Because MoE decode reads only ~3B active params, per-user speed stays **well
above** the 20–40 tok/s floors across the whole 1–120 range — so `max_num_seqs`
is bounded by VRAM for active KV, not by hitting a speed floor. Per-user p50
(tok/s) at a few concurrencies:

| 35B-A3B | mns 16 | mns 64 | mns 120 |
| --- | --- | --- | --- |
| 1×H200 / DP2-per-replica | 245 | 124 | 87 |
| 2×H200 TP2 | 441 | 223 | 157 |

The kink near mns ≈ 22 is the MoE **expert-union saturation**: once the batch
activates all experts (`n · w_route_pertok ≥ w_route_total`), extra concurrency
stops adding weight-read cost and only adds KV. **TP2 ≈ 1.8× per-user speed**
(parallel weight+KV reads); **DP2 leaves per-user speed at the 1×H200 curve but
doubles system aggregate** (≈15.9 vs 7.9 ktok/s at mns 64). Pick TP2 for latency,
DP2 for throughput-per-dollar — at the cost of a split cache.

### 4. Subagents and cache invalidation

![Subagent ratio and invalidation](../figures/scenario_subagent_invalidation.png)

**Subagents (left).** Because subagent prompts are shorter (median 8k vs 31k) and
carry a leaner 3k prefix, adding them *raises* the warm session **count** — at the
reference 1-per-10 ratio, 35B-A3B/TP2 holds ~636 warm; pushing to 1-per-1 reaches
~900. But that is more, smaller sessions, and the separate subagent prefix costs
one extra shared block. If subagents instead **share the user prefix** (toggle),
that block is saved.

**Invalidation (right).** A fraction *f* of unmatchable requests occupy KV
transiently but never become reusable, so warm capacity falls roughly linearly
(636 → 597 → 552 at *f* = 1% → 5% → 10%) and the achievable warm-hit rate is
capped at **1 − f**. At the assumed 1% this is a ~1.5% capacity haircut; it only
becomes a first-order effect past ~5%. (Modelled as always-cold, not as a global
flush — the harsher variant would also evict *other* sessions.)

## Bottom line

- **Go MoE.** At equal hardware the 35B-A3B holds ~2.5× more sessions warm and
  decodes far faster per token — the hybrid DeltaNet/attention layout (12 KiB/token
  KV) is the dominant lever, more than the extra GPU.
- **TP2 for latency, DP2 for throughput.** TP2 ~2.4× the KV pool and ~1.8× per-user
  speed with one shared cache; DP2 doubles aggregate serving but splits the cache,
  so it needs **sticky routing** or returning users go cold.
- **A large system prompt is not automatically waste** — if it is genuinely static
  and shared it *helps* warm capacity. Keep it lean (~3k) only to cut the
  miss/cold-start tax and to stay robust when sharing is imperfect.
- **Subagents are cheap** on capacity (short prompts); let them share the user
  prefix if their system prompt overlaps. **Invalidation** below ~5% is a
  second-order effect; above it, it caps the hit rate hard at `1 − f`.

## Interactive explorer

[`interactive/index.html`](../interactive/index.html) is a self-contained page
(no network needed) with sliders for the user/subagent log-normal parameters,
the subagent ratio, the system-prompt size, the invalidation rate, and switches
for model and topology. Every control re-runs the Monte Carlo live and updates
the distribution, warm-capacity, per-user-speed, and aggregate-throughput panels.

## What is modelled vs assumed

- **Measured / inherited:** the 1×H200+27B KV pool (2.77M tokens), FP8 KV bytes,
  HBM bandwidth, MTP speedup — all from the baseline study.
- **Derived:** the activation reserve (calibrated), and every other pool via the
  memory model above.
- **Assumed:** the TP comm haircut (0.90); the MoE expert-union weight growth; the
  35B-A3B architecture numbers (see research note); that DP routing is sticky; that
  invalidation is always-cold rather than a global flush; that arrival order is a
  random draw from the mixture (no correlation / bursts).

These are hypotheses, not measurements — but as in the baseline, they pin down a
usable decision space rather than a single point estimate.
