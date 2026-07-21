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
σ 0.81); subagents ~ log-normal(median 8k, σ 0.9); **1 subagent per 10 requests**
(`r = 0.1`); 15k system prompt; `f = 1%`; 180k `max_seq_len` cap.

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

<!-- RESULTS:constants -->
_The 27B row is taken directly from the baseline study. The 35B-A3B row is
finalized from the architecture research summarized in
[`research/model_35ba3b.md`](../research/model_35ba3b.md); until confirmed it is
marked provisional in `scenario_model.py`._

## Scenario results

### 1. Warm capacity by model × topology

![Warm capacity by model and topology](../figures/scenario_capacity.png)

<!-- RESULTS:capacity -->

### 2. System-prompt size — the 3k win

![System-prompt sweep](../figures/scenario_sysprompt.png)

<!-- RESULTS:sysprompt -->

### 3. `max_num_seqs` decode tradeoff (35B-A3B)

![max_num_seqs tradeoff](../figures/scenario_mns.png)

<!-- RESULTS:mns -->

### 4. Subagents and cache invalidation

![Subagent ratio and invalidation](../figures/scenario_subagent_invalidation.png)

<!-- RESULTS:subinv -->

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
