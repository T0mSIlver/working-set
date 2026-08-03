# Cold-spike model — queueing, bursts, and what a duty cycle hides

`research/prefill.md` § 5 opens its list of deliberate omissions with this:

> **Queueing.** `prefill_duty` reports utilisation, not latency. At duty 0.8
> the queue is already deep; the model says nothing about the TTFT
> distribution, only about where the ceiling is.

This note closes that gap, and a second one next to it — `docs/scenarios.md`
limitation 8, "arrival order is an i.i.d. draw — no burstiness, no correlated
invalidation (e.g. a prompt-template deploy that colds *every* session at
once)". Both are the same blind spot seen from two sides: § 8 prices prefill as
a **mean rate against a mean service time**, and a mean is exactly the statistic
that cannot see variance or correlation.

The headline consequence is that **f\* is not a planning number.** It is the
miss rate at which the queue has *already* diverged — the point where a
deployment can absorb no burst at all, not the point where trouble starts.

> **Status: analytic, UNVALIDATED**, and it inherits every uncertainty in
> `research/prefill.md` (MFU above all) plus new ones of its own. No queueing
> measurement exists in this repository either. The numbers rank
> configurations; they are not latency commitments.

---

## 1. The service-time distribution

The prefill server's service time `S` is the machine time one request occupies
it for. Both legs come straight from § 8's pricing, so nothing here re-derives
what the duty model already established:

| leg | service time | source |
| --- | --- | --- |
| a MISS | `(2 p L + 2 d_attn n_attn L²) / (peak × MFU)` | `cold_request_seconds` |
| a HIT | one new turn on top of `L` cached tokens — **affine in `L`** | `warm_request_seconds` |

with `L` the context length drawn from the study's lognormal user/subagent
mixture, and a request cold with probability `f = wl.invalidation`.

What is new is the **second** moment. `prefill_service_moments` samples `L` on
the same draws `context_moments` uses (same `n`, same seed), so `E[S | miss]`
and `E[S | hit]` reproduce § 8's numbers *exactly* rather than approximately —
the spike model extends the duty model instead of quietly re-deriving it. Then:

```
E[S]   = f E[S|miss] + (1-f) E[S|hit]           (= the duty cycle's mean)
E[S²]  = f E[S²|miss] + (1-f) E[S²|hit]         (new)
```

Mixing analytically at `f` rather than over a cold *subsample* matters: `is_cold`
is drawn independently of `L` in `Workload.sample`, so the analytic mix is exact
and spares the quadratic tail the sampling noise of a 1%-of-`n` subsample.

**The result that drives everything downstream:** squared coefficient of
variation `cv² = E[S²]/E[S]² − 1` lands at **5.5** (27B) to **8.3** (GLM-5.2).
An exponential service time would sit at 1. Service runs as `L²` on a lognormal
`L`, so the tail is heavy in a way the mean cannot show — and the P–K wait below
is proportional to `1 + cv²`.

---

## 2. The queue: M/G/1, and a bracket instead of a guess

**Arrivals: Poisson.** MEDIUM confidence, and deliberately the *mild* end —
real agentic traffic is burstier than Poisson (retry storms, fleet restarts,
a human hitting "run tests" across ten repos). That is precisely why § 4's
explicit-burst model exists alongside this one rather than instead of it.

**Waiting time: Pollaczek–Khinchine**, `E[W] = λ E[S²] / (2(1 − ρ))`, with
`ρ = λ E[S]` — the same ρ `prefill_duty` already reports. HIGH confidence *given*
the M/G/1 abstraction; it is a standard result, and the assertion suite checks
that ρ reproduces the published duty cycle exactly.

**Scheduling discipline: bracketed, not chosen.** vLLM is neither textbook
discipline, so the model reports both ends:

| discipline | TTFT | what it captures |
| --- | --- | --- |
| FCFS | `E[W] + E[S | class]` | one shared wait: a 2k-token turn queues behind a 180k re-prefill (**the convoy effect**). Sensitive to `E[S²]`. |
| processor sharing | `E[S | class] / (1 − ρ)` | chunked prefill time-shares admitted requests. **Insensitive to the distribution** — it never sees `E[S²]` at all. |

**Neither end is uniformly optimistic, and which is which flips by class.** PS
bills every request in proportion to its own size, so it is *dearer* for the
long misses and far *cheaper* for the short hits; FCFS bills one shared wait,
which the short hits cannot amortise. On the 27B/TP2 at f = 10% a miss waits
2.70 s (FCFS) vs 3.08 s (PS) — a 14% spread — while a hit waits 1,209 ms vs
160 ms, a **7.6×** spread. Reporting the pair per class is the honest move;
picking one would be a claim this study cannot support.

The truth sits between them: vLLM admits in arrival order but runs several
admitted prefills concurrently, bounded by `max_num_batched_tokens`.

**The planning number** falls out as `sla_miss_rate`: the largest `f` whose mean
TTFT still meets a latency budget. It always binds before `f*` does, and the gap
is pure queueing — at the SLA-limited miss rate the duty cycle still reads
**76–93%** on every binding configuration. A dashboard showing 80% utilisation is
showing a deployment whose latency has already gone.

---

## 3. Why hits pay the miss tax

The convoy column above is the sharpest thing in this note, and it is worth
stating on its own because it sharpens an existing limitation rather than
adding a new one.

`docs/scenarios.md` limitation 9 says "warm ≠ SLA: a warm hit still pays prefill
for the new turn's suffix". True, and § 8 priced that suffix — ~81 ms on the
27B/TP2. This note adds the larger term: under FCFS the same hit **waits behind
whatever misses are in front of it**, which at f = 10% is 15× its own service
time and at f = 20% is 74×. The cache-miss rate is not only a throughput
parameter for the users who miss; it is a latency parameter for the users who
*hit*. That is a different claim from "a hit is cheap, not free", and a worse one.

---

## 4. The burst: deterministic fluid drain

A spike of `B` **simultaneous** misses (the correlated invalidation limitation 8
excludes) is modelled as a fluid backlog draining against the standing load:

```
T_drain = B × E[S | miss] / (1 − ρ)
B*      = SLA × (1 − ρ) / E[S | miss]        (invert for the tolerance)
```

MEDIUM confidence. The `(1 − ρ)` denominator is the standard fluid argument —
the standing traffic keeps arriving during the drain, so only that fraction of
the server is free to work the backlog off.

The burst's **last** request sees `T_drain` under either discipline (FCFS serves
it last; PS finishes the whole burst together), which is why `B*` is
discipline-free even though § 2's TTFT is not. Their *mean* TTFT does differ:
~`T_drain / 2` under FCFS, ~`T_drain` under PS.

`B*` is **linear in the SLA**, so a different latency budget rescales every row
and moves no ranking — the reason the reference 10 s figure can be quoted
without much anxiety about the exact budget.

**Why the MoE gap compounds.** Two factors set `B*`, and on a MoE they move the
same way, because they are the same property seen twice: few active parameters
shrink `E[S | miss]`, and the cheap warm turns that follow from the same
property leave ρ low, widening the headroom the burst drains into. So the
spike-tolerance ratio *exceeds* the raw prefill-speed ratio — 7.2× against 5.9×
on TP2, and 8.8× against 5.9× on a single H200. The compounding is **larger on
the tighter machine**, which is the opposite of how most advantages behave.

---

## 5. The global flush, and why the drain figure is a floor

A prompt-template deploy or a cache wipe colds the *whole resident population*
at once — `B` = warm sessions, not a handful. Two things happen that the fluid
formula above understates:

1. **The MoE advantage stops compounding and starts cancelling.** The MoE holds
   a 3.3× larger warm population precisely because its KV is cheap, so it has
   more sessions to re-prefill. Capacity and prefill speed pull opposite ways,
   and the 7–9× per-request gap shrinks to **~2.2–2.7×**.
2. **The machine is at f = 100% until sessions re-warm.** The standing traffic
   is *also* all-cold during recovery, so `ρ` is not its steady-state value.
   The relevant question becomes whether an all-cold stream is servable at all:
   `all-cold duty = λ × E[S | miss]`, which exceeds 1 exactly when `f* < 100%`.

That second point re-reads an existing number in units that show what it means.
`f* > 100%` has always been reported as "prefill never binds at this rate"; what
it *is* is **"this configuration survives a global cache flush"**. At the
reference 2.13 req/s only the 35B-A3B on TP2 (56% all-cold duty) and on 2×B300
(25%) clear that bar. Every dense configuration — and the MoE on a single H200,
at 102%, marginally — enters a regime with no steady state, where recovery time
is set by admission control and load shedding rather than by drain rate.

**So the flush drain column is a FLOOR**, and on the "must shed load" rows it is
fiction: it prices the backlog while assuming the standing traffic stays at its
normal 1% miss rate, which is exactly what a flush makes untrue.

---

## 6. Token debt: what the drain costs the innocent

§ 8's ITL spike prices **one** forward pass — a chunk lands in the batch and
every concurrent decoder waits a prefill instead of a decode step. During a
drain the scheduler has a chunk to place in *every* pass, so that spike is not a
blip but the steady state for `T_drain` seconds:

```
tokens lost per user = T_drain × (1/ITL_normal − 1/ITL_mixed)
```

This is the metric that makes a cold spike legible on a latency dashboard rather
than in a duty cycle: a 32-miss spike on the 27B/TP2 costs each of 64 warm users
~4,300 output tokens across a 63-second plateau; the same spike on the 35B-A3B
TP2 costs ~1,140 tokens across 8.8 seconds.

---

## 7. Known weaknesses

1. **Poisson arrivals.** MEDIUM at best. Real agentic traffic correlates —
   which understates § 2's queueing. § 4's burst model is the mitigation, not a
   proof that § 2 is safe.
2. **The single-server abstraction.** One replica *group* is one queue; vLLM's
   scheduler admits multiple prefills concurrently and interleaves them with
   decode, which is what the FCFS/PS bracket is for. A real trace would land
   inside the bracket, but nothing here proves it lands in the middle.
3. **DP is not a single queue.** A DP deployment has `topo.replicas` queues, and
   a burst spreads across them only as well as the router balances it — with
   sticky routing (which H3 recommends for cache reasons) it may not spread at
   all. Every figure in this note is **per replica group**; treating a DP*n*
   deployment as *n* × `B*` assumes a balance that the caching strategy actively
   works against.
4. **Mean TTFT, not a percentile.** `sla_miss_rate` solves against the *mean*.
   A p95 budget binds at a lower miss rate still, and the heavy tail (cv² of
   5.5–8.3) means the gap between mean and p95 is large here. Every f_sla in
   this study is therefore an **upper** bound on the SLA-limited miss rate.
5. **No admission control, priority, or preemption.** Real servers shed load,
   prioritise short requests, and preempt — all of which change the outcome of
   exactly the events this note models. § 5's "must shed load" verdict is a
   diagnosis, not a simulation of the shedding.
6. **The drain ignores the decode batch it is degrading.** Each forward pass
   during a drain carries decode work too, stretching the drain by
   `t_decode / t_chunk` — 1–3% at reference settings. Small, and in the
   conservative direction (a longer real drain means a *smaller* real `B*`).
7. **Preemption/recompute is unpriced** — inherited from `research/prefill.md`
   § 5, and it bites hardest exactly during a spike, when the KV pool is
   fullest.

Weaknesses 1, 5, 6 and 7 all make the real machine **worse** than this model.
Weakness 4 does too. Weakness 3 can cut either way depending on routing. The
cold-spike figures are therefore optimistic bounds, in the same direction as
§ 8's thrash figures — with the same flagged exception, NVFP4 configurations,
whose FP8-rate pricing has unknown error direction (`research/prefill.md` § 1).

---

## 8. What would settle it

The same single measurement `research/prefill.md` asks for would anchor the
service-time scale here. Beyond that, one specific experiment settles the part
this note actually adds: **replay a burst.** Hold a warm population, invalidate
`B` sessions at once, and record TTFT for the burst and ITL for the bystanders.
`T_drain`, the convoy tax on hits, and the token debt are all directly
observable in that trace, and none of the three needs new hardware.
