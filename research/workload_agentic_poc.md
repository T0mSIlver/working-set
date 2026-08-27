# Measured agentic workload: 7 days of a production 27B on 4×H200 TP4

*Added 2026-08-27. Source: the deployment's Grafana over 2026-08-20 →
2026-08-27. Employer infrastructure — do not cite the instance's identity or
raw dashboards in public-facing material; this note carries the anonymised
numbers, and the explorer's defaults use rounded values only.*

The study's first end-to-end contact with a production agentic workload on
the baseline hardware (Qwen3.6-27B FP8, one TP4 H200 group, vLLM,
`max_num_batched_tokens` per the deployment config, speculative decoding
enabled). Two uses: (1) measured values for workload constants the model had
assumed, (2) a capacity placement of the instance against the four ceilings.

## 1. Measured workload (7 days)

| Quantity | Value | Notes |
|---|---|---|
| Total input tokens | 2.26 B | ≈ 39,400 requests |
| Mean input tokens / request | **57.4 k** | heatmap mass 10k → 100k+; fits the study lognormal at `user_median ≈ 47,400` (σ 0.81, sub mix unchanged) |
| Mean output tokens / request | **404.1** | → `OUT_TOKENS_DEFAULT` / `AVG_OUT_TOK` updated 1,000 → 400 (was the study's one confessed guess) |
| Prefix-cache savings | **87.5%** | computed tokens ≈ 7.2k/request |
| Implied miss rate | **~9.3%** | ⚠ jointly identified with the 2,000-token warm turn (one observable, two unknowns — see §3) |
| Request rate | 0.065/s (24/7) · ~0.22/s (office hours) · ~0.47/s (15-min peak) | bursty, office-hours profile |
| TTFT p99 (mean of) | 2.68 s | queue p99 ~1 s, occasional spikes |
| TPOT p99 | 12.9 ms (~78 tok/s/stream) | above the 40 tok/s floor with ~2× margin |
| Mean prefill time / request | 159 ms | the MFU cross-check input (prefill.md #1, second calibration point) |
| Speculative decoding | acceptance 78.9%, mean accepted length 2.57 (busy periods; 27% / 1.54 incl. idle) | |

## 2. Capacity placement (scenario_model, measured workload)

`Workload(user_median=47_400)` (fits the 57.4k mean), out_tokens 404,
chunk 32,768, MFU 0.45:

| Ceiling | Max concurrent users |
|---|---|
| **Cache (binding)** | **249** (warm-user p5; p50 271; 13.9 M-token pool) |
| Decode ≥ 40 tok/s | 309 |
| Saturation | 618 |
| Latency (10 s TTFT) | 718 |

Cache < decode < saturation < latency — the study's thesis (cache binds
before bandwidth before compute) holds on this workload. The measured load
(~10 office-hours users, ~20 at peak on the measured 43 s cycle) uses
4–8% of the binding ceiling: **~25× headroom, and the first wall is KV,
not FLOPs.**

Cold-side at the measured workload: E[cold request] 1.34 s vs 34 ms for a
2k warm turn (thrash ratio 39×); all-miss ceiling 0.745 req/s; below
~0.7 req/s **no miss rate can saturate the group** (f* > 100%); B* ≈ 7
simultaneous full-context misses inside a 10 s TTFT budget. Model cold
TTFT at the office-hours rate: 1.35 s — consistent with the measured
p99 2.68 s (p99 rides the context tail's quadratic term).

## 3. What this did and did not change

Changed (2026-08-27, this commit):
- `OUT_TOKENS_DEFAULT` / `AVG_OUT_TOK` and the explorer's Output-per-response
  default: 1,000 → **400** (measured 404.1). Re-pinned the steady-decode
  self-check (~6.4 decoders at ~370 tok/s → ~2.2 at ~430 tok/s).
- The MFU bracket: [0.30, 0.60] → **[0.35, 0.55]** (prefill.md #1, two
  calibration points). `validate_deployment.py`'s reference ms ranges still
  quote the wider band, flagged in its CONFIG notes.

Deliberately NOT changed:
- **Invalidation default stays 1.0%.** The measured ~9.3% is the *effective*
  total-miss rate — true invalidations + evictions + partial-match slippage —
  while the slider models i.i.d. can't-match-anything requests. The docs'
  10% adverse planning case turns out to be the production operating point;
  plan with it, but the default readouts keep the 1% semantics.
- **Warm-turn 2,000 and think-time 30 s reference.** The 87.5% savings is
  ONE observable decomposing over TWO unknowns (miss rate, turn size):
  f = 9.3% *assumes* the 2k turn and vice versa. Separating them needs the
  vLLM prefix-cache hit/query counters per request — a follow-up
  measurement, not a dashboard read.

## 4. Caveats

- All means are E[X]/E[Y] aggregates over uncontrolled traffic; only the
  MFU point in prefill.md #1 (isolated counter deltas) is a controlled
  measurement.
- The lognormal refit matches the *mean* only; the dashboard heatmap was
  not exported, so the tail (σ) is carried over from the earlier trace.
- Speculative-decoding acceptance is reported but not yet consumed by the
  model (the decode roofline prices no speculative speedup for the 27B).
- The capacity table is per replica group at MFU 0.45; the [0.35, 0.55]
  bracket moves the latency/saturation columns ~1.6× end to end and leaves
  cache and decode untouched.
