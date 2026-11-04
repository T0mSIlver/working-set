# One decode model: step time and draft speedup

Since 2026-11-04 (WS-9, WS-10) every decode figure comes from one step
equation, and draft speedup is computed from acceptance and depth. The MBU
fold, the opt-in latency pricing and the per-model `mtp` constants are gone.
Code: `src/workingset/model.py` (`decode_step_seconds`, `DecodeConstants`,
`decode_priors`, `tokens_per_step`), mirrored in `interactive/src/config.js`.

## The equation

A decode step of one replica group (tp GPUs) holding n sequences at draft
depth k costs

```
t_step = B / (eta · tp · hbm_bw)                  streamed bytes
       + n (1 + k) · c_tok                         verify GEMMs
       + (tau_0 + [tp > 1] tau_ar) (L + k L_d)     per-layer fixed cost
       + tau_rec · L_rec · (1 + k)                 serial recurrent updates

B = w_decode_shared + expert union over P = n (1 + k) positions
  + k · (draft layers + the lm_head they re-read + their experts over n)
  + KV reads + state traffic                        (independent of k)
c_tok = 2 · params_prefill / (tp · peak_flops_fp8 · MFU)
```

and a sequence advances `m = 1 + a + a² + … + a^k` tokens per step (or the
product form with one acceptance per position). Per-user tok/s = m / t_step.

Per model, the catalogue (`src/workingset/data/catalog.toml`) carries `layers`,
`rec_layers`, `draft_layers`, `w_draft`, `w_draft_route_pertok`,
`spec_tokens` (the recipe's num_speculative_tokens) and `acceptance`.

## Priors

| constant | value | source |
|---|---|---|
| eta | 0.29 | the held-batch sweep's 0.36 (old model convention) × 0.81, the TP4 haircut it was divided by |
| tau_0 | 5 µs | per-layer launch: a prior |
| tau_ar | 10 µs | two one-shot NVLink all-reduces at ~5 µs each (`research/gpu_catalogue.md`); WS-7's link model replaces it |
| tau_rec | 3.86 µs | solved: the 27B / 4×H200 TP4 / k = 3 anchor's measured 2.7 ms intercept, less its three draft passes priced as bytes and the two priors above, per recurrent update |
| MFU | 0.45 | prefill's, also prices c_tok |

`ws calibrate` fits all four from a decode sweep.

## Check against the measured sweep

The k = 3 held-batch sweep the priors were read on (27B, 4×H200 TP4, fp32
state, MFU 0.32), predicted from the physical terms, not fitted per rung
(`tests/test_config.py::test_decode_step_reproduces_the_measured_sweep`):

| n | context | measured ms | predicted ms | error |
|---|---|---|---|---|
| 1 | 46k | 8.6 | 8.66 | +0.7% |
| 8 | 47k | 11.1 | 11.54 | +3.9% |
| 32 | 40k | 19.7 | 20.07 | +1.9% |
| 64 | 37k | 31.3 | 30.76 | −1.7% |
| 8 | 111k | 14.4 | 14.55 | +1.0% |
| 32 | 105k | 31.7 | 32.31 | +1.9% |
| 64 | 103k | 56.0 | 55.62 | −0.7% |
| 1 | 114k | 10.2 | 9.06 | −11.2% (one long sequence alone; no term for it) |

## Decisions, and where the spec was challenged

1. **The draft pass is priced as bytes.** The spec's `k · W_draft` term is
   kept, and W_draft includes the lm_head each draft pass re-reads for its
   logits. On the 27B anchor a BF16 lm_head (2.54e9 B) plus the MTP layer
   would cost more than the measured 2.7 ms intercept leaves after the
   all-reduce floor, so tau_rec would come out negative. The measurement
   allows only the FP8 reading (1.77e9 B per pass), which the catalogue uses,
   flagged as an estimate.
2. **tau_rec is solved, the other two taus are priors.** The spec split the
   2.7 ms as 5 : 10 : 9 µs; with the draft bytes priced, that split
   overshoots the anchor. Fixing tau_0 and tau_ar from what they physically
   are (a launch, two all-reduces) and solving the one left from the one
   measurement keeps the anchor exact by construction.
3. **Acceptance: nearest measured sibling, not the 27B everywhere.** The
   spec carried the 27B's 0.971 to every model. Two outside measurements
   already in this repo disagree for the MoE rows: InferenceX pins GLM-5.3's
   accepted length at 2.94 with 3 drafts (a = 0.797), and Mistral-Large-4's
   vLLM recipe reports 2.79–2.98 with 3 DSpark drafts (a = 0.777 at 32k).
   So: the Qwen MTP heads (35B-A3B, Q3.8-Flash-Next) carry the 27B's 0.971,
   GLM-5.3-Flash carries GLM-5.3's 0.797, and the DeepSeek Flash DSpark
   drafters carry Mistral-Large-4's DSpark 0.777, each flagged unmeasured.
   At 0.971 GLM-5.3's five recipe drafts would give 5.58 tokens per step;
   at 0.797, 3.66.
4. **The k = 2 measurement and the k = 3 sweep disagree.** The retired fold
   was fitted on a k = 2 run that put the 27B / TP4 step at ~9.3 ms at
   n = 1 (314 tok/s). The equation follows the k = 3 sweep and reads 8.0 ms
   there (366 tok/s at k = 2). A step at k = 2 cannot be longer than at
   k = 3 on one server, so the two runs differ in something unrecorded
   (vLLM version, graph capture, date). The self-check no longer pins a k = 2
   band; a k-sweep on one server settles it.
5. **c_tok has no TP haircut**, as `architecture.md` §2.3 says: the
   all-reduces are tau_ar, and a haircut on the verify compute would charge
   them twice.
6. **The expert union defaults to coverage over P positions**, the spec's
   default; `union = "linear"` keeps the no-overlap upper bound.

The reported flat GLM-5.3 curve on 8×B300 (38 tok/s per user from n = 32 to
128 while aggregate rose) was not a max_num_seqs cap leaking into the
curve. Under the old linear union every expert is read from n = 32 on, and
the sparse attention reads little KV per sequence, so the step barely grows.
The new model shows the same physics with a slope: 82 → 74 tok/s.

## What moved

Default workload, `ws predict` and `decode_curves`, main → this change:

| row | per-user tok/s, n=1 | n=32 | decode ceiling, users | limit, binding |
|---|---|---|---|---|
| 27B 1xH200 TP1 | 97 → 136 | 40 → 57 | 70 → 130 | 70 decode → 71 slowed |
| 27B 4xH200 TP4 | 315 → 451 | 130 → 209 | 399 → 737 | 399 decode → 411 slowed |
| 35BA3B 1xH200 TP1 | 545 → 567 | 36 → 79 | 76 → 319 | 76 decode → 234 slowed |
| MM35 4xH200 TP4 | 26 → 40 | 10 → 15 | 0 → 0 | 0 decode → 0 decode |
| GLM52 8xH200 TP8 | 251 → 237 | 14 → 49 | 26 → 308 | 26 decode → 114 cold_wait |
| GLM52 8xB300 TP8 | 418 → 375 | 23 → 82 | 46 → 2810 | 46 decode → 414 cold_wait |
| DSV4F 4xH200 TP4 | 523 → 416 | 49 → 98 | 105 → 2266 | 105 decode → 773 slowed |
| DSV41F 4xB300 TP4 | 743 → 608 | 63 → 99 | 138 → 9233 | 138 decode → 1811 slowed |
| Q38FN 4xH200 TP4 | 522 → 714 | 65 → 150 | 221 → 3146 | 221 decode → 658 slowed |
| GLM53F 4xB300 TP4 | 429 → 402 | 33 → 91 | 67 → 3102 | 67 decode → 884 slowed |
| ML4 8xB300 TP8 | 327 → 283 | 16 → 48 | 29 → 708 | 29 decode → 579 slowed |

Decode stops binding on every MoE row: the old linear union read every
expert from n = 32 at a 1.7 speedup, and the new coverage union over
verify positions at the models' own draft depths does not. Single-stream
speed drops on the MoE rows that draft deeply: their draft passes now cost
bytes, and their acceptance is 0.78–0.80, not 0.97. The dense rows gain
from the raw eta and the 27B's measured k = 3 depth.

## Open

- The 27B's draft-pass bytes and lm_head dtype (decision 1): one shard-header
  read of Qwen3.6-27B-FP8's MTP layer settles them.
- Acceptance for every row flagged unmeasured: `ws decode-sweep
  --prompt-file` on coding text, `/metrics` per-position counts.
- Whether one DSpark draft pass runs all three stages; the equation assumes
  it does.
- The step jump past n ≈ 64 (#74) is still unpriced.
