# Prefill cost model — constants, sources, and what is *not* known

Everything else in this study is an **HBM roofline**: decode is memory-bound,
so bytes moved ÷ bandwidth is the whole model. Prefill is the exception. A
32k-token chunk reads the weights **once** and does ~2 × params × tokens FLOPs
on them, landing ~140× above the H200's roofline ridge point. Bytes are free
there; FLOPs are the budget.

This note carries the constants that made `scenario_model.py`'s prefill
section possible, and — more importantly — the confidence tier of each.

> **Status: analytic, partially calibrated (2026-08-27; was: unvalidated).**
> The baseline experiment collected prefill speeds (`docs/writeup.md`,
> "Collect prefill and decoding speeds") but the repository kept only the
> `ttft < 0.4 × cold` warm/cold classification heuristic. Since 2026-08-27,
> §1 carries two measured MFU calibration points (one controlled, one
> implied); every other figure below remains a projection. The single
> measurement that
> would move it most: one `vllm bench` prefill run at
> `max_num_batched_tokens=32768` on the 27B, TP2 — a few minutes of work on
> hardware that already exists.

---

## 1. Hardware: dense FP8 tensor-core throughput

| Part | `peak_flops_fp8` | Confidence | Derivation |
|---|---|---|---|
| H200 SXM | **1.979e15** | HIGH | NVIDIA's H200 datasheet leads with "3,958 TFLOPS FP8". That figure is **with 2:4 structured sparsity**, which no dense LLM GEMM achieves. The dense number is exactly half. |
| B300 (HGX form) | **4.5e15** | HIGH | The DGX B300 datasheet quotes 72 PFLOPS FP8 for 8 GPUs, sparse → 9 PFLOPS sparse/GPU → **4.5 dense** (third-party spec tables list HGX B300 FP8 dense at 4,500 TFLOPS directly). **Corrected 2026-08-01:** this note first derived 6.75 PFLOPS as "half the 13.5 dense-FP4 rate" — but Blackwell *Ultra*'s 1.5× FP4 uplift did not carry to FP8, so the Hopper-era "FP8 = FP4/2" rule over-credits the part by 1.5×. Caught in cross-review. |

**The sparsity trap is the main hazard here.** Every vendor spec sheet in this
class quotes the sparse number first. Using 3,958 TFLOPS would halve every
prefill time in this study and make the thrash finding look half as severe.

**A second trap, now stepped on once:** precision rates no longer scale by
neat powers of two across generations. Blackwell Ultra boosted FP4 only;
deriving one precision's rate from another's is how the B300 row above was
wrong for a revision.

**Weight dtype is NOT priced.** Every prefill figure uses the dense FP8 rate,
NVFP4 checkpoints included. The NVFP4 recipes are model-specific mixtures of
W4A4 layers (faster than FP8, up to the ~3× B300 dense-FP4 rate), FP8 layers,
and BF16 layers (*slower* than FP8 — roughly half rate), so the mixture's
true throughput could land on **either side** of the FP8 line depending on
the per-layer split; without per-layer benchmarks the direction is unknown.
For NVFP4 configurations the prefill figures are therefore a modeling
*choice*, not a bound — the one family where the section's "every bias
points against the hypothesis" bookkeeping cannot be claimed.

`gpu_b300.md` limitation 3 ("B300 FLOPS are not modelled") is now partially
retired: they are modelled, at HIGH confidence after the 2026-08-01
correction, for prefill only. No capacity or decode figure reads
`peak_flops_fp8`.

### Model FLOP Utilisation (MFU)

`MFU_LOW / MFU_DEFAULT / MFU_HIGH = 0.35 / 0.45 / 0.55` — *tightened from
0.30/0.60 on 2026-08-27, on the two calibration points below*. **Not
measured** ~~at all~~ — *see below*.
45% is a mid-range figure for FP8 prefill on Hopper-class parts with TP
collectives in the loop. This is the softest input in the section — the
plausible bracket moves every absolute time by **~1.6×** (was 2× before the
tightening).

**First measured calibration point (2026-08-27).** A production vLLM
deployment of a Qwen3.8-27B FP8-weight checkpoint — the 3.8-generation
*dense* 27B refresh, architecturally identical to the modelled Qwen3.6-27B
per the deployment's operator (NOT related to Qwen3.8-Flash-Next, this
study's hybrid-QSA model of the same generation) — (FP8 weight-only on
Ampere ⇒ BF16 compute, peak 2×312
TFLOP/s dense) on **2×A100, TP2, `max_num_batched_tokens=4096`**. Nine cold
single-chunk requests of 3,300–3,436 random-id tokens, each isolated via
`/metrics` counter deltas (Δcount=1, Δqueue≈0): `request_prefill_time`
0.666–0.700 s ⇒ effective MFU **0.396 ± 0.004 of the raw advertised dense
peak** (2×312 TF BF16; sample SD over n=9 runs on ONE deployment, SEM
±0.0013; range 38.9–40.1%) by this section's FLOP formula. **Convention
note (cross-review 2026-08-27):** the model's `mfu` parameter divides by
`peak_flops(topo)` = advertised × `tp_efficiency` (0.90 at TP2, 0.81 at
TP4), so the model-convention reading of this same measurement is
**44.4%**. State the convention whenever quoting either figure. Cross-check: vLLM's own `estimated_flops_per_gpu_total`
counter agrees with the formula within **3.5–3.6%** on every reconciled
window (the counter flushes mid-request, so reconcile over whole windows,
not per scrape step) — the FLOP accounting in §3 is validated against the
engine's bookkeeping, independent of timing. Reading: on this small model a
3.4k chunk already amortises the weight stream (overhead ≈1% of the pass),
so this deployment's anchor is chunk-size-robust, not a small-chunk
artifact. (An earlier draft read it as "~12% below the 0.45 central" — that
compared the advertised-convention measurement against the
model-convention central; retracted, see the convention note above.)
`MFU_DEFAULT` stays 0.45: the measurement is BF16-on-Ampere with
weight-dequant overhead, and the BF16→FP8, Ampere→Hopper transfer is
exactly the uncharacterised step. Do not cite the deployment's identity in
public-facing material (employer infrastructure); the numbers above are
anonymised.

**Second calibration point (2026-08-27, Hopper FP8, implied).** A 7-day
production dashboard of the 27B in FP8 on 4×H200 TP4
(`research/workload_agentic_poc.md`) reports a mean per-request prefill
time of 159 ms at 87.5% prefix-cache savings on a 57.4k-token mean prompt —
i.e. a mean warm pass of ~7.2k new tokens over ~50k cached context. Priced
with §3's formula, that implies an effective MFU of **40.0% of the raw
advertised peak** (4×1,979 TF; spread 32–48% across the plausible savings
reading), i.e. **49.4%** [40–59%] in the model convention (÷
`tp_efficiency(4)` = 0.81). Weaker than the A100 point (aggregate-mean
arithmetic — E[X]/E[Y], not E[X/Y] — over uncontrolled traffic), but it is
Hopper FP8, the anchor's own regime.

**The convention-consistent reading (corrected on cross-review — the
first draft's "the two points bracket 0.45 from both sides" mixed the
conventions and is retracted):** on the advertised-peak convention the two
deployments agree to under 1% — **39.6% and 40.0%** — across different
silicon (Ampere/Hopper), precision (BF16/FP8-native), TP width and chunk
size. In the model convention they read **44.4% (TP2) and 49.4% (TP4)**
against the 45% central: the central is ~1% high for TP2 and ~10%
conservative for TP4. Both readings support 0.45 as the central and
motivated tightening the bracket to [0.35, 0.55] — a **PROVISIONAL**
tightening, flagged as such: two deployments only, and the implied point's
own reading spread (40–59% model-convention) exceeds the new bracket's
top. A controlled Hopper-FP8 measurement (the
same `measure_mfu.py` protocol against an H200 backend) would settle it;
until then the wider [0.30, 0.60] remains defensible for cross-model,
cross-topology projections.

What MFU does *not* affect: the cold/warm cost ratio (`thrash_ratio`), which
cancels MFU and the GPU part entirely. That is why the ratio, not the
millisecond figure, is the load-bearing result. It is **not** invariant to
the attention model itself: on attention-heavy rows (GLM-5.2's dense-MLA
upper bound above all) the quadratic term drives both the miss cost and the
hit's cross-attention, so those rows inherit weakness #2 below rather than
escaping it.

---

## 2. Per-model prefill constants

`params_prefill` counts parameters doing a **GEMM per token**. Excluded:
embeddings (a lookup, no matmul) and `lm_head` (fires on the last token of a
prefill only). For MoE models it is the **active** count — a routed token
touches k of E experts however long the chunk is.

Excluding lm_head/embeddings biases prefill *cheaper*, i.e. **against** the
thrash hypothesis. That is deliberate: the finding should survive its own
conservative accounting.

| Model | `params_prefill` | `attn_layers` | `attn_d` | Source |
|---|---|---|---|---|
| Qwen3.6-27B | 24.5e9 | 16 | 24 × 256 = 6144 | 27B dense − ~2.5e9 (embed + untied lm_head, vocab **248,320** — the family figure `model_35ba3b.md` reads from config.json; a first revision used Qwen3's 151,936 — × hidden 5,120, ×2). 64 layers, interval 4 → 16 full-attention; Q/KV heads 24/4, head_dim 256. |
| Qwen3.6-35B-A3B | 2.44e9 | 10 | 16 × 256 = 4096 | From `model_35ba3b.md`'s **component ledger**, not the marketing round: per-step shared read 1.940e9 − lm_head 0.509e9 (fires once per prefill, not per token) + 8 routed experts 1.007e9 = 2.438e9. (The published "~3B active" includes embed + lm_head; using it overcharged the GEMM ~19%.) 40 layers (30 DeltaNet + 10 full-attention); Q/KV 16/2, head_dim 256. |
| Mistral-Medium-3.5-128B | 121.8e9 | 88 | 96 × 128 = 12288 | `model_mistral_medium35.md`'s shard ledger: **decoder layers exactly 121.8e9**. Embeddings/lm_head (3.22e9) are outside the layers, and the ~2.7e9-param vision tower is an encoder — never executed on these text re-prefills. (Subtracting embed+lm_head from the 128B *multimodal* total, as a first revision did, left the tower inside the GEMM.) **All 88 layers** are full GQA. |
| GLM-5.2 | 37.4e9 | 78 | 64 × 256 = 16384 | `model_glm52.md`'s derivation: **39.3e9 active excluding embeddings** (vLLM "39B") − lm_head 1.903e9 = 37.4e9. 78 layers, 64 heads, qk_nope 192 + rope 64, v_head_dim 256. |
| DeepSeek-V4-Flash-0731 *(added 2026-08-03)* | 12.70e9 | 41 | 26,624 / 41 ≈ 649 | `model_dsv4flash.md` #6: active GEMM params excl. embed/lm_head (12.703e9 ledger). Quadratic term = indexer over the compressed axis (1024-equiv × 21 CSA layers) + dense HCA (256-equiv × 20); the CSA top-512 and window reads are linear and left out — priced *cheaper*, biased against the thrash hypothesis. |
| Qwen3.8-Flash-Next *(added 2026-08-26)* | 6.04e9 | 12 | 24 × 256 = 6144 | `model_qwen38flashnext.md` §1: active GEMM params excl. embed/lm_head/n-gram lookups (shard-header ledger sum 6.036e9 ✓ the published "6B activated"). 48 layers (36 DeltaNet + 12 QSA full-attention); Q/KV 24/2, head_dim 256; dense upper bound on the 12 QSA layers (weakness 3). |

### Known weaknesses in this table

1. **Hidden sizes for the 27B are inferred, not read.** `model_35ba3b.md`
   records the 35B-A3B's `hidden_size` 2048 but not the 27B's; 5,120 is taken
   from the Qwen3-32B sibling. The embedding deduction is ~6% of
   `params_prefill`, so an error here is worth ~6% of the 27B's prefill time —
   inside the MFU bracket, and it does not touch the ratios.
2. **GLM-5.2's quadratic term is an UPPER BOUND.** `attn_layers=78` prices MLA
   as dense attention over every layer. DSA's top-2048 sparsity is
   characterised for *decode* (`kv_decode_bpt` / `kv_decode_const` /
   `kv_decode_topk` in `model_glm52.md` #3) but its **prefill** behaviour is
   not, so the sparse path is deliberately not claimed. Real GLM-5.2 prefill
   is somewhere at or below what this model reports.
3. **Qwen3.8-Flash-Next's quadratic term is an UPPER BOUND too**, for the
   same reason as GLM-5.2's: QSA's top-2048 sparsity is characterised for
   *decode* only (`model_qwen38flashnext.md` #3); its prefill behaviour is
   not, so all 12 QSA layers are priced as dense attention.
4. **Mistral-Medium-3.5's MTP absence does not help it here.** Prefill has no
   speculative path to lose; the model's prefill fragility (see below) is
   purely its dense 88-layer GQA geometry.
5. **~~Cross-chunk attention is not charged.~~ RESOLVED 2026-08-01 (cross
   review):** every chunk is now charged its attention over the tokens
   already cached (§3), warm hits included. Two consequences worth naming:
   a context's **total** prefill *FLOPs* became chunk-size *invariant* (the
   pair count telescopes — chunking bounds the per-pass spike, not the
   FLOP count), and the expected miss cost is priced on **E[L²]**, not the
   mean length, so the heavy tail is no longer underpriced. **Refined
   2026-08-02:** invariance holds for FLOPs only — per-pass overheads
   (the weight stream) do not telescope, so a miss's machine *time* is
   chunk-dependent after all; see "MFU as a function of chunk size" in §3.

---

## 3. FLOP arithmetic

```
gemm  = 2 × params_prefill × T               # multiply-accumulate = 2 FLOPs
attn  = 2 × T² × attn_d × attn_layers        # intra-chunk: QK^T + AV, causal halves
      + 4 × T × P × attn_d × attn_layers     # cross: T new queries × P cached tokens
```

Per attention layer, non-causal: QK^T is 2·pairs·d and A·V is 2·pairs·d
(summing over heads, d = n_q_heads × head_dim). Causal masking halves the
intra-chunk pairs; every new query attends to **all** `P` cached tokens — the
KV cache spares recomputing their keys and values, not attending over them.

The pair count telescopes over any partition (Σ TᵢPᵢ + Tᵢ²/2 = L²/2), so a
context's **total** prefill work is chunk-size invariant:

```
total(L) = 2 × params_prefill × L + 2 × L² × attn_d × attn_layers
E[cost]  = (2·params·E[L] + 2·attn_d·attn_layers·E[L²]) / (peak × MFU)
```

E[L²], not E[L]² — the quadratic term must be priced on the heavy tail. The
same accounting charges a warm hit its new turn's attention over the whole
cached context (a term linear in E[L]); the first revision priced hits at
turn² only, which *inflated* the thrash ratio (~22–37× then; 18–19× now).

Chunking (`max_num_batched_tokens`) therefore bounds the per-forward-pass
latency — the ITL spike a decode batch sees — *as far as FLOPs are
concerned*. At the 27B's 32k first chunk the attention term is **12%** of
that pass; a mid-re-prefill chunk carries its cross term on top, roughly
doubling by the last chunk of a mean-length context.

### MFU as a function of chunk size (added 2026-08-02)

The invariance above is a statement about FLOPs, and the first revision let
it stand for machine time too ("chunking trades spike size and nothing
else") — which made the explorer's chunk slider read as a free lunch:
shrink the chunk, shrink the spike, pay nothing. That is not how a real
machine behaves, and the reason is exactly what the flat-MFU model omits:
**per-pass costs do not telescope.** Every forward pass streams the full
resident weights once (`w_resident`, not the decode-side active subset — a
chunk of any practical size touches every routed expert on a MoE), plus
kernel launches and collectives the study has no constants for.

The model (`mfu_ceiling` / `prefill_pass_seconds` / `miss_context_seconds`
in `scenario_model.py`, mirrored in the explorer, drawn as its chart E):

```
pass(T, P) = flops(T, P) / (peak × MFU_ceil)  +  w_resident / effective_bw
```

an additive no-overlap roofline. `MFU_ceil` is solved per (model, topology)
so that a first chunk at the 32,768 default nets **exactly the calibrated
45%** effective MFU — the anchor absorbs whatever compute/stream overlap
the real machine achieves there, and every previously published figure
stays put: exactly for one first pass; within <0.2% for a chunked dense
context; within ~2% — *under*, i.e. the flat tables err conservative — for
the MoEs, whose later passes run at their higher solved ceiling. One class
of figures moves by design: a context *shorter* than the chunk is a single
small pass whose effective MFU sits below the anchor, so short misses now
cost genuinely more than the flat model said (~+10% at the reference p5
length for the 35B-A3B, ~+13% for GLM-5.2 — the explorer tile's p5 miss
cost reflects this). That increase is the model's content, not drift. The
flat-MFU tables in docs/ therefore remain valid, and the published Python
functions keep their old default behaviour — the overhead pricing is
opt-in there (`per_pass_overhead=True`, threaded through
`cold_request_seconds` / `max_cold_rate` / `thrash_ratio` /
`prefill_duty` / `breakeven_miss_rate`).

What it says: the **27B barely cares** (42.7% effective MFU at a 2,048
chunk — a dense pass has plenty of compute to hide its own weight stream);
the **35B-A3B falls to ~28%** and **GLM-5.2/TP8 to ~25%** at 2,048 — an MoE
prefills with its few active params but streams its entire expert bank per
pass, so small chunks cost it a fifth of its cold-request ceiling
(1.63 vs 2.10 req/s on 1×H200). That asymmetry, not the spike, is the real
content of the `max_num_batched_tokens` knob, and it is why vLLM's default
sits high: **the chunk buys ITL smoothness with prefill throughput, at a
price set by how MoE the model is.**

Pricing boundary, stated once: the overhead is charged to **miss-side
passes only** (a miss at the duty ceiling runs back-to-back prefill passes
with no host to share a weight stream with). The warm hit's small turn and
the ITL-spike chunk keep the *marginal* flat-MFU price — they join a decode
batch whose pass streams the weights regardless, so charging them the
stream again would double-count. Kernel-launch/scheduler per-pass costs
remain unpriced on both sides: the small-chunk end of the curve still reads
**better** than a real machine would, the honest direction for a model
whose message is "small chunks are not free".

---

## 4. Topology

`peak_flops(topo)` applies the same `tp_efficiency()` haircut (0.90 per
doubling) that the bandwidth model uses. **This is an assumption, and probably
a pessimistic one:** that haircut was fitted loosely to *bandwidth* scaling,
and prefill collectives amortise over far more compute per byte moved, so the
real prefill TP haircut is likely gentler.

The conservative choice for a thrash claim would be no haircut at all (faster
prefill, weaker claim). The study keeps the haircut for consistency with every
other topology figure and flags the direction here.

---

## 5. What this section deliberately does not model

- **Queueing.** *Addressed 2026-08-03 — see [`spike.md`](spike.md).*
  `prefill_duty` reports utilisation, not latency. At duty 0.8 the queue is
  already deep; this section says nothing about the TTFT distribution, only
  about where the ceiling is. `spike.md` adds the M/G/1 wait on top of these
  constants and finds the intuition was right and understated: the
  SLA-limited miss rate binds while duty still reads 76–93%, so `f*` is the
  point where burst tolerance reaches zero rather than a place to plan to sit.
  What remains missing there is a percentile (it solves against the mean) and
  any model of admission control.
- **Prefill/decode scheduling policy.** vLLM's chunked prefill batches a chunk
  with running decodes; the split of a forward pass's time between them is
  treated as additive (`itl_spike`), which is right to first order and wrong
  in detail (kernel overlap, the fact that the decode's KV read happens anyway).
- **Preemption and recompute.** A preempted sequence re-prefills. That is a
  *second* source of prefill load this model attributes to nobody.
- **PCIe restore.** CPU-offloaded sessions must be restored before decoding
  (`docs/scenarios.md` limitation 10). Restore competes with prefill for the
  same PCIe/HBM budget; not modelled in either place.

Every one of these makes the real machine **worse** than this model, not
better. The thrash finding is a lower bound — with one flagged exception:
NVFP4 configurations are priced at the FP8 tensor rate (§1), whose error
direction is unknown, so no bound can be claimed for them.
