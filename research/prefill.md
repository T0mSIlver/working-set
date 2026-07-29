# Prefill cost model — constants, sources, and what is *not* known

Everything else in this study is an **HBM roofline**: decode is memory-bound,
so bytes moved ÷ bandwidth is the whole model. Prefill is the exception. A
32k-token chunk reads the weights **once** and does ~2 × params × tokens FLOPs
on them, landing ~150× above the H200's roofline ridge point. Bytes are free
there; FLOPs are the budget.

This note carries the constants that made `scenario_model.py`'s prefill
section possible, and — more importantly — the confidence tier of each.

> **Status: analytic, UNVALIDATED.** The baseline experiment collected prefill
> speeds (`docs/writeup.md`, "Collect prefill and decoding speeds") but the
> repository kept only the `ttft < 0.4 × cold` warm/cold classification
> heuristic. **There is no measured prefill number in this repo to check
> against.** Every figure below is a projection. The single measurement that
> would move it most: one `vllm bench` prefill run at
> `max_num_batched_tokens=32768` on the 27B, TP2 — a few minutes of work on
> hardware that already exists.

---

## 1. Hardware: dense FP8 tensor-core throughput

| Part | `peak_flops_fp8` | Confidence | Derivation |
|---|---|---|---|
| H200 SXM | **1.979e15** | HIGH | NVIDIA's H200 datasheet leads with "3,958 TFLOPS FP8". That figure is **with 2:4 structured sparsity**, which no dense LLM GEMM achieves. The dense number is exactly half. |
| B300 (HGX form) | **6.75e15** | MEDIUM | `research/gpu_b300.md` records 13.5 PFLOPS *dense FP4* per GPU for the 8-GPU HGX B300 baseboard (108/8; the GB300 NVL72 form is 15). FP8 runs at half the FP4 rate on 5th-gen tensor cores → 6.75 PFLOPS. |

**The sparsity trap is the main hazard here.** Every vendor spec sheet in this
class quotes the sparse number first. Using 3,958 TFLOPS would halve every
prefill time in this study and make the thrash finding look half as severe.

`gpu_b300.md` limitation 3 ("B300 FLOPS are not modelled") is now partially
retired: they are modelled, at MEDIUM confidence, for prefill only. No
capacity or decode figure reads `peak_flops_fp8`.

### Model FLOP Utilisation (MFU)

`MFU_LOW / MFU_DEFAULT / MFU_HIGH = 0.30 / 0.45 / 0.60`. **Not measured.**
45% is a mid-range figure for FP8 prefill on Hopper-class parts with TP
collectives in the loop. This is the softest input in the section — the
plausible bracket moves every absolute time by **2×**.

What MFU does *not* affect: the cold/warm cost ratio (`thrash_ratio`), which
is a property of context length vs turn length and cancels MFU entirely. That
is why the ratio, not the millisecond figure, is the load-bearing result.

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
| Qwen3.6-27B | 25.4e9 | 16 | 24 × 256 = 6144 | 27B dense − ~1.6e9 (embed + untied lm_head, vocab 151,936 × hidden 5,120, ×2). 64 layers, interval 4 → 16 full-attention; Q/KV heads 24/4, head_dim 256 (`model_35ba3b.md` family cross-reference). |
| Qwen3.6-35B-A3B | 2.9e9 | 10 | 16 × 256 = 4096 | Published **A3B** = ~3B active, less router/embed. 40 layers (30 DeltaNet + 10 full-attention); Q/KV 16/2, head_dim 256 (`model_35ba3b.md`). |
| Mistral-Medium-3.5-128B | 124.8e9 | 88 | 96 × 128 = 12288 | Dense 128B − ~3.2e9 (vocab 131,072 × hidden 12,288, ×2). **All 88 layers** are full GQA — no linear layers, no sparsity (`model_mistral_medium35.md`). |
| GLM-5.2 | 38.1e9 | 78 | 64 × 256 = 16384 | 744B-**A40B** → ~40B active, less ~1.9e9 (vocab 154,880 × hidden 6,144, ×2). 78 layers, 64 heads, qk_nope 192 + rope 64, v_head_dim 256 (`model_glm52.md`). |

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
3. **Mistral-Medium-3.5's MTP absence does not help it here.** Prefill has no
   speculative path to lose; the model's prefill fragility (see below) is
   purely its dense 88-layer GQA geometry.
4. **The attention term assumes full causal attention within a chunk.** With
   chunked prefill the chunk also attends to all *previously* prefilled
   tokens of the same sequence, which this model does **not** charge — a real
   under-estimate that grows with prompt length. Another bias against the
   hypothesis.

---

## 3. FLOP arithmetic

```
gemm  = 2 × params_prefill × tokens          # multiply-accumulate = 2 FLOPs
attn  = 2 × tokens² × attn_d × attn_layers   # QK^T + AV, each 2·L²·d, causal halves
```

Per attention layer, non-causal: QK^T is 2·L²·d and A·V is 2·L²·d (summing
over heads, d = n_q_heads × head_dim). Causal masking halves the pair →
2·L²·d per layer.

At the 27B's 32k chunk the attention term is **11%** of the work. A single
un-chunked 180k pass would be ~40%. Chunked prefill caps the quadratic term by
construction, which is why this study prices **chunks** and not whole prompts.

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

- **Queueing.** `prefill_duty` reports utilisation, not latency. At duty 0.8
  the queue is already deep; the model says nothing about the TTFT
  distribution, only about where the ceiling is.
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
better. The thrash finding is a lower bound.
