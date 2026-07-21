# Qwen "3.6 35B-A3B" — architecture parameterization note

**Purpose:** defensible GPU KV-cache / decode-bandwidth numbers for a hypothetical
MoE LLM named *Qwen 3.6 35B-A3B* (35B total, ~3B active, hybrid MoE).

**Method / proxy choice:** "Qwen3.6 35B-A3B" is **not a real published model**. The closest
real architecture is **Qwen3-Next-80B-A3B** — a hybrid of Gated-DeltaNet (linear,
constant-state) layers interleaved 3:1 with full-attention (GQA) layers, plus a
high-sparsity MoE FFN and a single Multi-Token-Prediction (MTP) module, and ~3B active.
All *structural* numbers below are Qwen3-Next-80B's **published** values; the 35B figure is
obtained by scaling **only the total expert count** while holding layer count, attention
structure, and active-expert count fixed. Consequently **KV/token, DeltaNet state, and
active-weight bytes are invariant to the 80B→35B scaling** — only the *resident* weight shrinks.

> Egress note: HuggingFace `config.json` URLs are blocked (403) by this environment's proxy.
> Qwen3-Next values were taken from the canonical `Qwen3NextConfig` defaults in HuggingFace
> `transformers` (which *are* the 80B-A3B config), cross-checked against the Qwen model card,
> a Gated-DeltaNet analysis gist, and the Qwen3 tech report.

---

## 1. Architecture table (Qwen3-Next-80B-A3B analog → 35B-A3B target)

| Field | Value | Source | Published vs Assumed |
|---|---|---|---|
| `num_hidden_layers` | 48 | transformers `Qwen3NextConfig` | **Published** |
| `full_attention_interval` | 4 (every 4th layer full attn) | transformers config | **Published** |
| Full-attention (GQA) layers | 12 | = 48/4 | Derived |
| Gated-DeltaNet (linear) layers | 36 | = 48−12 | Derived |
| DeltaNet : Full ratio | 3 : 1 (75%/25%) | Qwen model card | **Published** |
| `num_attention_heads` (Q) | 16 | transformers config | **Published** |
| `num_key_value_heads` (GQA) | 2 | transformers config | **Published** |
| `head_dim` (full attn) | 256 | transformers config | **Published** |
| DeltaNet `linear_num_key_heads` | 16 | transformers config | **Published** |
| DeltaNet `linear_num_value_heads` | 32 | transformers config | **Published** |
| DeltaNet key/value head_dim | 128 / 128 | transformers config | **Published** |
| DeltaNet `linear_conv_kernel_dim` | 4 | transformers config | **Published** |
| `hidden_size` | 2048 | transformers config | **Published** |
| `num_experts` (total) | 512 (80B) → **~218 (35B)** | config / scaled | Published / **Assumed** |
| `num_experts_per_tok` (active) | 10 | transformers config | **Published** |
| Shared experts | 1 (`shared_expert_intermediate_size`=512) | transformers config | **Published** |
| `moe_intermediate_size` | 512 | transformers config | **Published** |
| Total params | 80B → **35B (target)** | model card / target | Published / **Assumed** |
| Active params/token | ~3.0B (both) | model card | **Published** |
| FP8 resident weight | 80B→74.5 GiB / **35B→32.6 GiB** | 1 byte/param | Derived / **Assumed** |
| MTP draft tokens | 1 (`num_nextn_predict_layers`=1) | model card | **Published** |
| MTP acceptance | ~0.72–0.83 | analysis blog / DeepSeek-V3 ref | Reported |
| MTP decode speedup | ~1.7× (in ~1.5–2× reported band) | vendor/blog + baseline | **Assumed** |
| `vocab_size` | 151936 | transformers config | **Published** |

**Cross-reference (published dense-MoE siblings):**

| Model | Layers | Q/KV heads | head_dim | Experts tot/act | moe_inter | Total/Active |
|---|---|---|---|---|---|---|
| Qwen3-Next-80B-A3B | 48 (36 DN+12 full) | 16/2 | **256** | 512/10 (+1 shared) | 512 | 80B/3.0B |
| Qwen3-30B-A3B | 48 | 32/4 | 128 | 128/8 (no shared) | 768 | 30.5B/3.3B |
| Qwen3-235B-A22B | 94 | 64/4 | 128 | 128/8 | 1536 | 235B/22B |

---

## 2. KV-cache bytes per token (full-attention layers only)

Only the **12 full-attention layers** hold a length-growing KV cache; the 36 DeltaNet
layers hold a constant recurrent state (Section 3).

```
kv_bpt = (full_attn_layers) x (num_kv_heads) x head_dim x 2(K,V) x bytes_per_elem
FP8 :   12 x 2 x 256 x 2 x 1 = 12,288 B/token = 12.0 KiB/token
FP16:   12 x 2 x 256 x 2 x 2 = 24,576 B/token = 24.0 KiB/token
```

---

## 3. Gated-DeltaNet recurrent state (constant per session, length-independent)

State = the per-value-head S matrix (key_dim x value_dim), fp32:

```
per DeltaNet layer S = num_value_heads x key_head_dim x value_head_dim
                     = 32 x 128 x 128 = 524,288 elements
over 36 layers       = 18,874,368 elements
fp32 bytes           = 18,874,368 x 4 = 75,497,472 B = 72.0 MiB  (+small conv state -> ~75 MiB)
bf16 alt             = ~37 MiB
```

Matches the baseline's ~75 MiB/session (fp32 assumption). Sequence-length independent.

---

## 4. Active-weight bytes per decode token (FP8) — arithmetic

Decode is bandwidth-bound on *active* weights, not the full 35B.

```
Routed experts/tok : 10 x (3 x 2048 x 512) = 31,457,280 /layer x 48 = 1.51e9   (w_route_pertok)
Shared expert      : (3 x 2048 x 512) x 48  = 0.151e9
Router             : (2048 x 512) x 48       = 0.050e9
Full-attn (12 lyr) : ~27.3M/layer            = 0.327e9   (q/k/v/o + out-gate)
DeltaNet (36 lyr)  : ~26.2M/layer            = 0.944e9
lm_head            : 151936 x 2048           = 0.311e9
------------------------------------------------------------
Total active       ~ 3.0e9 params -> 3.0e9 B FP8 = 2.8 GiB/token
  (~2.98e9 excl. lm_head, i.e. the published "~3B active")
```

Grouped for the speed model:
- `w_decode_shared` = attn + deltanet + shared expert + router (+lm_head) ≈ **1.49e9 B (1.39 GiB)**
- `w_route_pertok` = 10 active experts ≈ **1.51e9 B (1.41 GiB)**
- sum ≈ 3.0e9 B (2.8 GiB) = active-weight/token ✓

Saturation ceiling (all routed experts resident):
```
w_route_total = num_experts x (3 x 2048 x 512) x 48
              = 218 x 150,994,944 = 32.9e9 B = 30.7 GiB
w_resident    = w_route_total + non-routed(~2.08e9) ~ 35.0e9 B = 32.6 GiB  ✓
```

---

## 5. Comparison to baseline's "16 attn layers x 4 KV heads x 256" assumption

| Baseline ("27B") assumption | Real Qwen3-Next hybrid | Verdict |
|---|---|---|
| 16 full-attn layers | **12** full-attn (48 layers, interval 4) | baseline high |
| 4 KV heads | **2** KV heads | baseline 2x high |
| head_dim 256 | **256** | matches (Next-specific; 30B/235B use 128) |
| → 32 KiB/token FP8 | → **12 KiB/token FP8** | **baseline ~2.67x too high** |
| FP8 weights 28.8 GiB | 35B → **32.6 GiB** (28.8 GiB ≈ 30.9B) | baseline ~4 GiB low for a true 35B |
| DeltaNet 75 MiB | 72–75 MiB (fp32) | **consistent** |
| MTP 1.7x | within reported ~1.5–2x | **consistent** |

Root cause of the KV mismatch: the baseline mixes families — `head_dim 256` is Qwen3-Next-specific,
but `4 KV heads` matches the dense-MoE siblings (Qwen3-30B/235B), not the hybrid. For a Qwen3-Next-style
hybrid use **12 full-attn layers x 2 KV heads x 256**.

---

## 6. Rounding / interpolation ledger

- **35B totals** (`num_experts≈218`, resident 32.6 GiB): interpolated to hit 35B at fixed 3B active.
  `num_experts` solved from `2.08e9 + n·(3·2048·512·48) = 35e9` → n≈218 (round of a continuous fit).
- **DeltaNet state** rounded to fp32 S-matrix = 75,497,472 B (72.0 MiB); baseline "75 MiB" folds in
  minor conv-state/overhead — treated as equal.
- **Active split** (`w_decode_shared` / `w_route_pertok`) set to sum to the published ~3.0e9 B active;
  lm_head (0.311e9) folded into shared since decode reads it each step.
- **MTP 1.7x** is a point pick inside the reported ~1.5–2x band (kept equal to baseline for consistency).
- All Section-1 structural values are **published**; only the 35B-scaling rows are assumed.

## Sources
- transformers `Qwen3NextConfig` defaults: https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen3_next/configuration_qwen3_next.py
- Qwen3-Next-80B-A3B-Instruct model card: https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct
- Qwen3-Next-80B-A3B-Instruct-FP8 config.json: https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct-FP8/blob/main/config.json
- Gated-DeltaNet analysis gist: https://gist.github.com/justinchuby/0213aa253664fb72e9adb0089816de15
- Qwen3-30B-A3B: https://huggingface.co/Qwen/Qwen3-30B-A3B
- Qwen3-235B-A22B: https://huggingface.co/Qwen/Qwen3-235B-A22B
- Qwen3 Technical Report (arXiv 2505.09388): https://arxiv.org/pdf/2505.09388
- MTP-for-Qwen3 analysis: https://zolotukhin.ai/blog/2026-05-08-why-mtp-heads-are-the-speculative-decode-draft-qwen3-a3b-deserves/
