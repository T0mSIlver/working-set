# Qwen3.6-35B-A3B — architecture parameterization note

**Purpose:** defensible GPU KV-cache / decode-bandwidth constants for
**Qwen3.6-35B-A3B** (35B total, ~3B active, hybrid Gated-DeltaNet + full-attention MoE),
as used by `scripts/scenario_model.py` and the interactive explorer.

**Status: this is a real published model.** All structural numbers below are taken
from the published `Qwen/Qwen3.6-35B-A3B-FP8` `config.json` (cross-checked against an
independent architecture overview). An earlier revision of this note proxied the model
with Qwen3-Next-80B-A3B because the 3.6 release had not been checked; that proxy is now
retired — it disagreed with the real config on layer count (48 vs **40**), full-attention
layers (12 vs **10**), expert count (512 vs **256**), active experts (10 vs **8**) and
vocabulary (151,936 vs **248,320**).

> Egress note: `huggingface.co/.../config.json` is reachable through this environment's
> web-fetch path (the raw `resolve/` URLs 403 through the proxy). Values were read from
> the FP8 repo's config page and cross-checked against the architecture overview blog.

---

## 1. Architecture table (published config)

| Field | Value | Source |
|---|---|---|
| `num_hidden_layers` | 40 | config.json |
| `full_attention_interval` | 4 (every 4th layer is full attention) | config.json |
| Full-attention (GQA) layers | **10** | = 40/4 |
| Gated-DeltaNet (linear) layers | **30** | = 40−10 |
| `num_attention_heads` (Q) | 16 | config.json |
| `num_key_value_heads` (GQA) | **2** | config.json |
| `head_dim` (full attention) | **256** | config.json |
| DeltaNet `linear_num_key_heads` | 16 | config.json |
| DeltaNet `linear_num_value_heads` | 32 | config.json |
| DeltaNet key/value head_dim | 128 / 128 | config.json |
| DeltaNet `linear_conv_kernel_dim` | 4 | config.json |
| `hidden_size` | 2048 | config.json |
| `num_experts` (total) | **256** | config.json |
| `num_experts_per_tok` (routed, active) | **8** | config.json |
| Shared expert | 1 (`shared_expert_intermediate_size` = 512) | config.json |
| `moe_intermediate_size` | 512 | config.json |
| `vocab_size` | **248,320** | config.json |
| Total params | ~35B | model card |
| Active params/token | ~3B | model card |
| Context length | 262,144 native | model card |
| MTP module | 1 layer (`mtp_num_hidden_layers`: 1 on the 27B sibling; 2 tok/forward) | config / model card |
| FP8 quantization | e4m3, dynamic activation scheme | config.json (FP8 repo) |

**Family cross-reference** (the dense sibling anchors the baseline study):

| Model | Layers (DN+full) | Q/KV heads | head_dim | DN v-heads | Experts tot/act | Total/Active |
|---|---|---|---|---|---|---|
| Qwen3.6-35B-A3B | 40 (30+10) | 16/2 | 256 | 32×128 | 256/8 (+1 shared) | ~35B/~3B |
| Qwen3.6-27B (dense) | 64 (48+16) | 24/4 | 256 | 48×128 | — | ~27B |

The 27B's published config (64 layers, interval 4 → 16 full-attention layers × 4 KV
heads × 256 head_dim) reproduces the baseline study's 32 KiB/token FP8 assumption
**exactly** — the baseline's KV arithmetic needs no correction.

---

## 2. KV-cache bytes per token (full-attention layers only)

Only the **10 full-attention layers** hold a length-growing KV cache; the 30 DeltaNet
layers hold a constant recurrent state (Section 3).

```
kv_bpt = full_attn_layers × num_kv_heads × head_dim × 2(K,V) × bytes_per_elem
FP8 :   10 × 2 × 256 × 2 × 1 = 10,240 B/token = 10.0 KiB/token
FP16:   10 × 2 × 256 × 2 × 2 = 20,480 B/token = 20.0 KiB/token
```

For comparison: 27B dense = 16 × 4 × 256 × 2 × 1 = 32 KiB/token (FP8). The MoE's
KV/token is **3.2× smaller** than the 27B's.

---

## 3. Gated-DeltaNet recurrent state (constant per session, length-independent)

State = per-value-head S matrix (key_head_dim × value_head_dim) + causal-conv state.

```
per DeltaNet layer S = 32 v-heads × 128 × 128        = 524,288 elements
over 30 layers                                        = 15,728,640 elements
conv state: kernel 4 × conv_dim(2×2048 + 4096 = 8192) = 32,768 elem/layer → 983,040
total                                                 = 16,711,680 elements
bf16 : 33,423,360 B = 31.9 MiB      fp32 : 66,846,720 B = 63.8 MiB
```

**Dtype calibration:** the same arithmetic on the 27B (48 DN layers × 48 v-heads ×
128×128 + conv) gives **75.7 MiB in bf16** — matching the baseline study's measured-fit
"75 MiB/session" almost exactly, and 151 MiB in fp32, which does not. We therefore take
**bf16 state (31.9 MiB)** as the central value for the 35B-A3B and carry fp32 (63.8 MiB)
as a sensitivity case.

---

## 4. Weight bytes (FP8: 1 byte/param) — full arithmetic

Per-component parameter counts (biases/norms negligible):

```
embedding        : 248,320 × 2048                       = 0.509e9
lm_head (untied) : 248,320 × 2048                       = 0.509e9
full-attn ×10    : q+gate 2×(2048×4096) + k,v 2×(2048×512) + o 4096×2048
                 = 27.26e6/layer × 10                   = 0.273e9
DeltaNet  ×30    : in_proj_qkvz 2048×(2048+2048+4096+4096) + out 4096×2048
                   + ba/conv ≈ 33.7e6/layer × 30        = 1.012e9
shared expert×40 : 3×2048×512 = 3.146e6/layer           = 0.126e9
router ×40       : 2048×256                             = 0.021e9
routed experts   : 256 × 3.146e6 × 40                   = 32.212e9
MTP module (~1 layer: attn + MoE)                       ≈ 0.84e9
--------------------------------------------------------------------
TOTAL                                                   ≈ 35.5e9  ✓ "~35B"
```

**The published 256-expert config lands on ~35B total with no interpolation** — unlike
the retired 80B proxy, which needed a fictitious "218 experts" fit.

Grouped for the bandwidth model (FP8, bytes = params):

```
w_route_pertok  = 8 routed experts activated by one decoding token
                = 8 × 3.146e6 × 40                      = 1.007e9 B  (0.94 GiB)
w_route_total   = all routed experts (saturation ceiling)
                = 256 × 3.146e6 × 40                    = 32.212e9 B (30.0 GiB)
w_decode_shared = attn + deltanet + shared expert + router + lm_head
                = 0.273 + 1.012 + 0.126 + 0.021 + 0.509 = 1.940e9 B  (1.81 GiB)
active / token  = shared + pertok ≈ 2.95e9 B            ✓ "~3B active"
w_resident      ≈ 35.5e9 B = 33.1 GiB (all experts + MTP module resident)
```

Expert-union saturation: the *linear* union bound `n × 8` reaches all 256 experts at
exactly **n = 32** concurrent decoders. Under uniform independent routing the expected
union is `256 × (1 − (1 − 8/256)^n)` — only ~64% coverage at n = 32 — so the linear
model is a deliberate conservative (slow-side) bound; real routing is more correlated
still.

---

## 5. What changed vs the retired Qwen3-Next-80B proxy

| Quantity | 80B proxy (old) | Real Qwen3.6-35B-A3B | Effect |
|---|---|---|---|
| KV bytes/token (FP8) | 12 KiB | **10 KiB** | pool +20% |
| DeltaNet state | 72 MiB (fp32) | **31.9 MiB (bf16)** | cheaper offload + pool charge |
| Resident FP8 weights | 32.6 GiB (218-expert fit) | **33.1 GiB** (published 256 experts) | ~equal |
| w_decode_shared | 1.49e9 B | **1.94e9 B** (larger vocab lm_head, real DN arithmetic) | slightly slower |
| w_route_pertok | 1.51e9 B | **1.007e9 B** (8×, not 10×) | faster |
| Expert saturation | n ≈ 22 | **n = 32** | kink moves right |
| Vocabulary | 151,936 | **248,320** | lm_head 509M |

## 6. Remaining assumptions (everything not from the published config)

- **FP8 KV cache** (`--kv-cache-dtype fp8_e4m3`) — a serving choice, same as the baseline.
- **bf16 DeltaNet state** — inferred from the 27B/75 MiB consistency check (Section 3),
  not read from vLLM source; fp32 would double it (sensitivity case).
- **MTP decode speedup 1.7×** — kept equal to the baseline's measured-fit value; the
  35B-A3B ships an MTP module but its acceptance rate on our workload is unmeasured.
- **MTP module weight reads during decode are ignored** (folded into the 1.7×), as in
  the baseline.
- **DeltaNet per-layer minor params** (A, dt_bias, norms) rounded away; MTP module size
  estimated as ~1 full layer.

## Sources
- Qwen3.6-35B-A3B model card: https://huggingface.co/Qwen/Qwen3.6-35B-A3B
- Qwen3.6-35B-A3B-FP8 config.json: https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8/blob/main/config.json
- Architecture overview (independent cross-check): https://huggingface.co/blog/EXDai/qwen36-35b-a3b-architecture-overview
- Qwen3.6-27B-FP8 config.json (dense sibling / baseline anchor): https://huggingface.co/Qwen/Qwen3.6-27B-FP8/blob/main/config.json
- Specs summary: https://apxml.com/models/qwen36-35b-a3b
