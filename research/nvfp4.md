# NVFP4 weight quantization — format, gating policy, and Qwen3.6 constants

**Purpose:** defensible constants and policy decisions for adding **NVFP4 weights**
as a selectable quantization in `scripts/scenario_model.py` and the interactive
explorer: bytes/param arithmetic, the B300-only hardware gate, the exclusion of
4-bit KV cache, and derived weight constants for the two Qwen3.6 models.

> **Egress note (2026-07-27): weaker provenance than research/model_35ba3b.md.**
> Since that note was written, the environment's proxy policy has tightened:
> `huggingface.co` (including the previously-reachable config pages and the HF
> API), `docs.vllm.ai`, and `developer.nvidia.com` all now return 403 at the
> gateway. Everything below therefore rests on (a) search-index entries — real
> repo paths and issue titles, which cannot be hallucinated by a summarizer —
> (b) search-engine summaries of those pages, and (c) internal cross-checks
> against this repo's own published-config arithmetic. Each claim is tiered:
> **[corroborated]** (multiple independent sources and/or carried by titles/
> paths alone), **[summary]** (search-summary only — directionally reliable,
> not citable), **[assumption]**. The re-verification ledger is in § 7.

---

## 1. Format and bytes per parameter

**NVFP4** [corroborated — consistent across NVIDIA, Red Hat, and community
sources]: FP4 **E2M1** elements (4 bits), **16-element blocks**, one **FP8 E4M3
scale per block**, one **FP32 per-tensor scale**. Contrast **MXFP4**: 32-element
blocks with E8M0 (power-of-two) scales and no per-tensor scale. A checkpoint
with `group_size: 16` is NVFP4-family; 32 is MXFP4.

```
NVFP4 bytes/param (quantized tensors only):
  element       4 bits              = 0.5    B/param
  block scale   8 bits / 16 elem    = 0.0625 B/param
  tensor scale  32 bits / tensor    ≈ 0      (amortized)
  ------------------------------------------------------
  TOTAL         4.5 bits            = 0.5625 B/param
  vs FP8 (1 B/param)  : 1.778x smaller
  vs BF16 (2 B/param) : 3.556x smaller
```

**0.5625 B/param must NOT be applied checkpoint-wide** — see § 4: real NVFP4
checkpoints keep embeddings, lm_head, norms, MoE router gates and (for hybrid
models) the linear-attention blocks in **BF16**, which caps the whole-checkpoint
compression well below the per-tensor ratio.

## 2. Hardware gate: NVFP4 weights are **B300-only** in this study

Native FP4 GEMMs require Blackwell tensor cores (compute capability ≥ 10.0 —
SM100/SM103 datacenter, SM120 workstation) [corroborated]. On Hopper
(H100/H200), vLLM's software FP4 **emulation was removed** and NVFP4 checkpoints
fall back to a **Marlin weight-only W4A16** kernel — memory savings but no FP4
compute path, with a reported correctness bug on non-native hardware:

- vLLM PR #19563 — *"[Quantization] Remove FP4 emulation; Fall-back to marlin
  for device < 100"* [corroborated by title]
- vLLM issue #34694 — *"BF16 NVFP4 Marlin produces garbled output on GPUs
  without native FP4 support"* [corroborated by title]

**Policy decision (owner, 2026-07-27): NVFP4 weights are selectable only on the
B300.** The Hopper fallback is at best a storage-only win with no bandwidth-
model analogue (weights dequantize to 16-bit for compute) and at worst broken;
modelling it would misprice decode. `check_dtype_supported()` enforces this in
Python and the explorer greys the option out. Note this is a *policy* gate, not
a physical impossibility.

## 3. 4-bit KV cache: deliberately **excluded**

vLLM has grown a `--kv-cache-dtype nvfp4` flag, but it is tracked as an
in-progress feature (issue #32220 *"NVFP4 KV Cache Support"*) with an open
crash bug (issue #43562 — *"`--kv-cache-dtype nvfp4` crashes at first request
… instead of failing fast at init"*) [both corroborated by title]; vLLM's own
2026-04 blog on KV quantization scopes itself to **FP8** KV. **Owner decision:
the study keeps the KV dtype axis at FP8 (default) / FP16 and does not model
4-bit KV until it is stable in vLLM.** NVFP4 here means *weights only*.

## 4. What actually stays high-precision in NVFP4 checkpoints

Model cards consistently state that *only linear operators within transformer
blocks* are quantized [corroborated]. The reported `llm-compressor` ignore list
for `RedHatAI/Qwen3.6-35B-A3B-NVFP4` [summary, twice-corroborated across
independent write-ups]:

```
"re:.*lm_head", "re:visual.*", "re:model.visual.*", "re:.*mlp.gate$",
"re:.*embed_tokens$", "re:.*shared_expert_gate$", "re:.*linear_attn.*"
```

The load-bearing entry is **`linear_attn`**: the 30 Gated-DeltaNet layers stay
**BF16** (the checkpoint is quantized from the BF16 base, so excluded modules
are 2 B/param — *heavier* than in the FP8 checkpoint, where they are 1 B/param).
Quantized: full-attention projections, shared expert, routed experts.
Excluded: embeddings, lm_head, DeltaNet blocks, router gates, norms.

## 5. NVFP4 checkpoints exist for both Qwen study models [corroborated]

| Model | Repo (used for constants) | Also |
|---|---|---|
| Qwen3.6-27B (dense) | `nvidia/Qwen3.6-27B-NVFP4` (ModelOpt v0.45.0, ~**22 GB** weights vs 55.6 GB BF16, ≈ 2.53×) | `unsloth/`, `ocicek/` (20.6 GB), community 19.7 GB figures |
| Qwen3.6-35B-A3B (MoE) | `RedHatAI/Qwen3.6-35B-A3B-NVFP4` (llm-compressor, self-labelled **"early release"**; GSM8K-Platinum recovery 100.7%) | `nvidia/Qwen3.6-35B-A3B-NVFP4` |

Accuracy context [summary]: published NVFP4 results (e.g. DeepSeek-R1-0528
FP8→NVFP4) land within ~1% of baseline on standard evals; NVFP4 > MXFP4 at
equal bit-width (half the block size, mantissa-bearing scales). Sufficient to
treat NVFP4 as serving-viable; not sufficient to quote accuracy numbers.

## 6. Derived model constants

### 6.1 Qwen3.6-35B-A3B — from this repo's published-config param split

Param groups from `research/model_35ba3b.md` (config-derived), bytes under the
RedHatAI recipe (NVFP4 = 0.5625 B/param; excluded = BF16 = 2 B/param):

```
group                params    dtype     bytes
embedding            0.509e9   BF16      1.018e9   (excluded: embed_tokens)
lm_head              0.509e9   BF16      1.018e9   (excluded)
full-attn x10        0.273e9   NVFP4     0.154e9
DeltaNet x30         1.012e9   BF16      2.024e9   (excluded: linear_attn)
shared expert x40    0.126e9   NVFP4     0.071e9
router x40           0.021e9   BF16      0.042e9   (excluded: mlp.gate)
routed experts       32.212e9  NVFP4     18.119e9
MTP module          ~0.84e9    NVFP4     0.473e9   [assumption: quantized]
------------------------------------------------------------------
w_resident_nvfp4                         22.92e9 B  = 21.3 GiB
```

**Cross-check (non-circular):** vs the BF16 base (2 × 35.5e9 = 71.0e9 B) this
is a **3.10×** reduction; an independent write-up of the checkpoint reports
**"~3.06×"** [summary] — a 1.3% agreement, which the param split reached with
no fitting. Sensitivity: if the MTP module is *excluded* rather than quantized,
w_resident rises to 24.1e9 B (+5%).

Decode-side groups (same split):

```
w_decode_shared = attn 0.154 + DN 2.024 + shared-exp 0.071 + router 0.042
                + lm_head 1.018                        = 3.308e9 B
w_route_pertok  = 1.00663e9 x 0.5625                   = 0.56623e9 B
w_route_total   = 32.212e9  x 0.5625                   = 18.119e9  B
```

**Non-obvious result the model should surface:** NVFP4 makes the 35B-A3B's
*shared* per-step read **1.7× heavier** than FP8 (3.31 vs 1.94 GB) — the BF16
DeltaNet blocks and lm_head dominate it — while the *routed-expert* bytes drop
1.78×. Low-concurrency decode gets *slower*; the crossover to faster-than-FP8
decode comes as expert reads dominate (n ≳ 4 under the linear union; the
exact crossover on these constants is n = 3.1 — measured −0.7% at n = 3,
+5.4% at n = 4).

### 6.2 Qwen3.6-27B (dense) — from the reported checkpoint size

No public param-level split of the 27B exists in this repo, so the constant
comes from the **measured checkpoint size** (~22 GB [corroborated, NVIDIA
repo + the ModelOpt ~2.5× claim on the 55.6 GB BF16 base]), scaled by the
baseline's as-deployed convention (the study's FP8 figure, 28.8 GiB, read as
≈ 1.11× a raw 27.8e9 B [assumption — raw = half the 55.6 GB BF16 figure];
note `docs/scenarios.md` limitation 5 quotes the overhead as ~15%, which
implies a smaller ~26.9e9 raw count — the two conventions differ by ~3% on
the derived constant, see ledger item 5):

```
w_resident_nvfp4(27B) = 28.8 GiB x (22.0 GB / 27.8 GB) = 22.8 GiB = 24.47e9 B
w_decode_shared       = w_resident   (dense: every step reads all weights)
```

Checkpoint-size spread across quantizers (19.7–22 GB — recipes differ in what
they preserve, e.g. vision tower / MTP draft head in BF16) is carried as an
uncertainty note, central = the official NVIDIA repo's ~22 GB. Effective
whole-checkpoint compression 0.79× vs FP8 — the hybrid dense model keeps a
large BF16 share, so the dense NVFP4 win is real but far from 1.778×.

## 7. Re-verification ledger (when network access allows)

1. `config.json` + `hf_quant_config.json` of `nvidia/Qwen3.6-27B-NVFP4`,
   `RedHatAI/Qwen3.6-35B-A3B-NVFP4`: confirm `group_size: 16` and the exact
   ignore lists (§ 4 is the single most load-bearing summary-tier claim).
2. `model.safetensors.index.json` of both: sum per-dtype tensor bytes → true
   w_resident, replacing §§ 6.1–6.2 centrals.
3. vLLM supported-hardware matrix for nvfp4; status of issues #43562/#32220
   (KV) and #34694 (Hopper fallback) — whether the gates still hold.
4. Whether the 27B NVFP4 recipe quantizes the DeltaNet linears (the ~22 GB
   size implies mostly-quantized; the 35B recipe excludes them — recipes
   differ per quantizer).
5. The 27B's raw-vs-as-deployed weight ratio: § 6.2 uses 1.11× (raw =
   27.8e9 from the BF16 size), `docs/scenarios.md` limitation 5 says ~15%
   (raw ≈ 26.9e9). One vLLM startup log settles it and moves the NVFP4 27B
   constant ~3% (24.47e9 vs 25.19e9).

## Sources

- vLLM PR #19563 (FP4 emulation removed, Marlin fallback < CC 10.0):
  https://github.com/vllm-project/vllm/pull/19563
- vLLM issue #34694 (NVFP4 Marlin garbled output on non-native GPUs):
  https://github.com/vllm-project/vllm/issues/34694
- vLLM issue #32220 (NVFP4 KV cache — feature): https://github.com/vllm-project/vllm/issues/32220
- vLLM issue #43562 (nvfp4 KV crash at first request): https://github.com/vllm-project/vllm/issues/43562
- vLLM blog, "The State of FP8 KV-Cache and Attention Quantization in vLLM"
  (2026-04-22): https://vllm.ai/blog/2026-04-22-fp8-kvcache
- RedHatAI/Qwen3.6-35B-A3B-NVFP4 (llm-compressor; ignore list):
  https://huggingface.co/RedHatAI/Qwen3.6-35B-A3B-NVFP4
- Red Hat AI release note (early release, GSM8K-Platinum 100.69%):
  https://x.com/RedHat_AI/status/2045153791402520952
- nvidia/Qwen3.6-27B-NVFP4 (ModelOpt v0.45.0, ~22 GB):
  https://huggingface.co/nvidia/Qwen3.6-27B-NVFP4
- NVFP4 deployment guide w/ recipe details:
  https://knightli.com/en/2026/05/31/nvidia-qwen3-6-35b-a3b-nvfp4/
- DGX Spark vLLM recipe for the RedHatAI checkpoint:
  https://stevescargall.com/blog/2026/04/vllm-recipe-redhatai/qwen3.6-35b-a3b-nvfp4-on-dgx-spark/
- ocicek/Qwen3.6-27B-NVFP4 (20.6 GB variant, BF16 vision tower/MTP head):
  https://huggingface.co/ocicek/Qwen3.6-27B-NVFP4
- NVFP4 vs MXFP4 format comparison:
  https://www.spheron.network/blog/nvfp4-vs-mxfp4-gpu-cloud-4bit-quantization-guide/
- Red Hat: Accelerating LLMs with NVFP4 quantization:
  https://developers.redhat.com/articles/2026/02/04/accelerating-large-language-models-nvfp4-quantization
- NVIDIA, Pretraining LLMs with NVFP4: https://arxiv.org/pdf/2509.25149
