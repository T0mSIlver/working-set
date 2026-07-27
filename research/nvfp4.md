# NVFP4 weight quantization — format, gating policy, and Qwen3.6 constants

**Purpose:** defensible constants and policy decisions for adding **NVFP4 weights**
as a selectable quantization in `scripts/scenario_model.py` and the interactive
explorer: bytes/param arithmetic, the B300-only hardware gate, the exclusion of
4-bit KV cache, and derived weight constants for the two Qwen3.6 models.

> **Egress note (2026-07-27):** first written under a proxy policy that
> blocked `huggingface.co`, `docs.vllm.ai` and `developer.nvidia.com`, from
> search snippets + cross-checks, with per-claim confidence tiers.
>
> **Re-verified the same day after the block lifted** against the literal
> quantization configs, measured per-shard safetensors dtype splits (HTTP
> range reads of every shard header), `total_size` indexes, and the live
> vLLM issues/docs. Verdicts are inlined per section; the ledger in § 7
> records each item's resolution. Two constants were corrected (both Qwen
> w_resident figures — §§ 6.1–6.2); the B300-only weight gate held; the
> 4-bit-KV exclusion survives as an owner policy but its vLLM-instability
> rationale is now stale (§ 3).

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

## 3. 4-bit KV cache: deliberately **excluded** — now an owner POLICY, not a stability fact

**Status re-verified 2026-07-27 — the original instability rationale is
stale:** issue #32220 (*NVFP4 KV Cache Support*) was **closed completed
2026-05-04** (FlashInfer-backend support merged via PR #40177; working
`--kv-cache-dtype nvfp4` example given) and #43562 (the first-request crash)
was **closed completed 2026-06-02** — its root cause was a missing sm_120
kernel upstream; the B300 (sm_103) is on the supported path. NVIDIA has since
published an NVFP4-KV inference blog (≤1% accuracy loss on its evals; values
are dequantized NVFP4→FP8 *before* attention; Blackwell-only). The
docs.vllm.ai quantized-KV page still lists only fp8 variants (doc lag).

**Owner decision (upheld 2026-07-27): the study keeps the KV dtype axis at
FP8 (default) / FP16 and does not model 4-bit KV** — now purely a
conservatism/accuracy-margin choice on a young serving path, no longer a
capability constraint. NVFP4 here means *weights only*.

## 4. What actually stays high-precision in NVFP4 checkpoints

**Verified 2026-07-27 against the literal `config.json`:** the real
RedHatAI ignore list is 342 concrete module names + 1 regex (not the 7
regexes previously quoted from summaries). Collapsed:

```
lm_head                                                   (1)
model.language_model.layers.N.linear_attn.{in_proj_a,in_proj_b,
    in_proj_qkv,in_proj_z,out_proj}                       (30 each)
model.language_model.layers.N.mlp.gate                    (40)
model.language_model.layers.N.mlp.shared_expert_gate      (40)
model.visual.*  (attn.qkv/attn.proj/mlp linears, merger)  (110)
re:^mtp.*                                                 (1)
```

Two corrections vs the summary-tier version: **(a)** `embed_tokens` is NOT in
the ignore list — it stays BF16 anyway because `targets: ["Linear"]` never
touches the Embedding module (measured 1.017e9 B BF16; no byte impact).
**(b)** **`re:^mtp.*` IS ignored** — the MTP module is BF16 (measured
1.689e9 B), not quantized as § 6.1 previously assumed; this moves the
w_resident central (see § 6.1).

The load-bearing entry is still **`linear_attn`**: the 30 Gated-DeltaNet
layers stay **BF16** (measured 2.023e9 B — the checkpoint is quantized from
the BF16 base, so excluded modules are 2 B/param, *heavier* than the FP8
checkpoint's 1 B/param). Quantized: full-attention projections, shared
expert, routed experts. Excluded: lm_head, DeltaNet blocks, router gates,
MTP, visual, norms (+ embeddings via the Linear-only targeting).

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

### 6.1 Qwen3.6-35B-A3B — MEASURED per-shard safetensors bytes (2026-07-27)

Per-group bytes read from the shard headers of
`RedHatAI/Qwen3.6-35B-A3B-NVFP4` (HTTP range requests; index `total_size`
25,043,526,104 B incl. visual):

```
group                dtype     bytes (measured)
embedding            BF16      1.017e9   (Linear-only targets, not ignore)
lm_head              BF16      1.017e9   (ignored)
full-attn x10        NVFP4     0.153e9
DeltaNet x30         BF16      2.023e9   (ignored: linear_attn)
shared expert x40    NVFP4     0.071e9
router x40           BF16      0.042e9   (ignored: mlp.gate)
routed experts       NVFP4    18.120e9
MTP module           BF16      1.689e9   (ignored: re:^mtp.* — was assumed
                                          quantized; the note's old 22.92e9
                                          central used the wrong branch)
------------------------------------------------------------------
w_resident_nvfp4 (LM, excl. 0.893e9 visual)  = 24.13e9 B = 22.5 GiB
```

Vs the BF16 base (2 × 35.5e9 = 71.0e9 B) the LM checkpoint is a **2.94×**
reduction (the old derived 3.10× belonged to the MTP-quantized branch).

Decode-side groups (measured):

```
w_decode_shared = attn 0.153 + DN 2.023 + shared-exp 0.071 + router 0.042
                + lm_head 1.017                        = 3.3065e9 B
w_route_pertok  = measured                             = 0.56624e9 B
w_route_total   = measured                             = 18.120e9  B
```

(The MTP module's 1.689e9 BF16 bytes are resident but its decode reads are
folded into the speedup, as everywhere in the study.)

**Non-obvious result the model should surface:** NVFP4 makes the 35B-A3B's
*shared* per-step read **1.7× heavier** than FP8 (3.31 vs 1.94 GB) — the BF16
DeltaNet blocks and lm_head dominate it — while the *routed-expert* bytes drop
1.78×. Low-concurrency decode gets *slower*; the crossover to faster-than-FP8
decode comes as expert reads dominate (n ≳ 4 under the linear union; the
exact crossover on these constants is n = 3.1 — measured −0.7% at n = 3,
+5.4% at n = 4). All three decode constants survived measurement unchanged
(≤0.05%); only the resident total moved.

Note (owner-visible fork): `nvidia/Qwen3.6-35B-A3B-NVFP4` (23.41e9 B total)
uses a different recipe — attention/DeltaNet **FP8**, lm_head **NVFP4** — and
its shared read is only ~1.69e9 B, which would *flip* the shared-read-heavier
result. The study models the RedHatAI checkpoint; the effect is
checkpoint-specific, not format-inherent.

### 6.2 Qwen3.6-27B (dense) — MEASURED checkpoint bytes (2026-07-27)

`nvidia/Qwen3.6-27B-NVFP4` index `total_size` = **21,921,428,072 B**
(header sum exact). The old ×1.11 "as-deployed" scaling is retired: the
official `Qwen/Qwen3.6-27B-FP8` checkpoint measures 30.87e9 B = **28.75
GiB — within 0.2% of the study's 28.8 GiB baseline figure**, so the
baseline convention IS the raw checkpoint size and the NVFP4 constant is
simply the measured checkpoint under the same convention:

```
w_resident_nvfp4(27B) = 21.92e9 B = 20.4 GiB   (was derived 24.47e9: -10.4%)
w_decode_shared       = w_resident   (dense: every step reads all weights)
```

Recipe (from the literal `hf_quant_config.json`, MIXED_PRECISION): MLP
linears + lm_head **NVFP4** g16; `self_attn` AND the DeltaNet
(`linear_attn`) linears **FP8** (not BF16 as previously assumed — only
embed/visual/MTP stay BF16, 4.37 GB); `exclude_modules: ['mtp*']`;
`kv_cache_quant_algo: FP8` (supports the study's FP8-KV choice). Effective
whole-checkpoint compression 0.71× vs FP8.

## 7. Re-verification ledger — RESOLVED 2026-07-27 (proxy block lifted)

1. ✅ `group_size: 16` confirmed in all three repos; exact ignore lists read
   (§ 4 corrected: 342 concrete names; `re:^mtp.*` ignored; `embed_tokens`
   absent but BF16 via Linear-only targets).
2. ✅ Per-dtype bytes measured from every shard header; §§ 6.1–6.2 centrals
   replaced: 35B **24.13e9** (+5.3%), 27B **21.92e9** (−10.4%); decode-side
   constants confirmed ≤0.05%.
3. ✅ #34694 (Hopper Marlin garbled output) still OPEN (fix PR #47315
   unmerged as of 2026-07-27) → **B300-only weight gate holds**. #32220 and
   #43562 CLOSED-completed → § 3 reworded: no-4-bit-KV is now owner policy.
   Hopper docs row remains Marlin weight-only W4A16.
4. ✅ The 27B recipe quantizes DeltaNet linears to **FP8** (neither NVFP4 nor
   BF16) — § 6.2 updated.
5. ✅ Resolved without a startup log: the measured `Qwen/Qwen3.6-27B-FP8`
   checkpoint (30.87e9 B = 28.75 GiB) matches the study's 28.8 GiB within
   0.2%, so no as-deployed gross-up exists to transfer.

Open note for the owner (pre-existing, outside this axis): the study's 35B
FP8 `w_resident` = 35.5e9 (params×1B) understates the measured
`Qwen/Qwen3.6-35B-A3B-FP8` checkpoint (37.46e9 B; ~36.5e9 excl. visual) by
~1–2e9 B — a convention inconsistency inherited from the baseline, listed
here for a future recalibration pass.

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
