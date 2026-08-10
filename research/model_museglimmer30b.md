# Muse-Glimmer-30B (dense 30B, 3:1 sliding-window / NoPE-global hybrid) — parameterization note

**Purpose:** checkpoint-derived KV-cache / decode-bandwidth / weight constants
for **Muse-Glimmer-30B** (`meta-models/Muse-Glimmer-30B`, Apache-2.0, created
**2026-08-09**), so the model can be wired into `scripts/scenario_model.py` and
the explorer *when it becomes servable*.

> ## STATUS: NOT WIRED IN — vLLM CANNOT SERVE THIS MODEL YET
>
> As of **2026-08-10**, `muse_glimmer` is **absent from vLLM**: no entry in
> `vllm/model_executor/models/registry.py`, no row in
> `docs/models/supported_models.md`. Support is *in flight* — **vLLM PR #51655
> "Add Muse Glimmer model support", opened 2026-08-10, still open**. The
> architecture exists in `transformers` **main only**
> (`src/transformers/models/muse_glimmer/`, config stamped
> `transformers_version: 5.15.0.dev0` — an unreleased dev tree).
>
> Every other model in this study is priced from an **official FP8 checkpoint
> on a supported vLLM path**. Muse-Glimmer has **neither**: the only official
> weights are BF16, and the only official quantizations are **GGUF K-quants**
> (llama.cpp, consumer-targeted). The FP8 numbers in § 5 are therefore a
> **recipe projection, not a measured checkpoint** — the one place in this
> research directory where a weight constant is not read off a real file.
>
> Consequence: **no `MODELS[...]` entry is added by this note, and the explorer
> is unchanged.** The deploy recipe the explorer hands out is a set of vLLM
> flags; emitting one for a model vLLM refuses to load would be fiction.
> § 7 records the constants and the two code gaps a future wiring PR must
> close, so that PR is a mechanical follow-up once #51655 lands.

Everything in §§ 1–4 *is* measured: the architecture and the KV model come
from the literal `config.json` and the literal safetensors shard headers, and
the parameter ledger reconciles to the checkpoint's own index byte-for-byte.

## 1. Architecture table (literal `config.json`)

| Field | Value |
|---|---|
| `architectures` / `model_type` | `MuseGlimmerForConditionalGeneration` / `muse_glimmer` |
| Modality | **image+text → text** (`pipeline_tag: image-text-to-text`) |
| `text_config.num_hidden_layers` | **52** |
| `hidden_size` / `intermediate_size` | 6656 / 19968 (SwiGLU) |
| Attention | **GQA, 32 Q heads / 2 KV heads × `head_dim` 128** (16:1) |
| **`layer_types`** | **3 × `sliding_attention` + 1 × `full_attention`, repeating** → **39 sliding + 13 full** (full at layers 3, 7, 11, …, 51) |
| `sliding_window` | **2048** (local layers only) |
| **`layer_rope_theta`** | 500,000 on the 39 local layers, **`0` on all 13 global layers** — the global layers are **NoPE** |
| Gated attention | **yes** — extra `self_attn.gate_proj` [4096, 6656] per layer |
| `vocab_size` / `tie_word_embeddings` | 202,048 / **false** (untied `lm_head`) |
| `max_position_embeddings` | **131,072**; `rope_type: "default"` — **no rope scaling in the config** |
| Vision | `vision_tower` 50 layers, hidden 1536, patch 14, `merge_size` 2, GELU MLP; `max_position_embeddings` 1024 |
| Checkpoint dtype | **BF16, all 1,436 tensors** (no mixed precision, no quantized tensors) |
| MTP / draft module | **none** — no `mtp*` / `nextn` tensors exist in the checkpoint |

Numerical quirks worth recording (they do not enter this study's byte model,
but they constrain any re-implementation): `final_logit_softcapping: 20.0`,
`qk_scale_factor: 3.87`, `output_multiplier: 0.196…`, `post_norm_eps: 1e-8`
with both `pre_feedforward_layernorm` and `post_feedforward_layernorm` present.

## 2. Parameter ledger — MEASURED from the shard headers

Read by HTTP range-request over the two safetensors shards (header sampling,
same technique as the GLM-5.2 re-verification), 1,436 tensors, all `BF16`:

| Component | Params | Share |
|---|---:|---:|
| text `mlp` (52 × 398,721,024) | 20,733,493,248 | 69.63% |
| text `self_attn` (52 × 85,196,800) | 4,430,233,600 | 14.88% |
| `vision_tower` (50 layers + patch/pos embed + ln) | 1,852,639,744 | 6.22% |
| `embed_tokens` (202,048 × 6,656) | 1,344,831,488 | 4.52% |
| `lm_head` (untied, same shape) | 1,344,831,488 | 4.52% |
| `vision_adapter` + `vision_projection` | 69,206,016 | 0.23% |
| all RMSNorms (52 × 26,624 + 6,656) | 1,391,104 | 0.00% |
| **TOTAL** | **29,776,626,688** | |

**Cross-check:** the checkpoint's own `model.safetensors.index.json` metadata
declares `total_parameters: 29,776,626,688` and `total_size: 59,553,253,376`
— the sum above matches the first **exactly**, and `total_size` is exactly
`2 ×` it, confirming uniform BF16.

Per text layer = 483,944,448 params: attention 85,196,800
(`q` 27,262,976 + `k` 1,703,936 + `v` 1,703,936 + `o` 27,262,976 +
**`gate` 27,262,976**) + MLP 398,721,024 (3 × 6656 × 19968) + norms 26,624.
The gate projection is the term a config-only estimate misses — it is
**4.8% of the whole model** (52 × 27,262,976 = 1.418B params).

The card's "~29.6B, with ~1.8B dedicated to the perception encoder" is a mild
under-claim: measured total **29.78B**, `vision_tower` alone **1.853B**
(1.922B including the adapter + projection).

## 3. KV bytes per token — only 13 of 52 layers grow

Per attention layer, FP8: `2 KV heads × 128 head_dim × 2 (K,V) × 1 B` =
**512 B/token**. The 39 sliding layers never exceed their 2,048-token window,
so they contribute a **fixed per-session constant**, not a per-token rate:

```
kv_bpt (growing, 13 full-attention layers):
  13 layers x 512 B                    =  6,656 B/token  = 6.50 KiB/token

swa_window (fixed, 39 sliding layers at w=2048):
  39 x 512 B x 2048                    = 40,894,464 B    = 39.0 MiB/session
                                          (exactly 1 MiB per sliding layer)

naive dense equivalent (if all 52 layers were global):
  52 x 512 B                           = 26,624 B/token
  -> the 3:1 hybrid removes 75% of the growing cache
```

**This is the lightest growing cache of any dense model in the study** —
6.50 KiB/token against the 27B's 32 KiB and Mistral-Medium-3.5's 176 KiB
(27× lighter than MM35). A full 131,072-token session costs
`131,072 × 6,656 + 40,894,464` = **913,309,696 B = 0.85 GiB**.

**The fixed term is unusually large in the study's own currency.** Expressed
as KV-token-equivalents (`state / kv_bpt`, the ratio `warm_capacity` charges
per session), the window is worth **6,144 tokens** — against 2,400 for the
27B's DeltaNet state and 3,264 for the 35B-A3B's. Below ~6.1k tokens of
context a session's cache is **majority sliding-window**; the two terms cross
at exactly `39 × 2048 / 13 = 6,144` tokens. Short-context/subagent traffic is
therefore priced by the window, not by the context — the opposite of every
other model here.

## 4. Decode-bandwidth model — the window is read, not re-read

A sliding layer reads **only its window**, so decode does not scale with
context on 39 of 52 layers:

```
per decode step, per sequence:
  global layers   13 x 512 B  = 6,656 B per CONTEXT token   (grows with len)
  sliding layers  39 x 512 B x min(len, 2048) = up to 40,894,464 B per SEQUENCE
```

At the study's reference 31k-median context that is
`31,000 × 6,656 = 206.3 MB` + `40.9 MB` = **247.2 MB/seq/step**, against
**825.3 MB** for the dense-52-layer equivalent — the hybrid cuts decode KV
traffic **3.34×**.

Against the projected FP8 per-step weight read (§ 5, 27.86 GB, amortized over
the whole batch), KV traffic overtakes weights at
`27.856e9 / 247.2e6` ≈ **n = 113 concurrent decoders**. For comparison the
27B crosses far earlier. This model is **weight-bound over essentially its
whole useful batch range** — the practical consequence being that its decode
throughput should scale with `max_num_seqs` much further than the study's
other dense entries before flattening.

> **Modelling caveat — the 2× read/write charge does not apply here.**
> `decode_curves` charges the per-session constant as
> `2.0 × n × deltanet_state`, correct for a **Gated DeltaNet recurrent state**
> (read-modify-write every step). A sliding-window KV cache is **read once**
> per step; the only write is one appended token (~512 B/layer, negligible).
> Carrying the 39 MiB in `deltanet_state` unchanged would therefore
> **over-charge the dominant decode term by ~2×**. See § 7.

## 5. Weight bytes — PROJECTED, not measured

**BF16 (the only official checkpoint) — measured:**

```
w_resident_bf16 = 59,553,253,376 B = 55.46 GiB   (index total_size, literal)
```

**FP8 — a recipe projection, no such checkpoint exists.** Under the standard
vLLM/llm-compressor W8A8 treatment for a dense BF16 model (language-model
`Linear`s → FP8 at 1 B/param; embeddings, `lm_head`, norms and the vision
tower left BF16 — llm-compressor's default ignore set):

```
LM linears (attn 4.430e9 + mlp 20.733e9) x 1 B    = 25,163,726,848
norms            1,391,104 x 2                    =      2,782,208
embed_tokens 1,344,831,488 x 2                    =  2,689,662,976
lm_head      1,344,831,488 x 2                    =  2,689,662,976
vision (tower + adapter) 1,921,845,760 x 2        =  3,843,691,520
---------------------------------------------------------------
w_resident      (projected)                       = 34,389,526,528 B (32.03 GiB)
w_decode_shared (LM linears + norms + lm_head)    = 27,856,172,032 B
```

`w_decode_shared` excludes **`embed_tokens`** (a lookup, not a GEMM) and the
**vision tower** — the tower is an *encoder*, never executed when decoding or
re-prefilling a text context. This follows the Mistral-Medium-3.5 precedent
(`research/model_mistral_medium35.md`: "vision tower not read"), which is the
study's other multimodal entry.

**Dense model** → `w_route_pertok = w_route_total = 0`, and there is **no MTP
module** in the checkpoint, so the speculative-decode default is **`mtp=1.0`**
(the Mistral-Medium-3.5 treatment, not the transplanted 1.7× fit).

**NVFP4: none — `nvfp4_w = None`.** No first-party NVFP4 exists. The community
conversions are **not usable as a source**:

- `RadixArk/Muse-Glimmer-NVFP4` — its `quantization_config` addresses
  `model.layers.N.self_attn.output_gate_proj` and `model.embed_tokens`, but
  the real checkpoint's tensors are `model.language_model.layers.N.self_attn.gate_proj`
  and `model.language_model.embed_tokens`. **The layer paths do not match this
  architecture**, and it additionally quantizes the embedding table to NVFP4
  (`MIXED_PRECISION`, `lm_head` → MXFP8), which no recipe in this study does.
- `Inferact/Muse-Glimmer-30B-NVFP4-W4A4` — W4A4 *activations*, outside the
  weight-only scope `research/nvfp4.md` defines.

Same disposition as DeepSeek-V4-Flash (`nvfp4_w=None`): a community conversion
is not a checkpoint this study will price.

**Fit.** Both dtypes clear a single GPU with room to spare — 32.03 GiB (FP8)
or 55.46 GiB (BF16) against the H200's 141 GB and the B300's 288.4 GB, so
`min_tp = 1` everywhere, **including at BF16, a first for this study**.
Indicative FP8 pool on 1×H200 using the study's published reserve
(141e9 − 17.98 GiB − 34.39e9) / 6,656 ≈ **13.1M tokens** — ~4.7× the 27B's
2.77M, and ~353 warm sessions at the reference workload once each session's
39 MiB window is charged. *Computed by hand for scale only; not produced by
the wired model, since the model is not wired in.*

## 6. Serving notes

- **vLLM: unsupported as of 2026-08-10.** PR #51655 open, unmerged. Until it
  lands there is no `--kv-cache-dtype fp8` path, no FP8 quantization path, and
  no deploy recipe to emit. **Nothing in this note should be read as a claim
  that these numbers have been served.**
- **transformers `main` only** (`5.15.0.dev0`); no released version supports it.
- **Official quantizations are GGUF/llama.cpp**, consumer-targeted, and the
  sizes confirm the card's framing: `muse-glimmer-30B-kquant-17gb.gguf`
  **16.76 GB** (24 GB-VRAM target) and `muse-glimmer-30B-kquant-dynamic.gguf`
  **19.65 GB** (32 GB target), plus a separate `mmproj-kquant.gguf` (1.40 GB)
  vision projector. These are **not** modelled here — this study prices
  datacenter vLLM serving, and GGUF K-quants have no place in that byte model.
- The model card positions this as an **agentic model for consumer hardware**
  (multi-step reasoning, tool use, failure recovery). That makes it a genuinely
  interesting entry for this study's workload — but note the study's premise
  (H200/B300 multi-tenant serving) is *not* the deployment the model was
  designed for.

## 7. Proposed constants + the TWO gaps a wiring PR must close

Recorded so the follow-up is mechanical. **These are not yet in `MODELS`.**

```python
"MUSE30B": Model(
    name="Muse-Glimmer-30B (dense, 3:1 SWA)",
    kv_bpt=6_656,                    # 13 full-attn layers x 2 KV heads x 128 x 2(K,V) x 1B
    deltanet_state=40_894_464,       # 39 SWA layers x 512 B x 2048 window = 39.0 MiB/session
    w_resident=34_389_526_528,       # PROJECTED FP8 recipe (§5) — no FP8 checkpoint exists
    w_decode_shared=27_856_172_032,  # LM linears + norms + lm_head; vision tower not read
    w_route_pertok=0.0, w_route_total=0.0,
    mtp=1.0,                         # no MTP module in the checkpoint
    nvfp4_w=None,                    # no first-party NVFP4; community ones mismatch (§5)
    max_ctx=131_072,                 # native; config carries NO rope scaling
    params_prefill=25_165_117_952,   # text layers, excl. embed/lm_head/vision tower
    attn_layers=13, attn_d=32 * 128, # ONLY the 13 global layers pay O(L^2)
),
```

`attn_layers=13` is the point of the architecture: a sliding layer's cost is
`O(L × 2048)` — **linear** in context — so only the global layers belong in
the quadratic term, exactly as the 27B counts 16 of 64 and the 35B-A3B 10 of 40.

**Gap 1 — `deltanet_state` has the wrong semantics for a KV window.** The
field means "per-session constant recurrent-state bytes" and three behaviours
attached to it are wrong for sliding-window KV:

1. `decode_curves` charges `2.0 × n × deltanet_state` (read+write). A window is
   read once → **~2× over-charge on this model's dominant decode term** (§ 4).
2. `with_kv_dtype("fp16")` doubles `kv_bpt` only. The window **is** KV cache —
   FP16 must double it too, or the FP16 branch **understates VRAM by 39 MiB
   per session**.
3. `state_fp32_ok` gates a "fp32 recurrent state" toggle that is meaningless
   here; the window's precision is governed by the KV dtype.

Suggested fix: a `state_is_kv: bool = False` flag meaning *"the per-session
constant is sliding-window KV, not a recurrent state"* — when set, it scales
with the KV dtype, is read **1×** per decode step, and is excluded from the
fp32-state toggle. Must be mirrored in `interactive/index.html`.
(Note DSv4-Flash already partly stretches `deltanet_state` to cover fixed
windows; that entry is small enough — 3.4 MB of 15.6 MB — that the 2× charge
is minor there, and it is **not** affected by this proposal.)

**Gap 2 — `max_ctx` 131,072 is below the study's default workload cap.**
`Workload.cap` defaults to **180,000**, and `check_cap_allowed` **raises**
when `cap > model.max_ctx`. Muse-Glimmer would be the **first** model whose
native context is under the reference cap. The explorer already clamps its
slider (`capSliderMax()` reads `max_ctx`), so the UI is safe, but any script
using the default `wl()` — `scripts/tables.py`, `scripts/scenarios.py` — would
raise the moment this key is added to their model lists. A wiring PR must
either clamp the cap per model or keep `MUSE30B` out of `MODELS_K`/`MODELS_EXT_K`.

## 8. Assumption ledger

- **The FP8 constants in § 5 are a projection, not a checkpoint.** They assume
  llm-compressor's default ignore set (embed/`lm_head`/norms) and that the
  vision tower is left BF16. If vLLM's eventual Muse-Glimmer path quantizes
  the tower, or fuses/keeps `self_attn.gate_proj` differently, `w_resident`
  moves by up to ~1.9 GB and `w_decode_shared` by less. **Re-derive from the
  real checkpoint once one exists** — do not let this projection harden.
- The 2,048-token window is charged **in full** for every session, including
  ones shorter than 2,048 tokens (a sub-window session should pay
  `min(len, 2048)`). Invisible at the reference workload — every sampled
  length is floored well above 2,048 — but it would matter for a
  short-turn/subagent-heavy workload, which is exactly where § 3 shows the
  window dominates. The `kv_decode_topk` mechanism already implements this
  scaling shape for GLM-5.2/DSv4-Flash and would transfer directly.
- Prefill: the quadratic term counts only the 13 global layers; the sliding
  layers' `O(L × 2048)` term is **not** priced, consistent with this study
  pricing only the quadratic part elsewhere. This biases prefill **cheaper**,
  i.e. against the thrash hypothesis — the same deliberate direction as every
  other model note.
- The **vision tower is never executed** in this study's text-only agentic
  workload, so it is resident-but-unread. A genuinely multimodal workload
  would need image-token prefill priced separately; that is out of scope here.
- NoPE on the global layers is an architectural fact with **no byte-model
  consequence** (no rope cache is stored either way); recorded for completeness.
- No MTP module → `mtp=1.0`. No external draft model is known for this
  architecture, so unlike Mistral-Medium-3.5 there is not even an unmeasured
  EAGLE path to flag.

## Sources

All HF paths below were read **literally** on 2026-08-10 (no mirrors needed —
`huggingface.co` was reachable from this environment):

- `config.json` (architecture, `layer_types`, `sliding_window`,
  `layer_rope_theta`, vision config):
  https://huggingface.co/meta-models/Muse-Glimmer-30B/raw/main/config.json
- `model.safetensors.index.json` (`total_parameters` / `total_size`
  cross-check) + **both shard headers, read by HTTP range request** for the
  1,436-tensor dtype/shape ledger of § 2:
  https://huggingface.co/meta-models/Muse-Glimmer-30B/raw/main/model.safetensors.index.json
- Model card (release framing, license, "~29.6B / ~1.8B encoder", 131,072+
  context, K-Quant variants):
  https://huggingface.co/meta-models/Muse-Glimmer-30B
- Repo metadata (`createdAt: 2026-08-09T17:51:35Z`, `pipeline_tag`, tags,
  file list — note there are **no** `*.py` remote-code files):
  https://huggingface.co/api/models/meta-models/Muse-Glimmer-30B
- **vLLM non-support** (the load-bearing negative result): absent from
  https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/models/registry.py
  and from
  https://raw.githubusercontent.com/vllm-project/vllm/main/docs/models/supported_models.md ;
  in-flight support: **vllm-project/vllm PR #51655**, "Add Muse Glimmer model
  support", opened 2026-08-10, open at time of writing —
  https://github.com/vllm-project/vllm/pull/51655
- transformers support (main only):
  https://github.com/huggingface/transformers/tree/main/src/transformers/models/muse_glimmer
- Official GGUF sizes (§ 6):
  https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF
- Community NVFP4 conversions, **rejected** as sources (§ 5) — tensor-path
  mismatch / W4A4: https://huggingface.co/RadixArk/Muse-Glimmer-NVFP4 ·
  https://huggingface.co/Inferact/Muse-Glimmer-30B-NVFP4-W4A4
