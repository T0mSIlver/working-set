# Mistral Medium 3.5 (128B dense) — architecture parameterization note

**Purpose:** defensible KV-cache / decode-bandwidth constants for
**Mistral-Medium-3.5-128B** as used by `scripts/scenario_model.py` and the
explorer.

**Status: released 2026-04-28 with OPEN weights** — unlike Medium 3/3.1
(API-only), 3.5 ships on HF as `mistralai/Mistral-Medium-3.5-128B` under a
Modified MIT license. First-party evidence: Mistral's own docs-schema repo
(`mistralai/platform-docs-public`, model file `mistral-medium-3-5-26-04.ts`:
`type: 'Open'`, `parameters: '128'`, `active: '128'`, `contextLength: '256k'`,
weights URL = the HF repo).

> **Egress note (2026-07-27):** `huggingface.co` and `mistral.ai` are blocked
> at this environment's proxy, so the literal `config.json` bytes were NOT
> read. Every field below was instead cross-checked across ≥2 of: Mistral's
> first-party schema repo, NVIDIA's Model-Optimizer / NeMo repos, vLLM's
> recipes + model registry, SGLang's cookbook, and independent config mirrors
> (Kaito catalog, deployment profiles) — all on GitHub. The strongest single
> source for the layer count is **NVIDIA's quantization recipe enumerating
> decoder layers 0–87** for this exact model.

## 1. Architecture table

| Field | Value | Source |
|---|---|---|
| `architectures` | `Mistral3ForConditionalGeneration` (Pixtral vision tower + `ministral3` dense text decoder) | Kaito catalog; NeMo Automodel; vLLM registry |
| `num_hidden_layers` (text) | **88** | NVIDIA ModelOpt recipe ("MLP layers 4–86 … edge layers 0–3 and 87"); Kaito `numHiddenLayers: 88`; 5 more independent mirrors |
| `hidden_size` | 12288 | Kaito; SGLang; NeMo |
| `num_attention_heads` | 96 | Kaito; SGLang; NeMo |
| `num_key_value_heads` | **8** (GQA 12:1) | Kaito; SGLang; NeMo |
| `head_dim` | **128** | SGLang; config mirrors |
| `intermediate_size` | 28672 (SwiGLU) | config mirrors; sibling Devstral-2-123B config |
| Attention type | **plain GQA on ALL layers — no MLA, no MoE, no sliding window, no recurrent layers** (`sliding_window: null`, no `layer_types`) | SGLang ("Standard GQA, not MLA"); config mirrors |
| `vocab_size` | 131072 (Tekken) | config mirrors |
| `max_position_embeddings` | **262144** (YaRN ×64 over a 4k base, θ=1e6) | Kaito; vLLM recipe; Mistral schema "256k" |
| `tie_word_embeddings` | false | config mirrors |
| MTP module | **none** — speculative decoding via a separate **EAGLE-v1 draft repo** (`…-128B-EAGLE`, 2 layers, same GQA geometry) | vLLM recipe + registry |
| Total / active params | **128B / 128B (dense)** | Mistral first-party schema |
| Weights format | base repo ships **FP8 natively** (per-tensor E4M3); vision tower, projector, lm_head kept BF16 | vLLM recipe; NeMo Automodel |
| vLLM support | ≥ v0.20.0 (`Ministral3ForCausalLM` present; EAGLE class from v0.21.0) | tag probes of vLLM `registry.py` |

Parameter cross-check (config-derived, no fitting): per-layer attn
2·12288² + 2·(12288·1024) = 327.2M; MLP 3·12288·28672 = 1,056.9M; × 88 =
121.8B; + untied embeddings 2 × (131072·12288) = 3.22B → **125.0B text**;
+ Pixtral tower (~1.4B) ≈ **126–127B ≈ "128B"** ✓.

## 2. KV-cache bytes per token — all 88 layers grow

```
kv_bpt = 88 layers × 8 KV heads × 128 head_dim × 2(K,V) × bytes
FP8 :  88 × 8 × 128 × 2 × 1 = 180,224 B = 176.0 KiB/token
FP16:  88 × 8 × 128 × 2 × 2 = 360,448 B = 352.0 KiB/token
```

Independently confirmed formula-and-result by a third-party deployment
profile ("KV pro Token (FP8): 88 × 8 × 128 × 2 = 176 KB"). This is **5.5×
the 27B's** KV/token and **17.6× the 35B-A3B's** — a uniformly-full-attention
128B dense model is the study's KV-hungriest case by far, with no recurrent
state (`deltanet_state = 0`) and no per-layer discounts. A full 262k sequence
holds **44 GiB** of FP8 KV.

## 3. Weight bytes

### FP8 (as-shipped)

```
w_resident      = 133.6e9 B (124.43 GiB on-disk FP8 checkpoint, incl. BF16
                  vision tower / projector / lm_head)      [Kaito catalog]
w_decode_shared = 125.0e9 B — text-decoder read per step: 88 layers FP8
                  (121.8e9) + lm_head BF16 (2 x 1.61e9); the vision tower is
                  an encoder and is NOT read during decode; the input
                  embedding is a row lookup, not a matmul read
w_route_*       = 0 (dense)
```

Derived resident (125.0 text + BF16 extras + 2.8 vision ≈ 131e9) agrees with
the reported 133.6e9 within 2% — the gap is packaging/misc, carried as-is.

### NVFP4 — official `nvidia/Mistral-Medium-3.5-128B-NVFP4`

NVIDIA's recipe is **mixed-precision** (quoted from Model-Optimizer
`modelopt_recipes/ptq.md`): decoder **MLP layers 4–86 → NVFP4** W4A4; edge
MLP layers 0–3 & 87 → FP8; **all attention projections → FP8**; **KV cache →
FP8** (4-bit KV is not used even in the NVFP4 checkpoint — consistent with
this study's no-FP4-KV rule).

```
MLP x83 NVFP4 : 83 x 1.0569e9 x 0.5625 = 49.35e9
MLP x5  FP8   :  5 x 1.0569e9          =  5.28e9
attn x88 FP8  : 88 x 0.3272e9          = 28.79e9
lm_head BF16  :                           3.22e9
--------------------------------------------------
w_decode_shared_nvfp4                   = 86.6e9 B   (0.69x the FP8 read)
w_resident_nvfp4 = + embeddings 3.22 + vision tower ~2.8
                                        = 92.7e9 B   (86.3 GiB)
```

(The community `zdy1995love/…-NVFP4` all-linear variant is smaller, ~74 GB;
the official mixed recipe is modelled. Mistral's own schema quotes
`minGpuRam.fp4: 64` GB, which matches neither exactly — carried as an
unresolved vendor figure.)

## 4. Serving notes folded into the model

- **No MTP module → the model's default speculative speedup is 1.0×** (the
  Qwen models' 1.7× is their own measured MTP fit and does not transplant).
  The EAGLE-v1 draft head is the specultive path that exists: +4.0 KiB/token
  of draft KV (2 layers × 8 × 128 × 2 × FP8 = +2.3%, ignored) and an
  unmeasured acceptance rate on this workload — the explorer's speedup knob
  covers the what-if.
- vLLM caveats (from the official recipe): `--tokenizer-mode hf` workaround
  on v0.20 (tekken.json backend crash); 256k context OOMs on tight pools —
  cap `max_seq_len` (this study's 180k reference cap already does);
  `--language-model-only` frees the vision tower's ~2.8 GB.

## 5. Remaining assumptions

- Literal `config.json` unread (HF blocked): all fields are ≥2-source
  mirrors, but the byte-level file should be re-verified when reachable.
- `intermediate_size` 28672 and the YaRN parameters are HIGH/MEDIUM-HIGH
  confidence (config mirrors + sibling model), not first-party.
- NVFP4 embeddings assumed FP8-retained from the base checkpoint (recipe
  lists only MLP/attention treatment); ±1.6e9 B on w_resident either way.

## Sources

- Mistral first-party model schema (release date, Open, 128B, 256k, repo):
  https://github.com/mistralai/platform-docs-public/blob/main/src/schema/models/models/mistral-medium-3-5-26-04.ts
- NVIDIA Model-Optimizer PTQ recipes (NVFP4 scheme; layers 0–87):
  https://github.com/NVIDIA/Model-Optimizer/blob/main/modelopt_recipes/ptq.md
- NVIDIA NeMo Automodel coverage page (arch summary, FP8-native, lineage):
  https://github.com/NVIDIA-NeMo/Automodel/blob/main/docs/model-coverage/vlm/mistralai/mistral-medium-3-5.mdx
- vLLM official recipe (context, FP8 weights, EAGLE, caveats, MI300X numbers):
  https://github.com/vllm-project/recipes/blob/main/models/mistralai/Mistral-Medium-3.5-128B.yaml
- vLLM model registry (architecture classes, version probes):
  https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/registry.py
- SGLang cookbook (GQA-not-MLA, vision tower, EAGLE geometry):
  https://github.com/sgl-project/sglang/blob/main/docs_new/cookbook/autoregressive/Mistral/Mistral-Medium-3.5.mdx
- Kaito model catalog (88 layers, 12288 hidden, 96/8 heads, 262144, 124.43 GiB):
  https://github.com/kaito-project/kaito/blob/main/presets/workspace/models/model_catalog.yaml
- Independent deployment profiles / config mirrors (KV formula, vocab, SWA-null):
  https://github.com/MvdB/dgx-spark-vllm/blob/main/profiles/zdy1995love--Mistral-Medium-3.5-128B-NVFP4/vllm_profile.conf ·
  https://github.com/jjang-ai/jangq/blob/main/jang-tools/jang_tools/mistral3/config.py
- HF repos (blocked from this session, referenced):
  https://huggingface.co/mistralai/Mistral-Medium-3.5-128B ·
  https://huggingface.co/nvidia/Mistral-Medium-3.5-128B-NVFP4
