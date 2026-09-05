# GLM-5.3 — GLM-5.2's successor, checked tensor by tensor

**Purpose:** the study's GLM-5.2 row (`GLM52`, `research/model_glm52.md`)
was off the frontier once seats were priced at capacity; GLM-5.3 (released
2026-08-25) scores 83.9% on Terminal-Bench 2.1 against 77.9%. This note
checks whether it can take the row without touching a constant. Verdict:
**yes for everything the architecture determines; the MTP fit and the NVFP4
bytes stay projections, and the license changed.**

## 1. What was compared (read 2026-09-06)

| Artifact | GLM-5.2 (`zai-org/GLM-5.2-FP8`) | GLM-5.3 (`zai-org/GLM-5.3`) | Result |
|---|---|---|---|
| `config.json` | 56 keys | 56 keys | **55/56 identical**; only `transformers_version` (5.12.0 → 5.15.0) |
| `architectures` / `model_type` | `GlmMoeDsaForCausalLM` / `glm_moe_dsa` | same | same vLLM path |
| `model.safetensors.index.json` tensor names | 118,629 | 118,629 | identical sets |
| shard headers (all 141 shards, HTTP range reads) | dtype + shape per tensor | same | **0 differences**; parameter sum 753,375,793,584 both |
| FP8 checkpoint bytes | 755,632,050,320 (141 shards) | 755,632,050,320 (141 shards) | identical |
| quantization | fp8 e4m3, dynamic, block 128×128 | same | identical |
| vLLM recipe (recipes.vllm.ai) | MTP 5 drafts, fp8 KV, min vLLM 0.23 | MTP 5 drafts, fp8 KV, min vLLM 0.28 | same serving shape |
| License | MIT | **GLM-5.3 License** | see § 2 |

Architecture-determined constants that carry over untouched: `kv_bpt`
(47.3 KiB), `w_resident` (755.5e9), `w_decode_shared`, `w_route_pertok`,
`w_route_total`, `kv_decode_bpt`/`_const`/`_topk` (the DSA decode pricing),
`kv_fp16_ok: false`, `params_prefill`, `attn_layers`/`attn_d`, `max_ctx`.

## 2. What is not the same

- **Weights.** Different values, hence the score.
- **MTP (`mtp: 1.7`).** Was already a transplanted fit on 5.2 (unmeasured);
  the 5.3 recipe drafts 5 tokens as before. Stays unmeasured.
- **NVFP4.** `nvidia/GLM-5.2-NVFP4` (routed experts only, ~465 GB) was the
  measured recipe. No `nvidia/GLM-5.3-NVFP4` exists as of 2026-09-06. The
  explorer keeps the 5.2 bytes as a **projection** onto tensor-identical
  weights.
- **License.** GLM-5.2 was MIT. GLM-5.3 ships its own "GLM-5.3 License":
  MIT's grant and notice clause, plus one condition — a licensee (with
  affiliates) that operates a *Model-as-a-Service* business with aggregate
  revenue above US$10 billion over any 12 months must pass Z.AI's security
  review before commercial use. MaaS is defined as giving third parties
  meaningful control over inputs, parameters or training data via API;
  products with the model embedded in features, and relays to others'
  hosted models, are excluded. For the study's use case (self-hosted
  internal coding agents) the clause does not bite, but it is no longer
  MIT and the method page says so.
- **Recipe hardware list.** The 5.3 recipe marks B300 and an Ascend 950PR
  node validated; the 5.2 recipe listed B300, B200, MI300X, MI355X. Neither
  lists the H200; the explorer's H200 rows were already pool-arithmetic
  plus the recipe's 893 GB floor, unchanged.
- **Release.** 2026-08-25 (weights; AA lists 2026-08-18) vs 2026-06-16.

## 3. AA score

`glm-5-3`, "GLM-5.3 (max)", Terminal-Bench 2.1 **83.9%** = 224/267, index
48.6, effort max (the index run). GLM-5.2 (max) was 77.9% = 208/267.

## 4. What changed in the repo

`CONFIG.MODELS["GLM52"]` is named GLM-5.3 with the comments above (the key
stays `GLM52` so share links keep resolving); `CONFIG.QUALITY["GLM52"]` is
224/267; `scenario_model.py`'s mirror, the model button, the method page,
the README and the Terminal-Bench ledger follow. `research/model_glm52.md`
is untouched: it is the derivation, and the derivation is unchanged.
