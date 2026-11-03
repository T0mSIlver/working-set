# DeepSeek-V4-Flash-Vision-Exp (284B-A13B + 0.47B ViT, CSA/HCA) — parameterization note

**Purpose:** constants for **DeepSeek-V4-Flash-Vision-Exp**
(`deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`, MIT weights, created on the Hub
2026-08-31) as used by `workingset.model` (`MODELS["DSV4FV"]`) and the
explorer. Added 2026-10-02, alongside the two text-only DeepSeek Flash models.

> **Provenance (2026-10-02):** every primary artifact was read directly from
> huggingface.co: `config.json`, `inference/config.json`, `inference/model.py`,
> `inference/vision.py`, `README.md`, `model.safetensors.index.json`, the HF
> model API dtype histogram, and the raw safetensors header of **all 48
> shards** via HTTP range requests. The per-tensor byte sum matches
> `metadata.total_size` **exactly** (§ 3). `inference/model.py` and
> `inference/config.json` were diffed against DeepSeek-V4-Flash-0731's
> (`research/model_dsv4flash.md`) the same day (§ 1).

## 1. What it is: 0731's text backbone plus a vision tower

The model card: "builds on the DeepSeek-V4-Flash architecture by
incorporating visual modules and undergoing continued training". The
artifacts bear that out to the byte:

- **Text config.** Every text field of `inference/config.json` equals 0731's:
  43 layers + 3 DSpark stages, `compress_ratios` (2 SWA / 21 CSA ratio-4 / 20
  HCA ratio-128), `head_dim` 512, `window_size` 128, `index_topk` 512, 256
  experts / 6 routed, `max_position_embeddings` 1,048,576. The only additions
  are the ten `vision_*` keys (32 layers × 1024, 16 heads, patch 14, 3×3
  downsample, ≤ 384 tokens per image) and an explicit `norm_eps`.
- **Cache code.** `inference/model.py`'s diff against 0731's touches no cache
  shape: image tokens (ids ≥ `vocab_size`) get their own expert-gate bias
  (`bias_vl`, which also keeps a bias on the 3 hash-routed layers), and inside
  an image span the window mask widens to see the whole span bidirectionally
  (`get_window_topk_idxs_visible`; `width = window + max_image_tokens`, a
  *read* width — the ring buffer stays 128 entries). An image span must be
  prefilled in one chunk (an assert in `forward`).
- **Weights.** The HF dtype histogram differs from 0731's only in BF16
  (+466,393,088 params) and F32 (+12,544); FP8, I8 (packed FP4 experts) and
  I64 are identical.

So **every cache, decode and prefill constant is DSV4F's** — derived in
`research/model_dsv4flash.md` §§ 2–3 and not repeated here. The model's
self-check asserts that every `Model` field but `name`, `w_resident` and
`nvfp4_w` equals `MODELS["DSV4F"]`'s, so a later correction to one that
misses the other fails loudly.

## 2. Constants that differ

| Field | DSV4F | DSV4FV | Why |
|---|---|---|---|
| `w_resident` | 166,878,536,440 (rounded 166.88e9) | **167,811,372,792** | + vision tower and gate biases, § 3 |
| `nvfp4_w` | NVIDIA's 0731 repack | **None** | no official checkpoint, § 4 |
| `mtp` | 1.7 (7 DSpark drafts) | 1.7 (3 DSpark drafts) | same transplanted fit; the card's serve command drafts 3 |

`w_decode_shared` stays 7.66e9: the tower runs on image prefill only, so no
text decode step reads it. `params_prefill` stays 12.70e9: the workload is
text, and the study does not price image encoding (§ 5).

## 3. Weight bytes (all 48 shard headers)

```
main text layers + embed + head + norms   156,015,745,244
mtp.0-2 (DSpark stages)                    10,862,841,372
vision.*  (ViT, BF16)                         823,685,120
aligner.* (BF16)                              109,068,288
image_{start,end,newline,pad} (BF16)               32,768
------------------------------------------------------------
TOTAL = metadata.total_size               167,811,372,792 B  (156.3 GiB)
```

Against 0731's 166,878,536,440 B the delta is 932,836,352 B = the tower
(932,786,176) + 46 × `bias_vl` (256 × fp32 = 1,024 B each, 47,104) + the 3
hash layers' own gate bias (3,072). Every byte of the difference is accounted
for. As for every model in the study, the DSpark stages are charged in
`w_resident` and never in a per-step read.

Fit: from **2×H200 / 1×B300**, as 0731 (the tower moves neither boundary;
asserted).

## 4. NVFP4 — not priced

No NVIDIA or RedHatAI NVFP4 checkpoint exists (Hub search 2026-10-02: only
community repacks, `s-zaizen/…-NVFP4`, `msuiche/…-NVFP4`). The routed experts
already ship 4-bit with E8M0 block-32 scales; NVIDIA's repack of 0731 to E4M3
block-16 came out 5.2% heavier (`research/nvfp4_2026-09.md`), and its repack
of DSv4.1-Flash 3.3% heavier (`research/nvfp4_2026-10.md`). The policy
DSv4.1-Flash had until that checkpoint was measured: greyed out rather than
projected. (A projection would be 0731's NVFP4 total + 932,836,352 B, should
anyone want one.)

## 5. What is not modelled

- **Images.** The workload is text. An image costs a ViT forward (0.47B
  params, ≤ 384 tokens per image after the 3×3 downsample) plus its tokens in
  the backbone prefill, and the image span must be prefilled in a single
  chunk (`model.py` assert; vLLM forces `--disable_chunked_mm_input` for the
  bidirectional image attention). Image tokens then sit in the
  compressed caches like any other token. `ws test` can already send images
  (`image_share` in `[workload]`, `--api chat`) and split warm-turn TTFT by
  whether a turn carried one; the model does not price them.
- **Gate bias for image tokens** (`bias_vl`): changes which experts image
  tokens route to, not how many; no effect on the expert-union arithmetic for
  text.
- Everything listed in `research/model_dsv4flash.md` § 6 applies unchanged.

## 6. Quality

Terminal-Bench from Artificial Analysis, read 2026-10-02 from the
`deepseek-v4-flash-vision` page's embedded dataset ("DeepSeek V4 Flash
Vision (Max)", AA release date 2026-08-21): **2.1 = 0.741573 = 198/267
(74.2%)**, **4.0 = 0.121212 = 24/198 (12.1%)**. Below 0731 on 2.1 (210/267)
and equal on 4.0. DeepSeek's card puts it *above* 0731 on 2.1 (83.9 vs 82.7,
DeepSeek Harness minimal mode); per the one-lab rule
(`research/terminal_bench.md` § 1) the card is not used.

## Sources

- https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp — `config.json`,
  `inference/config.json`, `inference/model.py`, `inference/vision.py`,
  `README.md` (serve commands: vLLM TP4 on 4×GB300 with
  `--kv-cache-dtype fp8 --block-size 256` and DSpark
  `{"method":"dspark","num_speculative_tokens":3,"draft_sample_method":"probabilistic","enable_adaptive_verification":true}`;
  SGLang `--speculative-algorithm DSPARK`), `model.safetensors.index.json`
  (`total_size` 167,811,372,792) + the 48 shard headers ·
  https://huggingface.co/api/models/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp
  (createdAt 2026-08-31, dtype histogram)
- https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731 — `inference/model.py`
  and `inference/config.json`, for the diff in § 1
- vLLM 0.30.0 wheel (PyPI): `vllm/v1/core/kv_cache_utils.py`,
  `vllm/models/deepseek_v4/attention.py`, `compressor.py`
- https://artificialanalysis.ai/models/deepseek-v4-flash-vision (read 2026-10-02)
