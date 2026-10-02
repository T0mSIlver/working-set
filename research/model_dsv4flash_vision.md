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
> (`research/model_dsv4flash.md`) the same day (§ 1). A vLLM 0.30.0 startup
> log of the model on 2×B300 TP2 serves as an outside check (§ 5).

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
text, and the study does not price image encoding (§ 6).

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
block-16 came out 5.2% heavier (`research/nvfp4_2026-09.md`). Same policy as
DSv4.1-Flash: greyed out rather than projected. (A projection would be
0731's NVFP4 total + 932,836,352 B, should anyone want one.)

## 5. Outside check: a vLLM 0.30.0 startup log on 2×B300

A production deployment's startup log (2026-09-30; employer infrastructure —
the instance is not identified here) serves the model with `--tensor-parallel-size 2
--kv-cache-dtype fp8`, **no `--speculative-config`** and no
`--decode-context-parallel-size`, at `gpu_memory_utilization` 0.92 and
`max_model_len` 1,048,576. Per GPU:

| | vLLM log | model (`kv_pool_tokens`, 2×B300 TP2) |
|---|---|---|
| total HBM | 267.69 GiB | 268.59 GiB (`vram`) |
| weights | 74.28 GiB ("Model loading took") | 78.14 GiB (`w_resident` / 2) |
| everything else | 33.26 GiB (2.56 non-torch + 9.28 peak activation + 21.41 left by the 0.92 utilization) | 27.06 GiB (`ACT_RESERVE` + B300 `reserve_extra`) |
| **KV pool** | **160.15 GiB** ("Available KV cache memory") | **163.39 GiB** |

The pool bytes agree within **2.0%**, from two errors that partly offset:

- **Weights, −3.9 GiB.** Without a speculative config vLLM does not load the
  10.86e9 B of DSpark stages: half the main text weights (156,015,745,244 /
  2) plus the whole tower on each rank comes to 73.5 GiB, the remaining
  ~0.8 GiB being tensors vLLM replicates rather than shards. The study
  charges the stages always (§ 3); with speculation on, the log's figure
  would rise by ~5 GiB.
- **Reserve, +6.2 GiB.** The study's reserve is solved from the 1×H200 27B
  anchor (`_act_reserve`) and carried to B300 with its measured correction
  (`research/gpu_b300.md`); this deployment's 0.92 utilization leaves its
  own headroom (21.4 GiB of the 33.3) that the anchor does not know about.

**The token figure is not comparable.** The log also reports "GPU KV cache
size: 10,319,638 tokens, Maximum concurrency for 1,048,576 tokens per
request: 9.84x". In vLLM 0.30.0 that number is `concurrency × max_model_len`
(`vllm/v1/core/kv_cache_utils.py`, `get_kv_cache_capacity`), where
concurrency is the shared block pool divided by the blocks one
max-length request needs summed over all five cache groups (group block
sizes 64, 64, 256, 4, 8). It implies ~16.7 KB per token per GPU, against the
study's 3,450 B/token (each GPU holds a full copy under plain TP: the model's
"Replicated" arm prices 50.9M tokens, its default sharded arm 101.7M). The
study did not reconcile vLLM's per-group page accounting with the
reference-implementation layout. Two readings fit the log: either vLLM's
paged layout really costs ~4.8× the reference bytes per token (page padding
across groups, the compressor-state and fp8-indexer groups), in which case
the study overstates what this vLLM version admits, or vLLM's capacity
estimate is conservative and the allocator admits more. Settling it takes a
load test against the endpoint (`ws test --exclusive`), not more reading. **Open.**

## 6. What is not modelled

- **Images.** The workload is text. An image costs a ViT forward (0.47B
  params, ≤ 384 tokens per image after the 3×3 downsample) plus its tokens in
  the backbone prefill, and the image span must be prefilled in a single
  chunk (`model.py` assert; vLLM forces `--disable_chunked_mm_input` for the
  bidirectional image attention, per the log). Image tokens then sit in the
  compressed caches like any other token. `ws test` can already send images
  (`image_share` in `[workload]`, `--api chat`) and split warm-turn TTFT by
  whether a turn carried one; the model does not price them.
- **Gate bias for image tokens** (`bias_vl`): changes which experts image
  tokens route to, not how many; no effect on the expert-union arithmetic for
  text.
- Everything listed in `research/model_dsv4flash.md` § 6 applies unchanged.

## 7. Quality

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
