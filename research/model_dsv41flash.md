# DeepSeek-V4.1-Flash (552B + 196B Engram, CED + CSA2, FP4 main KV) — parameterization note

**Purpose:** defensible KV-cache / decode-bandwidth / prefill constants for
**DeepSeek-V4.1-Flash** (`deepseek-ai/DeepSeek-V4.1-Flash`, MIT weights,
released 2026-09-10) as used by `workingset.model` (`MODELS["DSV41F"]`) and the
explorer. It **replaces** DeepSeek-V4-Flash-0731 in the study
(`research/model_dsv4flash.md`, kept as the superseded record).

> **Provenance (2026-09-10, release day):** every primary artifact was read
> directly from huggingface.co: `config.json`, `inference/config.json`,
> `inference/model.py` (DeepSeek's reference implementation — authoritative for
> cache semantics), `model.safetensors.index.json`, the technical report PDF
> in the repo, and the raw safetensors header of **all 48 shards** via HTTP
> range requests (exact dtypes/shapes/byte offsets of the 96,085 tensors). The
> per-tensor byte sum matches `metadata.total_size` **exactly** (§ 4), and the
> config-derived cache arithmetic reproduces the paper's own **890 B/token**
> to the byte (§ 2), the 552B backbone / 196B Engram parameter counts and the
> 8B-prefill / 16B-decode active counts (§ 4, § 6). Serving facts come from
> the vLLM recipe (`recipes/models/deepseek-ai/DeepSeek-V4.1-Flash.yaml`,
> added 2026-09-09). Codex (gpt-5.6-sol, high) reviewed the derivation
> against the same cached sources on 2026-09-10; its six findings are
> applied (two constants: the per-step state traffic and the indexer's
> quadratic width) or recorded in § 6 (three provenance / convention
> points, one ledger typo).
>
> **What the reference implementation is and is not evidence for.**
> `inference/model.py` is a minimal single-process reference: it allocates
> every cache in **BF16** and its `fp4_act_quant(..., inplace=True)` /
> `act_quant(..., inplace=True)` calls quantize-then-dequantize into that
> BF16 tensor, discarding the scales. Read literally, it stores 1,280 B per
> compressed entry (3,200 B/token) and 5.66 MB of windows per session. It
> is therefore authoritative for **which tensors are cached, their shapes,
> which layers own them, and the quantization groups** (E2M1 per-16 with
> E4M3 scales on the latent, per-32 with E8M0 on the indexer keys and the
> fp8 windows) — and the technical report (§ 2.4.4, "FP4 main KV cache",
> "890 bytes per token") is the authority for the **packed byte layout** a
> production stack stores. The constants below price the packed layout.

## 1. Architecture table (config.json + inference/model.py + tech report § 2)

| Field | Value |
|---|---|
| `architectures` / `model_type` | `DeepseekV41ForCausalLM` / `deepseek_v41` (multimodal: `text_config` + `vision_config`) |
| `num_hidden_layers` | **40** = a 20-layer **causal encoder** (0–19) + a 20-layer **decoder** (20–39), plus 3 DSpark stages `mtp.0–2` (`num_nextn_predict_layers: 3`, layers 40–42 in `compress_ratios`) |
| `hidden_size` / `vocab_size` | 5120 / 129,280 |
| Attention | MQA over one **512-dim latent** (K = V = the latent, `num_key_value_heads` 1), 64 Q heads, `head_dim` 512 (448 nope + 64 rope), `q_lora_rank` 1280, `o_lora_rank` 1024 grouped ×8 |
| `compress_ratios` (per layer, MTP incl.) | `[0, 0, 2×18, 1×20, 0, 0, 0]` — layers 0–1 sliding-window only; encoder 2–19 **CSA2 ratio 2**; decoder 20–39 **CSA2 ratio 1** (uncompressed main KV); DSpark stages window-only. No HCA layers: "pure CSA2" |
| `kv_source_layer_ids` | **[2, 8, 14, 20]** — only these four layers compress their own KV and own an indexer-K cache ("Full Mode"); every other CSA2 layer reads the most recent source's cache (tech report § 2.3.1: Reindex / Reuse modes) |
| `index_source_layer_ids` | [2, 8, 14, 20, 24, 28, 32, 36] — 8 indexers (32 heads × 128 dims, top-512); 24/28/32/36 are "Reindex" layers restricted to the **candidate pool** |
| Hierarchical indexer | `candidate_source_layer_id` 20, `candidate_topk_blocks` 2048 × `candidate_block_size` 8 = **16,384 candidate positions**; the decoder's later indexers score only those (cost independent of context length) |
| `sliding_window` | **128** — a per-layer fp8 ring buffer on all 40 (+3) layers |
| `n_routed_experts` / `num_experts_per_tok` / shared | **384 / 6 / 1**, `moe_intermediate_size` 2304, no dense FFN layers; `noaux_tc` with a separate `bias_vl` for image tokens |
| Engram | 2 modules (layers 1, 14): hash tables of 384,006,168 and 384,016,682 rows × 256 dims (FP8 + E8M0 scales), n-gram orders 2–4 × 8 heads |
| `max_position_embeddings` | **1,048,576** (YaRN ×16 over `original_max_position_embeddings` 65,536; `compress_rope_theta` 160,000) |
| DSpark | 3 draft stages (128 routed experts / 3 active each), `dspark_block_size` **5**, Markov + confidence heads, adaptive verification |
| Vision | DeepSeek-ViT 32 layers × 1024 (0.49B params), 3×3 pixel-unshuffle, ≤ 1024 tokens/image; never executed on the study's text workload |
| Totals | **552B backbone + 196B Engram** (paper); 510.29 GB on disk (§ 4); active **8B prefill / 16B decode** (paper; reproduced § 6) |
| Checkpoint dtype | native mixed: routed experts **MXFP4** (I8-packed E2M1 + E8M0 block-32 scales), all other projections **FP8** (block 32×32, ue8m0), Engram tables FP8, embed/lm_head/gates/compressors/indexer-K BF16, mHC/sinks/gate biases FP32 |

The two ideas that set the constants (tech report § 2.2–2.3):

- **CED.** The decoder's global KV (the layer-20 cache that layers 20–39
  share) is projected from the **encoder's final hidden state** (`layers.20.
  attn.compressor.wkv` reads the input of layer 20), so a prompt token never
  has to run the decoder: prefill computes layers 0–19 only. The decoder's own
  sliding-window KV is rebuilt by **Decoder SWA Bounded Replay** of the last
  128 tokens at every prefill — a fixed cost, not per token.
- **CSA2 cross-layer reuse.** Four compressed caches serve 38 attention
  layers. The indexer keys are projected from the cached latent (not from the
  hidden state), so they are shared with it.

## 2. KV bytes per token — four shared FP4 caches: 890 B, the paper's figure

Cached-entry layout (tensors and quantization groups from `inference/model.py`;
packed byte widths from the technical report § 2.4.4, which the reference
implementation dequantizes into BF16 — see the provenance note):

- **Main KV latent**, 512 dims, quantized *after* RoPE to **E2M1 with one E4M3
  scale per 16 channels** (`fp4_act_quant(latent, 16, True, scale_dtype=
  float8_e4m3fn)`, tech report § 2.4.4 "following NVFP4 but omitting its
  second-level global scale"): 256 + 32 = **288 B/entry**.
- **Indexer K**, 128 dims, E2M1 with one **E8M0 scale per 32** (`fp4_act_quant
  (k, fp4_block_size=32, True)`): 64 + 4 = **68 B/entry**.
- **Sliding-window KV**, 512 dims **FP8** with one E8M0 scale per 32 ("retain
  FP8 for the SWA KV cache due to its sensitivity to quantization"): 512 + 16
  = **528 B/entry**, 128 entries per layer, fixed.

```
kv_bpt (per ORIGINAL token; only the four source caches grow):
  layers 2, 8, 14  ratio 2:  3 x (288 + 68) / 2  =  534
  layer 20         ratio 1:      (288 + 68)      =  356
  -----------------------------------------------------
  TOTAL                                          =  890 B/token   (= the paper's "890 bytes per token")

deltanet_state (fixed per resident session, reused field — not DeltaNet):
  windows   43 layers x 128 x 528 B (40 main + 3 DSpark) =  2,906,112
  fp32 compressor partial-group state: 3 ratio-2 compressors x
    (kv_state + score_state) x 2 slots x 512 x 4 B       =     24,576
  -----------------------------------------------------
  TOTAL                                                  =  2,930,688 B ≈ 2.8 MiB/session
```

A 262k-token session holds **222 MiB** (0731: 0.84 GiB; GLM-5.2: 11.8 GiB);
the full 1M context holds 0.87 GiB. The paper's "roughly 1/4 of V4-Flash" is
890 vs the 0731 card's 3,450 (3.9×) — consistent. Persistent (SSD) footprint
is lower still (SWA KV is never persisted, § 5); the study charges HBM-resident
sessions, where the windows exist.

**FP16-KV toggle disabled** (`kv_fp16_ok=False`): the FP4 main KV is the
model's *trained* cache format (QAT introduced in post-training, § 2.4.4); no
serving stack offers a BF16 main KV for it, and the vLLM recipe carries no
`--kv-cache-dtype` flag at all. The explorer's "FP8" arm therefore prices this
native FP4-main / FP8-window layout — this is the model's own format, **not**
the owner-policy `nvfp4` KV option that the study declines to model on other
models (`research/nvfp4.md` § 3).

## 3. Decode-bandwidth model — four scans, top-512 per layer, a candidate pool

Per decode step a query reads: (a) on the four Full-mode layers, the indexer-K
cache over the whole compressed axis (the scan); (b) on the four Reindex
layers, the indexer-K rows of the 16,384-position candidate pool (a constant
— the deployment the report describes, § 2.3.2: "changes the per-query cost
of deeper indexers from linear in context length to constant"; the reference
implementation instead scores every position and masks afterwards, § 6);
(c) on every CSA2 layer (2–39), its top-512 selected latent entries; (d) on
every layer, the 128-entry window.

```
kv_decode_bpt   = 3 x 68/2 + 68                    = 170 B per CONTEXT token per step
kv_decode_const = 38 x 512 x 288   (top-512 latent reads, layers 2-39) =  5,603,328
                + 4 x 16,384 x 68  (candidate-pool indexer reads)      =  4,456,448
                + 40 x 128 x 528   (window reads, main layers)         =  2,703,360
                                                                       = 12,763,136 B per ACTIVE SEQ per step
kv_decode_topk  = 1,024 original tokens (512 compressed entries x ratio 2 — the
                  encoder majority; the decoder's ratio-1 layers saturate at 512);
                  sequences shorter scale the constant by min(len, 1024)/1024
state_step_bytes = 40 x 528 (one ring slot written per main layer)      =  21,120
                 + 3 x 2 x 2 x 512 x 4 (compressor partial-group r/w)   =  24,576
                                                                        =  45,696 B per ACTIVE SEQ per step
```

`state_step_bytes` is new with this model: the study's DeltaNet models stream
their whole recurrent state twice per step (read + write, `2 x deltanet_state`,
the default when the field is unset). Here `deltanet_state` is fixed
*storage* whose reads are already inside `kv_decode_const`; charging 2 x 2.9 MB
on top would count the windows twice (codex finding F1).

At the reference 31k-median workload the scan is ~5.3 MB/seq and the constant
12.8 MB/seq — against 27.6 MB/seq if decode streamed the whole 890 B/token
cache, and ~1.1 GB/seq for a dense-attention model with V3-class MLA. The
candidate-pool component saturates only at 16,384 tokens: between 1k and 16k
tokens the constant over-charges it by ≤ 4.5 MB/seq (conservative, § 6).
Layers sharing one cache and one top-k set re-read the same 147 KB from HBM in
this accounting (sequential layers separated by hundreds of MB of weight
traffic; no L2 credit is taken), as the 0731 card did. Engram row lookups
(48 rows × 264 B = 12.7 KB/token) are omitted from the per-step read (1.5e-6
of the shared read per decoding sequence).

## 4. Weight bytes (all 48 shard headers, byte-exact)

```
attention (wq_a/wq_b/wkv/wo_a/wo_b, q/kv norms, sinks; FP8) 5,069,721,600
layer norms (attn_norm + ffn_norm, 40 layers; BF16)             819,200
compressors (4 sources; BF16)                                36,704,256
indexers (8; wq_b FP8, rest BF16)                            45,130,752
shared experts (40 x 3 x 2304x5120; FP8)                  1,416,960,000
gates (BF16 + fp32 biases)                                  157,409,280
hyper-connections (mHC, FP32)                               157,295,040
engram projections (2 x wkv 25600x6144 FP8 + q/k)           315,043,840
engram tables (768,022,850 rows x 264 B FP8+scale)      202,758,032,400
routed experts (40 x 384 x 18,800,640 B MXFP4)          288,777,830,400
embed (BF16)                                              1,323,827,200
lm_head (BF16)                                            1,323,827,200
final norm                                                       10,240
DSpark stages (3; 128 MXFP4 experts each)                 7,932,874,632
vision + aligner (BF16)                                     970,536,960
-------------------------------------------------------------------------
w_resident  = metadata.total_size                       510,286,023,000 B  (475.2 GiB)
```

Per routed expert: three MXFP4 matrices (w1/w3 2304×5120, w2 5120×2304 packed
to 5,898,240 B each) + E8M0 block-32 scales (368,640 B each) = **18,800,640 B**
(6.25% scale overhead, charged). Params: backbone 551.9B ✓ "552B"; Engram
196.6B ✓; routed experts 543.6B (40 × 384 × 35,389,440).

```
w_route_pertok  = 6   x 18,800,640 x 40 =   4,512,153,600 B
w_route_total   = 384 x 18,800,640 x 40 = 288,777,830,400 B   (= the ledger line exactly)
w_decode_shared = attn + compressors + indexers + shared experts + gates + mHC
                + engram projections + norms + lm_head        = 8,522,921,408 B
```

Expert-union saturation at n = 384/6 = **64** (an integer again, unlike 0731's
42.7). Active-parameter check: decode 16.13B incl. lm_head (5.06 attn + 1.42
shared + 8.49 routed + 0.08 gates + 0.04 mHC + 0.32 engram proj + 0.66 head)
✓ "16B".

**The Engram tables decide the fit.** They are 40% of the resident bytes, and
the vLLM recipe keeps them in HBM ("plan capacity for them, they dominate
everything except the experts"; `vram_minimum_gb: 614`; verified TP4 on a
GB200 NVL4 tray and 8×H200/TP8). Under the study's pool arithmetic (141 GB −
19.3 GB reserve per H200; 288.4 GB − 29.1 GB per B300) the minimum is **5×H200
(so TP8 on a node) and 2×B300** — where the 0731 model fitted from 2×H200 /
1×B300. DeepSeek's own deployment prefetches Engram rows from host DRAM over
RDMA (tech report § 2.4.2) — a 307.5 GB resident checkpoint that would fit
from 3×H200 / 2×B300 — but that path exists in no open stack, so it is not
modelled (§ 6). The ViT (0.97 GB) is charged although `--language-model-only`
drops it (0.2%, same convention as Qwen3.8-Flash-Next's tower).

### NVFP4 — none (`nvfp4_w = None`)

No official NVFP4 checkpoint exists on release day (three community repacks
were created 2026-09-10). The routed experts already ship 4-bit with E8M0
block-32 scales; NVIDIA's repack of the 0731 predecessor to E4M3 block-16
scales came out **5.2% heavier** (`research/nvfp4_2026-09.md`), and nothing in
this checkpoint changes that arithmetic. The explorer greys the option out
rather than project a heavier arm.

## 5. Serving notes

- **vLLM ≥ 0.30.0**, Docker image `vllm/vllm-openai:deepseekv41-flash-0909`
  only (no pip wheel), `--tokenizer-mode deepseek_v41`; verified on H200,
  GB200, GB300, MI350X. Strategies: single-node TP/TEP/DEP, multi-node,
  PD-disaggregated (1P1D on GB200 NVL4, TP4 each). Vision encoder separable
  (`--mm-encoder-tp-mode data`) or skipped (`--language-model-only`).
- **Speculative decoding: DSpark**, `num_speculative_tokens: 5` (the trained
  block size) with confidence-scheduled adaptive verification. The study keeps
  its 1.7× **transplanted** fit (module present, acceptance unmeasured on this
  workload).
- **Prefill under CED** (tech report § 3.2.2): the encoder runs over the
  uncached suffix (+ the last 128 cached tokens when the encoder SWA state is
  missing — Encoder SWA Bounded Replay); the decoder runs over the last 128
  tokens only. Global KV persists for ≥ 72 h; SWA KV lives in a host-DRAM pool
  with a minutes-scale TTL and is never written to SSD.
- Positioning: the paper reports Terminal-Bench 2.1 **90.6** (DeepSeek
  harness, max effort), DeepSWE v1.1 74.2, vs 0731's 82.7 / 54.4 on the same
  card. **No Artificial Analysis run exists yet** — the frontier's quality
  axis (`research/terminal_bench.md`) takes one lab under one protocol, so the
  model is *unscored* there until AA publishes (`CONFIG.QUALITY.DSV41F.tb21 =
  null`; the chart reports it as unscored).

## 6. Remaining assumptions / re-verification ledger

- **Engram placement**: charged in HBM (the vLLM recipe's deployment). If a
  stack ships host-resident Engram with RDMA prefetch, `w_resident` drops by
  202.8 GB and the H200 minimum falls from 5 to 3 GPUs. Biases capacity DOWN
  on Hopper (conservative); the decision-relevant B300 minimum (2) does not
  move.
- **Prefill = encoder only** (`params_prefill = 7.90e9`: encoder attention
  2.53 + shared 0.71 + routed 4.25 + gates/mHC 0.06 + engram projections 0.31
  + the layer-20 CED projection 0.003, excl. embed/lm_head ✓ paper "8B"). A
  stack that ran all 40 layers over the prompt would double it (15.9e9). The
  128-token decoder replay per prefill (~2 TFLOP) and the encoder replay on an
  SWA-state miss are fixed costs, left out: prefill priced cheaper, biased
  AGAINST the thrash hypothesis, per `research/prefill.md` convention.
- **Prefill quadratic term** `attn_layers=3, attn_d=1,024`: the three encoder
  Full-mode indexers (layers 2/8/14) score the full ratio-2 compressed axis at
  32 heads × 128 dims — 4,096 MACs per (query, compressed position), ÷ 2 for
  the ratio, ÷ 2 again because the study's `2 T² d` convention prices QK
  **and** AV and an indexer has no AV (a first draft wrote 2,048 and priced
  the term twice; codex F2). At a 32k chunk the whole quadratic term is 2%
  of the prefill FLOPs. The top-512 + window attention on all 18 encoder
  CSA2 layers is *linear* per token and left out; layer 20's indexer runs
  only in decode (and the replay). Same bias direction as above.
- **Candidate-pool indexing modelled as the paper's deployment, not the
  reference code.** `inference/model.py`'s Reindex layers compute scores over
  the *whole* indexer-K axis and only then mask to the candidate pool. Priced
  that way, the scan would be 3 × 68/2 + 5 × 68 = **442 B/ctx-token** with no
  4.46 MB candidate constant (8.31 MB/seq). The report's stated design and
  the reason the hierarchy exists is the gathered, constant-cost form, so 170
  B + 4.46 MB is modelled; at the 31k reference workload the two differ by
  4 MB/seq against a ≥ 297 GB weight read per step (< 0.1% at n = 64).
- **Single-KV-head caches under tensor parallelism.** Every cache here is
  one latent per position (MQA), and the reference implementation keeps
  the full `compress_kv_cache` / `k_cache` / `window_kv_cache` on every
  rank while splitting only the query heads. The study's pool arithmetic
  (`tp × VRAM − weights − tp × reserve`) assumes a deployment where each
  rank owns the caches of its *own* sequences — the attention-DP /
  expert-EP layout the vLLM recipe lists as `single_node_dep` and
  DeepSeek's own serving uses. Under plain TP the caches replicate and the
  pool divides by `tp` (8×H200: 520M → 65M tokens; 2×B300: 9.4M → 4.7M). The
  same convention already prices GLM-5.3 (MLA), GLM-5.3-Flash and the 0731
  model. **Resolved (research/kv_tp_sharding.md):** the study now carries
  the layout explicitly — `kv_heads=1, state_heads=1` here — with the
  default "dcp" layout pricing one copy (vLLM decode context parallelism,
  emitted in the recipe) and a "replicate" arm pricing plain TP's copies
  (8×B300: pool 1,758M → 220M tokens; the windows replicate too, so the
  decode ceiling there moves from "≥ 4,096" to 885). It moves no decision
  under the default: the cache never binds, the fit thresholds are
  weight-set, and the decode plateau is a weight read.
- **Cached-entry scale bytes are charged** (E4M3/16 on the latent, E8M0/32
  on the indexer keys and windows) — the layout `fp4_act_quant` writes. A
  stack storing scales elsewhere or at a different granularity moves
  `kv_bpt` by ±10%.
- **Windows counted for all 43 layers in HBM** (a GPU-resident session holds
  them; the study never priced the SSD tier). The fp32 compressor state
  (24.6 KB) is negligible either way.
- **`kv_decode_topk` = 1,024** maps the encoder's ratio-2 top-512 onto the
  study's token-space rule; the decoder's ratio-1 layers saturate at 512 and
  the candidate pool at 16,384. Sequences between 512 and 16k tokens are
  over-charged by < 4.5 MB/seq (conservative).
- **Cross-layer cache re-reads charged per layer** (no L2 credit), as in the
  0731 card. Under a kernel that kept the shared top-512 gather resident
  across a Reuse-mode run the constant would fall by up to 5.5 MB/seq.
- **DSpark-stage window reads and Engram lookups** folded into the MTP
  speedup / omitted, as everywhere in this study.
- **MTP 1.7× transplanted**; DSpark's 5-token block with adaptive
  verification could sit well above it (the slider covers it).
- The `w_decode_shared` ledger is an exact per-tensor sum (no sampled residual
  this time); the reference implementation promotes the ratio-2 compressors
  to fp32 and `wo_a` to bf16 at runtime — the checkpoint's bf16/fp8 bytes are
  charged, as "the explorer prices what the checkpoint weighs".

## Sources

Primary (read directly, exact bytes, 2026-09-10):
- https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/config.json ·
  https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/inference/config.json
- https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/inference/model.py
  — cache semantics: `Compressor` (ratio-1 bf16 projection, ratio-2 fp32
  pooling with `kv_state`/`score_state`), `Indexer` (`owns_k` on
  `kv_source_layers`, `k_cache` at `max_seq_len // ratio`, `fp4_act_quant(k,
  32)`, `select_candidate_blocks`), `Attention._window_kv` (fp8 ring of
  `window_size`), `Attention._compress_kv` (`fp4_act_quant(latent, 16, True,
  scale_dtype=float8_e4m3fn)`, `compress_kv_cache` only on sources,
  `shared_attn.compress_kv` read by the rest)
- https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/model.safetensors.index.json
  (`total_size` 510,286,023,000) + the 48 shard headers via HTTP range requests
  (96,085 tensors: I8 278.6e9, F8_E4M3 204.0e9, F8_E8M0 23.6e9, BF16 3.95e9,
  F32 0.17e9 — sum equals `total_size`)
- https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf
  — § 2.2 CED, § 2.3 CSA2 modes and the hierarchical indexer, § 2.4.2 Engram
  (196B, host prefetch), § 2.4.3 DSpark, § 2.4.4 FP4 main KV (E2M1 + E4M3/16,
  windows stay FP8), § 3.2 inference system and SWA Bounded Replay
- https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/README.md
  (552B / 8B-16B, 890 B/token, benchmarks, MIT license)
- https://huggingface.co/api/models?search=DeepSeek-V4.1-Flash (no `nvidia/`
  NVFP4 repo; three community repacks created 2026-09-10)

Secondary:
- vLLM recipe: https://github.com/vllm-project/recipes/blob/main/models/deepseek-ai/DeepSeek-V4.1-Flash.yaml
  (min version, Docker-only install, Engram in HBM and the 511 GB breakdown,
  `vram_minimum_gb` 614, DSpark 5 drafts, verified hardware and strategies)
- https://artificialanalysis.ai/models (no DeepSeek V4.1 entry as of
  2026-09-10)
- Predecessor card: `research/model_dsv4flash.md` (0731 constants, superseded)
