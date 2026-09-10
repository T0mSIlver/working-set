# KV cache across tensor-parallel ranks — replication, and the layout the study prices

**Purpose:** state what a replica group's `tp` ranks actually store when the
model's KV cache has fewer shardable heads than ranks, price both layouts,
and say which one every published number assumes. Opened by codex finding
F5 on the DeepSeek-V4.1-Flash swap (PR #70, 2026-09-10): the pool arithmetic
`tp × VRAM − weights − tp × reserve` counts the cache once per group, which is
only true of a layout that shards it.

## 1. The mechanism

Tensor parallelism splits attention by **head**. A cache with `h` KV heads
shards across up to `h` ranks; on a group wider than that, each of the
`tp / h` rank groups beyond the heads holds a **full copy** of every token's
cache and re-reads it on every decode step. For the single-latent caches
(MQA / MLA: one 512–576-byte latent per token shared by every query head) `h`
is 1, so plain `--tensor-parallel-size N` stores **N copies** — the reason
DeepSeek-class serving moved to attention-DP + expert-EP ("wide-EP"), and the
reason vLLM added **decode context parallelism** (DCP,
`--decode-context-parallel-size`), which splits those copies along the
*sequence* so each rank stores `1/dcp` of every request's cache.

vLLM's constraints (DCP blog, 2026-08-07): MLA needs `tp % dcp == 0`; GQA
needs `(tp // num_kv_heads) >= dcp`. Speculative decoding under DCP was "in
development" at that date and prefill/decode disaggregation "being
hardened" — the two caveats the explorer's tooltip carries.

## 2. Heads per model (from the model notes)

| Key | Model | Cache | `kv_heads` | Recurrent / fixed state | `state_heads` |
|---|---|---|---|---|---|
| `27B` | Qwen3.8-27B | GQA 24/4 on 16 full-attention layers | 4 | DeltaNet, 48 v-heads | shards (None) |
| `35BA3B` | Qwen3.6-35B-A3B | GQA 16/2 on 10 layers | 2 | DeltaNet, 32 v-heads | shards |
| `MM35` | Mistral-Medium-3.5 | GQA 96/8 on 88 layers | 8 | — | — |
| `GLM52` | GLM-5.3 | MLA, one 576-B latent | **1** | — | — |
| `DSV41F` | DeepSeek-V4.1-Flash | MQA, one 512-dim latent per cache | **1** | latent windows + compressor state | **1** (replicates with the cache) |
| `Q38FN` | Qwen3.8-Flash-Next | GQA 24/2 on 12 QSA layers | 2 | DeltaNet, 48 v-heads | shards |
| `GLM53F` | GLM-5.3-Flash | NoPE sparse-MLA, one latent | **1** | KDA, 64 heads | shards |

The DeltaNet / KDA states shard by value head at every width the study
prices (≤ 8), so only DeepSeek-V4.1-Flash's fixed per-session state — the
128-entry windows of the same single latent — replicates.

## 3. The two layouts, as priced

`kv_replication(model, topo)` returns `(r_kv, r_state)`:

- **`kv_shard = "dcp"`** (default): `r = 1` at every `tp`. The group stores
  one copy; the deploy recipe emits `--decode-context-parallel-size
  floor(tp / kv_heads)` whenever that is ≥ 2. This is the layout every
  published number has always assumed — nothing moves.
- **`kv_shard = "replicate"`** (plain TP): `r = max(1, tp / kv_heads)` on the
  cache (`kv_bpt`, `kv_decode_bpt`, `kv_decode_const`) and
  `max(1, tp / state_heads)` on the state (`deltanet_state`,
  `state_step_bytes`). Applied inside `kv_pool_tokens`, `warm_capacity`
  (both budgets: vLLM's native `--kv-offloading-size` is **rank-local**,
  each rank spilling its own blocks, so a replicated cache is replicated in
  host memory too — a deduplicating store that kept one copy is not
  modelled; codex F1) and `decode_curves` (every rank re-reads its copy per
  step). Nothing returns the multiplied model, so it cannot be applied
  twice. The topology name gains ` [KV replicated]`.

At an odd TP width the heads do not divide (TP6 on 4 heads): vLLM refuses
the width outright; the study keeps pricing it as sharded under "dcp"
(an extrapolation, unchanged from before this note) and charges the
continuous ratio (1.5×) under "replicate". The deploy recipe says so on
those widths — it cannot emit a command that realizes the priced layout
(codex F2).

**Speculative decoding under DCP** had no vLLM support at the cited date,
so the recipe never emits `--decode-context-parallel-size` and
`--speculative-config` together: with DCP ≥ 2 and the MTP slider above
1.0 the speculative flag is withheld and the recipe says why. The page's
numbers keep the slider's multiplier — the MTP transplant is an unmeasured
headroom assumption everywhere in the study, and the slider is where a
reader takes it out (codex F3).

**One approximation left in place (codex F6):** Qwen3.8-Flash-Next's
cache has two components with different head counts — the GQA main K/V
(2 heads, 12,288 B/token) and the compressed indexer keys (1 head,
384 B/token). A single `kv_heads = 2` under-replicates the indexer: at
TP8 the exact figures are 13,056 B/token under DCP (priced 12,672) and
52,224 under plain TP (priced 50,688), 0.5–3% on the pool and 2.6% on the
replicated per-user speed at n = 64. Splitting the two components would
add a second head field to every model for one model's 3%; recorded
instead.

## 4. What replication costs (reference workload, fp8 KV)

| Configuration | Pool, sharded | Pool, replicated | Warm p5, sharded → replicated |
|---|---|---|---|
| GLM-5.3 · 8×B300 TP8 | 27.3M tok | 3.4M tok | 957 → 105 sessions |
| DeepSeek-V4.1-Flash · 8×B300 TP8 | 1,758M tok | 220M tok | 58k → 7.2k sessions |
| DeepSeek-V4.1-Flash · 8×H200 TP8 | 520M tok | 65M tok | 17k → 2.1k sessions |
| Qwen3.6-35B-A3B · 8×B300 TP8 (2 heads) | ÷ 1 | ÷ 4 | |
| Mistral-Medium-3.5 · any TP ≤ 8 (8 heads) | same | same | — |

(`uv run python -m workingset.model` asserts the ratios; the counts above
are `warm_capacity` at 200 iterations on the reference workload.)

Decode moves with it: a replicated MLA cache on 8 ranks is read 8× per
step, which takes GLM-5.3's per-user p50 at n = 64 on 8×B300 from 23.1 to
20.6 tok/s (asserted as an ordering in `_selfcheck`); on DeepSeek-V4.1-Flash
the cache is so small that even 8 copies stay under the 297 GB expert read,
and its decode ceiling on 8×B300 moves from "≥ 4,096" to 885 only because
the replicated windows (8 × 2.9 MB per session) enter the step.

## 5. What is not modelled

- **Attention-DP + expert-EP** (vLLM `single_node_dep`, DeepSeek's own
  serving): each rank owns its sequences' caches (one copy, like DCP) but
  replicates the attention and shared-expert weights, and shards only the
  routed experts. Per-step reads per rank change shape (full attention
  weights + 1/tp of the expert union); the study's TP bandwidth model does
  not express it. DCP reaches the same cache capacity inside the existing
  TP model, which is why it is the layout priced.
- DCP's own costs: an all-gather of the partial attention outputs per step
  and the caveats in § 1. Not priced.
- Prefill-side replication: irrelevant to the FLOP roofline the prefill
  model prices.

## Sources

- vLLM, *Efficient Decode Context Parallelism for Long Context Workloads*
  (2026-08-07): https://vllm.ai/blog/2026-08-07-decode-context-parallelism —
  "under pure tensor parallelism, this latent KV cache is replicated in full
  on every TP rank"; flags and constraints; Kimi K2.6 on 8×B200 6,091 vs
  1,863 tok/s/GPU at full KV.
- vLLM, *Large Scale Serving: DeepSeek with Wide-EP* (2025-12-17):
  https://vllm.ai/blog/2025-12-17-large-scale-serving — DP attention to
  avoid the KV duplication of MLA under TP.
- DeepSeek-V4.1-Flash `inference/model.py`: `k_cache`, `compress_kv_cache`,
  `window_kv_cache` allocated at full width on every rank while only the
  query heads are `n_local_heads` (codex F5, PR #70).
- Model notes: `research/model_35ba3b.md` (Qwen family heads),
  `research/model_mistral_medium35.md`, `research/model_glm52.md`,
  `research/model_dsv41flash.md`, `research/model_qwen38flashnext.md`,
  `research/model_glm53flash.md`.
