# Mistral Large 4 (1.05T-A49B) — architecture parameterization note

**Purpose:** the constants behind the `ML4` row of `src/workingset/model.py`
and its mirror in `interactive/src/config.js`.

**Status (2026-11-04):** open weights since 2026-10-29, integrated as `ML4`.
This note replaces the release-day note of 2026-10-06 (PR #106), which kept
the model out until a checkpoint existed; § 9 says what that note got right.

Sources, all read 2026-11-04:

| Source | Revision | Read |
|---|---|---|
| [`mistralai/Mistral-Large-4-1T-A52B`](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B) | `af26e786` | `config.json`, `params.json`, the safetensors index, **all 54 shard headers**, README, LICENSE, technical report |
| [`mistralai/Mistral-Large-4-1T-A52B-NVFP4`](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B-NVFP4) | `bb6d00fe` | `params.json`, the index, **all 32 shard headers**, README |
| vLLM recipe `recipes/models/mistralai/Mistral-Large-4-1T-A52B.yaml` | `main`, `date_updated: 2026-10-30` | flags, hardware floors, measured capacities and latencies |
| vLLM [#64534](https://github.com/vllm-project/vllm/pull/64534) "Add Mistral Large 4" (open working branch) | head `28a4495d` | `vllm/models/mistral/{common,nvidia}/…`, `flashmla_sparse.py`, `kv_cache_interface.py` |
| Artificial Analysis model and Terminal-Bench 4.0 pages | — | the page's embedded dataset (§ 8) |

## 1. Architecture

| Field | Value | Source |
|---|---|---|
| `architectures` / `model_type` | `MistralLarge4ForConditionalGeneration` / `mistral_large4` (Pixtral vision tower + text decoder) | config |
| Layers / hidden | **72** / 6144 | config, params |
| Attention | **MLA + DSA on all 72 layers** (`layer_types` all `indexed_attention`): `kv_lora_rank` 512, `qk_rope_head_dim` 64, `qk_nope_head_dim` 64, `v_head_dim` 128, 64 heads, no q-LoRA | config |
| NoPE layers | every 4th (3, 7, …, 71: 18 layers); vLLM keeps the 576-wide latent row with a zero rope slot | `no_rope_layers`; vLLM `attention.py` docstring |
| **Cross-layer KV sharing** | layers 39, 43, …, 71 (**9**, all NoPE) attend to **layer 35's** latents and write none; **63 layers store a cache** | `kv_source_layers`; vLLM `kv_sharing_target_layer_name` |
| DSA indexer | 32 heads × 128, `index_topk` **2048**, fp8 keys + fp32 block scales = **132 B/token** per indexer | params `dsa`; vLLM `MistralIndexerCache` (`head_dim + head_dim/128 × 4`) |
| **Index sharing** | **44** layers run an indexer (`full`), **28** reuse the top-k of the nearest preceding one (`shared`); the 9 KV-sharing layers all run their own | `indexer_types`; vLLM `layer_spec.py` |
| MoE | 1 dense layer (0, `intermediate_size` 24576), **71** MoE layers: **128 routed experts, top-4**, 1 shared, `moe_intermediate_size` 6144 | config, params `moe` |
| Vocab / embeddings | 204,800, untied | config |
| Context | `max_position_embeddings` **1,048,576** (YaRN ×2048 over 512); the vLLM recipe validates 262,144 | config; recipe |
| Draft model | **DSpark**, embedded: 3 dense MLA layers in BF16, each windowed to 2,048 tokens, depth 3 | README; params `mtp`; vLLM `dspark.py` |
| Logit soft-cap | 30 | config |
| License | **Mistral Ecosystem Open-Weight License**: MIT plus a separate commercial licence for Model-as-a-Service operators; internal use is exempt | `LICENSE` |

**Parameters (shard-header ledger, elements excluding scales):** 1,053.2B
total; 49.82B active per token excluding embeddings and lm_head, **52.34B**
including them, which is the card's "49B active (52B including embeddings and
output layers)" and the repo name's A52B. Vision tower and adapter 1.70B, the
DSpark drafter 2.10B.

## 2. KV bytes per token

vLLM stores each latent as an `fp8_ds_mla` row of 656 B: 512 fp8 values,
16 B of fp32 scales and 64 rope values kept in BF16. That holds on both
parts. FlashInfer's sparse MLA, which Blackwell uses for GLM-5.3, requires
`qk_nope_head_dim` of 128–192, and Large 4's is 64, so FlashMLA serves it
on B300 too. The recipe says so ("FlashMLA sparse attention and an FP8
latent KV cache (`fp8_ds_mla`) by default").

```
kv_bpt (fp8):
  MLA latent  63 layers x 656 B   = 41,328
  indexer     44 layers x 132 B   =  5,808
  -------------------------------------------
  TOTAL                           = 47,136 B = 46.0 KiB/token
```

KV sharing removes 9 latent rows and index sharing 28 indexer rows: 3% below
GLM-5.3 on B300 (48,408 B, whose Blackwell row is 576 B) and 14% below it on
H200 (54,728 B). A 262k session holds 11.5 GiB, the full 1M 46.0 GiB.

The recipe's own measurements confirm the row. On 8 × GB300 at
`--gpu-memory-utilization 0.85`, without speculative decoding, it reports
2.38M (FP8) and 3.60M (NVFP4) KV tokens under TP8, and 18.88M / 28.76M under
DCP8. Solving each pair for one bytes-per-token figure and one per-GPU
overhead gives **46.2 kB** (TP8) and **45.6 kB** (DCP8). The 656-B row
leaves 0.9 and −0.3 GB/GPU of overhead (TP8) and 1.9 and 0.0 GB/GPU (DCP8).
A 576-B row would need 12.9 vs 17.9 GB/GPU and 13.8 vs 18.1 GB/GPU, so the
two checkpoints would need overheads 5 GB apart. (The row was priced at
576 B, GLM-5.3's convention, until issue #122 moved both models to vLLM's
real rows on 2026-11-04.)

**KV dtype.** FP8 only (`kv_fp16_ok=False`): every NVIDIA configuration
Mistral and vLLM document runs `fp8_ds_mla` (the default, so no flag needed),
the NVFP4 card calls `--kv-cache-dtype fp8_ds_mla` "needed for this
checkpoint", and `params.json` sets `fp8_kv_cache: true`. FlashMLA sparse also
accepts `bfloat16`, but only the ROCm recipe documents it as an opt-out, and
the study prices no AMD part. `fp8_ds_mla` runs on SM90 unchanged (recipe), so
the cache is not Blackwell-only.

## 3. Decode reads

```
kv_decode_bpt   = 44 x 132         =      5,808 B per CONTEXT token per step
                  (every indexer scans its own keys; the 28 shared layers do not)
kv_decode_const = 72 x 2048 x 656  = 96,731,136 B per ACTIVE SEQUENCE per step
                  (every layer gathers its own top-2048; the 9 KV-sharing layers
                   gather from layer 35's latents, a separate read each)
kv_decode_topk  = 2048
```

At the 31k median context the indexer scan (~180 MB/seq) is about twice the
sparse gather (97 MB/seq); both are far below a dense read (31k × 46.0 KiB ≈
1.5 GB/seq).

## 4. Weight bytes

### FP8, as shipped

`quant_method: fp8`, block 128 × 128, UE8M0 scales. Kept BF16
(`modules_to_not_convert`): vision tower, projector, lm_head, `kv_b_proj`,
`kv_a_proj_with_mqa`, `indexer.wk`, `indexer.weights_proj`. The DSpark drafter
is BF16.

| Component | Bytes | Note |
|---|---:|---|
| Routed experts | 1,029,432,803,328 | 9,088 experts × 113,273,856 B (3 × 6144² FP8 + 3 × 48² F32 scales) |
| MLA (72 layers) | 8,669,905,920 | `wkv_a` on the 63 cache-owning layers only |
| Shared experts | 8,042,443,776 | 71 × 3 × 6144² FP8 + scales |
| DSpark drafter | 4,199,938,048 | BF16 |
| Vision tower + adapter | 3,408,437,248 | BF16 |
| lm_head / embeddings | 2,516,582,400 each | BF16 |
| Indexers (44) | 1,195,183,616 | `wq_b` FP8, `wk` + `weights_proj` BF16 |
| Dense MLP (layer 0) | 453,095,424 | FP8 |
| Routers (71) | 111,673,344 | BF16 |
| Norms and misc | 1,781,824 | |
| **Total** | **1,060,548,427,328** | = the index's `total_size`, to the byte |

```
w_resident      = 1,060,548,427,328 B   (987.7 GiB)
w_decode_shared =    20,990,666,304 B   MLA + indexers + dense MLP + shared experts
                                        + routers + norms + lm_head
w_route_pertok  =    32,169,775,104 B   4 x 113,273,856 x 71
w_route_total   = 1,029,432,803,328 B   128 experts: the union saturates at n = 32
```

The drafter and the vision tower are resident but in no decode read, the
convention for every draft model in the study (`model_glm52.md` § 4). Per
token Large 4 reads 32.2 GB of routed experts against GLM-5.3's 22.6 GB and
DSv4.1-Flash's 4.5 GB, the heaviest in the study. Per-token reads, more than
the cache, are what limit how many users a node can carry.

**Not modelled:** vLLM replicates `kv_a_proj_with_mqa` and the indexer
projections on every TP rank (`ReplicatedLinear`), about 1.6 GB per rank
instead of 1/tp of it. That puts ~1.4 GB per GPU at TP8 outside the pool;
GLM-5.3 has the same shape and the same omission.

### NVFP4, official

ModelOpt NVFP4 on the routed experts of the 71 MoE layers only; attention,
indexers, shared experts and the dense MLP stay FP8 and the drafter, vision
and heads BF16, **byte-identical** to the FP8 checkpoint (every other
component's sum matches to the byte).

```
per expert = 3 x 6144^2 / 2   (E2M1 packed)    56,623,104
           + 3 x 6144^2 / 16  (E4M3 scales)     7,077,888
           + 6 x 4            (F32 globals)            24
           = 63,701,016 B   (x 0.5624 of FP8)

nvfp4_w = (610,030,457,408,     resident = the index's total_size
           20,990,666,304,      always-active read, unchanged
           18,091,088,544,      4 x 63,701,016 x 71
           578,914,833,408)     128 x 63,701,016 x 71
```

## 5. DSpark drafter

Three dense MLA layers in BF16 with the target's attention geometry and a
recurrent Markov head; depth 3 is fixed by the checkpoint. Each draft layer
caches latents over a sliding window of 2,048 context tokens (vLLM
`draft_attention_windows`), so the drafter's cache is a fixed per-session
charge, not a per-token one:

```
deltanet_state   = 3 x 2048 x 656 = 4,030,464 B (3.8 MiB) per session
state_step_bytes = 0   (the drafter's reads are folded into mtp)
state_heads      = 1   (a latent window: replicates with the cache under plain TP)
```

The rows are taken as `fp8_ds_mla`, the cache dtype the draft layers share
with the target. This is DSv4-Flash's convention for its three DSpark stages
(storage counted, decode left to the speedup). `mtp` stays at the study's transplanted 1.7, as
on every MoE row. The recipe reports DSpark acceptance lengths of 2.98 / 2.85
/ 2.79 at 8k / 32k / 60k input and "1.9–2.2× output throughput vs no spec" on
GB300; the technical report gives +90% decode interactivity at 8 concurrent
requests.

## 6. Prefill constants

`params_prefill = 49.82e9`, the active GEMM parameters excluding embeddings and
lm_head. `attn_layers = 72`, `attn_d = 64 × 128`: MLA priced as dense
attention on every layer, as GLM-5.3 is, an upper bound (DSA prefill
sparsity is not characterised). The KV-sharing layers still compute
attention, so they stay in the count. Large 4's heads are half as wide as
GLM-5.3's (qk 128 vs 256), so its quadratic term per layer is half: at the
same load it prefills cheaper than GLM-5.3 despite 33% more active
parameters.

## 7. Serving

| | Weights | Min TP (study arithmetic) | Recipe |
|---|---:|---:|---|
| H200, FP8 | 1,060.5 GB | **9** | 2 × 8 H200, TP16 or PP2 × TP8 (`strategy_min_gpus: h200: 16`); no performance numbers taken |
| B300, FP8 | 1,060.5 GB | **5** → TP8 | 8 × B300 TP8 validated |
| B300, NVFP4 | 610.0 GB | **3** → TP4 | one 4-GPU GB300 tray validated; NVFP4 is the default on Blackwell |

Large 4 is the first model that **fits no 8-GPU H200 node**. The explorer's
topologies stop at one node, so it prices Large 4 on B300 only: TP8 in FP8,
and TP8 or DP2 × TP4 with NVFP4.

The page's vLLM command follows the recipe: the
`vllm/vllm-openai:mistral-large-4` image (no vLLM release includes the model
yet), `--tokenizer-mode mistral --config-format mistral --load-format
mistral`, `--kv-cache-dtype fp8_ds_mla`, and DSpark
`{"method":"dspark","num_speculative_tokens":3}`. The recipe validates DSpark
together with DCP (TP8 DCP8 + DSpark on GB300, TP16 DCP8 + DSpark on H200),
so the page emits both flags for this model; elsewhere it drops the
speculative config under DCP.

## 8. Measured anchors against the model

All from the recipe, on 8 × GB300 (same HBM size and bandwidth as B300), with
DSpark. The model at its defaults (MBU 0.22, `mtp = 1.7`):

| | Measured | Model | Ratio |
|---|---:|---:|---:|
| NVFP4 TP8, 1 user, per-user tok/s | 319 (TPOT 3.13 ms) | 444 | model 1.39× high |
| FP8 TP8, 1 user | 281 (TPOT 3.56 ms) | 327 | model 1.16× high |
| NVFP4 TP8, 8 users at ≥100 tok/s/user, 8k input | 110 per user (110.2 tok/s/GPU) | 104 | model 0.94× |

The model is faster than measured at one user and close at eight. Both
comparisons mix the measured DSpark acceptance (~2.9) with the study's 1.7,
so they bound the model rather than calibrate it. The KV capacity check is in
§ 2.

## 9. Quality

The study takes Terminal-Bench from Artificial Analysis only
(`terminal_bench.md`). Read 2026-11-04 from AA's dataset:

| Version | Score | Runs | Note |
|---|---:|---:|---|
| 2.1 | **no run** | — | AA's legacy eval; it has not run 2.1 on Large 4 |
| 4.0 | **24.7%** | 49 / 198 | "Mistral Large 4 (High)", mini-swe-agent, 66 tasks × 3 |

Mistral's technical report claims **41.0%** on 4.0, labelled "AA Coding Agent
Index", and 90.2% on 2.1. The preview claimed 28.3% at launch.

**Where the 41.0 comes from** (AA `/agents/coding-agents` and
`/methodology/coding-agents-benchmarking`, raw page data, read 2026-11-04).
AA's Coding Agent Index v2.0 averages FrontierCode v1.1, Terminal-Bench 4.0
and SWE-Atlas-QnA-AA, each model run inside one agent. Its 4.0 component uses
the same 66 tasks × 3 (task image `sha256:39d9f44b…`) as the Intelligence
Index run the study reads; only the agent differs. Large 4 has one agent
there, Mistral's own:

| Run | Harness | Effort | 4.0 | Output tok / attempt |
|---|---|---|---:|---:|
| Intelligence Index (the study's axis) | mini-swe-agent, 500 steps, bash only, no compaction | high | **24.7%** (49) | 171k |
| Coding Agent Index | Mistral Vibe 3.0.0-v14 (AA lists no release date) | high | **41.4%** (82) | 294k |
| Coding Agent Index | Mistral Vibe 3.0.0-v14 | medium | 26.3% (52) | 194k |

Vibe at high passed 85 attempts; AA's reward-hacking judge flagged 3 and
zeroed them, leaving 82. Mistral's 41.0 is this row (an earlier snapshot or
81/198 rounded). The report's other 4.0 figures are the same index's rows,
each in a different agent: GLM-5.3 40.0 (Opencode 39.9), V4.1-Flash-1017
31.0 (DeepSeek Harness 30.8).

**The gap is an outlier.** For the models AA runs both ways, the agent row
minus the mini-swe-agent row: GLM-5.3 −2.0, V4.1-Flash −3.0 (1017, DeepSeek
Harness) and +1.5 (09-10, Codex), GPT-6 Astra −3.5 and GPT-6.1 Sol −3.1
(Codex), Gemini 4 Carbon −1.6 (Antigravity), Claude Sonnet 5.5 +2.6 and Haiku
5.5 −2.0 (Claude Code), Grok 4.7 +7.5 (Grok Build). Most sit within ±4
points; the next largest vendor-harness gain is Grok's +7.5. Large 4 gains
+16.7 in Vibe, and Vibe at medium (26.3) only matches mini-swe-agent at high.
(Effort levels are matched by name; AA's variant lists were not checked row
by row except for Large 4.) The opposite case exists too: Qwen3.8 Max loses
22 points in Claude Code (16.7 vs 38.9), a harness it was not tuned for.

Reading: Large 4 depends on Vibe far more than other models depend on their
vendor's agent, plausibly from training on Vibe's tool schema, which
mini-swe-agent's plain bash loop does not offer. That is an inference: AA
publishes no per-task trajectories to check it.

Outside views (read 2026-11-04). Vals AI's own mini-swe-agent run of 4.0
lands in the same place as AA's (low to mid 20s), so the 24.7 is not an AA
artefact. The release post says the model gained 13 points over the
preview's 28 on "Artificial Analysis Coding Agent Index - Terminal-Bench 4.0",
credits "continuing to scale reinforcement learning", and names neither the
harness nor the training tasks; a technical report is announced. AI Realist
(`airealist.ai/p/mistral-large-4-weights-score-comes-with-harness`) notes
Vibe moved from v2.26.0 (2026-10-06) to v3.0.0 (2026-10-19) between the
preview and the release, so the 13-point gain may sit in the weights (RL on
Vibe-shaped tasks), in Vibe 3.0 (prompting, thinking budget, sampling), or
both. A run of the release weights in another agent would split it: ≥ 38
puts the gain in the weights, 25–30 puts it in Vibe. No such run is public.

**The model repo** (`mistralai/Mistral-Large-4-1T-A52B`, discussions and
`assets/technical_report.pdf`, read 2026-11-04). No discussion raises the
gap. Two bear on it:
- #5: Claude Code through LiteLLM fails on every skill load, because the
  chat template refuses a `tool` message after a `user` message. Mistral's
  engineer keeps the check: "We have a chat completion protocol that enforces
  this order on which we train, so … behavior is undefined if we remove it",
  and recommends "our Vibe Harness". The model is trained on one message
  protocol and Mistral does not vouch for it outside that protocol.
- #7 (open, from HF staff): adds Terminal-Bench 4.0 = 41.0 to the Hub
  leaderboard, sourced to the announcement, with no harness named; the same
  PR labels its Terminal-Bench-Science row "mini-swe-agent harness".

The technical report's Table 3 labels the row "Terminal-Bench 4.0 (AA Coding
Agent Index)" and names no agent. § 4.1.2 builds terminal-agent RL tasks
partly from "internal terminal-agent traces"; § 4.3.2 says RL and evaluation
run "the same environment code" and that an agentic episode ends when "the
agent … produces a final message without tool calls". Neither says whether
that loop is Vibe's.

Asked in discussion #17 (opened 2026-11-04,
`https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B/discussions/17`):
whether the RL ran in Vibe, what Vibe 3.0 sets that mini-swe-agent does not,
and which score other harnesses should expect. Update this section when Mistral
answers.

**For the study:** the axis keeps 24.7. The one-harness rule
(`terminal_bench.md` § 1) holds, and the Coding Agent Index cannot replace it:
every model there runs in a different agent, and six of the study's ten
models (GLM-5.3-Flash, Qwen3.8-Flash-Next, Qwen3.8 27B, Qwen3.6 35B-A3B,
Mistral Medium 3.5, V4-Flash-Vision) have no row. The 41.4 is the right figure only for a
deployment whose users run Mistral Vibe; the EI users the study sizes for
run pi and opencode, which AA has not paired with Large 4.

On the study's axis Large 4 lands below GLM-5.3 (41.9), GLM-5.3-Flash (32.8)
and DeepSeek-V4.1-Flash (26.8), and level with Qwen3.8-Flash-Next (25.3).

## 10. What the preview note got right

The preview note (2026-10-06) had 1.05T total / 49B active, the A49B decode
read "about 1.2× GLM-5.3's" (it is 1.42× on routed experts and 1.11× on the
always-active read), FP8 ≈ 1.05 TB and NVFP4 ≈ 0.6 TB. It called 8 × H200 "a
marginal fit at best"; with the study's per-GPU reserve the model does not
fit at all (9 GPUs needed). It expected many small "granular" experts; the
release has 128 experts of 113 MB each, top-4, the same count and top-k as
Mistral Large 3. Its prior of a DeepSeek-V3-style MLA cache held, plus the
two sharing schemes.

## 11. Open

- DSpark acceptance is measured (~2.9) and `mtp` is not: the study keeps 1.7
  on every row until the k-sweep of issue #74 settles how the speedup is
  priced.
- 2 × 8 H200 (TP16 across nodes, or PP2 × TP8) is outside the explorer; the
  Python model can price TP16 with its cross-node penalty, unmeasured.
- Image input is not priced: the 1.70B vision tower is resident only.
