# SemiAnalysis InferenceX v2: what it measures and what we pulled

Read 2026-11-04. Verdict: the data is public and machine-readable with no auth. It has decode speed per user at conc 1-8 for 3 of our 9 models (DeepSeek-V4.1-Flash, GLM-5.3, Qwen3.8-Flash-Next), all from the AgentX agentic benchmark and all with speculative decoding on. Qwen3.8-27B and GLM-5.3-Flash are registered but have zero rows. Mistral and Qwen3.6-35B-A3B are absent. The older single-turn rows (DeepSeek-R1, Qwen3.5-397B, Llama-3.3-70B, gpt-oss) give a usable decode MBU anchor. Dense Llama-70B reaches 50-55% weights-only MBU, but MoE at conc 1 only 6-12%.

## Where the data lives

- Repos (commit-pinned): `SemiAnalysisAI/InferenceX` @ `6481779aae17` (Apache-2.0; configs, recipes, harness, no results) and `SemiAnalysisAI/InferenceX-app` @ `42da605b6d09` (dashboard + API; its LICENSE is GPL-3.0, not Apache). Results are not in git and not in GitHub releases (only a TileRT pre-release). They sit in a Neon Postgres behind a public API.
- API contract: `https://inferencex.semianalysis.com/api/openapi.json` (about 60 routes, `InferenceX-app/docs/inferencex-api-examples.md`).
- The route we use: `GET /api/v1/benchmarks?model=<display name>` returns raw rows, latest curve per config, `date` param = as-of cutoff. Model list: `/api/v1/views/options`. Params and total/active params: `/api/v1/views/model-architectures`. Other routes of interest: `/api/v1/agentic-aggregates`, `/api/v1/derived-agentic-metrics`, `/api/v1/tco-feed`, `/api/v1/views/calculator`.
- Refresh: `python3 scripts/external_inferencex.py fetch` (15 API calls into `$IX_RAW`, default `/tmp/inferencex-raw`), then `... build` rewrites `data/external/inferencex.json`. The API is live, not pinned: rows carry a per-curve `date` and a stable `id`; records cite `extra.row_id` and the GitHub Actions `run_url`. Raw snapshot of 2026-11-04 (sha256 prefixes: R1 f47d66cc, V4.1-Flash ea2ecba2, GLM-5.2/5.3 58b37eb7, Qwen3.8-Flash-Next 90e44f21).
- No energy per token in these rows. `power_*` keys exist on 2268 rows but we did not extract them; PowerX has its own export (`docs/inferencex-api-examples.md`, `powerValid=strictV2`).

## What is measured

- Single-turn ("fixed-seq"): `infx/bench/fixed_seq.py` runs `benchmark_serving` with `--dataset-name random`, ISL/OSL fixed, `--random-range-ratio 0.8` (`.github/workflows/benchmark-tmpl.yml:104`), `--ignore-eos`, `--request-rate inf`, `num_prompts = 10 * conc`, `2 * conc` warmups. Only 8192/1024 is active; 1k/1k deprecated 2026-07-17, 1k/8k deprecated 2026-03-27 (`docs/MODELS.md`). Old rows still exist: R1 and Qwen3.5 8k/1k up to 2026-10; Llama-70B only 2025-10-29.
- Concurrency sweep per config set in `configs/*-master.yaml` (`conc-start`/`conc-end`, powers of 2 plus lists). Single-turn conc starts at 1 or 4 depending on the config. Agentic conc values seen: 1, 2, 4, 6, 8, 12 ... 256, and 1228 to 10888 for multinode.
- Agentic (AgentX, trace replay, prefix caching ON, `kv_offloading` dram/hicache): median input per request 310K tokens on GLM-5.x B300 at conc 1, cache hit about 0.99. Details already in `research/agentx_kv_hits.md`. New models are onboarded agentic-only (`docs/MODELS.md`). Spec decode (MTP, DSpark, draft_model) is on for almost every agentic row.
- Engines: SGLang, vLLM, TRT-LLM, Dynamo (sglang/vllm/trt, used for disagg), llm-d vLLM, AMD ATOM and MoRI-SGLang, TileRT. Image tag is in each row (`image`); records keep the tag. Multi-node rows are mostly disaggregated prefill/decode.
- Metrics per row: `tput_per_gpu` (input+output tokens per GPU, as the dashboard plots it), `output_tput_per_gpu`, `median/p90/p99 ttft, tpot, itl, e2el`, `median_intvty` (= 1/TPOT, tok/s/user), agentic cache hit rates. Check: for DeepSeek-R1 B300 TP8 conc 1, `tput_per_gpu*8 = 1619` vs `9216/(ttft+1023*tpot) = 1637`, so the number is per GPU of the TP group.
- GPU count: the `num_decode_gpu` field is wrong for EP>1 single-node rows (TP8 EP8 reports 64). We use `total_tput_tps / tput_per_gpu` for agentic rows (exact), `decode_tp` for single-node single-turn, `prefill+decode` GPUs for disagg.
- TCO: no hidden model. Cost per Mtok = `costh / (tok/s/GPU * 3600 / 1e6)` (`InferenceX-app/packages/app/src/lib/shared/utils.ts:167`, `docs/tco-calculator.md:175`). `costh` = USD per GPU-hour "Owning at Large Hyperscaler Volume"; `costr` = a rental rate. Prices are from `packages/constants/src/gpu-keys.ts` (HW_REGISTRY), also in `/api/v1/views/options` `.hardware[].costPerHour`:

| GPU | costh | costr | TDP W |
|---|---|---|---|
| H100 | 1.17 | 2.00 | 700 |
| H200 | 1.22 | 2.90 | 700 |
| B200 | 1.73 | 3.70 | 1000 |
| B300 | 2.26 | 4.25 | 1100 |
| GB200 NVL72 | 1.86 | 4.00 | 1200 |
| GB300 NVL72 | 2.31 | 5.00 | 1400 |
| MI355X | 1.50 | 2.90 | 1400 |

The agentic dashboard cost is dominated by cached input tokens (tput_per_gpu of 1e5-1e6 tok/s/GPU), so we emit `usd_per_mtok` only for single-turn records.

## Overlap with Working Set models

| WS model | InferenceX | Rows (agentic) | Hardware with rows |
|---|---|---|---|
| DSV41F | DeepSeek-V4.1-Flash `dsv41flash`, 552B total / 16B active, FP4 | 204 | H100, H200, B200, B300, GB200, GB300, MI300X/325X/355X |
| GLM52 (= GLM-5.3) | `glm5.3` (a different model from `glm5.2`, 115 rows), 744B / 40B | 18 + 1 single-turn | B300 sglang, VR200 dynamo-trt, MI355X tilert only. No Hopper. |
| Q38FN | Qwen3.8-Flash-Next `qwen3.8next`, 176B / 6B, FP4 and FP8 | 47 | H100, H200, B200, B300, GB300, VR200, MI355X |
| 27B | Qwen3.8-27B (dense) and `-Eager` registered | 0 | none yet |
| GLM53F | GLM-5.3-Flash registered (`glm5.3flash`) | 0 | none yet |
| DSV4F, 35BA3B, MM35, ML4 | not in InferenceX | | |

The nearest non-overlapping rows for DSV4F: DeepSeek-V4-Pro (1600B / 49B, 935 rows, H200 FP8 agentic and FP4 on Blackwell).

## Low-concurrency decode speed, overlap models (agentic, MTP, median interactivity)

All figures are `metrics.median_intvty` of the lowest-conc point (conc 1 unless stated), spec decode on, so they are accepted-token rates, not single-token step rates. Context is long (hundreds of K tokens), cache hit about 99%. Full records in the JSON.

| Model | GPU | Engine | Config | conc | tok/s/user | Out tok/s/GPU |
|---|---|---|---|---|---|---|
| DSV41F | H100 | vllm | FP4 TP8 mtp | 1 | 361.0 | 19.4 |
| DSV41F | H200 | vllm | FP4 TP8 mtp | 2 | 349.7 | 22.4 |
| DSV41F | H200 | sglang | FP4 TP8 mtp | 2 | 264.6 | 19.5 |
| DSV41F | B200 | vllm | FP4 TP4 mtp | 1 | 617.3 | 46.5 |
| DSV41F | B300 | vllm | FP4 TP4 mtp | 1 | 847.5 | 49.7 |
| DSV41F | GB300 | sglang | FP4 TP4 mtp | 1 | 970.9 | 49.7 |
| DSV41F | MI355X | sglang | FP4 TP4 mtp | 1 | 666.7 | 44.3 |
| GLM-5.3 | B300 | sglang | FP4 TP8 mtp | 1 | 344.8 | 11.1 |
| GLM-5.3 | MI355X | tilert | FP8 disagg 8+8 | 1 | 480.8 | 20.0 |
| Q38FN | H100 | sglang | FP8 TP8 EP8 mtp | 1 | 239.8 | 9.7 |
| Q38FN | H200 | sglang | FP8 TP4 EP4 mtp | 1 | 252.5 | 37.2 |
| Q38FN | B200 | sglang | FP4 TP1 mtp | 1 | 292.4 | 159.7 |
| Q38FN | B300 | sglang | FP4 TP4 EP4 mtp | 1 | 393.7 | 48.6 |

For the DSV41F H100 pair, sglang TP8 EP8 gave only 134.6 tok/s/user against 361.0 for vllm, so engine choice moves this 2.7x.

## Max-throughput points, overlap models (agentic)

Chosen by `output_tput_per_gpu`. Interactivity collapses at the top, so these are bandwidth/cache limits, not user-facing speeds.

| Model | GPU | Engine | Config | conc | tok/s/user | Out tok/s/GPU |
|---|---|---|---|---|---|---|
| DSV41F | H100 | vllm | FP4 TP8 mtp | 28 | 117.8 | 128.9 |
| DSV41F | H200 | vllm | FP4 TP4 mtp | 64 | 41.6 | 444.9 |
| DSV41F | B200 | sglang | FP4 TP4 EP4 DPA | 256 | 114.5 | 2502.7 |
| DSV41F | B300 | vllm | FP4 TP2 EP2 DPA | 192 | 77.4 | 3261.5 |
| DSV41F | GB300 | dynamo-vllm | disagg, 40 GPUs | 9888 | 78.9 | 18820.0 |
| DSV41F | MI355X | sglang | FP4 TP2 DPA | 64 | 102.5 | 1343.3 |
| GLM-5.3 | B300 | sglang | FP4 TP8 mtp | 28 | 145.3 | 82.6 |
| Q38FN | H100 | sglang | FP8 TP8 EP8 | 16 | 126.9 | 72.2 |
| Q38FN | H200 | sglang | FP8 TP2 EP2 | 16 | 101.0 | 297.3 |
| Q38FN | B300 | sglang | FP4 TP1 | 80 | 36.2 | 2413.5 |

## Other models, single-turn 8k/1k (R1, Qwen3.5, Llama-70B, gpt-oss)

Low = best-interactivity point with conc <= 8 across all configs (often MTP, sometimes disagg, so it can use 30-60 GPUs). Max = top `tok_s_per_gpu`. USD/Mtok is derived with costh, all tokens counted.

| Model | GPU | Low: conc, tok/s/user, config | Max: conc, tok/s/GPU, USD/Mtok |
|---|---|---|---|
| R1 FP8 | H200 | n/a (conc 4 non-MTP sglang: 90.8) | 256, 3370, 0.101 (trt TP8 EP8 DPA, mtp) |
| R1 FP4 | B200 | 4, 300.8 (dynamo-sglang disagg, mtp) | 1691, 13430, 0.036 (dynamo-trt disagg) |
| R1 FP4 | B300 | 8, 318.6 (dynamo-trt disagg) | 2198, 15868, 0.040 |
| R1 FP4 | GB300 | 5, 312.4 | 2253, 18132, 0.035 |
| R1 FP8 | H100 | 6, 122.3 (disagg 64 GPUs) | 154, 850, 0.382 |
| Qwen3.5 FP8 | H100 | 4, 195.0 (TP8 EP8 mtp) | 256, 2561, 0.127 |
| Qwen3.5 FP8 | H200 | 4, 187.9 | 128, 2476, 0.137 |
| Qwen3.5 FP8 | B200 | 4, 361.6 (mtp) | 1024, 20571, 0.023 (FP4 trt) |
| gpt-oss-120b | H200 | 4, 255.6 (vllm TP8) | 64, 13091, 0.026 (TP1) |
| gpt-oss-120b | B200 | 4, 418.0 (trt TP8) | 256, 65459, 0.007 (TP1) |
| Llama-3.3-70B FP8 | H200 | 4, 138.9 (trt TP8) | 128, 5102, 0.066 (TP1) |

The cheapest tokens come from FP4 disaggregated Blackwell at very high concurrency (0.035-0.040 USD/Mtok on R1), roughly 10x cheaper than H100-class FP8. Engine and config choices are in every record.

## Implied decode MBU (derived)

Method: `MBU = active_params * bytes_per_param / (median_TPOT * n_GPUs * HBM_BW)`. Params from `/api/v1/views/model-architectures` (totalParams/activeParams). Bytes per param is our assumption: FP8 = 1.0, NVFP4 = 0.5625. HBM bandwidths as instructed: H100 3.35, H200 4.8, B200 8.0, B300 8.0 TB/s. Rows are single-node, TP-only, no spec decode, 8k/1k. Weights only: no KV, no activations, no quantization scales, so each number is a lower bound. At conc 1 an MoE reads only the routed experts, which is why we use active params. Where the sweep started at conc 4 (Llama) all 70B are read once per step anyway.

| Model | Total / active B | GPU | TP | Precision | conc | TPOT ms | MBU |
|---|---|---|---|---|---|---|---|
| Qwen3.5-397B-A17B | 397 / 17 | H100 | 8 | FP8 | 1 | 6.55 | 9.7% |
| Qwen3.5-397B-A17B | 397 / 17 | B200 | 8 | FP8 | 1 | 4.42 | 6.0% |
| DeepSeek-R1-0528 | 671 / 37 | B200 | 8 | FP8 | 1 | 7.39 | 7.8% |
| DeepSeek-R1-0528 | 671 / 37 | B300 | 8 | FP8 | 1 | 6.53 | 8.9% |
| DeepSeek-R1-0528 | 671 / 37 | B200 | 4 | NVFP4 | 1 | 5.27 | 12.3% |
| Llama-3.3-70B | 70 / 70 | H200 | 1 | FP8 | 4 | 26.33 | 55.4% |
| Llama-3.3-70B | 70 / 70 | H100 | 2 | FP8 | 4 | 20.72 | 50.4% |
| Llama-3.3-70B | 70 / 70 | B200 | 1 | FP8 | 4 | 16.73 | 52.3% |

Reading: a dense model on one GPU (TP1) reaches about 50-55% of HBM bandwidth even with the KV cache not counted. A MoE at conc 1 across 8 GPUs reaches only 6-12%: per-GPU reads are 2-5 GB, the step is 4-7 ms, and all-reduce, kernel launches and attention dominate (Qwen3.5 has linear-attention state reads we did not count either). So a flat MBU for MoE at low concurrency is wrong; a fixed-overhead term per step is needed. This is a shape check against our model, not a fit.

## Caveats

- Rows differ in date (2025-10 to 2026-11) and image. Compare within a row date, not across.
- Agentic `tok_s_per_gpu` counts cached input tokens; use `out_tok_s_per_gpu` for output.
- The single-turn "low" points often use MTP; MTP tok/s/user is accepted tokens, not step rate.
- Qwen3.8-27B and GLM-5.3-Flash: nothing to compare yet. Re-run `scripts/external_inferencex.py fetch` and check `Qwen3.8-27B.json` in the raw directory (currently `[]`).
- `n_gpus` for disagg rows is prefill+decode GPUs, sometimes 30-64; do not read these as single-node boxes.
- Licence: InferenceX repo Apache-2.0; the app repo that serves the API is GPL-3.0. We use only API data, with attribution to SemiAnalysis.
- Unchecked: whether FP4/FP8 checkpoints for DSV41F use KV in FP8; the API has no KV dtype field (`config.kv` is null).
