# MLPerf Inference v6.0

Read 2026-11-04. Records: `data/external/mlperf-inference-v6.0.json`, the DeepSeek-R1 and gpt-oss-120b rows on H200, B200, B300, GB200 and GB300 (75 of the 159 extracted).

## Verdict

- MLPerf v6.0 has no Working Set model and no H200 DeepSeek-R1 result. It has DeepSeek-R1 on B200, B300, GB200, GB300 (all fp4, TensorRT-LLM) and gpt-oss-120b on H200, B200, B300, MI355X. DeepSeek-R1 is the nearest architecture to DSV4F, DSV41F and GLM-5.3, and the only cross-check for B300 and B200.
- MLPerf measures single-turn requests at a fixed latency bound. It says nothing about agent traces, long context or prefix reuse. Use it as a hardware ceiling for decode, not as a seat count.
- MLPerf licensing allows citing, not blending. See "Rules for public use".

## Results

Source: `summary_results.json` in github.com/mlcommons/inference_results_v6.0 at commit 4d3916ac (520 rows, 19 benchmarks). Filter used: closed division, datacenter, token-throughput rows, GPUs H200, B200, B300, GB200, GB300, MI355X.

Benchmarks with LLMs in v6.0: DeepSeek-R1, gpt-oss-120b, Llama 3.1 405B, Llama 3.1 8B, Llama 2 70B (99 and 99.9), Mixtral 8x7B, Qwen3-VL-235B-A22B, Whisper. No Qwen3.x, GLM or Mistral models.

### Latency constraints (99th percentile, Server and Interactive only; Offline has none)

From `inference_rules.adoc` on inference_policies master (commit d3eba2f2), not from the results file. Master was read, not a v6.0 tag (the repo has no v6.0 tag, only `v6.0.0pre` on `mlcommons/inference`).

| Benchmark | Server TTFT / TPOT | Interactive TTFT / TPOT |
|---|---|---|
| DeepSeek-R1 | 2000 ms / 80 ms | 1500 ms / 15 ms |
| gpt-oss-120b | 3000 ms / 80 ms | 2000 ms / 20 ms |
| Llama 3.1 405B | 6000 ms / 175 ms | 4500 ms / 80 ms |
| Llama 2 70B | 2000 ms / 200 ms ("Conversational") | 450 ms / 40 ms |
| Llama 3.1 8B | 2000 ms / 100 ms ("Conversational") | 500 ms / 30 ms |

TPOT of 80 ms is 12.5 tok/s per user; 15 ms is 67 tok/s; 20 ms is 50 tok/s. So DeepSeek-R1 Server sits below the 20 tok/s floor Working Set's other sources use for agents, and only Interactive clears it. The score is the highest Poisson query rate that meets both bounds, so tok/s is whole-system output tokens/s under that load.

### 8-GPU single-node results, tokens/s (whole system; per GPU in brackets)

| Benchmark, scenario | GPU | tok/s range across submitters | Per GPU |
|---|---|---|---|
| DeepSeek-R1 Offline | B200 | 57.1k - 58.6k (3 systems) | 7.1k - 7.3k |
| DeepSeek-R1 Offline | B300 | 66.3k - 70.3k (7) | 8.3k - 8.8k |
| DeepSeek-R1 Server | B200 | 40.9k - 51.7k (3) | 5.1k - 6.5k |
| DeepSeek-R1 Server | B300 | 35.4k - 60.4k (7) | 4.4k - 7.6k |
| DeepSeek-R1 Interactive | B300 | 4,935 (1, GigaComputing) | 617 |
| gpt-oss-120b Server | H200 | 24.1k (1, RedHat, mxfp4, vLLM via llm-d) | 3.0k |
| gpt-oss-120b Server | B200 | 71.6k - 87.4k (2) | 8.9k - 10.9k |
| gpt-oss-120b Server | B300 | 100.3k - 110.7k (6) | 12.5k - 13.8k |
| gpt-oss-120b Server | MI355X | 75.4k - 82.1k (5) | 9.4k - 10.3k |
| gpt-oss-120b Interactive | B200 / B300 | 13.2k / 26.0k (1 each) | 1.6k / 3.3k |
| Llama 3.1 405B Server | B200 / B300 | 984 / 1.34k - 1.48k | 123 / 168 - 186 |
| Llama 3.1 405B Interactive | B300 | 765 (1) | 96 |
| Llama 2 70B Interactive | H200 NVL | 16.3k (1, Dell) | 2.0k |
| Llama 2 70B Interactive | B200 / B300 | 61.6k / 65.5k - 70.7k | 7.7k / 8.2k - 8.8k |
| Llama 2 70B Interactive | MI355X | 70.6k - 73.6k (6) | 8.8k - 9.2k |

The per-GPU column is our division (kind derived in the JSON), not an MLPerf metric. B300 submitters differ by up to 1.7x on the same benchmark and scenario (DeepSeek-R1 Server 35.4k to 60.4k): software stack and tuning, not hardware.

### Rack-scale results (not 8-GPU boxes)

| Benchmark, scenario | System | tok/s | Per GPU |
|---|---|---|---|
| DeepSeek-R1 Server | GB300 NVL72, 72 GPUs (NVIDIA / Nebius) | 400k / 576k | 5.6k / 8.0k |
| DeepSeek-R1 Server | GB300, 288 GPUs (NVIDIA) | 1.555M | 5.4k |
| DeepSeek-R1 Server | GB200 NVL72, 72 GPUs (NVIDIA) | 336k | 4.7k |
| DeepSeek-R1 Interactive | GB300 / GB200, 72 GPUs (NVIDIA) | 250.6k / 240.3k | 3.5k / 3.3k |
| DeepSeek-R1 Server | GB300, 8 GPUs on 2 nodes (CoreWeave / Nebius) | 44.6k / 64.5k | 5.6k / 8.1k |

The 3.3k to 3.5k per GPU at 15 ms TPOT (Interactive) against 5.6k to 8.0k at 80 ms (Server) shows the throughput cost of a tight per-user rate on the same hardware. This is the clearest published trade of tok/s per user against tok/s per GPU for this model class.

### Gaps

- No H200 DeepSeek-R1, no H200 B-series comparison on one benchmark except Llama 2 70B Interactive (H200 NVL only) and gpt-oss (one H200 SXM entry).
- AMD ran Llama 2 70B and gpt-oss-120b on MI355X, not DeepSeek-R1.
- Not read in this pass: DeepSeek-R1 input/output token lengths and the per-run `mlperf_log_detail` TTFT/TPOT percentiles. The records leave isl and osl null.
- Power results exist for some rows (`has_power`); not extracted.

### Rules for public use

Source: MLPerf Results Messaging Guidelines, github.com/mlcommons/policies, commit 45b8b625.

- "You may not use the MLPerf name or logo, except in a factual and non-trademark manner solely to accurately represent your benchmark score."
- Any disclosure names submitter, benchmark name and version, and system under test in the main text, table or figure.
- A footnote gives suite, version, division; benchmark and scenario; date and source of retrieval; result ID (e.g. 6.0-0073); and the trademark legend ("The MLPerf name and logo are registered and unregistered trademarks of MLCommons Association ... All rights reserved. Unauthorized use strictly prohibited. See www.mlcommons.org for more information.").
- Compare only the same benchmark and scenario from compatible versions. "MLPerf results may not be compared against non-MLPerf results." So a Working Set chart must not plot an MLPerf point against a Working Set model output as a like-for-like comparison. Show MLPerf as a separate, labelled reference.
- A comparison must name any difference in version, division, category, scenario or chip count.
- Derived metrics (our tok/s per GPU) "must not be presented as official or verified MLPerf metrics", and the basis must be explicit in text and a footnote.
- Results run by us with MLPerf code but not submitted must be called unverified, with "Result not verified by MLCommons Association."
- Do not imply MLCommons endorses a product.
- The results repo itself is Apache-2.0.

