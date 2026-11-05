# External self-hosted deployments (read 2026-11-04)

Records: `data/external/deployments.json`; how they are used: [README.md](README.md).

Bottom line: only two deployments give GPU count + model + a load figure from a readable source: Revolut (secondary press on a talk; talk itself not read) and DeepSeek V3/R1 (primary, day-level stats). Everything else I found lacks a GPU count or is a synthetic benchmark. Both are self-reported (kind "vendor").

## 1. Revolut: GLM-5.1 then GLM-5.3 on 56 B200 (coding agents, ~2,000 engineers)

Primary: AI Engineer NYC talk by David Abitbol (Revolut ops/infra lead), given 2026-10-13, video posted 2026-10-29: https://www.youtube.com/watch?v=Ok6QdZHCcI8 . I could not read the transcript (YouTube returned HTTP 429 to yt-dlp). I found no Revolut engineering blog or Revolut LinkedIn post. Everything below is from press about the talk.

Main source, The Register, Tobias Mann, 2026-11-03 (https://www.theregister.com/ai-and-ml/2026/11/03/british-fintech-revolut-slashes-ai-coding-costs-10x-by-cutting-out-claude-and-going-self-hosted/5307345). Read in full.

Figures checked against The Register:

| Note | Verdict | Quote |
|---|---|---|
| ~2,000 engineers | confirmed | "AI coding tools to roughly 2,000 engineers" |
| 56 B200, 7 DGX B200 | confirmed | "56 Nvidia B200 GPUs, equivalent in GPU count to seven eight-GPU DGX B200 systems" |
| GLM-5.1 then 5.3 | confirmed | "initially adopted GLM-5.1 before moving to GLM-5.3, which it now serves using 56 Nvidia B200 GPUs" |
| NVFP4, disaggregated P/D | confirmed | "switching to NVFP4 quantization, implementing disaggregated prefill and decode, and tuning context lengths" |
| Context ~256K | confirmed | "Even though you can serve one million tokens, 256 [thousand] is really what you need." |
| ~200 tok/s per stream | confirmed | "We serve, in terms of interactivity, 200 tokens per second." |
| ~800 after 5 pm | confirmed | "engineers who sign in after 5 pm are getting closer to 800 tokens a second" |
| TTFT < 3 s at 256K | partly: the 3 s is a target, and the article says it was reached | "able to achieve sub 3-second time to first token on context windows of around 256,000 tokens"; "As soon as you exceed 5 seconds, my Slack inbox starts to blow up" |
| $0.30/Mtok, 1/10 of Codex/Claude Code | confirmed, with a catch: the price is for GLM 5.1 | "At $0.30 per million input/output tokens, self hosting GLM 5.1 cost about a tenth of using OpenAI's Codex or Anthropic's Claude Code." |
| Images to Gemini/GPT | confirmed | "a model router to direct vision queries to models like Gemini or GPT Luna" |

Not in The Register: engine (vLLM/SGLang/TRT-LLM), parallelism, concurrent users, tokens/day, cache hit rate. Searches and three summaries (below) name no engine; no number of concurrent users or tokens/day anywhere.

Secondary summaries of the same talk (each is a rewrite; treat as unverified until the transcript is read):
- StartupHub, 2026-10-29 (https://www.startuphub.ai/ai-news/artificial-intelligence/2026/revolut-says-self-hosted-ai-cut-coding-costs-90): "improved performance 88% for 76% more demand without adding GPUs"; "NVFP4 on Blackwell saves about 40% GPU footprint versus FP8"; "GLM needs TP8 not TP4 to fit KV cache"; TTFT "from over 10 seconds to 1.6 seconds even as requests per second rose from 3 to over 7"; "OpenAI compatible and routed through tools like OpenCode and Codex"; cost math leaves out "cloud spend, staffing and observability". It says the model is "GLM 4.5", which contradicts The Register and looks like an error.
- CellCog (https://cellcog.ai/blog/revolut-self-hosted-glm/): repeats the above; adds "cache hit rate above 90% for coding traffic", which it labels as advice, not a Revolut measurement. Treat as weak.
- LinkedIn post by krehlik (https://www.linkedin.com/posts/krehlik_how-revolut-cut-ai-coding-costs-90-with-activity-7523128876093399040-M1Oc), as relayed by a fetch tool: GPUs "rented from Together AI". Third-party, one source only.

Reading: 2,000 is the rollout population, not concurrent users. The load figure that is comparable is requests/s (>7, secondary). Workday peak vs after-5-pm speeds (200 vs 800 tok/s) show the cluster is load-limited at peak. Cost claim excludes staffing and observability.

## 2. DeepSeek V3/R1 online service (Feb 2025), the only primary day-level stats

Primary: DeepSeek open-infra-index, Open Source Week day 6: https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md (raw file read). Published 2025-03-01; window is 2025-02-27 12:00 to 02-28 12:00 (UTC+8). Outside the 2025-26 coding-agent focus in that it is a chat/API mix (web, app, API), not agents; still the best primary of GPUs vs tokens.

- Config: H800, FP8 matmuls and dispatch, BF16 MLA and combine; prefill/decode disaggregated. Prefill unit: 4 nodes, EP32/DP32. Decode unit: 18 nodes, EP144/DP144. Engine is DeepSeek's own.
- GPUs: "peak node occupancy for V3 and R1 inference services reached 278, with an average occupancy of 226.75 nodes (each node contains 8 H800 GPUs)". Nodes shift to training at night.
- Load: "Total input tokens: 608B, of which 342B tokens (56.3%) hit the on-disk KV cache." "Total output tokens: 168B." "The average output speed was 20–22 tokens per second, and the average kvcache length per output token was 4,989 tokens."
- Per node: "~73.7k tokens/s input (including cache hits) during prefilling or ~14.8k tokens/s output during decoding."
- Cost claim: "$2 per hour" H800, "$87,072" a day; theoretical revenue "$562,027", "cost profit margin of 545%". Revenue is hypothetical (everything billed at R1 prices).
- No user count.

## 3. Kimi K2 on 128 H200 (Mooncake team, 2025-07-20): benchmark of a serving stack, not a user load

Primary: https://lmsys.org/blog/2025-07-20-k2-large-scale-ep ; Mooncake README https://github.com/kvcache-ai/Mooncake . Config from the blog via search summary (I did not read the blog page itself): SGLang + OME, PD disaggregated 1P1D, 4 prefill nodes and 12 decode nodes of 8 H200; test shape 2000 in / 100 out tokens, decode batch 480; "56k prefill tokens/sec and 24k decode tokens/sec per node". README (read): "Mooncake powers the deployment of Kimi K2 on 128 H200 GPUs with PD disaggregation and large-scale expert parallelism" and "224k tokens/sec prefill throughput and 288k tokens/sec decode throughput". No users. Short-output test, so tok/s per GPU is a ceiling, not an agent load. Useful as an upper bound on decode efficiency for a 1T MoE.

## 4. Vendor benchmark, not a deployment: GLM-5.2 NVFP4 on 24 B300 (DaoCloud, vLLM blog 2026-07-23)

https://vllm.ai/blog/2026-07-23-glm-5.2-nvfp4-b300-pd (read via fetch tool). 3 x 8-GPU B300; 4 prefill instances (TP1 DP4 EP, 16 GPUs) + 1 decode instance (TP1 DP8 EP, 8 GPUs); NIXL. SLA: "Mean TTFT <= 2.5 s and mean TPOT <= 20 ms". Concurrency: "roughly 700 concurrent requests at 8K, 300 at 16K, and 25 at 256K". Synthetic random data, no prefix hits. No named customer. Kept because it gives concurrency by context for a GLM-class model at an SLA near Revolut's.

## Looked at and dropped (no GPU count)

- Cloudflare, Kimi K2.5 on Workers AI (https://blog.cloudflare.com/workers-ai-large-models/, 2026-03-19): internal agent "processes over 7B tokens per day"; "we cut costs by 77%"; "$2.4M a year" avoided. No GPUs, engine or users. 7B tokens/day = 81 k tok/s average (derived, 7e9/86400).
- vLLM Korea Meetup (https://vllm.ai/blog/2026-04-14-vllm-korea-meetup-2026): "serves over 4,000 employees" on an air-gapped private LLM API (page attributes it to Samsung Electronics; a search summary said Upstage; check before citing). No GPUs, model or throughput.
- GWDG Chat AI (2024): 3.5K users, 3K messages/day; no GPU count tied to it; before the 2025 window.
- Public searches for bank, telco, university and government coding-agent deployments returned guides and vendor sizing, no case with GPU + users. Open: the Revolut talk transcript, which would settle engine, concurrency and hit rate.

## Derived table

All derived rows use primary or press figures above. Formulas in the JSON `note`.

| Deployment | Metric | Value | Formula |
|---|---|---|---|
| Revolut | engineers (enrolled) per GPU | 35.7 | 2,000 / 56 |
| Revolut | requests/s per GPU | ~0.125 | 7 / 56 (secondary, "over 7") |
| DeepSeek | average GPUs | 1,814 | 226.75 nodes x 8 |
| DeepSeek | output tok/s per GPU | ~1,072 | 168e9 / 86,400 / 1,814 |
| DeepSeek | input tok/s per GPU (incl. cache hits) | ~3,879 | 608e9 / 86,400 / 1,814 |
| DeepSeek | uncached input tok/s per GPU | ~1,697 | (608e9 - 342e9) / 86,400 / 1,814 |
| DeepSeek | concurrent streams, whole service | ~88k to ~97k | (168e9 / 86,400) / (22 or 20 tok/s) |
| DeepSeek | concurrent streams per GPU | ~49 to ~53 | above / 1,814 |
| Kimi K2 (bench) | decode tok/s per GPU | 2,250 | 288k / 128 |
| Kimi K2 (bench) | prefill tok/s per GPU | 1,750 | 224k / 128 |
| Kimi K2 (bench) | decode tok/s per GPU, decode GPUs only | 3,000 | 288k / 96 |

DeepSeek caveats: GPU average is over the whole service including prefill nodes; per-GPU output counts every GPU. Streams figure uses an average of 20-22 tok/s per stream, so it is a Little's-law estimate, not a reported count.
