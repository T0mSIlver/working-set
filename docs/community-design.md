# Community runs: design

Today `ws test` measures one deployment and writes a `run.json` that stays on the machine that ran it. This document designs a site where anyone downloads a config, runs it on their own hardware (llama.cpp or vLLM, a consumer GPU or a Mac), uploads the record, and finds the best config for their box among everyone's runs. The model then learns from those runs: every upload is a prediction checked against a measurement.

## What already exists elsewhere

Searched 19 Sep 2026. Nothing combines the three parts.

| Project | Agentic workload under concurrency | Consumer GPUs, Macs, llama.cpp | Community uploads |
|---|---|---|---|
| [SemiAnalysis InferenceX / AgentX](https://inferencex.semianalysis.com/) | yes: replays 393 Claude Code sessions, prefix-cache hit rate above 96% | no, datacenter parts | no, maintainer-run, vendor configs verified by them |
| [Artificial Analysis AA-AgentPerf](https://artificialanalysis.ai/articles/aa-agentperf) | yes: concurrent coding agents at latency SLOs | no, datacenter parts | no, vendor submissions |
| [Localmaxxing](https://localmaxxing.com) | no, single stream | yes | yes: 8,280 runs, 2,586 users |
| [LocalScore](https://www.localscore.ai/) | no, single stream | yes (llamafile) | yes |
| "Can I run it" calculators | predicted, never measured | yes | no |

None reports prediction against measurement.

The community sites run on the honour system: neither LocalScore nor Localmaxxing documents any check on uploads. MLPerf's peer review and audits ([rules](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc)) work because submitters are companies with a reputation at stake, and do not transfer to anonymous hobbyists.

The closest competitor is InferenceX: same workload idea, but datacenter hardware and no uploads from outside. Localmaxxing is the one with the users; its defence would be adding a concurrency test, so ours has to be the workload and the model, not the upload form.

## The run

A run is hardware × engine, version and flags × model and quantisation × workload profile, and it yields measured ceilings plus the prediction error.

Workload profiles make runs comparable. The site publishes a fixed, seeded set (`agentic-coding-v1` built from real session traces, `chat-v1`); users vary hardware and flags, never the workload. If SemiAnalysis's traces are public, they are a candidate source.

The headline number is **agents served**: the most concurrent sessions the box sustains before p95 TTFT or the inter-token gap breaks a stated bound. Single-stream tok/s, what llama-bench and every community site report, does not answer how many agents a box can run.

## Pages

1. **Explorer**, as today: predicts.
2. **Run it**: `ws detect` fills in the hardware; pick an engine and a model; get the toml, the engine launch line and the `ws` commands.
3. **Run page**: config, flags, per-rung charts, prediction against measurement, and a Reproduce button that downloads the same toml and engine command.
4. **Results**: one table per model × hardware class, filtered by engine, quantisation and flags.
5. **Best config for my box**: flag sets ranked by agents served on the given hardware.
6. **Model accuracy**: prediction error across all runs. Community runs refit the per-hardware constants, so predictions improve as runs arrive.

## Accounts

Browsing, downloading, running and uploading need no account. An anonymous run gets its page but stays out of the results tables. A GitHub login, which keeps spam down for this audience, adds:

- saved rigs, re-run on each new llama.cpp or vLLM release, with a regression chart;
- a notice when someone beats your config on the same hardware;
- your runs in the results tables, where they count as reproductions of others'.

## Keeping runs honest

An open-source CLI cannot be made tamper-proof and signing it proves nothing, so the site says so and checks instead:

1. **The server recomputes.** `ws submit` uploads the raw per-request timings; the server re-derives every verdict with the pinned `workingset` version, as `ws report` already does from a record, and rejects a record whose headline does not match.
2. **The CLI captures, the user does not type.** GPU and driver from nvidia-smi, rocm-smi or system_profiler; engine build and settings from llama.cpp's `/props` or vLLM's `/version`; the model by GGUF hash or Hugging Face revision. A mismatch with the claimed config is flagged.
3. **Physical bounds.** Decode cannot beat memory bandwidth over bytes read per token, and KV capacity cannot exceed the memory left after the weights. The model computes both; a run above its own hardware's roofline is rejected.
4. **Only exclusive runs rank.** `mode` is already in the record; a shared endpoint's numbers are not comparable.
5. **Outliers against their cluster.** A run far from the median of the same hardware and config is flagged. A result is **confirmed** once two accounts reproduce it.
6. **Privacy before upload.** Hostnames, URLs and IPs are stripped, and the CLI prints exactly what it sends.

That already exceeds every community site found above.

## The work

- **llama.cpp is a model change, not only an adapter.** Slots (`-np`) split the context unless `--kv-unified` is set, and its prompt-cache reuse differs from vLLM's paged prefix cache. It also needs a metrics adapter beside `metrics/vllm.py` (`llama-server --metrics`, `/slots`).
- **Consumer hardware in `GPUS`**, which holds H200 and B300 today: 3090, 4090, 5090, Apple M-series with unified memory, Strix Halo.
- **`ws detect` and `ws submit`**, and hardware and engine blocks in the run record, which captures neither today.
- **Stack**: the frontend stays on Cloudflare; the API is FastAPI and Postgres on an LXC behind Cloudflare; raw records in R2 or on disk.

## Order

1. llama.cpp support, `ws detect`, `ws submit`, run pages.
2. Seed with our own runs: an RTX 3090 and a Mac.
3. Post on r/LocalLLaMA.
4. Results tables once runs from others exist. An empty leaderboard at launch kills it.

## Open

- The bound behind "agents served": which p95 TTFT and inter-token gap, per workload profile.
- Whether SemiAnalysis's agentic traces are public and licensed for reuse.
