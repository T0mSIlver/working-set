# Artificial Analysis AgentPerf

Read 2026-11-04 from artificialanalysis.ai (methodology page, launch article
of 2026-06-12, the hardware pages). No data file: see Terms.

## What it measures

Real coding-agent trajectories recorded in the OpenCode harness, replayed
closed-loop: 5K to 131K input tokens per request (mean ~27K), up to 200
turns per session, tool delays of 0.1 to 5 s. The test set is private;
vendors get a tuning subset. Prefix caching, speculative decoding and
disaggregated prefill/decode are allowed; prefixes are randomised between
load phases. The headline is the most concurrent agents a system serves at a
tier: a P25 per-request output speed plus a P95 TTFT (for DeepSeek V4 Pro,
tier 1 is 20 tok/s and 10 s, tier 2 60 tok/s and 5 s, tier 3 180 tok/s and
3 s), also per accelerator, per megawatt of measured GPU power and per rental
dollar. Vendors submit tuned configs that AA runs and verifies, or AA builds
one; every result names its submitter. Systems: H100, H200, B200, B300 and
MI355X boxes of 8, GB200, GB300 and Vera Rubin NVL72 racks.

DeepSeek-V4.1-Flash and GLM-5.3 (labelled "GLM-5.2 & 5.3") have load phases
on H200, B200 and B300; none of our other models appear. AA publishes official
tier results only for DeepSeek V4 Pro and gpt-oss-120b.

## Fit with Working Set

The closest outside test of what Working Set predicts: a concurrency ceiling
under a latency SLO on a real agent workload. It is a **check**: run
`verdict` closed-loop at AA's floor and TTFT percentile on AA's prompt-length
distribution and compare the ceiling. AA's P25-speed tier is not our 40 tok/s
floor, so the mapping must be stated per check.

## Terms

The site's Terms of Use (2.2(d)): "no part of the Site may be copied,
reproduced, distributed, republished, downloaded, displayed, posted or
transmitted in any form or by any means", and section 4(vi) bars scraping.
The Data Platform Terms (2.3) allow "brief citations of individual Data
points in publicly available content" with the line "Source: Artificial
Analysis (artificialanalysis.ai)", but not "in a structured, tabular, or
machine-readable format", and (2.4, 2.5) bar combining the data into a
comparison or model-selection product without written consent.

So: no AgentPerf data in this repo or on the explorer. Checks run locally;
results reach the public repo only as a sentence citing one or two points
with the attribution line. Written permission (agentperf@artificialanalysis.ai)
would lift this.

The free API has no hardware or AgentPerf endpoint.
