# Self-host vs API cost calculators

Read 2026-11-04. Records: `data/external/calculators.json` (kind guess).


Findstack and Spheron are articles with a formula, not interactive tools. I found no Spheron calculator (`/tools/` and `/calculator/` return 404). InventiveHQ's calculator blocks automated fetches (Cloudflare 403), so it is unread.

| | BenchLM | GPUPerHour | AppScale | Findstack | Spheron |
|---|---|---|---|---|---|
| URL | benchlm.ai/tools/self-host-calculator | gpuperhour.com/tools/self-host-vs-api-cost-calculator | appscale.blog/en/tools/inference-cost-calculator | findstack.com/resources/self-hosted-llm-vs-api-cost | spheron.network/blog/migrate-from-openai-api-to-self-hosted-llms-cost-2026 |
| Type | tool | tool | tool | article | article |
| Default hardware | 8x H100, Q4, GLM-5.1 | 2x H100 FP8, Llama 3.3 70B | 2 GPUs, generic | 1x A100, 70B | 1x H100, 70B FP8 |
| GPU $/GPU-h | about 3.99 (derived from $23,302/mo) | 2.50 (live Hyperstack) | 2.50 | 1.19 (RunPod) | 2.64 (H100), 5.51 (H200) |
| Hours/month | 730 assumed | 730 | 730 | 730 | 24 h/day |
| Utilization | input, default 70%, no effect on cost or break-even | output (1.3% at default) | not modeled | 100% base; real 5-15% | full-use floor only |
| Throughput | not shown | 2,895 output tok/s at 100 concurrent (NVIDIA NIM) | 2,500 tok/s fleet (user input) | 20 tok/s (unmeasured) | about 400 tok/s "at typical concurrency" |
| Volume input | requests/day, tokens in/out | tokens/month in/out | requests/day, tokens in/out | tokens/month | tokens/day |
| Overhead | 20% default | excluded | excluded | excluded | $200/mo floor (cited) |
| Default result | break-even 405M tok/day | break-even 5,887M out tok/month; API wins | break-even about 8,004 req/day | no break-even vs cheap APIs | 12.2M tok/day vs GPT-5.4 |
| Concurrency | no | only via the benchmark's throughput | no | no | no |
| Seats / users | no | no | no | no (single user) | no |
| Latency | no | no | no | no | no (advises measuring TTFT, p95) |

### Formulas

- BenchLM: self-host/month = fixed node rent x (1 + overhead). API/month = tokens/day x 30 x blended $/Mtok (70% input, 30% output). Break-even = self-host / API per tok/day. Default API blended price works out to $2.30/Mtok (derived). Checked live on 2026-11-04: overhead 0% gave $23,302 and break-even 338M/day; utilization 35% or 100% changed nothing.
- GPUPerHour: always-on cost = servers x 730 h x GPU count x $/GPU-h, servers sized to monthly output at measured tok/s. Busy-hours cost = full-load hours only. Break-even = always-on cost / API $ per Mtok output at your mix.
- AppScale: self-host/month = GPUs x $/h x 730. Break-even is where it equals the API bill.
- Findstack: break-even tokens/month = monthly GPU cost / API price per token, then check against the card's ceiling. Result: "impossible" against 70B APIs at 20 tok/s.
- Spheron: break-even Mtok/day = (GPU $/h x 24) / blended API rate, flat until the throughput ceiling (about 34.5M tok/day on one H100).

All five treat the GPU bill as fixed and only count tokens. None sizes the node from a concurrent-user requirement.

### Overlap with Working Set

- Models: BenchLM names GLM-5.3, DeepSeek V4 Pro, Qwen3.8-27B, Kimi K3, Qwen3.8 Max in its text, but its default is GLM-5.1 and it shows no throughput. GPUPerHour, AppScale, Findstack, Spheron use Llama 70B only. No tool covers Qwen3.6-35B-A3B, Mistral-Medium-3.5, DeepSeek-V4(.1)-Flash, Qwen3.8-Flash-Next, GLM-5.3-Flash or Mistral-Large-4.
- GPUs: H100 everywhere; H200 only in Spheron; B300 only in BenchLM's FAQ ($48,000/mo for 8x B300, about $8.22/GPU-h derived); B200 nowhere.
- BenchLM's own FAQ says Qwen3.8-27B-class models "fit on a single consumer GPU" and GLM-5.3 class need a full 8-GPU node, which matches our deployments.

## Suggested stated defaults and ranges

GPU price (per GPU-h, rented, on-demand). Present as range with the source:

| GPU | Low | High | Source |
|---|---|---|---|
| H100 | 2.50 | 3.99 | GPUPerHour (Hyperstack live), Spheron 2.64, BenchLM derived 3.99 |
| H200 | n/a | 5.51 | Spheron live; no second source |
| B300 | n/a | 8.22 | BenchLM FAQ derived; one source, "about" |
| B200 | none found | | |

H200 and B300 each rest on one source; flag them as thin and keep Working Set's own price inputs user-editable. For owned hardware, BenchLM amortizes over 4 years plus electricity and PUE (stated in its FAQ; the inputs are not shown).

Other defaults:

- Hours per month: 730. Three tools agree.
- Overhead on top of GPU rent: 20% (BenchLM) is the only figure scaled to an 8-GPU node. The $200/month Spheron floor suits a single GPU and is a floor. Offer 0-20%, default 20%, and say it covers engineer time, monitoring, egress.
- Utilization: do not copy BenchLM's 70%, it does not drive its result. GPUPerHour computes it from volume and throughput, which is what Working Set should do. Findstack's 5-15% for single-user self-hosting is an unsupported claim but a useful floor for a "bad case" row. Spheron/GMI-style 40% duty cycle appears only in a secondary search summary, not checked.
- API blended price: 70% input / 30% output (BenchLM), 80/20 or 60/40 (Spheron). Working Set agent traces are input-heavy with prefix reuse, so 80/20 uncached overstates API cost unless cached-input pricing is modeled. Note both calculators say cached input (10% of price, per Findstack) is excluded or optional.
- Break-even is quoted as Mtok/day in BenchLM (405M with a 8xH100 default; 264M in a snippet for another config) and Spheron (12M to 27M for one H100), so a "break-even tokens/day" column in Working Set would sit in familiar units. Add seats and latency, which none provides.

## Cautions

- Calculator numbers are assumptions, kind guess. Vendor-hosted ones (Spheron sells GPUs, Findstack discloses affiliate links) lean toward self-hosting or rental respectively.
- Findstack's 20 tok/s is unmeasured and single-stream; its "impossible" verdicts do not apply to batched multi-user serving.
- AppScale's displayed verdict contradicts its own numbers at default volume.
- GPUPerHour's input/output volume defaults (300M/100M) are inferred from displayed outputs, not stated on the page.
- BenchLM's page changes with its selectors; the default was read from the rendered page on 2026-11-04.
