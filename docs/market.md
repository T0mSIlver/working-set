# Market prices against the model's cost

This is rent coverage for a hypothetical deployment at our workload, cache
assumptions, scheduler settings and latency limits; it does not estimate a
provider's actual utilisation, throughput capacity or profit.

`ws market MODEL` prices the model's predicted capacity at market GPU rent
and sets each provider's price for the same model beside it. The figure it
reports for each price is u\*, the utilisation at which that price covers the
rent, under the assumptions below.

```bash
uv run ws market GLM52                          # the report, on-demand rent, managed offers
uv run ws market GLM52 --term committed-long    # 12-month and longer commitments
uv run ws market GLM52 --rent-tiers all         # community-cloud and marketplace offers too
uv run ws market GLM52 --sensitivity            # which assumption moves u* most
uv run ws market GLM52 --workload no-cache      # the no-cache contrast (rough)
uv run ws market GLM52 --json                   # every configuration and rent level
uv run scripts/market_compare.py                # all models -> data/market/comparison-DATE.json
```

`--snapshot DIR` reads another snapshot directory. The default is
`data/market` in the checkout, and the command stops with an error naming
what is missing when it is not there.

## Inputs

| file | what | required |
|---|---|---|
| `provider_prices.json` | prices read from each provider's own page | one of the two |
| `openrouter-DATE.json` | OpenRouter's endpoint listing | one of the two |
| `gpu_rental-DATE.json` | USD per GPU-hour, per seller, term, tier and platform | yes |
| `chars_per_token.json` | characters per token per model and text kind | no: without it the per-character columns are left out |

Every price row names the file it came from. A missing field is an error that
names it; a null price is recorded and that endpoint gets no u\*. Rows are
endpoint/tier rows: one provider's regions, speed tiers, and list and current
prices are separate rows, not independent sellers.

## Derivation

All figures are in USD. Token prices and GPU rents are published in USD; a
seller listing in EUR is converted in the rental file at the ECB rate it
records.

1. **Workload.** The default `[workload]` of `ws init`, which the explorer
   also opens on. Per request: E[L] = 40,082 prompt tokens, of which some are
   read from a prefix cache (L − Δ on a hit) and the rest computed fresh (Δ
   on a hit, L on a miss), plus 400 output tokens. A live session sends 132
   requests an hour. Tool and human pauses are inside a live session, so
   utilisation is not GPU busy time.
2. **Mix.** As shares of every prompt token plus the output: at the model's
   1% miss rate (the capacity side), 93.1% cached input, 5.9% fresh, 1.0%
   output; under the assumed 300 s cache lifetime (the billing side), 3.70%
   misses and 90.6% cached.
3. **Capacity.** For each GPU, each TP width vLLM accepts up to one 8-GPU node
   (catalogue `vllm_tp`) and each priced weight arm: L =
   `predicted_limit_users` × replicas. That limit is the lowest of four
   ceilings at the explorer's default latency limits: slowed streams at most
   15% of wall time in cold-prefill steps, p95 cold queue wait at most 5 s,
   prefill saturation, and decode at a 40 tok/s floor under the recommended
   `max_num_seqs`. These are the explorer's defaults, not a provider's
   commitments. DP replicas multiply sessions and GPUs alike, so only DP1 is
   priced.
4. **Rent.** C = n_gpu × USD per GPU-hour, paid every hour, 24/7. Three
   levels: the lowest, median and highest seller, one figure per seller (its
   cheapest listing). The priced band counts managed-cloud offers only;
   `--rent-tiers all` adds community-cloud and marketplace offers, and the
   report prints the other band beside the priced one. GB300 instances never
   count: the catalogue prices HGX B300. Sold-out offers count and are named.
   Terms: `on-demand`, `committed-short` (under 3 months) and
   `committed-long` (12 months and more); other commitment lengths are
   counted and left out. Availability and eight-GPU fabric are not verified.
5. **Utilisation.** u is the time average of live sessions ÷ L over every
   rented hour. Cost per live session-hour at u: C / (L u).
6. **A provider's price for the workload.** R = what one live session-hour
   costs on the endpoint's price sheet: fresh tokens at input, cached tokens
   at the cache-read price (at input where none is published), cache writes
   at the write price where there is one, output at output. The price files
   carry no cache lifetime, so 300 s is assumed (`--cache-ttl`). A listed
   price is already after any published promotion, and the row says so. An
   endpoint with time-of-day prices is priced at its 24-hour time-weighted
   price, with the listed window's u\* beside it. An endpoint whose context is
   below the workload's 180k cap is flagged with the share of requests longer
   than it.
7. **u\*.** u\* = (C / L) / R, on the configuration with the lowest cost per
   session among those the endpoint's declared weights map to, at each rent
   level. u\* > 1 means the price does not cover the rent at the model's
   predicted limit even fully loaded, under these assumptions. The JSON also
   gives `u_star_no_cache_discount`: the same limit with cached tokens billed
   at input, a revenue-side sensitivity only.
8. **Per-token costs.** GPU time is split by the model's own duties
   (`model.power_draw`): prefill time to fresh input tokens, decode time to
   output tokens, idle time to both in proportion to their busy time. This
   split is a convention, and u\* does not depend on it. Cached tokens carry
   no prefill, but their KV holds HBM and decode reads it. The figure the page
   plots is USD per million tokens of this mix, cached input included.
9. **Model error.** The limit is re-predicted under each measured error of the
   same model in [docs/accuracy.md](accuracy.md): decode step time × (1 + e)
   for a decode row, prefill MFU × (1 + e) for a prefill row. Only a row on
   the binding quantity bounds the limit. Without one the row says "capacity
   unchecked"; another model's error is never borrowed. The error band and
   the rent band are shown side by side, never multiplied into one number.
10. **Fidelity.** Per endpoint: context offered ÷ the catalogue's maximum
    context, maximum output, and the declared quantization with the arm it
    is priced as. FP8 maps to the base arm. FP4 and MXFP4 map to the base arm
    of a model whose released checkpoint has MXFP4 experts (DeepSeek-V4,
    V4.1, Kimi-K3.1-Pro) and to the NVFP4 arm otherwise. BF16 is heavier than
    any priced arm, so its cost is a lower bound. An undeclared quantization
    takes every priced arm.
11. **Throughput.** Where a provider publishes tokens per second, it is
    printed beside the model's per-user decode range (one sequence alone, and
    at the predicted limit) over the configurations its weights map to. The
    hardware is unknown, so this is a sanity range, listed in
    [docs/accuracy.md](accuracy.md) apart from the statistics.

## Sensitivity

`--sensitivity` moves one assumption at a time and reports how far u\* moves,
holding each endpoint's cheapest configuration at median rent. A ±20% step
gives an elasticity, d ln u\* / d ln x by central difference. A named switch
gives the change ln(u\*_variant / u\*). The assumptions are rent, the limit
itself, decode step time and prefill MFU (through the re-predicted limit),
the cache lifetime (60 s and 3,600 s), cached tokens billed at input, output
tokens per request (200 and 800), the miss rate, the warm turn Δ, think time
and the prompt median. A workload change re-predicts the limit and re-bills
the price sheet. The report ranks the assumptions by the median over
endpoints of the largest |ln change|, and names for each endpoint the one its
u\* moves most with, rent and the limit itself left aside.

## The no-cache contrast

`--workload no-cache` is the default workload with every request a miss:
every prompt token is computed fresh and billed at the input price, and the
limit is re-predicted. It is a rough sensitivity, not an estimate. At 100%
misses, `predict()`'s recommended `max_num_seqs` falls to 1, because the
GPU-resident count it rests on counts only reusable sessions. The contrast
therefore sets `max_num_seqs` explicitly to the contexts of mean length the
HBM pool holds.

## What this does not say

A provider's hardware, engine, contract rents, traffic mix, batch or
priority tiers and its own cache lifetime are unknown. The limit is the
model's verdict at its default latency limits, which a provider need not
share. The gap between a price and the model's cost is not a provider's
margin, and u is not a provider's measured utilisation.
