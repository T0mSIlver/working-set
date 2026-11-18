# Market prices against the model's cost

`ws market MODEL` prices the model's predicted capacity at market GPU rent
and sets each provider's price for the same model beside it. The result is
not whether a provider loses money: it is the utilisation a provider must
reach for its price to cover the rent, under the assumptions below, with the
band they leave.

```bash
uv run ws market GLM52                      # the report, on-demand rent
uv run ws market GLM52 --term committed     # the committed-rent band
uv run ws market GLM52 --json               # every configuration and rent level
uv run scripts/market_compare.py            # all models -> data/market/comparison-DATE.json
```

`--snapshot DIR` reads another snapshot directory; the default is
`data/market` in the checkout, and the command stops with an error naming
what is missing when it is not there.

## Inputs

| file | what | required |
|---|---|---|
| `provider_prices.json` | prices read from each provider's own page | one of the two |
| `openrouter-DATE.json` | OpenRouter's endpoint listing | one of the two |
| `gpu_rental-DATE.json` | USD per GPU-hour, per seller and term | yes |
| `chars_per_token.json` | characters per token per model and text kind | no: without it the per-character columns are left out |

Every row names the file it came from. A missing field is an error that
names it; a null price is recorded and that endpoint gets no break-even
figure.

## Derivation

All figures are in USD. Token prices and GPU rents are published in USD; a
seller listing in EUR is converted in the rental file at the ECB rate it
records.

1. **Workload.** The default `[workload]` of `ws init`, which the explorer
   also opens on. Per request: E[L] prompt tokens, `read` of them from a
   prefix cache (L − Δ on a hit), `written` computed fresh (Δ on a hit, L on
   a miss), and 400 output tokens. A live session sends (1 + sub_ratio) /
   think_time_s requests per second, 132 an hour. These are the flows
   `ws api` bills (`api_cost.request_flows`).
2. **Capacity.** For each GPU, each TP width vLLM accepts up to one 8-GPU node
   (catalogue `vllm_tp`) and each priced weight arm: L =
   `predicted_limit_users` × replicas, the binding ceiling of the verdict.
   DP replicas multiply sessions and GPUs alike, so DPk × TPt costs the same
   per session as TPt, and only DP1 is priced.
3. **Rent.** C = n_gpu × USD per GPU-hour, paid every hour, 24/7. Three
   levels from the rental file: the lowest, median and highest seller (one
   figure per seller, its cheapest listing; aggregators left out), per term.
4. **Utilisation.** u is the time average of live sessions ÷ L over every
   rented hour. Cost per live session-hour at u: C / (L u).
5. **A provider's price for the workload.** R = what one live session-hour
   costs on the endpoint's price sheet: fresh tokens at input, cached tokens
   at the published cache-read price (at input where none is published),
   cache writes at the published write price where there is one, output at
   output. Providers publish no cache lifetime, so a 300 s TTL is assumed
   (`--cache-ttl`). A published discount or time-of-day price is flagged and
   not applied. An endpoint whose context is below the workload's 180k cap is
   flagged with the share of requests longer than it.
6. **Break-even utilisation.** u* = (C / L) / R, taken on the configuration
   with the lowest cost per session among those the endpoint's declared
   weights map to, at each rent level. u* > 1 means the price does not cover
   the rent at the model's predicted limit even fully loaded, under these
   assumptions; the provider's hardware, engine, contract prices and
   workload are unknown.
7. **Per-token costs.** GPU time is split by the model's own duties
   (`model.power_draw`): prefill time to fresh input tokens, decode time to
   output tokens, idle time to both in proportion to their busy time; cached
   tokens carry no compute. This split is a convention, and u* does not
   depend on it. The figure the page plots is USD per million tokens of the
   workload mix: every prompt token plus the output in the denominator.
8. **Model error.** For each configuration, the limit is re-predicted under
   every measured error of that model in [docs/accuracy.md](accuracy.md):
   decode step time × (1 + e) for a decode row, prefill MFU × (1 + e) for a
   prefill row. A model with no row is marked unchecked; another model's
   error is never borrowed. The error band and the rent band are shown side
   by side, never multiplied into one number.
9. **Fidelity.** Per endpoint: context offered ÷ the catalogue's maximum
   context, maximum output, and the declared quantization with the arm it
   is priced as. FP8 maps to the base arm; FP4 and MXFP4 map to the base arm
   of a model whose released checkpoint has MXFP4 experts (DeepSeek-V4,
   V4.1, Kimi-K3.1-Pro) and to the NVFP4 arm otherwise; BF16 is heavier than
   any priced arm, so its cost is a lower bound; an undeclared quantization
   takes every priced arm.
10. **Throughput.** Where a provider publishes tokens per second, it is
    printed beside the model's per-user decode range (one sequence alone, and
    at the predicted limit) over the configurations its weights map to. The
    hardware is unknown, so this is a sanity range;
    [docs/accuracy.md](accuracy.md) lists it apart from the statistics.
