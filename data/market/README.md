# data/market/

What providers charge for the catalogue models, and what GPUs rent for.

## OpenRouter

`scripts/market_openrouter.py` reads two documented routes of OpenRouter's
public API, `GET /api/v1/models` and `GET /api/v1/models/{author}/{slug}/endpoints`,
for the 11 catalogue models. It matches each model by the API's
`hugging_face_id` and writes every provider endpoint's prices (USD per million
tokens, converted with `decimal` from the per-token strings), context length,
quantization, uptime and status to `data/market/openrouter-<date>.json`.
Prices OpenRouter does not publish are `null`; `pricing.discount` and
time-of-day prices (`pricing.overrides`) are kept under `other_pricing`.
Where an endpoint has time-of-day prices, `time_weighted` gives the price
averaged over 24 hours of UTC windows; the listed price stays as published.

The snapshot is not in the repo. OpenRouter's
[Terms of Service](https://openrouter.ai/terms) (Last Updated: August 31, 2026)
forbid using "scripts, robots or any other means [...] to scrape or copy any
information on the Site or the Services" (§7), and say of the "compilation,
information, data" of the Service: "Except as expressly authorized by
OpenRouter, you may not make use of the Materials" (§12). Publishing it waits on
the owner's decision; `.git/info/exclude` keeps it out of commits.

To make one locally:

```
uv run scripts/market_openrouter.py            # re-uses the cache
uv run scripts/market_openrouter.py --refresh  # fetches again
```

Raw responses are cached in `~/.cache/workingset/market/openrouter/`
(`--cache DIR` to change). The file is dated by the fetch of the models list.

`provider_prices.json` holds what providers publish on their own price pages,
read by hand on 2026-11-18, each row with the quoted line and the page URL. It
is the published source. Each provider's terms were checked first: Alibaba,
Moonshot, Z.ai, Fireworks, Novita, Parasail, SiliconFlow, Cerebras, SambaNova
and Groq forbid scraping or reusing site content, so they have no rows (the
clause is under `providers`); Nebius's page returned 403. Rows come from
DeepInfra, Together AI and Mistral. A row whose model is a different checkpoint
from the catalogue's is under `dropped` with the reason: DeepSeek's
`deepseek-flash` is V4.1-Flash-1017, not DSV41F, and "Qwen3.8-Flash" is not
confirmed to be Qwen3.8-Flash-Next.

`price_crosscheck.json` sets 12 of those rows against OpenRouter's listing for
the same provider. OpenRouter's prices are the providers' current prices after
any promotion: DeepInfra's page shows GLM-5.3 at $0.90/$4.00 struck through and
$0.5625/$2.50 at "38% off", and OpenRouter lists $0.5625/$2.50 with
`pricing.discount` 0.375, the fraction off the list price.

## Tokenizers

A price per million tokens compares across models only once tokens are turned
back into text, because each model's tokenizer cuts the same text into a
different number of tokens. `scripts/market_tokenizers.py` tokenizes the fixed
corpus in `corpus/` (four kinds: `code`, `english`, `french`,
`json_tool_calls`; sources and licences in `corpus/README.md`) with each
catalogue model's own tokenizer and writes `chars_per_token.json`:
`models.<KEY>.chars_per_token.<kind>` is Unicode characters per token, so a
price per million tokens divided by it is a price per million characters.

Tokens are counted on plain text: no special tokens, no chat template. Each
model row names the Hugging Face repo, the commit the tokenizer files were read
at, and the SHA-256 of the tokenizer file, so two models share a tokenizer when
their `tokenizer_sha256` match. The script downloads tokenizer files only
(`tokenizer.json`, `tekken.json` for Mistral, `tiktoken.model` for Kimi), never
weights, and loads them as data; Kimi's split pattern is copied into the script
and checked against the repo's `tokenization_kimi.py`, which is never run.

```
uv run scripts/market_tokenizers.py --hf-cache /models/hub   # reads a local HF cache first
uv run scripts/market_tokenizers.py --offline                # fails when a file is missing
```

Downloads go to `~/.cache/workingset/market/tokenizers/<repo>/`
(`--cache DIR` to change).

Characters per token, 2026-11-18:

| Model | code | english | french | json_tool_calls |
|---|---|---|---|---|
| 27B | 3.29 | 3.99 | 3.50 | 2.80 |
| 35BA3B | 3.29 | 3.99 | 3.50 | 2.80 |
| MM35 | 3.29 | 3.99 | 3.71 | 2.69 |
| GLM52 | 3.50 | 4.07 | 3.40 | 3.01 |
| DSV4F | 3.31 | 4.02 | 3.33 | 3.04 |
| DSV4FV | 3.31 | 4.02 | 3.33 | 3.04 |
| DSV41F | 3.31 | 4.02 | 3.33 | 3.04 |
| Q38FN | 3.29 | 3.99 | 3.50 | 2.80 |
| GLM53F | 3.50 | 4.07 | 3.40 | 3.01 |
| ML4 | 3.41 | 3.94 | 3.71 | 3.05 |
| K31P | 3.48 | 4.01 | 3.03 | 3.12 |

The same text costs up to 22% more tokens on one model than another in French
(MM35 and ML4 3.71 chars/token, K31P 3.03), 16% in tool-call JSON (K31P 3.12,
MM35 2.69), 6% in code and 3% in English. Shared tokenizers: 27B and Q38FN
(identical file; 35BA3B differs by a few added tokens and counts the same),
GLM52 and GLM53F, DSV4FV and DSV41F (DSV4F's file differs slightly and counts
the same).

## GPU rental

`gpu_rental-2026-11-18.json` holds what an H200 and a B300 rent for per
GPU-hour, read on 2026-11-18 from each seller's own price page or price API,
with the cell quoted. `usd_per_gpu_h` is derived: the listed price divided by
the GPUs in the unit (and by the hours in the term for monthly or whole-term
prices), times the ECB rate of the day (1.0968 USD per EUR) for EUR prices;
`price_listed`, `currency_listed` and `unit_listed` keep what the page said.
Prices exclude VAT. A seller that shows "Contact sales" has a record with
`usd_per_gpu_h: null`. Verda's committed rows apply its listed term discounts
to its on-demand rate; their `note` says so.

Four fields keep unlike offers apart. `tier` is `managed` (the seller's own
fleet), `community` (third-party hosts under the seller's brand: RunPod
Community Cloud) or `marketplace` (Vast.ai: a median over independent
hosts), and `null` for aggregators. `platform` is `HGX`, `GB300` (a
Grace-Blackwell rack, not an HGX B300) or `unstated`, as the page names it.
`availability` holds the page's words when it marks an offer (Together's B300:
"sold out"), else `null`. `multi_gpu` is the largest GPU count per instance
the page offers at that price, `null` where the page does not say: eight
1-GPU instances are not one connected 8-GPU node.

One figure per seller, its cheapest offer; GB300 rows excluded by `platform`:

| per GPU-hour, low / median / high (sellers) | on-demand | committed | spot |
|---|---|---|---|
| H200, all tiers | $3.59 / $5.58 / $10.60 (12) | $3.40 / $5.18 / $7.89 (5) | $2.29 / $2.79 / $3.45 (4) |
| H200, managed | $4.29 / $5.76 / $10.60 (11) | $3.40 / $5.18 / $7.89 (5) | $2.29 / $2.79 / $3.45 (4) |
| B300, all tiers | $6.94 / $10.82 / $15.00 (8) | $5.87 / $9.59 / $16.15 (5) | $4.48 / $5.96 / $6.39 (4) |
| B300, managed | $8.96 / $11.49 / $15.00 (7) | $5.87 / $9.59 / $16.15 (5) | $4.48 / $5.96 / $6.39 (4) |

Nebius's preemptible "from $0.79" and "from $0.99" are floors of a market
price and stay out of spot.
The catalogue's `eur_gpu_h` (€3.80 for the H200, €6.80 for the B300, a
getdeploying.com median from 2026-09) comes to $4.17 and $7.46 at this rate:
below the managed sellers' on-demand medians by about 28% and 35%, and below
getdeploying's own medians of the day ($4.59 and $9.00), which rose 12% and
18% over 90 days by its account.

Not read: Google Cloud (prices drawn by JavaScript, the billing API needs a
key), Scaleway (bot check), Lambda (lists no H200 or B300), Azure for the
B300 (no B300 or GB300 rows in its Retail Prices API), AWS on-demand (drawn
by JavaScript; the AWS rows are Capacity Block prices from AWS's page, and
on-demand comes from the Vantage mirror, recorded as an aggregator).

`scripts/market_gpu_rental.py` re-fetches every page that serves its prices
in plain HTML or JSON and prints, per record, whether the listed price is
still next to its GPU (`same`), gone (`MOVED`: re-read that page), or needs a
browser. It never edits the JSON. Vast.ai is a live marketplace, so its line
prints the current median of the offers instead.

```
python3 scripts/market_gpu_rental.py [data/market/gpu_rental-<date>.json]
```

`source.terms` quotes each seller's website terms where they restrict
copying or automated collection of the site's content, with the URL; eight of
the fifteen have such a clause (`restrictive: true`), two were not read.

`gpu_rental_history.json` holds the same offers on earlier dates, read from
Internet Archive captures of the same pages (2025-01 to 2026-11). It has the
fields above plus `captured` and `archive_url`. A record is a capture where a
price was seen, so a price holds between two records of the same offer.
EUR prices use the 2026-11-18 rate.
