# API and coding-plan seats: prices, limits, and the cost model

This file prices one agentic-coding seat on a metered API from the workload
the self-host rows use, so the frontier table and chart H can set them side
by side, and states what each coding plan's published limits hold of that
workload (§ 5). The data ships as
`src/workingset/data/api_prices.toml`; the model is
`src/workingset/api_cost.py`; `ws api [CONFIG] [--hours H]` prints it.
Everything below was read on 2026-11-04. Prices are list, before tax, in
USD per million tokens unless stated, converted at the ECB reference rate of
that day, €1 = $1.1267.

## What the comparison assumes

Self-hosting and an API bill different things. A GPU costs the same every
hour whether it works or idles; an API costs per token sent. Comparing them
needs the assumptions below, each of which moves the answer. The explorer
sets every one of them with a control; this list is where they are written
down together.

Self-hosted (`model.energy_cost`, `research/power.md`):

1. **GPUs are rented by the hour**, €3.80 per H200-hour by default (the
   cross-provider median of September 2026). Owned hardware enters as its
   amortised cost per GPU-hour. Purchase, depreciation period and resale
   value are not modelled separately.
2. **The bill runs 720 hours a month**, busy or idle. Idle GPUs are paid
   for and do nothing else: batch jobs at night are not credited.
3. **Electricity** €0.19/kWh at a PUE of 1.5, from the duty cycle the load
   implies. On a rented GPU this double-counts power the rent already
   includes, about 7% at the defaults.
4. **The hardware is sized for the peak**: every seat live at once unless
   the population layer says otherwise (`peak_active_share`).
5. **Not counted:** staff, networking, storage, the engineering to run it,
   and what self-hosting buys outside the bill (data control, no rate limits).

API (this file, § 1 to 3):

6. **A seat is live H hours a month**, 84 by default (4 h a day, 21 days).
   This is the softest number on either side; Requesty's public data puts
   the median Claude Code user near 11 h and the 95th percentile near 137 h.
7. **List prices before tax**, read 2026-11-04, at the cheapest cache
   option for this workload, the cache lapsing on gaps longer than its TTL.
8. **No volume discount, committed-use price or rate limit.**

Both:

9. **The same workload**: a request every 30 s per live session plus
   subagents, the prompt lengths of the traced agent, 2,000 fresh tokens and
   400 output tokens a request.
10. **The same score**: Artificial Analysis's Terminal-Bench 4.0 at each
    model's index effort. Quantised self-hosted arms inherit the full model's
    score.

**The assumption-light comparison is the break-even.** The self-hosted bill
is flat in usage; the API bill is linear in it. So the question "which is
cheaper" has one free variable, the hours a seat runs, and the comparison
that needs no value for H is the H at which they cross:

    H* = self-hosted €/seat-month ÷ API €/active hour

At 64 seats on H200 and the defaults above: GLM-5.3-Flash 1,725 h (more than
the 720 h in a month, so the API is cheaper at any use), DeepSeek V4.1 Flash
3,521 h (the API serves the later 1017 checkpoint), Qwen3.8-27B 76 h,
Qwen3.6-35B-A3B 25 h (Alibaba serves it without a cache). Above H*
self-hosting wins; below it the API does.

## 1. What a seat is

A seat is one person with access. Self-hosting bills the hardware 24/7 and
the frontier table divides that bill by the concurrent sessions a
configuration carries. A metered API bills tokens, so its seat price is
linear in how many hours the seat is live: H active hours per month,
default 84 (4 h a day, 21 working days). With the population layer off,
every person is a live session at the peak, so the API €/seat and the
self-host €/seat are in the same unit. With `peak_active_share` ρ < 1 the
self-host side provisions only ρ of the people; the API seat still bills H
hours per person, so its price per concurrent session is €/seat ÷ ρ.

H is the comparison's softest input. Requesty's public gateway data
(requesty.ai/coding-agent-economy, May 2025 to April 2026, Claude Code: 92%
cache hits, mean $108, median $23, P95 $291 per active user-month) puts the
median user near 11 h and the P95 user near 137 h on this workload's request
cost (§ 3). 84 h is a heavy user, not a typical one; the explorer exposes it.

## 2. API prices

Vendor pages only; OpenRouter's public endpoint API
(`/api/v1/models/<id>/endpoints`) was read for the vendor's own endpoint as a
cross-check, never for a reseller's price. Cache write prices include the
cache's lifetime. ¹ DeepSeek bills half price off-peak.

| offer | API id | input | cache read | cache write (TTL) | output | above a threshold | OpenRouter, vendor endpoint |
|---|---|---|---|---|---|---|---|
| [Claude Fable 5.1](https://platform.claude.com/docs/en/about-claude/pricing) | `claude-fable-5-1` | 10 | 0.25 | 12.5 (5 min); 20 (1 h) | 50 | — | same |
| [Claude Opus 5.5](https://platform.claude.com/docs/en/about-claude/pricing) | `claude-opus-5-5` | 4 | 0.2 | 5 (5 min); 8 (1 h) | 20 | — | same |
| [Claude Sonnet 5.5](https://platform.claude.com/docs/en/about-claude/pricing) | `claude-sonnet-5-5` | 2 | 0.1 | 2.5 (5 min); 4 (1 h) | 10 | — | same |
| [Claude Haiku 5.5](https://platform.claude.com/docs/en/about-claude/pricing) | `claude-haiku-5-5` | 0.1 | 0.01 | 0.125 (5 min); 0.2 (1 h) | 0.5 | >100k: 0.5 / 2.5 | same |
| [GPT-6 Astra](https://platform.openai.com/docs/pricing) | `gpt-6-astra` | 10 | 1 | 12.5 (30 min); 25 (12 h) | 50 | >272k: 20 / 75 | same |
| [GPT-6.1 Sol](https://platform.openai.com/docs/pricing) | `gpt-6.1-sol` | 2 | 0.1 | 2.5 (30 min); 5 (12 h) | 10 | >272k: 4 / 15 | same |
| [GPT-6.1 Terra](https://platform.openai.com/docs/pricing) | `gpt-6.1-terra` | 1 | 0.05 | 1.25 (30 min); 2.5 (12 h) | 5 | >272k: 2 / 7.5 | same |
| [GPT-6 Luna](https://platform.openai.com/docs/pricing) | `gpt-6-luna` | 0.1 | 0.01 | 0.125 (30 min); 0.25 (12 h) | 0.5 | >272k: 0.2 / 0.75 | same |
| [Gemini 4 Argon](https://ai.google.dev/gemini-api/docs/pricing) | `gemini-4-argon` | 2 | 0.1 | 2 (5 min) | 10 | — | same |
| [Gemini 3.8 Flash](https://ai.google.dev/gemini-api/docs/pricing) | `gemini-3.8-flash` | 0.75 | 0.075 | 0.75 (5 min) | 3.75 | — | same |
| [GLM-5.3](https://docs.z.ai/guides/overview/pricing) | `glm-5.3` | 1.4 | 0.26 | 1.4 (5 min) | 4.4 | — | same |
| [GLM-5.3-Flash](https://docs.z.ai/guides/overview/pricing) | `glm-5.3-flash` | 0.15 | 0.03 | 0.15 (5 min) | 0.5 | — | same |
| [DeepSeek V4.1 Flash 1017 ¹](https://api-docs.deepseek.com/quick_start/pricing) | `deepseek-flash` | 0.3 | 0.006 | 0.3 (1 h) | 1.2 | — | differs: 0.15 / 0.6 |
| [DeepSeek V4 Pro 0813 ¹](https://api-docs.deepseek.com/quick_start/pricing) | `deepseek-v4-pro` | 1.32 | 0.044 | 1.32 (1 h) | 3.96 | — | differs: 0.66 / 1.98 |
| [Qwen3.8-27B](https://www.alibabacloud.com/help/en/model-studio/model-pricing) | `qwen3.8-27b` | 0.5 | 0.05 / 0.1 | 0.625 (5 min); 0.5 (5 min) | 3 | — | differs: 0.425 / 2.55 |
| [Qwen3.6-35B-A3B](https://www.alibabacloud.com/help/en/model-studio/model-pricing) | `qwen3.6-35b-a3b` | 0.375 | 0.375 | 0.375 (5 min) | 2.25 | — | none (resellers only) |
| [Qwen3.8 Max](https://www.alibabacloud.com/help/en/model-studio/model-pricing) | `qwen3.8-max` | 2 | 0.25 | 2.5 (5 min); 2 (5 min) | 6 | — | same |
| [Mistral Large 4](https://docs.mistral.ai/inference/pricing) | `mistral-large-4` | 0.7 | 0.07 | 0.7 (5 min) | 2.1 | — | same |
| [Kimi K3.1 Pro](https://platform.moonshot.ai/docs/pricing/chat) | `kimi-k3.1-pro` | 2 | 0.2 | 2 (5 min); 4 (1 h) | 10 | — | same |
| [Kimi K3](https://platform.moonshot.ai/docs/pricing/chat) | `kimi-k3` | 3 | 0.3 | 3 (5 min); 6 (1 h) | 15 | — | same |

Notes on the rows:

- **Anthropic.** Cache reads are 0.025× input on Fable 5.1, 0.05× on Opus
  and Sonnet 5.5, 0.1× on Haiku 5.5. Writes are 1.25× (5 min) or 2× (1 h).
  Only Haiku 5.5 has a long-context band: a whole request past 100,000
  prompt tokens bills at 5× ("A request's prompt length counts all of its
  input tokens, including cache reads and cache writes"). At the default
  cap that is 7.4% of main-agent requests.
- **OpenAI.** GPT-6 caching is opt-in per breakpoint; writes cost 1.25×
  input for 30 minutes or 2.5× for 12 hours. A request past 272K input
  tokens bills 2× input and 1.5× output. There is no GPT-6 "-pro" model;
  OpenRouter's `-pro` ids carry the same prices.
- **Google.** Gemini 4 Argon has implicit caching only (no write fee, hit not
  guaranteed, TTL not stated: priced at 5 min). Current Argon and 3.8 Flash
  prices are promotional: Argon doubles on 2026-12-01, 3.8 Flash on
  2027-01-01 (`price_changes` in the data file). Gemini 3.5 Flash-Lite
  is left out: 1.0% on Terminal-Bench 4.0.
- **Z.ai.** Cache storage is "Limited-time Free"; no write fee; TTL not
  stated (5 min assumed). GLM-5.3-FlashX and GLM-5.3-Prime are left out: no
  Artificial Analysis run, and Prime has no price on Z.ai's own page.
- **DeepSeek.** Only `deepseek-flash` (V4.1 Flash 1017) and `deepseek-v4-pro`
  are priced. The 2026-09-10 V4.1 Flash (the study's `DSV41F`) and V4 Flash
  Vision are retired aliases "served by the DeepSeek-V4.1-Flash-1017 model
  and billed at the Flash price". Peak is Monday to Friday, 01:00 to 04:00 and
  06:00 to 10:00 UTC. A 09:00 to 18:00 CET working day (08:00 to 17:00 UTC) spends
  2 of 9 hours at peak, so the model prices at 2/9 × peak + 7/9 × half =
  0.611 × the peak list (`time_blend`). OpenRouter's base rate is the
  off-peak price. The cache "will be automatically cleared, usually within
  a few hours to a few days" once unused: priced at a 1 h TTL.
- **Alibaba (Model Studio, International).** Qwen3.8-27B supports both cache
  modes: explicit (5 min, create 125%, hit 10%) and implicit (hit 20%, no
  write fee); the model takes the cheaper. Qwen3.6-35B-A3B is in neither
  supported-model list: no context cache, so every prompt token bills at
  input, which is why it costs as much as GLM-5.3. Qwen3.8 Max's hit price is
  "not 10%" and not printed ("see the console"); $0.25 is Alibaba's own
  endpoint on OpenRouter. That endpoint lists Qwen3.8-27B at $0.425 / $2.55,
  15% under the Model Studio page; the page's price is used. Qwen3.8-Flash
  is left out: it has no AA run, and secondary sources say Qwen3.8-Flash-Next
  (the study's `Q38FN`) is a different model.
- **Mistral.** Large 4 only. Medium 3.5 (deprecated 2026-10-31) and Devstral 2
  have no current price.
- **Moonshot.** Explicit cache writes at 1× (5 min) or 2× (1 h).

## 3. The cost of one request

Closed form in the normal CDF (`api_cost.py`, mirrored in
`interactive/src/apicost.js`); the derivation follows the API seat spec of
2026-11-04 with two changes to keep it on the self-host workload:

- **Request rate** is the self-host arrival rate, (1 + sub_ratio) /
  think_time per live session: 132 requests per hour at the 30 s default,
  not the spec's 91 from the measured 43.3 s cycle. The explorer's think
  slider moves both sides.
- **Fresh tokens per request** Δ is the warm turn the self-host side prefills
  on a hit (2,000 tokens, the turn slider), not a free parameter fitted to a
  public hit rate. It gives a 91.5% prompt-token hit rate on a 5-min cache,
  next to Requesty's 92% for Claude Code.

Per request of a class (main agent, or subagent at `sub_ratio`): the prompt
L is the clipped lognormal the self-host sampler draws. On a hit, L − Δ
tokens are read and Δ written; on a miss (probability q) all L are written.
Output is `out` tokens (400). A long-context band prices the whole request
by its prompt length; the share of requests and tokens in each band comes
from the same distribution (atoms at the prefix floor and the cap
included).

q is the configured invalidation plus cache expiry. The gap before a
request is a tool wait (6.8 of every 7.8 requests, lognormal median 0.61 s,
σ 2.43) or a human wait (median 58 s, mean 275 s, σ 1.76), the traced
anchors in `model.py`. A cache with TTL T lapses when the gap exceeds it:
q = 2.7% extra at 5 min, 0.4% at 30 min, 0.14% at 1 h. Each offer is priced at
the cheapest of its cache options for the workload; at Δ = 2,000 the 5-min
write beats Anthropic's 1-h write (2× vs 1.25× on every write), and
OpenAI's 30-min write beats its 12-h one.

The default workload's result (`ws api`, 84 h, before tax):

| offer | TB 4.0 | €/seat-month |
|---|---|---|
| DeepSeek V4.1 Flash 1017 | 33.8% | 8.6 |
| Claude Haiku 5.5 | 32.8% | 16.1 |
| GLM-5.3-Flash | 32.8% | 17.8 |
| GPT-6.1 Terra | 55.6% | 69.2 |
| GPT-6.1 Sol | 56.1% | 138 |
| Gemini 4 Argon | 57.1% | 143 |
| GLM-5.3 | 41.9% | 158 |
| Claude Sonnet 5.5 | 63.6% | 159 |
| Kimi K3.1 Pro | 67.2% | 179 |
| Claude Opus 5.5 | 59.6% | 319 |

Check against Requesty: at Sonnet-class prices one request costs $0.016,
so Requesty's median $23 a month is ~1,400 requests (~11 h at 132 per hour)
and its P95 $291 is ~18,000 requests (~137 h). The model's 84 h seat lands
between the two, as intended.

## 4. Terminal-Bench 4.0

Read from each model's Artificial Analysis page (embedded dataset field
`terminalBench40`), same protocol and one-lab rule as
`research/terminal_bench.md`: mini-SWE-agent v2.4.6, 66 tasks × 3 = 198
runs, the effort variant the Intelligence Index carries (the page's own
slug). AA's Claude labels read "(Max, Default Fallback)", in the page title
"max with fallback". The ten study models' scores were re-read the same day
and are unchanged.

| offer | AA name | effort | TB 4.0 | runs passed / 198 |
|---|---|---|---|---|
| Claude Fable 5.1 | [Claude Fable 5.1 (Max, Default Fallback)](https://artificialanalysis.ai/models/claude-fable-5-1) | max | 52.0% | 103 |
| Claude Opus 5.5 | [Claude Opus 5.5 (Max, Default Fallback)](https://artificialanalysis.ai/models/claude-opus-5-5) | max | 59.6% | 118 |
| Claude Sonnet 5.5 | [Claude Sonnet 5.5 (Max, Default Fallback)](https://artificialanalysis.ai/models/claude-sonnet-5-5) | max | 63.6% | 126 |
| Claude Haiku 5.5 | [Claude Haiku 5.5 (Max)](https://artificialanalysis.ai/models/claude-haiku-5-5) | max | 32.8% | 65 |
| GPT-6 Astra | [GPT-6 Astra (Max)](https://artificialanalysis.ai/models/gpt-6-astra) | max | 59.1% | 117 |
| GPT-6.1 Sol | [GPT-6.1 Sol (Max)](https://artificialanalysis.ai/models/gpt-6-1-sol) | max | 56.1% | 111 |
| GPT-6.1 Terra | [GPT-6.1 Terra (Max)](https://artificialanalysis.ai/models/gpt-6-1-terra) | max | 55.6% | 110 |
| GPT-6 Luna | [GPT-6 Luna (Max)](https://artificialanalysis.ai/models/gpt-6-luna) | max | 12.6% | 25 |
| Gemini 4 Argon | [Gemini 4 Argon (High)](https://artificialanalysis.ai/models/gemini-4-argon) | high | 57.1% | 113 |
| Gemini 3.8 Flash | [Gemini 3.8 Flash (High)](https://artificialanalysis.ai/models/gemini-3-8-flash) | high | 19.7% | 39 |
| GLM-5.3 | [GLM-5.3 (Max)](https://artificialanalysis.ai/models/glm-5-3) | max | 41.9% | 83 |
| GLM-5.3-Flash | [GLM-5.3-Flash](https://artificialanalysis.ai/models/glm-5-3-flash) | not labelled | 32.8% | 65 |
| DeepSeek V4.1 Flash 1017 | [DeepSeek V4.1 Flash 1017 (Max)](https://artificialanalysis.ai/models/deepseek-v4-1-flash-1017) | max | 33.8% | 67 |
| DeepSeek V4 Pro 0813 | [DeepSeek V4 Pro 0813 (Max)](https://artificialanalysis.ai/models/deepseek-v4-pro) | max | 14.1% | 28 |
| Qwen3.8-27B | [Qwen3.8 27B (Xhigh)](https://artificialanalysis.ai/models/qwen3-8-27b) | xhigh | 5.6% | 11 |
| Qwen3.6-35B-A3B | [Qwen3.6 35B A3B (Reasoning)](https://artificialanalysis.ai/models/qwen3-6-35b-a3b) | reasoning | 0.0% | 0 |
| Qwen3.8 Max | [Qwen3.8 Max (0902)](https://artificialanalysis.ai/models/qwen3-8-max) | not labelled | 38.9% | 77 |
| Mistral Large 4 | [Mistral Large 4 (High)](https://artificialanalysis.ai/models/mistral-large-4) | high | 24.7% | 49 |
| Kimi K3.1 Pro | [Kimi K3.1 Pro](https://artificialanalysis.ai/models/kimi-k3-1-pro) | not labelled | 67.2% | 133 |
| Kimi K3 | [Kimi K3 (Max)](https://artificialanalysis.ai/models/kimi-k3) | max | 12.6% | 25 |

## 5. Coding plans

A plan does not give the same usage as an API: it bills its price only while
the usage fits its limits, and how much fits is the vendor's to say. So a
plan gets no €/seat here. `ws api` states its price and the active hours of
this workload a month its published limits hold. The seat sends the API
seat's requests and the plan's meter prices them in its own unit; a
5-hour budget counts one window per working day and binds only if it holds
less than the 5 hours; a weekly one counts 30/7 weeks. Where the vendor
publishes no limit (Claude, ChatGPT) the plan says so and claims nothing.

At the default workload: GLM Coding Max 779 h, Pro 334 h, Lite 55 h (its
5-hour budget holds 2.6 h); Google AI Ultra 318 h, Pro 239 h (11 h a day);
Mistral Pro and Copilot Pro+ 33 h; Copilot Business 9 h.

| plan | model priced | $/month | budgets | over budget | basis |
|---|---|---|---|---|---|
| [Claude Pro](https://claude.com/pricing) | Claude Sonnet 5.5 | 20 | [$12 / 5h, $155 / week](https://wokey.ai/claude-usage-limits) | billed at API list | measured (one aggregator, 79 accounts), ±25% |
| [Claude Max](https://claude.com/pricing) | Claude Sonnet 5.5 | 100 | [$60 / 5h, $775 / week](https://wokey.ai/claude-usage-limits) | billed at API list | derived: documented 5x Pro x measured Pro (two single reports: $420 and $1,250 a week) |
| [Claude Ultra](https://support.claude.com/en/articles/17728952-what-is-the-ultra-plan) | Claude Sonnet 5.5 | 200 | [$1,549 / week](https://wokey.ai/claude-usage-limits) | billed at API list | measured (one aggregator, 18 accounts), equals the documented 10x Pro |
| [Claude Team, standard seat](https://support.claude.com/en/articles/9266767-what-is-the-team-plan) | Claude Sonnet 5.5 | 25 | [$20 / 5h, $181 / week](https://wokey.ai/claude-usage-limits) | billed at API list | measured (one aggregator, 258 accounts) |
| [Claude Team, premium seat](https://support.claude.com/en/articles/9266767-what-is-the-team-plan) | Claude Sonnet 5.5 | 125 | [$90 / 5h, $844 / week](https://wokey.ai/claude-usage-limits) | billed at API list | measured (one aggregator, 30 accounts) |
| [ChatGPT Plus (Codex)](https://help.openai.com/en/articles/9793128-about-chatgpt-pro-tiers) | GPT-6.1 Sol | 20 | [$55 / week](https://wokey.ai/codex-usage-limits) | billed at API list | measured (one aggregator, 12 accounts), low |
| [ChatGPT Pro 200 (Codex)](https://help.openai.com/en/articles/9793128-about-chatgpt-pro-tiers) | GPT-6.1 Sol | 200 | [$581 / week](https://wokey.ai/codex-usage-limits) | billed at API list | measured (one aggregator, 7 accounts) across a limit change; may be up to 2x |
| [GLM Coding Plan Lite](https://z.ai/subscribe) | GLM-5.3 | 18 | [2,000 credits / 5h, 10,000 credits / week](https://docs.z.ai/devpack/overview) | stops | documented |
| [GLM Coding Plan Pro](https://z.ai/subscribe) | GLM-5.3 | 80 | [12,000 credits / 5h, 60,000 credits / week](https://docs.z.ai/devpack/overview) | stops | documented |
| [GLM Coding Plan Max](https://z.ai/subscribe) | GLM-5.3 | 168 | [28,000 credits / 5h, 140,000 credits / week](https://docs.z.ai/devpack/overview) | stops | documented |
| [GitHub Copilot Pro+](https://github.com/features/copilot/plans) | Claude Sonnet 5.5 | 39 | [$70 / month](https://docs.github.com/en/copilot/concepts/billing/usage-based-billing-for-individuals) | billed at API list | documented (3,100 of the 7,000 credits are flex and can change) |
| [GitHub Copilot Business](https://github.com/features/copilot/plans) | Claude Sonnet 5.5 | 19 | [$19 / month](https://docs.github.com/en/copilot/concepts/billing/usage-based-billing-for-organizations-and-enterprises) | billed at API list | documented (pooled across the organisation) |
| [Google AI Pro (Gemini CLI)](https://one.google.com/about/google-ai-plans/) | Gemini 4 Argon | 19.99 | [1,500 requests / day](https://geminicli.com/docs/resources/quota-and-pricing/) | stops | documented (requests); which model the CLI routes to is not |
| [Google AI Ultra (Gemini CLI)](https://one.google.com/about/google-ai-plans/) | Gemini 4 Argon | 99.99 | [2,000 requests / day](https://geminicli.com/docs/resources/quota-and-pricing/) | stops | documented (requests) |
| [Mistral Pro (Vibe)](https://mistral.ai/pricing) | Mistral Large 4 | 14.99 | [$25.5 / month](https://mistral.ai/pricing) | billed at API list | documented allowance; that Vibe meters it at API list is not stated |

- **Claude and ChatGPT** publish only multiples of Pro or local-message
  estimates. The data file keeps, for the record and not in any figure,
  third-party budgets: API-equivalent spend per full week
  measured by wokey.ai (a pay-as-you-go reseller: a conflict of interest)
  on 79 Pro, 18 Ultra, 258 Team Standard, 30 Team Premium, 12 Plus and 7 Pro
  200 accounts,
  valued at a 92% cache-read mix, next to this workload's 91.5%. Max has no
  measurement: 5× Pro, documented, between two single reports ($420 and
  $1,250 a week). Claude plans are priced on Sonnet 5.5, which scores higher
  than Opus 5.5 on 4.0 for half the tokens; whether Anthropic's meter weighs
  models by their API price is not published. Max 5x is now "Max", Max 20x
  "Ultra" (no 5-hour limit for verified accounts). OpenAI moves Plus, Pro
  and Business to one token pool shared with ChatGPT in December; the Pro
  200 figure straddles the 2026-10-29 allowance doubling and may be up to
  2× low. Treat every Claude and ChatGPT budget as ±30%.
- **Z.ai** documents credits per 5 hours and per week and the per-token
  multipliers (GLM-5.3: 6.9 input, 1.7 cached, 24 output per 10,000
  tokens), so its meter is exact. Off-peak (outside 14:00 to 18:00 SGT) costs
  half, blended for a CET working day like DeepSeek's.
- **GitHub Copilot** meters AI credits at $0.01 each at model API list
  since 2026-06-01; part of each allowance is "flex" and can change.
  Business pools the $19 per seat across the organisation.
- **Gemini CLI** caps model requests per day (1,500 Pro, 2,000 Ultra); each
  request of the workload is one. The CLI picks the model; the row carries
  Gemini 4 Argon's score.
- **Mistral Pro** includes "$25.5 /mo in API credits" shared across Studio,
  the API and Vibe; that Vibe meters it at API list is not stated.

Left out, no defensible budget: Cursor (pools sized only as multiples of
Plus, third-party models at API list on top), Kimi Code (no quota number
published), Alibaba's Coding and Token Plans (the Coding Plan's models are
not in the comparison; the Token Plan's credit rates per model are not
published), Anthropic and OpenAI Enterprise (seat plus API usage; no
ceiling), Google Code Assist (priced by the hour) and ChatGPT Business (no
measurement of its own; the docs give it Plus's limits).

## 6. Re-reading

Prices move monthly (Gemini on 2026-12-01, OpenAI's pooled limits in
December). Re-read every row, the OpenRouter cross-check and the AA scores
together, rebuild the data file, and change `read` at the top of it. The
plan budgets are the weakest numbers here; the API rows are exact list
prices.
