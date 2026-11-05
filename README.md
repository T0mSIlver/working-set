# Working Set

Tools for scaling **local LLM deployments**: how many concurrent users or
agents a given GPU configuration can keep warm, where KV cache, decode
bandwidth, and prefill compute each become the binding constraint, and which
knob (topology, dtypes, `max_num_seqs`, prompt caching) buys the most headroom
for agentic coding workloads.

**Start with the [interactive explorer](https://workingset.tomvaucourt.com/)** —
live sliders for the workload, model (Qwen3.8-27B / 35B-A3B /
Mistral-Medium-3.5 / GLM-5.3 / DeepSeek-V4-Flash /
DeepSeek-V4-Flash-Vision-Exp / DeepSeek-V4.1-Flash / Qwen3.8-Flash-Next /
GLM-5.3-Flash / Mistral-Large-4 / Kimi-K3.1-Pro), GPU (H200 / B300), weight & KV dtypes, and DP × TP
topology. It answers as a decision tool: a binding-constraint verdict, a
deploy recipe (vLLM flags), the **bill** (€/GPU-hour and €/kWh sliders:
hardware plus a duty-cycle power model), a **sensitivity panel** showing which assumption
would flip the decision, the **steady-state decode point** (how many
sessions are actually decoding at your load, and how fast each one runs —
Little's law, not the all-warm stress test), **shareable links** that
encode the whole configuration, and a **download button** in the Test step
that hands out the configuration on screen as a `workingset.toml` — feed it to
`ws test` below and measure the real limits on a live vLLM endpoint.

## The `workingset` package

The model behind the explorer is a Python package (`src/workingset/`, the
source of truth; the explorer's JS mirrors it), published on PyPI as
[`workingset`](https://pypi.org/project/workingset/). It ships one console
script, `ws`, so the package name travels in `--from`:

```bash
uvx --from workingset ws init --model Q38FN --gpu B300 --tp 8 --weight-dtype nvfp4   # writes workingset.toml
uvx --from workingset ws predict workingset.toml        # the four ceilings, which one binds, the operating point
uvx --from workingset ws test workingset.toml --dry-run # the plan, the sampler self-check, no requests
uvx --from workingset ws test workingset.toml --all --exclusive --out run.json   # measure it
```

or, as a dependency, `pip install workingset` / `uv add workingset`. From a
checkout the same commands run under `uv run`:

```bash
uv run ws init --model Q38FN --gpu B300 --tp 8 --weight-dtype nvfp4   # writes workingset.toml
uv run ws predict workingset.toml        # the four ceilings, which one binds, the operating point
uv run ws predict workingset.toml --json # the same as a run record
uv run ws link workingset.toml           # the explorer URL showing this config
uv run ws hypotheses                     # the H-* and what each one needs
uv run ws test workingset.toml --dry-run # the plan, the sampler self-check, no requests
uv run ws test workingset.toml --exclusive --out run.json   # measure it
uv run ws calibrate sweep.json --config workingset.toml --write   # fit the decode constants to a ws decode-sweep run
uv run ws report run.json                # re-print the verdicts
uv run ws models                         # model / GPU keys
uv run ws api workingset.toml --hours 84 # one seat on each API; what each coding plan holds
uv run pytest                            # self-checks + config round-trips
```

The explorer's `workingset.toml` needs no checkout at all: the three `uvx`
lines above are exactly what its Test step prints. To run
the model at a commit PyPI does not have yet, point `--from` at git:
`uvx --from git+https://github.com/T0mSIlver/working-set ws …`.

Predictions live in no file: `ws predict` recomputes them from the config every
time, so a config can never carry a number the code did not produce. A harness
`.py` downloaded from the explorer before the package existed still loads (its
CONFIG block is extracted).

`ws link` goes the other way: it prints the explorer share URL that opens the
page on a config (`--base http://127.0.0.1:PORT/` for a local copy of
`interactive/`). The page has no control for `subagent_prefix_tokens` or
`[endpoint]`, and
it clamps each slider to its range and resets combinations it does not price,
so when a config sets one of those, or a value the page would move, `ws link`
names the field on stderr with what the page shows instead.

`ws test` puts the predictions to a live endpoint, one falsifiable hypothesis
at a time. Without `--exclusive` it runs only the hypotheses that need a
handful of requests (miss TTFT, the inter-token gap distribution, the steady
decode point) and lists the rest as skipped — a hypothesis that has to
generate its own population is never measured against someone else's load.
With `--exclusive` it drives the geometric load ladder once, and every ceiling
reads from it. `--burst N` adds the correlated-flush probe (B*).
`--tokenizer Qwen/Qwen3.8-27B` sizes the synthetic prompts with the model's
real tokenizer (via [toklen](https://pypi.org/project/toklen/); add
`--with toklen` to the `uvx` line) instead of the chars-per-token guess.
For a vision model, set `image_share` in `[workload]` (with
`images_per_request`, `image_width` and `image_height`) and pass `--api chat`:
that share of warm turns then attaches generated PNGs, which stay in the
session history, and the report splits warm-turn TTFT by whether the turn
sent a new image. The model does not price images yet.

## How many agent sessions an endpoint serves

`ws replay` answers that from the endpoint and recorded sessions alone, with
no config and nothing from the model:

```bash
uv run ws trace pi -o trace.json          # ~/.pi/agent/sessions -> sizes and waits, no text
uv run ws replay trace.json --base-url http://gpu:8000/v1 --out run.json
uv run ws report run.json                 # re-print it
```

1. **Alone.** One replayed session sends 40 requests back to back, one in
   flight at a time. The endpoint's own speed is fitted from them: time to
   first token from the uncached tokens and the context, then time per output
   token from the context.
2. **Load.** 1, 2, 4, ... replayed sessions run at once with their recorded
   waits, a 60 s ramp and a 300 s measured window each, until one step
   fails. Then the bracket between the last pass and the first fail is
   halved until it is within 10%.
3. **Verdict.** A step of N sessions passes when all of these hold over its
   window. Each threshold is a flag.
   - Mid-session evictions are at most 1% of warm requests
     (`--max-eviction-share`). A request is warm when its session's previous
     request exists, no compaction lies between, and it adds at most 8192
     tokens (`--warm-new-tokens`). The replay knows what it sent, so the
     tokens a request shares with its session's previous prompt (rounded down
     to a `--kv-block-tokens` block, 16) are what the server should have
     cached. It is an eviction when `usage.prompt_tokens_details.cached_tokens`
     is more than a block short. vLLM reports that with
     `--enable-prompt-tokens-details`; a server that does not leaves the check
     "unknown", with a warning, and it does not gate.
   - The p90 time to first token of warm requests is at most 3 s
     (`--max-warm-ttft-s`).
   - The p10 decode speed per request, (output tokens - 1) / (last token
     time - first token time) over requests of 32 output tokens or more, is
     at least 50 tok/s (`--min-decode-tok-s`).
   - At most 1% of requests failed or were left unfinished
     (`--max-failed-share`).

   A check judges only with `--min-samples` (10) requests: warm ones for
   evictions and time to first token, decode-eligible ones for decode speed.
   A step measures on, up to `--max-measure-s`, to reach them. If it still
   falls short and nothing failed, was unfinished, or queued on the server, the
   sessions sat idle (long think gaps): the step shows NO EVIDENCE, which is
   neither a pass nor a fail, and the search goes on to the next N. A short step
   with a queue or preempted requests is a fail. `--start-sessions N` begins
   the search at N.

   The turn slowdown (time its requests took over what the fitted speed gives
   alone) is reported, and gates only under an explicit `--max-slowdown`. A
   cold request (a session's first, or after a compaction) reports its time
   to first token less the idle-speed estimate, which is queueing; it is not
   gated.

The answer reads `serves 15 sessions within <the rule>; fails at 16`, naming
the checks the failing step missed, with the numbers at each step and, with
`--metrics-url`, what the server showed (queue, preemptions, KV use, running
requests, prefix hits).
`--config workingset.toml` prints the model's prediction for the trace's
averages next to it, and never feeds the verdict. Two deployments replayed
with the same trace and `--seed` compare directly.

Other flags shape the load:

- `--base-url` repeated benches a data-parallel deployment (give one
  `--metrics-url` per base URL, in the same order; without any, each
  defaults to `<origin of the base URL>/metrics`). `--routing sticky`, the
  default, pins a session tree to one URL by a stable hash of its seat id, as
  a session-affine router does; `--routing round-robin` sends each request to
  the next URL. The report counts requests and shows server metrics per URL.
- `--output-scale F` multiplies every recorded reply length and the context
  those replies add to the session's later prompts, to replay a model that
  writes more or less per task.
- `--chat-rate R` adds background chat requests (R per second, Poisson, each
  with its own prompt; mean lengths `--chat-prompt-tokens` and
  `--chat-output-tokens`, lognormal with `--chat-sigma`). They are reported
  apart (count, p90 time to first token, p10 decode speed) and never judged.

The synthetic sessions of `ws test` send a request every `think_time_s` on
average, spread evenly. A real coding agent works in bursts instead: a
request every second or so while it runs tools, then a long pause while the
person reads. `ws test --replay` puts the model's predictions to that load,
hypothesis by hypothesis:

```bash
uv run ws test workingset.toml --replay trace.json --exclusive --out run.json
```

`ws trace pi` keeps, for each request, the prompt and reply token counts, the
wait before it and what it waited on, and nothing of the text, so other people
can export their sessions and hand the file over. On a machine without
workingset, the module runs alone:

```bash
curl -fsSLO https://raw.githubusercontent.com/T0mSIlver/working-set/main/src/workingset/agent_trace.py
python3 agent_trace.py pi -o trace.json
```

`ws trace claude` does the same for Claude Code's session files
(`~/.claude/projects`): responses written as several lines are merged into one
request, background-task and subagent waits get the kind `background`, and each
subagent is its own session, linked to the one that launched it. Sessions run
with `claude -p` or the SDK are left out unless you pass `--include-headless`.
Claude Code keeps no request start time or permission log, so a request's
service time includes the client's overhead and a permission wait counts as
tool time (see the module's docstring).

Each wait gets a kind read from the session's events: a tool, a permission
prompt (timed from pi-permission-system's log when it is there), the agent's
question to the person, think time after an answer, a retry after an
interruption, a move in the session tree, a compaction, a failed request, a
restart of pi, or a session left open (a person's wait past 30 minutes with
no restart). `ws trace` prints the count of each.

A replay (either command) plays one recorded session per simulated session, from a random
request on, and a session that ends moves on to another recording, so a
ladder rung of any size draws from a small corpus. Prompts are synthetic text
of the recorded sizes that grows on top of the server's own replies, so the
prefix cache sees what it saw then without a byte of the real prompts.
Replies have the recorded length (`max_tokens` plus `ignore_eos`). Waits
are the recorded ones, except where the session sat unused (a failed request,
a restart, a session left open): `--replay-gap-policy` replaces those with a
think gap drawn from the trace (`sample`, the default), caps them at
`--replay-max-gap-s` (`cap`), or drops them (`drop`); `ws replay` always
samples. Under `ws test` the config's
`[workload]` is replaced by the trace's averages, so the predictions price the
same load arriving smoothly and the gap to the measurement is what the bursts
cost. `--dry-run` shows what each rung would offer at the recorded server's
speed. [research/replay_gaps.md](research/replay_gaps.md) has the gap counts
of the study's own sessions.

## Predicting AgentX's prefix-cache hit rates

SemiAnalysis published the Claude Code sessions behind AgentX
([`semianalysisai/cc-traces-weka-062126`](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126),
Apache-2.0) and the hit rates DeepSeek V4 Pro reached on them. `ws agentx`
replays those sessions against a model of the server's caches, per DP rank:
an HBM tier evicted the way vLLM does, a write-through DRAM tier, and
sliding-window retention points.

```bash
uv run ws agentx compact traces.jsonl agentx.json.gz   # 1.8 GB -> 2 MB, prefix structure only
uv run ws agentx predict agentx.json.gz --sweep       # both published points, with the sensitivity runs
```

On B300 at concurrency 384 it predicts 91.4% HBM hits (90.5-92.2% over ten
seeds) against a measured 91%, and 1.4% DRAM hits against 1.36%. On B200
at 196 it predicts 41% (28-48%) against 73%: a miss of about 32 points, at
a point on the knee of the curve, where 73% needs ~34M tokens of KV and the
bytes, priced from one vLLM anchor, give 24M. Method, sources and the misses:
[research/agentx_kv_hits.md](research/agentx_kv_hits.md).

## Contents

- [docs/writeup.md](docs/writeup.md) — baseline study: KV-cache capacity and
  the prompt-caching / offload / `max_num_seqs` trade-offs.
- [docs/scenarios.md](docs/scenarios.md) — extended scenario model: multi-GPU
  topologies, MoE vs dense, subagent workloads, the cost of a cache miss, and
  cold-spike tolerance.
- [docs/decode-sweep.md](docs/decode-sweep.md) — `ws decode-sweep`: decode step
  time at n held sequences, from a vLLM endpoint and its `/metrics`.
- [scripts/](scripts/) — everything is reproducible:

  ```bash
  uv run ws selfcheck                # the shared model's self-checks (src/workingset/model.py)
  uv run scripts/scenarios.py        # renders the scenario figures
  uv run scripts/tables.py           # regenerates every number in docs/scenarios.md
  ```

- [research/](research/) — sourced constants for each model and GPU;
  [research/api_prices.md](research/api_prices.md) for the API and coding-plan
  prices `ws api` reads.
- [interactive/](interactive/) — the explorer, a dependency-free page
  mirroring the Python model: `index.html` holds the markup and styles,
  `src/*.js` the model and the charts as ES modules (`src/main.js` is the
  entry and lists the layering). Browsers refuse module scripts from
  `file://`, so serve the folder to open it locally:

  ```sh
  python3 -m http.server 8000 --directory interactive   # then http://localhost:8000
  ```

Method, calibration, and caveats are laid out in the docs above.

MIT licensed; see [LICENSE](LICENSE).
