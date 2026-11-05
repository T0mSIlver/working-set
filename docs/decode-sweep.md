# Decode sweep: step time against held batch size

`ws decode-sweep` holds n sequences decoding at once, all at the same context,
and measures the decode step at each n. The output fits decode bandwidth
efficiency (bytes moved per step over step time) and shows where a MoE's
expert union saturates. It needs only a vLLM endpoint and its `/metrics`.

It replaces nothing in `scripts/`: `decode_probe.py` and `decode_mbu.py` serve
a shared instance with a watchdog and an offline fit; the sweep is the
exclusive-access run whose JSON they would fit.

## Run it

```bash
export VLLM_API_KEY=...        # if the endpoint needs one
uvx --from workingset ws decode-sweep \
  --base-url http://HOST:8000/v1 --model MODEL \
  --api-key-env VLLM_API_KEY \
  --metrics-url http://HOST:8000/metrics \
  --batch 1,8,32,64,128,256,512 --context 100000 \
  --out decode-sweep.json
```

- `--metrics-header 'Name: value'` (repeatable) adds a header to the
  `/metrics` request. The API key goes to `/metrics` only when it is on the
  same origin as `--base-url`.
- Under data parallelism pass `--engine 0`: engines step on independent
  clocks, so summed forward-pass counts give a step time too short by the
  engine count. Run the sweep per engine, or against one engine's port.
- `--ca-bundle FILE` or `--insecure` for a TLS interception proxy.
- Sequences share the context and differ by a 128-token random suffix, with
  `--temperature 0.6`. They must differ: identical text routes to identical
  experts and hides the union. `--distinct` gives each sequence its own
  context (n x context of KV; rungs over the KV pool are skipped).
- `--prompt-file FILE` tiles real text to the context. Use it when the
  speculative-decoding acceptance matters: random words are not
  representative.
- `--hold` (30 s) is the measured window, `--settle` (10 s) the wait after
  every sequence has its first token.

## What it records

One entry per n in the JSON, `rungs[]`:

| field | meaning |
|---|---|
| `server.step_time_s` | window seconds over forward passes (`iteration_tokens` count): the measurement |
| `server.tok_s_total`, `tok_s_seq_server` | generated tokens per second, total and per running sequence |
| `server.itl_mean_s`, `itl_p50_s`, `itl_p99_s` | inter-token latency histogram delta (one observation per output event) |
| `server.spec_decode` | drafts, accepted tokens, `acceptance_rate`, `mean_accepted_len`, per-position acceptance; `null` without speculative decoding |
| `server.est_read_bytes_per_step` | the engine's own estimate, when it exports one |
| `client.step_time_s`, `tok_s_seq_mean` | the same window from the streams; `step_ratio` is client over server step time |
| `running`, `batch_ok`, `why_not` | gauges over the window and whether the batch really was n |

Fit only rungs with `batch_ok: true`. A rung is skipped when it would not fit
the KV pool, with the reason in `skipped`.

## Mixed steps: prefill in the same forward pass

The worst step of a full batch carries a prefill chunk next to the decodes.
Two flags add it; without them the run is as before.

```bash
ws decode-sweep ... --batch 63 --prefill-tokens 2048,8192 --prefill-depth 100000 \
  --chunk-tokens 8192 --min-tok-s 50
```

Set the server's `max_num_seqs` above the largest n (the cold request needs a
slot; else it queues and measures a wait, not a step).

- `--prefill-tokens P,...`: after the pure window of each rung, and with the
  same n streams still decoding, cold requests of P random tokens
  (`max_tokens=1`, no prefix hit) run one at a time for `--hold` seconds, the
  next sent when the previous one's first token returns.
- `--prefill-depth D,...`: one cold request of D tokens, which the server
  chunks at its `max_num_batched_tokens`. Pass that value as `--chunk-tokens`
  to place each step at its depth; otherwise the chunk is estimated as D over
  the mixed steps seen. The deepest chunk is the worst case.

How mixed steps are told apart (`sweep_mixed.py`): each held stream logs the
time and running token count of every event, and each cold request logs
[send, first token]. Pure steps also fall inside that interval (tokenization,
HTTP, scheduling), so the interval is not the measurement. A P-token prefill
spans k = ceil(P / (chunk - n)) steps (`--chunk-tokens` is the server's
`max_num_batched_tokens`; n tokens of each step's budget go to the decodes).
Per stream and request, the mixed steps are the k longest gaps that start after
the send and end by the first token (20 ms tolerance); the rest are pure. For
`--prefill-depth` every chunk is a mixed step. Without `--chunk-tokens`, every
gap in the interval counts and the mean is diluted: pass it.

The headline is the median over requests of the mixed step (and the p90), so it
does not depend on how fast the client sends. Tokens per second per sequence at
a fully mixed step is delivered tokens per step over that step, so speculative
decoding counts accepted tokens.

Self-check: before the rungs, each P is sent once with nothing else running
(best of two, `alone_s`). The mixed step should be about the pure step plus
`alone_s / k`; a row whose ratio is off by more than 30% is marked.

Per rung, `mixed_windows[]` holds, per P or D: `mixed` (`mixed_step_s`,
`mixed_step_p90_s`, `tokens_per_step`, `tok_s_seq`, `tok_s_p90_step`,
`pure_in_window_s`, gap counts), `alone_s`, `check` (`expected_s`, `ratio`,
`ok`), `server` (the engine's window, which blends both kinds of step),
`depth_profile` (`[{depth, step_s}]`), `requests`, `ok` and `why_not`.
`pure_step_s` is the rung's pure server step. The run ends with a table of n,
context, P or D, pure, mixed and p90 ms, tok/s per sequence, cold prefill alone
and the self-check ratio; rows below `--min-tok-s` (default 50) at the median or
p90 step, or failing the self-check, are marked.

Output sizing: each stream's `max_tokens` covers the pure window, every P window
and a depth allowance at the last clean rungs' speed (step time extrapolated
linearly in n). When that over-reserves the KV pool it shrinks to what the
windows need; a rung is skipped only when even that does not fit, and the
message names `--max-tok-s`, `--hold`, `--context`, `--batch` and `--kv-fill`.

`ws calibrate` ignores these fields. The step equation has no prefill term
(the mixed step is `itl_spike`'s decode step plus the prefill pass), so there
is nothing to fit yet; the rows are kept for fitting later.

## Time

A rung is the warm-up prefill, the ramp, `settle + hold`, and the wait for
idle: about 1 to 2 minutes once the context is cached. At 100k context the
first prefill is the only long step (tens of seconds on a large node), and
shared-context rungs reuse it. Seven rungs take about 10 to 15 minutes. With
`--distinct`, add n prefills of 100k per rung, which is why the larger n are
skipped.

Local check (Qwen3-4B on one RTX 3090, 3k context, 88k-token pool): step time
13.7, 19.3, 26.2, 36.2 ms at n = 1, 8, 16, 32; client and server agree to 1%.

## Fitting the model: `ws calibrate`

```bash
uv run ws decode-sweep --base-url http://host:8000/v1 --model M --tensor-parallel 2 \
    --prompt-file coding.txt --out sweep_tp2.json
uv run ws calibrate sweep_tp2.json [sweep_tp1.json sweep_k0.json ...] \
    --config workingset.toml [--vllm-log server.log] --write
uv run ws predict workingset.toml
```

`ws calibrate` fits what the runs identify and keeps every other constant at
its prior: η from the step's bytes (more batch sizes or two contexts), the
per-layer cost from the intercept, τ_ar only when one run is at TP1 and one
above, τ_rec only when one run changes the draft depth (`--speculative-config`
with fewer drafts, or none), acceptance from the spec-decode counters, the
per-GPU reserve from the KV pool vLLM reports, and prefill MFU from `ws replay`
records. Each fit prints with its standard error and the reason when it is not
identified. `--write` stores the values in `[calibration]` and their provenance
under `[calibration.fit.<key>]`; `ws predict` prices with them and prints where
each came from. The equation and its priors: `research/decode-model.md`.
