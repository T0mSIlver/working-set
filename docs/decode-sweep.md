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

## Time

A rung is the warm-up prefill, the ramp, `settle + hold`, and the wait for
idle: about 1 to 2 minutes once the context is cached. At 100k context the
first prefill is the only long step (tens of seconds on a large node), and
shared-context rungs reuse it. Seven rungs take about 10 to 15 minutes. With
`--distinct`, add n prefills of 100k per rung, which is why the larger n are
skipped.

Local check (Qwen3-4B on one RTX 3090, 3k context, 88k-token pool): step time
13.7, 19.3, 26.2, 36.2 ms at n = 1, 8, 16, 32; client and server agree to 1%.
