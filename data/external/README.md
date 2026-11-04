# data/external/

Numbers Working Set did not produce: benchmarks, deployments, calculators.
One JSON file per source; `src/workingset/external.py` loads and validates
them (`uv run pytest tests/test_external.py`). How each source is read and
what it may be used for: [research/sources/](../../research/sources/).

| file | source | records | use |
|---|---|---|---|
| `inferencex.json` | SemiAnalysis InferenceX (API) | curve points per model × GPU × engine, derived MBU | calibrate, check |
| `mlperf-inference-v6.0.json` | MLCommons MLPerf Inference v6.0 | DeepSeek-R1 and gpt-oss-120b on H200/B200/B300/GB200/GB300 | context |
| `deployments.json` | press and operator posts | Revolut, DeepSeek, Kimi K2 on Mooncake, DaoCloud GLM-5.2 | context |
| `calculators.json` | self-host vs API calculators | each tool's default scenario | context |

Artificial Analysis AgentPerf has no file here: its terms forbid
redistributing its data ([research/sources/aa-agentperf.md](../../research/sources/aa-agentperf.md)).

## Format

```json
{
  "source": {
    "id": "inferencex", "name": "...", "publisher": "...", "url": "...",
    "kind": "measured", "refresh_days": 14,
    "redistribute": "cite", "terms": "the licence clause, quoted",
    "license": "...", "method": "what is run, by whom, how metrics are defined"
  },
  "records": [{
    "id": "inferencex/<slug>",
    "url": "page, file or API call the numbers come from",
    "read": "2026-11-04", "published": "2026-10-29",
    "kind": "measured", "use": "calibrate",
    "config": {"model": "GLM-5.3", "ws_model": "GLM52", "gpu": "B300", "n_gpus": 8,
               "weights": "fp4", "kv": null, "engine": "sglang <tag>", "parallel": "TP8",
               "spec_decode": "mtp", "max_ctx": null, "extra": {}},
    "workload": {"name": "AgentX trace replay", "isl": null, "osl": null, "prefix_cache": true, "extra": {}},
    "metrics": [{"name": "tok_s_per_user", "value": 344.8, "unit": "tok/s", "at": "conc 1"}],
    "inputs": [], "mapping": {},
    "quote": "the cell, JSON path or sentence the numbers are read from",
    "note": ""
  }]
}
```

- `kind`, per record and overridable per metric: `measured` (a third party
  ran it under a stated method), `vendor` (self-reported by a party with a
  stake: a seller, a deployer), `derived` (computed by us; `inputs` names the
  records or `note` gives the formula), `guess` (an assumption).
- `use`, per record and overridable per metric: `calibrate` (full config and
  workload; may set a constant by a fit), `check` (the model predicts the same
  quantity; the ratio is reported and no constant moves), `context` (shown
  with its provenance only). Default `context`.
- `redistribute`: `yes`, `cite` (public with attribution under `terms`), `no`
  (never in this directory; the loader refuses it).
- `refresh_days`: past it the record is stale (`external.stale`) and the page
  marks it.
- `ws_model`: the key in `workingset.model.MODELS` when the model is ours.
- Metric names are snake_case with the unit last (`tok_s_per_user`, `ttft_p50_s`,
  `tpot_ms`, `mbu`, `usd_per_gpu_h`). `ttft_s` is a TTFT whose percentile the
  source does not state. `op` is `=`, `<`, `<=`, `>`, `>=` or `~`; `low`/`high`
  give a band, `band` says whether the source reported it or we assumed it.
- `published` may be `YYYY-MM` when the source gives only a month.
