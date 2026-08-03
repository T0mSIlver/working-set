# GPU Scaling Experiments

Measuring the **true KV-cache capacity** of Qwen 3.6 27B (FP8) on a single
H200, and projecting how prompt-caching, CPU offload, and `max_num_seqs`
trade off against per-user decode speed for agentic coding workloads.

The full write-up — setup, method, results, and recommendations — is in
**[docs/writeup.md](docs/writeup.md)**.

> **Extended study:** [docs/scenarios.md](docs/scenarios.md) carries the same
> methodology to **2×H200** (tensor- vs data-parallel), the **35B-A3B MoE** model,
> **subagent** workloads, **system-prompt size**, and a **cache-invalidation** rate —
> with an interactive explorer, hosted at
> [workingset.tomvaucourt.com](https://workingset.tomvaucourt.com/) (source:
> [`interactive/index.html`](interactive/index.html)).
> The 2026-07 extension adds the **B300** (Blackwell Ultra) as a selectable GPU,
> **NVFP4 weight quantization** (B300-only, weights-never-KV, available for all
> models), and two more models: **Mistral-Medium-3.5-128B** (dense GQA,
> 176 KiB/token KV) and **GLM-5.2** (744B-A40B, MLA + DeepSeek Sparse Attention,
> sparse decode pricing). See docs/scenarios.md § Extension. The 2026-08
> extension prices the prefill roofline (§ 8) and then its **queue and its
> bursts** (§ 9) — what a cache miss costs, and how big a cold spike each
> deployment can absorb — and adds **DeepSeek-V4-Flash-0731** (284B-A13B,
> compressed sparse attention: 3.4 KiB/token — the study's KV-lightest big
> model).

## Key findings

- vLLM 0.19.0 mis-reports the KV cache size (`352k tokens`) due to a
  [known bug](https://github.com/vllm-project/vllm/issues/37121). Direct
  measurement puts the true capacity **P in [1139k, 1399k] tokens** — enough
  to hold the full KV of **4–5 max-length (262k) sequences**.
- The reported `Maximum concurrency 5.1×` (≈1337k tokens) *does* land inside
  the measured interval, so that figure looks correct even though the token
  count printed next to it does not.
- Lowering `max_seq_len` to 180k, enabling FP8 KV cache, and CPU/RAM offload
  each expand how many sessions can be kept warm; sharing a stable prefix
  across agent turns is the biggest production win.
- A 2×H200 tensor-parallel bring-up of the 27B (FP8 weights, FP16 KV) reported
  **3,233,564** KV-cache tokens at startup — within 0.3% of the extended
  model's prediction, its first non-circular validation (see
  [docs/scenarios.md](docs/scenarios.md), § Measured cross-check). It also
  **retires** the study's low-calibration-anchor hedge, which predicted that
  same pool 15% low: the stacked-adverse TP2 planning figure rises from 403 to
  **483** warm sessions with no change to the central case.
- **A cache miss costs 18–19× the machine time of a hit — near-identical
  across all four architectures** — and one 32k prefill chunk spikes the
  inter-token latency of *every* concurrently decoding user by **31–122×** —
  the study's founding hypothesis, finally priced (docs/scenarios.md § 8).
  Prefill is compute-bound where decode is memory-bound, so it imposes a
  cold-request ceiling no amount of KV pool can raise: Mistral-Medium-3.5 on
  TP4 saturates on prefill alone at a **3%** miss rate — while its 56 warm
  sessions also fall short of the 64-user reference load, a doubly-constrained
  deployment. Non-obvious corollary: the **35B-A3B MoE prefills ~7× faster
  than the smaller dense 27B**, because only its ~2.4B active GEMM parameters
  prefill.
- **That ceiling is not a place to sit** (docs/scenarios.md § 9). Adding the
  prefill queue turns it into a planning number: a miss's service time is
  quadratic in a log-normal context length (cv² of 5.5–8.3 where an
  exponential is 1), so mean TTFT breaches a 10 s budget while the duty cycle
  still reads a comfortable **76–93%**, and the burst a deployment absorbs hits
  zero exactly *at* f\*. The headline metric is **cold-spike tolerance B\***, the
  simultaneous cache misses a config survives inside that budget: **5.1** on
  the 27B/TP2, **36.4** on the 35B-A3B/TP2, and **0.5 — less than one request —**
  on Mistral-Medium-3.5/TP4. The MoE's prefill edge **compounds** here (7.2–8.8×
  against a 5.9× speed gap, wider on the tighter machine), and it is the only
  configuration modelled that can serve its own recovery after a **global cache
  flush** — every dense one has to shed load. Sharpest corollary: under FCFS the
  miss tax is paid by the users who *hit* the cache, whose TTFT reaches **74×**
  their own service time at a 20% miss rate.
- **All four constraints, in one unit.** The study reported capacity in sessions
  and prefill in work rates and declined to combine them. Converting each into
  **max concurrent users** — at the price of two stated assumptions (one user
  holds one session; a user turns every 30 s) — makes the binding constraint
  simply the smallest, and it *reproduces* the published decision table's warm-user
  and mns@40 columns rather than restating them. Two ceilings barely move with the
  miss rate and two collapse, so **which one binds changes hands at f ≈ 6%** on the
  27B/TP2. It also renames the tightest constraint on one configuration:
  Mistral-Medium-3.5/TP4 is **decode**-bound at 36 users — below the reference
  load — because it ships no MTP module, making the study's "without MTP the
  ordering flips" case its central one.

![Cache hit-rate sweep](figures/prefix_cache_sweep.png)

## Repository layout

```
.
├── README.md                 # this file
├── pyproject.toml            # project + dependencies (managed with uv)
├── uv.lock                   # pinned, reproducible environment
├── docs/
│   ├── writeup.md            # baseline experiment write-up
│   └── scenarios.md          # extended-scenario study (2xH200, MoE, subagents, …)
├── scripts/                  # figure-generating scripts (matplotlib, Agg)
│   ├── warm_capacity.py      # baseline: synthetic capacity/concurrency projections
│   ├── real_capacity.py      # baseline: clean → fit log-normal → MC warm-fill
│   ├── real_mns.py           # baseline: max_num_seqs speed/throughput tradeoff
│   ├── warm_whisker.py       # baseline: warm capacity p5/p50/p95 whiskers
│   ├── scenario_model.py     # extended study: shared capacity + decode model (+ self-checks)
│   ├── scenarios.py          # extended study: renders the scenario figures
│   └── tables.py             # extended study: regenerates every number in scenarios.md
├── interactive/
│   └── index.html            # self-contained interactive scenario explorer
├── research/
│   ├── model_35ba3b.md       # cited architecture parameterization for 35B-A3B
│   ├── model_mistral_medium35.md  # Mistral-Medium-3.5-128B constants + sources
│   ├── model_glm52.md        # GLM-5.2 (MLA+DSA) constants + sources
│   ├── model_dsv4flash.md    # DeepSeek-V4-Flash-0731 (CSA/HCA) constants + sources
│   ├── gpu_b300.md           # B300 (Blackwell Ultra) hardware constants
│   ├── nvfp4.md              # NVFP4 format, B300-only gate, Qwen NVFP4 bytes
│   ├── prefill.md            # prefill (compute-roofline) constants + confidence tiers
│   └── spike.md              # cold-spike model: M/G/1 queueing + burst drain
├── figures/                  # generated figures used in the write-ups
└── data/                     # provider CSVs (not committed — see data/README.md)
```

## Running the scripts

The project is managed with [uv](https://docs.astral.sh/uv/) — dependencies live
in `pyproject.toml` and are pinned by `uv.lock`. `uv run` creates/updates the
environment automatically on first use (or run `uv sync` explicitly):

```bash
# fully synthetic — runs out of the box, writes to figures/
uv run scripts/warm_capacity.py

# real-data scripts need the two CSVs described in data/README.md
uv run scripts/real_capacity.py
uv run scripts/real_mns.py
uv run scripts/warm_whisker.py
```

Each script writes its PNGs into `figures/`. Override the input and output
locations with environment variables:

```bash
DATA_DIR=/path/to/csvs OUT_DIR=/tmp/figs uv run scripts/real_mns.py
```

### Script → figure map

| Script              | Figures                                                             | Needs CSVs |
| ------------------- | ------------------------------------------------------------------ | ---------- |
| `warm_capacity.py`  | `warm_by_scenario.png`, `peruser_vs_mns.png`, `throughput_tradeoff.png` | no     |
| `real_capacity.py`  | `real_dist_fit.png`, `real_warm_mc.png`                             | yes        |
| `real_mns.py`       | `real_mns_tradeoff.png`                                             | yes        |
| `warm_whisker.py`   | `warm_whisker.png`                                                  | yes        |

The experimental result `figures/prefix_cache_sweep.png` comes from the live
prompt-caching sweep (not one of these scripts).

### Extended-scenario study

```bash
uv run scripts/scenario_model.py   # self-checks (calibration + published-config identities)
uv run scripts/scenarios.py        # renders scenario_capacity / sysprompt / mns / subagent_invalidation / prefill_thrash / cold_spike / binding_map .png
uv run scripts/tables.py           # regenerates every number quoted in docs/scenarios.md
```

`scripts/scenario_model.py` is the shared model (calibrated to the baseline's
2.77M-token FP8 anchor — projected from the measured FP16 pool, see
docs/scenarios.md limitations; 35B-A3B constants from the published
Qwen3.6-35B-A3B config — see `research/model_35ba3b.md`; an FP8/FP16 KV-cache
switch is available via `with_kv_dtype`, FP8 being the studied default, and an
FP8/NVFP4 *weight* switch via `with_weight_dtype` — NVFP4 being gated to the
B300's native FP4 tensor cores and never applied to the KV cache; topologies are
a `DP × TP` grid via `topology_grid(dp, tp, gpu)`, so a replica is a *group* of
GPUs — the only way a model that does not fit a single GPU can be data-parallel
at all (Mistral-Medium-3.5 needs TP2 on H200 though it fits one B300; GLM-5.2
needs TP7 on H200, TP3 on B300; DeepSeek-V4-Flash needs TP2 on H200, fits one
B300), with `min_tp_for` / `node_splits` giving the fitting splits of an
8-GPU node);
`scripts/scenarios.py` renders the static figures; and `interactive/index.html`
is a dependency-free page mirroring the same math with live sliders for the
workload, model (Qwen3.6-27B / 35B-A3B / Mistral-Medium-3.5 / GLM-5.2 /
DeepSeek-V4-Flash-0731), GPU
(H200 / B300), weight & KV dtypes, and topology — the **Split (DP × TP)** control
offers the legal splits of the chosen GPU count (its divisors), so hybrid grids
like GLM-5.2 `DP2×TP4` are reachable, with non-fitting splits struck through and
their TP threshold reported. The study's two remaining un-measured structural
assumptions are separate named controls — **Recurrent state dtype** (bf16/fp32)
and **Deployed-weight overhead** (as-published/+15%) — each disabled on the
models where it would be a no-op; the low-calibration-anchor case that used to
sit alongside them was retired once the 2×H200 log refuted it. See
[docs/scenarios.md](docs/scenarios.md).
