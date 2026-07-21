# GPU Scaling Experiments

Measuring the **true KV-cache capacity** of Qwen 3.6 27B (FP8) on a single
H200, and projecting how prompt-caching, CPU offload, and `max_num_seqs`
trade off against per-user decode speed for agentic coding workloads.

The full write-up — setup, method, results, and recommendations — is in
**[docs/writeup.md](docs/writeup.md)**.

## Key findings

- vLLM 0.19.0 mis-reports the KV cache size (`352k tokens`) due to a
  [known bug](https://github.com/vllm-project/vllm/issues/37121). Direct
  measurement puts the true capacity **P in [1139k, 1399k] tokens** — enough
  to hold the full KV of **4–5 full-length (140k) sequences**.
- The reported `Maximum concurrency 5.1×` (≈1337k tokens) *does* land inside
  the measured interval, so that figure looks correct even though the token
  count printed next to it does not.
- Lowering `max_seq_len` to 180k, enabling FP8 KV cache, and CPU/RAM offload
  each expand how many sessions can be kept warm; sharing a stable prefix
  across agent turns is the biggest production win.

![Cache hit-rate sweep](figures/prefix_cache_sweep.png)

## Repository layout

```
.
├── README.md                 # this file
├── requirements.txt
├── docs/
│   └── writeup.md            # full experiment write-up
├── scripts/                  # figure-generating scripts (matplotlib, Agg)
│   ├── warm_capacity.py      # synthetic capacity/concurrency projections (no data needed)
│   ├── real_capacity.py      # real-data: clean → fit log-normal → MC warm-fill
│   ├── real_mns.py           # real-data: max_num_seqs speed/throughput tradeoff
│   └── warm_whisker.py       # real-data: warm capacity p5/p50/p95 whiskers
├── figures/                  # generated figures used in the write-up
└── data/                     # provider CSVs (not committed — see data/README.md)
```

## Running the scripts

```bash
pip install -r requirements.txt

# fully synthetic — runs out of the box, writes to figures/
python scripts/warm_capacity.py

# real-data scripts need the two CSVs described in data/README.md
python scripts/real_capacity.py
python scripts/real_mns.py
python scripts/warm_whisker.py
```

Each script writes its PNGs into `figures/`. Override the input and output
locations with environment variables:

```bash
DATA_DIR=/path/to/csvs OUT_DIR=/tmp/figs python scripts/real_mns.py
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
