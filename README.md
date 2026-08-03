# Working Set

Tools for scaling **local LLM deployments**: how many concurrent users or
agents a given GPU configuration can keep warm, where KV cache, decode
bandwidth, and prefill compute each become the binding constraint, and which
knob (topology, dtypes, `max_num_seqs`, prompt caching) buys the most headroom
for agentic coding workloads.

**Start with the [interactive explorer](https://workingset.tomvaucourt.com/)** —
live sliders for the workload, model (Qwen3.6-27B / 35B-A3B / Mistral-Medium-3.5 /
GLM-5.2 / DeepSeek-V4-Flash), GPU (H200 / B300), weight & KV dtypes, and
DP × TP topology.

## Contents

- [docs/writeup.md](docs/writeup.md) — baseline study: KV-cache capacity and
  the prompt-caching / offload / `max_num_seqs` trade-offs.
- [docs/scenarios.md](docs/scenarios.md) — extended scenario model: multi-GPU
  topologies, MoE vs dense, subagent workloads, the cost of a cache miss, and
  cold-spike tolerance.
- [scripts/](scripts/) — everything is reproducible:

  ```bash
  uv run scripts/scenario_model.py   # shared model + self-checks
  uv run scripts/scenarios.py        # renders the scenario figures
  uv run scripts/tables.py           # regenerates every number in docs/scenarios.md
  ```

- [research/](research/) — sourced constants for each model and GPU.
- [interactive/index.html](interactive/index.html) — the explorer, a single
  dependency-free page mirroring the Python model.

Method, calibration, and caveats are laid out in the docs above.

---

By [Tom Vaucourt](https://www.tomvaucourt.com).
