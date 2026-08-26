# Working Set

Tools for scaling **local LLM deployments**: how many concurrent users or
agents a given GPU configuration can keep warm, where KV cache, decode
bandwidth, and prefill compute each become the binding constraint, and which
knob (topology, dtypes, `max_num_seqs`, prompt caching) buys the most headroom
for agentic coding workloads.

**Start with the [interactive explorer](https://workingset.tomvaucourt.com/)** —
live sliders for the workload, model (Qwen3.6-27B / 35B-A3B /
Mistral-Medium-3.5 / GLM-5.2 / DeepSeek-V4-Flash / Qwen3.8-Flash-Next /
GLM-5.3-Flash), GPU (H200 / B300), weight & KV dtypes, and DP × TP
topology. It answers as a decision tool: a binding-constraint verdict, a
deploy recipe (vLLM flags), the **electricity bill** (€/kWh slider,
duty-cycle power model), a **sensitivity panel** showing which assumption
would flip the decision, the **steady-state decode point** (how many
sessions are actually decoding at your load, and how fast each one runs —
Little's law, not the all-warm stress test), **shareable links** that
encode the whole configuration, and a **"Test these hypotheses" button**
that generates a standalone load-test script
([`scripts/validate_deployment.py`](scripts/validate_deployment.py)) preloaded
with the on-screen predictions — run it against a live vLLM endpoint to find
the real limits.

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
