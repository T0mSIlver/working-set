"""Write docs/accuracy.md: what the model predicts for every measured figure
the public repo may show, and the error.

Every predicted value is computed here from workingset.model; the measured
values carry their source and date. Outside numbers come through
workingset.external and keep their `use`: a `calibrate` record goes in the
fitted table, never in the error statistics; MLPerf numbers are never
compared (MLCommons rules); a `redistribute: no` source never loads.

    uv run scripts/accuracy.py           # rewrite docs/accuracy.md
    uv run scripts/accuracy.py --check   # exit 1 if it is stale (tests)
"""
from __future__ import annotations

import hashlib
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from importlib.metadata import version
from pathlib import Path

from workingset import external as X
from workingset import model as M
from workingset.config import RunConfig

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "accuracy.md"
CATALOG = ROOT / "src" / "workingset" / "data" / "catalog.toml"
GIB = 1024 ** 3

# InferenceX's AgentX replay: median input per request at conc 1
# (research/sources/inferencex.md, "What is measured").
AGENTX_CONTEXT = 310_000


@dataclass
class Row:
    quantity: str
    conditions: str
    measured: float
    predicted: float
    unit: str
    source: str
    date: str
    note: str = ""

    @property
    def error(self) -> float:
        return (self.predicted - self.measured) / self.measured


# ---------------------------------------------------------------- helpers

def arm(key: str, gpu: str, weights: str = "fp8") -> M.Model:
    """The model row as priced on `gpu`: NVFP4 weights when asked (B300
    only), the BF16 KV arm where fp8 KV is Blackwell-only, the GPU's own
    KV row."""
    m = M.MODELS[key]
    if weights == "nvfp4":
        m = M.with_weight_dtype(m, "nvfp4")
    if m.kv_fp8_blackwell_only and gpu == "H200":
        m = M.with_kv_dtype(m, "fp16")
    return M.with_gpu(m, gpu)


def fixed_context(ctx: float) -> M.Workload:
    """A workload whose every request is exactly `ctx` tokens."""
    return M.Workload(user_median=ctx, user_sigma=1e-9, sub_ratio=0.0,
                      sys_user=min(ctx, 15_000), sys_sub=0.0, cap=ctx,
                      min_tokens=0.0, invalidation=0.0)


def tok_s_per_user(m: M.Model, topo: M.Topology, n: int, ctx: float) -> float:
    _, p50, _, _ = M.decode_curves(m, topo, fixed_context(ctx), [n], n_iter=4)
    return float(p50[0])


def pct(x: float) -> str:
    return f"{x * 100:+.2f}%" if abs(x) < 0.01 else f"{x * 100:+.1f}%"


def num(x: float, unit: str) -> str:
    if unit == "tokens":
        return f"{x / 1e6:.3f}M" if x >= 1e6 else f"{x:,.0f}"
    if unit == "bytes":
        return f"{x / 1e9:.4f} GB" if x < 1e10 else f"{x / 1e9:.3f} GB"
    if unit == "ms" or unit == "s":
        return f"{x:.2f} {unit}"
    if unit == "tok/s":
        return f"{x:.0f}"
    if unit == "tokens/step":
        return f"{x:.2f}"
    return f"{x:g}"


def table(head: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(head) + " |",
           "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


# ---------------------------------------------------- independent checks

def check_rows() -> list[Row]:
    rows = []

    # KV pool: the TP2 FP16 vLLM startup log (docs/scenarios.md). Nothing
    # about TP2 or FP16 was fitted; the reserve came from the 1xH200 anchor.
    m27_16 = M.with_kv_dtype(M.MODELS["27B"], "fp16")
    rows.append(Row(
        "KV pool", "Qwen3.6-27B, 2×H200 TP2, FP8 weights, FP16 KV",
        M.MEASURED_POOL_TOKENS_27B_TP2_FP16,
        M.kv_pool_tokens(m27_16, M.TOPOLOGIES["2xH200-TP2"]), "tokens",
        "vLLM startup log, `GPU KV cache size` (docs/scenarios.md, Measured cross-check)",
        "2026-07-22"))

    # Weight bytes: the catalogue's config-derived NVFP4 ledger for the
    # 35B-A3B against the per-shard safetensors bytes (research/nvfp4.md 6.1)
    m35 = M.with_weight_dtype(M.MODELS["35BA3B"], "nvfp4")
    src = "RedHatAI/Qwen3.6-35B-A3B-NVFP4 shard headers (research/nvfp4.md § 6.1)"
    for what, measured, predicted in (
            ("always-read bytes per decode step", 3.3065e9, m35.w_decode_shared),
            ("one token's routed experts", 566_238_720, m35.w_route_pertok),
            ("all routed experts", 18_119_639_040, m35.w_route_total)):
        rows.append(Row("Weight bytes", f"Qwen3.6-35B-A3B NVFP4, {what}",
                        measured, predicted, "bytes", src, "2026-07-27"))

    # Decode on Blackwell Ultra: the Mistral-Large-4 vLLM recipe on 8×GB300
    # (same HBM size and bandwidth as B300), DSpark on, 8k input
    # (research/model_mistral_large4.md § 8)
    src = ("vLLM recipe `Mistral-Large-4-1T-A52B.yaml`, 8×GB300 priced as 8×B300 "
           "(research/model_mistral_large4.md § 8)")
    tp8 = M.topology_grid(1, 8, "B300", "replicate")
    note = "acceptance 0.777 is taken from this recipe (32k input)"
    for weights, label, n, measured in (("nvfp4", "NVFP4", 1, 319.0),
                                        ("fp8", "FP8", 1, 281.0),
                                        ("nvfp4", "NVFP4", 8, 110.0)):
        rows.append(Row(
            "Decode speed per user", f"Mistral-Large-4 {label}, 8×B300 TP8, "
            f"{n} user{'s' if n > 1 else ''}, 8k context"
            + ("" if n > 1 else " (stated for the 8-user row only)"),
            measured, tok_s_per_user(arm("ML4", "B300", weights), tp8, n, 8_000),
            "tok/s", src, "2026-10-30", note))

    # Prefill on B300: one 340k-token prompt on Kimi-K3.1-Pro, 8×B300 TP8
    # (research/model_kimi_k31pro.md § 8)
    k31 = M.MODELS["K31P"]
    rows.append(Row(
        "Prefill, idle cold TTFT", "Kimi-K3.1-Pro, 8×B300 TP8, one 340k-token prompt, "
        f"chunk {M.CHUNK_DEFAULT:,}",
        15.0, M.idle_cold_ttft(k31, M.topology_grid(1, 8, "B300"), 340_000, M.CHUNK_DEFAULT),
        "s", "vLLM recipe `Kimi-K3.1-Pro.yaml`, pre-release checkpoint, \"~15 s\" "
        "(research/model_kimi_k31pro.md § 8)", "2026-11-05",
        "the recipe's chunk size is not stated"))
    return rows


# ------------------------------------------------------- fitted anchors

def fitted_rows(recs: list[X.Record]) -> list[Row]:
    rows = []
    m27 = M.MODELS["27B"]
    t1 = M.TOPOLOGIES["1xH200"]
    rows.append(Row(
        "KV pool", "Qwen3.6-27B, 1×H200, FP8 KV (projected from the FP16 measurement)",
        M.BASELINE_POOL_TOKENS_27B_1GPU, M.kv_pool_tokens(m27, t1), "tokens",
        "baseline study (docs/writeup.md); the reserve is solved from it", "2026-07"))
    rows.append(Row(
        "KV pool", "Qwen3.6-27B, 1×H200, FP16 KV, best estimate in [1,139k, 1,399k]",
        1_337_000, M.kv_pool_tokens(M.with_kv_dtype(m27, "fp16"), t1), "tokens",
        "baseline study (docs/scenarios.md, Limitations 1); the FP8 anchor is projected from it",
        "2026-07"))

    # The held-batch sweep the decode priors were read on (eta and the
    # 2.7 ms intercept), as tests/test_config.py prices it
    dep = {"model": "27B", "gpu": "H200", "tensor_parallel": 4,
           "recurrent_state_dtype": "fp32"}
    cfg = RunConfig.from_dict({"deployment": dep, "calibration": {"mfu": 0.32}})
    m, t, dc = cfg.to_model(), cfg.to_topology(), cfg.decode_constants()
    for n, ctx, ms in ((1, 46e3, 8.6), (8, 47e3, 11.1), (32, 40e3, 19.7),
                       (64, 37e3, 31.3), (8, 111e3, 14.4), (32, 105e3, 31.7),
                       (64, 103e3, 56.0), (1, 114e3, 10.2)):
        kv = n * ctx * m.kv_bpt + n * m.state_traffic
        rows.append(Row(
            "Decode step time", f"Qwen3.6-27B, 4×H200 TP4, 3 drafts, n = {n}, {ctx / 1e3:.0f}k context",
            ms, 1e3 * M.decode_step_seconds(m, t, n, kv, dc), "ms",
            "held-batch sweep (research/decode-model.md); eta and the intercept were read on it",
            "2026-09"))

    # InferenceX accepted length per step at the model's own draft depth
    by_model = {"GLM52": "inferencex/golden-al-glm-5.3-nvfp4",
                "DSV41F": "inferencex/golden-al-deepseek-v4.1-flash",
                "Q38FN": "inferencex/golden-al-qwen3.8-flash-next-fp8"}
    ids = {r.id: r for r in recs}
    for key, rid in by_model.items():
        r = ids[rid]
        mk = M.MODELS[key]
        k = mk.spec_tokens
        for met in r.metrics:
            if met.name != "acceptance_length" or not met.at.startswith(f"{k} draft"):
                continue
            used = "set from this record (at 3 drafts)" if mk.acceptance_measured else \
                f"not set from it: the catalogue carries {mk.acceptance} from a sibling"
            rows.append(Row(
                "Tokens per step (acceptance)",
                f"{_short(key)}, {k} drafts, {met.at.split(', ')[1]}",
                met.value, M.tokens_per_step(mk.acceptance, k), "tokens/step",
                f"InferenceX golden accepted length (`{rid}`)", r.read.isoformat(), used))

    rows += inferencex_curve_rows(recs)
    return rows


def _short(key: str) -> str:
    return {"GLM52": "GLM-5.3", "DSV41F": "DeepSeek-V4.1-Flash",
            "Q38FN": "Qwen3.8-Flash-Next"}.get(key, M.MODELS[key].name)


def inferencex_curve_rows(recs: list[X.Record]) -> list[Row]:
    """Lowest-concurrency point of each InferenceX agent curve on a GPU and
    a plain TP or TP+EP layout the model prices (no DP attention, one node)."""
    curves: dict[str, list[X.Record]] = defaultdict(list)
    for r in recs:
        c = r.config
        cid = (c.get("extra") or {}).get("curve")
        if (r.source.id == "inferencex" and cid and c.get("ws_model")
                and c.get("gpu") in M.GPUS and "DPA" not in c.get("parallel", "")
                and c.get("n_gpus", 99) <= 8 and r.metric("tok_s_per_user")):
            curves[cid].append(r)
    rows = []
    for cid in sorted(curves):
        r = min(curves[cid], key=lambda q: q.metric("concurrency").value)
        c = r.config
        gpu, n_gpu = c["gpu"], c["n_gpus"]
        weights = "nvfp4" if c["weights"] == "fp4" and gpu == "B300" else "fp8"
        m = arm(c["ws_model"], gpu, weights)
        topo = M.topology_grid(1, n_gpu, gpu, "replicate")
        if M.kv_pool_tokens(m, topo) <= 0:
            continue
        n = int(r.metric("concurrency").value)
        note = ""
        if c["weights"] == "fp4" and gpu == "H200":
            note = "run in FP4 weights, priced in FP8 (no NVFP4 arm on Hopper)"
        rows.append(Row(
            "Decode speed per user (InferenceX AgentX)",
            f"{_short(c['ws_model'])} {c['weights'].upper()}, {n_gpu}×{gpu} {c['parallel']}, "
            f"{c['engine'].split()[0]}, {n} session{'s' if n > 1 else ''}",
            r.metric("tok_s_per_user").value,
            tok_s_per_user(m, topo, n, AGENTX_CONTEXT), "tok/s",
            f"InferenceX `{r.id}`", r.published or r.read.isoformat(), note))
    return rows


# ------------------------------------------------- inputs not taken

def memory_cap_rows() -> list[Row]:
    """vLLM recipe pools at a memory cap and draft setting the model does
    not take: the size of what those inputs move."""
    rows = []
    src = "vLLM recipe, 8×GB300 at `--gpu-memory-utilization 0.85`, no drafter (research/model_mistral_large4.md § 2)"
    for weights, shard, measured in (("fp8", "replicate", 2.38e6), ("nvfp4", "replicate", 3.60e6),
                                     ("fp8", "dcp", 18.88e6), ("nvfp4", "dcp", 28.76e6)):
        topo = M.topology_grid(1, 8, "B300", shard)
        label = "TP8" if shard == "replicate" else "DCP8"
        rows.append(Row("KV pool", f"Mistral-Large-4 {weights.upper()}, 8×B300 {label}",
                        measured, M.kv_pool_tokens(arm("ML4", "B300", weights), topo),
                        "tokens", src, "2026-10-30"))
    src = "vLLM recipe, no drafter, memory cap not stated, \"about\" (research/model_kimi_k31pro.md § 8)"
    for shard, measured in (("replicate", 2.6e6), ("dcp", 18.5e6)):
        topo = M.topology_grid(1, 8, "B300", shard)
        label = "TP8" if shard == "replicate" else "DCP8"
        rows.append(Row("KV pool", f"Kimi-K3.1-Pro, 8×B300 {label}", measured,
                        M.kv_pool_tokens(M.MODELS["K31P"], topo), "tokens", src, "2026-11-05"))
    return rows


def engine_spread(recs: list[X.Record]) -> tuple[float, float, str]:
    """Lowest-concurrency per-user speed of DeepSeek-V4.1-Flash on 8×H100
    under vLLM TP8 and under SGLang TP8 EP8 (same GPUs, same weights)."""
    def low(cid):
        pts = [r for r in recs if (r.config.get("extra") or {}).get("curve") == cid]
        r = min(pts, key=lambda q: q.metric("concurrency").value)
        return r.metric("tok_s_per_user").value, r
    v, rv = low("dsv41flash-h100-vllm-fp4-TP8-8g")
    s, _ = low("dsv41flash-h100-sglang-fp4-TP8EP8-8g")
    return v, s, rv.read.isoformat()


def thinking_spread(recs: list[X.Record]) -> tuple[float, float]:
    r = next(q for q in recs if q.id == "inferencex/golden-al-deepseek-v4.1-flash")
    k = M.MODELS["DSV41F"].spec_tokens
    get = {m.at: m.value for m in r.metrics}
    return get[f"{k} draft tokens, thinking off"], get[f"{k} draft tokens, thinking on"]


# ----------------------------------------------------------------- page

def _span(recs: list[X.Record]) -> str:
    lo, hi = min(r.read for r in recs), max(r.read for r in recs)
    return lo.isoformat() if lo == hi else f"{lo.isoformat()} to {hi.isoformat()}"


def summary(rows: list[Row]) -> list[list[str]]:
    by = defaultdict(list)
    for r in rows:
        by[r.quantity].append(r)
    out = []
    for q, rs in by.items():
        errs = [r.error for r in rs]
        out.append([q, str(len(rs)), pct(sum(errs) / len(errs)),
                    pct(sum(abs(e) for e in errs) / len(errs)).lstrip("+"),
                    pct(max(errs, key=abs))])
    return out


def row_table(rows: list[Row], with_note: bool = True) -> str:
    head = ["conditions", "measured", "predicted", "error", "source", "date"]
    if with_note and any(r.note for r in rows):
        head.append("note")
    body = []
    for r in rows:
        line = [r.conditions, num(r.measured, r.unit), num(r.predicted, r.unit),
                pct(r.error), r.source, r.date]
        if len(head) == 7:
            line.append(r.note)
        body.append(line)
    return table(head, body)


def render() -> str:
    recs = X.load()
    checks = check_rows()
    fitted = fitted_rows(recs)
    caps = memory_cap_rows()
    v_tok, s_tok, eng_date = engine_spread(recs)
    al_off, al_on = thinking_spread(recs)

    ix_cal = sum(1 for r in recs if r.source.id == "inferencex"
                 and any(r.use_of(m) == "calibrate" for m in r.metrics))
    ix_check = [r for r in recs if r.source.id == "inferencex"
                and any(r.use_of(m) == "check" for m in r.metrics)]
    n_src = Counter(r.source.id for r in recs)
    covered = {"27B", "35BA3B", "ML4", "K31P"}
    unchecked = [M.MODELS[k].name for k in M.MODELS if k not in covered]

    sha = hashlib.sha256(CATALOG.read_bytes()).hexdigest()[:12]
    p27 = M.MEASURED_POOL_TOKENS_27B_TP2_FP16
    hybrid_gap = 2 * 110.59 - p27 * M.with_kv_dtype(M.MODELS["27B"], "fp16").kv_bpt / GIB

    by_q = defaultdict(list)
    for r in checks:
        by_q[r.quantity].append(r)
    fit_by_q = defaultdict(list)
    for r in fitted:
        fit_by_q[r.quantity].append(r)

    lone = next(r for r in fitted if r.quantity == "Decode step time" and "114k" in r.conditions)

    parts = [
        "<!-- GENERATED by scripts/accuracy.py. Do not edit: change the script or the model "
        "and run `uv run scripts/accuracy.py`. -->",
        "# Accuracy: the model against measured figures",
        "",
        "`scripts/accuracy.py` computes every predicted value from `workingset.model`. "
        "Error = (predicted − measured) / measured. Positive means the model over-estimates.",
        "",
        "Few public measurements match a configuration the model prices, so the "
        "counts are small. A figure that set one of the model's constants is in "
        "[Fitted anchors](#fitted-anchors), outside the error statistics: "
        "agreement there proves nothing.",
        "",
        "## Summary",
        "",
        "Independent measurements only: none of these set a constant.",
        "",
        table(["quantity", "n", "mean error", "mean absolute error", "largest error"],
              summary(checks)),
        "",
        "Not yet checked against any independent public measurement:",
        "",
        "- **Prefill on H200.** The prefill MFU (0.45) rests on measurements this "
        "repo does not publish (research/prefill.md). The one prefill measurement, "
        "Kimi-K3.1-Pro's, is on B300.",
        "- **Decode step time on B300** for any model but Mistral-Large-4, and that "
        "one runs on GB300. InferenceX's B300 curves are tagged `calibrate` and sit "
        "in the fitted table.",
        "- **Decode step time on H200** beyond the 27B sweep the constants were read on.",
        "- **Cache hit rate and warm capacity.** research/agentx_kv_hits.md compares "
        "the AgentX simulation with InferenceX's hit rates; the simulation takes "
        "minutes per seed and is not rerun here, and those rates are tagged `calibrate`.",
        "- **TTFT under load, queue wait, users per box.** No public measurement.",
        "- **Power and energy.** The constants are H100 proxies and B300 "
        "extrapolations (research/power.md); nothing measured on H200 or B300.",
        f"- **Models with no independent row.** {', '.join(unchecked)}.",
        "",
    ]

    for q, rs in by_q.items():
        parts += [f"## {q}", "", f"n = {len(rs)}.", "", row_table(rs), ""]

    parts += [
        "## Fitted anchors",
        "",
        "These figures set a constant, or are tagged `use: calibrate` in "
        "`data/external/` and reserved for a fit (WS-26). Agreement proves "
        "nothing; a disagreement still shows where the model is off.",
        "",
    ]
    for q, rs in fit_by_q.items():
        parts += [f"### {q}", "", row_table(rs), ""]
        if q == "Decode speed per user (InferenceX AgentX)":
            parts += [
                f"The model prices {AGENTX_CONTEXT / 1e3:.0f}k-token sequences, InferenceX's "
                "median input per AgentX request, at its own acceptance and draft "
                "depth, not the run's. Above one session the x-axes also differ: "
                "InferenceX counts sessions replaying traces with tool waits, the "
                "model counts sequences decoding at once (WS-26).",
                "",
            ]

    parts += [
        "## What the model does not take as input",
        "",
        "### GPU memory cap and draft setting (KV pool)",
        "",
        "By default the model sizes the pool from the whole HBM less a fixed "
        "reserve. The memory cap is an input only through "
        "`calibration.gpu_memory_utilization`, one of the reserve terms "
        "`ws calibrate --vllm-log` reads from a start-up log "
        "(research/reserve_terms.md); the drafter is not an input. vLLM recipes "
        "report pools at `--gpu-memory-utilization 0.85` and without a drafter, "
        "whose cache rows the model charges, and the rows below are priced at the "
        "default reserve, so they mix both effects:",
        "",
        row_table(caps, with_note=False),
        "",
        "### Engine",
        "",
        f"One set of decode constants serves every engine. DeepSeek-V4.1-Flash on "
        f"8×H100 at the lowest concurrency: vLLM TP8 {v_tok:.0f} tok/s per user, "
        f"SGLang TP8 EP8 {s_tok:.0f} (InferenceX, read {eng_date}): ×{v_tok / s_tok:.1f} "
        "between engines on the same GPUs.",
        "",
        "### Workload behind the acceptance",
        "",
        f"Acceptance is one number per model. InferenceX measures DeepSeek-V4.1-Flash "
        f"at {M.MODELS['DSV41F'].spec_tokens} drafts at {al_off:.2f} tokens per step with "
        f"thinking off and {al_on:.2f} with it on, {pct(al_off / al_on - 1)} from the "
        "prompt alone.",
        "",
        "### One long sequence decoding alone",
        "",
        f"The step equation has no term for it. The sweep's n = 1, 114k row is "
        f"{pct(lone.error)} (fitted table).",
        "",
        "### Hybrid-allocator state pages",
        "",
        f"The 27B TP2 FP16 log reports 2 × 110.59 GiB of KV memory; its "
        f"{p27:,} tokens at FP16 fill {2 * 110.59 - hybrid_gap:.1f} GiB of it, leaving "
        f"{hybrid_gap:.1f} GiB the allocator holds for DeltaNet state. The model "
        "charges state per live session instead and does not predict that figure.",
        "",
        "### Also not taken, unsized",
        "",
        "- FP4 weights on Hopper: InferenceX runs them on H200; the model prices FP8 there.",
        "- Sparse attention in prefill: DSA layers are priced as dense attention, an "
        "upper bound on prefill time.",
        "- The decode step jump past n ≈ 64 (issue #74).",
        "- Per-deployment reserves (CUDA graphs, a lowered memory cap): `ws calibrate "
        "--vllm-log` reads the reserve, or its three terms, from the start-up log; "
        "the default does not.",
        "",
        "## Excluded",
        "",
        table(["what", "records", "why"], [
            ["MLPerf Inference v6.0", str(n_src.get("mlperf-inference-v6.0", 0)),
             "MLCommons rules: \"MLPerf results may not be compared against non-MLPerf results\""],
            ["Artificial Analysis AgentPerf", "0",
             "its terms forbid redistribution; it never ships (research/sources/aa-agentperf.md)"],
            ["InferenceX, `use: calibrate`", str(ix_cal),
             "reserved for a fit: shown in the fitted table, never in the statistics"],
            ["InferenceX, `use: check` (derived MBU)", str(len(ix_check)),
             "models outside the catalogue (" + ", ".join(sorted({r.config['model'] for r in ix_check}))
             + "), so the model has no row to predict them"],
            ["Deployments and calculators", str(n_src.get("deployments", 0) + n_src.get("calculators", 0)),
             "`use: context`: shown with provenance, never checked"],
            ["27B / 4×H200 TP4 step at 2 drafts (~9.3 ms)", "1",
             "context and vLLM version not recorded (research/decode-model.md, decision 4)"],
            ["DeepSeek-V4.1-Flash cache bytes per token from a vLLM log", "1",
             "the catalogue's cache row was set from it (research/model_dsv41flash.md § 7)"],
            ["Live-endpoint records", "not counted",
             "kept in a private repository; only figures already in this repo appear here"],
        ]),
        "",
        "## Pinned to",
        "",
        f"- workingset {version('workingset')}, catalogue `catalog.toml` sha256 `{sha}`.",
        "- The model takes no engine version. Measured rows name their engine: the "
        "27B rows ran vLLM on H200; the recipe rows are vLLM recipes dated as shown; "
        "InferenceX rows carry their image tag in `data/external/inferencex.json`.",
        f"- Outside records: {', '.join(f'{s} {n}' for s, n in sorted(n_src.items()))}, "
        f"read {_span(recs)}.",
        "",
    ]
    return "\n".join(parts)


def main(argv: list[str]) -> int:
    text = render()
    if "--check" in argv:
        if not OUT.exists() or OUT.read_text() != text:
            print(f"{OUT.relative_to(ROOT)} is stale: run `uv run scripts/accuracy.py`",
                  file=sys.stderr)
            return 1
        return 0
    OUT.write_text(text)
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
