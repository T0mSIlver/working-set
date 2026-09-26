"""`ws link`: the explorer share URL that reproduces a config.

The explorer (interactive/) encodes its state as a URL fragment holding only
the DIFFS from its defaults (main.js encodeStateURL / applyURLState). This
module writes that fragment from a RunConfig, so the page and `ws predict`
price the same question.

`KNOBS` is the whole mapping, one row per explorer URL key. The explorer's
decoder clamps every number to its control's range and parseInt-truncates the
integer ones; the rows carry those ranges so a value the page would move is
reported here instead of silently changed there. Config fields the page has no
control for are checked in `_unmapped`. tests/test_link.py pins the keys and
ranges to interactive/src/main.js and index.html.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Callable
from urllib.parse import urlencode

from . import model as M
from .config import Endpoint, RunConfig

DEFAULT_BASE = "https://workingset.tomvaucourt.com/"

# interactive/src/state.js fixes the subagent prefix and the reported
# percentile; everything else in [workload] and [slo] has a control
EXPLORER_SUB_PREFIX = 3000
EXPLORER_PERCENTILE = 95
CHUNKS = (2048, 4096, 8192, 16384, 32768, 65536)
WOVER = {0.0: "pub", 0.15: "p15"}


def _ctx_scale(cfg: RunConfig) -> float:
    return M.MODELS[cfg.deployment.model].max_ctx / 262_144


def _cap_max(cfg: RunConfig) -> int:
    """state.js capSliderMax(): the top stop of the max_model_len slider."""
    return round(M.MODELS[cfg.deployment.model].max_ctx / 1000)


def _cap(cfg: RunConfig) -> float:
    # currentWL(): the top stop means the model's exact max context, every
    # other stop means cap * 1000
    mml = cfg.deployment.max_model_len
    return _cap_max(cfg) if mml >= M.MODELS[cfg.deployment.model].max_ctx else mml / 1000


def _nearest_wover(x: float) -> float:
    return min(WOVER, key=lambda w: abs(w - x))


def _users(cfg: RunConfig) -> float | None:
    # the page's users slider is system-wide; the config's is per group
    w = cfg.workload
    return None if w.headcount is not None else w.users * cfg.deployment.replicas


@dataclass(frozen=True)
class Knob:
    key: str                                # explorer URL key (state.js)
    field: str                              # config field(s) it comes from
    get: Callable[[RunConfig], Any]         # config -> explorer value (None = omit)
    default: Any                            # state.js STATE_DEFAULTS[key]
    lo: float | Callable[[RunConfig], float] | None = None   # slider min / max
    hi: float | Callable[[RunConfig], float] | None = None
    integer: bool = False                   # decoded with parseInt
    unit: str = ""                          # how the page shows the value


D, W, S, C = "deployment.", "workload.", "slo.", "calibration."
KNOBS: tuple[Knob, ...] = (
    Knob("model", D + "model", lambda c: c.deployment.model, "27B"),
    Knob("gpu", D + "gpu", lambda c: c.deployment.gpu, "H200"),
    Knob("wdt", D + "weight_dtype", lambda c: c.deployment.weight_dtype, "fp8"),
    Knob("kv", D + "kv_dtype", lambda c: c.deployment.kv_dtype, "fp8"),
    Knob("state_dt", D + "recurrent_state_dtype",
         lambda c: c.deployment.recurrent_state_dtype, "bf16"),
    Knob("wover", D + "weight_overhead",
         lambda c: WOVER[_nearest_wover(c.deployment.weight_overhead)], "pub"),
    Knob("kvshard", D + "kv_sharding", lambda c: c.deployment.kv_sharding, "dcp"),
    Knob("chunk", D + "max_num_batched_tokens",
         lambda c: str(c.deployment.max_num_batched_tokens), "32768"),
    Knob("ngpu", D + "tensor_parallel x replicas",
         lambda c: c.deployment.tensor_parallel * c.deployment.replicas, 1, 1, 8, True),
    Knob("tp", D + "tensor_parallel", lambda c: c.deployment.tensor_parallel, 1, 1, 8, True),
    Knob("ram", D + "ram_gib x replicas",
         lambda c: c.deployment.ram_gib * c.deployment.replicas, 0, 0, 1024, True, " GiB"),
    Knob("cap", D + "max_model_len", _cap, 180, 30, _cap_max, True, "k tokens"),
    Knob("user_median", W + "user_prompt_median_tokens",
         lambda c: c.workload.user_prompt_median_tokens / 1000, 31,
         1, lambda c: round(120 * _ctx_scale(c)), unit="k tokens"),
    Knob("user_sigma", W + "user_prompt_sigma", lambda c: c.workload.user_prompt_sigma,
         0.81, 0.30, 1.40),
    Knob("sub_median", W + "subagent_median_tokens",
         lambda c: c.workload.subagent_median_tokens / 1000, 8,
         1, lambda c: round(60 * _ctx_scale(c)), unit="k tokens"),
    Knob("sub_sigma", W + "subagent_sigma", lambda c: c.workload.subagent_sigma,
         0.90, 0.30, 1.40),
    Knob("sub_ratio", W + "subagent_ratio", lambda c: c.workload.subagent_ratio,
         0.10, 0, 1.0),
    Knob("sys", W + "system_prefix_tokens",
         lambda c: c.workload.system_prefix_tokens / 1000, 15, 1, 40, unit="k tokens"),
    Knob("inval", W + "miss_rate", lambda c: c.workload.miss_rate * 100, 1.0, 0, 100,
         unit="%"),
    Knob("users", W + "users x replicas", _users, 64, 4, 1024, True),
    Knob("headcount", W + "headcount", lambda c: c.workload.headcount, None,
         0, 1_000_000_000, True),
    Knob("active", W + "peak_active_share",
         lambda c: c.workload.peak_active_share if c.workload.headcount is not None else None,
         1.0, 0.01, 1),
    Knob("spu", W + "sessions_per_active_user",
         lambda c: (c.workload.sessions_per_active_user
                    if c.workload.headcount is not None else None), 1.0, 1, 8),
    Knob("think", W + "think_time_s", lambda c: c.workload.think_time_s, 30, 5, 180,
         True, " s"),
    Knob("turn", W + "warm_turn_tokens", lambda c: c.workload.warm_turn_tokens, 2000,
         250, 16000, True),
    Knob("out", W + "max_output_tokens", lambda c: c.workload.max_output_tokens, 400,
         100, 8000, True),
    Knob("sub_shares_prefix", W + "sub_shares_prefix",
         lambda c: c.workload.sub_shares_prefix, False),
    Knob("sla", S + "ttft_budget_s", lambda c: c.slo.ttft_budget_s, 10, 1, 60, True, " s"),
    Knob("decode_floor", S + "itl_floor_tok_s", lambda c: c.slo.itl_floor_tok_s, 40,
         5, 100, True, " tok/s"),
    Knob("mfu", C + "mfu", lambda c: c.calibration.mfu, 0.45, 0.10, 1.00),
    Knob("mbu", C + "mbu", lambda c: c.calibration.mbu, 0.22, 0.10, 1.00),
    # an unset mtp is the model's own, which the page applies on its own when
    # the link names the model; encodeStateURL diffs mtp against it likewise
    Knob("mtp", C + "mtp", lambda c: c.calibration.mtp, None, 1.0, 3.0),
)


def _js(v: Any) -> str:
    """The string encodeStateURL writes for the same value: 1 not 1.0, a
    boolean as 1/0. Binary-float dust from a unit conversion (0.07 * 100 =
    7.000000000000001) is trimmed; any real precision is kept."""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (int, float)):
        v = float(v)
        r = round(v, 9)
        if abs(r - v) <= 1e-12 * max(1.0, abs(v)):
            v = r
        return str(int(v)) if v.is_integer() else repr(v)
    return str(v)


def _js_round(x: float) -> int:
    """JS Math.round: halves go up (Python's round() goes to even)."""
    return math.floor(x + 0.5)


def _bound(b, cfg):
    return b(cfg) if callable(b) else b


def _unmapped(cfg: RunConfig) -> list[str]:
    """Config fields the explorer has no control for, when they differ from
    what the page assumes."""
    out = []
    w, d, s, c = cfg.workload, cfg.deployment, cfg.slo, cfg.calibration
    if w.subagent_prefix_tokens != EXPLORER_SUB_PREFIX:
        out.append(f"workload.subagent_prefix_tokens = {w.subagent_prefix_tokens}: the "
                   f"explorer has no control for it and prices {EXPLORER_SUB_PREFIX}")
    if s.percentile != EXPLORER_PERCENTILE:
        out.append(f"slo.percentile = {s.percentile}: the explorer has no control for "
                   f"it and reports p{EXPLORER_PERCENTILE}")
    if d.max_num_seqs is not None:
        out.append(f"deployment.max_num_seqs = {d.max_num_seqs}: the explorer has no "
                   "control for it; its decode ceiling is the roofline's alone")
    if c.decode_pricing != "roofline":
        out.append(f"calibration.decode_pricing = {c.decode_pricing!r} (with "
                   "decode_bw_eff, decode_fixed_ms, spec_tokens): the explorer has "
                   "no control for it and prices decode with the roofline at mbu")
    if d.weight_overhead not in WOVER:
        near = _nearest_wover(d.weight_overhead)
        out.append(f"deployment.weight_overhead = {d.weight_overhead:g}: the explorer "
                   f"offers only 0 or 0.15; the link sets {near:g}")
    if d.max_num_batched_tokens not in CHUNKS:
        out.append(f"deployment.max_num_batched_tokens = {d.max_num_batched_tokens}: the "
                   f"explorer takes only {', '.join(map(str, CHUNKS))}; it will show "
                   "32768")
    m = M.MODELS[d.model]
    if d.kv_sharding == "replicate" and d.tensor_parallel <= (m.kv_heads or 1):
        # main.js enforceConstraints: the arm only prices something past the
        # model's KV heads, and the page resets it below that
        out.append(f"deployment.kv_sharding = 'replicate' at TP{d.tensor_parallel}: "
                   f"{d.model} has {m.kv_heads or 1} KV head(s), so the explorer "
                   "resets it to 'dcp' (both layouts store one copy here)")
    # the page's config download writes these defaults back, whatever the
    # config said (harness.js workingsetConfig)
    for f in fields(Endpoint):
        v, dflt = getattr(cfg.endpoint, f.name), getattr(Endpoint(), f.name)
        # the page's own download names the model with a "<your served
        # model id ...>" placeholder; that is its value, not the user's
        if v != dflt and not (f.name == "model" and str(v).startswith("<")):
            shown = "no value" if dflt is None else (
                "a placeholder" if dflt == "" else repr(dflt))
            out.append(f"endpoint.{f.name} = {v!r}: the explorer does not carry it; "
                       f"a config downloaded from the page will have {shown}")
    return out


def explorer_link(cfg: RunConfig, base: str = DEFAULT_BASE) -> tuple[str, list[str]]:
    """(share URL, warnings). A warning names a config field the page will
    not show as written, and what it shows instead."""
    cfg.validate()
    d = cfg.deployment
    if d.tensor_parallel * d.replicas > 8:
        # the page holds one 8-GPU node: it would clamp the count and re-derive
        # the split (clampTp), pricing a different topology under the same users
        raise ValueError(f"deployment: TP{d.tensor_parallel} x {d.replicas} replicas = "
                         f"{d.tensor_parallel * d.replicas} GPUs; the explorer shows at "
                         "most one 8-GPU node, so no link can show this config")
    warnings = _unmapped(cfg)
    params: dict[str, str] = {}
    for k in KNOBS:
        v = k.get(cfg)
        if v is None:
            continue
        if k.key == "chunk" and int(v) not in CHUNKS:
            continue
        lo, hi = _bound(k.lo, cfg), _bound(k.hi, cfg)
        shown = v
        if lo is not None and not isinstance(v, (bool, str)):
            shown = min(max(v, lo), hi)
            if k.integer:
                shown = int(shown)          # parseInt truncates
            if shown != v:
                why = "outside the control's range" if not lo <= v <= hi else \
                      "the control takes whole numbers"
                warnings.append(f"{k.field} -> {k.key} = {_js(v)}{k.unit}: {why} "
                                f"[{_js(lo)}, {_js(hi)}]; the explorer "
                                f"will show {_js(shown)}{k.unit}")
            if k.key == "cap" and shown >= hi and \
                    cfg.deployment.max_model_len != M.MODELS[cfg.deployment.model].max_ctx:
                # currentWL(): the top stop is the model's exact maximum
                warnings.append(
                    f"deployment.max_model_len = {cfg.deployment.max_model_len}: "
                    f"{_js(shown)}k is the explorer's top stop, which prices the "
                    f"model's maximum, {M.MODELS[cfg.deployment.model].max_ctx:,.0f} tokens")
        params[k.key] = _js(shown)
    if (w := cfg.workload).headcount is not None:
        # the page derives users from its sliders, which clamp active and spu
        knob = {k.key: k for k in KNOBS}
        active = min(max(w.peak_active_share, knob["active"].lo), knob["active"].hi)
        spu = min(max(w.sessions_per_active_user, knob["spu"].lo), knob["spu"].hi)
        raw = M.sessions_from_headcount(w.headcount, active, spu)
        page = min(max(_js_round(raw / 4) * 4, 4), 1024)    # main.js syncLabels
        if abs(raw - page) > 1e-6:
            warnings.append(f"workload.headcount -> {raw:g} sessions: the explorer prices "
                            f"{page} (its users slider takes multiples of 4 in "
                            "[4, 1024]); ws predict prices the exact load")
    # encodeStateURL's own diffing: a value at the page's default is not
    # written, and mtp / gpuh diff against the selected model / GPU
    model = params["model"]
    defaults = {k.key: k.default for k in KNOBS}
    defaults["mtp"] = M.MODELS[model].mtp
    frag = {k: v for k, v in params.items() if v != _js(defaults[k])}
    q = urlencode(frag, safe="")
    return base + ("#" + q if q else ""), warnings


def cmd_link(args) -> int:
    import sys
    from .config import load_config
    url, warnings = explorer_link(load_config(args.config), args.base.split("#")[0])
    for w in warnings:
        print(f"ws link: warning: {w}", file=sys.stderr)
    print(url)
    return 0
