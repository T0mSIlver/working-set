"""The run configuration: one file that names a deployment, a workload and an SLO.

This is the contract between the explorer, the CLI and the hypotheses. The
explorer writes it (`workingset.toml`, or JSON), `ws predict` prices it, `ws
test` measures it. Predictions are NEVER stored in the file: they are computed
from it by `workingset.model` at run time, so a file can never carry a number
the code did not produce.

Blocks (TOML tables) and their model counterparts:

  [deployment]  model key + dtype arms + GPU part + tp/dp + chunk + max_model_len
                -> (Model, Topology)                         [model.MODELS, topology_grid]
  [workload]    the closed-loop agentic workload             [model.Workload + turn/think/out]
  [slo]         what "served" means                          [operating_point kwargs]
  [endpoint]    where to send requests (measurement only)
  [calibration] mfu / mbu overrides (defaults = the study's calibrated values)

Field names follow the explorer's generated CONFIG block one-for-one (the
harness's `workload` keys), so a CONFIG block from a harness downloaded
before the package maps onto this schema without renaming.
"""
from __future__ import annotations

import json
import tomllib
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

from . import model as M

SCHEMA_VERSION = 1
STATE_DTYPES = ("bf16", "fp32")

# The model with no as-published weight overhead to add: its w_resident is the
# vendor's stated as-DEPLOYED footprint, and comparing it against the raw
# checkpoint bytes is where the +15% figure was measured in the first place.
# Mirrors main.js enforceConstraints (`woverOk = state.model !== "27B"`).
_NO_WEIGHT_OVERHEAD = ("27B",)


def _fp32_state_applies(m: M.Model) -> bool:
    """render.js modelFor()'s gate: a recurrent state exists and is a bf16
    buffer the toggle can meaningfully widen."""
    return m.deltanet_state > 0 and m.state_fp32_ok is not False


def _weight_overhead_applies(model_key: str | None) -> bool:
    return model_key not in _NO_WEIGHT_OVERHEAD


@dataclass(frozen=True)
class Endpoint:
    base_url: str = "http://localhost:8000/v1"
    model: str = ""                      # served model id
    api_key_env: str = "VLLM_API_KEY"    # env var NAME, never the key
    metrics_url: str | None = None       # vLLM /metrics, when reachable


@dataclass(frozen=True)
class Deployment:
    # key into workingset.model.MODELS. None = not stated (a legacy harness
    # CONFIG names only the served checkpoint): to_model() refuses to guess.
    model: str | None = "27B"
    gpu: str = "H200"                    # key into workingset.model.GPUS
    tensor_parallel: int = 1
    replicas: int = 1                    # data-parallel replica groups
    weight_dtype: str = "fp8"            # fp8 | nvfp4
    kv_dtype: str = "fp8"                # fp8 | fp16
    max_num_batched_tokens: int = M.CHUNK_DEFAULT
    max_model_len: int = 180_000
    ram_gib: float = 0.0                 # CPU KV offload per replica group (explorer's RAM knob)
    # Recurrent (Gated DeltaNet / KDA) state precision. "fp32" doubles the
    # per-session state bytes, which is a per-session charge against the KV
    # pool: the explorer's fp32-state control. No-op on a model with no
    # recurrent state, or whose state is not a bf16 buffer to widen
    # (state_fp32_ok=False) — validate() refuses those rather than no-opping.
    recurrent_state_dtype: str = "bf16"   # bf16 | fp32
    # Deployed-weight overhead over the checkpoint bytes, as a fraction:
    # the explorer's "+15% weights" arm is 0.15. Raises w_resident, which
    # costs KV pool and lengthens the prefill weight stream. Refused on the
    # 27B, whose 28.8 GiB is already the as-deployed footprint (that
    # measurement is where the 15% came from).
    weight_overhead: float = 0.0          # 0.0 = as published

    @property
    def gpus(self) -> str:
        return f"{self.tensor_parallel * self.replicas}x{self.gpu}"


@dataclass(frozen=True)
class WorkloadCfg:
    system_prefix_tokens: int = 15_000
    user_prompt_median_tokens: int = 31_000
    user_prompt_sigma: float = 0.81
    warm_turn_tokens: int = 2_000
    think_time_s: float = M.THINK_TIME_S
    subagent_ratio: float = 0.10
    subagent_median_tokens: int = 8_000
    subagent_sigma: float = 0.9
    subagent_prefix_tokens: int = 3_000
    sub_shares_prefix: bool = False
    miss_rate: float = 0.01
    max_output_tokens: int = M.OUT_TOKENS_DEFAULT
    # Optional population layer above the operating point. `headcount` is the
    # number of people with access; the other two factors turn it into
    # system-wide concurrent sessions. Both shares are operator-supplied and
    # unmeasured. When headcount is set, `users` must be omitted.
    headcount: int | None = None
    peak_active_share: float = 1.0
    sessions_per_active_user: float = 1.0
    # The operating point, PER REPLICA GROUP — and therefore fractional
    # whenever the load does not divide by the replica count. It is a LOAD, not
    # a population: everything downstream reads it as an arrival rate
    # (users / think_time_s), which is perfectly well defined at 0.5. Rounding
    # it here would double the rate a DP8 deployment is priced at when the page
    # shows 4 users across 8 groups. The two places that need a whole number of
    # sessions round it themselves, at the point they build one: the load
    # ladder (probe/ladder.build_ladder) and the burst's standing load
    # (hypotheses/context.RunContext._burst_pop).
    users: float | None = M.REF_USERS


@dataclass(frozen=True)
class SLO:
    ttft_budget_s: float = 10.0
    itl_floor_tok_s: float = M.DECODE_FLOOR_TOKS
    percentile: int = 95


@dataclass(frozen=True)
class Calibration:
    mfu: float = M.MFU_DEFAULT
    mbu: float = M.MBU_DEFAULT
    # Effective decode speedup from speculative decoding / MTP. Multiplies
    # per-user decode tok/s directly, so it moves the decode ceiling and the
    # steady point. None = the model's own value (M.MODELS[key].mtp), which is
    # measured for the 27B and transplanted everywhere else; the explorer
    # exposes it as a slider for exactly that reason. It travels with the MBU
    # it was fitted against — moving one without the other breaks the fit.
    mtp: float | None = None


@dataclass(frozen=True)
class RunConfig:
    deployment: Deployment = field(default_factory=Deployment)
    workload: WorkloadCfg = field(default_factory=WorkloadCfg)
    slo: SLO = field(default_factory=SLO)
    endpoint: Endpoint = field(default_factory=Endpoint)
    calibration: Calibration = field(default_factory=Calibration)
    schema_version: int = SCHEMA_VERSION

    # ---- model objects -------------------------------------------------
    def to_model(self) -> M.Model:
        d = self.deployment
        if d.model is None:
            raise ValueError("deployment.model is not set (a downloaded harness "
                             "names only the served checkpoint): pass --model KEY, "
                             f"one of {sorted(M.MODELS)}")
        if d.model not in M.MODELS:
            raise KeyError(f"unknown model key {d.model!r}; known: {sorted(M.MODELS)}")
        m = M.MODELS[d.model]
        if d.weight_dtype != "fp8":
            m = M.with_weight_dtype(m, d.weight_dtype)
        m = M.with_kv_dtype(m, d.kv_dtype)
        # The explorer's three remaining model knobs, in modelFor()'s own
        # order and under its own gates (interactive/src/render.js). They are
        # `replace` on the Model rather than functions in workingset.model
        # because that is all the explorer does: no new pricing, three field
        # edits. validate() refuses the combinations modelFor() would skip, so
        # a gate can never turn into a silently unapplied knob here.
        if d.recurrent_state_dtype == "fp32" and _fp32_state_applies(m):
            m = replace(m, deltanet_state=m.deltanet_state * 2,
                        name=m.name + " [fp32 state]")
        if d.weight_overhead and _weight_overhead_applies(d.model):
            m = replace(m, w_resident=m.w_resident * (1 + d.weight_overhead),
                        name=m.name + f" [+{d.weight_overhead:.0%} weights]")
        if self.calibration.mtp is not None:
            m = replace(m, mtp=self.calibration.mtp)
        return m

    def to_topology(self) -> M.Topology:
        d = self.deployment
        if d.gpu not in M.GPUS:
            raise KeyError(f"unknown GPU {d.gpu!r}; known: {sorted(M.GPUS)}")
        return M.topology_grid(d.replicas, d.tensor_parallel, d.gpu)

    def to_workload(self) -> M.Workload:
        w = self.workload
        return M.Workload(
            user_median=w.user_prompt_median_tokens, user_sigma=w.user_prompt_sigma,
            sub_median=w.subagent_median_tokens, sub_sigma=w.subagent_sigma,
            sub_ratio=w.subagent_ratio, sys_user=w.system_prefix_tokens,
            sys_sub=w.subagent_prefix_tokens, sub_shares_prefix=w.sub_shares_prefix,
            invalidation=w.miss_rate, cap=self.deployment.max_model_len)

    def users_per_group(self) -> float:
        """The concurrent-session load one replica group must serve."""
        w = self.workload
        if w.headcount is not None:
            sessions = M.sessions_from_headcount(
                w.headcount, w.peak_active_share, w.sessions_per_active_user)
            return sessions / self.deployment.replicas
        if w.users is None:
            raise ValueError("workload must set either users or headcount")
        return w.users

    def validate(self) -> None:
        """Raise on anything the model refuses to price."""
        m, t, wl = self.to_model(), self.to_topology(), self.to_workload()
        M.check_dtype_supported(m, t)
        M.check_cap_allowed(m, wl)
        if self.deployment.weight_dtype not in M.WEIGHT_DTYPES:
            raise ValueError(f"weight_dtype must be one of {M.WEIGHT_DTYPES}")
        if self.deployment.kv_dtype not in M.KV_DTYPES:
            raise ValueError(f"kv_dtype must be one of {M.KV_DTYPES}")
        w = self.workload
        if w.headcount is not None and w.users is not None:
            raise ValueError("workload.headcount and workload.users cannot both be set; "
                             "users is derived from headcount")
        if w.headcount is not None:
            M.sessions_from_headcount(w.headcount, w.peak_active_share,
                                      w.sessions_per_active_user)
        else:
            if w.users is None:
                raise ValueError("workload must set either users or headcount")
            if w.users < 0:
                raise ValueError("workload.users must be >= 0")
            # Validate the dormant defaults too. A malformed population factor
            # must not become valid merely because headcount is absent.
            M.sessions_from_headcount(0, w.peak_active_share,
                                      w.sessions_per_active_user)
        if self.deployment.ram_gib < 0:
            raise ValueError("deployment.ram_gib must be >= 0")
        d = self.deployment
        if d.recurrent_state_dtype not in STATE_DTYPES:
            raise ValueError(f"deployment.recurrent_state_dtype must be one of "
                             f"{STATE_DTYPES}, got {d.recurrent_state_dtype!r}")
        # the explorer snaps these two back rather than mis-pricing (main.js
        # enforceConstraints); at the file boundary they are refusals, so a
        # hand-written config cannot ask for a knob that would do nothing
        if d.recurrent_state_dtype == "fp32" and not _fp32_state_applies(M.MODELS[d.model]):
            raise ValueError(
                f"{M.MODELS[d.model].name}: recurrent_state_dtype 'fp32' does "
                "nothing here — the model carries no bf16 recurrent state to "
                "widen (deltanet_state 0, or a fixed mixed-precision buffer)")
        if d.weight_overhead < 0:
            raise ValueError("deployment.weight_overhead must be >= 0")
        if d.weight_overhead and not _weight_overhead_applies(d.model):
            raise ValueError(
                f"{d.model}: weight_overhead does not apply — its w_resident "
                "is already the as-deployed footprint, which is where the 15% "
                "figure came from")
        if self.calibration.mtp is not None and self.calibration.mtp <= 0:
            raise ValueError("calibration.mtp must be > 0")

    # ---- (de)serialisation --------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RunConfig":
        raw = dict(raw)
        version = raw.pop("schema_version", SCHEMA_VERSION)
        if version != SCHEMA_VERSION:
            raise ValueError(f"config schema_version {version} not supported "
                             f"(this workingset reads {SCHEMA_VERSION})")
        raw.pop("predictions", None)      # a harness CONFIG carries these; ignored
        raw.pop("hypotheses", None)
        blocks = {"deployment": Deployment, "workload": WorkloadCfg, "slo": SLO,
                  "endpoint": Endpoint, "calibration": Calibration}
        kw: dict[str, Any] = {}
        for name, typ in blocks.items():
            block = dict(raw.pop(name, {}) or {})
            _reject_unknown(block, typ, name)
            # The dataclass default keeps legacy direct-load configs at 64.
            # A headcount-only file selects the other representation, so make
            # the omitted direct load explicit before construction.
            if name == "workload" and "headcount" in block and "users" not in block:
                block["users"] = None
            kw[name] = typ(**block)
        if raw:
            raise ValueError(f"unknown top-level config keys: {sorted(raw)}")
        return cls(schema_version=version, **kw)

    def dumps(self, fmt: str = "toml") -> str:
        d = self.to_dict()
        if fmt == "json":
            return json.dumps(d, indent=2) + "\n"
        if fmt == "toml":
            return _dump_toml(d)
        raise ValueError(f"unknown format {fmt!r}")


_SCALAR = {"int": (int,), "float": (int, float), "bool": (bool,), "str": (str,),
           "int | None": (int, type(None)), "str | None": (str, type(None)),
           "float | None": (int, float, type(None))}


def _reject_unknown(block: dict, typ, name: str) -> None:
    """Unknown keys and wrong scalar types fail HERE, at the file boundary,
    not three calls deep inside the model with a numpy error."""
    known = {f.name: f.type for f in fields(typ)}
    for k, v in block.items():
        if k not in known:
            raise ValueError(f"unknown key {name}.{k}; known: {sorted(known)}")
        want = _SCALAR.get(str(known[k]))
        if want is None:
            continue
        ok = isinstance(v, want) and not (isinstance(v, bool) and bool not in want)
        if not ok:
            raise ValueError(f"{name}.{k}: expected {known[k]}, got {v!r}")


def _toml_scalar(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if v is None:
        raise TypeError("TOML has no null; omit the key instead")
    return json.dumps(str(v))


def _dump_toml(d: dict[str, Any]) -> str:
    if d["deployment"].get("model") is None:
        # an omitted key would read back as the dataclass default: a guess
        raise ValueError("deployment.model is not set; a config file cannot "
                         "be written without a model key")
    out = [f"schema_version = {d['schema_version']}", ""]
    for block in ("deployment", "workload", "slo", "endpoint", "calibration"):
        out.append(f"[{block}]")
        for k, v in d[block].items():
            if v is None:
                continue
            # Keep the two workload representations distinct on disk. Direct
            # session configs retain their old shape; headcount configs retain
            # the product inputs and never collapse to derived users.
            if block == "workload":
                has_headcount = d[block].get("headcount") is not None
                if k in ("peak_active_share", "sessions_per_active_user") and not has_headcount:
                    continue
                if k == "users" and has_headcount:
                    continue
            out.append(f"{k} = {_toml_scalar(v)}")
        out.append("")
    return "\n".join(out)


def load_config(path: str | Path) -> RunConfig:
    """Read a TOML or JSON config. A downloaded harness .py is accepted
    too: its CONFIG block is extracted, so an explorer download from before
    the package can still be priced."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix == ".json":
        return RunConfig.from_dict(_legacy_to_schema(json.loads(text)))
    if p.suffix == ".py":
        return RunConfig.from_dict(_legacy_to_schema(_config_from_harness(text)))
    return RunConfig.from_dict(tomllib.loads(text))


def _config_from_harness(src: str) -> dict:
    """The CONFIG between the harness's BEGIN/END markers: either the
    explorer's json.loads(r\"\"\"...\"\"\") form or the committed template's
    dict literal (comments and 4_096 underscores are fine for the AST)."""
    import ast
    import re
    m = re.search(r'CONFIG = json\.loads\(r"""\n([\s\S]*?)\n"""\)', src)
    if m:
        return json.loads(m.group(1))
    m = re.search(r"# --- BEGIN CONFIG[^\n]*\n([\s\S]*?)# --- END CONFIG", src)
    if not m:
        raise ValueError("no explorer CONFIG block found in this harness file")
    body = m.group(1)
    i = body.find("CONFIG = ")
    if i < 0:
        raise ValueError("CONFIG assignment not found inside the CONFIG block")
    try:
        return ast.literal_eval(body[i + len("CONFIG = "):].strip())
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"could not parse the harness CONFIG literal: {e}") from e


def _legacy_to_schema(raw: dict) -> dict:
    """Map an explorer/harness CONFIG (workload keys identical, deployment
    described by a 'gpus' string) onto this schema. Only the keys this
    schema lacks are synthesised; everything else passes through."""
    raw = dict(raw)
    dep = dict(raw.get("deployment", {}) or {})
    if "model" not in dep:
        # the harness block names the served checkpoint, not the model key.
        # None makes to_model()/validate() refuse until --model says which.
        dep["model"] = None
    if "gpu" not in dep and "gpus" in dep:
        dep["gpu"] = str(dep["gpus"]).split("x", 1)[-1]
    dep.pop("gpus", None)
    raw["deployment"] = dep
    wl = dict(raw.get("workload", {}) or {})
    if "context_cap_tokens" in wl:
        dep.setdefault("max_model_len", wl.pop("context_cap_tokens"))
    # the load the harness's predictions were computed at lives only in its
    # predictions block; it is the operating point, not a prediction
    preds = raw.get("predictions") or {}
    if "users" not in wl and "operating_point_users" in preds:
        wl["users"] = float(preds["operating_point_users"])
    raw["workload"] = wl
    return raw
