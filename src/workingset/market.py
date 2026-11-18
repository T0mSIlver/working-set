"""The model's cost of serving an open-weights model next to what providers
charge for it: the utilisation a provider must reach to cover GPU rent at its
price, under stated assumptions, with the band those assumptions leave.

Derivation (every figure in USD; rent is paid every hour, 24/7):

  workload   RunConfig()'s [workload], the explorer's opening workload. Per
             request: E[L] prompt tokens, of which `read` come from a prefix
             cache and `written` are computed fresh, and `out` decoded
             (api_cost.request_flows). A live session sends
             (1 + sub_ratio) / think_time_s requests per second.
  capacity   for each GPU, accepted TP width (catalogue vllm_tp, one 8-GPU
             node) and priced weight arm: L = predicted_limit_users x
             replicas, the verdict's binding ceiling (predict.predict). DP
             replicas multiply sessions and GPUs alike (balanced routing),
             so the cost per session of DPk x TPt is TPt's: only DP1 is
             priced.
  cost/hour  C = n_gpu x rent; rent low / median / high over sellers (not
             aggregators) for one term.
  u          time average of live sessions / L over every rented hour. Cost
             per live session-hour at u: C / (L u).
  price      R = what one live session-hour of the workload costs on a
             provider's price sheet (api_cost.request_cost: cached tokens at
             the published cache-read price, else at the input price).
  u*         (C / L) / R. u* > 1: the price does not cover the rent at the
             model's predicted limit even fully loaded, under these
             assumptions.

The per-token split (cost per million fresh input and output tokens) is a
convention: prefill time to fresh input, decode time to output, idle time to
both in proportion to their busy time. u* does not depend on it.
"""
from __future__ import annotations

import json
import math
import re
import statistics
from dataclasses import dataclass, replace
from pathlib import Path

from . import api_cost as A
from . import model as M
from .config import Calibration, Deployment, RunConfig
from .predict import predict

SCHEMA = "workingset.market/1"
NODE = 8                                  # one 8-GPU node: the explorer's frontier
UTILISATIONS = (0.2, 0.5, 0.8, 1.0)
U_GRID = tuple(round(0.05 * i, 2) for i in range(1, 21))
CACHE_TTL_DEFAULT = 300.0                 # s; providers publish no TTL (assumption)
TERMS = ("on-demand", "committed")
LEVELS = ("low", "median", "high")
DEFAULT_SNAPSHOT = Path(__file__).resolve().parents[2] / "data" / "market"

# The weights the catalogue's base ("fp8") arm prices, by model. The
# catalogue has no released-dtype field (proposed: one in catalog.toml); this
# reads its comments. DeepSeek-V4 and Kimi ship MXFP4 routed experts, and the
# base arm is that native checkpoint, so an FP4 / MXFP4 endpoint matches it.
NATIVE_FP4_BASE = frozenset({"DSV4F", "DSV4FV", "DSV41F", "K31P"})
ARM_LABEL = {"fp8": "base checkpoint", "nvfp4": "NVFP4"}

# accuracy rows that check the quantity each binding ceiling rests on
BINDING_QUANTITY = {"decode": "Decode speed per user",
                    "cold_wait": "Prefill, idle cold TTFT",
                    "slowed": "Prefill, idle cold TTFT",
                    "saturation": "Prefill, idle cold TTFT"}


class MarketDataError(ValueError):
    """A market file is missing, malformed, or lacks a field."""


# ---------------------------------------------------------------- inputs
def _need(d: dict, key: str, where: str):
    if key not in d:
        raise MarketDataError(f"{where}: field {key!r} missing")
    return d[key]


def _latest(dir_: Path, prefix: str) -> Path | None:
    hits = sorted(dir_.glob(f"{prefix}-*.json"))
    return hits[-1] if hits else None


def load_snapshot(dir_: str | Path | None = None) -> dict:
    """The market files of one snapshot directory. Prices come from
    `provider_prices.json` and/or the latest `openrouter-*.json`; rent from
    the latest `gpu_rental-*.json`; `chars_per_token.json` is optional."""
    d = Path(dir_) if dir_ is not None else DEFAULT_SNAPSHOT
    if not d.is_dir():
        raise MarketDataError(f"market snapshot directory {d} does not exist: pass "
                              "--snapshot DIR (a checkout has data/market)")
    price_files = [p for p in (d / "provider_prices.json", _latest(d, "openrouter"))
                   if p is not None and p.exists()]
    if not price_files:
        raise MarketDataError(f"{d}: no price file (provider_prices.json or "
                              "openrouter-*.json)")
    rental = _latest(d, "gpu_rental")
    if rental is None:
        raise MarketDataError(f"{d}: no gpu_rental-*.json, so no rent to price the "
                              "GPUs at")
    chars = d / "chars_per_token.json"
    out = {"dir": str(d), "prices": [], "rental_file": rental.name,
           "rental": json.loads(rental.read_text()),
           "chars_file": chars.name if chars.exists() else None,
           "chars": json.loads(chars.read_text()) if chars.exists() else None}
    for p in price_files:
        raw = json.loads(p.read_text())
        src = _need(raw, "source", p.name)
        if "models" in raw:               # openrouter-*: models -> endpoints
            models = raw["models"]
        elif "rows" in raw:               # provider_prices: one row per (model, provider, tier)
            by: dict = {}
            for i, r in enumerate(raw["rows"]):
                key = _need(r, "ws_model", f"{p.name} rows[{i}]")
                by.setdefault(key, []).append({**r, "tag": r.get("tier")})
            models = [{"ws_model": k, "endpoints": v} for k, v in by.items()]
        else:
            raise MarketDataError(f"{p.name}: neither 'models' nor 'rows'")
        out["prices"].append({"file": p.name, "source": src, "models": models})
    return out


def rent_levels(rental: dict, gpu: str, term: str) -> dict:
    """Low, median and high USD per GPU-hour over sellers for one GPU and
    term. Aggregator records are counted and left out."""
    recs = _need(rental, "records", "gpu_rental")
    best: dict = {}
    aggs, unpriced = 0, []
    for i, r in enumerate(recs):
        where = f"gpu_rental records[{i}]"
        if _need(r, "gpu", where) != gpu or _need(r, "term", where) != term:
            continue
        if _need(r, "kind", where) != "seller":
            aggs += 1
            continue
        v = _need(r, "usd_per_gpu_h", where)
        if v is None:                     # a recorded null: listed, no price read
            unpriced.append(_need(r, "seller", where))
            continue
        if not isinstance(v, (int, float)) or v <= 0:
            raise MarketDataError(f"{where}: usd_per_gpu_h is {v!r}")
        s = _need(r, "seller", where)
        # one figure per seller: its cheapest listing for this GPU and term,
        # what a buyer of that seller pays
        if s not in best or v < best[s]["usd_per_gpu_h"]:
            best[s] = r
    rows = list(best.values())
    base = {"gpu": gpu, "term": term, "n_sellers": len(rows),
            "n_aggregators_excluded": aggs, "sellers_without_price": sorted(set(unpriced)) }
    if not rows:
        return {**base, "low": None, "median": None, "high": None, "sellers": []}
    xs = sorted(r["usd_per_gpu_h"] for r in rows)
    return {**base, "low": xs[0], "median": statistics.median(xs), "high": xs[-1],
            "sellers": [{k: r.get(k) for k in ("seller", "usd_per_gpu_h", "commit_months",
                                               "read", "url")}
                        for r in sorted(rows, key=lambda r: r["usd_per_gpu_h"])]}


# ---------------------------------------------------------------- workload
@dataclass(frozen=True)
class WorkloadView:
    cfg: RunConfig
    wl: M.Workload
    turn: float
    out: float
    think: float
    req_per_hour: float            # per live session, subagents included
    prompt_mean: float
    tokens_per_request: float      # E[L] + out: the mixed-token denominator

    @property
    def mixed_per_session_hour(self) -> float:
        return self.req_per_hour * self.tokens_per_request


def default_workload() -> WorkloadView:
    cfg = RunConfig()
    w, wl = cfg.workload, cfg.to_workload()
    f = A.request_flows(wl, wl.invalidation, w.warm_turn_tokens, w.max_output_tokens)
    return WorkloadView(cfg, wl, float(w.warm_turn_tokens), float(w.max_output_tokens),
                        float(w.think_time_s), A.requests_per_hour(wl, w.think_time_s),
                        f.prompt, f.prompt + f.output)


# ---------------------------------------------------------------- capacity
def _base_deployment(model_key: str, gpu: str, tp: int, arm: str) -> Deployment:
    m = M.MODELS[model_key]
    kv = "fp16" if (m.kv_fp8_blackwell_only and not M.GPUS[gpu].supports_nvfp4) else "fp8"
    return replace(RunConfig().deployment, model=model_key, gpu=gpu, tensor_parallel=tp,
                   replicas=1, weight_dtype=arm, kv_dtype=kv)


def candidate_configs(model_key: str) -> list[dict]:
    """Every (GPU, TP, weight arm) on one node the model prices, and why the
    others were left out."""
    m = M.MODELS[model_key]
    if not m.vllm_tp:
        raise MarketDataError(f"{model_key}: catalogue vllm_tp is empty; the accepted "
                              "TP widths are unknown")
    out = []
    for gpu, g in M.GPUS.items():
        arms = ["fp8"] + (["nvfp4"] if (m.nvfp4_w is not None and g.supports_nvfp4) else [])
        for arm in arms:
            for tp in (t for t in m.vllm_tp if t <= NODE):
                out.append({"gpu": gpu, "tp": tp, "arm": arm})
    return out


class _ScaledDecode(RunConfig):
    """A RunConfig whose decode step runs `factor` times slower than the
    priors: eta and the verify GEMM's MFU divided, every tau multiplied."""
    _factor: float = 1.0

    def decode_constants(self) -> M.DecodeConstants:
        base = super().decode_constants()
        t0, tar, trec = base.taus()
        k = self._factor
        return replace(base, eta=base.eta / k, tau_0=t0 * k, tau_ar=tar * k,
                       tau_rec=trec * k, mfu=(base.mfu or M.MFU_DEFAULT) / k)


def _cfg(model_key: str, gpu: str, tp: int, arm: str, decode_factor: float = 1.0,
         prefill_factor: float = 1.0) -> RunConfig:
    dep = _base_deployment(model_key, gpu, tp, arm)
    cal = Calibration(mfu=M.MFU_DEFAULT / prefill_factor)
    if decode_factor == 1.0:
        return RunConfig(deployment=dep, calibration=cal)
    c = _ScaledDecode(deployment=dep, calibration=cal)
    object.__setattr__(c, "_factor", decode_factor)
    return c


class Evaluator:
    """Predictions per configuration, cached for one run."""

    def __init__(self, wv: WorkloadView, n_iter: int = 400):
        self.wv, self.n_iter = wv, n_iter
        self._cache: dict = {}

    def limit(self, model_key, gpu, tp, arm, decode_factor=1.0, prefill_factor=1.0):
        key = (model_key, gpu, tp, arm, decode_factor, prefill_factor)
        if key not in self._cache:
            cfg = _cfg(model_key, gpu, tp, arm, decode_factor, prefill_factor)
            m, t = cfg.to_model(), cfg.to_topology()
            if M.kv_pool_tokens(m, t) <= 0:
                self._cache[key] = {"skip": "the weights leave no KV pool"}
            else:
                p = predict(cfg, n_iter=self.n_iter)
                self._cache[key] = {"pred": p, "cfg": cfg}
        return self._cache[key]

    def decode_speeds(self, ev: dict) -> dict:
        """Per-user decode tok/s the model predicts on this configuration:
        one sequence alone (p50 over the workload's contexts), and the
        steady decode batch at the predicted limit's load."""
        p, cfg = ev["pred"], ev["cfg"]
        m, t, wl, w = cfg.to_model(), cfg.to_topology(), cfg.to_workload(), cfg.workload
        dc = cfg.decode_constants()
        _, p50, _, _ = M.decode_curves(m, t, wl, [1], n_iter=self.n_iter, decode=dc)
        users = p.predicted_limit_users
        rate = M.request_rate(users, w.think_time_s, wl.sub_ratio)
        # the resident cap predict.py passes: a draw sized to the pool
        draw = int(4000 + M.kv_pool_tokens(m, t) / 8000)
        resident = float(M.warm_capacity(m, t, wl, ram_gib=0, n_iter=self.n_iter,
                                         draw=draw, seed=0, which="gpu")[2])
        sp = M.steady_decode_point(m, t, M.workload_at(wl, users, p.warm_capacity_p5), rate,
                                   out_tokens=w.max_output_tokens, n_iter=self.n_iter,
                                   decode=dc, resident=resident)
        return {"n1_tok_s": float(p50[0]),
                "at_limit_tok_s": None if sp["saturated"] else float(sp["per_user_tok_s"]),
                "at_limit_batch": None if sp["saturated"] else float(sp["n"])}

    def duties(self, ev: dict, u: float) -> dict:
        """Prefill and decode duty of a group at u x its limit, the model's
        own split of GPU time (model.power_draw), and the fresh tokens per
        request at that load's miss share."""
        p, cfg = ev["pred"], ev["cfg"]
        wv, w = self.wv, cfg.workload
        m, t = cfg.to_model(), cfg.to_topology()
        users = u * p.predicted_limit_users
        rate = M.request_rate(users, w.think_time_s, cfg.to_workload().sub_ratio)
        wl_u = M.workload_at(cfg.to_workload(), users, p.warm_capacity_p5)
        pd = M.power_draw(m, t, wl_u, rate, p.decode_ceiling_users,
                          cfg.deployment.max_num_batched_tokens,
                          turn_tokens=w.warm_turn_tokens, mfu=cfg.calibration.mfu,
                          out_tokens=w.max_output_tokens, per_pass_overhead=True,
                          decode_floor=cfg.slo.itl_floor_tok_s)
        fl = A.request_flows(wl_u, wl_u.invalidation, wv.turn, wv.out)
        return {"u": u, "d_prefill": pd["d_p"], "d_decode": pd["d_d"],
                "miss": wl_u.invalidation, "fresh_per_request": fl.written,
                "cached_per_request": fl.read}


def split_cost(usd_per_hour: float, limit: float, u: float, d: dict,
               req_per_hour: float, out: float) -> dict:
    """USD per million fresh input and output tokens at utilisation u:
    prefill time to input, decode time to output, idle in proportion."""
    busy = d["d_prefill"] + d["d_decode"]
    if busy <= 0:
        return {"usd_per_mtok_input": None, "usd_per_mtok_output": None}
    idle = max(0.0, 1.0 - busy)
    c_in = usd_per_hour * (d["d_prefill"] + idle * d["d_prefill"] / busy)
    c_out = usd_per_hour * (d["d_decode"] + idle * d["d_decode"] / busy)
    reqs = limit * u * req_per_hour
    fresh, outs = reqs * d["fresh_per_request"], reqs * out
    return {"usd_per_mtok_input": c_in / fresh * 1e6 if fresh > 0 else None,
            "usd_per_mtok_output": c_out / outs * 1e6 if outs > 0 else None}


# ---------------------------------------------------------------- accuracy
def _accuracy_rows() -> list:
    """The independent rows of scripts/accuracy.py (a checkout only)."""
    import importlib.util
    path = Path(__file__).resolve().parents[2] / "scripts" / "accuracy.py"
    if not path.exists():
        raise MarketDataError(f"{path} not found: the error band needs a checkout")
    spec = importlib.util.spec_from_file_location("_ws_accuracy", path)
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = mod          # dataclasses look the module up
    spec.loader.exec_module(mod)
    return mod.check_rows()


_ROWS = None


ERROR_QUANTITIES = ("Decode speed per user", "Prefill, idle cold TTFT")


def accuracy_rows(model_key: str, quantity: str) -> list[dict]:
    """The independent accuracy rows (docs/accuracy.md) of this model for
    one quantity; empty when it has none. Never another model's rows."""
    global _ROWS
    if _ROWS is None:
        _ROWS = _accuracy_rows()
    return [{"conditions": r.conditions, "measured": r.measured, "predicted": r.predicted,
             "error": r.error, "source": r.source}
            for r in _ROWS if r.quantity == quantity and getattr(r, "model", "") == model_key]


def error_band(ev: Evaluator, model_key: str, cfg: dict, limit: int, binding: str) -> dict:
    """The limit re-predicted under each measured error of this model: for
    a decode row, every decode step (1 + e) x longer; for a prefill row,
    MFU x (1 + e) (time predicted (1 + e) x the measured). Errors are
    (predicted - measured) / measured; positive means the model is fast.

    `binding_status` says whether the quantity the binding ceiling rests on
    has a row for this model. The band spans every limit re-predicted from
    every quantity that has one, so a check on a non-binding quantity still
    shows whether it could take over; with no row at all it is unchecked."""
    q_bind = BINDING_QUANTITY.get(binding)
    if q_bind is None:
        raise MarketDataError(f"binding {binding!r} has no accuracy quantity")
    checked, lims = {}, []
    for q in ERROR_QUANTITIES:
        rows = accuracy_rows(model_key, q)
        if not rows:
            continue
        errs = sorted({min(r["error"] for r in rows), max(r["error"] for r in rows)})
        per = []
        for e in errs:
            if q.startswith("Decode"):
                r = ev.limit(model_key, cfg["gpu"], cfg["tp"], cfg["arm"], decode_factor=1.0 + e)
            else:
                r = ev.limit(model_key, cfg["gpu"], cfg["tp"], cfg["arm"],
                             prefill_factor=1.0 / (1.0 + e))
            lim = r["pred"].predicted_limit_users * r["pred"].replicas if "pred" in r else 0
            per.append({"error": e, "limit": lim,
                        "binding": r["pred"].binding_constraint if "pred" in r else None})
            lims.append(lim)
        checked[q] = {"rows": rows, "limits": per}
    return {"binding_quantity": q_bind,
            "binding_status": "checked" if q_bind in checked else "unchecked",
            "status": "checked" if checked else "unchecked",
            "quantities": checked, "limit": limit,
            "limit_low": min(lims) if lims else None,
            "limit_high": max(lims) if lims else None}


# ---------------------------------------------------------------- providers
def _arms_for(model_key: str, declared: str | None) -> tuple[list[str], str]:
    """The priced weight arms an endpoint's declared quantization maps to,
    and the note that says how."""
    m = M.MODELS[model_key]
    priced = ["fp8"] + (["nvfp4"] if m.nvfp4_w is not None else [])
    q = (declared or "unknown").lower()
    if q == "fp8":
        return ["fp8"], "declared fp8: base arm"
    if q in ("fp4", "mxfp4", "nvfp4"):
        if model_key in NATIVE_FP4_BASE and q != "nvfp4":
            return ["fp8"], f"declared {q}: the base arm is the native MXFP4-expert checkpoint"
        if "nvfp4" in priced:
            return ["nvfp4"], f"declared {q}: priced as the NVFP4 arm"
        if model_key in NATIVE_FP4_BASE:
            return ["fp8"], f"declared {q}: no NVFP4 arm; base arm (native MXFP4 experts)"
        return priced, f"declared {q}: no 4-bit arm priced; every priced arm"
    if q in ("bf16", "fp16", "fp32"):
        return priced, (f"declared {q}: heavier than any priced arm, so the cost "
                        "below is a lower bound")
    return priced, f"declared {q}: every priced arm ({', '.join(priced)})"


def _offer(ep: dict, where: str, ttl_s: float) -> tuple[A.ApiOffer | None, str]:
    """An api_cost offer from one endpoint's price sheet, or None with why."""
    inp = _need(ep, "usd_per_mtok_input", where)
    outp = _need(ep, "usd_per_mtok_output", where)
    cr = _need(ep, "usd_per_mtok_cache_read", where)
    cw = _need(ep, "usd_per_mtok_cache_write", where)
    if inp is None or outp is None:
        return None, "input or output price not published"
    read = inp if cr is None else cr
    write = inp if cw is None else cw
    note = ("cached tokens at the published cache-read price" if cr is not None
            else "no cache-read price: cached tokens billed as input")
    if cw is not None:
        note += "; cache writes at the published write price"
    band = A.PriceBand(0.0, float(inp), float(outp), (A.CacheOption(ttl_s, float(read),
                                                                     float(write)),))
    return A.ApiOffer(where, where, "", "", (band,)), note


def _flags(ep: dict) -> list[str]:
    out = []
    op = ep.get("other_pricing") or {}
    if op.get("discount"):
        out.append(f"discount {op['discount']:g} published; prices read as listed")
    if op.get("overrides"):
        out.append("time-of-day prices published; priced at the listed base")
    st = ep.get("status")
    if st not in (None, 0):
        out.append(f"status {st}")
    return out


def provider_rows(snapshot: dict, model_key: str, wv: WorkloadView,
                  ttl_s: float) -> list[dict]:
    rows = []
    for src in snapshot["prices"]:
        for mrow in src["models"]:
            if _need(mrow, "ws_model", src["file"]) != model_key:
                continue
            for i, ep in enumerate(_need(mrow, "endpoints", src["file"])):
                where = f"{src['file']} {model_key} endpoints[{i}]"
                offer, cache_note = _offer(ep, where, ttl_s)
                row = {"source": src["source"].get("id", src["file"]), "file": src["file"],
                       "provider": _need(ep, "provider", where), "tag": ep.get("tag"),
                       "usd_per_mtok_input": ep["usd_per_mtok_input"],
                       "usd_per_mtok_output": ep["usd_per_mtok_output"],
                       "usd_per_mtok_cache_read": ep["usd_per_mtok_cache_read"],
                       "usd_per_mtok_cache_write": ep["usd_per_mtok_cache_write"],
                       "cache_assumption": cache_note, "flags": _flags(ep),
                       "quantization": _need(ep, "quantization", where),
                       "context_length": _need(ep, "context_length", where),
                       "max_completion_tokens": ep.get("max_completion_tokens"),
                       "throughput_tok_s": ep.get("throughput_tok_s"),
                       "throughput_source": src["source"].get("id", src["file"])}
                if offer is None:
                    row["usd_per_session_hour"] = None
                else:
                    rc = A.request_cost(offer, wv.wl, wv.turn, wv.out)
                    fl = A.request_flows(wv.wl, rc.miss, wv.turn, wv.out)
                    row["usd_per_session_hour"] = rc.usd * wv.req_per_hour
                    row["usd_per_mtok_mixed"] = (row["usd_per_session_hour"]
                                                 / wv.mixed_per_session_hour * 1e6)
                    row["cache_ttl_s"] = ttl_s
                    row["miss"] = rc.miss
                    row["billed_per_request"] = {"cached": fl.read, "fresh": fl.written,
                                                 "output": fl.output}
                ctx = row["context_length"]
                cap = wv.wl.cap
                if not ctx:
                    row["fits_context"] = None
                    row["share_requests_over_context"] = None
                else:
                    row["fits_context"] = ctx >= cap
                    row["share_requests_over_context"] = _share_over(wv.wl, ctx)
                rows.append(row)
    return rows


def _share_over(wl: M.Workload, ctx: float) -> float:
    """Share of requests whose prompt is above `ctx` tokens."""
    s = wl.sub_ratio
    sub_prefix = wl.sys_user if wl.sub_shares_prefix else wl.sys_sub
    pm, _ = A.clipped_partial(wl.user_median, wl.user_sigma, min(wl.sys_user, wl.cap),
                              wl.cap, ctx, math.inf)
    ps, _ = A.clipped_partial(wl.sub_median, wl.sub_sigma, min(sub_prefix, wl.cap),
                              wl.cap, ctx, math.inf)
    return (pm + s * ps) / (1.0 + s)


# ---------------------------------------------------------------- the comparison
def compare(model_key: str, snapshot: dict, *, term: str = "on-demand",
            gpus: tuple | None = None, utilisations=UTILISATIONS,
            ttl_s: float = CACHE_TTL_DEFAULT, n_iter: int = 400,
            evaluator: Evaluator | None = None) -> dict:
    if model_key not in M.MODELS:
        raise KeyError(f"unknown model {model_key!r}; known: {sorted(M.MODELS)}")
    if term not in TERMS:
        raise ValueError(f"term must be one of {TERMS}, got {term!r}")
    wv = evaluator.wv if evaluator else default_workload()
    ev = evaluator or Evaluator(wv, n_iter)
    m = M.MODELS[model_key]
    rental = snapshot["rental"]
    gpus = tuple(gpus) if gpus else tuple(M.GPUS)
    rents = {g: {t: rent_levels(rental, g, t) for t in TERMS} for g in gpus}

    configs, skipped = [], []
    for c in candidate_configs(model_key):
        if c["gpu"] not in gpus:
            continue
        r = ev.limit(model_key, c["gpu"], c["tp"], c["arm"])
        cid = f"{c['gpu']}|TP{c['tp']}|{c['arm']}"
        if "skip" in r:
            skipped.append({**c, "id": cid, "why": r["skip"]})
            continue
        p = r["pred"]
        limit = p.predicted_limit_users * p.replicas
        n_gpu = c["tp"]
        row = {**c, "id": cid, "dp": 1, "n_gpu": n_gpu,
               "kv_dtype": r["cfg"].deployment.kv_dtype, "limit": limit,
               "binding": p.binding_constraint, "warm_capacity_p5": p.warm_capacity_p5,
               "decode_ceiling": p.decode_ceiling_users,
               "decode_censored": p.decode_ceiling_censored}
        if limit < 1:
            row["viable"] = False
            skipped.append({**c, "id": cid, "why": f"limit {limit} ({p.binding_constraint})"})
            continue
        row["viable"] = True
        row["duties"] = [ev.duties(r, u) for u in U_GRID]
        row["error"] = error_band(ev, model_key, c, limit, p.binding_constraint)
        row["decode"] = ev.decode_speeds(r)
        row["cost"] = {}
        for t in TERMS:
            lv = rents[c["gpu"]][t]
            row["cost"][t] = {}
            for level in LEVELS:
                rent = lv[level]
                if rent is None:
                    row["cost"][t][level] = None
                    continue
                c_h = n_gpu * rent
                at = {}
                for u in utilisations:
                    d = next(x for x in row["duties"] if abs(x["u"] - u) < 1e-9) \
                        if u in U_GRID else ev.duties(r, u)
                    at[f"{u:g}"] = {
                        "usd_per_session_hour": c_h / (limit * u),
                        "usd_per_mtok_mixed": c_h / (limit * u) / wv.mixed_per_session_hour * 1e6,
                        **split_cost(c_h, limit, u, d, wv.req_per_hour, wv.out)}
                row["cost"][t][level] = {"usd_per_gpu_h": rent, "usd_per_hour": c_h,
                                         "usd_per_session_hour_full": c_h / limit, "at": at}
        configs.append(row)

    def best(arms, t, level):
        cands = [c for c in configs if c["arm"] in arms and c["cost"][t][level] is not None]
        if not cands:
            return None
        return min(cands, key=lambda c: c["cost"][t][level]["usd_per_session_hour_full"])

    providers = provider_rows(snapshot, model_key, wv, ttl_s)
    for p in providers:
        arms, note = _arms_for(model_key, p["quantization"])
        p["arms"], p["arm_note"] = arms, note
        p["fidelity"] = {
            "context_offered": p["context_length"],
            "context_ratio": (p["context_length"] / m.max_ctx) if p["context_length"] else None,
            "model_max_context": m.max_ctx,
            "max_output": p["max_completion_tokens"],
            "quantization_declared": p["quantization"],
            "arms_priced": [ARM_LABEL[a] for a in (["fp8"] + (["nvfp4"] if m.nvfp4_w else []))]}
        p["u_star"] = {}
        rev = p.get("usd_per_session_hour")
        for t in TERMS:
            band = {}
            for level in LEVELS:
                # the cheapest configuration over the arms this endpoint maps to
                per_arm = []
                for a in arms:
                    b = best([a], t, level)
                    if b is not None:
                        per_arm.append(b)
                if rev is None or not per_arm:
                    band[level] = None
                    continue
                vals = [(b["cost"][t][level]["usd_per_session_hour_full"] / rev, b["id"])
                        for b in per_arm]
                band[level] = {"u_star": min(vals)[0], "config": min(vals)[1],
                               "u_star_max_over_arms": max(vals)[0]}
            p["u_star"][t] = band
            # model error on the best config at median rent, shown apart
            med = band.get("median")
            if med:
                cfg = next(c for c in configs if c["id"] == med["config"])
                e = cfg["error"]
                if e["status"] == "checked" and e["limit_low"]:
                    c_full = cfg["cost"][t]["median"]["usd_per_hour"]
                    band["model_error"] = {
                        "u_star_at_limit_low": c_full / e["limit_low"] / rev,
                        "u_star_at_limit_high": c_full / e["limit_high"] / rev}
                else:
                    band["model_error"] = {"status": "unchecked"}

    # throughput: an endpoint's published tok/s beside the model's per-user
    # decode range over the configurations its declared weights map to
    for p in providers:
        cs = [c for c in configs if c["arm"] in p["arms"]]
        speeds = [c["decode"]["n1_tok_s"] for c in cs] + \
                 [c["decode"]["at_limit_tok_s"] for c in cs
                  if c["decode"]["at_limit_tok_s"] is not None]
        p["throughput"] = {"published_tok_s": p["throughput_tok_s"],
                           "published_by": p["throughput_source"] if p["throughput_tok_s"]
                           else None,
                           "model_min_tok_s": min(speeds) if speeds else None,
                           "model_max_tok_s": max(speeds) if speeds else None}

    chars = None
    if snapshot["chars"] is not None:
        row = (snapshot["chars"].get("models") or {}).get(model_key)
        if row is not None:
            cpt = _need(row, "chars_per_token", f"{snapshot['chars_file']} {model_key}")
            chars = {"tokenizer_repo": row.get("tokenizer_repo"), "chars_per_token": cpt}
            # USD per million characters = USD per million tokens / chars per token
            for p in providers:
                mix = p.get("usd_per_mtok_mixed")
                p["usd_per_mchar_mixed"] = (None if mix is None else
                                            {k: mix / v for k, v in cpt.items()})
            for c in configs:
                for t in TERMS:
                    for lv in LEVELS:
                        cc = c["cost"][t][lv]
                        if cc is None:
                            continue
                        for a in cc["at"].values():
                            a["usd_per_mchar_mixed"] = {k: a["usd_per_mtok_mixed"] / v
                                                        for k, v in cpt.items()}

    return {
        "schema": SCHEMA, "model": model_key, "model_name": m.name,
        "snapshot": {"dir": snapshot["dir"], "rental_file": snapshot["rental_file"],
                     "rental_fx": rental.get("fx"),
                     "price_sources": [{"file": s["file"], **{k: s["source"].get(k)
                                       for k in ("id", "url")},
                                       "fetched": s["source"].get("fetched") or s["source"].get("read")}
                                       for s in snapshot["prices"]],
                     "chars_file": snapshot["chars_file"]},
        "assumptions": assumptions(wv, term, ttl_s),
        "workload": workload_block(wv),
        "rent": rents,
        "term": term,
        "utilisations": list(utilisations),
        "configs": configs, "skipped_configs": skipped,
        "best": {t: {a: {lv: (best([a], t, lv) or {}).get("id") for lv in LEVELS}
                     for a in sorted({c["arm"] for c in configs})} for t in TERMS},
        "providers": providers,
        "chars_per_token": chars,
    }


def workload_block(wv: WorkloadView) -> dict:
    f = A.request_flows(wv.wl, wv.wl.invalidation, wv.turn, wv.out)
    tot = wv.tokens_per_request
    return {"prompt_mean": wv.prompt_mean, "output": wv.out, "warm_turn": wv.turn,
            "think_time_s": wv.think, "sub_ratio": wv.wl.sub_ratio, "cap": wv.wl.cap,
            "miss_rate": wv.wl.invalidation,
            "requests_per_session_hour": wv.req_per_hour,
            "tokens_per_request": tot,
            "mixed_tokens_per_session_hour": wv.mixed_per_session_hour,
            "mix_at_miss_rate": {"fresh": f.written / tot, "cached": f.read / tot,
                                 "output": f.output / tot}}


def assumptions(wv: WorkloadView, term: str, ttl_s: float) -> list[str]:
    return [
        "Workload: RunConfig()'s [workload], the explorer's opening one.",
        "Rent is paid every hour, 24/7; u is the time-average of live sessions over "
        "the predicted limit across every rented hour.",
        f"Rent: low / median / high over sellers (aggregators left out), {term}.",
        "Capacity: one 8-GPU node, every TP width vLLM accepts (catalogue vllm_tp); "
        "DP replicas multiply sessions and GPUs alike, so only DP1 is priced.",
        "Limit: predicted_limit_users, the verdict's binding ceiling, at the default SLO.",
        f"Provider cache: a {ttl_s:g} s TTL (none is published); cached tokens at the "
        "cache-read price where published, else at the input price.",
        "Prices read as listed; a published discount or time-of-day override is "
        "flagged, not applied.",
        "u* = (rent per hour / limit) / provider price per live session-hour. "
        "u* > 1: the price does not cover the rent at the model's predicted limit "
        "even fully loaded, under these assumptions. The provider's hardware, "
        "engine, contract prices and workload are unknown.",
        "Per-token costs split GPU time by the model's duties (prefill to fresh "
        "input, decode to output, idle in proportion); a convention u* does not use.",
        "Model error is shown apart from the rent band, never multiplied into it.",
    ]


# ---------------------------------------------------------------- report
def _usd(x, d=2) -> str:
    if x is None or not math.isfinite(x):
        return "—"
    return f"${x:,.{d}f}"


def _pct(x) -> str:
    return "—" if x is None else f"{x * 100:.0f}%"


def render(res: dict) -> str:
    wv = res["workload"]
    t = res["term"]
    out = [f"{res['model_name']} ({res['model']}): GPU rent against provider prices, USD"]
    out.append("")
    out.append("assumptions")
    out += [f"  - {a}" for a in res["assumptions"]]
    mix = wv["mix_at_miss_rate"]
    out.append(f"  - per live session-hour: {wv['requests_per_session_hour']:.0f} requests, "
               f"{wv['tokens_per_request']:,.0f} tokens each (prompt {wv['prompt_mean']:,.0f}, "
               f"output {wv['output']:.0f}); mix {mix['fresh']:.1%} fresh input, "
               f"{mix['cached']:.1%} cached input, {mix['output']:.1%} output "
               f"at the workload's {wv['miss_rate']:.0%} miss rate")
    for s in res["snapshot"]["price_sources"]:
        out.append(f"  - prices: {s['file']} ({s['id']}, fetched {s['fetched']})")
    fx = res["snapshot"]["rental_fx"] or {}
    out.append(f"  - rent: {res['snapshot']['rental_file']}"
               + (f"; EUR listings converted at {fx.get('usd_per_eur')} USD/EUR "
                  f"(ECB {fx.get('date')})" if fx.get("usd_per_eur") else ""))
    for g, terms in res["rent"].items():
        for term, lv in terms.items():
            if lv["n_sellers"]:
                out.append(f"    {g} {term}: ${lv['low']:.2f} / ${lv['median']:.2f} / "
                           f"${lv['high']:.2f} per GPU-hour over {lv['n_sellers']} sellers")
            else:
                out.append(f"    {g} {term}: no seller record")
    if res["chars_per_token"] is None:
        out.append("  - per-character columns omitted: no chars_per_token.json for this model")
    out.append("")
    out.append(f"model cost per million mixed tokens, median rent, {t}; "
               "brackets: low-high rent")
    out.append(f"  {'config':22} {'limit':>6} {'binds':10} {'error':10} "
               + " ".join(f"{'u=' + format(u, 'g'):>18}" for u in (0.2, 0.5, 0.8)))
    for c in sorted(res["configs"], key=lambda c: (c["cost"][t]["median"] or {}).get(
            "usd_per_session_hour_full", math.inf)):
        cm = c["cost"][t]
        cells = []
        for u in (0.2, 0.5, 0.8):
            k = f"{u:g}"
            if cm["median"] is None or k not in cm["median"]["at"]:
                cells.append(f"{'—':>18}")
                continue
            lo = cm["low"]["at"][k]["usd_per_mtok_mixed"]
            hi = cm["high"]["at"][k]["usd_per_mtok_mixed"]
            cells.append(f"{_usd(cm['median']['at'][k]['usd_per_mtok_mixed'])} "
                         f"[{lo:.2f}-{hi:.2f}]".rjust(18))
        e = c["error"]
        et = ("unchecked" if e["status"] == "unchecked" else
              f"{e['limit_low']}-{e['limit_high']}"
              + ("" if e["binding_status"] == "checked" else "*"))
        out.append(f"  {c['id']:22} {c['limit']:>6} {c['binding']:10} {et:10} "
                   + " ".join(cells))
    for s in res["skipped_configs"]:
        out.append(f"  {s['id']:22} not priced: {s['why']}")
    out.append("  error: the limit re-predicted under this model's measured errors "
               "(docs/accuracy.md); * the binding quantity itself has no row; "
               "unchecked: no row for this model")
    out.append("")
    out.append(f"providers: price per live session-hour and per million mixed tokens; "
               f"u* at low / median / high rent, {t}, cheapest configuration")
    head = (f"  {'provider':26} {'in':>6} {'out':>6} {'cache':>6} {'$/s-h':>7} {'$/Mmix':>7} "
            f"{'u* low':>7} {'median':>7} {'high':>7} {'err':>11} {'ctx':>6} {'maxout':>7} {'quant':>8}")
    out.append(head)
    for p in sorted(res["providers"], key=lambda p: p.get("usd_per_session_hour") or math.inf):
        b = p["u_star"][t]
        us = [(b.get(lv) or {}).get("u_star") for lv in LEVELS]
        me = b.get("model_error") or {}
        err = ("unchecked" if me.get("status") == "unchecked" else
               f"{me['u_star_at_limit_high']:.2f}-{me['u_star_at_limit_low']:.2f}"
               if me else "—")
        f = p["fidelity"]
        name = f"{p['provider']} ({p['tag']})" if p.get("tag") else p["provider"]
        out.append(f"  {name[:26]:26} {_usd(p['usd_per_mtok_input']):>6} "
                   f"{_usd(p['usd_per_mtok_output']):>6} {_usd(p['usd_per_mtok_cache_read']):>6} "
                   f"{_usd(p.get('usd_per_session_hour')):>7} {_usd(p.get('usd_per_mtok_mixed')):>7} "
                   + " ".join(f"{(format(x, '.2f') if x is not None else '—'):>7}" for x in us)
                   + f" {err:>11} {_pct(f['context_ratio']):>6} "
                   + (f"{f['max_output'] / 1e3:>6.0f}k" if f['max_output'] else f"{'—':>7}")
                   + f" {str(f['quantization_declared'] or '—')[:8]:>8}")
        notes = [f"from {p['source']}", p["arm_note"]] + p["flags"]
        if p.get("usd_per_session_hour") is None:
            notes.append("input or output price not published: no u*")
        if p.get("fits_context") is False:
            notes.append(f"context {p['context_length']:,} below the workload's cap "
                         f"{res['workload']['cap']:,.0f}: "
                         f"{p['share_requests_over_context']:.2%} of requests longer")
        if p["cache_assumption"].startswith("no cache"):
            notes.append(p["cache_assumption"])
        out.append(f"  {'':26} {'; '.join(notes)}")
    ch = res["chars_per_token"]
    if ch is not None:
        cpt = ch["chars_per_token"]
        out.append("")
        out.append("per million characters = per million tokens / characters per token "
                   f"({ch['tokenizer_repo']}): "
                   + ", ".join(f"{k} {v:.2f}" for k, v in cpt.items()))
    out.append("")
    out.append("throughput, hardware unknown (a sanity range, not a check): model per-user "
               "decode, one sequence alone and at the predicted limit, over the "
               "configurations each endpoint's weights map to")
    tp = [p for p in res["providers"] if p["throughput"]["published_tok_s"]]
    srcs = {p["source"] for p in res["providers"]}
    if not tp:
        out.append(f"  not published by the API ({', '.join(sorted(srcs))})")
    for p in tp:
        th = p["throughput"]
        out.append(f"  {p['provider'][:26]:26} published {th['published_tok_s']:.0f} tok/s "
                   f"({th['published_by']}); model {th['model_min_tok_s']:.0f}-"
                   f"{th['model_max_tok_s']:.0f}")
    return "\n".join(out)


# ---------------------------------------------------------------- CLI
def cmd_market(args) -> int:
    snap = load_snapshot(args.snapshot)
    gpus = (args.gpu,) if args.gpu else None
    utils = UTILISATIONS
    if args.utilisation:
        utils = tuple(sorted(set(UTILISATIONS) | {float(x) for x in args.utilisation}))
        for u in utils:
            if not 0 < u <= 1:
                raise SystemExit(f"--utilisation {u}: must be in (0, 1]")
    res = compare(args.model, snap, term=args.term, gpus=gpus, utilisations=utils,
                  ttl_s=args.cache_ttl, n_iter=args.n_iter)
    if args.json:
        print(json.dumps(res, indent=1, allow_nan=False, default=_json_default))
    else:
        print(render(res))
    return 0


def _json_default(x):
    if isinstance(x, float) and not math.isfinite(x):
        return None
    raise TypeError(f"not JSON serialisable: {type(x).__name__}")


def add_subparser(sub) -> None:
    p = sub.add_parser("market", help="GPU rent against provider prices: break-even utilisation",
                       description="Price MODEL's predicted capacity at market GPU rent and "
                                   "set each provider's price beside it: the utilisation a "
                                   "provider must reach to cover the rent at its price, "
                                   "under the assumptions printed first.")
    p.add_argument("model", help="model key (ws models)")
    p.add_argument("--snapshot", default=None,
                   help="directory of market files (default: data/market in the checkout)")
    p.add_argument("--json", action="store_true")
    p.add_argument("--utilisation", nargs="*", help="extra utilisations to price, in (0, 1]")
    p.add_argument("--gpu", choices=sorted(M.GPUS), help="one GPU only")
    p.add_argument("--term", choices=TERMS, default="on-demand")
    p.add_argument("--cache-ttl", type=float, default=CACHE_TTL_DEFAULT,
                   help=f"provider prompt-cache TTL in seconds (default {CACHE_TTL_DEFAULT:g}; "
                        "none is published)")
    p.add_argument("--n-iter", type=int, default=400, help="Monte-Carlo iterations per prediction")
    p.set_defaults(fn=cmd_market)
