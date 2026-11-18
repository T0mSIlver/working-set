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
from .config import Calibration, Deployment, RunConfig, WorkloadCfg
from .predict import predict

SCHEMA = "workingset.market/2"
NODE = 8                                  # one 8-GPU node: the explorer's frontier
UTILISATIONS = (0.2, 0.5, 0.8, 1.0)
CACHE_TTL_DEFAULT = 300.0                 # s; providers publish no TTL (assumption)
# A committed listing is short (under 3 months: Together's 1-week and
# 1-month prices) or long (12 months and more); a commitment of 3 to 12
# months, or of unstated length, is counted and left out.
TERMS = ("on-demand", "committed-short", "committed-long")
SHORT_COMMIT_MONTHS = 3.0
LONG_COMMIT_MONTHS = 12.0
# Which offers a rent band counts. "managed": a provider's own managed cloud
# (the default); "all" adds community-cloud and marketplace offers, whose
# hosts, availability and eight-GPU fabric are not the seller's to state.
# GB300 (Grace-Blackwell) instances are a different platform from the HGX
# B300 the catalogue prices and never count, whichever the tier set.
RENT_TIERS = {"managed": ("managed",), "all": ("managed", "community", "marketplace")}
SELLER_TIERS = ("managed", "community", "marketplace")
PLATFORMS = ("HGX", "GB300", "unstated")
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


def _record_term(r: dict, where: str) -> str | None:
    """The TERMS entry a rental record counts under, None when none."""
    t = _need(r, "term", where)
    if t != "committed":
        return t
    months = r.get("commit_months")
    if months is None:
        return None
    if months < SHORT_COMMIT_MONTHS:
        return "committed-short"
    if months >= LONG_COMMIT_MONTHS:
        return "committed-long"
    return None


def rent_levels(rental: dict, gpu: str, term: str, tiers: str = "managed") -> dict:
    """Low, median and high USD per GPU-hour over sellers for one GPU and
    term, over the offers of `tiers` (RENT_TIERS). Aggregator records,
    other tiers and GB300 instances are counted and left out; a sold-out
    offer counts and is listed."""
    if tiers not in RENT_TIERS:
        raise ValueError(f"tiers must be one of {tuple(RENT_TIERS)}, got {tiers!r}")
    recs = _need(rental, "records", "gpu_rental")
    best: dict = {}
    aggs, unpriced, other_tier, gb300 = 0, [], [], []
    for i, r in enumerate(recs):
        where = f"gpu_rental records[{i}]"
        tier, platform = _need(r, "tier", where), _need(r, "platform", where)
        if _need(r, "kind", where) == "seller":
            if tier not in SELLER_TIERS:
                raise MarketDataError(f"{where} ({r.get('seller')}): tier {tier!r} is not "
                                      f"one of {SELLER_TIERS}")
            if platform not in PLATFORMS:
                raise MarketDataError(f"{where} ({r.get('seller')}): platform {platform!r} "
                                      f"is not one of {PLATFORMS}")
        if _need(r, "gpu", where) != gpu or _record_term(r, where) != term:
            continue
        if r["kind"] != "seller":
            aggs += 1
            continue
        if platform == "GB300":
            gb300.append(r.get("seller"))
            continue
        if tier not in RENT_TIERS[tiers]:
            other_tier.append(f"{r.get('seller')} ({tier})")
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
    left_out = sorted({r.get("seller") for i, r in enumerate(recs)
                       if r.get("gpu") == gpu and r.get("term") == "committed"
                       and term.startswith("committed")
                       and _record_term(r, f"gpu_rental records[{i}]") is None})
    base = {"gpu": gpu, "term": term, "tiers": tiers, "n_sellers": len(rows),
            "n_aggregators_excluded": aggs, "sellers_without_price": sorted(set(unpriced)),
            "other_tiers_excluded": sorted(set(other_tier)),
            "gb300_excluded": sorted(set(gb300)),
            "sold_out": sorted({r.get("seller") for r in rows if r.get("availability")
                                and "sold out" in str(r.get("availability"))}),
            "commitments_left_out": left_out,
            "commit_months": sorted({r.get("commit_months") for r in rows
                                     if r.get("commit_months") is not None})}
    if not rows:
        return {**base, "low": None, "median": None, "high": None, "sellers": []}
    xs = sorted(r["usd_per_gpu_h"] for r in rows)
    return {**base, "low": xs[0], "median": statistics.median(xs), "high": xs[-1],
            "sellers": [{k: r.get(k) for k in ("seller", "usd_per_gpu_h", "commit_months",
                                               "tier", "platform", "availability",
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


def workload_view(wcfg: WorkloadCfg | None = None) -> WorkloadView:
    cfg = RunConfig() if wcfg is None else RunConfig(workload=wcfg)
    w, wl = cfg.workload, cfg.to_workload()
    f = A.request_flows(wl, wl.invalidation, w.warm_turn_tokens, w.max_output_tokens)
    return WorkloadView(cfg, wl, float(w.warm_turn_tokens), float(w.max_output_tokens),
                        float(w.think_time_s), A.requests_per_hour(wl, w.think_time_s),
                        f.prompt, f.prompt + f.output)


def default_workload() -> WorkloadView:
    return workload_view(None)


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
         prefill_factor: float = 1.0, wcfg: WorkloadCfg | None = None) -> RunConfig:
    dep = _base_deployment(model_key, gpu, tp, arm)
    cal = Calibration(mfu=M.MFU_DEFAULT / prefill_factor)
    wcfg = WorkloadCfg() if wcfg is None else wcfg
    if decode_factor == 1.0:
        return RunConfig(deployment=dep, workload=wcfg, calibration=cal)
    c = _ScaledDecode(deployment=dep, workload=wcfg, calibration=cal)
    object.__setattr__(c, "_factor", decode_factor)
    return c


class Evaluator:
    """Predictions per configuration, cached for one run."""

    def __init__(self, wv: WorkloadView, n_iter: int = 400, explicit_cap: bool = False):
        """`explicit_cap`: set max_num_seqs to the contexts of mean length the
        HBM pool holds (hbm_contexts) instead of predict()'s recommendation,
        which counts only reusable sessions and so falls to 1 when every
        request misses."""
        self.wv, self.n_iter, self.explicit_cap = wv, n_iter, explicit_cap
        self._cache: dict = {}

    def limit(self, model_key, gpu, tp, arm, decode_factor=1.0, prefill_factor=1.0,
              wcfg: WorkloadCfg | None = None):
        """predict() on one configuration, at this evaluator's workload
        unless `wcfg` names another; cached per (configuration, workload)."""
        wcfg = self.wv.cfg.workload if wcfg is None else wcfg
        key = (model_key, gpu, tp, arm, decode_factor, prefill_factor, wcfg)
        if key not in self._cache:
            cfg = _cfg(model_key, gpu, tp, arm, decode_factor, prefill_factor, wcfg)
            m, t = cfg.to_model(), cfg.to_topology()
            if M.kv_pool_tokens(m, t) <= 0:
                self._cache[key] = {"skip": "the weights leave no KV pool"}
            else:
                if self.explicit_cap:
                    cap = hbm_contexts(m, t, workload_view(wcfg).prompt_mean)
                    capped = replace(cfg, deployment=replace(cfg.deployment, max_num_seqs=cap))
                    if isinstance(cfg, _ScaledDecode):     # replace() drops the factor
                        object.__setattr__(capped, "_factor", cfg._factor)
                    cfg = capped
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


def hbm_contexts(m: M.Model, t: M.Topology, prompt_mean: float) -> int:
    """Contexts of the mean prompt length (plus each session's recurrent
    state) one group's HBM pool holds: what a batch of cold requests, none
    sharing a prefix, can decode at once."""
    per = prompt_mean + m.deltanet_state / m.kv_bpt
    return max(1, int(M.kv_pool_tokens(m, t) // per))


# The no-cache contrast: the default workload with every request a miss, so
# every prompt token is computed fresh and billed at the input price.
NO_CACHE_NOTE = ("rough sensitivity, not an estimate: every prompt computed fresh "
                 "(miss rate 100%) and billed at the input price; capacity "
                 "re-predicted with max_num_seqs set to the contexts of mean "
                 "length the HBM pool holds, because predict()'s recommendation "
                 "counts only reusable sessions and falls to 1 at 100% misses")


def no_cache_workload() -> WorkloadCfg:
    return replace(WorkloadCfg(), miss_rate=1.0)


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


def _offer(ep: dict, where: str, ttl_s: float, cache_as_input: bool = False,
           no_cache: bool = False) -> tuple[A.ApiOffer | None, str]:
    """An api_cost offer from one endpoint's price sheet, or None with why.
    `cache_as_input` bills cached tokens at the input price (no discount).
    `no_cache` (every request a miss): nothing is reused, so no cache write is
    worth paying for and every prompt token is billed at the input price."""
    inp = _need(ep, "usd_per_mtok_input", where)
    outp = _need(ep, "usd_per_mtok_output", where)
    cr = _need(ep, "usd_per_mtok_cache_read", where)
    cw = _need(ep, "usd_per_mtok_cache_write", where)
    if inp is None or outp is None:
        return None, "input or output price not published"
    read = inp if (cr is None or cache_as_input) else cr
    write = inp if (cw is None or no_cache) else cw
    note = ("cached tokens at the published cache-read price" if cr is not None
            else "no cache-read price: cached tokens billed as input")
    if no_cache:
        note = "no cache: every prompt token billed at the input price"
    elif cw is not None:
        note += "; cache writes at the published write price"
    band = A.PriceBand(0.0, float(inp), float(outp), (A.CacheOption(ttl_s, float(read),
                                                                     float(write)),))
    return A.ApiOffer(where, where, "", "", (band,)), note


def _flags(ep: dict) -> list[str]:
    out = []
    op = ep.get("other_pricing") or {}
    if op.get("discount"):
        out.append(f"listed price is after a published promotion of {op['discount'] * 100:g}%")
    if ep.get("time_weighted"):
        out.append("time-of-day prices: u* at the 24-hour time-weighted price; the "
                   "listed price is the cheapest window")
    elif op.get("overrides"):
        out.append("time-of-day prices published, no time-weighted price recorded; "
                   "priced at the listed base")
    st = ep.get("status")
    if st not in (None, 0):
        out.append(f"status {st}")
    return out


def provider_rows(snapshot: dict, model_key: str, wv: WorkloadView,
                  ttl_s: float, cache_as_input: bool = False) -> list[dict]:
    rows = []
    no_cache = wv.wl.invalidation >= 1.0
    for src in snapshot["prices"]:
        for mrow in src["models"]:
            if _need(mrow, "ws_model", src["file"]) != model_key:
                continue
            for i, ep in enumerate(_need(mrow, "endpoints", src["file"])):
                where = f"{src['file']} {model_key} endpoints[{i}]"
                tw = ep.get("time_weighted")
                listed_offer, ep_listed = None, ep
                if tw:
                    for k in ("usd_per_mtok_input", "usd_per_mtok_output",
                              "usd_per_mtok_cache_read"):
                        _need(tw, k, f"{where} time_weighted")
                    listed_offer, _ = _offer(ep, where, ttl_s, cache_as_input, no_cache)
                    ep = {**ep, **{k: tw[k] for k in ("usd_per_mtok_input",
                                                       "usd_per_mtok_output",
                                                       "usd_per_mtok_cache_read")}}
                offer, cache_note = _offer(ep, where, ttl_s, cache_as_input, no_cache)
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
                    row["listed_window_prices"] = (None if listed_offer is None else {
                        k: ep_listed[k] for k in ("usd_per_mtok_input", "usd_per_mtok_output",
                                                  "usd_per_mtok_cache_read")})
                    row["usd_per_session_hour_listed_window"] = (
                        None if listed_offer is None else
                        A.request_cost(listed_offer, wv.wl, wv.turn, wv.out).usd
                        * wv.req_per_hour)
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
            evaluator: Evaluator | None = None, rent_tiers: str = "managed",
            workload_name: str = "default") -> dict:
    if model_key not in M.MODELS:
        raise KeyError(f"unknown model {model_key!r}; known: {sorted(M.MODELS)}")
    if term not in TERMS:
        raise ValueError(f"term must be one of {TERMS}, got {term!r}")
    wv = evaluator.wv if evaluator else default_workload()
    ev = evaluator or Evaluator(wv, n_iter)
    m = M.MODELS[model_key]
    rental = snapshot["rental"]
    gpus = tuple(gpus) if gpus else tuple(M.GPUS)
    rents = {g: {t: rent_levels(rental, g, t, rent_tiers) for t in TERMS} for g in gpus}
    other = "all" if rent_tiers == "managed" else "managed"
    rents_other = {g: {t: rent_levels(rental, g, t, other) for t in TERMS} for g in gpus}

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
        # the model's split of GPU time at each priced utilisation (the
        # per-token convention reads it; the page needs only u, limit, rent)
        row["duties"] = {f"{u:g}": ev.duties(r, u) for u in utilisations}
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
                    d = row["duties"][f"{u:g}"]
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
    # the revenue side alone with no cache discount: same limit, cached
    # tokens billed at the input price
    no_disc = provider_rows(snapshot, model_key, wv, ttl_s, cache_as_input=True)
    for p, nd in zip(providers, no_disc):
        p["_rev_no_disc"] = nd["usd_per_session_hour"]
    for p in providers:
        arms, note = _arms_for(model_key, p["quantization"])
        p["arms"], p["arm_note"] = arms, note
        p["fidelity"] = {
            "context_offered": p["context_length"],
            # the share of the model's native context (NATIVE_CTX)
            "context_ratio": ((p["context_length"] / native_context(model_key)[0])
                              if p["context_length"] else None),
            "model_max_context": m.max_ctx,
            "model_native_context": native_context(model_key)[0],
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
            # revenue-side variants at the same configuration and limit
            for key, r_alt in (("u_star_no_cache_discount", p["_rev_no_disc"]),
                               ("u_star_listed_window",
                                p.get("usd_per_session_hour_listed_window"))):
                p.setdefault(key, {})[t] = None if r_alt is None else {
                    lv: (None if band[lv] is None else band[lv]["u_star"] * rev / r_alt)
                    for lv in LEVELS}
            # model error on the best config at median rent, shown apart.
            # Only a row on the BINDING quantity bounds the limit; a row on
            # another quantity says only whether that one could take over.
            med = band.get("median")
            if med:
                cfg = next(c for c in configs if c["id"] == med["config"])
                e = cfg["error"]
                if e["binding_status"] == "checked":
                    c_full = cfg["cost"][t]["median"]["usd_per_hour"]
                    band["model_error"] = {
                        "status": "checked",
                        "u_star_at_limit_low": c_full / e["limit_low"] / rev,
                        "u_star_at_limit_high": c_full / e["limit_high"] / rev}
                else:
                    band["model_error"] = {"status": "capacity unchecked"}
        p.pop("_rev_no_disc")

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
            # a configuration's per-character cost is its usd_per_mtok_mixed
            # over the same chars_per_token, left to the reader

    return {
        "schema": SCHEMA, "model": model_key, "model_name": m.name,
        "snapshot": {"dir": snapshot["dir"], "rental_file": snapshot["rental_file"],
                     "rental_fx": rental.get("fx"),
                     "price_sources": [{"file": s["file"], **{k: s["source"].get(k)
                                       for k in ("id", "url")},
                                       "fetched": s["source"].get("fetched") or s["source"].get("read")}
                                       for s in snapshot["prices"]],
                     "chars_file": snapshot["chars_file"]},
        "caveat": CAVEAT,
        "workload_name": workload_name,
        "workload_note": NO_CACHE_NOTE if workload_name == "no-cache" else None,
        "model_note": MODEL_NOTES.get(model_key),
        "n_providers": len({p["provider"] for p in providers}),
        "n_rows": len(providers),
        "assumptions": assumptions(wv, term, ttl_s, rent_tiers),
        "workload": workload_block(wv, ttl_s),
        "rent": rents, "rent_tiers": rent_tiers,
        "rent_other_tiers": {"tiers": other, "rent": rents_other},
        "term": term,
        "utilisations": list(utilisations),
        "configs": configs, "skipped_configs": skipped,
        "best": {t: {a: {lv: (best([a], t, lv) or {}).get("id") for lv in LEVELS}
                     for a in sorted({c["arm"] for c in configs})} for t in TERMS},
        "providers": providers,
        "chars_per_token": chars,
        "fidelity_summary": fidelity_summary(model_key, providers),
    }


CAVEAT = ("This is rent coverage for a hypothetical deployment at our workload, cache "
          "assumptions, scheduler settings and latency limits; it does not estimate a "
          "provider's actual utilisation, throughput capacity or profit.")

# per-model notes on what a price row names
MODEL_NOTES = {
    "Q38FN": "OpenRouter's qwen/qwen3.8-flash names the Flash-Next repository but may "
             "be the later production release",
}


def billing_miss(wv: WorkloadView, ttl_s: float) -> float:
    """The miss share the price-sheet side bills at: the workload's own
    invalidation plus expiry of a ttl_s cache over the traced gaps."""
    inv = wv.wl.invalidation
    return min(1.0, inv + (1.0 - inv) * A.expiry_miss(ttl_s))


def workload_block(wv: WorkloadView, ttl_s: float = CACHE_TTL_DEFAULT) -> dict:
    tot = wv.tokens_per_request

    def mix(q):
        f = A.request_flows(wv.wl, q, wv.turn, wv.out)
        return {"miss": q, "fresh": f.written / tot, "cached": f.read / tot,
                "output": f.output / tot}

    return {"prompt_mean": wv.prompt_mean, "output": wv.out, "warm_turn": wv.turn,
            "think_time_s": wv.think, "sub_ratio": wv.wl.sub_ratio, "cap": wv.wl.cap,
            "miss_rate": wv.wl.invalidation,
            "requests_per_session_hour": wv.req_per_hour,
            "tokens_per_request": tot,
            "mixed_tokens_per_session_hour": wv.mixed_per_session_hour,
            "mix_denominator": "every prompt token plus the output, per request",
            # capacity side: the model's own miss rate
            "mix_at_miss_rate": mix(wv.wl.invalidation),
            # billing side: the assumed cache lifetime
            "mix_billed": {**mix(billing_miss(wv, ttl_s)), "cache_ttl_s": ttl_s}}


def assumptions(wv: WorkloadView, term: str, ttl_s: float, rent_tiers: str = "managed") -> list[str]:
    tiers = ("managed-cloud offers only (community-cloud and marketplace offers "
             "left out; `--rent-tiers all` counts them)" if rent_tiers == "managed"
             else "every tier: managed, community-cloud and marketplace offers")
    return [
        "Workload: RunConfig()'s [workload], the explorer's opening one. Tool and "
        "human pauses are inside a live session, so u is not GPU busy time.",
        "Rent is paid every hour, 24/7; u is the time average of live sessions over "
        "the predicted limit across every rented hour.",
        f"Rent: low / median / high over sellers, one figure per seller (its cheapest "
        f"listing), {term}, {tiers}; aggregators and GB300 instances left out; "
        "sold-out offers kept and named. Posted offers: availability and eight-GPU "
        "fabric are not verified.",
        "Capacity: one 8-GPU node, every TP width vLLM accepts (catalogue vllm_tp); "
        "DP replicas multiply sessions and GPUs alike, so only DP1 is priced.",
        "Limit: the lowest of four ceilings at the explorer's default latency limits: "
        "slowed streams at most 15% of wall time in cold-prefill steps, p95 cold "
        "queue wait at most 5 s, prefill saturation, and decode at a 40 tok/s floor "
        "under the recommended max_num_seqs. These are the explorer's defaults, not "
        "a provider's commitments.",
        f"Provider cache: the price files carry no cache lifetime; a {ttl_s:g} s TTL is "
        "assumed. Cached tokens at the cache-read price where published, else at the "
        "input price.",
        "Prices as listed. A listed price is already after any published promotion; "
        "an endpoint with time-of-day prices is priced at its 24-hour time-weighted "
        "price, the listed window beside it.",
        "u* = (rent per hour / limit) / provider price per live session-hour. "
        "u* > 1: the price does not cover the rent at the model's predicted limit "
        "even fully loaded, under these assumptions.",
        "Per-token costs split GPU time by the model's duties (prefill to fresh "
        "input, decode to output, idle in proportion); a convention u* does not use. "
        "Cached tokens carry no prefill, but their KV holds HBM and decode reads it.",
        "Model error is shown apart from the rent band, never multiplied into it; "
        "'capacity unchecked' where the binding quantity has no measurement for this model.",
    ]


# ---------------------------------------------------------------- fidelity
# The weights each model was released in, read from the research note that
# cites the checkpoint. A mixed checkpoint is compared at its routed experts'
# dtype, the bulk of its bytes. None: the note does not state the released
# repo's dtype (it reads an FP8 serving repo beside a base repo whose dtype
# it does not record), so the comparison is not made.
RELEASED_DTYPE = {
    "27B": (None, "not established: research/model_qwen38_27b.md reads Qwen/Qwen3.8-27B-FP8; "
                  "the base repo's dtype is not recorded"),
    "35BA3B": (None, "not established: research/model_35ba3b.md reads Qwen/Qwen3.6-35B-A3B-FP8; "
                     "the base repo's dtype is not recorded"),
    "MM35": ("fp8", "research/model_mistral_medium35.md:47: \"base repo ships FP8 natively "
                    "(per-tensor E4M3)\""),
    "GLM52": ("fp8", "research/model_glm53.md:13-20: zai-org/GLM-5.3, quantization \"fp8 e4m3, "
                     "dynamic, block 128×128\""),
    "DSV4F": ("fp4", "research/model_dsv4flash.md:57: \"native mixed: routed experts FP4 ..., "
                     "attention/shared-expert FP8\""),
    "DSV4FV": ("fp4", "research/model_dsv4flash_vision.md:36-37: dtype histogram as 0731's "
                      "(FP8, I8 packed FP4 experts) plus BF16 vision"),
    "DSV41F": ("fp4", "research/model_dsv41flash.md:59: \"native mixed: routed experts MXFP4 ..., "
                      "all other projections FP8\""),
    "Q38FN": (None, "not established: research/model_qwen38flashnext.md reads the base repo's "
                    "config.json and the Qwen/Qwen3.8-Flash-Next-FP8 shards; the base repo's "
                    "dtype is not recorded"),
    "GLM53F": ("fp8", "research/model_glm53flash.md:108: \"Native FP8 checkpoint "
                      "(zai-org/GLM-5.3-Flash)\""),
    "ML4": ("fp8", "research/model_mistral_large4.md:101: \"FP8, as shipped\" "
                   "(mistralai/Mistral-Large-4-1T-A52B)"),
    "K31P": ("fp4", "research/model_kimi_k31pro.md:40: \"MXFP4 weights ... on the routed "
                    "experts only; everything else BF16\""),
}
# The native context of a model whose catalogue max_ctx is reached only
# with YaRN rope scaling, as the catalogue's own comment on max_ctx records
# it (catalog.toml has no field for it). Every other model's max_ctx is its
# native context, so it is the denominator of the context share.
NATIVE_CTX = {
    "27B": (262_144, "catalog.toml [models.27B] max_ctx: \"262,144 native; 1M via YaRN\""),
    "35BA3B": (262_144, "catalog.toml [models.35BA3B] max_ctx: \"262,144 native; 1M via YaRN\"; "
                        "research/model_35ba3b.md:41"),
    "Q38FN": (262_144, "catalog.toml [models.Q38FN] max_ctx: \"262,144 native; 1M via YaRN\"; "
                       "research/model_qwen38flashnext.md:31"),
}


def native_context(model_key: str) -> tuple[float, str]:
    """(native context, how it reads in the output)."""
    m = M.MODELS[model_key]
    if model_key in NATIVE_CTX:
        n = NATIVE_CTX[model_key][0]
        return n, f"{n:,.0f} native, {m.max_ctx:,.0f} with YaRN"
    return m.max_ctx, f"{m.max_ctx:,.0f}"


# bits per weight of each declared quantization; a value outside it is an error
PRECISION_BITS = {"fp32": 32, "bf16": 16, "fp16": 16, "fp8": 8, "int8": 8,
                  "fp6": 6, "fp4": 4, "mxfp4": 4, "nvfp4": 4, "int4": 4}
CTX_BUCKETS = ("100%", "50-99%", "25-49%", "under 25%", "not published")
OUT_BUCKETS = ("none", "under 32k", "32k-128k", "over 128k")
QUANT_BUCKETS = ("same as released", "lower precision", "higher precision",
                 "not declared", "released dtype not established")


def _ctx_bucket(ctx, model_ctx: float) -> str:
    if not ctx:
        return "not published"
    r = ctx / model_ctx
    return ("100%" if r >= 1.0 else "50-99%" if r >= 0.5 else
            "25-49%" if r >= 0.25 else "under 25%")


def _out_bucket(out) -> str:
    if not out:
        return "none"
    return ("under 32k" if out < 32_768 else "32k-128k" if out <= 131_072 else
            "over 128k")


def _quant_bucket(declared, released) -> str:
    q = (declared or "unknown").lower()
    if q == "unknown":
        return "not declared"
    if q not in PRECISION_BITS:
        raise MarketDataError(f"quantization {declared!r} has no precision in "
                              f"PRECISION_BITS ({', '.join(PRECISION_BITS)})")
    if released is None:
        return "released dtype not established"
    a, b = PRECISION_BITS[q], PRECISION_BITS[released]
    return "same as released" if a == b else "lower precision" if a < b else "higher precision"


def fidelity_summary(model_key: str, providers: list[dict]) -> dict:
    """Endpoint/tier rows and distinct provider names per bucket: context
    offered as a share of the catalogue's max_ctx, max output, and declared
    quantization against the released checkpoint's dtype."""
    released, source = RELEASED_DTYPE[model_key]
    m = M.MODELS[model_key]
    axes = {"context": CTX_BUCKETS, "max_output": OUT_BUCKETS, "quantization": QUANT_BUCKETS}
    out = {ax: {b: {"rows": 0, "providers": set()} for b in bs} for ax, bs in axes.items()}
    native, ctx_label = native_context(model_key)
    for p in providers:
        for ax, b in (("context", _ctx_bucket(p["context_length"], native)),
                      ("max_output", _out_bucket(p["max_completion_tokens"])),
                      ("quantization", _quant_bucket(p["quantization"], released))):
            out[ax][b]["rows"] += 1
            out[ax][b]["providers"].add(p["provider"])
    for ax in out:
        for b in out[ax]:
            out[ax][b]["providers"] = len(out[ax][b]["providers"])
    return {"model": model_key, "model_max_context": m.max_ctx,
            "model_native_context": native, "context_label": ctx_label,
            "released_dtype": released, "released_source": source,
            "rows": len(providers), "providers": len({p["provider"] for p in providers}),
            "sources": sorted({p["source"] for p in providers}), **out}


def fidelity_table(summaries: list[dict]) -> str:
    """One markdown row per model: rows / providers in every bucket."""
    head = (["model", "rows / providers", "released", "native context"]
            + [f"context {b}" for b in CTX_BUCKETS] + [f"max output {b}" for b in OUT_BUCKETS]
            + [f"quant. {b}" for b in QUANT_BUCKETS])
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    for f in summaries:
        cells = [f["model"], f"{f['rows']} / {f['providers']}", f["released_dtype"] or "not established",
                 f["context_label"]]
        for ax, bs in (("context", CTX_BUCKETS), ("max_output", OUT_BUCKETS),
                       ("quantization", QUANT_BUCKETS)):
            cells += [f"{f[ax][b]['rows']} / {f[ax][b]['providers']}" for b in bs]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_fidelity(f: dict) -> str:
    rel = f["released_dtype"] or "not established"
    out = [f"{M.MODELS[f['model']].name} ({f['model']}): provider fidelity, "
           f"{f['rows']} endpoint/tier rows from {f['providers']} provider names "
           f"({', '.join(f['sources'])}); rows / providers",
           f"  released weights: {rel} ({f['released_source']})",
           f"  context offered as a share of the model's native context ({f['context_label']}):"]
    for ax, label in (("context", None), ("max_output", "max output:"),
                      ("quantization", "declared quantization against the released dtype:")):
        if label:
            out.append(f"  {label}")
        out.append("    " + "; ".join(f"{b} {v['rows']} / {v['providers']}"
                                       for b, v in f[ax].items()))
    return "\n".join(out)


# ---------------------------------------------------------------- sensitivity
STEP = 0.2                                # the ±20% finite difference


def _variants(w: WorkloadCfg, ttl_s: float) -> list[dict]:
    """Every assumption u* rests on, as its variants. `kind` says which side
    a variant moves: rent (C), limit (L, re-predicted unless `scale`),
    revenue (R) or workload (L and R both)."""
    lo, hi = 1.0 - STEP, 1.0 + STEP
    out = [
        {"key": "rent", "label": "GPU rent", "kind": "rent", "values": [lo, hi], "ref": 1.0},
        {"key": "limit", "label": "predicted limit (scaled directly)", "kind": "limit",
         "values": [lo, hi], "ref": 1.0},
        {"key": "decode_step", "label": "decode step time", "kind": "decode",
         "values": [lo, hi], "ref": 1.0},
        {"key": "prefill_mfu", "label": "prefill MFU", "kind": "mfu",
         "values": [lo, hi], "ref": 1.0},
        {"key": "cache_ttl", "label": "provider cache TTL (s)", "kind": "ttl",
         "values": [60.0, 3600.0], "ref": ttl_s, "switch": True},
        {"key": "cache_as_input", "label": "cached tokens billed at the input price",
         "kind": "cache_as_input", "values": [True], "ref": False, "switch": True},
        {"key": "output_tokens", "label": "output tokens per request", "kind": "workload",
         "field": "max_output_tokens", "values": [200, 800], "ref": w.max_output_tokens},
    ]
    for key, field_, label, typ in (
            ("miss_rate", "miss_rate", "miss rate", float),
            ("warm_turn", "warm_turn_tokens", "warm turn Δ (tokens)", int),
            ("think_time", "think_time_s", "think time (s)", float),
            ("prompt_median", "user_prompt_median_tokens", "prompt median (tokens)", int)):
        x = getattr(w, field_)
        out.append({"key": key, "label": label, "kind": "workload", "field": field_,
                    "values": [typ(round(x * lo, 6)) if typ is float else typ(round(x * lo)),
                               typ(round(x * hi, 6)) if typ is float else typ(round(x * hi))],
                    "ref": x})
    return out


def sensitivity(res: dict, snapshot: dict, ev: Evaluator, *, term: str = "on-demand",
                level: str = "median", ttl_s: float = CACHE_TTL_DEFAULT) -> dict:
    """How far each assumption moves each endpoint's u* at one rent level.

    The configuration is held at the endpoint's cheapest one (`res`): a
    variant re-prices that configuration and does not re-choose it. For a
    two-sided ±20% step the elasticity is d ln u* / d ln x by central
    difference; for a named switch (cache TTL, cache price) it is the
    change ln(u*_variant / u*). Rankings use the largest |ln change| over an
    assumption's variants, the median over endpoints."""
    model_key = res["model"]
    wv = ev.wv
    w0 = wv.cfg.workload
    cfgs = {c["id"]: c for c in res["configs"]}
    base = []
    for i, p in enumerate(res["providers"]):
        b = (p["u_star"].get(term) or {}).get(level)
        if b is not None:
            base.append((i, p, b))
    if not base:
        return {"term": term, "level": level, "assumptions": [], "providers": [],
                "note": "no endpoint has a u* at this term and rent level"}
    used = {b["config"] for _, _, b in base}

    def limit_of(cid, **kw):
        c = cfgs[cid]
        r = ev.limit(model_key, c["gpu"], c["tp"], c["arm"], **kw)
        if "pred" not in r:
            return None
        lim = r["pred"].predicted_limit_users * r["pred"].replicas
        return lim if lim >= 1 else None

    out_a, per_p = [], {i: {} for i, _, _ in base}
    for a in _variants(w0, ttl_s):
        rows = []
        for x in a["values"]:
            limits = {cid: cfgs[cid]["limit"] for cid in used}
            rent_f, wv_x, ttl_x, cai = 1.0, wv, ttl_s, False
            k = a["kind"]
            if k == "rent":
                rent_f = x
            elif k == "limit":
                limits = {cid: L * x for cid, L in limits.items()}
            elif k == "decode":
                limits = {cid: limit_of(cid, decode_factor=x) for cid in used}
            elif k == "mfu":
                limits = {cid: limit_of(cid, prefill_factor=1.0 / x) for cid in used}
            elif k == "ttl":
                ttl_x = x
            elif k == "cache_as_input":
                cai = True
            elif k == "workload":
                wcfg = replace(w0, **{a["field"]: x})
                wv_x = workload_view(wcfg)
                limits = {cid: limit_of(cid, wcfg=wcfg) for cid in used}
            revs = None
            if k in ("ttl", "cache_as_input", "workload"):
                revs = provider_rows(snapshot, model_key, wv_x, ttl_x, cai)
            dl = {}
            for i, p, b in base:
                c = cfgs[b["config"]]
                C = c["cost"][term][level]["usd_per_hour"] * rent_f
                L = limits[b["config"]]
                R = p["usd_per_session_hour"] if revs is None else revs[i]["usd_per_session_hour"]
                dl[i] = (None if (L is None or not R) else
                         math.log((C / L / R) / b["u_star"]))
            rows.append({"value": x, "limit": {cid: limits[cid] for cid in sorted(used)},
                         "dln": dl})
        # per endpoint: largest |ln change|, and the elasticity of a ±20% pair
        mags, els = [], []
        for i, _, _ in base:
            ds = [r["dln"][i] for r in rows]
            if any(d is None for d in ds):
                per_p[i][a["key"]] = None
                continue
            mag = max(abs(d) for d in ds)
            per_p[i][a["key"]] = mag
            mags.append(mag)
            if not a.get("switch") and len(ds) == 2:
                lo, hi = a["values"]
                ref = a["ref"]
                els.append((ds[1] - ds[0]) / (math.log(hi / ref) - math.log(lo / ref)))
        out_a.append({"key": a["key"], "label": a["label"], "kind": a["kind"],
                      "reference": a["ref"], "values": a["values"],
                      "switch": bool(a.get("switch")),
                      "median_abs_dln": statistics.median(mags) if mags else None,
                      "median_elasticity": statistics.median(els) if els else None,
                      "limit_by_value": [r["limit"] for r in rows],
                      "endpoints_without_value": sum(1 for i, _, _ in base
                                                     if per_p[i].get(a["key"]) is None)})
    out_a.sort(key=lambda a: -(a["median_abs_dln"] or 0.0))
    provs = []
    for i, p, b in base:
        vals = {k: v for k, v in per_p[i].items() if v is not None}
        # rent and the directly scaled limit move every u* alike; the
        # endpoint's own answer is the strongest of the others
        own = {k: v for k, v in vals.items() if k not in ("rent", "limit")}
        top = max(own, key=own.get) if own else None
        provs.append({"provider": p["provider"], "tag": p.get("tag"), "source": p["source"],
                      "config": b["config"], "u_star": b["u_star"],
                      "most_sensitive_to": top, "abs_dln": own.get(top) if top else None,
                      "abs_dln_by_assumption": per_p[i]})
    return {"term": term, "level": level,
            "held": "each endpoint's cheapest configuration at this rent level",
            "assumptions": out_a, "providers": provs}


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
    out.append(res["caveat"])
    if res.get("workload_note"):
        out.append(f"WORKLOAD {res['workload_name']}: {res['workload_note']}")
    out.append("")
    out.append("assumptions")
    out += [f"  - {a}" for a in res["assumptions"]]
    mix, bil = wv["mix_at_miss_rate"], wv["mix_billed"]
    out.append(f"  - per live session-hour: {wv['requests_per_session_hour']:.0f} requests, "
               f"{wv['tokens_per_request']:,.0f} tokens each (prompt {wv['prompt_mean']:,.0f}, "
               f"output {wv['output']:.0f})")
    out.append(f"  - mix, as shares of every prompt token plus the output: capacity side, at "
               f"the model's {mix['miss']:.0%} miss rate, {mix['fresh']:.1%} fresh input, "
               f"{mix['cached']:.1%} cached input, {mix['output']:.1%} output; billing "
               f"side, at the assumed {bil['cache_ttl_s']:g} s lifetime, {bil['miss']:.2%} "
               f"misses, {bil['fresh']:.1%} fresh, {bil['cached']:.1%} cached, "
               f"{bil['output']:.1%} output")
    for s in res["snapshot"]["price_sources"]:
        out.append(f"  - prices: {s['file']} ({s['id']}, fetched {s['fetched']})")
    fx = res["snapshot"]["rental_fx"] or {}
    out.append(f"  - rent: {res['snapshot']['rental_file']}"
               + (f"; EUR listings converted at {fx.get('usd_per_eur')} USD/EUR "
                  f"(ECB {fx.get('date')})" if fx.get("usd_per_eur") else ""))
    oth = res["rent_other_tiers"]

    def band(lv):
        if not lv["n_sellers"]:
            return "no seller record"
        return (f"${lv['low']:.2f} / ${lv['median']:.2f} / ${lv['high']:.2f} over "
                f"{lv['n_sellers']} seller{'s' if lv['n_sellers'] != 1 else ''}")

    out.append(f"    per GPU-hour, low / median / high: {res['rent_tiers']} offers "
               f"(priced) | {oth['tiers']} offers (beside)")
    for g, terms in res["rent"].items():
        for term, lv in terms.items():
            months = (f"; commitments of {', '.join(f'{m:g}' for m in lv['commit_months'])}"
                      " months" if lv["commit_months"] else "")
            extra = "".join([f"; sold out: {', '.join(lv['sold_out'])}" if lv["sold_out"] else "",
                             f"; GB300 left out: {', '.join(lv['gb300_excluded'])}"
                             if lv["gb300_excluded"] else ""])
            out.append(f"    {g} {term}: {band(lv)} | {band(oth['rent'][g][term])}"
                       + months + extra)
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
        et = ("unchecked" if e["binding_status"] == "unchecked" else
              f"{e['limit_low']}-{e['limit_high']}")
        out.append(f"  {c['id']:22} {c['limit']:>6} {c['binding']:10} {et:10} "
                   + " ".join(cells))
    for s in res["skipped_configs"]:
        out.append(f"  {s['id']:22} not priced: {s['why']}")
    out.append("  error: the limit re-predicted under this model's measured errors on the "
               "binding quantity (docs/accuracy.md); unchecked: no measurement of the "
               "binding quantity for this model (capacity unchecked)")
    out.append("")
    out.append(f"providers: {res['n_rows']} endpoint/tier rows from {res['n_providers']} "
               "provider names (regions, speed tiers, list and current prices of one "
               "provider are separate rows, not independent sellers)"
               + ("; a single provider" if res["n_providers"] == 1 else ""))
    if res.get("model_note"):
        out.append(f"  note: {res['model_note']}")
    out.append(f"  price per live session-hour and per million mixed tokens; u* at low / "
               f"median / high rent, {t}, cheapest configuration; err: u* at the "
               "measured-error limits, or capacity unchecked")
    head = (f"  {'provider':26} {'in':>6} {'out':>6} {'cache':>6} {'$/s-h':>7} {'$/Mmix':>7} "
            f"{'u* low':>7} {'median':>7} {'high':>7} {'err':>11} {'ctx':>6} {'maxout':>7} {'quant':>8}")
    out.append(head)
    for p in sorted(res["providers"], key=lambda p: p.get("usd_per_session_hour") or math.inf):
        b = p["u_star"][t]
        us = [(b.get(lv) or {}).get("u_star") for lv in LEVELS]
        me = b.get("model_error") or {}
        err = (f"{me['u_star_at_limit_high']:.2f}-{me['u_star_at_limit_low']:.2f}"
               if me.get("status") == "checked" else
               "unchecked" if me else "—")
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
        lw = (p.get("u_star_listed_window") or {}).get(t)
        if lw and lw.get("median") is not None:
            notes.append(f"u* at the listed window's price {lw['median']:.2f} (median rent)")
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


def render_sensitivity(sens: dict) -> str:
    held = sens.get("held", "nothing")
    out = [f"sensitivity of u* ({sens['term']}, {sens['level']} rent; {held} held)",
           "  ranked by the median over endpoints of the largest |ln change in u*|; "
           "elasticity = d ln u* / d ln x over ±20%"]
    if not sens["assumptions"]:
        return "\n".join(out + [f"  {sens.get('note', '')}"])
    out.append(f"  {'assumption':42} {'variants':>16} {'|ln Δu*|':>9} {'elasticity':>10}")
    for a in sens["assumptions"]:
        vals = "/".join(f"{v:g}" if not isinstance(v, bool) else "on" for v in a["values"])
        ref = a["reference"]
        if a["kind"] in ("rent", "limit", "decode", "mfu"):
            vals = "x" + "/".join(f"{v:g}" for v in a["values"])
        el = "switch" if a["switch"] else (
            "—" if a["median_elasticity"] is None else f"{a['median_elasticity']:+.2f}")
        mag = "—" if a["median_abs_dln"] is None else f"{a['median_abs_dln']:.3f}"
        lbl = a["label"] + ("" if isinstance(ref, bool) or a["kind"] in
                            ("rent", "limit", "decode", "mfu") else f" ({ref:g})")
        extra = (f"  [{a['endpoints_without_value']} endpoints: limit below 1]"
                 if a["endpoints_without_value"] else "")
        out.append(f"  {lbl[:42]:42} {vals:>16} {mag:>9} {el:>10}{extra}")
    out.append("")
    out.append("  per endpoint, the assumption its u* moves most with (rent and the "
               "limit itself left aside)")
    for p in sorted(sens["providers"], key=lambda p: p["u_star"]):
        name = f"{p['provider']} ({p['tag']})" if p.get("tag") else p["provider"]
        top = p["most_sensitive_to"] or "—"
        mag = "" if p["abs_dln"] is None else f" |ln Δu*| {p['abs_dln']:.2f}"
        out.append(f"  {name[:34]:34} u* {p['u_star']:.2f}  {top}{mag}")
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
    if args.fidelity:
        rows = provider_rows(snap, args.model, default_workload(), args.cache_ttl)
        if args.price_source:
            rows = [r for r in rows if r["source"] == args.price_source]
        f = fidelity_summary(args.model, rows)
        print(json.dumps(f, indent=1) if args.json else render_fidelity(f))
        return 0
    if args.sensitivity and args.workload != "default":
        raise SystemExit("--sensitivity ranks the assumptions of the default workload only")
    ev = evaluator_for(args.workload, args.n_iter)
    res = compare(args.model, snap, term=args.term, gpus=gpus, utilisations=utils,
                  ttl_s=args.cache_ttl, evaluator=ev, rent_tiers=args.rent_tiers,
                  workload_name=args.workload)
    if args.sensitivity:
        res["sensitivity"] = sensitivity(res, snap, ev, term=args.term, ttl_s=args.cache_ttl)
    if args.json:
        print(json.dumps(res, indent=1, allow_nan=False, default=_json_default))
    else:
        print(render(res))
        if args.sensitivity:
            print()
            print(render_sensitivity(res["sensitivity"]))
    return 0


WORKLOAD_NAMES = ("default", "no-cache")


def evaluator_for(name: str, n_iter: int = 400) -> Evaluator:
    """The evaluator of a named workload: the default, or the no-cache
    contrast (every request a miss, an explicit scheduler cap)."""
    if name == "default":
        return Evaluator(default_workload(), n_iter)
    if name == "no-cache":
        return Evaluator(workload_view(no_cache_workload()), n_iter, explicit_cap=True)
    raise ValueError(f"workload must be one of {WORKLOAD_NAMES}, got {name!r}")


def contrast_summary(res: dict) -> dict:
    """The compact no-cache contrast a page reads beside the default."""
    return {"workload_note": res["workload_note"], "workload": res["workload"],
            "configs": [{k: c[k] for k in ("id", "gpu", "tp", "arm", "n_gpu", "limit",
                                           "binding")} for c in res["configs"]],
            "providers": [{k: p.get(k) for k in ("provider", "tag", "source",
                                                 "usd_per_session_hour", "usd_per_mtok_mixed",
                                                 "u_star")} for p in res["providers"]]}


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
    p.add_argument("--term", choices=TERMS, default="on-demand",
                   help="committed-short: under 3 months; committed-long: 12 months and more")
    p.add_argument("--cache-ttl", type=float, default=CACHE_TTL_DEFAULT,
                   help=f"provider prompt-cache TTL in seconds (default {CACHE_TTL_DEFAULT:g}; "
                        "none is published)")
    p.add_argument("--n-iter", type=int, default=400, help="Monte-Carlo iterations per prediction")
    p.add_argument("--workload", choices=WORKLOAD_NAMES, default="default",
                   help="no-cache: every prompt computed fresh and billed at input, a rough "
                        "contrast (docs/market.md)")
    p.add_argument("--rent-tiers", choices=tuple(RENT_TIERS), default="managed",
                   help="offers the priced rent band counts (default managed; the other "
                        "band is printed beside)")
    p.add_argument("--fidelity", action="store_true",
                   help="only the provider fidelity counts: context, max output and declared "
                        "quantization against the released weights (no capacity run)")
    p.add_argument("--price-source", default=None,
                   help="with --fidelity: count only rows from this price source id "
                        "(provider_prices, openrouter)")
    p.add_argument("--sensitivity", action="store_true",
                   help="also rank the assumptions by how far each moves u*")
    p.set_defaults(fn=cmd_market)
