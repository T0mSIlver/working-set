"""`ws calibrate`: fit the model's constants from runs on one deployment.

    ws calibrate SWEEP.json [MORE.json ...] --config workingset.toml [--write]
    ws calibrate --vllm-log server.log --config workingset.toml [--reserve terms]

Reads `ws decode-sweep` records (and `ws replay` records, for prefill) taken
on the deployment the config names, fits what the runs identify, keeps every
other constant at its prior, and prints each with its standard error. With
`--write` it stores the fits in the config's [calibration] block, with their
provenance under [calibration.fit.<name>], and `ws predict` prices with them.

What each run identifies (research/calibrate.md has the estimators):

  eta, tau_0, tau_ar, tau_rec   decode sweep rungs, by non-negative least
      squares on the step equation (model.decode_step_seconds) with the
      verify compute pinned at the config's MFU. eta needs the step's bytes
      to vary (more sequences, or two contexts); tau_ar needs one run at
      tp = 1 and one at tp > 1, else tau_0 and tau_ar fit as one per-layer
      cost and tau_0 stays at its prior; tau_rec needs a recurrent model.
  acceptance        vLLM's spec-decode counters over each rung's window:
      per position when the per-position counter is there, else the
      geometric maximum-likelihood figure.
  reserve_gib       the KV pool vLLM reported (the sweep reads it from
      cache_config_info; `--vllm-log` reads "GPU KV cache size: N tokens"),
      turned into the per-GPU reserve that pool implies. It absorbs the
      activation reserve and any weight overhead in one read number.
  the reserve terms  `--vllm-log` only: what vLLM held back on each GPU, from
      its start-up lines (research/reserve_terms.md):
        gpu_memory_utilization, gpu_total   "Free memory on device (free/total
            GiB) on startup. Desired GPU memory utilization is (u, ...)";
            the holdback is (1 - u) x total
        activation   "... GiB for peak activation" on the same line
        non_torch    "... GiB for consumed memory (weights + non-torch)" less
            "Model loading took X GiB memory"
      They are printed with the reserve they sum to. `--write` stores
      reserve_gib (`--reserve pool`, the default) or these four
      (`--reserve terms`), never both.
  mfu               `ws replay` records' requests sent alone: time to first
      token against the model's prefill FLOPs.

A fit that a design does not identify is reported as such and not written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from . import model as M
from .config import RunConfig, load_config

GIB = 1024 ** 3
_KV_LOG = re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens")
_NUM = r"(-?[\d.]+)"
# vLLM 0.33 start-up lines (vllm/v1/worker/gpu_worker.py, gpu/model_runner.py,
# v1/core/kv_cache_utils.py at v0.33.0); research/reserve_terms.md
_LOG = {
    "version": re.compile(r"vLLM server version ([\w.+-]+)"),
    "tp": re.compile(r"tensor_parallel_size=(\d+)"),
    "weights": re.compile(r"Model loading took " + _NUM + r" GiB memory"),
    "kv_memory": re.compile(r"Available KV cache memory: " + _NUM + r" GiB"),
    "pool": re.compile(r"(GPU(?: \+ host)?) KV cache size:\s*([\d,]+)\s*tokens"),
    "concurrency": re.compile(r"Maximum concurrency for ([\d,]+) tokens per request: "
                              + _NUM + "x"),
    "free": re.compile(
        r"Free memory on device \(" + _NUM + "/" + _NUM + r" GiB\) on startup\. "
        r"Desired GPU memory utilization is \(" + _NUM + ", " + _NUM + r" GiB\)\. "
        r"Actual usage is " + _NUM + r" GiB for consumed memory \(weights \+ "
        r"non-torch\), " + _NUM + r" GiB for peak activation, and " + _NUM
        + r" GiB for CUDAGraph memory"),
    "util": re.compile(r"The current --gpu-memory-utilization=" + _NUM + " is equivalent"),
    "graph_estimate": re.compile(r"CUDA graph pool memory: " + _NUM
                                 + r" GiB \(actual\), " + _NUM + r" GiB \(estimated\)"),
    "fixed_kv": re.compile(r"reserved " + _NUM + r" GiB memory for KV Cache as "
                           r"specified by kv_cache_memory_bytes config"),
}
LINE_FREE = "Free memory on device (free/total GiB) on startup. Desired GPU memory utilization is ..."
LINE_WEIGHTS = "Model loading took X GiB memory"
LINE_POOL = "GPU KV cache size: N tokens"
# the [calibration] key each reserve term is written to
TERM_KEYS = {"gpu_memory_utilization": "gpu_memory_utilization",
             "gpu_total": "gpu_total_gib", "activation": "activation_gib",
             "non_torch": "non_torch_gib"}


@dataclass
class Fit:
    value: float
    stderr: float | None
    n: int
    status: str            # measured | not identified | prior
    note: str = ""

    def show(self, name: str, unit: str = "", scale: float = 1.0) -> str:
        v = self.value * scale
        se = "" if self.stderr is None else f" ± {self.stderr * scale:.3g}"
        tail = f" ({self.status}, {self.n} points)" if self.status == "measured" else f" ({self.status})"
        return f"  {name:<22} {v:.4g}{se}{unit}{tail}" + (f" — {self.note}" if self.note else "")


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
def _load_run(path: str) -> dict:
    rec = json.loads(Path(path).read_text(encoding="utf-8"))
    rec["_path"] = path
    rec["_sha"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]
    return rec


def _is_sweep(rec: dict) -> bool:
    return rec.get("format") == "workingset-decode-sweep"


def _is_replay(rec: dict) -> bool:
    return "speed" in rec and "calibration" in rec and "steps" in rec


def _run_tp(rec: dict, cfg: RunConfig) -> int:
    """The tensor-parallel width a sweep ran at, from its own record
    (`ws decode-sweep --tensor-parallel`). A record without it is refused:
    the config's width may not be the one the sweep measured."""
    tp = rec.get("tensor_parallel")
    if not tp:
        raise ValueError(
            f"{rec.get('_path', 'a decode-sweep record')} names no "
            "tensor_parallel: add \"tensor_parallel\": N to it (the server's "
            "--tensor-parallel-size), or rerun with --tensor-parallel")
    return int(tp)


def _rung_k(rung: dict) -> int:
    sp = (rung.get("server") or {}).get("spec_decode")
    if not sp or not sp.get("drafts"):
        return 0
    return int(round(sp["draft_tokens"] / sp["drafts"]))


def _rung_context(rec: dict, rung: dict) -> float | None:
    """Mean context of a rung's sequences over its window: the prompt plus
    what each had generated by mid-window. None when the rung measured no
    prompt tokens or no per-sequence speed."""
    prompt, rate = rung.get("prompt_tokens_mean"), rung.get("tok_s_seq_server")
    if not prompt or rate is None:
        return None
    return prompt + rate * (rec.get("settle_s", 0) + rec.get("hold_s", 0) / 2)


# ---------------------------------------------------------------------------
# decode constants
# ---------------------------------------------------------------------------
def decode_rows(runs: list[dict], cfg: RunConfig, mfu: float,
                notes: list | None = None) -> list[dict]:
    """One row per clean rung: the regressors of the step equation and the
    step time less the pinned verify compute. Every rung left out is said in
    `notes`, with why."""
    notes = [] if notes is None else notes
    base = cfg.to_model()
    rows = []
    for rec in runs:
        tp = _run_tp(rec, cfg)
        topo = M.topology_grid(1, tp, cfg.deployment.gpu, cfg.deployment.kv_sharding)
        m = M._replicated(base, topo)
        kv_bpt = m.kv_bpt if m.kv_decode_bpt is None else m.kv_decode_bpt
        name = rec.get("_path", "sweep")
        for r in rec.get("rungs", []):
            if r.get("skipped"):
                continue
            t = (r.get("server") or {}).get("step_time_s")
            if not r.get("batch_ok") or not t:
                notes.append(f"{name} n={r.get('n')}: left out ("
                             + ("; ".join(r.get("why_not") or [])
                                or "no step time") + ")")
                continue
            n, k = int(r["n"]), _rung_k(r)
            ctx = _rung_context(rec, r)
            if ctx is None:
                notes.append(f"{name} n={n}: left out (no measured prompt "
                             "tokens or per-sequence speed)")
                continue
            top = (m.kv_decode_const * min(ctx, m.kv_decode_topk) / m.kv_decode_topk
                   if m.kv_decode_const and m.kv_decode_topk else m.kv_decode_const)
            kv = n * (ctx * kv_bpt + top + m.state_traffic)
            union = cfg.calibration.union
            bytes_ = (m.w_decode(n * (1 + k), union)
                      + m.passes(k) * m.draft_pass_bytes(n, union) + kv)
            layers = m.layers + m.passes(k) * m.draft_layers
            rows.append({
                "x": [bytes_ / (tp * topo.gpu.hbm_bw), layers,
                      layers if tp > 1 else 0.0, m.rec_layers * (1 + k)],
                "y": t - n * (1 + k) * M.decode_token_seconds(m, topo, mfu),
                "n": n, "k": k, "tp": tp, "ctx": ctx, "t": t})
    return rows


def fit_decode(rows: list[dict]) -> dict[str, Fit]:
    """eta and the three taus from the rows, by NNLS; a column the design
    does not move (all zero, or identical to another) keeps its prior."""
    t0p, tarp, trecp = M.decode_priors()
    prior = {"eta": Fit(M.DECODE_ETA, None, 0, "prior"),
             "tau_0": Fit(t0p, None, 0, "prior"),
             "tau_ar": Fit(tarp, None, 0, "prior"),
             "tau_rec": Fit(trecp, None, 0, "prior")}
    if len(rows) < 3:
        for f in prior.values():
            f.note = f"{len(rows)} clean rungs; at least 3 needed"
        return prior
    X = np.array([r["x"] for r in rows], float)
    y = np.array([r["y"] for r in rows], float)
    names = ["eta", "tau_0", "tau_ar", "tau_rec"]
    cols = [0]
    fixed = np.zeros(len(rows))
    out = dict(prior)
    all_tp = {r["tp"] > 1 for r in rows}
    if all_tp == {True}:
        # tau_0 and tau_ar move together on every rung: fit one per-layer
        # cost and keep tau_0 at its prior
        cols.append(2)
        fixed += X[:, 1] * t0p
        out["tau_0"] = Fit(t0p, None, 0, "prior",
                           "every rung ran at tp > 1: fitted as part of tau_ar's column")
    else:
        cols.append(1)
        if all_tp == {True, False}:
            cols.append(2)
        else:
            out["tau_ar"] = Fit(tarp, None, 0, "prior", "no rung ran at tp > 1")
    if X[:, 3].any():
        cols.append(3)
    else:
        out["tau_rec"] = Fit(trecp, None, 0, "prior", "the model has no recurrent layers")
    # a fixed-cost column the design moves only in proportion to another
    # (one draft depth: L + k L_d and L_rec (1 + k) are both constant) is
    # pinned at its prior, tau_rec first: it needs a k sweep
    pin_note = {3: "every rung ran at one draft depth: a k sweep separates it",
                2: "the per-layer columns moved together", 1: "the per-layer columns moved together"}
    for c in (3, 2, 1):
        if c in cols and np.linalg.matrix_rank(X[:, cols]) < len(cols):
            cols.remove(c)
            pv = {1: t0p, 2: tarp, 3: trecp}[c]
            fixed += X[:, c] * pv
            out[names[c]] = Fit(pv, None, 0, "prior", pin_note[c])
    if np.ptp(X[:, 0]) <= 1e-12 * max(1.0, X[:, 0].max()):
        out["eta"] = Fit(M.DECODE_ETA, None, 0, "not identified",
                         "the step's bytes never varied: sweep more batch sizes or contexts")
        return out
    from .bench import _nnls
    Xa = X[:, cols]
    coef, _ = _nnls(Xa.tolist(), (y - fixed).tolist())
    coef = np.array(coef)
    active = [i for i, c in enumerate(coef) if c > 0]
    resid = (y - fixed) - Xa @ coef
    dof = len(rows) - len(active)
    cov = None
    if dof > 0 and active:
        s2 = float(resid @ resid) / dof
        A = Xa[:, active]
        try:
            cov = s2 * np.linalg.inv(A.T @ A)
        except np.linalg.LinAlgError:
            cov = None
    for j, col in enumerate(cols):
        name = names[col]
        c = coef[j]
        se = None
        if cov is not None and j in active:
            se = float(math.sqrt(max(cov[active.index(j), active.index(j)], 0.0)))
        if c <= 0 and name == "eta":
            out[name] = Fit(M.DECODE_ETA, None, len(rows), "not identified",
                            "NNLS put the byte slope at zero: the step does not grow with bytes")
            continue
        if c <= 0:
            # the design separates it (rank-deficient columns were pinned
            # above) and the data put it on the zero bound: keep the zero, so
            # the written constants reproduce what was fitted
            out[name] = Fit(0.0, None, len(rows), "measured", "at the zero bound")
            continue
        if name == "eta":
            eta = 1.0 / c
            out[name] = Fit(eta, None if se is None else se / c ** 2, len(rows), "measured")
        else:
            out[name] = Fit(float(c), se, len(rows), "measured")
    if out["eta"].status == "measured" and not 0.05 < out["eta"].value <= 1.0:
        out["eta"].status = "measured, flagged"
        out["eta"].note = "outside (0.05, 1]: the step equation misfits this run"
    return out


# ---------------------------------------------------------------------------
# acceptance, pool, MFU
# ---------------------------------------------------------------------------
def fit_acceptance(runs: list[dict]) -> Fit | None:
    """Per-draft acceptance from the spec-decode counters of every clean
    rung, each at its own draft depth k_r. With per-position counts the
    geometric MLE is successes / trials, trials = A_r + D_r - a_(k_r) summed
    (a draft run ends at its first rejection unless all k_r were accepted),
    and the per-position figures pool only rungs that drafted that far.
    Without them, a solves sum_r D_r (m(a, k_r) - 1) = A."""
    rungs = []
    for rec in runs:
        for r in rec.get("rungs", []):
            sp = (r.get("server") or {}).get("spec_decode")
            if r.get("batch_ok") and sp and sp.get("drafts"):
                rungs.append((_rung_k(r), sp))
    if not rungs:
        return None
    D = sum(sp["drafts"] for _, sp in rungs)
    A = sum(sp["accepted_tokens"] for _, sp in rungs)
    with_pos = [(k, sp) for k, sp in rungs if sp.get("accepted_per_position")]
    if with_pos and len(with_pos) == len(rungs):
        trials = 0.0
        num, den = {}, {}
        for k, sp in with_pos:
            d = sp["drafts"]
            a = {int(j): frac * d for j, frac in sp["accepted_per_position"].items()}
            last = a.get(k - 1, 0.0)
            trials += sp["accepted_tokens"] + d - last
            prev = d
            for j in range(k):
                num[j] = num.get(j, 0.0) + a.get(j, 0.0)
                den[j] = den.get(j, 0.0) + prev
                prev = a.get(j, 0.0)
        geo = A / trials
        se = math.sqrt(geo * (1 - geo) / trials)
        chain = [num[j] / den[j] if den[j] > 0 else 0.0 for j in sorted(num)]
        return Fit(geo, se, int(D), "measured",
                   "per position " + ", ".join(f"{x:.3f}" for x in chain))
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        got = sum(sp["drafts"] * (M.tokens_per_step(mid, k) - 1) for k, sp in rungs)
        lo, hi = (mid, hi) if got < A else (lo, mid)
    ks = sorted({k for k, _ in rungs})
    return Fit(lo, None, int(D), "measured",
               f"from the accepted lengths at {', '.join(map(str, ks))} drafts "
               "(no per-position counters)")


def pool_from_log(path: str) -> int | None:
    m = _KV_LOG.findall(Path(path).read_text(encoding="utf-8", errors="replace"))
    return int(m[-1].replace(",", "")) if m else None


@dataclass
class VllmMemory:
    """What a vLLM start-up log says about memory on one GPU, in bytes (the
    utilisation as a fraction). None: the log has no line for it."""
    version: str | None = None
    tensor_parallel: int | None = None
    weights: float | None = None          # Model loading took X GiB memory
    gpu_free: float | None = None         # free at vLLM's start-up snapshot
    gpu_total: float | None = None
    gpu_memory_utilization: float | None = None
    requested: float | None = None        # utilisation x total
    consumed: float | None = None         # weights + non-torch
    activation: float | None = None       # profile peak + CUDA graph estimate
    cuda_graph: float | None = None       # captured graph pool
    graph_estimate: float | None = None   # vLLM's CUDA graph estimate
    kv_memory: float | None = None        # Available KV cache memory
    pool_tokens: int | None = None
    concurrency: float | None = None      # requests at max_model_len the pool holds
    max_model_len: int | None = None
    fixed_kv: float | None = None         # --kv-cache-memory: no profile ran
    notes: list = None

    @property
    def non_torch(self) -> float | None:
        if self.consumed is None or self.weights is None:
            return None
        return self.consumed - self.weights

    def missing(self) -> dict[str, str]:
        """{term: the start-up line it needs} for every term the log lacks."""
        out = {}
        for term in ("gpu_memory_utilization", "gpu_total", "activation"):
            if getattr(self, term) is None:
                out[term] = LINE_FREE
        if self.non_torch is None:
            out["non_torch"] = LINE_FREE if self.consumed is None else LINE_WEIGHTS
        return out

    def terms(self) -> M.ReserveTerms:
        """The three reserve terms, or ValueError naming each missing line and
        the [calibration] key that supplies the term instead."""
        if self.fixed_kv is not None:
            raise ValueError(
                f"the server set its KV memory by hand (--kv-cache-memory, "
                f"{self.fixed_kv / GIB:g} GiB) and skipped vLLM's memory profile, "
                "so the log carries no reserve terms: use --reserve pool, or "
                "restart without --kv-cache-memory")
        miss = self.missing()
        if miss:
            raise ValueError(
                "the log lacks what the reserve terms need: " + "; ".join(
                    f"{t} needs \"{line}\" (or set calibration.{TERM_KEYS[t]})"
                    for t, line in miss.items())
                + ". vLLM prints the first line after CUDA graph capture, at "
                "INFO level, unless --kv-cache-memory is set")
        return M.ReserveTerms(self.gpu_memory_utilization, self.gpu_total,
                              self.activation, self.non_torch)


def read_vllm_log(text: str) -> VllmMemory:
    """Every start-up memory figure vLLM 0.33 prints at INFO level, per GPU.
    A log holding several starts is read from its last one; workers that
    print different figures are read from the one with the least KV memory
    (vLLM sizes every worker's cache to the smallest)."""
    G = GIB
    out = VllmMemory(notes=[])
    starts = [m.start() for m in _LOG["version"].finditer(text)]
    if len(starts) > 1:
        text = text[starts[-1]:]
        out.notes.append(f"the log holds {len(starts)} server starts: read the last")
    if m := _LOG["version"].search(text):
        out.version = m.group(1)
    if m := _LOG["fixed_kv"].search(text):
        out.fixed_kv = float(m.group(1)) * G
        out.notes.append("the server set its KV memory with --kv-cache-memory: "
                         "no memory profile ran, so no reserve terms")
    tps = {int(v) for v in _LOG["tp"].findall(text)}
    if len(tps) > 1:
        raise ValueError(f"the log names several tensor_parallel_size values "
                         f"({sorted(tps)}): pass one server's log")
    out.tensor_parallel = tps.pop() if tps else None
    weights = sorted({float(v) for v in _LOG["weights"].findall(text)})
    if weights:
        out.weights = weights[-1] * G
        if weights[-1] - weights[0] > 0.01:
            out.notes.append(f"workers loaded {weights[0]:g} to {weights[-1]:g} GiB "
                             "of weights: non_torch uses the largest")
    frees = [tuple(float(x) for x in m) for m in _LOG["free"].findall(text)]
    if frees:
        # the binding worker: least left for the cache
        free, total, util, req, cons, act, graph = min(
            frees, key=lambda f: f[3] - f[4] - f[5])
        if len(set(frees)) > 1:
            out.notes.append(f"{len(set(frees))} workers printed different "
                             "memory figures: read the one with the least KV memory")
        out.gpu_free, out.gpu_total = free * G, total * G
        out.gpu_memory_utilization, out.requested = util, req * G
        out.consumed, out.activation, out.cuda_graph = cons * G, act * G, graph * G
    utils = {float(v) for v in _LOG["util"].findall(text)}
    if (out.gpu_memory_utilization is not None and utils
            and any(abs(u - out.gpu_memory_utilization) > 1e-3 for u in utils)):
        raise ValueError(f"the log prints two utilisations ({sorted(utils)} and "
                         f"{out.gpu_memory_utilization:g}): pass one server's log")
    if out.gpu_memory_utilization is None and len(utils) == 1:
        out.gpu_memory_utilization = utils.pop()
        out.notes.append("gpu_memory_utilization read from the CUDA graph "
                         "profiling line; the log has no \"" + LINE_FREE + "\"")
    if est := _LOG["graph_estimate"].findall(text):
        out.graph_estimate = max(float(e) for _, e in est) * G
    kvs = [float(v) for v in _LOG["kv_memory"].findall(text)]
    if kvs:
        out.kv_memory = min(kvs) * G
    pools = _LOG["pool"].findall(text)
    if pools:
        where, n = pools[-1]
        if where != "GPU":
            raise ValueError(
                f"the log's pool is \"{where} KV cache size\": it counts host "
                "memory, which the model's HBM pool does not. Set "
                "calibration.reserve_gib by hand")
        out.pool_tokens = int(n.replace(",", ""))
    if conc := _LOG["concurrency"].findall(text):
        mml, c = conc[-1]
        out.max_model_len, out.concurrency = int(mml.replace(",", "")), float(c)
    if out.requested is not None and out.kv_memory is not None:
        gap = out.requested - out.consumed - out.activation - out.kv_memory
        if abs(gap) > 0.05 * G:
            out.notes.append(
                f"requested {out.requested / G:.2f} - consumed {out.consumed / G:.2f} "
                f"- activation {out.activation / G:.2f} GiB misses the available "
                f"KV memory {out.kv_memory / G:.2f} GiB by {gap / G:+.2f} GiB")
    return out


def fit_terms(mem: VllmMemory, cfg: RunConfig, notes: list) -> dict[str, Fit]:
    """The three reserve terms as fits, and the reserve they sum to. A term
    the log lacks is left out and said in `notes`."""
    m, topo = cfg.to_model(), cfg.to_topology()
    out: dict[str, Fit] = {}
    if mem.fixed_kv is not None:
        return out
    by_line: dict[str, list[str]] = {}
    for term, line in mem.missing().items():
        by_line.setdefault(line, []).append(term)
    for line, terms_ in by_line.items():
        notes.append(f"{', '.join(terms_)}: not read, the log has no \"{line}\" line "
                     f"({', '.join('calibration.' + TERM_KEYS[t] for t in terms_)} "
                     f"suppl{'ies' if len(terms_) == 1 else 'y'} "
                     f"{'it' if len(terms_) == 1 else 'them'})")
    G = GIB
    if mem.gpu_total is not None:
        out["gpu_total"] = Fit(mem.gpu_total, None, 1, "measured",
                               f"vLLM's CUDA total; catalogue vram {topo.gpu.vram / G:.2f} GiB")
    if mem.gpu_memory_utilization is not None:
        hb = ("" if mem.gpu_total is None else
              f"holdback {(1 - mem.gpu_memory_utilization) * mem.gpu_total / G:.2f} GiB")
        out["gpu_memory_utilization"] = Fit(mem.gpu_memory_utilization, None, 1,
                                            "measured", hb)
    if mem.activation is not None:
        note = ("" if mem.graph_estimate is None else
                f"profile peak {(mem.activation - mem.graph_estimate) / G:.2f} + CUDA "
                f"graph estimate {mem.graph_estimate / G:.2f} GiB")
        out["activation"] = Fit(mem.activation, None, 1, "measured", note)
    if mem.non_torch is not None:
        out["non_torch"] = Fit(mem.non_torch, None, 1, "measured",
                               f"consumed {mem.consumed / G:.2f} - weights "
                               f"{mem.weights / G:.2f} GiB")
    if mem.missing():
        return out
    terms = mem.terms()
    tp = topo.tp
    w_model = M.hbm_weights(m) / tp
    notes.append(
        f"reserve terms: holdback {terms.holdback / G:.2f} + activation "
        f"{terms.activation / G:.2f} + non_torch {terms.non_torch / G:.2f} = "
        f"{(terms.holdback + terms.activation + terms.non_torch) / G:.2f} GiB of "
        f"{terms.gpu_total / G:.2f} GiB per GPU; against the catalogue's vram "
        f"({topo.gpu.vram / G:.2f} GiB) the reserve is "
        f"{terms.reserve(topo.gpu) / G:.2f} GiB per GPU, the default "
        f"{M.per_gpu_reserve(replace(topo, reserve=None, reserve_terms=None)) / G:.2f}")
    if mem.weights is not None and abs(mem.weights - w_model) > 0.05 * G:
        notes.append(f"vLLM loaded {mem.weights / G:.2f} GiB of weights per GPU, the "
                     f"model holds {w_model / G:.2f}: the terms do not carry the "
                     "difference, the pool from them is off by that much per GPU")
    with_terms = M.kv_pool_tokens(m, replace(topo, reserve=None, reserve_terms=terms))
    if mem.pool_tokens:
        notes.append(f"pool from the terms {with_terms:,.0f} tokens; vLLM reported "
                     f"{mem.pool_tokens:,} ({with_terms / mem.pool_tokens - 1:+.1%})")
        if mem.kv_memory is not None and mem.concurrency and mem.max_model_len:
            # vLLM's token figure is concurrency x max_model_len, and one
            # request's blocks include its fixed pages (recurrent state, window
            # rings, padding layers): vllm/v1/core/kv_cache_utils.py,
            # get_max_concurrency_for_kv_cache_config, at v0.33.0
            kv_bpt = M._replicated(m, topo).kv_bpt
            per_req = mem.kv_memory * tp / mem.concurrency
            fixed = per_req - mem.max_model_len * kv_bpt
            if abs(fixed) > 0.01 * per_req:
                notes.append(
                    f"vLLM's token figure is {mem.concurrency:g} requests x "
                    f"{mem.max_model_len:,} tokens, and it charges each request "
                    f"{fixed / G:.2f} GiB beyond its tokens at the model's cache "
                    f"bytes (recurrent-state pages, window rings, padding layers"
                    + ("" if m.deltanet_state else
                       "; or check deployment.kv_sharding and deployment.kv_dtype")
                    + f"): a reserve fitted from the token figure absorbs "
                    f"{mem.concurrency * fixed / tp / G:.2f} GiB per GPU of it, an "
                    "amount that changes with max_model_len")
    return out


def fit_reserve(pool_tokens: float, cfg: RunConfig) -> Fit:
    """The per-GPU reserve the reported pool implies:
    reserve = (tp x vram - weights in HBM - N x kv_bpt) / tp."""
    m, topo = cfg.to_model(), cfg.to_topology()
    topo = replace(topo, reserve=None, reserve_terms=None)
    kv = M._replicated(m, topo).kv_bpt
    tp = topo.tp
    reserve = (tp * topo.gpu.vram - M.hbm_weights(m) - pool_tokens * kv) / tp
    if reserve <= 0:
        raise ValueError(
            f"the pool of {pool_tokens:,.0f} tokens leaves a negative reserve "
            f"({reserve / GIB:.1f} GiB per GPU): the model stores more per "
            "token or more weights than this server. Check "
            "deployment.kv_sharding (plain TP is \"replicate\"), "
            "deployment.kv_dtype and the model's kv_bpt")
    note = (f"pool {pool_tokens:,.0f} tokens; the default reserve gives "
            f"{M.kv_pool_tokens(m, topo):,.0f}")
    return Fit(reserve, 0.0, 1, "measured", note)


def fit_mfu(replays: list[dict], cfg: RunConfig) -> Fit | None:
    """Prefill MFU from requests sent alone: ttft = a + F / (peak x MFU)."""
    m, topo = cfg.to_model(), cfg.to_topology()
    peak = M.peak_flops(topo)
    rows, y = [], []
    for rec in replays:
        for r in rec.get("calibration", []):
            if not r.get("ok", True) or not r.get("ttft") or not r.get("uncached"):
                continue
            if r.get("phase", "single") != "single":    # a tree's streams share the GPU
                continue
            u, prompt = float(r["uncached"]), float(r.get("prompt", r["uncached"]))
            f = M.prefill_flops(m, u, prior=max(prompt - u, 0.0))[2]   # GEMM + attention
            rows.append([1.0, f / peak])
            y.append(float(r["ttft"]))
    if len(rows) < 4:
        return None
    from .bench import _nnls
    (a, b), _ = _nnls(rows, y)
    if b <= 0:
        return Fit(cfg.calibration.mfu, None, len(rows), "not identified",
                   "time to first token did not grow with the prefill FLOPs")
    X = np.array(rows)
    resid = np.array(y) - X @ np.array([a, b])
    se = None
    if len(rows) > 2:
        cov = float(resid @ resid) / (len(rows) - 2) * np.linalg.inv(X.T @ X)
        se = math.sqrt(max(cov[1, 1], 0.0)) / b ** 2
    return Fit(1.0 / b, se, len(rows), "measured", f"launch {a * 1e3:.0f} ms")


# ---------------------------------------------------------------------------
# the command
# ---------------------------------------------------------------------------
def calibrate(cfg: RunConfig, runs: list[dict], vllm_log: str | None = None,
              notes: list | None = None, reserve: str = "pool") -> dict[str, Fit]:
    """Every fit the runs and the log support. What a fit left out or fell
    back on goes in `notes`; what cannot be fitted honestly raises
    ValueError, and so does a log that lacks a reserve term when `reserve`
    is "terms" (the terms are then what --write stores)."""
    notes = [] if notes is None else notes
    sweeps = [r for r in runs if _is_sweep(r)]
    replays = [r for r in runs if _is_replay(r)]
    out: dict[str, Fit] = {}
    mfu = fit_mfu(replays, cfg) if replays else None
    if mfu is not None:
        out["mfu"] = mfu
    measured = mfu is not None and mfu.status == "measured"
    mfu_v = mfu.value if measured else cfg.calibration.mfu
    if sweeps and not measured:
        notes.append(f"the decode fit uses the config's mfu {mfu_v:g}, not a "
                     "measured one (no replay with 4+ requests sent alone)")
    if sweeps:
        out.update(fit_decode(decode_rows(sweeps, cfg, mfu_v, notes)))
        acc = fit_acceptance(sweeps)
        if acc is not None:
            out["acceptance"] = acc
    pool = None
    if vllm_log:
        mem = read_vllm_log(Path(vllm_log).read_text(encoding="utf-8", errors="replace"))
        tp = cfg.deployment.tensor_parallel
        if mem.tensor_parallel is not None and mem.tensor_parallel != tp:
            raise ValueError(
                f"{vllm_log} is a TP{mem.tensor_parallel} server and the config's "
                f"deployment.tensor_parallel is {tp}: its per-GPU figures and "
                "pool belong to that width. Set deployment.tensor_parallel to match")
        if mem.tensor_parallel is None:
            notes.append(f"{vllm_log} names no tensor_parallel_size: read as "
                         f"the config's TP{tp}")
        if reserve == "terms":
            mem.terms()          # raises on a missing line
        notes.extend(mem.notes)
        if mem.version:
            notes.append(f"vLLM {mem.version} start-up log")
        out.update(fit_terms(mem, cfg, notes))
        pool = mem.pool_tokens
        if pool is None:
            raise ValueError(f"{vllm_log} has no '{LINE_POOL}' line")
    if pool is None:
        tp = cfg.deployment.tensor_parallel
        other = [r.get("_path", "sweep") for r in sweeps if _run_tp(r, cfg) != tp]
        if other:
            notes.append(f"pool: left out {', '.join(other)} (not TP{tp})")
        pools = {r.get("kv_pool_tokens") for r in sweeps
                 if r.get("kv_pool_tokens") and _run_tp(r, cfg) == tp}
        if len(pools) > 1:
            raise ValueError(f"the TP{tp} sweeps report different KV pools "
                             f"({', '.join(f'{p:,}' for p in sorted(pools))}): "
                             "pass the server log with --vllm-log")
        pool = pools.pop() if pools else None
    if pool:
        out["reserve"] = fit_reserve(pool, cfg)
    return out


_KEYS = {"eta": ("eta", 1.0), "tau_0": ("tau_0_us", 1e6), "tau_ar": ("tau_ar_us", 1e6),
         "tau_rec": ("tau_rec_us", 1e6), "acceptance": ("acceptance", 1.0),
         "reserve": ("reserve_gib", 1 / GIB), "mfu": ("mfu", 1.0),
         "gpu_memory_utilization": ("gpu_memory_utilization", 1.0),
         "gpu_total": ("gpu_total_gib", 1 / GIB), "activation": ("activation_gib", 1 / GIB),
         "non_torch": ("non_torch_gib", 1 / GIB)}
RESERVE_FORMS = {"pool": ("reserve",), "terms": tuple(TERM_KEYS)}


def apply(cfg: RunConfig, fits: dict[str, Fit], sources: str,
          reserve: str = "pool") -> RunConfig:
    """The config with every measured fit written into [calibration] and its
    provenance under [calibration.fit.<name>]. `reserve` picks which form of
    the reserve is written, "pool" (reserve_gib) or "terms" (the four reserve
    term keys); writing one clears the other, since a config carries one."""
    cal = cfg.calibration
    kw, prov = {}, dict(cal.fit)
    today = time.strftime("%Y-%m-%d")
    other = [n for form, names in RESERVE_FORMS.items() if form != reserve for n in names]
    if any(n in fits for n in RESERVE_FORMS[reserve]):
        for n in other:
            kw[_KEYS[n][0]] = None
            prov.pop(_KEYS[n][0], None)
    for name, f in fits.items():
        if f.status != "measured" or name in other:
            continue
        key, scale = _KEYS[name]
        kw[key] = round(float(f.value) * scale, 6)
        prov[key] = {"value": kw[key], "status": f.status, "n": f.n, "source": sources,
                     "date": today}
        if f.stderr is not None:
            prov[key]["stderr"] = round(float(f.stderr) * scale, 6)
        if f.note:
            prov[key]["note"] = f.note
    return replace(cfg, calibration=replace(cal, fit=prov, **kw))


def cmd_calibrate(args) -> int:
    try:
        cfg = load_config(args.config)
        cfg.validate()
        runs = [_load_run(p) for p in args.runs]
        if not runs and not args.vllm_log:
            raise ValueError("nothing to read: pass decode-sweep or replay "
                             "records, or --vllm-log")
        if getattr(args, "reserve", "pool") == "terms" and not args.vllm_log:
            raise ValueError("--reserve terms reads the terms from a vLLM "
                             "start-up log: pass --vllm-log")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
        print(f"ws calibrate: {e}", file=sys.stderr)
        return 2
    unknown = [r["_path"] for r in runs if not (_is_sweep(r) or _is_replay(r))]
    if unknown:
        print(f"ws calibrate: not a decode-sweep or replay record: {', '.join(unknown)}",
              file=sys.stderr)
        return 2
    notes: list[str] = []
    reserve = getattr(args, "reserve", "pool")
    try:
        fits = calibrate(cfg, runs, args.vllm_log, notes, reserve)
    except ValueError as e:
        print(f"ws calibrate: {e}", file=sys.stderr)
        return 2
    m = cfg.to_model()
    print(f"{m.name} on {cfg.deployment.gpus} (TP{cfg.deployment.tensor_parallel}): "
          f"{len(runs)} run(s)")
    units = {"eta": ("", 1.0), "tau_0": (" us", 1e6), "tau_ar": (" us", 1e6),
             "tau_rec": (" us", 1e6), "acceptance": ("", 1.0),
             "reserve": (" GiB/GPU", 1 / GIB), "mfu": ("", 1.0),
             "gpu_memory_utilization": ("", 1.0), "gpu_total": (" GiB/GPU", 1 / GIB),
             "activation": (" GiB/GPU", 1 / GIB), "non_torch": (" GiB/GPU", 1 / GIB)}
    for name, f in fits.items():
        u, sc = units[name]
        print(f.show(name, u, sc))
    for n in notes:
        print(f"  note: {n}")
    unwritten = [k for k, f in fits.items() if f.status != "measured"]
    if args.write and unwritten:
        print("  not written (not measured): " + ", ".join(
            f"{k} ({fits[k].status})" for k in unwritten))
    if not fits:
        print("  nothing to fit: no clean rung, no spec-decode counters, no pool reading")
    sources = ",".join([r["_sha"] for r in runs] + (
        [hashlib.sha256(Path(args.vllm_log).read_bytes()).hexdigest()[:12]]
        if args.vllm_log else []))
    if args.write:
        new = apply(cfg, fits, sources, reserve)
        Path(args.config).write_text(new.dumps("toml"), encoding="utf-8")
        print(f"wrote [calibration] to {args.config}")
    return 0


def add_subparser(sub) -> None:
    p = sub.add_parser("calibrate", help="fit decode, acceptance, pool and MFU from runs",
                       description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="*", help="ws decode-sweep or ws replay JSON records")
    p.add_argument("--config", required=True,
                   help="workingset.toml naming the deployment the runs were taken on")
    p.add_argument("--vllm-log", help="vLLM server log: its 'GPU KV cache size' line "
                   "gives the KV pool, its start-up memory lines the reserve terms")
    p.add_argument("--reserve", choices=tuple(RESERVE_FORMS), default="pool",
                   help="what --write stores for the KV pool: 'pool', the single "
                   "reserve_gib the reported pool implies (default); 'terms', the "
                   "three reserve terms read from --vllm-log (refused if a line is "
                   "missing)")
    p.add_argument("--write", action="store_true",
                   help="store the measured fits in the config's [calibration]")
    p.set_defaults(fn=cmd_calibrate)
