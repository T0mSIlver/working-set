#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy>=1.26",
#     "httpx>=0.27",
# ]
# ///
"""Decode-side calibration: the MBU and per-pass overhead the roofline omits.

The study prices decode as a pure HBM roofline (docs/scenarios.md limitation
11): per-user tok/s = mtp x effective_bw / step_bytes, with NO efficiency
term and NO fixed cost. Prefill got its MFU anchor (research/prefill.md);
decode never got its counterpart, so every tok/s figure in the study is an
upper bound of unknown tightness. This script measures the counterpart from
vLLM's /metrics counters, on a SHARED production instance, with no
exclusive access and (in the passive tier) no added load.

THE FITTED MODEL — the roofline plus the two terms it is missing:

    t_pass  =  t0  +  step_bytes / (eta x bw_advertised)

    t0   fixed seconds per forward pass: TP collectives, kernel launches,
         recurrent-state updates, scheduler work. Independent of batch.
    eta  achieved fraction of ADVERTISED aggregate HBM bandwidth (MBU).
         Absorbs tp_efficiency, which a single-TP-width run cannot separate
         from it -- so eta is reported in BOTH conventions, exactly as the
         prefill MFU points are (research/prefill.md, "convention note").

    per-user tok/s = accepted_len / t_pass,  accepted_len measured, not fitted.

Why both terms: at n=1 they are indistinguishable -- ANY (t0, eta) pair on one
line through the observation fits -- and they diverge by >2x at n=64, which
is where the decode ceiling lives. Separating them is the whole point of the
batch-size sweep, and this script refuses to report eta when the accepted
windows lack the byte-range leverage to identify it.

ESTIMATORS (all from counter deltas between two /metrics scrapes):
    steps      d(vllm:iteration_tokens_total_count)      -- one per forward pass
    t_pass     d(wall) / d(steps), or d(model_forward_time_ms) if exported
    n          num_requests_running gauge (cross-checked against
               d(iteration_tokens_sum)/d(steps) / (1 + spec draft width))
    sum_L      kv_cache_usage_perc x pool_tokens, plus the shared-prefix
               correction (storage dedups a shared prefix; every sequence
               still READS it -- unless cascade attention is on: --cascade)
    accepted   1 + d(spec_accepted_tokens)/d(spec_drafts)

A window is used ONLY if it is pure decode and structurally stable:
    d(prompt_tokens_total) == 0        no prefill shared any pass
    num_requests_running equal at both ends and >= 1
    d(request_success_total) == 0      no request left mid-window
    d(steps) >= --min-steps            wall-clock jitter amortised
    d(generation_tokens) ~= steps x n x accepted_len   (+- --tol)
                                       catches idle gaps and batch churn
Rejections are counted by reason and reported: a harvest that yields nothing
tells you WHY, which is the difference between "wrong flags" and "wrong hour".

  uv run scripts/decode_mbu.py --probe http://backend:8000/metrics
  uv run scripts/decode_mbu.py mfu_scrape_*.log --pool-tokens 13900000
  uv run scripts/decode_mbu.py *.log --pool-tokens 13.9e6 --arms arms.jsonl --json out.json

Scrape logs come from scripts/scrape_metrics.py (any interval; 0.5-1 s is the
sweet spot). Active batch-size plateaus come from scripts/decode_probe.py,
whose arms.jsonl labels each window with the arm that produced it.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from glob import glob
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scenario_model as S  # noqa: E402  (path shim must precede the import)

# ---------------------------------------------------------------------------
# metric names. vLLM renamed several of these between v0 and v1, so every
# series this script needs is a CANDIDATE LIST, resolved once per log against
# what the deployment actually exports. --probe prints the resolution table.
# ---------------------------------------------------------------------------
SERIES = {
    # required
    "steps":     (["vllm:iteration_tokens_total_count"], True,
                  "forward passes per window -- the whole measurement"),
    "running":   (["vllm:num_requests_running"], True,
                  "batch size n"),
    "prompt":    (["vllm:prompt_tokens_total"], True,
                  "prefill detector: a window is decode-only iff this is flat"),
    "gen":       (["vllm:generation_tokens_total"], True,
                  "self-consistency check that rejects idle gaps"),
    # strongly wanted
    "kv":        (["vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"], False,
                  "sum of context lengths; without it sum_L is ASSUMED from "
                  "the workload model and the KV slope is not measured"),
    "success":   (["vllm:request_success_total"], False,
                  "rejects windows where a request finished mid-flight"),
    "iter_sum":  (["vllm:iteration_tokens_total_sum"], False,
                  "independent estimate of n (tokens per pass / draft width)"),
    "waiting":   (["vllm:num_requests_waiting"], False,
                  "queue depth, logged with each window for the politeness audit"),
    # speculative decoding
    "drafts":    (["vllm:spec_decode_num_drafts_total"], False,
                  "draft events; accepted_len = 1 + accepted/drafts"),
    "draft_tok": (["vllm:spec_decode_num_draft_tokens_total"], False,
                  "draft width k, and the per-draft acceptance alpha"),
    "accepted":  (["vllm:spec_decode_num_accepted_tokens_total"], False,
                  "the measured speculative speedup -- replaces the mtp slider"),
    "acc_pos":   (["vllm:spec_decode_num_accepted_tokens_per_pos_total"], False,
                  "per-position acceptance: tests the 1+a+a^2 geometric model"),
    # optional luxury: engine-side pass timing, immune to scrape jitter
    "fwd_sum":   (["vllm:model_forward_time_milliseconds_sum"], False,
                  "engine-side pass time; makes t_pass independent of wall clock"),
    "fwd_count": (["vllm:model_forward_time_milliseconds_count"], False, ""),
}

LINE = re.compile(r'^([a-zA-Z_:][\w:]*)(\{[^}]*\})?\s+([-+0-9.eENaninf]+)\s*$')
LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


def parse_line(line):
    m = LINE.match(line.strip())
    if not m:
        return None
    name, labels, val = m.group(1), m.group(2) or "", m.group(3)
    try:
        v = float(val)
    except ValueError:
        return None
    return name, dict(LABEL.findall(labels)), v


def parse_log(path, engine=None):
    """Scrape log -> [(t_unix, {name: value})], values summed over label sets.

    Counters and gauges are per-engine. Summing is correct for a single
    engine and WRONG for DP>1 (each engine steps on its own clock), so the
    engine label set is tracked and a multi-engine log is refused unless
    --engine selects one.
    """
    blocks, cur, t = [], None, None
    engines = set()
    for raw in Path(path).read_text(errors="replace").splitlines():
        if raw.startswith("====="):
            if cur is not None:
                blocks.append((t, cur))
            parts = raw.split()
            t, cur = float(parts[1]), defaultdict(float)
            # scrape_metrics records when the request was SENT plus its round
            # trip; the counters are as-of some instant inside that, so take
            # the midpoint. Older logs have no rtt field and lose ~rtt/2 of
            # timing sharpness -- harmless over multi-second windows.
            rtt = next((float(x[4:]) for x in parts[2:] if x.startswith("rtt=")
                        and x[4:] not in ("nan",)), 0.0)
            t += rtt / 2.0
            continue
        if cur is None or not raw.startswith("vllm:"):
            continue
        p = parse_line(raw)
        if p is None:
            continue
        name, labels, v = p
        eid = labels.get("engine") or labels.get("engine_index")
        if eid is not None:
            engines.add(eid)
            if engine is not None and eid != str(engine):
                continue
        # every other series is summed over its label sets (per-engine
        # counters); the per-POSITION acceptance histogram is the one place
        # where the label IS the measurement, so it keeps its own key
        if "position" in labels:
            name = f"{name}@{labels['position']}"
        cur[name] += v
    if cur is not None:
        blocks.append((t, cur))
    return blocks, engines


def resolve(sample):
    """Map logical series -> the exported name this deployment actually uses."""
    found = {}
    for key, (cands, _req, _why) in SERIES.items():
        for c in cands:
            if c in sample or any(k.startswith(c + "@") for k in sample):
                found[key] = c
                break
    return found


def get(block, names, key, default=None):
    n = names.get(key)
    return block.get(n, default) if n else default


# ---------------------------------------------------------------------------
# window extraction
# ---------------------------------------------------------------------------
def harvest(blocks, names, args):
    """Every scrape pair in [min_s, max_s] that survives the decode-only
    predicates. Pairs, not consecutive samples: a long window amortises
    scrape jitter, a short one is likelier to stay structurally stable, and
    which is available depends on the hour -- so try both and let the
    predicates choose."""
    rej = Counter()
    out = []
    for i in range(len(blocks)):
        t0, b0 = blocks[i]
        for j in range(i + 1, len(blocks)):
            t1, b1 = blocks[j]
            dt = t1 - t0
            if dt < args.min_window:
                continue
            if dt > args.max_window:
                break
            w = window(t0, b0, t1, b1, names, args, rej)
            if w:
                out.append(w)
    return out, rej


def window(t0, b0, t1, b1, names, args, rej):
    dt = t1 - t0
    d = lambda k: (get(b1, names, k, 0.0) or 0.0) - (get(b0, names, k, 0.0) or 0.0)

    steps = d("steps")
    if steps < args.min_steps:
        rej["too few steps"] += 1
        return None
    if d("prompt") != 0:
        rej["prefill in window"] += 1
        return None
    n0, n1 = get(b0, names, "running"), get(b1, names, "running")
    if n0 is None or n0 != n1 or n0 < 1:
        rej["batch size changed or idle"] += 1
        return None
    n = float(n0)
    if names.get("success") and d("success") != 0:
        rej["request finished mid-window"] += 1
        return None

    # speculative decoding: tokens emitted per sequence per pass
    drafts, acc = d("drafts"), d("accepted")
    if names.get("drafts") and drafts > 0:
        accepted_len = 1.0 + acc / drafts
        k_draft = d("draft_tok") / drafts if d("draft_tok") else float("nan")
    else:
        accepted_len, k_draft = 1.0, 0.0

    # the filter that earns its keep: delivered tokens must equal
    # steps x batch x accepted_len. An idle gap inflates dt without steps;
    # a batch that dipped and recovered passes the gauge check but not this.
    gen = d("gen")
    predicted = steps * n * accepted_len
    if gen <= 0 or abs(gen - predicted) > args.tol * max(gen, predicted):
        rej["token accounting mismatch"] += 1
        return None

    # per-pass time. Engine-side if exported (immune to scrape jitter),
    # else wall-clock over the window.
    if names.get("fwd_sum") and names.get("fwd_count") and d("fwd_count") > 0:
        t_pass, t_src = d("fwd_sum") / 1e3 / d("fwd_count"), "engine"
    else:
        t_pass, t_src = dt / steps, "wall"

    # Sum of context lengths, at the MIDPOINT of the window. Contexts grow
    # while the window runs -- n=32 decoding for 20 s adds ~300k tokens,
    # ~10 GB of step bytes -- so the end-of-window value would price a pass
    # that never happened. The midpoint is not an approximation here: step
    # bytes grow linearly in time and t_pass is affine in step bytes, so the
    # mean pass time over the window corresponds EXACTLY to the mean bytes.
    # Windows whose bytes swing too far are dropped anyway (--max-growth):
    # the fit wants points, not smears.
    kv0, kv1 = get(b0, names, "kv"), get(b1, names, "kv")
    if kv0 is not None and kv1 is not None and args.pool_tokens:
        L0, L1 = kv0 * args.pool_tokens, kv1 * args.pool_tokens
        if max(L0, L1) > 0 and abs(L1 - L0) / max(L0, L1) > args.max_growth:
            rej["context grew too much in window"] += 1
            return None
        sum_L, L_src = 0.5 * (L0 + L1), "metrics"
    else:
        sum_L, L_src = n * args.assumed_context, "assumed"

    pos = {}
    base = names.get("acc_pos")
    if base:
        for key, v1 in b1.items():
            if key.startswith(base + "@"):
                pos[key.split("@", 1)[1]] = v1 - b0.get(key, 0.0)

    return dict(t0=t0, t1=t1, dt=dt, steps=steps, n=n, t_pass=t_pass,
                t_src=t_src, sum_L=sum_L, L_stored=sum_L, L_src=L_src,
                accepted_len=accepted_len, k_draft=k_draft, drafts=drafts,
                pos_accepted=pos, gen=gen, waiting=get(b1, names, "waiting"))


# ---------------------------------------------------------------------------
# the fit
# ---------------------------------------------------------------------------
def step_bytes(model, n, sum_L):
    """The roofline's own byte ledger, called with MEASURED occupancy so the
    fit cannot disagree with the model it is calibrating."""
    return model.w_decode(int(round(n))) + sum_L * model.kv_bpt \
        + 2.0 * n * model.deltanet_state


def fit(points, bw_adv):
    """WLS of t_pass on roofline-seconds x = bytes / bw_advertised.

        t = t0 + x / eta          slope 1/eta, intercept t0

    Weight w = steps^2: the error in t_pass = dt/steps is scrape jitter
    (roughly constant per window) divided by steps, so precision grows
    linearly with window length and the WEIGHT with its square.
    """
    x = np.array([p["bytes"] / bw_adv for p in points])
    y = np.array([p["t_pass"] for p in points])
    w = np.array([p["steps"] ** 2 for p in points], dtype=float)
    A = np.vstack([np.ones_like(x), x]).T
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
    t0, slope = float(coef[0]), float(coef[1])
    return t0, (1.0 / slope if slope > 0 else float("nan"))


def cluster(points, gap):
    """Group windows into plateaus. Every scrape PAIR inside one plateau is a
    window, so windows overlap heavily and are anything but independent --
    resampling them individually would report a confidence interval an order
    of magnitude too tight. The plateau is the independent unit: one batch
    size, one stretch of wall clock, one set of thermal and neighbour
    conditions."""
    order = sorted(range(len(points)), key=lambda i: points[i]["t0"])
    cid, last_t, last_n = 0, None, None
    for i in order:
        p = points[i]
        if last_t is not None and (p["t0"] - last_t > gap or p["n"] != last_n):
            cid += 1
        p["cluster"] = cid
        last_t, last_n = p["t0"], p["n"]
    return cid + 1


def bootstrap(points, bw_adv, iters=2000, seed=0):
    """Block bootstrap over plateaus (see cluster())."""
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for p in points:
        groups[p.get("cluster", 0)].append(p)
    keys = list(groups)
    t0s, etas = [], []
    for _ in range(iters):
        draw = [p for k in rng.choice(keys, size=len(keys), replace=True)
                for p in groups[k]]
        try:
            a, b = fit(draw, bw_adv)
        except np.linalg.LinAlgError:
            continue
        if math.isfinite(a) and math.isfinite(b):
            t0s.append(a); etas.append(b)
    if not t0s:
        return (float("nan"),) * 2, (float("nan"),) * 2
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return q(t0s), q(etas)


def ceilings(model, topo, t0, eta, bw_adv, accepted_len, args):
    """What the calibration does to the study's verdict.

    The decode ceiling is where per-user p50 falls to the 40 tok/s floor. The
    cache ceiling does not move (bytes stored, not bytes read), so this is
    the whole question of which constraint binds -- and on the reference
    27B/TP4 row the two ceilings currently sit within ~25% of each other.
    """
    wl = S.Workload(user_median=args.wl_median)
    rng = np.random.default_rng(0)

    def bytes_at(n):
        full, _, _, _ = wl.sample(rng, (400, n))
        return float(np.percentile(model.w_decode(n) + full.sum(axis=1) * model.kv_bpt
                                   + 2 * n * model.deltanet_state, 50))

    def v_cal(n):
        return accepted_len / (t0 + bytes_at(n) / (eta * bw_adv))

    def v_model(n):
        return model.mtp * S.effective_bw(topo) / bytes_at(n)

    def cross(f, floor=S.DECODE_FLOOR_TOKS):
        if f(1) < floor:
            return 0
        lo, hi = 1, 4096
        while hi - lo > 1:
            mid = (lo + hi) // 2
            (lo, hi) = (mid, hi) if f(mid) >= floor else (lo, mid)
        return lo

    cache = S.max_users_cache(model, topo, wl)
    d_model, d_cal = cross(v_model), cross(v_cal)
    print("\n-- what this moves (stress convention, "
          f"{S.DECODE_FLOOR_TOKS:.0f} tok/s floor, user_median {args.wl_median:,.0f}) --")
    print(f"   per-user at n=1   : model {v_model(1):7.0f} -> calibrated {v_cal(1):7.0f} tok/s")
    print(f"   per-user at n=64  : model {v_model(64):7.0f} -> calibrated {v_cal(64):7.0f} tok/s")
    print(f"   DECODE ceiling    : model {d_model:7d} -> calibrated {d_cal:7d} users")
    print(f"   CACHE ceiling     : {cache:7.0f} users (unchanged -- storage, not reads)")
    verdict = ("cache still binds first -- the study's thesis survives"
               if cache < d_cal else
               "DECODE now binds before cache: the study's central ordering "
               "FLIPS for this configuration")
    print(f"   -> {verdict}")


# ---------------------------------------------------------------------------
# --self-test: recover known (t0, eta) from a synthetic log
# ---------------------------------------------------------------------------
def self_test(args):
    """Generate a scrape log from a KNOWN t0/eta with the arm-A shape, run the
    real harvester and fit over it, and check the recovery. Offline: proves
    the estimator inverts its own generative model before it is pointed at a
    production endpoint."""
    import tempfile
    T0, ETA, K, ALPHA, PREFIX, POOL = 0.0024, 0.47, 2, 0.55, 64_000, 13.9e6
    model = S.MODELS["27B"]
    bw_adv = args.tp * S.GPUS[args.gpu].hbm_bw
    rng = np.random.default_rng(11)
    a_len = 1 + ALPHA + ALPHA ** 2
    c = dict(steps=0.0, gen=0.0, prompt=0.0, drafts=0.0, dtok=0.0, acc=0.0, succ=0.0)
    t, out, arms = 1.7e9, [], []

    def emit(t, n, storage):
        out.append(f"===== {t:.3f} synthetic")
        for name, v in (("vllm:num_requests_running", n),
                        ("vllm:num_requests_waiting", 0),
                        ("vllm:iteration_tokens_total_count", c["steps"]),
                        ("vllm:prompt_tokens_total", c["prompt"]),
                        ("vllm:generation_tokens_total", c["gen"]),
                        ("vllm:kv_cache_usage_perc", storage / POOL),
                        ("vllm:request_success_total", c["succ"]),
                        ("vllm:spec_decode_num_drafts_total", c["drafts"]),
                        ("vllm:spec_decode_num_draft_tokens_total", c["dtok"]),
                        ("vllm:spec_decode_num_accepted_tokens_total", c["acc"])):
            out.append(f'{name}{{engine="0"}} {v}')

    for k in (1, 2, 4, 8, 16, 32):
        c["prompt"] += PREFIX; c["succ"] += 1
        grown, start = 0.0, t + 1
        for _ in range(45):
            reads = k * (PREFIX + grown)
            b = model.w_decode(k) + reads * model.kv_bpt + 2 * k * model.deltanet_state
            emit(t, k, PREFIX + k * grown)
            steps = 1.0 / (T0 + b / (ETA * bw_adv))
            c["steps"] += steps; c["gen"] += steps * k * a_len
            c["drafts"] += steps * k; c["dtok"] += steps * k * K
            c["acc"] += steps * k * (a_len - 1)
            grown += steps * a_len
            t += 1.0 + float(rng.normal(0, 0.004))     # scrape jitter
        emit(t, k, PREFIX + k * grown)
        arms.append({"arm": "A", "k": k, "shared_prefix_tokens": PREFIX,
                     "start": start, "end": t - 1})
        t += 15.0

    d = Path(tempfile.mkdtemp())
    (d / "s.log").write_text("\n".join(out) + "\n")
    (d / "arms.jsonl").write_text("\n".join(json.dumps(a) for a in arms) + "\n")
    args.pool_tokens, args.logs, args.arms = POOL, [str(d / "s.log")], str(d / "arms.jsonl")

    blocks, _ = parse_log(args.logs[0])
    names = resolve(blocks[-1][1])
    pts, rej = harvest(blocks, names, args)
    cluster(pts, args.max_window)
    label_arms(pts, args.arms)
    apply_shared_prefix(pts, args)
    for p in pts:
        p["bytes"] = step_bytes(model, p["n"], p["sum_L"])
    t0, eta = fit(pts, bw_adv)
    al = float(np.median([p["accepted_len"] for p in pts]))
    ok = (abs(t0 - T0) / T0 < 0.15 and abs(eta - ETA) / ETA < 0.10
          and abs(al - a_len) / a_len < 0.01)
    print(f"self-test: {len(pts):,} windows from {len(arms)} plateaus, "
          f"leverage {max(p['bytes'] for p in pts)/min(p['bytes'] for p in pts):.2f}x")
    print(f"  t0           truth {T0*1e3:6.2f} ms   recovered {t0*1e3:6.2f} ms")
    print(f"  eta          truth {ETA:6.3f}      recovered {eta:6.3f}")
    print(f"  accepted_len truth {a_len:6.3f}      recovered {al:6.3f}")
    print("PASS" if ok else "FAIL -- estimator does not invert its own model")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# --probe: does this deployment export what the plan needs?
# ---------------------------------------------------------------------------
def probe(url, api_key, verify=True):
    import httpx
    hdr = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    r = httpx.get(url, headers=hdr, timeout=10.0, verify=verify)
    r.raise_for_status()
    sample, engines = defaultdict(float), set()
    for line in r.text.splitlines():
        if not line.startswith("vllm:"):
            continue
        p = parse_line(line)
        if p:
            sample[p[0]] += p[2]
            eid = p[1].get("engine") or p[1].get("engine_index")
            if eid is not None:
                engines.add(eid)
    names = resolve(sample)
    print(f"{url}\n{len(sample)} vllm: series, {len(engines) or 1} engine(s)\n")
    blocking = []
    for key, (cands, req, why) in SERIES.items():
        if not why:
            continue
        got = names.get(key)
        mark = "ok  " if got else ("MISSING" if req else "absent ")
        print(f"  [{mark}] {key:<10} {got or cands[0]}")
        if why:
            print(f"             {why}")
        if req and not got:
            blocking.append(key)
    if len(engines) > 1:
        print(f"\n!! {len(engines)} engines (DP): pass --engine IDX; step counts "
              "summed across engines do not divide into a per-pass time.")
    print()
    if blocking:
        print(f"BLOCKED: {', '.join(blocking)} absent. The passive tier cannot run.")
        print("  vLLM v1 exports all four by default; check for a metrics allowlist "
              "in front of the endpoint, or scrape the engine directly.")
    else:
        spec = "yes" if names.get("accepted") else "NO -- speculative counters absent"
        print("READY. Passive tier can run.  speculative metrics: " + spec)
        if not names.get("kv"):
            print("  no KV-usage gauge: sum_L will be assumed, and the KV slope "
                  "(arm B) cannot be measured.")
        if not names.get("fwd_sum"):
            print("  no engine-side forward-time histogram (needs vLLM's detailed "
                  "traces): t_pass falls back to wall-clock, so use windows of "
                  ">= 200 steps and a steady scrape interval.")
    return 0 if not blocking else 2


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("logs", nargs="*", help="scrape logs from scrape_metrics.py")
    ap.add_argument("--probe", metavar="URL", help="capability check, then exit")
    ap.add_argument("--api-key")
    ap.add_argument("--ca-bundle", help="CA bundle for --probe over TLS")
    ap.add_argument("--insecure", action="store_true",
                    help="skip TLS verification on --probe")
    ap.add_argument("--model", default="27B", choices=sorted(S.MODELS))
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--gpu", default="H200", choices=sorted(S.GPUS))
    ap.add_argument("--engine", help="engine label to keep (DP deployments)")
    ap.add_argument("--pool-tokens", type=float, default=0.0,
                    help="KV pool size in tokens, from vLLM's startup line "
                         "'GPU KV cache size: N tokens'. Also validates "
                         "kv_pool_tokens() -- the study's binding ceiling.")
    ap.add_argument("--shared-prefix", type=float, default=0.0,
                    help="tokens of system prefix shared by all sessions "
                         "(storage dedups it, decode reads it per sequence)")
    ap.add_argument("--cascade", action="store_true",
                    help="cascade attention on: the shared prefix is read once "
                         "per batch, so skip the correction")
    ap.add_argument("--assumed-context", type=float, default=57_400,
                    help="fallback per-sequence context when no KV gauge exists")
    ap.add_argument("--min-window", type=float, default=0.8)
    ap.add_argument("--max-window", type=float, default=20.0)
    ap.add_argument("--min-steps", type=float, default=60)
    ap.add_argument("--tol", type=float, default=0.02,
                    help="token-accounting tolerance")
    ap.add_argument("--max-growth", type=float, default=0.05,
                    help="drop windows whose KV occupancy moves more than this "
                         "fraction end to end")
    ap.add_argument("--arms", help="arms.jsonl from decode_probe.py")
    ap.add_argument("--json", help="write the full result set here")
    ap.add_argument("--ceilings", action="store_true",
                    help="re-solve the decode ceiling under the fit and say "
                         "which constraint binds")
    ap.add_argument("--wl-median", type=float, default=47_400,
                    help="workload lognormal median for --ceilings (the "
                         "measured-trace fit)")
    ap.add_argument("--self-test", action="store_true",
                    help="recover known parameters from a synthetic log, offline")
    args = ap.parse_args()

    if args.self_test:
        return self_test(args)
    if args.probe:
        return probe(args.probe, args.api_key,
                     args.ca_bundle or (not args.insecure))
    if not args.logs:
        ap.error("give scrape logs, or --probe URL")

    # PowerShell does not expand wildcards for the program the way a POSIX
    # shell does, so a Windows caller hands us the literal "mfu_scrape_*.log".
    # Expand here: harmless where the shell already did it (a plain filename
    # globs to itself), and the difference between working and Errno 22 where
    # it did not.
    expanded = []
    for pat in args.logs:
        hits = sorted(glob(pat))
        if hits:
            expanded += hits
        elif Path(pat).exists():
            expanded.append(pat)
        else:
            ap.error(f"no log matches {pat!r}")
    args.logs = expanded
    print(f"reading {len(args.logs)} log(s): "
          + ", ".join(Path(x).name for x in args.logs[:4])
          + (" ..." if len(args.logs) > 4 else ""))

    model = S.MODELS[args.model]
    topo = S.topology_grid(1, args.tp, args.gpu)
    bw_adv = args.tp * S.GPUS[args.gpu].hbm_bw          # advertised aggregate
    bw_model = S.effective_bw(topo)                     # what the study prices

    pts, rej, engines_all = [], Counter(), set()
    for path in args.logs:
        blocks, engines = parse_log(path, args.engine)
        engines_all |= engines
        names = resolve(blocks[-1][1] if blocks else {})
        missing = [k for k, (_c, req, _w) in SERIES.items()
                   if req and k not in names]
        if missing:
            print(f"{path}: missing required series {missing} -- skipped",
                  file=sys.stderr)
            continue
        w, r = harvest(blocks, names, args)
        pts += w
        rej += r
    if len(engines_all) > 1 and not args.engine:
        sys.exit(f"log spans {len(engines_all)} engines {sorted(engines_all)}; "
                 "pass --engine IDX -- summed step counts are not a pass time")
    if not pts:
        print("no usable windows.\nrejections: " +
              ", ".join(f"{k} {v:,}" for k, v in rej.most_common()))
        return 1

    n_clusters = cluster(pts, args.max_window)
    if args.arms:
        hit = label_arms(pts, args.arms)
        print(f"arms: {hit:,}/{len(pts):,} windows fell inside a probe plateau")
    apply_shared_prefix(pts, args)
    for p in pts:
        p["bytes"] = step_bytes(model, p["n"], p["sum_L"])
        p["roofline_s"] = p["bytes"] / bw_model      # what the study predicts
        p["mbu_naive"] = (p["bytes"] / bw_adv) / p["t_pass"]
    if n_clusters < 4:
        print(f"WARNING: only {n_clusters} independent plateau(s) -- the "
              "interval below is a within-plateau interval and understates "
              "run-to-run spread.", file=sys.stderr)

    report(pts, rej, model, topo, args, bw_adv, bw_model)
    if args.ceilings:
        t0, eta = fit(pts, bw_adv)
        al = float(np.median([p["accepted_len"] for p in pts]))
        if max(p["bytes"] for p in pts) / min(p["bytes"] for p in pts) < 1.5:
            print("\n-- ceilings skipped: leverage too low, the fit is not "
                  "identified and the verdict would be an artefact --")
        else:
            ceilings(model, topo, t0, eta, bw_adv, al, args)
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"config": {"model": args.model, "tp": args.tp, "gpu": args.gpu,
                        "bw_advertised": bw_adv, "bw_model": bw_model,
                        "pool_tokens": args.pool_tokens},
             "windows": pts}, indent=1, default=float))
        print(f"\nwrote {args.json} ({len(pts)} windows)")
    return 0


def label_arms(pts, path):
    """Attach the decode_probe plateau each window fell inside. Carries the
    plateau's shared-prefix length, which the KV gauge cannot see."""
    arms = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    hit = 0
    for p in pts:
        a = next((a for a in arms
                  if a.get("start", 0) <= p["t0"] and p["t1"] <= a.get("end", 0)), None)
        p["arm"] = a.get("arm") if a else None
        p["arm_k"] = a.get("k") if a else None
        p["arm_shared_prefix"] = a.get("shared_prefix_tokens", 0) if a else None
        hit += a is not None
    return hit


def apply_shared_prefix(pts, args):
    """sum_L is a READ total; the KV gauge reports STORAGE. A prefix shared by
    the batch is stored once and read n times, so add the missing (n-1)
    copies -- unless cascade attention reads it once for the batch too, which
    is exactly what decode_probe's arm C is for. Per-window, because a probe
    plateau's prefix (arms.jsonl) and production's system prompt
    (--shared-prefix) are different lengths."""
    for p in pts:
        if p["L_src"] != "metrics":
            continue
        prefix = p.get("arm_shared_prefix")
        if prefix is None:
            prefix = args.shared_prefix
        p["shared_prefix_used"] = 0.0 if args.cascade else float(prefix or 0.0)
        p["sum_L"] = p["L_stored"] + (p["n"] - 1) * p["shared_prefix_used"]


def report(pts, rej, model, topo, args, bw_adv, bw_model):
    n_all = np.array([p["n"] for p in pts])
    by = np.array([p["bytes"] for p in pts])
    lev = by.max() / by.min()

    print(f"\n=== DECODE CALIBRATION  {model.name} / {topo.name} ===")
    print(f"plateaus         : {len(set(p.get('cluster') for p in pts))} "
          "independent (the bootstrap's resampling unit)")
    print(f"windows accepted : {len(pts):,}   "
          f"(rejected: {', '.join(f'{k} {v:,}' for k, v in rej.most_common(4))})")
    print(f"batch sizes      : n {int(n_all.min())} - {int(n_all.max())}  "
          f"(median {int(np.median(n_all))})")
    print(f"step bytes       : {by.min()/1e9:.1f} - {by.max()/1e9:.1f} GB  "
          f"-> leverage {lev:.2f}x")
    print(f"pass time source : "
          f"{Counter(p['t_src'] for p in pts).most_common()}")
    print(f"sum_L source     : "
          f"{Counter(p['L_src'] for p in pts).most_common()}")

    t0, eta = fit(pts, bw_adv)
    (t0lo, t0hi), (elo, ehi) = bootstrap(pts, bw_adv)
    tp_eff = S.tp_efficiency(args.tp, S.GPUS[args.gpu].nvlink_domain)

    print("\n-- fit  t_pass = t0 + bytes / (eta x bw_advertised) --")
    if lev < 1.5:
        print(f"!! LEVERAGE {lev:.2f}x IS TOO LOW TO IDENTIFY BOTH TERMS.")
        print("   Every (t0, eta) on a line through these points fits equally "
              "well; the split below is arbitrary, not measured.")
        print("   Fix: harvest windows spanning a wider batch range "
              "(decode_probe.py arm A, k = 1..64). Need >= 2x, want >= 3x.")
        naive = np.average([p["mbu_naive"] for p in pts],
                           weights=[p["steps"] for p in pts])
        print(f"   Reportable without the split: mean achieved bandwidth "
              f"{naive:.1%} of advertised, at n = {int(np.median(n_all))} "
              "(conflates t0 -- it is a point, not a model).")
    else:
        print(f"   t0  = {t0*1e3:8.2f} ms   [{t0lo*1e3:.2f}, {t0hi*1e3:.2f}]  "
              "per forward pass, batch-independent")
        print(f"   eta = {eta:8.3f}      [{elo:.3f}, {ehi:.3f}]  "
              f"of {bw_adv/1e12:.1f} TB/s advertised  (MBU)")
        print(f"         {eta/tp_eff:8.3f}      model convention "
              f"(/ tp_efficiency({args.tp}) = {tp_eff:.2f}), i.e. the number "
              "that replaces the study's implicit 1.00")
        print(f"   at n=1 the two terms are {t0/(t0 + by.min()/(eta*bw_adv)):.0%} "
              f"/ {1-t0/(t0 + by.min()/(eta*bw_adv)):.0%} of the pass; "
              f"at n={int(n_all.max())} they are "
              f"{t0/(t0 + by.max()/(eta*bw_adv)):.0%} / "
              f"{1-t0/(t0 + by.max()/(eta*bw_adv)):.0%}")

    # residuals against the UNCALIBRATED model -- the headline the user came for
    ratio = np.array([p["roofline_s"] / p["t_pass"] for p in pts])
    print(f"\n-- what the study currently predicts --")
    print(f"   measured pass / roofline pass: {1/np.median(ratio):.2f}x slower "
          f"(p5 {1/np.percentile(ratio,95):.2f}x, p95 {1/np.percentile(ratio,5):.2f}x)")
    print(f"   so every per-user tok/s figure at these batch sizes is that "
          "factor high, BEFORE the speculative slider.")

    spec = [p for p in pts if p["accepted_len"] > 1.0]
    if spec:
        al = np.array([p["accepted_len"] for p in spec])
        k = np.nanmedian([p["k_draft"] for p in spec])
        lo_n = [p["accepted_len"] for p in spec if p["n"] <= 2]
        hi_n = [p["accepted_len"] for p in spec if p["n"] >= 8]
        print(f"\n-- speculative decoding (measured, not fitted) --")
        print(f"   draft width k        : {k:.2f}")
        print(f"   accepted tokens/pass : {np.median(al):.3f}  "
              f"[p5 {np.percentile(al,5):.2f}, p95 {np.percentile(al,95):.2f}]")
        if lo_n:
            print(f"   at n <= 2            : {np.median(lo_n):.3f}   "
                  "<- what a lone user gets")
        if hi_n:
            print(f"   at n >= 8            : {np.median(hi_n):.3f}")
        a = (math.sqrt(max(4 * np.median(al) - 3, 0)) - 1) / 2
        print(f"   the study's mtp slider for this = {np.median(al):.2f}x "
              f"(alpha {a:.2f} under 1+a+a^2)")

        # The slider inverts speedup = 1 + a + a^2: accept-until-reject with
        # ONE acceptance rate at every draft position. vLLM counts acceptances
        # per position, so that assumption is directly falsifiable -- position
        # p is reached only if every earlier one was accepted, so a constant
        # alpha predicts A_p/D = alpha^(p+1).
        D = sum(p["drafts"] for p in spec)
        pos = defaultdict(float)
        for p in spec:
            for k_, v in (p.get("pos_accepted") or {}).items():
                pos[k_] += v
        if D > 0 and len(pos) >= 2:
            ks = sorted(pos, key=lambda x: int(x))
            rates = [pos[k_] / D for k_ in ks]
            print("   per-position acceptance: "
                  + ", ".join(f"pos{k_} {r:.3f}" for k_, r in zip(ks, rates)))
            pred = [rates[0] ** (i + 1) for i in range(len(rates))]
            print("   1+a+a^2 model predicts : "
                  + ", ".join(f"pos{k_} {q:.3f}" for k_, q in zip(ks, pred)))
            worst = max(abs(r - q) / max(q, 1e-9) for r, q in
                        zip(rates[1:], pred[1:]))
            print(f"   -> geometric acceptance {'HOLDS' if worst < 0.15 else 'FAILS'} "
                  f"(worst position off by {worst:.0%}); "
                  + ("the slider's alpha readout is meaningful"
                     if worst < 0.15 else
                     "the slider's alpha readout is a fiction -- quote the "
                     "measured speedup, not an implied acceptance"))

    arms = {p.get("arm") for p in pts if p.get("arm")}
    if "C" in arms:
        c = [p for p in pts if p.get("arm") == "C"]
        sh = [p["t_pass"] for p in c if p.get("shared_prefix_used")]
        un = [p["t_pass"] for p in c if not p.get("shared_prefix_used")]
        if sh and un:
            r = np.median(sh) / np.median(un)
            print(f"\n-- arm C: is a shared prefix read once or n times? --")
            print(f"   shared-prefix pass {np.median(sh)*1e3:.2f} ms vs "
                  f"unique-prompt pass {np.median(un)*1e3:.2f} ms  ({r:.2f}x)")
            print("   ~1.0x  => per-sequence reads, the study's sum_L is right"
                  if r > 0.9 else
                  "   <<1.0x => the shared prefix is read ONCE for the batch "
                  "(cascade attention): the study's KV term over-counts at "
                  "large n and this arm is the finding, not a nuisance.")

    if args.pool_tokens:
        pred = S.kv_pool_tokens(model, topo)
        print(f"\n-- KV pool cross-check --")
        print(f"   vLLM reports {args.pool_tokens:,.0f} tokens; "
              f"kv_pool_tokens() predicts {pred:,.0f} "
              f"({args.pool_tokens/pred:.2f}x)")
        print("   this is the study's BINDING ceiling; a 2x miss here moves "
              "every cache figure.")

    print("\n-- what this run does NOT establish --")
    print("   * eta and tp_efficiency are not separable at one TP width; the "
          "model-convention figure assumes the study's 0.90/doubling haircut.")
    print("   * t0 is this deployment's (its TP width, kernels, vLLM version, "
          "spec config) -- not a property of the model architecture.")
    print("   * acceptance measured here is THIS traffic's; it is a workload "
          "property and does not transplant to another prompt mix.")
    if any(p["L_src"] == "assumed" for p in pts):
        print("   * sum_L was assumed for some windows: the KV slope is "
              "unmeasured and folded into eta.")


if __name__ == "__main__":
    sys.exit(main())
