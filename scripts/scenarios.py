"""Canonical static figures for the extended scenario study.

Uses the shared model in scenario_model.py so every figure is consistent with
the interactive explorer. Writes PNGs into figures/ (override with OUT_DIR).

Scenarios covered:
  1. Warm capacity by model (27B vs 35B-A3B) x topology (1xH200 / TP2 / DP2)
  2. System-prompt-size sweep (3k / 15k / 30k) -- the "3k win"
  3. max_num_seqs decode tradeoff for 35B-A3B across topologies
  4. Subagent-ratio and cache-invalidation effects on warm reusable capacity
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scenario_model as M
from scenario_model import Workload, MODELS, TOPOLOGIES

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(_HERE, "..", "figures"))
os.makedirs(OUT_DIR, exist_ok=True)
out = lambda name: os.path.join(OUT_DIR, name)

# validated palette (light surface)
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
RED, GREEN, MUTED = "#d03b3b", "#0ca30c", "#898781"
plt.rcParams.update({"figure.dpi": 120, "font.size": 9, "axes.grid": True,
                     "grid.alpha": .28, "axes.spines.top": False,
                     "axes.spines.right": False})

MODEL_COLOR = {"27B": BLUE, "35BA3B": ORANGE}
TOPO_COLOR = {"1xH200": MUTED, "2xH200-TP2": BLUE, "2xH200-DP2": AQUA}
TOPO_LABEL = {"1xH200": "1xH200", "2xH200-TP2": "2xH200 TP", "2xH200-DP2": "2xH200 DP"}


def base_workload(**kw):
    """Reference workload: users median 31k/sigma .81, subagents 8k/sigma .9,
    1 subagent per 10 requests, 15k system prompt, 1% invalidation."""
    return Workload(**kw)


# ============================================================================
# FIG 1: warm reusable capacity  x (model, topology)
# ============================================================================
def fig_capacity():
    wl = base_workload()
    models = ["27B", "35BA3B"]
    topos = ["1xH200", "2xH200-TP2", "2xH200-DP2"]
    res = {(mk, tk): M.warm_capacity(MODELS[mk], TOPOLOGIES[tk], wl, ram_gib=0,
                                     n_iter=1500)
           for mk in models for tk in topos}

    fig, ax = plt.subplots(figsize=(10, 5.2))
    yrows = {tk: i for i, tk in enumerate(topos)}
    for mk, off in [("27B", 0.16), ("35BA3B", -0.16)]:
        col = MODEL_COLOR[mk]
        for tk in topos:
            p5, p50, p95 = res[(mk, tk)]
            y = yrows[tk] + off
            ax.plot([p5, p95], [y, y], color=col, lw=2.6, solid_capstyle="round", zorder=2)
            for xp in (p5, p95):
                ax.plot([xp, xp], [y - .06, y + .06], color=col, lw=2.6, zorder=2)
            ax.plot(p50, y, "o", color=col, ms=10, markeredgecolor="white",
                    markeredgewidth=1.2, zorder=3)
            ax.annotate(f"{p50:.0f}", (p50, y), xytext=(0, 10), textcoords="offset points",
                        ha="center", fontsize=8.5, color=col)
            # p5 is THE planning number (conservative tail) -> emphasized
            ax.annotate(f"p5 {p5:.0f}", (p5, y), xytext=(-7, 0), textcoords="offset points",
                        ha="right", va="center", fontsize=9, color=col, fontweight="bold")
            ax.annotate(f"{p95:.0f}", (p95, y), xytext=(7, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=8, color=col)
    ax.set_yticks(list(yrows.values()))
    ax.set_yticklabels([TOPO_LABEL[t] for t in topos], fontsize=10)
    ax.set_ylim(-0.6, len(topos) - 0.4)
    ax.set_xlabel("warm reusable sessions in one KV cache  (p5 - p50 - p95, 0 GB offload)")
    ax.set_xlim(left=0)
    from matplotlib.lines import Line2D
    leg = [Line2D([0], [0], color=MODEL_COLOR[m], lw=3, label=MODELS[m].name) for m in models]
    ax.legend(handles=leg, frameon=False, loc="lower right", fontsize=9)
    ax.set_title("Warm-cache capacity by model and topology\n"
                 "TP2 shares weights across both GPUs (pool >2x); DP2 keeps a per-replica cache "
                 "(returning users warm only on their home replica)", fontsize=11)
    fig.tight_layout(); fig.savefig(out("scenario_capacity.png"), bbox_inches="tight")
    plt.close(fig)
    return res


# ============================================================================
# FIG 2: system-prompt size sweep (the 3k win)
# ============================================================================
def fig_sysprompt():
    """The system-prompt tradeoff is two-sided: a bigger *shared* prefix dedups
    more (more warm sessions) but every cache miss must re-prefill it."""
    sizes = [3_000, 15_000, 30_000]
    mk = "35BA3B"
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))

    # LEFT: warm capacity rises with a bigger shared prefix (dedup), 1xH200 & TP2
    x = np.arange(len(sizes)); w = 0.34
    for i, tk in enumerate(["1xH200", "2xH200-TP2"]):
        p50s, los, his = [], [], []
        for s in sizes:
            p5, p50, p95 = M.warm_capacity(MODELS[mk], TOPOLOGIES[tk],
                                           base_workload(sys_user=s), n_iter=1500)
            p50s.append(p50); los.append(p50 - p5); his.append(p95 - p50)
        b = axL.bar(x + (i - .5) * w, p50s, w, yerr=[los, his], capsize=4,
                    color=TOPO_COLOR[tk], alpha=.9, label=TOPO_LABEL[tk],
                    error_kw=dict(lw=1.1, ecolor="#555"))
        axL.bar_label(b, labels=[f"{v:.0f}" for v in p50s], padding=3, fontsize=8)
    axL.set_xticks(x); axL.set_xticklabels([f"{s // 1000}k prefix" for s in sizes])
    axL.set_ylabel("warm reusable sessions (p50, whisker p5-p95)")
    axL.set_title("Upside: a bigger *shared* prefix dedups more\n"
                  "(stored once) -> more sessions kept warm")
    axL.legend(frameon=False, loc="upper left")

    # RIGHT: cost side -- expected re-prefill tokens per request due to the prefix,
    # paid on every cache miss (fraction f can't reuse it)
    xs = np.linspace(3_000, 30_000, 100)
    for f, col in [(0.01, BLUE), (0.05, ORANGE)]:
        axR.plot(xs / 1000, f * xs, color=col, lw=2.2, label=f"invalidation {f*100:.0f}%")
    for s in sizes:
        axR.axvline(s / 1000, ls=":", color=MUTED, lw=.8)
    axR.set_xlabel("shared system-prompt size (k tokens)")
    axR.set_ylabel("expected re-prefilled prefix tokens / request")
    axR.set_title("Cost: every cache miss re-prefills the whole prefix\n"
                  "miss tax = f x prefix -> a lean 3k prompt is cheap & robust")
    axR.legend(frameon=False, loc="upper left")
    axL.set_xlabel("shared system-prompt size")
    fig.suptitle(f"System-prompt size tradeoff  -  {MODELS[mk].name}", fontsize=12)
    fig.tight_layout(); fig.savefig(out("scenario_sysprompt.png"), bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# FIG 3: max_num_seqs decode tradeoff for 35B-A3B across topologies
# ============================================================================
def fig_mns():
    mk = "35BA3B"
    mns = np.arange(1, 121)
    wl = base_workload()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.6))
    for tk in ["1xH200", "2xH200-TP2", "2xH200-DP2"]:
        p5, p50, p95, agg = M.decode_curves(MODELS[mk], TOPOLOGIES[tk], wl, mns, n_iter=1500)
        col = TOPO_COLOR[tk]
        # per-user: DP2 is identical to 1xH200 (each replica is one GPU) -> skip it
        if tk != "2xH200-DP2":
            axL.plot(mns, p50, color=col, lw=2.2, label=TOPO_LABEL[tk] + " (linear union)")
            axL.fill_between(mns, p5, p95, color=col, alpha=.12)
            # optimistic bracket: expected expert union under uniform routing
            _, p50c, _, _ = M.decode_curves(MODELS[mk], TOPOLOGIES[tk], wl, mns,
                                            n_iter=1500, union="coverage")
            axL.plot(mns, p50c, color=col, lw=1.4, ls="--",
                     label=TOPO_LABEL[tk] + " (coverage union)")
        axR.plot(mns, agg / 1000, color=col, lw=2.2, label=TOPO_LABEL[tk])
    for t in (20, 30, 40):
        axL.axhline(t, ls="--", lw=.8, color="#bbb")
        axL.text(2, t + 1, f"{t} tok/s floor", fontsize=7, color=MUTED)
    axL.axvline(32, ls=":", color=MUTED, lw=1)
    axL.text(33, 5, "all 256 experts active\n(linear union, n=32)", fontsize=7, color=MUTED)
    axL.set_xlabel("max_num_seqs (concurrent decoders)")
    axL.set_ylabel("per-user decode speed (tok/s)")
    axL.set_title("Per-user speed  -  band = p5-p95  (DP2 per-user ≡ 1xH200)\n"
                  "solid = conservative no-overlap expert union; dashed = expected union")
    # cap the y-axis: the n=1 point is ~4.5 ktok/s and would crush the useful range
    axL.set_xlim(0, mns[-1]); axL.set_ylim(0, 700)
    axL.legend(frameon=False, loc="upper right")
    axR.set_xlabel("max_num_seqs (concurrent decoders)")
    axR.set_ylabel("aggregate throughput (ktok/s)  [DP = system total]")
    axR.set_title("Aggregate  -  DP2 doubles system throughput; TP2 speeds each user")
    axR.set_xlim(0, mns[-1]); axR.set_ylim(bottom=0)
    axR.legend(frameon=False, loc="lower right")
    fig.suptitle(f"max_num_seqs tradeoff  -  {MODELS[mk].name}  (mixed user+subagent workload)",
                 fontsize=13)
    fig.tight_layout(); fig.savefig(out("scenario_mns.png"), bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# FIG 4: subagent-ratio and cache-invalidation effects
# ============================================================================
def fig_subagent_invalidation():
    mk, tk = "35BA3B", "2xH200-TP2"
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))

    # left: warm p50 vs subagent ratio, for a few invalidation levels
    ratios = np.linspace(0, 1.0, 11)
    for f, col in [(0.0, AQUA), (0.01, BLUE), (0.05, ORANGE)]:
        p50s = []
        for r in ratios:
            wl = base_workload(sub_ratio=r, invalidation=f)
            _, p50, _ = M.warm_capacity(MODELS[mk], TOPOLOGIES[tk], wl, n_iter=900)
            p50s.append(p50)
        axL.plot(ratios, p50s, "o-", color=col, lw=2, ms=5, label=f"invalidation {f*100:.0f}%")
    axL.set_xlabel("subagent ratio (subagent requests per user request)")
    axL.set_ylabel("warm reusable sessions (p50)")
    axL.set_title("More (shorter) subagent requests raise warm session count\n"
                  "but a separate subagent prefix costs one extra shared block")
    axL.legend(frameon=False)
    axL.axvline(0.1, ls=":", color=MUTED, lw=1)
    axL.annotate("1 per 10", (0.1, axL.get_ylim()[0]), xytext=(4, 6),
                 textcoords="offset points", fontsize=8, color=MUTED)

    # right: warm p50 and hit-rate ceiling vs invalidation f
    fs = np.linspace(0, 0.10, 11)
    p50s = []
    for f in fs:
        wl = base_workload(invalidation=f)
        _, p50, _ = M.warm_capacity(MODELS[mk], TOPOLOGIES[tk], wl, n_iter=1200)
        p50s.append(p50)
    axR.plot(fs * 100, p50s, "o-", color=BLUE, lw=2, ms=5, label="warm reusable p50")
    axR.set_xlabel("cache-invalidation rate  (% of requests that match no KV)")
    axR.set_ylabel("warm reusable sessions (p50)", color=BLUE)
    axR.tick_params(axis="y", labelcolor=BLUE)
    axR.set_title("Unmatchable requests churn the pool and cap the hit rate\n"
                  "warm sessions fall; the achievable warm-hit ceiling is (1 - f)")
    for f in (0.01, 0.05):
        axR.axvline(f * 100, ls=":", color=MUTED, lw=1)
    axR.legend(frameon=False, loc="upper right")
    fig.tight_layout(); fig.savefig(out("scenario_subagent_invalidation.png"), bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# FIG 5: warm capacity vs number of H200s (TP one shared cache vs DP total)
# ============================================================================
def fig_scaling():
    """The planning view: warm sessions (p5 emphasized) as hardware scales.
    TP keeps ONE shared cache; DP's system total needs sticky routing."""
    wl = base_workload()
    ns = [1, 2, 3, 4, 6, 8]
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    for mk, col in [("35BA3B", ORANGE), ("27B", BLUE)]:
        tp5, tp50, tp95, dp5 = [], [], [], []
        for n in ns:
            it = max(200, 1200 // n)
            dr = 2000 + 1500 * n     # bigger pools hold more sessions than the default draw
            p5, p50, p95 = M.warm_capacity(MODELS[mk], M.topology("tp", n), wl,
                                           n_iter=it, draw=dr)
            tp5.append(p5); tp50.append(p50); tp95.append(p95)
            q5, _, _ = M.warm_capacity(MODELS[mk], M.topology("dp", n), wl, n_iter=it)
            dp5.append(q5 * n)
        ax.plot(ns, tp5, "o-", color=col, lw=2.4, ms=6,
                label=f"{MODELS[mk].name} — TP, one cache (p5)")
        ax.fill_between(ns, tp5, tp95, color=col, alpha=.10)
        ax.plot(ns, dp5, "s--", color=col, lw=1.5, ms=5, alpha=.75,
                label=f"{MODELS[mk].name} — DP system total (p5, sticky routing)")
        for n, v in zip(ns, tp5):
            ax.annotate(f"{v:.0f}", (n, v), xytext=(0, 9), textcoords="offset points",
                        ha="center", fontsize=8.5, color=col, fontweight="bold")
    ax.set_xlabel("number of H200 GPUs")
    ax.set_ylabel("warm sessions  (p5 = planning number; shading to p95)")
    ax.set_xticks(ns)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.set_title("Warm-session scaling with hardware\n"
                 "TP: weights stored once, one shared prefix cache. DP: cache splits — "
                 "total only reachable with session-sticky routing", fontsize=11)
    fig.tight_layout(); fig.savefig(out("scenario_scaling.png"), bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# FIG 6: the OTHER roofline — prefill duty cycle vs cache-miss rate
# ----------------------------------------------------------------------------
# Every other figure here answers "how many sessions fit?". This one answers
# "at what miss rate does fitting them stop being the question?" — the point
# where re-prefilling alone saturates the group and warm capacity is no longer
# the binding constraint. research/prefill.md.
# ============================================================================
def fig_prefill_thrash():
    CH, TURN, RATE = 32_768, 2_000, 2.13     # chunk, warm turn, 64 users @ 1/30s
    wl = base_workload()
    configs = [("27B", 1, 1, "H200", MUTED, "27B 1xH200"),
               ("27B", 1, 2, "H200", BLUE, "27B TP2"),
               ("35BA3B", 1, 2, "H200", ORANGE, "35B-A3B TP2"),
               ("MM35", 1, 4, "H200", RED, "Mistral-3.5 TP4"),
               ("GLM52", 1, 8, "H200", AQUA, "GLM-5.2 TP8")]
    fs = np.linspace(0, 0.5, 101)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.4, 5.0))

    # -- left: duty cycle vs miss rate ------------------------------------
    for mk, dp, tp, gk, col, lab in configs:
        m, t = MODELS[mk], M.topology_grid(dp, tp, gk)
        duty = [M.prefill_duty(m, t, base_workload(invalidation=f), RATE, CH, TURN)
                for f in fs]
        axL.plot(fs * 100, np.array(duty) * 100, color=col, lw=2, label=lab)
        fstar = M.breakeven_miss_rate(m, t, wl, RATE, CH, TURN)
        if 0 < fstar <= 0.5:
            axL.plot([fstar * 100], [100], "o", color=col, ms=7, zorder=5)
            # stagger the labels below the line: at 7% and 15% they would
            # otherwise sit on top of each other and on the threshold caption
            axL.annotate(f"{fstar:.0%}", (fstar * 100, 100), xytext=(0, -15),
                         textcoords="offset points", fontsize=8.5, ha="center",
                         color=col, fontweight="bold",
                         bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                   ec="none", alpha=.85))
    axL.axhline(100, color=RED, lw=1.4, ls="--")
    axL.text(49.4, 103, "prefill alone saturates the group", color=RED,
             fontsize=8.5, ha="right")
    axL.axvspan(0, 1, color=GREEN, alpha=.16)
    axL.annotate("reference f = 1%", (1, 4), xytext=(6, 0),
                 textcoords="offset points", fontsize=8, color=MUTED, va="bottom")
    axL.set_xlim(0, 50); axL.set_ylim(0, 185)
    axL.set_xlabel("cache-miss rate  f  (%)")
    axL.set_ylabel("share of the replica group spent prefilling (%)")
    axL.legend(frameon=False, fontsize=8.5, loc="upper left",
               bbox_to_anchor=(0.30, 1.0))
    axL.set_title(f"Prefill duty cycle at {RATE:.2f} req/s\n"
                  "(64 users, one turn every 30 s; warm turns included)",
                  fontsize=10.5)

    # -- right: what a miss costs everyone else ---------------------------
    labels, ratios, cols = [], [], []
    for mk, dp, tp, gk, col, lab in configs:
        m, t = MODELS[mk], M.topology_grid(dp, tp, gk)
        _, _, r = M.itl_spike(m, t, wl, 64, CH, n_iter=800)
        labels.append(lab); ratios.append(r); cols.append(col)
    y = np.arange(len(labels))
    axR.barh(y, ratios, color=cols, height=.62)
    for i, r in enumerate(ratios):
        axR.annotate(f"{r:.0f}x", (r, i), xytext=(4, 0), textcoords="offset points",
                     va="center", fontsize=9, fontweight="bold", color=cols[i])
    axR.set_yticks(y); axR.set_yticklabels(labels, fontsize=9)
    axR.invert_yaxis()
    axR.set_xlim(0, max(ratios) * 1.18)
    axR.set_xlabel("inter-token latency spike, x normal  (64 concurrent decoders)")
    axR.grid(axis="y", alpha=0)
    axR.set_title("What ONE cold chunk does to everyone else\n"
                  "a 32k prefill lands in the batch: every decoder waits a\n"
                  "prefill instead of a decode step", fontsize=10.5)

    fig.suptitle("Prefill is the other roofline — and the constraint the capacity "
                 "model cannot see", fontsize=12, y=1.005)
    fig.tight_layout()
    fig.savefig(out("scenario_prefill_thrash.png"), bbox_inches="tight")
    plt.close(fig)
    return {lab: M.breakeven_miss_rate(MODELS[mk], M.topology_grid(dp, tp, gk),
                                       wl, RATE, CH, TURN)
            for mk, dp, tp, gk, _, lab in configs}


def fig_cold_spike():
    """Cold-spike tolerance: the queue arrives long before the duty ceiling,
    and B* is where the MoE/dense gap compounds (research/spike.md)."""
    CH, TURN, RATE, SLA = 32_768, 2_000, 2.13, 10.0
    wl = base_workload()
    configs = [("27B", 1, 1, "H200", MUTED, "27B 1xH200"),
               ("27B", 1, 2, "H200", BLUE, "27B TP2"),
               ("35BA3B", 1, 2, "H200", ORANGE, "35B-A3B TP2"),
               ("MM35", 1, 4, "H200", RED, "Mistral-3.5 TP4"),
               ("GLM52", 1, 8, "H200", AQUA, "GLM-5.2 TP8")]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.4, 5.0))

    # -- left: TTFT vs miss rate, 27B TP2, both request classes -----------
    # the disciplines bracket vLLM from both sides and the bracket FLIPS by
    # class, so each class gets a band rather than a line
    m, t = MODELS["27B"], M.topology_grid(1, 2, "H200")
    fstar = M.breakeven_miss_rate(m, t, wl, RATE, CH, TURN)
    f_sla = M.sla_miss_rate(m, t, wl, RATE, CH, SLA, TURN)
    fs = np.linspace(0, fstar * 0.985, 160)
    curves = {}
    for cls in ("cold", "warm"):
        for disc in ("fcfs", "ps"):
            curves[cls, disc] = np.array([
                M.prefill_ttft_seconds(m, t, base_workload(invalidation=f),
                                       RATE, CH, TURN, request=cls,
                                       discipline=disc) for f in fs])
    for cls, col, lab in (("cold", BLUE, "a MISS waits"),
                          ("warm", ORANGE, "a HIT waits")):
        lo = np.minimum(curves[cls, "fcfs"], curves[cls, "ps"])
        hi = np.maximum(curves[cls, "fcfs"], curves[cls, "ps"])
        axL.fill_between(fs * 100, lo, hi, color=col, alpha=.22, lw=0)
        axL.plot(fs * 100, curves[cls, "fcfs"], color=col, lw=2, label=lab)
        axL.plot(fs * 100, curves[cls, "ps"], color=col, lw=1.2, ls=":")
    axL.axhline(SLA, color=RED, lw=1.4, ls="--")
    axL.text(0.4, SLA * 1.12, f"{SLA:.0f} s TTFT budget", color=RED, fontsize=8.5)
    axL.axvline(f_sla * 100, color=GREEN, lw=1.4, ls="--")
    axL.annotate(f"f_sla = {f_sla:.0%}\nlatency gone\n(duty only "
                 f"{M.prefill_duty(m, t, base_workload(invalidation=f_sla), RATE, CH, TURN):.0%})",
                 (f_sla * 100, 0.055), xytext=(-6, 0), textcoords="offset points",
                 fontsize=8.5, color=GREEN, ha="right", va="bottom")
    axL.axvline(fstar * 100, color=RED, lw=1.4)
    axL.annotate(f"f* = {fstar:.0%}\nduty 100%", (fstar * 100, 0.055),
                 xytext=(-6, 0), textcoords="offset points", fontsize=8.5,
                 color=RED, ha="right", va="bottom")
    axL.set_yscale("log")
    axL.set_xlim(0, fstar * 108); axL.set_ylim(0.05, 100)
    axL.set_xlabel("cache-miss rate  f  (%)")
    axL.set_ylabel("mean time to first token (s, log)")
    axL.legend(frameon=False, fontsize=8.5, loc="upper left")
    axL.set_title("27B TP2: the queue arrives before the ceiling\n"
                  "solid = FCFS, dotted = processor sharing; the band is the\n"
                  "scheduling bracket — and it flips sign by request class",
                  fontsize=10.5)

    # -- right: cold-spike tolerance vs miss rate -------------------------
    fs2 = np.linspace(0, 0.5, 151)
    for mk, dp, tp, gk, col, lab in configs:
        mm, tt = MODELS[mk], M.topology_grid(dp, tp, gk)
        b = [M.spike_tolerance(mm, tt, base_workload(invalidation=f), SLA,
                               RATE, CH, TURN) for f in fs2]
        axR.plot(fs2 * 100, np.maximum(b, 1e-3), color=col, lw=2, label=lab)
        b1 = M.spike_tolerance(mm, tt, wl, SLA, RATE, CH, TURN)
        axR.plot([1], [b1], "o", color=col, ms=6, zorder=5)
        axR.annotate(f"{b1:.1f}", (1, b1), xytext=(7, 0),
                     textcoords="offset points", fontsize=8.5, va="center",
                     color=col, fontweight="bold")
    axR.axhline(1, color=RED, lw=1.4, ls="--")
    axR.text(49.4, 1.1, "cannot absorb a SINGLE simultaneous miss", color=RED,
             fontsize=8.5, ha="right")
    axR.axvspan(0, 1, color=GREEN, alpha=.16)
    axR.set_yscale("log")
    axR.set_xlim(0, 50); axR.set_ylim(0.05, 120)
    axR.set_xlabel("standing cache-miss rate  f  (%)")
    axR.set_ylabel(f"cold-spike tolerance B*  (simultaneous misses, {SLA:.0f} s budget)")
    axR.legend(frameon=False, fontsize=8.5, loc="lower right")
    axR.set_title("How big a spike each deployment absorbs\n"
                  "B* is linear in the SLA, so another budget rescales every\n"
                  "curve and moves no ranking; each hits zero at its own f*",
                  fontsize=10.5)

    fig.suptitle("Cold-spike tolerance — the MoE's prefill advantage compounds "
                 "where it matters most", fontsize=12, y=1.005)
    fig.tight_layout()
    fig.savefig(out("scenario_cold_spike.png"), bbox_inches="tight")
    plt.close(fig)
    return {lab: M.spike_tolerance(MODELS[mk], M.topology_grid(dp, tp, gk),
                                   wl, SLA, RATE, CH, TURN)
            for mk, dp, tp, gk, _, lab in configs}


if __name__ == "__main__":
    res = fig_capacity()
    fig_sysprompt()
    fig_mns()
    fig_subagent_invalidation()
    fig_scaling()
    fstars = fig_prefill_thrash()
    spikes = fig_cold_spike()
    print("saved: scenario_capacity.png, scenario_sysprompt.png, scenario_mns.png, "
          "scenario_subagent_invalidation.png, scenario_scaling.png, "
          "scenario_prefill_thrash.png, scenario_cold_spike.png")
    print("\nbreakeven miss rate (prefill duty = 100% at 2.13 req/s):")
    for lab, f in fstars.items():
        print(f"  {lab:18} {f:6.0%}")
    print("\ncold-spike tolerance B* (10 s TTFT budget, f = 1%, 2.13 req/s):")
    for lab, b in spikes.items():
        print(f"  {lab:18} {b:6.1f} simultaneous misses")
    print("\nwarm reusable p50 (0GB offload), reference workload:")
    for (mk, tk), (p5, p50, p95) in res.items():
        print(f"  {mk:7} {tk:12} {p5:5.0f} / {p50:5.0f} / {p95:5.0f}")
