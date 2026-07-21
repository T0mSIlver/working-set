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
                        ha="center", fontsize=8.5, color=col, fontweight="bold")
            ax.annotate(f"{p5:.0f}", (p5, y), xytext=(-7, 0), textcoords="offset points",
                        ha="right", va="center", fontsize=8, color=col)
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
            axL.plot(mns, p50, color=col, lw=2.2, label=TOPO_LABEL[tk])
            axL.fill_between(mns, p5, p95, color=col, alpha=.12)
        axR.plot(mns, agg / 1000, color=col, lw=2.2, label=TOPO_LABEL[tk])
    for t in (20, 30, 40):
        axL.axhline(t, ls="--", lw=.8, color="#bbb")
        axL.text(2, t + 1, f"{t} tok/s floor", fontsize=7, color=MUTED)
    axL.set_xlabel("max_num_seqs (concurrent decoders)")
    axL.set_ylabel("per-user decode speed (tok/s)")
    axL.set_title("Per-user speed  -  band = p5-p95  (DP2 per-user ≡ 1xH200)")
    axL.set_xlim(0, mns[-1]); axL.set_ylim(0, 260)
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


if __name__ == "__main__":
    if MODELS["35BA3B"].provisional:
        print("NOTE: 35B-A3B constants are PROVISIONAL (pending architecture research).")
    res = fig_capacity()
    fig_sysprompt()
    fig_mns()
    fig_subagent_invalidation()
    print("saved: scenario_capacity.png, scenario_sysprompt.png, scenario_mns.png, "
          "scenario_subagent_invalidation.png")
    print("\nwarm reusable p50 (0GB offload), reference workload:")
    for (mk, tk), (p5, p50, p95) in res.items():
        print(f"  {mk:7} {tk:12} {p5:5.0f} / {p50:5.0f} / {p95:5.0f}")
