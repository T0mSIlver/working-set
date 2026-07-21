"""
Warm-cache + max_num_seqs projections - Qwen3.6-27B on 1x H200, FP8 KV cache.

Two axes now:
  CAPACITY  (memory)  : how many sessions' KV fit warm (VRAM + CPU offload)  -> TTFT-on-return
  CONCURRENCY (mns)   : how many decode in one step, and the per-user/aggregate
                        tok/s tradeoff that follows                          -> decode speed

Per (distribution x max_seq_len x RAM):
  vram_cap     : sessions whose KV physically fits in the VRAM pool (memory ceiling)
  warm NGB     : total users kept warm (vram_cap + offloaded to N GiB of CPU RAM)  [mns-independent]
  decoding_now : min(max_num_seqs, vram_cap)                                       [mns caps this]
  per-user / aggregate tok/s as a function of how many decode at once
  recommended mns per per-user speed floor (20 / 30 / 40 tok/s)

Edit the HYPOTHESES block and re-run.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(_HERE, "..", "figures"))
os.makedirs(OUT_DIR, exist_ok=True)
out = lambda name: os.path.join(OUT_DIR, name)

# ============================================================
# FIXED CONSTANTS  (hardware + model + your measured numbers)
# ============================================================
KIB, MIB, GIB   = 1024, 1024**2, 1024**3
BYTES_PER_TOKEN = 32 * KIB        # FP8 KV: 16 attn layers x 4 KV heads x 256 x 2(K,V) x 1B
SHARED_PREFIX   = 15_000          # tokens, stored ONCE per tier (content-hash dedup)
DELTANET_STATE  = 75 * MIB        # per-session recurrent-state checkpoint (CPU side), approx
GPU_POOL_TOKENS = 2.77e6          # measured 1.337M FP16 tokens x2 (FP8) + freed activation budget
WEIGHTS_BYTES   = 28.8 * GIB      # FP8 model weights
HBM_BW          = 4.8e12          # H200 HBM3e bandwidth, bytes/s
MTP_SPEEDUP     = 1.7             # effective output tokens/step from MTP2 acceptance
FLOOR_TOKENS    = SHARED_PREFIX   # a prompt is at least the shared prefix

# ============================================================
# HYPOTHESES   <-- edit these freely
# ============================================================
DISTRIBUTIONS = [
    # (name,              mean,     std)   std=0 => everyone at that length
    ("all 180k (worst)",  180_000,      0),
    ("all 140k (worst)",  140_000,      0),
    ("normal 70k/30k",     70_000, 30_000),
    ("normal 60k/40k",     60_000, 40_000),
    ("normal 70k/20k",     70_000, 20_000),
    ("normal 50k/35k",     50_000, 35_000),
]
MAX_SEQ_LENS  = [140_000, 180_000]    # truncation cap (tokens)
RAM_BUFFERS   = [0, 200, 600]         # GiB of CPU offload (LMCache buffer)
MIN_TOK_S     = [20, 30, 40]          # per-user decode-speed floors to test
MNS_SWEEP     = np.arange(1, 101)     # max_num_seqs range for the tradeoff curves
DECODE_CAP    = 180_000               # which cap to use for the decode-speed plots
N_MC          = 1_000_000
RNG           = np.random.default_rng(0)

# ============================================================
def sample_lengths(mean, std, cap):
    raw = np.full(N_MC, float(mean)) if std == 0 else RNG.normal(mean, std, N_MC)
    clipped = np.clip(raw, FLOOR_TOKENS, cap)
    return clipped, float(np.mean(raw > cap)) * 100.0

def per_user_tok_s(n, mean_full):
    """tok/s seen by each of n concurrently-decoding sessions of mean_full context."""
    step_bytes = WEIGHTS_BYTES + n * mean_full * BYTES_PER_TOKEN
    return MTP_SPEEDUP * HBM_BW / step_bytes

def speed_ceiling(mean_full, floor_tok_s):
    """max sequences that can decode while keeping per-user tok/s >= floor (ignoring VRAM)."""
    budget = MTP_SPEEDUP * HBM_BW / floor_tok_s          # allowed step_bytes
    k = mean_full * BYTES_PER_TOKEN
    return max(int((budget - WEIGHTS_BYTES) // k), 0)

def compute(mean, std, cap):
    L, pct_trunc = sample_lengths(mean, std, cap)
    unique, mean_unique, mean_full = L - SHARED_PREFIX, (L - SHARED_PREFIX).mean(), L.mean()
    p50, p95, p99 = np.percentile(L, [50, 95, 99])

    prefix_bytes   = SHARED_PREFIX * BYTES_PER_TOKEN
    sess_cpu_bytes = mean_unique * BYTES_PER_TOKEN + DELTANET_STATE

    vram_cap = int((GPU_POOL_TOKENS - SHARED_PREFIX) // max(mean_unique, 1))   # memory ceiling

    warm = {}
    for ram in RAM_BUFFERS:
        warm[ram] = vram_cap if ram == 0 else \
            vram_cap + max(int((ram * GIB - prefix_bytes) // sess_cpu_bytes), 0)

    # decode speed at the memory ceiling, and the speed-only ceiling per floor
    pu_at_cap  = per_user_tok_s(vram_cap, mean_full)
    agg_at_cap = vram_cap * pu_at_cap
    spd_ceil   = {t: speed_ceiling(mean_full, t) for t in MIN_TOK_S}
    # effective recommended mns per floor = min(memory, speed); record which binds
    rec = {t: min(vram_cap, spd_ceil[t]) for t in MIN_TOK_S}
    binds = {t: ("VRAM" if vram_cap <= spd_ceil[t] else "speed") for t in MIN_TOK_S}

    return dict(p50=p50/1e3, p95=p95/1e3, p99=p99/1e3, pct_trunc=pct_trunc,
                sess_GiB=sess_cpu_bytes/GIB, mean_full=mean_full,
                vram_cap=vram_cap, warm=warm,
                pu_at_cap=pu_at_cap, agg_at_cap=agg_at_cap,
                spd_ceil=spd_ceil, rec=rec, binds=binds)

R = {}
for cap in MAX_SEQ_LENS:
    for name, mean, std in DISTRIBUTIONS:
        R[(cap, name)] = compute(mean, std, cap)

# ---------- TABLE 1: capacity (memory) ----------
for cap in MAX_SEQ_LENS:
    rows = [[n, R[(cap,n)]['p50'], R[(cap,n)]['p95'], R[(cap,n)]['p99'], R[(cap,n)]['pct_trunc'],
             R[(cap,n)]['sess_GiB'], R[(cap,n)]['vram_cap'],
             R[(cap,n)]['warm'][0], R[(cap,n)]['warm'][200], R[(cap,n)]['warm'][600]]
            for n,_,_ in DISTRIBUTIONS]
    df = pd.DataFrame(rows, columns=["scenario","p50(k)","p95(k)","p99(k)","%trunc","sess GiB",
                                     "VRAM cap","warm 0GB","warm 200GB","warm 600GB"])
    print(f"\n========  CAPACITY  |  max_seq_len = {cap//1000}k  ========")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.1f}"))

# ---------- TABLE 2: concurrency / decode speed (mns) ----------
for cap in MAX_SEQ_LENS:
    rows = []
    for n,_,_ in DISTRIBUTIONS:
        r = R[(cap,n)]
        rows.append([n, r['vram_cap'], r['pu_at_cap'], r['agg_at_cap']/1000,
                     r['spd_ceil'][40], r['spd_ceil'][30], r['spd_ceil'][20],
                     r['rec'][40], r['binds'][40]])
    df = pd.DataFrame(rows, columns=["scenario","VRAM cap","tok/s/user @cap","agg ktok/s @cap",
                                     "mns@40(spd)","mns@30(spd)","mns@20(spd)",
                                     "rec mns@40","binds@40"])
    print(f"\n========  CONCURRENCY  |  max_seq_len = {cap//1000}k  ========")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.1f}"))

# ============================================================
# PLOTS
# ============================================================
plt.rcParams.update({"figure.dpi":120,"font.size":9,"axes.grid":True,"grid.alpha":0.3,
                     "axes.spines.top":False,"axes.spines.right":False})
ram_colors = {0:"#9aa0a6",200:"#2a78d6",600:"#1f9d6b"}

# ---- Plot 1 & 2: warm capacity (unchanged) ----
fig, axes = plt.subplots(1, 2, figsize=(12,4.6), sharey=True)
for ax, cap in zip(axes, MAX_SEQ_LENS):
    names=[d[0] for d in DISTRIBUTIONS]; x=np.arange(len(names)); w=0.26
    for i,ram in enumerate(RAM_BUFFERS):
        b=ax.bar(x+(i-1)*w,[R[(cap,n)]['warm'][ram] for n in names],w,label=f"{ram} GB",color=ram_colors[ram])
        ax.bar_label(b,fontsize=7,padding=1)
    ax.set_title(f"max_seq_len = {cap//1000}k",fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels(names,rotation=30,ha="right",fontsize=8); ax.set_ylabel("users kept warm")
axes[0].legend(title="CPU offload",frameon=False)
fig.suptitle("Warm-held users by prompt-length hypothesis (mns-independent)",fontsize=12)
fig.tight_layout(); fig.savefig(out("warm_by_scenario.png"),bbox_inches="tight")

# ---- Plot 3: per-user tok/s vs max_num_seqs, per distribution ----
fig3, ax = plt.subplots(figsize=(9,5.4))
line_colors = plt.cm.viridis(np.linspace(0.1,0.85,len(DISTRIBUTIONS)))
for (name,_,_), c in zip(DISTRIBUTIONS, line_colors):
    r = R[(DECODE_CAP,name)]; mf=r['mean_full']; cap_n=r['vram_cap']
    pu = per_user_tok_s(MNS_SWEEP, mf)
    feas = MNS_SWEEP <= cap_n
    ax.plot(MNS_SWEEP[feas], pu[feas], color=c, lw=2, label=f"{name}  (VRAM cap {cap_n})")
    ax.plot(MNS_SWEEP[~feas], pu[~feas], color=c, lw=1.2, ls=":", alpha=0.6)  # infeasible region
    ax.plot(cap_n, per_user_tok_s(cap_n, mf), "o", color=c, ms=6)             # operating point
for t in MIN_TOK_S:
    ax.axhline(t, ls="--", lw=1, color="#c0392b", alpha=0.6)
    ax.text(98, t+0.7, f"{t} tok/s floor", fontsize=7, color="#c0392b", ha="right")
ax.set_xlabel("max_num_seqs  (concurrently decoding)"); ax.set_ylabel("per-user decode speed (tok/s)")
ax.set_title("Per-user speed vs max_num_seqs  (cap 180k, MTP2)\nsolid = fits in VRAM, dotted = exceeds VRAM, dot = operating point")
ax.set_ylim(0,120); ax.set_xlim(0,100); ax.legend(frameon=False,fontsize=8,loc="upper right")
fig3.tight_layout(); fig3.savefig(out("peruser_vs_mns.png"),bbox_inches="tight")

# ---- Plot 4: throughput tradeoff (twin axis) for one representative workload ----
rep = "normal 70k/30k"; r = R[(DECODE_CAP,rep)]; mf=r['mean_full']; cap_n=r['vram_cap']
fig4, axL = plt.subplots(figsize=(8.5,5))
pu  = per_user_tok_s(MNS_SWEEP, mf); agg = MNS_SWEEP*pu/1000
axR = axL.twinx()
l1,=axL.plot(MNS_SWEEP, pu, color="#2a78d6", lw=2.2, label="per-user tok/s (left)")
l2,=axR.plot(MNS_SWEEP, agg, color="#1f9d6b", lw=2.2, label="aggregate ktok/s (right)")
axL.axvspan(cap_n, 100, color="#bbb", alpha=0.18)
axL.axvline(cap_n, color="#555", lw=1.4, ls="--"); axL.text(cap_n+1, 110, f"VRAM cap = {cap_n}", fontsize=8)
crossover = WEIGHTS_BYTES/(mf*BYTES_PER_TOKEN)
axL.axvline(crossover, color="#e08e0b", lw=1.2, ls=":"); axL.text(crossover+1, 95, f"KV=weight\ncrossover \u2248{crossover:.0f}", fontsize=7, color="#b6790a")
for t in MIN_TOK_S: axL.axhline(t, ls="--", lw=0.8, color="#c0392b", alpha=0.5)
axL.set_xlabel("max_num_seqs"); axL.set_ylabel("per-user tok/s", color="#2a78d6")
axR.set_ylabel("aggregate ktok/s", color="#1f9d6b"); axL.set_ylim(0,120); axL.set_xlim(0,100)
axL.set_title(f"Throughput vs per-user-latency tradeoff  ({rep}, cap 180k)\nshaded = exceeds VRAM (infeasible)")
axL.legend(handles=[l1,l2], frameon=False, loc="center right")
fig4.tight_layout(); fig4.savefig(out("throughput_tradeoff.png"),bbox_inches="tight")
print("\nsaved: warm_by_scenario.png, peruser_vs_mns.png, throughput_tradeoff.png")
