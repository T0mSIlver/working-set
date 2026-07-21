"""Warm-capacity from REAL provider data: clean -> fit log-normal -> Monte-Carlo fill."""
import os
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats

# Data in ../data (override with DATA_DIR), figures to ../figures (override with OUT_DIR).
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(_HERE, "..", "data"))
OUT_DIR  = os.environ.get("OUT_DIR",  os.path.join(_HERE, "..", "figures"))
os.makedirs(OUT_DIR, exist_ok=True)
out = lambda name: os.path.join(OUT_DIR, name)

# ---- constants (same model/hardware as before) ----
KIB,MIB,GIB = 1024,1024**2,1024**3
BYTES_PER_TOKEN = 32*KIB
SHARED_PREFIX   = 15_000
DELTANET_STATE  = 75*MIB
GPU_POOL_TOKENS = 2.77e6
RAM_BUFFERS     = [0,200,600]
CAP             = 180_000
MIN_TOKENS      = 1_000          # drop junk below this
N_ITER, MAX_DRAW = 4000, 1600
RNG = np.random.default_rng(0)

# ---- load + clean ----
U = DATA_DIR + os.sep
igp = pd.read_csv(U+"prompt_tokens_by_response_uid_igp.csv")['prompt_tokens'].values
wx  = pd.read_csv(U+"prompt_tokens_by_response_uid_watsonx.csv")['prompt_tokens'].values
raw = {"IGP":igp, "WatsonX":wx}
clean = {k: v[v>=MIN_TOKENS] for k,v in raw.items()}
combined = np.concatenate(list(clean.values()))
for k in raw:
    print(f"{k:8}: {len(raw[k])} -> {len(clean[k])} kept (dropped {len(raw[k])-len(clean[k])} < {MIN_TOKENS})")
print(f"COMBINED cleaned n={len(combined)}  mean={combined.mean():.0f} median={np.median(combined):.0f} "
      f"p95={np.percentile(combined,95):.0f} p99={np.percentile(combined,99):.0f}")

# ---- fit log-normal to combined cleaned lengths (params from real data) ----
shape, loc, scale = stats.lognorm.fit(combined, floc=0)
ln_median, ln_mu = scale, np.log(scale)
print(f"\nFitted LogNormal: sigma={shape:.3f}  median={scale:.0f}  "
      f"mean={stats.lognorm.mean(shape,scale=scale):.0f}  "
      f"p95={stats.lognorm.ppf(.95,shape,scale=scale):.0f}  p99={stats.lognorm.ppf(.99,shape,scale=scale):.0f}")
print(f"%>140k real={np.mean(combined>140e3)*100:.1f}  lognorm={(1-stats.lognorm.cdf(140e3,shape,scale=scale))*100:.1f}")

def to_unique_tokens(lengths):
    return np.maximum(np.clip(lengths,0,CAP)-SHARED_PREFIX, 0)

def warm_once(unique_tok, ram):
    cs = np.cumsum(unique_tok)
    n_gpu = int(np.searchsorted(cs, GPU_POOL_TOKENS-SHARED_PREFIX))
    if ram==0: return n_gpu
    rem = unique_tok[n_gpu:]
    cb = np.cumsum(rem*BYTES_PER_TOKEN + DELTANET_STATE)
    n_cpu = int(np.searchsorted(cb, ram*GIB - SHARED_PREFIX*BYTES_PER_TOKEN))
    return n_gpu + n_cpu

def mc(sampler, ram):
    counts = np.array([warm_once(sampler(), ram) for _ in range(N_ITER)])
    return np.percentile(counts,[5,50,95])

real_sampler = lambda: to_unique_tokens(RNG.choice(combined, MAX_DRAW, replace=True))
ln_sampler   = lambda: to_unique_tokens(stats.lognorm.rvs(shape, scale=scale, size=MAX_DRAW, random_state=RNG))

# mean-based (old method) for comparison
mean_unique = to_unique_tokens(combined).mean()
def mean_based(ram):
    n_gpu = int((GPU_POOL_TOKENS-SHARED_PREFIX)//mean_unique)
    if ram==0: return n_gpu
    return n_gpu + int((ram*GIB - SHARED_PREFIX*BYTES_PER_TOKEN)//(mean_unique*BYTES_PER_TOKEN+DELTANET_STATE))

print(f"\n{'RAM':>6} | {'real p5/p50/p95':>22} | {'lognorm p5/p50/p95':>22} | {'mean-based':>10}")
res = {}
for ram in RAM_BUFFERS:
    r = mc(real_sampler, ram); l = mc(ln_sampler, ram); m = mean_based(ram)
    res[ram] = (r,l,m)
    print(f"{ram:>4}GB | {r[0]:6.0f}/{r[1]:6.0f}/{r[2]:6.0f}       | "
          f"{l[0]:6.0f}/{l[1]:6.0f}/{l[2]:6.0f}       | {m:>10}")

# ============ PLOTS ============
plt.rcParams.update({"figure.dpi":120,"font.size":9,"axes.grid":True,"grid.alpha":.3,
                     "axes.spines.top":False,"axes.spines.right":False})

# Plot 1: real histograms + fitted lognormal
fig,ax=plt.subplots(figsize=(9,5))
bins=np.linspace(0,230_000,60)
for k,c in [("IGP","#2a78d6"),("WatsonX","#e08e0b")]:
    ax.hist(clean[k],bins=bins,density=True,alpha=.45,color=c,label=f"{k} (real, n={len(clean[k])})")
xs=np.linspace(1,230_000,1500)
ax.plot(xs,stats.lognorm.pdf(xs,shape,scale=scale),color="#c0392b",lw=2.4,
        label=f"fitted LogNormal (median {scale/1e3:.0f}k, \u03c3={shape:.2f})")
ax.axvline(CAP/1e3*1000,color="#555",ls="--",lw=1.2); ax.text(CAP+2000,ax.get_ylim()[1]*.9,"cap 180k",rotation=90,fontsize=8,va="top")
ax.set_xlabel("prompt length (tokens)"); ax.set_ylabel("density")
ax.set_title("Real request lengths (cleaned) vs fitted log-normal")
ax.legend(frameon=False); ax.set_xlim(0,230_000)
fig.tight_layout(); fig.savefig(out("real_dist_fit.png"),bbox_inches="tight")

# Plot 2: MC warm capacity with p5/p50/p95 whiskers, real vs lognorm, mean-based marker
fig2,ax=plt.subplots(figsize=(9,5.4))
x=np.arange(len(RAM_BUFFERS)); w=0.34
for i,(lab,idx,col) in enumerate([("real data",0,"#2a78d6"),("fitted log-normal",1,"#c0392b")]):
    p50=[res[r][idx][1] for r in RAM_BUFFERS]
    lo=[res[r][idx][1]-res[r][idx][0] for r in RAM_BUFFERS]
    hi=[res[r][idx][2]-res[r][idx][1] for r in RAM_BUFFERS]
    b=ax.bar(x+(i-.5)*w,p50,w,yerr=[lo,hi],capsize=5,color=col,alpha=.85,label=lab,
             error_kw=dict(lw=1.3,ecolor="#333"))
    ax.bar_label(b,labels=[f"{v:.0f}" for v in p50],padding=3,fontsize=8)
mb=[res[r][2] for r in RAM_BUFFERS]
ax.plot(x,mb,"D",color="#1f9d6b",ms=8,label="old mean-based estimate",zorder=5)
ax.set_xticks(x); ax.set_xticklabels([f"{r} GB RAM" for r in RAM_BUFFERS])
ax.set_ylabel("warm users (p50, whiskers = p5\u2013p95)")
ax.set_title("Warm-cache capacity from real data (cap 180k, FP8 KV)\nMonte-Carlo fill: bar=median, whisker=p5\u2013p95 tail spread")
ax.legend(frameon=False)
fig2.tight_layout(); fig2.savefig(out("real_warm_mc.png"),bbox_inches="tight")
print("\nsaved real_dist_fit.png, real_warm_mc.png")
