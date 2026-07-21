"""Warm capacity as labeled dot-and-whisker (p5 / p50 / p95) from real data + fitted log-normal."""
import os
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(_HERE, "..", "data"))
OUT_DIR  = os.environ.get("OUT_DIR",  os.path.join(_HERE, "..", "figures"))
os.makedirs(OUT_DIR, exist_ok=True)
out = lambda name: os.path.join(OUT_DIR, name)

KIB,MIB,GIB=1024,1024**2,1024**3
BPT=32*KIB; SHARED_PREFIX=15_000; DELTANET_STATE=75*MIB; GPU_POOL_TOKENS=2.77e6
RAM_BUFFERS=[0,200,600]; CAP=180_000; MIN_TOKENS=1000; N_ITER,MAX_DRAW=4000,1600
RNG=np.random.default_rng(0)

U=DATA_DIR+os.sep
combined=np.concatenate([pd.read_csv(U+f"prompt_tokens_by_response_uid_{p}.csv")['prompt_tokens'].values
                         for p in ("igp","watsonx")])
combined=combined[combined>=MIN_TOKENS]
shape,_,scale=stats.lognorm.fit(combined,floc=0)
to_unique=lambda L: np.maximum(np.clip(L,0,CAP)-SHARED_PREFIX,0)

def warm_once(u,ram):
    n_gpu=int(np.searchsorted(np.cumsum(u),GPU_POOL_TOKENS-SHARED_PREFIX))
    if ram==0: return n_gpu
    rem=u[n_gpu:]; cb=np.cumsum(rem*BPT+DELTANET_STATE)
    return n_gpu+int(np.searchsorted(cb,ram*GIB-SHARED_PREFIX*BPT))
def mc(sampler,ram):
    return np.percentile([warm_once(sampler(),ram) for _ in range(N_ITER)],[5,50,95])
real_s=lambda: to_unique(RNG.choice(combined,MAX_DRAW,replace=True))
ln_s  =lambda: to_unique(stats.lognorm.rvs(shape,scale=scale,size=MAX_DRAW,random_state=RNG))

res={ram:{"real":mc(real_s,ram),"ln":mc(ln_s,ram)} for ram in RAM_BUFFERS}

plt.rcParams.update({"figure.dpi":120,"font.size":9,"axes.grid":True,"grid.alpha":.25,
                     "axes.spines.top":False,"axes.spines.right":False,"axes.spines.left":False})
C={"real":"#2a78d6","ln":"#c0392b"}; LAB={"real":"real data","ln":"fitted log-normal"}
fig,ax=plt.subplots(figsize=(11,5.4))
ybase={0:1,200:3,600:5}
for ram in RAM_BUFFERS:
    for k,off in [("real",0.22),("ln",-0.22)]:
        p5,p50,p95=res[ram][k]; y=ybase[ram]+off; c=C[k]
        ax.plot([p5,p95],[y,y],color=c,lw=2.4,zorder=2,solid_capstyle="round")
        for xp in (p5,p95): ax.plot([xp,xp],[y-.1,y+.1],color=c,lw=2.4,zorder=2)
        ax.plot(p50,y,"o",color=c,ms=11,zorder=3,markeredgecolor="white",markeredgewidth=1.2)
        ax.annotate(f"{p5:.0f}",(p5,y),xytext=(-7,0),textcoords="offset points",ha="right",va="center",fontsize=8.5,color=c,fontweight="bold")
        ax.annotate(f"{p95:.0f}",(p95,y),xytext=(7,0),textcoords="offset points",ha="left",va="center",fontsize=8.5,color=c,fontweight="bold")
        ax.annotate(f"{p50:.0f}",(p50,y),xytext=(0,11),textcoords="offset points",ha="center",fontsize=8.5,color=c,fontweight="bold")
ax.set_yticks(list(ybase.values())); ax.set_yticklabels([f"{r} GB\nCPU offload" for r in RAM_BUFFERS],fontsize=10)
ax.set_xlabel("prompts servable from cache  (KV resident, VRAM + offload)")
ax.set_xlim(0,900); ax.set_ylim(0.3,5.9)
# legend + percentile key
from matplotlib.lines import Line2D
leg=[Line2D([0],[0],color=C["real"],lw=3,label="real data"),Line2D([0],[0],color=C["ln"],lw=3,label="fitted log-normal")]
ax.legend(handles=leg,frameon=False,loc="lower right")
ax.set_title("How many prompts we can keep warm  (real workload, FP8 KV, cap 180k)\n"
             "left tick = p5 (eviction-safe floor)   \u2022   dot = p50 (median)   \u2022   right tick = p95 (lucky)",fontsize=11)
fig.tight_layout(); fig.savefig(out("warm_whisker.png"),bbox_inches="tight")
print("results:",{r:{k:list(np.round(v)) for k,v in res[r].items()} for r in RAM_BUFFERS})
print("saved warm_whisker.png")
