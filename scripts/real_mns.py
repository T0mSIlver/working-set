"""max_num_seqs decode tradeoff: per-user speed band + aggregate throughput vs concurrency,
using the real workload distribution (with a fitted log-normal overlay)."""
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
BPT=32*KIB; WEIGHTS=28.8*GIB; BW=4.8e12; MTP=1.7
SHARED_PREFIX=15_000; GPU_POOL_TOKENS=2.77e6; CAP=180_000; MIN_TOKENS=1000
MNS=np.arange(1,151); N_ITER=3000; MIN_TOK_S=[20,30,40]; RNG=np.random.default_rng(0)

U=DATA_DIR+os.sep
combined=np.concatenate([pd.read_csv(U+f"prompt_tokens_by_response_uid_{p}.csv")['prompt_tokens'].values
                         for p in ("igp","watsonx")])
combined=combined[combined>=MIN_TOKENS]; full_real=np.clip(combined,0,CAP).astype(float)
shape,_,scale=stats.lognorm.fit(combined,floc=0)
to_unique=lambda L: np.maximum(np.clip(L,0,CAP)-SHARED_PREFIX,0)
warm_once=lambda u:int(np.searchsorted(np.cumsum(u),GPU_POOL_TOKENS-SHARED_PREFIX))
cap5,cap50,cap95=np.percentile([warm_once(to_unique(RNG.choice(combined,1600,True))) for _ in range(4000)],[5,50,95]).astype(int)

def curve(sampler):
    a,b,c=[],[],[]
    for n in MNS:
        pu=MTP*BW/(WEIGHTS+sampler((N_ITER,n)).sum(1)*BPT)
        x,y,z=np.percentile(pu,[5,50,95]); a.append(x);b.append(y);c.append(z)
    return map(np.array,(a,b,c))
r5,r50,r95=curve(lambda sz:RNG.choice(full_real,sz,True))
l50=list(curve(lambda sz:np.clip(stats.lognorm.rvs(shape,scale=scale,size=sz,random_state=RNG),0,CAP)))[1]
pu=lambda n: MTP*BW/(WEIGHTS+n*full_real.mean()*BPT)
i=cap5-1; worst,med,best=r5[i],r50[i],r95[i]   # speed band at the recommended operating point

plt.rcParams.update({"figure.dpi":120,"font.size":9,"axes.grid":True,"grid.alpha":.3,
                     "axes.spines.top":False,"axes.spines.right":False})
GREEN,AMBER,RED,BLUE="#1f9d6b","#e08e0b","#c0392b","#2a78d6"
fig,(axL,axR)=plt.subplots(1,2,figsize=(14,6))

# ================= LEFT: per-user =================
axL.axvspan(cap5,cap95,color=GREEN,alpha=.07,zorder=0)
axL.fill_between(MNS,r5,r95,color=BLUE,alpha=.15)
axL.plot(MNS,r50,color=BLUE,lw=2.2,label="real: median speed")
axL.plot(MNS,r95,color=BLUE,lw=.8,ls=":",alpha=.7)
axL.plot(MNS,r5,color=BLUE,lw=.8,ls=":",alpha=.7)
axL.plot(MNS,l50,color=RED,lw=1.3,ls="--",alpha=.6,label="log-normal: median")
# three capacity verticals + median dots
for c,col in [(cap5,GREEN),(cap50,AMBER),(cap95,RED)]:
    axL.axvline(c,color=col,ls=":",lw=1.5,zorder=1)
    axL.plot(c,pu(c),"o",color=col,ms=8,zorder=6,markeredgecolor="white",markeredgewidth=1)
# best/worst error bar at the recommended (p5) point
axL.errorbar(cap5,med,yerr=[[med-worst],[best-med]],color=GREEN,elinewidth=2,capsize=6,capthick=2,zorder=5)
axL.annotate(f"best {best:.0f}",(cap5,best),xytext=(8,2),textcoords="offset points",color=GREEN,fontsize=8.5,fontweight="bold")
axL.annotate(f"worst {worst:.0f}",(cap5,worst),xytext=(8,-10),textcoords="offset points",color=GREEN,fontsize=8.5,fontweight="bold")
# capacity labels, spread out
axL.annotate(f"p5 cap = {cap5} seqs\nmedian {pu(cap5):.0f} tok/s",(cap5,med),xytext=(-12,46),
             textcoords="offset points",ha="right",color=GREEN,fontsize=8.5,fontweight="bold")
axL.annotate(f"p50 = {cap50}\n{pu(cap50):.0f} tok/s\n(~50% evict)",(cap50,pu(cap50)),xytext=(7,30),
             textcoords="offset points",ha="left",color=AMBER,fontsize=8.5,fontweight="bold")
axL.annotate(f"p95 = {cap95}\n{pu(cap95):.0f} tok/s\n(overflow)",(cap95,pu(cap95)),xytext=(8,-2),
             textcoords="offset points",ha="left",color=RED,fontsize=8.5,fontweight="bold")
for t in MIN_TOK_S: axL.axhline(t,ls="--",lw=.8,color="#bbb"); axL.text(2,t+1.2,f"{t} tok/s floor",fontsize=7,color="#999")
axL.text((cap5+cap95)/2,134,"capacity operating zone (p5\u2013p95)",ha="center",fontsize=8.5,color="#137a52")
axL.set_xlabel("max_num_seqs (concurrent decoders)"); axL.set_ylabel("per-user decode speed (tok/s)")
axL.set_title("Per-user speed  \u2014  band = best/worst over which sessions co-decode")
axL.set_ylim(0,145); axL.set_xlim(0,150)
axL.legend(frameon=True,framealpha=.92,edgecolor="#ddd",fontsize=8.5,loc="lower left")

# ================= RIGHT: aggregate =================
axR.axvspan(cap5,cap95,color=GREEN,alpha=.07,zorder=0)
axR.plot(MNS,MNS*r50/1000,color=BLUE,lw=2.2,label="real: aggregate")
axR.plot(MNS,MNS*l50/1000,color=RED,lw=1.3,ls="--",alpha=.6,label="log-normal")
agg=lambda n: n*pu(n)/1000
for c,col in [(cap5,GREEN),(cap50,AMBER),(cap95,RED)]:
    axR.axvline(c,color=col,ls=":",lw=1.5); axR.plot(c,agg(c),"o",color=col,ms=8,zorder=6,markeredgecolor="white",markeredgewidth=1)
axR.annotate(f"p5 = {cap5}\n{agg(cap5):.1f} ktok/s\nsafe",(cap5,agg(cap5)),xytext=(-14,-40),
             textcoords="offset points",ha="right",color=GREEN,fontsize=8.5,fontweight="bold",
             arrowprops=dict(arrowstyle="-",color=GREEN,lw=.8))
axR.annotate(f"p50 = {cap50}\n{agg(cap50):.1f} ktok/s\nmedian",(cap50,agg(cap50)),xytext=(-6,28),
             textcoords="offset points",ha="center",color=AMBER,fontsize=8.5,fontweight="bold")
axR.annotate(f"p95 = {cap95}\n{agg(cap95):.1f} ktok/s\noverflow",(cap95,agg(cap95)),xytext=(10,-6),
             textcoords="offset points",ha="left",color=RED,fontsize=8.5,fontweight="bold")
axR.set_xlabel("max_num_seqs (concurrent decoders)"); axR.set_ylabel("aggregate throughput (ktok/s)")
axR.set_title("Aggregate  \u2014  ~flat across the zone: the safe p5 costs ~nothing")
axR.set_xlim(0,150); axR.set_ylim(0,6.2)
axR.legend(frameon=True,framealpha=.92,edgecolor="#ddd",fontsize=8.5,loc="lower right")

fig.suptitle("max_num_seqs tradeoff  (real workload, 1\u00d7H200, FP8 KV, MTP2)",fontsize=13)
fig.tight_layout(); fig.savefig(out("real_mns_tradeoff.png"),bbox_inches="tight")
print(f"cap p5/p50/p95={cap5}/{cap50}/{cap95}  speed@p5: worst={worst:.0f} med={med:.0f} best={best:.0f}")
print("saved real_mns_tradeoff.png")
