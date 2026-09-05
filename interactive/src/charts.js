import { CONFIG, is_moe } from './config.js';
import { PREFILL_CHUNK, coldRequestSeconds, decodeComfort, decodeFloor, mfuCeil, mfuEff,
         peakFlops, prefillChunk, prefillFlops, prefillOverheadSeconds, prefillSeconds } from './prefill.js';
import { clip } from './mathlib.js';
import { currentWL, state } from './state.js';
import { NS, cssv, esc, fmt, linScale, logScale, logTicks, niceTicks, niceTicksIn,
         svgEl } from './svg.js';
import { modelForCompare } from './render.js';

/* ---- Chart A: prompt-length PDFs ---- */
function lognormPdf(x, median, sigma){
  if (x<=0) return 0;
  const mu = Math.log(median);
  const z = (Math.log(x)-mu)/sigma;
  return Math.exp(-0.5*z*z)/(x*sigma*Math.sqrt(2*Math.PI));
}
export function renderChartA(){
  const W=560,H=300, mL=44,mR=14,mT=12,mB=40;
  const pw=W-mL-mR, ph=H-mT-mB;
  const wl=currentWL();
  // Frame on the DATA, not on the truncation cap: a 180k cap with a 31k median
  // left almost half the panel empty. p99 of the wider distribution, still
  // showing the cap when it lands inside the frame.
  const pct = (med, sig, z) => med*Math.exp(z*sig);
  // p95 of the wider distribution: past that the density is visually zero, and
  // framing on p99 (6.5x the median at sigma 0.81) left the panel half empty
  const dataMax = Math.max(pct(wl.user_median, wl.user_sigma, 1.645),
                           pct(wl.sub_median, wl.sub_sigma, 1.645));
  const xmax = Math.max(Math.min(dataMax, wl.cap)*1.08, 12000);
  const capIn = wl.cap <= xmax;
  const s1=cssv('--s1'), s2=cssv('--s2'), grid=cssv('--grid'), axis=cssv('--axis');
  const N=220;
  const xs=[], u=[], sub=[];
  let ymax=0;
  for(let i=0;i<=N;i++){
    const x = (i/N)*xmax;
    xs.push(x);
    const a=lognormPdf(x, wl.user_median, wl.user_sigma);
    const b=lognormPdf(x, wl.sub_median, wl.sub_sigma);
    u.push(a); sub.push(b);
    if(a>ymax)ymax=a; if(b>ymax)ymax=b;
  }
  ymax = ymax*1.08 || 1;
  const sx=linScale(0,xmax,mL,mL+pw);
  const sy=linScale(0,ymax,mT+ph,mT);
  let g='';
  // grid + x ticks (k)
  const xstep=xmax>500000?250000:(xmax>160000?50000:(xmax>80000?25000:10000));
  const xticks=[]; for(let t=0;t<xmax;t+=xstep) xticks.push(t);
  xticks.forEach(t=>{
    const X=sx(t);
    g+=`<line x1="${X}" y1="${mT}" x2="${X}" y2="${mT+ph}" stroke="${grid}" stroke-width="1"/>`;
    g+=`<text class="axtick" x="${X}" y="${mT+ph+15}" text-anchor="middle">${t/1000}k</text>`;
  });
  g+=`<line x1="${mL}" y1="${mT+ph}" x2="${mL+pw}" y2="${mT+ph}" stroke="${axis}" stroke-width="1"/>`;
  g+=`<text class="axlbl" x="${mL+pw/2}" y="${H-6}" text-anchor="middle">prompt length (tokens)</text>`;
  g+=`<text class="axlbl" transform="translate(12,${mT+ph/2}) rotate(-90)" text-anchor="middle">density</text>`;
  // area + line builder
  function path(arr, color){
    let dLine=`M ${sx(xs[0])} ${sy(arr[0])}`;
    for(let i=1;i<arr.length;i++) dLine+=` L ${sx(xs[i])} ${sy(arr[i])}`;
    let dArea=dLine+` L ${sx(xs[xs.length-1])} ${sy(0)} L ${sx(xs[0])} ${sy(0)} Z`;
    return `<path d="${dArea}" fill="${color}" opacity="0.12"/><path d="${dLine}" fill="none" stroke="${color}" stroke-width="2" stroke-linejoin="round"/>`;
  }
  g+=path(u,s1);
  g+=path(sub,s2);
  // the numbers every other panel is computed from, marked in place: capacity
  // follows the MEAN, and the p95 is the tail that drives the heavy-miss cost
  [[wl.user_median, wl.user_sigma, s1, 'user'],
   [wl.sub_median, wl.sub_sigma, s2, 'subagent']].forEach(([med,sig,col,who])=>{
    const mean = med*Math.exp(sig*sig/2), p95 = med*Math.exp(1.645*sig);
    [[med,'median'],[mean,'mean'],[p95,'p95']].forEach(([v,lab],k)=>{
      if (v > xmax) return;
      const X = sx(v);
      g+=`<line x1="${X}" y1="${mT+ph}" x2="${X}" y2="${mT+ph-9}" stroke="${col}" stroke-width="1.6"/>`;
      // staggered: median and mean sit within a few k of each other at the
      // study's sigma, so a single row of labels overlaps
      if (who === 'user')
        g+=`<text class="axtick" x="${X}" y="${mT+ph-13-k*11}" text-anchor="middle" fill="${col}">${lab} ${fmt(v/1000,0)}k</text>`;
    });
  });
  // cap line — only when the truncation cap falls inside the framed range;
  // otherwise say where it is rather than drawing it off-plot
  if (capIn){
    const cx=sx(wl.cap);
    g+=`<line x1="${cx}" y1="${mT}" x2="${cx}" y2="${mT+ph}" stroke="${cssv('--crit')}" stroke-width="1.5" stroke-dasharray="4 3"/>`;
    g+=`<text class="dlabel" x="${cx-4}" y="${mT+11}" text-anchor="end" fill="${cssv('--crit')}">cap ${Math.round(wl.cap/1000)}k</text>`;
  } else {
    g+=`<text class="dlabel" x="${mL+pw-2}" y="${mT+11}" text-anchor="end" fill="${cssv('--muted')}">max_seq_len cap ${Math.round(wl.cap/1000)}k — beyond this range</text>`;
  }
  document.getElementById('chartA').innerHTML = svgEl(g,W,H,'Prompt-length distribution: log-normal densities for user and subagent prompts with the max_seq_len cap');
}

/* ---- Chart B: warm capacity (p5) vs number of H200s, TP vs DP ---- */
export function renderChartB(sc){
  // sc: {ns, tp5, tp95, dp5, cur:{n, split, p5system}} — dp5 = SYSTEM total (sticky)
  const W=560,H=300, mL=52,mR=16,mT=16,mB=40;
  const pw=W-mL-mR, ph=H-mT-mB;
  const grid=cssv('--grid'), axis=cssv('--axis'), s1=cssv('--s1'), s3=cssv('--s3'), muted=cssv('--muted');
  let ymax=0;
  sc.tp95.forEach(v=>{ if(v>ymax)ymax=v; });
  sc.dp5.forEach(v=>{ if(v>ymax)ymax=v; });
  ymax=Math.max(ymax*1.12,10);
  const nmax=sc.ns[sc.ns.length-1];
  // round the top up to the last nice tick, so the axis ends on a readable
  // number instead of on the data's own maximum
  const bTicks = niceTicks(ymax, 5);
  ymax = Math.max(ymax, bTicks[bTicks.length-1]);
  const sx=linScale(1,nmax,mL,mL+pw), sy=linScale(0,ymax,mT+ph,mT);
  let g='';
  for(const v of bTicks){
    const Y=sy(v);
    g+=`<line x1="${mL}" y1="${Y}" x2="${mL+pw}" y2="${Y}" stroke="${grid}" stroke-width="1"/>`;
    g+=`<text class="axtick" x="${mL-8}" y="${Y+3}" text-anchor="end">${fmt(v,0)}</text>`;
  }
  sc.ns.forEach(n=>{
    g+=`<text class="axtick" x="${sx(n)}" y="${mT+ph+15}" text-anchor="middle">${n}</text>`;
  });
  g+=`<line x1="${mL}" y1="${mT+ph}" x2="${mL+pw}" y2="${mT+ph}" stroke="${axis}" stroke-width="1"/>`;
  g+=`<text class="axlbl" x="${mL+pw/2}" y="${H-5}" text-anchor="middle">number of ${CONFIG.GPUS[state.gpu].name} GPUs</text>`;
  g+=`<text class="axlbl" transform="translate(13,${mT+ph/2}) rotate(-90)" text-anchor="middle">warm sessions (p5)</text>`;
  // TP band p5..p95 + p5 line (the planning series)
  let dband=`M ${sx(sc.ns[0])} ${sy(sc.tp5[0])}`;
  for(let i=1;i<sc.ns.length;i++) dband+=` L ${sx(sc.ns[i])} ${sy(sc.tp5[i])}`;
  for(let i=sc.ns.length-1;i>=0;i--) dband+=` L ${sx(sc.ns[i])} ${sy(sc.tp95[i])}`;
  g+=`<path d="${dband} Z" fill="${s1}" opacity="0.10"/>`;
  let dtp=`M ${sx(sc.ns[0])} ${sy(sc.tp5[0])}`;
  for(let i=1;i<sc.ns.length;i++) dtp+=` L ${sx(sc.ns[i])} ${sy(sc.tp5[i])}`;
  g+=`<path d="${dtp}" fill="none" stroke="${s1}" stroke-width="2.4" stroke-linejoin="round"/>`;
  let ddp=`M ${sx(sc.ns[0])} ${sy(sc.dp5[0])}`;
  for(let i=1;i<sc.ns.length;i++) ddp+=` L ${sx(sc.ns[i])} ${sy(sc.dp5[i])}`;
  g+=`<path d="${ddp}" fill="none" stroke="${s3}" stroke-width="1.8" stroke-dasharray="5 4" stroke-linejoin="round"/>`;
  sc.ns.forEach((n,i)=>{
    g+=`<circle cx="${sx(n)}" cy="${sy(sc.tp5[i])}" r="4" fill="${s1}" stroke="${cssv('--surface')}" stroke-width="1.5"/>`;
    // the annotated point gets its value from the "you" callout instead, so
    // the two labels stop stacking in the same few pixels
    if (!(sc.cur && n === sc.cur.n))
      g+=`<text class="dlabel" x="${sx(n)}" y="${sy(sc.tp5[i])-9}" text-anchor="middle" fill="${s1}">${fmt(sc.tp5[i],0)}</text>`;
    g+=`<circle cx="${sx(n)}" cy="${sy(sc.dp5[i])}" r="3.2" fill="${s3}" stroke="${cssv('--surface')}" stroke-width="1.2"/>`;
  });
  // marker for the current configuration, offset up-and-right with a leader so
  // it clears the axis lines at n = 1
  if (sc.cur){
    const cy=sy(sc.cur.p5system), cx=sx(sc.cur.n);
    const lx=Math.min(cx+34, mL+pw-52), ly=Math.max(cy-30, mT+12);
    g+=`<line x1="${cx+7}" y1="${cy-5}" x2="${lx-4}" y2="${ly+3}" stroke="${cssv('--muted')}" stroke-width="1"/>`;
    g+=`<circle cx="${cx}" cy="${cy}" r="8" fill="none" stroke="${cssv('--text')}" stroke-width="1.6"/>`;
    g+=`<text class="dlabel" x="${lx}" y="${ly}" fill="${cssv('--text')}">you · ${fmt(sc.cur.p5system,0)}</text>`;
  }
  document.getElementById('chartB').innerHTML = svgEl(g,W,H,'Warm sessions (p5) versus number of GPUs, tensor-parallel and data-parallel');
}

// Round, human-readable x ticks for a dynamic 1..nMax axis (~4 intervals).
function xTicks(nMax){
  const raw=nMax/4;
  const pow=Math.pow(10, Math.floor(Math.log10(raw)));
  const step=([1,2,2.5,5,10].find(m=>raw<=m*pow) ?? 10)*pow;
  const out=[1];
  for(let t=step;t<=nMax+1e-9;t+=step){
    const v=Math.round(t);
    if(v-out[out.length-1] >= step*0.6) out.push(v);   // never crowd the "1"
  }
  return out;
}

/* ---- Chart C: per-user decode with band + capacity zone ---- */
// A zero pool means the weights don't fit the configuration at all — a decode
// curve there prices reads from weights that were never loaded. Charts C and D
// show this empty state instead (the tiles already say "—"), and null geometry
// switches the hover layer off.
export function renderNoFit(divId, what){
  // collapse to a message-sized box: a 300-tall empty card holding one
  // sentence reads as a rendering failure rather than an answer
  const W=560,H=120, curve = what || 'decode curve';
  const g=`<text x="${W/2}" y="${H/2-8}" text-anchor="middle" class="axlbl" font-size="13">model weights do not fit this configuration</text>`+
          `<text x="${W/2}" y="${H/2+12}" text-anchor="middle" class="axtick">no ${curve} — add GPUs, or switch GPU / weight dtype</text>`;
  document.getElementById(divId).innerHTML = svgEl(g,W,H,`Model weights do not fit this configuration; no ${curve}`);
}
export let chartCGeom=null;
export function renderChartC(dc, zone, stress, steady){
  const W=560,H=300, mL=46,mR=16,mT=14,mB=40;
  const pw=W-mL-mR, ph=H-mT-mB;
  // axis ends at the last SAMPLED n, so the curve reaches the right edge and
  // the hover clamp (which also uses the last sample) agrees with the axis
  const xmax=dc.ns[dc.ns.length-1];
  // LOG y-scale. Per-user speed is a roofline hyperbola: once the axis reaches
  // the all-warm-decoding point it spans two orders of magnitude (~4,500 tok/s
  // at n=1 vs ~34 at n=800 on TP2), and a linear axis presses everything that
  // matters — the 40/50 floors, the capacity zone — flat onto the baseline.
  let yhi=0, ylo=Infinity;
  dc.p95.forEach(v=>{ if(v>yhi)yhi=v; });
  dc.p5.forEach(v=>{ if(v<ylo)ylo=v; });
  // always keep both threshold markers in frame, whatever the curve does
  ylo=Math.min(ylo, 38)/1.15;
  yhi=Math.max(yhi, 60)*1.10;
  const sx=linScale(1,xmax,mL,mL+pw);
  const sy=logScale(ylo,yhi,mT+ph,mT);
  const s1=cssv('--s1'), s3=cssv('--s3'), grid=cssv('--grid'), axis=cssv('--axis'), muted=cssv('--muted');
  let g='';
  // capacity operating zone (vertical span from warm p5..p95 of current topo)
  if(zone){
    const zx0=sx(clip(zone.p5,1,xmax)), zx1=sx(clip(zone.p95,1,xmax));
    if(zx1>zx0+0.5){
      g+=`<rect x="${zx0}" y="${mT}" width="${zx1-zx0}" height="${ph}" fill="${s3}" opacity="0.13"/>`;
      g+=`<text class="axtick" x="${(zx0+zx1)/2}" y="${mT+11}" text-anchor="middle" fill="${s3}">capacity zone</text>`;
    }
  }
  // y grid (log decades, 1-2-5 or finer when the range is narrow)
  logTicks(ylo,yhi).forEach(v=>{
    const Y=sy(v);
    g+=`<line x1="${mL}" y1="${Y}" x2="${mL+pw}" y2="${Y}" stroke="${grid}" stroke-width="1"/>`;
    g+=`<text class="axtick" x="${mL-8}" y="${Y+3}" text-anchor="end">${fmt(v,0)}</text>`;
  });
  // x ticks (dynamic range)
  xTicks(xmax).forEach(t=>{
    const X=sx(t);
    g+=`<text class="axtick" x="${X}" y="${mT+ph+15}" text-anchor="middle">${t}</text>`;
  });
  g+=`<line x1="${mL}" y1="${mT+ph}" x2="${mL+pw}" y2="${mT+ph}" stroke="${axis}" stroke-width="1"/>`;
  g+=`<text class="axlbl" x="${mL+pw/2}" y="${H-5}" text-anchor="middle">max_num_seqs</text>`;
  g+=`<text class="axlbl" transform="translate(12,${mT+ph/2}) rotate(-90)" text-anchor="middle">per-user tok/s (log)</text>`;
  // decode-speed thresholds: the hard floor is the slider (40 tok/s by
  // default — the study's value, which every published figure uses), and the
  // comfortable mark tracks it at 1.25x (50 against 40). Planning happens
  // against the floor.
  [[decodeFloor(),'hard floor'],[decodeComfort(),'comfortable']].forEach(([f,what])=>{
    if(f>=ylo && f<=yhi){ const Y=sy(f);
      g+=`<line x1="${mL}" y1="${Y}" x2="${mL+pw}" y2="${Y}" stroke="${muted}" stroke-width="1" stroke-dasharray="2 3" opacity="0.7"/>`;
      g+=`<text class="axtick" x="${mL+3}" y="${Y-3}" text-anchor="start" fill="${muted}">${fmt(f,0)} tok/s — ${what}</text>`;
    }
  });
  // band p5..p95
  let up=`M ${sx(dc.ns[0])} ${sy(dc.p95[0])}`;
  for(let i=1;i<dc.ns.length;i++) up+=` L ${sx(dc.ns[i])} ${sy(dc.p95[i])}`;
  let dn=` L ${sx(dc.ns[dc.ns.length-1])} ${sy(dc.p5[dc.p5.length-1])}`;
  for(let i=dc.ns.length-2;i>=0;i--) dn+=` L ${sx(dc.ns[i])} ${sy(dc.p5[i])}`;
  g+=`<path d="${up+dn} Z" fill="${s1}" opacity="0.20"/>`;
  // p50 line
  let ln=`M ${sx(dc.ns[0])} ${sy(dc.p50[0])}`;
  for(let i=1;i<dc.ns.length;i++) ln+=` L ${sx(dc.ns[i])} ${sy(dc.p50[i])}`;
  g+=`<path d="${ln}" fill="none" stroke="${s1}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>`;
  // THE stress point: every GPU-resident p5-warm session decoding at once. This
  // is the tile's headline number and the docs' v@warm column, so the axis is
  // sized to always contain it — mark it where it lands on the p50 curve.
  if(stress && stress.n>=1 && stress.n<=xmax){
    const X=sx(stress.n), Y=sy(clip(stress.pu,ylo,yhi));
    g+=`<line x1="${X}" y1="${mT}" x2="${X}" y2="${mT+ph}" stroke="${s3}" stroke-width="1.2" stroke-dasharray="4 3"/>`;
    g+=`<circle cx="${X}" cy="${Y}" r="4.5" fill="${s3}" stroke="${cssv('--surface')}" stroke-width="1.5"/>`;
    const flip = X > mL+pw*0.62;    // keep the label inside the plot
    g+=`<text class="dlabel" x="${X+(flip?-8:8)}" y="${Y-9}" text-anchor="${flip?'end':'start'}" fill="${s3}">all warm decoding · ${fmt(stress.pu,0)} tok/s</text>`;
  }
  // ...and the point the CHOSEN LOAD actually sits at (steadyDecodePoint): the
  // stress marker's honest counterpart, normally far to its left. Labelled
  // BELOW the dot — at small n the curve runs along the top of the axis, where
  // an above-the-dot label would leave the plot.
  if(steady && steady.real && steady.n>0){
    const s2=cssv('--s2'), n=Math.max(1, Math.min(steady.n, xmax));
    const X=sx(n), Y=sy(clip(steady.pu,ylo,yhi));
    g+=`<line x1="${X}" y1="${mT}" x2="${X}" y2="${mT+ph}" stroke="${s2}" stroke-width="1.2" stroke-dasharray="1 3"/>`;
    g+=`<circle cx="${X}" cy="${Y}" r="4.5" fill="${s2}" stroke="${cssv('--surface')}" stroke-width="1.5"/>`;
    const flip = X > mL+pw*0.62;
    g+=`<text class="dlabel" x="${X+(flip?-8:8)}" y="${Y+16}" text-anchor="${flip?'end':'start'}" fill="${s2}">`
      +`your load · n ≈ ${fmt(steady.n, steady.n<10?1:0)} · ${fmt(steady.pu,0)} tok/s</text>`;
  }
  document.getElementById('chartC').innerHTML = svgEl(g,W,H,'Per-user decode speed versus max_num_seqs with capacity zone, the all-warm stress point and the steady-state point at the chosen load');
  // ylog tells the hover to interpolate GEOMETRICALLY between samples, matching
  // the straight segments actually drawn on a log axis (see interpAt)
  chartCGeom={ W,H,mL,mR,mT,mB,pw,ph,xmax, ylo,yhi, ylog:true, sxDom:[1,xmax], dc,
    sx:(x)=>sx(x), sy:(y)=>sy(y) };
}

/* ---- Chart D: aggregate ---- */
export let chartDGeom=null;
export function clearChartGeomCD(){ chartCGeom=null; chartDGeom=null; }
export function renderChartD(dc, stress, kink, steady){
  const W=560,H=300, mL=54,mR=16,mT=14,mB=40;
  const pw=W-mL-mR, ph=H-mT-mB;
  const xmax=dc.ns[dc.ns.length-1];   // dynamic, same range as chart C
  let ymax=0; dc.agg.forEach(v=>{ if(v>ymax)ymax=v; });
  ymax=Math.max(ymax*1.1, 1);
  const sx=linScale(1,xmax,mL,mL+pw);
  const dTicks = niceTicks(ymax, 5);
  ymax = Math.max(ymax, dTicks[dTicks.length-1]);
  const sy=linScale(0,ymax,mT+ph,mT);
  const s3=cssv('--s3'), grid=cssv('--grid'), axis=cssv('--axis');
  let g='';
  for(const v of dTicks){
    const Y=sy(v);
    g+=`<line x1="${mL}" y1="${Y}" x2="${mL+pw}" y2="${Y}" stroke="${grid}" stroke-width="1"/>`;
    const lab = v>=1000 ? fmt(v/1000, v%1000?1:0)+'k' : fmt(v,0);
    g+=`<text class="axtick" x="${mL-8}" y="${Y+3}" text-anchor="end">${lab}</text>`;
  }
  xTicks(xmax).forEach(t=>{
    const X=sx(t);
    g+=`<text class="axtick" x="${X}" y="${mT+ph+15}" text-anchor="middle">${t}</text>`;
  });
  g+=`<line x1="${mL}" y1="${mT+ph}" x2="${mL+pw}" y2="${mT+ph}" stroke="${axis}" stroke-width="1"/>`;
  g+=`<text class="axlbl" x="${mL+pw/2}" y="${H-5}" text-anchor="middle">max_num_seqs</text>`;
  g+=`<text class="axlbl" transform="translate(13,${mT+ph/2}) rotate(-90)" text-anchor="middle">aggregate tok/s</text>`;
  // expert-union kink (MoE only): from n_sat on, every routed expert is
  // already read each step — the per-step weight read stops growing, so the
  // aggregate curve's slope visibly breaks here. Drawn behind the line.
  if(kink && kink>1 && kink<xmax){
    const X=sx(kink), mut=cssv('--muted');
    g+=`<line x1="${X}" y1="${mT}" x2="${X}" y2="${mT+ph}" stroke="${mut}" stroke-width="1" stroke-dasharray="2 3" opacity="0.65"/>`;
    g+=`<text class="axtick" x="${X+5}" y="${mT+10}" text-anchor="start" fill="${mut}">expert-union kink · n = ${kink}</text>`;
  }
  let ln=`M ${sx(dc.ns[0])} ${sy(dc.agg[0])}`;
  for(let i=1;i<dc.ns.length;i++) ln+=` L ${sx(dc.ns[i])} ${sy(dc.agg[i])}`;
  g+=`<path d="${ln}" fill="none" stroke="${s3}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>`;
  // same stress point as chart C: system throughput when all warm decode at once
  if(stress && stress.n>=1 && stress.n<=xmax){
    const X=sx(stress.n), Y=sy(clip(stress.agg,0,ymax));
    g+=`<line x1="${X}" y1="${mT}" x2="${X}" y2="${mT+ph}" stroke="${s3}" stroke-width="1.2" stroke-dasharray="4 3" opacity="0.75"/>`;
    g+=`<circle cx="${X}" cy="${Y}" r="4.5" fill="${s3}" stroke="${cssv('--surface')}" stroke-width="1.5"/>`;
    const flip = X > mL+pw*0.62;
    g+=`<text class="dlabel" x="${X+(flip?-8:8)}" y="${Y-9}" text-anchor="${flip?'end':'start'}" fill="${s3}">all warm decoding · ${fmt(stress.agg/1000,1)} ktok/s</text>`;
  }
  // the chosen load's own point. On THIS chart it is a flow balance made
  // visible: the y value is the output-token demand the load generates, and
  // the x value is the batch size the curve needs to retire exactly that.
  if(steady && steady.real && steady.n>0){
    const s2=cssv('--s2'), n=Math.max(1, Math.min(steady.n, xmax));
    const X=sx(n), Y=sy(clip(steady.demanded,0,ymax));
    g+=`<line x1="${X}" y1="${mT}" x2="${X}" y2="${mT+ph}" stroke="${s2}" stroke-width="1.2" stroke-dasharray="1 3" opacity="0.85"/>`;
    g+=`<circle cx="${X}" cy="${Y}" r="4.5" fill="${s2}" stroke="${cssv('--surface')}" stroke-width="1.5"/>`;
    const flip = X > mL+pw*0.62;
    g+=`<text class="dlabel" x="${X+(flip?-8:8)}" y="${Y-9}" text-anchor="${flip?'end':'start'}" fill="${s2}">`
      +`your load · ${steady.demanded>=1000?fmt(steady.demanded/1000,1)+' ktok/s':fmt(steady.demanded,0)+' tok/s'}</text>`;
  }
  document.getElementById('chartD').innerHTML = svgEl(g,W,H,'Aggregate throughput versus max_num_seqs, with the all-warm stress point and the chosen load\'s steady-state demand');
  chartDGeom={ W,H,mL,mR,mT,mB,pw,ph,xmax,ymax, dc, sx:(x)=>sx(x), sy:(y)=>sy(y) };
}

/* ---- Chart E: the max_num_batched_tokens trade (replaces the old slider —
   a slider showed one operating point; the trade is a curve) ---- */
const CHUNK_LO = 2048, CHUNK_HI = 65536;   // the retired slider's range
export let chartEGeom=null, lastChartE=null;
// One sample per eighth-octave so the 32,768 default lands exactly on the
// grid (i = 32). Everything here is analytic given the shared context draw,
// so the sweep costs ~41 closed forms + 41 mean-pass scans per render.
export function chartEData(model, topo, wl, cs, stress){
  const Cs=[], mfu=[], cold=[], spike=[];
  const stepS = (stress && stress.pu > 0) ? model.mtp / stress.pu : 0;
  for(let i=0;i<=40;i++){
    const C = Math.round(Math.pow(2, 11 + i/8));
    Cs.push(C);
    mfu.push(mfuEff(model, topo, C));
    cold.push(1/coldRequestSeconds(model, topo, wl, cs, C));
    // the spike prices the chunk's MARGINAL FLOPs (its host pass streams the
    // weights for the decode batch anyway), mid-re-prefill (prior = E[L]/2);
    // past C = E[L] the "chunk" is one whole-context pass — same clamp as the
    // tile and itlSpikeRatio, so the dot and the tile cannot disagree
    const step = Math.min(C, cs.mean), prior = step < C ? 0 : cs.mean/2;
    spike.push(stepS > 0
      ? (stepS + prefillSeconds(model, topo, step, prior)) / stepS : 0);
  }
  return {Cs, mfu, cold, spike, hasSpike: stepS > 0};
}
// linear interpolation on the drawn segments (x is log-spaced, so screen-x
// linear == linear in ln C — exactly what the polyline shows)
function interpE(data, key, C){ return interpChunk(data[key], C); }
export function renderChartE(data){
  lastChartE = data;
  if(!data){
    renderNoFit('chartE', 'chunk-size trade');
    chartEGeom=null;
    document.getElementById('ttE').style.opacity=0;  // clear a stale hover
    return;
  }
  const W=560, H=404, mL=52, mR=14, mT=20, sh=92, sg=32, mB=34;
  const pw=W-mL-mR;
  const grid=cssv('--grid'), axis=cssv('--axis'), muted=cssv('--muted');
  const surface=cssv('--surface');
  const sx=logScale(CHUNK_LO, CHUNK_HI, mL, mL+pw);
  // three stacked strips, one measure each — same knob, three consequences
  // (a dual axis is how this would go wrong as a single panel)
  // These two panels used to be pinned to zero, which flattened both curves
  // into apparently horizontal lines — while the caption asserted that they
  // FALL. Framing them on their own range is the only way the chart can show
  // what it claims; the axis label says it does not start at zero so nobody
  // reads the slope as bigger than it is.
  const frame = (arr, pad) => {
    const lo = Math.min(...arr), hi = Math.max(...arr), span = (hi-lo) || Math.abs(hi) || 1;
    return { lo: Math.max(0, lo - span*pad), hi: hi + span*pad };
  };
  const fMfu = frame(data.mfu, 0.28), fCold = frame(data.cold, 0.28);
  const strips=[
    { y0:mT,            key:'mfu',   color:cssv('--s1'), lo:fMfu.lo, hi:fMfu.hi,
      title:'effective MFU of a first chunk  (axis does not start at zero)',
      nticks:3, tfmt:v=>fmt(v*100,0)+'%', dfmt:v=>fmt(v*100,0)+'%' },
    { y0:mT+sh+sg,      key:'cold',  color:cssv('--s2'), lo:fCold.lo, hi:fCold.hi,
      title:'max cold req/s per replica group  (axis does not start at zero)',
      nticks:3, tfmt:v=>fmt(v,2), dfmt:v=>fmt(v,2)+' req/s' },
    { y0:mT+2*(sh+sg),  key:'spike', color:cssv('--s3'), lo:1,
      hi:Math.max(...data.spike, 2)*1.08,
      title:'ITL spike every decoder sees (× one inter-token gap)',
      nticks:3, tfmt:v=>'×'+fmt(v,0), dfmt:v=>'×'+fmt(v,0) },
  ];
  let g='';
  // x grid: powers of two, labels under the bottom strip only
  const xt=[2048,4096,8192,16384,32768,65536];
  const yBot = mT + 3*sh + 2*sg;
  xt.forEach(t=>{
    const X=sx(t);
    strips.forEach(s=>{
      if(s.key==='spike' && !data.hasSpike) return;  // no half-empty panel
      g+=`<line x1="${X}" y1="${s.y0}" x2="${X}" y2="${s.y0+sh}" stroke="${grid}" stroke-width="1"/>`;
    });
    g+=`<text class="axtick" x="${X}" y="${yBot+15}" text-anchor="middle">${t/1024}k</text>`;
  });
  // dot = the SELECTED chunk (what every tile and the deploy recipe price);
  // dashed line = the study's 32,768 default. They coincide at the default.
  const CH=prefillChunk(), XD=sx(CH), XRef=sx(PREFILL_CHUNK);
  for(const s of strips){
    if(s.key==='spike' && !data.hasSpike) continue;
    const sy=linScale(s.lo, s.hi, s.y0+sh, s.y0);
    const tks = s.ticks || niceTicksIn(s.lo, s.hi, s.nticks || 4);
    tks.forEach(v=>{
      const Y=sy(v);
      g+=`<line x1="${mL}" y1="${Y}" x2="${mL+pw}" y2="${Y}" stroke="${grid}" stroke-width="1"/>`;
      g+=`<text class="axtick" x="${mL-8}" y="${Y+3}" text-anchor="end">${s.tfmt(v)}</text>`;
    });
    g+=`<line x1="${mL}" y1="${s.y0+sh}" x2="${mL+pw}" y2="${s.y0+sh}" stroke="${axis}" stroke-width="1"/>`;
    g+=`<text class="axlbl" x="${mL}" y="${s.y0-10}" text-anchor="start">${s.title}</text>`;
    let ln=`M ${sx(data.Cs[0])} ${sy(clip(data[s.key][0],s.lo,s.hi))}`;
    for(let i=1;i<data.Cs.length;i++)
      ln+=` L ${sx(data.Cs[i])} ${sy(clip(data[s.key][i],s.lo,s.hi))}`;
    g+=`<path d="${ln}" fill="none" stroke="${s.color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>`;
    // the operating point every tile is priced at, direct-labeled
    const vD=interpE(data, s.key, CH), YD=sy(clip(vD,s.lo,s.hi));
    g+=`<circle cx="${XD}" cy="${YD}" r="4" fill="${s.color}" stroke="${surface}" stroke-width="1.5"/>`;
    g+=`<text class="dlabel" x="${XD-8}" y="${YD-7}" text-anchor="end" fill="${s.color}">${s.dfmt(vD)}</text>`;
    s.sy=sy;
  }
  // the 32,768 default, across all strips (the selected chunk carries the dots)
  g+=`<line x1="${XRef}" y1="${mT}" x2="${XRef}" y2="${yBot}" stroke="${muted}" stroke-width="1.2" stroke-dasharray="4 3" opacity="0.8"/>`;
  g+=`<text class="axtick" x="${XRef+5}" y="${mT+10}" text-anchor="start" fill="${muted}">default 32,768</text>`;
  g+=`<text class="axlbl" x="${mL+pw/2}" y="${H-4}" text-anchor="middle">max_num_batched_tokens (log)</text>`;
  document.getElementById('chartE').innerHTML =
    svgEl(g,W,H,'Effective MFU, max cold request rate and ITL spike versus max_num_batched_tokens');
  chartEGeom={ W,H,mL,mR,mT,pw, ph:yBot-mT, data, strips,
    sx:(x)=>sx(x) };
}
// bespoke hover: chart E's x is logarithmic and its crosshair spans three
// strips, so setupHover's linear invert does not apply
(function hoverE(){
  const box=document.getElementById('chartE'), tt=document.getElementById('ttE');
  function handler(e){
    const geom=chartEGeom; if(!geom) return;
    const svg=box.querySelector('svg'); if(!svg) return;
    const rect=svg.getBoundingClientRect();
    const vx=(e.clientX-rect.left)/rect.width*geom.W;
    if(vx<geom.mL || vx>geom.mL+geom.pw){ tt.style.opacity=0; removeCross(svg); chartE2Hover(null); return; }
    const C=Math.round(clip(
      Math.exp(Math.log(CHUNK_LO) + (vx-geom.mL)/geom.pw*Math.log(CHUNK_HI/CHUNK_LO)),
      CHUNK_LO, CHUNK_HI));
    const px=geom.sx(C), d=geom.data;
    const marks=geom.strips.filter(s=>s.sy)
      .map(s=>({v:interpE(d,s.key,C), color:s.color, r:3.5,
                y:s.sy(clip(interpE(d,s.key,C), s.lo, s.hi))}));
    drawCross(svg, px, geom, marks);
    const rows=geom.strips.filter(s=>s.sy).map(s=>
      `<div class="row"><span class="sw" style="background:${s.color}"></span>`
      +`${esc(s.title.split(' (')[0])} <b class="tnum">${s.dfmt(interpE(d,s.key,C))}</b></div>`).join('');
    tt.innerHTML=`<div class="tth tnum">chunk = ${fmt(C,0)} tok</div>`+rows;
    const par=tt.offsetParent||box, parRect=par.getBoundingClientRect();
    const scale=rect.width/geom.W;
    const cx=(rect.left-parRect.left)+px*scale;
    const cy=(rect.top-parRect.top)+(marks.length?marks[0].y:geom.mT)*scale;
    const tw=tt.offsetWidth, th=tt.offsetHeight, pad=4, gap=12;
    let left=cx+gap;
    if(left+tw>parRect.width-pad) left=cx-gap-tw;
    let top=cy-th-gap;
    if(top<pad) top=cy+gap;
    tt.style.left=Math.max(pad,Math.min(left,parRect.width-tw-pad))+'px';
    tt.style.top =Math.max(pad,Math.min(top, parRect.height-th-pad))+'px';
    tt.style.opacity=1;
    chartE2Hover(C);          // E2's live third bar tracks this cursor
  }
  function leave(){
    tt.style.opacity=0; const svg=box.querySelector('svg'); if(svg) removeCross(svg);
    chartE2Hover(null);
  }
  box.addEventListener('mousemove', handler);
  box.addEventListener('mouseleave', leave);
  box.addEventListener('touchmove',(e)=>{ if(e.touches[0]) handler(e.touches[0]); },{passive:true});
  box.addEventListener('touchend', leave); box.addEventListener('touchcancel', leave);
})();

/* ---- Charts E2-E4: chart E's three strips, taken apart -------------------
   Same closed forms as chart E, three more angles on the one knob. E2
   anatomises one miss-side pass (why the top strip falls), E3 draws the
   middle strip for every model at once (who pays for small chunks), E4 shows
   the bottom strip as a decoder lives it (the chunk sets the texture of the
   stall, not its total). Nothing here prices a tile: the tiles, the deploy
   card and the harness read prefillChunk() through the functions above.
   ------------------------------------------------------------------------- */
const TEXTURE_CS = [2048, 8192, 32768];   // E4's rows

// Pass times (s) of ONE mean-length miss chunked at C — MARGINAL pricing (the
// interleaved chunk rides a decode pass that streams the weights regardless:
// research/prefill.md's pricing boundary), prior starting at 0 (a miss
// re-prefills from empty), so later passes carry more cross-attention and the
// last FULL chunk is the worst freeze. The sum telescopes -> chunk-invariant,
// which is E4's whole point. The worst freeze here therefore differs slightly
// from chart E's spike, which prices one midpoint chunk (prior = E[L]/2)
// rather than the max over the real passes.
function chunkPassTimes(m, topo, context, C){
  const ts=[]; let done=0;
  while(done<context){
    const step=Math.min(C, context-done);
    ts.push(prefillSeconds(m, topo, step, done));
    done+=step;
  }
  return ts;
}
// value at C read off chart E's eighth-octave grid (i = 0..40 from 2^11);
// interpE above is this over one of chart E's series
function interpChunk(vals, C){
  const f=clip((Math.log2(C)-11)*8, 0, 40);
  const i=Math.min(Math.floor(f), 39), t=f-i;
  return vals[i]+t*(vals[i+1]-vals[i]);
}
function renderChartMsg(divId, msg, W){
  W=W||560; const H=90;
  document.getElementById(divId).innerHTML=
    svgEl(`<text x="${W/2}" y="${H/2+4}" text-anchor="middle" class="axtick">${esc(msg)}</text>`, W, H, msg);
}

// ---- E2: anatomy of one MISS-side pass. A fixed weight-stream toll plus the
// compute (GEMM + intra-chunk attention) that grows with the chunk: the same gray bytes in every bar, the chunk only
// choosing how much math amortises them. Rows: 2k, the 32k default, the priced
// chunk when a share link pins another, and a live bar that follows the
// cursor on chart E or E3 (chartE2Hover). ----
let chartE2State=null;
function renderChartE2(model, topo, hoverC){
  const el=document.getElementById('chartE2'); if(!el) return;
  let ceil;
  try{ ceil=mfuCeil(model, topo); }
  catch(e){ renderChartMsg('chartE2','MFU anchor unsolvable on this topology — no pass anatomy'); chartE2State=null; return; }
  const toll=prefillOverheadSeconds(model, topo);
  chartE2State={model, topo};
  const pf=peakFlops(topo), CH=prefillChunk();
  const rows=[{C:2048},{C:PREFILL_CHUNK}];
  if(CH!==2048 && CH!==PREFILL_CHUNK) rows.push({C:CH, priced:true});
  const slots=rows.length+1;                  // one slot reserved for the hover bar
  if(hoverC) rows.push({C:Math.round(hoverC), hover:true});
  rows.forEach(r=>{ r.g=prefillFlops(model, r.C, 0)/(pf*ceil); r.t=r.g+toll; });
  // the scale spans the largest chunk the hover can reach, so the bars never
  // rescale under the cursor
  const Tmax=Math.max(...rows.map(r=>r.t), prefillFlops(model, CHUNK_HI, 0)/(pf*ceil)+toll);
  const W=560, mL=52, mR=134, mT=16, rh=24, rgap=16;
  const pw=W-mL-mR, H=mT+slots*(rh+rgap)+24;
  const sw=linScale(0, Tmax, 0, pw);
  const muted=cssv('--muted'), s1=cssv('--s1');
  let g='';
  rows.forEach((r,i)=>{
    const y=mT+i*(rh+rgap);
    g+=`<text class="axtick" x="${mL-8}" y="${y+rh/2+3}" text-anchor="end">${r.C>=1024?fmt(r.C/1024,r.C%1024?1:0)+'k':r.C}</text>`;
    g+=`<rect x="${mL}" y="${y}" width="${Math.max(1,sw(toll))}" height="${rh}" fill="${muted}" opacity="0.6"/>`;
    g+=`<rect x="${mL+sw(toll)+1}" y="${y}" width="${Math.max(1,sw(r.g)-1)}" height="${rh}" fill="${s1}" opacity="0.85"/>`;
    const tp=100*toll/r.t;
    const tag=r.hover?' (hover)':r.priced?' (priced)':'';
    g+=`<text class="axtick" x="${mL+sw(r.t)+7}" y="${y+rh/2+3}" text-anchor="start">${fmt(r.t*1000,0)} ms, toll ${fmt(tp,tp<10?1:0)}%${tag}</text>`;
  });
  if(!hoverC){
    const y=mT+(slots-1)*(rh+rgap);
    g+=`<text class="axtick" x="${mL}" y="${y+rh/2+3}" fill="${muted}">hover chart E or E3 — a third bar tracks the cursor</text>`;
  }
  const yAx=mT+slots*(rh+rgap)-rgap+6;
  niceTicksIn(0, Tmax*1000, 5).forEach(v=>{
    g+=`<text class="axtick" x="${mL+sw(v/1000)}" y="${yAx+8}" text-anchor="middle">${fmt(Math.abs(v)<1e-9?0:v,0)}</text>`;
  });
  g+=`<text class="axlbl" x="${mL+pw/2}" y="${H-2}" text-anchor="middle">one miss-side pass, ms — the gray toll never changes</text>`;
  el.innerHTML=svgEl(g, W, H,
    'One prefill pass as a fixed weight-stream toll plus GEMM time that grows with the chunk');
}
function chartE2Hover(C){
  if(chartE2State) renderChartE2(chartE2State.model, chartE2State.topo, C);
}

// ---- E3: every model's cold ceiling vs chunk, normalised to its own value at
// the 32,768 default. Priced through modelForCompare so each row carries the
// selected GPU, weight and KV dtypes and its OWN mtp, exactly as the frontier
// does; a model whose MFU anchor cannot be solved on this topology is left
// out. Colour is the architecture (dense vs MoE) — the argument the panel
// makes — and the left-edge label is the identity. ----
function chartE3Data(topo, wl, cs){
  const Cs=[]; for(let i=0;i<=40;i++) Cs.push(Math.round(Math.pow(2, 11+i/8)));
  const models=[];
  for(const key of Object.keys(CONFIG.MODELS)){
    const m=modelForCompare(key);
    let ref;
    try{ ref=coldRequestSeconds(m, topo, wl, cs, PREFILL_CHUNK); }
    catch(e){ continue; }
    // meanPasses is memoised on `cs`, so the seven sweeps share chart E's scans
    models.push({key, dense:!is_moe(m), sel:key===state.model,
                 rel:Cs.map(C=>ref/coldRequestSeconds(m, topo, wl, cs, C))});
  }
  return {Cs, models};
}
let chartE3Geom=null;
function renderChartE3(d){
  const el=document.getElementById('chartE3'), tt=document.getElementById('ttE3');
  if(!d || !d.models.length){
    renderChartMsg('chartE3','the MFU anchor solves for no model on this topology');
    chartE3Geom=null; tt.style.opacity=0; return;
  }
  const W=560, H=220, mL=52, mR=14, mT=22, mB=34;
  const pw=W-mL-mR, ph=H-mT-mB;
  const grid=cssv('--grid'), axis=cssv('--axis'), muted=cssv('--muted');
  const sx=logScale(CHUNK_LO, CHUNK_HI, mL, mL+pw);
  // ratios can top 100% past the default (a mean context just over 32k drops
  // from two passes to one at 64k), so the frame follows the data both ways
  const all=d.models.flatMap(m=>m.rel), lo0=Math.min(...all), hi0=Math.max(...all);
  const hi=Math.max(1.04, hi0+(hi0-lo0)*0.10), lo=Math.max(0, lo0-(hi0-lo0)*0.10);
  const sy=linScale(lo, hi, mT+ph, mT);
  let g='';
  [2048,4096,8192,16384,32768,65536].forEach(t=>{
    const X=sx(t);
    g+=`<line x1="${X}" y1="${mT}" x2="${X}" y2="${mT+ph}" stroke="${grid}" stroke-width="1"/>`;
    g+=`<text class="axtick" x="${X}" y="${mT+ph+15}" text-anchor="middle">${t/1024}k</text>`;
  });
  niceTicksIn(lo, hi, 4).forEach(v=>{
    const Y=sy(v);
    g+=`<line x1="${mL}" y1="${Y}" x2="${mL+pw}" y2="${Y}" stroke="${grid}" stroke-width="1"/>`;
    g+=`<text class="axtick" x="${mL-8}" y="${Y+3}" text-anchor="end">${fmt(v*100,0)}%</text>`;
  });
  g+=`<line x1="${mL}" y1="${mT+ph}" x2="${mL+pw}" y2="${mT+ph}" stroke="${axis}" stroke-width="1"/>`;
  g+=`<text class="axlbl" x="${mL}" y="${mT-10}" text-anchor="start">cold-request ceiling, % of the model's own 32k value</text>`;
  const XD=sx(PREFILL_CHUNK);
  g+=`<line x1="${XD}" y1="${mT}" x2="${XD}" y2="${mT+ph}" stroke="${muted}" stroke-width="1.2" stroke-dasharray="4 3" opacity="0.8"/>`;
  g+=`<text class="axtick" x="${XD+5}" y="${mT+10}" text-anchor="start" fill="${muted}">default 32,768</text>`;
  const series=d.models.map(m=>({...m, color:cssv(m.dense?'--s1':'--s2'), w:m.sel?2.6:1.6}));
  // unselected first, so the selected line sits on top of the bundle
  for(const s of series.slice().sort((a,b)=>(a.sel?1:0)-(b.sel?1:0))){
    let ln=`M ${sx(d.Cs[0])} ${sy(clip(s.rel[0],lo,hi))}`;
    for(let i=1;i<d.Cs.length;i++) ln+=` L ${sx(d.Cs[i])} ${sy(clip(s.rel[i],lo,hi))}`;
    g+=`<path d="${ln}" fill="none" stroke="${s.color}" stroke-width="${s.w}" stroke-linecap="round" stroke-linejoin="round"/>`;
  }
  // left-edge direct labels, de-overlapped downward in y order
  const labels=series.map(s=>({y:sy(clip(s.rel[0],lo,hi)), color:s.color,
                               txt:fmt(s.rel[0]*100,0)+'% '+s.key+(s.sel?' ◀':'')}))
                     .sort((a,b)=>a.y-b.y);
  let prev=-1e9;
  for(const L of labels){
    const y=Math.max(L.y, prev+11); prev=y;
    g+=`<text class="dlabel" x="${mL+6}" y="${clip(y,mT+8,mT+ph-2)}" text-anchor="start" fill="${L.color}">${esc(L.txt)}</text>`;
  }
  g+=`<text class="axlbl" x="${mL+pw/2}" y="${H-4}" text-anchor="middle">max_num_batched_tokens (log)</text>`;
  el.innerHTML=svgEl(g, W, H,
    'Cold-request ceiling versus chunk size for every model, normalised to each model’s value at 32,768');
  chartE3Geom={W, mL, pw, mT, ph, sx, sy, lo, hi, series};
}
// hover: same log invert as chart E's, crosshair through every model
(function hoverE3(){
  const box=document.getElementById('chartE3'), tt=document.getElementById('ttE3');
  function handler(e){
    const geom=chartE3Geom; if(!geom) return;
    const svg=box.querySelector('svg'); if(!svg) return;
    const rect=svg.getBoundingClientRect();
    const vx=(e.clientX-rect.left)/rect.width*geom.W;
    if(vx<geom.mL || vx>geom.mL+geom.pw){ tt.style.opacity=0; removeCross(svg); chartE2Hover(null); return; }
    const C=Math.round(clip(
      Math.exp(Math.log(CHUNK_LO) + (vx-geom.mL)/geom.pw*Math.log(CHUNK_HI/CHUNK_LO)),
      CHUNK_LO, CHUNK_HI));
    const px=geom.sx(C);
    const marks=geom.series.map(s=>({color:s.color, r:3.5,
      y:geom.sy(clip(interpChunk(s.rel,C), geom.lo, geom.hi))}));
    drawCross(svg, px, geom, marks);
    tt.innerHTML=`<div class="tth tnum">chunk = ${fmt(C,0)} tok</div>`
      +geom.series.map(s=>`<div class="row"><span class="sw" style="background:${s.color}"></span>`
        +`${esc(s.key)}${s.sel?' (selected)':''} <b class="tnum">${fmt(interpChunk(s.rel,C)*100,0)}%</b></div>`).join('');
    chartE2Hover(C);
    const par=tt.offsetParent||box, parRect=par.getBoundingClientRect();
    const scale=rect.width/geom.W;
    const cx=(rect.left-parRect.left)+px*scale;
    const cy=(rect.top-parRect.top)+(marks.length?marks[0].y:geom.mT)*scale;
    const tw=tt.offsetWidth, th=tt.offsetHeight, pad=4, gap=12;
    let left=cx+gap;
    if(left+tw>parRect.width-pad) left=cx-gap-tw;
    let top=cy-th-gap;
    if(top<pad) top=cy+gap;
    tt.style.left=Math.max(pad,Math.min(left,parRect.width-tw-pad))+'px';
    tt.style.top =Math.max(pad,Math.min(top, parRect.height-th-pad))+'px';
    tt.style.opacity=1;
  }
  function leave(){
    tt.style.opacity=0; const svg=box.querySelector('svg'); if(svg) removeCross(svg);
    chartE2Hover(null);
  }
  box.addEventListener('mousemove', handler);
  box.addEventListener('mouseleave', leave);
  box.addEventListener('touchmove',(e)=>{ if(e.touches[0]) handler(e.touches[0]); },{passive:true});
  box.addEventListener('touchend', leave); box.addEventListener('touchcancel', leave);
})();

// ---- E4: the token stream itself. Ticks = tokens arriving; blocks = the
// stream frozen behind an interleaved prefill chunk (one token rides each
// pass, so a gap is stepS + chunk time). Same green total in every row. Drawn
// at the solo-panel width so its type matches chart G's. ----
function renderChartE4(model, topo, mean, stepS){
  const W=1120, rh=24, lab=13, rgap=16, mL=52, mR=150, mT=8, rows=TEXTURE_CS;
  const slot=lab+rh+rgap;
  const H=mT + rows.length*slot - rgap + 40;
  const pw=W-mL-mR;
  const grid=cssv('--grid'), axis=cssv('--axis'), muted=cssv('--muted');
  const s2=cssv('--s2'), s3=cssv('--s3');
  const data=rows.map(C=>{
    const ts=chunkPassTimes(model, topo, mean, C);
    return {C, ts, stall:ts.reduce((a,b)=>a+b,0)};
  });
  const lead=0.25, tail=0.35;   // s of normal decode either side of the miss
  const Tmax=Math.max(...data.map(r=>lead+r.ts.length*stepS+r.stall+tail));
  const sx=linScale(0, Tmax, mL, mL+pw);
  let g='';
  data.forEach((r,ri)=>{
    const y=mT+ri*slot+lab, yl=y-4;
    // a run of normal decode is ONE thick horizontal line dashed at the token
    // pitch (1 px on, pitch-1 off), so the tick count never enters the DOM:
    // at a 10 ms step over a 60-s stall a per-tick element would be thousands
    // of nodes, and below a 1 px pitch the dashes merge into the solid bar
    // the eye would see anyway
    const yc=y+rh/2, th=rh-6, stepPx=sx(stepS)-sx(0);
    const run=(t0,t1)=>{
      if(t1<=t0) return;
      const dash=stepPx>1.5?` stroke-dasharray="1 ${(stepPx-1).toFixed(2)}"`:'';
      g+=`<line x1="${sx(t0).toFixed(1)}" y1="${yc}" x2="${sx(t1).toFixed(1)}" y2="${yc}" stroke="${axis}" stroke-width="${th}"${dash}/>`;
    };
    g+=`<text class="axtick" x="${mL-8}" y="${y+rh/2+3}" text-anchor="end">${r.C/1024}k</text>`;
    g+=`<rect x="${mL}" y="${y}" width="${pw}" height="${rh}" fill="${grid}" opacity="0.30"/>`;
    run(0, lead);
    // the miss's first chunk rides the pass after the LAST token drawn, so the
    // first labelled gap is p + stepS like every other, not p + 2 stepS
    let t=Math.floor((lead-1e-9)/stepS)*stepS;
    g+=`<text class="axtick" x="${mL}" y="${yl}" text-anchor="start" fill="${muted}">${fmt(stepS*1000,0)} ms</text>`;
    const blocks=[];
    for(const p of r.ts){
      const x0=sx(t), x1=sx(t+p);
      g+=`<rect x="${x0}" y="${y}" width="${Math.max(1, x1-x0)}" height="${rh}" fill="${s3}" opacity="0.8"/>`;
      blocks.push({x0, x1, ms:(stepS+p)*1000});   // the full token gap: step + chunk
      t+=p+stepS;      // the next token: one decode step after the chunk clears
    }
    run(t, Tmax);      // the stream resumes with that token
    // gap labels: one per block when they fit, else one summary for the row
    const avgW=blocks.reduce((a,b)=>a+b.x1-b.x0,0)/blocks.length;
    if(avgW>=40){
      blocks.forEach(b=>{
        g+=`<text class="axtick" x="${(b.x0+b.x1)/2}" y="${yl}" text-anchor="middle">${fmt(b.ms,0)} ms</text>`;
      });
    } else {
      const meanMs=blocks.reduce((a,b)=>a+b.ms,0)/blocks.length;
      g+=`<text class="axtick" x="${(blocks[0].x0+blocks[blocks.length-1].x1)/2}" y="${yl}" text-anchor="middle">${blocks.length} gaps of ≈ ${fmt(meanMs,0)} ms</text>`;
    }
    g+=`<text class="axtick" x="${mL+pw+7}" y="${y+8}" text-anchor="start">worst ${fmt(Math.max(...r.ts)+stepS,2)} s</text>`;
    g+=`<text class="axtick" x="${mL+pw+7}" y="${y+19}" text-anchor="start" fill="${muted}">Σ frozen ${fmt(r.stall,2)} s</text>`;
    // the miss's own side of the trade: its first token arrives when its LAST
    // chunk clears — later at small chunks, one interleaved decode step per
    // extra pass
    const tft=r.stall+(r.ts.length-1)*stepS;
    const xEnd=blocks[blocks.length-1].x1;
    g+=`<path d="M ${xEnd} ${y+rh+2} L ${xEnd-4} ${y+rh+8} L ${xEnd+4} ${y+rh+8} Z" fill="${s2}"/>`;
    g+=`<text class="axtick" x="${mL+pw+7}" y="${y+30}" text-anchor="start" fill="${s2}">▲ 1st tok ${fmt(tft,2)} s</text>`;
  });
  const yBot=mT+rows.length*slot-rgap;          // bottom of the last row's bar
  niceTicksIn(0, Tmax, 8).forEach(v=>{
    g+=`<text class="axtick" x="${sx(v)}" y="${yBot+15}" text-anchor="middle">${fmt(Math.abs(v)<1e-9?0:v,1)}</text>`;
  });
  g+=`<text class="axlbl" x="${mL+pw/2}" y="${yBot+32}" text-anchor="middle">seconds — one mean-length miss (${fmt(mean/1000,0)}k tokens) mid-stream; labels = gap between consecutive tokens</text>`;
  document.getElementById('chartE4').innerHTML=
    svgEl(g, W, H, 'Token stream of a decoder while one miss re-prefills, at three chunk sizes, each waiting gap labelled in milliseconds');
}

// Runs on every frame, drafts included, so each panel is rewritten only when
// its inputs moved: the three panels' JS is ~1 ms, but replacing three SVGs
// of text on a frame that then reads layout cost ~25 ms of forced layout on
// every settle. The signatures are the closed-form scalars each panel is
// drawn from (probe pass times stand in for the MFU anchor and the FLOP
// scale), so a load slider — which moves none of them — leaves all three
// untouched, and a theme flip clears them (redrawCharts) so colours repaint.
const companionSig={E2:null, E3:null, E4:null};
let lastCompanionArgs=null;
// E3 and E4 are rebuilt on SETTLED frames only (chart B's trade: a drag keeps
// the last settled picture, the 120 ms settle redraws it) — their rewrite is
// ~1.5 ms of a ~14 ms draft frame, and E4 is priced off the stress point,
// which a draft frame only approximates anyway. E2 is live: it moves with the
// model, topology and chunk, not with any dragged slider, and costs 0.2 ms.
export function renderChartECompanions(model, topo, wl, cs, stress, noFit, draft){
  lastCompanionArgs=[model, topo, wl, cs, stress, noFit];
  if(noFit){
    if(companionSig.E2!=='nofit'){
      renderNoFit('chartE2','pass anatomy'); renderNoFit('chartE3','chunk-size trade');
      renderNoFit('chartE4','token stream');
      document.getElementById('ttE3').style.opacity=0;
    }
    chartE2State=null; chartE3Geom=null;
    companionSig.E2=companionSig.E3=companionSig.E4='nofit';
    return;
  }
  const CH=prefillChunk();
  let sig2;
  try{ sig2=[model.name, CH, mfuCeil(model, topo), prefillOverheadSeconds(model, topo),
             prefillFlops(model, 2048, 0), peakFlops(topo)].join('|'); }
  catch(e){ sig2='unsolvable|'+model.name; }
  if(sig2!==companionSig.E2){ companionSig.E2=sig2; renderChartE2(model, topo, null); }
  if(draft) return;
  const d3=chartE3Data(topo, wl, cs);
  const sig3=d3.models.map(m=>m.key+(m.sel?'*':'')+m.rel.map(v=>v.toFixed(4)).join(',')).join(';');
  if(sig3!==companionSig.E3){ companionSig.E3=sig3; renderChartE3(d3); }
  const stepS=(stress && stress.pu>0) ? model.mtp/stress.pu : 0;   // chart E's own gap
  const sig4=[model.name, cs.mean, stepS, prefillSeconds(model, topo, 2048, 0),
              prefillSeconds(model, topo, 2048, cs.mean)].join('|');
  if(sig4!==companionSig.E4){
    companionSig.E4=sig4;
    if(stepS>0) renderChartE4(model, topo, cs.mean, stepS);
    else renderChartMsg('chartE4','no decode operating point at this load — the token stream needs one', 1120);
  }
}
// theme flip: repaint all three from the last frame's inputs, no new draws
export function redrawChartECompanions(){
  if(!lastCompanionArgs) return;
  companionSig.E2=companionSig.E3=companionSig.E4=null;
  renderChartECompanions(...lastCompanionArgs);
}

/* ============================================================================
   TOOLTIP / crosshair for C and D
   ========================================================================== */
// Value of a plotted series at an arbitrary x, read off the SAME straight
// segments the chart draws (the sweep samples ~60 points across the dynamic
// axis — decodePlan() picks the stride — so the
// curve between samples IS a line). Clamped to the sampled range at the ends.
// `log` = the chart plots this series on a log y-axis, where the drawn segment
// is the GEOMETRIC interpolation of its endpoints — interpolate the same way or
// the readout would not be the value the segment shows.
export function interpAt(dc, key, x, log){
  const ns=dc.ns, arr=dc[key], last=ns.length-1;
  if(x<=ns[0]) return arr[0];
  if(x>=ns[last]) return arr[last];
  let i=last-1;
  for(let j=0;j<last;j++){ if(ns[j+1]>=x){ i=j; break; } }
  const t=(x-ns[i])/(ns[i+1]-ns[i]);
  if(log && arr[i]>0 && arr[i+1]>0)
    return Math.exp(Math.log(arr[i]) + t*(Math.log(arr[i+1])-Math.log(arr[i])));
  return arr[i] + t*(arr[i+1]-arr[i]);
}
// markFn(n, dc, geom) -> [{v, color, r}] : the plotted series values the crosshair
// crosses at this x. The FIRST entry is the tooltip's anchor point, so the
// readout sits next to where the bar actually meets the curve.
export function setupHover(chartId, ttId, getGeom, fmtFn, markFn){
  const box = document.getElementById(chartId);
  const tt = document.getElementById(ttId);
  function handler(e){
    const geom=getGeom(); if(!geom) return;
    const svg = box.querySelector('svg'); if(!svg) return;
    const rect = svg.getBoundingClientRect();
    const fx=(e.clientX-rect.left)/rect.width;
    const vx=fx*geom.W;                 // viewBox x
    // invert to data n
    const nData = 1 + (vx-geom.mL)/geom.pw*(geom.xmax-1);
    if(vx<geom.mL || vx>geom.mL+geom.pw){ tt.style.opacity=0; removeCross(svg); return; }
    // Track the pointer instead of snapping to the sweep's sample points (which
    // made the bar and the readout jump one sample stride at a time). max_num_seqs
    // is an integer setting, so the bar lands on the nearest whole seq and every
    // series is INTERPOLATED along its drawn segment at that x — the readout is
    // then exactly what the eye reads off the curve under the bar.
    const dc=geom.dc;
    // clamp to the LAST SAMPLED n, not the axis max: the sweep ends at 119
    // (117 in draft mode) while the axis runs to 120, and interpAt clamps Y
    // to the last sample — so an X past it would pair X=120 with Y(119).
    const nMax=dc.ns[dc.ns.length-1];
    const n=clip(Math.round(nData), 1, nMax);
    const px=geom.sx(n);
    // crosshair + a dot on every curve it crosses
    const marks=(markFn?markFn(n,dc,geom):[]).map(m=>({...m, y:geom.sy(m.v)}));
    drawCross(svg, px, geom, marks);
    tt.innerHTML = fmtFn(n, dc, geom);
    // Position in CHARTBOX-relative px, anchored on the intersection point and
    // clamped inside the box: the card clips overflow, so a tooltip placed
    // above the plot (the old transform) was cut off or invisible entirely.
    const par = tt.offsetParent || box;
    const parRect = par.getBoundingClientRect();
    const scale = rect.width/geom.W;    // viewBox units -> css px (uniform)
    const cx = (rect.left-parRect.left) + px*scale;
    const cy = (rect.top-parRect.top) + (marks.length?marks[0].y:geom.mT)*scale;
    const tw=tt.offsetWidth, th=tt.offsetHeight, pad=4, gap=12;
    let left = cx+gap;
    if (left+tw > parRect.width-pad) left = cx-gap-tw;   // flip to the left
    let top = cy-th-gap;
    if (top < pad) top = cy+gap;                          // flip below
    tt.style.left = Math.max(pad, Math.min(left, parRect.width -tw-pad))+'px';
    tt.style.top  = Math.max(pad, Math.min(top,  parRect.height-th-pad))+'px';
    tt.style.opacity=1;
  }
  function leave(){ tt.style.opacity=0; const svg=box.querySelector('svg'); if(svg) removeCross(svg); }
  box.addEventListener('mousemove', handler);
  box.addEventListener('mouseleave', leave);
  box.addEventListener('touchmove', (e)=>{ if(e.touches[0]){ handler(e.touches[0]); } }, {passive:true});
}
export function drawCross(svg, px, geom, marks){
  removeCross(svg);
  const gEl=document.createElementNS(NS,'g');
  gEl.setAttribute('class','xcross');
  const c=document.createElementNS(NS,'line');
  c.setAttribute('x1',px); c.setAttribute('x2',px);
  c.setAttribute('y1',geom.mT); c.setAttribute('y2',geom.mT+geom.ph);
  c.setAttribute('stroke',cssv('--axis')); c.setAttribute('stroke-width','1');
  c.setAttribute('stroke-dasharray','3 2');
  gEl.appendChild(c);
  (marks||[]).forEach(m=>{
    // clamp to the plot area so a dot never renders outside the frame
    const y=clip(m.y, geom.mT, geom.mT+geom.ph);
    const d=document.createElementNS(NS,'circle');
    d.setAttribute('cx',px); d.setAttribute('cy',y); d.setAttribute('r',m.r??4);
    d.setAttribute('fill',m.color); d.setAttribute('stroke',cssv('--surface'));
    d.setAttribute('stroke-width','1.5');
    gEl.appendChild(d);
  });
  svg.appendChild(gEl);
}
export function removeCross(svg){ const e=svg.querySelector('.xcross'); if(e) e.remove(); }
