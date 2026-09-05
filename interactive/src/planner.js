import { PREFILL_MFU_HI, PREFILL_MFU_LO } from './config.js';
import { decodeFloor, maxUsersLatency, maxUsersSaturation, prefillChunk,
         prefillServiceMoments, serverRate } from './prefill.js';
import { clip } from './mathlib.js';
import { p_sub } from './workload.js';
import { warmCapacity } from './capacity.js';
import { state } from './state.js';
import { cssv, esc, fmt, linScale, logScale, logTicks, niceTicks, svgEl } from './svg.js';
import { renderNoFit } from './charts.js';
import { paintTiles, pendingLoadTiles } from './render.js';

/* ============================================================================
   THE PLANNER PANEL — spike tiles, charts F & G, the frontier table
   ========================================================================== */
// Four series that must stay separable for colour-vision deficiency: green,
// orange and red all collapse to the same olive under deuteranopia, so the
// ramp is blue / orange / purple / grey, and every line also carries its own
// DASH pattern — hue is never the only channel that distinguishes them.
export const PLANNER_COLORS = () => ({
  cache: cssv('--s1'), decode: cssv('--s4'),
  latency: cssv('--s2'), saturation: cssv('--muted'),
});
const PLANNER_DASH = { cache:'', decode:'7 3', latency:'2 3', saturation:'11 4' };
export const PLANNER_LABEL = { cache:'cache', decode:'decode', latency:'latency',
                        saturation:'saturation' };

/* Right-hand end of the miss-rate axis charts F and G sweep and draw.
   The study's planning range is 0–50% and every published figure uses it, so
   that stays the default: widening the axis for everyone would compress the
   0–20% region the decision actually turns on. A workload with NO prefix
   reuse at all (each request a fresh conversation, or a per-request salt in
   the system prompt) is a real configuration the slider now reaches, so the
   axis extends to 100% only once the slider is actually past 50%. The 0.01
   grid is preserved either way — renderSpikeChart indexes the sweep by
   Math.round(f*100) and would read the wrong point off a rescaled grid. */
export function fAxisMax(){ return state.inval > 50 ? 1.0 : 0.5; }
// six evenly spaced ticks — [0,10,...,50]% on the default axis, [0,20,...,100]%
// on the widened one, so the labels stay round numbers in both
function fAxisTicks(fMax){
  return [0,1,2,3,4,5].map(i => i*fMax/5);
}

// Warm USER capacity as a function of f, from a few Monte-Carlo anchors.
// Warm capacity falls smoothly and near-linearly with the miss rate (H6:
// ~1.5 x f), so five fills plus the CURRENT f — included as an anchor so the
// curve passes exactly through the value the tiles quote — beat 50 fills for
// any resolution this chart can show. Interpolation is stated in the caption
// rather than hidden: this is the one series on the panel that is sampled.
export function warmUsersCurve(model, topo, ram, iter, budget, wl0, fNow, warmNow){
  const fMax = fAxisMax();
  const anchors = [0, 0.25, 0.5, 0.75, 1.0].map(a => a*fMax);
  if (!anchors.includes(fNow)) anchors.push(fNow);
  anchors.sort((a,b)=>a-b);
  const pts = anchors.map(f => {
    if (f === fNow && warmNow !== undefined) return {f, u: warmNow};
    const wl = {...wl0, invalidation: f};
    const wc = warmCapacity(model, topo, wl, ram, iter, budget);
    return { f, u: wc.all[0] * (1 - p_sub(wl)) };
  });
  return f => {
    if (f <= pts[0].f) return pts[0].u;
    for (let i=1;i<pts.length;i++)
      if (f <= pts[i].f){
        const a=pts[i-1], b=pts[i];
        return a.u + (b.u-a.u)*(f-a.f)/((b.f-a.f)||1);
      }
    return pts[pts.length-1].u;
  };
}

// Everything charts F and G draw, for the CURRENT configuration.
export function plannerData(model, topo, wl, cs, warmFn, decodeUsers, mo){
  const fs = [];
  // 1-point-per-percent grid out to the current axis end — renderSpikeChart
  // reads its marker back as fs[Math.round(f*100)], so the step must stay 0.01
  for (let i=0;i<=Math.round(fAxisMax()*100);i++) fs.push(i/100);
  const think = state.think, sla = state.sla, reps = topo.replicas || 1;
  // the ceilings are per replica GROUP, so the load one group sees is the
  // system population divided by the replica count
  const perGroup = u => u * reps;
  const series = { cache:[], decode:[], latency:[], saturation:[], env:[], binding:[] };
  const spike = { mid:[], lo:[], hi:[] };
  const moLo = prefillServiceMoments(model, topo, wl, cs, prefillChunk(), PREFILL_MFU_LO);
  const moHi = prefillServiceMoments(model, topo, wl, cs, prefillChunk(), PREFILL_MFU_HI);
  for (const f of fs){
    const c = {
      cache: warmFn(f) * reps,
      decode: decodeUsers * reps,
      latency: perGroup(maxUsersLatency(mo, f, sla, think, undefined,
                                        wl.sub_ratio)),
      saturation: perGroup(maxUsersSaturation(mo, f, think, wl.sub_ratio)),
    };
    let bind='cache'; for (const k in c) if (c[k] < c[bind]) bind = k;
    for (const k in c) series[k].push(c[k]);
    series.env.push(c[bind]); series.binding.push(bind);
    const rate = serverRate(state.users, think, wl.sub_ratio)/reps;
    spike.mid.push(bStar(mo,   f, sla, rate));
    spike.lo .push(bStar(moLo, f, sla, rate));
    spike.hi .push(bStar(moHi, f, sla, rate));
  }
  return { fs, ...series, spike, mo, moLo, moHi };
}

// B* from a moments bundle — the same arithmetic spikeMetrics does, factored
// out so the MFU bracket and the f sweep can reuse it without re-sampling.
export function bStar(mo, f, sla, rate){
  const rho = rate * (f*mo.miss + (1-f)*mo.hit);
  return rho >= 1 ? 0 : Math.max(0, sla*(1-rho)/mo.miss);
}

export function renderSpikeTiles(op, sp, model, topo, wl, cs, noFit, fitHint){
  if (noFit){
    const msg = `<div class="tile wide"><div class="k">The operating point</div>`
      + `<div class="v tnum">—</div><div class="sub2">model weights do not fit this `
      + `configuration — ${esc(fitHint)}</div></div>`;
    document.getElementById('decisionTiles').innerHTML = msg;
    // act 2's capacity-side tiles are already painted and still meaningful;
    // only the cold-traffic readouts are missing, so append rather than wipe
    return;
  }
  const C = PLANNER_COLORS();
  const bind = op.binding, reps = topo.replicas || 1;
  const drain = op.burstDrain, lastTTFT = drain;
  const good=cssv('--good'), warn=cssv('--warn'), crit=cssv('--crit');
  const headClass = op.headroom >= 1 ? crit : (op.headroom >= 0.8 ? warn : good);
  const others = Object.keys(op.ceilings).filter(k=>k!==bind)
    .sort((a,b)=>op.ceilings[a]-op.ceilings[b])
    .map(k=>`${PLANNER_LABEL[k]} ${fmt(op.ceilings[k],0)}`).join(' · ');
  const tiles = [
    {decision:true, k:'Binding constraint', hero:true, v:PLANNER_LABEL[bind].toUpperCase(),
     u: op.limit >= 1 ? `at ${fmt(op.limit,0)} users` : 'at no load at all',
     sub: op.limit >= 1
        ? `you are running ${fmt(op.users,0)} — ${fmt(op.headroom*100,0)}% of the limit`
          + ` · next: ${others}`
        : `the ${fmt(state.sla,0)} s budget is below one miss's own prefill `
          + `(${fmt(sp.mo.miss,1)} s) — unachievable at any load`
          + ` · next: ${others}`,
     cls: op.headroom>=1?'crit':(op.headroom>=0.8?'warn':'good'),
     tip:`All four ceilings in ONE unit — max concurrent users — so the binding one is simply the smallest. cache = the warm p5 population that fits the pool; decode = where per-user p50 hits the ${fmt(decodeFloor(),0)} tok/s floor; latency = where a miss's mean TTFT hits the budget; saturation = where prefill duty hits 100%. The conversion rests on the Concurrent-users assumptions; chart G shows where the binding constraint changes hands.`},
    {hero:true, k:'Cold-spike tolerance B*', v:fmt(op.bstar,1), u:'misses at once',
     sub:`MFU 35–55% band: ${fmt(op.bstarLo,1)}–${fmt(op.bstarHi,1)}`
        + ` · zero at f* ${op.fstar>10?'> 1,000':fmt(op.fstar*100,0)+'%'}`
        + ` · latency ceiling f_sla ${op.fsla>=1?'> 100':fmt(op.fsla*100,0)}%`,
     cls: op.bstar<1?'crit':(op.bstar<5?'warn':'good'),
     tip:"The largest burst of SIMULTANEOUS misses whose last request still gets a first token inside the TTFT budget — linear in that budget. The band is the MFU [35–55%] bracket; B* reaches zero exactly at f*, and f_sla (mean TTFT = budget) binds earlier still."},
    {k:`A burst of ${fmt(state.burst,0)} at once`,
     v: !isFinite(drain) ? 'never' : (drain>=90? fmt(drain/60,1) : fmt(drain,1)),
     u: !isFinite(drain) ? 'clears at this load' : (drain>=90?'min to clear':'s to clear'),
     sub: !isFinite(drain)
        ? `the standing load already saturates prefill, so a burst on top of it never drains`
        : `last request waits ${lastTTFT>=90?fmt(lastTTFT/60,1)+' min':fmt(lastTTFT,1)+' s'}`
          + ` (budget ${fmt(state.sla,0)} s) · every warm user loses ~${fmt(op.tokensLost,0)} output tokens`,
     cls: drain>state.sla?'crit':'good',
     tip:"What a correlated invalidation event costs. The backlog drains at (1 − duty) seconds of work per second — standing traffic keeps arriving — so the last request's TTFT IS the drain time; meanwhile the ITL spike is the steady state, and the tokens-lost figure integrates what the warm users stop receiving over the drain."},
    {k:'TTFT now (miss / hit)',
     v: isFinite(op.ttftMiss) ? fmt(op.ttftMiss,2) : '∞', u: isFinite(op.ttftMiss) ? 's for a miss' : 'queue unbounded',
     sub: op.duty >= 1
        ? `prefill duty ${fmt(op.duty*100,0)}% — the queue is unbounded at this load, `
          + `so there is no steady-state TTFT to quote`
        : `a HIT waits ${fmt(op.ttftHitFcfs*1000,0)} ms FCFS | ${fmt(op.ttftHitPs*1000,0)} ms sharing`
          + ` · duty ${fmt(op.duty*100,0)}%`,
     cls: op.ttftMiss>state.sla?'crit':(op.ttftMiss>state.sla/2?'warn':'good'),
     tip:"Mean TTFT at the current load, from the M/G/1 queue. The HIT column is the sharpest thing on this page: under FCFS a warm hit waits behind whatever misses are in front of it, so the miss rate is a latency parameter for the hitting users too; processor sharing is the bracket's other end. Solved against the MEAN — a p95 budget binds sooner."},
  ];
  const colMap = {good, warn, crit};
  // the binding constraint IS the decision, so it heads act 3 alone; the cold-
  // traffic readouts join act 2's tiles, which renderTiles has already painted
  paintTiles('decisionTiles', tiles.filter(t=>t.decision), colMap);
  // B* is act 2's headline, so it leads and spans two columns; the row then
  // reads 2+1 / 3 and fills exactly, with the capacity-side tiles following
  const mine = tiles.filter(t=>!t.decision);
  const bstar = mine.filter(t=>t.hero), rest = mine.filter(t=>!t.hero);
  paintTiles('tilesLoad', [...bstar, ...pendingLoadTiles, ...rest], colMap);
}

/* ---- The binding-constraint chart (act 3, rendered as 'G'): the four
   ceilings, in users, vs the miss rate ---- */
let bindingGeom = null;
export function renderBindingChart(d, op){
  if (!d){ renderNoFit('chartG','planner'); bindingGeom=null;
           document.getElementById('ttG').style.opacity=0; return; }
  // act 3 gives this chart the full page width, so its viewBox has to track
  // the width it will actually be painted at: a 560-wide box stretched to
  // 1,100 px scales the type UP, and a 1,120-wide box squeezed into 430 px
  // scales it down to ~4 px. Pick the box to keep labels at parity with every
  // other chart on the page.
  const wide = (typeof window !== 'undefined' ? window.innerWidth : 1400) >= 900;
  const W = wide ? 1120 : 560, H = wide ? 400 : 330;
  const mL = wide ? 64 : 52, mR = wide ? 22 : 16, mT = 16, mB = wide ? 46 : 42;
  const pw=W-mL-mR, ph=H-mT-mB;
  const grid=cssv('--grid'), axis=cssv('--axis'), muted=cssv('--muted');
  const surface=cssv('--surface'), text=cssv('--text');
  const C = PLANNER_COLORS();
  const all = [...d.cache, ...d.decode, ...d.latency, ...d.saturation, op.users]
                .filter(v=>isFinite(v) && v>0);
  const yLo = Math.max(1, Math.min(...all, op.users)*0.6);
  const yHi = Math.max(...all)*1.25;
  const fMax = fAxisMax();
  const sx = linScale(0, fMax, mL, mL+pw), sy = logScale(yLo, yHi, mT+ph, mT);
  let g='';
  for (const t of logTicks(yLo,yHi)){
    const Y=sy(t);
    g+=`<line x1="${mL}" y1="${Y}" x2="${mL+pw}" y2="${Y}" stroke="${grid}" stroke-width="1"/>`;
    g+=`<text class="axtick" x="${mL-8}" y="${Y+3}" text-anchor="end">${fmt(t,0)}</text>`;
  }
  for (const t of fAxisTicks(fMax)){
    const X=sx(t);
    g+=`<line x1="${X}" y1="${mT}" x2="${X}" y2="${mT+ph}" stroke="${grid}" stroke-width="1"/>`;
    g+=`<text class="axtick" x="${X}" y="${mT+ph+16}" text-anchor="middle">${fmt(t*100,0)}%</text>`;
  }
  const path = arr => arr.map((v,i)=>`${i?'L':'M'} ${sx(d.fs[i])} ${sy(clip(v,yLo,yHi))}`).join(' ');
  // safe region: everything under the envelope
  g+=`<path d="${path(d.env)} L ${sx(fMax)} ${mT+ph} L ${mL} ${mT+ph} Z" fill="${C.cache}" opacity="0.07"/>`;
  // the envelope goes UNDERNEATH as a wide halo: drawn on top as an opaque
  // line it covered whichever series was binding — i.e. always hid the one
  // the reader came for
  g+=`<path d="${path(d.env)}" fill="none" stroke="${text}" stroke-width="7" stroke-linejoin="round" opacity="0.14"/>`;
  for (const k of ['cache','decode','latency','saturation'])
    g+=`<path d="${path(d[k])}" fill="none" stroke="${C[k]}" stroke-width="2" stroke-linejoin="round"`
      +`${PLANNER_DASH[k]?` stroke-dasharray="${PLANNER_DASH[k]}"`:''}/>`;
  // ...and direct-label each line at its right terminus, so the chart is
  // readable without cross-referencing a legend
  for (const k of ['cache','decode','latency','saturation']){
    const v = d[k][d[k].length-1];
    if (!isFinite(v)) continue;
    g+=`<text class="dlabel" x="${mL+pw-3}" y="${sy(clip(v,yLo,yHi))-5}" text-anchor="end" fill="${C[k]}">${esc(PLANNER_LABEL[k])}</text>`;
  }
  // where the envelope changes hands
  for (let i=1;i<d.fs.length;i++){
    if (d.binding[i] !== d.binding[i-1]){
      const X=sx(d.fs[i]);
      g+=`<line x1="${X}" y1="${mT}" x2="${X}" y2="${mT+ph}" stroke="${muted}" stroke-width="1.2" stroke-dasharray="3 3"/>`;
      g+=`<text class="dlabel" x="${X+5}" y="${mT+12}" text-anchor="start" fill="${muted}">`
        +`${esc(d.binding[i-1])} → ${esc(d.binding[i])} at ${fmt(d.fs[i]*100,0)}%</text>`;
      break;
    }
  }
  // your load
  const X0=sx(clip(state.inval/100,0,fMax)), Y0=sy(clip(op.users,yLo,yHi));
  const col0=op.fits?cssv('--good'):cssv('--crit');
  g+=`<circle cx="${X0}" cy="${Y0}" r="5" fill="${col0}" stroke="${surface}" stroke-width="1.5"/>`;
  // flip the label to the left near the right edge, and say when the load is
  // pinned to the top of the axis rather than silently drawing it off-plot
  const flip = X0 > mL + pw*0.82;
  const over = op.users > yHi;
  g+=`<text class="dlabel" x="${X0+(flip?-9:9)}" y="${Y0+4}" text-anchor="${flip?'end':'start'}" fill="${col0}">`
    +`${over?'≥ ':''}${fmt(op.users,0)} users</text>`;
  g+=`<line x1="${mL}" y1="${mT+ph}" x2="${mL+pw}" y2="${mT+ph}" stroke="${axis}" stroke-width="1"/>`;
  g+=`<text class="axlbl" x="${mL+pw/2}" y="${H-6}" text-anchor="middle">cache-miss rate f</text>`;
  g+=`<text class="axlbl" x="${12}" y="${mT+ph/2}" text-anchor="middle" transform="rotate(-90 12 ${mT+ph/2})">max concurrent users (log)</text>`;
  document.getElementById('chartG').innerHTML =
    svgEl(g,W,H,'The four ceilings in max concurrent users versus the cache-miss rate');
  bindingGeom = { W,H,mL,mR,mT,pw,ph, d, sx, sy };
}

/* ---- The spike chart (act 2, rendered as 'F'): cold-spike tolerance,
   with the MFU bracket as a band ---- */
let spikeGeom = null;
export function renderSpikeChart(d, others){
  if (!d){ renderNoFit('chartF','spike tolerance'); spikeGeom=null;
           document.getElementById('ttF').style.opacity=0; return; }
  const W=560,H=320, mL=52,mR=16,mT=14,mB=42;
  const pw=W-mL-mR, ph=H-mT-mB;
  const grid=cssv('--grid'), axis=cssv('--axis'), muted=cssv('--muted');
  const surface=cssv('--surface'), s1=cssv('--s1'), crit=cssv('--crit');
  const ctx = (others||[]).flatMap(o=>o.b).filter(v=>isFinite(v)&&v>0);
  const yLo=0.1, yHi=Math.max(4, ...d.spike.hi.filter(isFinite), ...ctx)*1.3;
  const fMax = fAxisMax();
  const sx=linScale(0,fMax,mL,mL+pw), sy=logScale(yLo,yHi,mT+ph,mT);
  let g='';
  for (const t of logTicks(yLo,yHi)){
    const Y=sy(t);
    g+=`<line x1="${mL}" y1="${Y}" x2="${mL+pw}" y2="${Y}" stroke="${grid}" stroke-width="1"/>`;
    g+=`<text class="axtick" x="${mL-8}" y="${Y+3}" text-anchor="end">${t<1?t.toFixed(1):fmt(t,0)}</text>`;
  }
  for (const t of fAxisTicks(fMax)){
    const X=sx(t);
    g+=`<line x1="${X}" y1="${mT}" x2="${X}" y2="${mT+ph}" stroke="${grid}" stroke-width="1"/>`;
    g+=`<text class="axtick" x="${X}" y="${mT+ph+16}" text-anchor="middle">${fmt(t*100,0)}%</text>`;
  }
  const path = (arr,fs) => arr.map((v,i)=>`${i?'L':'M'} ${sx((fs||d.fs)[i])} ${sy(clip(v,yLo,yHi))}`).join(' ');
  // other topologies of the same model, for context
  for (const o of (others||[]))
    g+=`<path d="${path(o.b, o.fs)}" fill="none" stroke="${muted}" stroke-width="1.2" opacity="0.55"/>`;
  // MFU bracket band
  const up = d.spike.hi.map((v,i)=>[d.fs[i],v]);
  const dn = d.spike.lo.map((v,i)=>[d.fs[i],v]).reverse();
  g+=`<path d="${up.map(([f,v],i)=>`${i?'L':'M'} ${sx(f)} ${sy(clip(v,yLo,yHi))}`).join(' ')} `
    + `${dn.map(([f,v])=>`L ${sx(f)} ${sy(clip(v,yLo,yHi))}`).join(' ')} Z" fill="${s1}" opacity="0.20"/>`;
  g+=`<path d="${path(d.spike.mid)}" fill="none" stroke="${s1}" stroke-width="2.4" stroke-linejoin="round"/>`;
  // "cannot absorb one" line
  const Y1=sy(1);
  g+=`<line x1="${mL}" y1="${Y1}" x2="${mL+pw}" y2="${Y1}" stroke="${crit}" stroke-width="1.4" stroke-dasharray="4 3"/>`;
  g+=`<text class="dlabel" x="${mL+pw-4}" y="${Y1-6}" text-anchor="end" fill="${crit}">cannot absorb a single simultaneous miss</text>`;
  // your miss rate
  const f0=clip(state.inval/100,0,fMax), i0=Math.round(f0*100);
  const X0=sx(f0), Y0=sy(clip(d.spike.mid[i0],yLo,yHi));
  g+=`<circle cx="${X0}" cy="${Y0}" r="5" fill="${s1}" stroke="${surface}" stroke-width="1.5"/>`;
  g+=`<text class="dlabel" x="${X0+9}" y="${Y0+4}" text-anchor="start" fill="${s1}">B* ${fmt(d.spike.mid[i0],1)}</text>`;
  g+=`<line x1="${mL}" y1="${mT+ph}" x2="${mL+pw}" y2="${mT+ph}" stroke="${axis}" stroke-width="1"/>`;
  g+=`<text class="axlbl" x="${mL+pw/2}" y="${H-6}" text-anchor="middle">standing cache-miss rate f</text>`;
  g+=`<text class="axlbl" x="${12}" y="${mT+ph/2}" text-anchor="middle" transform="rotate(-90 12 ${mT+ph/2})">B* — simultaneous misses (log)</text>`;
  document.getElementById('chartF').innerHTML =
    svgEl(g,W,H,'Cold-spike tolerance versus the standing cache-miss rate, with the MFU bracket');
  spikeGeom = { W,H,mL,mR,mT,pw,ph, d, sx, sy };
}

/* ---- The four ceilings, side by side -------------------------------------
   Act 3's thesis is "every constraint in one unit, so the binding one is
   simply the smallest". Rendering that as a comma-separated sub-line asked the
   reader to do the comparison in their head; a shared linear axis does it for
   them, and shows the HEADROOM (how far the load sits from each ceiling),
   which the ranking alone does not.
   -------------------------------------------------------------------------- */
export function renderCeilingBars(op){
  const box = document.getElementById('ceilingBars');
  if (!box) return;
  if (!op){ box.innerHTML = '<p class="cs">model weights do not fit this configuration.</p>'; return; }
  const C = PLANNER_COLORS(), keys = ['cache','decode','latency','saturation'];
  const W=1120, rowH=46, mT=10, mL=112, mR=92, H=mT+rowH*keys.length+34;
  const pw = W-mL-mR;
  const top = Math.max(op.users, ...keys.map(k=>op.ceilings[k]).filter(isFinite))*1.08 || 1;
  const sx = linScale(0, top, mL, mL+pw);
  const grid=cssv('--grid'), muted=cssv('--muted'), text=cssv('--text');
  const crit=cssv('--crit'), tile=cssv('--tile');
  let g='';
  for (const t of niceTicks(top, 5)){
    g+=`<line x1="${sx(t)}" y1="${mT}" x2="${sx(t)}" y2="${mT+rowH*keys.length}" stroke="${grid}" stroke-width="1"/>`;
    g+=`<text class="axtick" x="${sx(t)}" y="${mT+rowH*keys.length+16}" text-anchor="middle">${fmt(t,0)}</text>`;
  }
  keys.forEach((k,i)=>{
    const y = mT + i*rowH + 9, v = op.ceilings[k], bind = k===op.binding;
    g+=`<rect x="${mL}" y="${y}" width="${pw}" height="${rowH-20}" fill="${tile}" rx="4"/>`;
    if (isFinite(v))
      g+=`<rect x="${mL}" y="${y}" width="${Math.max(2,sx(v)-mL)}" height="${rowH-20}" fill="${C[k]}" opacity="${bind?0.95:0.4}" rx="4"/>`;
    g+=`<text class="axlbl" x="${mL-10}" y="${y+18}" text-anchor="end" fill="${bind?C[k]:muted}"`
      +`${bind?' font-weight="700"':''}>${esc(PLANNER_LABEL[k])}</text>`;
    g+=`<text class="dlabel" x="${mL+pw+8}" y="${y+18}" text-anchor="start" fill="${bind?C[k]:muted}"`
      +`${bind?' font-weight="700"':''}>${isFinite(v)?fmt(v,0):'—'}${bind?' ← binds':''}</text>`;
  });
  // the load you asked for, across all four
  const X = sx(Math.min(op.users, top));
  g+=`<line x1="${X}" y1="${mT-4}" x2="${X}" y2="${mT+rowH*keys.length+2}" stroke="${op.fits?text:crit}" stroke-width="2" stroke-dasharray="4 3"/>`;
  g+=`<text class="dlabel" x="${X+6}" y="${mT+rowH*keys.length+14}" text-anchor="start" fill="${op.fits?text:crit}">your load ${fmt(op.users,0)}${op.fits?'':' — over'}</text>`;
  box.innerHTML = svgEl(g, W, H, 'The four ceilings compared in max concurrent users, with the current load marked');
}
