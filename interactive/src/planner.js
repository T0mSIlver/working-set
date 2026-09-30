import { PREFILL_MFU_HI, PREFILL_MFU_LO } from './config.js';
import { SPIKE_SLA_S, decodeFloor, maxUsersColdWait, maxUsersSaturation, maxUsersSlowed, prefillChunk,
         prefillServiceMoments, serverRate, slowedShare, slowedSteps, stretchPct, coldWait,
         idleColdTtftTable } from './prefill.js';
import { clip } from './mathlib.js';
import { p_sub } from './workload.js';
import { warmCapacity } from './capacity.js';
import { hasHeadcount, peopleFromSessions, sessionsFromHeadcount, state } from './state.js';
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
  cold_wait: cssv('--s2'), slowed: cssv('--s3'), saturation: cssv('--muted'),
});
const PLANNER_DASH = { cache:'', decode:'7 3', cold_wait:'2 3', slowed:'5 2 1 2',
                       saturation:'11 4' };
export const PLANNER_LABEL = { cache:'cache', decode:'decode', cold_wait:'cold wait',
                        slowed:'slowed generation', saturation:'saturation' };
// the ceilings in the verdict (prefill.js operatingPoint), in display order
export const CEILING_KEYS = ['cache', 'decode', 'cold_wait', 'slowed', 'saturation'];
const ceilingText = value => hasHeadcount()
  ? `${fmt(value,0)} sessions ≈ ${fmt(peopleFromSessions(value),0)} people`
  : fmt(value,0);

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
export function warmUsersNow(p5, wl){ return p5 * (1 - p_sub(wl)); }
export function warmUsersCurve(model, topo, ram, iter, budget, wl0, fNow, warmNow){
  const fMax = fAxisMax();
  const anchors = [0, 0.25, 0.5, 0.75, 1.0].map(a => a*fMax);
  if (!anchors.includes(fNow)) anchors.push(fNow);
  anchors.sort((a,b)=>a-b);
  const pts = anchors.map(f => {
    if (f === fNow && warmNow !== undefined) return {f, u: warmNow};
    const wl = {...wl0, invalidation: f};
    const wc = warmCapacity(model, topo, wl, ram, iter, budget);
    return { f, u: warmUsersNow(wc.all[0], wl) };
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
  const think = state.think, sla = SPIKE_SLA_S, reps = topo.replicas || 1;
  // the ceilings are per replica GROUP, so the load one group sees is the
  // system population divided by the replica count
  const perGroup = u => u * reps;
  const series = { cache:[], decode:[], cold_wait:[], slowed:[], saturation:[], env:[], binding:[] };
  const steps = slowedSteps(model, topo, cs, prefillChunk());
  const spike = { mid:[], lo:[], hi:[] };
  const moLo = prefillServiceMoments(model, topo, wl, cs, prefillChunk(), PREFILL_MFU_LO);
  const moHi = prefillServiceMoments(model, topo, wl, cs, prefillChunk(), PREFILL_MFU_HI);
  for (const f of fs){
    const c = {
      cache: warmFn(f) * reps,
      decode: decodeUsers * reps,
      cold_wait: perGroup(maxUsersColdWait(mo, f, state.cold_wait, think,
                                           wl.sub_ratio, state.ttft_pct)),
      slowed: perGroup(maxUsersSlowed(mo, steps, f, state.slowed_pct/100, think,
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
  const waitStat = state.ttft_pct === 'mean' ? 'mean' : `p${state.ttft_pct}`;
  const drain = op.burstDrain, lastTTFT = drain;
  const good=cssv('--good'), warn=cssv('--warn'), crit=cssv('--crit');
  const headClass = op.headroom >= 1 ? crit : (op.headroom >= 0.8 ? warn : good);
  const others = Object.keys(op.ceilings).filter(k=>k!==bind)
    .sort((a,b)=>op.ceilings[a]-op.ceilings[b])
    .map(k=>`${PLANNER_LABEL[k]} ${ceilingText(op.ceilings[k])}`).join(' · ');
  const tiles = [
    {decision:true, k:'Binding constraint', hero:true, v:PLANNER_LABEL[bind].toUpperCase(),
     u: op.limit >= 1
        ? (hasHeadcount() ? `at ${ceilingText(op.limit)}` : `at ${fmt(op.limit,0)} users`)
        : 'at no load at all',
     sub: op.limit >= 1
        ? (hasHeadcount()
          ? `your ${fmt(state.headcount,0)} people produce ${fmt(sessionsFromHeadcount(state.headcount, state.active, state.spu),0)} sessions`
            + (Math.abs(sessionsFromHeadcount(state.headcount, state.active, state.spu) - op.users) < 1e-6 ? '' : ` (priced at the slider's ${fmt(op.users,0)})`)
            + ` — ${fmt(op.headroom*100,0)}% of the limit`
          : `you are running ${fmt(op.users,0)} — ${fmt(op.headroom*100,0)}% of the limit`)
          + ` · next: ${others}`
        : `${PLANNER_LABEL[bind]} is over its limit even for one user — unachievable at any load`
          + ` · next: ${others}`,
     cls: op.headroom>=1?'crit':(op.headroom>=0.8?'warn':'good'),
     tip:hasHeadcount()
       ? `All five ceilings are concurrent sessions, with people equivalents from the population inputs. The smallest binds. cache = the warm p5 population that fits the pool; decode = where per-session p50 hits the ${fmt(decodeFloor(),0)} tok/s floor; cold wait = where the ${waitStat} of a cold request's wait hits ${fmt(state.cold_wait,0)} s; slowed generation = where the share of generation time slowed by a cold prefill hits ${fmt(state.slowed_pct,0)}%; saturation = where prefill duty hits 100%.`
       : `All five ceilings in ONE unit — max concurrent users — so the binding one is simply the smallest. cache = the warm p5 population that fits the pool; decode = where per-user p50 hits the ${fmt(decodeFloor(),0)} tok/s floor; cold wait = where the ${waitStat} of a cold request's wait hits ${fmt(state.cold_wait,0)} s; slowed generation = where the share of generation time slowed by a cold prefill hits ${fmt(state.slowed_pct,0)}%; saturation = where prefill duty hits 100%. The conversion rests on the Concurrent-users assumptions; chart G shows where the binding constraint changes hands.`},
    {hero:true, k:'Cold-spike tolerance B*', v:fmt(op.bstar,1), u:'misses at once',
     sub:`MFU 30–55% band: ${fmt(op.bstarLo,1)}–${fmt(op.bstarHi,1)}`
        + ` · zero at f* ${op.fstar>10?'> 1,000':fmt(op.fstar*100,0)+'%'}`
        + ` · a miss's mean first token reaches ${fmt(SPIKE_SLA_S,0)} s at f ${op.fsla>=1?'never':fmt(op.fsla*100,0)+'%'}`,
     cls: op.bstar<1?'crit':(op.bstar<5?'warn':'good'),
     tip:"The largest burst of SIMULTANEOUS misses whose last request still gets a first token inside a fixed 10 s time to first token — linear in that figure. It is a burst measure and does not enter the verdict. The band is the MFU [30–55%] bracket; B* reaches zero exactly at f*."},
    {k:'Slowed generation',
     v: fmt(op.slowedNow*100, op.slowedNow < 0.1 ? 1 : 0), u:'% of generation time',
     sub: `each stream drops to ${fmt(op.steps.speed,1)} tok/s while a chunk rides · limit ${fmt(state.slowed_pct,0)}%`,
     cls: op.slowedNow*100>state.slowed_pct?'crit':(op.slowedNow*100>state.slowed_pct*0.8?'warn':'good'),
     tip:"The share of wall time in which a cold prompt is being prefilled. Every GPU step of that prefill carries one chunk on top of the running decodes, so each stream runs at the tok/s shown instead of its normal speed. Warm-turn prefills are left out."},
    {k:'One long cold request',
     v: isFinite(op.stretchNow) ? fmt(op.stretchNow, op.stretchNow < 10 ? 1 : 0) : '∞', u:'s slowed',
     sub: `a p${stretchPct(state.ttft_pct)}-length cold prompt keeps every stream at ${fmt(op.steps.speed,1)} tok/s`,
     cls: 'good',
     tip:"How long one cold prompt of the p-th percentile length keeps every stream on the group slowed: its decode steps plus its chunked prefill, at the speed shown."},
    {k:'Cold request wait',
     v: isFinite(op.coldWaitNow) ? fmt(op.coldWaitNow,2) : '∞', u: isFinite(op.coldWaitNow) ? `s (${waitStat})` : 'queue unbounded',
     sub: sp.rho >= 1
        ? `prefill duty ${fmt(op.duty*100,0)}% — the queue is unbounded at this load`
        : `mean ${fmt(op.coldWaitMean,2)} s · limit ${fmt(state.cold_wait,0)} s · duty ${fmt(op.duty*100,0)}%`,
     cls: op.coldWaitNow>state.cold_wait?'crit':(op.coldWaitNow>state.cold_wait*0.8?'warn':'good'),
     tip:"How long a cold request queues behind the prefills already waiting before its own starts, at the current load. The percentile reads an exponential tail off the queue's mean wait; an approximation."},
    {full:true, k:`A burst of ${fmt(state.burst,0)} at once`,
     v: !isFinite(drain) ? 'never' : (drain>=90? fmt(drain/60,1) : fmt(drain,1)),
     u: !isFinite(drain) ? 'clears at this load' : (drain>=90?'min to clear':'s to clear'),
     sub: !isFinite(drain)
        ? `the standing load already saturates prefill, so a burst on top of it never drains`
        : `last request waits ${lastTTFT>=90?fmt(lastTTFT/60,1)+' min':fmt(lastTTFT,1)+' s'}`
          + ` (reference ${fmt(SPIKE_SLA_S,0)} s) · every warm user loses ~${fmt(op.tokensLost,0)} output tokens`,
     cls: drain>SPIKE_SLA_S?'crit':'good',
     tip:"What a correlated invalidation event costs. The backlog drains at (1 − duty) seconds of work per second — standing traffic keeps arriving — so the last request's TTFT IS the drain time; meanwhile the ITL spike is the steady state, and the tokens-lost figure integrates what the warm users stop receiving over the drain."},
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

/* ---- The binding-constraint chart (act 3, rendered as 'G'): the five
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
  const all = [...CEILING_KEYS.flatMap(k => d[k]), op.users]
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
  for (const k of CEILING_KEYS)
    g+=`<path d="${path(d[k])}" fill="none" stroke="${C[k]}" stroke-width="2" stroke-linejoin="round"`
      +`${PLANNER_DASH[k]?` stroke-dasharray="${PLANNER_DASH[k]}"`:''}/>`;
  // ...and direct-label each line at its right terminus, so the chart is
  // readable without cross-referencing a legend
  for (const k of CEILING_KEYS){
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
        +`${esc(PLANNER_LABEL[d.binding[i-1]])} → ${esc(PLANNER_LABEL[d.binding[i]])} at ${fmt(d.fs[i]*100,0)}%</text>`;
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
    +`${over?'≥ ':''}${fmt(op.users,0)} ${hasHeadcount()?'sessions':'users'}</text>`;
  g+=`<line x1="${mL}" y1="${mT+ph}" x2="${mL+pw}" y2="${mT+ph}" stroke="${axis}" stroke-width="1"/>`;
  g+=`<text class="axlbl" x="${mL+pw/2}" y="${H-6}" text-anchor="middle">cache-miss rate f</text>`;
  g+=`<text class="axlbl" x="${12}" y="${mT+ph/2}" text-anchor="middle" transform="rotate(-90 12 ${mT+ph/2})">max concurrent ${hasHeadcount()?'sessions':'users'} (log)</text>`;
  document.getElementById('chartG').innerHTML =
    svgEl(g,W,H,`The five ceilings in max concurrent ${hasHeadcount()?'sessions':'users'} versus the cache-miss rate`);
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

/* ---- The five ceilings, side by side -------------------------------------
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
  const C = PLANNER_COLORS(), keys = CEILING_KEYS;
  const W=1120, rowH=46, mT=10, mL=112, mR=(hasHeadcount()?230:92)+(op.decodeCapped?150:0);
  const H=mT+rowH*keys.length+34;
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
      +`${bind?' font-weight="700"':''}>${isFinite(v)?ceilingText(v):'—'}`
      +`${k==='decode'&&op.decodeCapped?` (max_num_seqs ${fmt(op.mns,0)} at p99)`:''}${bind?' ← binds':''}</text>`;
  });
  // the load you asked for, across all five
  const X = sx(Math.min(op.users, top));
  g+=`<line x1="${X}" y1="${mT-4}" x2="${X}" y2="${mT+rowH*keys.length+2}" stroke="${op.fits?text:crit}" stroke-width="2" stroke-dasharray="4 3"/>`;
  g+=`<text class="dlabel" x="${X+6}" y="${mT+rowH*keys.length+14}" text-anchor="start" fill="${op.fits?text:crit}">your load ${fmt(op.users,0)}${op.fits?'':' — over'}</text>`;
  box.innerHTML = svgEl(g, W, H, hasHeadcount()
    ? 'The five concurrent-session ceilings with people equivalents and the current load marked'
    : 'The five ceilings compared in max concurrent users, with the current load marked');
}

/* ---- Slowed generation and cold request wait against load (act 2, 'I'):
   the two limits users feel, each with its limit as a horizontal line and
   the current load as a vertical marker. Two panels rather than two y-axes:
   the units (% of time, seconds) share nothing. Per replica group at
   rate = serverRate(users)/reps, as the ceilings are. ---- */
function latencyPanel(divId, label, ylab, ys, yMax, limit, users, xMax, color, fmtY, fmtNow){
  const W=560, H=300, mL=52, mR=16, mT=14, mB=42;
  const pw=W-mL-mR, ph=H-mT-mB;
  const grid=cssv('--grid'), axis=cssv('--axis'), muted=cssv('--muted');
  const surface=cssv('--surface'), crit=cssv('--crit');
  const sx=linScale(0,xMax,mL,mL+pw), sy=linScale(0,yMax,mT+ph,mT);
  let g='';
  for (const t of niceTicks(yMax,4)){
    const Y=sy(t);
    g+=`<line x1="${mL}" y1="${Y}" x2="${mL+pw}" y2="${Y}" stroke="${grid}" stroke-width="1"/>`;
    g+=`<text class="axtick" x="${mL-8}" y="${Y+3}" text-anchor="end">${fmtY(t)}</text>`;
  }
  for (const t of niceTicks(xMax,5)){
    g+=`<text class="axtick" x="${sx(t)}" y="${mT+ph+16}" text-anchor="middle">${fmt(t,0)}</text>`;
  }
  const pts = ys.map(([u,y]) => `${sx(u)} ${sy(clip(y,0,yMax))}`);
  g+=`<path d="M ${pts.join(' L ')}" fill="none" stroke="${color}" stroke-width="2.4" stroke-linejoin="round"/>`;
  // the limit
  if (limit <= yMax){
    const Y=sy(limit);
    g+=`<line x1="${mL}" y1="${Y}" x2="${mL+pw}" y2="${Y}" stroke="${crit}" stroke-width="1.4" stroke-dasharray="4 3"/>`;
    g+=`<text class="dlabel" x="${mL+pw-4}" y="${Y-6}" text-anchor="end" fill="${crit}">limit ${fmtY(limit)}</text>`;
  }
  // the current load
  const X0=sx(clip(users,0,xMax));
  g+=`<line x1="${X0}" y1="${mT}" x2="${X0}" y2="${mT+ph}" stroke="${muted}" stroke-width="1.2" stroke-dasharray="3 3"/>`;
  const flip = X0 > mL + pw*0.7;
  g+=`<text class="dlabel" x="${X0+(flip?-5:5)}" y="${mT+12}" text-anchor="${flip?'end':'start'}" fill="${muted}">now ${fmt(users,0)} · ${fmtNow}</text>`;
  g+=`<line x1="${mL}" y1="${mT+ph}" x2="${mL+pw}" y2="${mT+ph}" stroke="${axis}" stroke-width="1"/>`;
  g+=`<text class="axlbl" x="${mL+pw/2}" y="${H-6}" text-anchor="middle">concurrent ${hasHeadcount()?'sessions':'users'}</text>`;
  g+=`<text class="axlbl" x="${12}" y="${mT+ph/2}" text-anchor="middle" transform="rotate(-90 12 ${mT+ph/2})">${esc(ylab)}</text>`;
  document.getElementById(divId).innerHTML = svgEl(g,W,H,label);
}

export function renderLatencyCharts(op, model, topo, wl, mo){
  const tbl = document.getElementById('idleTtftBody');
  if (!op){
    renderNoFit('chartI1','slowed generation'); renderNoFit('chartI2','cold request wait');
    tbl.innerHTML = '<tr><td colspan="2">the model does not fit</td></tr>';
    return;
  }
  const reps = topo.replicas || 1, f = wl.invalidation;
  const fin = [op.ceilings.slowed, op.ceilings.cold_wait].filter(v => isFinite(v) && v > 0);
  const xMax = Math.max(8, op.users*1.3, fin.length ? Math.max(...fin)*1.5 : 0);
  const N = 60, us = Array.from({length:N+1}, (_,i) => xMax*i/N);
  const rateAt = u => serverRate(u, state.think, wl.sub_ratio)/reps;
  const waitStat = state.ttft_pct === 'mean' ? 'mean' : `p${state.ttft_pct}`;
  const C = PLANNER_COLORS();
  const sPct = us.map(u => [u, 100*slowedShare(mo, op.steps, f, rateAt(u))]);
  latencyPanel('chartI1', 'Share of generation time slowed versus concurrent users, with its limit and the current load',
    'generation slowed, % of time', sPct, Math.min(100, Math.max(state.slowed_pct*2.5, op.slowedNow*130, 10)),
    state.slowed_pct, op.users, xMax, C.slowed, v => `${fmt(v,0)}%`, `${fmt(op.slowedNow*100,1)}%`);
  const wMax = Math.max(state.cold_wait*2.5, 10);
  const wS = us.map(u => [u, coldWait(mo, f, rateAt(u), state.ttft_pct)]);
  latencyPanel('chartI2', 'Cold request wait versus concurrent users, with its limit and the current load',
    `cold request wait (${waitStat}), s`, wS, wMax, state.cold_wait, op.users, xMax, C.cold_wait,
    v => `${fmt(v,0)} s`, isFinite(op.coldWaitNow) ? `${fmt(op.coldWaitNow,1)} s` : '∞');
  const rows = idleColdTtftTable(model, topo, prefillChunk(), wl.cap);
  tbl.innerHTML = rows.map(([L,sec]) =>
    `<tr><td>${fmt(L/1000,0)}k tokens${L === Math.trunc(wl.cap) ? ' (max_model_len)' : ''}</td>`
    + `<td class="num">${fmt(sec, sec < 10 ? 1 : 0)} s</td></tr>`).join('');
}
