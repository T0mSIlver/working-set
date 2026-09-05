import { DECODE_MBU, GIB, PREFILL_MFU, PREFILL_MFU_HI, PREFILL_MFU_LO, effective_bw,
         kv_pool_tokens, w_decode } from './config.js';
import { contextStats, decodeFloor, liveTurn, maxUsersLatency, maxUsersSaturation,
         prefillChunk, prefillServiceMoments, setLiveTurn } from './prefill.js';
import { clip, seedFor } from './mathlib.js';
import { p_sub, sampleReqInto } from './workload.js';
import { ramPerCache, state } from './state.js';
import { cssv, esc, fmt, linScale, svgEl } from './svg.js';
import { PLANNER_COLORS, PLANNER_LABEL, fAxisMax } from './planner.js';

/* ---- "What would flip this decision" — the sensitivity panel --------------
   Chart G already sweeps ONE assumption (the miss rate) and marks where the
   binding constraint changes hands; this panel does the same for every soft
   input at once, in one glance. Each row sweeps one assumption across its
   plausible range holding the rest fixed, colours the track by which
   constraint binds there, underlines where the current load stops fitting,
   and names the nearest handover. Rows sort by distance-to-flip: the top row
   is the assumption the current decision leans on hardest.

   The two sampled ceilings (cache, decode) cannot be Monte-Carlo re-filled at
   every point of every sweep, so the workload axes use MEAN-FIELD stand-ins —
   expected pool cost per session, mean-context decode speed — each CALIBRATED
   so it passes exactly through the MC value at the current settings. The
   approximation is stated in the panel caption; the directly-priced axes (miss
   rate via the same sampled warmFn chart G uses, think time, budget, turn,
   MFU) carry no such caveat. */

// Mean-field warm USER capacity of one replica group: expected pool cost per
// resident session against both budgets, mirroring warmOnce's accounting
// (cold sessions occupy their full length but do not count as warm; every
// resident holds its recurrent state; one reserved block per distinct prefix).
// Returns a MEAN, not a p5 — the flip calibration ratio absorbs the gap.
function warmUsersApprox(model, topo, wl, nSamp){
  const pool = kv_pool_tokens(model, topo);
  if (pool <= 0) return 0;
  let reserved = wl.sys_user;
  if (!wl.sub_shares_prefix && wl.sub_ratio > 0) reserved += wl.sys_sub;
  const stateTok = model.deltanet_state / model.kv_bpt;
  let s = 0; const r = {full:0, prefix:0, isCold:false};
  for (let i = 0; i < nSamp; i++){
    sampleReqInto(wl, r);
    s += (r.isCold ? r.full : Math.max(r.full - r.prefix, 0)) + stateTok;
  }
  const eU = s / nSamp || 1;
  const ram = ramPerCache(topo);
  const gpuB = Math.max(0, pool - reserved);
  const ramB = ram > 0 ? Math.max(0, ram * GIB / model.kv_bpt - reserved) : 0;
  return (gpuB + ramB) / eU * (1 - wl.invalidation) * (1 - p_sub(wl));
}

// Mean-context decode ceiling of one replica group: the concurrency at which
// per-user speed hits the floor, with every sequence priced at the MEAN
// context instead of its own draw (the MC bisection prices the spread; the
// calibration ratio absorbs the difference). Monotone, so plain bisection.
function decodeUsersApprox(model, topo, samples, floor){
  // default to the LIVE floor, not the constant: every caller here omits the
  // argument, and a fixed 40 would leave the sensitivity panel's stand-in on a
  // different threshold from the MC ceiling it is calibrated against — calD
  // would silently absorb the mismatch at the anchor and be wrong everywhere else
  const F = floor || decodeFloor(), bw = effective_bw(topo) * (model.decode_mbu || DECODE_MBU);
  const tk = (model.kv_decode_const && model.kv_decode_topk) ? model.kv_decode_topk : 0;
  let mL = 0, mT = 0;
  for (let i = 0; i < samples.length; i++){
    mL += samples[i];
    if (tk) mT += Math.min(samples[i], tk);
  }
  mL /= samples.length; if (tk) mT /= samples.length;
  const kvReadBpt = model.kv_decode_bpt ?? model.kv_bpt;
  const perSeq = mL * kvReadBpt + 2 * model.deltanet_state
    + (model.kv_decode_const ? (tk ? mT * (model.kv_decode_const / tk)
                                   : model.kv_decode_const) : 0);
  const speed = n => model.mtp * bw / (w_decode(model, n) + n * perSeq);
  if (speed(1) < F) return 0;
  let lo = 1, hi = 2;
  while (speed(hi) >= F && hi < 1e7){ lo = hi; hi *= 2; }
  for (let i = 0; i < 50; i++){ const mid = (lo + hi) / 2; if (speed(mid) >= F) lo = mid; else hi = mid; }
  return lo;
}

export let lastFlipAxes = null;
export function setLastFlipAxes(v){ lastFlipAxes = v; }
export function computeFlipData(model, topo, wl, cs, mo, warmFn, op, reps){
  const f0 = wl.invalidation, subR = wl.sub_ratio;
  const cacheNow = op.ceilings.cache, decodeNow = op.ceilings.decode;
  const evalAt = (mo_, f_, sla_, think_, cacheU, decodeU) => {
    const c = { cache: cacheU, decode: decodeU,
      latency: reps * maxUsersLatency(mo_, f_, sla_, think_, undefined, subR),
      saturation: reps * maxUsersSaturation(mo_, f_, think_, subR) };
    let bind = 'cache';
    for (const k in c) if (c[k] < c[bind]) bind = k;
    return { bind, limit: c[bind] };
  };
  // calibration anchors: the stand-ins, forced through the MC values at the
  // current settings, so every sweep passes exactly through the decision tile
  seedFor('flipCal');
  const aC = warmUsersApprox(model, topo, wl, 4000);
  const calC = aC > 0 ? (cacheNow / reps) / aC : 0;
  const aD = decodeUsersApprox(model, topo, cs.samples);
  const calD = aD > 0 ? (decodeNow / reps) / aD : 0;

  const axes = [];
  const sweep = (K, lo, hi, at, cur) => {
    const vs = [];
    for (let i = 0; i < K; i++) vs.push(lo + (hi - lo) * i / (K - 1));
    // the sweep must contain the CURRENT value exactly: the stand-ins are
    // calibrated to pass through the MC ceilings there, and a coarse grid
    // whose nearest sample sits half a stride away can contradict the
    // decision tile (fits at a 31k median, sampled at 45k → "does not fit")
    if (cur !== undefined && cur >= lo && cur <= hi && !vs.includes(cur)){
      vs.push(cur); vs.sort((a, b) => a - b);
    }
    return vs.map((v, i) => ({ v, ...at(v, i) }));
  };
  // 1 · cache-miss rate — warmFn is the SAME sampled curve chart G uses, so
  //     this row has to sweep the SAME axis: at f0 > 0.5 a 0–0.5 sweep never
  //     contains the current value (sweep() only injects `cur` when it falls
  //     inside [lo,hi]), so the marker pins to the endpoint and the handover
  //     and fits verdicts are read off a baseline that is not the live one.
  //     Point count scales with the range to hold the 0.0125 stride.
  const fMax = fAxisMax();
  axes.push({ label: 'Cache-miss rate', cur: f0, lo: 0, hi: fMax,
    fmt: v => fmt(v * 100, v > 0 && v < 0.02 ? 1 : 0) + '%', approx: false,
    pts: sweep(Math.round(80*fMax) + 1, 0, fMax, v =>
      evalAt(mo, v, state.sla, state.think, warmFn(v) * reps, decodeNow), f0) });
  // 2 · think time — enters every rate linearly, closed form
  axes.push({ label: 'Think time', cur: state.think, lo: 5, hi: 180,
    fmt: v => fmt(v, 0) + ' s', approx: false,
    pts: sweep(36, 5, 180, v =>
      evalAt(mo, f0, state.sla, v, cacheNow, decodeNow), state.think) });
  // 3 · TTFT budget — moves the latency ceiling only, closed form
  axes.push({ label: 'TTFT budget', cur: state.sla, lo: 1, hi: 60,
    fmt: v => fmt(v, 0) + ' s', approx: false,
    pts: sweep(40, 1, 60, v =>
      evalAt(mo, f0, v, state.think, cacheNow, decodeNow), state.sla) });
  // 4 · warm turn size — re-prices the hit leg of the service moments; the
  //     moments read the liveTurn mirror, so swap it per point and restore
  {
    // try/finally: an exception mid-sweep must not leave the liveTurn mirror
    // poisoned for every later render until the next syncState()
    const saved = liveTurn;
    let pts;
    try {
      pts = sweep(17, 250, 16000, v => {
        setLiveTurn(v);
        const mo2 = prefillServiceMoments(model, topo, wl, cs, prefillChunk());
        return evalAt(mo2, f0, state.sla, state.think, cacheNow, decodeNow);
      }, state.turn);
    } finally { setLiveTurn(saved); }
    axes.push({ label: 'Warm turn size', cur: state.turn, lo: 250, hi: 16000,
      fmt: v => fmt(v / 1000, v < 1000 ? 2 : 1) + 'k tok', approx: false, pts });
  }
  // 5 · prefill MFU — the study's softest input; not a slider, a structural
  //     unknown, so the sweep runs over the whole stated [35%, 55%] bracket
  axes.push({ label: 'Prefill MFU', cur: PREFILL_MFU, lo: PREFILL_MFU_LO, hi: PREFILL_MFU_HI,
    fmt: v => fmt(v * 100, 0) + '%', approx: false,
    pts: sweep(16, PREFILL_MFU_LO, PREFILL_MFU_HI, v => {
      const mo2 = prefillServiceMoments(model, topo, wl, cs, prefillChunk(), v);
      return evalAt(mo2, f0, state.sla, state.think, cacheNow, decodeNow);
    }, PREFILL_MFU) });
  // 6 · speculative-decode speedup — decode speed is exactly linear in it,
  //     so the mean-context stand-in only has to move the crossing point
  if (calD > 0)
    axes.push({ label: 'Speculative speedup', cur: state.mtp, lo: 1.0, hi: 3.0,
      fmt: v => '×' + v.toFixed(2), approx: true,
      pts: sweep(21, 1.0, 3.0, v =>
        evalAt(mo, f0, state.sla, state.think, cacheNow,
               decodeUsersApprox({ ...model, mtp: v }, topo, cs.samples) * calD * reps), state.mtp) });
  // 6a · decode efficiency (MBU) — one constant for every row, measured on
  //      one deployment; decode speed is exactly linear in it, like the
  //      speedup, so the same mean-context stand-in serves
  if (calD > 0)
    axes.push({ label: 'Decode MBU', cur: state.mbu, lo: 0.10, hi: 1.00,
      fmt: v => fmt(v * 100, 0) + '%', approx: true,
      pts: sweep(19, 0.10, 1.00, v =>
        evalAt(mo, f0, state.sla, state.think, cacheNow,
               decodeUsersApprox({ ...model, decode_mbu: v }, topo, cs.samples) * calD * reps), state.mbu) });
  // 6b · per-user decode floor — moves the DECODE ceiling only, and steeply
  //      (the ceiling is the reciprocal of per-user speed). This is the row
  //      that answers "is my deployment actually decode-bound, or just judged
  //      against a floor fitted to a different workload?"
  if (calD > 0)
    axes.push({ label: 'Decode floor', cur: state.decode_floor, lo: 5, hi: 100,
      fmt: v => fmt(v, 0) + ' tok/s', approx: true,
      pts: sweep(20, 5, 100, v =>
        evalAt(mo, f0, state.sla, state.think, cacheNow,
               decodeUsersApprox(model, topo, cs.samples, v) * calD * reps),
        state.decode_floor) });
  // 7 & 8 · the prompt-length distribution — median and shape. Both re-draw
  //     the context stats (4k samples: flip resolution, not tile precision)
  //     and move ALL FOUR ceilings through them.
  const wlAxis = (label, key, lo, hi, cur, fmtFn) => {
    // seeds are keyed by VALUE, not sweep index: inserting the current value
    // into the grid must not shift every later point onto different draws
    // (cross-state stability), and the anchor below must reproduce the cur
    // sample bit-for-bit
    const at = v => {
      const wl2 = { ...wl, [key]: key === 'user_median' ? v * 1000 : v };
      seedFor('flip|' + key + '|' + v.toFixed(4));
      const cs2 = contextStats(wl2, 4000);
      const mo2 = prefillServiceMoments(model, topo, wl2, cs2, prefillChunk());
      seedFor('flip|' + key + 'w|' + v.toFixed(4));
      return { mo2, warm: warmUsersApprox(model, topo, wl2, 3000),
               dec: decodeUsersApprox(model, topo, cs2.samples) };
    };
    // per-axis anchor at the CURRENT value, same seeds and sample counts as
    // the sweep — so the calibrated curve passes exactly through the MC
    // ceilings at cur (the global flipCal anchor uses different counts and
    // would leave ~1/sqrt(N) daylight between the ▼ and the decision tile)
    const a = at(cur);
    const cC = a.warm > 0 ? (cacheNow / reps) / a.warm : 0;
    const cD = a.dec > 0 ? (decodeNow / reps) / a.dec : 0;
    if (!(cC > 0) || !(cD > 0)) return;
    axes.push({ label, cur, lo, hi, fmt: fmtFn, approx: true,
      pts: sweep(13, lo, hi, v => {
        const q = at(v);
        return evalAt(q.mo2, f0, state.sla, state.think,
                      q.warm * cC * reps, q.dec * cD * reps);
      }, cur) });
  };
  {
    const el = document.getElementById('s-user_median');
    wlAxis('User prompt median', 'user_median',
           parseFloat(el.min), parseFloat(el.max), state.user_median,
           v => fmt(v, 0) + 'k');
    wlAxis('User prompt sigma', 'user_sigma', 0.30, 1.40, state.user_sigma,
           v => v.toFixed(2));
  }
  // nearest flip per axis: first sample (scanning outward from the current
  // value, both directions) where the binding IDENTITY or the fits VERDICT
  // changes; distance is normalised by the range so the sort is unitless
  const users = op.users;
  for (const ax of axes){
    let ci = 0;
    for (let i = 1; i < ax.pts.length; i++)
      if (Math.abs(ax.pts[i].v - ax.cur) < Math.abs(ax.pts[ci].v - ax.cur)) ci = i;
    ax.ci = ci;
    const base = ax.pts[ci], fits0 = base.limit >= users;
    let best = null;
    for (const dir of [1, -1])
      for (let i = ci + dir; i >= 0 && i < ax.pts.length; i += dir){
        const p = ax.pts[i];
        const bindFlip = p.bind !== base.bind, fitFlip = (p.limit >= users) !== fits0;
        if (bindFlip || fitFlip){
          const d = Math.abs(p.v - ax.cur) / (ax.hi - ax.lo);
          if (!best || d < best.dist)
            best = { v: p.v, prev: ax.pts[i - dir].v, bind: p.bind,
                     bindFlip, fitFlip, dir, dist: d, nowFits: fits0 };
          break;
        }
      }
    ax.flip = best;
  }
  axes.sort((a, b) => (a.flip ? a.flip.dist : Infinity) - (b.flip ? b.flip.dist : Infinity));
  // stamp the population every verdict was judged against: a draft render
  // reuses this sweep while state.users moves, and the underline must not
  // slide against flip labels that still say the old count
  for (const ax of axes) ax.users = users;
  return axes;
}

export function renderFlipPanel(axes){
  const box = document.getElementById('flipRows'), leg = document.getElementById('flipLegend');
  if (!box) return;
  if (!axes){
    leg.innerHTML = '';
    box.innerHTML = '<p class="cs">model weights do not fit this configuration — nothing binds because nothing runs.</p>';
    return;
  }
  const C = PLANNER_COLORS(), crit = cssv('--crit'), muted = cssv('--muted');
  const text = cssv('--text'), good = cssv('--good');
  leg.innerHTML = ['cache', 'decode', 'latency', 'saturation'].map(k =>
    `<span class="li"><span class="sw" style="background:${C[k]}"></span>${PLANNER_LABEL[k]} binds</span>`).join('')
    + `<span class="li"><span class="sw" style="background:${crit};height:3px"></span>your load does not fit</span>`
    + `<span class="li"><span style="color:${text};font-size:10px">▼</span>&nbsp;your value</span>`
    + `<span class="li"><span class="sw" style="width:0;height:12px;border-left:2px dashed ${muted};background:none;border-radius:0"></span>nearest flip — named at the right, in its own colour</span>`;
  const W = 1120, rowH = 44, mT = 6, mL = 172, mR = 320, H = mT + rowH * axes.length + 6;
  const pw = W - mL - mR;
  let g = '';
  axes.forEach((ax, ri) => {
    const y = mT + ri * rowH, yc = y + 24;
    const sx = linScale(ax.lo, ax.hi, mL, mL + pw);
    // label + current value
    g += `<text class="axlbl" x="${mL - 10}" y="${yc - 2}" text-anchor="end" fill="${text}">${esc(ax.label)}${ax.approx ? ' <tspan fill="' + muted + '">≈</tspan>' : ''}</text>`;
    g += `<text class="axtick" x="${mL - 10}" y="${yc + 11}" text-anchor="end">now ${ax.fmt(ax.cur)}</text>`;
    // track: contiguous runs of the same binding constraint
    const pts = ax.pts;
    let i = 0;
    while (i < pts.length){
      let j = i;
      while (j + 1 < pts.length && pts[j + 1].bind === pts[i].bind) j++;
      const x0 = i === 0 ? sx(pts[0].v) : (sx(pts[i - 1].v) + sx(pts[i].v)) / 2;
      const x1 = j === pts.length - 1 ? sx(pts[j].v) : (sx(pts[j].v) + sx(pts[j + 1].v)) / 2;
      g += `<rect x="${x0}" y="${yc - 5}" width="${Math.max(1, x1 - x0)}" height="10" fill="${C[pts[i].bind]}" opacity="0.8"/>`;
      i = j + 1;
    }
    // the does-not-fit underline, same segment logic on the verdict
    i = 0;
    while (i < pts.length){
      if (pts[i].limit >= ax.users){ i++; continue; }
      let j = i;
      while (j + 1 < pts.length && pts[j + 1].limit < ax.users) j++;
      const x0 = i === 0 ? sx(pts[0].v) : (sx(pts[i - 1].v) + sx(pts[i].v)) / 2;
      const x1 = j === pts.length - 1 ? sx(pts[j].v) : (sx(pts[j].v) + sx(pts[j + 1].v)) / 2;
      g += `<rect x="${x0}" y="${yc + 7}" width="${Math.max(1, x1 - x0)}" height="3" rx="1.5" fill="${crit}"/>`;
      i = j + 1;
    }
    // range end labels
    g += `<text class="axtick" x="${mL - 2}" y="${yc + 20}" text-anchor="start">${ax.fmt(ax.lo)}</text>`;
    g += `<text class="axtick" x="${mL + pw + 2}" y="${yc + 20}" text-anchor="end">${ax.fmt(ax.hi)}</text>`;
    // current-value marker
    const xc = sx(clip(ax.cur, ax.lo, ax.hi));
    g += `<path d="M ${xc - 5} ${yc - 13} L ${xc + 5} ${yc - 13} L ${xc} ${yc - 6} Z" fill="${text}"/>`;
    // flip marker + right-hand verdict
    if (ax.flip){
      // the tick marks the VISUAL transition, so it must share the bands'
      // convention: segment edges sit at the midpoint between adjacent
      // samples, not at the first sample past the flip (half a stride off)
      const xf = (sx(ax.flip.prev) + sx(ax.flip.v)) / 2;
      const side = ax.flip.dir > 0 ? '≥' : '≤';
      const what = ax.flip.fitFlip
        ? (ax.flip.nowFits ? `stops fitting your ${fmt(ax.users, 0)} users at ${side} ${ax.fmt(ax.flip.v)}`
                           : `fits your ${fmt(ax.users, 0)} users at ${side} ${ax.fmt(ax.flip.v)}`)
        : `${PLANNER_LABEL[ax.flip.bind]} takes over at ${side} ${ax.fmt(ax.flip.v)}`;
      // the tick, its label and the thing that changes at it share ONE colour:
      // a verdict flip is red/green (and lands where the red underline starts
      // or stops), a binding handover wears the incoming constraint's colour
      const col = ax.flip.fitFlip ? (ax.flip.nowFits ? crit : good) : C[ax.flip.bind];
      g += `<line x1="${xf}" y1="${yc - 10}" x2="${xf}" y2="${yc + 9}" stroke="${col}" stroke-width="1.6" stroke-dasharray="3 2"/>`;
      g += `<text class="dlabel" x="${mL + pw + 14}" y="${yc + 3}" text-anchor="start" fill="${col}">${esc(what)}</text>`;
    } else {
      g += `<text class="dlabel" x="${mL + pw + 14}" y="${yc + 3}" text-anchor="start" fill="${muted}" font-weight="400">no handover in this range</text>`;
    }
  });
  box.innerHTML = svgEl(g, W, H,
    'Sensitivity of the decision: which constraint binds as each assumption sweeps its range, with the nearest flip named per row');
}
