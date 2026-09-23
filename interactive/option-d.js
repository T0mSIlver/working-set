// Option D: the headcount ruler. Reads the four ceilings the page already
// computed (the #ceilingBars labels, in concurrent sessions), adds the latency
// ceiling's MFU bracket from the same closed form, and redraws every time the
// page repaints #ceilingBars.
import { CONFIG, PREFILL_MFU_HI, PREFILL_MFU_LO } from './src/config.js';
import { state, currentTopo, currentWL } from './src/state.js';
import { activeModel, lastCS } from './src/render.js';
import { prefillServiceMoments, prefillChunk, maxUsersLatency } from './src/prefill.js';
import { PLANNER_COLORS, PLANNER_LABEL } from './src/planner.js';
import { cssv, esc, fmt, niceTicks } from './src/svg.js';

const KEYS = ['cache', 'decode', 'latency', 'saturation'];
const $ = id => document.getElementById(id);
const todayEl = $('dToday');

// "today" lives only on this page; carry it in the fragment next to the
// shared state so a reload keeps it
{
  const p = new URLSearchParams(location.hash.replace(/^#/, ''));
  const t = parseInt(p.get('today'), 10);
  if (Number.isFinite(t) && t >= 0) todayEl.value = String(t);
}
todayEl.addEventListener('input', () => {
  const p = new URLSearchParams(location.hash.replace(/^#/, ''));
  if (todayEl.value === '') p.delete('today'); else p.set('today', todayEl.value);
  const q = p.toString();
  try { history.replaceState(null, '', location.pathname + location.search + (q ? '#' + q : '')); } catch (e) {}
  draw();
});

const num = el => { const v = parseInt(el.value, 10); return Number.isFinite(v) && v > 0 ? v : null; };
const shortName = mk => CONFIG.MODELS[mk].name.replace(/\s*\(.*\)\s*$/, '');

function splitLabel(){
  const n = state.ngpu, tp = state.tp;
  if (n === 1) return '';
  if (tp === n) return ` TP${tp}`;
  if (tp === 1) return ` DP${n}`;
  return ` DP${n / tp}×TP${tp}`;
}

// the four ceilings, in concurrent sessions, as the page printed them
function readCeilings(){
  const labels = [...document.querySelectorAll('#ceilingBars text.dlabel')].slice(0, 4);
  if (labels.length < 4) return null;
  const c = {};
  KEYS.forEach((k, i) => {
    const m = labels[i].textContent.match(/^\s*([\d,]+(?:\.\d+)?)/);
    c[k] = m ? parseFloat(m[1].replace(/,/g, '')) : Infinity;
  });
  return c;
}

// latency ceiling at the two ends of the prefill-MFU bracket, scaled onto the
// page's own figure so the range always contains the point the page quotes
function latencyRange(pageValue){
  try {
    if (!lastCS || !isFinite(pageValue)) return null;
    const model = activeModel(), topo = currentTopo(), wl = currentWL();
    const reps = topo.replicas || 1, f = wl.invalidation;
    const lat = mfu => reps * maxUsersLatency(
      prefillServiceMoments(model, topo, wl, lastCS, prefillChunk(), mfu),
      f, state.sla, state.think, undefined, wl.sub_ratio);
    const mid = lat(undefined), lo = lat(PREFILL_MFU_LO), hi = lat(PREFILL_MFU_HI);
    if (!(mid > 0)) return null;
    const s = pageValue / mid;
    return { lo: Math.min(lo, hi) * s, hi: Math.max(lo, hi) * s };
  } catch (e) { return null; }
}

function verdictFor(h, dev, range, bind){
  const over = KEYS.filter(k => dev[k] < h).sort((a, b) => dev[a] - dev[b]);
  const inRange = range && h > range.lo && h <= range.hi;
  const rangeNote = inRange
    ? ` <span class="maybe">Latency is uncertain here</span>: the limit falls between ${fmt(range.lo, 0)} and ${fmt(range.hi, 0)} developers depending on prefill efficiency, so the load test decides.`
    : '';
  if (!over.length){
    return `<span class="ok">fits</span> · the first limit is <b>${PLANNER_LABEL[bind]}</b> at ${fmt(dev[bind], 0)} developers (${fmt(dev[bind] / h, 1)}× headroom).` + rangeNote;
  }
  return `<span class="over">over</span> on ${over.map(k => `<b>${PLANNER_LABEL[k]}</b> (limit ${fmt(dev[k], 0)})`).join(', ')}. This setup carries up to ${fmt(dev[bind], 0)} developers.` + rangeNote;
}

function draw(){
  const k = state.active * state.spu;
  const target = state.headcount !== null && state.headcount > 0 ? state.headcount : null;
  const today = num(todayEl);
  $('dDeploy').textContent = `${shortName(state.model)} on ${state.ngpu}×${state.gpu}${splitLabel()}`;
  $('dSlo').textContent = `${fmt(state.sla, 0)} s TTFT · ${fmt(state.decode_floor, 0)} tok/s`;
  $('dMap').textContent = `1 developer = ${fmt(state.active, 2)} peak active share × ${fmt(state.spu, 1)} sessions each = ${fmt(k, 2)} concurrent sessions at peak.`;

  const box = $('rulerSvg'), list = $('verdict');
  const ses = readCeilings();
  if (!ses){
    box.innerHTML = '<p class="cs">The model weights do not fit this deployment, so there is no headcount it can serve.</p>';
    list.innerHTML = '';
    return;
  }
  const dev = {}; for (const key of KEYS) dev[key] = ses[key] / k;
  let bind = KEYS[0]; for (const key of KEYS) if (dev[key] < dev[bind]) bind = key;
  const lr = latencyRange(ses.latency);
  const range = lr ? { lo: lr.lo / k, hi: lr.hi / k } : null;

  const W = Math.max(320, Math.round(box.clientWidth || 1000)), narrow = W < 600;
  const mL = narrow ? 78 : 118, mR = narrow ? 14 : 28, mT = 40, rowH = narrow ? 34 : 40;
  const H = mT + rowH * KEYS.length + 30;
  // 1.5x the largest of today, target and the first limit; stretched to show
  // the second limit too when it sits within 3x of that
  const base = Math.max(today || 0, target || 0, isFinite(dev[bind]) ? dev[bind] : 0) || 100;
  const second = KEYS.map(key => dev[key]).filter(isFinite).sort((a, b) => a - b)[1];
  const top = Math.max(1.5 * base, second ? Math.min(1.12 * second, 3 * base) : 0);
  const pw = W - mL - mR;
  const sx = v => mL + Math.min(v, top) / top * pw;
  const C = PLANNER_COLORS();
  const text = cssv('--text'), muted = cssv('--muted'), grid = cssv('--grid'), axis = cssv('--axis');
  const good = cssv('--good'), crit = cssv('--crit'), tile = cssv('--tile');
  const yEnd = mT + rowH * KEYS.length;
  let g = '';
  // fits / over shading
  const xb = isFinite(dev[bind]) ? sx(dev[bind]) : mL + pw;
  g += `<rect x="${mL}" y="${mT - 6}" width="${xb - mL}" height="${yEnd - mT + 6}" fill="${good}" fill-opacity=".09"/>`;
  if (xb < mL + pw) g += `<rect x="${xb}" y="${mT - 6}" width="${mL + pw - xb}" height="${yEnd - mT + 6}" fill="${crit}" fill-opacity=".08"/>`;
  for (const t of niceTicks(top, narrow ? 4 : 7)){
    if (t > top) continue;
    g += `<line x1="${sx(t)}" y1="${mT - 6}" x2="${sx(t)}" y2="${yEnd}" stroke="${grid}"/>`;
    g += `<text class="mono" x="${sx(t)}" y="${yEnd + 16}" font-size="11" fill="${muted}" text-anchor="middle">${fmt(t, 0)}</text>`;
  }
  g += `<line x1="${mL}" y1="${yEnd}" x2="${mL + pw}" y2="${yEnd}" stroke="${axis}"/>`;
  g += `<text class="mono" x="${mL - 10}" y="${yEnd + 16}" font-size="10" fill="${muted}" text-anchor="end">DEVELOPERS</text>`;
  KEYS.forEach((key, i) => {
    const y = mT + i * rowH + rowH / 2, v = dev[key], isB = key === bind, col = C[key];
    g += `<text x="${mL - 10}" y="${y + 4}" font-size="${narrow ? 12 : 13}" text-anchor="end" fill="${isB ? text : muted}" font-weight="${isB ? 700 : 400}">${esc(PLANNER_LABEL[key])}</text>`;
    if (key === 'latency' && range && range.lo < top){
      const x0 = sx(range.lo), x1 = sx(range.hi);
      g += `<rect x="${x0}" y="${y - 9}" width="${Math.max(2, x1 - x0)}" height="18" fill="${col}" fill-opacity=".22" stroke="${col}" stroke-opacity=".6"/>`;
    }
    const xe = isFinite(v) ? sx(v) : mL + pw;
    g += `<line x1="${mL}" y1="${y}" x2="${xe}" y2="${y}" stroke="${col}" stroke-width="${isB ? 4 : 2}" stroke-opacity="${isB ? 1 : .55}"/>`;
    const clipped = !isFinite(v) || v > top;
    if (!clipped) g += `<line x1="${xe}" y1="${y - 9}" x2="${xe}" y2="${y + 9}" stroke="${col}" stroke-width="${isB ? 3 : 2}"/>`;
    const lbl = !isFinite(v) ? 'none' : (clipped ? `${fmt(v, 0)} →` : fmt(v, 0));
    const right = clipped || xe > mL + pw - 60;
    g += `<text class="mono" x="${right ? (clipped ? mL + pw : xe - 6) : xe + 6}" y="${y - 6}" font-size="11" text-anchor="${right ? 'end' : 'start'}" fill="${isB ? text : muted}" font-weight="${isB ? 700 : 400}">${lbl}${isB ? ' · binds' : ''}</text>`;
  });
  // today / target markers
  const marks = [];
  if (today) marks.push({ v: today, name: 'today', dash: '4 3' });
  if (target) marks.push({ v: target, name: 'target', dash: '' });
  marks.sort((a, b) => a.v - b.v);
  marks.forEach((m, i) => {
    const x = sx(m.v), fits = m.v <= dev[bind], col = fits ? text : crit;
    g += `<line x1="${x}" y1="${mT - 14}" x2="${x}" y2="${yEnd}" stroke="${col}" stroke-width="2" ${m.dash ? `stroke-dasharray="${m.dash}"` : ''}/>`;
    const anchor = x > mL + pw - 90 ? 'end' : (i === 0 && marks.length > 1 && sx(marks[1].v) - x < 110 ? 'end' : 'start');
    g += `<text class="mono" x="${anchor === 'end' ? x - 5 : x + 5}" y="${mT - 18}" font-size="11" font-weight="700" fill="${col}" text-anchor="${anchor}">${m.name.toUpperCase()} ${fmt(m.v, 0)}</text>`;
  });
  box.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="Headcount at which each limit is reached, with today and target marked">${g}</svg>`;
  void tile;

  const rows = [];
  for (const m of [today && { h: today, n: 'today' }, target && { h: target, n: 'target' }].filter(Boolean))
    rows.push(`<li><span class="h">${m.n} ${fmt(m.h, 0)}</span><span>${verdictFor(m.h, dev, range, bind)}</span></li>`);
  if (!target) rows.push(`<li><span class="h">target</span><span>Enter a target headcount in the question above to get a verdict. Without one this setup carries up to <b>${fmt(dev[bind], 0)}</b> developers, limited by <b>${PLANNER_LABEL[bind]}</b>.</span></li>`);
  list.innerHTML = rows.join('');
}

new MutationObserver(() => draw()).observe($('ceilingBars'), { childList: true, subtree: true });
let rt = 0;
window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(draw, 120); });
draw();
