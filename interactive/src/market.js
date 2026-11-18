import { MARKET } from './market-data.js';

/* Rent against token prices: one shared log axis, one row per model.
   Band: cost per million tokens of the workload mix for the model's cheapest viable
   configuration, from the lowest to the highest seller rent, tick at the median,
   at the utilisation the slider sets. Dots: each endpoint's listed price for the
   same mix. All costs are recomputed here from n_gpu, limit and the rents. */

const M = MARKET;
const LEVELS = ['low', 'median', 'high'];
const KIND_NAME = { code: 'code', english: 'English', french: 'French', json_tool_calls: 'tool-call JSON' };
const BIND = {
  decode: 'decode speed per user',
  cold_wait: 'wait for a cold (uncached) prompt',
  slowed: 'generation slowed by other sessions’ prefill',
};
const TERM_NAME = { 'on-demand': 'on-demand', 'committed-short': 'committed under 3 months', 'committed-long': 'committed 12 months or more' };
const termName = (t) => TERM_NAME[t] ?? t;
const TERM_SHORT = { 'on-demand': 'on-demand', 'committed-short': 'committed < 3 mo', 'committed-long': 'committed ≥ 12 mo' };
const S = { fresh: false, u: 0.5, term: 'on-demand', unit: 'tok', kind: 'code', pinned: null, sort: ['u1', 1] };
const TPH = M.workload.mixed_tokens_per_session_hour;
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

// ---------- the model ----------
const rent = (gpu, term, level) => M.rent[gpu][term][level];
const HAS_NC = M.models.every((m) => m.nc);
const cfgs = (m, fresh = S.fresh) => (fresh ? m.nc.configs : m.configs);
const price = (p, fresh = S.fresh) => (fresh ? p.nc.mix : p.mix);
const pSH = (p) => (S.fresh ? p.nc.sh : p.sh);
const pU = (p) => (S.fresh ? p.nc.u[S.term] : p.u[S.term]);
function best(m, term, fresh = S.fresh) {
  let b = null, bc = Infinity;
  for (const c of cfgs(m, fresh)) {
    const v = c.n * rent(c.gpu, term, 'median') / c.limit;
    if (v < bc) { bc = v; b = c; }
  }
  return b;
}
const costTok = (c, term, level, u) => c.n * rent(c.gpu, term, level) / (c.limit * u) / TPH * 1e6;
const div = (m) => (S.unit === 'char' ? m.cpt[S.kind] : 1);
const shortName = (m) => m.name.split(' (')[0];

// ---------- formatting ----------
function usd(x) {
  if (x == null) return 'not listed';
  if (x >= 100) return '$' + x.toFixed(0);
  if (x >= 1) return '$' + x.toFixed(2);
  if (x >= 0.01) return '$' + x.toFixed(3);
  return '$' + x.toPrecision(2);
}
function pct(x) {
  if (x == null) return 'n/a';
  if (x > 1) return 'above 100%';
  return (x < 0.1 ? (x * 100).toFixed(1) : (x * 100).toFixed(0)) + '%';
}
const tickLabel = (v) => '$' + (v >= 1 ? v.toFixed(0) : String(+v.toPrecision(1)));

// ---------- axis domain, fixed per unit so the slider moves bands, not the axis ----------
function domainFor(unit, fresh) {
  let lo = Infinity, hi = -Infinity;
  const kinds = unit === 'char' ? Object.keys(KIND_NAME) : [null];
  for (const m of M.models) for (const k of kinds) {
    const d = k ? m.cpt[k] : 1;
    for (const term of M.terms) {
      const c = best(m, term, fresh);
      if (c) { lo = Math.min(lo, costTok(c, term, 'low', 1) / d); hi = Math.max(hi, costTok(c, term, 'high', 0.05) / d); }
    }
    for (const p of m.prov) if (price(p, fresh) != null) { lo = Math.min(lo, price(p, fresh) / d); hi = Math.max(hi, price(p, fresh) / d); }
  }
  return [lo / 1.25, hi * 1.25];
}
const DOMAIN = { tok: domainFor('tok', false), char: domainFor('char', false) };
const DOMAIN_NC = HAS_NC ? { tok: domainFor('tok', true), char: domainFor('char', true) } : null;
const dom = () => (S.fresh ? DOMAIN_NC : DOMAIN)[S.unit];
function ticks([lo, hi]) {
  const out = [];
  for (let e = Math.floor(Math.log10(lo)) - 1; e <= Math.ceil(Math.log10(hi)); e++)
    for (const f of [1, 3]) { const v = f * 10 ** e; if (v >= lo && v <= hi) out.push(+v.toPrecision(1)); }
  return out;
}

// ---------- the count sentence ----------
function covered(u) {
  let k = 0, n = 0;
  const who = new Set();
  for (const m of M.models) {
    const c = best(m, S.term);
    for (const p of m.prov) {
      if (price(p) == null || !c) continue;
      n++;
      who.add(p.p);
      if (price(p) >= costTok(c, S.term, 'median', u)) k++;
    }
  }
  return [k, n, who.size];
}
function countSentence() {
  const [k, n, np] = covered(S.u), [k1] = covered(1);
  const up = Math.round(S.u * 100);
  let s = `${S.fresh ? 'With every prompt computed fresh (rough sensitivity): a' : 'A'}t <b>${up}%</b> utilisation and median ${termName(S.term)} rent, <b>${k}</b> of ${n} listed prices (endpoint and tier rows from ${np} providers) cover the GPU rent the model predicts.`;
  if (up < 100) s += ` At 100%, <b>${k1}</b> do.`;
  $('count').innerHTML = s;
}

// ---------- the chart ----------
let rows = [], marks = [], bands = [];
function render() {
  const wrap = $('chartwrap');
  const W = Math.max(280, wrap.clientWidth - 28);
  const narrow = W < 600, small = W < 520;
  const R = small ? 3 : 4.5, RING = R + 2, LANE = small ? 5 : 7, GAP = 2 * R + 1;
  const labelW = narrow ? 0 : 220;
  const rowH = narrow ? 78 : 50, cy = narrow ? 54 : 25, top = 22;
  const [d0, d1] = dom();
  const x0 = labelW + 8, x1 = W - 12;
  const X = (v) => x0 + (Math.log(v) - Math.log(d0)) / (Math.log(d1) - Math.log(d0)) * (x1 - x0);

  rows = M.models.map((m) => {
    const c = best(m, S.term);
    return { m, c, key: c ? costTok(c, S.term, 'median', 1) / div(m) : Infinity };
  }).sort((a, b) => a.key - b.key);

  const H = top + rows.length * rowH + 6;
  const svg = $('chart');
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.setAttribute('height', H);
  const unitTxt = (S.fresh ? 'Every prompt fresh. ' : '') + (S.unit === 'tok' ? 'USD per million tokens of this mix (cached input included)' : `USD per million characters of ${KIND_NAME[S.kind]} text, same mix (cached input included)`);
  $('axisTitle').textContent = unitTxt + ', log scale';
  let g = '';
  for (const t of ticks(dom())) {
    const x = X(t);
    g += `<line x1="${x}" x2="${x}" y1="${top - 6}" y2="${H - 6}" stroke="var(--grid)" stroke-width="1"/>`;
    g += `<text class="tick" x="${x}" y="${top - 11}" text-anchor="middle">${tickLabel(t)}</text>`;
  }
  marks = []; bands = [];
  rows.forEach((r, i) => {
    const { m, c } = r, d = div(m), y0 = top + i * rowH, y = y0 + cy;
    g += `<g class="row${S.pinned === m.key ? ' pinned' : ''}" data-row="${esc(m.key)}">`;
    g += `<rect class="rowbg" x="0" y="${y0}" width="${W}" height="${rowH}" fill="transparent"/>`;
    g += `<line x1="0" x2="${W}" y1="${y0 + rowH}" y2="${y0 + rowH}" stroke="var(--grid)"/>`;
    const provNames = [...new Set(m.prov.map((p) => p.p))];
    let sub = c ? `${c.n}×${c.gpu} TP${c.tp} ${c.arm}` : 'no viable configuration';
    if (provNames.length === 1) sub += ' · 1 provider';
    else if (!provNames.length) sub += ' · no listed price';
    if (narrow) {
      g += `<text class="rowname" x="2" y="${y0 + 17}">${esc(shortName(m))}</text>`;
      g += `<text class="rowsub" x="2" y="${y0 + 31}">${esc(sub)}</text>`;
    } else {
      g += `<text class="rowname" x="0" y="${y - 2}">${esc(shortName(m))}</text>`;
      g += `<text class="rowsub" x="0" y="${y + 12}">${esc(sub)}</text>`;
    }
    if (c) {
      const gl = X(costTok(c, S.term, 'low', 1) / d), gh = X(costTok(c, S.term, 'high', 1) / d);
      g += `<rect x="${gl}" y="${y - 7}" width="${Math.max(2, gh - gl)}" height="14" fill="var(--ghost)" stroke="var(--band)" stroke-opacity=".45" stroke-dasharray="3 3"/>`;
      const bl = X(costTok(c, S.term, 'low', S.u) / d), bh = X(costTok(c, S.term, 'high', S.u) / d), bm = X(costTok(c, S.term, 'median', S.u) / d);
      g += `<rect x="${bl}" y="${y - 7}" width="${Math.max(2, bh - bl)}" height="14" fill="var(--band)" fill-opacity=".32"/>`;
      g += `<line x1="${bm}" x2="${bm}" y1="${y - 13}" y2="${y + 13}" stroke="var(--band)" stroke-width="3"/>`;
      bands.push({ r });
      g += `<rect class="hit" data-band="${bands.length - 1}" tabindex="0" role="button" aria-label="${esc(shortName(m))}: modelled cost band; press Enter to list its prices" x="${bl - 6}" y="${y - 15}" width="${Math.max(14, bh - bl + 12)}" height="30"/>`;
    }
    // beeswarm: deterministic lanes
    const med = c ? costTok(c, S.term, 'median', S.u) : null;
    const pts = m.prov.filter((p) => price(p) != null).map((p) => ({ p, x: X(price(p) / d) }))
      .sort((a, b) => a.x - b.x || a.p.p.localeCompare(b.p.p));
    const lanes = [0, -1, 1, -2, 2, -3, 3, -4, 4].map((k) => k * LANE).filter((o) => Math.abs(o) <= Math.min(cy - 4, narrow ? 3 * LANE : 3 * LANE));
    const placed = lanes.map(() => []);
    for (const pt of pts) {
      let li = lanes.findIndex((_, j) => placed[j].every((px) => Math.abs(px - pt.x) >= GAP));
      if (li < 0) {
        let bestGap = -1;
        lanes.forEach((_, j) => { const gap = Math.min(...placed[j].map((px) => Math.abs(px - pt.x))); if (gap > bestGap) { bestGap = gap; li = j; } });
      }
      placed[li].push(pt.x);
      pt.y = y + lanes[li];
    }
    for (const pt of pts) {
      const covers = med != null && price(pt.p) >= med;
      marks.push({ p: pt.p, m, c });
      const mi = marks.length - 1;
      g += `<circle cx="${pt.x}" cy="${pt.y}" r="${RING}" fill="var(--surface)"/>`;
      g += covers
        ? `<circle class="dot" cx="${pt.x}" cy="${pt.y}" r="${R}" fill="var(--text)"/>`
        : `<circle class="dot" cx="${pt.x}" cy="${pt.y}" r="${R - 0.5}" fill="var(--surface)" stroke="var(--warm)" stroke-width="${small ? 1.5 : 2}"/>`;
      g += `<circle class="hit" data-mark="${mi}" tabindex="0" role="button" aria-label="${esc(pt.p.p)}, ${esc(usd(price(pt.p) / d))}" cx="${pt.x}" cy="${pt.y}" r="10"/>`;
    }
    g += '</g>';
  });
  svg.innerHTML = g;
  countSentence();
  if (S.pinned) renderPin();
}

// ---------- legend ----------
function footnote() {
  const parts = [];
  const r0 = M.rent[Object.keys(M.rent)[0]][M.terms[0]];
  if (M.rent_tiers) {
    const outTiers = [...new Set(Object.values(M.rent).flatMap((g) => Object.values(g).flatMap((t) => (t.out_tiers || []).map((x) => (x.match(/\(([^)]+)\)$/) || [])[1]).filter(Boolean))))];
    const gb = Object.values(M.rent).some((g) => Object.values(g).some((t) => (t.gb300 || []).length));
    let s = `Rent band: ${M.rent_tiers} sellers only`;
    const outs = [];
    if (outTiers.length) outs.push(`${outTiers.join(' and ')} tiers out`);
    if (gb) outs.push('GB300 out');
    if (outs.length) s += ` (${outs.join(', ')})`;
    parts.push(s + '.');
  }
  void r0;
  parts.push('Rows are endpoint and tier rows, so one provider can appear several times.');
  $('foot').textContent = parts.join(' ');
}
function legend() {
  $('legend').innerHTML = `
    <span><svg width="34" height="16" aria-hidden="true"><rect x="1" y="2" width="32" height="12" fill="var(--band)" fill-opacity=".32"/><line x1="17" x2="17" y1="0" y2="16" stroke="var(--band)" stroke-width="3"/></svg>GPU rent at this utilisation: lowest to highest seller, tick at the median</span>
    <span><svg width="34" height="16" aria-hidden="true"><rect x="1" y="2" width="32" height="12" fill="var(--ghost)" stroke="var(--band)" stroke-opacity=".45" stroke-dasharray="3 3"/></svg>the same at 100%</span>
    <span><svg width="14" height="14" aria-hidden="true"><circle cx="7" cy="7" r="4.5" fill="var(--text)"/></svg>listed price at or above the median tick</span>
    <span><svg width="14" height="14" aria-hidden="true"><circle cx="7" cy="7" r="4" fill="var(--surface)" stroke="var(--warm)" stroke-width="2"/></svg>listed price below it</span>`;
}

// ---------- tooltips ----------
const tt = $('tt');
let tipLocked = false;
function sourceLine(p) {
  return p.src === 'openrouter'
    ? `OpenRouter API, read ${esc(p.read)}`
    : `${esc(p.p)}’s price page, read ${esc(p.read)}`;
}
function markTip({ p, m }) {
  const us = pU(p);
  const d = div(m);
  const unitWord = S.unit === 'tok' ? 'M tokens' : `M chars of ${KIND_NAME[S.kind]}`;
  let h = `<h4>${esc(p.p)}</h4><div class="sub">${esc(p.tag)}</div><dl>`;
  h += `<dt>Input</dt><dd>${usd(p.in)} per M tokens</dd><dt>Output</dt><dd>${usd(p.out)}</dd><dt>Cached input</dt><dd>${usd(p.cr)}</dd>`;
  if (p.cw != null) h += `<dt>Cache write</dt><dd>${usd(p.cw)}</dd>`;
  h += `<dt>${S.fresh ? 'All fresh' : 'This mix'}</dt><dd>${usd(price(p) / d)} per ${esc(unitWord)} · ${usd(pSH(p))} per session-hour</dd>`;
  h += `<dt>Quantization</dt><dd>${esc(p.q ?? 'not declared')}</dd><dt>Priced as</dt><dd>${esc(p.arm)}</dd>`;
  if (p.ctx) h += `<dt>Context</dt><dd>${p.ctx.toLocaleString('en-US')} tokens, ${pct(p.ctxr)} of the model’s own</dd>`;
  if (us) h += `<dt>Break-even u</dt><dd>${us.map(pct).join(' / ')}<br><span class="muted">at low / median / high ${esc(termName(S.term))} rent</span></dd>`;
  h += `<dt>Source</dt><dd>${sourceLine(p)}</dd></dl>`;
  if (p.lw && !S.fresh) {
    const lu = p.lw.u[S.term];
    h += `<p>Time-of-day prices: plotted at the 24-hour time-weighted price. The cheapest listed window is ${usd(p.lw.in)} in, ${usd(p.lw.out)} out, ${usd(p.lw.cr)} cached${lu ? `; break-even u there ${lu.map(pct).join(' / ')}` : ''}.</p>`;
  }
  if (p.src === 'openrouter' && m.note) h += `<p>${esc(m.note)}</p>`;
  if (us && us.some((x) => x != null && x > 1))
    h += `<p>Above 100%: does not cover the rent at the predicted limit even fully loaded, under these assumptions.</p>`;
  if (!p.fits) h += `<p>Context below the workload’s ${M.workload.cap.toLocaleString('en-US')}-token cap: ${pct(p.over)} of requests are longer.</p>`;
  for (const f of p.flags) h += `<p>${esc(f)}</p>`;
  return h;
}
function errText(e) {
  if (e.s === 'unchecked') return 'capacity unchecked: no independent measurement of this model';
  if (e.bs !== 'checked') return `capacity unchecked: what binds (${esc(e.bq.toLowerCase())}) has no independent measurement for this model`;
  if (e.lo == null || e.hi == null) return `checked against measurements of ${esc(e.bq.toLowerCase())}`;
  if (e.lo === e.hi) return `checked against measurements of ${esc(e.bq.toLowerCase())}: the measured errors leave the limit at ${e.lo} sessions`;
  return `checked against measurements of ${esc(e.bq.toLowerCase())}: re-predicted under each measured error, the limit is ${e.lo} to ${e.hi} sessions`;
}
function bandTip({ r }) {
  const { m, c } = r, d = div(m);
  const rr = M.rent[c.gpu][S.term];
  const unitWord = S.unit === 'tok' ? 'M tokens' : `M chars`;
  let h = `<h4>${esc(shortName(m))}</h4><div class="sub">${c.n}×${esc(c.gpu)}, TP${c.tp}, ${esc(c.arm)} weights: the cheapest viable configuration</div><dl>`;
  h += `<dt>Limit</dt><dd>${c.limit.toLocaleString('en-US')} live sessions</dd><dt>Binds</dt><dd>${esc(BIND[c.bind] ?? c.bind)}</dd>`;
  h += `<dt>${esc(c.gpu)} rent</dt><dd>${LEVELS.map((l) => usd(rr[l])).join(' / ')} per GPU-hour<br><span class="muted">low / median / high of ${rr.n} ${esc(termName(S.term))} sellers</span></dd>`;
  h += `<dt>At ${Math.round(S.u * 100)}%</dt><dd>${LEVELS.map((l) => usd(costTok(c, S.term, l, S.u) / d)).join(' / ')} per ${unitWord}</dd>`;
  h += `<dt>At 100%</dt><dd>${LEVELS.map((l) => usd(costTok(c, S.term, l, 1) / d)).join(' / ')}</dd>`;
  h += `<dt>Model error</dt><dd>${c.err ? errText(c.err) : 'capacity unchecked: rough sensitivity'}</dd></dl>`;
  if (m.sens && !S.fresh) h += `<p>Most sensitive to: ${esc(m.sens.words)}.</p>`;
  if (S.fresh && m.nc.note) h += `<p>${esc(m.nc.note)}.</p>`;
  if (m.note && m.prov.some((p) => p.src === 'openrouter')) h += `<p>${esc(m.note)}</p>`;
  return h;
}
function showTip(el) {
  const wrap = $('chartwrap');
  let html;
  if (el.dataset.mark != null) html = markTip(marks[+el.dataset.mark]);
  else if (el.dataset.band != null) html = bandTip(bands[+el.dataset.band]);
  else return;
  tt.innerHTML = html;
  tt.hidden = false;
  const wr = wrap.getBoundingClientRect(), er = el.getBoundingClientRect();
  const tw = tt.offsetWidth, th = tt.offsetHeight;
  let left = er.left + er.width / 2 - wr.left - tw / 2;
  left = Math.max(4, Math.min(left, wr.width - tw - 4));
  let topY = er.top - wr.top - th - 8;
  if (topY < 0) topY = er.bottom - wr.top + 8;
  tt.style.left = left + 'px';
  tt.style.top = topY + 'px';
}
function hideTip() { tt.hidden = true; tipLocked = false; }

const svg = $('chart');
svg.addEventListener('pointerover', (e) => { if (!tipLocked && e.pointerType === 'mouse' && e.target.classList.contains('hit')) showTip(e.target); });
svg.addEventListener('pointerout', (e) => { if (!tipLocked && e.pointerType === 'mouse' && e.target.classList.contains('hit')) hideTip(); });
svg.addEventListener('focusin', (e) => { if (e.target.classList.contains('hit')) showTip(e.target); });
svg.addEventListener('focusout', () => { if (!tipLocked) tt.hidden = true; });
svg.addEventListener('click', (e) => {
  const t = e.target;
  if (t.dataset && t.dataset.mark != null) { showTip(t); tipLocked = true; e.stopPropagation(); return; }
  const row = t.closest('.row');
  if (row) {
    if (t.dataset.band != null) { showTip(t); tipLocked = true; }
    else hideTip();
    pin(row.dataset.row);
  }
});
svg.addEventListener('keydown', (e) => {
  const t = e.target;
  if ((e.key === 'Enter' || e.key === ' ') && t.dataset.band != null) { e.preventDefault(); pin(t.closest('.row').dataset.row); }
  if (e.key === 'Escape') hideTip();
});
document.addEventListener('click', (e) => { if (!svg.contains(e.target)) hideTip(); });

// ---------- pinned table ----------
function pin(key) {
  S.pinned = S.pinned === key ? null : key;
  for (const g of svg.querySelectorAll('.row')) g.classList.toggle('pinned', g.dataset.row === S.pinned);
  renderPin();
}
const COLS = [
  ['p', 'Provider', (p) => p.p], ['src', 'Source', (p) => p.src], ['in', 'In', (p) => p.in], ['out', 'Out', (p) => p.out],
  ['cr', 'Cached', (p) => p.cr], ['sh', '$/session-h', (p) => pSH(p)],
  ['u0', 'u* low', (p) => pU(p)?.[0]], ['u1', 'u* median', (p) => pU(p)?.[1]], ['u2', 'u* high', (p) => pU(p)?.[2]],
  ['q', 'Quant', (p) => p.q], ['ctxr', 'Context', (p) => p.ctxr], ['mo', 'Max out', (p) => p.mo],
];
function renderPin() {
  const box = $('pin');
  const m = M.models.find((x) => x.key === S.pinned);
  if (!m) { box.hidden = true; box.innerHTML = ''; return; }
  const [sk, dir] = S.sort;
  const col = COLS.find((c) => c[0] === sk);
  const rowsP = [...m.prov].sort((a, b) => {
    const va = col[2](a), vb = col[2](b);
    if (va == null && vb == null) return 0;
    if (va == null) return 1;
    if (vb == null) return -1;
    return (typeof va === 'string' ? va.localeCompare(vb) : va - vb) * dir;
  });
  let h = `<h2>${esc(shortName(m))}: ${m.prov.length} listed price${m.prov.length === 1 ? '' : 's'}</h2>`;
  if (m.note && m.prov.some((p) => p.src === 'openrouter')) h += `<div class="muted">${esc(m.note)}</div>`;
  h += `<div class="muted">USD per million tokens. u* is the utilisation at which the price covers the ${esc(termName(S.term))} rent at low / median / high seller prices.</div>`;
  if (!m.prov.length) { box.innerHTML = h + '<p class="muted">No listed price in this data set.</p>'; box.hidden = false; return; }
  h += '<div class="tablewrap"><table><thead><tr>';
  for (const [k, label] of COLS) h += `<th${k === 'p' || k === 'src' || k === 'q' ? ' class="l"' : ''}><button type="button" id="sort-${k}" data-sort="${k}" data-active="${k === sk}">${esc(label)}${k === sk ? (dir > 0 ? ' ↑' : ' ↓') : ''}</button></th>`;
  h += '<th class="l">Flags</th></tr></thead><tbody>';
  for (const p of rowsP) {
    const u = pU(p) || [null, null, null];
    h += `<tr><td>${esc(p.p)}<div class="muted">${esc(p.tag)}</div></td><td class="l">${p.src === 'openrouter' ? 'OpenRouter' : `<a href="${esc(p.url)}" target="_blank" rel="noopener">own page</a>`}</td>`;
    h += `<td>${usd(p.in)}</td><td>${usd(p.out)}</td><td>${usd(p.cr)}</td><td>${usd(pSH(p))}</td>`;
    h += u.map((x) => `<td>${pct(x)}</td>`).join('');
    h += `<td class="l">${esc(p.q ?? 'not declared')}</td><td>${p.ctxr != null ? pct(p.ctxr) : 'n/a'}</td><td>${p.mo != null ? Math.round(p.mo / 1000) + 'k' : '—'}</td>`;
    h += `<td class="wrap">${esc([...p.flags, ...(p.fits ? [] : [`${pct(p.over)} of requests exceed its context`])].join('; '))}</td></tr>`;
  }
  h += '</tbody></table></div>';
  box.innerHTML = h;
  box.hidden = false;
}
$('pin').addEventListener('click', (e) => {
  const b = e.target.closest('button[data-sort]');
  if (!b) return;
  const k = b.dataset.sort;
  S.sort = [k, S.sort[0] === k ? -S.sort[1] : 1];
  renderPin();
  document.getElementById('sort-' + k)?.focus();
});

// ---------- sources ----------
function sources() {
  const w = M.workload, mix = M.mix;
  const n = (x, d = 0) => Number(x).toLocaleString('en-US', { maximumFractionDigits: d });
  let h = `<h3>Workload</h3><p>The explorer’s default agentic coding session: prompts of ${n(w.prompt_mean)} tokens on average, capped at ${n(w.cap)}; ${n(w.output)} output tokens per request; a ${n(w.think_time_s)} s think time and ${n(w.sub_ratio * 100)}% extra sub-agent requests, so ${n(w.requests_per_session_hour)} requests per live session-hour and ${n(w.mixed_tokens_per_session_hour / 1e6, 2)} M tokens. With ${n(w.miss_rate * 100)}% prefix-cache misses the mix is ${n(mix.fresh * 100, 1)}% fresh input, ${n(mix.cached * 100, 1)}% cached input and ${n(mix.output * 100, 1)}% output.</p>`;
  h += `<h3>Derivation</h3><ol>
    <li>For each GPU (H200, B300), each tensor-parallel width vLLM accepts on one 8-GPU node and each weight format, the model predicts how many live sessions fit (the limit L) and which ceiling binds.</li>
    <li>Rent C = GPUs × USD per GPU-hour, paid every hour, at the lowest, median and highest seller.</li>
    <li>Utilisation u = live sessions ÷ L, averaged over every rented hour.</li>
    <li>Cost per million tokens = C ÷ (L × u) ÷ tokens per session-hour × 10⁶, on the configuration with the lowest C ÷ L.</li>
    <li>An endpoint’s price for the mix applies its listed input, cached-input and output prices to the same session, assuming cache entries live 300 s.</li>
    <li>Break-even u* = cost per session-hour at full load ÷ the endpoint’s price per session-hour. Above 100%, the price does not cover the rent at the predicted limit even fully loaded, under these assumptions.</li></ol>
    <p>Per million characters divides both by the model’s characters per token, measured with its own tokenizer on fixed public text (code, English, French, tool-call JSON). The provider’s hardware, engine, contract rents and traffic are unknown.</p>`;
  if (HAS_NC) h += `<h3>Every prompt fresh</h3><p>Rough sensitivity: limits re-predicted with an explicit max_num_seqs, every prompt computed fresh and billed at the input price. It moves both the bands and the dots.</p>`;
  h += `<h3>GPU rent, USD per GPU-hour</h3>`;
  if (M.rent_tiers) h += `<p>The band counts ${esc(M.rent_tiers)} offers only${M.rent_other ? `; rows marked “not counted” are ${esc(M.rent_other.tiers)}-tier offers left out of it` : ''}. Aggregators and GB300 instances are left out.</p>`;
  for (const gpu of Object.keys(M.rent)) for (const term of M.terms) {
    const r = M.rent[gpu][term];
    const seen = new Set(r.sellers.map((x) => x[0] + '|' + x[1]));
    const extra = M.rent_other ? (M.rent_other.rent[gpu]?.[term]?.sellers ?? []).filter((x) => !seen.has(x[0] + '|' + x[1])) : [];
    h += `<div class="tablewrap"><table><thead><tr><th class="l">${esc(gpu)} ${esc(termName(term))}</th><th>USD/GPU-h</th><th class="l">Tier</th><th>Commitment</th><th class="l">Read</th><th class="l">Note</th></tr></thead><tbody>`;
    const line = (x, counted) => {
      const [s, v, cm, read, url, tier, avail] = x;
      return `<tr><td><a href="${esc(url)}" target="_blank" rel="noopener">${esc(s)}</a></td><td>${usd(v)}</td><td class="l">${esc(tier ?? 'n/a')}</td><td>${cm ? n(cm, 1) + ' months' : 'none'}</td><td class="l">${esc(read)}</td><td class="l">${counted ? '' : 'not counted'}${avail ? (counted ? '' : '; ') + esc(avail) : ''}</td></tr>`;
    };
    for (const x of r.sellers) h += line(x, true);
    for (const x of extra) h += line(x, false);
    h += `</tbody></table></div><p class="muted">Band: low ${usd(r.low)}, median ${usd(r.median)}, high ${usd(r.high)} over ${r.n} sellers; ${r.agg} aggregator${r.agg === 1 ? '' : 's'} left out${r.gb300?.length ? `; GB300 left out: ${esc(r.gb300.join(', '))}` : ''}.</p>`;
  }
  if (M.fx) h += `<p class="muted">Sellers listing in EUR are converted at ${M.fx.usd_per_eur} USD per EUR (ECB, ${esc(M.fx.date)}).</p>`;
  h += `<h3>Token prices</h3><ul>`;
  for (const [p, url, read] of M.price_pages) h += `<li><a href="${esc(url)}" target="_blank" rel="noopener">${esc(p)}</a>, read ${esc(read)}</li>`;
  if (M.openrouter) h += `<li><a href="${esc(M.openrouter.url)}" target="_blank" rel="noopener">OpenRouter API</a> endpoint listing, read ${esc(M.openrouter.read)}</li>`;
  h += `</ul><p>Published discounts and time-of-day prices are flagged and not applied.</p>`;
  h += `<h3>How far the model has been checked</h3><p>Against independent measurements (docs/accuracy.md), the model predicts decode speed per user 10.5% too fast on average and 23.4% too fast at worst, over 3 rows. On the one prefill row it predicts a cold prompt 14.4% too fast. Most models have no independent row, and their bands are marked unchecked.</p>`;
  $('srcbody').innerHTML = h;
}

// ---------- controls ----------
$('term').innerHTML = M.terms.map((t) => `<button type="button" id="term-${t}" data-v="${t}" aria-pressed="${t === S.term}" title="${esc(termName(t))}">${esc(TERM_SHORT[t] ?? t)}</button>`).join('');
function setSeg(id, v) { for (const b of $(id).querySelectorAll('button')) b.setAttribute('aria-pressed', String(b.dataset.v === v)); }
$('u').addEventListener('input', (e) => { S.u = +e.target.value / 100; $('uOut').textContent = e.target.value + '%'; hideTip(); render(); });
$('term').addEventListener('click', (e) => { const b = e.target.closest('button'); if (!b) return; S.term = b.dataset.v; setSeg('term', S.term); hideTip(); render(); });
$('unit').addEventListener('click', (e) => { const b = e.target.closest('button'); if (!b) return; S.unit = b.dataset.v; setSeg('unit', S.unit); $('kindWrap').hidden = S.unit !== 'char'; hideTip(); render(); });
if (HAS_NC) {
  $('cacheWrap').hidden = false;
  $('cache').addEventListener('click', (e) => { const b = e.target.closest('button'); if (!b) return; S.fresh = b.dataset.v === 'fresh'; setSeg('cache', b.dataset.v); hideTip(); render(); });
}
$('kind').addEventListener('change', (e) => { S.kind = e.target.value; hideTip(); render(); });

const mix = M.mix;
const bill = M.billing;
const p1 = (x) => (x * 100).toFixed(1) + '%';
$('lede').textContent = `An agentic coding session re-reads its history on every request: at the model’s ${p1(M.workload.miss_rate)} cache-miss rate, ${p1(mix.cached)} of all prompt-plus-output tokens are cache reads, ${p1(mix.fresh)} fresh input and ${p1(mix.output)} output. `
  + `Listed prices are applied with an assumed ${bill.ttl_s} s cache lifetime, which gives ${p1(bill.miss)} misses and ${p1(bill.cached)} cached tokens.`;
S.u = +$('u').value / 100;
$('uOut').textContent = $('u').value + '%';
legend();
footnote();
sources();
render();
let lastW = $('chartwrap').clientWidth;
new ResizeObserver(() => { const w = $('chartwrap').clientWidth; if (w !== lastW) { lastW = w; render(); } }).observe($('chartwrap'));
