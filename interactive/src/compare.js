/* The Compare page: every self-hosted configuration, scored and priced per
   seat, on one chart and one table. Rows come from compare-worker.js, which
   runs the explorer's own frontier math (frontier-rows.js) per GPU. */

const USERS = [8, 16, 32, 64, 128, 256, 512, 1024, 2048];
// short names for narrow screens: the explorer's model buttons (frontier.js FRONTIER_SHORT)
const SHORT = { '27B': 'Qwen3.8-27B', '35BA3B': 'Qwen3.6-35B', 'MM35': 'Mistral-Med-3.5', 'GLM52': 'GLM-5.3', 'DSV4F': 'DSv4-Flash',
  'DSV4FV': 'DSv4-Flash-Vis', 'DSV41F': 'DSv4.1-Flash', 'Q38FN': 'Q3.8-Flash', 'GLM53F': 'GLM-5.3-Flash', 'ML4': 'Mistral-Large-4' };
const BIND = { cache: 'cache', decode: 'decode', cold_wait: 'cold wait', slowed: 'slowed generation', saturation: 'saturation' };
const $ = id => document.getElementById(id);
const fmtInt = v => Math.round(v).toLocaleString('en-US');
const fmtEur = v => v == null ? '—' : '€' + (v < 10 ? v.toFixed(1) : fmtInt(v));
const fmtPct = v => v == null ? '—' : (v * 100).toFixed(1) + '%';
const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const statusEl = document.getElementById('status');   // re-attached after each chart redraw

/* ---- page state, mirrored in the URL hash ---- */
const S = { users: 64, bench: 'tb40', gpus: ['H200', 'B300'], x: 'seat', scale: 'log', pareto: true,
            hidden: new Set(), hl: null, pin: null, sort: 'seat', dir: 1 };
function readHash(){
  const h = new URLSearchParams(location.hash.slice(1));
  if (USERS.includes(+h.get('users'))) S.users = +h.get('users');
  if (['tb40', 'tb21'].includes(h.get('bench'))) S.bench = h.get('bench');
  if (['seat', 'users'].includes(h.get('x'))) S.x = h.get('x');
  if (['log', 'lin'].includes(h.get('scale'))) S.scale = h.get('scale');
  if (h.has('gpus')) S.gpus = h.get('gpus').split(',').filter(g => ['H200', 'B300'].includes(g));
  if (h.has('hide')) S.hidden = new Set(h.get('hide').split(',').filter(Boolean));
}
function writeHash(){
  const h = new URLSearchParams();
  if (S.users !== 64) h.set('users', S.users);
  if (S.bench !== 'tb40') h.set('bench', S.bench);
  if (S.x !== 'seat') h.set('x', S.x);
  if (S.scale !== 'log') h.set('scale', S.scale);
  if (S.gpus.join() !== 'H200,B300') h.set('gpus', S.gpus.join());
  if (S.hidden.size) h.set('hide', [...S.hidden].join());
  const s = h.toString();
  history.replaceState(null, '', s ? '#' + s : location.pathname);
}

/* ---- data from the worker ---- */
let META = null;                     // models, quality, benches
const cache = new Map();             // `${gpu}|${users}` -> rows
const worker = new Worker(new URL('./compare-worker.js', import.meta.url), { type: 'module' });
let reqId = 0;
worker.onmessage = ({ data }) => {
  if (data.ready){ META = data; buildStatic(); request(); return; }
  if (data.error){ statusEl.textContent = 'Could not price the configurations: ' + data.error.split('\n')[0]; return; }
  cache.set(`${data.gpu}|${data.users ?? pending.get(data.id)}`, data.rows);
  pending.delete(data.id);
  if (!pending.size) draw();
};
const pending = new Map();
function request(){
  const need = S.gpus.filter(g => !cache.has(`${g}|${S.users}`));
  if (!need.length){ draw(); return; }
  statusEl.hidden = false;
  statusEl.textContent = `Pricing ${need.join(' and ')} at ${S.users} users…`;
  for (const gpu of need){ const id = ++reqId; pending.set(id, S.users); worker.postMessage({ id, gpu, users: S.users }); }
}

/* ---- derived rows ---- */
const modelName = mk => META.models[mk].name;
const score = mk => { const q = (META.quality[mk] || {})[S.bench]; return Number.isFinite(q) ? q : null; };
const color = mk => css(`--m-${mk}`) || css('--accent');
function cfgName(r){
  const split = r.dp > 1 && r.tp > 1 ? ` DP${r.dp}×TP${r.tp}` : r.dp > 1 ? ' DP' : r.tp > 1 ? ' TP' : '';
  return `${r.ngpu}×${r.gpu}${split}${r.tags ? ' ' + r.tags : ''}`;
}
const xOf = r => S.x === 'seat' ? (r.eurSeat != null && r.limit >= 1 ? r.eurSeat : null) : (r.limit >= 1 ? r.limit : null);
const planUrl = r => `index.html#model=${r.mk}&gpu=${r.gpu}&ngpu=${r.ngpu}&tp=${r.tp}&users=${S.users}`;
function allRows(){
  return S.gpus.flatMap(g => (cache.get(`${g}|${S.users}`) || []))
    .map(r => ({ ...r, score: score(r.mk), name: modelName(r.mk), cfg: cfgName(r) }));
}
// one row per model and GPU: the cheapest split that carries the load, else
// the one that comes closest; the rest ride along as `alts`
function grouped(rows){
  const by = new Map();
  for (const r of rows){ const k = `${r.mk}|${r.gpu}`; if (!by.has(k)) by.set(k, []); by.get(k).push(r); }
  return [...by.values()].map(rs => {
    const fit = rs.filter(r => r.fits && r.eurSeat != null).sort((a, b) => a.eurSeat - b.eurSeat);
    const best = fit[0] || [...rs].sort((a, b) => b.limit - a.limit)[0];
    return { ...best, alts: rs.length - 1 };
  });
}
// the efficient set among rows that carry the load: no other is both
// stronger and better on the x axis (cheaper, or more users)
function efficient(rows){
  const c = rows.filter(r => r.fits && r.score != null && xOf(r) != null)
    .sort((a, b) => S.x === 'seat' ? xOf(a) - xOf(b) : xOf(b) - xOf(a));
  const out = []; let top = -1;
  for (const r of c) if (r.score > top + 1e-9){ out.push(r); top = r.score; }
  return out;
}

/* ---- static controls ---- */
function buildStatic(){
  $('bench').innerHTML = Object.entries(META.benches).reverse().map(([k, b]) =>
    `<button type="button" data-v="${k}" aria-pressed="${k === S.bench}">${esc(b.label)}</button>`).join('');
  $('gpus').innerHTML = META.gpus.map(g => `<button type="button" data-v="${g}" aria-pressed="${S.gpus.includes(g)}">${g}</button>`).join('')
    + `<button type="button" disabled title="WS-7 adds more GPUs">B200</button><button type="button" disabled title="WS-7 adds more GPUs">MI355X</button>`;
  $('users').value = USERS.indexOf(S.users);
  syncControls();
}
function syncControls(){
  $('usersOut').textContent = S.users.toLocaleString('en-US');
  for (const b of $('bench').children) b.setAttribute('aria-pressed', b.dataset.v === S.bench);
  for (const b of $('gpus').querySelectorAll('[data-v]')) b.setAttribute('aria-pressed', S.gpus.includes(b.dataset.v));
  for (const b of document.querySelectorAll('[data-scale]')) b.setAttribute('aria-pressed', b.dataset.scale === S.scale);
  for (const b of document.querySelectorAll('[role=tab]')) b.setAttribute('aria-selected', b.dataset.x === S.x);
  $('pareto').checked = S.pareto;
}

/* ---- the chart ---- */
let geom = null;   // last drawn points in px, for hover
function niceLogTicks(lo, hi){
  const t = [];
  for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++)
    for (const m of [1, 2, 5]){ const v = m * 10 ** e; if (v >= lo * 0.999 && v <= hi * 1.001) t.push(v); }
  return t.length > 9 ? t.filter(v => /^1/.test(String(v)) || /^5/.test(String(v))) : t;
}
function niceLinTicks(hi){
  const step = 10 ** Math.floor(Math.log10(hi / 5)); const m = hi / step > 25 ? 5 : hi / step > 10 ? 2 : 1;
  const out = []; for (let v = 0; v <= hi * 1.0001; v += step * m) out.push(v); return out;
}
const tickLabel = v => S.x === 'seat' ? (v >= 1000 ? `€${v / 1000}k` : `€${v}`) : (v >= 1000 ? `${v / 1000}k` : String(v));

function drawChart(rows, eff){
  const host = $('chart'), W = host.clientWidth, H = host.clientHeight;
  const narrow = W < 560;
  const labW = narrow ? 118 : 230;
  const m = { l: narrow ? 36 : 48, r: labW, t: narrow ? 14 : 28, b: 42 };
  const pts = rows.filter(r => !S.hidden.has(r.mk) && r.score != null && xOf(r) != null);
  if (!pts.length){ host.querySelector('svg')?.remove(); geom = null; statusEl.hidden = false; statusEl.textContent = 'Nothing to plot: every model is hidden or no configuration carries one user.'; return; }
  statusEl.hidden = true;
  const xs = pts.map(xOf);
  let x0, x1, X;
  if (S.scale === 'log'){
    x0 = 10 ** (Math.floor(Math.log10(Math.min(...xs)) * 2) / 2);
    x1 = 10 ** (Math.ceil(Math.log10(Math.max(...xs)) * 2) / 2);
    X = v => m.l + (Math.log10(v) - Math.log10(x0)) / (Math.log10(x1) - Math.log10(x0)) * (W - m.l - m.r);
  } else {
    x0 = 0; x1 = Math.max(...xs) * 1.05;
    X = v => m.l + v / x1 * (W - m.l - m.r);
  }
  const ymax = Math.max(0.1, Math.ceil((Math.max(...pts.map(r => r.score)) + 0.02) * 10) / 10);
  const Y = v => H - m.b - v / ymax * (H - m.t - m.b);
  const ink = css('--ink'), mut = css('--muted'), grid = css('--grid'), line = css('--line'), sheet = css('--sheet'), acc = css('--accent');
  const fs = narrow ? 10.5 : 11.5;

  // models sharing a score get their rails nudged apart by a few px
  const models = [...new Set(pts.map(r => r.mk))];
  const bySc = new Map();
  for (const mk of models){ const k = score(mk).toFixed(4); if (!bySc.has(k)) bySc.set(k, []); bySc.get(k).push(mk); }
  const yOf = {};
  for (const grp of bySc.values()) grp.forEach((mk, i) => { yOf[mk] = Y(score(mk)) + (i - (grp.length - 1) / 2) * 7; });

  let s = `<svg viewBox="0 0 ${W} ${H}" aria-label="${esc(META.benches[S.bench].name)} score against ${S.x === 'seat' ? 'euro per seat per month' : 'maximum users'} for ${pts.length} configurations">`;
  // grid and axes
  let xt = S.scale === 'log' ? niceLogTicks(x0, x1) : niceLinTicks(x1);
  // keep tick labels ~56 px apart
  const room = (W - m.l - m.r) / 56;
  if (xt.length > room) xt = S.scale === 'log' ? xt.filter(v => /^1/.test(String(v))) : xt.filter((v, i) => i % Math.ceil(xt.length / room) === 0);
  for (const t of xt){
    const x = X(t); if (x < m.l - 1 || x > W - m.r + 1) continue;
    s += `<line x1="${x}" x2="${x}" y1="${m.t}" y2="${H - m.b}" stroke="${grid}"/>`;
    s += `<text x="${x}" y="${H - m.b + 16}" text-anchor="middle" font-size="11" fill="${mut}" font-family="var(--f-num)">${tickLabel(t)}</text>`;
  }
  for (let v = 0; v <= ymax + 1e-9; v += 0.1){
    s += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="${v === 0 ? line : grid}"/>`;
    s += `<text x="${m.l - 8}" y="${Y(v) + 4}" text-anchor="end" font-size="11" fill="${mut}" font-family="var(--f-num)">${Math.round(v * 100)}%</text>`;
  }
  s += `<text x="${m.l + (W - m.l - m.r) / 2}" y="${H - 6}" text-anchor="middle" font-size="12" fill="${mut}">${S.x === 'seat' ? (narrow ? '€ per seat per month' : `€ per seat per month, configuration full${S.scale === 'log' ? ' (log)' : ''} · cheaper to the left`) : (narrow ? 'Max users' : `Max concurrent users${S.scale === 'log' ? ' (log)' : ''} · more to the right`)}</text>`;
  if (!narrow) s += `<text x="${m.l}" y="${m.t - 2}" font-size="12" fill="${mut}">${esc(META.benches[S.bench].name)}, pass@1</text>`;

  // rails: one per model, through every configuration it has on screen
  const g = [];
  for (const mk of models){
    const mine = pts.filter(r => r.mk === mk), c = color(mk), y = yOf[mk];
    const a = Math.min(...mine.map(xOf)), b = Math.max(...mine.map(xOf));
    s += `<g class="m" data-mk="${mk}">`;
    s += `<line class="rail" x1="${X(a)}" x2="${W - m.r + 6}" y1="${y}" y2="${y}" stroke="${c}" stroke-width="1.5" stroke-opacity=".35"/>`;
    s += `<line class="rail" x1="${X(a)}" x2="${X(b)}" y1="${y}" y2="${y}" stroke="${c}" stroke-width="2.5" stroke-opacity=".7"/>`;
    for (const r of mine.sort((p, q) => p.fits - q.fits)){
      const x = X(xOf(r)), rr = 5;
      const fill = r.fits ? c : sheet;
      s += r.gpu === 'H200'
        ? `<circle class="mk" cx="${x}" cy="${y}" r="${rr}" fill="${fill}" stroke="${r.fits ? sheet : c}" stroke-width="${r.fits ? 1.5 : 1.75}"/>`
        : `<rect class="mk" x="${x - rr}" y="${y - rr}" width="${rr * 2}" height="${rr * 2}" transform="rotate(45 ${x} ${y})" fill="${fill}" stroke="${r.fits ? sheet : c}" stroke-width="${r.fits ? 1.5 : 1.75}"/>`;
      g.push({ r, x, y });
    }
    s += `</g>`;
  }
  // the efficient line
  if (S.pareto && eff.length){
    const e = eff.filter(r => !S.hidden.has(r.mk) && xOf(r) != null);
    if (e.length){
      let d = `M${X(xOf(e[0]))},${yOf[e[0].mk]}`;
      for (let i = 1; i < e.length; i++) d += ` H${X(xOf(e[i]))} V${yOf[e[i].mk]}`;
      s += `<path d="${d}" fill="none" stroke="${acc}" stroke-width="2" stroke-dasharray="1 0" opacity=".9"/>`;
      for (const r of e) s += `<circle cx="${X(xOf(r))}" cy="${yOf[r.mk]}" r="9" fill="none" stroke="${acc}" stroke-width="1.5"/>`;
    }
  }
  // labels in a column at the right, spread so none overlap
  const lab = models.map(mk => ({ mk, y: yOf[mk] })).sort((a, b) => a.y - b.y);
  const gap = fs + 5;
  for (let i = 1; i < lab.length; i++) if (lab[i].y - lab[i - 1].y < gap) lab[i].y = lab[i - 1].y + gap;
  const over = lab.length ? lab[lab.length - 1].y - (H - m.b) : 0;
  if (over > 0){ lab[lab.length - 1].y -= over; for (let i = lab.length - 2; i >= 0; i--) if (lab[i + 1].y - lab[i].y < gap) lab[i].y = lab[i + 1].y - gap; }
  const lx = W - m.r + 14;
  for (const L of lab){
    const best = pts.filter(r => r.mk === L.mk && r.fits).sort((a, b) => S.x === 'seat' ? xOf(a) - xOf(b) : xOf(b) - xOf(a))[0];
    let name = modelName(L.mk);
    if (narrow) name = SHORT[L.mk] || name;
    const val = best ? (S.x === 'seat' ? fmtEur(xOf(best)) : fmtInt(xOf(best))) : 'over';
    s += `<g class="lbl" data-mk="${L.mk}">`;
    s += `<path d="M${W - m.r + 6},${yOf[L.mk]} L${lx - 4},${L.y}" stroke="${line}" fill="none"/>`;
    s += `<text x="${lx}" y="${L.y + fs * 0.35}" font-size="${fs}" font-weight="600" fill="${color(L.mk)}">${esc(name)}`
       + (narrow ? '' : `<tspan font-weight="400" fill="${mut}" font-family="var(--f-num)" dx="6">${val}</tspan>`) + `</text></g>`;
  }
  s += `<g id="hover"></g></svg>`;
  const tip = host.querySelector('.tip');
  host.innerHTML = s;
  host.appendChild(statusEl);
  if (tip) host.appendChild(tip);
  geom = { pts: g, m, W, H, X, Y };
  applyHighlight();
}

function applyHighlight(){
  const hl = S.pin ? S.pin.r.mk : S.hl;
  for (const el of $('chart').querySelectorAll('[data-mk]')) el.style.opacity = !hl || el.dataset.mk === hl ? '' : '.18';
  for (const el of $('models').querySelectorAll('button')) el.classList.toggle('hl', el.dataset.mk === hl);
  for (const el of $('rows').querySelectorAll('tr')) el.classList.toggle('hl', el.dataset.mk === hl);
}

/* ---- hover and the tooltip ---- */
function nearest(ev){
  if (!geom) return null;
  const box = $('chart').getBoundingClientRect(), px = ev.clientX - box.left, py = ev.clientY - box.top;
  let best = null, bd = 26 * 26;
  for (const p of geom.pts){ const d = (p.x - px) ** 2 + (p.y - py) ** 2; if (d < bd){ bd = d; best = p; } }
  return best;
}
function showTip(p, sticky){
  const host = $('chart');
  let tip = host.querySelector('.tip');
  if (!p){ tip?.remove(); $('hover') && ($('hover').innerHTML = ''); return; }
  if (!tip){ tip = document.createElement('div'); tip.className = 'tip'; host.appendChild(tip); }
  const r = p.r;
  tip.style.pointerEvents = sticky ? 'auto' : 'none';
  tip.innerHTML = `<h4 style="color:${color(r.mk)}">${esc(r.name)}</h4><div class="cfg">${esc(r.cfg)}</div>
    <dl><dt>${esc(META.benches[S.bench].label)}</dt><dd>${fmtPct(r.score)}${r.score === 0 ? ' measured' : ''}</dd>
    <dt>€ per seat</dt><dd>${fmtEur(r.eurSeat)}</dd>
    <dt>Max users</dt><dd>${r.censored ? '≥ ' : ''}${fmtInt(r.limit)}</dd>
    <dt>${S.users} users</dt><dd style="color:${r.fits ? 'var(--good)' : 'var(--crit)'}">${r.fits ? 'fits' : 'over'}</dd>
    <dt>Binds on</dt><dd>${BIND[r.binding] || r.binding}</dd></dl>
    <div class="hint">${sticky ? `<a href="${planUrl(r)}">Open in Plan →</a>` : 'Click to open in Plan'}</div>`;
  const tw = tip.offsetWidth, th = tip.offsetHeight;
  let left = p.x + 16, top = p.y - th / 2;
  if (left + tw > geom.W - 4) left = p.x - tw - 16;
  top = Math.max(0, Math.min(geom.H - th, top));
  if (left < 0){ left = Math.max(0, Math.min(geom.W - tw, p.x - tw / 2)); top = p.y + 18 > geom.H - th ? p.y - th - 18 : p.y + 18; }
  tip.style.left = left + 'px'; tip.style.top = top + 'px';
  const mut = css('--muted');
  $('hover').innerHTML = `<line x1="${p.x}" x2="${p.x}" y1="${geom.m.t}" y2="${geom.H - geom.m.b}" stroke="${mut}" stroke-dasharray="3 3"/>`
    + `<circle cx="${p.x}" cy="${p.y}" r="10" fill="none" stroke="${css('--ink')}" stroke-width="1.5"/>`;
}
$('chart').addEventListener('pointermove', ev => {
  if (ev.pointerType === 'touch' || S.pin) return;
  const p = nearest(ev);
  S.hl = p ? p.r.mk : null; applyHighlight(); showTip(p, false);
  $('chart').style.cursor = p ? 'pointer' : '';
});
$('chart').addEventListener('pointerleave', () => { if (S.pin) return; S.hl = null; applyHighlight(); showTip(null); });
$('chart').addEventListener('click', ev => {
  if (ev.target.closest('.tip a')) return;
  const p = nearest(ev);
  // a tap pins the card (it holds the link); a mouse click goes straight to Plan
  if (ev.pointerType === 'mouse' && p){ location.href = planUrl(p.r); return; }
  S.pin = p || null; S.hl = p ? p.r.mk : null; applyHighlight(); showTip(p, !!p);
});

/* ---- side list and table ---- */
function drawSide(groups){
  const q = $('find').value.trim().toLowerCase();
  const mks = Object.keys(META.models).sort((a, b) => (score(b) ?? -1) - (score(a) ?? -1));
  $('models').innerHTML = mks.filter(mk => !q || modelName(mk).toLowerCase().includes(q)).map(mk => {
    const best = groups.filter(r => r.mk === mk && r.fits && xOf(r) != null).sort((a, b) => S.x === 'seat' ? xOf(a) - xOf(b) : xOf(b) - xOf(a))[0];
    return `<li><button type="button" data-mk="${mk}" aria-pressed="${!S.hidden.has(mk)}" title="${esc(META.models[mk].arch)}">
      <span class="sw" style="background:${color(mk)}"></span><span>${esc(modelName(mk))}</span>
      <span class="v">${fmtPct(score(mk))}</span></button></li>`;
  }).join('');
}
function drawTable(groups){
  const cols = [
    ['name', 'Model · runs on', r => r.name],
    ['score', META.benches[S.bench].label, r => r.score ?? -1, true],
    ['seat', '€ / seat / month', r => r.eurSeat ?? 1e12, true],
    ['users', 'Max users', r => r.limit, true],
    ['fit', `${S.users} users`, r => (r.fits ? 0 : 1)],
    ['bind', 'Binds on', r => BIND[r.binding] || ''],
  ];
  const col = cols.find(c => c[0] === S.sort) || cols[2];
  const rows = [...groups].sort((a, b) => (b.limit >= 1) - (a.limit >= 1) || (col[2](a) > col[2](b) ? 1 : col[2](a) < col[2](b) ? -1 : 0) * S.dir);
  $('thead').innerHTML = cols.map(([k, h, , n]) => `<th class="${n ? 'n' : ''}"${k === S.sort ? ` aria-sort="${S.dir > 0 ? 'ascending' : 'descending'}"` : ''}><button type="button" data-sort="${k}">${esc(h)}${k === S.sort ? (S.dir > 0 ? ' ↑' : ' ↓') : ''}</button></th>`).join('');
  $('rows').innerHTML = rows.map(r => `<tr data-mk="${r.mk}" class="${r.fits ? '' : 'over'}" data-href="${planUrl(r)}">
    <td class="c-name"><span class="sw2" style="background:${color(r.mk)}"></span><a class="mname" href="${planUrl(r)}" style="color:inherit;text-decoration:none">${esc(r.name)}</a><br><span class="cfg2">${esc(r.cfg)}${r.alts ? ` · ${r.alts} other split${r.alts > 1 ? 's' : ''}` : ''}</span></td>
    <td class="n c-fig" data-h="${esc(META.benches[S.bench].label)}">${fmtPct(r.score)}${r.score === 0 ? '<span class="zero">measured</span>' : ''}<span class="badge" title="Artificial Analysis, read 2026-11-04">AA</span></td>
    <td class="n c-fig" data-h="€/seat/mo">${r.limit >= 1 ? fmtEur(r.eurSeat) : '—'}</td>
    <td class="n c-fig" data-h="max users">${r.limit >= 1 ? (r.censored ? '≥ ' : '') + fmtInt(r.limit) : 'not viable'}</td>
    <td class="c-fit"><span class="fit ${r.fits ? 'y' : 'x'}">${r.fits ? '✓ fits' : '✗ over'}</span></td>
    <td class="c-bind">${esc(BIND[r.binding] || r.binding)}</td></tr>`).join('');
  $('count').textContent = `${rows.length} rows · best split per model and GPU · ${rows.filter(r => r.fits).length} carry ${S.users} users`;
  $('ledgerCaption').textContent = `Each self-hosted model on each GPU, at its cheapest split that carries ${S.users} concurrent users. Click a row to open it in Plan.`;
}
function drawFinding(groups){
  const fit = groups.filter(r => r.fits && r.eurSeat != null && r.score != null);
  if (!fit.length){ $('finding').textContent = `No configuration of up to 8 GPUs carries ${S.users} users.`; return; }
  const cheap = [...fit].sort((a, b) => a.eurSeat - b.eurSeat)[0];
  const strong = [...fit].sort((a, b) => b.score - a.score || a.eurSeat - b.eurSeat)[0];
  const b = META.benches[S.bench].label;
  $('finding').innerHTML = `At ${S.users} users, the cheapest seat is <b>${fmtEur(cheap.eurSeat)}</b> a month: ${esc(cheap.name)} on ${esc(cheap.cfg)}, <b>${fmtPct(cheap.score)}</b> on ${b}.`
    + (strong !== cheap ? ` The strongest model that carries the load is ${esc(strong.name)} on ${esc(strong.cfg)}: <b>${fmtPct(strong.score)}</b> for <b>${fmtEur(strong.eurSeat)}</b> a seat.` : '');
}
function keyline(){
  const mut = css('--muted'), ink = css('--ink-2'), sheet = css('--sheet'), acc = css('--accent');
  $('keyline').innerHTML =
    `<span><svg width="12" height="12"><circle cx="6" cy="6" r="4.5" fill="${ink}"/></svg>H200</span>`
  + `<span><svg width="12" height="12"><rect x="2.5" y="2.5" width="7" height="7" transform="rotate(45 6 6)" fill="${ink}"/></svg>B300</span>`
  + `<span><svg width="12" height="12"><circle cx="6" cy="6" r="4.5" fill="${sheet}" stroke="${ink}" stroke-width="1.5"/></svg>cannot carry ${S.users} users</span>`
  + `<span><svg width="20" height="12"><path d="M1,9 H10 V3 H19" stroke="${acc}" stroke-width="2" fill="none"/></svg>best for the money: nothing is both stronger and ${S.x === 'seat' ? 'cheaper' : 'larger'}</span>`
  + `<span>Each line is one model; its dots are its GPU counts and splits.</span>`;
}

function draw(){
  if (!META) return;
  writeHash(); syncControls();
  if (S.gpus.some(g => !cache.has(`${g}|${S.users}`))){ request(); return; }
  const rows = allRows(), groups = grouped(rows), eff = efficient(rows);
  drawFinding(groups);
  drawChart(rows, eff);
  drawSide(groups);
  drawTable(groups);
  keyline();
  applyHighlight();
}

/* ---- wiring ---- */
$('users').addEventListener('input', e => { S.users = USERS[+e.target.value]; $('usersOut').textContent = S.users.toLocaleString('en-US'); });
$('users').addEventListener('change', () => { S.pin = null; showTip(null); draw(); });
$('bench').addEventListener('click', e => { const b = e.target.closest('button'); if (!b) return; S.bench = b.dataset.v; draw(); });
$('gpus').addEventListener('click', e => {
  const b = e.target.closest('button[data-v]'); if (!b) return;
  const on = new Set(S.gpus); on.has(b.dataset.v) ? on.delete(b.dataset.v) : on.add(b.dataset.v);
  if (!on.size) return;   // at least one GPU stays on
  S.gpus = META.gpus.filter(g => on.has(g)); S.pin = null; showTip(null); draw();
});
document.querySelector('.tabs').addEventListener('click', e => { const b = e.target.closest('[data-x]'); if (!b) return; S.x = b.dataset.x; S.sort = S.x === 'seat' ? 'seat' : 'users'; S.dir = S.x === 'seat' ? 1 : -1; S.pin = null; showTip(null); draw(); });
for (const b of document.querySelectorAll('[data-scale]')) b.addEventListener('click', () => { S.scale = b.dataset.scale; draw(); });
$('pareto').addEventListener('change', e => { S.pareto = e.target.checked; draw(); });
$('models').addEventListener('click', e => { const b = e.target.closest('button'); if (!b) return; S.hidden.has(b.dataset.mk) ? S.hidden.delete(b.dataset.mk) : S.hidden.add(b.dataset.mk); draw(); });
$('models').addEventListener('pointerover', e => { const b = e.target.closest('button'); if (S.pin) return; S.hl = b ? b.dataset.mk : null; applyHighlight(); });
$('models').addEventListener('pointerleave', () => { if (S.pin) return; S.hl = null; applyHighlight(); });
$('find').addEventListener('input', () => draw());
$('allOn').addEventListener('click', () => { S.hidden.clear(); draw(); });
$('allOff').addEventListener('click', () => { S.hidden = new Set(Object.keys(META.models)); draw(); });
$('thead').addEventListener('click', e => { const b = e.target.closest('[data-sort]'); if (!b) return; const k = b.dataset.sort;
  S.dir = S.sort === k ? -S.dir : (['score', 'users'].includes(k) ? -1 : 1); S.sort = k; draw(); });
$('rows').addEventListener('click', e => { if (e.target.closest('a')) return; const tr = e.target.closest('tr'); if (tr) location.href = tr.dataset.href; });
$('rows').addEventListener('pointerover', e => { const tr = e.target.closest('tr'); if (S.pin) return; S.hl = tr ? tr.dataset.mk : null; applyHighlight(); });
$('rows').addEventListener('pointerleave', () => { if (S.pin) return; S.hl = null; applyHighlight(); });
document.addEventListener('keydown', e => { if (e.key === 'Escape'){ S.pin = null; S.hl = null; showTip(null); applyHighlight(); } });
new ResizeObserver(() => { if (META && geom !== undefined) draw(); }).observe($('chart'));
// theme: follows the system; the button pins the other one for this visit
const theme = $('theme');
const isDark = () => document.documentElement.dataset.theme ? document.documentElement.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
const syncTheme = () => { theme.textContent = isDark() ? 'Light' : 'Dark'; };
theme.addEventListener('click', () => { document.documentElement.dataset.theme = isDark() ? 'light' : 'dark'; syncTheme(); draw(); });
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { syncTheme(); draw(); });
syncTheme();
readHash();
