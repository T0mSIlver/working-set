// The notebook layout. Reads the page the main modules render and
// re-presents parts of it; it never computes a number of its own.
//  1. the hypothesis as one sentence (from state)
//  2. the load test's predictions (the list #testBody renders) as a table,
//     with the binding one marked as the one that decides
//  3. a copy of the verdict tiles next to the MFU / MBU calibration sliders
//  4. clicking a frontier row or dot selects that model + split by driving
//     the existing controls (the same click/input events they listen to)
import { state, STATE_DEFAULTS } from './state.js';
import { CONFIG } from './config.js';
import { frontierChartGeom, frontierRowName } from './frontier.js';

const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const n0 = v => Number(v).toLocaleString('en-US', { maximumFractionDigits: 0 });

function hypothesis(){
  const m = CONFIG.MODELS[state.model];
  const name = m ? m.name.replace(/\s*\(.*\)\s*$/, '') : state.model;
  const dp = state.ngpu / state.tp;
  const split = dp > 1 && state.tp > 1 ? `DP${dp}×TP${state.tp}` : state.tp > 1 ? `TP${state.tp}` : dp > 1 ? `DP${dp}` : 'one GPU';
  const who = state.headcount !== null && state.headcount !== undefined
    ? `<span class="q">${n0(state.headcount)} people</span>`
    : `<span class="q">${n0(state.users)} concurrent sessions</span>`;
  const chunk = Number(state.chunk);
  $('hypSentence').innerHTML =
    `<b>${esc(name)}</b> (${esc(String(state.wdt).toUpperCase())}) on <b>${state.ngpu}×${esc(state.gpu)}</b>, ${split}`
    + (chunk ? `, chunk ${n0(chunk / 1024)}k` : '')
    + `, serves ${who} with a cache miss answered within <b>${state.sla} s</b> `
    + `and every user decoding at <b>${state.decode_floor} tok/s</b> or faster.`;
}

// "H-binding: the binding constraint is 'cache' — …" -> the row id it names
const BIND = { cache: 'H-cache', decode: 'H-decode', latency: 'H-latency', saturation: 'H-saturation' };
function predictions(){
  const lis = [...document.querySelectorAll('#testBody details li')].map(li => li.textContent);
  const tb = document.querySelector('#predTable tbody');
  if (!lis.length){
    tb.innerHTML = `<tr><td colspan="3">${esc($('testBody').textContent.trim() || 'No predictions for this configuration.')}</td></tr>`;
    return;
  }
  const rows = lis.map(t => {
    const i = t.indexOf(':');
    return { id: t.slice(0, i).trim(), text: t.slice(i + 1).trim() };
  });
  const bindRow = rows.find(r => r.id === 'H-binding');
  const bm = bindRow && bindRow.text.match(/'([a-z]+)'/);
  const binds = bm && BIND[bm[1]];
  tb.innerHTML = rows.map(r => {
    const decides = r.id === 'H-binding' || r.id === binds;
    const role = r.id === 'H-binding' ? 'decides' : decides ? 'decides · binds' : '';
    return `<tr class="${decides ? 'decides' : ''}"><td class="pid">${esc(r.id)}</td>`
      + `<td>${esc(r.text)}</td><td class="role">${role}</td></tr>`;
  }).join('');
}

let lastVerdict = '';
function readout(){
  const src = $('decisionTiles'), out = $('calReadout');
  const clone = src.cloneNode(true);
  clone.removeAttribute('id');
  clone.querySelectorAll('[id]').forEach(e => e.removeAttribute('id'));
  out.replaceChildren(clone);
  const text = clone.textContent;
  if (lastVerdict && text !== lastVerdict){
    out.classList.remove('calflash'); void out.offsetWidth; out.classList.add('calflash');
  }
  lastVerdict = text;
  const d = STATE_DEFAULTS;
  $('calDefaults').textContent =
    `defaults: prefill ${Number(d.mfu).toFixed(2)} · decode ${Number(d.mbu).toFixed(2)}`;
}

function refresh(){ hypothesis(); predictions(); readout(); }

let queued = false;
const schedule = () => { if (queued) return; queued = true; requestAnimationFrame(() => { queued = false; refresh(); }); };
const mo = new MutationObserver(schedule);
mo.observe($('decisionTiles'), { childList: true, subtree: true, characterData: true });
mo.observe($('testBody'), { childList: true, subtree: true });
schedule();

const shortName = mk => CONFIG.MODELS[mk].name.replace(/\s*\(.*\)\s*$/, '');

// "Qwen3.8-27B [FP16 KV] · 4×H200 DP2×TP2" -> {mk, ngpu, tp}
function parse(name){
  const [left, grid] = name.replace(/ — yours$/, '').split(' · ');
  if (!grid) return null;
  const model = left.replace(/\s*\[[^\]]+\]/g, '').trim();
  const mk = Object.keys(CONFIG.MODELS).find(k => shortName(k) === model);
  const g = grid.match(/^(\d+)×\S+(?:\s+(.*))?$/);
  if (!mk || !g) return null;
  const ngpu = +g[1], s = (g[2] || '').trim();
  let tp = ngpu;
  const hyb = s.match(/DP\d+×TP(\d+)/);
  if (hyb) tp = +hyb[1];
  else if (/^DP$/.test(s)) tp = 1;
  return { mk, ngpu, tp };
}

function select(name){
  const p = parse(name);
  if (!p) return;
  const mb = document.querySelector(`#seg-model button[data-v="${p.mk}"]`);
  if (mb && mb.getAttribute('aria-pressed') !== 'true' && !mb.disabled) mb.click();
  const sl = document.getElementById('s-ngpu');
  if (sl && +sl.value !== p.ngpu){
    sl.value = String(p.ngpu);
    sl.dispatchEvent(new Event('input', { bubbles: true }));
  }
  const sb = document.querySelector(`#seg-split button[data-v="${p.tp}"]`);
  if (sb && sb.getAttribute('aria-pressed') !== 'true') sb.click();
}

document.getElementById('frontierTable').addEventListener('click', e => {
  const tr = e.target.closest('tr');
  if (!tr || !tr.cells.length || tr.querySelector('th')) return;
  select(tr.cells[0].textContent.trim());
});

// the nearest dot within the hover handler's radius (main.js), found from
// the click itself: a tap closes the tooltip before its click arrives
document.getElementById('chartH').addEventListener('click', e => {
  const g = frontierChartGeom, svg = e.currentTarget.querySelector('svg');
  if (!g || !svg) return;
  const rect = svg.getBoundingClientRect(), scale = rect.width / g.W;
  const vx = (e.clientX - rect.left) / scale, vy = (e.clientY - rect.top) / scale;
  let best = null, bd = Infinity;
  for (const p of g.pts){ const d = Math.hypot(p.x - vx, p.y - vy); if (d < bd){ bd = d; best = p; } }
  if (best && bd <= Math.max(12, 24 / scale)) select(frontierRowName(best.r));
});
