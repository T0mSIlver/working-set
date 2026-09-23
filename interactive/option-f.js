// Option F: the decision brief. Everything here READS what the explorer has
// already rendered (the ceiling bars, the verdict tile, the sensitivity axes)
// and restates it as sentences and two small tables; nothing is recomputed.
import { CONFIG } from './src/config.js';
import { state, hasHeadcount, sessionsFromHeadcount } from './src/state.js';
import { lastFlipAxes } from './src/sensitivity.js';

const $ = id => document.getElementById(id);
const fmt = (x, d = 0) => x.toLocaleString('en-US', { maximumFractionDigits: d, minimumFractionDigits: d });
const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const num = s => { const m = String(s).replace(/,/g, '').match(/-?\d+(\.\d+)?/); return m ? +m[0] : NaN; };

const KEYS = ['cache', 'decode', 'latency', 'saturation'];
const NAME = {
  cache: 'KV-cache capacity', decode: 'decode speed',
  latency: 'cache-miss latency', saturation: 'prefill saturation',
};
const WHAT = {
  cache: 'warm sessions whose context still fits in GPU memory',
  decode: `every user still decodes at the speed floor`,
  latency: 'a cache miss still gets its first token inside the TTFT budget',
  saturation: 'prefill keeps up with arriving requests (100% busy)',
};

const modelName = () => CONFIG.MODELS[state.model].name.replace(/\s*\(.*\)\s*$/, '');
function setup(){
  const split = document.querySelector('#seg-split button[aria-pressed="true"]');
  const s = split ? split.textContent.trim() : '';
  const grid = `${state.ngpu} × ${state.gpu}` + (s && state.ngpu > 1 ? ` (${s})` : '');
  return `${grid} running <span class="nw">${esc(modelName())}</span>`;
}
const people = () => hasHeadcount();
const unit = n => people() ? (n === 1 ? 'developer' : 'developers') : (n === 1 ? 'concurrent user' : 'concurrent users');

// The ceiling bars carry all four ceilings in the binding order the model
// uses; each value label reads "N" or "N sessions ≈ M people".
function readCeilings(){
  const box = $('ceilingBars');
  const labels = [...box.querySelectorAll('text.axlbl')].map(t => t.textContent.trim());
  const vals = [...box.querySelectorAll('text.dlabel')];
  if (labels.length < 4 || vals.length < 5) return null;
  const out = {};
  labels.slice(0, 4).forEach((k, i) => {
    const t = vals[i].textContent;
    if (/^—/.test(t.trim())){ out[k] = { sessions: Infinity, people: Infinity }; return; }
    const [a, b] = t.split('≈');
    out[k] = { sessions: num(a), people: b ? num(b) : num(a) };
  });
  const loadTxt = vals[4].textContent;
  return { c: out, load: num(loadTxt.replace(/^your load/, '')) };
}

function render(){
  const lead = $('fLead'), ans = $('fAnswer'), dek = $('fDek');
  const tiles = $('decisionTiles').textContent;
  const sessionsNow = people() ? sessionsFromHeadcount(state.headcount, state.active, state.spu) : state.users;
  const loadShown = people() ? state.headcount : state.users;
  renderSlo(sessionsNow);
  $('fStamp').textContent = `Model projection · ${modelName()} ${state.wdt.toUpperCase()} weights, ${state.kv.toUpperCase()} KV · `
    + `${state.ngpu}×${state.gpu} · batched-token chunk ${fmt(+state.chunk)} · ${new Date().toISOString().slice(0, 10)}`;

  if (/do not fit/.test(tiles)){
    const hint = tiles.split('configuration —')[1] || '';
    lead.className = 'lead no';
    ans.innerHTML = `<span class="yn">No.</span> ${esc(modelName())}'s weights do not fit on ${setup().split(' running')[0]}.`;
    dek.textContent = hint ? `To fix it: ${hint.trim()}` : '';
    $('fWhyLede').textContent = 'Nothing runs, so no constraint binds.';
    $('fWhy').innerHTML = ''; $('fLean').innerHTML = '';
    openNo(true);
    return;
  }
  const r = readCeilings();
  if (!r) return;
  const order = KEYS.filter(k => r.c[k]).sort((a, b) => r.c[a].sessions - r.c[b].sessions);
  const bind = order[0], lim = r.c[bind];
  const load = r.load;                       // sessions the model priced
  const fits = lim.sessions >= load;
  const limShown = people() ? lim.people : lim.sessions;
  const next = order[1];

  lead.className = 'lead ' + (fits ? 'yes' : 'no');
  if (lim.sessions < 1){
    ans.innerHTML = `<span class="yn">No.</span> ${setup()} cannot meet a ${fmt(state.sla)} s first-token budget at any load: `
      + `one cache miss takes longer than that to prefill on its own.`;
  } else if (fits){
    ans.innerHTML = `<span class="yn">Yes.</span> ${setup()} carries ${fmt(loadShown)} ${unit(loadShown)}; `
      + `the first limit is ${NAME[bind]} at ${fmt(limShown)}.`;
  } else {
    ans.innerHTML = `<span class="yn">No.</span> ${setup()} runs out of ${NAME[bind]} at ${fmt(limShown)} ${unit(limShown)}, `
      + `short of the ${fmt(loadShown)} asked for.`;
  }
  const pct = load / lim.sessions * 100;
  const nx = next && isFinite(r.c[next].sessions)
    ? ` After that comes ${NAME[next]} at ${fmt(people() ? r.c[next].people : r.c[next].sessions)}.` : '';
  dek.textContent = lim.sessions < 1 ? nx.trim()
    : (fits ? `The load uses ${fmt(pct)}% of that limit, so ${fmt(Math.max(0, limShown - loadShown))} more ${unit(2)} fit before it binds.`
            : `The load is ${fmt(pct)}% of that limit.`) + nx;
  if (people() && Math.abs(sessionsNow - load) >= 1)
    dek.textContent += ` (${fmt(sessionsNow)} sessions requested, priced at the nearest step, ${fmt(load)}.)`;

  // Why: the four ceilings
  $('fWhyLede').innerHTML = `Four things cap how many ${unit(2)} one deployment carries. The smallest decides; `
    + `headroom is how far each sits above the load of <b>${fmt(loadShown)}</b>.`;
  $('fWhy').innerHTML = `<thead><tr><th>Constraint</th><th class="num">Ceiling</th><th class="num">Headroom</th></tr></thead><tbody>`
    + order.map(k => {
      const v = r.c[k], shown = people() ? v.people : v.sessions;
      const h = isFinite(v.sessions) ? (v.sessions / load - 1) * 100 : Infinity;
      const cls = h < 0 ? 'crit' : h < 25 ? 'warn' : 'good';
      return `<tr class="${k === bind ? 'bind' + (fits ? '' : ' over') : ''}"><th>${NAME[k]}<span class="note">${WHAT[k]}</span></th>`
        + `<td class="num">${isFinite(shown) ? fmt(shown) : 'none'}</td>`
        + `<td class="num ${cls}">${isFinite(h) ? (h >= 0 ? '+' : '−') + fmt(Math.abs(h)) + '%' : '—'}</td></tr>`;
    }).join('') + '</tbody>';

  renderLean();
  openNo(!fits);
}

function renderSlo(sessionsNow){
  const loadRow = people()
    ? `${fmt(state.headcount)} developers → ${fmt(sessionsNow)} concurrent sessions`
    : `${fmt(state.users)} concurrent users`;
  $('fSlo').innerHTML = `<thead><tr><th>Requirement</th><th class="num">Target</th></tr></thead><tbody>`
    + `<tr><th>First token on a cache miss<span class="note">mean time to first token; a p95 budget binds sooner</span></th><td class="num">≤ ${fmt(state.sla)} s</td></tr>`
    + `<tr><th>Decode speed per user<span class="note">with every warm session decoding at once</span></th><td class="num">≥ ${fmt(state.decode_floor)} tok/s</td></tr>`
    + `<tr><th>Load to carry<span class="note">one turn every ${fmt(state.think)} s, ${fmt(state.inval, 1)}% of requests miss the cache</span></th><td class="num">${loadRow}</td></tr>`
    + `</tbody>`;
}

function renderLean(){
  const box = $('fLean');
  const ax = lastFlipAxes && lastFlipAxes[0];
  if (!ax){ box.innerHTML = ''; box.hidden = true; return; }
  box.hidden = false;
  let what;
  if (!ax.flip) what = 'no assumption changes the answer anywhere in its plausible range.';
  else {
    const side = ax.flip.dir > 0 ? 'rises to' : 'falls to';
    what = ax.flip.fitFlip
      ? (ax.flip.nowFits ? `if it ${side} ${ax.fmt(ax.flip.v)}, the answer turns to no.`
                         : `if it ${side} ${ax.fmt(ax.flip.v)}, the answer turns to yes.`)
      : `if it ${side} ${ax.fmt(ax.flip.v)}, ${NAME[ax.flip.bind]} becomes the first limit instead.`;
  }
  box.innerHTML = `<span class="lbl">The assumption it leans on hardest</span>`
    + `<b>${esc(ax.label)}</b>, now ${esc(ax.fmt(ax.cur))}: ${esc(what)} `
    + `Check it against measured traffic before relying on the answer.`;
}

let lastNo = null;
function openNo(no){
  if (no !== lastNo){ $('fNoDet').open = no; lastNo = no; }
}

let raf = 0;
const schedule = () => { cancelAnimationFrame(raf); raf = requestAnimationFrame(render); };
const mo = new MutationObserver(schedule);
for (const id of ['decisionTiles', 'ceilingBars', 'flipRows', 'seg-split'])
  mo.observe($(id), { childList: true, subtree: true, characterData: true, attributes: id === 'seg-split' });
schedule();

// the shared share button resets its label to the explorer's wording
const st = $('shareTxt');
new MutationObserver(() => {
  if (st.textContent === 'Share this configuration') st.textContent = 'Copy link to this brief';
}).observe(st, { childList: true, characterData: true, subtree: true });
$('fPrint').addEventListener('click', () => window.print());
