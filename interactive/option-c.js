// Option C: clicking a frontier row or dot selects that model + split by
// driving the existing controls (the same click/input events they listen to).
import { CONFIG } from './src/config.js';

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

document.getElementById('chartH').addEventListener('click', () => {
  const tt = document.getElementById('ttH');
  const h = tt && +getComputedStyle(tt).opacity > 0 && tt.querySelector('.tth');
  if (h) select(h.textContent.trim());
});
