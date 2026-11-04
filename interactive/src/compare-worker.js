/* Prices every frontier row for one GPU at one load, off the main thread.
   The Compare page posts {id, gpu, users}; this answers with slim rows. */
import { CONFIG } from './config.js';
import { setLiveThink, setLiveTurn } from './prefill.js';
import { STATE_DEFAULTS, currentWL, state } from './state.js';
import { computeFrontierRows } from './frontier-rows.js';

self.onmessage = ({ data }) => {
  const { id, gpu, users } = data;
  try {
    // the GPU-hour price follows the GPU, as the explorer re-seeds it on a switch
    Object.assign(state, STATE_DEFAULTS, { gpu, users, gpuh: CONFIG.GPUS[gpu].eur_gpu_h });
    setLiveTurn(state.turn); setLiveThink(state.think);
    const rows = computeFrontierRows(currentWL()).map(r => ({
      mk: r.mk, gpu, dp: r.dp, tp: r.tp, reps: r.reps,
      ngpu: r.dp * r.tp,
      tags: (r.label.split(' · ')[0].match(/\[[^\]]+\]/g) || []).join(' '),
      limit: r.op.limit, binding: r.op.binding, fits: r.op.fits,
      censored: !!r.censored, eurSeat: isFinite(r.eurSeat) ? r.eurSeat : null,
      eurLoad: r.eur, warmUsers: r.warmUsers, decodeUsers: r.decodeUsers,
    }));
    self.postMessage({ id, gpu, rows });
  } catch (e) {
    self.postMessage({ id, gpu, error: String(e && e.stack || e) });
  }
};
// the static model facts the page needs once: names and scores
self.postMessage({ ready: true,
  models: Object.fromEntries(Object.entries(CONFIG.MODELS).map(([k, m]) => [k, {
    name: m.name.replace(/\s*\(.*\)\s*$/, ''), arch: (m.name.match(/\((.*)\)/) || [])[1] || '' }])),
  quality: CONFIG.QUALITY, benches: CONFIG.BENCHES, gpus: Object.keys(CONFIG.GPUS) });
