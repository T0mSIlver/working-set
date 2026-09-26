/* Decode an explorer share URL into a full state, the way main.js
   applyURLState() does for a link whose values are already in range. main.js
   is UI wiring and cannot load under Node (see stub-main.mjs), so this is the
   decoder the Node tests share. It throws on a key the page does not keep in
   its state, where the real decoder would ignore it silently. */
import { CONFIG } from '../../interactive/src/config.js';
import { STATE_DEFAULTS, ttftPctFromFragment } from '../../interactive/src/state.js';
import { TTFT_PCTS } from '../../interactive/src/prefill.js';

// Inverse of main.js encodeStateURL(): the fragment carries only the DIFFS
// from STATE_DEFAULTS, so the type of each key is read off the defaults
// rather than from a second copy of the enum tables. The two keys that are
// re-seeded from the selection rather than defaulted globally get the same
// treatment encodeStateURL gives them when it decides not to write them.
export function decodeStateURL(url){
  const st = { ...STATE_DEFAULTS };
  const q = url.includes('#') ? url.slice(url.indexOf('#') + 1) : '';
  for (const [k, v] of new URLSearchParams(q)){
    if (!(k in STATE_DEFAULTS)) throw new Error(`unknown state key in URL: ${k}`);
    st[k] = STATE_DEFAULTS[k] === null ? Number(v)
          : typeof STATE_DEFAULTS[k] === 'boolean' ? v === '1'
          : typeof STATE_DEFAULTS[k] === 'number' ? Number(v)
          : v;
  }
  const p = new URLSearchParams(q);
  if (!p.has('mtp')) st.mtp = CONFIG.MODELS[st.model].mtp;
  if (!p.has('gpuh')) st.gpuh = CONFIG.GPUS[st.gpu].eur_gpu_h;
  // a non-bare link without ttft_pct predates the control: a miss's mean
  st.ttft_pct = ttftPctFromFragment(p, TTFT_PCTS);
  return st;
}
