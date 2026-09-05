import { state } from './state.js';

/* ============================================================================
   RNG + math primitives
   ========================================================================== */
let _spare = null;
/* ---- deterministic sampling ------------------------------------------------
   Every Monte-Carlo draw used to come from the platform RNG, so the headline
   "planning number" moved by a session or two on every render — including
   renders triggered by controls that cannot affect capacity at all — and the
   hero tile and chart B could print different values for the SAME statistic,
   having each drawn independently. scenario_model.py has always used fixed
   seeds; this brings the page into line.

   mulberry32, seeded per PURPOSE from a hash of only those state fields that
   change the sampling. Load and latency knobs are excluded, so dragging the
   TTFT budget or the user count leaves every capacity figure rock steady.
   -------------------------------------------------------------------------- */
let _rngState = 1;
export function rnd(){
  _rngState = _rngState + 0x6D2B79F5 | 0;
  let t = Math.imul(_rngState ^ _rngState >>> 15, 1 | _rngState);
  t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
  return ((t ^ t >>> 14) >>> 0) / 4294967296;
}
function _hash(str){
  let h = 2166136261 >>> 0;
  for (let i=0;i<str.length;i++){ h ^= str.charCodeAt(i); h = Math.imul(h, 16777619) >>> 0; }
  return h >>> 0;
}
// the fields that change what gets sampled — deliberately NOT users/think/
// sla/turn/burst, none of which touch the workload or the pool
export function samplingSig(){
  return [state.model,state.gpu,state.wdt,state.kv,state.state_dt,state.wover,
          state.mtp,state.mbu,state.mfu,state.ngpu,state.tp,state.ram,state.cap,
          state.user_median,state.user_sigma,state.sub_median,state.sub_sigma,
          state.sub_ratio,state.sub_shares_prefix,state.sys,state.inval].join('|');
}
// _spare must be dropped too: a half-used Box-Muller pair carried across a
// reseed makes the section's draws depend on whatever sampled BEFORE it, so
// "deterministic given the state" silently wasn't
// `sig` pins the sampling signature to a caller-chosen snapshot: the sliced
// frontier rebuild passes the one it captured at job start, so a seed-only
// knob (model, MTP, ngpu/tp — none of which change what the rows contain)
// moved mid-rebuild cannot shift the draws of the rows still queued
export function seedFor(purpose, sig){ _rngState = _hash((sig || samplingSig()) + '#' + purpose) || 1; _spare = null; }
export function seedRng(seed){ _rngState = seed; _spare = null; }   // an explicit seed (the steady-state checks)

function randn(){                        // standard normal (Box-Muller)
  if (_spare !== null){ const v=_spare; _spare=null; return v; }
  let u=0, v=0;
  while(u===0) u=rnd();
  while(v===0) v=rnd();
  const r = Math.sqrt(-2*Math.log(u));
  const t = 2*Math.PI*v;
  _spare = r*Math.sin(t);
  return r*Math.cos(t);
}
export function lognormal(median, sigma){ return Math.exp(Math.log(median) + sigma*randn()); }
export function clip(x, lo, hi){ return x<lo?lo:(x>hi?hi:x); }

// numpy-style linear percentile; sorts a copy
export function percentiles(arr, ps){
  const a = Float64Array.from(arr).sort();
  const N = a.length;
  return ps.map(p=>{
    if (N===0) return 0;
    if (N===1) return a[0];
    const rank = (p/100)*(N-1);
    const lo = Math.floor(rank), hi = Math.ceil(rank), frac = rank-lo;
    return a[lo] + frac*(a[hi]-a[lo]);
  });
}
