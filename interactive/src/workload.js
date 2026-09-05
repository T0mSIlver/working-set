import { clip, lognormal, rnd } from './mathlib.js';

/* ============================================================================
   WORKLOAD sampling  (mirrors Workload.sample)
   wl fields already in absolute tokens.
   ========================================================================== */
export function p_sub(wl){ return wl.sub_ratio / (1 + wl.sub_ratio); }

// sample one request into `out` -> {full, prefix, isCold}. A warm fill draws
// one request per resident session — millions per render at large pools — so
// the hot path writes into a caller-owned object instead of allocating one.
export function sampleReqInto(wl, out){
  const isSub = rnd() < p_sub(wl);
  let length = isSub ? lognormal(wl.sub_median, wl.sub_sigma)
                     : lognormal(wl.user_median, wl.user_sigma);
  let prefix = isSub ? (wl.sub_shares_prefix ? wl.sys_user : wl.sys_sub) : wl.sys_user;
  // a prompt always contains at least its shared system prefix, never > cap
  const full = clip(length, Math.min(prefix, wl.cap), wl.cap);
  const isCold = rnd() < wl.invalidation;
  if (isCold) prefix = 0;   // cold: matches nothing (occupies full length)
  out.full=full; out.prefix=prefix; out.isCold=isCold;
  return out;
}
function sampleReq(wl){ return sampleReqInto(wl, {full:0, prefix:0, isCold:false}); }
// sample just a length (decode only needs full)
export function sampleFull(wl){
  const isSub = rnd() < p_sub(wl);
  let length = isSub ? lognormal(wl.sub_median, wl.sub_sigma)
                     : lognormal(wl.user_median, wl.user_sigma);
  const prefix = isSub ? (wl.sub_shares_prefix ? wl.sys_user : wl.sys_sub) : wl.sys_user;
  return clip(length, Math.min(prefix, wl.cap), wl.cap);
}
