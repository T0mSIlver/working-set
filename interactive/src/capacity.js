import { CONFIG, DECODE_MBU, GIB, PREFILL_MFU, effective_bw, is_moe, kv_pool_tokens,
         w_decode } from './config.js';
import { percentiles } from './mathlib.js';
import { sampleFull, sampleReqInto } from './workload.js';
import { state } from './state.js';

/* ============================================================================
   CAPACITY — warm reusable sessions in one cache  (mirrors _warm_once)
   ========================================================================== */
const _warmReq = {full:0, prefix:0, isCold:false};   // warmOnce scratch
function warmOnce(pool, ram_gib, model, wl){
  // reserve one block per distinct prefix a request class can actually use
  let reserved = wl.sys_user;
  if (!wl.sub_shares_prefix && wl.sub_ratio > 0) reserved += wl.sys_sub;
  const gpu_budget = pool - reserved;
  const ram_budget = ram_gib > 0 ? ram_gib*GIB - reserved*model.kv_bpt : 0;

  // every resident session also holds its constant DeltaNet recurrent state
  // (a warm hit needs the state, not just the attention KV) — charged in
  // KV-token equivalents, mirroring _warm_once in scenario_model.py
  const state_tok = model.deltanet_state / model.kv_bpt;

  // arrival-order STREAMING fill (GPU, then optional CPU offload) — samples
  // until both budgets are exhausted, so results are never draw-capped.
  // SAFETY guards against runaway budgets; hitting it CENSORS the result (the
  // budgets still had room), which is reported up so the readout can say "at
  // least this many" instead of quietly under-reporting. Reachable: a 40k
  // shared prefix at a 5k median makes every session cost only its recurrent
  // state, so a TP8 pool plus 1 TiB of offload holds >60k of them.
  const SAFETY = 60000;
  let warm = 0, warmGpu = 0, gpu = 0, ram = 0, gpuFull = false, done = false, i = 0;
  const r = _warmReq;                       // scratch, refilled every iteration
  for (; i<SAFETY && !done; i++){
    sampleReqInto(wl, r);
    const u = r.isCold ? r.full : Math.max(r.full - r.prefix, 0);
    if (!gpuFull){
      gpu += u + state_tok;
      if (gpu <= gpu_budget){ if (!r.isCold){ warm++; warmGpu++; } continue; }
      gpuFull = true;   // this request overflows the GPU -> spills to CPU below
    }
    if (ram_gib > 0){
      ram += u*model.kv_bpt + model.deltanet_state;
      if (ram <= ram_budget){ if (!r.isCold) warm++; } else done = true;
    } else done = true;
  }
  if (!done) console.warn("warmOnce: SAFETY cap reached — result censored");
  // censored = the fill ran out of DRAWS, not budget: counts are lower bounds
  // warm    = warm STORAGE (HBM + offloaded to host RAM)
  // warmGpu = HBM-resident only — the ONLY sessions that can decode without a
  //           PCIe restore first, so decode concurrency is computed over these
  // drawn   = requests sampled to fill both budgets — the cost of this fill,
  //           used to size the iteration count (see warmCapacity)
  return { warm, warmGpu, drawn:i, censored:!done };
}

// Returns {all, gpu, off}, each [p5,p50,p95]. `all` counts offloaded sessions
// as warm storage (the capacity view); `gpu` counts only HBM-resident sessions
// (the decode view); `off` is the offloaded count taken PER DRAW (all-gpu
// within the same fill) and then percentiled, so it is a statistic of the
// offloaded population rather than a difference of marginal percentiles.
// Mirrors warm_capacity(which=...) in scenario_model.py.
export function warmCapacity(model, topo, wl, ram_gib, n_iter, budget){
  const pool = kv_pool_tokens(model, topo);
  // Weights (+ reserve) leave no KV space -> there is no engine to serve or
  // to restore offloaded sessions into: capacity is zero REGARDLESS of the
  // CPU-offload buffer (mirrors warm_capacity in scenario_model.py).
  if (pool <= 0)
    return { all:[0,0,0], gpu:[0,0,0], off:[0,0,0], censored:false };
  // One fill costs one draw per resident session, so a big pool full of small
  // sessions (a 91M-token TP8 pool at a 5k median holds ~30k of them) makes a
  // 700-iteration run a multi-second freeze. Probe one fill for its true cost,
  // then spend at most `budget` draws. This only bites above a few thousand
  // sessions per fill — where the count's relative spread is ~1/sqrt(count),
  // i.e. under 1%, so a dozen fills already pin p5/p50/p95 to a fraction of a
  // percent. Ordinary configs (~100 sessions) keep every requested iteration.
  const first = warmOnce(pool, ram_gib, model, wl);
  const cap = Math.round((budget ?? CONFIG.WARM_BUDGET) / Math.max(1, first.drawn));
  // A censored fill is pinned at the draw cap, so every repeat returns the same
  // floor: sampling it many times buys no distribution, only cost.
  const floor = first.censored ? 3 : CONFIG.WARM_ITER_MIN;
  const iters = Math.max(Math.min(n_iter, floor), Math.min(n_iter, cap));
  const all = new Float64Array(iters), gpu = new Float64Array(iters),
        off = new Float64Array(iters);
  let censored = false;
  for (let i=0;i<iters;i++){
    const r = i===0 ? first : warmOnce(pool, ram_gib, model, wl);
    all[i] = r.warm; gpu[i] = r.warmGpu; off[i] = r.warm - r.warmGpu;
    censored = censored || r.censored;
  }
  return { all: percentiles(all, [5,50,95]), gpu: percentiles(gpu, [5,50,95]),
           off: percentiles(off, [5,50,95]), censored };
}

/* ============================================================================
   CONCURRENCY — decode tok/s vs max_num_seqs  (mirrors decode_curves)
   ========================================================================== */
// Size the decode sweep for the current config. `gpuP95` is the top of the
// GPU-resident warm band, so the axis always contains the "every warm session
// decodes at once" stress point with headroom to spare. A fixed sample COUNT
// (rather than a fixed step) keeps the curve equally smooth at any range, and
// iterations taper as the range grows. Percentiles hold up: each draw at
// concurrency n already averages n contexts, so its spread tightens as n rises.
export function decodePlan(gpuP95, draft){
  const q = draft ? CONFIG.DRAFT : CONFIG;
  const nMax = Math.max(CONFIG.DECODE_NMIN,
                        Math.ceil(gpuP95 * CONFIG.DECODE_HEADROOM));
  const step = Math.max(1, Math.round(nMax / q.DECODE_SAMPLES));
  const iter = Math.max(q.DECODE_ITER_MIN,
                        Math.min(q.DECODE_ITER, Math.round(q.DECODE_BUDGET / nMax)));
  return { nMax, step, iter };
}

// The two efficiency knobs, read through helpers so nothing reaches past the
// slider to the constant. Each falls back to the single global constant when
// state has not been seeded yet (the load-time unit checks run that early).
// The load-time unit checks and steadyChecks call decodeCurves/prefillSeconds
// BEFORE `state` is declared (steadyChecks says as much in its own comment), so
// these must survive being called during that window and fall back to the
// constant -- which is also exactly the value those checks pin against.
//
// try/catch, NOT `typeof state !== "undefined"`: typeof is only safe for
// UNDECLARED identifiers. `state` is a const declared later in the same scope,
// so it is in the temporal dead zone and typeof throws a ReferenceError just
// like a plain read would. That guard looked right, parsed fine, and silently
// killed every load-time check before it ran.
export function decodeMbu(){
  try { return state.mbu > 0 ? state.mbu : DECODE_MBU; }
  catch (e) { return DECODE_MBU; }
}
export function prefillMfu(){
  try { return state.mfu > 0 ? state.mfu : PREFILL_MFU; }
  catch (e) { return PREFILL_MFU; }
}

export function decodeCurves(model, topo, wl, nMax, step, n_iter){
  // MEASURED efficiency, not a roofline: see DECODE_MBU.
  const bw = effective_bw(topo) * (model.decode_mbu || DECODE_MBU), scale = topo.replicas;
  const ns=[]; for (let n=1; n<=nMax; n+=step) ns.push(n);
  // LOW-n anchors. Per-user speed is a hyperbola in n, so the first regular
  // interval (1 -> 1+step) contains most of the curve's entire drop: at the
  // widest axes that one segment spans 4,500 tok/s down to a few hundred, and
  // both charts draw it as a single straight line. It is also exactly where
  // the steady-state point lands at any sane load, so inverting the aggregate
  // curve there inherits the whole error. A geometric ladder fixes both, and
  // costs only O(n_iter) per point — the context pool is drawn once and read
  // cumulatively, so extra sample POSITIONS are nearly free.
  const firstGap = ns.length>1 ? ns[1] : nMax+1;
  for (const a of [2,3,4,6,8,11,16,22,32,45,64,90])
    if (a < Math.min(firstGap, nMax+1)) ns.push(a);
  if (ns.length > 1) ns.sort((a,b)=>a-b);
  // The conservative expert union kinks HARD at n_sat = total/pertok (the
  // batch activates every expert; n_sat = 32 for both MoE models). A wide-
  // step sweep that straddles n_sat interpolates straight through the kink
  // and draws a spurious notch around it — sample the kink point exactly so
  // charts C and D show the true piecewise shape.
  if (is_moe(model)){
    const nSat = Math.round(model.w_route_total / model.w_route_pertok);
    if (nSat > 1 && nSat < nMax && !ns.includes(nSat))
      ns.push(nSat), ns.sort((a,b)=>a-b);
  }
  const K=ns.length, last=ns[K-1];
  // ONE pool of contexts per iteration, read CUMULATIVELY: the sample for
  // concurrency n is the running sum of the pool's first n contexts. Drawing
  // each sampled n its own pool costs O(iter x K x nMax) — 69M draws and a
  // 3.5 s synchronous freeze at the widest legal axis — while the running sum
  // costs O(iter x nMax), ~K times less, for the same statistics: every n
  // still sees n i.i.d. contexts, so its marginal distribution is unchanged.
  // Neighbouring n now share draws (common random numbers), which only makes
  // the curve smoother. scenario_model.py redraws per n; that is a difference
  // in MC noise structure, not in the model.
  // DSA top-k reads are capped at each sequence's own length (a context
  // shorter than the 2,048-token window only reads its own tokens), so the
  // per-sequence top-k bytes need their own running sum over min(len, topk).
  const topk = (model.kv_decode_const && model.kv_decode_topk) ? model.kv_decode_topk : 0;
  const kvsum = Array.from({length:K}, ()=>new Float64Array(n_iter));
  const tksum = topk ? Array.from({length:K}, ()=>new Float64Array(n_iter)) : null;
  for (let it=0; it<n_iter; it++){
    let sum=0, tsum=0, k=0;
    for (let s=1; s<=last; s++){
      const L = sampleFull(wl);
      sum += L;
      if (topk) tsum += Math.min(L, topk);
      if (ns[k]===s){ kvsum[k][it]=sum; if (topk) tksum[k][it]=tsum; k++; }
    }
  }
  const p5=[], p50=[], p95=[], agg=[];
  const buf = new Float64Array(n_iter);
  // dense attention reads the whole cache (kv_bpt per context token); a
  // sparse-attention model (GLM-5.2/DSA) reads kv_decode_bpt per context
  // token (indexer scan) plus the length-capped top-k read per sequence
  const kvReadBpt = model.kv_decode_bpt ?? model.kv_bpt;
  const perTok = topk ? model.kv_decode_const/topk : 0;
  for (let k=0;k<K;k++){
    const n=ns[k];
    // weights + DeltaNet recurrent-state read+write for every active sequence
    const wd = w_decode(model, n) + 2*n*model.deltanet_state
             + (topk ? 0 : n*(model.kv_decode_const ?? 0));
    const col=kvsum[k], tcol=topk?tksum[k]:null;
    for (let it=0; it<n_iter; it++)
      buf[it] = model.mtp * bw / (wd + col[it]*kvReadBpt + (topk ? tcol[it]*perTok : 0));
    const [a,b,c] = percentiles(buf, [5,50,95]);
    p5.push(a); p50.push(b); p95.push(c);
    agg.push(n*b*scale);
  }
  return { ns, p5, p50, p95, agg };
}
