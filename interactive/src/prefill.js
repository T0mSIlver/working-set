import { CONFIG, DECODE_COMFORT_RATIO, DECODE_FLOOR_TOKS, effective_bw, tpEff } from './config.js';
import { clip, percentiles } from './mathlib.js';
import { sampleFull } from './workload.js';
import { decodeCurves, prefillMfu } from './capacity.js';
import { state } from './state.js';
import { interpAt } from './charts.js';

/* ============================================================================
   PREFILL — the compute roofline (mirrors scenario_model.py's prefill section;
   research/prefill.md). Everything else on this page prices HBM bytes; a cache
   MISS prices FLOPs: it reads the weights once and does 2 x params x tokens on
   them. ANALYTIC, MFU CALIBRATED 2026-08-27 — MFU is the soft input (the 35–55%
   bracket moves every absolute figure by ~1.6x, though not the ratios).
   ========================================================================== */
// 0.45 is now a CALIBRATION POINT, not a flat rate: it is the effective MFU
// of a first chunk at the study's 32,768-token default. Away from that point
// the effective MFU moves with the chunk size (see mfuCeil/prefillPassSeconds
// below and chart E) — smaller chunks amortise the per-pass overhead over
// fewer FLOPs and achieve less of peak, which is the real reason vLLM's
// default is high. At C = 32,768 the published figures reproduce the old
// flat-45% model: exactly for one first pass, within <0.2% for a chunked
// dense context and within ~2% (under — the flat tables err conservative)
// for the MoEs, whose later passes run at their higher solved ceiling.
// Deliberate exception: a context SHORTER than the chunk is one small pass
// below the anchor's efficiency, so short (p5-ish) misses now read ~10%
// dearer on the MoEs — that is the model's content, not drift.
// PREFILL_MFU and its [LO, HI] bracket live with the study constants in the
// config section: the state object's defaults read them at load.
export const PREFILL_CHUNK = 32768;   // vLLM max_num_batched_tokens (the study default)
export const WARM_TURN_TOK = 2000;    // tokens a warm hit still prefills (the new turn).
// The study's reference value and the Warm turn size slider's default.
// The prefill functions below are defined ABOVE the state object and are also
// exercised by the unit checks, which run before any slider exists — so they
// read these two live mirrors rather than `state` directly (a const in its
// temporal dead zone throws even for typeof). syncState() keeps them equal to
// the sliders; at their defaults these ARE the published reference values, so
// the unit checks pin the page to the docs' tables by construction.
export let liveTurn = WARM_TURN_TOK, liveThink = 30;
export function setLiveTurn(v){ liveTurn = v; }
export function setLiveThink(v){ liveThink = v; }
export const REF_REQ_RATE  = 2.13;    // req/s: 64 users, one turn every 30 s (docs' f* load)

// FLOPs of ONE forward pass over T tokens with P tokens already cached:
// GEMM + intra-chunk attention (causal halves the pair) + cross-attention of
// the new queries over ALL cached tokens (the KV cache saves recomputing
// their keys/values, not attending over them). Pair-counting telescopes, so
// a context's TOTAL FLOPs are chunk-size invariant: chunking bounds the
// per-pass spike, not the FLOP count. Machine TIME is another matter — the
// per-pass overhead below does not telescope (missContextSeconds).
export function prefillFlops(m, T, P){
  P = P || 0;
  return 2*m.params_prefill*T
       + 2*T*T*m.attn_d*m.attn_layers
       + 4*T*P*m.attn_d*m.attn_layers;
}
// Dense FP8 FLOP/s of one replica group. Reusing the BANDWIDTH-fitted tpEff
// haircut is an assumption (prefill collectives likely amortise better); it
// errs toward slower prefill, i.e. against the deployment.
export function peakFlops(topo){
  const gpu = topo.gpu || CONFIG.GPUS["H200"];
  return topo.tp * gpu.peak_flops_fp8 * tpEff(topo.tp, gpu.nvlink_domain);
}
// MARGINAL pass time at the calibrated flat MFU — the FLOPs-only price, no
// per-pass overhead. Still the right price for work that PIGGYBACKS on a
// forward pass someone else already pays for: the ITL-spike chunk joins a
// decode batch whose pass streams the weights regardless, and a warm hit's
// small turn rides the serving steady state the same way. A MISS at the
// prefill-duty ceiling has no such host — its passes run back-to-back and
// each pays the overhead — so misses are priced by prefillPassSeconds below.
export function prefillSeconds(m, topo, T, P, mfu){
  return prefillFlops(m, T, P) / (peakFlops(topo) * (mfu || prefillMfu()));
}
// Whole context, chunked, on top of `prior` cached tokens: each chunk is
// priced with the cache it actually attends over, so later chunks cost more.
// MARGINAL (no per-pass overhead): the total equals one unchunked pass
// exactly (telescoping pairs), i.e. chunk-size invariant. Used for the warm
// hit; a miss's chunked cost is missContextSeconds, which is NOT invariant.
export function prefillContextSeconds(m, topo, context, chunk, prior, mfu){
  const C = chunk || PREFILL_CHUNK;
  let s = 0, done = 0, P = prior || 0;
  while (done < context){
    const step = Math.min(C, context - done);
    s += prefillSeconds(m, topo, step, P + done, mfu);
    done += step;
  }
  return s;
}

/* ---- MFU as a function of chunk size (chart E; research/prefill.md #3) ----
   Why a real machine keeps max_num_batched_tokens HIGH: the FLOPs of a
   context telescope across chunks, but each forward pass also pays costs
   that do NOT — it must stream the full resident weights once (on a MoE a
   chunk of any practical size touches every routed expert, so the whole
   expert bank streams; this is why the MoE curve degrades hardest), plus
   kernel launches and collectives. Model: one pass costs
       FLOPs/(peak x MFU_CEIL)  +  w_resident / effective_bw(topo)
   an additive (no-overlap) roofline — conservative at small chunks, where
   there is too little compute to hide the stream, which is the regime the
   term exists to price. MFU_CEIL is solved per (model, topo) so that the
   EFFECTIVE MFU of a first chunk at the 32,768 default is exactly the
   calibrated 0.45 — the anchor absorbs whatever overlap the real machine
   achieves there, and every published figure stays put. The anchor chunk
   and MFU are deliberately hard-wired below (Python's mirrors expose them
   as mfu_anchor/chunk parameters; the explorer has exactly one anchor).
   Kernel-launch and
   scheduler overheads are still unpriced (no constant for them exists in
   this study), so the small-chunk end still reads somewhat BETTER than a
   real machine would: the honest direction for a curve whose message is
   "small chunks are not free". */
export function prefillOverheadSeconds(m, topo){
  return m.w_resident / effective_bw(topo);
}
export function mfuCeil(m, topo, anchor){
  const A = anchor || prefillMfu();
  // PREFILL_CHUNK here is the CALIBRATION ANCHOR, never state.chunk: the
  // ceiling is solved so a 32,768 first chunk reads exactly 0.45. If this
  // followed the selected chunk, choosing 4,096 would re-solve the ceiling
  // and the very MFU penalty chart E measures would vanish.
  const ref = prefillFlops(m, PREFILL_CHUNK, 0);
  const inv = 1/A
            - peakFlops(topo) * prefillOverheadSeconds(m, topo) / ref;
  if (!(inv > 0)) throw new Error(
    `${m.name}: per-pass overhead exceeds the whole ${(A*100).toFixed(0)}% budget `
    + `at C=32,768 — the anchor cannot be solved on this topology`);
  return 1/inv;
}
// One MISS-side forward pass: FLOPs at the solved ceiling + the overhead.
function prefillPassSeconds(m, topo, T, P){
  return prefillFlops(m, T, P) / (peakFlops(topo) * mfuCeil(m, topo))
       + prefillOverheadSeconds(m, topo);
}
// Effective MFU of a first (cache-empty) chunk of C tokens — chart E's top
// strip, and the quantity the 0.45 anchor is stated in.
export function mfuEff(m, topo, C){
  return prefillFlops(m, C, 0)
       / (peakFlops(topo) * prefillPassSeconds(m, topo, C, 0));
}
// A miss's whole context, chunked at C: the FLOPs part telescopes as ever,
// the overhead multiplies by the pass count — smaller chunks now cost MORE
// total machine time, which is the throughput side of the chunk-size trade.
export function missContextSeconds(m, topo, context, chunk, prior){
  const C = chunk || PREFILL_CHUNK;
  const ceil = mfuCeil(m, topo), over = prefillOverheadSeconds(m, topo);
  let s = 0, done = 0, P = prior || 0;
  while (done < context){
    const step = Math.min(C, context - done);
    s += prefillFlops(m, step, P + done) / (peakFlops(topo) * ceil) + over;
    done += step;
  }
  return s;
}
// E[ceil(L/C)] over the sampled context lengths — the expected pass count a
// miss's overhead multiplies by. Sampled, not E[L]/C + 0.5: the truncation
// at the cap makes the fractional parts anything but uniform.
// Memoised on the draw itself: a contextStats object is immutable once made,
// and one render asks for the same C many times over (chart E's 41-point
// sweep, E3's seven models on the same grid, every tile at the priced chunk).
// The scan is 20,000 ceils, so the memo is what keeps E3 free on drag frames.
function meanPasses(cs, C){
  const memo = cs.passMemo || (cs.passMemo = new Map());
  let v = memo.get(C);
  if (v !== undefined) return v;
  const a = cs.samples;
  let s = 0;
  for (let i = 0; i < a.length; i++) s += Math.ceil(a[i]/C);
  v = s / a.length;
  memo.set(C, v);
  return v;
}
// Context-length statistics of the workload. A miss re-prefills the WHOLE
// context of whichever session missed, so its cost inherits this spread:
// the MEAN is what a sustained cold-request rate averages over (same reason
// the mean, not the median, drives warm capacity), while p5/p95 bound what a
// single cheap/expensive miss costs. Monte-Carlo like everything else on the
// page — ~1% sampling noise against the true distribution — but the draws are
// seeded from the configuration, so the same knobs always render the same
// numbers.
export function contextStats(wl, n){
  n = n || 20000;
  const a = new Float64Array(n);
  let s = 0, s2 = 0;
  for (let i = 0; i < n; i++){ a[i] = sampleFull(wl); s += a[i]; s2 += a[i]*a[i]; }
  const [p5, p95] = percentiles(a, [5, 95]);
  // samples ride along for meanPasses (E[ceil(L/C)] has no clean closed form)
  return { mean: s/n, meanSq: s2/n, p5, p95, samples: a };
}
function meanContext(wl, n){ return contextStats(wl, n).mean; }
// EXPECTED machine time of one MISS, averaged over the context-length
// distribution. The FLOPs part keeps its closed form via the telescoping
// pair-count (a chunked context L costs gemm(L) + 2 L^2 d n_layers however
// it is chunked); the per-pass overhead does NOT telescope and multiplies
// by the expected pass count instead:
//   E[cost] = (2 params E[L] + 2 d n_layers E[L^2]) / (peak x MFU_CEIL)
//           + E[ceil(L/C)] x overhead
// E[L^2], not E[L]^2 — the quadratic term is priced on the heavy tail.
export function coldRequestSeconds(m, topo, wl, cs, chunk){
  const st = cs || contextStats(wl);
  const C = chunk || PREFILL_CHUNK;
  const flops = 2*m.params_prefill*st.mean
              + 2*m.attn_d*m.attn_layers*st.meanSq;
  return flops / (peakFlops(topo) * mfuCeil(m, topo))
       + meanPasses(st, C) * prefillOverheadSeconds(m, topo);
}
// Cold req/s at which prefill alone consumes the whole replica group — the
// hard ceiling the capacity model cannot see: set by FLOPs, so no KV pool,
// CPU offload or warm-session headroom raises it. 1/MEAN-miss-cost, not a
// percentile: a sustained rate averages over many misses, so the mean is the
// only statistic that yields a throughput ceiling (a "p5 rate" would just be
// the reciprocal cost of an unusually long single miss, not a capacity).
function maxColdRate(m, topo, wl, chunk){
  return 1 / coldRequestSeconds(m, topo, wl, null, chunk);
}
// Miss rate at which prefill duty hits 100% at `rate` req/s, warm turns
// included (duty is linear in f: f x cold + (1-f) x warm = 1/rate). Can
// exceed 1: prefill cannot saturate at this request rate (which says nothing
// about which OTHER constraint binds — KV capacity is judged separately).
export function breakevenMissRate(m, topo, wl, rate, cs, chunk){
  const st = cs || contextStats(wl);
  const cold = coldRequestSeconds(m, topo, wl, st, chunk);
  // a warm hit's new turn attends over the whole cached context (prior=E[L])
  const warm = prefillContextSeconds(m, topo, liveTurn, 0, st.mean);
  return (1/(rate || REF_REQ_RATE) - warm) / (cold - warm);
}

/* ---- COLD SPIKES (research/spike.md) --------------------------------------
   Everything above prices prefill as a DUTY CYCLE: a mean rate against a mean
   service time. A mean cannot see variance or correlation, and both bite:
   a miss's service time runs as L^2 on a lognormal L (so the queue diverges
   well below f*), and invalidation arrives in CLUMPS, not one request at a
   time. f* turns out to be the miss rate at which burst tolerance reaches
   ZERO — not a place to plan to sit.
   -------------------------------------------------------------------------- */
export const SPIKE_SLA_S = 10;   // TTFT budget B* is quoted against — B* is LINEAR in it
const SPIKE_BURST = 32;   // reference simultaneous-miss spike, for the drain line

// First AND second moments of the prefill service time, per request class.
// The second moment is the new quantity: the Pollaczek-Khinchine wait is
// proportional to it, and here it is dominated by the rare long miss.
export function prefillServiceMoments(m, topo, wl, cs, chunk, mfuAnchor){
  const st = cs || contextStats(wl);
  const C = chunk || PREFILL_CHUNK;
  const a = st.samples, n = a.length;
  const peak = peakFlops(topo), ceil = mfuCeil(m, topo, mfuAnchor);
  const over = prefillOverheadSeconds(m, topo);
  const gemm = 2*m.params_prefill, quad = 2*m.attn_d*m.attn_layers;
  // a hit's cost is AFFINE in the cached length it attends over (the cross
  // term is linear in `prior` chunk by chunk), so two evaluations pin the line
  const TURN = liveTurn;
  const w0 = prefillContextSeconds(m, topo, TURN, C, 0, mfuAnchor);
  const wSlope = (prefillContextSeconds(m, topo, TURN, C, 1e6, mfuAnchor) - w0)/1e6;
  let c1=0, c2=0, h1=0, h2=0;
  for (let i=0; i<n; i++){
    const L = a[i];
    const cold = (gemm*L + quad*L*L)/(peak*ceil) + Math.ceil(L/C)*over;
    const hit  = w0 + wSlope*L;
    c1 += cold; c2 += cold*cold; h1 += hit; h2 += hit*hit;
  }
  return { miss:c1/n, missSq:c2/n, hit:h1/n, hitSq:h2/n };
}

// Queueing + burst metrics for one configuration, at `rate` req/s PER GROUP.
//   rho      the duty cycle, unchanged from the tile above
//   wait     M/G/1 FCFS mean wait (Pollaczek-Khinchine), lam E[S^2]/(2(1-rho))
//   bstar    COLD-SPIKE TOLERANCE: largest simultaneous burst of misses whose
//            LAST request still gets a first token inside SPIKE_SLA_S
//   drain    how long a SPIKE_BURST-miss spike takes to clear
//   fsla     miss rate at which a MISS's mean TTFT reaches the budget. Closed
//            form: E[S^2] and E[S] are both LINEAR in f, so
//              lam (B + f(A-B)) / (2 (1 - lam(D + f(C-D)))) + C = SLA
//            is one linear equation in f. It always binds BELOW f*.
//   allCold  duty if every request missed — what a global cache flush means.
//            Exceeds 1 exactly when f* < 100%, i.e. "this configuration cannot
//            serve its own recovery and must shed load".
export function spikeMetrics(m, topo, wl, cs, rate, chunk){
  const mo = prefillServiceMoments(m, topo, wl, cs, chunk);
  const f = wl.invalidation, lam = rate || REF_REQ_RATE;
  const eS  = f*mo.miss   + (1-f)*mo.hit;
  const eS2 = f*mo.missSq + (1-f)*mo.hitSq;
  const rho = lam*eS;
  const live = rho < 1;
  const A = mo.missSq, B = mo.hitSq, Cc = mo.miss, D = mo.hit;
  const k = 2*(SPIKE_SLA_S - Cc);          // budget left after its own prefill
  const den = lam*(A - B) + k*lam*(Cc - D);
  const fsla = k <= 0 ? 0
             : (den > 0 ? Math.max(0, (k*(1 - lam*D) - lam*B)/den) : Infinity);
  return {
    rho, mo, fsla,
    wait:    live ? lam*eS2/(2*(1-rho)) : Infinity,
    bstar:   live ? Math.max(0, SPIKE_SLA_S*(1-rho)/mo.miss) : 0,
    drain:   live ? SPIKE_BURST*mo.miss/(1-rho) : Infinity,
    allCold: lam*mo.miss,
  };
}

/* ---- THE OPERATING POINT (research/spike.md) -------------------------------
   The study reports its constraints in DIFFERENT UNITS and has always refused
   to combine them — section 8 says so outright. Two assumptions make them
   commensurable, and both are stated on the Concurrent users control rather
   than hidden here:
     1. one user holds one session      (a session count becomes a user count)
     2. a user's MAIN requests come every `think` s — the full open-loop
        interval (measured 43 s on a role-tagged pi-agent trace; the 30 s
        reference is the conservative side). Each main request tows
        `sub_ratio` subagent requests through the prefill server, so the
        arrival rate carries (1 + r) — the same mixture the service moments
        already price. (The Python model also offers a CLOSED conversion,
        operating_point(closed=True); the explorer stays open-loop.)
   Under those, all four ceilings become MAX CONCURRENT USERS and the binding
   one is simply the smallest. Mirrors operating_point() in scenario_model.py.
   -------------------------------------------------------------------------- */
// DECODE_FLOOR_TOKS, DECODE_COMFORT_RATIO and AVG_OUT_TOK live with the study
// constants in the config section: the state object's defaults read them.
export function decodeFloor(){ return state.decode_floor; }
// The SELECTED max_num_batched_tokens (state is a string enum; consumers want
// the number). Everything priced at the deployment's chunk reads this; the
// mfuCeil anchor and the unit checks stay on the PREFILL_CHUNK const.
export function prefillChunk(){ return parseInt(state.chunk, 10); }
export function decodeComfort(){ return state.decode_floor * DECODE_COMFORT_RATIO; }
export function requestRate(users, think){ return users / (think || liveThink); }

// TOTAL arrival rate at the prefill server: each main request tows subR
// subagent requests — the same mixture the service moments price. Every
// queue metric (duty, TTFT, f*, f_sla, B*) must use THIS rate; the plain
// requestRate is the main-agent figure the readouts display.
export function serverRate(users, think, subR){
  return requestRate(users, think) * (1 + (subR || 0));
}

// Concurrency at which per-user p50 falls to `floor` tok/s. Bisection: speed
// is monotone decreasing in n (every extra sequence adds KV bytes to the same
// step), so ~11 evaluations replace a linear scan. Mirrors max_users_decode.
export function maxUsersDecode(model, topo, wl, floor, n_iter, hi){
  const F = floor || DECODE_FLOOR_TOKS;
  // The search cap was sized for the study's 40 tok/s floor. Per-user speed is
  // aggregate/n, so the crossing sits at ~1/F — at a chat-shaped 5 tok/s floor
  // it lands ~8x higher and the old fixed 4,096 censored routinely. Censoring
  // is not just a display '≥': powerDraw prices decodeUsersGroup x floor as the
  // aggregate capacity, so a censored ceiling silently UNDERSTATES capacity and
  // overstates the energy bill. Scaling the cap by 40/F keeps HI at exactly
  // 4,096 for the default floor — no published number moves — and the doubling
  // search only pays for the extra probes it actually needs (iterations are
  // scaled by 128/n, so each probe costs about the same).
  const HI = hi || Math.max(4096, Math.round(4096 * DECODE_FLOOR_TOKS / F));
  // one probe costs O(n_iter x n) — decodeCurves builds a cumulative context
  // sum up to n — so the ORDER of probes dominates the cost. Doubling up from
  // 1 keeps the largest probe within 2x the answer; the old "probe HI first"
  // bisection paid for a 4,096-wide sweep even when the crossing was at 118,
  // which was ~30x of this function and most of a settled render.
  // A probe at concurrency n already averages over n sampled contexts, so its
  // relative MC error falls as n grows — spending the same iteration count at
  // n = 4,096 as at n = 8 buys precision nobody reads. Scaling iterations by
  // 128/n makes every probe cost about the same (n_iter x n stays flat) and
  // keeps the wide end of the search from dominating: the 35B-A3B on 8xH200,
  // whose crossing sits near 2,400 concurrent decoders, went from ~2.9 s to
  // ~0.2 s with no visible change to the answer.
  const base = n_iter || 220;
  const p50 = n => decodeCurves(model, topo, wl, n, Math.max(1,n-1),
                                clip(Math.round(base*128/n), 30, base)
                               ).p50.slice(-1)[0];
  if (!(p50(1) >= F)) return { n: 0, censored: false };
  let lo = 1, up = 2;
  while (up <= HI && p50(up) >= F){ lo = up; up *= 2; }
  // a crossing beyond HI is CENSORED, not "exactly HI" — the study does not
  // do silent caps, so the flag rides along and the readouts print "≥"
  if (up > HI) return { n: HI, censored: true };
  while (up - lo > 1){ const mid = (lo+up)>>1; if (p50(mid) >= F) lo = mid; else up = mid; }
  return { n: lo, censored: false };
}

/* ---- the STEADY-STATE decode point -----------------------------------------
   Every other decode readout on this page is a STRESS test: it asks what one
   user gets when the whole warm population decodes at once. That is the right
   worst case and the wrong expectation. Arrivals are open-loop — a user sends
   one request every `think` seconds and then spends most of that interval
   waiting on a tool or a human — so the number of sequences in the decode
   batch at any instant is far below the warm population.

   Little's law on the DECODE phase alone (a request queueing for prefill, or
   being prefilled, is not yet decoding):

       E[n_decode] = lambda_group x E[time spent decoding]
                   = lambda_group x out / v(n_decode)

   with v(n) the per-user p50 speed at batch n. Multiply through and the fixed
   point reads as a flow balance that needs no inversion:

       n x v(n)  =  lambda_group x out
       ^ delivered output tok/s      ^ demanded output tok/s

   n x v(n) is the aggregate decode curve, INCREASING in n over the region
   this page models (chart D), so a plain bisection finds the crossing. It is
   increasing in EXPECTATION rather than strictly: adding sequence n+1 raises
   the aggregate only while W/kv_bpt + S_n > n x L_{n+1}, so on a heavy-tailed
   prompt distribution a longer-than-mean draw can lower it, and sampled curves
   do show descending segments out on the plateau. Every reference-load point
   sits far to the left of that, where the curve is steep and the inversion is
   well conditioned; in the plateau the root is badly conditioned and the
   bisection may return one of several. If the
   demand exceeds the aggregate at the sweep's widest n, no crossing exists
   INSIDE what this page models and the result is flagged `saturated` — the
   axis is sized to the GPU-resident warm band, so that says the load asks for
   more output than this cache's whole resident population retires. (Aggregate
   throughput does have a true asymptote, at bandwidth / mean-context bytes;
   scenario_model.py searches to n = 4,096 and reaches the same verdict on the
   one configuration where this bites, Mistral-3.5/TP4.)

   Approximations, both stated on the tile:
     - MEAN FIELD. v is evaluated at the MEAN batch, not averaged over the
       batch-size distribution. v(n) is convex, so E[v(N)] >= v(E[N]): the
       readout is the conservative side of Jensen.
     - DECODE ONLY. Prefill chunks sharing a forward pass are priced separately
       (the ITL spike); this is the clean-decode speed between those spikes.
   Mirrors steady_decode_point() in scenario_model.py. */
export function steadyDecodePoint(dc, topo, rateGroup, outTok){
  const reps = topo.replicas || 1;
  const demand = Math.max(0, rateGroup) * Math.max(0, outTok);   // tok/s asked
  const nMax = dc.ns[dc.ns.length-1];
  // the AGGREGATE series, per replica group. dc.agg is already x replicas, and
  // its samples are exactly ns[i] x p50[i] — so inverting the LINEARLY
  // interpolated agg curve (increasing and concave between samples) is both
  // monotone and exactly self-consistent, where inverting n x interp(p50) on a
  // log axis is neither at the wide end of the sweep.
  const aggAt = n => interpAt(dc, 'agg', n) / reps;
  const out = { demand, nMax, saturated:false };
  if (!(demand > 0))                     // no load: nothing is decoding at all
    return { ...out, n:0, pu:interpAt(dc,'p50',1,true), delivered:0, demanded:0 };
  if (demand >= aggAt(nMax))             // beyond the sampled axis — say so
    return { ...out, n:nMax, pu:aggAt(nMax)/nMax,
             delivered:aggAt(nMax)*reps, demanded:demand*reps, saturated:true };
  // below n = 1 the batch is a single sequence whenever anyone is decoding, so
  // v stays v(1) and the fixed point is exact rather than interpolated
  if (demand <= aggAt(1))
    return { ...out, n: demand/aggAt(1), pu: aggAt(1),
             delivered: demand*reps, demanded: demand*reps };
  let lo = 1, hi = nMax;
  for (let i=0;i<60;i++){                 // 60 halvings: exact to ~1e-16 of nMax
    const mid = (lo+hi)/2;
    if (aggAt(mid) < demand) lo = mid; else hi = mid;
  }
  const n = (lo+hi)/2;
  // v = agg/n by construction: at the fixed point the batch delivers exactly
  // what the load demands, so per-user speed is the demand shared out
  return { ...out, n, pu: demand/n, delivered: demand*reps, demanded: demand*reps };
}

// Users at which prefill duty reaches 100% — section 8's f*, in users. Each
// user's (1 + subR) requests per interval all land on the prefill server.
export function maxUsersSaturation(mo, f, think, subR){
  const eS = (f*mo.miss + (1-f)*mo.hit) * (1 + (subR || 0));
  return eS > 0 ? (think||liveThink)/eS : Infinity;
}

// Users at which a MISS's mean TTFT reaches the budget. Closed form in both
// disciplines (E[S] and E[S^2] do not depend on the arrival rate):
//   FCFS  lam a/(2(1-lam b)) + c = SLA  ->  lam = k/(a + k b),  k = 2(SLA - c)
//   PS    c/(1-lam b) = SLA            ->  lam = (1 - c/SLA)/b
// ALWAYS strictly inside saturation, since k/(a+kb) < 1/b for any a > 0 — the
// algebraic form of "the queue diverges before the server does".
export function maxUsersLatency(mo, f, sla, think, discipline, subR){
  const b = f*mo.miss + (1-f)*mo.hit;
  const a = f*mo.missSq + (1-f)*mo.hitSq, c = mo.miss;
  if (c >= sla || b <= 0) return 0;
  const lam = discipline === 'ps' ? (1 - c/sla)/b : (2*(sla-c))/(a + 2*(sla-c)*b);
  // lam is the TOTAL rate (the moments mix both classes); a user contributes
  // (1 + subR) requests per interval, so the user count divides it back out
  return Math.max(0, lam*(think||liveThink)/(1 + (subR || 0)));
}

// All four ceilings in one unit, plus which binds. `warmUsers` is passed in
// because the caller has already paid for the Monte-Carlo warm fill.
export function operatingPoint(model, topo, wl, cs, opts){
  const o = opts || {};
  const sla = o.sla ?? state.sla, think = o.think ?? state.think;
  const mo = o.mo || prefillServiceMoments(model, topo, wl, cs, o.chunk);
  const f = wl.invalidation;
  // UNITS: every ceiling here is SYSTEM-wide concurrent users, because that is
  // what the Concurrent users control means (its load is split across replicas
  // at groupRate = rate / replicas). cache and decode arrive already scaled by
  // the caller; latency and saturation come out of the closed forms PER GROUP,
  // so they are scaled here. Mixing the two conventions silently mislabelled
  // which constraint binds on every DP row. scenario_model.py keeps all four
  // per group instead — a different, also-consistent choice, documented in its
  // operating_point() docstring; on TP topologies (replicas = 1) they agree.
  const reps = o.reps || 1;
  const ceilings = {
    cache:      o.warmUsers,
    decode:     o.decodeUsers,
    latency:    reps * maxUsersLatency(mo, f, sla, think, o.discipline,
                                       wl.sub_ratio),
    saturation: reps * maxUsersSaturation(mo, f, think, wl.sub_ratio),
  };
  let binding = 'cache';
  for (const k in ceilings) if (ceilings[k] < ceilings[binding]) binding = k;
  const limit = ceilings[binding];
  const users = o.users ?? state.users;
  return { ceilings, binding, limit, users,
           reqRate: requestRate(users, think),
           headroom: limit > 0 ? users/limit : Infinity,
           fits: users <= limit };
}
