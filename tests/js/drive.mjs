/* Drive the explorer's JS mirror from a golden-vector state.
   One exported function, driveState(state) -> {quantity: number}, whose keys
   match scripts/golden.py's `out` block exactly.

   The state is assigned onto interactive/src/state.js's live `state` object and
   the mirror is entered through the same doors the page uses:
   render.js activeModel(), state.js currentTopo()/currentWL(), and the seeded
   RNG (mathlib.js seedFor) that computeAndRender() calls before each sampled
   section.

   FOUR quantities are DUPLICATED here rather than called, because the
   explorer computes them somewhere this module cannot reach. Each is marked
   DUPLICATED at its site, named as such in vectors.json's `mapping`, and
   listed in tests/golden/README.md under coverage limitations. What that
   costs: for these four the test pins the MODEL, not the explorer's own line
   of code — an edit to the render.js expression would not be caught.

     ttft_miss_fcfs / ttft_hit_fcfs / ttft_hit_ps   inline in render.js's
         Object.assign onto the operating point (~line 298)
     max_users_cache                                render.js warmUsersNow,
         inline (~line 249)
     mean_passes                                    prefill.js meanPasses is
         module-private

   Moving those three render.js expressions into prefill.js would close it, and
   is a follow-up rather than part of this change: it touches the render path,
   which AGENTS.md requires be verified byte-identical in a headless browser.

   Everything else comes out of a function the page itself calls. */
import { CONFIG, effective_bw, kv_pool_tokens, tpEff, w_decode }
  from '../../interactive/src/config.js';
import { state, currentTopo, currentWL, ramPerCache }
  from '../../interactive/src/state.js';
import { activeModel } from '../../interactive/src/render.js';
import { seedFor } from '../../interactive/src/mathlib.js';
import { p_sub } from '../../interactive/src/workload.js';
import { decodeCurves, decodePlan, warmCapacity }
  from '../../interactive/src/capacity.js';
import { bStar, warmUsersCurve } from '../../interactive/src/planner.js';
import { energyCost } from '../../interactive/src/cost.js';
import {
  breakevenMissRate, coldRequestSeconds, contextStats, maxUsersDecode,
  maxUsersLatency, maxUsersSaturation, mfuCeil, mfuEff, missContextSeconds,
  peakFlops, prefillContextSeconds, prefillFlops, prefillOverheadSeconds,
  prefillSeconds, prefillServiceMoments, serverRate, setLiveThink, setLiveTurn,
  spikeMetrics, steadyDecodePoint,
} from '../../interactive/src/prefill.js';

// the three lengths and the one context golden.py prices every state at
const PREFILL_LENGTHS = [2048, 32768, 180000];
const CONTEXT_LEN = 120000;
// CONFIG.WARM_ITER / DECODE_ITER: the explorer's own full-quality sample
// counts, i.e. what a settled render actually spends.
const WARM_ITER = CONFIG.WARM_ITER, DECODE_ITER = CONFIG.DECODE_ITER;

export function driveState(v){
  // 1. install the state, exactly as a share link + enforceConstraints would
  for (const [k, val] of Object.entries(v)) state[k] = val;
  // syncState() in render.js: the prefill module's two live mirrors of the
  // sliders it is declared above
  setLiveTurn(state.turn); setLiveThink(state.think);

  const m = activeModel(), topo = currentTopo(), wl = currentWL();
  const chunk = parseInt(state.chunk, 10);
  const reps = topo.replicas || 1;
  const rate = serverRate(state.users, state.think, wl.sub_ratio) / reps;
  const o = {};

  // ---- exact ---------------------------------------------------------
  o.kv_pool_tokens = kv_pool_tokens(m, topo);
  o.effective_bw = effective_bw(topo);
  o.peak_flops = peakFlops(topo);
  o.tp_efficiency = tpEff(topo.tp, topo.gpu.nvlink_domain);
  o.w_decode_n64 = w_decode(m, 64);
  o.prefill_overhead_seconds = prefillOverheadSeconds(m, topo);
  o.mfu_ceiling = mfuCeil(m, topo);
  o.mfu_effective_at_chunk = mfuEff(m, topo, chunk);
  for (const L of PREFILL_LENGTHS){
    o[`prefill_flops_${L}`] = prefillFlops(m, L, 0);
    o[`prefill_seconds_${L}`] = prefillSeconds(m, topo, L, 0);
  }
  o.prefill_context_seconds_120k =
    prefillContextSeconds(m, topo, CONTEXT_LEN, chunk, 0);
  o.miss_context_seconds_120k =
    missContextSeconds(m, topo, CONTEXT_LEN, chunk, 0);
  o.warm_pass_seconds_turn =
    prefillContextSeconds(m, topo, state.turn, chunk, 0);

  // ---- the shared context draw --------------------------------------
  // render.js draws this once per render under seedFor('context') and hands
  // the same object to every prefill readout, so the tiles cannot disagree
  // with the charts within a frame.
  seedFor('context');
  const cs = contextStats(wl);
  o.ctx_mean = cs.mean;
  o.ctx_mean_sq = cs.meanSq;
  // DUPLICATED: meanPasses is module-private in prefill.js. E[ceil(L/C)] over
  // the same cs.samples draw is exactly what it computes.
  let passes = 0;
  for (let i = 0; i < cs.samples.length; i++) passes += Math.ceil(cs.samples[i]/chunk);
  o.mean_passes = passes / cs.samples.length;

  o.cold_request_seconds = coldRequestSeconds(m, topo, wl, cs, chunk);
  o.warm_request_seconds =
    prefillContextSeconds(m, topo, state.turn, chunk, cs.mean);

  const mo = prefillServiceMoments(m, topo, wl, cs, chunk);
  o.moments_miss = mo.miss;
  o.moments_miss_sq = mo.missSq;
  o.moments_hit = mo.hit;
  o.moments_hit_sq = mo.hitSq;

  const sp = spikeMetrics(m, topo, wl, cs, rate, chunk);
  const f = wl.invalidation;
  const eS2 = f*mo.missSq + (1-f)*mo.hitSq;
  o.prefill_duty = sp.rho;
  o.queue_wait_seconds = sp.wait;
  // spikeMetrics computes fsla against the same hard-wired SPIKE_SLA_S = 10
  o.sla_miss_rate_sla10 = sp.fsla;
  // spikeMetrics hard-wires SPIKE_BURST = 32; the vectors price Python's
  // burst_drain_seconds at the same 32 so the two are comparable at all
  o.burst_drain_seconds_b32 = sp.drain;
  // DUPLICATED: render.js stamps these three onto the operating point inline
  // (renderPlanner's Object.assign), so there is no function to call.
  o.ttft_miss_fcfs = sp.rho >= 1 ? Infinity
    : rate*eS2/(2*(1-sp.rho)) + mo.miss;
  o.ttft_hit_fcfs = sp.rho >= 1 ? Infinity
    : rate*eS2/(2*(1-sp.rho)) + mo.hit;
  o.ttft_hit_ps = sp.rho >= 1 ? Infinity : mo.hit/(1-sp.rho);
  o.breakeven_miss_rate = breakevenMissRate(m, topo, wl, rate, cs, chunk);
  o.spike_tolerance = bStar(mo, f, state.sla, rate);
  o.spike_tolerance_sla10 = sp.bstar;
  o.max_users_latency = maxUsersLatency(mo, f, state.sla, state.think,
                                        'fcfs', wl.sub_ratio);
  o.max_users_saturation = maxUsersSaturation(mo, f, state.think, wl.sub_ratio);

  // ---- warm fill -----------------------------------------------------
  seedFor('warm');
  const wc = warmCapacity(m, topo, wl, ramPerCache(topo), WARM_ITER,
                          CONFIG.WARM_BUDGET);
  o.warm_p5_all = wc.all[0];
  // DUPLICATED: render.js computes warmUsersNow inline. It is also a standing
  // APPROXIMATION either way — the explorer scales the whole warm p5 by the
  // non-subagent share, where Python counts user-class sessions inside each
  // fill (warm_capacity which="user").
  o.max_users_cache = wc.all[0] * (1 - p_sub(wl));

  // ---- decode --------------------------------------------------------
  seedFor('decodeCeil');
  const dec = maxUsersDecode(m, topo, wl, state.decode_floor, DECODE_ITER);
  o.max_users_decode = dec.n;
  o.max_users_decode_censored = dec.censored;

  seedFor('decode');
  const dcPts = decodeCurves(m, topo, wl, 64, 63, DECODE_ITER);
  for (const n of [1, 8, 64]){
    const i = dcPts.ns.indexOf(n);
    o[`decode_p50_n${n}`] = i >= 0 ? dcPts.p50[i] : NaN;
  }

  // steadyDecodePoint inverts the sweep the page actually drew, so the sweep
  // has to be the page's: decodePlan sizes it from the GPU-resident warm band.
  const plan = decodePlan(wc.gpu[2], false);
  seedFor('decode');
  const dc = decodeCurves(m, topo, wl, plan.nMax, plan.step, plan.iter);
  const sd = steadyDecodePoint(dc, topo, rate, state.out);
  o.steady_n = sd.n;
  o.steady_per_user_tok_s = sd.pu;
  o.steady_saturated = sd.saturated;

  // ---- power and the bill --------------------------------------------
  const e = energyCost(topo, mo, f, rate, dec.n);
  o.power_d_p = e.dP;
  o.power_d_d = e.dD;
  o.power_per_gpu_w = e.perGpu;
  o.power_kw = e.kw;
  o.energy_eur_month = e.eurMonth;
  o.energy_hw_month = e.hwMonth;
  o.energy_total_month = e.totalMonth;
  o.energy_eur_user = e.eurUser;
  o.energy_eur_mtok = e.eurMtok;
  return o;
}

// warmUsersCurve is the frontier/chart-G sampler; the golden set does not
// price a whole f sweep, but the test exercises it at the state's own f so a
// mirror change there cannot pass unseen.
export function warmUsersAt(v){
  for (const [k, val] of Object.entries(v)) state[k] = val;
  const m = activeModel(), topo = currentTopo(), wl = currentWL();
  seedFor('warmCurve');
  const fn = warmUsersCurve(m, topo, ramPerCache(topo),
                            Math.max(80, CONFIG.WARM_ITER/3),
                            CONFIG.WARM_BUDGET_SCAN, wl, wl.invalidation,
                            undefined);
  return fn(wl.invalidation);
}
