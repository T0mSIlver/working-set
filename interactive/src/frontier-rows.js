/* The frontier table's rows, without the DOM: every model × DP×TP split on
   the selected GPU, each priced at your load (op), at its own ceiling (€/seat)
   and on its own warm pool. render.js caches and time-slices these steps for
   the explorer; computeFrontierRows() runs them in one go for any page that
   needs the whole set, such as the Compare page. */
import { CONFIG, kv_pool_tokens, makeGrid } from './config.js';
import { capDecodeUsers, contextStats, decodePowerUsers, decodeSlotMean, decodeSlotUsers,
         maxUsersDecode, missShareAt, operatingPoint, p50AtCap, prefillServiceMoments,
         recommendedMns, serverRate, SPIKE_SLA_S, decodeFloor } from './prefill.js';
import { samplingSig, seedFor } from './mathlib.js';
import { p_sub } from './workload.js';
import { warmCapacity } from './capacity.js';
import { energyCost } from './cost.js';
import { ramPerCache, state } from './state.js';
import { modelForCompare } from './models.js';

export function warmUsersNow(p5, wl){ return p5 * (1 - p_sub(wl)); }

// B* from a moments bundle — the same arithmetic spikeMetrics does, factored
// out so the MFU bracket and the f sweep can reuse it without re-sampling.
export function bStar(mo, f, sla, rate){
  const rho = rate * (f*mo.miss + (1-f)*mo.hit);
  return rho >= 1 ? 0 : Math.max(0, sla*(1-rho)/mo.miss);
}

// The rows to price: every model on every split of up to 8 GPUs whose pool
// holds at least one token after weights.
export function frontierPlan(){
  const plan = [];
  for (const mk of Object.keys(CONFIG.MODELS))
    for (const [dp, tp] of [[1,1],[1,2],[1,4],[1,8],[2,1],[2,2]]){
      if (dp*tp > 8) continue;
      const m2 = modelForCompare(mk), t2 = makeGrid(dp, tp, state.gpu, state.kvshard);
      if (kv_pool_tokens(m2, t2) <= 0) continue;
      plan.push({ mk, dp, tp, key: `${mk}|${dp}x${tp}`, m2, t2, ram: ramPerCache(t2) });
    }
  return plan;
}

// The warm half of a row: how many sessions its pool keeps warm.
export function frontierWarm(p, wl, q, ssig){
  const m2 = p.m2, t2 = p.t2, reps = t2.replicas || 1;
  // Seed PER ROW, so a row's fill depends only on that row (and slicing
  // cannot move a draw). One stream shared across the table made every
  // row's draws depend on how many the earlier rows consumed.
  seedFor(`frontier|${p.mk}|${p.dp}x${p.tp}`, ssig);
  const wc2 = warmCapacity(m2, t2, wl, p.ram,
                           Math.max(60, q.WARM_ITER/6), q.WARM_BUDGET_SCAN);
  return { key: p.key, mk: p.mk, dp: p.dp, tp: p.tp,
           label: `${m2.name} · ${t2.name}`, reps,
           warmUsers: warmUsersNow(wc2.all[0], wl)*reps,
           resident5: wc2.gpu[0] };
}

// The decode half: the decode ceiling, the batch cap and the per-user speed
// at it.
export function frontierDecode(p, res5, wl, floor0, mns0, ssig){
  const m2 = p.m2, t2 = p.t2;
  seedFor(`frontierDec|${p.key}`, ssig);
  const raw = maxUsersDecode(m2, t2, wl, floor0, 140);
  const mns = mns0 === null ? recommendedMns(raw.n, res5) : mns0;
  seedFor(`frontierDecPow|${p.key}`, ssig);
  const pu = p50AtCap(m2, t2, wl, mns, 140);
  seedFor(`frontierDecSlot|${p.key}`, ssig);
  const slotPu = p50AtCap(m2, t2, wl, Math.max(1, Math.ceil(decodeSlotMean(mns))), 140);
  // the cap is applied at assembly: where the steady batch reaches it
  // moves with think time and output length, which this cache is not
  // keyed on
  return { raw, pu, slotPu, mns, floor: floor0 };
}

// €/seat/month: the bill with the configuration carrying its max users,
// divided by them. Infinity where there is no ceiling to fill. Its prefill
// duty is priced at the miss share at that load, which past the warm pool
// is more than f.
function seatPrice(op, topo, mo, f, wl, r){
  if (!isFinite(op.limit) || op.limit < 1) return Infinity;
  const rateMax = serverRate(op.limit, state.think, wl.sub_ratio) / r.reps;
  const fMax = missShareAt(op.limit / r.reps, r.warmUsers / r.reps, f);
  return energyCost(topo, mo, fMax, rateMax, (r.decodePower ?? r.decodeUsers) / r.reps).totalMonth / op.limit;
}

// One row from its warm half r0, decode half d and prefill moments mo2, at
// your load. csSteps is the shared context draw for the slowed-generation
// steps.
export function frontierAssemble(r0, d, mo2, wl, csSteps){
  const f = wl.invalidation;
  const dcap = capDecodeUsers(d.raw, d.mns,
                              decodeSlotUsers(d.slotPu, d.mns, state.think, state.out, wl.sub_ratio));
  const r = { ...r0, decodeUsers: dcap.n*r0.reps, censored: dcap.censored, capped: dcap.capped,
              decodePower: decodePowerUsers(dcap, d.mns, d.pu, d.floor)*r0.reps };
  const m2 = modelForCompare(r.mk), t2 = makeGrid(r.dp, r.tp, state.gpu, state.kvshard);
  const r2 = serverRate(state.users, state.think, wl.sub_ratio)/r.reps;
  const op = operatingPoint(m2, t2, wl, csSteps, {
             mo: mo2, reps: r.reps,
             warmUsers: r.warmUsers, decodeUsers: r.decodeUsers });
  // each row at the share that misses with YOUR load on ITS pool
  return { ...r, op,
           bstar: bStar(mo2, op.missShare, SPIKE_SLA_S, r2),
           // the whole bill at YOUR load — hardware plus energy
           // (research/power.md) — comparable across rows because every
           // row is priced at the same demand and the same €/GPU-hour
           eur: energyCost(t2, mo2, op.missShare, r2, (r.decodePower ?? r.decodeUsers) / r.reps).totalMonth,
           // and the bill at the row's OWN ceiling, per seat: what one
           // user costs when the configuration is full. At your load the
           // bill is the GPU count and every row on a topology prices the
           // same; at capacity a row that carries twice the users halves
           // the seat. Chart H's y-axis (research/power.md for the terms)
           eurSeat: seatPrice(op, t2, mo2, f, wl, r) };
}

// Every row for the current state in one synchronous pass, sorted by max
// users. Same per-row seeds and steps as the explorer's sliced rebuild; only
// the shared context draw is seeded here (the explorer's rides the page's RNG
// stream), so figures can sit within sampling noise of the explorer's table.
export function computeFrontierRows(wl, q = CONFIG){
  const ssig = samplingSig(), floor0 = decodeFloor(), mns0 = state.mns;
  seedFor('frontierRows|ctx', ssig);
  const cs = contextStats(wl), csSteps = cs;
  return frontierPlan().map(p => {
    const r0 = frontierWarm(p, wl, q, ssig);
    const d = frontierDecode(p, r0.resident5, wl, floor0, mns0, ssig);
    return frontierAssemble(r0, d, prefillServiceMoments(p.m2, p.t2, wl, cs), wl, csSteps);
  }).sort((a,b)=>b.op.limit-a.op.limit);
}
