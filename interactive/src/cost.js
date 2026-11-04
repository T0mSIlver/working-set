import { CONFIG } from './config.js';
import { decodeFloor, serverRate } from './prefill.js';
import { state } from './state.js';
import { cssv, fmt } from './svg.js';

/* ---- THE ELECTRICITY BILL (research/power.md) -----------------------------
   Wall power from the duty cycle the model already computes. Three GPU
   states, priced per research/power.md's measured anchors:
     prefill   d_p = the prefill duty rho — compute-bound, power-cap-limited
               (~0.90 x TDP, FLAT across the MFU band: the cap binds first)
     decode    d_d — bandwidth-bound, well under TDP (~0.55 x, the softest
               constant, band 0.45–0.75)
     idle      the remainder, at warm-idle watts
   d_p and d_d PARTITION time (no double counting): d_d is the decode demand
   the load implies against the decode capacity at decodeFloor() — 40 tok/s by
   default, and the decode-floor slider moves it, on this side and on
   model.power_draw's `decode_floor` alike — capped at whatever prefill
   leaves. P_total = n_gpu x (P_gpu + host) x PUE.
   -------------------------------------------------------------------------- */
// AVG_OUT_TOK (declared with the decode constants) is the REFERENCE output
// length; the live figure is state.out, which the Output-per-response slider
// sets. Scales d_d linearly; the €/1M-token figure moves hyperbolically (fixed
// idle/prefill watts amortise as outputs lengthen); the €/month bill moves
// least (decode is one term of three).
//
// Average draw of one GPU and the whole system, at the current load.
// rateGroup = TOTAL req/s one replica group sees; decodeUsersGroup = the
// per-group decode ceiling at the floor (its capacity proxy).
function powerDraw(topo, mo, f, rateGroup, decodeUsersGroup){
  const g = topo.gpu || CONFIG.GPUS["H200"];
  const eS = f * mo.miss + (1 - f) * mo.hit;
  const dP = Math.min(1, rateGroup * eS);
  const demand = rateGroup * state.out;                         // tok/s asked
  const cap = decodeUsersGroup * decodeFloor();                 // tok/s at floor
  const dD = Math.min(Math.max(0, 1 - dP), cap > 0 ? demand / cap : 1);
  const perGpu = dP * g.p_prefill_w + dD * g.p_decode_w
               + Math.max(0, 1 - dP - dD) * g.idle_w;
  const pue = parseFloat(state.pue) || 1.5;
  const kw = topo.n_gpu * (perGpu + g.host_w) * pue / 1000;
  return { dP, dD, perGpu, kw, pue };
}
// €/GPU-hour of any part at the current slider. The slider names the price of
// the SELECTED part; the other part keeps its own list price scaled by the
// same ratio. Today every frontier row is on the selected part (ratio 1), so
// the scaling only matters if the frontier ever spans both parts.
function gpuHourPrice(g){
  return g.eur_gpu_h * state.gpuh / CONFIG.GPUS[state.gpu].eur_gpu_h;
}
// Months an owned server is written off over (model.AMORT_MONTHS_DEFAULT)
export const AMORT_MONTHS = 36;
// The monthly bill, power counted once (model.energy_cost). state.pay:
//   'rent'  the GPU-hour price x n_gpu x state.work: GPUs up, and billed,
//           only the hours the team works; power is in the rent, so no
//           electricity line (eurMonth 0).
//   'own'   the purchase price x n_gpu / AMORT_MONTHS, plus electricity at
//           the meter: the load's draw for state.work hours, the idle draw
//           for the rest of the 720-hour month.
// The per-user and per-token figures divide the TOTAL; the token rate is
// the load's, over the hours it runs.
export function energyCost(topo, mo, f, rateGroup, decodeUsersGroup){
  const p = powerDraw(topo, mo, f, rateGroup, decodeUsersGroup);
  const g = topo.gpu || CONFIG.GPUS["H200"], work = state.work;
  let eurMonth, hwMonth;
  if (state.pay === 'own'){
    const kwIdle = topo.n_gpu * (g.idle_w + g.host_w) * p.pue / 1000;
    eurMonth = (p.kw * work + kwIdle * (720 - work)) * state.ekwh;
    hwMonth = topo.n_gpu * g.eur_capex_gpu / AMORT_MONTHS;
  } else {
    eurMonth = 0;
    hwMonth = topo.n_gpu * gpuHourPrice(g) * work;
  }
  const totalMonth = eurMonth + hwMonth;
  const outTokS = rateGroup * (topo.replicas || 1) * state.out;
  const eurMtok = outTokS > 0 ? (totalMonth / work / 3600) / outTokS * 1e6 : Infinity;
  return { ...p, eurMonth, hwMonth, totalMonth,
           eurUser: totalMonth / Math.max(1, state.users), eurMtok };
}

export function renderCostCard(op, model, topo, wl, mo, decodeUsersGroup){
  const box = document.getElementById('costBody'); if (!box) return;
  if (!op){
    box.innerHTML = '<p class="cs">model weights do not fit this configuration — no deployment, no bill.</p>';
    return;
  }
  const reps = topo.replicas || 1;
  const rate = serverRate(state.users, state.think, wl.sub_ratio) / reps;
  const c = energyCost(topo, mo, wl.invalidation, rate, decodeUsersGroup);
  const g = topo.gpu || CONFIG.GPUS["H200"];
  const idleFrac = Math.max(0, 1 - c.dP - c.dD);
  const draw = [
    ['Average draw', `${fmt(c.kw, 2)} kW`],
    ['Per GPU (of TDP)', `${fmt(c.perGpu, 0)} W (${fmt(c.perGpu / g.tdp_w * 100, 0)}%)`],
    ['Time split', `${fmt(c.dP * 100, 0)}% prefill · ${fmt(c.dD * 100, 0)}% decode · ${fmt(idleFrac * 100, 0)}% idle`],
    ['Host + PUE', `${fmt(g.host_w, 0)} W/GPU · ×${c.pue}`],
  ].map(([k, v]) => `<dt>${k}</dt><dd class="tnum">${v}</dd>`).join('');
  const muted = cssv('--muted');
  const hw = state.pay === 'own'
    ? ['Hardware write-off / month', `€${fmt(c.hwMonth, 0)} <span style="color:${muted};font-weight:400">(${topo.n_gpu} × €${fmt(g.eur_capex_gpu, 0)} ÷ ${AMORT_MONTHS} months)</span>`]
    : ['Rent / month', `€${fmt(c.hwMonth, 0)} <span style="color:${muted};font-weight:400">(${topo.n_gpu} × €${fmt(gpuHourPrice(g), 2)}/h × ${fmt(state.work, 0)} h)</span>`];
  const bill = [
    hw,
    ['Electricity / month', state.pay === 'own' ? `€${fmt(c.eurMonth, 0)}` : `<span style="color:${muted};font-weight:400">in the rent</span>`],
    ['Total / month', `€${fmt(c.totalMonth, 0)}`],
    ['Per user / month', `€${fmt(c.eurUser, 2)}`],
    ['Per 1M output tokens', isFinite(c.eurMtok) ? `€${fmt(c.eurMtok, 2)}` : '—'],
  ].map(([k, v]) => `<dt>${k}</dt><dd class="tnum">${v}</dd>`).join('');
  box.innerHTML =
    `<div class="deploy">`
    + `<div><h4>Draw</h4><dl class="spec">${draw}</dl></div>`
    + `<div><h4>Bill</h4><dl class="spec">${bill}</dl></div>`
    + `</div>`;
}
