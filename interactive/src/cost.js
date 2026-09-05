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
   the load implies against the decode capacity at the 40 tok/s floor, capped
   at whatever prefill leaves. P_total = n_gpu x (P_gpu + host) x PUE.
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
// € figures on top of the draw. 720 h/month. eurMonth is the ELECTRICITY
// alone (the power model's own output); hwMonth is GPU-hours at the rental
// rate; totalMonth their sum, and the per-user / per-token figures divide the
// TOTAL — a €/Mtok that priced the watts but not the silicon would read an
// order of magnitude too cheap.
export function energyCost(topo, mo, f, rateGroup, decodeUsersGroup){
  const p = powerDraw(topo, mo, f, rateGroup, decodeUsersGroup);
  const eurMonth = p.kw * 720 * state.ekwh;
  const hwMonth = topo.n_gpu * gpuHourPrice(topo.gpu || CONFIG.GPUS["H200"]) * 720;
  const totalMonth = eurMonth + hwMonth;
  const outTokS = rateGroup * (topo.replicas || 1) * state.out;
  const eurMtok = outTokS > 0 ? (totalMonth / 720 / 3600) / outTokS * 1e6 : Infinity;
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
  // the idle floor keeps the hardware line: a rented GPU bills whether or
  // not it decodes, and an owned one depreciates just the same
  const idleMonth = topo.n_gpu * (g.idle_w + g.host_w) * c.pue / 1000 * 720 * state.ekwh + c.hwMonth;
  const bill = [
    ['Hardware / month', `€${fmt(c.hwMonth, 0)} <span style="color:${cssv('--muted')};font-weight:400">(${topo.n_gpu} × €${fmt(gpuHourPrice(g), 2)}/h)</span>`],
    ['Electricity / month', `€${fmt(c.eurMonth, 0)}`],
    ['Total / month', `€${fmt(c.totalMonth, 0)}`],
    ['Per user / month', `€${fmt(c.eurUser, 2)}`],
    ['Per 1M output tokens', isFinite(c.eurMtok) ? `€${fmt(c.eurMtok, 2)}` : '—'],
    ['At idle (floor)', `€${fmt(idleMonth, 0)} /mo`],
  ].map(([k, v]) => `<dt>${k}</dt><dd class="tnum">${v}</dd>`).join('');
  box.innerHTML =
    `<div class="deploy">`
    + `<div><h4>Draw</h4><dl class="spec">${draw}</dl></div>`
    + `<div><h4>Bill</h4><dl class="spec">${bill}</dl></div>`
    + `<p class="verdictline" style="color:${cssv('--muted')};font-weight:400">`
    + `Output assumed at ${fmt(state.out, 0)} tokens per request (unmeasured). `
    + `Per-user and per-token figures divide the total. Two seemingly equal configurations rarely bill equally: hardware and idle watts scale with GPU count, decode watts with the load.</p>`
    + `</div>`;
}
