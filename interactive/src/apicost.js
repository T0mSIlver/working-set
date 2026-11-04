import { API_PRICES } from './api_prices.js';

/* ============================================================================
   API AND PLAN SEATS  (mirrors src/workingset/api_cost.py)
   What one agentic-coding seat costs on a metered API or a coding plan, from
   the same workload the self-host rows price. Closed form in the normal CDF,
   the same polynomial as the Python side, so the golden vectors agree to
   rounding. research/api_prices.md for the derivation and every price.
   ========================================================================== */
export const ACTIVE_HOURS_DEFAULT = 84;
export const WORKDAYS_PER_MONTH = 21;
export const WEEKS_PER_MONTH = 30 / 7;
const REQ_PER_TURN = 7.8;                 // model.MEASURED_REQ_PER_TURN
const TOOL_GAP_MEDIAN_S = 0.61, TOOL_GAP_SIGMA = 2.43;
const HUMAN_GAP_MEDIAN_S = 58, HUMAN_GAP_SIGMA = Math.sqrt(2 * Math.log(275 / 58));

// Numerical Recipes' erfcc: the Python side runs the same polynomial
function erfc(x){
  const z = Math.abs(x), t = 1 / (1 + 0.5 * z);
  const ans = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196
            + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398
            + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? ans : 2 - ans;
}
export function phi(x){
  if (x === Infinity) return 1;
  if (x === -Infinity) return 0;
  return 0.5 * erfc(-x / Math.SQRT2);
}
function gapExceeds(ttl, median, sigma){
  return ttl <= 0 ? 1 : 1 - phi((Math.log(ttl) - Math.log(median)) / sigma);
}
export function expiryMiss(ttl, reqPerTurn = REQ_PER_TURN){
  return ((reqPerTurn - 1) * gapExceeds(ttl, TOOL_GAP_MEDIAN_S, TOOL_GAP_SIGMA)
          + gapExceeds(ttl, HUMAN_GAP_MEDIAN_S, HUMAN_GAP_SIGMA)) / reqPerTurn;
}
// L = clip(X, lo, hi), X lognormal(median, sigma): [P(a < L <= b), E[L; a < L <= b]]
export function clippedPartial(median, sigma, lo, hi, a, b){
  if (lo > hi) lo = hi;
  if (b <= a) return [0, 0];
  const mu = Math.log(median), meanX = median * Math.exp(sigma * sigma / 2);
  const z = t => t <= 0 ? -Infinity : t === Infinity ? Infinity : (Math.log(t) - mu) / sigma;
  let p = 0, e = 0;
  if (a < lo && lo <= b){ const w = phi(z(lo)); p += w; e += lo * w; }
  const u = Math.max(a, lo), v = Math.min(b, hi);
  if (v > u){
    p += phi(z(v)) - phi(z(u));
    e += meanX * (phi(z(v) - sigma) - phi(z(u) - sigma));
  }
  if (a < hi && hi <= b && hi > lo){ const w = 1 - phi(z(hi)); p += w; e += hi * w; }
  return [p, e];
}

// ---- the catalogue ----
function priceScale(a){
  const b = a.time_blend;
  return b ? b.peak_share + (1 - b.peak_share) * b.offpeak_factor : 1;
}
function band(d, k){
  return { above: d.above_tokens || 0, input: d.input * k, output: d.output * k,
           cache: d.cache.map(c => ({ ttl: c.ttl_s, read: c.read * k, write: c.write * k })) };
}
export const API_OFFERS = Object.fromEntries(API_PRICES.apis.map(a => {
  const k = priceScale(a);
  return [a.key, { ...a, priced: a.bands.map(b => band(b, k)).sort((x, y) => x.above - y.above),
                   tb40: a.quality ? a.quality.tb40 : null }];
}));
// a plan's meter prices one request in its own unit (vendor credits, model
// requests); none means the API offer's own list prices, in dollars
export const PLANS = Object.fromEntries((API_PRICES.plans || []).map(p => {
  const m = p.meter, k = m ? priceScale(m) : 1;
  return [p.key, { ...p, meterOffer: m ? { priced: m.bands.map(b => band(b, k)).sort((x, y) => x.above - y.above) } : null,
                   perRequest: m ? (m.per_request || 0) : 0, unit: m ? m.unit : 'usd' }];
}));
export const USD_PER_EUR = API_PRICES.fx.usd_per_eur;

function classCost(offer, k, median, sigma, prefix, cap, delta, q, out){
  const lo = Math.min(prefix, cap), bands = offer.priced;
  let total = 0;
  for (let i = 0; i < bands.length; i++){
    const top = i + 1 < bands.length ? bands[i + 1].above : Infinity;
    const [p, e] = clippedPartial(median, sigma, lo, cap, i ? bands[i].above : -Infinity, top);
    if (p <= 0) continue;
    const o = bands[i].cache[k], d = Math.min(delta, lo);
    const readTok = (1 - q) * (e - d * p), writeTok = (1 - q) * d * p + q * e;
    total += (o.read * readTok + o.write * writeTok + bands[i].output * out * p) / 1e6;
  }
  return total;
}
// mean $ per request, main and subagent mixed, at the cheapest cache option
export function requestCost(offer, wl, turn, out){
  const s = wl.sub_ratio, subPrefix = wl.sub_shares_prefix ? wl.sys_user : wl.sys_sub;
  let best = null;
  offer.priced[0].cache.forEach((o, k) => {
    const q = Math.min(1, wl.invalidation + (1 - wl.invalidation) * expiryMiss(o.ttl));
    const main = classCost(offer, k, wl.user_median, wl.user_sigma, wl.sys_user, wl.cap, turn, q, out);
    const sub = classCost(offer, k, wl.sub_median, wl.sub_sigma, subPrefix, wl.cap, turn, q, out);
    const usd = (main + s * sub) / (1 + s);
    if (best === null || usd < best.usd - 1e-15) best = { usd, ttl: o.ttl, miss: q, discounted: o.read < offer.priced[0].input };
  });
  const lm = clippedPartial(wl.user_median, wl.user_sigma, Math.min(wl.sys_user, wl.cap), wl.cap, -Infinity, Infinity)[1];
  const ls = clippedPartial(wl.sub_median, wl.sub_sigma, Math.min(subPrefix, wl.cap), wl.cap, -Infinity, Infinity)[1];
  const lbar = (lm + s * ls) / (1 + s);
  return { usd: best.usd, ttl: best.ttl, miss: best.miss,
           hitRate: best.discounted ? (1 - best.miss) * Math.max(0, 1 - Math.min(turn, lbar) / lbar) : 0 };
}
export const requestsPerHour = (wl, think) => (1 + wl.sub_ratio) / think * 3600;

// opts: { think, turn, out, hours, active, spu }
export function apiSeatCost(offer, wl, o){
  const rc = requestCost(offer, wl, o.turn, o.out);
  const n = requestsPerHour(wl, o.think) * o.hours * o.spu;
  const eur = n * rc.usd / USD_PER_EUR;
  return { key: offer.key, name: offer.name, kind: 'api', eurSeat: eur, eurSession: eur / (o.active * o.spu),
           usdRequest: rc.usd, requestsMonth: n, ttl: rc.ttl, hitRate: rc.hitRate,
           fits: true, plansPerSeat: 1, overflowEur: 0, binds: '' };
}
export function planSeatCost(plan, wl, o){
  const offer = API_OFFERS[plan.api];
  const lam = requestsPerHour(wl, o.think) * o.spu;
  const rc = requestCost(offer, wl, o.turn, o.out);
  const unitReq = (plan.meterOffer ? requestCost(plan.meterOffer, wl, o.turn, o.out).usd : rc.usd) + plan.perRequest;
  const rateH = lam * unitReq, usdH = lam * rc.usd, hoursDay = o.hours / WORKDAYS_PER_MONTH;
  const full = Math.floor(hoursDay / 5), rem = hoursDay - 5 * full;
  const win5 = [...Array(full).fill(5), ...(rem > 1e-12 ? [rem] : [])];
  const spans = { '5h': [win5, WORKDAYS_PER_MONTH], day: [[hoursDay], WORKDAYS_PER_MONTH],
                  week: [[o.hours / WEEKS_PER_MONTH], WEEKS_PER_MONTH], month: [[o.hours], 1] };
  const need = {}, over = {};
  for (const [w, budget] of Object.entries(plan.budgets || {})){
    const [hours, times] = spans[w];
    need[w] = budget > 0 ? Math.max(...hours.map(h => h * rateH)) / budget : Infinity;
    over[w] = times * hours.reduce((acc, h) => h * rateH > 0
      ? acc + Math.max(0, h * rateH - budget) / (h * rateH) * h * usdH : acc, 0);
  }
  const keys = Object.keys(need);
  const top = keys.length ? keys.reduce((x, y) => need[y] > need[x] ? y : x) : '';
  const binds = top && need[top] > 1 ? top : '';
  let plans = 1, overflowEur = 0;
  if (binds){
    if (plan.overage === 'api') overflowEur = Math.max(...Object.values(over)) * (plan.overage_factor ?? 1) / USD_PER_EUR;
    else plans = Math.ceil(need[binds] - 1e-9);
  }
  const eur = plan.usd_month / USD_PER_EUR * plans + overflowEur;
  return { key: plan.key, name: plan.name, kind: 'plan', eurSeat: eur, eurSession: eur / (o.active * o.spu),
           usdRequest: rc.usd, requestsMonth: lam * o.hours, ttl: rc.ttl, hitRate: rc.hitRate,
           fits: !binds, plansPerSeat: plans, overflowEur, binds };
}
