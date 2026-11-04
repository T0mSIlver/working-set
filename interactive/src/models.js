/* The model arms the page prices: the selected model with every dtype,
   overhead, MTP and efficiency knob applied, and the same switches applied to
   any other model for cross-configuration comparison. Moved out of render.js
   so pages and workers that never touch the DOM can price a model. */
import { CONFIG, servableKv, withGpu, withKvDtype } from './config.js';
import { decodeLatency, decodeMbu } from './capacity.js';
import { state } from './state.js';

// active model with the dtype, weight-overhead and MTP-speedup knobs applied.
// FP16 doubles kv_bpt (with_kv_dtype in scenario_model.py); fp32 doubles the
// recurrent state; +15% covers deployed-weight overhead on the models whose
// resident bytes are raw/checkpoint figures.
//
// These two are what remains of the old bundled "Conservative" set. Its third
// ingredient — a calibration anchor at 2x the measured FP16 LOWER bound
// (2.278M, a 33.0 GiB per-GPU reserve) — was RETIRED 2026-07-29: the TP2 FP16
// startup log pins the real per-GPU reserve at 18.24 GiB and the equivalent
// FP8 anchor at 2.762M tokens. The low anchor would have predicted a 2.750M-
// token TP2 pool against 3,233,564 measured (-14.9%), so it is not a
// plausible-adverse case any more, it is a refuted one. See docs/scenarios.md
// § Measured cross-check.
export function activeModel(){ return modelFor(state.model); }
// The same dtype / deployed-overhead switches applied to an ARBITRARY model —
// what the frontier table needs to price every configuration on equal terms
// (and what activeModel() has always done for the selected one).
// Cross-MODEL comparison variant: the same dtype / overhead switches, but each
// model keeps its OWN speculative-decoding capability instead of inheriting the
// MTP slider. Ranking configurations against each other must not hand one of
// them a speedup it does not ship — Mistral-Medium-3.5 has no MTP module at
// all, and pricing it at the slider's 1.7x doubles its decode ceiling (74 vs
// 36 users) and hides that it is decode-bound below the reference load. The
// slider stays a SENSITIVITY control on the selected configuration, which is
// what its own tooltip says it is.
export function modelForCompare(key){
  const own = CONFIG.MODELS[key].mtp;
  const saved = state.mtp;
  state.mtp = own;
  try { return modelFor(key); } finally { state.mtp = saved; }
}
function modelFor(key){
  let m = CONFIG.MODELS[key];
  // NVFP4 weights (B300-only; the UI disables the option elsewhere): swap in
  // the NVFP4-checkpoint weight bytes — KV and recurrent state are untouched
  if (state.wdt === "nvfp4" && m.nvfp4_w){
    const [wr, wds, wpt, wtot] = m.nvfp4_w;
    m = { ...m, w_resident: wr, w_decode_shared: wds, w_route_pertok: wpt,
          w_route_total: wtot, weight_dtype: "nvfp4", name: m.name + " [NVFP4]" };
  }
  // FP16 KV is not servable on Mistral-Large-4 and the DeepSeek Flash models;
  // the UI disables the toggle — withKvDtype's guard keeps the math honest
  // anyway (and prices the sparse-decode doubling on Q38FN). servableKv
  // first resolves GPU-coupled dtypes (GLM-5.3-Flash: BF16 forced on H200),
  // and withGpu GPU-coupled rows (GLM-5.3: fp8_ds_mla on H200), so no code
  // path — the frontier included — ever prices an unservable arm.
  m = withGpu(withKvDtype(m, servableKv(m, state.kv, state.gpu)), state.gpu);
  // fp32 recurrent state. No-op on the pure-attention models (MM35, GLM-5.3
  // carry no state at all) and on the DeepSeek Flash models (their per-session state is a
  // fixed mixed-precision buffer, state_fp32_ok) — the UI disables both.
  if (state.state_dt === "fp32" && m.deltanet_state > 0 && m.state_fp32_ok !== false)
    m = { ...m, deltanet_state: m.deltanet_state*2, name: m.name + " [fp32 state]" };
  // +15% deployed-weight overhead. Never applied to the 27B, whose 28.8 GiB is
  // already the as-deployed footprint (that is where the 15% came from) — the
  // UI disables the control on that model.
  if (state.wover === "p15" && state.model !== "27B")
    m = { ...m, w_resident: m.w_resident*1.15, name: m.name + " [+15% weights]" };
  m = { ...m, mtp: state.mtp };   // MTP speedup knob (1.0 = off)
  // ...and the decode-efficiency knob, baked onto the model so decodeCurves
  // and decodeUsersApprox read one field whichever path built the model.
  // Deliberately the SLIDER for every row, the frontier included (see
  // DECODE_MBU); frontierDecSig carries state.mbu so the cached decode
  // ceilings rebuild when it moves.
  m = { ...m, decode_mbu: decodeMbu() };
  // ...and the opt-in latency pricing, null under the default roofline
  const lat = decodeLatency();
  if (lat) m = { ...m, decode_lat: lat };
  return m;
}
