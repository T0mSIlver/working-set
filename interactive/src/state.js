import { AVG_OUT_TOK, CONFIG, DECODE_FLOOR_TOKS, DECODE_MBU, PREFILL_MFU, makeGrid } from './config.js';

/* ============================================================================
   STATE
   ========================================================================== */
export const state = {
  // tp = TENSOR-PARALLEL WIDTH of one replica group; the replica count is the
  // derived dp = ngpu / tp, so tp must always divide ngpu (enforceConstraints
  // clamps it). tp === ngpu is pure TP (the old par:"tp"), tp === 1 is pure DP.
  model: "27B", gpu: "H200", wdt: "fp8", ngpu: 1, tp: 1, ram: 0,
  kv: "fp8", cap: 180,
  // state_dt: recurrent-state dtype (bf16 = the study default, inferred from
  // the baseline's 75 MiB); wover: deployed-weight overhead over the raw or
  // on-disk checkpoint bytes ("pub" = as published, "p15" = +15%).
  // mtp follows the SELECTED model's own value (the model segment resets it);
  // the seed here is the 27B's measured 2.94 (decode_mbu.md), was 1.7.
  state_dt: "bf16", wover: "pub", mtp: 2.94,
  // Efficiency knobs. mbu (decode) and mfu (prefill) are the study's two
  // measured-efficiency constants, exposed because both are single-deployment
  // anchors rather than brackets and the reader should be able to argue with
  // them. Both are GLOBAL: one value for every row, never reseeded per model.
  mbu: DECODE_MBU, mfu: PREFILL_MFU,
  // max_num_batched_tokens. A string enum (the pue idiom) over the powers of
  // two chart E ticks: an A/B arm selector, not a free knob. The tiles, the
  // deploy recipe and the generated harness CONFIG all follow it; the MFU
  // calibration anchor and the published tables stay pinned to PREFILL_CHUNK.
  // Reachable through share links only (`#chunk=4096`): chart E's segmented
  // control was retired 2026-09-03 once charts E2-E4 told the chunk story as
  // curves, so the page has no control for it. Chart E's dot and the deploy
  // card still show which chunk a link priced.
  chunk: "32768",
  user_median: 31, user_sigma: 0.81,
  sub_median: 8, sub_sigma: 0.90,
  sub_ratio: 0.10, sub_shares_prefix: false,
  sys: 15, inval: 1.0,
  // load & latency budget. The defaults ARE the study's reference load
  // (64 users x one turn / 30 s = 2.13 req/s, a 2,000-token warm turn), so at
  // slider defaults every readout reproduces the published tables exactly.
  // `out` = output tokens one response decodes (AVG_OUT_TOK, the reference).
  // It sets the steady-state decode point and the power model's decode duty;
  // it is the ONE assumption those two readouts cannot be honest without.
  users: 64, think: 30, sla: 10, turn: 2000, out: AVG_OUT_TOK, burst: 32,
  // per-user decode speed the DECODE ceiling is solved against. A workload
  // property, not a hardware one: the study's 40 is an agentic-coding comfort
  // standard, and a chat deployment judged at it can read as decode-bound
  // while every human reader is served comfortably.
  decode_floor: DECODE_FLOOR_TOKS,
  // UI-only: whether the frontier table shows the four per-ceiling columns.
  // Not part of any compute signature — toggling re-renders from cached rows.
  showCeil: false,
  // electricity price and facility PUE the cost card multiplies by — pricing
  // constants, so deliberately NOT in samplingSig (they move no sampled draw).
  // Defaults: Eurostat EU non-household average and the Uptime-survey colo
  // figure (research/power.md).
  ekwh: 0.19, pue: "1.5",
  // €/GPU-hour of the SELECTED part (the other part rides along at its own
  // list price scaled by the same ratio — gpuHourPrice). Seeded from
  // CONFIG.GPUS[gpu].eur_gpu_h and re-seeded on every GPU switch, the way mtp
  // follows the model; a pricing constant, so NOT in samplingSig either.
  gpuh: CONFIG.GPUS["H200"].eur_gpu_h,
};

// Snapshot of the published reference configuration, taken BEFORE any URL
// state is applied: the share link encodes only the DIFFS from this, so a
// default page shares as a bare URL and every link stays readable.
export const STATE_DEFAULTS = { ...state };

// top stop of the max_seq_len slider for the active model, derived from its
// max_ctx (1049 for 1M models, 262 for Mistral). At the top stop the workload
// cap maps to the EXACT model max (1,048,576 / 262,144), everywhere else to
// cap*1000 — so the same slider position means slightly different tokens on
// different models at the very top, by design. The clamp on model switch is
// one-way: a 1024k cap snaps down to 262 for Mistral and is NOT restored on
// switching back (the user sees the clamped value and can re-raise it).
export function currentTopo(){ return makeGrid(state.ngpu / state.tp, state.tp, state.gpu); }
export function capSliderMax(){ return Math.round(CONFIG.MODELS[state.model].max_ctx/1000); }
// the host-RAM offload buffer is shared: DP replicas each get 1/N of it
export function ramPerCache(topo){ return topo.replicas>1 ? state.ram/topo.replicas : state.ram; }

export function currentWL(){
  return {
    user_median: state.user_median*1000, user_sigma: state.user_sigma,
    sub_median: state.sub_median*1000, sub_sigma: state.sub_sigma,
    sub_ratio: state.sub_ratio,
    // subagents use a lean 3k separate prefix by default (matches scenario_model.py);
    // the "share prefix" toggle instead points them at the user prefix.
    sys_user: state.sys*1000, sys_sub: 3000,
    sub_shares_prefix: state.sub_shares_prefix,
    invalidation: state.inval/100,
    // top slider stop maps to the active model's exact context max
    cap: state.cap>=capSliderMax() ? CONFIG.MODELS[state.model].max_ctx
                                   : state.cap*1000, min_tokens: 1000,
  };
}
