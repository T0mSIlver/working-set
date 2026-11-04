/* ============================================================================
   CONFIG — ALL physical constants in one place (mirrors scenario_model.py).
   Update MoE / provisional numbers here; everything downstream re-derives.
   ========================================================================== */
import { CATALOG } from './catalog.js';
export const KIB = 1024, MIB = 1024**2, GIB = 1024**3;

// The catalogue rows with the Python dataclass defaults filled in, so every
// field reads as it does in workingset.model (None stays absent). The model
// row keeps its explorer table under `explorer`; its scores go to QUALITY.
const CATALOG_GPUS = Object.fromEntries(Object.entries(CATALOG.gpus)
  .map(([k, g]) => [k, { ...CATALOG.defaults.gpu, ...g }]));
const CATALOG_MODELS = Object.fromEntries(Object.entries(CATALOG.models)
  .map(([k, { quality, ...m }]) => [k, { ...CATALOG.defaults.model, explorer: {}, ...m }]));
// a score is [passed, runs] (an exact AA run count) or a vendor fraction; a
// benchmark the catalogue leaves out is null (no run)
const CATALOG_QUALITY = Object.fromEntries(Object.entries(CATALOG.models)
  .map(([k, { quality = {} }]) => [k, { ...quality,
    ...Object.fromEntries(["tb21", "tb40"].map(b => [b,
      quality[b] === undefined ? null
      : Array.isArray(quality[b]) ? quality[b][0] / quality[b][1] : quality[b]])) }]));

// DECODE efficiency -- MFU's counterpart, and the newest constant here.
// Decode was first a PURE roofline (bytes / effective_bw, no efficiency term
// at all) while prefill carried an anchor. A measurement not published in
// this repo found it optimistic on the reference row, and ONE constant fits
// it across n = 1-25.
// MODEL CONVENTION (already divided by tp_efficiency), so it multiplies
// effective_bw directly.
// It is fitted on a HYBRID model running speculative decode at k=2, so it
// absorbs both streaming inefficiency and whatever the speculative verify
// costs on a sequential recurrence. It therefore travels WITH the model's
// mtp: change one and the fit breaks. Mirrors MBU_DEFAULT in scenario_model.py.
// EXTRAPOLATION BIAS: fitted over n = 1-25, where weights are 80-95% of the
// step. The decode ceiling lives at n ~ 100-250, where KV dominates and a
// whole-ledger multiplier over-charges. The 27B/TP4 ceiling reads 108 here,
// 238 under reading A (the verify multiplies the weight reads) and 154 under
// reading B (it serialises as latency) -- against a
// cache ceiling of 249, so the BINDING ORDER is not identified. Per-user
// speed near the fitted mix is; the ceiling is not. See MBU_DEFAULT.
// ONE value for EVERY row, from the slider. A dense or MLA model has no
// recurrence to serialise and should sit higher, but nothing in this study
// measures one, and a per-model guess would rank the frontier on the guess.
// So the slider is a GLOBAL sensitivity knob: drag it and watch which
// configurations survive. (Not the MTP rule -- own capability per row --
// because MTP is a shipped feature and MBU is an unmeasured efficiency.)
// Per-row values come back the day a second architecture is measured.
export const DECODE_MBU    = 0.22;

// OPT-IN decode pricing. Mirrors DECODE_BW_EFF,
// DECODE_FIXED_S and DECODE_SPEC_TOKENS in workingset.model: a decode step is
//   step_bytes / (effective_bw x bw_eff) + n x (1 + spec) x t_token + fixed
// with t_token = 2 x params_prefill / (peak x mfu). Read on ONE deployment
// (TP4, MTP depth 3); the default pricing stays the single DECODE_MBU fold.
export const DECODE_BW_EFF      = 0.36;
export const DECODE_FIXED_MS    = 2.7;
export const DECODE_SPEC_TOKENS = 3;

export const CONFIG = {
  // ---- HARDWARE ----
  // The GPU is a selectable part; H200 is the calibrated baseline. Rows and
  // their provenance (HBM, bandwidth, dense FP8 peak, power, rental price):
  // src/workingset/data/catalog.toml.
  GPUS: CATALOG_GPUS,
  VRAM_PER_GPU: 141e9,     // calibration-anchor GPU (H200) HBM bytes
  HBM_BW:       4.8e12,    // calibration-anchor GPU bandwidth, bytes/s
  TP_EFFICIENCY: 0.90,     // tensor-parallel aggregate-BW haircut
  // extra haircut per doubling once a TP group spans nodes. UNMEASURED and
  // deliberately pessimistic; nothing in the study crosses the boundary.
  CROSS_DOMAIN_EFFICIENCY: 0.65,
  // Calibration anchor: 1xH200 + 27B FP8 KV pool (tokens). PROJECTED from the
  // baseline's measured FP16 pool (~1.337M x2 + freed activations), not measured.
  BASELINE_POOL_TOKENS_27B_1GPU: 2.77e6,

  // ---- MODELS ----
  // Built from src/workingset/data/catalog.toml (through the generated
  // catalog.js); field meanings and provenance live there and on
  // workingset.model.Model. Each row carries its explorer-only table
  // (labels, tips, recipe rules) as `explorer`.
  MODELS: CATALOG_MODELS,

  // ---- QUALITY ----
  // Coding-agent scores per model, so the frontier ranks on what a
  // configuration delivers and not only on what it costs. TWO Terminal-Bench
  // versions, both pass@1 as MEASURED BY ARTIFICIAL ANALYSIS (one lab, one
  // harness per version) — vendor cards quote other harnesses and other
  // versions and are not comparable, so they are deliberately not used, with
  // ONE exception by owner decision (2026-09-10): DSV41F carries its vendor
  // card's 2.1 figure because AA has no 2.1 run of it (its card's 4.0 figure
  // gave way to AA's run on 2026-11-04). That one is a percentage, not a run
  // count; `source[bench]` names it. A model AA has not run on a version
  // carries null there (ML4 on 2.1). Reasoning variant at
  // the effort AA ran for its index. The FP16-KV / NVFP4 arms inherit the base
  // score: the quantisation loss is unmeasured, and a per-arm guess would rank
  // the frontier on the guess. Ledger and protocol: research/terminal_bench.md.
  //
  // 2.1 is AA's LEGACY agentic-coding eval and 4.0 the one in the
  // Intelligence Index (since v4.3; v5.0.1 on 2026-11-04); they disagree about
  // the top of this frontier.
  // Among the AA-MEASURED eight, 2.1 ranks Q38FN first and GLM52 third while
  // 4.0 reverses them: GLM52 leads Q38FN by 16.7 points there and trails it
  // by 2.2 on 2.1. (DSV41F's vendor 2.1 figure tops 2.1 outright, but it is
  // not measured like the others — § source.) That is why the explorer offers
  // both rather than picking one. Read 2026-09-10, re-read 2026-11-04.
  BENCHES: {
    tb21: { label: "TB 2.1", name: "Terminal-Bench 2.1", runs: 267,
            harness: "Terminus 2", protocol: "89 tasks × 3 repeats, Terminus 2",
            status: "AA legacy eval" },
    tb40: { label: "TB 4.0", name: "Terminal-Bench 4.0", runs: 198,
            harness: "mini-SWE-agent v2.4.6", protocol: "66 tasks × 3 repeats, mini-SWE-agent v2.4.6",
            status: "AA Intelligence Index v5" },
  },
  QUALITY: CATALOG_QUALITY,

  // ---- TOPOLOGY ----
  // A DP x TP grid: the unit DP replicates is a GROUP of `tp` GPUs, not a GPU.
  // n_gpu/kind/replicas are derived (dp*tp / label / dp) and spelled out here
  // only because these literals are built before the helpers exist.
  // `gpu` is filled in below (the GPUS literal is not in scope yet), so these
  // are structurally identical to makeGrid(dp, tp) — asserted in unitChecks.
  TOPOLOGIES: {
    "1xH200":     { name: "1×H200",            dp: 1, tp: 1, n_gpu: 1, kind: "single", replicas: 1 },
    "2xH200-TP2": { name: "2×H200 tensor-par", dp: 1, tp: 2, n_gpu: 2, kind: "tp",     replicas: 1 },
    "2xH200-DP2": { name: "2×H200 data-par",   dp: 2, tp: 1, n_gpu: 2, kind: "dp",     replicas: 2 },
  },

  // ---- MONTE-CARLO sizes (full quality) ----
  WARM_ITER: 700,
  // The decode x-axis is DYNAMIC: it must always reach the stress point this
  // study cares about — every GPU-resident warm session decoding at once — which
  // is ~94 seqs for the 27B on one H200 but ~680 for the 35B-A3B on TP2. NMIN
  // keeps small configs on a familiar axis; HEADROOM leaves room past warm p95.
  DECODE_ITER: 600, DECODE_NMIN: 120, DECODE_HEADROOM: 1.15, DECODE_SAMPLES: 60,
  // Sweep cost is O(iter x nMax) (see decodeCurves' shared draw), so iterations
  // are capped by a WORK budget rather than a flat floor: every realistic config
  // keeps the full 600, and only a pathologically wide axis tapers toward the
  // minimum. A fixed sample COUNT keeps the curve smooth at any range.
  DECODE_BUDGET: 1_400_000, DECODE_ITER_MIN: 40,
  // Warm-fill work budget (draws per warmCapacity call). A fill samples one
  // request per resident session, so pool size — not iteration count — sets the
  // cost; these caps keep the widest legal config interactive. SCAN is chart B's
  // 12-point survey, which is coarse by design.
  WARM_BUDGET: 600_000, WARM_BUDGET_SCAN: 120_000, WARM_ITER_MIN: 8,
  // draft (while dragging) — lighter for responsiveness
  DRAFT: { WARM_ITER: 220, DECODE_ITER: 200, DECODE_SAMPLES: 30,
           DECODE_BUDGET: 400_000, DECODE_ITER_MIN: 30,
           WARM_BUDGET: 200_000, WARM_BUDGET_SCAN: 35_000 },
};

// Backfill the GPU object onto the legacy topology literals — GPUS is not in
// scope inside the CONFIG object literal, and without this `topo.gpu` is
// undefined there and only survives via `|| GPUS["H200"]` fallbacks. Python's
// Topology carries the GPU directly; this makes the mirror structurally exact.
for (const t of Object.values(CONFIG.TOPOLOGIES)) t.gpu = CONFIG.GPUS["H200"];

/* ---- model helpers ---- */
export function is_moe(m){ return m.w_route_total > 0; }
// concurrency at which the conservative expert union saturates (every routed
// expert read each step): n_sat = total/pertok = E/k = 32 for both MoE models
export function unionKink(m){ return is_moe(m) ? Math.round(m.w_route_total / m.w_route_pertok) : null; }
// Conservative (no-overlap) expert-union bound, the study's planning default;
// the expected-union "coverage" model is the optimistic bracket (see docs).
export function w_decode(m, n){ return m.w_decode_shared + Math.min(n * m.w_route_pertok, m.w_route_total); }
// recurrent / fixed-state bytes moved per active sequence per decode step:
// read + write of the whole state unless the model says otherwise (mirrors
// Model.state_traffic — the DeepSeek Flash models' windows are read inside kv_decode_const)
export function state_traffic(m){ return m.state_step_bytes ?? 2 * m.deltanet_state; }
// FP16-KV transform (mirrors Python's with_kv_dtype): doubles kv_bpt AND the
// top-k main-KV gathers (kv_decode_const) of a sparse-decode model; the
// quantized indexer scan (kv_decode_bpt) keeps its own width. Identity for
// fp8 and for models whose FP16 arm is not servable (kv_fp16_ok false).
// ONE implementation, called by modelFor and by the load-time unit checks.
export function withKvDtype(m, kv){
  // one-way door (mirrors with_kv_dtype): an fp16 arm must not be converted
  // again or "back" — a round-trip would double-double and un-stamp kv_dtype
  if (m.kv_dtype === "fp16")
    throw new Error("withKvDtype expects a base (fp8) model");
  if (m.kv_gpu)
    throw new Error(`${m.name}: withKvDtype must run before withGpu`);
  if (kv !== "fp16" || m.kv_fp16_ok === false) return m;
  return { ...m, kv_bpt: m.kv_bpt*2, kv_decode_const: (m.kv_decode_const||0)*2,
           kv_dtype: "fp16", name: m.name + " [FP16 KV]" };
}
// The FP8 cache row this GPU serves (mirrors with_gpu). Identity for every
// model with one row on both parts; GLM-5.3 (kv_hopper_fp8) takes FlashMLA's
// fp8_ds_mla row on a Hopper part when it prices the FP8 arm. Every arm of
// such a model is stamped with the part, so checkDtypeSupported can refuse a
// model resolved for the other one. Apply it after withKvDtype.
export function withGpu(m, gpuKey){
  if (!m.kv_hopper_fp8) return m;
  if (m.kv_gpu) throw new Error(`${m.name}: already resolved for ${m.kv_gpu}`);
  if (m.kv_dtype !== "fp16" && !CONFIG.GPUS[gpuKey].supports_nvfp4){
    const [kv_bpt, kv_decode_const] = m.kv_hopper_fp8;
    return { ...m, kv_bpt, kv_decode_const, kv_gpu: gpuKey };
  }
  return { ...m, kv_gpu: gpuKey };
}
// The KV dtype actually SERVABLE for this model on this GPU (mirrors
// check_dtype_supported): GLM-5.3-Flash's fp8 KV cache is Blackwell-only —
// "Hopper ... must run BF16 KV" (vLLM recipe) — so on H200 the fp8 request
// resolves to fp16. Keeps the frontier table and tiles from ever pricing an
// unservable arm, whatever the global toggle says.
export function servableKv(m, kv, gpuKey){
  return (kv === "fp8" && m.kv_fp8_blackwell_only === true
          && !CONFIG.GPUS[gpuKey].supports_nvfp4) ? "fp16" : kv;
}

/* ---- derived: activation reserve solved from the anchor ---- */
function actReserve(){
  const m = CONFIG.MODELS["27B"];
  const poolBytes = CONFIG.BASELINE_POOL_TOKENS_27B_1GPU * m.kv_bpt;
  return CONFIG.VRAM_PER_GPU - m.w_resident - poolBytes;
}
// 17.98 GiB. Independently corroborated: the 2xH200 TP2 FP16 startup log
// implies 18.24 GiB per GPU (+1.4%), with nothing about TP2 or FP16 fitted.
// There is deliberately no second, "conservative" reserve any more — see
// activeModel() for why the low anchor was retired.
export const ACT_RESERVE = actReserve();

/* ---- study constants: the reference values every published figure is quoted
   at (mirrors scenario_model.py). They sit here rather than beside the prefill
   and decode code that uses them because the state object's DEFAULTS read
   them at load, and the module that holds them must not depend on anything
   that depends on state. ---- */
export const PREFILL_MFU   = 0.45;    // the central of the plausible [0.30, 0.55] FP8 bracket
                               // (prefill.md §1; the bracket rests on
                               // measurements not published in this repo)
export const PREFILL_MFU_LO = 0.30, PREFILL_MFU_HI = 0.55;  // the bracket itself:
// an ERROR BAR on every prefill-derived figure, not a knob anyone chooses
export const DECODE_FLOOR_TOKS = 40;      // the study's hard per-user floor — now the
                                   // DEFAULT of state.decode_floor, kept as a
                                   // named constant because every published
                                   // figure is quoted at it
// the comfortable mark has always been 1.25x the hard floor (50 against 40);
// keeping the ratio means the chart-C guide lines and the tile's green/amber
// split still mean the same thing at any floor the slider is set to
export const DECODE_COMFORT_RATIO = 1.25;
// Output tokens ONE REQUEST decodes (every request, subagents included) —
// ~400 mean, from an agentic trace not published in this repo (was assumed
// 1,000). At 50-90 tok/s this decodes in ~5-8 s of the measured 10.8 s
// served per turn.
// The reference value; the `out` slider starts here and STATE_DEFAULTS pins it,
// so a default page still reproduces every published number. Drives the
// steady-state decode point below and the power model's decode duty.
export const AVG_OUT_TOK = 400;


// NVFP4 weights require native FP4 tensor cores (B300) — the UI greys the
// option out on the H200; this guard backstops any programmatic path.
function checkDtypeSupported(m, topo){
  const gpu = topo.gpu || CONFIG.GPUS["H200"];
  if (m.weight_dtype === "nvfp4" && !gpu.supports_nvfp4)
    throw new Error(`NVFP4 weights require native FP4; ${gpu.name} has none`);
  // GPU-coupled KV dtype (GLM-5.3-Flash): fp8 KV is Blackwell-only — the
  // BF16-KV arm (withKvDtype "fp16", which stamps kv_dtype) must be priced
  // on Hopper instead (mirrors Python's check_dtype_supported)
  if (m.kv_fp8_blackwell_only === true && m.kv_dtype !== "fp16"
      && !gpu.supports_nvfp4)
    throw new Error(`${m.name}: FP8 KV is Blackwell-only — price the BF16-KV arm on ${gpu.name}`);
  // GPU-coupled FP8 row (GLM-5.3): the base constants are the Blackwell row,
  // so a Hopper topology needs withGpu's swap and a swapped model may price
  // Hopper only (mirrors Python's check_dtype_supported)
  if (m.kv_hopper_fp8 && m.kv_dtype !== "fp16"){
    const hopperRow = !!m.kv_gpu && !CONFIG.GPUS[m.kv_gpu].supports_nvfp4;
    if (hopperRow !== !gpu.supports_nvfp4)
      throw new Error(`${m.name}: its FP8 cache row depends on the GPU — price withGpu(m, "${gpu.name}") on this topology`);
  }
}

// KV pool of ONE replica group. A group owns topo.tp GPUs: the weights are
// sharded across them and counted once, the reserve is scratch and charged per
// GPU -> pool = tp*(vram - reserve) - weights. Per-GROUP, i.e. per-cache; a
// dp>1 deployment has topo.replicas of them. A model too large for its group
// still clamps to 0 (the study's "does not fit" sentinel); minTpFor() reports
// the group size that would fit.
export function kv_pool_tokens(m, topo){
  checkDtypeSupported(m, topo);
  const gpu = topo.gpu || CONFIG.GPUS["H200"];
  const R = ACT_RESERVE + (gpu.reserve_extra || 0);
  // NOT factored as tp*(vram - R) - w: the three terms in this order make every
  // pre-existing configuration bit-identical to the pre-grid code (and to
  // Python's kv_pool_tokens), rather than merely equal after formatting.
  const poolBytes = topo.tp * gpu.vram - m.w_resident - topo.tp * R;
  // ...in tokens of the cache the group STORES: a replicated layout pays
  // kv_bpt x copies per token (kvReplication) — mirrors _replicated()
  return Math.max(poolBytes, 0) / replicated(m, topo).kv_bpt;
}

// KV sharding under TP (mirrors kv_replication / dcp_size / _replicated in
// workingset.model; research/kv_tp_sharding.md). A cache with h shardable
// heads splits across min(tp, h) ranks and is REPLICATED on the tp/h rank
// groups beyond that. topo.kv_shard "dcp" (the default — decode context
// parallelism, --decode-context-parallel-size floor(tp/h)) splits that
// dimension along the sequence: ONE copy at every tp (an odd width the heads
// do not divide is extrapolated as sharded, as before this layout existed);
// "replicate" (plain --tensor-parallel-size) pays every copy, tp/h.
export const KV_SHARDS = ["dcp", "replicate"];
export function kvReplication(m, topo){
  const factor = heads => {
    if ((topo.kv_shard || "dcp") === "dcp" || heads === undefined || heads === null || topo.tp <= heads) return 1.0;
    return topo.tp / heads;
  };
  return [factor(m.kv_heads), factor(m.state_heads)];
}
export function dcpSize(m, topo){
  if ((topo.kv_shard || "dcp") !== "dcp") return 1;
  return Math.max(1, Math.floor(topo.tp / Math.max(1, m.kv_heads)));
}
// the model with its cache and state bytes multiplied by the copies the group
// holds — what the pool and the decode step pay. Applied INSIDE kv_pool_tokens,
// warmCapacity and decodeCurves to the caller's model; never returned to a
// caller, so it cannot be applied twice.
export function replicated(m, topo){
  const [rKv, rSt] = kvReplication(m, topo);
  if (rKv === 1.0 && rSt === 1.0) return m;
  return { ...m,
    kv_bpt: m.kv_bpt * rKv,
    kv_decode_bpt: (m.kv_decode_bpt === undefined || m.kv_decode_bpt === null) ? m.kv_decode_bpt : m.kv_decode_bpt * rKv,
    kv_decode_const: (m.kv_decode_const ?? 0) * rKv,
    deltanet_state: m.deltanet_state * rSt,
    state_step_bytes: (m.state_step_bytes === undefined || m.state_step_bytes === null) ? m.state_step_bytes : m.state_step_bytes * rSt };
}

// Smallest TP group size with a non-empty pool on this part. Shares the single
// ACT_RESERVE with the kv_pool_tokens sitting next to it, so the two always
// agree — and, now that the low anchor is gone, so does Python's min_tp_for
// (the reserve no longer depends on any UI state, so JS and Python cannot
// diverge here at all). Pass the model from activeModel() so a +15% weight
// bump is reflected in the threshold the Split control reports.
export function minTpFor(m, gpuKey){
  const gpu = CONFIG.GPUS[gpuKey || "H200"];
  return Math.floor(m.w_resident / (gpu.vram - (ACT_RESERVE + (gpu.reserve_extra || 0)))) + 1;
}

// topology for arbitrary N GPUs of a part (mirrors topology()/tp_efficiency()
// in scenario_model.py): TP = one engine, weights sharded, one shared cache,
// 0.90 bandwidth haircut per GPU-count doubling inside the node, then a steeper
// CROSS_DOMAIN_EFFICIENCY penalty per doubling once a TP group spans nodes.
export function tpEff(n, domain){
  // an explicit 0/negative domain is an ERROR, not a request for the default —
  // `domain || 8` would silently mask a non-NVLink part as an 8-GPU node while
  // Python's tp_efficiency raises. Only undefined/null falls back.
  const d = (domain === undefined || domain === null) ? 8 : domain;
  if (!(d >= 1) || Math.floor(d) !== d)
    throw new Error(`nvlink_domain must be a positive integer, got ${domain}`);
  if (n<=1) return 1.0;
  let eff = Math.pow(CONFIG.TP_EFFICIENCY, Math.log2(Math.min(n, d)));
  if (n > d) eff *= Math.pow(CONFIG.CROSS_DOMAIN_EFFICIENCY, Math.log2(n/d));
  return eff;
}
// DP x TP grid (mirrors topology_grid() in scenario_model.py). DP replicates
// whole GROUPS of `tp` GPUs — the only way the models that fit no single GPU
// (Mistral-Medium-3.5, GLM-5.3) can be data-parallel at all.
export function makeGrid(dp, tp, gpuKey, kvShard){
  // mirror topology_grid()'s validation: a silently-accepted dp=0 would build a
  // zero-GPU topology whose pool still reads non-empty (the pool depends on tp)
  for (const [label, v] of [["dp", dp], ["tp", tp]])
    if (!(v >= 1) || Math.floor(v) !== v)
      throw new Error(`${label} must be a positive integer, got ${v}`);
  const key = gpuKey || "H200";
  if (!CONFIG.GPUS[key]) throw new Error(`unknown gpu ${key}`);
  const shard = kvShard || "dcp";
  if (!KV_SHARDS.includes(shard)) throw new Error(`kv_shard must be one of ${KV_SHARDS}, got ${kvShard}`);
  const g = CONFIG.GPUS[key], n = dp*tp;
  // names match Python's topology_grid() exactly (× for x)
  let name = n===1 ? `1×${g.name}`
           : dp===1 ? `${tp}×${g.name} tensor-par`
           : tp===1 ? `${dp}×${g.name} data-par`
           : `${n}×${g.name} DP${dp}×TP${tp}`;
  if (shard === "replicate" && tp > 1) name += " [KV replicated]";
  const kind = n===1 ? "single" : dp===1 ? "tp" : tp===1 ? "dp" : "hybrid";
  return {name, dp, tp, n_gpu:n, kind, replicas:dp, gpu:g, kv_shard:shard};
}
// single-axis helper for the explorer's TP/DP toggle: the two grid edges
export function makeTopo(kind, n, gpuKey, kvShard){
  if (kind!=="tp" && kind!=="dp") throw new Error(`kind must be 'tp' or 'dp', got ${kind}`);
  return kind==="tp" ? makeGrid(1, n, gpuKey, kvShard) : makeGrid(n, 1, gpuKey, kvShard);
}
// divisors of n, ascending — the TP widths that evenly split n GPUs
export function divisors(n){ const d=[]; for (let i=1;i<=n;i++) if (n%i===0) d.push(i); return d; }
// Snap a TP width to a legal one for `ngpu`: the largest divisor <= tp, so
// shrinking the GPU count keeps as much TP as still divides evenly rather than
// resetting the user to pure TP or pure DP. Always returns a divisor of ngpu.
export function clampTp(tp, ngpu){
  const ds = divisors(ngpu);
  return ds.includes(tp) ? tp : (ds.filter(d => d <= tp).pop() ?? 1);
}
// Bandwidth seen by ONE replica group. DP adds groups, it never widens one.
export function effective_bw(topo){
  const gpu = topo.gpu || CONFIG.GPUS["H200"];
  return topo.tp * gpu.hbm_bw * tpEff(topo.tp, gpu.nvlink_domain);
}
