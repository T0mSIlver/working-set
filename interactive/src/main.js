/* ============================================================================
   ENTRY — the explorer's modules, bottom-up (each imports only what it names):

     config.js       every physical constant, the model/GPU/topology registry and
                     the derived activation reserve. No imports: the state
                     object's defaults read it at load.
     prefill.js      the compute roofline: MFU(chunk), cold spikes, the operating
                     point, the steady-state decode point.
     mathlib.js      the seeded RNG and the math primitives.
     workload.js     prompt-length sampling (mirrors Workload.sample).
     capacity.js     warm capacity in one cache and decode tok/s vs max_num_seqs.
     selfcheck.js    the three load-time checks against scenario_model.py.
     state.js        the state object, its defaults, and the current-workload view.
     svg.js          scales, ticks, formatting, the SVG shell.
     charts.js       charts A–E4 and the crosshair tooltip.
     render.js       the render pipeline: what recomputes when, tiles, frontier
                     rebuild scheduling.
     planner.js      the planner panel: ceilings, charts F & G.
     harness.js      the generated validation harness ("Test these hypotheses").
     cost.js         the bill: power draw and €/month.
     sensitivity.js  "what would flip this decision".
     deploy.js       the deploy card and its vLLM recipe.
     frontier.js     the frontier table and chart H.
     main.js         control wiring, share links, theme, hover, init.

   Cycles between the runtime modules are fine (imports are live bindings and
   nothing but state.js reads another module at evaluation time), but a module
   that state.js depends on must never depend on state.js.
   ========================================================================== */
import { CONFIG, MIB, clampTp, divisors, is_moe, kv_pool_tokens, minTpFor, unionKink } from './config.js';
import { decodeComfort, decodeFloor, requestRate } from './prefill.js';
import { prefillSampledChecks, steadyChecks, unitChecks } from './selfcheck.js';
import { clip } from './mathlib.js';
import { STATE_DEFAULTS, capSliderMax, currentTopo, currentWL, state } from './state.js';
import { cssv, esc, fmt } from './svg.js';
import { chartCGeom, chartDGeom, clearChartGeomCD, drawCross, interpAt, lastChartE,
         redrawChartECompanions, removeCross, renderChartA, renderChartB, renderChartC,
         renderChartD, renderChartE, renderNoFit, setupHover } from './charts.js';
import { activeModel, computeAndRender, frontierDecodeDeferred, lastCS, lastDC,
         lastSteady, lastStress, lastWarmAll, lastWarmCur, renderTiles } from './render.js';
import { PLANNER_LABEL } from './planner.js';
import { lastFlipAxes, renderFlipPanel } from './sensitivity.js';
import { deployCmdText, lastDeploy, renderDeployCard } from './deploy.js';
import { frontierChartGeom, frontierRowName, frontierScore, lastFrontierCurKey, lastFrontierRows, renderFrontierChart, renderFrontierTable } from './frontier.js';

/* ============================================================================
   CONTROL WIRING
   ========================================================================== */
let debTimer=null, frontierTimer=null;
// The frontier's decode half re-runs a bisection per row (~27 rows). Folding
// it into the 120 ms settle made the decode-floor slider feel laggy in a way
// no other slider does, so it gets its own, longer tier: the verdict, tiles
// and charts land at 120 ms as before, and the "every alternative ranked"
// table — which is below the fold and is not what you are watching while
// dragging — catches up at 500 ms. The rebuild itself is time-sliced (see
// renderFrontierSection), so what either tier costs the main thread in one
// task is ~one slice, not the whole table.
// Deliberately NOT a numeric shortcut: no probe count is reduced and no
// published frontier number moves. Only WHEN it recomputes changes.
const FRONTIER_SETTLE_MS = 500;
function scheduleFull(){
  clearTimeout(debTimer);
  debTimer=setTimeout(()=>{ computeAndRender(false, true); }, 120);
  clearTimeout(frontierTimer);
  // only arm the second tier if the first one actually deferred something —
  // otherwise every keystroke schedules a redundant full re-render
  frontierTimer=setTimeout(()=>{
    if (frontierDecodeDeferred) computeAndRender(false);
  }, FRONTIER_SETTLE_MS);
}
function onInput(){
  syncLabels();
  computeAndRender(true);   // draft while dragging
  scheduleFull();           // full on settle
}

// Rebuild the DP×TP split control for the current GPU count. The valid TP
// widths are exactly the divisors of ngpu, so the button set changes whenever
// the slider moves — this is where the hybrid grid becomes reachable from the
// UI at all (pure TP and pure DP are just its two ends).
function renderSplitControl(){
  const seg = document.getElementById('seg-split');
  if (!seg) return;
  const need = minTpFor(activeModel(), state.gpu);
  seg.innerHTML = '';
  for (const tp of divisors(state.ngpu)){
    const dp = state.ngpu / tp;
    const b = document.createElement('button');
    b.dataset.v = String(tp);
    b.textContent = state.ngpu === 1 ? '1 GPU'
                  : dp === 1 ? `TP${tp}`
                  : tp === 1 ? `DP${dp}`
                  : `DP${dp}×TP${tp}`;
    b.setAttribute('aria-pressed', tp === state.tp ? 'true' : 'false');
    const fits = tp >= need;
    if (!fits) b.classList.add('nofit');
    b.title = fits
      ? `${dp} replica group${dp>1?'s':''} × ${tp} GPU${tp>1?'s':''} — one KV cache per group`
      : `weights need TP ≥ ${need} on ${state.gpu}; a ${tp}-GPU group holds no cache`;
    b.addEventListener('click', ()=>{
      state.tp = tp;
      enforceConstraints(); syncLabels(); computeAndRender(false);
    });
    seg.appendChild(b);
  }
}

// Cross-control constraints (mirrors check_dtype_supported / kv_fp16_ok in
// scenario_model.py): NVFP4 weights only on GPUs with native FP4 (B300);
// FP16 KV never on GLM-5.2 (vLLM's DSA path asserts a quantized cache).
// Illegal selections snap back to FP8 rather than silently mis-pricing.
// The two structural knobs are disabled where they would be no-ops — fp32
// state on a model with no recurrent state, +15% weights on the 27B — and
// snapped back, so a stale selection can never read as if it were applied.
// Also snaps the TP width to a divisor of the GPU count and rebuilds the split
// control (whose dimmed buttons depend on the active model and GPU part).
function enforceConstraints(){
  const m = CONFIG.MODELS[state.model];
  state.tp = clampTp(state.tp, state.ngpu);
  renderSplitControl();
  // NVFP4 needs BOTH a capable GPU and an existing checkpoint (DSv4-Flash
  // has none: its experts are already FP4 natively — research note #4)
  const nvOk = CONFIG.GPUS[state.gpu].supports_nvfp4 && !!m.nvfp4_w;
  if (!nvOk && state.wdt === "nvfp4") state.wdt = "fp8";
  const fp16Ok = m.kv_fp16_ok !== false;
  if (!fp16Ok && state.kv === "fp16") state.kv = "fp8";
  // GLM-5.3-Flash on Hopper: fp8 KV is not servable (Blackwell-only), so
  // the selection snaps to FP16/BF16 and the fp8 button locks — the same
  // "illegal selections snap back rather than silently mis-pricing" policy
  // as NVFP4-on-Hopper
  const fp8Ok = !(m.kv_fp8_blackwell_only === true
                  && !CONFIG.GPUS[state.gpu].supports_nvfp4);
  if (!fp8Ok && state.kv === "fp8") state.kv = "fp16";
  const stateOk = m.deltanet_state > 0 && m.state_fp32_ok !== false;
  if (!stateOk && state.state_dt === "fp32") state.state_dt = "bf16";
  const woverOk = state.model !== "27B";
  if (!woverOk && state.wover === "p15") state.wover = "pub";
  // reflect state + disabled flags back into the segmented controls
  const sync = (segId, val)=>document.querySelectorAll(`#${segId} button`).forEach(
    b=>b.setAttribute('aria-pressed', b.dataset.v===val?'true':'false'));
  sync('seg-wdt', state.wdt); sync('seg-kv', state.kv);
  sync('seg-state', state.state_dt); sync('seg-wover', state.wover);
  document.querySelector('#seg-wdt button[data-v="nvfp4"]').disabled = !nvOk;
  document.querySelector('#seg-kv button[data-v="fp16"]').disabled = !fp16Ok;
  document.querySelector('#seg-kv button[data-v="fp8"]').disabled = !fp8Ok;
  document.querySelector('#seg-state button[data-v="fp32"]').disabled = !stateOk;
  document.querySelector('#seg-wover button[data-v="p15"]').disabled = !woverOk;
  // control explanations live in the label tooltips, not under the knobs;
  // the state-dependent ones are re-stamped here on every constraint pass
  document.getElementById('kv-tip').setAttribute('data-tip', fp16Ok
    ? (state.model === "GLM53F"
      ? "GLM-5.3-Flash inverts the usual constraint: fp8 KV is BLACKWELL-ONLY — vLLM's recipe says Hopper must run BF16 KV, so on H200 the toggle LOCKS to FP16 (double bytes/token AND doubled top-k reads; the compressed indexer scan stays fp8). On B300 both arms are selectable."
      : "All constants assume the FP8 KV cache (--kv-cache-dtype fp8_e4m3), as tested in the baseline. FP16 doubles KV bytes/token: half the pool, slower decode. Weights & DeltaNet state unaffected.")
    : (state.model === "DSV4F"
      ? "DeepSeek-V4-Flash: vLLM's V4 path asserts a quantized (FP8) main KV cache and SGLang's BF16 KV-decode is unfinished — FP16 KV is not servable, so the toggle is disabled."
      : "GLM-5.2: vLLM's sparse-MLA path requires a quantized (FP8) KV cache — FP16 KV is not servable, so the toggle is disabled."));
  document.getElementById('state-tip').setAttribute('data-tip', stateOk
    ? `Dtype of the Gated-DeltaNet recurrent state — a flat ${fmt(m.deltanet_state/MIB,1)} MiB per-session charge at bf16, double at fp32. bf16 is ASSUMED, not measured: the largest un-measured structural knob on the Qwens (~−10% warm capacity at fp32) — evidence: method page, "The two un-measured structural knobs".`
    : (state.model === "DSV4F"
      ? `DeepSeek-V4-Flash pays a fixed ${fmt(m.deltanet_state/MIB,1)} MiB/session, but its precision is set by the serving stack — no bf16/fp32 knob to turn, so the control is disabled.`
      : `${m.name.split(" (")[0]} keeps no recurrent state — pure ${state.model==="GLM52" ? "MLA" : "GQA"} attention — so this control is disabled.`));
  document.getElementById('wover-tip').setAttribute('data-tip', woverOk
    ? "Slack between a checkpoint's stated bytes and the server's real resident footprint. This model's figure is raw/on-disk — an under-estimate by an unknown margin; +15% is the one calibrated data point (the 27B's as-deployed 28.8 GiB vs its raw params), transferred here as an extrapolation."
    : "The 27B's 28.8 GiB is already the measured AS-DEPLOYED footprint — the +15% was derived from it, so applying it here would double-count. Its NVFP4 figure is a measured checkpoint total under the same convention, so the knob stays disabled on this model.");
  const modelTips = {
    "27B": "Qwen3.8-27B — dense hybrid, the calibrated baseline (measured on Qwen3.6-27B, whose checkpoint is tensor-for-tensor the same shape): 64 layers (48 DeltaNet + 16 full-attn) → 32 KiB/token FP8 KV plus a 75 MiB recurrent state per session. 262k native context, 1M via YaRN.",
    "35BA3B": "Qwen3.6-35B-A3B — MoE, ~3B active: 40 layers (30 DeltaNet + 10 full-attn) → 10 KiB/token KV; 256 experts / 8 routed (expert-union kink at n = 32). 262k native context, 1M via YaRN.",
    "MM35": "Mistral-Medium-3.5-128B — dense, 88 uniform full-attention layers → 176 KiB/token KV (17.6× the 35B-A3B), the study's KV-hungriest model. No MTP module: speculative default 1.0× (external EAGLE draft unmeasured). Hard 262k context max.",
    "GLM52": "GLM-5.2 — 744B-A40B MLA+DSA: 47.3 KiB/token stored, but decode reads only top-2048 tokens/layer + an indexer scan. FP8 weights fit from 7×H200 / 4×B300 (recipe floor; 3×B300 by pool arithmetic alone); NVFP4 from 2×B300. 1M native context.",
    "DSV4F": "DeepSeek-V4-Flash-0731 — 284B-A13B, compressed sparse attention (21 CSA ratio-4 + 20 HCA ratio-128 + 2 SWA layers): only 3.4 KiB/token stored — ~10× below V3-class MLA — plus a fixed 14.9 MiB/session. The study's KV-lightest big model: fits from 2×H200 / 1×B300 (native FP4 experts; no NVFP4 variant). 1M native context.",
    "Q38FN": "Qwen3.8-Flash-Next — MoE 125B-A6B (+51B FP8 n-gram table): 48 layers (36 DeltaNet + 12 QSA sparse attention) → 12.4 KiB/token stored, decode reads top-2048 tokens + an indexer scan; 512 experts / 10 routed (deepest expert-union kink, n = 51.2) and a 56.8 MiB bf16 state per session. Fits from 2×H200; 1×B300 by pool arithmetic (recipe floor TP2 — TP1 hit compilation OOM on GB300). No NVFP4 variant. 262k native context, 1M via YaRN.",
    "GLM53F": "GLM-5.3-Flash — MoE 320B-A18B hybrid (34 KDA linear + 11 NoPE sparse-MLA layers, +1 MTP DSA layer whose cache is charged): 6.4 KiB/token stored — 7.4× below GLM-5.2 — plus a 74.4 MiB bf16 KDA state per session; decode reads top-2048 latents + a kpool-4 compressed indexer scan. 288 experts / 8 routed (kink n = 36). Fits from 3×H200 / 2×B300 (pool arithmetic; no recipe floor published). fp8 KV is Blackwell-only, so on H200 the KV toggle locks to BF16. No NVFP4 variant. 1M native context.",
  };
  document.getElementById('model-tip').setAttribute('data-tip', modelTips[state.model]);
  document.getElementById('v-gpuname').textContent = CONFIG.GPUS[state.gpu].name;
  // per-model ranges: max_seq_len cap (mirrors check_cap_allowed) and the
  // prompt-median sliders both scale with the model's context window — the
  // 1M-context models (Qwens via YaRN, GLM-5.2) allow 4× the Mistral ranges.
  // Clamps are one-way, like the cap: switching to a smaller-context model
  // snaps the value down and does NOT restore it on switching back.
  const capEl = document.getElementById('s-cap'), capMax = capSliderMax();
  capEl.max = capMax;
  if (state.cap > capMax){ state.cap = capMax; capEl.value = capMax; }
  const ctxScale = m.max_ctx / 262144;
  for (const [id, key, base] of [['s-user_median','user_median',120],
                                 ['s-sub_median','sub_median',60]]){
    const el = document.getElementById(id), mx = Math.round(base*ctxScale);
    el.max = mx;
    if (state[key] > mx){ state[key] = mx; el.value = mx; }
  }
}

// segmented controls
document.querySelectorAll('.seg').forEach(seg=>{
  const key=seg.dataset.key;
  // #seg-split has no data-key: its buttons are rebuilt per GPU count and carry
  // their own listeners (renderSplitControl). Without this guard the generic
  // handler below would assign state[undefined].
  if (!key) return;
  seg.querySelectorAll('button').forEach(btn=>{
    btn.addEventListener('click',()=>{
      if (btn.disabled) return;
      seg.querySelectorAll('button').forEach(b=>b.setAttribute('aria-pressed', b===btn?'true':'false'));
      state[key]=btn.dataset.v;
      // switching model adopts ITS speculative-decode default (1.7x = the
      // Qwen fit / GLM transplant; 1.0x = Mistral, which has no MTP module) —
      // the label's value readout changes visibly, and the MTP tooltip
      // documents the reset behaviour
      if (key==='model'){
        state.mtp = CONFIG.MODELS[state.model].mtp;
        document.getElementById('s-mtp').value = state.mtp;
      }
      // switching GPU adopts ITS on-demand price the same way — the slider
      // names the selected part's price, so leaving it at the old part's
      // would silently re-price the new one
      if (key==='gpu'){
        state.gpuh = CONFIG.GPUS[state.gpu].eur_gpu_h;
        document.getElementById('s-gpuh').value = state.gpuh;
      }
      enforceConstraints();
      syncLabels();
      computeAndRender(false);
    });
  });
});

// sliders
const sliderMap=[
  ['s-ngpu','ngpu',v=>parseInt(v,10)],
  ['s-mtp','mtp',v=>parseFloat(v)],
  ['s-mbu','mbu',v=>parseFloat(v)],
  ['s-mfu','mfu',v=>parseFloat(v)],
  ['s-ram','ram',v=>parseInt(v,10)],
  ['s-cap','cap',v=>parseInt(v,10)],
  ['s-user_median','user_median',v=>parseFloat(v)],
  ['s-user_sigma','user_sigma',v=>parseFloat(v)],
  ['s-sub_median','sub_median',v=>parseFloat(v)],
  ['s-sub_sigma','sub_sigma',v=>parseFloat(v)],
  ['s-sub_ratio','sub_ratio',v=>parseFloat(v)],
  ['s-sys','sys',v=>parseFloat(v)],
  ['s-inval','inval',v=>parseFloat(v)],
  ['s-users','users',v=>parseInt(v,10)],
  ['s-think','think',v=>parseInt(v,10)],
  ['s-sla','sla',v=>parseInt(v,10)],
  ['s-decode_floor','decode_floor',v=>parseInt(v,10)],
  ['s-turn','turn',v=>parseInt(v,10)],
  ['s-out','out',v=>parseInt(v,10)],
  ['s-burst','burst',v=>parseInt(v,10)],
  ['s-ekwh','ekwh',v=>parseFloat(v)],
  ['s-gpuh','gpuh',v=>parseFloat(v)],
];
sliderMap.forEach(([id,key,cast])=>{
  const el=document.getElementById(id);
  // associate the field's visible label with its slider for AT/click-to-focus
  const lab=el.closest('.field')?.querySelector('label.flabel');
  if (lab) lab.setAttribute('for', id);
  // the GPU count changes which TP widths are legal, so it must re-run the
  // constraint pass (clamp tp, rebuild the split buttons) before rendering.
  // Other sliders don't touch the topology — kept off the per-frame drag path.
  el.addEventListener('input',()=>{
    state[key]=cast(el.value);
    if (key==='ngpu') enforceConstraints();
    onInput();
  });
});
// metric/control explainer tooltips are keyboard-reachable (CSS shows them on
// :focus-visible); tile tips are stamped in renderTiles' generated markup
document.querySelectorAll('.tip').forEach(t=>{ t.tabIndex=0; });
document.getElementById('t-sub_shares_prefix').addEventListener('change',e=>{
  state.sub_shares_prefix=e.target.checked; computeAndRender(false);
});
// frontier detail columns: display-only, re-renders from the cached rows
document.getElementById('t-ceilcols').addEventListener('change',e=>{
  state.showCeil = e.target.checked;
  if (lastFrontierRows) renderFrontierTable(lastFrontierRows, lastFrontierCurKey);
});
// deploy-card copy button lives in re-generated markup, so delegate the click
document.getElementById('deployCard').addEventListener('click',e=>{
  if (e.target.id !== 'copyCmd' || !deployCmdText) return;
  // navigator.clipboard is UNDEFINED (sync TypeError, not a rejection) on
  // insecure origins — guard before touching it
  if (!navigator.clipboard?.writeText){
    e.target.textContent = 'blocked';
    setTimeout(()=>{ e.target.textContent = 'copy'; }, 1200);
    return;
  }
  navigator.clipboard.writeText(deployCmdText).then(()=>{
    e.target.textContent = 'copied';
    setTimeout(()=>{ e.target.textContent = 'copy'; }, 1200);
  }).catch(()=>{
    // rejected in non-secure contexts / unfocused documents — say so
    e.target.textContent = 'blocked';
    setTimeout(()=>{ e.target.textContent = 'copy'; }, 1200);
  });
});
// burst presets. "= full flush" is the correlated event limitation 8 names —
// a prompt-template deploy colding the WHOLE resident population at once — so
// it reads the current warm count rather than any fixed number.
document.querySelectorAll('.preset button[data-burst]').forEach(b=>{
  b.addEventListener('click',()=>{
    const v = b.dataset.burst;
    const flush = lastWarmCur ? Math.round(lastWarmCur.p5) : 32;
    const el = document.getElementById('s-burst');
    state.burst = clip(v==='flush' ? flush : parseInt(v,10),
                       parseInt(el.min,10), parseInt(el.max,10));
    el.value = state.burst;
    syncLabels(); computeAndRender(false);
  });
});

function syncLabels(){
  // sliders follow state, not the other way round: without this a change made
  // in code (a preset button, a future URL-state feature) leaves the thumb
  // parked at its old position while the readout moves
  for (const [id,key] of sliderMap.map(([i,k])=>[i,k])){
    const el=document.getElementById(id);
    if (el && String(state[key]) !== el.value) el.value = state[key];
  }
  document.getElementById('v-users').textContent=fmt(state.users,0);
  document.getElementById('v-think').textContent=fmt(state.think,0);
  document.getElementById('v-sla').textContent=fmt(state.sla,0);
  document.getElementById('v-decode_floor').textContent=fmt(state.decode_floor,0);
  // chart C's dashed guide lines move with the slider, so its caption has to
  // name the thresholds actually drawn rather than the study's 40/50
  const csC = document.getElementById('cs-C');
  if (csC) csC.innerHTML = 'p50 line, p5\u2013p95 band, <b>log</b> axis. Shaded = the '
    + 'GPU-resident warm-capacity zone; dashed = the ' + fmt(decodeFloor(),0)
    + ' tok/s floor and ' + fmt(decodeComfort(),0) + ' tok/s comfortable mark.';
  document.getElementById('v-turn').textContent=fmt(state.turn,0);
  document.getElementById('v-out').textContent=fmt(state.out,0);
  document.getElementById('v-burst').textContent=fmt(state.burst,0);
  // the derived quantity the whole planner runs on — shown next to the users
  // slider so the 64 / 30 s = 2.13 req/s reference is never a hidden constant
  document.getElementById('v-rate').textContent=
    requestRate(state.users, state.think).toFixed(2);
  document.getElementById('v-ngpu').textContent=fmt(state.ngpu,0);
  document.getElementById('v-gpuname').textContent=CONFIG.GPUS[state.gpu].name;
  document.getElementById('v-mtp').textContent=state.mtp.toFixed(2);
  document.getElementById('v-mbu').textContent=state.mbu.toFixed(2);
  document.getElementById('v-mfu').textContent=state.mfu.toFixed(2);
  // invert speedup = 1 + a + a^2 (MTP-2, accept-until-reject) for the implied
  // per-draft acceptance — the base quantity the speedup is computed from
  const acc = state.mtp<=1 ? 0 : (Math.sqrt(4*state.mtp-3)-1)/2;
  // α inverts the 2-draft MTP model — exact for the two 2-draft Qwen models only.
  // GLM-5.2 drafts 5 tokens, DSv4-Flash 7, Q3.8-Flash-Next 3, and Mistral
  // has no MTP module (its slider models EAGLE) — indicative on all of those.
  const aNote = state.model==="GLM52" ? ' GLM-5.2 drafts 5 tokens — α is indicative only.'
              : state.model==="DSV4F" ? ' DSv4-Flash drafts 7 tokens (DSpark) — α is indicative only.'
              : state.model==="Q38FN" ? ' Q3.8-Flash-Next drafts 3 tokens (vLLM recipe) — α is indicative only.'
              : state.model==="GLM53F" ? ' GLM-5.3-Flash drafts 5 tokens (vLLM recipe) — α is indicative only.'
              : state.model==="MM35" ? ' Mistral has no MTP module — the slider models an EAGLE-style speedup; α does not apply.' : '';
  const aLine = (state.mtp<=1 ? 'Currently OFF — implied per-draft acceptance α = 0%.'
    : `Current implied per-draft acceptance α ≈ ${(acc*100).toFixed(0)}%.`) + aNote;
  document.getElementById('mtp-tip').setAttribute('data-tip',
    'Speculative-decode speedup (1.0× = off). With 2 draft tokens speedup = 1 + α + α², so 2.94× ⇔ α ≈ 98% — the 27B\'s measured accepted length (research/decode_mbu.md: per-position acceptance 0.971 then 0.944, which the two-draft inversion rounds up to 98%; 1.7× ⇔ α ≈ 47% was the pre-measurement fit); switching model resets the slider to that model\'s default. Applies to the selected configuration only. ' + aLine);
  // MBU provenance, stated on the control: one measurement, applied to every
  // row, the frontier included -- the slider is how a reader argues with it.
  const mbuProv = state.model === "27B"
    ? 'Measured on this model (research/decode_mbu.md): the uncalibrated roofline was ~4× optimistic; this constant brings it to a median 13% error.'
    : (is_moe(CONFIG.MODELS[state.model]) || CONFIG.MODELS[state.model].deltanet_state > 0
        ? 'Measured on the 27B and applied here unchanged; unmeasured on this model.'
        : 'Measured on a recurrent hybrid and applied here unchanged; this model has no recurrence to serialise and may well sit higher, but nothing in this study measures a dense one.');
  document.getElementById('mbu-tip').setAttribute('data-tip',
    'Memory Bandwidth Utilisation — the fraction of the group\'s advertised HBM bandwidth a decode pass achieves. Decode\'s counterpart to the prefill MFU below, and the study\'s newest constant: before 2026-08-28 decode was priced at 1.00, a pure roofline. MODEL CONVENTION (already divided by tp_efficiency). ONE value for every row, the frontier included: a global sensitivity knob, not a per-model property — drag it and watch which configurations survive. It absorbs both streaming inefficiency and whatever speculative verification costs on a sequential recurrence, so on the 27B it TRAVELS WITH the MTP slider. ' + mbuProv);
  document.getElementById('v-ram').textContent=fmt(state.ram,0);
  document.getElementById('v-cap').textContent=fmt(state.cap,0);
  // the 0.1k step is only reachable at the chat end of the range, so show the
  // decimal only when there is one — the reference 31k must not read "31.0k"
  document.getElementById('v-user_median').textContent=
    fmt(state.user_median, Number.isInteger(state.user_median) ? 0 : 1);
  document.getElementById('v-user_sigma').textContent=state.user_sigma.toFixed(2);
  document.getElementById('v-sub_median').textContent=fmt(state.sub_median,0);
  document.getElementById('v-sub_sigma').textContent=state.sub_sigma.toFixed(2);
  const r=state.sub_ratio;
  // keep this readout short — it shares one grid cell with its label; the
  // r/(1+r) mix share is spelled out in the control's tooltip
  const perTxt = r>0 ? ` · 1 per ${fmt(1/r,0)}` : ' · none';
  document.getElementById('v-sub_ratio').textContent=r.toFixed(2)+perTxt;
  document.getElementById('v-sys').textContent=fmt(state.sys,0);
  document.getElementById('v-inval').textContent=state.inval.toFixed(1);
  document.getElementById('v-ekwh').textContent=state.ekwh.toFixed(2);
  document.getElementById('v-gpuh').textContent=state.gpuh.toFixed(2);
}

/* ---- share link: the whole configuration in the URL fragment ------------
   Encodes only the DIFFS from STATE_DEFAULTS, so a default page shares as a
   bare URL. The fragment (not the query string) keeps every permutation on
   one cached asset and out of server logs. Decode VALIDATES everything: enum
   keys against their real option sets, numbers clamped to each slider's own
   min/max — a mangled link degrades to the nearest legal state instead of
   rendering garbage, and enforceConstraints() then applies the cross-control
   rules (tp | ngpu, NVFP4 gate, per-model caps) exactly as a click would. */
const URL_ENUMS = {
  model: () => Object.keys(CONFIG.MODELS),
  gpu:   () => Object.keys(CONFIG.GPUS),
  wdt:   () => ["fp8","nvfp4"], kv: () => ["fp8","fp16"],
  state_dt: () => ["bf16","fp32"], wover: () => ["pub","p15"],
  pue: () => ["1.2","1.5","2.0"],
  chunk: () => ["2048","4096","8192","16384","32768","65536"],
};
const URL_BOOLS = ["sub_shares_prefix", "showCeil"];
// numeric keys ride the slider map where a slider exists; tp has none
const URL_EXTRA_NUM = { tp: [1, 8] };
export function encodeStateURL(){
  const p = new URLSearchParams();
  const put = (k, v) => { if (String(v) !== String(STATE_DEFAULTS[k])) p.set(k, String(v)); };
  for (const k of Object.keys(URL_ENUMS)) put(k, state[k]);
  for (const [, k] of sliderMap) put(k, state[k]);
  // mtp diffs against the SELECTED model's own default, not the global one:
  // the decoder resets an absent mtp to CONFIG.MODELS[model].mtp, so diffing
  // globally silently flipped a 1.7x Mistral link back to that model's 1.0x
  if (String(state.mtp) !== String(CONFIG.MODELS[state.model].mtp)) p.set('mtp', String(state.mtp));
  else p.delete('mtp');
  // gpuh likewise diffs against the SELECTED part's list price: the decoder
  // re-seeds an absent gpuh from CONFIG.GPUS[gpu], so a B300 link at the
  // B300 median must stay a bare link
  if (String(state.gpuh) !== String(CONFIG.GPUS[state.gpu].eur_gpu_h)) p.set('gpuh', String(state.gpuh));
  else p.delete('gpuh');
  for (const k of Object.keys(URL_EXTRA_NUM)) put(k, state[k]);
  // booleans encode as 1/0, so the diff test must compare the BOOLEANS —
  // comparing "0" to "false" would stamp the key into every default link
  for (const k of URL_BOOLS) if (state[k] !== STATE_DEFAULTS[k]) p.set(k, state[k] ? '1' : '0');
  const q = p.toString();
  return location.origin === "null"   // file:// — origin is unusable
    ? location.href.split('#')[0] + (q ? '#' + q : '')
    : location.origin + location.pathname + (q ? '#' + q : '');
}
function applyURLState(){
  const h = location.hash.replace(/^#/, '');
  if (!h) return;
  let p; try { p = new URLSearchParams(h); } catch (e) { return; }
  for (const [k, opts] of Object.entries(URL_ENUMS)){
    const v = p.get(k);
    if (v !== null && opts().includes(v)) state[k] = v;
  }
  // a shared model carries its own MTP default unless the link pins one —
  // the same reset the model buttons apply on click
  if (p.get('model') !== null && p.get('mtp') === null)
    state.mtp = CONFIG.MODELS[state.model].mtp;
  // and a shared GPU carries its own price unless the link pins one
  if (p.get('gpu') !== null && p.get('gpuh') === null)
    state.gpuh = CONFIG.GPUS[state.gpu].eur_gpu_h;
  // the prompt-median sliders get model-scaled maxima (up to 4× the HTML
  // attribute, ctxScale) from enforceConstraints AFTER this runs — clamp
  // against the widest legal range here and let the constraint pass do the
  // per-model clamp. cap's real ceiling is capSliderMax() (1,049 at a 1M
  // context — NOT 4 × 262): state.model is already applied above, so the
  // exact per-model maximum is known here.
  const SCALED = { 'user_median': 4, 'sub_median': 4 };
  for (const [id, k, cast] of sliderMap){
    const v = p.get(k);
    if (v === null) continue;
    const el = document.getElementById(id), n = cast(v);
    if (Number.isFinite(n))
      state[k] = clip(n, parseFloat(el.min),
                      k === 'cap' ? capSliderMax() : parseFloat(el.max) * (SCALED[k] || 1));
  }
  for (const [k, [lo, hi]] of Object.entries(URL_EXTRA_NUM)){
    const v = p.get(k);
    if (v === null) continue;
    const n = parseFloat(v);
    if (Number.isFinite(n)) state[k] = k === 'tp' ? Math.round(clip(n, lo, hi)) : clip(n, lo, hi);
  }
  for (const k of URL_BOOLS) if (p.get(k) !== null) state[k] = p.get(k) === '1';
}
// reflect enum state into the two segmented controls enforceConstraints does
// not manage (it owns wdt/kv/state/wover; model and gpu are click-only)
function syncEnumSegs(){
  for (const [segId, key] of [['seg-model','model'], ['seg-gpu','gpu'], ['seg-pue','pue']])
    document.querySelectorAll(`#${segId} button`).forEach(
      b => b.setAttribute('aria-pressed', b.dataset.v === state[key] ? 'true' : 'false'));
  document.getElementById('t-sub_shares_prefix').checked = state.sub_shares_prefix;
  document.getElementById('t-ceilcols').checked = state.showCeil;
}
document.getElementById('shareBtn').addEventListener('click', e => {
  const url = encodeStateURL();
  // keep the address bar in step with what was just copied (no history entry)
  try { history.replaceState(null, '', url); } catch (err) { /* file:// */ }
  const txt = document.getElementById('shareTxt'), old = 'Share this configuration';
  const done = ok => { txt.textContent = ok ? 'Link copied' : url === location.href ? 'Link is in the address bar' : 'Copy blocked — use the address bar';
                       setTimeout(() => { txt.textContent = old; }, 1800); };
  if (!navigator.clipboard?.writeText){ done(false); return; }
  navigator.clipboard.writeText(url).then(() => done(true)).catch(() => done(false));
});

/* ---- theme toggle ---- */
const themeBtn=document.getElementById('themeBtn');
function currentTheme(){
  const attr=document.documentElement.getAttribute('data-theme');
  if(attr) return attr;
  return matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light';
}
function applyThemeLabel(){
  const t=currentTheme();
  document.getElementById('themeIco').textContent = t==='dark'?'☾':'☀';
  document.getElementById('themeTxt').textContent = t==='dark'?'Switch to light':'Switch to dark';
  themeBtn.setAttribute('aria-label', t==='dark'?'Switch to light theme':'Switch to dark theme');
}
// repaint every panel from the LAST computed results (no new MC draws — a
// theme change must not re-roll the numbers); keeps the no-fit empty state
function redrawCharts(){
  if(!lastWarmAll) return;
  renderChartA(); renderChartB(lastWarmAll);
  if(!lastDC) return;
  if (kv_pool_tokens(activeModel(),currentTopo()) <= 0){
    renderNoFit('chartC'); renderNoFit('chartD');
    clearChartGeomCD();
  } else {
    renderChartC(lastDC,{p5:lastWarmCur.g5,p95:lastWarmCur.g95},lastStress,lastSteady);
    renderChartD(lastDC,lastStress,unionKink(activeModel()),lastSteady);
  }
  renderChartE(lastChartE);   // re-render from cached series, no new draws
  redrawChartECompanions();
  renderTiles(activeModel(),currentTopo(),currentWL(),lastDC,lastWarmCur,lastStress,lastCS,lastSteady);
  // the frontier bakes chart-palette colors (PLANNER_COLORS -> cssv) into
  // inline styles, so a theme flip must repaint it; the deploy card only uses
  // var() colors but is repainted too so its copy-button state never goes stale
  if (lastFrontierRows){
    renderFrontierTable(lastFrontierRows, lastFrontierCurKey);
    renderFrontierChart(lastFrontierRows, lastFrontierCurKey);
  }
  if (lastDeploy) renderDeployCard(lastDeploy.op, lastDeploy.model, lastDeploy.topo,
                                   lastDeploy.wl, lastDeploy.mo, lastDeploy.decodeUsers);
  // the flip panel bakes chart-palette colors into the SVG, so repaint it
  // from the cached sweep (no recompute — a theme flip must not re-roll)
  if (lastFlipAxes) renderFlipPanel(lastFlipAxes);
}
themeBtn.addEventListener('click',()=>{
  const next = currentTheme()==='dark'?'light':'dark';
  document.documentElement.setAttribute('data-theme',next);
  applyThemeLabel();
  redrawCharts();
});
matchMedia('(prefers-color-scheme: dark)').addEventListener('change',()=>{
  if(!document.documentElement.getAttribute('data-theme')){ applyThemeLabel(); redrawCharts(); }
});

/* ---- hover setup ---- */
setupHover('chartC','ttC',()=>chartCGeom,(n,dc,g)=>{
  const s1=cssv('--s1'), L=g.ylog;
  return `<div class="tth tnum">max_num_seqs = ${n}</div>`+
    `<div class="row"><span class="sw" style="background:${s1}"></span>p50 <b class="tnum">${fmt(interpAt(dc,'p50',n,L),1)}</b> tok/s</div>`+
    `<div class="row"><span class="sw" style="background:${s1};opacity:.4"></span>p5–p95 <b class="tnum">${fmt(interpAt(dc,'p5',n,L),1)}–${fmt(interpAt(dc,'p95',n,L),1)}</b> tok/s</div>`;
}, (n,dc,g)=>{
  const s1=cssv('--s1'), L=g.ylog;
  return [{v:interpAt(dc,'p50',n,L), color:s1, r:4},   // anchor: the p50 curve
          {v:interpAt(dc,'p95',n,L), color:s1, r:2.4},
          {v:interpAt(dc,'p5', n,L), color:s1, r:2.4}];
});
setupHover('chartD','ttD',()=>chartDGeom,(n,dc)=>{
  const s3=cssv('--s3');
  return `<div class="tth tnum">max_num_seqs = ${n}</div>`+
    `<div class="row"><span class="sw" style="background:${s3}"></span>aggregate <b class="tnum">${fmt(interpAt(dc,'agg',n)/1000,2)}</b> ktok/s</div>`;
}, (n,dc)=>[{v:interpAt(dc,'agg',n), color:cssv('--s3'), r:4}]);

// chart H is a scatter, so the crosshair snaps to the NEAREST dot rather
// than to an x: the hit radius is ~24 css px around each mark, not the
// 10 px dot itself (the ring is part of the target)
(function(){
  const box=document.getElementById('chartH'), tt=document.getElementById('ttH');
  if (!box || !tt) return;
  function leave(){ tt.style.opacity=0; const svg=box.querySelector('svg'); if(svg) removeCross(svg); }
  function handler(e){
    const geom=frontierChartGeom; if(!geom) return;
    const svg=box.querySelector('svg'); if(!svg) return;
    const rect=svg.getBoundingClientRect();
    const scale=rect.width/geom.W;                     // css px per viewBox unit
    const vx=(e.clientX-rect.left)/scale, vy=(e.clientY-rect.top)/scale;
    let best=null, bd=Infinity;
    for (const p of geom.pts){ const d=Math.hypot(p.x-vx, p.y-vy); if (d<bd){ bd=d; best=p; } }
    if (!best || bd > Math.max(12, 24/scale)){ leave(); return; }
    drawCross(svg, best.x, geom, [{ y: best.y, color: best.color, r: 6 }]);
    const r=best.r, on=geom.par.has(r);
    const cen = r.op.binding==='decode' && r.censored ? '≥ ' : '';
    tt.innerHTML=`<div class="tth">${esc(frontierRowName(r))}${r.key===geom.curKey?' — yours':''}</div>`
      +`<div class="row"><span class="sw" style="background:${best.color}"></span>binds on ${esc(PLANNER_LABEL[r.op.binding])} · ${on?'efficient':'dominated'}</div>`
      +`<div class="row">Terminal-Bench 2.1 <b class="tnum">${fmt(frontierScore(r)*100,1)}%</b></div>`
      // every plotted row carries the load (renderFrontierChart draws `live`
      // only), so the seat price always exists
      +`<div class="row">€/seat/month, full <b class="tnum">${fmt(r.eurSeat,2)}</b></div>`
      +`<div class="row">€/user/month at your load <b class="tnum">${fmt(r.eur/Math.max(1,state.users),2)}</b></div>`
      +`<div class="row">€/month <b class="tnum">${fmt(r.eur,0)}</b> · max users <b class="tnum">${cen}${fmt(r.op.limit,0)}</b></div>`;
    const par=tt.offsetParent||box, parRect=par.getBoundingClientRect();
    const cx=(rect.left-parRect.left)+best.x*scale, cy=(rect.top-parRect.top)+best.y*scale;
    const tw=tt.offsetWidth, th=tt.offsetHeight, pad=4, gap=12;
    let left=cx+gap; if (left+tw>parRect.width-pad) left=cx-gap-tw;
    let top=cy-th-gap; if (top<pad) top=cy+gap;
    tt.style.left=Math.max(pad,Math.min(left,parRect.width-tw-pad))+'px';
    tt.style.top =Math.max(pad,Math.min(top, parRect.height-th-pad))+'px';
    tt.style.opacity=1;
  }
  box.addEventListener('mousemove', handler);
  box.addEventListener('mouseleave', leave);
  box.addEventListener('touchstart',(e)=>{ if(e.touches[0]) handler(e.touches[0]); },{passive:true});
  box.addEventListener('touchmove',(e)=>{ if(e.touches[0]) handler(e.touches[0]); },{passive:true});
  box.addEventListener('touchend', leave); box.addEventListener('touchcancel', leave);
})();

/* ---- init ---- */
// the load-time checks first, in the order they were written: unitChecks
// draws nothing, prefillSampledChecks draws from the RNG's initial state (no
// module draws at evaluation time, so it sees the same seed the single file
// gave it), steadyChecks seeds explicitly
unitChecks();
prefillSampledChecks();
steadyChecks();
applyURLState();       // a shared link restores every control before first paint
syncEnumSegs();
enforceConstraints();
syncLabels();
applyThemeLabel();
computeAndRender(false);
