import { CONFIG, PREFILL_MFU_HI, PREFILL_MFU_LO, kv_pool_tokens, makeGrid, makeTopo,
         minTpFor, servableKv, unionKink, withKvDtype } from './config.js';
import { breakevenMissRate, coldRequestSeconds, contextStats, decodeComfort,
         decodeFloor, maxUsersDecode, missContextSeconds, operatingPoint, prefillChunk,
         prefillContextSeconds, prefillSeconds, prefillServiceMoments, serverRate,
         setLiveThink, setLiveTurn, spikeMetrics, steadyDecodePoint } from './prefill.js';
import { samplingSig, seedFor } from './mathlib.js';
import { p_sub } from './workload.js';
import { decodeCurves, decodeMbu, decodePlan, warmCapacity } from './capacity.js';
import { currentTopo, currentWL, ramPerCache, state } from './state.js';
import { cssv, esc, fmt } from './svg.js';
import { chartEData, clearChartGeomCD, interpAt, renderChartA, renderChartB,
         renderChartC, renderChartD, renderChartE, renderChartECompanions, renderNoFit } from './charts.js';
import { bStar, fAxisMax, plannerData, renderBindingChart, renderCeilingBars,
         renderSpikeChart, renderSpikeTiles, warmUsersCurve } from './planner.js';
import { renderTestCard } from './harness.js';
import { energyCost, renderCostCard } from './cost.js';
import { computeFlipData, lastFlipAxes, renderFlipPanel, setLastFlipAxes } from './sensitivity.js';
import { renderDeployCard } from './deploy.js';
import { renderFrontierChart, renderFrontierTable } from './frontier.js';

/* ============================================================================
   RENDER pipeline
   ========================================================================== */
export let lastDC=null, lastWarmCur=null, lastWarmAll=null, lastStress=null, lastCS=null;
// the chosen load's decode point — recomputed every frame from lastDC
export let lastSteady=null;

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
  // FP16 KV is not servable for GLM-5.3 (DSA asserts a quantized cache);
  // the UI disables the toggle — withKvDtype's guard keeps the math honest
  // anyway (and prices the sparse-decode doubling on Q38FN). servableKv
  // first resolves GPU-coupled dtypes (GLM-5.3-Flash: BF16 forced on H200),
  // so no code path — the frontier included — ever prices an unservable arm.
  m = withKvDtype(m, servableKv(m, state.kv, state.gpu));
  // fp32 recurrent state. No-op on the pure-attention models (MM35, GLM-5.3
  // carry no state at all) and on DSv4-Flash (its per-session state is a
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
  return m;
}

function syncState(){ setLiveTurn(state.turn); setLiveThink(state.think); }

export function computeAndRender(draft, deferFrontierDecode){
  syncState();
  const model=activeModel();
  const topo=currentTopo();
  const wl=currentWL();
  const q = draft ? CONFIG.DRAFT : CONFIG;
  const warmIter = q.WARM_ITER;

  // Chart A (analytic, cheap)
  renderChartA();

  // current configuration: warm capacity of ONE cache (DP: per replica).
  // g5/g95 = the HBM-resident subset (excludes CPU-offloaded sessions), which
  // is what any decode-concurrency readout must use.
  seedFor('warm');
  const wc=warmCapacity(model,topo,wl,ramPerCache(topo),warmIter,q.WARM_BUDGET);
  const [p5,p50,p95]=wc.all, [g5,g50,g95]=wc.gpu, [o5,o50,o95]=wc.off;
  lastWarmCur={p5,p50,p95, g5,g50,g95, o5,o50,o95, censored:wc.censored,
               system5:p5*topo.replicas, topo};

  // Chart B: warm p5 vs number of GPUs (TP one cache vs DP system total),
  // under the SAME offload setting as the current config (TP: whole buffer to
  // the one cache; DP: buffer split across the N replicas). Larger pools hold
  // more sessions, so fewer MC iterations suffice per point.
  // Chart B is 12 warm fills — by far the most expensive panel at large pools.
  // While a slider is being DRAGGED we keep the last settled scan and only move
  // its "you" marker; the full scan runs when the drag settles (scheduleFull's
  // 120 ms debounce). Everything else still updates live.
  const scaleNs=[1,2,3,4,6,8];
  let sc;
  if (draft && lastWarmAll && lastWarmAll.ns.length===scaleNs.length){
    sc=lastWarmAll;                       // reuse the last settled scan
  } else {
    const dpIt=Math.max(60, Math.round(warmIter/4));
    const scanB=q.WARM_BUDGET_SCAN;
    seedFor('scanB');
    const tp5=[],tp95=[],dp5=[];
    for(const n of scaleNs){
      const it=Math.max(80, Math.round(warmIter/n));
      const [a,,c]=warmCapacity(model,makeTopo("tp",n,state.gpu),wl,state.ram,it,scanB).all;
      tp5.push(a); tp95.push(c);
      const [r5]=warmCapacity(model,makeTopo("dp",n,state.gpu),wl,state.ram/n,dpIt,scanB).all;
      dp5.push(r5*n);
    }
    sc={ns:scaleNs, tp5, tp95, dp5};
  }
  sc={...sc, cur:{n:state.ngpu, split:`DP${state.ngpu/state.tp}×TP${state.tp}`,
                  p5system:lastWarmCur.system5}};
  lastWarmAll=sc;
  renderChartB(sc);

  // Decode curves for current topo, over an axis sized to reach the stress
  // point (all GPU-resident warm sessions decoding at once).
  const plan=decodePlan(lastWarmCur.g95, draft);
  seedFor('decode');
  const dc=decodeCurves(model,topo,wl,plan.nMax,plan.step,plan.iter);
  lastDC=dc;
  // the stress point itself: shared by both charts and both tiles, so it is
  // computed once here rather than three times with three different draws
  lastStress=stressPoint(dc,topo,lastWarmCur);
  // ...and the point the CHOSEN LOAD actually sits at. Pure interpolation over
  // the curves already drawn, so it costs nothing and can be recomputed on
  // every frame — including while the users / think / output sliders are being
  // dragged, which are exactly the knobs that move it.
  lastSteady=steadyDecodePoint(dc, topo,
                serverRate(state.users, state.think, wl.sub_ratio)
                  / (topo.replicas || 1),
                state.out);
  // the C zone is a DECODE-concurrency span -> GPU-resident sessions only
  const noFit = kv_pool_tokens(model,topo) <= 0;

  // ONE context-length draw feeds chart E and every prefill readout on the
  // tiles, so the two can never disagree within a render. Drawn BEFORE the
  // charts because the steady point's validity predicate needs the prefill
  // duty, which needs these samples — the draw is seeded, so its position in
  // the render order does not change what it returns.
  seedFor('context');
  lastCS = noFit ? null : contextStats(wl);

  // Whether the steady point EXISTS, decided once and stamped on the shared
  // object. There is no steady state when prefill duty >= 1 (the queue is
  // unbounded, so requests never reach the decode batch at all) or when the
  // demand runs past the sampled axis. The tiles already refused to quote a
  // figure in those cases; the charts and the deploy card each carried their
  // own weaker `n > 0` test and published one anyway — the deploy card being
  // the copy-paste artifact, which is the worst place to invent a number.
  // One predicate, so the three cannot disagree again.
  lastSteady.real = !noFit && !lastSteady.saturated
    && (spikeMetrics(model, topo, wl, lastCS,
                     serverRate(state.users, state.think, wl.sub_ratio)
                       / (topo.replicas || 1), prefillChunk()).rho < 1);

  if (noFit){
    renderNoFit('chartC'); renderNoFit('chartD');
    clearChartGeomCD();
  } else {
    renderChartC(dc, {p5:lastWarmCur.g5, p95:lastWarmCur.g95}, lastStress, lastSteady);
    renderChartD(dc, lastStress, unionKink(model), lastSteady);
  }
  updateCsD(model);

  renderChartE(noFit ? null : chartEData(model, topo, wl, lastCS, lastStress));
  renderChartECompanions(model, topo, wl, lastCS, lastStress, noFit, draft);

  renderTiles(model,topo,wl,dc,lastWarmCur,lastStress,lastCS,lastSteady);
  renderPlanner(model, topo, wl, lastCS, noFit, draft, q, deferFrontierDecode);
}

/* ---- the planner: tiles, charts F & G, the frontier table ------------------
   The expensive halves — the warm-capacity-vs-f anchors, the decode-ceiling
   bisections and the whole frontier table — run only on a SETTLED render and
   are reused while a slider is being dragged, the same split chart B uses.
   The closed-form halves (latency, saturation, B*, TTFT) are cheap and update
   live on every frame, so dragging the load or budget sliders is immediate.
   -------------------------------------------------------------------------- */
let lastPlanner = null, lastFrontier = null, lastFrontierMo = null, lastFrontierDec = null;
export let pendingLoadTiles = [];   // act-2 tiles from renderTiles, painted by the planner
function renderPlanner(model, topo, wl, cs, noFit, draft, q, deferFrontierDecode){
  if (noFit){
    renderSpikeTiles(null, null, model, topo, wl, cs, true, fitHintFor(model, topo));
    renderBindingChart(null); renderSpikeChart(null); renderCeilingBars(null);
    renderDeployCard(null); renderTestCard(null); renderCostCard(null);
    renderFlipPanel(null); setLastFlipAxes(null);
    // The frontier enumerates every configuration and does not depend on the
    // current one fitting — when the answer to "can I run this?" is no, it is
    // the only thing on screen that answers "then what?". It now carries
    // load-dependent verdicts, so it must keep updating here (cs is null on
    // the no-fit path; the workload stats it needs are fit-independent).
    // cs is null here; only draw the 20k-sample context stats when some row's
    // moments cache is actually cold — otherwise the draw is discarded
    // A signature change rebuilds the table and RESETS the moments cache
    // inside renderFrontierSection — so testing only the caches that exist
    // right now says "no samples needed" and then hands a null cs to a full
    // rebuild, leaving every row to sample its own moments off whatever the
    // RNG stream happened to be left at. That made non-decode columns
    // (latency, B*, ranking, €/mo) move when only the decode floor changed.
    const wsigNow = frontierWarmSig(wl);
    const needCs = !lastFrontier || !lastFrontierMo
                || (!draft && (lastFrontier.sig !== wsigNow
                               || lastFrontierMo.sig !== `${wsigNow}|${state.turn}`))
                || lastFrontier.base.some(r => !lastFrontierMo.mo[r.key]);
    renderFrontierSection(wl, cs || (needCs ? contextStats(wl) : null), draft, q,
                          deferFrontierDecode);
    lastPlanner = null;
    return;
  }
  const reps = topo.replicas || 1;
  const rate = serverRate(state.users, state.think, wl.sub_ratio)/reps;
  const mo = prefillServiceMoments(model, topo, wl, cs, prefillChunk());
  const moLo = prefillServiceMoments(model, topo, wl, cs, prefillChunk(), PREFILL_MFU_LO);
  const moHi = prefillServiceMoments(model, topo, wl, cs, prefillChunk(), PREFILL_MFU_HI);
  const f = wl.invalidation;
  const warmUsersNow = lastWarmCur.p5 * (1 - p_sub(wl));

  // --- the sampled halves, on settled renders only ---
  // A DRAFT render reuses the last settled results UNCONDITIONALLY (the same
  // split chart B uses): the knobs being dragged are exactly the ones that
  // would change them, so any staleness check re-runs the Monte-Carlo on
  // every input event — ~0.5 s per frame — and the 120 ms settle corrects
  // the numbers anyway.
  let warmFn, decodeUsers, decodeCensored = false;
  if (!draft || !lastPlanner){
    seedFor('decodeCeil');
    decodeUsers = maxUsersDecode(model, topo, wl, decodeFloor(), q.DECODE_ITER || 220);
    decodeCensored = decodeUsers.censored; decodeUsers = decodeUsers.n;
    seedFor('warmCurve');
    warmFn = warmUsersCurve(model, topo, ramPerCache(topo), Math.max(80, q.WARM_ITER/3),
                            q.WARM_BUDGET_SCAN, wl, f, warmUsersNow);
    lastPlanner = { warmFn, decodeUsers, decodeCensored, fMax: fAxisMax() };
  } else {
    warmFn = lastPlanner.warmFn; decodeUsers = lastPlanner.decodeUsers;
    decodeCensored = lastPlanner.decodeCensored;
  }
  // a draft render reuses the cached curve, but warmUsersCurve's anchors only
  // span the axis it was built for and it CLAMPS beyond its last anchor. So a
  // drag across the 50% boundary would draw chart G's cache line flat from
  // 0.5 to 1.0 until the 120 ms settle rebuilt it — a plausible-looking curve
  // that is simply wrong. Rebuild immediately when the axis changed under it.
  if (lastPlanner.fMax !== fAxisMax()){
    seedFor('warmCurve');
    warmFn = warmUsersCurve(model, topo, ramPerCache(topo), Math.max(80, q.WARM_ITER/3),
                            q.WARM_BUDGET_SCAN, wl, f, warmUsersNow);
    lastPlanner = { ...lastPlanner, warmFn, fMax: fAxisMax() };
  }

  const d = plannerData(model, topo, wl, cs, warmFn, decodeUsers, mo);
  const op = operatingPoint(model, topo, wl, cs, {
    mo, reps, warmUsers: warmUsersNow*reps, decodeUsers: decodeUsers*reps });
  // per-group quantities the tiles quote alongside the user ceilings
  const sp = spikeMetrics(model, topo, wl, cs, rate, prefillChunk());
  const drain = state.burst * mo.miss / Math.max(1e-9, 1 - sp.rho);
  const dm = itlSpikeRatio(model, topo, wl, cs);
  Object.assign(op, {
    bstar:   bStar(mo,   f, state.sla, rate),
    bstarLo: bStar(moLo, f, state.sla, rate),
    bstarHi: bStar(moHi, f, state.sla, rate),
    fstar:   breakevenMissRate(model, topo, wl, rate, cs, prefillChunk()),
    fsla:    sp.fsla, duty: sp.rho,
    burstDrain: sp.rho >= 1 ? Infinity : drain,
    tokensLost: (sp.rho >= 1 || !(dm.ratio > 1)) ? 0
              : drain * (1/dm.decodeS - 1/dm.mixedS),
    ttftMiss:    sp.rho >= 1 ? Infinity : (rate*(f*mo.missSq+(1-f)*mo.hitSq))/(2*(1-sp.rho)) + mo.miss,
    ttftHitFcfs: sp.rho >= 1 ? Infinity : (rate*(f*mo.missSq+(1-f)*mo.hitSq))/(2*(1-sp.rho)) + mo.hit,
    ttftHitPs:   sp.rho >= 1 ? Infinity : mo.hit/(1-sp.rho),
  });
  renderSpikeTiles(op, sp, model, topo, wl, cs, false, '');
  renderCeilingBars(op);
  renderDeployCard(op, model, topo, wl, mo, decodeUsers);
  renderTestCard(op, model, topo, wl);
  renderCostCard(op, model, topo, wl, mo, decodeUsers);
  renderBindingChart(d, op);

  // context lines on the spike chart: the same model at other widths that fit
  const others = [];
  for (const n of [1,2,4,8]){
    const t2 = makeTopo('tp', n, state.gpu);
    if (t2.name === topo.name) continue;   // skip only the CURRENT topology:
    // skipping every topology with the same GPU count hid the TP curve from
    // DP users, which is exactly the comparison they are looking for
    if (kv_pool_tokens(model, t2) <= 0) continue;
    const m2 = prefillServiceMoments(model, t2, wl, cs, prefillChunk());
    const r2 = serverRate(state.users, state.think, wl.sub_ratio);
    others.push({ fs: d.fs, b: d.fs.map(x => bStar(m2, x, state.sla, r2)) });
  }
  renderSpikeChart(d, others);
  renderFrontierSection(wl, cs, draft, q, deferFrontierDecode);

  // the sensitivity panel re-sweeps every soft input, so like the other
  // sampled halves it runs on SETTLED renders only and drafts reuse the last
  // result (the 120 ms settle corrects it)
  if (!draft || !lastFlipAxes)
    setLastFlipAxes(computeFlipData(model, topo, wl, cs, mo, warmFn, op, reps));
  renderFlipPanel(lastFlipAxes);
}

/* ---- the frontier section --------------------------------------------------
   Factored out of renderPlanner so the no-fit path can still update it: the
   table enumerates every configuration and carries verdicts at the CURRENT
   load, so it must not go stale while the selected configuration has no pool.
   Split by cost, not by render mode. The Monte-Carlo half (warm fill +
   decode-ceiling search per configuration) is keyed on a signature that
   EXCLUDES the selected model, the GPU count and every load/budget knob —
   none of which change any other row's warm capacity or decode ceiling — so
   the two most common interactions (switching model, dragging load) never
   pay for it. The closed-form half is recomputed every render. */
/* The frontier's Monte-Carlo signatures, hoisted out of renderFrontierSection
   because the no-fit path has to know whether the table is ABOUT to be rebuilt
   before it decides whether to draw context samples for it.

   TWO signatures, because the table's two sampled halves have different
   dependencies and VERY different costs. The warm fill is the expensive one
   (a Monte-Carlo fill per row) and does not depend on the decode floor at all;
   the decode search is the cheap one (a doubling bisection per row) and is the
   only thing the floor moves. Keying both on one signature — as a single
   frontierSig including the floor did — made every floor step re-run the warm
   fills for nothing, which measured ~1,000 ms per settle against ~190 ms for a
   budget knob that touches neither. Splitting them puts the floor slider back
   in the cheap class. */
function frontierWarmSig(wl){
  return `${state.gpu}|${state.wdt}|${state.kv}|${state.state_dt}|${state.wover}|`
       + `${state.cap}|${state.ram}|${state.user_median}|${state.user_sigma}|`
       + `${state.sub_median}|${state.sub_sigma}|${state.sub_ratio}|`
       + `${state.sub_shares_prefix}|${state.sys}|${wl.invalidation}`;
}
// the decode ceilings additionally move with the floor and the MBU slider —
// and ONLY they do (both are closed-form scalings of the same search)
function frontierDecSig(wl){ return `${frontierWarmSig(wl)}|${state.decode_floor}|${state.mbu}`; }
/* Chunked rebuild: a settle used to recompute every stale row in ONE task —
   measured at 500–900 ms of main-thread block once the table reached 27 rows
   (7 models) — so the page froze after each drag. The rebuild now walks the
   rows in ~40 ms time slices on the macrotask queue: the longest block is one
   slice, the table repaints when the last row lands, and a knob moved
   mid-rebuild abandons the run (the next settle starts a fresh one).
   Deliberately NOT a numeric shortcut: the per-row seeding is unchanged, so
   slicing cannot move a draw and every published number is bit-identical to
   the one-task rebuild's. Only HOW LONG the main thread blocks changes. */
const FRONTIER_SLICE_MS = 40;
let frontierGen = 0;        // bumping this abandons any in-flight rebuild
export let frontierDecodeDeferred = false;   // set here, read by the 500 ms settle tier
let frontierJobSig = null;  // signature of the in-flight rebuild, if any

function renderFrontierSection(wl, cs, draft, q, deferDecode){
  const wsig = frontierWarmSig(wl), dsig = frontierDecSig(wl);
  const msig = `${wsig}|${state.turn}`, jobSig = `${dsig}|${state.turn}`;
  const cachesComplete = () =>
    lastFrontier && lastFrontierDec && lastFrontierMo &&
    !lastFrontier.base.some(r => !lastFrontierDec.dec[r.key]
                              || !lastFrontierMo.mo[r.key]);
  // like the sampled halves above, a DRAFT render never re-runs either
  // Monte-Carlo half — the signatures contain the very knobs a drag is
  // changing, so checking them here would re-sample on every input event.
  // It re-ranks from the settled caches when they are complete, leaves the
  // last painted table alone mid-rebuild (the finishing job repaints), and
  // abandons a rebuild the drag has already invalidated.
  if (draft){
    if (frontierJobSig && frontierJobSig !== jobSig){ frontierGen++; frontierJobSig = null; }
    // stale-while-DRAGGING re-ranks are the page's design (the settle
    // corrects them); but while a settled change's rebuild is in flight the
    // warm half is a whole epoch behind — repainting would mix old-GPU row
    // labels with new-GPU prices, so hold the last paint until the commit
    if (cachesComplete() && (!frontierJobSig || lastFrontier.sig === wsig))
      assembleFrontier(wl, cs);
    return;
  }
  const warmStale = !lastFrontier    || lastFrontier.sig    !== wsig;
  const moStale   = !lastFrontierMo  || lastFrontierMo.sig  !== msig
                 || (!warmStale && lastFrontier.base.some(r => !lastFrontierMo.mo[r.key]));
  const decStale  = !lastFrontierDec || lastFrontierDec.sig !== dsig
                 || (!warmStale && lastFrontier.base.some(r => !lastFrontierDec.dec[r.key]));
  if (!warmStale && !moStale && !decStale){
    // sigs can revert mid-rebuild (a drag back to the exact starting values):
    // the caches are already right, so the leftover job must not commit —
    // and a deferral pending for the reverted floor is moot, so drop it or
    // the 500 ms tier fires one pointless full render
    if (frontierJobSig){ frontierGen++; frontierJobSig = null; }
    frontierDecodeDeferred = false;
    assembleFrontier(wl, cs);
    return;
  }
  if (frontierJobSig === jobSig){
    // this exact rebuild is already in flight — keep showing the last settled
    // table, re-ranked for whichever cheap knob (model highlight, users)
    // moved. Checked BEFORE the deferral branch: a settle while the deferred
    // decode rebuild is already running must not re-arm the flag it cannot
    // clear, or every later settle drags a spurious 500 ms full render.
    // Same epoch guard as the draft path: only re-rank when the warm half
    // still matches the live signature (decode/moments staleness is the
    // accepted stale-by-one-tier case; a stale warm half is a mixed table)
    if (cachesComplete() && lastFrontier.sig === wsig) assembleFrontier(wl, cs);
    return;
  }
  // the decode half is the only thing the floor moves. Stale is allowed for
  // exactly one 500 ms tier, and only when a previous result exists to show
  // meanwhile — never on the first paint, and never when the warm half is
  // itself stale (then the whole rebuild runs now and includes it)
  if (!warmStale && !moStale && deferDecode && lastFrontierDec){
    frontierDecodeDeferred = true;   // the 500 ms tier will pick this up
    assembleFrontier(wl, cs);        // fresh warm + stale-by-one-tier decode
    return;
  }
  startFrontierRebuild(wl, cs, q, wsig, dsig, msig, jobSig);
}

function startFrontierRebuild(wl, cs, q, wsig, dsig, msig, jobSig){
  frontierJobSig = jobSig;
  frontierDecodeDeferred = false;
  const gen = ++frontierGen;
  // the row plan is pool arithmetic only — the Monte-Carlo work is in the
  // slices. Each entry FREEZES its model arm, grid, RAM share and the floor
  // at job start: today every state mutation renders synchronously
  // (abandoning this job before the next slice can run), but a future
  // programmatic write with no render must not commit rows mixed across two
  // knob values under one signature
  const plan = [];
  const floor0 = decodeFloor();
  for (const mk of Object.keys(CONFIG.MODELS))
    for (const [dp, tp] of [[1,1],[1,2],[1,4],[1,8],[2,1],[2,2]]){
      if (dp*tp > 8) continue;
      const m2 = modelForCompare(mk), t2 = makeGrid(dp, tp, state.gpu);
      if (kv_pool_tokens(m2, t2) <= 0) continue;
      plan.push({ mk, dp, tp, key: `${mk}|${dp}x${tp}`, m2, t2, ram: ramPerCache(t2) });
    }
  // per-key reuse of whichever caches are still valid for their own signature:
  // a floor step keeps warm + moments and re-searches decode; a turn step
  // keeps warm + decode and re-derives moments (they depend on the workload
  // and the turn size only — NOT on the decode floor)
  const warmOld = (lastFrontier && lastFrontier.sig === wsig)
    ? Object.fromEntries(lastFrontier.base.map(r => [r.key, r])) : {};
  const decOld = (lastFrontierDec && lastFrontierDec.sig === dsig) ? lastFrontierDec.dec : {};
  const moOld  = (lastFrontierMo  && lastFrontierMo.sig  === msig) ? lastFrontierMo.mo  : {};
  const base = [], dec = {}, mo = {};
  // snapshot the sampling signature: rows sampled in later slices must draw
  // exactly what a one-task rebuild at job start would have drawn, even if a
  // seed-only knob (model, MTP, ngpu/tp) moves while slices are queued
  const ssig = samplingSig();
  let i = 0;
  const slice = () => {
    if (gen !== frontierGen) return;         // superseded or abandoned
    const t0 = performance.now();
    try {
    // elapsed is checked BETWEEN rows, so a slice is FRONTIER_SLICE_MS plus at
    // most one row's overshoot — the block never approaches whole-table cost
    while (i < plan.length && performance.now() - t0 < FRONTIER_SLICE_MS){
      const p = plan[i++];
      const m2 = p.m2, t2 = p.t2, reps = t2.replicas || 1;
      if (warmOld[p.key]){ base.push(warmOld[p.key]); }
      else {
        // Seed PER ROW, so a row's fill depends only on that row (and slicing
        // cannot move a draw). One stream shared across the table made every
        // row's draws depend on how many the earlier rows consumed.
        seedFor(`frontier|${p.mk}|${p.dp}x${p.tp}`, ssig);
        const wc2 = warmCapacity(m2, t2, wl, p.ram,
                                 Math.max(60, q.WARM_ITER/6), q.WARM_BUDGET_SCAN);
        base.push({ key: p.key, mk: p.mk, dp: p.dp, tp: p.tp,
                    label: `${m2.name} · ${t2.name}`, reps,
                    warmUsers: wc2.all[0]*(1-p_sub(wl))*reps });
      }
      if (decOld[p.key]){ dec[p.key] = decOld[p.key]; }
      else {
        seedFor(`frontierDec|${p.key}`, ssig);
        const du2 = maxUsersDecode(m2, t2, wl, floor0, 140);
        dec[p.key] = { decodeUsers: du2.n*reps, censored: du2.censored };
      }
      mo[p.key] = moOld[p.key] || prefillServiceMoments(m2, t2, wl, cs);
    }
    } catch (e) {
      // the one-task rebuild re-threw on every render — noisy but it never
      // latched. Match that: unlatch so the next settle retries, then rethrow
      // (a latched frontierJobSig would freeze the table for the session)
      frontierGen++; frontierJobSig = null;
      throw e;
    }
    if (i < plan.length){ setTimeout(slice, 0); return; }
    // commit all three caches together — a draft assembling mid-rebuild must
    // never see a fresh warm half spliced onto a stale decode half
    lastFrontier    = { sig: wsig, base };
    lastFrontierDec = { sig: dsig, dec };
    lastFrontierMo  = { sig: msig, mo };
    frontierJobSig = null;
    assembleFrontier(wl, cs);
  };
  slice();   // first slice runs inside the settle render, the rest queue behind it
}

// €/seat/month: the bill with the configuration carrying its max users,
// divided by them. Infinity where there is no ceiling to fill.
function seatPrice(op, topo, mo, f, wl, r){
  if (!isFinite(op.limit) || op.limit < 1) return Infinity;
  const rateMax = serverRate(op.limit, state.think, wl.sub_ratio) / r.reps;
  return energyCost(topo, mo, f, rateMax, r.decodeUsers / r.reps).totalMonth / op.limit;
}
function assembleFrontier(wl, cs){
  const f = wl.invalidation;
  const rows = lastFrontier.base.map(r0=>{
    const r = { ...r0, ...lastFrontierDec.dec[r0.key] };
    const m2 = modelForCompare(r.mk), t2 = makeGrid(r.dp, r.tp, state.gpu);
    const r2 = serverRate(state.users, state.think, wl.sub_ratio)/r.reps;
    const mo2 = lastFrontierMo.mo[r.key]
             || (lastFrontierMo.mo[r.key] = prefillServiceMoments(m2, t2, wl, cs));
    const op = operatingPoint(m2, t2, wl, cs, {
               mo: mo2, reps: r.reps,
               warmUsers: r.warmUsers, decodeUsers: r.decodeUsers });
    return { ...r, op,
             bstar: bStar(mo2, f, state.sla, r2),
             // the whole bill at YOUR load — hardware plus energy
             // (research/power.md) — comparable across rows because every
             // row is priced at the same demand and the same €/GPU-hour
             eur: energyCost(t2, mo2, f, r2, r.decodeUsers / r.reps).totalMonth,
             // and the bill at the row's OWN ceiling, per seat: what one
             // user costs when the configuration is full. At your load the
             // bill is the GPU count and every row on a topology prices the
             // same; at capacity a row that carries twice the users halves
             // the seat. Chart H's y-axis (research/power.md for the terms)
             eurSeat: seatPrice(op, t2, mo2, f, wl, r) };
  }).sort((a,b)=>b.op.limit-a.op.limit);
  const curKey = `${state.model}|${state.ngpu/state.tp}x${state.tp}`;
  renderFrontierTable(rows, curKey);
  // chart H draws EXACTLY the rows the table just painted — this is the one
  // commit point, so it can never show a half-rebuilt set
  renderFrontierChart(rows, curKey);
}

// chart D's subtitle explains the n=32 slope break, but only when the active
// model is MoE and actually has one
function updateCsD(model){
  const kink = unionKink(model);
  document.getElementById('cs-D').textContent =
    "System p50 tok/s (×replicas for DP). Where the load's output-token demand"
    + " crosses this curve is the steady-state batch size." + (kink
      ? ` Slope break at n = ${kink}: the expert-union kink — past it every routed expert is read each step.`
      : "");
}

// The study's worst case: every GPU-resident p5-warm session decoding at once.
// Read off the decode sweep at that concurrency — the axis is sized to contain
// it by construction (decodePlan), so this is never an extrapolation, it costs
// nothing, and the tile can no longer disagree with the point chart C marks
// (both are the p50 curve at the same x, interpolated the same way).
// The fit hint the tiles show, factored out so the planner panel can say the
// same thing rather than a second, subtly different thing.
export function fitHintFor(model, topo){
  const needTp = minTpFor(model, topo.gpu.name);
  return `needs TP ≥ ${needTp} on ${topo.gpu.name}`
       + (needTp <= 8 ? ` (this node has 8)` : ` — more than one 8-GPU node`);
}
// Inter-token latency with and without a prefill chunk sharing the forward
// pass, at the current stress point — section 8's ITL spike, returned in
// seconds so the planner can integrate it over a whole drain. Mirrors
// itl_spike() in scenario_model.py.
function itlSpikeRatio(model, topo, wl, cs){
  const st = lastStress;
  const pu = st && st.pu > 0 ? st.pu : 0;
  if (!(pu > 0)) return { decodeS: 0, mixedS: 0, ratio: 0 };
  const decodeS = model.mtp / pu;
  // priced mid-re-prefill: prior = E[L]/2, the average cache a chunk attends
  // over during a full cold re-prefill. A chunk larger than the mean context
  // is a single whole-context pass with no cache behind it.
  const C = prefillChunk();
  const step = Math.min(C, cs.mean), prior = step < C ? 0 : cs.mean/2;
  const mixedS = decodeS + prefillSeconds(model, topo, step, prior);
  return { decodeS, mixedS, ratio: mixedS/decodeS };
}

function stressPoint(dc, topo, warm){
  const n=Math.max(1,Math.round(warm.g5));
  const pu=interpAt(dc,'p50',n,true);
  return { n, pu, agg:n*pu*topo.replicas };
}

export function renderTiles(model,topo,wl,dc,warm,stress,cs,steady){
  const pool=kv_pool_tokens(model,topo);
  const ceiling=(1-wl.invalidation)*100;
  // Stress point (computed once in stressPoint(), marked on charts C and D):
  // every GPU-RESIDENT p5-warm session decoding at once. CPU-offloaded sessions
  // are storage — their KV lives in host RAM, so they cannot decode until they
  // are restored over PCIe. Raising the offload slider must therefore NOT change
  // per-user decode speed, only warm STORAGE.
  const st=stress || stressPoint(dc,topo,warm);
  const nOp=st.n, puOp=st.pu, aggOp=st.agg/1000;
  // a zero pool means the weights don't fit this configuration at all — a
  // decode readout evaluated at n=1 would show a meaningless green number
  const noFit = pool <= 0;
  // ...and if it doesn't fit, say what WOULD. minTpFor gets the same model
  // object the pool was computed from, so a +15% weight bump is reflected in
  // the reported threshold (see minTpFor's contract).
  const needTp = noFit ? minTpFor(model, topo.gpu.name) : 0;
  const fitHint = noFit
    ? `needs TP ≥ ${needTp} on ${topo.gpu.name}`
      + (needTp <= 8 ? ` (this node has 8)` : ` — more than one 8-GPU node`)
    : "";
  const floorClass = noFit ? undefined : (puOp>=decodeComfort()?'good':(puOp>=decodeFloor()?'warn':'crit'));
  // true whenever the cache SPLITS across >1 replica group (pure DP or a DP×TP
  // grid) — that is what makes the headline a per-replica number
  const dp = topo.replicas>1;
  // Prefill ceiling (analytic; MFU calibrated 2026-08-27, remainder
  // unvalidated — research/prefill.md). ONE
  // context-length draw (shared with chart E via computeAndRender) feeds the
  // ceiling, the p5/p95 miss costs and f*, so they stay consistent within a
  // render; the colour compares the invalidation slider f to f*. The tile is
  // priced at the SELECTED max_num_batched_tokens (32,768 unless a share link
  // pins another); the full trade curve still lives in chart E's sweep.
  const CH = prefillChunk();
  if (cs === undefined) cs = noFit ? null : contextStats(wl);
  // per-miss cost is monotone in context length, so its percentiles are the
  // priced percentiles of the length distribution
  // expected miss cost prices the heavy tail (E[L^2]); the p5/p95 costs are
  // the priced percentiles of the length draw (cost is monotone in length)
  const coldS   = noFit ? 0 : coldRequestSeconds(model, topo, wl, cs, CH);
  const coldP5  = noFit ? 0 : missContextSeconds(model, topo, cs.p5,  CH);
  const coldP95 = noFit ? 0 : missContextSeconds(model, topo, cs.p95, CH);
  // a warm hit's new turn attends over the whole cached context; MARGINAL
  // pricing (no per-pass overhead) — it rides the serving steady state
  const warmS = noFit ? 0 : prefillContextSeconds(model, topo, state.turn, CH, cs.mean);
  const coldRate = noFit ? 0 : 1/coldS;
  // the load is SYSTEM-wide; each of the `replicas` groups takes its share, so
  // f* for a DP grid is solved at the per-group rate. Driven by the Concurrent
  // users / Think time controls since 2026-08-03. TOTAL rate incl. subagent
  // tow (2026-08-04): at the defaults (64 users, 30 s, r = 0.10) this is
  // 2.35 req/s — the doc's § 8 tables stay parameterized at 2.13 req/s TOTAL,
  // which corresponds to ~58 users here, not 64.
  const groupRate = serverRate(state.users, state.think, wl.sub_ratio)
                    / (topo.replicas || 1);
  const fstar = noFit ? 0 : (1/groupRate - warmS)/(coldS - warmS);
  const prefillClass = noFit ? undefined
    : (wl.invalidation >= fstar ? 'crit'
      : (wl.invalidation >= fstar/2 ? 'warn' : 'good'));
  // the ITL spike (mirrors itl_spike): one chunk lands in the decode batch,
  // and every decoder at the stress point sees one inter-token gap of
  // step-time + chunk-time instead of step-time (a forward pass yields ~mtp
  // tokens, so step-time = mtp / per-user speed)
  const stepS = (noFit || !(puOp > 0)) ? 0 : model.mtp / puOp;
  // the chunk is priced mid-re-prefill (prior = E[L]/2, the average cache it
  // attends over); the LAST chunk of a mean context carries ~2x this cross
  // term. A chunk larger than the mean context is one whole-context pass.
  const spikeStep = stepS > 0 ? Math.min(CH, cs.mean) : 0;
  const spike = stepS > 0
    ? (stepS + prefillSeconds(model, topo, spikeStep,
                              spikeStep < CH ? 0 : cs.mean/2)) / stepS : 0;
  // cold-SPIKE tolerance (research/spike.md): the duty cycle above is a mean
  // rate against a mean service time, and misses arrive in clumps
  const sp = noFit ? null : spikeMetrics(model, topo, wl, cs, groupRate, CH);
  const spikeClass = noFit ? undefined
    : ((sp.bstar < 1 || wl.invalidation >= sp.fsla) ? 'crit'
      : ((sp.bstar < 5 || wl.invalidation >= sp.fsla/2) ? 'warn' : 'good'));
  // p5 of the OFFLOADED population itself (per-draw all-gpu, then percentiled).
  // Not p5(all)-p5(gpu): that subtracts two marginal percentiles taken from
  // different draws and is not a count of anything.
  const offl = warm.o5;
  // ---- the STEADY-STATE decode point (act 2's honest counterpart to act 1's
  // stress test). Every quantity below is per replica GROUP, like groupRate.
  const sd = noFit ? null : (steady || steadyDecodePoint(dc, topo, groupRate, state.out));
  // prefill duty at this load. Above 1 the queue is unbounded, so there is no
  // steady state to be in: requests never reach the decode batch at all, and
  // quoting a decode speed for them would be the most misleading number on the
  // page. sp.rho is exactly this product — reuse it rather than recompute.
  const dutyNow = sp ? sp.rho : 0;
  // the shared predicate, computed once in computeAndRender; the local duty
  // is still needed below to say WHICH way it failed
  const sdReal = !!sd && (sd.real !== undefined ? sd.real : (dutyNow < 1 && !sd.saturated));
  const sdClass = !sd ? undefined : (!sdReal ? 'crit'
    : (sd.pu >= 50 ? 'good' : (sd.pu >= 40 ? 'warn' : 'crit')));
  // share of the machine's OWN decode ceiling this load asks for: its output
  // demand against the aggregate the whole warm pool would deliver (chart D's
  // stress point). The same ratio the power model bills decode watts on.
  const stAggGroup = st.n * st.pu;
  const sdUse = sd && stAggGroup > 0 ? sd.demand/stAggGroup : 0;
  const sdN = sd ? fmt(sd.n, sd.n < 10 ? 1 : 0) : "—";
  // "distinct users" ≈ warm × (1−p_sub) — an expected-share approximation of
  // scenario_model.py's exact which="user" count; accurate near the reference
  // workload, drifts at extreme subagent ratios
  const tiles=[
    // a censored fill ran out of draws before it ran out of budget, so the count
    // is a floor, not an estimate — say so rather than quietly under-reporting
    {act:1, k:"Warm sessions · p5", hero:true, v:noFit?"—":(warm.censored?"≥ ":"")+fmt(warm.p5,0),
     u:noFit?"":(dp?"per replica":"sessions"),
     sub: noFit ? `model weights do not fit this configuration (${fitHint}) — zero at any CPU offload`
        : (dp ? `system total ${fmt(warm.p5*topo.replicas,0)} (sticky routing) · ≈ ${fmt(warm.p5*topo.replicas*(1-p_sub(wl)),0)} distinct users`
              : `the planning number — 95% of draws hold at least this many · ≈ ${fmt(warm.p5*(1-p_sub(wl)),0)} distinct users`)
          + (offl>=1 ? ` · by tier, p5 each: ${fmt(warm.g5,0)} HBM-resident · ${fmt(offl,0)} offloaded (storage)` : ''),
     tip:"THE number this study is built to produce: sessions whose KV + state sit in cache, so their next request prefills only its new turn. p5 = conservative tail — 95% of draws keep at least this many warm. With CPU offload on this counts the host-RAM tier too, which is storage, not decode."},
    {act:1, k:"Warm p50 / p95", v:noFit?"—":fmt(warm.p50,0), u:noFit?"":"sessions",
     sub:noFit?`model weights do not fit this configuration — ${fitHint}`:`p95 ${fmt(warm.p95,0)} — median and optimistic-tail views`,
     tip:"Same metric, median (p50) and optimistic (p95) draws. Plan on p5; use p50 to compare configurations."},
    {act:1, k:"KV pool", v:fmt(pool/1e6,2), u:"M tok", sub:topo.name,
     tip:"KV-token capacity of one cache: what's left of VRAM after resident weights and the calibrated ~18 GiB activation reserve, divided by KV bytes/token. Every warm session and every active decode lives inside this pool."},
    {act:2, k:"Per-user decode, at this load", v: sdReal ? fmt(sd.pu,0) : "—",
     u: sdReal ? "tok/s" : "", cls:sdClass,
     sub: noFit ? `model weights do not fit this configuration — ${fitHint}`
        : dutyNow >= 1
          ? `prefill duty ${fmt(dutyNow*100,0)}% — the queue is unbounded here, so nothing reaches a steady state to decode in`
        : sd.saturated
          ? `the load demands ${fmt(sd.demand,0)} output tok/s — more than this cache's whole GPU-resident population retires, so the decode batch has no steady size`
        : sd.n < 1
          ? `under one sequence decoding on average — the cache sits idle between requests, so a request that arrives has the machine to itself`
        : `${sdN} sessions decoding at once${dp?" per replica":""}, not ${fmt(st.n,0)}`
          + ` · the all-warm stress test says ${fmt(st.pu,0)} tok/s`,
     tip:"What a user actually sees at the load you set — the counterpart to act 1's stress test. Requests are open-loop: a user waits out most of the think-time interval, so the decode batch holds far fewer sessions than the warm pool and each one runs much faster. Little's law on the decode phase fixes the batch size where the aggregate decode curve (chart D) delivers exactly the output tokens the load demands. Mean-field: the speed is priced at the MEAN batch, and per-user speed is convex in batch size, so this is the conservative side. Excludes the ITL spike from prefill chunks, which is priced separately."},
    {act:2, k:"Decoders in flight", v: sdReal ? sdN : "—",
     u: sdReal ? (dp?"per replica":"sessions") : "",
     sub: !sdReal ? `no steady state at this load — see the tile to the left`
        : `${fmt(sd.demand,0)} output tok/s demanded${dp?` × ${topo.replicas}`:""}`
          + ` = ${fmt(sdUse*100, sdUse<0.1?2:1)}% of the all-warm decode aggregate`
          + ` · at ${fmt(state.out,0)} output tok per response`,
     tip:"Mean number of sequences in the decode batch at any instant — Little's law: arrival rate × seconds spent decoding. This is an EXPECTED occupancy, not a setting: max_num_seqs is a cap and the deploy card rightly leaves it at the ceiling, so the batch can absorb bursts. Scales linearly with the request rate (Concurrent users ÷ Think time) and with output length, and only weakly with anything else — which is why the percentage beside it, not the warm count, is the honest read on how loaded the decode side is. Output length is the one ASSUMED input here (the workload model fits prompt lengths, never output lengths)."},
    {act:2, k:"Warm-hit ceiling", v:fmt(ceiling,1), u:"%", sub:`invalidation ${fmt(wl.invalidation*100,1)}%`,
     tip:"Upper bound on the warm-hit rate: the invalidating fraction f can never match cached KV (edited prefix, changed tools...), so at best (1−f) of requests avoid the cold-prefill thrash."},
    {act:2, k:"Max cold req/s", v:noFit?"—":fmt(coldRate,2), u:noFit?"":"req/s",
     sub:noFit?`model weights do not fit this configuration — ${fitHint}`
        :`miss ≈ ${fmt(coldS,1)} s (p5–p95 ${fmt(coldP5,1)}–${fmt(coldP95,1)}) · f* ${fstar>10?"> 1,000":"= "+fmt(fstar*100,0)}% at ${fmt(groupRate,2)} req/s${dp?"/replica":""}`
         + (spike>0?` · chunk spikes ITL ×${fmt(spike,0)}`:"")
         + (dp?` · ×${topo.replicas} replicas`:""), cls:prefillClass,
     tip:"The prefill (compute) ceiling the capacity model cannot see: cold requests/s at which re-prefilling misses alone consumes 100% of one replica group — set by FLOPs, so no KV pool, offload or warm headroom raises it. Priced on the heavy tail (E[L²]) at the priced chunk (32,768 unless a share link pins another) and the calibrated 45% MFU; f* is the miss rate that saturates the group at the current load; colour: green f < f*/2, amber from f*/2, red at f* and beyond. Analytic and UNVALIDATED — details on the method page."},
    {act:1, k:"Per-user, all GPU-resident p5 warm active", v:noFit?"—":fmt(puOp,1), u:noFit?"":"tok/s",
     sub:noFit?`model weights do not fit this configuration — ${fitHint}`:`at n = ${nOp} GPU-resident concurrent`, cls:floorClass,
     tip:`Stress test: every HBM-resident p5-warm session decoding at once — each user's median speed. Green ≥${fmt(decodeComfort(),0)} tok/s, amber ≥${fmt(decodeFloor(),0)} (the hard floor), red below; if green, the cache — not bandwidth — binds. Both thresholds follow the decode-floor slider. CPU-offloaded sessions are excluded, so the offload slider does not move this.`},
    {act:1, k:"Aggregate, all GPU-resident p5 warm active", v:noFit?"—":fmt(aggOp,1), u:noFit?"":"ktok/s",
     sub:noFit?`model weights do not fit this configuration — ${fitHint}`:`n = ${nOp} × ${topo.replicas} replica${topo.replicas>1?'s':''}`,
     tip:"System-wide token throughput at that same stress point — the GPU-resident p5-warm sessions all decoding at once, summed over DP replicas. A capacity ceiling for batch/offline work, not a latency promise."},
  ];
  const good=cssv('--good'),warn=cssv('--warn'),crit=cssv('--crit');
  const colMap={good,warn,crit};
  // one tile vocabulary, three destinations: act 1 answers "can I hold them?",
  // act 2 "can I serve them?". A tile without an act stays with act 1.
  paintTiles('tiles',     tiles.filter(t=>(t.act||1)===1), colMap);
  // act 2 is painted once, by the planner, so it can lead with its headline
  // metric instead of appending to whatever renderTiles left behind
  pendingLoadTiles = tiles.filter(t=>t.act===2);
}

// shared tile markup — used by both the capacity tiles above and the planner
// good/warn/crit was carried by hue alone, which fails for colour-vision
// deficiency and in greyscale; every status now also carries a word
const CLS_WORD = { good:'ok', warn:'tight', crit:'over' };

export function paintTiles(id, tiles, colMap){
  const el = document.getElementById(id);
  if (!el) return;
  el.innerHTML = tiles.map(t=>{
    const vc = t.cls? `style="color:${colMap[t.cls]}"`:'';
    const badge = t.cls? `<span class="badge" style="color:${colMap[t.cls]};border-color:${colMap[t.cls]}">${CLS_WORD[t.cls]}</span>`:'';
    const kk = t.tip? `<span class="tip" tabindex="0" data-tip="${esc(t.tip)}">${esc(t.k)}</span>` : esc(t.k);
    return `<div class="tile${t.hero?' hero':''}${t.wide?' wide':''}"><div class="k">${kk}${badge}</div><div class="v tnum" ${vc}>${esc(t.v)}<span class="u">${esc(t.u)}</span></div><div class="sub2 tnum">${esc(t.sub)}</div></div>`;
  }).join('');
}
