import { ACT_RESERVE, AVG_OUT_TOK, CONFIG, GIB, PREFILL_MFU, clampTp, divisors,
         effective_bw, kv_pool_tokens, makeGrid, makeTopo, minTpFor, servableKv, tpEff,
         withKvDtype } from './config.js';
import { PREFILL_CHUNK, REF_REQ_RATE, SPIKE_SLA_S, WARM_TURN_TOK, coldRequestSeconds,
         contextStats, liveThink, liveTurn, maxUsersLatency, maxUsersSaturation, mfuEff,
         missContextSeconds, prefillContextSeconds, prefillSeconds,
         prefillServiceMoments, requestRate, serverRate, spikeMetrics,
         steadyDecodePoint } from './prefill.js';
import { seedRng } from './mathlib.js';
import { decodeCurves, warmCapacity } from './capacity.js';
import { interpAt } from './charts.js';
import { bStar } from './planner.js';

/* ============================================================================
   UNIT CHECKS — must match scenario_model.py derived numbers
   ========================================================================== */
export function unitChecks(){
  const approx = (a,b,rt)=>Math.abs(a-b) <= Math.abs(b)*rt;
  console.assert(approx(ACT_RESERVE/GIB, 17.98, 0.01), "ACT_RESERVE ~ 17.98 GiB, got "+(ACT_RESERVE/GIB).toFixed(3));
  const p1  = kv_pool_tokens(CONFIG.MODELS["27B"],    CONFIG.TOPOLOGIES["1xH200"]);
  const p2  = kv_pool_tokens(CONFIG.MODELS["27B"],    CONFIG.TOPOLOGIES["2xH200-TP2"]);
  const p3  = kv_pool_tokens(CONFIG.MODELS["35BA3B"], CONFIG.TOPOLOGIES["1xH200"]);
  console.assert(approx(p1, 2.770e6, 0.001), "27B/1xH200 pool ~2.770e6, got "+p1.toExponential(4));
  console.assert(approx(p2, 6.484e6, 0.001), "27B/TP2 pool ~6.484e6, got "+p2.toExponential(4));
  console.assert(approx(p3, 8.417e6, 0.001), "35BA3B/1xH200 pool ~8.417e6, got "+p3.toExponential(4));
  // measured cross-check: 27B + FP16 KV + TP2 reported 3,233,564 tokens (vLLM startup log)
  const m27fp16 = { ...CONFIG.MODELS["27B"], kv_bpt: CONFIG.MODELS["27B"].kv_bpt*2 };
  const pMeas = kv_pool_tokens(m27fp16, CONFIG.TOPOLOGIES["2xH200-TP2"]);
  console.assert(approx(pMeas, 3233564, 0.01), "27B/TP2/FP16 pool should match measured 3,233,564, got "+pMeas.toExponential(4));
  // published-config identities (mirror scenario_model.py _selfcheck)
  const m35 = CONFIG.MODELS["35BA3B"];
  console.assert(m35.kv_bpt === 10*2*256*2*1, "35BA3B kv_bpt = 10 KiB/token");
  console.assert(m35.w_route_pertok === 8*(3*2048*512)*40, "8 routed experts x 40 layers");
  console.assert(m35.w_route_total === 256*(3*2048*512)*40, "256 experts total");
  console.assert(approx(m35.w_route_total/m35.w_route_pertok, 32, 1e-9), "linear saturation at n=32");
  const active = m35.w_decode_shared + m35.w_route_pertok;
  console.assert(active > 2.8e9 && active < 3.1e9, "active bytes/token ~3B, got "+active.toExponential(3));
  // N-GPU topology helpers
  console.assert(Math.abs(tpEff(2)-CONFIG.TP_EFFICIENCY)<1e-12 && tpEff(1)===1, "tpEff anchors");
  console.assert(tpEff(8)<tpEff(4) && tpEff(4)<tpEff(2), "tpEff decreases per doubling");
  const p4 = kv_pool_tokens(CONFIG.MODELS["35BA3B"], makeTopo("tp",4));
  console.assert(p4 > 2*kv_pool_tokens(CONFIG.MODELS["35BA3B"], makeTopo("tp",2)),
    "TP4 pool must exceed 2x TP2 (weights amortized)");
  // ---- DP x TP grid: a replica is a GROUP, not a GPU (mirrors _selfcheck) ----
  const g24 = makeGrid(2,4,"B300");
  console.assert(g24.n_gpu===8 && g24.replicas===2 && g24.kind==="hybrid" &&
    g24.gpu===CONFIG.GPUS["B300"], "makeGrid(2,4,B300) shape");
  console.assert(makeTopo("tp",4).tp===4 && makeTopo("tp",4).dp===1, "TP4 = edge (1,4)");
  console.assert(makeTopo("dp",4).dp===4 && makeTopo("dp",4).tp===1, "DP4 = edge (4,1)");
  const m35g = CONFIG.MODELS["35BA3B"];
  for (const tp of [1,2,4]){
    const basePool = kv_pool_tokens(m35g, makeGrid(1,tp)), baseBw = effective_bw(makeGrid(1,tp));
    for (const dp of [1,3,8]){
      console.assert(approx(kv_pool_tokens(m35g, makeGrid(dp,tp)), basePool, 1e-12),
        "per-group pool must be independent of dp");
      console.assert(effective_bw(makeGrid(dp,tp))===baseBw, "DP must not widen a group");
    }
  }
  // legacy literals must be STRUCTURALLY exact mirrors of makeGrid — name and
  // gpu included, not just the numeric fields
  for (const [k,t] of Object.entries(CONFIG.TOPOLOGIES)){
    const g = makeGrid(t.dp, t.tp);
    console.assert(g.n_gpu===t.n_gpu && g.kind===t.kind && g.replicas===t.replicas,
      "TOPOLOGIES literal "+k+" must match makeGrid(dp,tp)");
    console.assert(g.name===t.name, `TOPOLOGIES ${k} name "${t.name}" != makeGrid "${g.name}"`);
    console.assert(t.gpu===CONFIG.GPUS["H200"], `TOPOLOGIES ${k} must carry its GPU object`);
  }
  // validation parity with topology_grid()/topology(): bad input throws
  for (const [dp,tp] of [[0,2],[2,0],[1.5,2],[2,-1]]){
    let threw=false; try { makeGrid(dp,tp); } catch(e){ threw=true; }
    console.assert(threw, `makeGrid(${dp},${tp}) must throw`);
  }
  {
    let threw=false; try { makeGrid(1,2,"H100"); } catch(e){ threw=true; }
    console.assert(threw, "makeGrid with an unknown gpu key must throw");
    threw=false; try { makeTopo("bogus",2); } catch(e){ threw=true; }
    console.assert(threw, "makeTopo with a bad kind must throw, not silently mean DP");
    threw=false; try { tpEff(4,0); } catch(e){ threw=true; }
    console.assert(threw, "tpEff with domain 0 must throw, not fall back to 8");
    console.assert(tpEff(4,undefined)===tpEff(4,8), "undefined domain falls back to 8");
  }
  // the grid is what makes DP expressible for the models that fit no single GPU
  console.assert(minTpFor(CONFIG.MODELS["MM35"],"H200")===2 &&
                 minTpFor(CONFIG.MODELS["MM35"],"B300")===1 &&
                 minTpFor(CONFIG.MODELS["GLM52"],"H200")===7 &&
                 minTpFor(CONFIG.MODELS["GLM52"],"B300")===3 &&
                 minTpFor(CONFIG.MODELS["DSV4F"],"H200")===2 &&
                 minTpFor(CONFIG.MODELS["DSV4F"],"B300")===1 &&
                 minTpFor(CONFIG.MODELS["Q38FN"],"H200")===2 &&
                 minTpFor(CONFIG.MODELS["Q38FN"],"B300")===1 &&
                 minTpFor(withKvDtype(CONFIG.MODELS["GLM53F"],"fp16"),"H200")===3 &&
                 minTpFor(CONFIG.MODELS["GLM53F"],"B300")===2, "min TP per model/part");
  // (GLM-5.3-Flash prices as its BF16-KV arm on any H200 topology — the fp8
  // arm THROWS there, asserted below with the rest of its identities)
  for (const mk of ["MM35","GLM52","DSV4F","Q38FN","GLM53F"])
    for (const n of [1,2,4,8])
      console.assert(kv_pool_tokens(
          mk==="GLM53F" ? withKvDtype(CONFIG.MODELS[mk],"fp16") : CONFIG.MODELS[mk],
          makeTopo("dp",n))===0,
        mk+": pure DP of single H200s must stay a 0 pool");
  console.assert(kv_pool_tokens(CONFIG.MODELS["MM35"], makeGrid(4,2)) > 0,
    "MM35 DP4xTP2 on H200 must hold a real pool");
  console.assert(kv_pool_tokens(CONFIG.MODELS["GLM52"], makeGrid(2,4,"B300")) > 0,
    "GLM-5.2 DP2xTP4 on B300 must hold a real pool");
  // widening TP raises the SYSTEM total: each DP group re-pays for the weights
  const sys = tp => tp>0 ? (8/tp)*kv_pool_tokens(CONFIG.MODELS["GLM52"], makeGrid(8/tp,tp,"B300")) : 0;
  console.assert(sys(8) > sys(4), "GLM-5.2 TP8 must beat DP2xTP4 on system total");
  // TP past the node's NVLink domain must cliff, not extrapolate
  for (const g of Object.values(CONFIG.GPUS))
    console.assert(g.nvlink_domain===8, "both parts are 8-GPU nodes");
  console.assert(tpEff(16)/tpEff(8) < tpEff(8)/tpEff(4),
    "crossing nodes must cost more than an in-node doubling");
  console.assert(approx(tpEff(16)/tpEff(8), CONFIG.CROSS_DOMAIN_EFFICIENCY, 1e-9), "cliff factor");
  for (const n of [1,2,4,8])
    console.assert(approx(tpEff(n), Math.pow(CONFIG.TP_EFFICIENCY, Math.log2(n)), 1e-12),
      "in-node tpEff unchanged at n="+n);
  console.assert(tpEff(8,4) < tpEff(8,8), "a smaller domain moves the cliff earlier");
  // ---- split control: the TP widths offered, and the clamp on GPU-count change
  console.assert(divisors(8).join()==="1,2,4,8" && divisors(6).join()==="1,2,3,6" &&
                 divisors(1).join()==="1" && divisors(7).join()==="1,7", "divisors");
  // every clamp result must divide ngpu and never exceed the requested width
  for (let n=1;n<=8;n++) for (let tp=1;tp<=8;tp++){
    const c = clampTp(tp, n);
    console.assert(n % c === 0 && c <= tp && c >= 1,
      `clampTp(${tp},${n})=${c} must be a divisor of ${n} no greater than ${tp}`);
  }
  console.assert(clampTp(4,8)===4 && clampTp(4,6)===3 && clampTp(8,6)===6 &&
                 clampTp(3,8)===2 && clampTp(1,8)===1,
    "clampTp keeps as much TP as still divides evenly");
  // a legal split is exactly a grid whose product is the GPU count
  for (const tp of divisors(8)){
    const t = makeGrid(8/tp, tp, "B300");
    console.assert(t.n_gpu===8 && t.tp===tp && t.replicas===8/tp, "split grid "+tp);
  }
  // new models: published-config KV identities (mirror _selfcheck)
  const mm = CONFIG.MODELS["MM35"], glm = CONFIG.MODELS["GLM52"];
  console.assert(mm.kv_bpt === 88*8*128*2*1, "MM35 kv_bpt = 176 KiB/token");
  console.assert(glm.kv_bpt === 79*576 + 22*132, "GLM52 kv_bpt = MLA latent + indexer");
  console.assert(mm.mtp === 1.0, "MM35 has no MTP module");
  console.assert(Math.abs(glm.w_route_total/glm.w_route_pertok - 32) < 1e-9, "GLM kink n=32");
  // B300 + NVFP4: pool ordering, gate, and the ~465 GB GLM checkpoint check
  const b1 = makeTopo("tp",1,"B300");
  console.assert(kv_pool_tokens(CONFIG.MODELS["27B"], b1) > p1, "B300 pool > H200 pool");
  console.assert(Math.abs(glm.nvfp4_w[0]/465e9 - 1) < 0.005,
    "GLM-5.2 NVFP4 resident must match the vLLM recipe's ~465 GB");
  for (const mk of ["27B","35BA3B"])
    console.assert(!!CONFIG.MODELS[mk].nvfp4_w, mk+" must be NVFP4-selectable");
  let gateThrew = false;
  try { kv_pool_tokens({...CONFIG.MODELS["27B"], weight_dtype:"nvfp4"}, makeTopo("tp",1,"H200")); }
  catch(e){ gateThrew = true; }
  console.assert(gateThrew, "NVFP4 on H200 must throw (B300-only gate)");
  console.assert(kv_pool_tokens(glm, makeTopo("tp",1,"H200")) === 0 &&
                 kv_pool_tokens(glm, makeTopo("tp",8,"H200")) > 0,
    "GLM-5.2 FP8 must not fit 1 GPU but fit 8xH200");
  console.assert(kv_pool_tokens(CONFIG.MODELS["DSV4F"], makeTopo("tp",1,"H200")) === 0 &&
                 kv_pool_tokens(CONFIG.MODELS["DSV4F"], makeTopo("tp",2,"H200")) > 0 &&
                 kv_pool_tokens(CONFIG.MODELS["DSV4F"], makeTopo("tp",1,"B300")) > 0,
    "DSv4-Flash must fit from 2xH200 and a single B300");
  console.assert(kv_pool_tokens(CONFIG.MODELS["Q38FN"], makeTopo("tp",1,"H200")) === 0 &&
                 kv_pool_tokens(CONFIG.MODELS["Q38FN"], makeTopo("tp",2,"H200")) > 0 &&
                 kv_pool_tokens(CONFIG.MODELS["Q38FN"], makeTopo("tp",1,"B300")) > 0,
    "Qwen3.8-Flash-Next must fit from 2xH200 and a single B300");
  const g53bf16 = withKvDtype(CONFIG.MODELS["GLM53F"], "fp16");
  console.assert(kv_pool_tokens(g53bf16, makeTopo("tp",2,"H200")) === 0 &&
                 kv_pool_tokens(g53bf16, makeTopo("tp",3,"H200")) > 0 &&
                 kv_pool_tokens(CONFIG.MODELS["GLM53F"], makeTopo("tp",1,"B300")) === 0 &&
                 kv_pool_tokens(CONFIG.MODELS["GLM53F"], makeTopo("tp",2,"B300")) > 0,
    "GLM-5.3-Flash must fit from 3xH200 (BF16-KV arm) and 2xB300");
  // the GPU-coupled KV gate (mirrors check_dtype_supported): the fp8-KV arm
  // must THROW on H200, price on B300, and servableKv must resolve it
  let g53Threw = false;
  try { kv_pool_tokens(CONFIG.MODELS["GLM53F"], makeTopo("tp",4,"H200")); }
  catch(e){ g53Threw = true; }
  console.assert(g53Threw, "GLM53F fp8 KV on H200 must throw (Blackwell-only)");
  console.assert(servableKv(CONFIG.MODELS["GLM53F"], "fp8", "H200") === "fp16" &&
                 servableKv(CONFIG.MODELS["GLM53F"], "fp8", "B300") === "fp8" &&
                 servableKv(CONFIG.MODELS["Q38FN"], "fp8", "H200") === "fp8",
    "servableKv: GLM53F fp8->fp16 on Hopper only; other models untouched");
  // NVFP4 swap identities (mirror _selfcheck): checkpoint shrinks, but the
  // BF16-kept blocks make the SHARED per-step read heavier on the MoE models
  console.assert(m35.nvfp4_w[0] < m35.w_resident && m35.nvfp4_w[1] > m35.w_decode_shared,
    "35B NVFP4: smaller resident, heavier shared read");
  console.assert(glm.nvfp4_w[0] < glm.w_resident && glm.nvfp4_w[1] > glm.w_decode_shared,
    "GLM NVFP4: smaller resident, heavier shared read");
  // sparse-decode pricing identities (research/model_glm52.md #3)
  console.assert(glm.kv_decode_bpt === 21*132, "GLM indexer scan = 21 x 132 B/tok");
  console.assert(Math.abs(glm.kv_decode_const/(78*2048*576) - 1) < 0.001,
    "GLM top-k read = 78 layers x top-2048 x 576 B");
  console.assert(glm.kv_decode_topk === 2048, "GLM top-k window");
  // DSv4-Flash identities (research/model_dsv4flash.md; mirror _selfcheck)
  const dsf = CONFIG.MODELS["DSV4F"];
  console.assert(dsf.kv_bpt === 21*576/4 + 20*576/128 + 21*64/4,
    "DSV4F kv_bpt = CSA + HCA + fp4 indexer = 3,450 B/token");
  console.assert(dsf.deltanet_state === 46*128*576 + 12206080,
    "DSV4F per-session state = windows + fp32 compressor buffers");
  console.assert(dsf.state_fp32_ok === false && dsf.kv_fp16_ok === false,
    "DSV4F: fixed-precision state, quantized-only main KV");
  console.assert(dsf.w_route_pertok === 6*13369344*43 &&
                 dsf.w_route_total === 256*13369344*43 &&
                 Math.abs(dsf.w_route_total/dsf.w_route_pertok - 256/6) < 1e-9,
    "DSV4F expert bytes (FP4 packed + E8M0 scales); kink at n = 256/6");
  console.assert(dsf.kv_decode_bpt === 21*64/4 + 20*576/128 &&
                 dsf.kv_decode_const === 21*512*576 + 43*128*576 &&
                 dsf.kv_decode_topk === 2048, "DSV4F sparse-decode pricing");
  console.assert(Math.abs(dsf.attn_layers*dsf.attn_d - (21*1024 + 20*256)) < 1e-9,
    "DSV4F prefill quadratic term = indexer + dense-HCA equivalents");
  console.assert(!dsf.nvfp4_w, "DSV4F must not be NVFP4-selectable");
  // Qwen3.8-Flash-Next identities (research/model_qwen38flashnext.md; mirror _selfcheck)
  const q38 = CONFIG.MODELS["Q38FN"];
  console.assert(q38.kv_bpt === 12*2*256*2 + 12*128/4,
    "Q38FN kv_bpt = 12 attn layers' KV + ratio-4 compressed indexer keys");
  console.assert(q38.deltanet_state === 36*48*128*128*2 + 36*10240*4*2,
    "Q38FN per-session state = bf16 DeltaNet SSM + conv state");
  console.assert(q38.w_route_pertok === 10*4915800*48 &&
                 q38.w_route_total === 512*4915800*48 &&
                 Math.abs(q38.w_route_total/q38.w_route_pertok - 51.2) < 1e-9,
    "Q38FN expert bytes (FP8 + block scales); deepest kink at n = 512/10 = 51.2");
  console.assert(q38.kv_decode_bpt === 12*128/4 &&
                 q38.kv_decode_const === 12*2048*1024 &&
                 q38.kv_decode_topk === 2048, "Q38FN QSA sparse-decode pricing");
  console.assert(q38.w_decode_shared === 8623999000,
    "Q38FN shared per-step read = the exact BF16 ledger sum");
  console.assert(q38.w_resident > q38.w_route_total + q38.w_decode_shared + 50e9,
    "Q38FN resident (n-gram table, embed, vision, MTP) far exceeds the touchable bytes");
  console.assert(!q38.nvfp4_w, "Q38FN must not be NVFP4-selectable (no official ckpt)");
  console.assert(q38.kv_fp16_ok === true && q38.state_fp32_ok === true,
    "Q38FN: FP16-KV and fp32-state toggles stay enabled (explicit flags)");
  // FP16-KV on the sparse path — exercises the REAL transform modelFor uses
  // (withKvDtype), not a local copy: pool bytes AND the top-k main-KV gathers
  // double; the fp8 indexer scan does not; GLM-5.2 stays refused (identity)
  const q38fp16 = withKvDtype(q38, "fp16");
  console.assert(q38fp16.kv_bpt === 2*q38.kv_bpt &&
                 q38fp16.kv_decode_const === 2*q38.kv_decode_const &&
                 q38fp16.kv_decode_bpt === q38.kv_decode_bpt,
    "Q38FN FP16-KV: kv_bpt and kv_decode_const double, indexer scan unchanged");
  console.assert(withKvDtype(glm, "fp16") === glm && withKvDtype(q38, "fp8") === q38,
    "withKvDtype: identity on refused-FP16 models and on fp8");
  // GLM-5.3-Flash identities (research/model_glm53flash.md; mirror _selfcheck)
  const g53 = CONFIG.MODELS["GLM53F"];
  console.assert(g53.kv_bpt === 12*512 + 12*132/4,
    "GLM53F kv_bpt = 12 DSA stacks (11 main + MTP layer, the GLM-5.2 storage "
    + "convention) of nope-only latents + kpool-4 compressed indexer keys");
  console.assert(g53.deltanet_state === 34*64*128*128*2 + 34*3*8192*4*2,
    "GLM53F per-session state = bf16 KDA SSM + q/k/v conv state");
  console.assert(g53.w_route_pertok === 8*25171968*42 &&
                 g53.w_route_total === 288*25171968*42 &&
                 Math.abs(g53.w_route_total/g53.w_route_pertok - 36) < 1e-9,
    "GLM53F expert bytes (FP8 + F32 block scales); kink at n = 288/8 = 36");
  console.assert(g53.kv_decode_bpt === 11*132/4 &&
                 g53.kv_decode_const === 11*2048*512 &&
                 g53.kv_decode_topk === 2048, "GLM53F DSA sparse-decode pricing");
  console.assert(g53.w_decode_shared === 13957216504,
    "GLM53F shared per-step read = the exact ledger sum");
  console.assert(g53.w_decode_shared + g53.w_route_total
                 + 1268776960 + 7493399168 + 1127254016 === g53.w_resident,
    "GLM53F closing identity: shared + routed + embed + MTP + vision = total_size");
  console.assert(!g53.nvfp4_w, "GLM53F must not be NVFP4-selectable (no official ckpt)");
  console.assert(g53.kv_fp16_ok === true && g53.state_fp32_ok === true,
    "GLM53F: FP16-KV live (REQUIRED on Hopper) and fp32-state live");
  const g53fp16 = withKvDtype(g53, "fp16");
  console.assert(g53fp16.kv_bpt === 2*g53.kv_bpt &&
                 g53fp16.kv_decode_const === 2*g53.kv_decode_const &&
                 g53fp16.kv_decode_bpt === g53.kv_decode_bpt,
    "GLM53F FP16-KV: kv_bpt and kv_decode_const double, indexer scan unchanged");
  // zero-pool honesty: weights that don't fit report ZERO warm even with a
  // large CPU-offload buffer (the guard fires before the workload is touched)
  const zr = warmCapacity(glm, makeTopo("tp",1,"H200"), null, 512, 3, 1000);
  console.assert(zr.all[2] === 0 && zr.gpu[2] === 0,
    "no-fit config must stay at zero warm regardless of offload");
  // ---- prefill ceiling, deterministic part (mirrors scenario_model.py's
  // prefill section; the sampled-mean checks live in a second block AFTER the
  // RNG section — its `let _spare` state is TDZ-dead while this one runs) ----
  const t1H = CONFIG.TOPOLOGIES["1xH200"];
  console.assert(approx(prefillContextSeconds(CONFIG.MODELS["27B"], t1H, WARM_TURN_TOK, 0, 40141), 0.1464, 0.02),
    "27B/1xH200 warm hit (turn on 40.1k cached) ~146 ms");
  console.assert(prefillSeconds(CONFIG.MODELS["35BA3B"], t1H, PREFILL_CHUNK) * 5
                 < prefillSeconds(CONFIG.MODELS["27B"], t1H, PREFILL_CHUNK),
    "the MoE must prefill a chunk >5x faster than the smaller dense 27B (active params)");
  // chunk-size behaviour: one isolated pass is superlinear in its size; the
  // MARGINAL whole-context cost is chunk-size invariant (the pair-count
  // telescopes); the MISS cost is not — its per-pass overhead multiplies by
  // the pass count, so smaller chunks cost MORE total machine time
  console.assert(prefillSeconds(CONFIG.MODELS["27B"], t1H, 65536)
                 > 2 * prefillSeconds(CONFIG.MODELS["27B"], t1H, 32768),
    "isolated prefill pass must be superlinear in chunk size (quadratic term)");
  console.assert(approx(prefillContextSeconds(CONFIG.MODELS["27B"], t1H, 180000, 16384),
                        prefillContextSeconds(CONFIG.MODELS["27B"], t1H, 180000, 65536), 1e-9),
    "marginal whole-context cost must be chunk-size invariant (telescoping pairs)");
  console.assert(prefillSeconds(CONFIG.MODELS["27B"], t1H, PREFILL_CHUNK, PREFILL_CHUNK)
                 > prefillSeconds(CONFIG.MODELS["27B"], t1H, PREFILL_CHUNK),
    "a later chunk pays cross-attention over the cache");
  console.assert(prefillContextSeconds(CONFIG.MODELS["27B"], t1H, 180000)
                 === prefillContextSeconds(CONFIG.MODELS["27B"], t1H, 180000, PREFILL_CHUNK),
    "omitted chunk must default to the study's 32,768");
  // ---- MFU(chunk) model: anchor, monotonicity, and who pays most ----
  for (const mk of ["27B","35BA3B"])
    console.assert(approx(mfuEff(CONFIG.MODELS[mk], t1H, PREFILL_CHUNK), PREFILL_MFU, 1e-9),
      `${mk}: effective MFU at the 32,768 anchor must be exactly the calibrated 0.45`);
  console.assert(mfuEff(CONFIG.MODELS["27B"], t1H, 2048)
                 < mfuEff(CONFIG.MODELS["27B"], t1H, 8192)
              && mfuEff(CONFIG.MODELS["27B"], t1H, 8192)
                 < mfuEff(CONFIG.MODELS["27B"], t1H, 32768),
    "effective MFU must rise with chunk size (overhead amortises)");
  console.assert(mfuEff(CONFIG.MODELS["35BA3B"], t1H, 2048)
                 < mfuEff(CONFIG.MODELS["27B"], t1H, 2048),
    "the MoE must degrade harder at small chunks (little compute, full expert-bank stream)");
  console.assert(missContextSeconds(CONFIG.MODELS["27B"], t1H, 180000, 2048)
                 > missContextSeconds(CONFIG.MODELS["27B"], t1H, 180000, 16384)
              && missContextSeconds(CONFIG.MODELS["27B"], t1H, 180000, 16384)
                 > missContextSeconds(CONFIG.MODELS["27B"], t1H, 180000, 65536),
    "miss cost must fall as the chunk grows (fewer passes pay the overhead)");
  console.assert(approx(missContextSeconds(CONFIG.MODELS["27B"], t1H, 180000, PREFILL_CHUNK),
                        prefillContextSeconds(CONFIG.MODELS["27B"], t1H, 180000, PREFILL_CHUNK), 0.01),
    "at the 32,768 default the dense miss cost must reproduce the flat-45% model within 1%");
  console.assert(approx(missContextSeconds(CONFIG.MODELS["35BA3B"], t1H, 180000, PREFILL_CHUNK),
                        prefillContextSeconds(CONFIG.MODELS["35BA3B"], t1H, 180000, PREFILL_CHUNK), 0.02),
    "…and the MoE within 2% (under: its later passes run at the higher solved ceiling)");
  console.log("[unit checks] ACT_RESERVE=%s GiB | pools: 27B/1x=%s TP2=%s 35BA3B/1x=%s",
    (ACT_RESERVE/GIB).toFixed(2), p1.toExponential(3), p2.toExponential(3), p3.toExponential(3));
}


// prefill-ceiling unit checks, sampled part — a separate function because it
// draws from the RNG (unitChecks does not). Expected values are
// docs/scenarios.md § 8's table, so the tile can never silently drift from
// the numbers the study quotes. Runs from init, before the first render.
export function prefillSampledChecks(){
  const approx = (a,b,rt)=>Math.abs(a-b) <= Math.abs(b)*rt;
  // the reference workload (Python Workload defaults); the mean over 200k
  // draws has ~0.2% sampling error, so a 5% tolerance is all jitter headroom
  const wlRef = { user_median:31000, user_sigma:0.81, sub_median:8000, sub_sigma:0.9,
                  sub_ratio:0.10, sys_user:15000, sys_sub:3000, sub_shares_prefix:false,
                  invalidation:0.01, cap:180000 };
  const csRef = contextStats(wlRef, 200000);
  const t1H = CONFIG.TOPOLOGIES["1xH200"];
  console.assert(approx(1/coldRequestSeconds(CONFIG.MODELS["27B"], t1H, wlRef, csRef), 0.354, 0.05),
    "27B/1xH200 max cold rate ~0.35 req/s");
  const tp4H = makeTopo("tp", 4);
  const coldMM = coldRequestSeconds(CONFIG.MODELS["MM35"], tp4H, wlRef, csRef);
  const warmMM = prefillContextSeconds(CONFIG.MODELS["MM35"], tp4H, WARM_TURN_TOK, 0, csRef.mean);
  console.assert(approx(1/coldMM, 0.182, 0.05), "MM35/TP4 max cold rate ~0.18 req/s");
  console.assert(approx((1/REF_REQ_RATE - warmMM)/(coldMM - warmMM), 0.034, 0.15),
    "MM35/TP4 f* ~3% — prefill saturates in-slider-range (cache is ALSO <64 users there)");
  // ---- cold spikes (research/spike.md) ----
  const tp2H = CONFIG.TOPOLOGIES["2xH200-TP2"];
  const sp1 = spikeMetrics(CONFIG.MODELS["27B"], t1H, wlRef, csRef, REF_REQ_RATE);
  const sp2 = spikeMetrics(CONFIG.MODELS["27B"], tp2H, wlRef, csRef, REF_REQ_RATE);
  const spM = spikeMetrics(CONFIG.MODELS["35BA3B"], tp2H, wlRef, csRef, REF_REQ_RATE);
  const spX = spikeMetrics(CONFIG.MODELS["MM35"], tp4H, wlRef, csRef, REF_REQ_RATE);
  console.assert(approx(sp2.rho, 0.205, 0.05), "27B/TP2 prefill duty ~20.5% at f=1%");
  // service time is far more variable than exponential (cv^2 = 1): it runs as
  // L^2 on a lognormal L, which is why the queue diverges below f*
  const cv2 = sp2.mo.missSq*wlRef.invalidation + sp2.mo.hitSq*(1-wlRef.invalidation);
  const eS  = sp2.mo.miss*wlRef.invalidation + sp2.mo.hit*(1-wlRef.invalidation);
  console.assert(cv2/(eS*eS) - 1 > 3, "27B/TP2 service cv^2 ~5.5 (an M/M/1 would be 1)");
  console.assert(approx(sp2.bstar, 5.06, 0.08), "27B/TP2 B* ~5 simultaneous misses");
  console.assert(approx(sp1.bstar, 2.23, 0.08), "27B/1xH200 B* ~2 simultaneous misses");
  console.assert(approx(spM.bstar, 36.4, 0.08), "35B-A3B/TP2 B* ~36 simultaneous misses");
  console.assert(spX.bstar < 1, "MM35/TP4 cannot absorb ONE simultaneous miss in 10 s");
  // the MoE gap COMPOUNDS: B* ratio must exceed the raw prefill-speed ratio
  console.assert(spM.bstar/sp2.bstar > sp2.mo.miss/spM.mo.miss,
    "MoE spike tolerance must beat dense by MORE than its prefill-speed ratio");
  // the latency ceiling binds below the duty ceiling, always
  console.assert(approx(sp2.fsla, 0.215, 0.10), "27B/TP2 f_sla ~21% at a 10 s budget");
  console.assert(sp2.fsla < (1/REF_REQ_RATE - prefillContextSeconds(CONFIG.MODELS["27B"], tp2H, WARM_TURN_TOK, 0, csRef.mean))
                          / (coldRequestSeconds(CONFIG.MODELS["27B"], tp2H, wlRef, csRef) - prefillContextSeconds(CONFIG.MODELS["27B"], tp2H, WARM_TURN_TOK, 0, csRef.mean)),
    "f_sla must bind before f*");
  // a global flush = f 100%: only the MoE on TP2 can serve its own recovery
  console.assert(sp2.allCold > 1 && spX.allCold > 1 && spM.allCold < 1,
    "all-cold duty: dense configs must shed load on a flush, the MoE/TP2 serves it");
  // ---- the operating point (research/spike.md, the two-axis planner) ----
  // these run before any slider exists, so they price the PUBLISHED reference
  // values; the live mirrors start equal to them by construction
  console.assert(liveTurn === WARM_TURN_TOK && liveThink === 30,
    "unit checks must run at the published reference turn size and think time");
  console.assert(approx(requestRate(64, 30), REF_REQ_RATE, 0.01),
    "64 users at one turn per 30 s must reproduce the 2.13 req/s reference");
  // the algebraic heart of the section: the queue diverges before the server,
  // so the latency ceiling is ALWAYS strictly inside saturation
  for (const [mk, tt] of [["27B", t1H], ["27B", tp2H], ["35BA3B", tp2H]]){
    const mo = prefillServiceMoments(CONFIG.MODELS[mk], tt, wlRef, csRef);
    const lat = maxUsersLatency(mo, wlRef.invalidation, 10, 30);
    const sat = maxUsersSaturation(mo, wlRef.invalidation, 30);
    console.assert(lat > 0 && lat < sat,
      `${mk}: latency ceiling ${lat.toFixed(0)} must sit inside saturation ${sat.toFixed(0)}`);
    console.assert(maxUsersLatency(mo, wlRef.invalidation, 10, 30, 'ps') < lat,
      "processor sharing must admit fewer users than FCFS on the MISS side");
    console.assert(maxUsersLatency(mo, wlRef.invalidation, mo.miss*0.5, 30) === 0,
      "a budget below one miss's own prefill cannot be met at any load");
    console.assert(approx(maxUsersLatency(mo, wlRef.invalidation, 10, 60)/lat, 2, 1e-9),
      "think time scales the user ceilings linearly");
    console.assert(approx(maxUsersSaturation(mo, wlRef.invalidation, 30, 0.10),
                          sat/1.10, 1e-9),
      "each main request tows sub_ratio subagent requests: ceilings / (1+r)");
    console.assert(approx(maxUsersLatency(mo, wlRef.invalidation, 10, 30,
                                          undefined, 0.10), lat/1.10, 1e-9),
      "the latency ceiling carries the same (1+r) as saturation");
    console.assert(approx(serverRate(64, 30, 0.10), REF_REQ_RATE*1.10, 0.01),
      "serverRate must be the (1+r)-scaled main-agent rate");
  }
  // The planner's two SAMPLED ceilings (cache, decode) are pinned against
  // section 7's published table in scenario_model.py's self-checks, not here:
  // this browser suite is deliberately the closed-form subset, since the
  // Monte-Carlo assertions run only in Python (docs/scenarios.md,
  // Reproducibility) — and warmCapacity's scratch buffer is not even
  // initialised this early in the file.
  // B* factored out of spikeMetrics must agree with it exactly
  console.assert(approx(bStar(sp2.mo, wlRef.invalidation, SPIKE_SLA_S, REF_REQ_RATE),
                        sp2.bstar, 1e-9), "bStar must agree with spikeMetrics");
}


// Steady-state decode point checks. They live HERE, not with the other sampled
// unit checks, because they need a real decode sweep — and the identities they
// pin are exact, so a coarse 24-sample sweep is enough to catch a sign error,
// a per-group/system unit slip, or a broken inversion.
export function steadyChecks(){
  const approx = (a,b,rt)=>Math.abs(a-b) <= Math.abs(b)*rt;
  const wlRef = { user_median:31000, user_sigma:0.81, sub_median:8000, sub_sigma:0.9,
                  sub_ratio:0.10, sys_user:15000, sys_sub:3000, sub_shares_prefix:false,
                  invalidation:0.01, cap:180000 };
  const m27 = CONFIG.MODELS["27B"], tp2 = CONFIG.TOPOLOGIES["2xH200-TP2"];
  // seeded by hand, not seedFor: samplingSig() reads `state`, which does not
  // exist this early. A fixed seed makes the pinned values below reproducible
  // rather than a function of whatever drew from the stream before this ran.
  seedRng(20260807);
  const dc = decodeCurves(m27, tp2, wlRef, 480, 20, 400);
  const OUT = AVG_OUT_TOK;
  // the reference load, as one replica group sees it (TP2 => replicas = 1)
  const rate = serverRate(64, 30, wlRef.sub_ratio);
  const sd = steadyDecodePoint(dc, tp2, rate, OUT);
  // 1. THE identity: at the fixed point the batch delivers exactly what the
  //    load demands. Everything else on the tile is a reading of this line.
  console.assert(approx(sd.n * sd.pu, rate*OUT, 1e-9),
    "steady point must balance: n x per-user speed = arrival rate x output tokens");
  // 2. ...which is Little's law read the other way: n = lambda x time decoding
  console.assert(approx(sd.n, rate * (OUT/sd.pu), 1e-9),
    "steady n must equal arrival rate x seconds spent decoding");
  // 3. the whole point of the tiles, PINNED: the 27B on TP2 at the published
  //    reference load (64 users, 30 s, r = 0.10, 400 output tokens) decodes
  //    ~2 sequences at a time at ~430 tok/s — against ~196 warm sessions at
  //    ~46 tok/s if they all decoded at once. Stable to ~1% from step 1 to
  //    step 40, so the tolerance is MC jitter only; a units slip in the
  //    arrival rate or the output length moves it by multiples.
  //    (Re-pinned 2026-08-27 with AVG_OUT_TOK 1000 -> 400, measured: fewer
  //    decode-seconds per request, smaller faster batch. Was ~6.4 at ~365 —
  //    the JS pin; the Python mirror's historical pin was 370.)
  // Re-pinned 2026-08-28 for DECODE_MBU + the 27B's measured mtp: slower
  // decode holds each request in the batch longer, so the steady batch GROWS
  // and each member runs slower. 2.2 at 430 -> ~6.5 at ~135. Mirrors the
  // same re-pin in scenario_model.py's _selfcheck.
  console.assert(approx(sd.n, 6.8, 0.08) && approx(sd.pu, 138, 0.09),
    "27B/TP2 at the reference load: ~6.8 sequences decoding, ~138 tok/s each");
  console.assert(sd.n < 480/5 && sd.pu > 4*interpAt(dc,'p50',480,true),
    "the steady batch must sit far left of the sweep, and far above its speed");
  // 4. monotone and linear in the load, exactly (demand is linear, and the
  //    inversion is of a fixed curve)
  console.assert(steadyDecodePoint(dc, tp2, 2*rate, OUT).n > sd.n
              && steadyDecodePoint(dc, tp2, rate, 2*OUT).n > sd.n,
    "steady concurrency must rise with both arrival rate and output length");
  console.assert(approx(steadyDecodePoint(dc, tp2, 2*rate, OUT).demand,
                        steadyDecodePoint(dc, tp2, rate, 2*OUT).demand, 1e-12),
    "only the PRODUCT rate x output tokens can move the demand");
  // 5. no load, no decoders — and no division by zero on the way there
  console.assert(steadyDecodePoint(dc, tp2, 0, OUT).n === 0,
    "zero arrival rate must leave the decode batch empty");
  // 6. a demand past the curve's top is CENSORED, not clamped silently
  const hot = steadyDecodePoint(dc, tp2, 1e6, OUT);
  console.assert(hot.saturated && hot.n === dc.ns[dc.ns.length-1],
    "demand beyond the decode curve must flag saturated rather than invent a batch");
  // the two aggregates are DIFFERENT quantities on the saturated path — that is
  // the whole point of splitting them, and chart D must plot the demand
  console.assert(hot.demanded > hot.delivered,
    "a saturated point demands more than it delivers");
  console.assert(Math.abs(sd.demanded - sd.delivered) < 1e-6,
    "an unsaturated point delivers exactly what it demands");
  // 7. DP: the batch is PER GROUP, the aggregate is the system. A 2-group grid
  //    at the same per-group rate must report the same n and twice the tokens.
  // a bare {replicas} stub, not makeTopo: this block runs before `state`
  // exists, and steadyDecodePoint reads nothing else off the topology
  const sdDp = steadyDecodePoint({...dc, agg:dc.agg.map(v=>v*2)},
                                 {replicas:2}, rate, OUT);
  console.assert(approx(sdDp.n, sd.n, 1e-9)
              && approx(sdDp.demanded, 2*sd.demanded, 1e-9)
              && approx(sdDp.delivered, 2*sd.delivered, 1e-9),
    "steady n is per replica group; only the aggregates carry the replica count");
}
