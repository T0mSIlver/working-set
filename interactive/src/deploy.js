import { CONFIG } from './config.js';
import { prefillChunk } from './prefill.js';
import { p_sub } from './workload.js';
import { currentTopo, ramPerCache, state } from './state.js';
import { esc, fmt } from './svg.js';
import { activeModel, fitHintFor, lastSteady, lastStress, lastWarmCur } from './render.js';
import { PLANNER_LABEL } from './planner.js';

/* ---- The deploy card ------------------------------------------------------
   The page's answer as a recipe: the current controls spelled out as the vLLM
   flags they imply, next to the outcomes the model projects for them. The
   checkpoint id is a placeholder on purpose — the study models several
   quantization recipes per model and does not pin registry paths. */
export let deployCmdText = '', lastDeploy = null;
export function renderDeployCard(op, model, topo, wl, mo, decodeUsers){
  const box = document.getElementById('deployBody'); if (!box) return;
  if (!op){
    box.innerHTML = `<p class="cs">model weights do not fit this configuration — `
      + `${esc(fitHintFor(activeModel(), currentTopo()))}</p>`;
    deployCmdText = ''; lastDeploy = null;
    return;
  }
  lastDeploy = { op, model, topo, wl, mo, decodeUsers };   // theme flips re-render from this
  const reps = topo.replicas||1, dp = topo.dp||1;
  const m0 = CONFIG.MODELS[state.model];
  const base = m0.name.split(" (")[0];
  const warmSys = lastWarmCur.p5*reps;
  const userSys = warmSys*(1-p_sub(wl));
  // largest decode batch that keeps per-user p50 at the 40 tok/s floor, capped
  // at the GPU-resident warm population — beyond it there is nobody warm to admit
  const sug = Math.max(1, Math.round(Math.min(decodeUsers, lastWarmCur.g5)));
  // draft counts mirror the research notes; a model missing here must not
  // emit num_speculative_tokens:undefined, so it gates specOn too
  const specDrafts = {"27B":2, "35BA3B":2, "GLM52":5, "DSV4F":7, "Q38FN":3, "GLM53F":5}[state.model];
  const specOn = state.mtp>1 && m0.mtp>1 && specDrafts !== undefined;
  const eagle = state.mtp>1 && m0.mtp<=1;   // Mistral: the slider models an external draft
  const ramGrp = ramPerCache(topo);

  // ONE replica group's command. The model prices DP as INDEPENDENT groups
  // behind a sticky router — which vLLM's internal DP load balancer is not —
  // so DP is "run N of these", never --data-parallel-size.
  const lines = [
    `vllm serve "<${base} ${state.wdt==='nvfp4'?'NVFP4':'FP8'} checkpoint>"`,
    `  --tensor-parallel-size ${topo.tp}`,
    `  --kv-cache-dtype ${state.kv==='fp8'?'fp8_e4m3':'auto'}`,
    `  --max-model-len ${wl.cap}`,
    `  --max-num-seqs ${sug}`,
    `  --max-num-batched-tokens ${prefillChunk()}`,
  ];
  // Q3.8-Flash-Next's vLLM recipe: plain TP8 is incompatible with the FP8
  // checkpoint (128-wide quantization blocks) — TP8 must run as TEP8. The
  // triton MoE backend is the recipe's HOPPER guidance only; Blackwell rows
  // never mention it, so it is not emitted on B300.
  if (state.model==="Q38FN" && topo.tp>=8)
    lines.splice(2, 0, `  --enable-expert-parallel`,
                 ...(state.gpu==="H200" ? [`  --moe-backend triton`] : []));
  if (state.state_dt==='fp32' && m0.deltanet_state>0 && m0.state_fp32_ok!==false)
    lines.push(`  --mamba-ssm-cache-dtype float32`);
  if (specOn)
    lines.push(`  --speculative-config '{"method":"mtp","num_speculative_tokens":${specDrafts}}'`);
  if (state.ram>0)
    lines.push(`  --kv-offloading-size ${fmt(ramGrp,0)}`);
  const cmts = [];
  if (dp>1) cmts.push(`# one replica GROUP — run ${dp} of these behind a sticky (session-affinity) router`);
  if (state.model==="Q38FN" && topo.tp>=8)
    cmts.push(`# TEP8: plain TP8 is incompatible with this FP8 checkpoint (vLLM recipe)`);
  if (state.model==="Q38FN" && topo.tp===1 && state.gpu==="B300")
    cmts.push(`# 1×B300 passes the pool arithmetic; the recipe's validated floor is TP2 (TP1 hit compilation OOM on GB300)`);
  if (state.model==="GLM53F" && state.gpu==="H200")
    cmts.push(`# BF16 KV is REQUIRED on Hopper for this model — fp8 KV is Blackwell-only (vLLM recipe); the KV toggle locks accordingly`);
  if (state.model==="GLM53F" && topo.tp<4)
    cmts.push(`# TP<4 is this study's pool arithmetic; the vLLM recipe only demonstrates TP4 (one GB200 tray)`);
  if (state.kv==='fp16') cmts.push(`# auto = the checkpoint's 16-bit dtype (the study's FP16-KV case)`);
  if (specOn) cmts.push(`# speculative decoding modelled at ${state.mtp.toFixed(2)}×${state.model==='DSV4F'?' (DSpark drafts)':''}`);
  if (eagle) cmts.push(`# the modelled ${state.mtp.toFixed(2)}× speedup assumes an EXTERNAL EAGLE-style draft (no MTP module; unmeasured)`);
  if (state.ram>0) cmts.push(`# --kv-offloading-size is GiB per group${dp>1?` (${fmt(state.ram,0)} GiB total across ${dp} groups)`:''}; a storage tier — restore latency unpriced`);
  deployCmdText = lines.join(' \\\n') + (cmts.length ? '\n'+cmts.join('\n') : '');

  const spec = [
    ['Model', esc(base)],
    ['Weights', (state.wdt==='nvfp4'?'NVFP4':'FP8') + (state.wover==='p15'?' (+15% deployed)':'')],
    ['Hardware', esc(topo.name)],
    ['KV cache', state.kv==='fp8'?'FP8 (fp8_e4m3)':'FP16 (auto)'],
    ['max_model_len', `${fmt(wl.cap,0)} tok`],
    ['max_num_seqs', `${fmt(sug,0)}${reps>1?' per group':''}`],
    ['CPU offload', state.ram>0?`${fmt(state.ram,0)} GiB${dp>1?` (${fmt(ramGrp,0)}/group)`:''}`:'off'],
    ['Speculative', specOn?`MTP ${state.mtp.toFixed(2)}× (${specDrafts} drafts)`
                   :(eagle?`EAGLE-style ${state.mtp.toFixed(2)}× (external, unmeasured)`:'off')],
  ].map(([k,v])=>`<dt>${k}</dt><dd class="tnum">${v}</dd>`).join('');
  const out = [
    ['Warm sessions (p5)', `${fmt(warmSys,0)} · ≈ ${fmt(userSys,0)} users`],
    ['Per-user decode, at this load', lastSteady && lastSteady.real && lastSteady.n>0
       ? `${fmt(lastSteady.pu,0)} tok/s · ${fmt(lastSteady.n, lastSteady.n<10?1:0)} decoding at once`
       : '—'],
    ['Per-user decode, all warm active', `${fmt(lastStress.pu,1)} tok/s`],
    ['Max cold req/s', `${fmt(1/mo.miss,2)}${reps>1?' per group':''}`],
    ['Cold-spike tolerance B*', `${fmt(op.bstar,1)} misses at once${reps>1?' per group':''}`],
  ].map(([k,v])=>`<dt>${k}</dt><dd class="tnum">${v}</dd>`).join('');
  // same thresholds as the binding-constraint tile, so the two act-3 panels
  // can never disagree about the same operating point (headroom = users/limit)
  const vcol = op.headroom>=1 ? 'var(--crit)'
             : (op.headroom>=0.8 ? 'var(--warn)' : 'var(--good)');
  const verdict = op.limit>=1
    ? `${op.fits?'✓ serves':'✗ cannot serve'} your ${fmt(op.users,0)} users — `
      + `${PLANNER_LABEL[op.binding]} binds at ${fmt(op.limit,0)} (×${fmt(op.limit/op.users,1)} your load)`
    : `✗ not viable at any load — ${PLANNER_LABEL[op.binding]} binds below one user`;
  box.innerHTML =
    `<div class="deploy">`
    + `<div><h4>Run</h4><dl class="spec">${spec}</dl></div>`
    + `<div><h4>Expect</h4><dl class="spec">${out}</dl></div>`
    + `<p class="verdictline" style="color:${vcol}">${esc(verdict)}</p>`
    + `<div class="cmdwrap"><pre class="cmd">${esc(lines.join(' \\\n'))}`
    + (cmts.length?`\n<span class="cmt">${cmts.map(esc).join('\n')}</span>`:'')
    + `</pre><button type="button" class="copybtn" id="copyCmd">copy</button></div>`
    + `</div>`;
}
