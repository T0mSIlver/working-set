import { CONFIG, tokensPerStep } from './config.js';
import { prefillChunk } from './prefill.js';
import { p_sub } from './workload.js';
import { currentTopo, ramPerCache, state } from './state.js';
import { dcpSize, kvReplication } from './config.js';
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
  // the recommended cap the planner priced (recommendedMns): the largest decode
  // batch that keeps per-user p50 at the floor, within the GPU-resident warm
  // population — beyond it there is nobody warm to admit
  const sug = op.mns;
  // the draft count, method and any non-default speculative config come
  // from the catalogue; a model without drafts emits no speculative flags
  const ex = m0.explorer;
  // the page's draft depth (the model's recipe unless moved) is the one the
  // command starts the server with
  const specDrafts = state.dspec;
  const dspark = m0.draft_method === "dspark";
  const specOn = specDrafts > 0 && m0.spec_tokens > 0;
  const eagle = specDrafts > 0 && m0.spec_tokens === 0;   // Mistral-Medium: an external draft
  const tps = tokensPerStep(state.acc, specDrafts).toFixed(2);
  const ramGrp = ramPerCache(topo);
  // the catalogue's recipe rules that hold for this GPU and TP width
  const rules = (ex.recipe || []).filter(({ when = {} }) =>
    (when.gpu === undefined || when.gpu === state.gpu)
    && (when.tp_min === undefined || topo.tp >= when.tp_min)
    && (when.tp_max === undefined || topo.tp <= when.tp_max));
  // --kv-cache-dtype: fp8_e4m3 / auto unless the catalogue names the
  // model's own (a Hopper-only fp8 spelling, a bfloat16 BF16 arm)
  const kvDt = { fp8: 'fp8_e4m3', fp16: 'auto', ...ex.kv_cache_dtype };
  const hopper = !CONFIG.GPUS[state.gpu].supports_nvfp4;
  const kvFlag = state.kv === 'fp8' ? (hopper && kvDt.fp8_hopper || kvDt.fp8) : kvDt.fp16;
  // the base checkpoint's weight format: FP8 unless the catalogue names the
  // model's own (Kimi-K3.1-Pro ships MXFP4 experts and BF16 elsewhere)
  const wName = state.wdt === 'nvfp4' ? 'NVFP4' : (ex.weights ?? 'FP8');

  // ONE replica group's command. The model prices DP as INDEPENDENT groups
  // behind a sticky router — which vLLM's internal DP load balancer is not —
  // so DP is "run N of these", never --data-parallel-size.
  const lines = [
    `vllm serve "<${base} ${wName} checkpoint>"`,
    `  --tensor-parallel-size ${topo.tp}`,
    // Mistral-Large-4's FlashMLA sparse path names its own FP8 layout (its
    // NVFP4 card: "needed for this checkpoint"). GLM-5.3's recipe passes fp8
    // (FlashMLA's fp8_ds_mla) on Hopper and fp8_e4m3 on Blackwell, where
    // vLLM also turns auto into fp8, so its BF16 arm must say bfloat16.
    `  --kv-cache-dtype ${kvFlag}`,
    `  --max-model-len ${wl.cap}`,
    `  --max-num-seqs ${state.mns ?? sug}`,
    `  --max-num-batched-tokens ${prefillChunk()}`,
  ];
  // model flags from the catalogue's recipe rules, each rule's flags at its
  // `at` line (default 2: after --tensor-parallel-size)
  for (const at of [...new Set(rules.filter(r => r.flags).map(r => r.at ?? 2))].sort((a, b) => b - a))
    lines.splice(at, 0, ...rules.filter(r => r.flags && (r.at ?? 2) === at).flatMap(r => r.flags.map(f => `  ${f}`)));
  if (state.state_dt==='fp32' && m0.deltanet_state>0 && m0.state_fp32_ok!==false)
    lines.push(`  --mamba-ssm-cache-dtype float32`);
  // KV cache under TP: ranks beyond the model's KV heads either split the
  // cache along the sequence (DCP — the layout the page's capacity prices by
  // default) or keep a full copy each (plain TP, the "Replicated" arm)
  const dcp = dcpSize(m0, topo), [rKv] = kvReplication(m0, topo);
  if (dcp > 1) lines.splice(2, 0, `  --decode-context-parallel-size ${dcp}`);
  // a width the KV heads neither divide nor are divided by is one vLLM
  // refuses outright (TP6 on 4 heads); the page prices it as sharded, an
  // extrapolation this recipe cannot realize
  const heads = m0.kv_heads || 1;
  const illegalTp = topo.tp > 1 && (topo.tp % heads !== 0) && (heads % topo.tp !== 0);
  // speculative decoding under DCP was still in development in vLLM at the
  // cited date (2026-08-07): never emit the two flags together — except on a
  // model whose recipe validates the pair (spec_with_dcp)
  const specBlocked = specOn && dcp > 1 && !ex.spec_with_dcp;
  if (specOn && !specBlocked)
    lines.push(`  --speculative-config '${ex.speculative_config
        ?.replace(/"num_speculative_tokens":\d+/, `"num_speculative_tokens":${specDrafts}`)
      ?? `{"method":"mtp","num_speculative_tokens":${specDrafts}}`}'`);
  if (state.ram>0)
    lines.push(`  --kv-offloading-size ${fmt(ramGrp,0)}`);
  const cmts = [];
  if (dp>1) cmts.push(`# one replica GROUP — run ${dp} of these behind a sticky (session-affinity) router`);
  for (const r of rules) if (r.note && !r.after_spec) cmts.push(`# ${r.note}`);
  if (state.kv==='fp16' && kvFlag==='auto') cmts.push(`# auto = the checkpoint's 16-bit dtype (the study's FP16-KV case)`);
  if (dcp > 1) cmts.push(`# DCP ${dcp}: ${m0.kv_heads===1?'a single-latent (MQA/MLA) cache':`${m0.kv_heads} KV heads`} would replicate on ${dcp}× the ranks under plain TP${topo.tp}; the page prices ONE copy${ex.spec_with_dcp ? '' : ' (speculative decoding under DCP is still in development in vLLM, 2026-08)'}`);
  if (rKv > 1) cmts.push(`# KV REPLICATED: plain TP${topo.tp} keeps ${rKv}× copies of every session's cache — the page prices the pool ÷ ${rKv} and ${rKv}× per-step cache reads; --decode-context-parallel-size ${Math.max(1, Math.floor(topo.tp/(m0.kv_heads||1)))} would remove them`);
  if (specOn && !specBlocked) cmts.push(`# speculative decoding modelled at ${tps} tokens per step (${specDrafts} drafts at acceptance ${state.acc.toFixed(3)}${dspark?', DSpark':''})`);
  for (const r of rules) if (r.note && r.after_spec) cmts.push(`# ${r.note}`);
  if (specBlocked) cmts.push(`# NOT EMITTED: --speculative-config — the page prices ${specDrafts} draft tokens, but vLLM's DCP path had no speculative-decoding support as of 2026-08; set Draft tokens to 0 for a recipe that matches, or choose Replicated`);
  if (illegalTp) cmts.push(`# TP${topo.tp} is not a width vLLM accepts for ${heads} KV heads (tp must divide the heads or the heads divide tp): the page prices this as if the cache sharded — an extrapolation no command realizes`);
  if (eagle) cmts.push(`# the modelled ${tps} tokens per step assume an EXTERNAL EAGLE-style draft (no MTP module; its bytes are unpriced)`);
  if (state.ram>0) cmts.push(`# --kv-offloading-size is GiB per group${dp>1?` (${fmt(state.ram,0)} GiB total across ${dp} groups)`:''}; a storage tier — restore latency unpriced`);
  deployCmdText = lines.join(' \\\n') + (cmts.length ? '\n'+cmts.join('\n') : '');

  const spec = [
    ['Model', esc(base)],
    ['Weights', wName + (state.wover==='p15'?' (+15% deployed)':'')],
    ['Hardware', esc(topo.name)],
    ['KV cache', state.kv==='fp8'?'FP8 (fp8_e4m3)':'FP16 (auto)'],
    ['max_model_len', `${fmt(wl.cap,0)} tok`],
    ['max_num_seqs', state.mns !== null
       ? `${fmt(state.mns,0)}${reps>1?' per group':''} (set; ${op.decodeCapped?'sets the decode ceiling':'does not set the decode ceiling'})`
       : `${fmt(sug,0)}${reps>1?' per group':''} (recommended)`],
    ['CPU offload', state.ram>0?`${fmt(state.ram,0)} GiB${dp>1?` (${fmt(ramGrp,0)}/group)`:''}`:'off'],
    ['Speculative', specOn?`${dspark ? 'DSpark' : 'MTP'} ${specDrafts} drafts · ${tps} tok/step`
                   :(eagle?`EAGLE-style ${specDrafts} drafts · ${tps} tok/step (external, unmeasured)`:'off')],
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
