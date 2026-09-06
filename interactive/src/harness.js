import { CONFIG, PREFILL_MFU_HI, PREFILL_MFU_LO } from './config.js';
import { PREFILL_CHUNK, coldRequestSeconds, decodeFloor, prefillChunk, prefillSeconds } from './prefill.js';
import { p_sub } from './workload.js';
import { ramPerCache, state } from './state.js';
import { esc, fmt } from './svg.js';
import { lastCS, lastSteady, lastWarmCur } from './render.js';
import { encodeStateURL } from './main.js';

/* ---- "TEST THESE HYPOTHESES" — the run configuration ----------------------
   The deploy card says what to run; this card says how to CHECK it. The
   button hands out a `workingset.toml`: the configuration on screen and
   nothing else. Predictions never travel in the file — the package computes
   them from it, so a config can never carry a number the code did not
   produce. The H-* statements the run will score stay ON THIS PAGE, next to
   the sliders that made them.

   The field names below are `workingset.config.RunConfig` one-for-one; the
   Python side is the source of truth for what each one means. */

// The three commands the card prints. `workingset` publishes ONE console
// script, `ws`, so the package name has to travel in --from: `uvx workingset`
// resolves the package but finds no executable of that name.
const WS_CMDS = [
  'uvx --from workingset ws predict workingset.toml',
  'uvx --from workingset ws test workingset.toml --dry-run',
  'uvx --from workingset ws test workingset.toml --all --exclusive --out run.json',
];
const WS_CMD_NOTE =
  '# until the PyPI release, read the package straight from git:\n'
  + '#   uvx --from git+https://github.com/T0mSIlver/working-set ws predict workingset.toml';

/* ---- workingset.toml ------------------------------------------------------ */
// TOML scalars, matching workingset.config._dump_toml: JSON string quoting
// (so a quote or a backslash in a served-model id cannot break the file),
// lower-case booleans, bare numbers.
function tomlScalar(v){
  if (typeof v === 'boolean') return v ? 'true' : 'false';
  if (typeof v === 'number'){
    if (!isFinite(v)) throw new Error('TOML has no non-finite number');
    return String(v);
  }
  return JSON.stringify(String(v));
}
function tomlBlock(name, rows){
  return [`[${name}]`, ...rows.map(([k, v]) => `${k} = ${tomlScalar(v)}`), ''].join('\n');
}

// The run configuration on screen, as the file `ws predict` / `ws test` read.
// Every figure is PER REPLICA GROUP — one endpoint is one group — so the
// per-group user count and the per-group share of the offload buffer are what
// go in, exactly as the ceilings on this page are quoted.
export function workingsetConfig(state, model, topo, wl){
  const reps = topo.replicas || 1;
  const base = CONFIG.MODELS[state.model].name.split(" (")[0];
  const int = x => Math.round(x);
  // trim binary-float dust (0.7000000000000001) without moving any value the
  // sliders can produce: their finest step is far coarser than 1e-6
  const flt = x => Math.round(x * 1e6) / 1e6;

  const head = [
    `# workingset.toml — the configuration on screen in the interactive explorer.`,
    `# ${base} · ${topo.name} · max_num_batched_tokens ${prefillChunk()}`
      + ` · generated ${new Date().toISOString().slice(0, 10)}`,
    `# reproduce this page: ${encodeStateURL()}`,
  ];
  if (reps > 1) head.push(
    `# DP${reps}: every figure is PER REPLICA GROUP — point base_url at ONE group.`);
  head.push(`# No predictions live here: \`ws predict workingset.toml\` computes them.`);

  const blocks = [
    // what the server under test is started with — every prediction is priced
    // at THIS chunk, and a run record carries the whole block, so an A/B run
    // (4,096 vs 32,768) is self-labeling
    ['deployment', [
      ['model', state.model],                       // key into workingset.model.MODELS
      ['gpu', state.gpu],
      ['tensor_parallel', int(topo.tp)],
      ['replicas', int(reps)],
      ['weight_dtype', state.wdt],
      ['kv_dtype', state.kv],
      ['max_num_batched_tokens', int(prefillChunk())],
      ['max_model_len', int(wl.cap)],
      ['ram_gib', flt(ramPerCache(topo))],
    ]],
    ['workload', [
      ['system_prefix_tokens', int(wl.sys_user)],
      ['user_prompt_median_tokens', int(wl.user_median)],
      ['user_prompt_sigma', flt(wl.user_sigma)],
      ['warm_turn_tokens', int(state.turn)],
      ['think_time_s', flt(state.think)],
      ['subagent_ratio', flt(wl.sub_ratio)],
      ['subagent_median_tokens', int(wl.sub_median)],
      ['subagent_sigma', flt(wl.sub_sigma)],
      ['subagent_prefix_tokens', int(wl.sys_sub)],
      ['sub_shares_prefix', !!wl.sub_shares_prefix],
      ['miss_rate', flt(wl.invalidation)],
      ['max_output_tokens', int(state.out)],
      ['users', Math.max(1, Math.round(state.users / reps))],
    ]],
    ['slo', [
      ['ttft_budget_s', flt(state.sla)],
      ['itl_floor_tok_s', flt(decodeFloor())],
      ['percentile', 95],
    ]],
    ['endpoint', [
      ['base_url', 'http://localhost:8000/v1'],
      ['model', `<your served model id — a ${base} `
                + `${state.wdt === 'nvfp4' ? 'NVFP4' : 'FP8'} checkpoint>`],
      ['api_key_env', 'VLLM_API_KEY'],              // env var NAME, never the key
    ]],
    // the study's two measured-efficiency constants, as the sliders have them
    ['calibration', [
      ['mfu', flt(state.mfu)],
      ['mbu', flt(state.mbu)],
    ]],
  ];
  return head.join('\n') + '\n\nschema_version = 1\n\n'
       + blocks.map(([n, rows]) => tomlBlock(n, rows)).join('\n');
}

/* ---- the predictions the hypotheses quote -------------------------------- */
// All figures are PER REPLICA GROUP — a run drives one endpoint, and one
// endpoint is one group.
// Worst ITL freeze, ms: the LAST chunk of a full-context cold re-prefill
// joins the decode batch, so every decoder sees one gap of step-time +
// chunk-time. MARGINAL pricing (the host pass streams the weights anyway),
// prior = cap - C — mirrors itl_spike / workingset.predict.freeze_ms.
function freezeMs(model, topo, wl, C, pu, mfu){
  // a chunk larger than the whole context is one cap-sized pass with no
  // cache behind it — never a full C with a negative prior
  const step = Math.min(C, wl.cap);
  return 1000 * (model.mtp / pu + prefillSeconds(model, topo, step, wl.cap - step, mfu));
}
function harnessPredictions(op, model, wl, topo){
  const reps = topo.replicas || 1, perG = x => Math.round(x / reps);
  const P = {
    warm_capacity_p5: Math.round(lastWarmCur.p5 * (1 - p_sub(wl))),
    decode_ceiling_users: perG(op.ceilings.decode),
    latency_ceiling_users: perG(op.ceilings.latency),
    saturation_ceiling_users: isFinite(op.ceilings.saturation) ? perG(op.ceilings.saturation) : 999999,
    binding_constraint: op.binding,
    predicted_limit_users: perG(op.limit),
    // the load ttft_miss_s and bstar_misses are computed AT — a run ladders
    // through it and reads those verdicts there, not at the limit
    operating_point_users: Math.max(1, perG(op.users)),
    ttft_miss_s: Math.round(op.ttftMiss * 10) / 10,
    bstar_misses: Math.round(op.bstar * 10) / 10,
  };
  // the ITL / steady-decode predictions exist only where the steady point
  // does (duty < 1 and the demand is on the sampled axis) — every hypothesis
  // that quotes them is dropped without them
  if (lastSteady && lastSteady.real && lastSteady.pu > 0){
    const C = prefillChunk();
    P.steady_decode_seqs = Math.round(lastSteady.n * 100) / 100;
    P.steady_decode_tok_s = Math.round(lastSteady.pu);
    P.itl_normal_ms = Math.round(1000 * model.mtp / lastSteady.pu * 10) / 10;
    P.itl_worst_freeze_ms = Math.round(freezeMs(model, topo, wl, C, lastSteady.pu));
  }
  return P;
}
function harnessHypotheses(P, model, topo, wl, reps){
  const grp = reps > 1 ? ' (per replica group)' : '';
  const out = [
    `H-cache: >= ${fmt(P.warm_capacity_p5, 0)} user sessions stay warm (p5)${grp}. `
      + `A run bounds this below unless load reaches eviction.`,
    `H-decode: per-user p50 decode holds >= ${fmt(decodeFloor(), 0)} tok/s up to `
      + `~${fmt(P.decode_ceiling_users, 0)} concurrent users${grp}.`,
    `H-latency: a cache miss's mean TTFT reaches the ${fmt(state.sla, 0)} s budget `
      + `near ~${fmt(P.latency_ceiling_users, 0)} users${grp}.`,
    `H-saturation: prefill duty reaches 100% near ~${fmt(P.saturation_ceiling_users, 0)} `
      + `users${grp}; above it the queue has no steady state.`,
    `H-binding: the binding constraint is '${P.binding_constraint}' — measured SLO `
      + `capacity should land near ${fmt(P.predicted_limit_users, 0)} users${grp}.`,
    `H-ttft-miss: a forced miss's mean TTFT at the ~${fmt(P.operating_point_users, 0)}-user `
      + `operating point is ~${P.ttft_miss_s} s (read at the ladder rung nearest that load).`,
    `H-burst (needs --burst): at the ~${fmt(P.operating_point_users, 0)}-user standing load, `
      + `a simultaneous flush of <= ${Math.floor(P.bstar_misses)} misses (B* = ${P.bstar_misses}) `
      + `drains inside the TTFT budget; a larger one does not.`,
  ];
  // the ITL / steady-decode hypotheses, present only when the steady point is
  // real (same guard as harnessPredictions). One arm each: an A/B against
  // another chunk setting is two configs, compared by their run records.
  if (P.itl_worst_freeze_ms != null){
    const C = prefillChunk(), pu = lastSteady.pu;
    // higher MFU = shorter freeze, so the HI anchor is the bracket's low edge
    const brLo = Math.round(freezeMs(model, topo, wl, C, pu, PREFILL_MFU_HI));
    const brHi = Math.round(freezeMs(model, topo, wl, C, pu, PREFILL_MFU_LO));
    const ref = Math.round(freezeMs(model, topo, wl, PREFILL_CHUNK, pu));
    const vsRef = C !== PREFILL_CHUNK
      ? `vs ~${fmt(ref, 0)} ms at the 32,768 study default. The spike MAGNITUDE `
        + `scales ~inversely with the unvalidated MFU (bracket above), but the `
        + `RATIO between the two settings does not — quote the ratio, not the `
        + `milliseconds. `
      : ``;
    // per-pass overhead: total machine time a full miss costs at this chunk
    // vs the 32,768 default (research/prefill.md — FLOPs telescope, the
    // weight-stream overhead does not). lastCS = the render's shared context
    // draw; an ad-hoc contextStats() here would disturb the seeded RNG stream.
    const overheadPct = lastCS
      ? (coldRequestSeconds(model, topo, wl, lastCS, C)
         / coldRequestSeconds(model, topo, wl, lastCS, PREFILL_CHUNK) - 1) * 100
      : 0;
    out.push(
      `H-steady: at the ~${fmt(P.operating_point_users, 0)}-user operating point the `
        + `decode batch holds ~${P.steady_decode_seqs} sequences at `
        + `~${fmt(P.steady_decode_tok_s, 0)} tok/s each — NOT the whole warm pool at `
        + `the stress figure. Read against the measured concurrent-decode count, `
        + `not the population.`,
      `H-itl-spike: the worst freeze behind one chunk of a `
        + `${fmt(wl.cap / 1000, 0)}k cold re-prefill is ~${fmt(P.itl_worst_freeze_ms, 0)} ms `
        + `[${fmt(brLo, 0)}-${fmt(brHi, 0)}, the MFU 35-55% bracket] at `
        + `max_num_batched_tokens=${fmt(C, 0)} ${vsRef}`
        + `Read it on the INTER-TOKEN GAPS table, never on decode p50 (a mean `
        + `over the stream is nearly blind to a freeze); make sure `
        + `--freeze-threshold-ms sits below the predicted freeze.`,
      `H-itl-mean: the normal inter-token gap (~${P.itl_normal_ms} ms) is `
        + `~unchanged across chunk settings — prefill FLOPs telescope; only the `
        + `per-pass overhead does not (a full miss costs `
        + `${overheadPct >= 0 ? '+' : ''}${fmt(overheadPct, 1)}% total machine time at `
        + `${fmt(C, 0)} vs the 32,768 default). NOTE a few ms is near the `
        + `client-side timing floor — check the 'floor' column before reading `
        + `anything into this row.`);
  }
  return out;
}
let lastHarnessArgs = null;
export function renderTestCard(op, model, topo, wl){
  const box = document.getElementById('testBody'); if (!box) return;
  if (!op){
    lastHarnessArgs = null;
    box.innerHTML = `<p class="cs">model weights do not fit this configuration — `
      + `nothing to test.</p>`;
    return;
  }
  lastHarnessArgs = { op, model, topo, wl };
  const reps = topo.replicas || 1;
  const hyp = harnessHypotheses(harnessPredictions(op, model, wl, topo),
                                model, topo, wl, reps);
  // the list is the longest text block in act 3, so it folds: the summary
  // line still says how many hypotheses there are and names the binding one
  box.innerHTML =
    `<details class="assump" style="margin:0 0 10px">`
    + `<summary>The ${hyp.length} hypotheses it will test — headline: ${esc(hyp[4])}</summary>`
    + `<div class="body"><ul style="margin:6px 0 4px;padding-left:18px;font-size:12.5px;color:var(--text-2)">`
    + hyp.map(h => `<li style="margin:3px 0">${esc(h)}</li>`).join('') + `</ul></div></details>`
    + `<div class="preset"><button type="button" id="dlHarness">⬇ download workingset.toml</button>`
    + `<button type="button" id="cpHarness">copy config</button>`
    + `<span class="plab" style="text-transform:none;letter-spacing:0">`
    + (reps > 1
        ? `every figure is per replica group — point base_url at ONE group`
        : `then, next to the file:`)
    + `</span></div>`
    + `<div class="cmdwrap" style="margin-top:10px"><pre class="cmd">`
    + esc(WS_CMDS.join('\n'))
    + `\n<span class="cmt">${esc(WS_CMD_NOTE)}</span></pre>`
    + `<button type="button" class="copybtn" id="cpCmds">copy</button></div>`;
}
document.getElementById('testCard').addEventListener('click', e => {
  if (!lastHarnessArgs) return;
  const a = lastHarnessArgs;
  const flash = (btn, t, back) => { btn.textContent = t;
    setTimeout(() => { btn.textContent = back; }, 1500); };
  const copy = (text, btn, back) => {
    if (!navigator.clipboard?.writeText){ flash(btn, 'blocked', back); return; }
    navigator.clipboard.writeText(text)
      .then(() => flash(btn, 'copied', back)).catch(() => flash(btn, 'blocked', back));
  };
  if (e.target.id === 'dlHarness'){
    const text = workingsetConfig(state, a.model, a.topo, a.wl);
    const blob = new Blob([text], { type: 'application/toml' });
    const el = document.createElement('a');
    el.href = URL.createObjectURL(blob);
    el.download = 'workingset.toml';
    document.body.appendChild(el); el.click(); el.remove();
    setTimeout(() => URL.revokeObjectURL(el.href), 5000);
  } else if (e.target.id === 'cpHarness'){
    copy(workingsetConfig(state, a.model, a.topo, a.wl), e.target, 'copy config');
  } else if (e.target.id === 'cpCmds'){
    copy(WS_CMDS.join('\n'), e.target, 'copy');
  }
});
