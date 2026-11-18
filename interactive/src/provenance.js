/* Where the inputs of the configuration on screen come from.
   The catalogue (src/workingset/data/catalog.toml) gives every model and GPU
   quantity that feeds a verdict a provenance; workingset.provenance turns each
   into the words shown for the GPU on screen, and scripts/catalog.py writes
   them into catalog.js as provenance_labels. Nothing here decides a label: it
   looks one up, and a quantity without one throws. */
import { CATALOG } from './catalog.js';
import { esc } from './svg.js';

const LABELS = CATALOG.provenance_labels, WORDS = CATALOG.provenance_words;

// the inputs each ceiling is priced from, as [side, quantity]. A cold request
// is priced by prefill, and how many requests are cold by the cache pool.
const DECODE = [['gpu','decode'], ['model','decode'], ['model','kv'], ['model','draft'], ['model','acceptance']];
const COLD = [['gpu','prefill'], ['model','prefill'], ['gpu','memory'], ['gpu','reserve'], ['model','weights'], ['model','kv']];
export const CEILING_INPUTS = {
  decode: DECODE,
  cold_wait: COLD,
  saturation: COLD,
  slowed: [...COLD, ...DECODE.filter(([s, q]) => !(s === 'model' && q === 'kv'))],
};

function entry(model, gpu, wdt, side, q){
  if (side === 'model' && q === 'weights' && wdt === 'nvfp4') q = 'weights_nvfp4';
  const label = side === 'gpu' ? LABELS[gpu]?.gpu[q] : LABELS[gpu]?.models[model]?.[q];
  const row = side === 'gpu' ? CATALOG.gpus[gpu]?.provenance?.[q] : CATALOG.models[model]?.provenance?.[q];
  if (!label || !row) throw new Error(`no provenance for ${side} ${side === 'gpu' ? gpu : model} ${q}: catalog.toml needs one`);
  return { label, kind: row.kind, note: row.note, ref: row.ref || '', words: WORDS[side][q] };
}

// strongest first: the reader's eye ends on the weakest inputs
const RANK = l => l.startsWith('measured') ? 0 : l.startsWith('derived') ? 1
  : l === 'vendor figure' ? 2 : l.startsWith('fitted on') ? 3 : 4;

/* [{label, items: [{words, note, ref}]}], one group per label, strongest
   first; `inputs` is a list of [side, quantity]. */
export function provenanceGroups(model, gpu, wdt, inputs){
  const groups = new Map();
  for (const [side, q] of inputs){
    const e = entry(model, gpu, wdt, side, q);
    if (!groups.has(e.label)) groups.set(e.label, []);
    groups.get(e.label).push(e);
  }
  return [...groups].map(([label, items]) => ({ label, items }))
    .sort((a, b) => RANK(a.label) - RANK(b.label));
}

export const MODEL_INPUTS = Object.keys(WORDS.model).filter(q => q !== 'weights_nvfp4').map(q => ['model', q]);
export const GPU_INPUTS = Object.keys(WORDS.gpu).map(q => ['gpu', q]);

// one line of plain text for a tooltip: "measured: weights · guess: draft pass"
export function provenanceText(groups){
  return groups.map(g => `${g.label}: ${g.items.map(i => i.words).join(', ')}`).join(' · ');
}

// the same groups as chips, for the binding-constraint tile: one badge per
// label, each input a tooltip with its note
export function provenanceHtml(groups){
  return '<div class="prov"><span class="plead">inputs</span>' + groups.map(g =>
    `<span class="pgroup"><span class="badge">${esc(g.label)}</span>`
    + g.items.map(i => `<span class="tip" tabindex="0" data-tip="${esc(i.note)}">${esc(i.words)}</span>`).join(', ')
    + '</span>').join('') + '</div>';
}
