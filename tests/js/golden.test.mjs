/* Golden vectors: the JS mirror must agree with Python.
 *
 *   node --import ./tests/js/register.mjs --test tests/js/golden.test.mjs
 *   node --import ./tests/js/register.mjs tests/js/golden.test.mjs   (same, direct)
 *
 * tests/golden/vectors.json carries a few hundred explorer states priced by
 * src/workingset/model.py, the tolerance class of every quantity, and the
 * Python -> JS function map. This file replays each state through
 * interactive/src/*.js and asserts agreement.
 *
 * The first run of anything like this is red: constants get typed twice,
 * rounding conventions drift, an approximation is taken on one side only.
 * Loosening the tolerances would hide exactly what the fixture exists to
 * expose, so the known differences live in tests/golden/known_disagreements.json
 * instead — one entry per (quantity, where), each with the relative error
 * actually observed and a one-line hypothesis. CI is green on that set and red
 * on anything new.
 *
 *   GOLDEN_NO_ALLOWLIST=1   ignore the allowlist (shows the raw failure set)
 *   GOLDEN_QUIET=1          skip the tables, print only the verdict
 *   GOLDEN_DUMP=path.json   write every over-tolerance pair out, grouped by
 *                           quantity — how an allowlist entry gets written from
 *                           what was measured rather than from a guess
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import assert from 'node:assert/strict';

import { driveState, warmUsersAt } from './drive.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
const GOLDEN = resolve(HERE, '..', 'golden');
const NO_ALLOWLIST = process.env.GOLDEN_NO_ALLOWLIST === '1';
const QUIET = process.env.GOLDEN_QUIET === '1';

const doc = JSON.parse(readFileSync(resolve(GOLDEN, 'vectors.json'), 'utf8'));
const allow = NO_ALLOWLIST
  ? { entries: [] }
  : JSON.parse(readFileSync(resolve(GOLDEN, 'known_disagreements.json'), 'utf8'));

/* ---- numbers ------------------------------------------------------------ */
// golden.py writes non-finite values as strings so the file stays valid JSON
function num(v){
  if (v === 'Infinity') return Infinity;
  if (v === '-Infinity') return -Infinity;
  if (v === 'NaN') return NaN;
  return v;
}
// Symmetric relative difference, on the larger magnitude. Both-zero is 0;
// one-zero is 1 (a full disagreement), which is the honest reading when one
// side collapses a quantity to nothing.
function relErr(a, b){
  a = num(a); b = num(b);
  if (typeof a === 'boolean' || typeof b === 'boolean') return a === b ? 0 : 1;
  if (Number.isNaN(a) && Number.isNaN(b)) return 0;
  if (Number.isNaN(a) || Number.isNaN(b)) return Infinity;
  if (a === b) return 0;                       // covers +-Infinity === +-Infinity
  if (!Number.isFinite(a) || !Number.isFinite(b)) return Infinity;
  const d = Math.max(Math.abs(a), Math.abs(b));
  return d === 0 ? 0 : Math.abs(a - b) / d;
}

/* ---- tolerance ---------------------------------------------------------- */
const MC = new Set(doc.mc_quantities);
const FLAGS = new Set(doc.flag_quantities);
function klass(q){
  if (FLAGS.has(q)) return 'flag';
  return MC.has(q) ? 'mc' : 'exact';
}
function tolerance(q){
  if (FLAGS.has(q)) return 0;
  if (!MC.has(q)) return doc.tolerances.exact.rel;
  return doc.tolerances.mc[q] ?? doc.tolerances.mc_default;
}

/* ---- the allowlist ------------------------------------------------------ */
// An entry matches a (quantity, vector) pair when the quantity matches (exact
// name, or a trailing '*' glob) and every key of `where` matches the vector's
// state or its derived topology. `max_rel` is the ceiling the entry buys: a
// known 4% difference does not license a new 40% one.
// A `where` clause matches the state's own fields, the derived topology
// (n_gpu / replicas), and the vector's `cond` block under a `_` prefix — the
// conditioning diagnostics scripts/golden.py emits alongside the outputs, so
// an entry can say "where the queue is within 10% of saturation" or "where the
// load runs past the mirror's own decode axis" instead of naming a model.
// They come out of the committed fixture, so they are as deterministic as the
// state itself. See compute()'s `cond` block for what each one means.
function derived(vec){
  const cond = {};
  for (const [k, v] of Object.entries(vec.cond || {})) cond['_' + k] = num(v);
  return {
    ...vec.state,
    n_gpu: vec.state.ngpu,
    replicas: vec.state.ngpu / vec.state.tp,
    ...cond,
  };
}
function qMatch(pattern, q){
  return pattern.endsWith('*') ? q.startsWith(pattern.slice(0, -1)) : pattern === q;
}
function whereMatch(where, d){
  if (!where) return true;
  for (const [k, want] of Object.entries(where)){
    const have = d[k];
    if (Array.isArray(want)){ if (!want.includes(have)) return false; }
    else if (want && typeof want === 'object'){
      if (want.gt !== undefined && !(have > want.gt)) return false;
      if (want.gte !== undefined && !(have >= want.gte)) return false;
      if (want.lt !== undefined && !(have < want.lt)) return false;
      if (want.lte !== undefined && !(have <= want.lte)) return false;
      if (want.ne !== undefined && have === want.ne) return false;
      if (want.nin !== undefined && want.nin.includes(have)) return false;
    }
    else if (have !== want) return false;
  }
  return true;
}
const used = new Set();
function allowed(q, d, rel){
  for (let i = 0; i < (allow.entries || []).length; i++){
    const e = allow.entries[i];
    if (!qMatch(e.quantity, q)) continue;
    if (!whereMatch(e.where, d)) continue;
    // "Infinity" is a real ceiling, not a missing one: it says the two sides
    // disagree about whether a quantity is finite at all (a queue at
    // rho -> 1). It is only ever reachable behind a `where`.
    if (rel <= num(e.max_rel)){ used.add(i); return true; }
  }
  return false;
}

/* ---- run ---------------------------------------------------------------- */
function fmt(x){
  if (x === 0) return '0';
  if (!Number.isFinite(x)) return String(x);
  return x < 1e-4 || x >= 1e4 ? x.toExponential(2) : x.toPrecision(4);
}
function pad(s, n, right){
  s = String(s);
  return right ? s.padStart(n) : s.padEnd(n);
}
function table(title, head, rows){
  if (QUIET || !rows.length) return;
  const w = head.map((h, i) => Math.max(h.length,
    ...rows.map(r => String(r[i]).length)));
  const line = (cells) => '  ' + cells.map((c, i) =>
    pad(c, w[i], i > 0 && i < cells.length - 1)).join('  ');
  console.log(`\n${title}`);
  console.log(line(head));
  console.log('  ' + w.map(n => '-'.repeat(n)).join('  '));
  for (const r of rows) console.log(line(r));
}

test('golden vectors: the JS mirror agrees with workingset.model', (t) => {
  const worstQ = new Map();      // quantity -> {rel, label, py, js, ok}
  const worstM = new Map();      // `${model}|${quantity}` -> {rel, label}
  const failures = [];
  const excused = [];
  const over = [];               // everything past tolerance, allowlisted or not
  const missing = new Set();     // quantities the JS driver does not produce
  let compared = 0, declined = 0;

  for (const vec of doc.vectors){
    let js;
    try {
      js = driveState(vec.state);
    } catch (err){
      failures.push({ q: '(driveState threw)', label: vec.label,
                      rel: Infinity, py: '', js: String(err && err.message) });
      continue;
    }
    const d = derived(vec);
    for (const [q, pyRaw] of Object.entries(vec.out)){
      const py = num(pyRaw);
      // An EXPLICIT null is the fixture declining to state a reference for
      // this state — deliberate, and recorded as such. A quantity simply
      // absent from the driver is a HOLE, and is asserted away below rather
      // than silently reducing what this test covers.
      if (py === null){ declined++; continue; }
      if (!(q in js)){ missing.add(q); continue; }
      const jsv = js[q];
      compared++;
      const rel = relErr(py, jsv);
      const tol = tolerance(q);
      const ok = rel <= tol || allowed(q, d, rel);
      const rec = { q, rel, label: vec.label, py, js: jsv, ok,
                    cls: klass(q), tol };
      const cur = worstQ.get(q);
      if (!cur || rel > cur.rel) worstQ.set(q, rec);
      const mk = `${vec.state.model}|${q}`;
      const curM = worstM.get(mk);
      if (!curM || rel > curM.rel) worstM.set(mk, rec);
      if (!ok) failures.push(rec);
      else if (rel > tol) excused.push(rec);
      if (rel > tol) over.push({ ...rec, state: vec.state });
    }
  }

  // warmUsersCurve: exercised on one state per model so a change to the
  // frontier's sampler cannot pass unseen. Not a golden quantity (the curve is
  // an interpolation over anchors, with no Python counterpart of its own), so
  // it is only asserted to be a finite non-negative number.
  const seenModels = new Set();
  for (const vec of doc.vectors){
    if (seenModels.has(vec.state.model)) continue;
    seenModels.add(vec.state.model);
    const u = warmUsersAt(vec.state);
    assert.ok(Number.isFinite(u) && u >= 0,
      `warmUsersCurve returned ${u} for ${vec.label}`);
  }

  /* ---- coverage: no quantity may go uncompared by accident ---- */
  // Without this, deleting a line from drive.mjs turns a comparison into a
  // silent skip and the suite passes greener than it did before.
  assert.equal(missing.size, 0,
    `tests/js/drive.mjs produces no value for: ${[...missing].sort().join(', ')}`
    + ". Every key of a vector's `out` block must have a JS counterpart; if a "
    + 'quantity genuinely cannot be driven, scripts/golden.py must emit null '
    + 'for it (and say why in the mapping) rather than the driver omitting it.');
  // ...and every quantity the mapping advertises must actually be in `out`
  const outKeys = new Set(Object.keys(doc.vectors[0].out));
  const unmapped = (doc.mapping || [])
    .map(m => m.quantity)
    .filter(q => !q.startsWith('(') && !q.includes('/'))
    .filter(q => q.endsWith('*')
      ? ![...outKeys].some(k => k.startsWith(q.slice(0, -1)))
      : !outKeys.has(q));
  assert.equal(unmapped.length, 0,
    'vectors.json mapping advertises quantities absent from every vector\'s '
    + `out block: ${unmapped.join(', ')}`);

  /* ---- the tables ---- */
  const rows = [...worstQ.values()]
    .sort((a, b) => b.rel - a.rel)
    .map(r => [r.q, fmt(r.rel), fmt(r.tol), r.cls,
               r.ok ? (r.rel > r.tol ? 'allowlisted' : 'ok') : 'FAIL',
               r.label]);
  table('WORST relative disagreement per quantity',
        ['quantity', 'rel', 'tol', 'class', 'status', 'where'], rows);

  const perModel = new Map();
  for (const [key, r] of worstM){
    const model = key.split('|')[0];
    const cur = perModel.get(model);
    if (!cur || r.rel > cur.rel) perModel.set(model, r);
  }
  table('WORST relative disagreement per model',
        ['model', 'rel', 'quantity', 'status', 'where'],
        [...perModel.entries()].sort((a, b) => b[1].rel - a[1].rel)
          .map(([mk, r]) => [mk, fmt(r.rel), r.q,
                             r.ok ? (r.rel > r.tol ? 'allowlisted' : 'ok') : 'FAIL',
                             r.label]));

  const unused = (allow.entries || [])
    .map((e, i) => [e, i]).filter(([, i]) => !used.has(i));
  if (!QUIET){
    console.log(`\n  ${doc.vectors.length} vectors, ${compared} comparisons, `
      + `${declined} declined by the fixture, ${excused.length} allowlisted, `
      + `${failures.length} failing`
      + (NO_ALLOWLIST ? '   [GOLDEN_NO_ALLOWLIST=1: allowlist ignored]' : ''));
    if (unused.length && !NO_ALLOWLIST)
      console.log(`  ${unused.length} allowlist entr${unused.length === 1 ? 'y' : 'ies'} `
        + `matched nothing: ${unused.map(([e]) => e.quantity).join(', ')}`);
  }

  if (process.env.GOLDEN_DUMP){
    const byQ = {};
    for (const r of over){
      (byQ[r.q] ||= []).push({
        rel: r.rel, tol: r.tol, python: r.py, js: r.js, label: r.label,
        state: r.state,
      });
    }
    for (const q of Object.keys(byQ)) byQ[q].sort((a, b) => b.rel - a.rel);
    writeFileSync(process.env.GOLDEN_DUMP,
      JSON.stringify({ n_over: over.length, by_quantity: byQ }, null, 1));
    if (!QUIET) console.log(`  wrote ${process.env.GOLDEN_DUMP}`);
  }

  if (failures.length){
    const shown = failures.slice(0, 40).map(r =>
      [r.q, fmt(r.rel), fmt(r.tol ?? 0), fmt(r.py), fmt(r.js), r.label]);
    table('FAILURES', ['quantity', 'rel', 'tol', 'python', 'js', 'where'], shown);
  }
  // A stale entry is not harmless: it is a standing licence to disagree on a
  // quantity nothing currently disagrees on. Asserted, not logged — QUIET
  // hides logs, and the whole point is that this cannot be missed.
  if (!NO_ALLOWLIST)
    assert.equal(unused.length, 0,
      `${unused.length} entr${unused.length === 1 ? 'y' : 'ies'} in `
      + 'tests/golden/known_disagreements.json matched nothing: '
      + unused.map(([e]) => `${e.quantity} ${JSON.stringify(e.where ?? {})}`)
          .join('; ')
      + '. A difference that no longer happens must have its entry DELETED, '
      + 'not left standing.');

  assert.equal(failures.length, 0,
    `${failures.length} quantity/state pairs disagree beyond tolerance and are `
    + 'not in tests/golden/known_disagreements.json. Either the JS mirror has '
    + 'drifted from src/workingset/model.py (fix the JS), or the Python model '
    + 'changed and the mirror has not caught up yet (add an entry with a '
    + 'hypothesis, or port the change).');
});
