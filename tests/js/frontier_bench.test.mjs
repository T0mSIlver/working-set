/* The frontier's quality axis, checked without a browser.

   Two invariants are easy to break by hand-editing CONFIG.QUALITY and
   invisible until someone reads chart H:

   1. A MEASURED zero is a score. Two models score 0/198 on Terminal-Bench
      4.0, and every truthiness test on a score (`q ? … : '—'`,
      `if (!q) continue`) silently turns that into "no run" — the dot
      vanishes from the chart and the table prints an em dash. frontierScore
      must return 0, not NaN, and only a null/absent entry may give NaN.

   2. Every figure is a run count over its version's denominator (267 on
      2.1 = 89 tasks x 3 repeats, 198 on 4.0 = 66 x 3). A percentage typed
      in by hand — 0.842 for GLM-5.3-Flash's 225/267 — is off by a tenth of
      a point and cannot be traced back to a run count. The only entries
      exempt are the vendor-card ones, which carry a `source` string.

   Ledger and protocols: research/terminal_bench.md.

   Run: node --import ./tests/js/register.mjs --test tests/js/frontier_bench.test.mjs
*/
import assert from 'node:assert/strict';
import { test } from 'node:test';

import { CONFIG } from '../../interactive/src/config.js';
import { state } from '../../interactive/src/state.js';
import { frontierScore } from '../../interactive/src/frontier.js';

const BENCH_KEYS = Object.keys(CONFIG.BENCHES);

test('every model carries an entry on every benchmark', () => {
  for (const mk of Object.keys(CONFIG.MODELS)){
    const q = CONFIG.QUALITY[mk];
    assert.ok(q, `CONFIG.QUALITY is missing ${mk}`);
    for (const b of BENCH_KEYS)
      assert.ok(b in q, `CONFIG.QUALITY.${mk} is missing ${b}`);
  }
  // and nothing scored that is not a model (a renamed key would strand a score)
  for (const mk of Object.keys(CONFIG.QUALITY))
    assert.ok(CONFIG.MODELS[mk], `CONFIG.QUALITY.${mk} names no model`);
});

test('AA scores are exact run counts over their version denominator', () => {
  for (const [mk, q] of Object.entries(CONFIG.QUALITY)){
    for (const b of BENCH_KEYS){
      const v = q[b];
      if (v === null || v === undefined) continue;
      assert.ok(v >= 0 && v <= 1, `${mk}.${b} = ${v} is not a pass rate`);
      if (q.source) continue;                       // vendor card: a percentage
      const runs = CONFIG.BENCHES[b].runs, k = v * runs;
      assert.ok(Math.abs(k - Math.round(k)) < 1e-9,
        `${mk}.${b} = ${v} is not k/${runs} (k = ${k}) — see research/terminal_bench.md`);
    }
  }
});

/* research/terminal_bench.md § 2, transcribed: runs passed per version, or a
   percentage for the one row scored from a vendor card. Pinned rather than
   derived from CONFIG so the two copies must be changed together — these are
   hand-read off Artificial Analysis's model pages, and the failure mode is a
   digit, not a formula. Going red after a ledger update is the intended
   signal: re-read the note, then update this table. */
const LEDGER = {
  "27B":    { tb21: 213, tb40:  11 },
  "35BA3B": { tb21: 120, tb40:   0 },
  "MM35":   { tb21: 135, tb40:   0 },
  "GLM52":  { tb21: 224, tb40:  83 },
  "DSV41F": { tb21: 0.906, tb40: 0.312, vendor: true },
  "Q38FN":  { tb21: 230, tb40:  50 },
  "GLM53F": { tb21: 225, tb40:  65 },
};

test('CONFIG.QUALITY is the ledger in research/terminal_bench.md', () => {
  assert.deepEqual(Object.keys(CONFIG.QUALITY).sort(), Object.keys(LEDGER).sort());
  for (const [mk, row] of Object.entries(LEDGER)){
    for (const b of BENCH_KEYS){
      const want = row.vendor ? row[b] : row[b] / CONFIG.BENCHES[b].runs;
      assert.equal(CONFIG.QUALITY[mk][b], want,
        `${mk}.${b}: config says ${CONFIG.QUALITY[mk][b]}, the ledger says `
        + (row.vendor ? row[b] : `${row[b]}/${CONFIG.BENCHES[b].runs}`));
    }
    assert.equal(!!CONFIG.QUALITY[mk].source, !!row.vendor,
      `${mk}: vendor provenance disagrees with the ledger`);
  }
});

test('a measured zero is a score, an absent one is not', () => {
  // pinned from the ledger, not discovered from CONFIG: a zero mistyped as
  // null would otherwise just shrink the set this test walks
  const zeroes = Object.entries(LEDGER).flatMap(([mk, row]) =>
    BENCH_KEYS.filter(b => !row.vendor && row[b] === 0).map(b => [mk, b]));
  assert.equal(zeroes.length, 2, 'the ledger pins two measured zeroes on 4.0');
  const was = state.bench;
  try {
    for (const [mk, b] of zeroes){
      state.bench = b;
      assert.equal(frontierScore({ mk }), 0, `${mk}.${b}: measured zero lost`);
      assert.ok(isFinite(frontierScore({ mk })), `${mk}.${b}: zero reads as unscored`);
    }
    // the other half of the contract: no entry at all is NaN, not 0
    state.bench = BENCH_KEYS[0];
    assert.ok(Number.isNaN(frontierScore({ mk: '__absent__' })));
    assert.ok(Number.isNaN(frontierScore({ mk: Object.keys(CONFIG.QUALITY)[0] }, '__nobench__')));
  } finally { state.bench = was; }
});

test('frontierScore reads the benchmark it is handed, not the selected one', () => {
  const was = state.bench;
  try {
    // a model whose two versions differ, so a wrong lookup cannot pass
    const [mk, q] = Object.entries(CONFIG.QUALITY)
      .find(([, v]) => v[BENCH_KEYS[0]] !== v[BENCH_KEYS[1]]);
    for (const sel of BENCH_KEYS){
      state.bench = sel;
      assert.equal(frontierScore({ mk }), q[sel], `default arg ignored state.bench=${sel}`);
      for (const b of BENCH_KEYS)
        assert.equal(frontierScore({ mk }, b), q[b], `explicit ${b} read as ${sel}`);
    }
  } finally { state.bench = was; }
});

test('the default axis is a benchmark that exists', () => {
  assert.ok(CONFIG.BENCHES[state.bench], `state.bench = ${state.bench} names no benchmark`);
});
