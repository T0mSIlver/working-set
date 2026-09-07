/* tests/fixtures/explorer_*.toml are real downloads from the explorer, and
   tests/test_config.py asserts what `workingset.config` reads out of them.
   Nothing kept them in step with the page that writes them: a field added to
   `workingsetConfig` (or dropped from it) left the fixtures green and stale,
   which is the same class of bug as a config that silently omits a knob the
   page prices.

   This closes it from the other side. Each fixture's own header carries the
   share URL of the state it was taken at; that URL is decoded back into
   `state`, `workingsetConfig` is called on it, and the result must equal the
   fixture byte for byte below the header.

   Why the header is excluded: interactive/src/main.js is UI wiring that
   cannot load under Node, so tests/js/loader.mjs swaps it for a stub whose
   encodeStateURL() returns "". The header's URL line is therefore the one
   line this process cannot reproduce — and it is also the line being used as
   the INPUT here, so a URL that no longer produces the body fails anyway.

   Run: node --import ./tests/js/register.mjs --test tests/js/explorer_config.test.mjs
*/
import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { dirname, join, resolve } from 'node:path';

import { CONFIG } from '../../interactive/src/config.js';
import { STATE_DEFAULTS, currentTopo, currentWL, state } from '../../interactive/src/state.js';
import { setLiveThink, setLiveTurn } from '../../interactive/src/prefill.js';
import { activeModel } from '../../interactive/src/render.js';
import { workingsetConfig } from '../../interactive/src/harness.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const FIXTURES = resolve(HERE, '../fixtures');
const URL_LINE = /^# reproduce this page: (\S+)\s*$/m;

// Inverse of main.js encodeStateURL(): the fragment carries only the DIFFS
// from STATE_DEFAULTS, so the type of each key is read off the defaults
// rather than from a second copy of the enum tables. The two keys that are
// re-seeded from the selection rather than defaulted globally get the same
// treatment encodeStateURL gives them when it decides not to write them.
function decodeStateURL(url){
  const st = { ...STATE_DEFAULTS };
  const q = url.includes('#') ? url.slice(url.indexOf('#') + 1) : '';
  for (const [k, v] of new URLSearchParams(q)){
    if (!(k in STATE_DEFAULTS)) throw new Error(`unknown state key in URL: ${k}`);
    st[k] = STATE_DEFAULTS[k] === null ? Number(v)
          : typeof STATE_DEFAULTS[k] === 'boolean' ? v === '1'
          : typeof STATE_DEFAULTS[k] === 'number' ? Number(v)
          : v;
  }
  const p = new URLSearchParams(q);
  if (!p.has('mtp')) st.mtp = CONFIG.MODELS[st.model].mtp;
  if (!p.has('gpuh')) st.gpuh = CONFIG.GPUS[st.gpu].eur_gpu_h;
  return st;
}

const body = text => text.replace(/^(#[^\n]*\n)+/, '');

const names = readdirSync(FIXTURES)
  .filter(f => f.startsWith('explorer_') && f.endsWith('.toml')).sort();
assert.ok(names.length >= 5, `expected the explorer fixtures, found ${names}`);

for (const name of names) test(`${name} is what the page emits for its own share URL`, () => {
  const fixture = readFileSync(join(FIXTURES, name), 'utf8');
  const m = fixture.match(URL_LINE);
  assert.ok(m, `${name}: header carries no "reproduce this page" URL`);

  Object.assign(state, decodeStateURL(m[1]));
  setLiveTurn(state.turn); setLiveThink(state.think);
  const emitted = workingsetConfig(state, activeModel(), currentTopo(), currentWL());

  assert.equal(body(emitted), body(fixture),
    `tests/fixtures/${name} is stale: re-download it from the explorer at the `
    + 'URL in its header, or fix workingsetConfig');
});
