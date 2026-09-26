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

import { currentTopo, currentWL, state, stampTtftPct, ttftPctFromFragment } from '../../interactive/src/state.js';
import { TTFT_PCTS } from '../../interactive/src/prefill.js';
import { setLiveThink, setLiveTurn } from '../../interactive/src/prefill.js';
import { activeModel } from '../../interactive/src/render.js';
import { workingsetConfig } from '../../interactive/src/harness.js';
import { decodeStateURL } from './decode-url.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
const FIXTURES = resolve(HERE, '../fixtures');
const URL_LINE = /^# reproduce this page: (\S+)\s*$/m;

test('share links: legacy fragments decode as mean, bare URLs as p95, new links name it', () => {
  const dec = q => ttftPctFromFragment(new URLSearchParams(q), TTFT_PCTS);
  assert.equal(dec(''), '95');                       // bare URL: today's default
  assert.equal(dec('users=128'), 'mean');            // pre-control link
  assert.equal(dec('users=128&ttft_pct=99'), '99');
  assert.equal(dec('ttft_pct=mean'), 'mean');
  assert.equal(dec('users=128&ttft_pct=bogus'), '95');
  const enc = (q, pct) => stampTtftPct(new URLSearchParams(q), pct).toString();
  assert.equal(enc('', '95'), '');                   // default page stays bare
  assert.equal(enc('users=128', '95'), 'users=128&ttft_pct=95');
  assert.equal(enc('ttft_pct=mean', 'mean'), 'ttft_pct=mean');
  // round trip: a new non-bare link decodes to the statistic it was made at
  for (const pct of TTFT_PCTS) assert.equal(dec(enc('users=128', pct)), pct);
});

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
