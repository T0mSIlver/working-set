/* `ws link` round trip: config file -> `ws link` URL -> explorer state ->
   the config the page would download.

   For each tests/fixtures/explorer_*.toml, the URL `ws link` prints is decoded
   into the page state (decode-url.mjs, which refuses a key the page does not
   hold) and harness.js workingsetConfig() writes that state back out. It must
   equal the fixture below its header: every field `ws link` maps reaches the
   page with the value the file holds.

   Needs `uv` on PATH to run the CLI; skipped without it.
   Run: node --import ./tests/js/register.mjs --test tests/js/explorer_link.test.mjs
*/
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { readFileSync, readdirSync } from 'node:fs';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { dirname, join, resolve } from 'node:path';

import { currentTopo, currentWL, state } from '../../interactive/src/state.js';
import { setLiveThink, setLiveTurn } from '../../interactive/src/prefill.js';
import { activeModel } from '../../interactive/src/render.js';
import { workingsetConfig } from '../../interactive/src/harness.js';
import { decodeStateURL } from './decode-url.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, '../..');
const FIXTURES = resolve(ROOT, 'tests/fixtures');

let uv = true;
try { execFileSync('uv', ['--version'], { stdio: 'ignore' }); } catch { uv = false; }

const body = text => text.replace(/^(#[^\n]*\n)+/, '');
const names = readdirSync(FIXTURES)
  .filter(f => f.startsWith('explorer_') && f.endsWith('.toml')).sort();

for (const name of names) test(`ws link ${name} opens the page on that config`,
  { skip: !uv && 'uv not on PATH' }, () => {
  const path = join(FIXTURES, name);
  const url = execFileSync('uv', ['run', '--quiet', 'ws', 'link', path,
                                  '--base', 'http://127.0.0.1:1/'],
                           { cwd: ROOT, encoding: 'utf8' }).trim();
  Object.assign(state, decodeStateURL(url));
  setLiveTurn(state.turn); setLiveThink(state.think);
  const emitted = workingsetConfig(state, activeModel(), currentTopo(), currentWL());
  assert.equal(body(emitted), body(readFileSync(path, 'utf8')));
});
