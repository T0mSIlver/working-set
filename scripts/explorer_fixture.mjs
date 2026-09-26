/* Write a tests/fixtures/explorer_*.toml the way the page's download button
   does, for a state given as a share-URL fragment.

   node --import ./tests/js/register.mjs scripts/explorer_fixture.mjs \
        'model=Q38FN&gpu=B300' > tests/fixtures/explorer_NAME.toml

   The body is harness.js workingsetConfig() on the decoded state, the header
   carries the URL it was taken at: tests/js/explorer_config.test.mjs checks
   the pair, tests/test_link.py checks `ws link` gives the URL back. */
import { currentTopo, currentWL, state } from '../interactive/src/state.js';
import { setLiveThink, setLiveTurn } from '../interactive/src/prefill.js';
import { activeModel } from '../interactive/src/render.js';
import { workingsetConfig } from '../interactive/src/harness.js';
import { decodeStateURL } from '../tests/js/decode-url.mjs';

const BASE = 'https://workingset.tomvaucourt.com/';
const frag = (process.argv[2] || '').replace(/^#/, '');
Object.assign(state, decodeStateURL(BASE + '#' + frag));
setLiveTurn(state.turn); setLiveThink(state.think);
const text = workingsetConfig(state, activeModel(), currentTopo(), currentWL());
// the stub encodeStateURL() returns "", so the URL line is stamped here
process.stdout.write(text.replace(/^# reproduce this page: .*$/m,
  `# reproduce this page: ${BASE}${frag ? '#' + frag : ''}`));
