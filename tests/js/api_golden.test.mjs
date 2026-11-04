/* The API and plan seats: the mirror (interactive/src/apicost.js) against
   the Python model (workingset.api_cost), on tests/golden/api_vectors.json.
   Regenerate with `uv run scripts/golden_api.py`. Both sides are closed form
   on the same erfc polynomial, so every figure agrees to float rounding.

   Run: node --import ./tests/js/register.mjs --test tests/js/api_golden.test.mjs
*/
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import { API_OFFERS, PLANS, apiSeatCost, planSeatCost } from '../../interactive/src/apicost.js';

const G = JSON.parse(readFileSync(new URL('../golden/api_vectors.json', import.meta.url), 'utf8'));
const REL = 1e-9;
const FIELDS = { eur_seat: 'eurSeat', eur_session: 'eurSession', usd_request: 'usdRequest',
                 requests_month: 'requestsMonth', ttl_s: 'ttl', hit_rate: 'hitRate',
                 fits: 'fits', plans_per_seat: 'plansPerSeat', overflow_eur: 'overflowEur', binds: 'binds' };

function close(a, b){
  if (typeof a !== 'number') return a === b;
  return Math.abs(a - b) <= REL * Math.max(1, Math.abs(a), Math.abs(b));
}

test('every catalogue row has a vector and every vector a row', () => {
  const keys = new Set([...Object.keys(API_OFFERS).map(k => k + ':api'), ...Object.keys(PLANS).map(k => k + ':plan')]);
  for (const v of G.vectors) assert.deepEqual(new Set(Object.keys(v.out)), keys, v.name);
});

for (const v of G.vectors){
  test(`seat prices agree: ${v.name}`, () => {
    for (const [key, want] of Object.entries(v.out)){
      const [k, kind] = key.split(':');
      const got = kind === 'api' ? apiSeatCost(API_OFFERS[k], v.workload, v.opts)
                                 : planSeatCost(PLANS[k], v.workload, v.opts);
      for (const [py, js] of Object.entries(FIELDS))
        assert.ok(close(want[py], got[js]), `${v.name} ${key} ${py}: python ${want[py]} js ${got[js]}`);
    }
  });
}
