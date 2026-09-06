/* Passed to node with --import, before any explorer module loads.
   Two jobs, both about getting the explorer's MATH to run under Node without
   editing a byte of interactive/src/:

   1. loader.mjs's resolve hook swaps interactive/src/main.js — pure UI
      wiring, which calls computeAndRender() at module scope — for a stub, and
      its load hook stamps format "module" on the explorer's .js sources so the
      ESM parse does not depend on Node's syntax detection;
   2. the DOM shim below covers the one remaining module-scope DOM call in a
      module the math pulls in: harness.js line 213 attaches a click listener
      to the Test card. Nothing here is ever read; the shim exists so that
      registration is a no-op instead of a ReferenceError.

   Everything that computes a number — config.js, mathlib.js, workload.js,
   capacity.js, prefill.js, planner.js, cost.js, charts.js (interpAt),
   render.js (modelFor) — loads unmodified. */
import { register } from 'node:module';
import { pathToFileURL } from 'node:url';

register('./loader.mjs', pathToFileURL(import.meta.filename));

const noopEl = new Proxy({}, {
  get: (_, k) => (k === 'style' || k === 'dataset' || k === 'classList'
    ? noopEl
    : (k === Symbol.toPrimitive || k === 'then' ? undefined : () => noopEl)),
  set: () => true,
});
globalThis.document = {
  getElementById: () => noopEl,
  querySelector: () => noopEl,
  querySelectorAll: () => [],
  createElement: () => noopEl,
  createElementNS: () => noopEl,
  documentElement: noopEl,
};
