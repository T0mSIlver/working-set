/* Stand-in for interactive/src/main.js while the golden test drives the mirror
   under Node.
   main.js is the explorer's UI wiring: it binds sliders, reads the URL
   fragment, installs hover handlers and calls computeAndRender() at module
   scope. None of that is model arithmetic, but all of it needs a live DOM, so
   importing it under Node throws before a single formula runs.
   harness.js is the only module that imports from it, and the only symbol it
   wants is encodeStateURL (a share link stamped into the generated CONFIG
   block). The test never reads that link, so an empty string is enough.
   tests/js/loader.mjs redirects interactive/src/main.js here; every module
   that actually computes something is loaded unmodified. */
export function encodeStateURL(){ return ""; }
