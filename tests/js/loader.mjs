/* Module hooks for the golden test. Registered by tests/js/register.mjs, which
   the runner passes with --import.

   resolve: interactive/src/main.js -> tests/js/stub-main.mjs. See
   stub-main.mjs for why main.js cannot load under Node.

   load: stamp format "module" on everything under interactive/src/. Those
   files are `.js` with `import`/`export`, and the repo has no package.json
   declaring {"type":"module"} — deliberately, since the nearest one to
   interactive/ would ship to the Worker. Without this hook Node would fall
   back to its ESM syntax detection, which only became the default in 22.x;
   with it the format is explicit, so the real floor for this test is
   module.register (Node 20.6) rather than a version-dependent heuristic.
   (CI pins 24, which is the only version verified end to end.) */
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, resolve as resolvePath } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const STUB = pathToFileURL(resolvePath(HERE, 'stub-main.mjs')).href;
const EXPLORER_SRC = '/interactive/src/';

export async function resolve(specifier, context, nextResolve){
  const r = await nextResolve(specifier, context);
  return r.url.endsWith('/interactive/src/main.js')
    ? { ...r, url: STUB, shortCircuit: true }
    : r;
}

export async function load(url, context, nextLoad){
  return url.includes(EXPLORER_SRC) && url.endsWith('.js')
    ? nextLoad(url, { ...context, format: 'module' })
    : nextLoad(url, context);
}
