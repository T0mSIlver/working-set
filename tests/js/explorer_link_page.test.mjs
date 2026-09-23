/* `ws link` through the real page: the URL is opened in headless Chromium, so
   main.js decodes it (applyURLState), applies its cross-control rules
   (enforceConstraints) and snaps the derived controls (syncLabels) exactly as
   for a user. explorer_link.test.mjs cannot see those three: it decodes with
   a Node-side copy of the decoder.

   Each case checks the page state against the config and, where the page
   moves a value, that `ws link` said so on stderr.

   Needs `uv` and a Chromium: $CHROME, else chromium / google-chrome on PATH,
   else Playwright's headless shell. Skipped without them.
   Run: node --test tests/js/explorer_link_page.test.mjs
*/
import assert from 'node:assert/strict';
import { spawn, spawnSync } from 'node:child_process';
import { createServer } from 'node:http';
import { existsSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { homedir, tmpdir } from 'node:os';
import { dirname, extname, join, resolve } from 'node:path';
import { after, before, test } from 'node:test';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const SITE = join(ROOT, 'interactive');
const FIXTURES = join(ROOT, 'tests/fixtures');
const sleep = ms => new Promise(r => setTimeout(r, ms));

function findChrome(){
  if (process.env.CHROME) return process.env.CHROME;
  for (const bin of ['chromium', 'chromium-browser', 'google-chrome', 'google-chrome-stable']){
    const r = spawnSync('which', [bin], { encoding: 'utf8' });
    if (r.status === 0) return r.stdout.trim();
  }
  const pw = join(homedir(), '.cache/ms-playwright');
  if (existsSync(pw)) for (const d of readdirSync(pw).filter(d => d.startsWith('chromium_headless_shell')).sort().reverse()){
    const bin = join(pw, d, 'chrome-headless-shell-linux64/chrome-headless-shell');
    if (existsSync(bin)) return bin;
  }
  return null;
}
const CHROME = findChrome();
const UV = spawnSync('uv', ['--version']).status === 0;
const skip = !CHROME ? 'no Chromium found (set $CHROME)' : !UV ? 'uv not on PATH' : false;

const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css',
                '.svg': 'image/svg+xml', '.json': 'application/json', '.png': 'image/png' };
let server, base, chrome, profile, sock, tmp;
let nextId = 0; const pending = new Map(); let loaded = null;

function cdp(method, params = {}, sessionId){
  return new Promise((res, rej) => {
    const id = ++nextId;
    pending.set(id, m => m.error ? rej(new Error(`${method}: ${m.error.message}`)) : res(m.result));
    sock.send(JSON.stringify({ id, method, params, sessionId }));
  });
}
let session;

before(async () => {
  if (skip) return;
  tmp = mkdtempSync(join(tmpdir(), 'wslink-'));
  server = createServer((req, res) => {
    const p = join(SITE, decodeURIComponent(new URL(req.url, 'http://x').pathname));
    const file = p.endsWith('/') ? join(p, 'index.html') : p;
    if (!file.startsWith(SITE) || !existsSync(file)){ res.writeHead(404); res.end(); return; }
    res.writeHead(200, { 'content-type': TYPES[extname(file)] || 'application/octet-stream' });
    res.end(readFileSync(file));
  });
  await new Promise(r => server.listen(0, '127.0.0.1', r));
  base = `http://127.0.0.1:${server.address().port}/`;

  profile = mkdtempSync(join(tmpdir(), 'wslink-chrome-'));
  chrome = spawn(CHROME, ['--headless=new', '--no-sandbox', '--disable-gpu',
    '--remote-debugging-port=0', `--user-data-dir=${profile}`, 'about:blank'], { stdio: 'ignore' });
  const portFile = join(profile, 'DevToolsActivePort');
  for (let i = 0; i < 100 && !existsSync(portFile); i++) await sleep(100);
  const [port, path] = readFileSync(portFile, 'utf8').trim().split('\n');
  sock = new WebSocket(`ws://127.0.0.1:${port}${path}`);
  await new Promise((res, rej) => { sock.onopen = res; sock.onerror = rej; });
  sock.onmessage = e => {
    const m = JSON.parse(e.data);
    if (m.id && pending.has(m.id)){ pending.get(m.id)(m); pending.delete(m.id); }
    else if (m.method === 'Page.loadEventFired' && loaded){ loaded(); loaded = null; }
  };
  const { targetId } = await cdp('Target.createTarget', { url: 'about:blank' });
  ({ sessionId: session } = await cdp('Target.attachToTarget', { targetId, flatten: true }));
  await cdp('Page.enable', {}, session);
  await cdp('Runtime.enable', {}, session);
});

after(async () => {
  if (skip) return;
  sock?.close(); server?.close();
  // Chrome keeps writing its profile until it has exited, so removing the
  // directory straight after kill() races it (ENOTEMPTY on CI)
  if (chrome && chrome.exitCode === null && chrome.signalCode === null){
    const exited = new Promise(r => chrome.once('exit', r));
    chrome.kill();
    await Promise.race([exited, sleep(5000)]);
  }
  for (const d of [tmp, profile]) if (d)
    rmSync(d, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 });
});

function wsLink(path){
  const r = spawnSync('uv', ['run', '--quiet', 'ws', 'link', path, '--base', base],
                      { cwd: ROOT, encoding: 'utf8' });
  assert.equal(r.status, 0, r.stderr);
  return { url: r.stdout.trim(), stderr: r.stderr };
}

// open the link on a fresh document and read what the page ended up holding
async function openPage(url){
  const load = () => new Promise(r => { loaded = r; });
  let l = load(); await cdp('Page.navigate', { url: 'about:blank' }, session); await l;
  l = load(); await cdp('Page.navigate', { url }, session); await l;
  const expression = `(async () => {
    const S = await import('./src/state.js');
    const H = await import('./src/harness.js');
    const R = await import('./src/render.js');
    return JSON.stringify({ state: S.state, cap: S.currentWL().cap,
      toml: H.workingsetConfig(S.state, R.activeModel(), S.currentTopo(), S.currentWL()) });
  })()`;
  const r = await cdp('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true }, session);
  if (r.exceptionDetails) throw new Error(JSON.stringify(r.exceptionDetails));
  return JSON.parse(r.result.value);
}

function config(name, text){
  const p = join(tmp, name + '.toml');
  writeFileSync(p, 'schema_version = 1\n' + text);
  return p;
}

const body = text => text.replace(/^(#[^\n]*\n)+/, '');
const fixtures = readdirSync(FIXTURES).filter(f => f.startsWith('explorer_') && f.endsWith('.toml')).sort();

for (const name of fixtures) test(`page opens ws link ${name} on the same config`, { skip }, async () => {
  const path = join(FIXTURES, name);
  const { url, stderr } = wsLink(path);
  assert.equal(stderr, '');
  const page = await openPage(url);
  assert.equal(body(page.toml), body(readFileSync(path, 'utf8')));
});

test('a headcount on a tie snaps half up, and ws link names the load', { skip }, async () => {
  const { url, stderr } = wsLink(config('tie', '[workload]\nheadcount = 10\n'));
  const page = await openPage(url);
  assert.equal(page.state.headcount, 10);
  assert.equal(page.state.users, 12);
  assert.match(stderr, /workload\.headcount -> 10 sessions: the explorer prices 12/);
});

test('replicate below the KV heads is reset by the page, and ws link says so', { skip }, async () => {
  const { url, stderr } = wsLink(config('repl', '[deployment]\nkv_sharding = "replicate"\n'));
  const page = await openPage(url);
  assert.equal(page.state.kvshard, 'dcp');
  assert.match(stderr, /deployment\.kv_sharding = 'replicate' at TP1/);
});

test("the cap's top stop prices the model maximum, and ws link says so", { skip }, async () => {
  const { url, stderr } = wsLink(config('cap',
    '[deployment]\nmodel = "MM35"\ngpu = "B300"\nmax_model_len = 262000\n'));
  const page = await openPage(url);
  assert.equal(page.cap, 262144);
  assert.match(stderr, /max_model_len = 262000: .*262,144 tokens/);
});

test('precision past six decimals reaches the page unwarned', { skip }, async () => {
  const { url, stderr } = wsLink(config('mfu', '[calibration]\nmfu = 0.45000049\n'));
  const page = await openPage(url);
  assert.equal(page.state.mfu, 0.45000049);
  assert.equal(stderr, '');
});
