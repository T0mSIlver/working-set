// Auth gate for PR preview deploys (see preview-explorer.yml). Runs in front
// of the static assets (run_worker_first) so every path requires auth.
// Production (workingset.tomvaucourt.com) never uses this script.
//
// Ways in, ordered by friction:
//   1. cookie      — <expiry>.<hmac(password, expiry)>, stateless; rotating
//                    the PREVIEW_PASSWORD secret revokes every session
//   2. magic link  — ?key=<derived token> (or the raw password) sets the
//                    cookie and redirects with the key stripped; the derived
//                    token keeps the password itself out of URLs and logs
//   3. login form  — what a browser sees instead of the basic-auth POPUP: a
//                    real password field, so password managers save/autofill
//   4. basic auth  — user "preview" (curl and other non-HTML clients only)
const COOKIE = "ws_preview";
const TTL_S = 60 * 60 * 24 * 30; // 30 days
const LOGIN_PATH = "/__preview-login";

async function sign(secret, msg) {
  const key = await crypto.subtle.importKey(
    "raw", new TextEncoder().encode(secret),
    { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
  const mac = await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(msg));
  return [...new Uint8Array(mac)].map(b => b.toString(16).padStart(2, "0")).join("");
}

// constant-time string compare — the cookie check recomputes an HMAC and
// compares attacker-supplied input against it, which is exactly where a
// short-circuiting === becomes a byte-at-a-time oracle
function ctEqual(a, b) {
  const ab = new TextEncoder().encode(String(a));
  const bb = new TextEncoder().encode(String(b));
  let d = ab.length ^ bb.length;
  const n = Math.max(ab.length, bb.length);
  for (let i = 0; i < n; i++) d |= (ab[i] ?? 0) ^ (bb[i] ?? 0);
  return d === 0;
}

// URL-safe magic-link token: password-derived, so links never carry the
// password itself; rotating the password rotates every link
const magicToken = async pw => (await sign(pw, "magic-link-v1")).slice(0, 32);

async function cookieValid(header, pw) {
  const m = (header || "").match(new RegExp(`(?:^|;\\s*)${COOKIE}=([^;]+)`));
  if (!m) return false;
  const [exp, sig] = m[1].split(".");
  if (!exp || !sig || Number(exp) < Date.now()) return false;
  return ctEqual(sig, await sign(pw, exp));
}

async function authCookie(pw) {
  const exp = String(Date.now() + TTL_S * 1000);
  return `${COOKIE}=${exp}.${await sign(pw, exp)}; Max-Age=${TTL_S}; `
       + `Path=/; Secure; HttpOnly; SameSite=Lax`;
}

const escAttr = s => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;")
                              .replace(/"/g, "&quot;");
// only same-origin destinations survive as a post-login target — URL-parse
// instead of prefix checks, which let /\evil.com through (browsers treat \
// as / in the authority position)
const safeNext = (n, base) => {
  try {
    const u = new URL(String(n ?? "/"), base);
    return u.origin === new URL(base).origin ? u.pathname + u.search : "/";
  } catch { return "/"; }
};

function loginPage(next, wrong) {
  return new Response(`<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>workingset preview — sign in</title>
<style>
:root{--page:#f9f9f7;--surface:#fcfcfb;--text:#0b0b0b;--muted:#6f6d66;
  --border:#e1e0d9;--s1:#2a78d6;--crit:#c22c2c}
@media (prefers-color-scheme:dark){:root{--page:#0d0d0d;--surface:#1a1a19;
  --text:#fff;--muted:#898781;--border:#2c2c2a;--s1:#3987e5;--crit:#d03b3b}}
body{margin:0;min-height:100vh;display:grid;place-items:center;
  background:var(--page);color:var(--text);
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif}
form{background:var(--surface);border:1px solid var(--border);border-radius:12px;
  padding:26px 28px;width:min(320px,88vw);display:grid;gap:10px}
h1{font-size:15px;font-weight:650;margin:0}
p{font-size:12.5px;color:var(--muted);margin:0}
.err{color:var(--crit);font-weight:600}
input[type=password]{font:inherit;padding:8px 10px;border:1px solid var(--border);
  border-radius:8px;background:var(--page);color:var(--text)}
button{font:inherit;font-weight:600;padding:8px;border:0;border-radius:8px;
  background:var(--s1);color:#fff;cursor:pointer}
</style></head><body>
<form method="POST" action="${LOGIN_PATH}">
  <h1>workingset preview</h1>
  <p>${wrong ? '<span class="err">Wrong password — try again.</span>'
             : 'This is a private PR preview.'}</p>
  <input type="text" name="username" value="preview" autocomplete="username" hidden>
  <input type="password" name="key" autocomplete="current-password"
         placeholder="preview password" autofocus required>
  <input type="hidden" name="next" value="${escAttr(next)}">
  <button type="submit">Enter</button>
</form></body></html>`, {
    status: 401,
    headers: { "Content-Type": "text/html;charset=utf-8", "Cache-Control": "no-store" },
  });
}

export default {
  async fetch(request, env) {
    if (!env.PREVIEW_PASSWORD)
      return new Response("preview password not configured", { status: 503 });
    const pw = env.PREVIEW_PASSWORD;
    const url = new URL(request.url);
    const cookieOk = await cookieValid(request.headers.get("Cookie"), pw);

    // login-form submission; an already-authed session just gets redirected
    if (request.method === "POST" && url.pathname === LOGIN_PATH) {
      const form = await request.formData();
      const next = safeNext(form.get("next"), url);
      if (cookieOk || ctEqual(form.get("key") ?? "", pw))
        return new Response(null, {
          status: 303,
          headers: { Location: next, "Set-Cookie": await authCookie(pw) },
        });
      return loginPage(next, true);
    }

    // magic link: strip the key from the URL so it stays out of
    // history/referrers, and hand back a session cookie
    const key = url.searchParams.get("key");
    if (key !== null && (ctEqual(key, await magicToken(pw)) || ctEqual(key, pw))) {
      url.searchParams.delete("key");
      return new Response(null, {
        status: 302,
        headers: { Location: url.toString(), "Set-Cookie": await authCookie(pw) },
      });
    }

    // session cookie
    if (cookieOk) return env.ASSETS.fetch(request);

    // basic auth (curl etc.) — and upgrade it to a cookie
    const expected = "Basic " + btoa("preview:" + pw);
    if (ctEqual(request.headers.get("Authorization") ?? "", expected)) {
      const asset = await env.ASSETS.fetch(request);
      const resp = new Response(asset.body, asset);   // unfreeze the headers
      resp.headers.append("Set-Cookie", await authCookie(pw));
      return resp;
    }

    // unauthenticated: browsers get the login form (no WWW-Authenticate, so no
    // popup — and password managers can save the form); everything else gets
    // the challenge header for curl-style clients
    if ((request.headers.get("Accept") || "").includes("text/html"))
      return loginPage(url.pathname + url.search, false);
    return new Response("authentication required", {
      status: 401,
      headers: { "WWW-Authenticate": 'Basic realm="workingset preview"' },
    });
  },
};
