// Auth gate for PR preview deploys (see preview-explorer.yml). Runs in front
// of the static assets (run_worker_first) so every path requires auth.
// Production (workingset.tomvaucourt.com) never uses this script.
//
// Three ways in, ordered by friction:
//   1. magic link  — ?key=<password> sets a signed 30-day cookie and
//                    redirects to the same URL without the key
//   2. cookie      — <expiry>.<hmac(password, expiry)>, stateless; rotating
//                    the PREVIEW_PASSWORD repo secret revokes every session
//   3. basic auth  — user "preview" / the password (curl, fallback)
const COOKIE = "ws_preview";
const TTL_S = 60 * 60 * 24 * 30; // 30 days

async function sign(secret, msg) {
  const key = await crypto.subtle.importKey(
    "raw", new TextEncoder().encode(secret),
    { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
  const mac = await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(msg));
  return [...new Uint8Array(mac)].map(b => b.toString(16).padStart(2, "0")).join("");
}

async function cookieValid(header, pw) {
  const m = (header || "").match(new RegExp(`(?:^|;\\s*)${COOKIE}=([^;]+)`));
  if (!m) return false;
  const [exp, sig] = m[1].split(".");
  if (!exp || !sig || Number(exp) < Date.now()) return false;
  return sig === await sign(pw, exp);
}

async function authCookie(pw) {
  const exp = String(Date.now() + TTL_S * 1000);
  return `${COOKIE}=${exp}.${await sign(pw, exp)}; Max-Age=${TTL_S}; `
       + `Path=/; Secure; HttpOnly; SameSite=Lax`;
}

export default {
  async fetch(request, env) {
    if (!env.PREVIEW_PASSWORD)
      return new Response("preview password not configured", { status: 503 });
    const pw = env.PREVIEW_PASSWORD;
    const url = new URL(request.url);

    // 1. magic link: strip the key from the URL so it stays out of
    // history/referrers, and hand back a session cookie
    if (url.searchParams.get("key") === pw) {
      url.searchParams.delete("key");
      return new Response(null, {
        status: 302,
        headers: { Location: url.toString(), "Set-Cookie": await authCookie(pw) },
      });
    }

    // 2. session cookie
    if (await cookieValid(request.headers.get("Cookie"), pw))
      return env.ASSETS.fetch(request);

    // 3. basic auth — and upgrade it to a cookie so the dialog is one-time
    const expected = "Basic " + btoa("preview:" + pw);
    if (request.headers.get("Authorization") === expected) {
      const asset = await env.ASSETS.fetch(request);
      const resp = new Response(asset.body, asset);   // unfreeze the headers
      resp.headers.append("Set-Cookie", await authCookie(pw));
      return resp;
    }
    return new Response("authentication required", {
      status: 401,
      headers: { "WWW-Authenticate": 'Basic realm="workingset preview"' },
    });
  },
};
