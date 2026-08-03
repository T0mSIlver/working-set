// Basic-auth gate for PR preview deploys (see preview-explorer.yml).
// Runs in front of the static assets (run_worker_first) so every path —
// index, method.html, anything added later — requires the shared password.
// Production (workingset.tomvaucourt.com) never uses this script.
export default {
  async fetch(request, env) {
    if (!env.PREVIEW_PASSWORD)
      return new Response("preview password not configured", { status: 503 });
    const expected = "Basic " + btoa("preview:" + env.PREVIEW_PASSWORD);
    if (request.headers.get("Authorization") !== expected)
      return new Response("authentication required", {
        status: 401,
        headers: { "WWW-Authenticate": 'Basic realm="workingset preview"' },
      });
    return env.ASSETS.fetch(request);
  },
};
