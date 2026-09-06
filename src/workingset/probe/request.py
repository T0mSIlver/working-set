"""One streamed completion, timed per SSE event.

Ported from `send_turn` in scripts/validate_deployment.py, with three
additions and one deviation, all deliberate:

  ADDED  `usage.prompt_tokens_details.cached_tokens` is read back when the
         server reports it (vLLM does). That is the server's own answer to
         "did the prefix cache hit", next to the harness's TTFT heuristic.
  ADDED  `covariates`: whatever a metrics sampler saw at SEND time
         (requests_running / requests_waiting / kv_cache_usage). The sampler
         is duck-typed on `at(t) -> snapshot`; nothing here imports it.
  ADDED  the `stream_options` fallback from scripts/measure_mfu.py: an
         endpoint that rejects `stream_options` gets one retry without it
         (and the whole run then stops sending it). Without usage readback
         the achieved/intended prompt-token ratio is unavailable, which the
         report says out loud.
  DEVIATION  the harness only ever posts to /completions. This posts there by
         default — same prompt bytes, same `ignore_eos`, same accounting — and
         supports /chat/completions under `api="chat"` for endpoints that
         serve nothing else. Chat mode wraps the prompt in one user message,
         so the served chat template adds a handful of tokens either side of
         the byte-stable prefix; the prefix is still byte-stable, but the
         achieved/intended ratio shifts by the template's length.

Gaps are measured per SSE EVENT, not per token; `n_chunks` vs `ctok` says how
far apart those two are on a given endpoint (the report's tok/evt column).
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass, field

from .stats import FREEZE_LADDER_MS, pct


@dataclass(frozen=True)
class EndpointSpec:
    """Where to send, and what the endpoint turned out to accept.

    Frozen: a probe cache is keyed on it, and a spec that could be mutated
    after a probe ran would hand back a result measured against a different
    endpoint. The one thing that DOES change at run time — whether this
    endpoint accepts `stream_options` — lives in `_flags`, excluded from
    equality and the hash, so learning it does not change the spec's
    identity.
    """
    base_url: str = "http://localhost:8000/v1"
    model: str = ""
    api_key: str = ""                    # resolved value, never the env NAME
    api: str = "completions"             # "completions" | "chat"
    _flags: dict = field(default_factory=lambda: {"stream_options": True},
                         compare=False, repr=False)

    @property
    def use_stream_options(self) -> bool:
        return self._flags["stream_options"]

    def disable_stream_options(self) -> None:
        """Remember, for every FUTURE request, that this endpoint rejects the
        usage extension. Requests already in flight carry their own payload
        and decide their own retry from it — see `send_request`."""
        self._flags["stream_options"] = False

    def identity(self) -> tuple:
        """What makes this a different endpoint. The API key is included by
        presence only; it never reaches a cache key or a record."""
        return (self.base_url, self.model, self.api, bool(self.api_key))

    @classmethod
    def from_config(cls, ep, api: str = "completions",
                    base_url: str | None = None, model: str | None = None,
                    api_key_env: str | None = None) -> "EndpointSpec":
        env = api_key_env or ep.api_key_env or ""
        return cls(base_url=(base_url or ep.base_url).rstrip("/"),
                   model=model or ep.model,
                   api_key=os.environ.get(env, "") if env else "",
                   api=api)

    def redacted(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k != "_flags"}
        d["api_key"] = "<set>" if self.api_key else ""
        d["use_stream_options"] = self.use_stream_options
        return d


@dataclass
class RequestTrace:
    """Everything one streamed request establishes."""
    uid: int = 0
    is_sub: bool = False
    kind: str = "hit"                 # "first" | "hit" | "miss"
    t_send: float = 0.0               # monotonic clock at POST
    ttft: float | None = None
    decode_tps: float | None = None
    # decode rate with the FREEZES TAKEN OUT: (tokens - 1) / (span - stall).
    # steady_decode_point predicts the clean decode speed BETWEEN prefill
    # spikes (its own docstring says so), and decode_tps is a mean over the
    # whole stream, spikes included — comparing the two prices a freeze as a
    # decode slowdown. Threshold-dependent by construction: `stall_s` is the
    # time inside gaps at or above ProbeOptions.freeze_threshold_ms.
    clean_decode_tps: float | None = None
    ptok_intended: int = 0
    ptok_achieved: int | None = None
    ctok: int | None = None
    cached_tokens: int | None = None  # usage.prompt_tokens_details.cached_tokens
    status: int | None = None
    error: str | None = None
    # --- inter-token gap stats (the ITL-spike evidence) ---------------------
    # decode_tps above is a MEAN over the whole stream and is nearly blind to a
    # freeze: a single 1.3 s stall inside a 1,000-token response otherwise
    # running at 200 tok/s reads as ~159 tok/s — a 20% dip standing in for a
    # 260x spike. These fields keep the distribution the mean throws away.
    itl_p50: float | None = None      # s, median gap within this response
    itl_max: float | None = None      # s, worst gap within this response
    itl_min: float | None = None      # s, smallest gap — the CLIENT FLOOR probe
    n_gaps: int = 0
    n_freeze: int = 0                 # gaps >= the headline freeze threshold
    stall_s: float = 0.0              # total time inside those gaps
    n_freeze_at: tuple = ()           # counts, one per FREEZE_LADDER_MS entry
    stall_at: tuple = ()              # summed seconds, same order
    span_s: float | None = None       # t_last - t_first
    t_end: float | None = None        # monotonic end-of-stream (last chunk)
    n_chunks: int = 0
    gaps_ms: list = field(default_factory=list, repr=False)
    covariates: dict | None = None    # metrics snapshot at SEND time

    def to_dict(self, gaps: bool = False) -> dict:
        d = asdict(self)
        d["n_freeze_at"] = list(self.n_freeze_at)
        d["stall_at"] = list(self.stall_at)
        if not gaps:
            d.pop("gaps_ms")
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RequestTrace":
        d = dict(d)
        d.setdefault("gaps_ms", [])
        d["n_freeze_at"] = tuple(d.get("n_freeze_at") or ())
        d["stall_at"] = tuple(d.get("stall_at") or ())
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    def summarise_gaps(self, freeze_threshold_ms: float) -> None:
        """Fill the gap fields from `gaps_ms`. Split out of the stream loop so
        a synthetic trace can be scored by exactly the same code."""
        gaps = [g / 1e3 for g in self.gaps_ms]
        if not gaps:
            return
        freeze_s = freeze_threshold_ms / 1e3
        big = [g for g in gaps if g >= freeze_s]
        self.itl_p50 = pct(gaps, 50)
        self.itl_max = max(gaps)
        self.itl_min = min(gaps)
        self.n_gaps = len(gaps)
        self.n_freeze = len(big)
        self.stall_s = sum(big)
        self.n_freeze_at = tuple(
            sum(1 for g in gaps if g >= t / 1e3) for t in FREEZE_LADDER_MS)
        self.stall_at = tuple(
            sum(g for g in gaps if g >= t / 1e3) for t in FREEZE_LADDER_MS)

    def finish_decode_rates(self, n_tok: int | None = None) -> None:
        """Fill `clean_decode_tps` from the span and the stall. Split out of
        the stream loop so a synthetic trace is scored by the same code."""
        n_tok = n_tok if n_tok else (self.ctok or self.n_chunks)
        clean = (self.span_s or 0.0) - self.stall_s
        if n_tok and n_tok > 1 and clean > 0:
            self.clean_decode_tps = (n_tok - 1) / clean


def _covariates(metrics, t: float) -> dict | None:
    """Duck-typed read of a metrics sampler: `at(t) -> snapshot-like`. The
    sampler is built by another module; nothing here touches its internals."""
    if metrics is None:
        return None
    try:
        snap = metrics.at(t)
    except Exception:
        return None
    return _plain(snap, ("requests_running", "requests_waiting",
                         "kv_cache_usage"))


def _plain(obj, keys=None) -> dict | None:
    """A snapshot/delta as a JSON-safe dict, whatever shape it arrived in."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        d = dict(obj)
    elif hasattr(obj, "to_dict"):
        d = dict(obj.to_dict())
    elif hasattr(obj, "__dataclass_fields__"):
        d = asdict(obj)
    elif hasattr(obj, "__dict__"):
        d = {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    else:
        return None
    if keys is not None:
        d = {k: d.get(k) for k in keys if k in d}
    return {k: v for k, v in d.items()
            if isinstance(v, (int, float, str, bool, type(None)))}


def make_client(max_connections: int = 4096, max_keepalive: int = 256):
    """An AsyncClient sized for the ladder. httpx is imported here so that
    `--dry-run` stays importable on a machine without it."""
    import httpx
    return httpx.AsyncClient(limits=httpx.Limits(
        max_connections=max_connections,
        max_keepalive_connections=max_keepalive))


def _payload(ep: EndpointSpec, opts, prompt: str, max_tokens: int) -> tuple[str, dict]:
    body = {"model": ep.model, "max_tokens": max_tokens, "temperature": 0.7,
            "stream": True}
    if ep.use_stream_options:
        body["stream_options"] = {"include_usage": True}
    if opts.ignore_eos:
        # vLLM extension: guarantees max_output_tokens of decode per turn, so
        # decode tok/s is measured over a fixed-length stream, not over
        # whatever an early EOS leaves. Strict non-vLLM endpoints: disable.
        body["ignore_eos"] = True
    if ep.api == "chat":
        return f"{ep.base_url}/chat/completions", {
            **body, "messages": [{"role": "user", "content": prompt}]}
    return f"{ep.base_url}/completions", {**body, "prompt": prompt}


async def _sse_events(response):
    """Yield one SSE event's data payload per iteration.

    The spec allows an event to carry SEVERAL `data:` lines, joined with a
    newline, and terminates the event with a blank line. Treating every
    `data:` line as its own JSON document raised JSONDecodeError on the first
    such stream; vLLM does not split its frames today, but a proxy in front of
    it may, and a crashed run loses the whole rung. Other field lines (`event:`,
    `id:`, `:` comments) are skipped, as are the keep-alive blank lines that
    separate nothing.
    """
    buf: list[str] = []
    async for line in response.aiter_lines():
        line = line.rstrip("\r")
        if line == "":
            if buf:
                yield "\n".join(buf)
                buf = []
            continue
        if line.startswith(":"):          # comment / keep-alive
            continue
        if line.startswith("data:"):
            buf.append(line[5:].lstrip(" "))
    if buf:
        yield "\n".join(buf)


def _delta_text(obj: dict) -> str:
    ch = obj.get("choices") or []
    if not ch:
        return ""
    c = ch[0]
    if c.get("text"):
        return c["text"]
    d = c.get("delta") or {}
    return d.get("content") or ""


async def send_request(client, ep: EndpointSpec, opts, prompt: str,
                       trace: RequestTrace, max_tokens: int,
                       metrics=None, _retries_left: int = 1) -> str:
    """Stream one completion; fills `trace` in place, returns the response
    text (appended to the session so the next warm turn extends the cached
    run)."""
    url, payload = _payload(ep, opts, prompt, max_tokens)
    headers = {"Authorization": f"Bearer {ep.api_key}"} if ep.api_key else {}

    t0 = time.monotonic()
    trace.t_send = t0
    # WALL clock for the sampler, monotonic for the trace. `t_send` does span
    # arithmetic and must not step backwards; a metrics sampler stamps its
    # snapshots with `time.time()`, and asking it for a MONOTONIC instant
    # (uptime seconds, ~3 orders of magnitude smaller than a unix time) always
    # landed before the whole series, so every request got the FIRST snapshot's
    # gauges instead of its own. Shared mode fits on these numbers.
    trace.covariates = _covariates(metrics, time.time())
    t_first = t_last = None
    n_chunks, text, usage = 0, [], None
    gaps: list[float] = []
    try:
        async with client.stream("POST", url, json=payload, headers=headers,
                                 timeout=opts.request_timeout_s) as r:
            trace.status = r.status_code
            if r.status_code != 200:
                body = (await r.aread()).decode("utf-8", "replace")
                # measure_mfu.py's fallback: the endpoint rejects the usage
                # extension. Eligibility is decided from THIS request's own
                # payload, not from the endpoint-wide flag — under the ladder
                # many requests are in flight, and the first rejection clears
                # that flag while the rest are still waiting on their own 400s.
                # Reading the flag then made every one of them a hard error
                # (8 concurrent: 1 retried, 7 failed). The flag is still
                # cleared, for FUTURE requests.
                if ("stream_options" in payload and _retries_left > 0
                        and "stream_options" in body):
                    ep.disable_stream_options()
                    return await send_request(client, ep, opts, prompt, trace,
                                              max_tokens, metrics,
                                              _retries_left - 1)
                trace.error = f"HTTP {r.status_code}: {body[:180]}"
                return ""
            async for data in _sse_events(r):
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                if obj.get("usage"):
                    usage = obj["usage"]
                if _delta_text(obj):
                    now = time.monotonic()
                    if t_first is None:
                        t_first = now
                        trace.ttft = now - t0
                    else:
                        gaps.append(now - t_last)
                    t_last = now
                    n_chunks += 1
                    text.append(_delta_text(obj))
    except asyncio.CancelledError:
        raise
    except Exception as e:  # connection, HTTP status, timeout, bad JSON
        trace.error = f"{type(e).__name__}: {e}"[:200]
        return ""

    if usage:
        trace.ptok_achieved = usage.get("prompt_tokens")
        trace.ctok = usage.get("completion_tokens")
        details = usage.get("prompt_tokens_details") or {}
        if isinstance(details, dict) and details.get("cached_tokens") is not None:
            trace.cached_tokens = details["cached_tokens"]
    n_tok = trace.ctok if trace.ctok else n_chunks
    # span_s is set whenever the stream had >= 2 content chunks — the SAME
    # condition that makes `gaps` non-empty. A server reporting
    # completion_tokens=1 on a multi-chunk stream must not carry stall_s with
    # no span: decode_tps keeps the n_tok gate (a rate needs a token count),
    # span does not.
    if t_first is not None and t_last is not None and t_last > t_first:
        trace.span_s = t_last - t_first
        trace.t_end = t_last
        if n_tok > 1:
            trace.decode_tps = (n_tok - 1) / (t_last - t_first)
    trace.n_chunks = n_chunks
    trace.gaps_ms = [g * 1e3 for g in gaps]
    trace.summarise_gaps(opts.freeze_threshold_ms)
    trace.finish_decode_rates(n_tok)
    return "".join(text)
