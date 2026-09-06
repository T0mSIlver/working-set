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


@dataclass
class EndpointSpec:
    """Where to send, and what the endpoint turned out to accept."""
    base_url: str = "http://localhost:8000/v1"
    model: str = ""
    api_key: str = ""                    # resolved value, never the env NAME
    api: str = "completions"             # "completions" | "chat"
    use_stream_options: bool = True      # flipped off by the fallback

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
        d = asdict(self)
        d["api_key"] = "<set>" if self.api_key else ""
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
                       metrics=None) -> str:
    """Stream one completion; fills `trace` in place, returns the response
    text (appended to the session so the next warm turn extends the cached
    run)."""
    url, payload = _payload(ep, opts, prompt, max_tokens)
    headers = {"Authorization": f"Bearer {ep.api_key}"} if ep.api_key else {}

    t0 = time.monotonic()
    trace.t_send = t0
    trace.covariates = _covariates(metrics, t0)
    t_first = t_last = None
    n_chunks, text, usage = 0, [], None
    gaps: list[float] = []
    try:
        async with client.stream("POST", url, json=payload, headers=headers,
                                 timeout=opts.request_timeout_s) as r:
            trace.status = r.status_code
            if r.status_code != 200:
                body = (await r.aread()).decode("utf-8", "replace")
                if ep.use_stream_options and "stream_options" in body:
                    # measure_mfu.py's fallback: the endpoint rejects the
                    # usage extension. Retry once without it, for this run.
                    ep.use_stream_options = False
                    return await send_request(client, ep, opts, prompt, trace,
                                              max_tokens, metrics)
                trace.error = f"HTTP {r.status_code}: {body[:180]}"
                return ""
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
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
    return "".join(text)
