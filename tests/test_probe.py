"""Probe layer: the generators, the statistics, and the request loop against a
fake SSE server. No test here needs a real endpoint."""
from __future__ import annotations

import asyncio
import json
import math
import random
from dataclasses import replace

import httpx
import pytest

from workingset.config import RunConfig, WorkloadCfg
from workingset.probe import (ProbeOptions, RequestTrace, build_ladder,
                              build_prefixes, draw_session_tokens, eval_burst,
                              eval_rung, eval_sample, make_session, make_text,
                              pct, run_population, run_sample,
                              sampler_selfcheck, sub_prefix_floor)
from workingset.probe.burst import run_burst
from workingset.probe.request import EndpointSpec, send_request


# ============================================================================
# fixtures: a small config and a fake streaming server
# ============================================================================
def small_cfg(**wl) -> RunConfig:
    """The reference workload shrunk ~150x so a test can generate it."""
    base = dict(system_prefix_tokens=80, user_prompt_median_tokens=200,
                user_prompt_sigma=0.5, warm_turn_tokens=20, think_time_s=0.05,
                subagent_ratio=0.5, subagent_median_tokens=60,
                subagent_sigma=0.6, subagent_prefix_tokens=20,
                miss_rate=0.5, max_output_tokens=6, users=4)
    base.update(wl)
    return RunConfig(workload=WorkloadCfg(**base))


def small_opts(**kw) -> ProbeOptions:
    base = dict(ramp_s=0.02, measure_s=0.25, context_cap_tokens=2_000,
                request_timeout_s=5.0, freeze_threshold_ms=20.0,
                sample_requests=2, sample_warm_turns=1, seed=7)
    base.update(kw)
    return ProbeOptions(**base)


def _sse(obj) -> bytes:
    return f"data: {json.dumps(obj)}\n\n".encode()


def fake_server(n_tokens: int = 6, delay: float = 0.001, ttft: float = 0.003,
                usage: bool = True, cached: int = 0,
                freeze_at: int | None = None, freeze_s: float = 0.05,
                reject_stream_options: bool = False, seen: list | None = None):
    """An OpenAI-compatible streaming endpoint that streams N tokens with a
    configurable per-event delay, one optional long freeze, and a usage
    trailer."""
    state = {"rejected": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if seen is not None:
            seen.append(body)
        if reject_stream_options and "stream_options" in body:
            state["rejected"] += 1
            return httpx.Response(
                400, json={"error": "stream_options is not supported"})
        n = min(n_tokens, body.get("max_tokens") or n_tokens)
        prompt = body.get("prompt")
        if prompt is None:
            prompt = body["messages"][0]["content"]
        ptok = int(len(prompt) / 4)

        async def gen():
            await asyncio.sleep(ttft)
            for i in range(n):
                if i:
                    await asyncio.sleep(
                        freeze_s if freeze_at == i else delay)
                yield _sse({"choices": [{"text": "tok ", "index": 0}]})
            if usage and "stream_options" in body:
                yield _sse({"choices": [],
                            "usage": {"prompt_tokens": ptok,
                                      "completion_tokens": n,
                                      "prompt_tokens_details":
                                          {"cached_tokens": cached}}})
            yield b"data: [DONE]\n\n"

        return httpx.Response(200, content=gen(),
                              headers={"content-type": "text/event-stream"})

    handler.state = state
    return handler


def client_for(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class FakeMetrics:
    """The duck type the probes accept: at(t) and window(t0, t1)."""

    def __init__(self):
        self.at_calls, self.window_calls = 0, 0

    def at(self, t):
        self.at_calls += 1
        return {"requests_running": 3, "requests_waiting": 1,
                "kv_cache_usage": 0.42, "t": t}

    def window(self, t0, t1):
        self.window_calls += 1
        return {"prompt_tokens": 1234, "generation_tokens": 567,
                "seconds": t1 - t0}


# ============================================================================
# session generator
# ============================================================================
def test_make_text_hits_the_token_budget():
    rng = random.Random(0)
    for tokens, cpt in ((100, 4.0), (2_500, 4.0), (700, 3.2)):
        txt = make_text(rng, tokens, cpt)
        assert len(txt) == int(tokens * cpt)
        # the intent the probe records is exactly len/cpt
        assert int(len(txt) / cpt) == pytest.approx(tokens, abs=1)


def test_draw_session_tokens_clips_to_prefix_and_cap():
    rng = random.Random(1)
    draws = [draw_session_tokens(rng, 1_000, 1.5, 300, 4_000)
             for _ in range(2_000)]
    assert min(draws) >= 300 and max(draws) <= 4_000
    # the unclipped median must still be the configured one
    rng = random.Random(1)
    raw = [rng.lognormvariate(math.log(1_000), 1.5) for _ in range(2_000)]
    assert pct(raw, 50) == pytest.approx(1_000, rel=0.06)


def test_session_generation_is_deterministic():
    cfg, opts = small_cfg(), small_opts()
    pre = build_prefixes(cfg.workload, opts.chars_per_token)
    a = make_session(cfg.workload, opts, pre, uid=5, is_sub=False)
    b = make_session(cfg.workload, opts, pre, uid=5, is_sub=False)
    c = make_session(cfg.workload, opts, pre, uid=6, is_sub=False)
    assert a.ctx == b.ctx and a.ctx != c.ctx
    assert a.next_turn(force_miss=False)[0] == b.next_turn(force_miss=False)[0]


def test_prefixes_are_byte_stable_across_processes():
    cfg, opts = small_cfg(), small_opts()
    p1 = build_prefixes(cfg.workload, opts.chars_per_token)
    p2 = build_prefixes(cfg.workload, opts.chars_per_token)
    assert p1 == p2 and p1.user != p1.sub


def test_every_session_shares_one_prefix_per_class():
    cfg, opts = small_cfg(), small_opts()
    pre = build_prefixes(cfg.workload, opts.chars_per_token)
    users = [make_session(cfg.workload, opts, pre, uid=i, is_sub=False)
             for i in range(4)]
    subs = [make_session(cfg.workload, opts, pre, uid=100 + i, is_sub=True)
            for i in range(4)]
    assert {s.prefix_text for s in users} == {pre.user}
    assert {s.prefix_text for s in subs} == {pre.sub}
    # ...unless the config points subagents at the user prefix
    shared = small_cfg(sub_shares_prefix=True)
    s = make_session(shared.workload, opts, pre, uid=1, is_sub=True)
    assert s.prefix_text == pre.user


def test_miss_salt_sits_ahead_of_the_prefix():
    cfg, opts = small_cfg(), small_opts()
    pre = build_prefixes(cfg.workload, opts.chars_per_token)
    s = make_session(cfg.workload, opts, pre, uid=1, is_sub=False)
    first, kind = s.next_turn()
    assert kind == "first" and first.startswith(pre.user)
    s.commit("reply")
    miss, kind = s.next_turn(force_miss=True)
    assert kind == "miss" and miss.startswith("[miss-salt ")
    assert pre.user in miss


def test_warm_turn_extends_the_cached_run():
    cfg, opts = small_cfg(), small_opts()
    pre = build_prefixes(cfg.workload, opts.chars_per_token)
    s = make_session(cfg.workload, opts, pre, uid=1, is_sub=False)
    p1, _ = s.next_turn()
    s.commit("REPLY")
    p2, kind = s.next_turn(force_miss=False)
    assert kind == "hit"
    assert p2.startswith(p1)          # byte-identical prefix run, then more
    assert "REPLY" in p2 and len(p2) > len(p1)


def test_sampler_selfcheck_reproduces_median_and_sigma():
    rows, ok = sampler_selfcheck(RunConfig().workload, ProbeOptions())
    assert ok
    user = rows[0]
    assert user["median_smp"] == pytest.approx(user["median_cfg"], rel=0.03)
    assert user["sigma_smp"] == pytest.approx(user["sigma_cfg"], rel=0.03)
    # the subagent row clips at ITS OWN prefix, not the user prefix
    assert rows[1]["p5"] == sub_prefix_floor(RunConfig().workload)


def test_sampler_selfcheck_fails_on_a_drifted_sampler(monkeypatch):
    import workingset.probe.session as S
    real = random.Random.lognormvariate
    monkeypatch.setattr(random.Random, "lognormvariate",
                        lambda self, mu, sigma: real(self, mu, sigma) * 1.5)
    _, ok = S.sampler_selfcheck(RunConfig().workload, ProbeOptions())
    assert not ok


# ============================================================================
# ladder
# ============================================================================
def test_build_ladder_brackets_the_prediction():
    assert build_ladder(403, "0.25,0.5,1,2", 1024, 64) == [64, 101, 202, 403, 806]


def test_build_ladder_always_includes_the_operating_point():
    lad = build_ladder(400, "1", 1024, 17)
    assert 17 in lad and 400 in lad


def test_build_ladder_respects_max_users_and_the_floor_of_one():
    assert build_ladder(400, "0.5,1,2", max_users=500) == [200, 400]
    assert build_ladder(2, "0.001,1", max_users=1024) == [1, 2]


# ============================================================================
# rung statistics on synthetic traces
# ============================================================================
def trace(uid, kind, t_send, ttft, tps=None, gaps_ms=(), ctok=None,
          ptok=None, intended=None, err=None, cached=None) -> RequestTrace:
    t = RequestTrace(uid=uid, kind=kind, t_send=t_send, ttft=ttft,
                     decode_tps=tps, ctok=ctok, ptok_achieved=ptok,
                     ptok_intended=intended or 0, error=err,
                     cached_tokens=cached)
    t.gaps_ms = list(gaps_ms)
    t.n_chunks = len(gaps_ms) + 1 if gaps_ms else 0
    if gaps_ms:
        t.span_s = sum(gaps_ms) / 1e3
        t.t_end = t_send + (ttft or 0) + t.span_s
    t.summarise_gaps(100.0)
    return t


def test_eval_rung_percentiles_and_per_user_decode():
    cfg, opts = small_cfg(), small_opts(measure_s=10.0)
    tr = [
        # user 1: two hits, decode 100 and 200 -> within-user p50 = 150
        trace(1, "hit", 1.0, 0.5, tps=100, ctok=10, ptok=100, intended=100),
        trace(1, "hit", 2.0, 1.5, tps=200, ctok=10, ptok=100, intended=100),
        # user 2: one hit at 50, one miss at 3.0 s TTFT
        trace(2, "hit", 1.0, 2.5, tps=50, ctok=10, ptok=100, intended=100),
        trace(2, "miss", 2.0, 3.0, tps=60, ctok=10, ptok=100, intended=100),
        # before the measure window: ignored
        trace(3, "hit", 0.1, 9.9, tps=1, ctok=10),
        # first turns are never measured
        trace(4, "first", 1.0, 9.9, tps=1, ctok=10),
    ]
    r = eval_rung(4, 2, tr, measure_start=0.5, cfg=cfg, opts=opts)
    assert r.n_turns == 4 and r.n_hit == 3 and r.n_miss == 1 and r.n_err == 0
    assert r.ttft_hit_p50 == pytest.approx(1.5)
    assert r.ttft_miss_mean == pytest.approx(3.0)
    assert r.ttft_all_pX == pytest.approx(pct([0.5, 1.5, 2.5, 3.0], 95))
    # per-user p50 = p50 across [150, 55] = 102.5
    assert r.decode_p50 == pytest.approx(102.5)
    assert r.ptok_ratio == pytest.approx(1.0)
    assert r.offered_rps == pytest.approx(6 / cfg.workload.think_time_s)
    assert r.achieved_rps == pytest.approx(4 / 10.0)


def test_eval_rung_slo_verdict_and_blown():
    cfg = small_cfg()
    slo = replace(cfg.slo, ttft_budget_s=1.0, itl_floor_tok_s=40, percentile=95)
    cfg = replace(cfg, slo=slo)
    opts = small_opts(measure_s=1.0)

    ok = [trace(1, "hit", 1.0, 0.5, tps=100, ctok=10)]
    assert eval_rung(1, 0, ok, 0.0, cfg, opts).passed

    slow = [trace(1, "hit", 1.0, 1.5, tps=100, ctok=10)]
    r = eval_rung(1, 0, slow, 0.0, cfg, opts)
    assert not r.passed and "TTFT" in r.reasons[0] and not r.blown

    dead = [trace(1, "hit", 1.0, 2.5, tps=100, ctok=10)]
    assert eval_rung(1, 0, dead, 0.0, cfg, opts).blown

    starved = [trace(1, "hit", 1.0, 0.5, tps=10, ctok=10)]
    r = eval_rung(1, 0, starved, 0.0, cfg, opts)
    assert not r.passed and "decode" in r.reasons[0] and r.blown

    errs = [trace(i, "hit", 1.0, None, err="boom") for i in range(4)]
    r = eval_rung(4, 0, errs, 0.0, cfg, opts)
    assert not r.passed and r.blown and r.n_err == 4


def test_eval_rung_gap_statistics_are_per_event_and_per_token():
    cfg, opts = small_cfg(), small_opts(measure_s=1.0, freeze_threshold_ms=100.0)
    # one response: gaps 2, 2, 2, 500 ms over 100 decoded tokens in 4 events
    tr = [trace(1, "hit", 1.0, 0.1, tps=100, gaps_ms=[2, 2, 2, 500], ctok=100)]
    r = eval_rung(1, 0, tr, 0.0, cfg, opts)
    assert r.itl_p50_ms == pytest.approx(2.0)
    assert r.itl_worst_p50_ms == pytest.approx(500.0)
    assert r.itl_floor_ms == pytest.approx(2.0)
    # one freeze over 100 tokens -> 10 per 1k tokens
    assert r.freeze_per_ktok == pytest.approx(10.0)
    assert r.stall_ms_per_ktok == pytest.approx(5_000.0)
    # 100 tokens over 5 SSE events
    assert r.chunk_tok_ratio == pytest.approx(20.0)
    assert r.stall_frac == pytest.approx(0.5 / 0.506, rel=1e-3)
    ladder = {e["threshold_ms"]: e["per_ktok"] for e in r.freeze_ladder}
    assert ladder[50.0] == pytest.approx(10.0)
    assert ladder[1000.0] == pytest.approx(0.0)


def test_eval_rung_eviction_and_cached_fractions():
    cfg, opts = small_cfg(), small_opts(measure_s=1.0)
    tr = [
        trace(1, "miss", 1.0, 10.0, tps=100, ctok=10),
        trace(2, "hit", 1.0, 0.1, tps=100, ctok=10, ptok=1000, cached=990),
        # >= 0.4 x 10 s -> the classifier calls this a re-prefill
        trace(3, "hit", 1.0, 5.0, tps=100, ctok=10, ptok=1000, cached=0),
    ]
    r = eval_rung(3, 0, tr, 0.0, cfg, opts)
    assert r.evict_frac == pytest.approx(0.5)
    assert r.cached_frac == pytest.approx(0.5)


def test_rung_round_trips_through_a_dict():
    cfg, opts = small_cfg(), small_opts(measure_s=1.0)
    r = eval_rung(1, 0, [trace(1, "hit", 1.0, 0.5, tps=100,
                               gaps_ms=[2, 300], ctok=10)], 0.0, cfg, opts)
    from workingset.probe.population import Rung
    back = Rung.from_dict(json.loads(json.dumps(r.to_dict())))
    assert back.pop == r.pop and back.passed == r.passed
    assert back.itl_p50_ms == pytest.approx(r.itl_p50_ms)
    assert len(back.traces) == 1 and back.traces[0].kind == "hit"
    # compact by default: the per-event gap list is summarised, not stored
    assert "gaps_ms" not in r.to_dict()["traces"][0]


# ============================================================================
# the request loop, against the fake SSE server
# ============================================================================
def test_send_request_reads_ttft_gaps_and_usage():
    async def go():
        handler = fake_server(n_tokens=5, delay=0.004, ttft=0.02, cached=7)
        async with client_for(handler) as c:
            ep = EndpointSpec(base_url="http://x/v1", model="m")
            tr = RequestTrace(uid=1, kind="miss", ptok_intended=25)
            txt = await send_request(c, ep, small_opts(), "p" * 100, tr, 5)
        assert txt == "tok " * 5
        assert tr.status == 200 and tr.error is None
        assert tr.ttft >= 0.02
        assert tr.n_chunks == 5 and tr.n_gaps == 4
        assert tr.ctok == 5 and tr.ptok_achieved == 25 and tr.cached_tokens == 7
        assert tr.decode_tps and tr.decode_tps > 0
        assert tr.span_s and tr.span_s > 0
    asyncio.run(go())


def test_send_request_counts_a_freeze():
    async def go():
        handler = fake_server(n_tokens=6, delay=0.001, freeze_at=3,
                              freeze_s=0.08)
        async with client_for(handler) as c:
            ep = EndpointSpec(base_url="http://x/v1", model="m")
            tr = RequestTrace(uid=1, kind="hit")
            await send_request(c, ep, small_opts(freeze_threshold_ms=50.0),
                               "p" * 40, tr, 6)
        assert tr.n_freeze == 1 and tr.itl_max >= 0.08
        assert tr.stall_s >= 0.08
        # the ladder counts it at 50 ms, not at 250 ms
        assert tr.n_freeze_at[0] == 1 and tr.n_freeze_at[2] == 0
    asyncio.run(go())


def test_stream_options_fallback_retries_once_without_it():
    seen: list = []
    async def go():
        handler = fake_server(reject_stream_options=True, seen=seen)
        async with client_for(handler) as c:
            ep = EndpointSpec(base_url="http://x/v1", model="m")
            tr = RequestTrace(uid=1, kind="hit")
            txt = await send_request(c, ep, small_opts(), "p" * 40, tr, 4)
        assert txt and tr.error is None
        assert ep.use_stream_options is False
        assert "stream_options" in seen[0] and "stream_options" not in seen[1]
        # no usage trailer without the extension: the ratio is unavailable
        assert tr.ptok_achieved is None
    asyncio.run(go())


def test_http_error_is_recorded_not_raised():
    def handler(_request):
        return httpx.Response(503, json={"error": "overloaded"})

    async def go():
        async with client_for(handler) as c:
            ep = EndpointSpec(base_url="http://x/v1", model="m")
            tr = RequestTrace(uid=1, kind="hit")
            txt = await send_request(c, ep, small_opts(), "p", tr, 4)
        assert txt == "" and tr.status == 503 and "503" in tr.error
    asyncio.run(go())


def test_chat_route_sends_messages():
    seen: list = []
    async def go():
        handler = fake_server(seen=seen)
        async with client_for(handler) as c:
            ep = EndpointSpec(base_url="http://x/v1", model="m", api="chat")
            tr = RequestTrace(uid=1, kind="hit")
            await send_request(c, ep, small_opts(), "hello", tr, 4)
        assert seen[0]["messages"][0]["content"] == "hello"
        assert tr.ttft is not None
    asyncio.run(go())


def test_ignore_eos_is_droppable():
    seen: list = []
    async def go():
        handler = fake_server(seen=seen)
        async with client_for(handler) as c:
            ep = EndpointSpec(base_url="http://x/v1", model="m")
            await send_request(c, ep, small_opts(), "x", RequestTrace(), 4)
            await send_request(c, ep, small_opts(ignore_eos=False), "x",
                               RequestTrace(), 4)
        assert seen[0]["ignore_eos"] is True and "ignore_eos" not in seen[1]
    asyncio.run(go())


def test_metrics_covariates_land_on_the_trace():
    async def go():
        m = FakeMetrics()
        async with client_for(fake_server()) as c:
            ep = EndpointSpec(base_url="http://x/v1", model="m")
            tr = RequestTrace(uid=1, kind="hit")
            await send_request(c, ep, small_opts(), "x", tr, 4, metrics=m)
        assert tr.covariates == {"requests_running": 3, "requests_waiting": 1,
                                 "kv_cache_usage": 0.42}
        assert m.at_calls == 1
    asyncio.run(go())


def test_a_broken_metrics_object_never_breaks_a_request():
    class Broken:
        def at(self, t):
            raise RuntimeError("scrape failed")

        def window(self, a, b):
            raise RuntimeError("scrape failed")

    async def go():
        async with client_for(fake_server()) as c:
            ep = EndpointSpec(base_url="http://x/v1", model="m")
            tr = RequestTrace(uid=1, kind="hit")
            await send_request(c, ep, small_opts(), "x", tr, 4, metrics=Broken())
        assert tr.covariates is None and tr.error is None and tr.ttft is not None
    asyncio.run(go())


# ============================================================================
# population / sample / burst against the fake server
# ============================================================================
def test_run_population_measures_a_rung():
    async def go():
        # sessions stagger over max(ramp_s, 1.0) s — the harness's floor, so
        # session establishment is never instantaneous. The measure window has
        # to outlast it for the rung to see any turns at all.
        cfg, opts = small_cfg(), small_opts(ramp_s=0.05, measure_s=1.3)
        pre = build_prefixes(cfg.workload, opts.chars_per_token)
        m = FakeMetrics()
        async with client_for(fake_server(n_tokens=6, delay=0.002)) as c:
            ep = EndpointSpec(base_url="http://x/v1", model="m")
            r = await run_population(c, ep, cfg, opts, 2, pre, metrics=m)
        assert r.pop == 2 and r.n_sub == 1
        assert r.n_turns > 0 and r.n_err == 0
        assert math.isfinite(r.decode_p50) and r.decode_p50 > 0
        assert r.ptok_ratio == pytest.approx(1.0, rel=0.05)
        assert r.server == {"prompt_tokens": 1234, "generation_tokens": 567,
                            "seconds": pytest.approx(r.server["seconds"])}
        assert all(t.covariates for t in r.traces if t.ttft is not None)
    asyncio.run(go())


def test_run_sample_fires_warm_turns_and_a_forced_miss():
    async def go():
        cfg = small_cfg(miss_rate=0.0)
        opts = small_opts(sample_requests=2, sample_warm_turns=2)
        pre = build_prefixes(cfg.workload, opts.chars_per_token)
        async with client_for(fake_server(n_tokens=5)) as c:
            ep = EndpointSpec(base_url="http://x/v1", model="m")
            s = await run_sample(c, ep, cfg, opts, pre)
        kinds = [t.kind for t in s.traces]
        assert kinds.count("first") == 2 and kinds.count("miss") == 2
        assert kinds.count("hit") == 4
        assert s.n_err == 0 and math.isfinite(s.ttft_miss_mean)
        assert math.isfinite(s.decode_p50)
    asyncio.run(go())


def test_run_burst_times_the_drain():
    async def go():
        cfg, opts = small_cfg(), small_opts(ramp_s=0.05)
        pre = build_prefixes(cfg.workload, opts.chars_per_token)
        async with client_for(fake_server(n_tokens=6, delay=0.003)) as c:
            ep = EndpointSpec(base_url="http://x/v1", model="m")
            b = await run_burst(c, ep, cfg, opts, n=3, standing_users=2,
                                prefixes=pre)
        assert b.n == 3 and b.n_ok == 3 and b.n_err == 0
        assert b.drain_s is not None and b.drain_s > 0
        assert b.last_ttft_s is not None
        assert all(t.kind == "miss" for t in b.traces)
    asyncio.run(go())


def test_eval_burst_reads_the_standing_load_in_flight():
    burst = [trace(990_000, "miss", 10.0, 1.0), trace(990_001, "miss", 10.0, 2.0)]
    standing = [
        # in flight across the fire instant
        trace(1, "hit", 9.0, 0.1, gaps_ms=[2, 400], ctok=50),
        # finished before it
        trace(2, "hit", 1.0, 0.1, gaps_ms=[2, 2], ctok=50),
    ]
    standing[0].t_end = 12.0
    standing[1].t_end = 2.0
    b = eval_burst(2, 4, burst, standing, t_fire=10.0)
    assert b.n_ok == 2 and b.last_ttft_s == pytest.approx(2.0)
    assert b.drain_s == pytest.approx(2.0)
    assert b.standing_n == 1
    assert b.standing_worst_max_ms == pytest.approx(400.0)


def test_eval_sample_on_synthetic_traces():
    s = eval_sample([
        trace(1, "miss", 1.0, 2.0, tps=90, gaps_ms=[3, 3], ctok=10),
        trace(1, "hit", 2.0, 0.4, tps=110, gaps_ms=[3, 3], ctok=10,
              ptok=100, intended=100, cached=99),
        trace(2, "hit", 2.0, 0.6, tps=None, err="boom"),
    ])
    assert s.n == 3 and s.n_ok == 2 and s.n_err == 1
    assert s.ttft_miss_mean == pytest.approx(2.0)
    assert s.decode_p50 == pytest.approx(100.0)
    assert s.cached_frac == pytest.approx(1.0)
    assert s.ptok_ratio == pytest.approx(1.0)
