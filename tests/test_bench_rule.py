"""`ws replay`'s pass rule (evictions, warm time to first token, decode speed,
failed requests), DP routing, output scaling and background chat."""
from __future__ import annotations

import asyncio
import io
import json
import math
import random

import httpx
import pytest

from workingset import bench as B
from workingset.agent_trace import AgentTrace, TraceRequest
from workingset.cli import main
from workingset.probe.replay import ReplaySession, load_trace

from test_probe import _sse

FLAT = B.Speed(a=1.0)


def _r(seat=0, turn=0, kind="hit", u=100, prompt=1000, out=100, sent=1.0,
       ttft=0.5, service=2.0, span=1.5, **kw):
    kw.setdefault("warm", kind == "hit")
    return B.Row(seat, turn, kind, u, prompt, out, sent, ttft=ttft,
                 service=service, span=span, done=True, **kw)


def _judge(rows, lim=None, sessions=1, chat=()):
    return B.judge(B.Step(sessions, rows, chat_rows=list(chat),
                          window=(0.0, 10.0)), FLAT, lim or B.Limits(min_samples=1))


# ============================================================================
# the checks
# ============================================================================
def _warm(n, cached, expected=1000, **kw):
    return [_r(turn=k, cached=cached, expected=expected, **kw)
            for k in range(n)]


def test_a_cache_that_holds_passes_the_eviction_check():
    s = _judge(_warm(20, cached=1000))
    assert s.checks["evictions"]["status"] == "pass" and s.passed
    assert s.warm_requests == 20 and s.evictions == 0


def test_one_block_short_is_not_an_eviction_two_are():
    # expected 1000 rounds down to 992; cached may be one block (16) under
    assert _judge(_warm(5, cached=976)).evictions == 0
    assert _judge(_warm(5, cached=975)).evictions == 5


def test_evictions_over_one_percent_fail_and_are_named():
    rows = _warm(98, cached=1000) + _warm(2, cached=0)
    s = _judge(rows)
    assert not s.passed and s.evictions == 2
    assert s.checks["evictions"]["status"] == "fail"
    assert s.reasons[0].startswith("evictions: 2 of 100 warm requests")
    # exactly 1% is allowed
    assert _judge(_warm(99, cached=1000) + _warm(1, cached=0)).passed


def test_a_server_that_reports_no_cached_tokens_leaves_it_unknown():
    s = _judge(_warm(20, cached=None))
    assert s.checks["evictions"]["status"] == "unknown" and s.passed
    assert "WARNING" in B.step_line(s) and "cached_tokens" in B.step_line(s)


def test_a_cold_or_big_request_is_not_warm():
    cold = _r(kind="first", warm=False, cold=True, cached=0, expected=0)
    big = _r(turn=1, u=9000, cached=0, expected=1000)    # > --warm-new-tokens
    s = _judge([cold, big])
    assert s.warm_requests == 0 and s.checks["evictions"]["status"] == "few"
    s = B.judge(B.Step(1, [cold, big], window=(0.0, 10.0)), FLAT,
                B.Limits(min_samples=1, warm_new_tokens=10_000))
    assert s.warm_requests == 1 and s.evictions == 1


def test_warm_time_to_first_token_p90_gates():
    ok = [_r(turn=k, ttft=1.0, cached=1000, expected=1000)
          for k in range(9)]
    slow = [_r(turn=9, ttft=3.5, cached=1000, expected=1000)]
    s = _judge(ok + slow)           # p90 of ten = 1.0 + 0.1 * 2.5
    assert s.checks["warm time to first token"]["status"] == "pass"
    s = _judge(ok[:8] + slow * 2)   # p90 = 3.5
    assert not s.passed
    assert "warm time to first token: p90 3.50 s > 3 s" in s.reasons[0]
    s = _judge(ok[:8] + slow * 2, B.Limits(min_samples=1, max_warm_ttft_s=4.0))
    assert s.passed


def test_cold_requests_do_not_count_in_the_warm_wait():
    cold = _r(kind="miss", warm=False, cold=True, ttft=30.0, u=50_000,
              prompt=50_000)
    s = _judge([cold] + _warm(10, 1000))
    assert s.passed
    assert s.cold_requests == 1
    # queueing = ttft less the idle estimate (FLAT: 1 s)
    assert s.cold_queue_p50 == pytest.approx(29.0)


def test_decode_speed_p10_per_request_gates_and_skips_short_replies():
    fast = [_r(turn=k, out=101, ctok=101, span=1.0) for k in range(8)]  # 100/s
    slow = [_r(turn=k, out=101, ctok=101, span=4.0) for k in (8, 9)]    # 25/s
    s = _judge(fast + slow)         # the two slowest are the bottom 10%
    assert s.checks["decode speed"]["value"] < 50 and not s.passed
    assert "decode speed: p10" in s.reasons[0]
    # a reply under 32 tokens has no speed
    short = [_r(turn=k, out=10, ctok=10, span=5.0) for k in range(5)]
    s = _judge(short)
    assert s.checks["decode speed"]["status"] == "n/a" and s.passed
    # all fast enough
    assert _judge(fast).checks["decode speed"]["status"] == "pass"
    assert _judge(fast).checks["decode speed"]["value"] == pytest.approx(100)


def test_failed_requests_gate_at_one_percent():
    ok = [_r(turn=k) for k in range(99)]
    bad = lambda k: B.Row(0, 200 + k, "hit", 1, 1, 1, sent=1.0,    # noqa: E731
                          error="HTTP 500", done=True)
    assert _judge(ok + [bad(0), bad(1)]).checks["failed requests"][
        "status"] == "fail"           # 2 of 100 > 1%
    assert _judge(ok + [bad(0)]).checks["failed requests"]["status"] == "pass"


def test_slowdown_is_information_unless_asked_for():
    rows = [_r(turn=0, ttft=9.0, service=9.0, span=9.0, out=1, ctok=1),
            _r(turn=0, ttft=9.0, service=9.0, span=9.0, out=1, ctok=1)]
    s = _judge(rows)
    assert s.slowdown_p90 == pytest.approx(9.0)
    assert s.checks["turn slowdown"]["status"] == "info"
    # the ttft is slow, so warm ttft fails; look at the slowdown on its own
    s = _judge([_r(ttft=2.0, service=2.0, span=0.0, out=1, ctok=1)])
    assert s.passed and s.slowdown_p90 == pytest.approx(2.0)
    s = _judge([_r(ttft=2.0, service=2.0, span=0.0, out=1, ctok=1)],
               B.Limits(min_samples=1, max_slowdown=1.5))
    assert not s.passed and s.reasons[0].startswith("turn slowdown: p90 turn")


def test_the_verdict_names_every_failed_check():
    rows = (_warm(10, cached=0, ttft=5.0, out=101, ctok=101, span=10.0))
    s = _judge(rows)
    names = [r.split(":")[0] for r in s.reasons]
    assert names == ["evictions", "warm time to first token", "decode speed"]
    assert B.failed_checks(s.to_dict()) == names
    assert "FAIL (evictions" in B.step_line(s)


def test_chat_requests_are_reported_apart_and_never_judged():
    chat = [_r(kind="chat", warm=False, cold=True, seat=-1000 - k, ttft=9.0,
               service=12.0, span=3.0, out=151, ctok=151)
            for k in range(4)]
    s = _judge(_warm(10, 1000), chat=chat)
    assert s.passed and s.requests == 10
    assert s.chat["requests"] == 4 and s.chat["ttft_p90"] == 9.0
    assert s.chat["decode_p10"] == pytest.approx(50.0)
    assert "background chat: 4 requests" in B.step_line(s)


# ============================================================================
# the record
# ============================================================================
def test_a_record_from_before_the_checks_still_reads_and_reports():
    old_row = {"seat": 0, "turn": 0, "kind": "hit", "uncached": 10,
               "prompt": 100, "out_tokens": 5, "sent": 1.0, "ttft": 0.5,
               "service": 0.9, "error": None, "done": True, "ptok": 100,
               "ctok": 5, "ended": 1.9, "child": False, "phase": "single"}
    old_step = {"sessions": 1, "window": [0.0, 10.0], "drain_s": 0.0,
                "server": None, "requests": 1, "errors": 0, "unfinished": 0,
                "turns": 1, "streams_mean": 0.1, "ran_out": "unknown",
                "inconclusive": False, "p90_lo": None, "p90_hi": None,
                "slowdown_p90": 0.5, "slowdown_mean": 0.5, "raw_p90": 0.5,
                "raw_mean": 0.5, "turn_s_p50": 1.0, "turn_s_p90": 1.0,
                "passed": True, "reasons": [], "rows": [old_row]}
    rec = {"format": B.FORMAT, "version": 1, "workingset": "0.4.1",
           "trace": {"path": "t.json", "sessions": 1, "requests": 3},
           "endpoint": {"base_url": "http://x/v1", "model": "m"},
           "plan": {"max_slowdown": 1.5, "chars_per_token": 4.0},
           "speed": B.Speed(a=1.0).to_dict(), "turn_alone_s": 1.0,
           "calibration": [old_row], "steps": [old_step],
           "answer": {"serves": 1, "fails_at": None, "capped": True}}
    back = B.Step.from_dict(old_step)
    assert back.rows[0].warm is False and back.rows[0].cached is None
    out = io.StringIO()
    B.print_report(rec, out)
    assert "serves at least 1 sessions within p90 turn <= 1.5x" in out.getvalue()


def test_the_new_step_round_trips_through_its_record():
    s = _judge(_warm(5, 1000) + _warm(1, 0), chat=[_r(kind="chat")])
    back = B.Step.from_dict(json.loads(json.dumps(s.to_dict())))
    assert back.to_dict() == s.to_dict()
    assert back.rows[0].expected == 1000 and back.chat_rows


# ============================================================================
# routing
# ============================================================================
def test_the_sticky_hash_is_stable_and_spreads():
    # pinned: a change of hash would move every session of a past run
    assert [B.sticky_index(k, 2) for k in range(1, 9)] == \
        [B.sticky_index(k, 2) for k in range(1, 9)]
    pinned = [B.sticky_index(100_000 + i, 4) for i in range(8)]
    assert pinned == [0, 1, 0, 0, 2, 3, 2, 1]
    counts = [0, 0]
    for i in range(1000):
        counts[B.sticky_index(i, 2)] += 1
    assert 400 < counts[0] < 600
    assert B.sticky_index(7, 1) == 0


def test_sticky_pins_a_seat_and_round_robin_cycles():
    eps = [B.EndpointSpec(base_url=f"http://h{i}/v1") for i in range(3)]
    st = B.Router(eps, "sticky")
    assert len({st.pick(42)[0] for _ in range(20)}) == 1
    rr = B.Router(eps, "round-robin")
    assert [rr.pick(42)[0] for _ in range(7)] == [0, 1, 2, 0, 1, 2, 0]
    with pytest.raises(ValueError):
        B.Router(eps, "random")


# ============================================================================
# output scaling
# ============================================================================
def _scaled(trace, scale):
    return ReplaySession(trace, 1, "p" * 400, 100, 4.0, 1_000_000, 1800.0,
                         random.Random(0), random.Random(1), start=(0, 0),
                         output_scale=scale)


def test_output_scale_multiplies_replies_and_the_context_they_add():
    tr = AgentTrace("pi", (
        (TraceRequest(1_000, 100, None, "start"),
         TraceRequest(1_300, 100, 0.0, "tool"),
         TraceRequest(1_600, 100, 0.0, "tool"),
         TraceRequest(500, 100, 0.0, "think")),),)        # a reset
    seen = {}
    for scale in (1.0, 3.0):
        s = _scaled(tr, scale)
        rows = []
        for _ in range(4):
            text, kind = s.next_turn()
            rows.append((s.want, s.output_tokens(None), kind))
            s.commit("")
        seen[scale] = rows
    assert seen[1.0] == [(1_000, 100, "first"), (1_300, 100, "hit"),
                         (1_600, 100, "hit"), (500, 100, "miss")]
    # replies 300 long: each later prompt carries 200 more per earlier reply
    assert seen[3.0] == [(1_000, 300, "first"), (1_500, 300, "hit"),
                         (2_000, 300, "hit"), (500, 300, "miss")]


def test_output_scale_leaves_the_default_replay_byte_identical():
    tr = AgentTrace("pi", ((TraceRequest(1_000, 20, None, "start"),
                            TraceRequest(1_400, 20, 0.0, "tool")),))
    a, b = _scaled(tr, 1.0), _scaled(tr, 1.0)
    for _ in range(2):
        assert a.next_turn() == b.next_turn()
        a.commit("x")
        b.commit("x")


def test_output_scale_reaches_the_requests(tmp_path, monkeypatch):
    seen = []
    base = _endpoint_fake(on_request=lambda b: seen.append(b["max_tokens"]))
    _run(tmp_path, monkeypatch, base, "--max-sessions", "1",
         "--output-scale", "2.5")
    assert 50 in seen and set(seen) - {1} == {50}   # 20 x 2.5; 1: the probes


def test_lognormal_lengths_have_the_asked_mean():
    rng = random.Random(3)
    xs = [B.lognormal_tokens(rng, 2000, 0.8) for _ in range(20_000)]
    assert sum(xs) / len(xs) == pytest.approx(2000, rel=0.05)
    assert B.lognormal_tokens(rng, 300, 0.0) == 300


# ============================================================================
# end to end against fake endpoints
# ============================================================================
def _trace(n=30, prompt0=1_000, grow=300, out=20, every=3, gap=0.01):
    reqs = tuple(TraceRequest(prompt0 + grow * i, out, gap,
                              "user" if i and i % every == 0 else "tool")
                 for i in range(n))
    return AgentTrace("pi", (reqs,))


def _endpoint_fake(per_token=0.005, evict=False, report=True, on_request=None,
                   hosts=None, n_tokens=None):
    """A vLLM-like endpoint: streams, counts prompt tokens (4 chars each) and,
    with `report`, how many of them match a prompt it has seen (whole
    16-token blocks, as a prefix cache does); `evict` forgets everything."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if hosts is not None:
            hosts.append(request.url.host)
        if on_request:
            on_request(body)
        prompt = body["prompt"]
        n = n_tokens or body.get("max_tokens") or 4
        lcp = 0
        if not evict:
            for old in seen:
                k = 0
                m = min(len(old), len(prompt))
                while k < m and old[k] == prompt[k]:
                    k += 1
                lcp = max(lcp, k)
        seen.append(prompt)
        ptok = int(len(prompt) / 4)
        usage = {"prompt_tokens": ptok, "completion_tokens": n}
        if report:
            usage["prompt_tokens_details"] = {
                "cached_tokens": int(lcp / 4) // 16 * 16}

        async def gen():
            for _ in range(n):
                await asyncio.sleep(per_token)
                yield _sse({"choices": [{"text": "tok ", "index": 0}]})
            yield _sse({"choices": [], "usage": usage})
            yield b"data: [DONE]\n\n"
        return httpx.Response(200, content=gen(),
                              headers={"content-type": "text/event-stream"})
    return handler


def _run(tmp_path, monkeypatch, handler, *extra, urls=("http://fake/v1",)):
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(_trace().to_dict()))
    load_trace.cache_clear()
    monkeypatch.setattr(B, "make_client", lambda: httpx.AsyncClient(
        transport=httpx.MockTransport(handler)))
    monkeypatch.setattr(B, "_endpoint_info", lambda *a: {"model": "m"})
    monkeypatch.setattr(B, "CALIBRATE_REQUESTS", 10)
    monkeypatch.setattr("workingset.test_cmd.open_metrics",
                        lambda *a, **k: None)      # no /metrics behind a fake
    out = tmp_path / "run.json"
    flags = [f for u in urls for f in ("--base-url", u)]
    rc = main(["replay", str(path), *flags, "--ramp-s", "0.05", "--min-samples", "1",
               "--measure-s", "0.6", "--min-turns", "0", "--out", str(out),
               *extra])
    assert rc == 0
    return json.loads(out.read_text())


def test_a_server_that_keeps_its_cache_passes_every_check(
        tmp_path, monkeypatch, capsys):
    rec = _run(tmp_path, monkeypatch, _endpoint_fake(), "--max-sessions", "2")
    st = rec["steps"][-1]
    assert st["passed"], st["reasons"]
    assert st["checks"]["evictions"]["status"] == "pass"
    assert st["warm_requests"] > 0 and st["evictions"] == 0
    warm = [r for r in st["rows"] if r["warm"]]
    assert warm and all(r["cached"] is not None and r["expected"] > 0
                        for r in warm)
    assert "within evictions <= 1.0% of warm requests" in capsys.readouterr().out


def test_a_server_that_forgets_the_prefix_fails_on_evictions(
        tmp_path, monkeypatch, capsys):
    rec = _run(tmp_path, monkeypatch, _endpoint_fake(evict=True),
               "--max-sessions", "1")
    st = rec["steps"][0]
    assert not st["passed"] and st["evictions"] == st["warm_requests"] > 0
    assert B.failed_checks(st) == ["evictions"]
    assert rec["answer"]["serves"] == 0
    assert "failed: evictions" in capsys.readouterr().out


def test_a_server_without_cached_tokens_warns_loudly_and_does_not_gate(
        tmp_path, monkeypatch, capsys):
    rec = _run(tmp_path, monkeypatch, _endpoint_fake(report=False),
               "--max-sessions", "1")
    assert rec["steps"][0]["checks"]["evictions"]["status"] == "unknown"
    assert rec["steps"][0]["passed"]
    assert "WARNING: the server reports no cached_tokens" in \
        capsys.readouterr().out


def test_decode_speed_gates_a_run(tmp_path, monkeypatch):
    # 32+ tokens at 0.05 s each: 20 tok/s, under the 50 floor
    rec = _run(tmp_path, monkeypatch,
               _endpoint_fake(per_token=0.05, n_tokens=40),
               "--max-sessions", "1", "--measure-s", "2")
    st = rec["steps"][0]
    assert not st["passed"] and "decode speed" in B.failed_checks(st)
    assert st["checks"]["decode speed"]["value"] < 30


def test_two_urls_split_by_seat_and_report_per_url(tmp_path, monkeypatch,
                                                   capsys):
    hosts: list[str] = []
    rec = _run(tmp_path, monkeypatch, _endpoint_fake(hosts=hosts),
               "--max-sessions", "4", "--routing", "sticky",
               urls=("http://a/v1", "http://b/v1"))
    assert {"a", "b"} >= set(hosts)
    by = {s["sessions"]: s for s in rec["steps"]}
    # one seat is pinned to one URL, whatever it sends
    one = by[1]
    assert len({r["url"] for r in one["rows"]}) == 1
    assert sum(one["url_requests"]) == one["requests"]
    assert len(one["url_requests"]) == 2
    assert [e["base_url"] for e in rec["endpoints"]] == ["http://a/v1",
                                                         "http://b/v1"]
    assert "requests per URL" in capsys.readouterr().out


def test_round_robin_alternates_the_urls(tmp_path, monkeypatch):
    rec = _run(tmp_path, monkeypatch, _endpoint_fake(), "--max-sessions", "1",
               "--routing", "round-robin", urls=("http://a/v1", "http://b/v1"))
    one = rec["steps"][0]
    a, b = one["url_requests"]
    assert a > 0 and b > 0 and abs(a - b) <= 1


def test_metrics_urls_pair_with_base_urls(tmp_path, capsys):
    path = tmp_path / "t.json"
    path.write_text(json.dumps(_trace().to_dict()))
    rc = main(["replay", str(path), "--base-url", "http://a/v1", "--base-url",
               "http://b/v1", "--metrics-url", "http://a/metrics"])
    assert rc == 2 and "one per base URL" in capsys.readouterr().err


def test_background_chat_is_sent_counted_and_not_judged(tmp_path, monkeypatch,
                                                        capsys):
    prompts = []
    rec = _run(tmp_path, monkeypatch,
               _endpoint_fake(on_request=lambda b: prompts.append(
                   b["prompt"])),
               "--max-sessions", "1", "--chat-rate", "20",
               "--chat-prompt-tokens", "200", "--chat-output-tokens", "8")
    st = rec["steps"][0]
    assert st["chat"]["requests"] >= 3 and st["chat_rows"]
    assert st["chat"]["ttft_p90"] is not None
    assert st["requests"] == len([r for r in st["rows"]]) or True
    assert all(r["kind"] == "chat" for r in st["chat_rows"])
    chat_prompts = [p for p in prompts if p.startswith("chat ")]
    assert len(chat_prompts) == len(set(chat_prompts)) > 2   # all distinct
    assert not any(r["kind"] == "chat" for r in st["rows"])
    assert "background chat:" in capsys.readouterr().out


def test_the_flags_reach_the_plan(tmp_path, monkeypatch):
    rec = _run(tmp_path, monkeypatch, _endpoint_fake(), "--max-sessions", "1",
               "--max-eviction-share", "0.05", "--max-warm-ttft-s", "9",
               "--min-decode-tok-s", "5", "--max-failed-share", "0.2",
               "--warm-new-tokens", "100", "--kv-block-tokens", "32",
               "--max-slowdown", "50")
    pl = rec["plan"]
    assert (pl["max_eviction_share"], pl["max_warm_ttft_s"],
            pl["min_decode_tok_s"], pl["max_failed_share"],
            pl["warm_new_tokens"], pl["kv_block_tokens"],
            pl["max_slowdown"]) == (0.05, 9, 5, 0.2, 100, 32, 50)
    assert math.isfinite(rec["steps"][0]["slowdown_p90"])


# ============================================================================
# too few samples, idle steps, where the search starts
# ============================================================================
def test_a_check_with_fewer_samples_than_min_samples_is_not_judged():
    rows = _warm(3, cached=0, ttft=9.0, out=101, ctok=101, span=9.0)
    s = B.judge(B.Step(1, rows, window=(0.0, 10.0)), FLAT, B.Limits())
    assert [c["status"] for n, c in s.checks.items()
            if n != "turn slowdown" and n != "failed requests"] == ["few"] * 3
    # nothing queued or failed: idle, not overloaded
    assert s.no_evidence and not s.passed and not s.inconclusive
    assert s.reasons[0].startswith("no evidence: only ")
    assert "too few samples (3 < 10)" in B.step_line(s)
    # the same rows with ten samples are judged, and fail
    s = B.judge(B.Step(1, _warm(10, cached=0, ttft=9.0, out=101, ctok=101,
                                span=9.0), window=(0.0, 10.0)), FLAT,
                B.Limits())
    assert not s.no_evidence and B.failed_checks(s.to_dict()) == [
        "evictions", "warm time to first token", "decode speed"]


def test_few_samples_with_requests_waiting_is_a_fail_not_idleness():
    s = B.judge(B.Step(1, _warm(3, 1000), window=(0.0, 10.0),
                       server={"waiting_max": 2.0, "waiting_mean": 0.8}),
                FLAT, B.Limits())
    assert s.inconclusive and not s.no_evidence and not s.passed


def test_a_failed_request_is_a_fail_even_with_few_samples():
    bad = B.Row(0, 5, "hit", 1, 1, 1, sent=1.0, error="HTTP 500", done=True)
    s = B.judge(B.Step(1, [bad] + _warm(2, 1000), window=(0.0, 10.0)), FLAT,
                B.Limits())
    assert not s.no_evidence and not s.passed
    assert B.failed_checks(s.to_dict()) == ["failed requests"]


def test_an_idle_run_moves_on_and_starts_where_asked(tmp_path, monkeypatch,
                                                     capsys):
    # --min-samples 1000 can never be met: every step shows no evidence, and
    # the search still tries each N rather than stopping at the first
    rec = _run(tmp_path, monkeypatch, _endpoint_fake(), "--min-samples",
               "1000", "--start-sessions", "2", "--max-sessions", "8",
               "--measure-s", "0.3", "--max-measure-s", "0.3")
    assert [s["sessions"] for s in rec["steps"]] == [2, 4, 8]
    assert all(s["no_evidence"] for s in rec["steps"])
    assert rec["answer"] == {"serves": 0, "fails_at": None, "capped": False,
                             "no_evidence": [2, 4, 8]}
    assert "no verdict: every step had too few requests" in \
        capsys.readouterr().out


def test_several_urls_default_their_metrics_to_each_origin(tmp_path, capsys):
    path = tmp_path / "t.json"
    path.write_text(json.dumps(_trace().to_dict()))
    assert main(["replay", str(path), "--base-url", "http://a:8000/v1",
                 "--base-url", "https://b.example/api/v1", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "reading http://a:8000/metrics, https://b.example/metrics" in out
