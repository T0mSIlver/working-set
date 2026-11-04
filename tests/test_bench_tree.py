"""`ws replay`: a seat is a session tree (subagents launched from their
parent), what the step measures about it, and the server-side verdict."""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from workingset import bench as B
from workingset.agent_trace import AgentTrace, SessionLink, TraceRequest
from workingset.probe.replay import load_trace, root_sessions

from test_probe import _sse


def _tree_trace():
    # the last request follows a person's message: the replay draws its
    # pause from these, so a seat that wraps round does not sit for 30 s
    parent = tuple(TraceRequest(1_000 + 200 * i, 10, None if i == 0 else 0.05,
                                "start" if i == 0 else
                                "user" if i == 3 else "tool")
                   for i in range(4))
    child = tuple(TraceRequest(500 + 100 * i, 5, None if i == 0 else 0.02,
                               "start" if i == 0 else "tool")
                  for i in range(3))
    return AgentTrace("claude", (parent, child),
                      links=(None, SessionLink(0, 1.0, 1)))


def test_a_subagent_starts_after_the_request_that_spawned_it():
    tr = _tree_trace()
    assert B.spawn_points(tr) == {0: [(1, 1)]}
    # no spawn request recorded: after the last request begun by the offset
    # (parent requests begin at 0, 0.05, 0.10, 0.15 s)
    tr2 = AgentTrace("claude", tr.sessions,
                     links=(None, SessionLink(0, 0.06, None)))
    assert B.spawn_points(tr2) == {0: [(1, 1)]}


def test_seats_never_play_a_subagent_recording_themselves():
    tr = _tree_trace()
    assert root_sessions(tr) == (0,)
    plan = B.Plan("t", context_cap_tokens=100_000)
    for uid in range(20):
        assert B._seat(tr, plan, "p" * 100, uid, "n")._s == 0


def _server_stub():
    def handler(request):
        body = json.loads(request.content)

        async def gen():
            yield _sse({"choices": [{"text": "tok ", "index": 0}]})
            yield _sse({"choices": [], "usage": {
                "prompt_tokens": int(len(body["prompt"]) / 4),
                "completion_tokens": 1}})
            yield b"data: [DONE]\n\n"
        return httpx.Response(200, content=gen(),
                              headers={"content-type": "text/event-stream"})
    return handler


def test_a_step_launches_each_subagent_from_its_parent(tmp_path):
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(_tree_trace().to_dict()))
    load_trace.cache_clear()
    tr = load_trace(str(path))
    plan = B.Plan(str(path), ramp_s=0.05, measure_s=0.6,
                  context_cap_tokens=100_000)

    async def go():
        async with httpx.AsyncClient(
                transport=httpx.MockTransport(_server_stub())) as c:
            return await B.run_step(
                c, B.EndpointSpec(base_url="http://fake/v1", model="m"),
                plan, tr, 1, "p" * 100, "nonce")
    step = asyncio.run(go())
    kids = [r for r in step.rows if r.child]
    mains = [r for r in step.rows if not r.child]
    assert kids and mains
    # the child is filed under its tree's seat and the turn that spawned it
    assert {r.seat for r in step.rows} == {100_000}
    assert {k.turn for k in kids} <= {m.turn for m in mains}
    # it starts only once the parent's second request has finished
    spawn_at = sorted(m.end for m in mains)[1] if len(mains) > 1 else 0
    assert min(k.sent for k in kids) >= spawn_at - 1e-6
    # and runs on its own recording, three requests per launch
    assert len(kids) % 3 == 0


def test_a_childs_requests_join_the_turn_that_spawned_it():
    main = _row_(0, 0, 1.0)
    kid = B.Row(0, 0, "first", 1, 1, 1, sent=1.0, ttft=3.0, service=3.0,
                done=True, ended=4.0, child=True)
    s = B.judge(B.Step(1, [main, kid], window=(0.0, 10.0)),
                B.Speed(a=1.0), 1.5)
    assert s.turns == 1 and s.slowdown_mean == pytest.approx(2.0)


def _row_(seat, turn, t):
    return B.Row(seat, turn, "hit", 1, 1, 1, sent=0.0, ttft=t, service=t,
                 done=True, ended=t)


# ---- what ran out ----------------------------------------------------------
def test_the_verdict_names_what_ran_out():
    assert B.ran_out(None)[0] == "unknown"
    kv = {"kv_max": 0.99, "preemptions": 3, "waiting_mean": 4,
          "waiting_max": 9, "running_max": 8}
    assert B.ran_out(kv)[0] == "kv"                       # before the queue
    assert B.ran_out({**kv, "kv_max": 0.5, "preemptions": 0})[0] == "queue"
    quiet = {"kv_max": 0.4, "preemptions": 0, "waiting_mean": 0.0,
             "waiting_max": 1, "running_max": 8}
    assert B.ran_out(quiet)[0] == "compute"


def test_the_failing_step_reports_what_ran_out():
    row = _row_(0, 0, 9.0)
    st = B.judge(B.Step(1, [row], window=(0.0, 10.0),
                        server={"kv_max": 0.97, "preemptions": 1.0}),
                 B.Speed(a=1.0), 1.5)
    assert st.ran_out == "kv" and not st.passed
    rec = {"steps": [st.to_dict()], "answer": {"serves": 0, "fails_at": 1,
                                               "capped": False},
           "plan": {"max_slowdown": 1.5}}
    assert "the limit is the KV cache" in B.headline(rec)


# ---- chars per token, from the server ------------------------------------
def test_chars_per_token_is_measured_from_the_servers_counts():
    def handler(request):
        body = json.loads(request.content)

        async def gen():
            yield _sse({"choices": [{"text": "t", "index": 0}]})
            # this tokenizer spends 5 characters per token
            yield _sse({"choices": [], "usage": {
                "prompt_tokens": int(len(body["prompt"]) / 5),
                "completion_tokens": 1}})
            yield b"data: [DONE]\n\n"
        return httpx.Response(200, content=gen(),
                              headers={"content-type": "text/event-stream"})
    plan = B.Plan("t", context_cap_tokens=100_000, cpt_source="default")

    async def go():
        async with httpx.AsyncClient(
                transport=httpx.MockTransport(handler)) as c:
            return await B.measure_chars_per_token(
                c, B.EndpointSpec(base_url="http://fake/v1", model="m"),
                plan, "n")
    assert asyncio.run(go()) == pytest.approx(5.0, rel=0.01)


# ---- calibration spreads over the trace ----------------------------------
def test_calibration_starts_spread_over_sessions_and_sizes():
    sessions = tuple(
        tuple(TraceRequest(1_000 * (k + 1) + 100 * i, 10,
                           None if i == 0 else 1.0, "start" if i == 0
                           else "tool") for i in range(10))
        for k in range(8))
    tr = AgentTrace("pi", sessions)
    plan = B.Plan("t", context_cap_tokens=1_000_000)
    starts = B.calibration_starts(tr, plan, 8)
    assert len(starts) == 8 and len({s for s, _ in starts}) >= 6
    sizes = sorted(tr.sessions[i][j].prompt_tokens for i, j in starts)
    assert sizes[0] < 2_000 and sizes[-1] > 7_000


# ---- /metrics credentials -------------------------------------------------
def test_metrics_headers():
    h = B.metrics_headers([], "k", "http://gpu:8000/v1", "http://gpu:8000/m")
    assert h == {"Authorization": "Bearer k"}
    # never to another host
    assert B.metrics_headers([], "k", "http://gpu:8000/v1",
                             "http://other:9/m") == {}
    h = B.metrics_headers(["X-Token: abc", "Authorization: Basic zz"], "k",
                          "http://gpu:8000/v1", "http://gpu:8000/m")
    assert h == {"X-Token": "abc", "Authorization": "Basic zz"}


def test_named_rungs_run_without_the_operating_point():
    from workingset.probe.ladder import build_ladder
    assert build_ladder(78, "0.03,0.05", 1024, 64, rungs_given=True) == [2, 4]
    assert build_ladder(78, "0.03,0.05", 1024, 64) == [2, 4, 64]


def test_a_dry_run_reads_the_context_cap_from_the_endpoint(
        tmp_path, monkeypatch, capsys):
    from workingset.cli import main
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(_tree_trace().to_dict()))
    load_trace.cache_clear()
    monkeypatch.setattr(B, "_endpoint_info", lambda *a: {
        "model": "served", "max_model_len": 4096})
    assert main(["replay", str(path), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "context cap 4,096 tokens (from /models)" in out
    assert "model=served" in out
    assert "1 sessions start a seat; the other 1 are subagents" in out


# ---- the baseline of a tree is that tree alone -----------------------------
FLAT1 = B.Speed(a=1.0)                 # every request: 1 s alone


def _kid(seat, turn, sent, took, child=False):
    return B.Row(seat, turn, "hit", 1, 1, 1, sent=sent, ttft=took,
                 service=took, done=True, ended=sent + took, child=child)


def test_a_trees_concurrency_counts_by_alone_time_not_load():
    # three requests of one tree sent together: each sees the other two. A
    # fourth sent long after sees none, however slow the first three were.
    rows = [_kid(0, 0, 0.0, 9.0), _kid(0, 0, 0.1, 9.0, True),
            _kid(0, 0, 0.2, 9.0, True), _kid(0, 0, 5.0, 1.0)]
    k = B.tree_k(rows, FLAT1)
    assert [k[id(r)] for r in rows] == [2, 2, 2, 0]


def test_the_per_stream_slowdown_is_fitted_on_a_tree_alone():
    # each request took (1 + 0.25 k) s alone, k the others in flight
    rows = [_kid(0, 0, 0.0, 1.5), _kid(0, 0, 0.1, 1.5, True),
            _kid(0, 0, 0.2, 1.5, True), _kid(0, 0, 5.0, 1.0)]
    # (k = 2 for the first three: 1.5 s is 1 + 0.25 * 2)
    beta, n = B.fit_tree_beta(rows, FLAT1)
    assert beta == pytest.approx(0.25) and n == 3


def test_a_trees_turn_is_judged_against_the_tree_alone():
    rows = [_kid(0, 0, 0.0, 1.5), _kid(0, 0, 0.1, 1.5, True),
            _kid(0, 0, 0.2, 1.5, True)]
    speed = B.Speed(a=1.0, tree_beta=0.25)
    s = B.judge(B.Step(1, rows, window=(0.0, 10.0)), speed, 1.5)
    assert s.slowdown_p90 == pytest.approx(1.0)      # as slow as it is alone
    assert s.raw_p90 == pytest.approx(1.5)           # vs one stream: kept
    assert s.passed


def test_the_cap_is_reported_with_its_share():
    tr = AgentTrace("pi", ((TraceRequest(900, 50, None, "start"),
                            TraceRequest(2_000, 100, 1.0, "tool"),
                            TraceRequest(960, 50, 1.0, "tool"),
                            TraceRequest(100, 5, 1.0, "tool")),))
    cp = B.capped_requests(tr, 1_000)
    assert cp["requests"] == 2 and cp["share"] == 0.5     # 2000 and 960+50


def test_the_speed_line_names_the_cap_not_40k():
    assert "at 32k context" in B._rates(B.Speed(a=1.0), cap=32_768)
    assert "at 40k context" in B._rates(B.Speed(a=1.0), cap=200_000)


def test_tree_calibration_replays_a_whole_tree_from_its_first_spawn(
        tmp_path, monkeypatch):
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(_tree_trace().to_dict()))
    load_trace.cache_clear()
    tr = load_trace(str(path))
    plan = B.Plan(str(path), context_cap_tokens=100_000)
    monkeypatch.setattr(B, "CALIBRATE_TREE_S", 5.0)

    async def go():
        async with httpx.AsyncClient(
                transport=httpx.MockTransport(_server_stub())) as c:
            return await B.calibrate_trees(
                c, B.EndpointSpec(base_url="http://fake/v1", model="m"),
                plan, tr, "p" * 100, "n")
    rows = asyncio.run(go())
    kids = [r for r in rows if r.child]
    assert len(kids) == 3                    # the child recording, once
    assert [r.prompt for r in rows if not r.child][0] == 1_200   # request 1


# ---- a step measures until it can say something -----------------------------
def test_a_step_with_too_few_turns_is_inconclusive_not_a_pass():
    rows = [_row_(k, 0, 1.0) for k in range(4)]
    s = B.judge(B.Step(4, rows, window=(0.0, 10.0)), B.Speed(a=1.0), 1.5,
                min_turns=30)
    assert s.inconclusive and not s.passed
    assert "inconclusive: 4 turns judged" in s.reasons[0]
    # a clear fail stays a fail
    slow = [_row_(k, 0, 9.0) for k in range(4)]
    f = B.judge(B.Step(4, slow, window=(0.0, 10.0)), B.Speed(a=1.0), 1.5,
                min_turns=30)
    assert not f.inconclusive and not f.passed
    # enough turns: a pass
    many = [_row_(k, 0, 1.0) for k in range(30)]
    assert B.judge(B.Step(4, many, window=(0.0, 10.0)), B.Speed(a=1.0), 1.5,
                   min_turns=30).passed


def test_the_search_treats_inconclusive_as_a_fail_and_the_headline_says_so():
    rows = [_row_(k, 0, 1.0) for k in range(4)]
    st = B.judge(B.Step(4, rows, window=(0.0, 10.0)), B.Speed(a=1.0), 1.5,
                 min_turns=30)
    assert B.answer({2: True, 4: st.passed}, 64) == {
        "serves": 2, "fails_at": 4, "capped": False}
    rec = {"steps": [st.to_dict()], "plan": {"max_slowdown": 1.5},
           "answer": {"serves": 2, "fails_at": 4, "capped": False}}
    assert "INCONCLUSIVE" in B.headline(rec)


def test_the_p90_has_a_bootstrap_interval():
    lo, hi = B.bootstrap_p90([1.0 + 0.1 * k for k in range(40)])
    assert lo < B.PERCENTILE and 1.0 < lo < hi <= 4.9
    assert B.bootstrap_p90([1.0])[0] != B.bootstrap_p90([1.0])[0]   # nan


def test_a_step_measures_past_the_window_until_it_has_the_turns(tmp_path):
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(_tree_trace().to_dict()))
    load_trace.cache_clear()
    tr = load_trace(str(path))
    plan = B.Plan(str(path), ramp_s=0.05, measure_s=0.3, min_turns=10 ** 6,
                  max_measure_s=1.2, context_cap_tokens=100_000)

    async def go():
        async with httpx.AsyncClient(
                transport=httpx.MockTransport(_server_stub())) as c:
            return await B.run_step(
                c, B.EndpointSpec(base_url="http://fake/v1", model="m"),
                plan, tr, 1, "p" * 100, "n", speed=B.Speed(a=0.01))
    step = asyncio.run(go())
    w0, w1 = step.window
    assert 1.1 <= w1 - w0 <= 1.5              # stopped at the cap, not 0.3
