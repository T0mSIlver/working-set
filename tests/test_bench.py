"""`ws replay`: the endpoint alone, the step search, the turn verdict, and a
run against fake endpoints with and without contention."""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from workingset import bench as B
from workingset.agent_trace import AgentTrace, TraceRequest
from workingset.cli import main
from workingset.probe.replay import load_trace

from test_probe import _sse


# ============================================================================
# the endpoint alone
# ============================================================================
def _row(u, prompt, out, ttft, service, seat=0, turn=0, sent=1.0, **kw):
    return B.Row(seat, turn, "hit", u, prompt, out, sent, ttft=ttft,
                 service=service, done=True, **kw)


def test_fit_recovers_the_endpoints_speed():
    truth = B.Speed(a=0.05, b=1 / 8000, c=2e-11, d=0.02, e=1e-7)
    rows = [_row(u, ctx, out, truth.ttft(u, ctx), truth.service(u, ctx, out))
            for u, ctx, out in ((500, 10_000, 200), (2_000, 30_000, 50),
                                (40_000, 40_000, 400), (1_000, 80_000, 800),
                                (90_000, 90_000, 100), (300, 120_000, 300))]
    got = B.fit_speed(rows)
    for k in "abcde":
        assert getattr(got, k) == pytest.approx(getattr(truth, k), rel=1e-6)
    assert got.n == 6


def test_the_fit_never_goes_negative():
    # a prefill that looks faster on longer prompts (noise): the coefficient
    # is dropped, not made negative
    rows = [_row(u, u, 10, t, t + 0.1) for u, t in
            ((1_000, 0.50), (2_000, 0.48), (4_000, 0.52), (8_000, 0.49))]
    got = B.fit_speed(rows)
    assert min(got.a, got.b, got.c, got.d, got.e) >= 0
    assert got.a == pytest.approx(0.4975, abs=1e-3)


# ============================================================================
# the search
# ============================================================================
def _search(knee: int, cap: int = 256) -> tuple[list[int], dict]:
    seen, results = [], {}
    while (n := B.next_sessions(results, cap)) is not None:
        seen.append(n)
        results[n] = n <= knee
    return seen, B.answer(results, cap)


def test_steps_double_then_halve_to_ten_percent():
    seen, a = _search(40)
    assert seen == [1, 2, 4, 8, 16, 32, 64, 48, 40, 44]
    assert a == {"serves": 40, "fails_at": 44, "capped": False}


def test_a_small_knee_is_found_exactly():
    assert _search(5)[1] == {"serves": 5, "fails_at": 6, "capped": False}
    assert _search(1)[1] == {"serves": 1, "fails_at": 2, "capped": False}


def test_one_failing_session_stops_at_once():
    seen, a = _search(0)
    assert seen == [1] and a["serves"] == 0 and a["fails_at"] == 1


def test_the_cap_is_reported_as_a_floor():
    seen, a = _search(10_000, cap=12)
    assert seen == [1, 2, 4, 8, 12]
    assert a == {"serves": 12, "fails_at": None, "capped": True}


# ============================================================================
# the verdict
# ============================================================================
FLAT = B.Speed(a=1.0)              # alone, every request takes 1 s


def test_a_turn_is_scored_by_its_total_time():
    # seat 0, turn 0: 1 s + 3 s = 2x; seat 1, turn 0: 1 s = 1x
    rows = [_row(1, 1, 1, 1.0, 1.0), _row(1, 1, 1, 3.0, 3.0),
            _row(1, 1, 1, 1.0, 1.0, seat=1)]
    s = B.judge(B.Step(2, rows, window=(0.0, 10.0)), FLAT, 1.5)
    assert s.turns == 2 and s.requests == 3
    assert s.slowdown_mean == pytest.approx(5 / 3)
    assert s.slowdown_p90 == pytest.approx(1.9)
    assert not s.passed and "p90 turn 1.90x" in s.reasons[0]


def test_a_request_in_flight_during_the_window_counts_wherever_it_was_sent():
    """A request sent in the ramp and finishing in the window is the slowest
    one an overloaded server has; dropping it for its send time let an
    overloaded step pass on the few requests that happened to start later."""
    ramp_slow = _row(1, 1, 1, 9.0, 9.0, sent=0.5, seat=1)      # ends at 9.5
    ramp_done = _row(1, 1, 1, 0.2, 0.2, sent=0.1, seat=2)      # ends at 0.3
    fast = _row(1, 1, 1, 1.0, 1.0)
    s = B.judge(B.Step(2, [ramp_slow, ramp_done, fast],
                       window=(1.0, 10.0)), FLAT, 1.5)
    assert s.requests == 2 and s.turns == 2
    assert s.slowdown_p90 > 1.5 and not s.passed


def test_requests_left_running_fail_the_step_and_are_censored():
    ok = [_row(1, 1, 1, 1.0, 1.0, turn=k, sent=2.0 + k) for k in range(5)]
    stuck = B.Row(9, 0, "hit", 1, 1, 1, sent=1.0)      # never finished
    s = B.judge(B.Step(1, ok + [stuck], window=(0.0, 10.0), drain_s=5.0),
                FLAT, 1.5)
    assert s.unfinished == 1 and not s.passed
    assert any("1 of 6 failed" in r and "still running" in r
               for r in s.reasons)
    assert s.slowdown_p90 > 5            # the stuck one counts 14x, not nothing


def test_a_step_with_few_turns_for_its_sessions_fails():
    rows = [_row(1, 1, 1, 1.0, 1.0, seat=k) for k in range(2)]
    s = B.judge(B.Step(8, rows, window=(0.0, 10.0)), FLAT, 1.5)
    assert not s.passed
    assert "only 2 turns judged for 8 session(s)" in s.reasons[0]


def test_the_mean_requests_in_flight_over_the_window():
    rows = [_row(1, 1, 1, 5.0, 5.0, sent=2.0, seat=0),     # 2..7
            _row(1, 1, 1, 10.0, 10.0, sent=0.0, seat=1)]   # 0..10
    s = B.judge(B.Step(2, rows, window=(0.0, 10.0)), FLAT, 99)
    assert s.streams_mean == pytest.approx(1.5)


def test_a_request_still_running_counts_with_its_time_so_far():
    # sent at 4 s into a window ending at 10 s, still running after a 6 s
    # drain: at least 12 s, against 1 s alone
    running = B.Row(0, 0, "hit", 1, 1, 1, sent=4.0)
    s = B.judge(B.Step(1, [running], window=(0.0, 10.0), drain_s=6.0),
                FLAT, 1.5)
    assert s.slowdown_p90 == pytest.approx(12.0) and not s.passed


def test_failed_requests_fail_the_step_past_one_percent():
    ok = [_row(1, 1, 1, 1.0, 1.0, turn=k) for k in range(10)]
    bad = B.Row(0, 99, "hit", 1, 1, 1, sent=1.0, error="HTTP 500", done=True)
    s = B.judge(B.Step(1, ok + [bad], window=(0.0, 10.0)), FLAT, 1.5)
    assert s.errors == 1 and not s.passed
    assert s.reasons == ["failed requests: 1 of 11 failed (first: HTTP 500)"]


def test_a_step_where_every_request_failed_says_why():
    bad = [B.Row(0, 0, "hit", 1, 1, 1, sent=1.0, error="HTTP 400: too long",
                 done=True) for _ in range(3)]
    s = B.judge(B.Step(1, bad, window=(0.0, 10.0)), FLAT, 1.5)
    assert not s.passed
    assert "HTTP 400: too long" in " ".join(s.reasons)


def test_calibration_stops_when_the_endpoint_rejects_the_requests(
        tmp_path, monkeypatch):
    """19 of 20 calibration requests got a 400 and the fit went ahead on the
    one that passed: the speed (and every verdict) came from nothing."""
    def handler(request):
        return httpx.Response(400, json={"error": {"message": "too long"}})
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(_trace(n=30).to_dict()))
    load_trace.cache_clear()
    monkeypatch.setattr(B, "make_client", lambda: httpx.AsyncClient(
        transport=httpx.MockTransport(handler)))
    monkeypatch.setattr(B, "_endpoint_info", lambda *a: {"model": "m"})
    monkeypatch.setattr(B, "CALIBRATE_REQUESTS", 5)
    rc = main(["replay", str(path), "--base-url", "http://fake/v1"])
    assert rc == 3


def test_the_step_round_trips_through_its_record():
    rows = [_row(1, 1, 1, 1.0, 1.0), B.Row(0, 1, "hit", 1, 1, 1, sent=2.0)]
    s = B.judge(B.Step(1, rows, window=(0.0, 10.0), drain_s=1.0,
                       server={"kv_max": 0.5}), FLAT, 1.5)
    back = B.Step.from_dict(json.loads(json.dumps(s.to_dict())))
    assert back.to_dict() == s.to_dict()


# ============================================================================
# the session: uncached tokens and turns
# ============================================================================
def _trace(n=12, prompt0=1_000, grow=300, out=20, every=3, gap=0.01):
    """Each request extends the previous one by `grow` tokens; every
    `every`-th request follows a message from the person."""
    reqs = tuple(TraceRequest(prompt0 + grow * i, out, gap,
                              "user" if i and i % every == 0 else "tool")
                 for i in range(n))
    return AgentTrace("pi", (reqs,))


def test_a_seat_marks_uncached_tokens_and_turns():
    tr = _trace()
    seat = B._Seat(tr, uid=1, prefix_text="p" * 400, prefix_tokens=100,
                   cpt=4.0, cap_tokens=100_000, max_gap_s=1800.0,
                   rng=__import__("random").Random(0),
                   text_rng=__import__("random").Random(1))
    seat._s, seat._pos = 0, 0
    marks = []
    for _ in range(7):
        seat.next_turn()
        marks.append(seat.last)
        seat.commit("r" * 80)
    # first request: cold past the shared prefix, a turn starts
    assert marks[0] == (900, 1_000, 20, True)
    # then the growth since the previous prompt, reply included
    assert [m[0] for m in marks[1:]] == [300] * 6
    # a turn starts where the person wrote (requests 3 and 6)
    assert [m[3] for m in marks] == [True, False, False, True, False, False,
                                     True]


def test_the_default_shared_prefix_is_the_smallest_first_prompt():
    tr = AgentTrace("pi", ((TraceRequest(8_000, 1, None, "start"),),
                           (TraceRequest(5_000, 1, None, "start"),
                            TraceRequest(9_000, 1, 1.0, "tool"))))
    assert B.default_shared_prefix(tr) == 5_000


# ============================================================================
# a run against fake endpoints
# ============================================================================
def _endpoint(per_token=0.03, n_tokens=4, free=4, penalty=3.0):
    """Streams `n_tokens`, `per_token` apart while at most `free` requests
    are in flight and `penalty` times slower beyond: a knee sharp enough
    that the event loop's own jitter cannot move it."""
    state = {"inflight": 0, "seen": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        n = min(n_tokens, body.get("max_tokens") or n_tokens)
        state["seen"] += 1

        async def gen():
            # counted down before the trailer: httpx does not close the
            # generator once the client stops reading, so a `finally` here
            # would never run and the count would only grow
            state["inflight"] += 1
            for i in range(n):
                await asyncio.sleep(per_token * (
                    1.0 if state["inflight"] <= free else penalty))
                yield _sse({"choices": [{"text": "tok ", "index": 0}]})
            state["inflight"] -= 1
            yield _sse({"choices": [], "usage": {
                "prompt_tokens": int(len(body["prompt"]) / 4),
                "completion_tokens": n}})
            yield b"data: [DONE]\n\n"
        return httpx.Response(200, content=gen(),
                              headers={"content-type": "text/event-stream"})
    handler.state = state
    return handler


def _run(tmp_path, monkeypatch, handler, *extra):
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(_trace(n=30).to_dict()))
    load_trace.cache_clear()
    monkeypatch.setattr(B, "make_client", lambda: httpx.AsyncClient(
        transport=httpx.MockTransport(handler)))
    monkeypatch.setattr(B, "_endpoint_info", lambda *a: {"model": "m"})
    monkeypatch.setattr(B, "CALIBRATE_REQUESTS", 10)
    out = tmp_path / "run.json"
    rc = main(["replay", str(path), "--base-url", "http://fake/v1",
               "--ramp-s", "0.05", "--measure-s", "0.5", "--min-turns", "0",
               "--out", str(out),
               *extra])
    assert rc == 0
    return json.loads(out.read_text())


def test_an_endpoint_without_contention_serves_up_to_the_cap(
        tmp_path, monkeypatch, capsys):
    # a loose target: this checks the cap, not the client's own jitter on a
    # busy machine
    rec = _run(tmp_path, monkeypatch, _endpoint(free=10_000),
               "--max-sessions", "8", "--max-slowdown", "3")
    assert rec["answer"] == {"serves": 8, "fails_at": None, "capped": True}
    assert [s["sessions"] for s in rec["steps"]] == [1, 2, 4, 8]
    assert all(s["passed"] and s["turns"] for s in rec["steps"])
    assert "serves at least 8 sessions" in capsys.readouterr().out


def test_a_contended_endpoint_is_bracketed_at_its_knee(tmp_path, monkeypatch):
    # 4 sessions run at full speed, past 4 in flight everything is 6x
    # slower: a 2.5x target lands on the knee whatever the machine's load
    rec = _run(tmp_path, monkeypatch, _endpoint(free=4, penalty=6.0),
               "--max-sessions", "64", "--max-slowdown", "2.5")
    a = rec["answer"]
    # sessions wait between requests, so a few more than 4 still mostly
    # find room; 8 never do
    assert 3 <= a["serves"] <= 7 and a["fails_at"] == a["serves"] + 1
    by_n = {s["sessions"]: s for s in rec["steps"]}
    assert by_n[1]["slowdown_p90"] < 1.5
    assert by_n[8]["slowdown_p90"] > 2.5 and not by_n[8]["passed"]
    # the record holds every request, so the verdict can be re-derived
    assert sum(len(s["rows"]) for s in rec["steps"]) > 20
    assert rec["speed"]["n"] == 10


def test_ws_report_reprints_a_replay_record(tmp_path, monkeypatch, capsys):
    rec = _run(tmp_path, monkeypatch, _endpoint(free=10_000),
               "--max-sessions", "2", "--max-slowdown", "3")
    capsys.readouterr()
    assert main(["report", str(tmp_path / "run.json")]) == 0
    out = capsys.readouterr().out
    assert "REPLAY" in out and "serves at least 2 sessions" in out
    assert rec["format"] == B.FORMAT


def test_the_dry_run_sends_nothing(tmp_path, monkeypatch, capsys):
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(_trace().to_dict()))
    load_trace.cache_clear()
    monkeypatch.setattr(B, "make_client", lambda: pytest.fail("sent"))
    assert main(["replay", str(path), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "dry run: nothing sent" in out and "warm time to first token p90 <= 3 s" in out


def test_the_models_prediction_is_printed_not_used(tmp_path, monkeypatch,
                                                    capsys):
    cfg = tmp_path / "ws.toml"
    assert main(["init", "--model", "27B", "--gpu", "H200", "--tp", "4",
                 "-o", str(cfg)]) == 0
    path = tmp_path / "trace.json"
    # agent-sized: 30k prompts growing by 2k a request, 20 s waits
    path.write_text(json.dumps(_trace(prompt0=30_000, grow=2_000,
                                      out=400, gap=20.0).to_dict()))
    load_trace.cache_clear()
    assert main(["replay", str(path), "--dry-run", "--config",
                 str(cfg)]) == 0
    out = capsys.readouterr().out
    assert "printed, not used" in out
    users = out.split("the model (")[1].split("): ")[1].split(" users")[0]
    assert int(users) > 0


def test_a_config_the_model_cannot_price_does_not_stop_the_run(
        tmp_path, monkeypatch, capsys):
    cfg = tmp_path / "ws.toml"
    cfg.write_text("not toml [")
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(_trace().to_dict()))
    load_trace.cache_clear()
    assert main(["replay", str(path), "--dry-run", "--config",
                 str(cfg)]) == 0
    assert "could not price this trace" in capsys.readouterr().out


def test_the_request_timeout_reaches_the_requests(tmp_path, monkeypatch):
    rec = _run(tmp_path, monkeypatch, _endpoint(free=10_000),
               "--max-sessions", "1", "--request-timeout-s", "7")
    assert rec["plan"]["request_timeout_s"] == 7
    assert B.Plan("t", request_timeout_s=7).options().request_timeout_s == 7
