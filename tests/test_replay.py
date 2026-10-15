"""Agent traces: extraction from pi sessions and their replay by `ws test`."""
from __future__ import annotations

import asyncio
import json
import random

from workingset.agent_trace import (AgentTrace, TraceRequest,
                                    equivalent_workload, extract_pi,
                                    extract_pi_session)
from workingset.probe import build_prefixes, run_population
from workingset.probe.replay import ReplaySession, load_trace
from workingset.probe.request import EndpointSpec

from test_probe import client_for, fake_server, small_cfg, small_opts

SECRET = "internal-billing-db.corp.example"


def _pi_lines() -> list[str]:
    """A pi session: one prompt, two tool rounds (bash, then an MCP tool),
    a final answer, the person's next message after 40 s, one more answer.
    Request starts are the message's ms timestamp, ends the entry's ISO one."""
    def msg(ts_iso, message):
        return json.dumps({"type": "message", "timestamp": ts_iso,
                           "message": message})

    def asst(start_ms, end_iso, p, cached, out, tools=()):
        return msg(end_iso, {
            "role": "assistant", "timestamp": start_ms,
            "content": [{"type": "thinking", "thinking": SECRET}]
            + [{"type": "toolCall", "name": t, "arguments": {"q": SECRET}}
               for t in tools],
            "usage": {"input": p, "cacheRead": cached, "cacheWrite": 0,
                      "output": out}})

    t0 = 1_800_000_000_000                       # ms
    iso = lambda ms: (__import__("datetime").datetime.fromtimestamp(
        ms / 1000, __import__("datetime").timezone.utc)
        .isoformat().replace("+00:00", "Z"))
    return [
        json.dumps({"type": "session", "cwd": SECRET}),
        msg(iso(t0), {"role": "user", "content": [{"type": "text",
                                                    "text": SECRET}]}),
        asst(t0 + 100, iso(t0 + 4_100), 1_000, 9_000, 300, ["bash"]),
        msg(iso(t0 + 6_100), {"role": "toolResult", "toolName": "bash",
                              "content": [{"type": "text", "text": SECRET}]}),
        asst(t0 + 6_200, iso(t0 + 9_200), 500, 10_300, 200,
             ["mcp__corp__billing"]),
        msg(iso(t0 + 9_700), {"role": "toolResult",
                              "toolName": "mcp__corp__billing",
                              "content": [{"type": "text", "text": SECRET}]}),
        asst(t0 + 9_800, iso(t0 + 12_800), 400, 10_800, 100),
        # an aborted request: no usage, dropped
        msg(iso(t0 + 13_000), {"role": "assistant", "timestamp": t0 + 12_900,
                               "content": [], "usage": {}}),
        msg(iso(t0 + 52_800), {"role": "user", "content": [{"type": "text",
                                                             "text": SECRET}]}),
        asst(t0 + 52_900, iso(t0 + 55_900), 600, 11_300, 150),
    ]


# ============================================================================
# extraction
# ============================================================================
def test_extract_pi_keeps_sizes_waits_and_tool_kinds():
    reqs = extract_pi_session(_pi_lines())
    assert [r.prompt_tokens for r in reqs] == [10_000, 10_800, 11_200, 11_900]
    assert [r.output_tokens for r in reqs] == [300, 200, 100, 150]
    assert [r.after for r in reqs] == ["start", "tool", "tool", "user"]
    # the wait runs from the previous reply's end to this request's start
    assert [r.gap_s for r in reqs] == [None, 2.1, 0.6, 40.1]
    # builtin names kept; an MCP tool can name an internal service
    assert reqs[1].tools == ("bash",) and reqs[2].tools == ("other",)


def test_a_trace_file_holds_no_session_text(tmp_path):
    f = tmp_path / "s.jsonl"
    f.write_text("\n".join(_pi_lines()) + "\n")
    blob = json.dumps(extract_pi([tmp_path]).to_dict())
    assert SECRET not in blob and "corp" not in blob
    assert AgentTrace.from_dict(json.loads(blob)).n_requests == 4


def test_equivalent_workload_has_the_trace_averages():
    s = (TraceRequest(10_000, 300, None, "start"),
         TraceRequest(10_800, 200, 2.0, "tool"),
         TraceRequest(11_200, 100, 1.0, "tool"),
         TraceRequest(4_000, 100, 5_000.0, "user"))   # shrank; parked wait
    wl = equivalent_workload(AgentTrace("pi", (s,)), park_s=1800)
    assert wl["think_time_s"] == round((2 + 1 + 1800) / 3, 1)
    # growth beyond the previous reply, over the requests that grew
    assert wl["warm_turn_tokens"] == round((500 + 200) / 2)
    assert wl["max_output_tokens"] == 175
    assert wl["miss_rate"] == 0.0 and wl["subagent_ratio"] == 0.0


# ============================================================================
# replay
# ============================================================================
def _session(reqs, cap=50_000, max_gap=1800.0):
    trace = AgentTrace("pi", (tuple(reqs),))
    pre = build_prefixes(small_cfg(system_prefix_tokens=80).workload, 4.0)
    s = ReplaySession(trace, uid=1, prefix_text=pre.user, prefix_tokens=80,
                      cpt=4.0, cap_tokens=cap, max_gap_s=max_gap,
                      rng=random.Random(0), text_rng=random.Random(1))
    s._s, s._pos = 0, 0                       # start at the recording's start
    return s


def test_replay_extends_the_cached_prompt_to_the_recorded_size():
    s = _session([TraceRequest(400, 50, None, "start"),
                  TraceRequest(600, 30, 2.0, "tool"),
                  TraceRequest(700, 10, 9_000.0, "user")], max_gap=60)
    p1, k1 = s.next_turn()
    assert k1 == "first" and abs(len(p1) / 4 - 400) < 5
    assert s.output_tokens(None) == 50
    reply = "r" * 200                          # ~50 tokens the server sent
    s.commit(reply)
    assert s.think_s(None) == 2.0
    p2, k2 = s.next_turn()
    # the previous prompt and the reply, unchanged, then the new text
    assert k2 == "hit" and p2.startswith(p1 + reply)
    assert abs(len(p2) / 4 - 600) < 5
    s.commit("r" * 120)
    assert s.think_s(None) == 60               # a parked wait is capped


def test_a_shrinking_prompt_restarts_from_the_shared_prefix():
    s = _session([TraceRequest(400, 50, None, "start"),
                  TraceRequest(200, 30, 1.0, "user")])
    p1, _ = s.next_turn()
    s.commit("r" * 200)
    p2, kind = s.next_turn()
    assert kind == "miss" and not p2.startswith(p1)
    assert p2[:320] == p1[:320]                # the 80-token prefix is shared


def test_replay_moves_on_to_another_recording_from_its_start():
    s = _session([TraceRequest(400, 50, None, "start")])
    s.next_turn()
    s.commit("r")
    _, kind = s.next_turn()
    assert kind == "first" and s._pos == 0


def test_a_replayed_rung_sends_the_recorded_sizes(tmp_path):
    reqs = tuple(TraceRequest(300 + 40 * i, 5 + i, 0.01, "tool")
                 for i in range(6))
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(AgentTrace("pi", (reqs,)).to_dict()))
    load_trace.cache_clear()
    cfg = small_cfg(subagent_ratio=0.0)
    # sessions stagger over at least 1 s, so the window must outlast it
    opts = small_opts(replay=str(path), ramp_s=0.05, measure_s=1.3)
    seen: list = []

    async def go():
        async with client_for(fake_server(n_tokens=20, seen=seen)) as c:
            ep = EndpointSpec(base_url="http://fake/v1", model="m")
            return await run_population(c, ep, cfg, opts, 2,
                                        build_prefixes(cfg.workload, 4.0))
    asyncio.run(go())
    by_out = {r.output_tokens: r.prompt_tokens for r in reqs}
    assert seen and all(b["max_tokens"] in by_out for b in seen)
    # each prompt is within a few tokens of the recorded size it replays
    assert all(abs(len(b["prompt"]) / 4 - by_out[b["max_tokens"]]) < 10
               for b in seen)
