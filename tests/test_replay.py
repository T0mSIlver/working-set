"""Agent traces: extraction from pi sessions and their replay by `ws test`."""
from __future__ import annotations

import asyncio
import json
import random

import pytest

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


def test_a_torn_last_line_does_not_lose_the_session(tmp_path):
    f = tmp_path / "s.jsonl"
    f.write_text("\n".join(_pi_lines()) + '\n{"type": "mess')
    assert extract_pi([tmp_path]).n_requests == 4


def test_the_gap_cap_reaches_the_predicted_workload(tmp_path):
    from workingset.test_cmd import apply_replay
    s = tuple(TraceRequest(1_000, 10, g, "user") for g in (None,) + (2.0,) * 9
              + (7_200.0,))
    path = tmp_path / "t.json"
    path.write_text(json.dumps(AgentTrace("pi", (s,)).to_dict()))
    cfg, err = apply_replay(small_cfg(), str(path), 300.0)
    assert err is None and cfg.workload.think_time_s == round((18 + 300) / 10, 1)


# ============================================================================
# gap kinds and the gap policy
# ============================================================================
def _tree_session():
    """One pi session that meets every gap kind, in this order: tool,
    permission, ask, think, abort, error, resume, branch, compaction, idle.
    Entries chain by id/parentId as pi writes them."""
    from datetime import datetime, timezone
    rows, last = [], [None]
    t = [1_800_000_000.0]

    def iso(s):
        return (datetime.fromtimestamp(s, timezone.utc).isoformat()
                .replace("+00:00", "Z"))

    def add(row, dt=0.0, parent=...):
        t[0] += dt
        row = {"id": f"e{len(rows)}", "parentId": last[0] if parent is ...
               else parent, "timestamp": iso(t[0]), **row}
        rows.append(json.dumps(row))
        last[0] = row["id"]
        return row["id"]

    def asst(tools=(), dur=2.0, usage=True, stop=None):
        start = t[0]
        msg = {"role": "assistant", "timestamp": int(start * 1000),
               "content": [{"type": "toolCall", "id": f"c{len(rows)}{i}",
                            "name": n, "arguments": {}}
                           for i, n in enumerate(tools)],
               "usage": ({"input": 1_000 + 10 * len(rows), "cacheRead": 0,
                          "cacheWrite": 0, "output": 50} if usage else {})}
        if stop:
            msg["stopReason"] = stop
        add({"type": "message", "message": msg}, dt=dur)
        return msg["content"]

    def tool(name, dt, text="ok"):
        add({"type": "message", "message": {
            "role": "toolResult", "toolName": name,
            "content": [{"type": "text", "text": text}]}}, dt=dt)

    def user(dt):
        return add({"type": "message", "message": {
            "role": "user", "content": [{"type": "text", "text": "x"}]}},
            dt=dt)

    add({"type": "session", "cwd": "/w"}, parent=None)
    add({"type": "custom", "customType": "ext", "data": {}})
    user(1.0)
    calls = asst(["bash"])                       # 1 start
    tool("bash", 0.5)
    calls = asst(["bash"])                       # 2 tool
    tool("bash", 30.0)
    perm_id = calls[0]["id"]
    asst(["ask_user_question"])                  # 3 permission (25 s of it)
    tool("ask_user_question", 40.0)
    asst()                                       # 4 ask
    user(12.0)
    asst()                                       # 5 think
    user(1.0)
    asst(usage=False, stop="aborted", dur=3.0)   # dropped; restarts the clock
    user(9.0)
    asst()                                       # 6 abort
    user(2.0)
    asst(usage=False, stop="error", dur=1.0)     # dropped; an outage
    user(5.0)
    asst()                                       # 7 error
    add({"type": "custom", "customType": "ext", "data": {}}, dt=600.0)
    user(4.0)
    asst()                                       # 8 resume
    fork_at = last[0]
    user(3.0)
    asst()                                       # 9 think
    add({"type": "message", "message": {"role": "user", "content": []}},
        dt=7.0, parent=fork_at)                  # back up the tree
    asst()                                       # 10 branch
    add({"type": "compaction", "summary": "s"}, dt=20.0)
    user(3.0)
    asst()                                       # 11 compaction
    user(7_200.0)
    asst()                                       # 12 idle
    return rows, {perm_id: 25.0}


def test_each_gap_is_classified_from_the_session_events():
    rows, waits = _tree_session()
    counts = {}
    reqs = extract_pi_session(rows, waits=waits, counts=counts)
    assert [r.kind for r in reqs] == [
        "start", "tool", "permission", "ask", "think", "abort", "error",
        "resume", "think", "branch", "compaction", "idle"]
    perm = reqs[2]
    assert perm.human_s == 25.0 and perm.gap_s == pytest.approx(30.0, abs=0.01)
    # the abort's wait runs from the interruption, not from the last reply
    assert reqs[5].gap_s == pytest.approx(9.0, abs=0.01)
    assert counts["dropped_aborted"] == 1 and counts["dropped_error"] == 1


def test_a_denied_prompt_marks_the_gap_without_the_log():
    rows, _ = _tree_session()
    rows = [r.replace('"text": "ok"',
                      '"text": "[pi-permission-system] User denied it"', 1)
            for r in rows]
    reqs = extract_pi_session(rows)
    assert reqs[1].kind == "permission"


def test_the_gap_policy_applies_to_inactive_gaps_only():
    from workingset.agent_trace import mean_gap, replay_gap
    think = TraceRequest(1_000, 10, 20.0, "user")
    idle = TraceRequest(1_000, 10, 90_000.0, "user", kind="idle")
    tool = TraceRequest(1_000, 10, 4_000.0, "tool")
    rng, pool = random.Random(0), [20.0]
    assert replay_gap(idle, "sample", 1800.0, rng, pool) == 20.0
    assert replay_gap(idle, "cap", 1800.0, rng, pool) == 1800.0
    assert replay_gap(idle, "drop", 1800.0, rng, pool) == 0.0
    # an active gap is the recording's, whatever the policy, under the cap
    assert replay_gap(tool, "drop", 1800.0, rng, pool) == 1800.0
    assert replay_gap(think, "sample", 1800.0, rng, pool) == 20.0
    tr = AgentTrace("pi", ((think, think, idle),))
    assert mean_gap(tr, "sample") == pytest.approx(20.0)
    assert mean_gap(tr, "drop") == pytest.approx(40.0 / 3)
    assert mean_gap(tr, "cap") == pytest.approx((20.0 + 20.0 + 1800.0) / 3)


def test_a_version_1_trace_still_loads():
    d = {"format": "workingset-agent-trace", "version": 1, "source": "pi",
         "sessions": [{"requests": [
             {"prompt_tokens": 9, "output_tokens": 1, "gap_s": None,
              "after": "start"},
             {"prompt_tokens": 12, "output_tokens": 1, "gap_s": 3.0,
              "after": "user"}]}]}
    tr = AgentTrace.from_dict(d)
    assert [r.kind for r in tr.sessions[0]] == ["start", "think"]
