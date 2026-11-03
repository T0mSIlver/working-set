"""`ws trace claude`: Claude Code session files -> the trace `ws replay` plays.
The fixture is synthetic; SECRET stands for text that must never leave."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from workingset import agent_trace
from workingset.agent_trace import (AgentTrace, KINDS, extract_claude,
                                    extract_claude_session, summarize)

SECRET = "sk-secret-do-not-keep"
T0 = 1_700_000_000.0


def iso(t):
    return datetime.fromtimestamp(T0 + t, timezone.utc).isoformat().replace("+00:00", "Z")


def human(t, text=SECRET, entry="claude-desktop"):
    return {"type": "user", "timestamp": iso(t), "entrypoint": entry,
            "origin": {"kind": "human"}, "promptSource": "typed",
            "message": {"role": "user", "content": text}}


def notification(t):
    return {"type": "user", "timestamp": iso(t), "promptSource": "system",
            "origin": {"kind": "task-notification"},
            "message": {"role": "user", "content": "<task-notification>" + SECRET}}


def result(t, *ids):
    return {"type": "user", "timestamp": iso(t),
            "toolUseResult": {"stdout": SECRET},
            "message": {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": i, "content": SECRET}
                for i in ids]}}


def block(t, mid, blocks, out, inp=2, cr=0, cw=0, model="claude-opus-5-5"):
    return {"type": "assistant", "timestamp": iso(t), "requestId": "req_" + mid,
            "message": {"id": mid, "model": model, "role": "assistant",
                        "content": blocks,
                        "usage": {"input_tokens": inp, "output_tokens": out,
                                  "cache_read_input_tokens": cr,
                                  "cache_creation_input_tokens": cw}}}


def use(i, name, **inp):
    return {"type": "tool_use", "id": i, "name": name, "input": {"x": SECRET, **inp}}


TEXT = {"type": "text", "text": SECRET}


def main_lines(entry="claude-desktop"):
    return [
        human(0, entry=entry),
        # one response in two lines: the first carries a partial output count
        block(3, "m1", [{"type": "thinking", "thinking": SECRET}], 10, 2, 5000, 1000),
        block(4, "m1", [use("tA", "Task"), use("tB", "Bash")], 120, 2, 5000, 1000),
        result(4.2, "tA"),
        result(6, "tB"),
        block(8, "m2", [use("tC", "mcp__internal__lookup")], 60, 2, 6500, 100),
        result(9.1, "tC"),
        block(12, "m3", [use("tD", "Bash")], 40, 2, 6600, 50),
        result(15, "tD"),
        block(18, "m4", [TEXT], 300, 2, 6650, 10),
        notification(100),
        block(103, "m5", [TEXT], 80, 2, 6700, 200),
        human(400),
        block(405, "m6", [TEXT], 90, 2, 6900, 400),
        # an API error: not replayed
        {"type": "assistant", "timestamp": iso(500),
         "message": {"id": "e1", "model": "<synthetic>", "content": [TEXT],
                     "usage": {"input_tokens": 0, "output_tokens": 0}}},
        human(520),
        block(526, "m7", [TEXT], 70, 2, 7300, 100),
    ]


def sub_lines():
    return [human(5), block(7, "s1", [use("uA", "Read")], 30, 2, 800, 200),
            result(7.5, "uA"), block(9, "s2", [TEXT], 50, 2, 1000, 20)]


def write(tmp, entry="claude-desktop", sub=True):
    proj = tmp / "proj"
    proj.mkdir(exist_ok=True)
    f = proj / "11111111.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in main_lines(entry)) + "\n")
    if sub:
        d = proj / "11111111" / "subagents"
        d.mkdir(parents=True, exist_ok=True)
        (d / "agent-a1.jsonl").write_text(
            "\n".join(json.dumps(r) for r in sub_lines()) + "\n")
        (d / "agent-a1.meta.json").write_text(json.dumps({"toolUseId": "tA"}))
    return proj


def test_blocks_merge_into_one_request_with_the_largest_output():
    s = extract_claude_session(json.dumps(r) for r in main_lines())
    reqs = s.requests
    assert [r.output_tokens for r in reqs] == [120, 60, 40, 300, 80, 90, 70]
    assert reqs[0].prompt_tokens == 2 + 5000 + 1000      # input + read + write


def test_gaps_kinds_and_service_times():
    r = extract_claude_session(json.dumps(x) for x in main_lines()).requests
    assert [x.kind for x in r] == ["start", "tool", "tool", "tool",
                                   "background", "think", "think"]
    assert [x.after for x in r] == ["start", "tool", "tool", "tool", "tool",
                                    "user", "user"]
    assert r[0].gap_s is None and r[0].service_s == 4.0   # human at 0, last block 4
    assert (r[1].gap_s, r[1].service_s) == (2.0, 2.0)     # 4 -> tool result 6 -> 8
    assert r[1].tools == ("bash", "task")
    assert r[2].tools == ("other",)          # an MCP tool is not named
    assert r[4].gap_s == 82.0 and r[4].service_s == 3.0    # notification at 100
    assert r[5].gap_s == 297.0               # the person at 400 after m5 at 103
    assert r[6].gap_s == 115.0 and r[6].kind == "think"    # the person's wait
    assert "background" in KINDS


def test_compaction_and_interrupt_mark_the_gap():
    lines = main_lines()
    lines.insert(11, {"type": "system", "subtype": "compact_boundary",
                      "timestamp": iso(99)})
    r = extract_claude_session(json.dumps(x) for x in lines).requests
    assert r[4].kind == "compaction"


def test_subagent_is_its_own_session_linked_to_its_parent(tmp_path):
    trace = extract_claude([write(tmp_path)])
    assert len(trace.sessions) == 2
    link = trace.links[1]
    assert (link.parent, link.start_offset_s, link.spawn_request) == (0, 5.0, 0)
    assert trace.links[0] is None
    back = AgentTrace.from_dict(json.loads(json.dumps(trace.to_dict())))
    assert back.links == trace.links
    assert summarize(trace)["subagent_sessions"] == 1
    # a subagent's start is not a message from the person
    assert summarize(trace)["requests_per_user_message"] == 9 / 3


def test_headless_is_left_out_unless_asked(tmp_path):
    proj = write(tmp_path, entry="sdk-cli")
    try:
        extract_claude([proj])
    except ValueError as e:
        assert "no Claude Code request" in str(e)
    else:
        raise AssertionError("headless session kept")
    kept = extract_claude([proj], include_headless=True)
    assert len(kept.sessions) == 2


def test_no_text_no_path_no_absolute_time_leaves_the_file(tmp_path):
    trace = extract_claude([write(tmp_path)])
    blob = json.dumps(trace.to_dict())
    assert SECRET not in blob and "mcp__internal" not in blob
    assert str(tmp_path) not in blob and "11111111" not in blob
    assert str(int(T0))[:6] not in blob
    assert trace.notes["permission_waits"].startswith("no log")


def test_old_traces_load_and_the_cli_writes_a_trace(tmp_path, capsys):
    proj = write(tmp_path)
    out = tmp_path / "t.json"
    assert agent_trace.main(["claude", str(proj), "-o", str(out)]) == 0
    t = agent_trace.load(out)
    assert t.n_requests == 9 and t.source == "claude"
    d = json.loads(out.read_text())
    assert "parent" not in d["sessions"][0] and d["sessions"][1]["parent"] == 0
    for s in d["sessions"]:           # links are optional: version 2 readers
        s.pop("parent", None), s.pop("start_offset_s", None), s.pop("spawn_request", None)
    assert AgentTrace.from_dict(d).links == ()
    assert "subagents" in capsys.readouterr().out


def test_an_api_error_marks_a_tool_gap_but_not_the_persons_wait():
    err = {"type": "assistant", "timestamp": iso(8),
           "message": {"id": "e", "model": "<synthetic>", "content": [TEXT],
                       "usage": {"input_tokens": 0, "output_tokens": 0}}}
    lines = [human(0), block(2, "a", [use("t1", "Bash")], 20, 2, 900, 100),
             err, result(30, "t1"), block(33, "b", [TEXT], 20, 2, 1000, 10)]
    r = extract_claude_session(json.dumps(x) for x in lines).requests
    assert [x.kind for x in r] == ["start", "error"] and r[1].gap_s == 28.0
