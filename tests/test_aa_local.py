"""AA AgentPerf-Local lanes: the reader's prefix accounting, the rotation,
and the simulator's bounds."""
from __future__ import annotations

import json
from dataclasses import replace

from workingset import aa_local as A


def _replay(tmp_path):
    sys_msg = {"role": "system", "content": "s" * 400}
    rows = {
        "a": [{"messages": [sys_msg, {"role": "user", "content": "u1"}],
               "recorded_prompt_tokens": 100, "target_output_tokens": 10},
              {"messages": [sys_msg, {"role": "user", "content": "u1"},
                            {"role": "assistant", "content": "x" * 40},
                            {"role": "user", "content": "u2"}],
               "recorded_prompt_tokens": 120, "target_output_tokens": 10}],
        "b": [{"messages": [sys_msg, {"role": "user", "content": "v1"}],
               "recorded_prompt_tokens": 100, "target_output_tokens": 20}],
    }
    tasks = []
    for tid, rs in rows.items():
        (tmp_path / f"{tid}.jsonl").write_text(
            "\n".join(json.dumps({**r, "simulated_tool_delay_ms_after": 0.0}) for r in rs))
        tasks.append({"task_id": tid, "trace": f"{tid}.jsonl"})
    (tmp_path / "manifest-x86_64.json").write_text(json.dumps({"tasks": tasks}))
    return tmp_path


def test_lcp_follows_the_serialized_prefix(tmp_path):
    turns = A.load(_replay(tmp_path))
    assert [t.task for t in turns] == ["a", "a", "b"]
    assert turns[0].lcp == 0
    # the second turn repeats all of the first request's messages
    assert 95 <= turns[1].lcp <= 100
    # across the task boundary only the system prompt is shared
    assert 0 < turns[2].lcp < turns[1].lcp


def test_rotation_spreads_lanes():
    order = list("abcdefgh")
    assert A.rotated(order, 0, 2) == order
    assert A.rotated(order, 1, 2)[0] == "e"
    assert [A.rotated(order, k, 8)[0] for k in range(8)] == order


def _box(**kw):
    return replace(A.BOXES["qwen38-27b-q4km-3090"], **kw)


def _lane(n_turns=6, prompt0=2000, step=500, out=50, tool_s=0.0):
    ts, p = [], prompt0
    for i in range(n_turns):
        ts.append(A.Turn("t", p, out, tool_s, 0 if i == 0 else p - step))
        p += step
    return ts


def test_more_lanes_more_throughput_slower_streams():
    one = A.simulate(_box(restore="lcp"), [_lane()])
    four = A.simulate(_box(restore="lcp"), [_lane() for _ in range(4)])
    assert four["agg_out_tps"] > one["agg_out_tps"]
    assert four["gen_tps_mean"] < one["gen_tps_mean"]
    assert one["failed_turns"] == 0 and one["evictions"] == 0


def test_hits_never_exceed_the_common_prefix():
    lane = _lane()
    d = A.simulate(_box(restore="lcp"), [lane])
    reusable = sum(min(t.lcp, t.prompt) for t in lane)
    assert d["hit_rate"] * sum(t.prompt for t in lane) <= reusable + 1e-6
    # a checkpoint restore can only lose hits against the exact prefix
    c = A.simulate(_box(restore="checkpoint"), [lane])
    assert c["hit_rate"] <= d["hit_rate"] + 1e-9


def test_a_small_pool_evicts_idle_slots():
    # tool waits of different lengths leave some slots idle while others run
    lanes = [_lane(tool_s=1.0 + 3 * k) for k in range(3)]
    small = _box(restore="lcp", pool_tokens=9000, cache_idle_slots=False, ram_tokens=0)
    d = A.simulate(small, lanes, tool_delays=True)
    assert d["evictions"] > 0
    big = A.simulate(replace(small, pool_tokens=10 ** 7), lanes, tool_delays=True)
    assert big["evictions"] == 0 and big["hit_rate"] >= d["hit_rate"]


def test_kv_exhaustion_fails_the_active_requests():
    lanes = [_lane() for _ in range(3)]          # in step: never an idle slot to clear
    d = A.simulate(_box(restore="lcp", pool_tokens=9000), lanes)
    assert d["failed_turns"] > 0
    ok = A.simulate(_box(restore="lcp", pool_tokens=10 ** 7), lanes)
    assert ok["failed_turns"] == 0 and ok["turns"] == 18


def test_idle_slots_come_back_from_ram():
    lanes = [_lane(tool_s=1.0 + 3 * k) for k in range(2)]
    d = A.simulate(_box(restore="lcp"), lanes, tool_delays=True)
    assert d["ram_restores"] > 0
    off = A.simulate(_box(restore="lcp", cache_idle_slots=False), lanes, tool_delays=True)
    assert off["ram_restores"] == 0 and off["hit_rate"] == d["hit_rate"]
