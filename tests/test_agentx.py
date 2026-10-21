"""AgentX traces: the compact reader keeps the prefix structure, the cache
simulator respects its bounds."""
from __future__ import annotations

import json

import pytest

from workingset import agentx as A
from workingset.agentx_sim import Replay, Server, simulate


def _req(t, ids, out=100, typ="n", api=2.0):
    return {"t": t, "type": typ, "model": "m", "in": len(ids) * 64 + 10, "out": out,
            "hash_ids": ids, "api_time": api, "stop": "tool_use"}


def _trace(tid="a"):
    return {"id": tid, "models": ["m"], "block_size": 64, "hash_id_scope": "local",
            "requests": [
                _req(0, [1, 2, 3]), _req(5, [1, 2, 3, 4, 5]),
                {"t": 6, "type": "subagent", "agent_id": "agent_001",
                 "subagent_type": "Explore", "status": "completed", "models": ["m"],
                 "requests": [_req(6, [1, 2, 9]), _req(8, [1, 2, 9, 10])]},
                _req(10, [1, 2, 3, 4, 5, 6]),
                _req(12, [1, 2, 3, 7])]}       # pull-back: shares 3 blocks


def test_compact_keeps_prefix_structure(tmp_path):
    src = tmp_path / "t.jsonl"
    src.write_text(json.dumps(_trace()) + "\n")
    out = tmp_path / "c.json.gz"
    assert A.compact_jsonl(src, out) == {"traces": 1, "requests": 6}
    (tr,) = A.load(out)
    root, sa = tr.streams
    assert [(r.lcp_prev, r.shared, r.nblocks) for r in root.requests] == \
        [(0, 0, 3), (3, 3, 5), (5, 5, 6), (3, 3, 4)]
    # the sub-agent's first request shares the parent's first two blocks,
    # which only the trace-wide trie sees
    assert [(r.lcp_prev, r.shared) for r in sa.requests] == [(0, 2), (3, 3)]
    assert sa.parent_turn == 2 and sa.spawn_t == 6
    assert A.load(src)[0].streams[0].requests == root.requests   # jsonl path agrees


def test_compact_rejects_global_scope():
    tr = _trace()
    tr["hash_id_scope"] = "global"
    with pytest.raises(ValueError):
        A.compact_trace(tr)


def _growing_traces(n=40, turns=60):
    """Sessions whose every request extends the previous by 20 blocks."""
    out = []
    for i in range(n):
        reqs, t, blocks = [], 0.0, 1000
        for k in range(turns):
            reqs.append(A.Request(t, blocks * 64, 200, 4.0, None,
                                  blocks - 20 if k else 0, blocks - 20 if k else 0,
                                  blocks, 1))
            blocks += 20
            t += 10.0
        out.append(A.Trace(f"s{i}", 64, [A.Stream("root", reqs)]))
    return out


def test_unbounded_cache_reaches_the_ceiling():
    r = simulate(_growing_traces(), Server(hbm_tokens=1e12),
                 Replay(concurrency=16, duration_s=600)).to_dict()
    assert r["requests"] > 0
    assert r["hbm_hit_rate"] == pytest.approx(r["ceiling_hit_rate"])
    assert r["dram_hit_rate"] == 0


def test_hit_rate_is_monotone_in_capacity_and_bounded():
    traces = _growing_traces()
    rates = []
    for cap in (2e6, 8e6, 32e6, 1e9):
        d = simulate(traces, Server(hbm_tokens=cap, ranks=4),
                     Replay(concurrency=32, duration_s=900)).to_dict()
        assert 0 <= d["hbm_hit_rate"] <= d["ceiling_hit_rate"] + 1e-12
        rates.append(d["hbm_hit_rate"])
    assert rates == sorted(rates)


def test_dram_tier_recovers_what_hbm_evicts():
    traces = _growing_traces()
    small = Server(hbm_tokens=4e6, ranks=4)
    no = simulate(traces, small, Replay(concurrency=32, duration_s=900)).to_dict()
    yes = simulate(traces, Server(hbm_tokens=4e6, ranks=4, dram_tokens=1e9),
                   Replay(concurrency=32, duration_s=900)).to_dict()
    assert yes["dram_hit_rate"] > 0
    assert yes["overall_hit_rate"] > no["overall_hit_rate"]
    assert yes["overall_hit_rate"] <= yes["ceiling_hit_rate"] + 1e-12


def test_retention_resumes_only_at_kept_points():
    from workingset.agentx_sim import _usable
    # append-only turn, prefix intact: resume at the previous prompt's end
    assert _usable(100_000, 100_000, 100_000, 32_768) == 100_000
    # pull-back to 90k: only the 65,536 checkpoint is kept
    assert _usable(90_000, 100_000, 100_000, 32_768) == 65_536
    # tail trimmed to 80k by eviction: the end is gone, back to 65,536
    assert _usable(100_000, 80_000, 100_000, 32_768) == 65_536
    # no retention policy: every block counts
    assert _usable(90_000, 100_000, 100_000, None) == 90_000
