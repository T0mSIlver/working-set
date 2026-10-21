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


def _session(n=6, gap=10.0, blocks=4):
    """One root stream of `n` small requests, `gap` seconds apart."""
    reqs = [A.Request(k * gap, blocks * 64, 0, 1.0, None, blocks if k else 0,
                      blocks if k else 0, blocks, 1) for k in range(n)]
    return [A.Trace("one", 64, [A.Stream("root", reqs)])]


def _one_lane(**kw):
    return Replay(concurrency=1, start_min=0.0, start_max=0.0, **kw)


def test_warmup_removes_gaps_and_is_not_measured():
    srv = Server(ranks=1, hbm_tokens=1e9, prefill_tps=1e9, decode_tps=1e9)
    d = simulate(_session(n=10), srv, _one_lane(warmup_requests_per_lane=3,
                                                duration_s=25.0)).to_dict()
    # three warmup turns ran back to back, not 10 s apart
    assert 0 < d["window_open_s"] < 10.0
    # the fourth turn waits its recorded 9 s, the fifth 9 s more; the sixth
    # falls past the 25 s window
    assert d["requests"] == 2
    cold = simulate(_session(n=10), srv, _one_lane(warmup_requests_per_lane=0,
                                                   duration_s=25.0)).to_dict()
    assert cold["window_open_s"] == 0 and cold["requests"] == 3


def test_lanes_recycle_when_their_session_drains():
    srv = Server(ranks=1, hbm_tokens=1e9, prefill_tps=1e9, decode_tps=1e9)
    d = simulate(_session(n=3), srv, _one_lane(warmup_requests_per_lane=0,
                                               duration_s=200.0)).to_dict()
    assert d["lanes_recycled"] >= 5
    # a recycled session starts cold at turn 0 (cache-busted first turn)
    assert d["hbm_hit_rate"] < d["ceiling_hit_rate"] + 1e-12


def test_sub_agent_spawns_on_its_turn_and_parent_joins_it(monkeypatch):
    from workingset import agentx_sim as S
    order = []
    real_pin = S._Cache.pin

    def pin(self, s, keep, total):
        order.append(s.rsplit(":", 1)[-1])
        return real_pin(self, s, keep, total)

    monkeypatch.setattr(S._Cache, "pin", pin)
    tr = A.from_compact(A.compact_trace(_trace()))
    simulate([tr], Server(ranks=1, hbm_tokens=1e9, dram_tokens=0.0),
             _one_lane(warmup_requests_per_lane=0, duration_s=12.0))
    # root turns 0-1, then the sub-agent (spawned after turn 2 was sent),
    # and the root turn recorded after the sub-agent ended waits for it
    first = order[:6]
    assert first[:2] == ["root", "root"]
    assert first.index("agent_001") < first.index("root", 2)
    sa_last = max(i for i, k in enumerate(first) if k == "agent_001")
    assert first.count("agent_001") == 2 and sa_last < 4


def test_cache_never_holds_more_than_its_capacity(monkeypatch):
    from workingset import agentx_sim as S
    worst = [0.0]
    for name in ("pin", "unpin", "store"):
        real = getattr(S._Cache, name)

        def wrapped(self, *a, _real=real):
            out = _real(self, *a)
            worst[0] = max(worst[0], self.used - self.cap)
            assert self.used >= -1e-6
            return out

        monkeypatch.setattr(S._Cache, name, wrapped)
    simulate(_growing_traces(), Server(hbm_tokens=4e6, ranks=4, dram_tokens=8e6),
             Replay(concurrency=32, duration_s=900))
    assert worst[0] <= 1e-6


def test_block_size_comes_from_the_trace():
    def session(bs, nblocks):
        reqs = [A.Request(k * 10.0, 512, 0, 1.0, None, nblocks if k else 0,
                          nblocks if k else 0, nblocks, 1) for k in range(4)]
        return [A.Trace("one", bs, [A.Stream("root", reqs)])]

    srv = Server(ranks=1, hbm_tokens=1e9, block=64, retention=None)
    rp = _one_lane(warmup_requests_per_lane=0, duration_s=60.0)
    # 4 blocks of 128 tokens share as much as 8 blocks of 64
    a = simulate(session(128, 4), srv, rp).to_dict()
    b = simulate(session(64, 8), srv, rp).to_dict()
    assert a["ceiling_hit_rate"] == b["ceiling_hit_rate"] > 0.5
    assert a["hbm_hit_rate"] == b["hbm_hit_rate"]
