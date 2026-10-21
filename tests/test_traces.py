"""Public-trace replay: exact LRU, Denning, the planner and the fill rules
on small traces whose answers are known by hand."""
from __future__ import annotations

import json
import math

import pytest

from workingset import traces as T


def _cyclic(n_sessions, rounds, pages=4, bs=256, gap=10.0):
    """n sessions taking turns in a fixed order, each prompt growing by one
    page per turn and sharing nothing with other sessions."""
    reqs, t = [], 0.0
    for r in range(rounds):
        for s in range(n_sessions):
            blocks = [s * 1000 + k for k in range(pages + r)]
            reqs.append(T.Request(t, str(s), len(blocks) * bs, 10, blocks, bs))
            t += gap / n_sessions
    return reqs


def test_cyclic_lru_collapses_past_the_sessions_that_fit():
    reqs = _cyclic(4, 6)
    c = T.lru_curve(reqs, 256)
    # every request after the first round reuses its previous prompt whole
    reuse = sum(4 + r - 1 for r in range(1, 6)) * 4
    assert c.ceiling == pytest.approx(reuse * 256 / sum(r.tokens_in for r in reqs))
    big = 4 * 10 * 256                         # all four sessions' largest prompts
    assert c.hit(big) == pytest.approx(c.ceiling)
    # with less room than the other three sessions' prompts, each session is
    # evicted just before its turn comes round
    assert c.hit(12 * 256) == 0.0


def test_planner_gives_the_cliff_on_equal_gaps():
    reqs = _cyclic(4, 6)
    st = T.session_stats(reqs, 256)
    plan = T.planner_curve(st)
    ceiling = sum(st.reusable) / sum(st.prompt_tokens)
    assert plan(1e9) == pytest.approx(ceiling)
    assert plan(1.0) == 0.0
    # equal gaps: the fill rule and the planner agree it is all or nothing
    assert T.fill_hit(st, 1e9) == pytest.approx(ceiling)
    assert T.fill_hit(st, 100.0) == 0.0


def test_denning_bounds_on_cyclic_trace():
    # Denning averages the window over time, so on a strict cycle it smooths
    # LRU's cliff: it is only checked at the two ends here
    reqs = _cyclic(4, 6)
    c, d = T.lru_curve(reqs, 256), T.denning_curve(reqs, 256)
    assert d(1e9) == pytest.approx(c.ceiling)
    assert d(1.0) == 0.0


def test_page_smaller_than_block_is_refused():
    r = T.Request(0, None, 1024, 1, [1, 2], 512)
    with pytest.raises(ValueError):
        T.prefix_pages([r], 256)


def test_prefix_hit_needs_every_earlier_page():
    # same tail, different head: no prefix hit even though block 2 repeats
    a = T.Request(0, "a", 512, 1, [1, 2], 256)
    b = T.Request(1, "b", 512, 1, [9, 2], 256)
    assert T.lru_curve([a, b], 256).ceiling == 0.0
    c = T.Request(2, "c", 512, 1, [1, 2], 256)
    assert T.lru_curve([a, c], 256).ceiling == pytest.approx(0.5)


def test_shared_prefix_counts_as_hot():
    # two sessions sharing a one-page system prompt: the second session's
    # first request hits it although it reuses nothing of its own
    a = T.Request(0, "a", 512, 1, [1, 2], 256)
    b = T.Request(1, "b", 512, 1, [1, 3], 256)
    st = T.session_stats([a, b], 256)
    assert st.shared_any == [0, 256] and st.reusable == [0, 0]
    assert T.planner_curve(st)(1.0) == pytest.approx(256 / 1024)


def test_proportional_rule_is_warm_over_live():
    st = T.SessionStats(100.0, [1000] * 10, [0] + [1000] * 9, [math.inf] + [10.0] * 9,
                        [10.0] * 9 + [math.inf], [0] + [1000] * 9)
    warm, live = T.fill_users(st, 450.0)
    assert live == pytest.approx(0.9)
    assert warm == pytest.approx(0.45)
    assert T.proportional_hit(st, 450.0) == pytest.approx(0.9 * 0.5)
    assert T.fill_hit(st, 450.0) == 0.0


def test_planner_ttl_caps_reuse():
    st = T.SessionStats(1000.0, [100, 100, 100], [0, 100, 100],
                        [math.inf, 60.0, 600.0], [60.0, 600.0, math.inf], [0, 100, 100])
    assert T.planner_curve(st)(math.inf) == pytest.approx(200 / 300)
    assert T.planner_curve(st, ttl_s=300)(math.inf) == pytest.approx(100 / 300)


def test_readers(tmp_path):
    mc = tmp_path / "m.jsonl"
    mc.write_text("\n".join(json.dumps(d) for d in [
        {"timestamp": 2000, "input_length": 1024, "output_length": 5, "hash_ids": [1, 2]},
        {"timestamp": 0, "input_length": 512, "output_length": 5, "hash_ids": [1]}]))
    m = T.read_mooncake(mc)
    assert [r.t for r in m] == [0.0, 2.0] and m[0].session is None and m[0].bs == 512
    bl = tmp_path / "b.jsonl"
    bl.write_text("\n".join(json.dumps(d) for d in [
        {"chat_id": 1, "parent_chat_id": -1, "timestamp": 0.0, "input_length": 16,
         "output_length": 1, "hash_ids": [7]},
        {"chat_id": 2, "parent_chat_id": 1, "timestamp": 1.0, "input_length": 32,
         "output_length": 1, "hash_ids": [7, 8]},
        {"chat_id": 3, "parent_chat_id": 2, "timestamp": 2.0, "input_length": 48,
         "output_length": 1, "hash_ids": [7, 8, 9]}]))
    b = T.read_bailian(bl)
    assert {r.session for r in b} == {"1"}
    tl = tmp_path / "t.jsonl"
    rows = [("claude", "s", 0, "2026-01-01T00:00:00Z", 100, 0),
            ("claude", "s", 1, "2026-01-01T00:01:00Z", 150, 100),
            ("claude", "s", 2, "2026-01-01T01:01:00Z", 50, 0)]   # compaction
    tl.write_text("\n".join(json.dumps({
        "provider": p, "session_id": s, "round_index": i,
        "timing_events": [{"timestamp": ts}], "input_tokens_total": n,
        "prefix_tokens": c}) for p, s, i, ts, n, c in rows))
    stats, measured = T.read_tracelab(tl)
    st = stats["claude"]
    assert st.reusable == [0, 100, 0]
    assert st.gap_prev[1] == pytest.approx(60.0)
    assert measured["claude"] == pytest.approx(100 / 300)
    assert T.planner_curve(st, ttl_s=300)(math.inf) == pytest.approx(100 / 300)
