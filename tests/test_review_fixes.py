from __future__ import annotations

import random

import pytest

from workingset.agent_trace import (AgentTrace, TraceRequest, equivalent_workload,
                                    mean_gap)
from workingset.probe.population import histogram_evict_frac
from workingset.probe.replay import ReplaySession
from workingset.probe.request import RequestTrace


def test_mean_gap_averages_the_capped_pool_values():
    a = TraceRequest(1_000, 10, 1.0, "user")
    b = TraceRequest(1_000, 10, 100.0, "user")
    idle = TraceRequest(1_000, 10, 90_000.0, "user", kind="idle")
    tr = AgentTrace("pi", ((a, b, idle),))
    # the idle gap is replaced by a pool draw capped at 10: mean(1, 10) = 5.5
    assert mean_gap(tr, "sample", max_gap_s=10.0) == pytest.approx(
        (1.0 + 10.0 + 5.5) / 3)


def test_short_waits_keep_a_positive_think_time():
    reqs = tuple(TraceRequest(1_000, 10, 0.02, "tool") for _ in range(3))
    wl = equivalent_workload(AgentTrace("pi", (reqs,)))
    assert wl["think_time_s"] > 0


def test_a_replay_trims_a_retained_context_longer_than_the_next_prompt():
    tr = AgentTrace("pi", ((TraceRequest(400, 200, None, "user"),
                            TraceRequest(500, 10, 1.0, "user")),))
    s = ReplaySession(tr, 0, "", 0, 1.0, 10_000, 100.0, random.Random(0),
                      random.Random(1))
    s._s, s._pos = 0, 0
    s.next_turn()
    s.commit("x" * 200)
    text, kind = s.next_turn()
    assert kind == "hit" and len(text) <= 500


def test_cold_requests_below_the_bucket_cutoff_are_not_subtracted():
    miss = [RequestTrace(kind="miss", ttft=1.0, ptok_achieved=24_000)]
    cold = [RequestTrace(kind="miss", ttft=1.0, ptok_achieved=13_000), miss[0]]
    hit = [RequestTrace(kind="hit", ttft=1.0, ptok_achieved=24_000)]
    # the 12k threshold sits in the 16384 bucket, which counts as below: the
    # 13k request is not in `above`, so it must not be subtracted
    server = {"histograms": {"prefill_kv_computed_tokens_hist": {
        "count": 3, "buckets": {"16384": 1, "+Inf": 3}}}}
    assert histogram_evict_frac(server, hit, miss, cold) == 1.0
