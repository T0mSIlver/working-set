"""`ws decode-sweep --prefill-tokens/--prefill-depth`: which steps are mixed,
and the bookkeeping around them, on synthetic event logs and a fake server."""
from __future__ import annotations

import argparse
import asyncio
import time

import httpx
import pytest

from workingset import decode_sweep as ds
from workingset.metrics import detect_adapter, parse_text
from workingset.sweep_mixed import (chunk_steps, classify, depth_profile, format_row,
                                    self_check, stream_gaps, summarize, summary_rows)


def events(times, step_tok=1):
    return [(t, i * step_tok) for i, t in enumerate(times)]


def test_chunk_steps_reserve_the_decodes_budget():
    assert chunk_steps(2048, 2048, 1) == 2          # n takes a token of the budget
    assert chunk_steps(512, 2048, 32) == 1
    assert chunk_steps(8192, 2048, 64) == 5
    assert chunk_steps(512, None, 8) is None


def test_pure_steps_inside_the_interval_are_not_mixed():
    # steps every 20 ms; a request sent at 0.105, first token back at 0.285. Its
    # prefill spans ONE 120 ms step (0.14 -> 0.26); the gaps either side of it
    # inside [send, first] are pure steps waiting on HTTP and scheduling.
    t = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.26, 0.28, 0.30]
    g = stream_gaps(events(t), 0.0, 1.0)
    c = classify(g, [(0.105, 0.285)], k=1, tol=0.01)
    assert [round(e - s, 3) for s, e, _ in c["mixed"]] == [0.12]
    assert c["ambiguous"] == 1 + 1      # 0.10->0.12 straddles the send; 0.28->0.30 starts at the first token
    # with k unknown every candidate counts, which is the dilution
    assert len(classify(g, [(0.105, 0.285)], k=None, tol=0.01)["mixed"]) == 3


def test_headline_is_the_median_over_intervals_not_the_mean_over_gaps():
    def stream(per_interval):
        by = {j: [(0.0, d, 2)] for j, d in enumerate(per_interval)}
        return {"by_interval": by, "mixed": [g for j in by for g in by[j]],
                "pure": [(9.0, 9.02, 1)], "ambiguous": 0}
    # three intervals; one a slow outlier; accepted tokens (2 per step) speculation
    s = summarize([stream([.1, .1, .4])] * 2)
    assert s["mixed_step_s"] == pytest.approx(0.1)
    assert s["mixed_step_p90_s"] == pytest.approx(0.1)    # index 1 of 3 by int(q*(n-1))
    assert s["tok_s_seq"] == pytest.approx(20.0)           # 2 accepted tokens / 0.1 s
    assert s["pure_in_window_s"] == pytest.approx(0.02)


def test_depth_profile_aligns_step_j_across_streams():
    def stream(ms):
        t, ev = 0.0, []
        for d in ms:
            ev.append((t, t + d, 1))
            t += d
        return {"mixed": ev, "pure": [], "ambiguous": 0, "by_interval": {0: ev}}
    p = depth_profile([stream([.05, .06, .08]), stream([.05, .07, .08]), stream([.05])],
                      depth=3000, chunk=1024)
    assert [x["depth"] for x in p] == [1024, 2048, 3000]
    assert [x["step_s"] for x in p] == pytest.approx([.05, .065, .08])
    # without a chunk size: depth over the steps seen
    assert [x["depth"] for x in depth_profile([stream([.1, .1])], 2000)] == [1000, 2000]


def test_self_check_flags_a_diluted_or_wrong_step():
    # pure 14 ms + 270 ms prefill alone over one step = 284 ms expected
    assert self_check(0.28, 0.014, 0.270, 1)["ok"] is True
    bad = self_check(0.0915, 0.014, 0.270, 1)
    assert bad["ok"] is False and bad["ratio"] == pytest.approx(0.0915 / 0.284)
    assert self_check(0.1, 0.01, None, 1)["ok"] is None
    # two steps share the prefill
    assert self_check(0.15, 0.015, 0.270, 2)["ok"] is True


def test_summary_rows_mark_the_floor_and_old_records_have_none():
    w = {"kind": "tokens", "prefill_tokens": 8192, "ok": True, "alone_s": 0.27,
         "check": {"ratio": 1.0, "ok": True},
         "mixed": {"mixed_step_s": 0.03, "mixed_step_p90_s": 0.04, "tok_s_seq": 33.0,
                   "tok_s_p90_step": 25.0}}
    res = {"rungs": [{"n": 64, "prompt_tokens_mean": 1000, "pure_step_s": 0.01,
                      "mixed_windows": [w]}, {"n": 8}]}     # a v1 rung has no windows
    (r,) = summary_rows(res, 50.0)
    assert len(r["flags"]) == 2 and "FLAG" in format_row(r)
    assert not summary_rows(res, 20.0)[0]["flags"]
    w["check"] = {"ratio": 0.3, "ok": False}
    assert any("self-check" in f for f in summary_rows(res, 20.0)[0]["flags"])


def dump(steps, gen):
    lab = '{engine="0",model_name="m"}'
    lines = [f"vllm:generation_tokens_total{lab} {gen}",
             f'vllm:iteration_tokens_total_bucket{{engine="0",le="8.0",model_name="m"}} {steps}',
             f'vllm:iteration_tokens_total_bucket{{engine="0",le="+Inf",model_name="m"}} {steps}',
             f"vllm:iteration_tokens_total_count{lab} {steps}",
             f"vllm:iteration_tokens_total_sum{lab} {steps}",
             f"vllm:num_requests_running{lab} 4", f"vllm:num_requests_waiting{lab} 0"]
    return parse_text("\n".join(lines))


def test_mixed_window_against_a_fake_server():
    """Cold requests take 0.12 s; while one is in flight the held streams step at
    60 ms, otherwise at 10 ms. The window must report the slow steps as mixed."""
    busy = {"on": False}

    async def handler(req):
        busy["on"] = True
        await asyncio.sleep(0.12)
        busy["on"] = False
        body = ('data: {"choices":[{"text":"x"}],"usage":{"prompt_tokens":512,'
                '"completion_tokens":1}}\n\ndata: [DONE]\n\n')
        return httpx.Response(200, text=body)

    calls = []

    class Ep:
        ad = None

        def __init__(self, c):
            self.c = c

        async def scrape(self):
            calls.append(1)
            return time.monotonic(), dump(0, 0) if len(calls) == 1 else dump(40, 400), ""

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler),
                                     base_url="http://x/v1") as c:
            ep = Ep(c)
            ep.ad = detect_adapter(dump(0, 0))
            a = argparse.Namespace(model="m", seed=0, hold=0.5, prefill_timeout=5.0,
                                   chunk_tokens=None)
            pr = ds.Prompts.__new__(ds.Prompts)
            pr.a, pr.cpt, pr.file = a, 4.0, None
            st = ds.Streams(2, record=True)
            stop = asyncio.Event()

            async def ticker(i):
                k = 0
                while not stop.is_set():
                    k += 1
                    st.tok[i] = k
                    st.ts[i].append((time.monotonic(), k))
                    await asyncio.sleep(0.06 if busy["on"] else 0.01)
            ts = [asyncio.create_task(ticker(i)) for i in range(2)]
            w = await ds.mixed_window(ep, a, pr, 2, st, "tokens", 512, 0.01, 0)
            stop.set()
            await asyncio.gather(*ts)
            return w
    w = asyncio.run(run())
    assert w["requests"] >= 3 and w["prompt_tokens_mean"] == 512
    m = w["mixed"]
    assert m["mixed_step_s"] == pytest.approx(0.06, abs=0.03)
    assert m["tok_s_seq"] < 40            # ~16 tok/s on mixed steps, not the 100 of pure ones
    assert m["intervals"] >= 3
    assert w["ok"], w["why_not"]


def test_rate_bound_extrapolates_step_time_so_big_rungs_fit_the_pool():
    assert ds.rate_bound([], 8, 300) == 300
    assert ds.rate_bound([(1, 60.0)], 8, 300) == 60.0
    # 1/rate: 16.7 ms at n=1, 20.1 ms at n=8 -> about 33 ms at n=32
    b = ds.rate_bound([(1, 60.0), (8, 50.0)], 32, 300)
    assert 28 < b < 36
