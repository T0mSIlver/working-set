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
from workingset.sweep_mixed import (classify, depth_profile, format_row, stream_gaps,
                                    summarize, summary_rows)


def events(times, step_tok=1):
    return [(t, i * step_tok) for i, t in enumerate(times)]


def test_gaps_are_classified_by_the_request_interval():
    # decode steps every 20 ms; a request sent at 0.105, first token at 0.165;
    # the step in flight at the send (0.100 -> 0.120) is ambiguous, the steps
    # 0.120 -> 0.140 -> 0.165 ran with the request in the batch
    t = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.165, 0.185, 0.205]
    g = stream_gaps(events(t), 0.0, 1.0)
    c = classify(g, [(0.105, 0.165)], tol=0.01)
    assert [round(e - s, 3) for s, e, _ in c["mixed"]] == [0.02, 0.025]
    assert c["ambiguous"] == 2           # 0.10->0.12 straddles the send; 0.165->0.185 starts at the end
    assert len(c["pure"]) == 6 and len(c["last"]) == 1
    assert c["last"][0][1] == pytest.approx(0.165)


def test_summary_is_per_sequence_on_mixed_steps_only():
    # 100 ms mixed steps delivering 3 tokens each (speculation), 20 ms pure steps of 1
    mixed = [(0.0, 0.1, 3), (0.1, 0.2, 3)]
    pure = [(1.0, 1.02, 1)] * 4
    s = summarize([{"mixed": mixed, "pure": pure, "ambiguous": 2, "last": mixed[-1:]}] * 2)
    assert s["mixed_step_s"] == pytest.approx(0.1)
    assert s["tok_s_seq_mean"] == pytest.approx(30.0)      # accepted tokens, not 1/step
    assert s["pure_step_s"] == pytest.approx(0.02)
    assert s["share_mixed"] == pytest.approx(4 / 16)


def test_depth_profile_aligns_step_j_across_streams():
    def stream(ms):
        t, ev = 0.0, []
        for d in ms:
            ev.append((t, t + d, 1))
            t += d
        return {"mixed": ev, "pure": [], "ambiguous": 0, "last": []}
    p = depth_profile([stream([.05, .06, .08]), stream([.05, .07, .08]), stream([.05])],
                      depth=3000, chunk=1024)
    assert [x["depth"] for x in p] == [1024, 2048, 3000]
    assert [x["step_s"] for x in p] == pytest.approx([.05, .065, .08])
    # without a chunk size: depth over the steps seen
    assert [x["depth"] for x in depth_profile([stream([.1, .1])], 2000)] == [1000, 2000]


def test_summary_rows_mark_the_floor_and_old_records_have_none():
    w = {"kind": "tokens", "prefill_tokens": 8192, "ok": True,
         "mixed": {"mixed_step_s": 0.03, "mixed_step_p99_s": 0.04, "tok_s_seq_mean": 33.0,
                   "tok_s_worst_step": 25.0, "share_mixed": 0.8}}
    res = {"rungs": [{"n": 64, "prompt_tokens_mean": 1000, "pure_step_s": 0.01,
                      "mixed_windows": [w]}, {"n": 8}]}     # a v1 rung has no windows
    (r,) = summary_rows(res, 50.0)
    assert len(r["flags"]) == 2 and "BELOW" in format_row(r)
    assert not summary_rows(res, 20.0)[0]["flags"]


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
    assert m["tok_s_seq_mean"] < 40            # ~16 tok/s on mixed steps, not the 100 of pure ones
    assert m["pure_gaps"] >= 0 and 0 < m["share_mixed"] <= 1
    assert w["ok"], w["why_not"]
