"""`ws decode-sweep`'s arithmetic on synthetic /metrics dumps and stream counts."""
from __future__ import annotations

import pytest

from workingset.decode_sweep import (client_window, judge, kv_need,
                                     output_budget, parse_batches,
                                     server_window)
from workingset.metrics import detect_adapter, parse_text


def dump(steps, gen, drafts=None, accepted=None, drafted=None, itl_fast=0):
    """A /metrics body: `steps` forward passes, `gen` tokens, and (when given)
    speculative-decoding counters. `itl_fast` observations fall under 0.05 s."""
    lab = '{engine="0",model_name="m"}'
    h = lambda base, total, fast, bounds: [
        f'{base}_bucket{{engine="0",le="{b}",model_name="m"}} {fast if b == 0.05 else total}'
        for b in bounds] + [f'{base}_bucket{{engine="0",le="+Inf",model_name="m"}} {total}',
                            f'{base}_count{lab} {total}', f'{base}_sum{lab} {total * 0.04}']
    lines = [f"vllm:generation_tokens_total{lab} {gen}"]
    lines += h("vllm:iteration_tokens_total", steps, steps, [1.0, 8.0])
    lines += h("vllm:inter_token_latency_seconds", gen, itl_fast, [0.025, 0.05, 0.1])
    if drafts is not None:
        lines += [f"vllm:spec_decode_num_drafts_total{lab} {drafts}",
                  f"vllm:spec_decode_num_draft_tokens_total{lab} {drafted}",
                  f"vllm:spec_decode_num_accepted_tokens_total{lab} {accepted}"]
    return parse_text("\n".join(lines))


def test_server_window_step_time_and_acceptance():
    a = dump(1000, 8000, drafts=500, drafted=1500, accepted=300)
    b = dump(1250, 10000, drafts=700, drafted=2100, accepted=700)
    w = server_window(detect_adapter(b), a, b, 10.0)
    assert w["steps"] == 250 and w["step_time_s"] == pytest.approx(0.04)
    assert w["tok_s_total"] == pytest.approx(200.0)
    assert w["tokens_per_step"] == pytest.approx(8.0)
    s = w["spec_decode"]
    assert s["acceptance_rate"] == pytest.approx(400 / 600)
    assert s["mean_accepted_len"] == pytest.approx(1 + 400 / 200)


def test_server_window_without_spec_decode_says_none():
    a, b = dump(0, 0), dump(100, 3200)
    assert server_window(detect_adapter(b), a, b, 5.0)["spec_decode"] is None


def test_client_window_matches_per_stream_rates():
    tok0, tok1 = [0, 0, 0], [100, 300, 200]
    ev0, ev1 = [0, 0, 0], [50, 150, 100]
    c = client_window(tok0, tok1, ev0, ev1, 10.0, [True, True, False])
    assert c["streams"] == 2 and c["tok_s_seq_mean"] == pytest.approx(20.0)
    assert c["step_time_s"] == pytest.approx(0.1)


def test_judge_flags_a_short_batch():
    ok, why = judge(32, [32, 32], [0, 0], 0, 0)
    assert ok and not why
    ok, why = judge(32, [32, 20], [0, 5], 1, 2)
    assert not ok and len(why) == 4


def test_planning_arithmetic():
    assert parse_batches("8,1,8,32") == [1, 8, 32]
    with pytest.raises(ValueError):
        parse_batches("0")
    # a shared context is stored once, distinct ones n times
    assert kv_need(10, 1000, 100, 400, True) == 1000 + 10 * 500
    assert kv_need(10, 1000, 100, 400, False) == 10 * 1500
    assert output_budget(100, 40) == 5000


def test_cross_origin_metrics_scrape_carries_no_api_token():
    import argparse
    import asyncio

    import httpx

    from workingset.decode_sweep import Endpoint

    seen = []

    def handler(req):
        seen.append(req)
        return httpx.Response(200, text='vllm:generation_tokens_total{engine="0"} 1\n')

    async def run():
        t = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=t, base_url="https://api.example/v1",
                                     headers={"Authorization": "Bearer SECRET"}) as c, \
                httpx.AsyncClient(transport=t) as mc:
            a = argparse.Namespace(metrics_url="https://metrics.other/metrics",
                                   timeout=5.0, engine=None)
            ep = Endpoint(c, a, {"X-Scrape": "1"}, mc)
            try:
                await ep.scrape()
            except Exception:
                pass

    asyncio.run(run())
    assert seen and "authorization" not in seen[0].headers
    assert seen[0].headers["X-Scrape"] == "1"


# ---------------------------------------------------------------------------
# what the sweep cannot read stops it before any load
# ---------------------------------------------------------------------------
CACHE = ('vllm:cache_config_info{block_size="16",enable_prefix_caching="True",'
         'engine="0",num_gpu_blocks="1000"} 1.0\n'
         'vllm:num_requests_running{engine="0"} 0\n')
MODELS = {"data": [{"id": "m", "max_model_len": 200_000}]}


def _sweep(monkeypatch, capsys, *flags, metrics=CACHE, models=MODELS):
    import httpx

    from workingset.cli import main
    sent = []

    def handler(req):
        if req.url.path.endswith("/metrics"):
            return httpx.Response(200, text=metrics)
        if req.url.path.endswith("/models"):
            return httpx.Response(200, json=models)
        sent.append(req)
        return httpx.Response(500, text="no load in these tests")
    real = httpx._client.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(
        transport=httpx.MockTransport(handler),
        **{k: v for k, v in kw.items() if k != "verify"}))
    rc = main(["decode-sweep", "--base-url", "http://h:8000/v1", "--model", "m",
               "--batch", "1", "--context", "1000", *flags])
    assert not sent
    return rc, capsys.readouterr().err


def test_the_sweep_needs_the_tensor_parallel_width(monkeypatch, capsys):
    rc, err = _sweep(monkeypatch, capsys)
    assert rc == 2 and "--tensor-parallel is required" in err


def test_several_engines_need_one_named(monkeypatch, capsys):
    two = CACHE + 'vllm:num_requests_running{engine="1"} 0\n'
    rc, err = _sweep(monkeypatch, capsys, "--tensor-parallel", "4",
                     metrics=two)
    assert rc == 2 and "2 engines" in err and "--engine" in err


def test_unknown_pool_or_prefix_caching_stops_the_sweep(monkeypatch, capsys):
    bare = 'vllm:num_requests_running{engine="0"} 0\n'
    rc, err = _sweep(monkeypatch, capsys, "--tensor-parallel", "4",
                     metrics=bare)
    assert rc == 2 and "--kv-tokens" in err
    rc, err = _sweep(monkeypatch, capsys, "--tensor-parallel", "4",
                     "--kv-tokens", "16000", metrics=bare)
    assert rc == 2 and "does not say whether prefix caching" in err


def test_a_model_the_server_does_not_list_stops_the_sweep(monkeypatch,
                                                         capsys):
    rc, err = _sweep(monkeypatch, capsys, "--tensor-parallel", "4",
                     models={"data": [{"id": "other"}]})
    assert rc == 2 and "'m' is not served" in err and "other" in err
    rc, err = _sweep(monkeypatch, capsys, "--tensor-parallel", "4",
                     models={"data": [{"id": "m"}]})
    assert rc == 2 and "--max-model-len" in err


def test_judge_fails_a_rung_whose_streams_did_not_all_start():
    ok, why = judge(32, [32, 32], [0, 0], 0, 0, started=30)
    assert not ok and "only 30 of 32 streams started" in why[0]
