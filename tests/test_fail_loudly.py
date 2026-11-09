"""The rule: a fact about the measured server or the input is read or asked
for, never assumed. Each test is an error case and its override, or the line
that says what was left out."""
from __future__ import annotations

import asyncio
import json
import math
from argparse import Namespace
from dataclasses import replace

import pytest

from test_probe import (client_for, fake_server, small_cfg, small_opts, trace)
from workingset import cli
from workingset import server_check as SC
from workingset.agent_trace import AgentTrace, TraceRequest, equivalent_workload
from workingset.config import RunConfig, defaulted_deployment_keys
from workingset.hypotheses.context import RunContext
from workingset.probe import RequestTrace, eval_rung
from workingset.probe.ladder import build_ladder
from workingset.probe.population import check_prompt_tokens, warm_share
from workingset.probe.request import EndpointSpec, send_request
from workingset.predict import predict
from workingset.test_cmd import (apply_replay, check_server,
                                 replaced_workload_keys)

CACHE_INFO = ('vllm:cache_config_info{block_size="16",'
              'enable_prefix_caching="True",num_gpu_blocks="%d"} 1.0\n')


def _server(**kw):
    s = {"max_model_len": 180_000, "model": "m", "block_size": 16,
         "pool_tokens": None, "prefix_caching": True, "spec_decode": None,
         "resolved": ["requests_running", "kv_cache_usage"], "errors": []}
    s.update(kw)
    return s


def _args(**kw):
    base = dict(metrics_url="http://x/metrics", engine=None,
                metrics_header=[], allow_config_mismatch=False,
                allow_blind_metrics=False)
    base.update(kw)
    return Namespace(**base)


def _ep():
    return EndpointSpec(base_url="http://x/v1", model="m")


def _pool(cfg):
    return SC.M.kv_pool_tokens(cfg.to_model(), cfg.to_topology())


# ---- 1. ws test against the server it measures -----------------------------
def test_cache_info_gives_block_pool_and_prefix_caching():
    got = SC.cache_info(CACHE_INFO % 1000)
    assert got == {"block_size": 16, "pool_tokens": 16_000,
                   "prefix_caching": True}


def test_a_max_model_len_that_differs_stops_the_run_and_the_flag_overrides(
        monkeypatch, capsys):
    cfg = RunConfig()
    monkeypatch.setattr(SC, "read_server", lambda *a, **k: _server(
        max_model_len=131_072, pool_tokens=int(_pool(cfg))))
    opts, notes, err = check_server(cfg, small_opts(), _ep(), _args(),
                                    "http://x/metrics")
    assert "deployment.max_model_len" in err and "131,072" in err
    assert "--allow-config-mismatch" in err
    opts, notes, err = check_server(cfg, small_opts(), _ep(), _args(
        allow_config_mismatch=True), "http://x/metrics")
    assert err is None
    assert any("ALLOWED MISMATCH" in n and "131,072" in n for n in notes)


def test_a_pool_off_by_over_15_percent_names_the_layout_and_reserve(
        monkeypatch):
    cfg = RunConfig()
    pred = _pool(cfg)
    monkeypatch.setattr(SC, "read_server", lambda *a, **k: _server(
        pool_tokens=int(pred * 0.8)))
    _, _, err = check_server(cfg, small_opts(), _ep(), _args(),
                             "http://x/metrics")
    for name in ("deployment.kv_sharding", "calibration.reserve_gib",
                 "ws calibrate --vllm-log", "--allow-config-mismatch"):
        assert name in err
    monkeypatch.setattr(SC, "read_server", lambda *a, **k: _server(
        pool_tokens=int(pred * 0.9)))
    _, _, err = check_server(cfg, small_opts(), _ep(), _args(),
                             "http://x/metrics")
    assert err is None


def test_a_server_that_gives_no_pool_or_length_is_an_error_not_a_pass(
        monkeypatch):
    cfg = RunConfig()
    monkeypatch.setattr(SC, "read_server", lambda *a, **k: _server(
        max_model_len=None, pool_tokens=None))
    _, _, err = check_server(cfg, small_opts(), _ep(), _args(),
                             "http://x/metrics")
    assert "no max_model_len" in err and "KV pool is unchecked" in err


def test_without_metrics_the_pool_is_noted_as_uncompared(monkeypatch):
    cfg = RunConfig()
    monkeypatch.setattr(SC, "read_server", lambda *a, **k: _server(
        block_size=None, resolved=None))
    _, notes, err = check_server(cfg, small_opts(kv_block_tokens=0), _ep(),
                                  _args(metrics_url=None), None)
    assert err is None and any("pool was not compared" in n for n in notes)
    assert any("KV block is unknown" in n for n in notes)


def test_the_kv_block_comes_from_metrics_and_a_flag_wins(monkeypatch):
    cfg = RunConfig()
    monkeypatch.setattr(SC, "read_server", lambda *a, **k: _server(
        block_size=1600, pool_tokens=int(_pool(cfg))))
    opts, _, _ = check_server(cfg, small_opts(kv_block_tokens=0), _ep(),
                              _args(), "http://x/metrics")
    assert opts.kv_block_tokens == 1600
    opts, notes, _ = check_server(cfg, small_opts(kv_block_tokens=32), _ep(),
                                  _args(), "http://x/metrics")
    assert opts.kv_block_tokens == 32 and any("1600" in n for n in notes)


def test_ws_test_exits_2_on_a_mismatch(tmp_path, monkeypatch, capsys):
    cfg = tmp_path / "w.toml"
    cfg.write_text(RunConfig().dumps("toml"))
    monkeypatch.setattr(SC, "read_server", lambda *a, **k: _server(
        max_model_len=1000))
    assert cli.main(["test", str(cfg)]) == 2
    assert "deployment.max_model_len" in capsys.readouterr().err


def test_defaulted_deployment_keys_are_named(tmp_path, capsys):
    cfg = tmp_path / "w.toml"
    cfg.write_text('[deployment]\nmodel = "27B"\ngpu = "H200"\n')
    keys = defaulted_deployment_keys(cfg)
    assert "max_model_len" in keys and "model" not in keys
    cli.main(["predict", str(cfg), "--max-model-len", "100000"])
    first = capsys.readouterr().out.splitlines()[0]
    assert first.startswith("deployment keys not set in") and \
        "kv_sharding=dcp" in first and "max_model_len" not in first


# ---- 2. the warm test follows the server's KV block ------------------------
def _hit(ptok, cached):
    return trace(1, "hit", 1.0, 0.1, tps=10, ctok=5, ptok=ptok, cached=cached)


def _warm(prev, cached, block=1600):
    """warm_share of one hit whose session sent `prev` tokens before."""
    first = trace(1, "first", 0.0, 0.1, ptok=prev)
    hit = trace(1, "hit", 1.0, 0.1, tps=10, ctok=5, ptok=prev + 2000,
                cached=cached)
    return warm_share([hit], block, [first, hit])


def test_a_warm_request_is_judged_against_the_block_not_a_fraction():
    # 1,600-token blocks: a 3,300-token previous prompt is two whole blocks;
    # one block short (1,600 cached) is still warm, none cached is not
    assert _warm(3300, 1600) == pytest.approx(1.0)
    assert _warm(3300, 0) == pytest.approx(0.0)
    # 40,000 tokens: 25 blocks, floor 24 x 1600 = 38,400; cached 38,400 warm
    assert _warm(40_000, 38_400) == pytest.approx(1.0)
    assert _warm(40_000, 20_000) == pytest.approx(0.0)
    # under two blocks the test cannot judge
    assert math.isnan(_warm(1700, 1600))


def test_an_unknown_block_with_cached_tokens_names_the_flag():
    with pytest.raises(ValueError, match="--kv-block-tokens"):
        warm_share([_hit(3300, 1600)], 0)
    cfg = small_cfg()
    with pytest.raises(ValueError, match="--kv-block-tokens"):
        eval_rung(1, 0, [_hit(1000, 990)], 0.0, cfg,
                  small_opts(kv_block_tokens=0))


# ---- 3. no usage, no decode rate -------------------------------------------
def test_a_stream_without_usage_is_counted_and_has_no_decode_rate():
    async def go():
        async with client_for(fake_server(n_tokens=6, usage=False)) as c:
            tr = RequestTrace(uid=1, kind="hit", ptok_intended=25)
            await send_request(c, _ep(), small_opts(), "p" * 100, tr, 6)
        return tr
    tr = asyncio.run(go())
    assert tr.ctok is None and tr.n_chunks == 6
    assert tr.tokens_estimated is True
    assert tr.decode_tps is None and tr.clean_decode_tps is None
    r = eval_rung(1, 0, [tr], 0.0, small_cfg(), small_opts())
    assert r.n_tokens_estimated == 1 and math.isnan(r.decode_p50)


# ---- 4. chars per token ----------------------------------------------------
def _ratio_traces(ratio, n=10):
    return [RequestTrace(uid=i, kind="hit", ptok_intended=1000,
                         ptok_achieved=int(1000 * ratio)) for i in range(n)]


def test_a_prompt_token_ratio_outside_the_band_stops_an_assumed_cpt():
    with pytest.raises(ValueError) as e:
        check_prompt_tokens(small_opts(), _ratio_traces(1.8))
    assert "--tokenizer" in str(e.value) and "--chars-per-token" in str(e.value)
    check_prompt_tokens(small_opts(), _ratio_traces(1.05))
    check_prompt_tokens(small_opts(), _ratio_traces(1.8)[:3])    # too few
    # the figure was given: the report prints the ratio, nothing stops
    check_prompt_tokens(small_opts(chars_per_token_given=True),
                        _ratio_traces(1.8))


def test_build_options_records_whether_the_cpt_was_given():
    from workingset.test_cmd import build_options
    assert not build_options(Namespace(), RunConfig()).chars_per_token_given
    assert build_options(Namespace(chars_per_token=3.2),
                         RunConfig()).chars_per_token_given


# ---- 5. the cold wait's idle prefill is the model's ------------------------
def test_the_cold_wait_verdict_says_its_idle_prefill_is_the_models():
    from workingset.hypotheses import REGISTRY
    from workingset.hypotheses.base import Measurement
    h = REGISTRY.select(["H-cold-wait"])[0]
    cfg = RunConfig()
    pr = h.predict(cfg, predict(cfg, n_iter=20))
    m = Measurement(text="not separable", data={
        "idle_source": "model", "reason": "no rung crossed",
        "idle_prefill_note": "idle prefill from the model (mfu 0.3), not "
                             "measured"})
    assert "idle prefill from the model (mfu 0.3), not measured" in \
        h.verdict(pr, m).text


# ---- 6. a replay says what it replaced -------------------------------------
def _trace_file(tmp_path, sessions):
    p = tmp_path / f"t{len(list(tmp_path.glob('t*.json')))}.json"
    p.write_text(json.dumps(AgentTrace("pi", sessions).to_dict()))
    return str(p)


def test_a_trace_with_no_waits_or_no_growth_is_an_error():
    one = AgentTrace("pi", ((TraceRequest(1000, 10, None, "start"),),))
    with pytest.raises(ValueError, match="think_time_s"):
        equivalent_workload(one)
    shrinking = AgentTrace("pi", ((TraceRequest(1000, 10, None, "start"),
                                   TraceRequest(400, 10, 2.0, "tool")),))
    with pytest.raises(ValueError, match="warm_turn_tokens"):
        equivalent_workload(shrinking)


def test_apply_replay_returns_the_cause_and_lists_what_it_replaced(tmp_path):
    cfg, err = apply_replay(small_cfg(), _trace_file(tmp_path, (
        (TraceRequest(1000, 10, None, "start"),),)), 300.0)
    assert err and "think_time_s" in err
    s = (TraceRequest(1000, 10, None, "start"),
         TraceRequest(1200, 10, 2.0, "tool"))
    new, err = apply_replay(small_cfg(), _trace_file(tmp_path, (s,)), 300.0)
    lines = replaced_workload_keys(small_cfg().workload, new.workload)
    assert err is None
    assert any(l.startswith("miss_rate: 0.5 -> 0.0") for l in lines)
    assert any(l.startswith("subagent_ratio: 0.5 -> 0.0") for l in lines)


def test_the_image_check_runs_on_the_users_config_before_a_replay(tmp_path,
                                                                  capsys):
    cfg = tmp_path / "w.toml"
    cfg.write_text(replace(RunConfig(), workload=replace(
        RunConfig().workload, image_share=0.5)).dumps("toml"))
    s = (TraceRequest(1000, 10, None, "start"),
         TraceRequest(1200, 10, 2.0, "tool"))
    rc = cli.main(["test", str(cfg), "--exclusive", "--replay",
                   _trace_file(tmp_path, (s,)), "--dry-run"])
    assert rc == 2 and "--api chat" in capsys.readouterr().err
    assert cli.main(["test", str(cfg), "--exclusive", "--dry-run", "--api",
                     "chat", "--replay", _trace_file(tmp_path, (s,))]) == 0
    assert "image_share: 0.5 -> 0.0" in capsys.readouterr().out


# ---- 7. the ladder and the burst's standing load ---------------------------
def test_rungs_above_max_users_are_reported_and_an_empty_ladder_is_an_error():
    dropped: list = []
    got = build_ladder(100, "0.5,1,4", max_users=150, dropped=dropped)
    assert got == [50, 100] and dropped == [400]
    with pytest.raises(ValueError, match="--max-users"):
        build_ladder(100, "2,4", max_users=50)
    with pytest.raises(ValueError, match="--rungs"):
        build_ladder(100, "", max_users=50)


def test_the_burst_standing_load_names_its_fallback():
    cfg = RunConfig()
    preds = predict(cfg, n_iter=20)
    ctx = RunContext(cfg, replace(preds, operating_point_users=0.0),
                     small_opts(), _ep())
    pop, why = ctx._burst_pop_why()
    assert pop >= 1 and "half the predicted limit" in why
    ctx = RunContext(cfg, preds, small_opts(), _ep(), burst_users=7)
    assert ctx._burst_pop_why() == (7, "--burst-users")


def test_ws_test_stops_on_an_empty_ladder(tmp_path, capsys):
    cfg = tmp_path / "w.toml"
    cfg.write_text(RunConfig().dumps("toml"))
    assert cli.main(["test", str(cfg), "--exclusive", "--max-users", "0",
                     "--dry-run"]) == 2
    assert "ladder is empty" in capsys.readouterr().err


# ---- 8. blind metrics ------------------------------------------------------
def test_metrics_with_neither_gauge_stop_the_run_unless_allowed(monkeypatch):
    cfg = RunConfig()
    monkeypatch.setattr(SC, "read_server", lambda *a, **k: _server(
        resolved=["ttft_hist"], pool_tokens=int(_pool(cfg))))
    _, _, err = check_server(cfg, small_opts(), _ep(), _args(),
                             "http://x/metrics")
    assert "requests_running" in err and "--allow-blind-metrics" in err
    _, notes, err = check_server(cfg, small_opts(), _ep(), _args(
        allow_blind_metrics=True), "http://x/metrics")
    assert err is None and any("--allow-blind-metrics" in n for n in notes)


# ---- 9. speculative decoding on one side only ------------------------------
def test_spec_decode_on_one_side_only_is_noted():
    cfg = RunConfig()
    spec = cfg.to_model().spec_tokens
    server = _server(spec_decode=(spec == 0), pool_tokens=int(_pool(cfg)))
    _, notes = SC.check_config(cfg, server, "http://x/metrics")
    assert any("spec_decode" in n for n in notes)
    server = _server(spec_decode=(spec > 0), pool_tokens=int(_pool(cfg)))
    _, notes = SC.check_config(cfg, server, "http://x/metrics")
    assert not any("spec_decode" in n for n in notes)


# ---- 10. small ones --------------------------------------------------------
def test_load_jsonl_counts_what_it_skips(tmp_path, capsys):
    from workingset.metrics.sampler import load_jsonl, load_jsonl_counted
    from test_metrics import _series
    p = tmp_path / "log.jsonl"
    p.write_text("\n".join(s.to_json() for s in _series())
                 + '\nnot json\n{"t_sent": 5.0, "rtt": 0.0, "li')
    snaps, skipped = load_jsonl_counted(p)
    assert len(snaps) == 5 and skipped == 2
    load_jsonl(p)
    assert "skipped 2 malformed line(s)" in capsys.readouterr().err


def test_an_into_file_that_does_not_parse_is_an_error_not_empty(tmp_path):
    from workingset.workload import existing_workload
    assert existing_workload(tmp_path / "missing.toml") == {}
    bad = tmp_path / "w.toml"
    bad.write_text("[workload\nusers = 4\n")
    with pytest.raises(ValueError, match="not valid TOML"):
        existing_workload(bad)


def test_turns_without_server_counts_are_left_out_and_counted(tmp_path,
                                                              capsys):
    from workingset.aa_local import read_measured
    d = tmp_path / "lane0"
    d.mkdir()

    def turn(**tok):
        return json.dumps({"success": True, "task_id": "a", "tokens": tok,
                           "timing": {"e2e_latency_ms": 2000,
                                      "generation_ms": 1000,
                                      "time_to_first_token_ms": 500}})
    full = dict(server_prompt_tokens=1000, server_output_tokens=11,
                server_cached_prompt_tokens=900)
    (d / "turns.jsonl").write_text("\n".join([
        turn(**full), turn(recorded_prompt_tokens=1000,
                           local_output_tokens=11)]))
    out = read_measured([d])
    assert out["turns"] == 1 and out["turns_skipped_no_server_counts"] == 1
    assert "1 successful turn(s) left out" in capsys.readouterr().err


def test_a_skipped_turn_keeps_its_time_in_the_lane(tmp_path):
    from workingset.aa_local import read_measured
    d = tmp_path / "lane0"
    d.mkdir()

    def turn(**tok):
        return json.dumps({"success": True, "task_id": "a", "tokens": tok,
                           "timing": {"e2e_latency_ms": 2000,
                                      "generation_ms": 1000,
                                      "time_to_first_token_ms": 500}})
    full = dict(server_prompt_tokens=1000, server_output_tokens=11,
                server_cached_prompt_tokens=900)
    (d / "turns.jsonl").write_text("\n".join([
        turn(**full), turn(server_prompt_tokens=1000), turn(**full)]))
    out = read_measured([d])
    assert out["turns"] == 2 and out["wall_s"] == pytest.approx(6.0)


def test_a_warm_hit_is_judged_against_the_previous_prompt():
    from workingset.probe.population import warm_share
    from workingset.probe.request import RequestTrace
    first = RequestTrace(uid=1, kind="first", t_send=0.0, ptok_achieved=31_000)
    hit = RequestTrace(uid=1, kind="hit", t_send=1.0, ptok_achieved=33_000,
                       cached_tokens=30_992)
    assert warm_share([hit], 16, [first, hit]) == 1.0
    hit.cached_tokens = 20_000
    assert warm_share([hit], 16, [first, hit]) == 0.0


def test_replies_without_completion_tokens_stop_the_run():
    from workingset.probe.population import RATIO_MIN_N, check_decode_counts
    from workingset.probe.request import RequestTrace
    ts = [RequestTrace(uid=i, ttft=0.1, t_end=1.0, tokens_estimated=True)
          for i in range(RATIO_MIN_N)]
    with pytest.raises(ValueError, match="completion_tokens"):
        check_decode_counts(ts)
    ts[0].tokens_estimated = False
    check_decode_counts(ts)


def test_models_are_read_for_the_named_model(monkeypatch):
    import httpx

    from workingset import bench as B
    body = {"data": [{"id": "small", "max_model_len": 8192},
                     {"id": "big", "max_model_len": 180_000}]}
    monkeypatch.setattr(httpx, "get", lambda *a, **k: httpx.Response(
        200, json=body, request=httpx.Request("GET", "http://h/v1/models")))
    assert B._endpoint_info("http://h/v1", "", "big")["max_model_len"] == 180_000
    assert "pass --model" in B._endpoint_info("http://h/v1", "")["error"]
    assert "does not serve 'x'" in B._endpoint_info("http://h/v1", "", "x")["error"]
