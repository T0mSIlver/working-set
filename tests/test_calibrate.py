"""`ws calibrate` recovers constants it is handed, from records shaped like
`ws decode-sweep`'s, and what it writes is what `ws predict` prices with."""
import json
import tomllib
from dataclasses import replace

import pytest

from workingset import calibrate as C
from workingset import model as M
from workingset.config import RunConfig, load_config
from workingset.predict import predict

TRUE = M.DecodeConstants(eta=0.34, tau_0=7e-6, tau_ar=14e-6, tau_rec=5e-6)
ALPHA, K = 0.85, 3


def _sweep(cfg: RunConfig, tp: int, contexts=(20_000, 80_000), noise=0.0,
           pool: int | None = None, k: int = K) -> dict:
    """A decode-sweep record whose step times come from TRUE, at draft depth k."""
    m0 = replace(cfg.to_model(), spec_tokens=k)
    topo = M.topology_grid(1, tp, cfg.deployment.gpu)
    m = M._replicated(m0, topo)
    rungs = []
    for ctx in contexts:
        for i, n in enumerate((1, 2, 4, 8, 16, 32)):
            kv = n * (ctx * m.kv_bpt + m.state_traffic)
            dc = replace(TRUE, mfu=cfg.calibration.mfu)
            t = M.decode_step_seconds(m, topo, n, kv, dc) * (1 + noise * (-1) ** i)
            drafts = 1000 * n
            acc = [drafts * ALPHA ** (j + 1) for j in range(k)]
            spec = None if k == 0 else {
                "drafts": drafts, "draft_tokens": drafts * k, "accepted_tokens": sum(acc),
                "accepted_per_position": {str(j): acc[j] / drafts for j in range(k)}}
            rungs.append({
                "n": n, "batch_ok": True, "prompt_tokens_mean": ctx,
                "tok_s_seq_server": 0.0,
                "server": {"step_time_s": t, "spec_decode": spec}})
    return {"format": "workingset-decode-sweep", "version": 1, "context": contexts[0],
            "settle_s": 10, "hold_s": 30, "kv_pool_tokens": pool,
            "tensor_parallel": tp, "rungs": rungs}


def _cfg(tp: int = 2) -> RunConfig:
    return RunConfig.from_dict({"deployment": {"model": "27B", "gpu": "H200",
                                               "tensor_parallel": tp}})


def test_two_widths_and_two_depths_recover_every_decode_constant():
    cfg = _cfg(2)
    fits = C.calibrate(cfg, [_sweep(cfg, 1), _sweep(cfg, 2), _sweep(cfg, 2, k=0)])
    assert fits["eta"].value == pytest.approx(TRUE.eta, rel=1e-6)
    assert fits["tau_0"].value == pytest.approx(TRUE.tau_0, rel=1e-4)
    assert fits["tau_ar"].value == pytest.approx(TRUE.tau_ar, rel=1e-4)
    assert fits["tau_rec"].value == pytest.approx(TRUE.tau_rec, rel=1e-4)
    assert all(fits[k].status == "measured" for k in ("eta", "tau_0", "tau_ar", "tau_rec"))
    # geometric acceptance from the per-position counters
    assert fits["acceptance"].value == pytest.approx(ALPHA, abs=1e-9)
    assert "0.850, 0.850, 0.850" in fits["acceptance"].note


def test_one_width_fits_one_per_layer_cost_and_names_it():
    cfg = _cfg(2)
    fits = C.calibrate(cfg, [_sweep(cfg, 2), _sweep(cfg, 2, k=0)])
    # tau_0 and tau_ar move together: tau_0 stays at its prior and tau_ar
    # carries the rest of the per-layer cost
    assert fits["tau_0"].status == "prior" and "tp > 1" in fits["tau_0"].note
    p0 = M.decode_priors()[0]
    assert fits["tau_ar"].value == pytest.approx(TRUE.tau_0 + TRUE.tau_ar - p0, rel=1e-4)
    assert fits["eta"].value == pytest.approx(TRUE.eta, rel=1e-6)


def test_noise_gives_standard_errors_and_a_constant_byte_load_is_refused():
    cfg = _cfg(2)
    fits = C.calibrate(cfg, [_sweep(cfg, 1, noise=0.01), _sweep(cfg, 2, noise=0.01),
                             _sweep(cfg, 2, noise=0.01, k=0)])
    assert fits["eta"].stderr and 0 < fits["eta"].stderr < 0.1
    assert abs(fits["eta"].value / TRUE.eta - 1) < 0.1
    # every rung at one n and one context: the bytes never vary
    flat = _sweep(cfg, 2, contexts=(20_000,))
    flat["rungs"] = [r for r in flat["rungs"] if r["n"] == 8] * 4
    assert C.calibrate(cfg, [flat])["eta"].status == "not identified"


def test_the_reported_pool_sets_the_reserve(tmp_path):
    cfg = _cfg(2)
    m, t = cfg.to_model(), cfg.to_topology()
    want = M.kv_pool_tokens(m, t) * 0.9           # a server with 10% less pool
    fits = C.calibrate(cfg, [_sweep(cfg, 2, pool=int(want))])
    r = fits["reserve"]
    assert r.status == "measured" and r.value > M.ACT_RESERVE
    got = M.kv_pool_tokens(m, replace(t, reserve=r.value))
    assert got == pytest.approx(int(want), rel=1e-9)
    # vLLM's start-up line wins over the sweep's reading
    log = tmp_path / "vllm.log"
    log.write_text("INFO GPU KV cache size: 1,234,567 tokens\n", encoding="utf-8")
    assert C.pool_from_log(str(log)) == 1_234_567


def test_write_reaches_predict(tmp_path):
    cfg = _cfg(2)
    path = tmp_path / "workingset.toml"
    path.write_text(cfg.dumps("toml"), encoding="utf-8")
    runs = []
    for tp, k in ((1, K), (2, K), (2, 0)):
        f = tmp_path / f"sweep_tp{tp}_k{k}.json"
        f.write_text(json.dumps(_sweep(cfg, tp, pool=3_000_000, k=k)), encoding="utf-8")
        runs.append(str(f))
    args = type("A", (), {"config": str(path), "runs": runs, "vllm_log": None,
                          "write": True})()
    assert C.cmd_calibrate(args) == 0
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    cal = raw["calibration"]
    assert cal["eta"] == pytest.approx(TRUE.eta, rel=1e-5)
    assert cal["tau_ar_us"] == pytest.approx(TRUE.tau_ar * 1e6, rel=1e-4)
    assert cal["acceptance"] == pytest.approx(ALPHA, abs=1e-6)
    assert cal["fit"]["eta"]["status"] == "measured" and cal["fit"]["eta"]["n"] == 36
    assert "reserve_gib" in cal
    back = load_config(path)
    back.validate()
    assert back == RunConfig.from_dict(tomllib.loads(back.dumps("toml")))
    # predict prices with the fitted constants: the pool and the decode
    # constants are the calibrated ones
    dc = back.decode_constants()
    assert dc.eta == pytest.approx(TRUE.eta, rel=1e-5)
    assert M.kv_pool_tokens(back.to_model(), back.to_topology()) == pytest.approx(3_000_000, rel=1e-6)
    assert predict(back, n_iter=60).decode_ceiling_users != predict(cfg, n_iter=60).decode_ceiling_users


def test_a_record_of_another_kind_is_refused(tmp_path, capsys):
    cfg = _cfg(2)
    path = tmp_path / "workingset.toml"
    path.write_text(cfg.dumps("toml"), encoding="utf-8")
    bad = tmp_path / "x.json"
    bad.write_text('{"format": "something-else"}', encoding="utf-8")
    args = type("A", (), {"config": str(path), "runs": [str(bad)], "vllm_log": None,
                          "write": False})()
    assert C.cmd_calibrate(args) == 2
    assert "not a decode-sweep or replay record" in capsys.readouterr().err


def test_acceptance_pools_rungs_at_their_own_depth():
    """Equal k = 2 and k = 3 sweeps at 0.85: a draft that never tried
    position 3 must not count as a rejection there (Codex review)."""
    cfg = _cfg(2)
    fits = C.calibrate(cfg, [_sweep(cfg, 2, k=2), _sweep(cfg, 2, k=3)])
    assert fits["acceptance"].value == pytest.approx(ALPHA, abs=1e-9)
    assert "0.850, 0.850, 0.850" in fits["acceptance"].note
    # without per-position counters: solved from the accepted lengths
    runs = [_sweep(cfg, 2, k=2), _sweep(cfg, 2, k=3)]
    for rec in runs:
        for r in rec["rungs"]:
            r["server"]["spec_decode"].pop("accepted_per_position")
    assert C.fit_acceptance(runs).value == pytest.approx(ALPHA, abs=1e-9)


def test_a_zero_tau_stays_zero():
    """A full-rank design that puts a tau on the zero bound keeps the zero, so
    the written constants reproduce the fit (Codex review)."""
    global TRUE
    saved = TRUE
    TRUE = replace(TRUE, tau_rec=0.0)
    try:
        cfg = _cfg(2)
        fits = C.calibrate(cfg, [_sweep(cfg, 1), _sweep(cfg, 2), _sweep(cfg, 2, k=0)])
    finally:
        TRUE = saved
    assert fits["tau_rec"].value == 0.0 and fits["tau_rec"].status == "measured"
    assert fits["eta"].value == pytest.approx(saved.eta, rel=1e-6)


def test_mfu_counts_attention_flops():
    """Replay timings generated at MFU 0.45 with long prompts fit back to 0.45
    (GEMM-only FLOPs read 0.33; Codex review)."""
    cfg = _cfg(2)
    m, t = cfg.to_model(), cfg.to_topology()
    rows = []
    for u, prompt in ((4_000, 4_000), (16_000, 16_000), (64_000, 64_000), (20_000, 120_000)):
        f = M.prefill_flops(m, u, prior=prompt - u)[2]
        rows.append({"ok": True, "uncached": u, "prompt": prompt,
                     "ttft": 0.05 + f / (M.peak_flops(t) * 0.45)})
    fit = C.fit_mfu([{"speed": {}, "calibration": rows, "steps": []}], cfg)
    assert fit.value == pytest.approx(0.45, rel=1e-6)


def test_mfu_ignores_concurrent_tree_rows():
    """Rows of a session tree replayed with its streams in flight are slower
    than a request alone; they must not enter the single-stream MFU fit."""
    cfg = _cfg(2)
    m, t = cfg.to_model(), cfg.to_topology()
    rows = []
    for u, prompt in ((4_000, 4_000), (16_000, 16_000), (64_000, 64_000), (20_000, 120_000)):
        f = M.prefill_flops(m, u, prior=prompt - u)[2]
        base = {"ok": True, "uncached": u, "prompt": prompt}
        rows.append(dict(base, ttft=0.05 + f / (M.peak_flops(t) * 0.45)))
        rows.append(dict(base, phase="tree", ttft=0.05 + 3 * f / (M.peak_flops(t) * 0.45)))
    fit = C.fit_mfu([{"speed": {}, "calibration": rows, "steps": []}], cfg)
    assert fit.value == pytest.approx(0.45, rel=1e-6)
