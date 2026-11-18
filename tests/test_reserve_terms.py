"""vLLM's three reserve terms: `ws calibrate --vllm-log` reads them from the
start-up lines vLLM 0.33 prints, and the pool they give is vLLM's own
arithmetic (utilisation x total - weights - activation - non-torch, per GPU)."""
import tomllib
from dataclasses import replace

import pytest

from workingset import calibrate as C
from workingset import model as M
from workingset.config import RunConfig, load_config

G = 1024 ** 3


def _log(tp=2, util=0.90, total=140.4, weights=14.4, non_torch=1.0, act=3.0,
         graph_est=1.5, tokens=None, version="0.33.0", free_line=True,
         loading=True, concurrency=40.0) -> str:
    """Start-up lines shaped like vLLM 0.33's (gpu_worker.py, kv_cache_utils.py),
    one per worker where vLLM prints one per worker."""
    req = round(util * total, 2)
    cons = round(weights + non_torch, 2)
    kv = round(req - cons - act, 2)
    tokens = tokens or int(kv * G * tp / 40_000)
    lines = [f"INFO vllm.entrypoints.launchers.launcher : vLLM server version {version}, "
             "serving model /m",
             f"INFO vllm.v1.engine.core : Initializing a V1 LLM engine (v{version}) with "
             f"config: model='/m', dtype=torch.bfloat16, tensor_parallel_size={tp}, "
             "pipeline_parallel_size=1"]
    if loading:
        lines += [f"(Worker_TP{r}) INFO vllm.v1.worker.gpu.model_runner : Model loading "
                  f"took {weights} GiB memory and 35.0 seconds" for r in range(tp)]
    lines += [f"INFO vllm.v1.worker.gpu_worker : Available KV cache memory: {kv} GiB",
              "INFO vllm.v1.worker.gpu_worker : CUDA graph memory profiling is enabled "
              f"(default since v0.21.0). The current --gpu-memory-utilization={util:.4f} "
              "is equivalent to --gpu-memory-utilization=0.8800 without CUDA graph "
              "memory profiling.",
              f"INFO vllm.v1.core.kv_cache_utils : GPU KV cache size: {tokens:,} tokens, "
              f"Maximum concurrency for 262,144 tokens per request: {concurrency:.2f}x"]
    if free_line:
        for _ in range(tp):
            lines += [
                f"INFO vllm.v1.worker.gpu_worker : CUDA graph pool memory: 0.80 GiB "
                f"(actual), {graph_est} GiB (estimated), difference: 0.70 GiB (46.7%).",
                f"INFO vllm.v1.worker.gpu_worker : Free memory on device "
                f"({total - 1.6:.2f}/{total} GiB) on startup. Desired GPU memory "
                f"utilization is ({util}, {req} GiB). Actual usage is {cons} GiB for "
                f"consumed memory (weights + non-torch), {act} GiB for peak "
                "activation, and 0.80 GiB for CUDAGraph memory. Replace "
                "gpu_memory_utilization config with `--kv-cache-memory=1` (0.0 GiB) "
                "to fit into requested memory, or `--kv-cache-memory=2` (0.0 GiB) to "
                f"fully utilize gpu memory. Current kv cache memory in use is {kv} GiB."]
    return "\n".join(lines) + "\n"


def _cfg(tp=2, gpu="H200", model="27B", **cal) -> RunConfig:
    return RunConfig.from_dict({"deployment": {"model": model, "gpu": gpu,
                                               "tensor_parallel": tp},
                                "calibration": cal})


def test_every_start_up_line_is_read_per_gpu():
    mem = C.read_vllm_log(_log())
    assert mem.version == "0.33.0" and mem.tensor_parallel == 2
    assert mem.gpu_total == pytest.approx(140.4 * G)
    assert mem.gpu_memory_utilization == 0.90
    assert mem.weights == pytest.approx(14.4 * G)
    assert mem.non_torch == pytest.approx(1.0 * G)
    assert mem.activation == pytest.approx(3.0 * G)
    assert mem.graph_estimate == pytest.approx(1.5 * G)
    t = mem.terms()
    assert t.holdback == pytest.approx(0.10 * 140.4 * G)
    # vLLM's own identity: requested - consumed - activation = KV memory
    assert (mem.requested - mem.consumed - mem.activation) == pytest.approx(mem.kv_memory,
                                                                         abs=0.01 * G)
    assert not mem.notes


def test_a_log_restarted_twice_is_read_from_its_last_start():
    mem = C.read_vllm_log(_log(act=9.9) + _log(act=3.0))
    assert mem.activation == pytest.approx(3.0 * G)
    assert "2 server starts" in mem.notes[0]


def test_a_missing_line_names_the_line_and_the_key():
    mem = C.read_vllm_log(_log(free_line=False))
    # the utilisation still comes from the CUDA graph profiling line
    assert mem.gpu_memory_utilization == 0.90
    with pytest.raises(ValueError) as e:
        mem.terms()
    msg = str(e.value)
    assert "Free memory on device" in msg
    for key in ("gpu_total_gib", "activation_gib", "non_torch_gib"):
        assert f"calibration.{key}" in msg
    mem = C.read_vllm_log(_log(loading=False))
    with pytest.raises(ValueError, match="Model loading took.*calibration.non_torch_gib"):
        mem.terms()


def test_logs_the_terms_cannot_describe_are_refused():
    fixed = _log() + ("INFO vllm.v1.worker.gpu_worker : Initial free memory 138.2 GiB, "
                      "reserved 100.0 GiB memory for KV Cache as specified by "
                      "kv_cache_memory_bytes config and skipped memory profiling.\n")
    mem = C.read_vllm_log(fixed)
    assert mem.pool_tokens and "--kv-cache-memory" in mem.notes[0]
    with pytest.raises(ValueError, match="--kv-cache-memory"):
        mem.terms()
    host = _log().replace("GPU KV cache size", "GPU + host KV cache size")
    with pytest.raises(ValueError, match="host memory"):
        C.read_vllm_log(host)


def test_calibrate_reports_the_terms_and_refuses_another_width(tmp_path):
    log = tmp_path / "vllm.log"
    log.write_text(_log(tp=2), encoding="utf-8")
    notes = []
    fits = C.calibrate(_cfg(2), [], str(log), notes)
    for name in ("gpu_memory_utilization", "gpu_total", "activation", "non_torch",
                 "reserve"):
        assert fits[name].status == "measured"
    assert any(n.startswith("reserve terms: holdback") for n in notes)
    with pytest.raises(ValueError, match="deployment.tensor_parallel"):
        C.calibrate(_cfg(4), [], str(log))
    # without the terms' line: the pool still fits, the terms are said not read,
    # and --reserve terms refuses
    log.write_text(_log(free_line=False), encoding="utf-8")
    notes = []
    fits = C.calibrate(_cfg(2), [], str(log), notes)
    assert "reserve" in fits and "activation" not in fits
    assert any("not read" in n for n in notes)
    with pytest.raises(ValueError, match="Free memory on device"):
        C.calibrate(_cfg(2), [], str(log), reserve="terms")


@pytest.mark.parametrize("model,gpu,tp,total", [("27B", "H200", 1, 150e9),
                                                ("K31P", "B300", 8, 287e9)])
def test_the_terms_price_vllms_pool(model, gpu, tp, total):
    """At TP1 and TP8 the pool is tp x (utilisation x total - activation -
    non_torch) - weights, in the bytes of the cache the group stores."""
    terms = M.ReserveTerms(gpu_memory_utilization=0.9, gpu_total=total,
                           activation=3 * G, non_torch=2 * G)
    m = M.MODELS[model]
    topo = replace(M.topology_grid(1, tp, gpu, "replicate"), reserve_terms=terms)
    per_gpu = 0.9 * total - 3 * G - 2 * G
    assert tp * per_gpu > M.hbm_weights(m)
    want = (tp * per_gpu - M.hbm_weights(m)) / M._replicated(m, topo).kv_bpt
    assert M.kv_pool_tokens(m, topo) == pytest.approx(want, rel=1e-12)
    # the holdback scales with the GPU's memory; activation and non_torch do not
    bigger = replace(topo, reserve_terms=replace(terms, gpu_total=2 * total))
    grow = (M.kv_pool_tokens(m, bigger) - M.kv_pool_tokens(m, topo)) * \
        M._replicated(m, topo).kv_bpt
    assert grow == pytest.approx(tp * 0.9 * total, rel=1e-9)
    with pytest.raises(ValueError, match="not both"):
        M.kv_pool_tokens(m, replace(topo, reserve=20 * G))


def test_config_carries_all_four_terms_or_none(tmp_path):
    full = dict(gpu_memory_utilization=0.90, gpu_total_gib=140.4, activation_gib=3.0,
                non_torch_gib=1.0)
    cfg = _cfg(2, **full)
    cfg.validate()
    t = cfg.to_topology()
    assert t.reserve is None and t.reserve_terms.gpu_total == pytest.approx(140.4 * G)
    assert RunConfig.from_dict(tomllib.loads(cfg.dumps("toml"))) == cfg
    with pytest.raises(ValueError, match="but not activation_gib"):
        _cfg(2, **{k: v for k, v in full.items() if k != "activation_gib"}).validate()
    with pytest.raises(ValueError, match="keep one"):
        _cfg(2, reserve_gib=18.0, **full).validate()
    with pytest.raises(ValueError, match="gpu_memory_utilization must be"):
        _cfg(2, **{**full, "gpu_memory_utilization": 1.5}).validate()
    # no terms: the default reserve, unchanged
    plain = _cfg(2)
    assert M.per_gpu_reserve(plain.to_topology()) == M.ACT_RESERVE


def test_write_stores_one_reserve_form(tmp_path):
    path = tmp_path / "workingset.toml"
    path.write_text(_cfg(2, reserve_gib=30.0).dumps("toml"), encoding="utf-8")
    log = tmp_path / "vllm.log"
    log.write_text(_log(), encoding="utf-8")

    def run(reserve):
        args = type("A", (), {"config": str(path), "runs": [], "vllm_log": str(log),
                              "write": True, "reserve": reserve})()
        assert C.cmd_calibrate(args) == 0
        return load_config(path)

    back = run("terms")
    cal = back.calibration
    assert cal.reserve_gib is None and "reserve_gib" not in cal.fit
    assert cal.gpu_total_gib == pytest.approx(140.4)
    assert cal.non_torch_gib == pytest.approx(1.0)
    assert cal.fit["activation_gib"]["status"] == "measured"
    back.validate()
    terms = back.to_topology().reserve_terms
    m = back.to_model()
    per_gpu = 0.90 * 140.4 * G - 3.0 * G - 1.0 * G
    want = (2 * per_gpu - M.hbm_weights(m)) / M._replicated(m, back.to_topology()).kv_bpt
    assert terms is not None
    assert M.kv_pool_tokens(m, back.to_topology()) == pytest.approx(want, rel=1e-6)
    # back to the single reserve: the terms are cleared
    back = run("pool")
    assert back.calibration.reserve_gib is not None
    assert back.calibration.activation_gib is None
    assert "activation_gib" not in back.calibration.fit
    assert back.to_topology().reserve_terms is None


def test_the_token_figure_is_concurrency_times_max_model_len(tmp_path):
    """vLLM 0.33 prints concurrency x max_model_len as its pool, with each
    request's fixed pages (recurrent state, windows) inside the concurrency;
    calibrate says how much of the fitted reserve that is."""
    cfg = _cfg(2)
    m, topo = cfg.to_model(), cfg.to_topology()
    kv = round(0.90 * 140.4, 2) - round(14.4 + 1.0, 2) - 3.0
    L, fixed = 262_144, 1.0 * G
    conc = round(kv * 2 * G / (L * M._replicated(m, topo).kv_bpt + fixed), 2)
    log = tmp_path / "vllm.log"
    log.write_text(_log(concurrency=conc, tokens=int(conc * L)), encoding="utf-8")
    notes = []
    C.calibrate(cfg, [], str(log), notes)
    note = next(n for n in notes if n.startswith("vLLM's token figure"))
    assert f"{conc:g} requests x 262,144 tokens" in note
    assert "charges each request 1.00 GiB" in note
