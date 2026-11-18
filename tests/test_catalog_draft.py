"""`ws catalog`: the draft reproduces the catalogue from each model's public
repo, withholds what it cannot derive, and fails loudly on what it cannot read.

The fixtures under tests/fixtures/hf/ are each repo's config.json and its
safetensors headers reduced to a census (name pattern, dtype, shape, count),
written by `ws catalog REPO --save tests/fixtures/hf`. The live test re-reads
one repo and is skipped unless WS_NETWORK=1.
"""
from __future__ import annotations

import copy
import json
import os
import struct
from pathlib import Path

import httpx
import pytest

from workingset import hf
from workingset import model as M
from workingset.catalog_draft import draft, render
from workingset.cli import main
from workingset.vllm_tp import accepted_tps, dcp_refusals, tp_facts, tp_refusals

FIX = Path(__file__).parent / "fixtures" / "hf"

# catalogue key -> the public repo its constants were measured on
REPOS = {
    "27B": "Qwen/Qwen3.8-27B-FP8",
    "35BA3B": "Qwen/Qwen3.6-35B-A3B-FP8",
    "MM35": "mistralai/Mistral-Medium-3.5-128B",
    "GLM52": "zai-org/GLM-5.3",
    "DSV4F": "deepseek-ai/DeepSeek-V4-Flash-0731",
    "DSV4FV": "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp",
    "DSV41F": "deepseek-ai/DeepSeek-V4.1-Flash",
    "Q38FN": "Qwen/Qwen3.8-Flash-Next-FP8",
    "GLM53F": "zai-org/GLM-5.3-Flash",
    "ML4": "mistralai/Mistral-Large-4-1T-A52B",
    "K31P": "moonshotai/Kimi-K3.1-Pro",
}

# Where the draft and the catalogue differ, and why. Every other printed
# field must equal the catalogue's value exactly.
EXPLAINED = {
    # owner decision: the Qwens are priced at 1M via YaRN over 262,144 native
    ("27B", "max_ctx"): "yarn",
    ("35BA3B", "max_ctx"): "yarn",
    ("Q38FN", "max_ctx"): "yarn",
    # the baseline's stated as-deployed 28.8 GiB; the checkpoint is 30.87e9,
    # within 0.2% (catalog.toml says so)
    ("27B", "w_resident"): 0.002,
    # the catalogue's "~35.5B params x 1B FP8" estimate. The checkpoint is
    # 37.45e9 (BF16 attention, DeltaNet, embeddings and a 0.9 GB vision
    # tower), so the catalogue holds 1.95 GB less than vLLM loads: a real gap,
    # left for a modelling change rather than papered over here
    ("35BA3B", "w_resident"): 0.056,
    # rounded to four significant figures in the catalogue
    ("MM35", "w_resident"): 1e-4,
    ("GLM52", "w_resident"): 2e-4,
    ("DSV4F", "w_resident"): 1e-5,
    # the catalogue adds the DSpark drafter, a separate repo
    # (Inferact/Kimi-K3.1-Pro-DSpark-preview, 5,373,504,514 B)
    ("K31P", "w_resident"): "drafter",
}


def _draft(key):
    return draft(hf.load_fixture(hf.fixture_dir(FIX, REPOS[key])), key)


@pytest.mark.parametrize("key", REPOS)
def test_draft_reproduces_the_catalogue(key):
    d = _draft(key)
    m = M.MODELS[key]
    for name, (value, _comment) in d.fields.items():
        if name == "name":
            continue
        want = getattr(m, name)
        why = EXPLAINED.get((key, name))
        if why is None:
            assert value == want, f"{key}.{name}: draft {value}, catalogue {want}"
        elif isinstance(why, float):
            assert abs(value / want - 1) <= why, f"{key}.{name}: {value} vs {want}"
        elif why == "yarn":
            assert (value, want) == (262_144, 1_048_576)
        elif why == "drafter":
            assert want - value == 5_373_504_514
    # a field the draft cannot derive is never printed
    assert not set(d.fields) & {n for n, *_ in d.todo}


@pytest.mark.parametrize("key", ["DSV41F", "Q38FN"])
def test_engram_tables_are_named_not_guessed(key):
    """The tables vLLM offloads are a TODO, but the bytes it quotes are the
    catalogue's to the byte."""
    d = _draft(key)
    m = M.MODELS[key]
    assert "w_resident" not in d.fields
    assert d.measured["engram"] == m.w_engram
    assert d.measured["total"] - d.measured["engram"] == m.w_resident


def test_what_the_draft_withholds():
    todo = {k: {n for n, *_ in _draft(k).todo} for k in REPOS}
    # latent, compressed and indexed caches never get a kv_bpt
    for k in ("GLM52", "DSV4F", "DSV41F", "Q38FN", "GLM53F", "ML4", "K31P"):
        assert "kv_bpt" in todo[k], k
    # recurrent state and the MoE split are always a person's
    for k in ("27B", "35BA3B", "Q38FN", "GLM53F", "K31P"):
        assert {"rec_layers", "deltanet_state"} <= todo[k], k
    for k in ("35BA3B", "GLM52", "DSV4F", "ML4", "K31P"):
        assert {"w_decode_shared", "w_route_pertok", "w_route_total"} <= todo[k], k
    # a DSpark drafter's windows are state: Mistral-Large-4 has no recurrence
    # but must not be told deltanet_state = 0
    assert "deltanet_state" in todo["ML4"]
    # no draft module in the repo is not "no draft": Kimi's is a separate repo
    assert "draft_layers" in todo["K31P"]


def _mm35():
    return hf.load_fixture(hf.fixture_dir(FIX, REPOS["MM35"]))


def test_unrecognised_layer_type_gets_no_kv_bpt():
    ck = _mm35()
    t = ck.config["text_config"]
    t["layer_types"] = ["full_attention"] * 87 + ["chunked_attention"]
    d = draft(ck)
    assert "kv_bpt" not in d.fields and "attn_layers" not in d.fields
    assert "chunked_attention" in render(d)


def test_unrecognised_cache_field_gets_no_kv_bpt():
    ck = _mm35()
    ck.config["text_config"]["num_kv_shared_layers"] = 20
    d = draft(ck)
    assert "kv_bpt" not in d.fields
    assert "num_kv_shared_layers" in render(d)


def test_missing_layer_count_stops():
    ck = _mm35()
    del ck.config["text_config"]["num_hidden_layers"]
    with pytest.raises(ValueError, match="num_hidden_layers"):
        draft(ck)


def test_unknown_dtype_stops():
    ck = _mm35()
    ck.census.append(["x.weight", "F4_E2M1", [4], 1])
    with pytest.raises(hf.HFError, match="F4_E2M1"):
        draft(ck)


def test_cli_prints_a_draft(capsys):
    assert main(["catalog", "--fixture", str(hf.fixture_dir(FIX, REPOS["27B"])),
                 "--key", "Q27"]) == 0
    out = capsys.readouterr().out
    assert "[models.Q27]" in out and "kv_bpt = 32_768" in out
    assert "accepts --tensor-parallel-size 1, 2, 4, 8\n" in out


# ---- the reader --------------------------------------------------------------

def _safetensors(tensors: dict) -> bytes:
    hdr = json.dumps({k: {"dtype": dt, "shape": sh, "data_offsets": [0, 0]}
                      for k, (dt, sh) in tensors.items()}).encode()
    return struct.pack("<Q", len(hdr)) + hdr


def _mock(monkeypatch, handler):
    monkeypatch.setattr(hf, "_client", lambda: httpx.Client(
        transport=httpx.MockTransport(handler), follow_redirects=True))


def test_reader_takes_headers_by_range(monkeypatch):
    blob = _safetensors({"model.layers.0.w": ("BF16", [2, 3]),
                         "model.layers.1.w": ("BF16", [2, 3])})

    def handler(req):
        p = req.url.path
        if p.startswith("/api/models/"):
            return httpx.Response(200, json={"sha": "abc", "siblings": [
                {"rfilename": "config.json"}, {"rfilename": "model.safetensors"}]})
        if p.endswith("config.json"):
            return httpx.Response(200, json={"num_hidden_layers": 2})
        lo, hi = map(int, req.headers["Range"].removeprefix("bytes=").split("-"))
        return httpx.Response(206, content=blob[lo:hi + 1])

    _mock(monkeypatch, handler)
    ck = hf.read_repo("org/m")
    assert ck.revision == "abc"
    assert ck.census == [["model.layers.#.w", "BF16", [2, 3], 2]]
    assert ck.total_bytes() == 24


@pytest.mark.parametrize("status", [401, 403])
def test_gated_repo_names_the_token(monkeypatch, status):
    _mock(monkeypatch, lambda req: httpx.Response(status))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(hf.HFError, match="HF_TOKEN"):
        hf.read_repo("org/gated")


def test_two_layouts_pick_the_transformers_index():
    files = ["consolidated.safetensors.index.json", "model.safetensors.index.json"]
    assert hf.pick_weights(files) == ("model.safetensors.index.json", None)
    with pytest.raises(hf.HFError, match="--index"):
        hf.pick_weights(["a.safetensors.index.json", "b.safetensors.index.json"])


@pytest.mark.skipif(not os.environ.get("WS_NETWORK"), reason="reads Hugging Face; WS_NETWORK=1")
def test_live_repo_matches_its_fixture():
    fx = hf.load_fixture(hf.fixture_dir(FIX, REPOS["MM35"]))
    live = hf.read_repo(fx.repo, fx.revision)
    assert live.census == fx.census and live.config == fx.config


# ---- vLLM's TP rules ---------------------------------------------------------

def _facts(key):
    return tp_facts(hf.load_fixture(hf.fixture_dir(FIX, REPOS[key])).config)


@pytest.mark.parametrize("key", REPOS)
def test_one_node_tp_sizes_vllm_accepts(key):
    """Within one 8-GPU node vLLM 0.33 accepts the powers of two for every
    catalogue model, and no odd width."""
    assert [tp for tp in accepted_tps(_facts(key), 8)] == [1, 2, 4, 8]


def test_tp_rules_name_the_dimension():
    # Mistral-Medium-3.5: 96 heads divide by 3, its 8 KV heads do not
    assert tp_refusals(_facts("MM35"), 3)[0] == "8 KV heads do not divide by 3"
    # Qwen3.6-35B-A3B at TP8 replicates its 2 KV heads: accepted
    assert tp_refusals(_facts("35BA3B"), 8) == []
    # MLA models skip the KV rule (GLM-5.3 states 64 KV heads)
    assert _facts("GLM52").kv_heads is None
    # expert parallelism lifts the expert-width rule only
    f = _facts("35BA3B")
    assert any("moe_intermediate_size" in r for r in tp_refusals(f, 3))
    assert not any("moe_intermediate_size" in r
                   for r in tp_refusals(f, 3, expert_parallel=True))


@pytest.mark.parametrize("key", REPOS)
def test_the_study_dcp_layout_is_one_vllm_accepts(key):
    """The "dcp" layout runs --decode-context-parallel-size tp // kv_heads
    (model.dcp_size); vLLM must accept it at every TP it accepts."""
    f = _facts(key)
    m = M.MODELS[key]
    for tp in accepted_tps(f, 8):
        topo = M.topology_grid(1, tp, "B300")
        assert dcp_refusals(f, tp, M.dcp_size(m, topo)) == [], (key, tp)
