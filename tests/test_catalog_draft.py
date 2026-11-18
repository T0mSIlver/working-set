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


# ---- minimal configs from the atlas sweep (open-weights-atlas, 2026-11-18) ----

_BASE = {"num_hidden_layers": 4, "num_attention_heads": 8, "num_key_value_heads": 2,
         "head_dim": 16, "vocab_size": 128, "max_position_embeddings": 4096}


def _ck(extra=None, census=(("model.layers.#.mlp.up_proj.weight", "BF16", [64, 16], 4),)):
    return hf.Checkpoint("org/m", "0" * 40, {**_BASE, **(extra or {})},
                         [list(r) for r in census], 1, None)


def test_a_plain_config_drafts_its_cache():
    d = draft(_ck())
    assert d.fields["kv_bpt"][0] == 4 * 2 * 16 * 2
    assert d.fields["w_route_pertok"][0] == 0.0 and d.fields["deltanet_state"][0] == 0.0


@pytest.mark.parametrize("extra, named", [
    # LFM2-1.2B: attention layers listed by index, the rest short convolutions
    ({"full_attn_idxs": [1, 3], "conv_L_cache": 3}, "full_attn_idxs"),
    # Granite-4.0-H: Mamba2 and attention layer types
    ({"layer_types": ["mamba", "attention", "mamba", "mamba"], "mamba_d_state": 128},
     "mamba"),
    # Ling-3.0-tiny: KDA groups of layer_group_size around MLA layers
    ({"kda_safe_gate": True, "layer_group_size": 4, "short_conv_kernel_size": 4},
     "layer_group_size"),
    # MiMo-V2.6-Pro: hybrid_layer_pattern, 0 full and 1 sliding-window
    ({"hybrid_layer_pattern": [0, 1, 1, 1], "sliding_window_size": 128},
     "hybrid_layer_pattern"),
    # Inkling-2-Small: local_layer_ids and short convolutions, no model_type
    ({"local_layer_ids": [0, 1, 2], "sconv_kernel_size": 4,
      "sliding_window_size": 1024}, "local_layer_ids"),
])
def test_a_hybrid_layout_it_cannot_read_is_not_all_attention(extra, named):
    """Granite-4.0-H-Small and LFM2-1.2B drafted every layer as attention or
    printed rec_layers = 0 and deltanet_state = 0.0 for Mamba / conv layers."""
    d = draft(_ck(extra))
    assert not {"attn_layers", "rec_layers", "deltanet_state", "kv_bpt"} & set(d.fields)
    assert named in render(d)


@pytest.mark.parametrize("extra, census", [
    # Mixtral, gpt-oss, Granite-MoE name the count num_local_experts
    ({"num_local_experts": 8, "num_experts_per_tok": 2}, None),
    # gpt-oss also says experts_per_token
    ({"num_local_experts": 128, "experts_per_token": 4}, None),
    # a top-k without a count, and expert tensors without either
    ({"num_experts_per_tok": 2}, None),
    ({}, (("model.layers.#.block_sparse_moe.input_linear.weight", "BF16", [8, 16], 4),)),
])
def test_a_mixture_of_experts_is_never_drafted_dense(extra, census):
    """gpt-oss-120b and Granite-4.0-H-Small were drafted w_route_pertok = 0.0
    "dense: no routed experts"."""
    d = draft(_ck(extra, census) if census else _ck(extra))
    assert not {"w_route_pertok", "w_route_total"} & set(d.fields)
    assert "w_route_pertok" in {n for n, *_ in d.todo}


def test_an_unused_sliding_window_does_not_withhold_the_cache():
    """Qwen2.5 ships sliding_window 131072 with use_sliding_window false;
    transformers drops the window, so the cache is plain K/V."""
    qwen = {"model_type": "qwen2", "use_sliding_window": False,
            "sliding_window": 131072, "max_window_layers": 28}
    d = draft(_ck(qwen))
    assert d.fields["kv_bpt"][0] == 256 and d.fields["deltanet_state"][0] == 0.0
    # Fara-7B: the flag sits in a Qwen2.5-VL text_config
    vl = {"model_type": "qwen2_5_vl", "text_config": {**_BASE, **qwen, "sliding_window": 1024,
                                                      "model_type": "qwen2_5_vl_text"}}
    assert draft(hf.Checkpoint("org/m", "0" * 40, vl, [], 1, None)).fields["kv_bpt"][0] == 256
    d = draft(_ck({**qwen, "use_sliding_window": True, "sliding_window": 1024}))
    assert "kv_bpt" not in d.fields and "deltanet_state" not in d.fields


def _proj(name, rows, cols=128, n=4):
    return (f"model.layers.#.self_attn.{name}.weight", "BF16", [rows, cols], n)


@pytest.mark.parametrize("extra, census, kv_heads, hd", [
    # Llama-style separate projections, no head_dim (Mistral-7B, Yi, SmolLM2)
    ({}, [_proj("q_proj", 128), _proj("k_proj", 32)], 2, 16),
    # Phi-3: packed qkv_proj, no num_key_value_heads (MHA)
    ({"num_key_value_heads": None}, [_proj("qkv_proj", (8 + 16) * 16)], 8, 16),
    # Falcon-7B: multi_query, packed query_key_value
    ({"model_type": "falcon", "multi_query": True, "num_key_value_heads": None},
     [("transformer.h.#.self_attention.query_key_value.weight", "BF16", [160, 128], 4)],
     1, 16),
    # Falcon-40B: new_decoder_architecture names num_kv_heads
    ({"model_type": "falcon", "new_decoder_architecture": True, "num_kv_heads": 2,
      "num_key_value_heads": None},
     [("transformer.h.#.self_attention.query_key_value.weight", "BF16", [192, 128], 4)],
     2, 16),
    # DCLM-7B (open_lm): attention.in_proj
    ({"num_key_value_heads": None},
     [("model.layers.#.attention.in_proj.weight", "F32", [3 * 128, 128], 4)], 8, 16),
    # Baichuan's W_pack, Molmo's att_proj (MHA, no num_key_value_heads)
    ({"num_key_value_heads": None}, [_proj("W_pack", 3 * 128)], 8, 16),
    ({}, [("model.transformer.blocks.#.att_proj.weight", "F32", [(8 + 4) * 16, 128], 4)],
     2, 16),
    # GPT-2 Conv1D c_attn stored [in, out]
    ({"num_key_value_heads": None},
     [("transformer.h.#.attn.c_attn.weight", "BF16", [128, 3 * 128], 4)], 8, 16),
    # MiniCPM-o: Whisper and TTS projections beside the decoder's
    ({}, [_proj("q_proj", 128), _proj("k_proj", 32),
          ("apm.layers.#.self_attn.k_proj.weight", "BF16", [32, 32], 3),
          ("tts.model.layers.#.self_attn.k_proj.weight", "BF16", [24, 24], 2)], 2, 16),
    # another tower's projections, other width and count: the decoder's are
    # the ones found once per layer (MiniCPM-V's SigLIP before vpm was named)
    ({}, [_proj("q_proj", 128), _proj("k_proj", 32),
          ("tower.layers.#.self_attn.q_proj.weight", "BF16", [72, 72], 3),
          ("tower.layers.#.self_attn.k_proj.weight", "BF16", [72, 72], 3)], 2, 16),
])
def test_head_size_is_measured_from_the_projections(extra, census, kv_heads, hd):
    """No head_dim in config.json (most 2023-2024 models): the head size
    comes from the key or packed qkv projection rows in the shard headers,
    and the KV heads from the config's own convention."""
    d = draft(_ck({"head_dim": None, **extra}, census))
    assert d.fields["kv_heads"][0] == kv_heads
    assert d.fields["kv_bpt"][0] == 4 * kv_heads * hd * 2
    assert "measured" in d.fields["kv_bpt"][1]


def test_a_head_dim_the_projections_contradict_gets_no_kv_bpt():
    d = draft(_ck({"head_dim": 32}, [_proj("q_proj", 128), _proj("k_proj", 32)]))
    assert "kv_bpt" not in d.fields
    assert "config head_dim 32 but key projection 32 rows / 2 KV heads = 16" in render(d)


def test_no_head_dim_and_no_projection_gets_no_kv_bpt():
    d = draft(_ck({"head_dim": None}))
    assert "kv_bpt" not in d.fields and "no q/k or packed qkv projection" in render(d)


def test_a_null_kv_head_count_is_one_per_query_head():
    """phi-1_5 ships num_key_value_heads: null; transformers reads it as MHA."""
    d = draft(_ck({"num_key_value_heads": None}))
    assert d.fields["kv_heads"][0] == 8 and d.fields["kv_bpt"][0] == 4 * 8 * 16 * 2


@pytest.mark.parametrize("cfg, match", [
    # Mamba-Codestral-7B: no attention heads
    ({"model_type": "mamba2", "num_hidden_layers": 64, "state_size": 128,
      "vocab_size": 128}, "a state-space model"),
    # aya-101 (T5)
    ({**_BASE, "model_type": "t5", "is_encoder_decoder": True}, "an encoder-decoder"),
    # OpenELM: num_transformer_layers, with per-layer head counts
    ({**{k: v for k, v in _BASE.items() if k != "num_hidden_layers"},
      "num_transformer_layers": 16},
     r"no num_hidden_layers; it has \{'num_transformer_layers': 16\}"),
    # SmolVLM2: text_config leaves the head count to LlamaConfig's default
    ({"text_config": {k: v for k, v in _BASE.items() if k != "num_attention_heads"}},
     "config.json text_config: num_attention_heads is None"),
    # a nested config this command does not look in
    ({"speech_config": {"decoder_config": dict(_BASE)}},
     r"nested configs \['speech_config.decoder_config'\]"),
    # Llama-3.3-Nemotron-Super-49B: per-layer attention in block_configs
    ({**_BASE, "num_key_value_heads": None,
      "block_configs": [{"attention": {"n_heads_in_group": 8, "no_op": False}}]},
     "block_configs sets attention per layer"),
    # xgen-mm: a text_config stub without sizes
    ({"text_config": {"model_type": "phi3", "sliding_window": 2047},
      "vision_encoder_config": {}}, "config.json text_config: no layer count"),
])
def test_an_unmodelled_config_is_named(cfg, match):
    ck = hf.Checkpoint("org/m", "0" * 40, cfg, [], 1, None)
    with pytest.raises(ValueError, match=match):
        draft(ck)


def test_audio_towers_and_molmo_vision_are_measured():
    """Gemma-4-E2B's audio tower and Molmo's vision_backbone were left in
    the "dense: header total less 0 B of vision" sum."""
    d = draft(_ck({"vision_config": {"x": 1}, "audio_config": {"x": 1}},
                  census=[_proj("q_proj", 128),
                          ("model.audio_tower.layers.#.w.weight", "BF16", [8, 8], 2),
                          ("model.vision_backbone.image_projector.w", "BF16", [8, 8], 1)]))
    assert d.measured["audio"] == 2 * 64 * 2 and d.measured["vision"] == 64 * 2
    assert "unmeasured" not in render(d)


def test_a_tower_the_config_names_but_no_tensor_matches_is_said():
    d = draft(_ck({"vision_config": {"x": 1}},
                  census=[_proj("q_proj", 128), ("model.eyes.w", "BF16", [8, 8], 1)]))
    out = render(d)
    assert "the config names ['vision_config'] but no tensor name matched" in out
    assert "less 0 B" not in out


def test_minicpm_v_towers_count_as_vision():
    d = draft(_ck(census=[_proj("q_proj", 128),
                          ("vpm.encoder.layers.#.mlp.fc1.weight", "F16", [8, 8], 2),
                          ("resampler.kv_proj.weight", "F16", [8, 8], 1)]))
    assert d.measured["vision"] == 3 * 64 * 2


@pytest.mark.parametrize("window", [128, None])
def test_sliding_window_layers_are_attention_with_a_window_not_recurrence(window):
    """gpt-oss-120b and Gemma 3 mark layers "sliding_attention": the draft
    asked for recurrent layers to be counted by hand. They are attention with
    a fixed window: no recurrence, a window state, no plain per-token cache."""
    d = draft(_ck({"layer_types": ["sliding_attention", "full_attention"] * 2,
                   "sliding_window": window}))
    assert d.fields["rec_layers"][0] == 0
    assert not {"attn_layers", "kv_bpt", "deltanet_state"} & set(d.fields)
    assert "2 full-attention and 2 sliding-window layers" in render(d)


def test_a_window_as_long_as_the_context_is_plain_kv():
    """Phi-4-mini: sliding_window 262144 over max_position_embeddings 131072."""
    d = draft(_ck({"sliding_window": 262144, "max_position_embeddings": 131072}))
    assert d.fields["kv_bpt"][0] == 4 * 2 * 16 * 2
    assert d.fields["deltanet_state"][0] == 0.0
    assert "never binds" in d.fields["kv_bpt"][1]


def test_typed_sliding_layers_stay_windowed_at_any_length():
    """Ministral-8B-2410 types its sliding layers with a window equal to
    max_position_embeddings (32768) but is served at 128k."""
    d = draft(_ck({"layer_types": ["full_attention", "sliding_attention"] * 2,
                   "sliding_window": 4096}))
    assert not {"attn_layers", "kv_bpt", "deltanet_state"} & set(d.fields)


def test_cross_attention_layers_hold_no_per_token_cache():
    """Llama-3.2-11B-Vision: 8 of 40 layers cross-attend the image tokens;
    the draft counted them in kv_bpt (81,920 for 65,536)."""
    d = draft(_ck({"cross_attention_layers": [1]}))
    assert d.fields["attn_layers"][0] == 3 and d.fields["kv_bpt"][0] == 3 * 2 * 16 * 2
    assert "deltanet_state" not in d.fields


@pytest.mark.parametrize("extra", [
    {"sliding_window": 1024},                                 # Mistral-7B-v0.1
    {"sliding_window": 1024, "sliding_window_pattern": 6},    # Gemma 3
])
def test_a_window_without_layer_types_leaves_the_attention_layers_open(extra):
    """Gemma-3-12B was drafted attn_layers = 48 though 40 of its layers are
    sliding-window."""
    d = draft(_ck(extra))
    assert not {"attn_layers", "kv_bpt", "deltanet_state"} & set(d.fields)
    assert "no layer_types to say which" in render(d)


def test_cross_layer_attention_gets_no_kv_bpt():
    """Hunyuan's use_cla shares one cache between cla_share_factor layers;
    use_cla false (Hunyuan-1.8B) is plain K/V."""
    assert "kv_bpt" in draft(_ck({"use_cla": False, "cla_share_factor": 2})).fields
    d = draft(_ck({"use_cla": True, "cla_share_factor": 2}))
    assert "kv_bpt" not in d.fields and "use_cla" in render(d)


_NO_LAYERS = {k: v for k, v in _BASE.items() if k != "num_hidden_layers"}


@pytest.mark.parametrize("cfg", [
    # OLMo-7B (hf_olmo), DCLM-7B (openlm): n_layers and n_heads
    {**{k: v for k, v in _NO_LAYERS.items() if k != "num_attention_heads"},
     "n_layers": 4, "n_heads": 8},
    # EXAONE-3: num_layers
    {**_NO_LAYERS, "num_layers": 4},
    # InternVL, NVLM: llm_config; Qwen2.5-Omni: thinker_config.text_config
    {"llm_config": dict(_BASE), "vision_config": {"num_hidden_layers": 27}},
    {"thinker_config": {"text_config": dict(_BASE)}, "talker_config": dict(_BASE)},
])
def test_a_layer_count_under_another_name_or_config_is_read(cfg):
    """The sweep refused OLMo, DCLM, EXAONE-3, chatglm, InternVL, NVLM and
    Qwen-Omni on a missing num_hidden_layers."""
    d = draft(hf.Checkpoint("org/m", "0" * 40, cfg, [_proj("q_proj", 128),
                                                      _proj("k_proj", 32)], 1, None))
    assert d.fields["layers"][0] == 4 and d.fields["kv_bpt"][0] == 4 * 2 * 16 * 2


def test_an_aliased_layer_count_the_headers_contradict_stops():
    cfg = {**_NO_LAYERS, "n_layers": 6}
    with pytest.raises(ValueError, match="n_layers = 6 is read as num_hidden_layers"):
        draft(hf.Checkpoint("org/m", "0" * 40, cfg, [_proj("q_proj", 128)], 1, None))


def test_chatglm2_reads_its_own_names():
    """chatglm2-6b: num_layers, multi_query_group_num, seq_length and
    padded_vocab_size (vLLM configs/chatglm.py, models/chatglm.py:61-65)."""
    cfg = {"model_type": "chatglm", "num_layers": 4, "num_attention_heads": 8,
           "multi_query_attention": True, "multi_query_group_num": 2,
           "seq_length": 32768, "padded_vocab_size": 128, "ffn_hidden_size": 64}
    qkv = ("transformer.encoder.layers.#.self_attention.query_key_value.weight",
           "BF16", [(8 + 4) * 16, 128], 4)
    d = draft(hf.Checkpoint("org/m", "0" * 40, cfg, [qkv], 1, None))
    assert d.fields["kv_heads"][0] == 2 and d.fields["kv_bpt"][0] == 4 * 2 * 16 * 2
    assert d.fields["max_ctx"] == (32768, "seq_length")
    assert d.fields["layers"] == (4, "num_layers")


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
    assert "vllm_tp = [1, 2, 4, 8] " in out


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


def test_a_bare_401_says_the_repo_may_not_exist(monkeypatch):
    """mosaicml/mpt-7b: a bare anonymous 401 (no GatedRepo code) was reported
    as gated; Hugging Face answers a missing repo the same way."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    _mock(monkeypatch, lambda req: httpx.Response(401))
    with pytest.raises(hf.HFError, match="does not exist, is private, or is gated"):
        hf.read_repo("org/gone")
    _mock(monkeypatch, lambda req: httpx.Response(401, headers={"X-Error-Code": "GatedRepo"}))
    with pytest.raises(hf.HFError, match="The repo is gated: set HF_TOKEN"):
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


ARMS = [(k, g, w) for k in REPOS for g in M.GPUS for w in ("fp8", "nvfp4")
        if w == "fp8" or (M.MODELS[k].nvfp4_w and M.GPUS[g].supports_nvfp4)]


def _arm(key, gpu, wdt):
    m = M.with_gpu(M.MODELS[key], gpu)
    if wdt == "nvfp4":
        m = M.with_weight_dtype(m, "nvfp4")
    if m.kv_fp8_blackwell_only and gpu == "H200":
        m = M.with_kv_dtype(m, "fp16")
    return m


@pytest.mark.parametrize("key, gpu, wdt", ARMS)
def test_every_offered_topology_is_one_vllm_accepts(key, gpu, wdt):
    """node_splits offers only widths vLLM accepts, and so does the
    explorer's Split control, which drops the rest: TP 3, 5, 6 and 7 for
    every catalogue model."""
    m = _arm(key, gpu, wdt)
    for t in M.node_splits(m, gpu, node=8):
        assert t.tp in m.vllm_tp and M.vllm_tp_refused(m, t.tp) is None, (key, t.tp)
    need = M.min_tp_for(m, gpu)
    reachable = {tp for n in range(1, 9) for tp in range(1, n + 1)
                 if n % tp == 0 and tp >= need}
    assert {tp for tp in reachable if M.vllm_tp_refused(m, tp)} == reachable & {3, 5, 6, 7}


def test_a_config_at_a_refused_width_stops(capsys):
    assert main(["init", "--model", "27B", "--gpu", "H200", "--tp", "3", "-o", "-"]) == 2
    assert "vLLM refuses tensor_parallel 3 for Qwen3.8-27B" in capsys.readouterr().err
    assert main(["init", "--model", "27B", "--gpu", "H200", "--tp", "4", "-o", "-"]) == 0


def test_drafted_provenance_names_what_is_missing_then_passes():
    """The draft labels what it read itself (weights measured off the shard
    headers, kv derived from the config, dated, with the repo as ref) and
    leaves the rest as TODOs: the completeness check names exactly those,
    and passes once they are filled."""
    import tomllib
    from workingset.provenance import ProvenanceError, check_model_row
    d = draft(_mm35(), "MM35X", read_on="2026-11-18")
    row = tomllib.loads(render(d))["models"]["MM35X"]
    prov = row["provenance"]
    assert prov["weights"]["kind"] == "measured" and prov["kv"]["basis"] == "config"
    assert "read 2026-11-18" in prov["kv"]["ref"]
    with pytest.raises(ProvenanceError,
                       match="missing decode, draft, acceptance, prefill, quality"):
        check_model_row("MM35X", row)
    for q in ("decode", "draft", "acceptance", "prefill", "quality"):
        prov[q] = {"kind": "guess", "note": "filled by hand"}
    assert set(check_model_row("MM35X", row)) == set(prov)


def test_draft_never_labels_what_it_left_to_a_person():
    """GLM-5.3-Flash: no kv_bpt drafted (latent cache), so no kv provenance."""
    d = _draft("GLM53F")
    assert "kv_bpt" not in d.fields and "kv" not in d.provenance
    assert d.provenance["weights"]["kind"] == "measured"
