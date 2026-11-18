"""The tensor-parallel sizes vLLM accepts for a model, from its config.json.

vLLM splits attention heads, MLP and expert widths, the vocabulary and the
linear-attention heads evenly over the TP ranks, and refuses a
--tensor-parallel-size that does not divide them. Each rule below cites the
line of vLLM v0.33.0 (commit 4ba787ec) that raises; research/kv_tp_sharding.md
has the table.

Not checked, and why:
- FP8 block-scale alignment of a TP shard (fp8_utils.py:1296-1372): in
  0.33 the MoE path refines the scale grid or pads the shard to whole
  blocks depending on the kernel backend
  (fused_moe/oracle/fp8.py:resolve_fp8_moe_weight_block_shape), so whether
  a width loads is a property of the backend, not of the config.
  Qwen3.8-Flash-Next's recipe note (TP8 needs expert parallelism) is that
  case and stays a recipe note.
- Architecture-specific splits (DeepSeek-V4's grouped output projection,
  indexer heads, vision towers): only the shared layers are checked, so an
  accepted size is necessary, not sufficient, on a model vLLM implements
  with its own layers.
"""
from __future__ import annotations

from dataclasses import dataclass

VLLM_VERSION = "0.33.0"
VOCAB_PAD = 64   # DEFAULT_VOCAB_PADDING_SIZE, vocab_parallel_embedding.py:36


# where a wrapper keeps its language model's config, in the order vLLM 0.33
# finds it: transformers' get_text_config (text_config), vLLM's
# _CONFIG_ATTRS_MAPPING llm_config -> text_config (InternVL, NVLM;
# transformers_utils/config.py:195) and an Omni model's thinker
# (thinker_config.text_config, config.py:744-748; the registry serves
# Qwen2_5OmniModel as its thinker)
NESTED_TEXT = (("text_config",), ("llm_config",), ("thinker_config", "text_config"))

# the names a config class gives the standard keys (its attribute_map, or the
# attribute its model code reads), filled in only where the standard key is
# absent: ChatGLM num_layers, seq_length, padded_vocab_size, ffn_hidden_size
# (vLLM configs/chatglm.py:11-14, 54; models/chatglm.py:158, 331); GPT-2
# n_layer, n_head, n_positions (configuration_gpt2.py:71-76); OLMo's hf_olmo
# and DCLM's openlm n_layers, n_heads, n_kv_heads, max_sequence_length or
# seq_len; EXAONE-3 num_layers. A layer count read this way is checked
# against the shard headers by `ws catalog`.
ALIASES = {
    "num_hidden_layers": ("num_layers", "n_layers", "n_layer"),
    "num_attention_heads": ("n_heads", "n_head"),
    "num_key_value_heads": ("n_kv_heads",),
    "max_position_embeddings": ("seq_length", "max_sequence_length", "n_positions",
                                "seq_len"),
    "vocab_size": ("padded_vocab_size",),
    "intermediate_size": ("ffn_hidden_size",),
}


def text_config(config: dict) -> dict:
    """The language model's config, with the standard key names.

    A copy: `_path` says where it was found ("" for the top level) and
    `_read_as` maps each standard key filled from an alias to that alias."""
    t, path = config, ""
    for keys in NESTED_TEXT:
        c = config
        for k in keys:
            c = c.get(k) if isinstance(c, dict) else None
        if isinstance(c, dict) and c:
            t, path = c, ".".join(keys)
            break
    out = dict(t)
    read_as = {}
    for std, names in ALIASES.items():
        if out.get(std) is None:
            for a in names:
                if out.get(a) is not None:
                    out[std] = out[a]
                    read_as[std] = a
                    break
    out["_path"], out["_read_as"] = path, read_as
    return out


@dataclass(frozen=True)
class TPFacts:
    """The sizes vLLM divides by the TP width, read from config.json."""
    heads: int
    kv_heads: int | None          # None: MLA (one latent per token, no KV split)
    ffn: dict                     # name -> width split over ranks (MLP, experts)
    vocab: int
    linear_heads: dict            # name -> linear-attention head count


def kv_heads_of(t: dict) -> tuple:
    """(KV head count, the config fact it comes from) of a non-MLA text config.

    Falcon names it num_kv_heads under new_decoder_architecture and has one
    with multi_query (vLLM models/falcon.py:116-121). An absent or null
    num_key_value_heads is one per query head: transformers' configs set it
    so (configuration_llama.py:89, configuration_phi.py:85 in 5.20.2), and
    vLLM reads that config."""
    heads = t.get("num_attention_heads")
    if isinstance(t.get("block_configs"), list):
        # Nemotron-NAS (DeciLM): each layer's attention has its own
        # n_heads_in_group, and some layers have none (no_op)
        raise ValueError("config.json: block_configs sets attention per layer "
                         "(n_heads_in_group, no_op): the KV heads and the "
                         "attention layers differ by layer, which `ws catalog` "
                         "and the TP check do not model")
    if t.get("model_type") == "falcon":
        if t.get("new_decoder_architecture"):
            kv, how = t.get("num_kv_heads"), "num_kv_heads (Falcon new_decoder_architecture)"
        elif t.get("multi_query", True):
            kv, how = 1, "Falcon multi_query: one KV head"
        else:
            kv, how = heads, "Falcon without multi_query: one KV head per query head"
    elif t.get("model_type") == "chatglm" and "multi_query_attention" in t:
        # vLLM models/chatglm.py:61-65
        kv, how = ((t.get("multi_query_group_num"),
                    "multi_query_group_num (ChatGLM multi_query_attention)")
                   if t["multi_query_attention"] else
                   (heads, "ChatGLM without multi_query_attention: one per query head"))
    elif t.get("num_key_value_heads") is None:
        kv, how = heads, ("num_key_value_heads is " + ("null" if "num_key_value_heads" in t
                          else "absent") + ": one per query head (MHA)")
    else:
        kv, how = t["num_key_value_heads"], t.get("_read_as", {}).get(
            "num_key_value_heads", "num_key_value_heads")
    if not isinstance(kv, int) or isinstance(kv, bool) or kv <= 0:
        raise ValueError(f"config.json: the KV head count ({how}) is {kv!r}")
    return kv, how


def tp_facts(config: dict) -> TPFacts:
    t = text_config(config)
    for k in ("num_attention_heads", "vocab_size"):
        if not isinstance(t.get(k), int) or t[k] <= 0:
            raise ValueError(f"config.json: {k} missing or not a positive integer "
                             f"({t.get(k)!r}); the TP rules need it")
    # vLLM's use_mla (config/model.py:2068-2081) is a list of model types
    # with kv_lora_rank set, or compress_ratios for DeepSeek-V4
    # (transformers_utils/model_arch_config_convertor.py:294-327); then
    # get_num_kv_heads returns 1 (config/model.py:1673-1675). The key test
    # alone is used here: a type missing from 0.33's list (a model shipped in
    # its own image) is served by that image's MLA path.
    mla = t.get("kv_lora_rank") is not None or bool(t.get("compress_ratios"))
    kv = None if mla else kv_heads_of(t)[0]
    ffn = {k: t[k] for k in ("intermediate_size", "moe_intermediate_size",
                             "shared_expert_intermediate_size")
           if isinstance(t.get(k), int) and t[k] > 0}
    lin = {k: t[k] for k in ("linear_num_key_heads", "linear_num_value_heads")
           if isinstance(t.get(k), int)}
    kda = t.get("linear_attn_config")
    if isinstance(kda, dict) and "num_heads" in kda:
        lin["linear_attn_config.num_heads"] = kda["num_heads"]
    return TPFacts(t["num_attention_heads"], kv, ffn, t["vocab_size"], lin)


def pad_vocab(v: int) -> int:
    return -(-v // VOCAB_PAD) * VOCAB_PAD


def tp_refusals(f: TPFacts, tp: int, expert_parallel: bool = False) -> list:
    """Why vLLM refuses --tensor-parallel-size tp; empty when it accepts it."""
    if tp < 1:
        raise ValueError(f"tp must be >= 1, got {tp}")
    out = []
    # config/model.py:1557-1564 (verify_with_parallel_config)
    if f.heads % tp:
        out.append(f"{f.heads} attention heads do not divide by {tp}")
    # layers/linear.py:1065-1070 (QKVParallelLinear): KV heads shard when
    # tp < kv, replicate when tp >= kv; either way one divides the other
    if f.kv_heads is not None:
        if tp >= f.kv_heads and tp % f.kv_heads:
            out.append(f"TP {tp} is not a multiple of the {f.kv_heads} KV heads")
        if tp < f.kv_heads and f.kv_heads % tp:
            out.append(f"{f.kv_heads} KV heads do not divide by {tp}")
    for name, width in f.ffn.items():
        # routed experts under --enable-expert-parallel are not TP-split
        # (fused_moe/config.py:1211-1251); the shared expert still is
        if expert_parallel and name == "moe_intermediate_size":
            continue
        # dense MLP: layers/linear.py:497 (ColumnParallelLinear divide);
        # experts: fused_moe/config.py:1352
        if width % tp:
            out.append(f"{name} {width} does not divide by {tp}")
    # vocab_parallel_embedding.py:95 pads to 64, :111 divides by the ranks
    if pad_vocab(f.vocab) % tp:
        padded = "" if pad_vocab(f.vocab) == f.vocab else f" (padded to {pad_vocab(f.vocab)})"
        out.append(f"vocabulary {f.vocab}{padded} does not divide by {tp}")
    # Qwen GatedDeltaNet: mamba/gdn/qwen_gdn_linear_attn.py:651 (value
    # heads) and :946 (the key/value projection split by rank); KDA:
    # mamba/gdn/kimi_gdn_linear_attn.py:189-190
    for name, heads in f.linear_heads.items():
        if heads % tp:
            out.append(f"{name} {heads} does not divide by {tp}")
    return out


def dcp_refusals(f: TPFacts, tp: int, dcp: int) -> list:
    """Why vLLM refuses --decode-context-parallel-size dcp at this TP."""
    out = []
    if dcp <= 1:
        return out
    # config/parallel.py:597-600
    if tp % dcp:
        out.append(f"TP {tp} does not divide by DCP {dcp}")
    if f.kv_heads is not None:   # GQA / MQA: config/model.py:1579-1609
        if tp <= f.kv_heads:
            out.append(f"DCP needs TP {tp} above the {f.kv_heads} KV heads")
        elif dcp > tp // f.kv_heads:
            out.append(f"DCP {dcp} exceeds TP {tp} // {f.kv_heads} KV heads")
        q_per_kv = f.heads // f.kv_heads
        if q_per_kv % dcp:
            out.append(f"{q_per_kv} query heads per KV head do not divide by DCP {dcp}")
    return out


def accepted_tps(f: TPFacts, upto: int = 16, expert_parallel: bool = False) -> list:
    return [tp for tp in range(1, upto + 1) if not tp_refusals(f, tp, expert_parallel)]
