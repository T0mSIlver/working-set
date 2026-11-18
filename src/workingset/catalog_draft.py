"""`ws catalog REPO`: a draft catalog.toml model table from a Hugging Face repo.

Prints every field that follows from config.json and the safetensors
headers, and a commented placeholder for every field a person must still
determine, saying what and where the sibling model's research note is. A
field the command cannot read is printed as unset with the reason; it never
falls back to a default.

    ws catalog Qwen/Qwen3.8-27B-FP8
    ws catalog deepseek-ai/DeepSeek-V4.1-Flash --key DSV41F --revision 314113f8
    ws catalog Qwen/Qwen3.8-27B-FP8 --save tests/fixtures/hf   # also write the fixture
    ws catalog --fixture tests/fixtures/hf/Qwen--Qwen3.8-27B-FP8  # no network
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field

from datetime import date

from .hf import Checkpoint, HFError, load_fixture, read_repo, save_fixture
from .provenance import MODEL_QUANTITIES
from .vllm_tp import (VLLM_VERSION, accepted_tps, pad_vocab, text_config,
                      tp_facts, tp_refusals)

# layer_types values: attention layers (each keeps a per-token cache) and
# recurrent ones (a fixed per-session state). Anything else is unrecognised.
ATTENTION_TYPES = {"full_attention", "deepseek_sparse_attention", "indexed_attention"}
RECURRENT_TYPES = {"linear_attention"}

# config keys that change what a token's cache holds beyond plain K/V heads
# (latents, compressed or indexed caches, windows, shared layers, tables).
# Any of them with a truthy value withholds kv_bpt.
_CACHE_KEY = re.compile(
    r"^(kv_lora_rank|compress_|index_|indexer_|kv_source|kv_shar|num_kv_shared|"
    r"cross_layer|attention_chunk|sliding_window|use_sliding_window|swa_|"
    r"engram_|ngram|linear_attn_config|dsa_|no_rope_layer|mla_|candidate_)")

_VISION = re.compile(r"(^|\.)(visual|vision_tower|vision_encoder|vision_model|"
                     r"multi_modal_projector|mm_projector|vision_language_adapter|"
                     r"patch_merger|pre_mm_projector_norm|aligner)\.")
_IMAGE_TOKENS = re.compile(r"^image_(start|end|newline|pad)$")
_DRAFT = re.compile(r"(^|\.)(mtp|mtp_dspark|dspark)(\.|_)")
# the hashed lookup tables themselves (DeepSeek-V4.1-Flash's Engram embed,
# Qwen3.8-Flash-Next's n-gram embedding), not the projections around them
_ENGRAM = re.compile(r"(^|\.)(engram\.embed|ngram_embedding)\.")
_ROUTED = re.compile(r"(^|\.)experts\.#\.")
# routed-expert counts as configs name them (DeepSeek, Qwen, Mixtral / gpt-oss /
# Granite, DBRX-style)
_EXPERT_COUNT = ("n_routed_experts", "num_experts", "num_local_experts", "moe_num_experts")
_EXPERT_TOPK = ("num_experts_per_tok", "num_experts_per_token", "experts_per_token",
                "moe_top_k")
# a tensor name a mixture of experts leaves even when the config names no count
_MOE_TENSOR = re.compile(r"(^|\.)(experts?|block_sparse_moe|moe|router)(\.|_)")
# config keys that lay out non-attention layers in a form _layer_counts does
# not read (LFM2's full_attn_idxs, Jamba's attn_layer_period, Nemotron-H's
# hybrid_override_pattern, Zamba's layers_block_type, Mamba / SSM state sizes)
_HYBRID_KEY = re.compile(
    r"^(full_attn_idxs|attn_layer_|attn_type_list|hybrid_override_pattern|"
    r"layers_block_type|layer_block_types|block_types|mamba_|ssm_|state_size|"
    r"conv_L_cache|conv_kernel|rwkv_|time_step)")
# model types whose transformers config drops sliding_window when
# use_sliding_window is false (transformers 5.20.2, `if self.use_sliding_window else None`)
_WINDOW_FLAG_TYPES = {"qwen2", "qwen3", "qwen3_moe", "qwen2_vl", "qwen2_5_vl",
                      "qwen2_5_omni"}

NOTES = {
    "mla": "research/model_glm52.md (MLA + indexer) and research/model_mistral_large4.md",
    "compressed": "research/model_dsv41flash.md (CSA2) and research/model_dsv4flash.md",
    "deltanet": "research/model_35ba3b.md (Qwen DeltaNet)",
    "kda": "research/model_glm53flash.md and research/model_kimi_k31pro.md (KDA)",
    "moe": "research/model_35ba3b.md (component ledger)",
    "draft_mtp": "research/model_35ba3b.md (MTP module)",
    "draft_dspark": "research/model_mistral_large4.md and research/model_dsv41flash.md (DSpark)",
    "engram": "research/model_dsv41flash.md (Engram) and research/model_qwen38flashnext.md (n-gram)",
    "vision": "research/model_dsv4flash_vision.md",
    "window": "research/model_dsv4flash.md (windows)",
    "nvfp4": "research/nvfp4.md",
    "prefill": "research/prefill.md",
    "kv_tp": "research/kv_tp_sharding.md",
}


@dataclass
class Draft:
    repo: str
    revision: str
    key: str
    fields: dict = field(default_factory=dict)       # name -> (value, comment)
    todo: list = field(default_factory=list)         # (name, what, note path)
    facts: list = field(default_factory=list)        # comment lines
    tp_ok: list = field(default_factory=list)
    tp_refused: dict = field(default_factory=dict)
    measured: dict = field(default_factory=dict)     # byte sums the TODOs quote
    provenance: dict = field(default_factory=dict)   # quantity -> inline table
    read_on: str = ""                                # the date the repo was read

    def set(self, name, value, comment=""):
        self.fields[name] = (value, comment)

    def need(self, name, what, note=None):
        self.todo.append((name, what, note))


def _gb(b: float) -> str:
    return f"{b / 1e9:,.2f} GB"


def _layer_counts(t: dict) -> tuple:
    """(attention layers, recurrent layers, unrecognised layer types)."""
    n = t.get("num_hidden_layers")
    lt = t.get("layer_types")
    if isinstance(lt, list):
        if len(lt) != n:
            raise ValueError(f"config.json: layer_types has {len(lt)} entries for "
                             f"num_hidden_layers {n}")
        unknown = sorted(set(lt) - ATTENTION_TYPES - RECURRENT_TYPES)
        attn = sum(x in ATTENTION_TYPES for x in lt)
        rec = sum(x in RECURRENT_TYPES for x in lt)
        iv = t.get("full_attention_interval")
        if iv and lt.count("full_attention") != n // iv:
            raise ValueError(f"config.json: layer_types has {lt.count('full_attention')} "
                             f"full_attention layers, full_attention_interval {iv} "
                             f"implies {n // iv}")
        return attn, rec, unknown
    lac = t.get("linear_attn_config")
    if isinstance(lac, dict) and "full_attn_layers" in lac:
        attn = len(lac["full_attn_layers"])
        rec = len(lac.get("kda_layers", []))
        if attn + rec != n:
            raise ValueError(f"config.json: linear_attn_config lists {attn} full and "
                             f"{rec} KDA layers for num_hidden_layers {n}")
        return attn, rec, []
    iv = t.get("full_attention_interval")
    if iv:
        if n % iv:
            raise ValueError(f"config.json: num_hidden_layers {n} does not divide by "
                             f"full_attention_interval {iv}")
        return n // iv, n - n // iv, []
    return n, 0, []


def draft(ck: Checkpoint, key: str = "NEW", tp_max: int = 16,
          read_on: str | None = None) -> Draft:
    cfg = ck.config
    t = text_config(cfg)
    d = Draft(ck.repo, ck.revision, key, read_on=read_on or date.today().isoformat())
    d.set("name", ck.repo.split("/")[-1], "the repo name: write the catalogue's display form")

    n = t.get("num_hidden_layers")
    if not isinstance(n, int) or n <= 0:
        raise ValueError(f"config.json: num_hidden_layers is {n!r}")
    d.set("layers", n, "num_hidden_layers")
    attn, rec, unknown = _layer_counts(t)

    # transformers sets sliding_window to None for these model types when
    # use_sliding_window is false (configuration_qwen2.py:84 in 5.20.2), and
    # vLLM reads that config: Qwen2.5 ships sliding_window 131072 unused
    unused_window = (t.get("use_sliding_window") is False
                     and (t.get("model_type") or cfg.get("model_type")) in _WINDOW_FLAG_TYPES)
    unused = {"sliding_window", "max_window_layers"} if unused_window else set()
    cache_keys = sorted(k for k, v in t.items() if _CACHE_KEY.match(k) and v not in
                        (None, False, 0, [], {}) and k not in unused)
    mla = t.get("kv_lora_rank") is not None
    compressed = bool(t.get("compress_ratios"))
    kda = isinstance(t.get("linear_attn_config"), dict)
    heads, kvh, hd = (t.get("num_attention_heads"), t.get("num_key_value_heads"),
                      t.get("head_dim"))
    hybrid = sorted(k for k, v in t.items() if _HYBRID_KEY.match(k)
                    and v not in (None, False, 0, [], {}))
    # what stops the layer census: unknown layer types, or a hybrid layout in
    # keys _layer_counts does not read (then every layer would count as attention)
    layout = (f"layer_types {unknown} are not ones this command knows "
              f"({sorted(ATTENTION_TYPES | RECURRENT_TYPES)})" if unknown
              else f"the config lays out non-attention layers with {hybrid}, which "
              f"this command does not read"
              if hybrid and not isinstance(t.get("layer_types"), list) and not kda
              else None)

    # ---- attention cache ----------------------------------------------------
    if layout:
        d.need("attn_layers", f"{layout}: count the attention layers by hand")
    elif compressed:
        d.need("attn_layers", "compressed caches (compress_ratios): which layers the "
               "prefill quadratic term prices", NOTES["compressed"])
    else:
        d.set("attn_layers", attn, "attention layers"
              + (f" of {n} ({rec} recurrent)" if rec else ""))

    if mla or compressed:
        d.set("kv_heads", 1, "one latent per token: vLLM's MLA path has 1 KV head "
              "(config/model.py:1673), replicated under plain TP")
    elif isinstance(kvh, int):
        d.set("kv_heads", kvh, f"GQA {heads}/{kvh}" if kvh != heads else "MHA")
    else:
        d.need("kv_heads", f"num_key_value_heads is {kvh!r}")

    standard = not (layout or cache_keys or mla or compressed)
    if standard and isinstance(hd, int) and hd > 0 and isinstance(kvh, int):
        d.set("attn_d", heads * hd, f"= {heads} * {hd}")
        d.set("kv_bpt", attn * kvh * hd * 2,
              f"{attn} attn layers x {kvh} KV heads x {hd} x 2(K,V) x 1B (fp8 KV)")
    else:
        why = (layout if layout
               else f"cache fields {cache_keys}" if cache_keys
               else f"head_dim is {hd!r}" if standard
               else "MLA latent cache")
        note = (NOTES["compressed"] if compressed else NOTES["mla"] if mla
                else NOTES["kv_tp"])
        d.need("kv_bpt", f"not plain K/V heads ({why}): size the per-token cache "
               f"in vLLM's fp8 layout", note)
        d.need("attn_d", "query heads x the attention width the prefill term prices",
               NOTES["prefill"])
        if mla or compressed:
            d.need("kv_decode_bpt / kv_decode_const / kv_decode_topk",
                   "what a decode step reads of the cache (indexer scan, top-k rows)",
                   NOTES["compressed"] if compressed else NOTES["mla"])

    nextn = t.get("num_nextn_predict_layers") or t.get("mtp_num_hidden_layers") or 0
    dspark = (any(k.startswith("dspark_") for k in t)
              or any(p.startswith("mtp_dspark") for p, *_ in ck.census))

    # ---- recurrent state ----------------------------------------------------
    window = None if unused_window else t.get("sliding_window")
    if layout:
        d.need("rec_layers", f"{layout}: count the recurrent layers (state-space, "
               f"short-conv, linear-attention) by hand", NOTES["deltanet"])
        d.need("deltanet_state", "per-session state of those layers"
               + (f" and of the sliding_window {window} layers" if window else ""),
               NOTES["deltanet"])
    elif rec:
        kind = "kda" if kda else "deltanet"
        d.need("rec_layers", f"the config lists {rec} linear-attention layers: "
               f"confirm the recurrent kind", NOTES[kind])
        d.need("deltanet_state", "per-session recurrent state bytes (heads x k x v "
               "x dtype + conv)", NOTES[kind])
    else:
        d.set("rec_layers", 0, "no linear-attention layers in the config")
        if window or compressed or dspark:
            d.need("deltanet_state", "fixed per-session window state"
                   + (f" (sliding_window {window})" if window else "")
                   + (" (a DSpark drafter keeps latent windows)" if dspark else ""),
                   NOTES["draft_dspark"] if dspark and not window else NOTES["window"])
        else:
            d.set("deltanet_state", 0.0, "no recurrent layers, no window")

    # ---- weights ------------------------------------------------------------
    total = ck.total_bytes()
    by = ck.bytes_by_dtype()
    d.facts.append(f"safetensors headers ({ck.shards} shards): {total:,} B = "
                   + ", ".join(f"{k} {_gb(v)}" for k, v in by.items()))
    if ck.index_total_size is not None and ck.index_total_size != total:
        d.facts.append(f"the index's metadata.total_size {ck.index_total_size:,} B "
                       f"differs from the header sum by {total - ck.index_total_size:+,} B")
    vision = ck.bytes_matching(_VISION.pattern) + ck.bytes_matching(_IMAGE_TOKENS.pattern)
    draft_b = ck.bytes_matching(_DRAFT.pattern)
    engram = ck.bytes_matching(_ENGRAM.pattern)
    d.measured.update(total=total, vision=vision, draft=draft_b, engram=engram)
    if engram:
        d.need("w_resident", f"the Engram / n-gram tables hold {engram:,} B; vLLM keeps "
               f"them in host memory by default, which leaves {total - engram:,} B "
               f"on the GPUs; confirm the tables are what this model offloads",
               NOTES["engram"])
        d.need("w_engram", f"the offloaded table bytes ({engram:,} B by tensor name)",
               NOTES["engram"])
    else:
        incl = [f"vision {_gb(vision)}"] * bool(vision) + [f"draft {_gb(draft_b)}"] * bool(draft_b)
        d.set("w_resident", total, "safetensors header total, every tensor"
              + (f" (incl. {', '.join(incl)})" if incl else ""))
    if vision:
        d.facts.append(f"vision tower and projector: {vision:,} B, resident but never "
                       f"in a text decode read ({NOTES['vision']})")

    n_exp = next((t[k] for k in _EXPERT_COUNT if t.get(k)), None)
    k_exp = next((t[k] for k in _EXPERT_TOPK if t.get(k)), None)
    moe_tensors = sorted({p for p, *_ in ck.census if _MOE_TENSOR.search(p)})
    if not n_exp and (k_exp or moe_tensors):
        # a mixture of experts whose count this command cannot read: never "dense"
        seen = (f"{k_exp} experts per token" if k_exp else
                f"tensors like {moe_tensors[0]}")
        for f_, what in (("w_decode_shared", "always-active bytes one decode step reads"),
                         ("w_route_pertok", "routed experts' bytes per token x MoE layers"),
                         ("w_route_total", "all routed experts' bytes x MoE layers")):
            d.need(f_, f"{what}: the config shows a mixture of experts ({seen}) but "
                   f"names no expert count in {list(_EXPERT_COUNT)}", NOTES["moe"])
        d.need("params_prefill", "active GEMM params excluding embeddings and lm_head",
               NOTES["prefill"])
    elif n_exp:
        routed = ck.bytes_matching(_ROUTED.pattern)
        d.facts.append(f"MoE: {n_exp} routed experts, {k_exp} per token; tensors under "
                       f"experts.#. hold {routed:,} B (main and draft layers)")
        for f_, what in (("w_decode_shared", "always-active bytes one decode step reads"),
                         ("w_route_pertok", f"{k_exp} routed experts' bytes x MoE layers"),
                         ("w_route_total", f"all {n_exp} routed experts' bytes x MoE layers")):
            d.need(f_, what, NOTES["moe"])
        d.need("params_prefill", "active GEMM params excluding embeddings and lm_head",
               NOTES["prefill"])
    else:
        d.set("w_route_pertok", 0.0, "dense: no routed experts")
        d.set("w_route_total", 0.0)
        d.need("w_decode_shared", "dense: the bytes every step reads (all but vision, "
               f"draft and embedding tables: header total less {vision + draft_b:,} B "
               f"of vision and draft is {total - vision - draft_b:,})", NOTES["moe"])
        d.need("params_prefill", "decoder-layer params excluding embeddings and lm_head",
               NOTES["prefill"])

    # ---- draft module -------------------------------------------------------
    if nextn or dspark or draft_b:
        method = "dspark" if dspark else "mtp"
        d.need("draft_layers / w_draft / w_draft_route_pertok / spec_tokens / "
               "draft_method / acceptance",
               f"{method} draft module: the config states {nextn or 'no'} predict layers; "
               + (f"tensors named mtp/dspark hold {draft_b:,} B" if draft_b else
                  f"no tensor is named mtp/dspark (look past layer {n - 1})")
               + "; its cache rows join kv_bpt on some models",
               NOTES[f"draft_{method}"])
    else:
        d.need("draft_layers", "no MTP or draft module in this repo: a drafter "
               "published in its own repo is read with its own `ws catalog` "
               "(Kimi-K3.1-Pro's DSpark draft is one)", NOTES["draft_dspark"])

    # ---- context, vocabulary, TP -------------------------------------------
    ctx = t.get("max_position_embeddings")
    if isinstance(ctx, int) and ctx > 0:
        rope = t.get("rope_scaling") or t.get("rope_parameters") or {}
        rt = rope.get("rope_type") or rope.get("type")
        d.set("max_ctx", ctx, "max_position_embeddings"
              + (f" ({rt} x{rope.get('factor')} over "
                 f"{rope.get('original_max_position_embeddings')})"
                 if rt == "yarn" else ""))
    else:
        d.need("max_ctx", f"max_position_embeddings is {ctx!r}")

    facts = tp_facts(cfg)
    d.tp_ok = accepted_tps(facts, tp_max)
    d.tp_refused = {tp: tp_refusals(facts, tp) for tp in range(1, tp_max + 1)
                    if tp not in d.tp_ok}
    d.set("vllm_tp", tuple(d.tp_ok), f"TP sizes vLLM {VLLM_VERSION} accepts, of 1..{tp_max}")
    padded = pad_vocab(facts.vocab)
    d.facts.append(f"vocabulary {facts.vocab:,}"
                   + (f" (vLLM pads it to {padded:,})" if padded != facts.vocab else ""))
    d.need("nvfp4_w", "run this command on the NVFP4 checkpoint and take its totals",
           NOTES["nvfp4"])

    # provenance (workingset.provenance) for what this command read itself;
    # every other quantity stays a TODO for whoever fills its fields
    ref = f"https://huggingface.co/{ck.repo}/tree/{ck.revision[:12]}, read {d.read_on}"
    if "w_resident" in d.fields:
        d.provenance["weights"] = {"kind": "measured", "note": f"safetensors total, "
                                   f"every shard header ({ck.shards} shards)", "ref": ref}
    if "kv_bpt" in d.fields:
        _, how = d.fields["kv_bpt"]
        d.provenance["kv"] = {"kind": "derived", "basis": "config", "note": how, "ref": ref}
    return d


def _val(v) -> str:
    if isinstance(v, tuple):
        return "[" + ", ".join(map(str, v)) + "]"
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, float):
        return repr(v)
    return f"{v:_}" if abs(v) >= 10_000 else str(v)


def render(d: Draft) -> str:
    out = [f"# DRAFT from {d.repo} at {d.revision[:12]} (`ws catalog`).",
           "# '# TODO' lines are fields a person must determine; the rest follow",
           "# from config.json and the safetensors headers.",
           f"[models.{d.key}]"]
    for name, (v, c) in d.fields.items():
        line = f"{name} = {_val(v)}"
        out.append(f"{line:<32} # {c}" if c else line)
    for line in d.facts:
        out.append(f"# {line}")
    if d.tp_refused:
        out.append(f"# vLLM {VLLM_VERSION} refuses " + ", ".join(map(str, d.tp_refused)))
    for tp, why in d.tp_refused.items():
        if tp <= 8:   # one node: the widths a deployment of this study can pick
            out.append(f"#   {tp}: {'; '.join(why)}")
    for name, what, note in d.todo:
        out.append(f"# TODO {name}: {what}" + (f" (see {note})" if note else ""))
    out.append(f"# TODO [models.{d.key}.quality] and [models.{d.key}.explorer]: "
               "Terminal-Bench scores, short name, tooltip")
    out += ["", f"[models.{d.key}.provenance]"]
    for q, row in d.provenance.items():
        out.append(f"{q} = {_inline(row)}")
    for q in MODEL_QUANTITIES:
        if q not in d.provenance:
            when = " (once nvfp4_w is set)" if q == "weights_nvfp4" else ""
            out.append(f"# TODO {q}{when} = {{ kind = measured|vendor|derived|guess, "
                       f"note = how, ref = where }}: see src/workingset/provenance.py")
    return "\n".join(out) + "\n"


def _inline(row: dict) -> str:
    return "{ " + ", ".join(f"{k} = {json.dumps(v, ensure_ascii=False)}"
                            for k, v in row.items()) + " }"


def cmd_catalog(args) -> int:
    if (args.repo is None) == (args.fixture is None):
        raise ValueError("give a REPO or --fixture DIR, not both")
    if args.fixture:
        ck = load_fixture(args.fixture)
    else:
        try:
            ck = read_repo(args.repo, args.revision, index_file=args.index)
        except HFError as e:
            raise ValueError(str(e)) from e
        if args.save:
            print(f"wrote {save_fixture(ck, args.save)}", file=sys.stderr)
    sys.stdout.write(render(draft(ck, args.key, args.tp_max)))
    return 0


def add_subparser(sub) -> None:
    p = sub.add_parser(
        "catalog", help="draft a catalog.toml model table from a Hugging Face repo",
        description=__doc__.split("\n\n")[0])
    p.add_argument("repo", nargs="?", help="Hugging Face repo id, e.g. Qwen/Qwen3.8-27B-FP8 "
                   "(gated repos read HF_TOKEN)")
    p.add_argument("--revision", default="main", help="branch, tag or commit (default main)")
    p.add_argument("--key", default="NEW", help="catalogue key for the table (default NEW)")
    p.add_argument("--index", help="the safetensors index to read when the repo ships "
                   "several layouts")
    p.add_argument("--tp-max", type=int, default=16,
                   help="check TP sizes 1..N against vLLM's rules (default 16)")
    p.add_argument("--save", metavar="DIR", help="also write config.json and the tensor "
                   "census under DIR/<org>--<name>/, the test fixture format")
    p.add_argument("--fixture", metavar="DIR", help="read a saved fixture instead of the network")
    p.set_defaults(fn=cmd_catalog)
