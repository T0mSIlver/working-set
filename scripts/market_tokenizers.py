# /// script
# requires-python = ">=3.11"
# dependencies = ["tokenizers>=0.21", "mistral-common>=1.9", "tiktoken>=0.8"]
# ///
"""Characters per token for every catalogue model on the fixed corpus in data/market/corpus/.

A price per million tokens compares across models only once each model's tokens are
converted to text: this script counts, for each corpus kind, how many tokens each
model's own tokenizer makes of the same text, and writes data/market/chars_per_token.json.

    uv run scripts/market_tokenizers.py [--cache DIR] [--hf-cache /models/hub] [--offline]

Only tokenizer files are fetched (one HTTPS GET per file, pinned to the repo's current
commit), never weights. They are loaded as data: `tokenizers.Tokenizer.from_file`,
mistral-common's Tekkenizer for tekken.json, tiktoken for Kimi's tiktoken.model. No code
from a model repo runs; Kimi's split pattern is copied below and checked against the
repo's tokenization_kimi.py as text.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data/market/corpus"
OUT = ROOT / "data/market/chars_per_token.json"
KINDS = ("code", "english", "french", "json_tool_calls")

# key -> (Hugging Face repo, loader). Repos as in research/model_*.md.
MODELS = {
    "27B": ("Qwen/Qwen3.8-27B", "hf"),
    "35BA3B": ("Qwen/Qwen3.6-35B-A3B", "hf"),
    "MM35": ("mistralai/Mistral-Medium-3.5-128B", "tekken"),
    "GLM52": ("zai-org/GLM-5.3", "hf"),
    "DSV4F": ("deepseek-ai/DeepSeek-V4-Flash-0731", "hf"),
    "DSV4FV": ("deepseek-ai/DeepSeek-V4-Flash-Vision-Exp", "hf"),
    "DSV41F": ("deepseek-ai/DeepSeek-V4.1-Flash", "hf"),
    "Q38FN": ("Qwen/Qwen3.8-Flash-Next", "hf"),
    "GLM53F": ("zai-org/GLM-5.3-Flash", "hf"),
    "ML4": ("mistralai/Mistral-Large-4-1T-A52B", "tekken"),
    "K31P": ("moonshotai/Kimi-K3.1-Pro", "kimi"),
}
FILES = {"hf": ["tokenizer.json"], "tekken": ["tekken.json"],
         "kimi": ["tiktoken.model", "tokenization_kimi.py"]}

# moonshotai/Kimi-K3.1-Pro tokenization_kimi.py, TikTokenTokenizer.pat_str
KIMI_PAT_PARTS = [
    r"""[\p{Han}]+""",
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""\p{N}{1,3}""",
    r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
    r"""\s*[\r\n]+""",
    r"""\s+(?!\S)""",
    r"""\s+""",
]
KIMI_RESERVED_SPECIAL = 256  # TikTokenTokenizer.num_reserved_special_tokens

METHOD = (
    "Each corpus file is read as UTF-8 and encoded on its own with the model's own "
    "tokenizer as plain text: no special tokens (add_special_tokens=False, bos/eos off, "
    "tiktoken encode_ordinary), no chat template. A kind's tokens and characters are summed "
    "over its files. Characters are Unicode code points, bytes are UTF-8 bytes. "
    "tokenizer.json is loaded with tokenizers.Tokenizer.from_file; tekken.json with "
    "mistral-common's Tekkenizer; Kimi's tiktoken.model with tiktoken and the split "
    "pattern from the repo's tokenization_kimi.py (read as text, not run). "
    "vocab_size counts added/special tokens."
)


def http_json(url: str) -> dict:
    req = urllib.request.Request(url)
    if os.environ.get("HF_TOKEN"):
        req.add_header("Authorization", "Bearer " + os.environ["HF_TOKEN"])
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def fetch(repo: str, files: list[str], cache: Path, hf_cache: Path | None, offline: bool) -> tuple[str, dict[str, Path]]:
    """Return (revision, {file: local path}). Raises PermissionError when gated."""
    # 1. the shared HF cache on this box, only for this exact repo
    if hf_cache:
        d = hf_cache / ("models--" + repo.replace("/", "--"))
        ref = d / "refs/main"
        if ref.is_file():
            rev = ref.read_text().strip()
            paths = {f: d / "snapshots" / rev / f for f in files}
            if all(p.is_file() for p in paths.values()):
                return rev, paths
    # 2. our own cache, written by a previous run
    d = cache / repo
    rev_file = d / ".revision"
    if rev_file.is_file() and all((d / f).is_file() for f in files):
        return rev_file.read_text().strip(), {f: d / f for f in files}
    if offline:
        raise FileNotFoundError(f"{repo}: tokenizer files not in {d} or the HF cache, and --offline is set")
    try:
        info = http_json(f"https://huggingface.co/api/models/{repo}")
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise PermissionError("gated") from e
        raise
    rev = info["sha"]
    d.mkdir(parents=True, exist_ok=True)
    for f in files:
        url = f"https://huggingface.co/{repo}/resolve/{rev}/{f}"
        req = urllib.request.Request(url)
        if os.environ.get("HF_TOKEN"):
            req.add_header("Authorization", "Bearer " + os.environ["HF_TOKEN"])
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                (d / f).write_bytes(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise PermissionError("gated") from e
            raise
    rev_file.write_text(rev)
    return rev, {f: d / f for f in files}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(loader: str, paths: dict[str, Path]):
    """Return (encode(text) -> list[int], vocab_size)."""
    if loader == "hf":
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(paths["tokenizer.json"]))
        return (lambda s: tok.encode(s, add_special_tokens=False).ids), tok.get_vocab_size(with_added_tokens=True)
    if loader == "tekken":
        from mistral_common.tokens.tokenizers.tekken import Tekkenizer
        tok = Tekkenizer.from_file(str(paths["tekken.json"]))
        return (lambda s: tok.encode(s, bos=False, eos=False)), tok.n_words
    if loader == "kimi":
        import tiktoken
        from tiktoken.load import load_tiktoken_bpe
        src = paths["tokenization_kimi.py"].read_text(encoding="utf-8")
        missing = [p for p in KIMI_PAT_PARTS if p not in src]
        if missing or f"num_reserved_special_tokens = {KIMI_RESERVED_SPECIAL}" not in src:
            raise ValueError("tokenization_kimi.py no longer carries the split pattern copied in "
                             "KIMI_PAT_PARTS / KIMI_RESERVED_SPECIAL; update them from the repo")
        ranks = load_tiktoken_bpe(str(paths["tiktoken.model"]))
        enc = tiktoken.Encoding(name="kimi", pat_str="|".join(KIMI_PAT_PARTS), mergeable_ranks=ranks, special_tokens={})
        return enc.encode_ordinary, len(ranks) + KIMI_RESERVED_SPECIAL
    raise ValueError(loader)


def read_corpus() -> dict:
    corpus = {}
    for kind in KINDS:
        files = sorted(p for p in (CORPUS / kind).rglob("*") if p.is_file())
        if not files:
            raise FileNotFoundError(f"no corpus files under {CORPUS / kind}")
        texts = [(p, p.read_text(encoding="utf-8")) for p in files]
        raw = b"".join(p.read_bytes() for p, _ in texts)
        corpus[kind] = {
            "files": [str(p.relative_to(ROOT)) for p, _ in texts],
            "chars": sum(len(t) for _, t in texts),
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "_texts": [t for _, t in texts],
        }
    return corpus


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "workingset" / "market" / "tokenizers",
                    help="where tokenizer files are downloaded, one directory per repo")
    ap.add_argument("--hf-cache", type=Path, default=None,
                    help="a Hugging Face hub cache to read first (e.g. /models/hub); used only for the exact repo")
    ap.add_argument("--offline", action="store_true", help="never download; fail when a file is missing")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    corpus = read_corpus()
    models = {}
    for key, (repo, loader) in MODELS.items():
        files = FILES[loader]
        try:
            rev, paths = fetch(repo, files, args.cache, args.hf_cache, args.offline)
        except PermissionError:
            models[key] = {"tokenizer_repo": repo, "error": "gated"}
            print(f"{key}: {repo} is gated; set HF_TOKEN", file=sys.stderr)
            continue
        encode, vocab = load(loader, paths)
        main_file = paths[files[0]]
        row = {"tokenizer_repo": repo, "revision": rev, "tokenizer_files": files,
               "tokenizer_sha256": sha256(main_file), "vocab_size": vocab,
               "tokens": {}, "chars_per_token": {}, "bytes_per_token": {}}
        for kind, c in corpus.items():
            n = sum(len(encode(t)) for t in c["_texts"])
            row["tokens"][kind] = n
            row["chars_per_token"][kind] = round(c["chars"] / n, 4)
            row["bytes_per_token"][kind] = round(c["bytes"] / n, 4)
        models[key] = row
        print(f"{key:7s} " + "  ".join(f"{k} {row['chars_per_token'][k]:.3f}" for k in KINDS))

    for c in corpus.values():
        del c["_texts"]
    from importlib.metadata import version
    libs = {p: version(p) for p in ("tokenizers", "mistral-common", "tiktoken")}
    out = {"source": {"read": dt.date.today().isoformat(), "script": "scripts/market_tokenizers.py", "method": METHOD,
                      "libraries": libs},
           "corpus": corpus, "models": models}
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.out.relative_to(ROOT) if args.out.is_relative_to(ROOT) else args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
