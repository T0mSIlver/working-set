"""Read a Hugging Face model repo: config.json and every safetensors header.

A safetensors file starts with an 8-byte little-endian header length and a
JSON header naming each tensor's dtype, shape and byte offsets, so two HTTP
range requests per shard give every tensor's dtype and shape without
downloading weights. Gated repos read HF_TOKEN from the environment.

The tensor list is kept as a census: tensor names with their numeric
indices replaced by '#' ("model.layers.#.mlp.experts.#.down_proj.weight"),
grouped by (pattern, dtype, shape) with a count. Byte totals and name
detection need nothing more, and the census of a 100k-tensor checkpoint is a
few hundred rows, small enough to commit as a test fixture.
"""
from __future__ import annotations

import json
import math
import os
import re
import struct
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import httpx

HF = "https://huggingface.co"
TOKEN_ENV = "HF_TOKEN"

# bytes per element of each safetensors dtype (safetensors' Dtype enum)
DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}
# a header longer than this is a corrupt length field, not a header
MAX_HEADER = 256 * 1024 * 1024


class HFError(RuntimeError):
    """A repo, file or header that could not be read."""


@dataclass
class Checkpoint:
    repo: str
    revision: str                    # the commit the files were read at
    config: dict
    # [pattern, dtype, shape, count] rows, sorted
    census: list = field(default_factory=list)
    shards: int = 0
    index_total_size: int | None = None   # metadata.total_size of the index, if any

    def bytes_by_dtype(self) -> dict:
        out: dict = {}
        for _pat, dt, shape, n in self.census:
            out[dt] = out.get(dt, 0) + tensor_bytes(dt, shape) * n
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def total_bytes(self) -> int:
        return sum(self.bytes_by_dtype().values())

    def bytes_matching(self, regex: str) -> int:
        r = re.compile(regex)
        return sum(tensor_bytes(dt, shape) * n
                   for pat, dt, shape, n in self.census if r.search(pat))

    def to_fixture(self) -> dict:
        return {"repo": self.repo, "revision": self.revision,
                "shards": self.shards, "index_total_size": self.index_total_size,
                "census": self.census}


def tensor_bytes(dtype: str, shape) -> int:
    if dtype not in DTYPE_BYTES:
        raise HFError(f"unknown safetensors dtype {dtype!r}: add its element "
                      f"size to workingset.hf.DTYPE_BYTES")
    return DTYPE_BYTES[dtype] * math.prod(shape)


_INDEX = re.compile(r"(?<=\.)\d+(?=\.|$)")


def pattern(name: str) -> str:
    """'model.layers.12.mlp.experts.7.w1' -> 'model.layers.#.mlp.experts.#.w1'."""
    return _INDEX.sub("#", name)


def census(tensors: dict) -> list:
    """{name: (dtype, shape)} -> sorted [pattern, dtype, shape, count] rows."""
    rows: dict = {}
    for name, (dt, shape) in tensors.items():
        key = (pattern(name), dt, tuple(shape))
        rows[key] = rows.get(key, 0) + 1
    return [[p, dt, list(shape), n] for (p, dt, shape), n in sorted(rows.items())]


def _client() -> httpx.Client:
    headers = {"User-Agent": "workingset"}
    token = os.environ.get(TOKEN_ENV)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return httpx.Client(follow_redirects=True, timeout=120, headers=headers)


def _denied(url: str, r: httpx.Response) -> str:
    """The message for a 401/403. Hugging Face sends X-Error-Code GatedRepo
    for a gated repo; to an anonymous read it answers a missing or private
    repo with the same bare 401, so without that code the repo may not exist."""
    have = "is set" if os.environ.get(TOKEN_ENV) else "is not set"
    if r.headers.get("x-error-code") == "GatedRepo":
        return (f"{url}: HTTP {r.status_code}. The repo is gated: set {TOKEN_ENV} "
                f"to a token with access ({TOKEN_ENV} {have})")
    return (f"{url}: HTTP {r.status_code}. The repo does not exist, is private, "
            f"or is gated (Hugging Face answers all three alike without a token): "
            f"check the repo id, or set {TOKEN_ENV} to a token with access "
            f"({TOKEN_ENV} {have})")


def _get(client: httpx.Client, repo: str, rev: str, path: str,
         byte_range: tuple | None = None) -> httpx.Response:
    url = f"{HF}/{repo}/resolve/{rev}/{path}"
    headers = {"Range": f"bytes={byte_range[0]}-{byte_range[1]}"} if byte_range else {}
    try:
        r = client.get(url, headers=headers)
    except httpx.HTTPError as e:
        raise HFError(f"{url}: {e}") from e
    if r.status_code in (401, 403):
        raise HFError(_denied(url, r))
    if r.status_code == 404:
        raise HFError(f"{url}: not found (HTTP 404)")
    if r.status_code >= 400:
        raise HFError(f"{url}: HTTP {r.status_code}")
    if byte_range and r.status_code != 206:
        raise HFError(f"{url}: the server ignored the range request "
                      f"(HTTP {r.status_code}), refusing to download the shard")
    return r


def read_header(client: httpx.Client, repo: str, rev: str, shard: str) -> dict:
    """{tensor name: (dtype, shape)} from one safetensors file's header."""
    head = _get(client, repo, rev, shard, (0, 7)).content
    if len(head) != 8:
        raise HFError(f"{repo}/{shard}: header length field is {len(head)} bytes, not 8")
    (n,) = struct.unpack("<Q", head)
    if not 0 < n <= MAX_HEADER:
        raise HFError(f"{repo}/{shard}: header length {n} is not a safetensors header")
    raw = _get(client, repo, rev, shard, (8, 8 + n - 1)).content
    if len(raw) != n:
        raise HFError(f"{repo}/{shard}: read {len(raw)} of {n} header bytes")
    hdr = json.loads(raw)
    hdr.pop("__metadata__", None)
    return {k: (v["dtype"], v["shape"]) for k, v in hdr.items()}


def _repo_files(client: httpx.Client, repo: str, rev: str) -> tuple:
    """(commit sha, file names) from the model API."""
    url = f"{HF}/api/models/{repo}/revision/{rev}"
    try:
        r = client.get(url)
    except httpx.HTTPError as e:
        raise HFError(f"{url}: {e}") from e
    if r.status_code in (401, 403):
        raise HFError(_denied(url, r))
    if r.status_code >= 400:
        raise HFError(f"{url}: HTTP {r.status_code}")
    d = r.json()
    return d["sha"], [s["rfilename"] for s in d.get("siblings", [])]


def pick_weights(files: list, index: str | None = None) -> tuple:
    """(index file or None, shard files) the checkpoint is read from.

    The transformers index model.safetensors.index.json wins when a repo
    ships two layouts (Mistral's consolidated-*.safetensors beside it);
    otherwise the one *.safetensors.index.json, otherwise the one
    *.safetensors. Anything else stops and names --index."""
    if index is not None:
        if index not in files:
            raise HFError(f"--index {index}: not in the repo")
        return index, None
    indexes = [f for f in files if f.endswith(".safetensors.index.json")]
    if "model.safetensors.index.json" in indexes:
        return "model.safetensors.index.json", None
    if len(indexes) == 1:
        return indexes[0], None
    if indexes:
        raise HFError(f"several safetensors indexes {indexes}: pick one with --index")
    single = [f for f in files if f.endswith(".safetensors")]
    if len(single) == 1:
        return None, single
    raise HFError(f"no safetensors index and {len(single)} .safetensors files "
                  f"({single[:4]}): pick the index with --index")


def read_repo(repo: str, revision: str = "main", workers: int = 8,
              index_file: str | None = None) -> Checkpoint:
    """config.json, the safetensors index and every shard header of `repo`."""
    with _client() as c:
        sha, files = _repo_files(c, repo, revision)
        config = _get(c, repo, sha, "config.json").json()
        index_name, single = pick_weights(files, index_file)
        index = _get(c, repo, sha, index_name).json() if index_name else None
        if index is not None:
            if "weight_map" not in index:
                raise HFError(f"{repo}: model.safetensors.index.json has no weight_map")
            shards = sorted(set(index["weight_map"].values()))
            total = (index.get("metadata") or {}).get("total_size")
        else:
            shards, total = single, None
        with ThreadPoolExecutor(workers) as ex:
            parts = list(ex.map(lambda s: read_header(c, repo, sha, s), shards))
    tensors: dict = {}
    for part in parts:
        tensors.update(part)
    if index is not None:
        missing = set(index["weight_map"]) - set(tensors)
        if missing:
            raise HFError(f"{repo}: {len(missing)} tensors in the index are in no "
                          f"shard header, e.g. {sorted(missing)[:3]}")
    return Checkpoint(repo, sha, config, census(tensors), len(shards),
                      int(total) if total is not None else None)


# ---- fixtures: config.json + census.json in one directory per repo ----------

def fixture_dir(root: Path, repo: str) -> Path:
    return Path(root) / repo.replace("/", "--")


def save_fixture(ck: Checkpoint, root: Path) -> Path:
    d = fixture_dir(root, ck.repo)
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(ck.config, indent=2) + "\n")
    meta = ck.to_fixture()
    rows = meta.pop("census")
    head = json.dumps(meta)[:-1]   # one census row per line: diffs stay readable
    body = ",\n".join(json.dumps(r) for r in rows)
    (d / "census.json").write_text(f'{head}, "census": [\n{body}\n]}}\n')
    return d


def load_fixture(path: Path) -> Checkpoint:
    path = Path(path)
    for f in ("config.json", "census.json"):
        if not (path / f).is_file():
            raise HFError(f"{path}: no {f}; write one with `ws catalog REPO --save DIR`")
    meta = json.loads((path / "census.json").read_text())
    return Checkpoint(meta["repo"], meta["revision"],
                      json.loads((path / "config.json").read_text()),
                      meta["census"], meta["shards"], meta["index_total_size"])
