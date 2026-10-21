"""SemiAnalysis AgentX traces: read them, compact them, rebuild their blocks.

The corpus (Hugging Face `semianalysisai/cc-traces-weka-062126`, Apache-2.0)
is 393 Claude Code sessions in the Weka kv-cache-tester format: one JSON
object per line, each request carrying its timestamp, token counts, server
time and the hash ids of its 64-token blocks, with sub-agents nested under
`type: "subagent"` markers. No text. 1.8 GB, almost all of it hash ids.

A prefix cache only ever matches a request's blocks up to the first block it
has not seen, so the ids themselves carry nothing a cache can use beyond
"which earlier prefix does this request extend, and by how many blocks". The
compact form keeps exactly that, per request: the blocks it shares with its
own stream's previous request (`lcp_prev`), the blocks it shares with any
earlier request of the trace (`shared`, from a trie over the block lists in
file order) and its block count. That is the prefix structure a cache can
exploit, in a small fraction of the file.

Hash scope is per trace (`hash_id_scope: "local"`), and AgentX busts the
first-turn prefix of every replayed session, so no block is shared across
sessions: one trie per trace is the whole structure.
"""
from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path

SOURCE = "semianalysisai/cc-traces-weka-062126"
FORMAT = "workingset-agentx-compact/1"
# request tuple layout in the compact file
FIELDS = ("t", "in", "out", "api_time", "think_time", "lcp_prev", "shared", "blocks", "stop")
_STOPS = {"": 0, "tool_use": 1, "end_turn": 2}


def _lcp(a: list[int], b: list[int]) -> int:
    """Length of the common prefix of two id lists (binary search on slices)."""
    hi = min(len(a), len(b))
    if a[:hi] == b[:hi]:
        return hi
    lo = 0  # a[:lo] == b[:lo] holds, a[:hi] != b[:hi]
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid
    return lo


class _Trie:
    """Block-prefix trie of one trace. Node 0 is the empty prefix."""

    def __init__(self) -> None:
        self.child: dict[tuple[int, int], int] = {}
        self.n = 1

    def insert(self, ids: list[int], prev_ids: list[int] | None,
               prev_path: list[int] | None) -> tuple[int, int, list[int]]:
        """Insert `ids`; return (blocks shared with the stream's previous
        request, blocks shared with any earlier request, the node path).

        `prev_ids`/`prev_path` shortcut the walk: the common prefix with the
        stream's previous request is already in the trie.
        """
        path: list[int] = []
        node, depth, k = 0, 0, 0
        if prev_ids is not None and prev_path:
            k = _lcp(ids, prev_ids)
            if k:
                path = prev_path[:k]
                node, depth = path[-1], k
        # walk on through nodes another stream may have created
        while depth < len(ids):
            nxt = self.child.get((node, ids[depth]))
            if nxt is None:
                break
            node = nxt
            path.append(node)
            depth += 1
        shared = depth
        # everything below a new node is new: no more lookups
        for h in ids[depth:]:
            self.child[(node, h)] = self.n
            node = self.n
            path.append(node)
            self.n += 1
        return k, shared, path


def _req_row(r: dict, trie: _Trie, state: dict) -> list:
    ids = r.get("hash_ids") or []
    lcp_prev, shared, path = trie.insert(ids, state.get("ids"), state.get("path"))
    state["ids"], state["path"] = ids, path
    return [r["t"], r["in"], r["out"], r.get("api_time"), r.get("think_time"),
            lcp_prev, shared, len(ids), _STOPS.get(r.get("stop", ""), 3)]


def compact_trace(tr: dict) -> dict:
    """One Weka trace object -> its compact form (streams of request rows)."""
    if tr.get("hash_id_scope", "local") != "local":
        raise ValueError(f"trace {tr.get('id')}: only local hash scope is supported")
    trie = _Trie()
    root_state: dict = {}
    root: list = []
    subs: list = []
    for r in tr["requests"]:
        if r["type"] == "subagent":
            st: dict = {}
            subs.append({
                "agent_id": r.get("agent_id"), "subagent_type": r.get("subagent_type"),
                "t": r["t"], "parent_turn": len(root), "status": r.get("status"),
                "duration_ms": r.get("duration_ms"),
                "requests": [_req_row(q, trie, st) for q in r.get("requests", [])],
            })
        else:
            root.append(_req_row(r, trie, root_state))
    return {"id": tr["id"], "block_size": tr.get("block_size", 64),
            "tool_tokens": tr.get("tool_tokens", 0),
            "system_tokens": tr.get("system_tokens", 0),
            "nodes": trie.n, "root": root, "subagents": subs}


def compact_jsonl(src: str | Path, out: str | Path, *, limit: int | None = None) -> dict:
    """Stream `traces.jsonl` into a gzipped compact file; return its summary."""
    traces = []
    n_req = 0
    with open(src, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            c = compact_trace(json.loads(line))
            n_req += len(c["root"]) + sum(len(s["requests"]) for s in c["subagents"])
            traces.append(c)
            if limit is not None and len(traces) >= limit:
                break
    doc = {"format": FORMAT, "source": SOURCE, "license": "Apache-2.0",
           "fields": list(FIELDS), "traces": traces}
    with gzip.open(out, "wt", encoding="utf-8") as g:
        json.dump(doc, g, separators=(",", ":"))
    return {"traces": len(traces), "requests": n_req}


# --- reading the compact form ------------------------------------------------

@dataclass
class Request:
    t: float
    tokens_in: int
    tokens_out: int
    api_time: float | None
    think_time: float | None
    lcp_prev: int        # blocks shared with the stream's previous request
    shared: int          # blocks shared with any earlier request of the trace
    nblocks: int
    stop: int


@dataclass
class Stream:
    """The root agent or one sub-agent: requests in order."""
    key: str
    requests: list[Request]
    spawn_t: float | None = None
    parent_turn: int | None = None   # sub-agents: root requests sent before the spawn


@dataclass
class Trace:
    id: str
    block_size: int
    streams: list[Stream]            # [root, *subagents]

    @property
    def root(self) -> Stream:
        return self.streams[0]


def _stream(key: str, rows: list, **kw) -> Stream:
    return Stream(key, [Request(*row) for row in rows], **kw)


def from_compact(c: dict) -> Trace:
    subs = [_stream(f"sa:{s['agent_id']}", s["requests"], spawn_t=s["t"],
                    parent_turn=s["parent_turn"]) for s in c["subagents"]]
    return Trace(c["id"], c["block_size"], [_stream("root", c["root"]), *subs])


def load(path: str | Path) -> list[Trace]:
    """Read a compact file (or, for small tests, a raw traces.jsonl)."""
    p = Path(path)
    if p.suffix == ".jsonl":
        with open(p, "rb") as f:
            cs = [compact_trace(json.loads(line)) for line in f if line.strip()]
    else:
        with gzip.open(p, "rt", encoding="utf-8") as g:
            doc = json.load(g)
        if doc.get("format") != FORMAT:
            raise ValueError(f"{p}: not a {FORMAT} file")
        cs = doc["traces"]
    return [from_compact(c) for c in cs]


if __name__ == "__main__":  # python -m workingset.agentx traces.jsonl out.json.gz
    import sys
    print(compact_jsonl(sys.argv[1], sys.argv[2]))
