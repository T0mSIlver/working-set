"""Agent-session traces: the timing and sizes of real coding-agent sessions,
without their text.

A trace is what `ws test --replay` plays back. `extract_pi` reads pi session
files (`~/.pi/agent/sessions/*/*.jsonl`) and keeps, for every model request,
only numbers and tool names:

    prompt_tokens   input + cacheRead + cacheWrite of the request's usage
    output_tokens   tokens generated, reasoning included
    gap_s           seconds from the end of the previous request to the start
                    of this one: a tool running, or the person reading and
                    typing (None on a session's first request)
    after           what the gap waited on: "tool", "user" or "start"
    tools           the tools the previous request called, builtin names kept
                    and every other name reported as "other"

`extract_claude` does the same for Claude Code's session files
(`~/.claude/projects/*/*.jsonl`, with the subagents' files beside them). Its
differences from pi, all from what the files hold:

    request         Claude Code writes one line per content block; the lines
                    of one API response share a message id and are merged
                    (output tokens: the largest of them, since the first
                    lines carry a partial count)
    service_s       the files have no request start time. A request starts at
                    the tool result, person's message or background
                    notification that triggered it and ends at its last
                    block's line, so service_s also holds the client's own
                    overhead (a few hundred ms) and the first-token wait, and
                    is never shorter than the true one. The first request of
                    a session starts at its first block
    permission      no log exists: a wait on a permission prompt is counted as
                    tool time (the trace's notes say so), so the long tail of
                    "tool" gaps is partly a person
    background      a background task or subagent finished, or another agent
                    sent a message (kind "background"); inside a turn, no
                    person typed
    subagents       each subagent is its own session in the trace, linked to
                    the session that launched it (`SessionLink`): which
                    session, when its first request started after the
                    parent's first request, and the parent request that
                    spawned it. `ws replay` replays them as independent
                    sessions (it does not launch them from their parent)
    headless        sessions run with `claude -p` or the SDK (entrypoint
                    sdk-cli, sdk-ts) are left out unless asked for

No prompt, reply, path or tool argument leaves the file, so a colleague can
export their sessions and hand over the trace. The tokenizer is the one of
whatever model served the session; the replay re-sizes nothing.

`equivalent_workload` turns a trace into the `[workload]` values with the same
averages, which is what the model prices: a replay measured against those
predictions shows what the bursts cost over a smooth arrival.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

FORMAT = "workingset-agent-trace"
VERSION = 2

# pi's builtin tools plus the common ones every harness has. Anything else (MCP
# servers, extensions) can name an internal service, so it is reported as
# "other".
KNOWN_TOOLS = frozenset({"bash", "read", "edit", "write", "grep", "find", "ls",
                         "glob", "todo", "web_fetch", "web_search", "task",
                         "ask_user_question"})

# A person's wait longer than this, with nothing in the file saying pi was
# restarted, means the session was left open (overnight, a meeting): the one
# gap kind a threshold decides. The same boundary as
# scripts/think_time_trace.py's PARK_S.
PARK_S = 1800.0

# What a gap waited on, decided from the session's events (`classify`):
#   start       the recording's first request; nothing before it
#   tool        tools ran, nobody was asked anything
#   permission  a tool waited on a person answering a permission prompt
#   ask         the agent asked the person a question (ask_user_question)
#   think       the agent finished; the person read and typed the next message
#   abort       the person interrupted a request, then typed again; the wait
#               runs from the interruption
#   branch      the person moved to another point of the session tree
#   compaction  pi compacted the history; the gap holds the summary request
#   error       a request failed (timeout, server down) and was retried: the
#               gap is an outage, not anyone's think time
#   resume      pi was closed and the session reopened later
#   idle        a person's wait past PARK_S with no restart: left open
#   background  (Claude Code) a background task or subagent finished, or
#               another agent wrote to the session, and nobody typed
KINDS = ("start", "tool", "permission", "ask", "think", "abort", "branch",
         "compaction", "error", "resume", "idle", "background")
# The kinds where the session was not in use. `ws test --replay` applies the
# gap policy to these only; every other gap is replayed as recorded.
INACTIVE = frozenset({"error", "resume", "idle"})
# What a replay does with an inactive gap (`--replay-gap-policy`):
#   sample  a human wait drawn from the trace's own think gaps (default)
#   cap     the recorded wait, capped at --replay-max-gap-s
#   drop    no wait: the next request follows at once
GAP_POLICIES = ("sample", "cap", "drop")
DEFAULT_GAP_POLICY = "sample"

PERMISSION_LOG = ("~/.pi/agent/extensions/pi-permission-system/logs/"
                  "pi-permission-system-permission-review.jsonl")
_DENIED = "[pi-permission-system] User denied"


@dataclass(frozen=True)
class TraceRequest:
    prompt_tokens: int
    output_tokens: int
    gap_s: float | None
    after: str
    tools: tuple[str, ...] = ()
    # what the gap waited on (KINDS) and, for a tool wait, the part of it a
    # person spent answering (a permission prompt, an agent's question)
    kind: str = ""
    human_s: float = 0.0
    # how long the recorded server took on it, first token to last
    # included: what the dry run sizes the offered load with
    service_s: float | None = None

    def __post_init__(self):
        if not self.kind:
            object.__setattr__(self, "kind", {"user": "think"}.get(
                self.after, self.after))

    def to_dict(self) -> dict:
        d = {"prompt_tokens": self.prompt_tokens,
             "output_tokens": self.output_tokens,
             "gap_s": None if self.gap_s is None else round(self.gap_s, 3),
             "after": self.after, "kind": self.kind}
        if self.tools:
            d["tools"] = list(self.tools)
        if self.human_s:
            d["human_s"] = round(self.human_s, 3)
        if self.service_s is not None:
            d["service_s"] = round(self.service_s, 3)
        return d


@dataclass(frozen=True)
class SessionLink:
    """A subagent session's place among the trace's sessions: the index of
    the session that launched it, seconds from that session's first request
    to the subagent's first, and the index of the parent request that
    spawned it (None when unknown)."""
    parent: int
    start_offset_s: float
    spawn_request: int | None = None

    def to_dict(self) -> dict:
        d = {"parent": self.parent,
             "start_offset_s": round(self.start_offset_s, 3)}
        if self.spawn_request is not None:
            d["spawn_request"] = self.spawn_request
        return d


@dataclass(frozen=True)
class AgentTrace:
    source: str
    sessions: tuple[tuple[TraceRequest, ...], ...]
    # dropped requests and other counts `ws trace` reports (not replayed)
    notes: dict | None = None
    # per session: its SessionLink when a subagent, else None; () when the
    # trace has none (pi, and every trace of version 2 before links)
    links: tuple[SessionLink | None, ...] = ()

    @property
    def n_requests(self) -> int:
        return sum(len(s) for s in self.sessions)

    def to_dict(self) -> dict:
        links = self.links or (None,) * len(self.sessions)
        d = {"format": FORMAT, "version": VERSION, "source": self.source,
             "sessions": [{"requests": [r.to_dict() for r in s],
                           **(l.to_dict() if l else {})}
                          for s, l in zip(self.sessions, links)]}
        if self.notes:
            d["notes"] = self.notes
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "AgentTrace":
        if d.get("format") != FORMAT:
            raise ValueError(f"not a {FORMAT} file")
        if d.get("version") not in (1, VERSION):
            raise ValueError(f"trace version {d.get('version')!r}, "
                             f"this ws reads versions 1 and {VERSION}")
        sessions, raw_links, kept = [], [], {}
        for i, s in enumerate(d["sessions"]):
            reqs = tuple(TraceRequest(
                prompt_tokens=int(r["prompt_tokens"]),
                output_tokens=int(r["output_tokens"]),
                gap_s=None if r.get("gap_s") is None else float(r["gap_s"]),
                after=r["after"], tools=tuple(r.get("tools", ())),
                kind=r.get("kind", ""), human_s=float(r.get("human_s", 0.0)),
                service_s=(None if r.get("service_s") is None
                           else float(r["service_s"])))
                for r in s["requests"])
            if reqs:
                kept[i] = len(sessions)
                sessions.append(reqs)
                raw_links.append(s if "parent" in s else None)
        if not sessions:
            raise ValueError("the trace holds no request")
        links = []
        for s in raw_links:
            if s is not None and s["parent"] in kept:
                sr = s.get("spawn_request")
                links.append(SessionLink(
                    kept[s["parent"]], float(s.get("start_offset_s", 0.0)),
                    None if sr is None else int(sr)))
            else:
                links.append(None)
        return cls(source=d.get("source", ""), sessions=tuple(sessions),
                   notes=d.get("notes"),
                   links=tuple(links) if any(links) else ())


def load(path: str | Path) -> AgentTrace:
    return AgentTrace.from_dict(
        json.loads(Path(path).read_text(encoding="utf-8")))


def _iso_s(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def _tool_name(name: str) -> str:
    return name if name in KNOWN_TOOLS else "other"


def permission_waits(path: str | Path | None = None) -> dict[str, float]:
    """pi-permission-system's review log -> {tool call id: seconds the prompt
    stood before the person answered}. Empty when the log is absent."""
    p = Path(path or PERMISSION_LOG).expanduser()
    shown: dict[str, float] = {}
    waits: dict[str, float] = {}
    try:
        fh = p.open(encoding="utf-8")
    except OSError:
        return waits
    with fh:
        for line in fh:
            try:
                row = json.loads(line)
                ev, rid = row.get("event", ""), row.get("toolCallId")
                t = _iso_s(row["timestamp"])
            except (ValueError, KeyError, TypeError, AttributeError):
                continue
            if not rid:
                continue
            if ev == "permission_request.waiting":
                shown[rid] = t
            elif ev in ("permission_request.approved",
                        "permission_request.denied") and rid in shown:
                waits[rid] = waits.get(rid, 0.0) + t - shown.pop(rid)
    return waits


def classify(after: str, human_s: float, gap: float, events: set,
             park_s: float = PARK_S) -> str:
    """One gap's kind from what happened in it. `events` holds the markers
    seen since the previous reply: "error", "resume", "compaction", "branch",
    "abort", "permission", "ask"."""
    for k in ("error", "resume", "compaction", "branch"):
        if k in events:
            return k
    if "background" in events:
        return "background"
    if after == "tool":
        if "permission" in events or "ask" in events:
            if human_s > park_s:
                return "idle"
            return "permission" if "permission" in events else "ask"
        return "tool"
    if gap > park_s:
        return "idle"
    return "abort" if "abort" in events else "think"


def extract_pi_session(lines, waits: dict[str, float] | None = None,
                       park_s: float = PARK_S, counts: dict | None = None
                       ) -> tuple[TraceRequest, ...]:
    """One pi session file's lines -> its requests, in file order.

    A request starts at the assistant message's `timestamp` (ms, when pi sent
    it) and ends at the entry's ISO `timestamp` (when the reply was written).
    A request without usage (aborted, errored) is not replayed: the server's
    work on it is unknown. An aborted one restarts the wait clock, since the
    person's wait began there; an errored one folds into the next gap, which
    is then an outage (kind "error"). `waits` maps a tool call id to its
    permission-prompt time (`permission_waits`); without it a denial in the
    tool's result still marks the gap, with the whole wait counted as human.
    `counts`, when given, collects the dropped requests and fork headers.
    """
    waits = waits or {}
    counts = counts if counts is not None else {}
    out: list[TraceRequest] = []
    last_end: float | None = None
    waited_on, tools, call_ids = "start", (), ()
    events: set[str] = set()
    denied = False
    prev_id = None
    early_custom: set[str] = set()          # extensions that write on start
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            counts["torn_lines"] = counts.get("torn_lines", 0) + 1
            continue           # a torn append (pi crashed mid-write)
        typ = row.get("type")
        pid, rid = row.get("parentId"), row.get("id")
        if (pid is not None and prev_id is not None and pid != prev_id
                and typ != "session"):
            events.add("branch")
        if rid is not None:
            prev_id = rid
        if typ == "session":
            if row.get("parentSession"):
                counts["forked_sessions"] = counts.get("forked_sessions", 0) + 1
            if out:
                events.add("resume")
            continue
        if typ == "compaction":
            events.add("compaction")
            continue
        if typ == "branch_summary":
            events.add("branch")
            continue
        if typ == "custom":
            ct = row.get("customType", "")
            if not out:
                early_custom.add(ct)
            elif ct in early_custom and waited_on != "tool":
                # an extension that writes its state when pi opens a session
                # wrote it again: pi was restarted on this file
                events.add("resume")
            continue
        if typ != "message":
            continue
        msg = row["message"]
        role = msg.get("role")
        if role == "user":
            waited_on = "user"
        elif role == "assistant":
            usage = msg.get("usage") or {}
            prompt = (usage.get("input", 0) + usage.get("cacheRead", 0)
                      + usage.get("cacheWrite", 0))
            if prompt <= 0:
                why = msg.get("stopReason") or "error"
                counts[f"dropped_{why}"] = counts.get(f"dropped_{why}", 0) + 1
                if why == "aborted":
                    events.add("abort")
                    last_end = _iso_s(row["timestamp"])
                else:
                    events.add("error")
                continue
            start = msg["timestamp"] / 1000.0
            gap = (None if last_end is None
                   else round(max(0.0, start - last_end), 3))
            after = waited_on if out else "start"
            human = 0.0
            if after == "tool":
                human = sum(waits.get(c, 0.0) for c in call_ids)
                if human > 0:
                    events.add("permission")
                elif denied:
                    events.add("permission")
                    human = gap or 0.0
                if "ask_user_question" in tools:
                    events.add("ask")
                    human = gap or 0.0
            kind = ("start" if after == "start"
                    else classify(after, human, gap or 0.0, events, park_s))
            out.append(TraceRequest(
                prompt_tokens=int(prompt),
                output_tokens=int(usage.get("output", 0)),
                gap_s=gap, after=after,
                tools=tools if waited_on == "tool" else (),
                kind=kind, human_s=round(min(human, gap or 0.0), 3),
                service_s=max(0.0, _iso_s(row["timestamp"]) - start)))
            last_end = _iso_s(row["timestamp"])
            calls = [c for c in msg["content"] if c.get("type") == "toolCall"]
            tools = tuple(sorted({_tool_name(c["name"]) for c in calls}))
            call_ids = tuple(c.get("id") for c in calls if c.get("id"))
            waited_on = "tool" if tools else "user"
            events, denied = set(), False
        elif role == "toolResult":
            waited_on = "tool"
            for c in msg.get("content") or ():
                if (isinstance(c, dict)
                        and str(c.get("text", "")).startswith(_DENIED)):
                    denied = True
    return tuple(out)


def extract_pi(paths, permission_log: str | None = None,
               exclude: tuple[str, ...] = ()) -> AgentTrace:
    """pi session files, or directories searched for `*.jsonl`. `exclude`
    holds fnmatch patterns matched against each file's path (a test
    harness's scratch sessions, say)."""
    import fnmatch
    files: list[Path] = []
    for p in map(Path, paths):
        files += sorted(p.rglob("*.jsonl")) if p.is_dir() else [p]
    files = [f for f in files
             if not any(fnmatch.fnmatch(str(f), pat) for pat in exclude)]
    waits = permission_waits(permission_log)
    counts: dict = {"permission_log": bool(waits)}
    sessions = []
    for f in files:
        try:
            with f.open(encoding="utf-8") as fh:
                reqs = extract_pi_session(fh, waits=waits, counts=counts)
        except (OSError, ValueError, KeyError, TypeError) as e:
            raise ValueError(f"{f}: {type(e).__name__}: {e}") from e
        if reqs:
            sessions.append(reqs)
    if not sessions:
        raise ValueError("no pi request with token usage in "
                         + ", ".join(map(str, paths)))
    return AgentTrace(source="pi", sessions=tuple(sessions), notes=counts)


# ============================================================================
# Claude Code session files
# ============================================================================
DEFAULT_CLAUDE_SESSIONS = "~/.claude/projects"

# Claude Code's builtin tool names in the vocabulary of KNOWN_TOOLS.
_CLAUDE_TOOLS = {"Bash": "bash", "Read": "read", "Edit": "edit",
                 "MultiEdit": "edit", "NotebookEdit": "edit",
                 "Write": "write", "Grep": "grep", "Glob": "glob",
                 "LS": "ls", "TodoWrite": "todo", "WebFetch": "web_fetch",
                 "WebSearch": "web_search", "Task": "task", "Agent": "task",
                 "AskUserQuestion": "ask_user_question"}
# entrypoints of `claude -p` and the SDK: scripts, not a person at a prompt
HEADLESS_ENTRYPOINTS = frozenset({"sdk-cli", "sdk-ts"})


@dataclass
class ClaudeSession:
    """One Claude Code file, parsed: its requests and what links it."""
    requests: tuple[TraceRequest, ...]
    t0: float                                  # first request's start
    entrypoint: str = ""
    tool_use_ids: dict = field(default_factory=dict)   # id -> request index
    spawned_by: str = ""                       # tool_use id (subagent files)


def _claude_user_kind(row: dict) -> str | None:
    """What a user line is: "tool" (a tool result), "human", "background"
    (task notification, another agent's message, a plugin), "abort" (the
    person interrupted), or None for lines that trigger nothing (meta lines,
    local commands). Only the structure is read, never kept."""
    if row.get("isMeta"):
        return None
    msg = row.get("message") or {}
    content = msg.get("content")
    if "toolUseResult" in row or (
            isinstance(content, list) and content
            and all(isinstance(b, dict) and b.get("type") == "tool_result"
                    for b in content)):
        return "tool"
    origin = row.get("origin")
    okind = origin.get("kind") if isinstance(origin, dict) else None
    text = content if isinstance(content, str) else "".join(
        b.get("text", "") for b in content or ()
        if isinstance(b, dict) and b.get("type") == "text")
    head = text.lstrip()[:40]
    if head.startswith("[Request interrupted"):
        return "abort"
    if okind in ("task-notification", "peer", "plugin", "system") or \
            row.get("promptSource") == "system" or \
            head.startswith("<task-notification"):
        return "background"
    if head.startswith(("<command-name", "<local-command")):
        return None
    return "human"


def extract_claude_session(lines, counts: dict | None = None
                           ) -> ClaudeSession | None:
    """One Claude Code file's lines -> its requests, in file order.

    Lines of one API response share `message.id` and are merged (first
    timestamp, last timestamp, the largest output count). A request starts at
    the latest event that could have triggered it since the previous request
    ended (a tool result, the person's message, a background notification),
    else at its own first line, and ends at its last line. Responses without
    usage (API errors are logged as `<synthetic>`) are not replayed; they
    mark the next gap as an outage.
    """
    counts = counts if counts is not None else {}
    reqs: dict[str, dict] = {}              # message id -> merged lines
    order: list[str] = []
    events: list[tuple[int, float, str]] = []   # (line no, ts, kind)
    entry = ""
    spawned_by = ""
    for n, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            counts["torn_lines"] = counts.get("torn_lines", 0) + 1
            continue
        typ = row.get("type")
        if not entry and row.get("entrypoint"):
            entry = row["entrypoint"]
        try:
            t = _iso_s(row["timestamp"]) if row.get("timestamp") else None
        except ValueError:
            t = None
        if typ == "assistant":
            msg = row.get("message") or {}
            usage = msg.get("usage") or {}
            mid = msg.get("id") or row.get("requestId") or f"line{n}"
            prompt = (usage.get("input_tokens", 0) or 0) + (
                usage.get("cache_read_input_tokens", 0) or 0) + (
                usage.get("cache_creation_input_tokens", 0) or 0)
            if msg.get("model") == "<synthetic>" or (prompt <= 0 and
                                                     not usage.get("output_tokens")):
                if t is not None:
                    counts["dropped_error"] = counts.get("dropped_error", 0) + 1
                    events.append((n, t, "error"))
                continue
            if t is None:
                continue
            r = reqs.get(mid)
            if r is None:
                r = reqs[mid] = {"first": n, "last": n, "t0": t, "t1": t,
                                 "prompt": 0, "out": 0, "calls": []}
                order.append(mid)
            r["last"] = n
            r["t0"], r["t1"] = min(r["t0"], t), max(r["t1"], t)
            r["prompt"] = max(r["prompt"], prompt)
            r["out"] = max(r["out"], usage.get("output_tokens", 0) or 0)
            for b in msg.get("content") or ():
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    r["calls"].append((b.get("id", ""), b.get("name", "")))
        elif typ == "user" and t is not None:
            k = _claude_user_kind(row)
            if k:
                events.append((n, t, k))
        elif typ == "system" and t is not None:
            if row.get("subtype") == "compact_boundary":
                events.append((n, t, "compaction"))
    out: list[TraceRequest] = []
    ids: dict[str, int] = {}
    prev = None
    t_first = None
    for mid in order:
        r = reqs[mid]
        if r["prompt"] <= 0:
            counts["dropped_error"] = counts.get("dropped_error", 0) + 1
            continue
        cand, marks = [], set()
        lo = prev["last"] if prev else -1
        for n, t, k in events:
            if lo < n < r["first"]:
                if k in ("tool", "human", "background"):
                    if t <= r["t0"] + 1.0:
                        cand.append((t, k))
                elif k == "abort":
                    marks.add("abort")
                    cand.append((t, "abort"))
                else:
                    marks.add(k)
        if prev is None:
            start, after, kind, gap, human = r["t0"], "start", "start", None, 0.0
            if cand:
                start = min(r["t0"], max(t for t, _ in cand))
        else:
            if cand:
                start, trig = max(cand, key=lambda c: c[0])
            else:
                start, trig = r["t0"], ("tool" if prev["calls"] else "human")
            if trig == "background":
                marks.add("background")
            elif trig == "human":
                # the person's own wait is not an outage or a compaction,
                # even if one happened somewhere in it
                marks -= {"error", "compaction"}
            after = "tool" if trig in ("tool", "background") else "user"
            gap = round(max(0.0, start - prev["t1"]), 3)
            tools = tuple(sorted({_CLAUDE_TOOLS.get(n_, "other")
                                  for _, n_ in prev["calls"]}))
            human = 0.0
            if after == "tool" and "ask_user_question" in tools:
                marks.add("ask")
                human = gap
            kind = classify(after, human, gap, marks)
        tools = (tuple(sorted({_CLAUDE_TOOLS.get(n_, "other")
                               for _, n_ in prev["calls"]}))
                 if prev is not None and after == "tool" else ())
        if t_first is None:
            t_first = start
        out.append(TraceRequest(
            prompt_tokens=int(r["prompt"]), output_tokens=int(r["out"]),
            gap_s=gap, after=after, tools=tools, kind=kind,
            human_s=round(min(human, gap or 0.0), 3),
            service_s=max(0.0, round(r["t1"] - start, 3))))
        for cid, _ in r["calls"]:
            if cid:
                ids[cid] = len(out) - 1
        prev = r
    if not out:
        return None
    return ClaudeSession(tuple(out), t_first or 0.0, entry, ids)


def _claude_files(paths) -> list[Path]:
    files: list[Path] = []
    for p in map(Path, paths):
        files += sorted(p.rglob("*.jsonl")) if p.is_dir() else [p]
    return files


def extract_claude(paths, include_headless: bool = False,
                   exclude: tuple[str, ...] = ()) -> AgentTrace:
    """Claude Code session files, or directories searched for `*.jsonl`
    (the subagents' files under `<session>/subagents/` are found there and
    linked to the session that launched them)."""
    import fnmatch
    counts: dict = {"permission_log": False,
                    "permission_waits": "no log: counted as tool time"}
    units: dict[Path, ClaudeSession] = {}
    for f in _claude_files(paths):
        if any(fnmatch.fnmatch(str(f), pat) for pat in exclude):
            continue
        try:
            with f.open(encoding="utf-8") as fh:
                u = extract_claude_session(fh, counts)
        except (OSError, ValueError, KeyError, TypeError) as e:
            raise ValueError(f"{f}: {type(e).__name__}: {e}") from e
        if u is None:
            continue
        if f.parent.name == "subagents":
            try:
                meta = json.loads(f.with_suffix(".meta.json").read_text())
                u.spawned_by = str(meta.get("toolUseId", ""))
            except (OSError, ValueError):
                pass
        units[f] = u
    # headless sessions and their subagents are dropped together
    def root_of(f: Path) -> Path:
        return f.parent.parent.with_suffix(".jsonl") \
            if f.parent.name == "subagents" else f
    dropped = {f for f in units
               if not include_headless
               and (units[root_of(f)] if root_of(f) in units
                    else units[f]).entrypoint in HEADLESS_ENTRYPOINTS}
    if dropped:
        counts["headless_sessions_left_out"] = sum(
            1 for f in dropped if f.parent.name != "subagents")
        counts["headless_subagents_left_out"] = sum(
            1 for f in dropped if f.parent.name == "subagents")
    kept = {f: u for f, u in units.items() if f not in dropped}
    # order: a session, then the subagents it launched, by start
    roots = [f for f in kept if f.parent.name != "subagents"]
    orphans = [f for f in kept if f.parent.name == "subagents"
               and root_of(f) not in kept]
    seq: list[Path] = []
    for r in roots + orphans:
        seq.append(r)
        fam = sorted((f for f in kept if f.parent.name == "subagents"
                      and root_of(f) == r), key=lambda f: kept[f].t0)
        seq += fam
    index = {f: i for i, f in enumerate(seq)}
    links: list[SessionLink | None] = []
    for f in seq:
        if f.parent.name != "subagents" or root_of(f) not in kept:
            links.append(None)
            continue
        u = kept[f]
        par = root_of(f)
        # the nearest owner of the spawning tool call: another subagent of
        # the same session, else the session itself
        for g in seq:
            if g is not f and root_of(g) == par and u.spawned_by in kept[g].tool_use_ids:
                par = g
                break
        spawn = kept[par].tool_use_ids.get(u.spawned_by) if u.spawned_by else None
        links.append(SessionLink(index[par], max(0.0, u.t0 - kept[par].t0), spawn))
    if not seq:
        raise ValueError("no Claude Code request with token usage in "
                         + ", ".join(map(str, paths)))
    counts["subagent_sessions"] = sum(1 for l in links if l)
    return AgentTrace(source="claude",
                      sessions=tuple(kept[f].requests for f in seq),
                      notes=counts,
                      links=tuple(links) if any(links) else ())


# ============================================================================
# what the trace says, and the smooth workload with the same averages
# ============================================================================
def _pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    return xs[min(len(xs) - 1, int(q / 100 * len(xs)))]


def think_pool(trace: AgentTrace, park_s: float = PARK_S) -> list[float]:
    """The trace's own human waits after a finished answer: what an inactive
    gap is replaced by under the "sample" policy."""
    return [r.gap_s for s in trace.sessions for r in s
            if r.kind == "think" and r.gap_s is not None and r.gap_s <= park_s]


def replay_gap(r: TraceRequest, policy: str, max_gap_s: float, rng,
               pool: list[float]) -> float | None:
    """The wait a replay puts before request `r` (None for a recording's
    first request, which the caller draws). Inactive gaps (INACTIVE) follow
    `policy`; every other gap is the recorded one, capped at max_gap_s."""
    if r.gap_s is None:
        return None
    if r.kind in INACTIVE:
        if policy == "drop":
            return 0.0
        if policy == "sample":
            if not pool:
                raise ValueError(NO_THINK_POOL)
            return min(pool[rng.randrange(len(pool))], max_gap_s)
    return min(r.gap_s, max_gap_s)


NO_THINK_POOL = ("the gap policy 'sample' draws inactive gaps from the "
                 "trace's think gaps, and this trace has none: use "
                 "--replay-gap-policy cap or drop")


def mean_gap(trace: AgentTrace, policy: str = DEFAULT_GAP_POLICY,
             max_gap_s: float = PARK_S) -> float | None:
    """The mean wait a replay under `policy` puts between requests."""
    pool = think_pool(trace)
    sampled = (sum(min(g, max_gap_s) for g in pool) / len(pool)
               if pool else None)
    gaps = []
    for s in trace.sessions:
        for r in s:
            if r.gap_s is None:
                continue
            if r.kind in INACTIVE and policy == "drop":
                gaps.append(0.0)
            elif r.kind in INACTIVE and policy == "sample":
                if sampled is None:
                    raise ValueError(NO_THINK_POOL)
                gaps.append(sampled)
            else:
                gaps.append(min(r.gap_s, max_gap_s))
    return sum(gaps) / len(gaps) if gaps else None


def offered_load(trace: AgentTrace, n_sessions: int,
                 policy: str = DEFAULT_GAP_POLICY,
                 max_gap_s: float = PARK_S) -> dict | None:
    """What n replayed sessions send if the server is as fast as the recorded
    one: each session loops request -> wait, so it sends one request every
    (mean service + mean wait) seconds. None when the trace has no service
    times (a version-1 trace)."""
    reqs = [r for s in trace.sessions for r in s]
    svc = [r.service_s for r in reqs if r.service_s is not None]
    gap = mean_gap(trace, policy, max_gap_s)
    if not svc or gap is None:
        return None
    cycle = sum(svc) / len(svc) + gap
    rps = n_sessions / cycle
    out = sum(r.output_tokens for r in reqs) / len(reqs)
    busy = sum(svc) / len(svc) / cycle
    return {"cycle_s": cycle, "requests_per_s": rps,
            "output_tok_per_s": rps * out, "generating_share": busy}


def _stats(xs):
    return {"n": len(xs), "p50": _pct(xs, 50), "p90": _pct(xs, 90),
            "mean": sum(xs) / len(xs) if xs else float("nan")}


def gaps_by_kind(trace: AgentTrace) -> dict[str, dict]:
    """Every gap, uncapped, by kind: count, p50, p90, max and total hours."""
    by: dict[str, list[float]] = {k: [] for k in KINDS if k != "start"}
    for s in trace.sessions:
        for r in s:
            if r.gap_s is not None:
                by.setdefault(r.kind, []).append(r.gap_s)
    return {k: {"n": len(v), "p50": _pct(v, 50), "p90": _pct(v, 90),
                "max": max(v) if v else float("nan"),
                "total_h": sum(v) / 3600, "human_s": sum(
                    r.human_s for s in trace.sessions for r in s
                    if r.kind == k)}
            for k, v in by.items()}


def summarize(trace: AgentTrace, park_s: float = PARK_S) -> dict:
    reqs = [r for s in trace.sessions for r in s]
    prompts = [r.prompt_tokens for r in reqs]
    outs = [r.output_tokens for r in reqs]
    tool_gaps = [min(r.gap_s, park_s) for r in reqs
                 if r.after == "tool" and r.gap_s is not None
                 and r.kind not in INACTIVE and r.kind != "background"]
    user_gaps = [min(r.gap_s, park_s) for r in reqs
                 if r.after == "user" and r.gap_s is not None
                 and r.kind not in INACTIVE]
    by_tool: dict[str, list[float]] = {}
    for r in reqs:
        if r.after == "tool" and r.gap_s is not None and r.kind == "tool":
            for t in r.tools or ("other",):
                by_tool.setdefault(t, []).append(min(r.gap_s, park_s))
    links = trace.links or (None,) * len(trace.sessions)
    n_user = sum(1 for sess, l in zip(trace.sessions, links) for r in sess
                 if r.after == "user" or (r.after == "start" and l is None))
    return {"sessions": len(trace.sessions), "requests": len(reqs),
            "subagent_sessions": sum(1 for l in links if l),
            "requests_per_user_message": len(reqs) / max(n_user, 1),
            "prompt_tokens": _stats(prompts), "output_tokens": _stats(outs),
            "tool_wait_s": _stats(tool_gaps), "user_wait_s": _stats(user_gaps),
            "tool_wait_by_tool_s": {t: _stats(v) for t, v in
                                    sorted(by_tool.items(),
                                           key=lambda kv: -len(kv[1]))},
            "gaps_by_kind": gaps_by_kind(trace)}


def equivalent_workload(trace: AgentTrace, park_s: float = PARK_S,
                        policy: str = DEFAULT_GAP_POLICY) -> dict:
    """`[workload]` overrides with the trace's averages.

    prompt length   log-normal fitted to every request's prompt (median and
                    log-sd), the distribution `Workload.sample` draws
    think time      mean wait between requests as the replay sends them:
                    inactive gaps under `policy`, the rest capped at park_s
    turn tokens     mean prompt growth beyond the previous reply
    output          mean output tokens
    misses          none forced: a replay only goes cold where the recording
                    did (a session's first request, a shrinking prompt)
    """
    reqs = [r for s in trace.sessions for r in s]
    logs = [math.log(r.prompt_tokens) for r in reqs]
    mu = sum(logs) / len(logs)
    sigma = (math.sqrt(sum((x - mu) ** 2 for x in logs) / (len(logs) - 1))
             if len(logs) > 1 else 0.0)
    think = mean_gap(trace, policy, park_s)
    growth = []
    for s in trace.sessions:
        for prev, cur in zip(s, s[1:]):
            d = cur.prompt_tokens - prev.prompt_tokens - prev.output_tokens
            if cur.prompt_tokens >= prev.prompt_tokens:
                growth.append(max(d, 0))
    # no wait and no growth to average are not zero or 30 s: the predictions
    # priced on them would be a guess dressed as the trace's
    if think is None:
        raise ValueError(
            "the trace has no wait between two requests (every session is "
            f"one request, or the {policy!r} gap policy leaves none), so "
            "workload.think_time_s cannot be read from it")
    if not growth:
        raise ValueError(
            "no session in the trace has two requests whose prompt did not "
            "shrink, so workload.warm_turn_tokens (the prompt growth per "
            "turn) cannot be read from it")
    return {
        "user_prompt_median_tokens": int(round(math.exp(mu))),
        "user_prompt_sigma": round(max(sigma, 0.01), 3),
        "think_time_s": _round_think(think),
        "warm_turn_tokens": int(round(sum(growth) / len(growth))),
        "max_output_tokens": max(1, int(round(
            sum(r.output_tokens for r in reqs) / len(reqs)))),
        "subagent_ratio": 0.0,
        "miss_rate": 0.0,
        "image_share": 0.0,
    }


def _round_think(think: float) -> float:
    """One decimal, but never down to zero: a trace of short waits keeps a
    positive think time (the config rejects 0)."""
    r = round(think, 1)
    return r if r > 0 else max(round(think, 3), 0.001)


# ============================================================================
# `ws trace pi`
# ============================================================================
DEFAULT_PI_SESSIONS = "~/.pi/agent/sessions"


def _fmt_stats(s: dict, unit: str = "") -> str:
    f = ",.1f" if unit else ",.0f"
    return (f"p50 {s['p50']:{f}}{unit}, p90 {s['p90']:{f}}{unit}, "
            f"mean {s['mean']:{f}}{unit} (n={s['n']:,})") if s["n"] else "none"


def cmd_trace(args) -> int:
    import sys
    claude = args.source == "claude"
    paths = args.paths or [str(Path(
        DEFAULT_CLAUDE_SESSIONS if claude else DEFAULT_PI_SESSIONS
    ).expanduser())]
    try:
        if claude:
            trace = extract_claude(paths, args.include_headless,
                                   exclude=tuple(args.exclude or ()))
        else:
            trace = extract_pi(paths, permission_log=args.permission_log,
                               exclude=tuple(args.exclude or ()))
    except (OSError, ValueError, KeyError) as e:
        print(f"ws trace: error: {e}", file=sys.stderr)
        return 2
    out = json.dumps(trace.to_dict(), indent=1)
    if args.output == "-":
        print(out)
    else:
        Path(args.output).write_text(out + "\n", encoding="utf-8")
    s = summarize(trace)
    say = (lambda m: print(m, file=sys.stderr)) if args.output == "-" else print
    say(f"{s['requests']:,} requests in {s['sessions']} sessions"
        + ("" if args.output == "-" else f" -> {args.output}")
        + "; no prompt or reply text kept")
    say(f"  prompt tokens      {_fmt_stats(s['prompt_tokens'])}")
    say(f"  output tokens      {_fmt_stats(s['output_tokens'])}")
    say(f"  requests per user message  {s['requests_per_user_message']:.1f}")
    say(f"  wait on a tool     {_fmt_stats(s['tool_wait_s'], ' s')}")
    for t, ts in list(s["tool_wait_by_tool_s"].items())[:8]:
        say(f"    {t:<16} {_fmt_stats(ts, ' s')}")
    say(f"  wait on the person {_fmt_stats(s['user_wait_s'], ' s')}")
    notes = trace.notes or {}
    dropped = {k[len("dropped_"):]: v for k, v in notes.items()
               if k.startswith("dropped_")}
    if s.get("subagent_sessions"):
        say(f"  {s['subagent_sessions']} of the sessions are subagents "
            "(linked to their parent; `ws replay` plays them independently)")
    say("gaps by what they waited on (uncapped; "
        + ("permission prompts timed from pi-permission-system's log"
           if notes.get("permission_log") else
           "no permission log: a permission wait is counted as tool time"
           if claude else
           "no permission log: a denied prompt marks its gap, an approved "
           "one is counted as tool time") + "):")
    say(f"  {'kind':<11} {'n':>5} {'p50':>9} {'p90':>9} {'max':>10} "
        f"{'total':>8}")
    for k, g in s["gaps_by_kind"].items():
        if not g["n"]:
            continue
        say(f"  {k:<11} {g['n']:>5} {g['p50']:>8.1f}s {g['p90']:>8.1f}s "
            f"{g['max']:>9.0f}s {g['total_h']:>7.2f}h"
            + ("  (replay: --replay-gap-policy)" if k in INACTIVE else ""))
    if notes.get("headless_sessions_left_out"):
        say(f"  left out: {notes['headless_sessions_left_out']} headless "
            "session(s) (--include-headless keeps them)")
    if dropped or notes.get("forked_sessions") or notes.get("torn_lines"):
        say("  not replayed: " + ", ".join(
            [f"{v} {k} request(s)" for k, v in sorted(dropped.items())]
            + ([f"{notes['torn_lines']} torn line(s)"]
               if notes.get("torn_lines") else []))
            + (f"; {notes['forked_sessions']} forked session(s), replayed "
               "from their own first request" if notes.get("forked_sessions")
               else ""))
    say("the same averages as a smooth [workload] (what `ws test --replay` "
        "predicts against):")
    try:
        eq = equivalent_workload(trace)
    except ValueError as e:      # the trace is written; its averages need a policy
        say(f"  none under the default gap policy: {e}")
        return 0
    for k, v in eq.items():
        say(f"  {k} = {v}")
    return 0


def add_subparser(sub) -> None:
    import argparse
    p = sub.add_parser(
        "trace", help="extract an agent-session trace for `ws test --replay`",
        description="Read coding-agent session files and write their timing "
                    "and token counts, without any text, as a trace that "
                    "`ws test --replay` plays back.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  ws trace pi -o trace.json\n"
               "  ws trace pi ~/.pi/agent/sessions/--home-me-work-repo-- "
               "-o trace.json\n"
               "  ws trace claude -o trace.json\n")
    tsub = p.add_subparsers(dest="source", required=True)
    q = tsub.add_parser("pi", help="pi session files (*.jsonl)")
    q.add_argument("paths", nargs="*", metavar="PATH",
                   help=f"session files or directories (default "
                        f"{DEFAULT_PI_SESSIONS})")
    q.add_argument("--permission-log", metavar="PATH",
                   help="pi-permission-system's review log, which times each "
                        f"permission prompt (default {PERMISSION_LOG}; "
                        "skipped when absent)")
    q.add_argument("--exclude", action="append", metavar="GLOB",
                   help="skip session files whose path matches (repeatable), "
                        "e.g. '*scratchpad*' for a test harness's sessions")
    q.add_argument("-o", "--output", default="trace.json",
                   help="trace file to write ('-' for stdout; default "
                        "trace.json)")
    q.set_defaults(fn=cmd_trace)
    c = tsub.add_parser("claude", help="Claude Code session files (*.jsonl)")
    c.add_argument("paths", nargs="*", metavar="PATH",
                   help=f"session files or directories (default "
                        f"{DEFAULT_CLAUDE_SESSIONS})")
    c.add_argument("--include-headless", action="store_true",
                   help="keep `claude -p` and SDK sessions (entrypoint "
                        "sdk-cli, sdk-ts), left out by default")
    c.add_argument("--exclude", action="append", metavar="GLOB",
                   help="skip session files whose path matches (repeatable)")
    c.add_argument("-o", "--output", default="trace.json",
                   help="trace file to write ('-' for stdout; default "
                        "trace.json)")
    c.set_defaults(fn=cmd_trace)


def main(argv=None) -> int:
    """This file alone, without the package, on a machine that only has the
    sessions:

        curl -fsSLO https://raw.githubusercontent.com/T0mSIlver/working-set/main/src/workingset/agent_trace.py
        python3 agent_trace.py pi -o trace.json

    It needs Python 3.10+ and nothing else, and writes the same trace as
    `ws trace pi`."""
    import argparse
    parser = argparse.ArgumentParser(prog="agent_trace.py")
    sub = parser.add_subparsers(dest="cmd", required=True)
    add_subparser(sub)
    args = parser.parse_args(["trace", *(argv if argv is not None
                                         else __import__("sys").argv[1:])])
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
