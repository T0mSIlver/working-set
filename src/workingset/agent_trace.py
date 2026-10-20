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
from dataclasses import dataclass
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
KINDS = ("start", "tool", "permission", "ask", "think", "abort", "branch",
         "compaction", "error", "resume", "idle")
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
class AgentTrace:
    source: str
    sessions: tuple[tuple[TraceRequest, ...], ...]
    # dropped requests and other counts `ws trace` reports (not replayed)
    notes: dict | None = None

    @property
    def n_requests(self) -> int:
        return sum(len(s) for s in self.sessions)

    def to_dict(self) -> dict:
        d = {"format": FORMAT, "version": VERSION, "source": self.source,
             "sessions": [{"requests": [r.to_dict() for r in s]}
                          for s in self.sessions]}
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
        sessions = []
        for s in d["sessions"]:
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
                sessions.append(reqs)
        if not sessions:
            raise ValueError("the trace holds no request")
        return cls(source=d.get("source", ""), sessions=tuple(sessions),
                   notes=d.get("notes"))


def load(path: str | Path) -> AgentTrace:
    return AgentTrace.from_dict(json.loads(Path(path).read_text()))


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
        fh = p.open()
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
            with f.open() as fh:
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
        if policy == "sample" and pool:
            return min(pool[rng.randrange(len(pool))], max_gap_s)
    return min(r.gap_s, max_gap_s)


def mean_gap(trace: AgentTrace, policy: str = DEFAULT_GAP_POLICY,
             max_gap_s: float = PARK_S) -> float | None:
    """The mean wait a replay under `policy` puts between requests."""
    pool = think_pool(trace)
    sampled = (min(sum(pool) / len(pool), max_gap_s) if pool else None)
    gaps = []
    for s in trace.sessions:
        for r in s:
            if r.gap_s is None:
                continue
            if r.kind in INACTIVE and policy == "drop":
                gaps.append(0.0)
            elif r.kind in INACTIVE and policy == "sample" and sampled is not None:
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
                 and r.kind not in INACTIVE]
    user_gaps = [min(r.gap_s, park_s) for r in reqs
                 if r.after == "user" and r.gap_s is not None
                 and r.kind not in INACTIVE]
    by_tool: dict[str, list[float]] = {}
    for r in reqs:
        if r.after == "tool" and r.gap_s is not None and r.kind == "tool":
            for t in r.tools or ("other",):
                by_tool.setdefault(t, []).append(min(r.gap_s, park_s))
    n_user = sum(1 for r in reqs if r.after in ("user", "start"))
    return {"sessions": len(trace.sessions), "requests": len(reqs),
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
    return {
        "user_prompt_median_tokens": int(round(math.exp(mu))),
        "user_prompt_sigma": round(max(sigma, 0.01), 3),
        "think_time_s": round(think, 1) if think is not None else 30.0,
        "warm_turn_tokens": int(round(sum(growth) / len(growth))) if growth else 0,
        "max_output_tokens": max(1, int(round(
            sum(r.output_tokens for r in reqs) / len(reqs)))),
        "subagent_ratio": 0.0,
        "miss_rate": 0.0,
        "image_share": 0.0,
    }


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
    paths = args.paths or [str(Path(DEFAULT_PI_SESSIONS).expanduser())]
    try:
        trace = extract_pi(paths, permission_log=args.permission_log,
                           exclude=tuple(args.exclude or ()))
    except (OSError, ValueError, KeyError) as e:
        print(f"ws trace: error: {e}", file=sys.stderr)
        return 2
    out = json.dumps(trace.to_dict(), indent=1)
    if args.output == "-":
        print(out)
    else:
        Path(args.output).write_text(out + "\n")
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
    say("gaps by what they waited on (uncapped; "
        + ("permission prompts timed from pi-permission-system's log"
           if notes.get("permission_log") else
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
    for k, v in equivalent_workload(trace).items():
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
               "-o trace.json\n")
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
