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
VERSION = 1

# pi's builtin tools plus the common ones every harness has. Anything else (MCP
# servers, extensions) can name an internal service, so it is reported as
# "other".
KNOWN_TOOLS = frozenset({"bash", "read", "edit", "write", "grep", "find", "ls",
                         "glob", "todo", "web_fetch", "web_search", "task"})

# A gap longer than this means the person left; `ws test --replay` caps the
# wait there. The same boundary as scripts/think_time_trace.py's PARK_S.
PARK_S = 1800.0


@dataclass(frozen=True)
class TraceRequest:
    prompt_tokens: int
    output_tokens: int
    gap_s: float | None
    after: str
    tools: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        d = {"prompt_tokens": self.prompt_tokens,
             "output_tokens": self.output_tokens,
             "gap_s": None if self.gap_s is None else round(self.gap_s, 3),
             "after": self.after}
        if self.tools:
            d["tools"] = list(self.tools)
        return d


@dataclass(frozen=True)
class AgentTrace:
    source: str
    sessions: tuple[tuple[TraceRequest, ...], ...]

    @property
    def n_requests(self) -> int:
        return sum(len(s) for s in self.sessions)

    def to_dict(self) -> dict:
        return {"format": FORMAT, "version": VERSION, "source": self.source,
                "sessions": [{"requests": [r.to_dict() for r in s]}
                             for s in self.sessions]}

    @classmethod
    def from_dict(cls, d: dict) -> "AgentTrace":
        if d.get("format") != FORMAT:
            raise ValueError(f"not a {FORMAT} file")
        if d.get("version") != VERSION:
            raise ValueError(f"trace version {d.get('version')!r}, "
                             f"this ws reads version {VERSION}")
        sessions = []
        for s in d["sessions"]:
            reqs = tuple(TraceRequest(
                prompt_tokens=int(r["prompt_tokens"]),
                output_tokens=int(r["output_tokens"]),
                gap_s=None if r.get("gap_s") is None else float(r["gap_s"]),
                after=r["after"], tools=tuple(r.get("tools", ())))
                for r in s["requests"])
            if reqs:
                sessions.append(reqs)
        if not sessions:
            raise ValueError("the trace holds no request")
        return cls(source=d.get("source", ""), sessions=tuple(sessions))


def load(path: str | Path) -> AgentTrace:
    return AgentTrace.from_dict(json.loads(Path(path).read_text()))


def _iso_s(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def _tool_name(name: str) -> str:
    return name if name in KNOWN_TOOLS else "other"


def extract_pi_session(lines) -> tuple[TraceRequest, ...]:
    """One pi session file's lines -> its requests, in file order.

    A request starts at the assistant message's `timestamp` (ms, when pi sent
    it) and ends at the entry's ISO `timestamp` (when the reply was written).
    Requests without usage (aborted, errored) are dropped, and the gap after
    them is folded into the next one.
    """
    out: list[TraceRequest] = []
    last_end: float | None = None
    waited_on, tools = "start", ()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("type") != "message":
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
                continue
            start = msg["timestamp"] / 1000.0
            gap = (None if last_end is None
                   else round(max(0.0, start - last_end), 3))
            out.append(TraceRequest(
                prompt_tokens=int(prompt),
                output_tokens=int(usage.get("output", 0)),
                gap_s=gap, after=waited_on if out else "start",
                tools=tools if waited_on == "tool" else ()))
            last_end = _iso_s(row["timestamp"])
            tools = tuple(sorted({_tool_name(c["name"]) for c in msg["content"]
                                  if c.get("type") == "toolCall"}))
            waited_on = "tool" if tools else "user"
        elif role == "toolResult":
            waited_on = "tool"
    return tuple(out)


def extract_pi(paths) -> AgentTrace:
    """pi session files, or directories searched for `*.jsonl`."""
    files: list[Path] = []
    for p in map(Path, paths):
        files += sorted(p.rglob("*.jsonl")) if p.is_dir() else [p]
    sessions = []
    for f in files:
        with f.open() as fh:
            reqs = extract_pi_session(fh)
        if reqs:
            sessions.append(reqs)
    if not sessions:
        raise ValueError("no pi request with token usage in "
                         + ", ".join(map(str, paths)))
    return AgentTrace(source="pi", sessions=tuple(sessions))


# ============================================================================
# what the trace says, and the smooth workload with the same averages
# ============================================================================
def _pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    return xs[min(len(xs) - 1, int(q / 100 * len(xs)))]


def summarize(trace: AgentTrace, park_s: float = PARK_S) -> dict:
    reqs = [r for s in trace.sessions for r in s]
    prompts = [r.prompt_tokens for r in reqs]
    outs = [r.output_tokens for r in reqs]
    tool_gaps = [min(r.gap_s, park_s) for r in reqs
                 if r.after == "tool" and r.gap_s is not None]
    user_gaps = [min(r.gap_s, park_s) for r in reqs
                 if r.after == "user" and r.gap_s is not None]
    by_tool: dict[str, list[float]] = {}
    for r in reqs:
        if r.after == "tool" and r.gap_s is not None:
            for t in r.tools or ("other",):
                by_tool.setdefault(t, []).append(min(r.gap_s, park_s))
    n_user = sum(1 for r in reqs if r.after in ("user", "start"))

    def stats(xs):
        return {"n": len(xs), "p50": _pct(xs, 50), "p90": _pct(xs, 90),
                "mean": sum(xs) / len(xs) if xs else float("nan")}
    return {"sessions": len(trace.sessions), "requests": len(reqs),
            "requests_per_user_message": len(reqs) / max(n_user, 1),
            "prompt_tokens": stats(prompts), "output_tokens": stats(outs),
            "tool_wait_s": stats(tool_gaps), "user_wait_s": stats(user_gaps),
            "tool_wait_by_tool_s": {t: stats(v) for t, v in
                                    sorted(by_tool.items(),
                                           key=lambda kv: -len(kv[1]))}}


def equivalent_workload(trace: AgentTrace, park_s: float = PARK_S) -> dict:
    """`[workload]` overrides with the trace's averages.

    prompt length   log-normal fitted to every request's prompt (median and
                    log-sd), the distribution `Workload.sample` draws
    think time      mean wait between requests, parked waits capped at park_s
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
    gaps = [min(r.gap_s, park_s) for r in reqs if r.gap_s is not None]
    growth = []
    for s in trace.sessions:
        for prev, cur in zip(s, s[1:]):
            d = cur.prompt_tokens - prev.prompt_tokens - prev.output_tokens
            if cur.prompt_tokens >= prev.prompt_tokens:
                growth.append(max(d, 0))
    return {
        "user_prompt_median_tokens": int(round(math.exp(mu))),
        "user_prompt_sigma": round(max(sigma, 0.01), 3),
        "think_time_s": round(sum(gaps) / len(gaps), 1) if gaps else 30.0,
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
        trace = extract_pi(paths)
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
    say(f"  wait on the person {_fmt_stats(s['user_wait_s'], ' s')}"
        f" (capped at {PARK_S:g} s)")
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
    q.add_argument("-o", "--output", default="trace.json",
                   help="trace file to write ('-' for stdout; default "
                        "trace.json)")
    q.set_defaults(fn=cmd_trace)
