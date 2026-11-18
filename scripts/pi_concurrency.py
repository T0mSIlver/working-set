#!/usr/bin/env python3
"""How many pi sessions one person has in use at the same moment.

    python pi_concurrency.py
    python pi_concurrency.py --model glm --since 2026-09-01 --idle-min 20

`ws replay` answers "this deployment serves N agent sessions in use at once".
This script gives, from one person's own pi history, how many sessions that
person has in use at once, so N can be turned into people. It reads
~/.pi/agent/sessions/*/*.jsonl (directories named --tmp* are skipped), and
prints and writes numbers only: no prompt text, no file path, no project
name. Standard library only; runs with plain python 3.7+ on Windows and Linux.

Definitions
  event       any `message` line of a session file with a readable timestamp
              (user, assistant, tool result).
  busy period a session is in use from its first event to its last, cut
              wherever two consecutive events are more than --idle-min
              minutes apart (default 30: the cap `ws replay` puts on a
              recorded gap, PARK_S in workingset/agent_trace.py). Each piece
              is a busy period. A piece with one event is in use for 0 s: it
              adds nothing to any duration, but is counted in the totals.
  subagent    a session file whose header line names a parent session file
              (`parentSession`, the only parent link pi writes) is not a
              separate session in use: it is merged into its root session
              (union of busy periods), as `ws replay` counts a session and
              what it spawned as one. `ws trace pi` itself reads no such
              link (it only counts these headers as forked_sessions), so
              this is the one place the link is followed. A header whose
              parent file is not in the set read, is ambiguous or loops is
              counted ("parent link unresolved") and the file is kept as a
              session of its own.
  concurrency at an instant, the number of root sessions with a busy period
              covering it.

Output (printed, and written as JSON to --out)
  1. totals: files read, root sessions, subagent files merged, files skipped
     and why, first and last day covered.
  2. over the time with at least one session in use: the time-weighted mean
     concurrency, its p50, p90 and max (p50 and p90: the smallest
     concurrency that the session-time at or below it reaches 50% and 90%
     of the in-use time), and the share of that time at 1, 2, 3, 4, 5+.
  3. per day (machine's local time) with any use: hours with at least one
     session in use, and peak concurrency; median and p90 across days (linear
     interpolation). The JSON also holds the list, dates only.
  4. the mean and p90 again for --idle-min 10, 30 and 60.

A file that cannot be read or parsed, a session with no timestamped message,
and a line that is not JSON are counted and printed, never dropped silently.
"""
import argparse
import collections
import json
import re
import sys
from datetime import datetime, time as dtime, timedelta
from pathlib import Path

SENSITIVITY = (10, 30, 60)
UNRESOLVED = "parent link unresolved"
SHARES = ("1", "2", "3", "4", "5+")


def ts(d):
    """The line's ISO `timestamp` as epoch seconds, or None."""
    t = d.get("timestamp")
    try:
        return datetime.fromisoformat(str(t).replace("Z", "+00:00")).timestamp()
    except (ValueError, OverflowError, OSError):
        return None


def _num(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) else 0


class Session:
    """What one session file gave: its event times, its requests' models,
    the parent file name its header names, and what did not parse."""

    def __init__(self, path):
        self.name = path.name
        self.events, self.models = [], []
        self.parent = None
        self.lines_ok = self.lines_bad = self.events_untimed = 0
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                except ValueError:
                    d = None
                if not isinstance(d, dict):
                    self.lines_bad += 1
                    continue
                self.lines_ok += 1
                typ = d.get("type")
                if typ == "session":
                    if self.parent is None and d.get("parentSession"):
                        self.parent = str(d["parentSession"])
                    continue
                m = d.get("message") if typ == "message" else None
                if not isinstance(m, dict):
                    continue
                now = ts(d)
                if now is None:
                    self.events_untimed += 1
                    continue
                self.events.append(now)
                u = m.get("usage")
                u = u if isinstance(u, dict) else {}
                if (m.get("role") == "assistant" and _num(u.get("input"))
                        + _num(u.get("cacheRead")) + _num(u.get("cacheWrite")) > 0):
                    self.models.append(str(m.get("model") or "?"))
        self.events.sort()


def busy_periods(events, idle_s):
    """[(start, end, n_events)] of sorted event times, cut at gaps > idle_s."""
    out, start, prev, n = [], events[0], events[0], 1
    for t in events[1:]:
        if round(t - prev, 3) > idle_s:
            out.append((start, prev, n))
            start, n = t, 0
        prev, n = t, n + 1
    out.append((start, prev, n))
    return out


def union(intervals):
    out = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def segments(per_root):
    """[(start, end, concurrency)] over the time with >= 1 root in use.
    per_root: one list of disjoint (start, end) per root session."""
    pts = []
    for ivs in per_root:
        for s, e in ivs:
            if e > s:
                pts += [(s, 1), (e, -1)]
    pts.sort()
    segs, c, prev, i = [], 0, None, 0
    while i < len(pts):
        t = pts[i][0]
        if prev is not None and c > 0 and t > prev:
            segs.append((prev, t, c))
        while i < len(pts) and pts[i][0] == t:
            c += pts[i][1]
            i += 1
        prev = t
    return segs


def pct(xs, q):
    """q-th percentile (0-1) of xs by linear interpolation."""
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def stats(segs):
    """Time-weighted figures over the in-use time; None when there is none."""
    by = collections.defaultdict(float)
    for a, b, c in segs:
        by[c] += b - a
    tot = sum(by.values())
    if tot <= 0:
        return None

    def wq(q):
        cum = 0.0
        for c in sorted(by):
            cum += by[c]
            if cum >= q * tot * (1 - 1e-12):
                return c

    share = {"1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0, "5+": 0.0}
    for c, d in by.items():
        share[str(c) if c < 5 else "5+"] += d / tot
    return {"in_use_hours": tot / 3600,
            "mean": sum(c * d for c, d in by.items()) / tot,
            "p50": wq(0.5), "p90": wq(0.9), "max": max(by), "share": share}


def per_day(segs):
    """{date: [hours in use, peak concurrency]} in the machine's local time."""
    days = {}
    for a, b, c in segs:
        while a < b:
            day = datetime.fromtimestamp(a).date()
            nxt = datetime.combine(day + timedelta(days=1), dtime()).timestamp()
            if nxt <= a:
                nxt = a + 86400
            e = min(b, nxt)
            h = days.setdefault(day, [0.0, 0])
            h[0] += (e - a) / 3600
            h[1] = max(h[1], c)
            a = e
    return days


def families_segments(families, idle_s):
    """Segments of the root sessions, each the union of its members' busy
    periods. families: [[Session, ...]], events already cut by --since."""
    per_root = []
    for fam in families:
        ivs = [(s, e) for m in fam if m.events
               for s, e, _ in busy_periods(m.events, idle_s)]
        per_root.append(union(ivs))
    return segments(per_root)


def resolve_roots(live, counts):
    """{id(session): its root session}, following `parentSession`. A file
    whose link cannot be followed (parent not in the set, ambiguous name, a
    loop) is counted and is its own root."""
    by_name = collections.defaultdict(list)
    for s in live:
        by_name[s.name].append(s)

    def parent_of(s):
        if not s.parent:
            return None
        name = re.split(r"[\\/]", s.parent.rstrip("\\/"))[-1]
        c = by_name.get(name, [])
        return c[0] if len(c) == 1 and c[0] is not s else None

    root = {}
    for s in live:
        cur, seen = s, {id(s)}
        while True:
            p = parent_of(cur)
            if p is None:
                if cur is s and s.parent:
                    counts[UNRESOLVED] += 1
                break
            if id(p) in seen:
                cur = s
                counts[UNRESOLVED] += 1
                break
            seen.add(id(p))
            cur = p
        root[id(s)] = cur
    return root


def local_day(t):
    return datetime.fromtimestamp(t).date()


def analyse(a):
    counts = collections.Counter()
    files = sorted(a.sessions.glob("*/*.jsonl"))
    if not files:
        sys.exit(f"no session file in {a.sessions}/*/*.jsonl: point --sessions "
                 "at the folder that holds pi's session folders "
                 "(default ~/.pi/agent/sessions)")
    live, torn_files, torn_lines, untimed = [], 0, 0, 0
    for f in files:
        if f.parent.name.startswith("--tmp"):
            counts["in a --tmp folder"] += 1
            continue
        try:
            s = Session(f)
        except OSError:
            counts["unreadable"] += 1
            continue
        if s.lines_ok == 0 and s.lines_bad:
            counts["not parseable (no line is JSON)"] += 1
            continue
        torn_files += bool(s.lines_bad)
        torn_lines += s.lines_bad
        untimed += s.events_untimed
        if not s.events:
            counts["no message with a timestamp"] += 1
            continue
        live.append(s)

    root = resolve_roots(live, counts)
    fams = collections.OrderedDict()
    for s in live:
        fams.setdefault(id(root[id(s)]), []).append(s)
    sel = a.model.lower() if a.model else None
    cut = a.since_ts
    families, merged, n_single, n_periods = [], 0, 0, 0
    for rid, fam in fams.items():
        r = next(m for m in fam if id(m) == rid)
        if sel is not None:
            mine = sum(1 for m in r.models if sel in m.lower())
            if not r.models or mine < 0.9 * len(r.models):
                counts["left out by --model"] += len(fam)
                continue
        if cut is not None:
            for m in fam:
                m.events = [t for t in m.events if t >= cut]
            if not any(m.events for m in fam):
                counts["before --since"] += len(fam)
                continue
        families.append(fam)
        merged += len(fam) - 1
        for m in fam:
            if m.events:
                for _, _, n in busy_periods(m.events, a.idle_min * 60):
                    n_periods += 1
                    n_single += n == 1
    days_seen = sorted({local_day(t) for fam in families for m in fam
                        for t in m.events})
    totals = {
        "session_files_found": len(files),
        "session_files_read": len(files) - counts["in a --tmp folder"]
        - counts["unreadable"],
        "root_sessions": len(families),
        "subagent_files_merged": merged,
        "parent_link_unresolved": counts.pop(UNRESOLVED, 0),
        "files_skipped": dict(counts),
        "files_with_lines_that_are_not_json": torn_files,
        "lines_that_are_not_json": torn_lines,
        "messages_without_a_readable_timestamp": untimed,
        "busy_periods": n_periods,
        "busy_periods_of_one_event": n_single,
        "first_day": str(days_seen[0]) if days_seen else None,
        "last_day": str(days_seen[-1]) if days_seen else None,
    }
    res = {"parameters": {"idle_min": a.idle_min, "since": a.since,
                          "model": a.model},
           "totals": totals}
    segs = families_segments(families, a.idle_min * 60)
    st = stats(segs)
    res["concurrency"] = st
    if st:
        days = per_day(segs)
        hrs = [h for h, _ in days.values()]
        pk = [p for _, p in days.values()]
        res["days"] = {
            "with_use": len(days),
            "hours_median": pct(hrs, 0.5), "hours_p90": pct(hrs, 0.9),
            "peak_median": pct(pk, 0.5), "peak_p90": pct(pk, 0.9),
            "list": [{"date": str(d), "hours": round(h, 3), "peak": p}
                     for d, (h, p) in sorted(days.items())]}
    sens = []
    for m in SENSITIVITY:
        s = stats(families_segments(families, m * 60))
        sens.append({"idle_min": m, "mean": s and s["mean"],
                     "p90": s and s["p90"]})
    res["idle_sensitivity"] = sens
    return res


def render(res):
    t, c = res["totals"], res["concurrency"]
    L = [f"session files: {t['session_files_found']} found, "
         f"{t['session_files_read']} read",
         f"root sessions: {t['root_sessions']}   busy periods: "
         f"{t['busy_periods']} ({t['busy_periods_of_one_event']} of one "
         "event, in use for 0 s)",
         f"subagent files merged into their parent: {t['subagent_files_merged']}",
         f"parent link unresolved (kept as sessions of their own): "
         f"{t['parent_link_unresolved']}"]
    sk = t["files_skipped"]
    L.append("files skipped: " + (", ".join(f"{n} {why}" for why, n in
                                            sk.items()) if sk else "none"))
    L.append(f"lines that are not JSON: {t['lines_that_are_not_json']} in "
             f"{t['files_with_lines_that_are_not_json']} files; messages "
             f"without a readable timestamp: "
             f"{t['messages_without_a_readable_timestamp']}")
    L.append(f"days covered: {t['first_day']} to {t['last_day']}")
    if not c:
        L.append("\nno time with a session in use (no busy period longer "
                 "than 0 s)")
    else:
        L.append(f"\nover the {c['in_use_hours']:.1f} h with at least one "
                 f"session in use: mean {c['mean']:.2f}, p50 {c['p50']}, "
                 f"p90 {c['p90']}, max {c['max']}")
        L.append("share of that time at " + ", ".join(
            f"{k}: {c['share'][k]:.1%}" for k in SHARES))
        d = res["days"]
        L.append(f"per day with any use ({d['with_use']} days): hours in "
                 f"use median {d['hours_median']:.1f}, p90 "
                 f"{d['hours_p90']:.1f}; peak sessions median "
                 f"{d['peak_median']:g}, p90 {d['peak_p90']:g}")
    L.append("\nmean and p90 concurrency by --idle-min:")
    for s in res["idle_sensitivity"]:
        L.append(f"  {s['idle_min']:3d} min: " + (
            f"mean {s['mean']:.2f}, p90 {s['p90']}" if s["mean"] is not None
            else "no time in use"))
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sessions", type=Path,
                    default=Path.home() / ".pi/agent/sessions",
                    help="folder holding pi's session folders")
    ap.add_argument("--idle-min", type=float, default=30,
                    help="a gap longer than this many minutes ends a busy "
                         "period (default 30)")
    ap.add_argument("--since", help="only use from this day on, YYYY-MM-DD "
                                    "(local time)")
    ap.add_argument("--model", help="keep sessions with at least 90%% of "
                                    "their requests on a model id containing "
                                    "this (case-insensitive; default all)")
    ap.add_argument("-o", "--out", default="pi-concurrency.json")
    a = ap.parse_args(argv)
    if a.idle_min <= 0:
        ap.error("--idle-min must be positive")
    a.since_ts = None
    if a.since:
        try:
            a.since_ts = datetime.strptime(a.since, "%Y-%m-%d").timestamp()
        except ValueError:
            ap.error("--since is YYYY-MM-DD")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    res = analyse(a)
    print(render(res))
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\nwrote {a.out}")
    return 0 if res["concurrency"] else 1


if __name__ == "__main__":
    sys.exit(main())
