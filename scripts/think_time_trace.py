"""Regenerate the study's think-time anchors from an agent-session trace.

    uv run python scripts/think_time_trace.py <inter_event_gaps.csv>

The CSV is a role-tagged inter-event-gap export of agent sessions (one row
per consecutive event pair) with at least these columns:

    session, transition, role_transition, gap_seconds, from_timestamp

where `role_transition` is "<from> -> <to>" over roles {user, assistant,
toolResult, ...}. Rows may arrive in any order; cycles are rebuilt
chronologically per session. The trace itself is NOT committed — it carries
session names and timestamps — only the aggregate anchors below go into
scripts/scenario_model.py (the MEASURED_* block).

Why roles matter (and why a role-less export misleads): one request cycle
spans several gaps, so per-GAP means understate the per-REQUEST interval;
and without roles the heavy tool tail (builds) is indistinguishable from
human gaps while generation time pollutes the tool bucket. With roles the
split is exact:

    assistant -> toolResult / bashExecution   waiting on a tool    (think, Z)
    toolResult -> toolResult (>= ART_S)       chained tool waits   (think, Z)
    assistant -> user                         waiting on the human (think, Z)
    *         -> assistant                    being served         (service, R)

Sub-ART_S gaps are events flushed in the same write (parallel tool results,
one request emitting two messages): zero-width bookkeeping, not a wait
anyone experienced — they merge into the surrounding cycle instead of
counting as requests or waits. Gaps above PARK_S leave the active
population (the user parked the session; cold on return), so both the gap
and, for a service gap, the request it would have counted are excluded
rather than priced into every active user's cadence.

Anchors printed: requests/turn, mean tool and human waits, Z, R, and the
open-loop inter-request interval Z + R. R is whatever backend served the
trace — it does not transfer to an on-prem deployment, which is why the
model's closed-loop variants take Z alone and supply their own R.
"""
import csv
import math
import sys
from collections import defaultdict

PARK_S = 1800.0   # parked session boundary (see module docstring)
ART_S = 0.05      # same-write flush threshold (see module docstring)

TOOL_ROLES = ("toolResult", "bashExecution")


def load(path):
    rows = [r for r in csv.DictReader(open(path, newline=""))
            if r["transition"] == "message -> message"]
    rows.sort(key=lambda r: (r["session"], r["from_timestamp"]))
    return rows


def classify(rows):
    """One pass per (chronologically ordered) session; returns per-session
    lists of tool waits, human waits, and service gaps, plus request/turn
    counts. Artifact gaps (< ART_S) merge into the cycle they interrupt;
    parked gaps close it."""
    S = defaultdict(lambda: {"tool": [], "human": [], "serve": [],
                             "n_req": 0, "n_turns": 0, "n_parked": 0,
                             "n_artifact": 0})
    for r in rows:
        d = S[r["session"]]
        g = float(r["gap_seconds"])
        frm, to = r["role_transition"].split(" -> ")
        if g > PARK_S:
            # excluded BEFORE any counting: a request preceded by a parked
            # gap would otherwise enter the denominator with its service
            # time discarded, biasing R low
            d["n_parked"] += 1
            continue
        if g < ART_S and (frm == to or to in TOOL_ROLES and frm in TOOL_ROLES):
            # same-write flush: assistant -> assistant is one request
            # emitting two messages, toolResult -> toolResult a parallel
            # batch — neither is a new request nor a wait
            d["n_artifact"] += 1
            continue
        if to == "assistant":
            d["n_req"] += 1
            d["serve"].append(g)
            if frm == "user":
                d["n_turns"] += 1
        elif frm == "assistant" and to == "user":
            d["human"].append(g)
        elif frm == "assistant" and to in TOOL_ROLES:
            d["tool"].append(g)
        elif frm in TOOL_ROLES and to in TOOL_ROLES:
            # >= ART_S survivor of the artifact filter: sequentially chained
            # tool executions — waiting, same as the assistant -> tool leg
            d["tool"].append(g)
        # user -> user etc.: no wait class; ignore
    return S


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def median(v):
    s = sorted(v)
    return s[len(s) // 2] if s else float("nan")


def main(path):
    S = classify(load(path))
    tool = [g for d in S.values() for g in d["tool"]]
    human = [g for d in S.values() for g in d["human"]]
    serve = [g for d in S.values() for g in d["serve"]]
    n_req = sum(d["n_req"] for d in S.values())
    n_turns = sum(d["n_turns"] for d in S.values())
    assert len(serve) == n_req, "every counted request has its service gap"
    z = (sum(tool) + sum(human)) / n_req
    r = sum(serve) / n_req
    lt = [math.log(g) for g in tool if g > 0]
    lmu = sum(lt) / len(lt)
    lsd = math.sqrt(sum((v - lmu) ** 2 for v in lt) / len(lt))

    print(f"trace: {len(S)} sessions; excluded "
          f"{sum(d['n_parked'] for d in S.values())} parked gap(s) > "
          f"{PARK_S:.0f}s and {sum(d['n_artifact'] for d in S.values())} "
          f"same-write artifact(s) < {ART_S}s")
    print(f"\nMEASURED_REQ_PER_TURN = {n_req / n_turns:.1f}   "
          f"({n_req} requests / {n_turns} human turns)")
    print(f"MEASURED_T_TOOL_S     = {mean(tool):.1f}   "
          f"(n={len(tool)}, median {median(tool):.2f}s, "
          f"lognormal sigma {lsd:.2f})")
    print(f"MEASURED_T_HUMAN_S    = {mean(human):.1f}  "
          f"(n={len(human)}, median {median(human):.1f}s — tail-dominated)")
    print(f"MEASURED_THINK_Z_S    = {z:.1f}   "
          f"(tool share {sum(tool) / (sum(tool) + sum(human)):.0%})")
    print(f"MEASURED_SERVICE_R_S  = {r:.1f}   (backend-specific: do not port)")
    print(f"MEASURED_CYCLE_S      = {z + r:.1f}")

    print(f"\n{'session':30s} {'req':>4} {'Z/req':>7} {'R/req':>7} {'cycle':>7}")
    for s, d in sorted(S.items()):
        nr = d["n_req"] or 1
        zz = (sum(d["tool"]) + sum(d["human"])) / nr
        print(f"{s:30s} {d['n_req']:4d} {zz:6.1f}s "
              f"{mean(d['serve']):6.1f}s {zz + mean(d['serve']):6.1f}s")
    print("\nthe per-session spread above is the strongest finding: a fleet is")
    print("a MIXTURE of interactive and near-autonomous sessions, not one Z —")
    print("carried as a stated limitation in docs/scenarios.md § 9.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
