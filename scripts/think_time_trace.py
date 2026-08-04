"""Regenerate the study's think-time anchors from an agent-session trace.

    uv run python scripts/think_time_trace.py <inter_event_gaps.csv>

The CSV is a role-tagged inter-event-gap export of agent sessions (one row
per consecutive event pair) with at least these columns:

    session, transition, role_transition, gap_seconds

where `role_transition` is "<from> -> <to>" over roles {user, assistant,
toolResult, ...}. The trace itself is NOT committed — it carries session
names and timestamps — only the aggregate anchors below go into
scripts/scenario_model.py (the MEASURED_* block).

Why roles matter (and why a role-less export misleads): one request cycle
spans several gaps, so per-GAP means understate the per-REQUEST interval;
and without roles the heavy tool tail (builds) is indistinguishable from
human gaps while generation time pollutes the tool bucket. With roles the
split is exact:

    assistant -> toolResult   waiting on a tool          (think, Z)
    assistant -> user         waiting on the human       (think, Z)
    *         -> assistant    being served: generation   (service, R)

Anchors printed: requests/turn, mean tool and human waits, Z, R, and the
open-loop inter-request interval Z + R. R is whatever backend served the
trace — it does not transfer to an on-prem deployment, which is why the
model's closed-loop variants take Z alone and supply their own R.
"""
import csv
import math
import sys
from collections import defaultdict

PARK_S = 1800.0   # a gap this long is a parked session (cold on return), not
                  # think time: it leaves the active population, so pricing it
                  # into Z would understate every active user's request rate
ART_S = 0.05      # events flushed in the same write (parallel tool results):
                  # zero-width bookkeeping, not a wait anyone experienced

THINK_TRANSITIONS_TOOL = ("assistant -> toolResult", "assistant -> bashExecution")
THINK_TRANSITIONS_HUMAN = ("assistant -> user",)


def load(path):
    with open(path, newline="") as fh:
        return [r for r in csv.DictReader(fh)
                if r["transition"] == "message -> message"]


def analyze(rows):
    tool, human, serve = [], [], []
    per_session = defaultdict(lambda: defaultdict(list))
    n_req = n_turns = n_parked = 0
    for r in rows:
        g = float(r["gap_seconds"])
        tr = r["role_transition"]
        to_role = tr.split(" -> ")[1]
        if to_role == "assistant":
            n_req += 1                      # every assistant message = 1 request
            if tr.startswith("user"):
                n_turns += 1
        if g > PARK_S:
            n_parked += 1
            continue
        if tr in THINK_TRANSITIONS_TOOL:
            tool.append(g)
        elif tr in THINK_TRANSITIONS_HUMAN:
            human.append(g)
        elif to_role == "assistant":
            serve.append(g)
        per_session[r["session"]][tr].append(g)
    return tool, human, serve, n_req, n_turns, n_parked, per_session


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def median(v):
    s = sorted(v)
    return s[len(s) // 2] if s else float("nan")


def main(path):
    rows = load(path)
    tool, human, serve, n_req, n_turns, n_parked, per_session = analyze(rows)
    z = (sum(tool) + sum(human)) / n_req
    r = sum(serve) / n_req
    lt = [math.log(g) for g in tool if g > 0]
    lmu = sum(lt) / len(lt)
    lsd = math.sqrt(sum((v - lmu) ** 2 for v in lt) / len(lt))

    print(f"trace: {len(rows)} message gaps, {len(per_session)} sessions, "
          f"{n_parked} parked gap(s) > {PARK_S:.0f}s excluded")
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
    for s, d in sorted(per_session.items()):
        st = sum(sum(d.get(t, [])) for t in THINK_TRANSITIONS_TOOL)
        sh = sum(sum(d.get(t, [])) for t in THINK_TRANSITIONS_HUMAN)
        sv = [g for tr, gs in d.items() for g in gs
              if tr.endswith("-> assistant")]
        nr = len(sv) or 1
        print(f"{s:30s} {len(sv):4d} {(st + sh) / nr:6.1f}s "
              f"{mean(sv):6.1f}s {(st + sh) / nr + mean(sv):6.1f}s")
    print("\nthe per-session spread above is the strongest finding: a fleet is")
    print("a MIXTURE of interactive and near-autonomous sessions, not one Z —")
    print("carried as a stated limitation in docs/scenarios.md § 9.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
