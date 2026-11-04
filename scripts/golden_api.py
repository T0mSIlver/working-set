#!/usr/bin/env python3
"""Golden vectors for the API and plan seats: Python prices, the mirror agrees.

workingset.api_cost is the source of truth; interactive/src/apicost.js is
its hand-written mirror. This writes every API offer and plan in the price
catalogue, priced at the reference workload and at one knob moved at a time,
to tests/golden/api_vectors.json; tests/js/api_golden.test.mjs replays them.
Closed form on both sides (the same erfc polynomial), so the tolerance is
float rounding, not sampling.

    uv run scripts/golden_api.py            # rewrite the vectors
    uv run scripts/golden_api.py --check    # exit 1 if they are stale
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, replace
from pathlib import Path

from workingset import api_cost as A
from workingset import model as M

OUT = Path(__file__).resolve().parents[1] / "tests" / "golden" / "api_vectors.json"
BASE_OPTS = {"think": M.THINK_TIME_S, "turn": 2000.0, "out": float(M.OUT_TOKENS_DEFAULT),
             "hours": A.ACTIVE_HOURS_DEFAULT, "active": 1.0, "spu": 1.0}
# one knob at a time, each inside its explorer control's range
WL_SWEEP = [("user_median", 8_000), ("user_median", 120_000), ("user_sigma", 1.4),
            ("sub_ratio", 0.0), ("sub_ratio", 0.5), ("sys_user", 40_000), ("invalidation", 0.10),
            ("cap", 262_144), ("cap", 1_000_000), ("sub_shares_prefix", True)]
OPT_SWEEP = [("think", 10.0), ("think", 120.0), ("turn", 500.0), ("turn", 8000.0),
             ("out", 2000.0), ("hours", 8.0), ("hours", 160.0), ("hours", 400.0),
             ("active", 0.25), ("spu", 2.0)]


def sig(x):
    return float(f"{x:.12g}") if isinstance(x, float) and math.isfinite(x) else x


def states():
    wl0 = M.Workload()
    yield "reference", wl0, BASE_OPTS
    for k, v in WL_SWEEP:
        yield f"{k}={v}", replace(wl0, **{k: v}), BASE_OPTS
    for k, v in OPT_SWEEP:
        yield f"{k}={v}", wl0, {**BASE_OPTS, k: v}


def price(wl, o):
    kw = dict(think_time_s=o["think"], turn_tokens=o["turn"], out_tokens=o["out"],
              active_hours=o["hours"], peak_active_share=o["active"],
              sessions_per_active_user=o["spu"])
    cat = A.catalogue()
    rows = [A.api_seat_cost(a, wl, **kw) for a in cat["apis"].values()]
    rows += [A.plan_seat_cost(p, wl, **kw) for p in cat["plans"].values()]
    return {r.key + ":" + r.kind: {k: sig(v) for k, v in asdict(r).items()
                                   if k not in ("key", "name", "kind")} for r in rows}


def render() -> str:
    vec = []
    for name, wl, o in states():
        w = {k: sig(float(v)) if isinstance(v, (int, float)) and not isinstance(v, bool) else v
             for k, v in asdict(wl).items()}
        vec.append({"name": name, "workload": w, "opts": o, "out": price(wl, o)})
    return json.dumps({"read": A.catalogue()["read"], "vectors": vec},
                      indent=1, sort_keys=True, allow_nan=False) + "\n"


def main(argv):
    text = render()
    if "--check" in argv:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != text:
            print(f"{OUT} is stale: run `uv run scripts/golden_api.py`", file=sys.stderr)
            return 1
        return 0
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT}: {text.count('\"name\"')} states")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
