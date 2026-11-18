"""Write every catalogue model's `ws market --json` result into one file.

    uv run scripts/market_compare.py                       # data/market -> data/market/comparison-DATE.json
    uv run scripts/market_compare.py --snapshot DIR --out FILE

DATE is the rental snapshot's. Both terms are in each model's result; the
text report's term is on-demand.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from workingset import market as MK
from workingset import model as M


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-iter", type=int, default=400)
    ap.add_argument("--models", nargs="*", default=None, help="default: every catalogue model")
    a = ap.parse_args(argv)
    snap = MK.load_snapshot(a.snapshot)
    date = re.search(r"(\d{4}-\d{2}-\d{2})", snap["rental_file"])
    if date is None:
        raise SystemExit(f"{snap['rental_file']}: no date in the file name")
    out = Path(a.out) if a.out else Path(snap["dir"]) / f"comparison-{date.group(1)}.json"
    wv = MK.default_workload()
    results = {}
    for key in a.models or list(M.MODELS):
        t0 = time.time()
        ev = MK.Evaluator(wv, a.n_iter)
        results[key] = MK.compare(key, snap, evaluator=ev)
        print(f"{key}: {len(results[key]['configs'])} configurations, "
              f"{len(results[key]['providers'])} endpoints, {time.time() - t0:.0f} s",
              file=sys.stderr)
    doc = {"schema": MK.SCHEMA + "+all", "date": date.group(1), "models": results}
    out.write_text(json.dumps(doc, indent=1, allow_nan=False, default=MK._json_default) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
