"""Run N agentperf-local replays at once against one server (ws aa's lanes).

    uv run scripts/aa_concurrency.py --aa ~/aa-agentperf-local \
        --base-url http://127.0.0.1:8080/v1 --model served --n 4 --out runs/n4

Lane k replays the default tasks rotated to start k/N of the way through the
list (`aa_local.rotated`), each lane its own `agentperf-local run` process
with its own cache namespace. The rotated manifests are written next to the
bundled ones in a copy of the replay directory, so trace paths still resolve.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from workingset.aa_local import rotated  # noqa: E402

REPLAY = "agentperf_local/data/replays/agentperf-default-v1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aa", required=True, type=Path, help="aa-agentperf-local checkout")
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--manifest", default="manifest-x86_64.json")
    ap.add_argument("--no-rotate", action="store_true")
    ap.add_argument("extra", nargs="*", help="passed to agentperf-local run after --")
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=False)
    rep = a.out / "replay"
    shutil.copytree(a.aa / REPLAY, rep, ignore=shutil.ignore_patterns("workspaces"))
    man = json.loads((rep / a.manifest).read_text())
    order = [t["task_id"] for t in man["tasks"]]
    by_id = {t["task_id"]: t for t in man["tasks"]}
    procs = []
    t0 = time.time()
    for k in range(a.n):
        ids = order if a.no_rotate else rotated(order, k, a.n)
        mk = rep / f"manifest-lane{k}.json"
        mk.write_text(json.dumps({**man, "tasks": [by_id[i] for i in ids]}, indent=2))
        cmd = ["uv", "run", "--project", str(a.aa), "agentperf-local", "run",
               "--base-url", a.base_url, "--model", a.model,
               "--manifest", str(mk), "--output-dir", str(a.out / f"lane{k}"), *a.extra]
        log = open(a.out / f"lane{k}.log", "w")
        procs.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT))
    codes = [p.wait() for p in procs]
    (a.out / "driver.json").write_text(json.dumps(
        {"n": a.n, "rotate": not a.no_rotate, "started_unix_s": t0,
         "wall_s": time.time() - t0, "exit_codes": codes}, indent=2))
    print(json.dumps({"n": a.n, "wall_s": round(time.time() - t0, 1), "exit_codes": codes}))
    return 0 if all(c == 0 for c in codes) else 1


if __name__ == "__main__":
    sys.exit(main())
