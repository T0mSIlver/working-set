#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///
"""
Embed scripts/validate_deployment.py into the explorer.

The explorer's "Test these hypotheses" button hands out the validation
harness with its CONFIG block rewritten from the live controls. The page is
a single dependency-free file that must also work from file://, so the
harness travels INSIDE it — as a JSON string literal (no template-literal
escaping hazards: the Python source contains both backticks and `${`).

This script is the only writer of that string. Run it after every edit to
validate_deployment.py:

    uv run scripts/sync_harness.py           # re-embed
    uv run scripts/sync_harness.py --check   # verify freshness (exit 1 if stale)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HARNESS = ROOT / "scripts" / "validate_deployment.py"
PAGE = ROOT / "interactive" / "index.html"
BEGIN, END = "/* @@HARNESS_EMBED_BEGIN@@ */", "/* @@HARNESS_EMBED_END@@ */"


def embedded_literal() -> str:
    src = HARNESS.read_text(encoding="utf-8")
    # ensure_ascii keeps the page byte-safe in any charset handling; the </
    # escape keeps a literal "</script>" (present or future) from ever
    # terminating the surrounding <script> element
    lit = json.dumps(src, ensure_ascii=True).replace("</", "<\\/")
    return f"{BEGIN}\nconst HARNESS_TEMPLATE = {lit};\n{END}"


def main() -> int:
    check = "--check" in sys.argv[1:]
    page = PAGE.read_text(encoding="utf-8")
    pat = re.compile(re.escape(BEGIN) + r"[\s\S]*?" + re.escape(END))
    if not pat.search(page):
        print(f"sync_harness: sentinel block not found in {PAGE}", file=sys.stderr)
        return 1
    fresh = pat.sub(lambda _: embedded_literal(), page, count=1)
    if fresh == page:
        print("sync_harness: embed is current")
        return 0
    if check:
        print("sync_harness: STALE — run `uv run scripts/sync_harness.py` "
              "after editing validate_deployment.py", file=sys.stderr)
        return 1
    PAGE.write_text(fresh, encoding="utf-8")
    print(f"sync_harness: embedded {HARNESS.stat().st_size:,} bytes into {PAGE.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
