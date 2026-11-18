"""Re-read the GPU rental price pages and report which recorded prices moved.

The prices in data/market/gpu_rental-<date>.json were read by hand, each with
the cell quoted. This script re-fetches every page that serves its prices in
plain HTML or JSON, checks that each record's listed price still appears there
next to its GPU, and prints one line per record:

    same     the listed price is still on the page next to the GPU name
    MOVED    the page loads but the listed price is no longer next to the GPU
    browser  the page needs a browser (prices drawn by JavaScript, or a bot check)
    error    the fetch failed (the HTTP error is printed)

It never edits the JSON: a MOVED line means re-read that page and record it
again with its new date. Vast.ai is a live marketplace, so its line prints the
current median beside the recorded one instead.

    uv run scripts/market_gpu_rental.py [data/market/gpu_rental-<date>.json]

Pages are cached under ~/.cache/workingset/market/pages-check/ for the day.
"""

from __future__ import annotations

import datetime as dt
import html
import json
import re
import statistics
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CACHE = Path.home() / ".cache" / "workingset" / "market" / "pages-check"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"

# Pages whose prices only a browser shows; listed so a run says what it skipped.
NEEDS_BROWSER = {
    "https://cloud.google.com/compute/gpus-pricing": "Google Cloud: prices drawn by JavaScript",
    "https://www.scaleway.com/en/pricing/gpu/": "Scaleway: bot check (HTTP 403)",
    "https://nebius.com/spot-prices": "Nebius live spot prices: drawn by JavaScript",
    "https://aws.amazon.com/ec2/pricing/on-demand/": "AWS on-demand: drawn by JavaScript (Vantage mirrors it)",
}
WINDOW = 1500  # characters between the GPU name and its price on a flattened page


def fetch(url: str) -> str:
    CACHE.mkdir(parents=True, exist_ok=True)
    key = re.sub(r"[^A-Za-z0-9]+", "_", url)[:150] + "-" + dt.date.today().isoformat()
    path = CACHE / key
    if path.exists():
        return path.read_text(errors="ignore")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        body = r.read(8_000_000).decode("utf-8", "ignore")
    path.write_text(body)
    return body


def flatten(page: str) -> str:
    """Tags out, entities and JSON escapes decoded, whitespace collapsed."""
    t = page.replace("\\u003C", "<").replace("\\u003E", ">").replace("\\u002F", "/").replace('\\"', '"')
    t = html.unescape(re.sub(r"<[^>]+>", " ", t))
    return re.sub(r"\s+", " ", t)


def price_tokens(rec: dict) -> list[str]:
    """Ways the listed price can be written: $5.40, €42, "price":42, "value": 10."""
    p = rec["price_listed"]
    forms = {f"{p:.2f}", f"{p:.3f}", f"{p:,.2f}", f"{p:g}", f"{p:,.1f}" if p >= 1000 else f"{p:g}"}
    forms |= {f.rstrip("0").rstrip(".") for f in list(forms) if "." in f}
    out = []
    for f in forms:
        out += [f"${f}", f"$ {f}", f"€{f}", f'"price":{f}', f'"value": {f}', f'"value":{f}',
                f'"retailPrice":{f}', f"{f}/h"]
    return out


def gpu_names(rec: dict) -> list[str]:
    inst = rec["instance"]
    names = [w for w in re.findall(r"[A-Za-z0-9_.-]+", inst) if len(w) >= 4 and re.search(r"\d", w)]
    return names + [rec["gpu"]]


def check(rec: dict, page: str) -> bool:
    text = flatten(page)
    toks = price_tokens(rec)
    for name in gpu_names(rec):
        for m in re.finditer(re.escape(name), text):
            window = text[max(0, m.start() - WINDOW): m.end() + WINDOW]
            if any(t in window for t in toks):
                return True
    return False


def vast_median(page: str, gpu: str) -> tuple[float, int]:
    offers = json.loads(page).get("offers", [])
    per = [o["dph_total"] / o["num_gpus"] for o in offers if o["gpu_name"] == gpu]
    if not per:
        raise SystemExit(f"Vast.ai returned no rentable {gpu} offers; re-read the query in the record's url")
    return statistics.median(per), len(per)


def main() -> None:
    files = sorted((REPO / "data/market").glob("gpu_rental-*.json"))
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else (files[-1] if files else None)
    if path is None:
        raise SystemExit("no data/market/gpu_rental-*.json; pass the file to check")
    doc = json.loads(path.read_text())
    print(f"{path.name}: {len(doc['records'])} records read {doc['source']['read']}; checking today {dt.date.today()}\n")
    counts: dict[str, int] = {}
    for rec in doc["records"]:
        url, label = rec["url"], f"{rec['gpu']:5} {rec['seller'][:24]:24} {rec['term']:9} {rec['instance'][:36]:36}"
        if rec["price_listed"] is None:
            status, extra = "skip", "no listed price (contact sales)"
        elif url in NEEDS_BROWSER:
            status, extra = "browser", NEEDS_BROWSER[url]
        else:
            try:
                page = fetch(url)
            except Exception as e:  # report and go on: one dead page must not hide the others
                status, extra = "error", f"{type(e).__name__}: {e}"
            else:
                if rec["seller"] == "Vast.ai":
                    med, n = vast_median(page, rec["gpu"])
                    status = "same" if abs(med - rec["price_listed"]) < 0.005 else "MOVED"
                    extra = f"median now {med:.3f} over {n} offers (recorded {rec['price_listed']})"
                else:
                    status = "same" if check(rec, page) else "MOVED"
                    extra = "" if status == "same" else f"listed {rec['price_listed']} {rec['currency_listed']} not found next to {gpu_names(rec)}"
        counts[status] = counts.get(status, 0) + 1
        print(f"{status:7} {label} {extra}")
    print("\n" + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    print("needs a browser, not checked here:")
    for url, why in NEEDS_BROWSER.items():
        print(f"  {why}: {url}")


if __name__ == "__main__":
    main()
