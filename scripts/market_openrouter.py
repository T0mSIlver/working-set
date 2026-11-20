"""Snapshot OpenRouter's per-provider prices for the catalogue models.

Reads only two documented routes of OpenRouter's public API (no key):
GET /api/v1/models and GET /api/v1/models/{author}/{slug}/endpoints. It reads
no web page. Writes data/market/openrouter-<fetch date>.json, a snapshot for
local use: OpenRouter's terms (quoted in source.terms) restrict copying and
reusing its data, so the file stays out of the repo (data/market/README.md).

    uv run scripts/market_openrouter.py [--cache DIR] [--refresh] [--out DIR]

Raw responses are cached outside the repo and re-used unless --refresh.
A required field missing from a response is an error naming it; an optional
one is stored as null and counted in the printed summary.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import urllib.request
from collections import Counter
from decimal import Decimal
from pathlib import Path

API = "https://openrouter.ai/api/v1"
REPO = Path(__file__).resolve().parent.parent
TERMS_URL = "https://openrouter.ai/terms"

# Catalogue key -> (OpenRouter id, Hugging Face repo the catalogue was built
# from, as cited in research/). The API's hugging_face_id must equal the repo.
MODELS = {
    "27B": ("qwen/qwen3.8-27b", "Qwen/Qwen3.8-27B"),
    "35BA3B": ("qwen/qwen3.6-35b-a3b", "Qwen/Qwen3.6-35B-A3B"),
    "MM35": ("mistralai/mistral-medium-3-5", "mistralai/Mistral-Medium-3.5-128B"),
    "GLM52": ("z-ai/glm-5.3", "zai-org/GLM-5.3"),  # key is historical
    "DSV4F": ("deepseek/deepseek-v4-flash-0731", "deepseek-ai/DeepSeek-V4-Flash-0731"),
    "DSV4FV": ("deepseek/deepseek-v4-flash-vision-exp", "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp"),
    "DSV41F": ("deepseek/deepseek-v4.1-flash", "deepseek-ai/DeepSeek-V4.1-Flash"),
    "Q38FN": ("qwen/qwen3.8-flash", "Qwen/Qwen3.8-Flash-Next"),
    "GLM53F": ("z-ai/glm-5.3-flash", "zai-org/GLM-5.3-Flash"),
    "ML4": ("mistralai/mistral-large-4-0", "mistralai/Mistral-Large-4-1T-A52B"),
    "K31P": ("moonshotai/kimi-k3.1-pro", "moonshotai/Kimi-K3.1-Pro"),
}

# pricing keys stored in their own field; every other key goes to other_pricing
PRICE_FIELDS = {
    "prompt": "usd_per_mtok_input",
    "completion": "usd_per_mtok_output",
    "input_cache_read": "usd_per_mtok_cache_read",
    "input_cache_write": "usd_per_mtok_cache_write",
}

TERMS = (
    f"OpenRouter Terms of Service, {TERMS_URL}, Last Updated: August 31, 2026. "
    '§7 Prohibited Conduct: "BY USING THE SERVICE, YOU AGREE NOT TO: [...] develop, '
    "support or use software, devices, scripts, robots or any other means or processes "
    "(such as crawlers, browser plugins, add-ons or any other automated technology) to "
    "scrape or copy any information on the Site or the Services; bypass any technical "
    'measures implemented by OpenRouter that are designed to prevent scraping". '
    '§12 Ownership; Proprietary Rights: "The visual interfaces, graphics, design, '
    "compilation, information, data, computer code (including source code or object "
    'code), products, software, services, and all other elements of the Service ("Materials") '
    "provided by OpenRouter are protected by intellectual property and other laws. All "
    "Materials included in the Service are the property of OpenRouter or our third-party "
    "licensors. Except as expressly authorized by OpenRouter, you may not make use of the "
    'Materials." This snapshot is for local use; it is not published.'
)

METHOD = (
    "Prices: GET /api/v1/models/{author}/{slug}/endpoints, pricing.* in USD per "
    "token as decimal strings, multiplied by 1e6 with Python decimal. "
    "pricing.discount and any other pricing key are kept under other_pricing as "
    "published (OpenRouter does not document whether prompt/completion are "
    "before or after the discount). throughput_tok_s = throughput_last_30m and "
    "latency_s = latency_last_30m from the same endpoint object: OpenRouter's "
    "rolling 30-minute figures for requests routed through it; when the field "
    "is an object of percentiles, the p50 is stored and the object kept under "
    "raw. uptime = uptime_last_1d (percent of successful requests over the last "
    "day; 5m and 30m kept under raw). status = the endpoint's status code as "
    "published (0 = up)."
)


def fetch(url: str, cache: Path, refresh: bool) -> dict:
    name = url.removeprefix(API + "/").replace("/", "__") + ".json"
    path = cache / name
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    req = urllib.request.Request(url, headers={"User-Agent": "working-set-market/1", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        body = r.read()
    data = json.loads(body)
    cache.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return data


def require(obj: dict, key: str, where: str):
    if key not in obj:
        sys.exit(f"error: {where} has no field '{key}'")
    return obj[key]


def per_mtok(s) -> float | None:
    if s is None:
        return None
    return float(Decimal(str(s)) * 1_000_000)


def p50(v):
    """A rolling figure is either a number or an object of percentiles."""
    if v is None or isinstance(v, (int, float)):
        return v
    if isinstance(v, dict):
        if "p50" not in v:
            sys.exit(f"error: percentile object without p50: {v}")
        return v["p50"]
    sys.exit(f"error: unexpected throughput/latency value {v!r}")


def other_pricing(pricing: dict, where: str) -> dict:
    """Pricing keys beyond the four token prices, converted where their unit is known.

    discount: a fraction as published. overrides: time-of-day prices, a list of
    {utc_start, utc_end (HHMM), prompt, completion, ...} windows in USD per token.
    Any other key is kept raw: its unit is not documented.
    """
    other = {}
    for k, v in pricing.items():
        if k in PRICE_FIELDS:
            continue
        if k == "discount":
            other[k] = float(Decimal(str(v)))
        elif k == "overrides":
            if not isinstance(v, list):
                sys.exit(f"error: {where}.pricing.overrides is not a list: {v!r}")
            other[k] = [
                {kk: (per_mtok(vv) if kk in PRICE_FIELDS else vv) for kk, vv in w.items()}
                | {"unit": "usd_per_mtok"}
                for w in v
            ]
        else:
            other[k] = {"raw": v, "unit": "unknown"}
    return other


def _minutes(hhmm: int) -> int:
    h, m = divmod(int(hhmm), 100)
    if not (0 <= h <= 24 and 0 <= m < 60):
        sys.exit(f"error: override window time {hhmm!r} is not HHMM")
    return h * 60 + m


def time_weighted(rec: dict, where: str) -> dict | None:
    """Derived: the 24-hour time-weighted price when pricing.overrides has UTC windows.

    Each window [utc_start, utc_end) is HHMM UTC, wrapping past midnight when
    utc_end <= utc_start. Minutes no window covers bill at the listed base price.
    A price a window omits falls back to the base price. Overlapping windows are an error.
    """
    windows = rec["other_pricing"].get("overrides")
    if not windows:
        return None
    covered = [None] * 1440
    used = []
    for i, w in enumerate(windows):
        for k in ("utc_start", "utc_end"):
            if k not in w:
                sys.exit(f"error: {where}.pricing.overrides[{i}] has no field '{k}'")
        a, b = _minutes(w["utc_start"]), _minutes(w["utc_end"])
        span = (b - a) % 1440 or 1440
        for t in range(a, a + span):
            if covered[t % 1440] is not None:
                sys.exit(f"error: {where}.pricing.overrides windows overlap at minute {t % 1440}")
            covered[t % 1440] = i
        used.append({"utc_start": w["utc_start"], "utc_end": w["utc_end"], "hours": span / 60,
                     **{f: w.get(k, rec[f]) for k, f in PRICE_FIELDS.items() if k != "input_cache_write"}})
    base_minutes = covered.count(None)
    out = {"what": "24-hour time-weighted price over the UTC windows below; the listed base price "
                   "covers any minute no window covers", "windows": used,
           "base_hours": base_minutes / 60}
    for k, f in PRICE_FIELDS.items():
        if k == "input_cache_write":
            continue
        vals = [(used[i][f] if i is not None else rec[f]) for i in covered]
        out[f] = None if any(v is None for v in vals) else round(sum(vals) / 1440, 9)
    return out


def endpoint_record(e: dict, where: str, nulls: Counter) -> dict:
    pricing = require(e, "pricing", where)
    for k in ("prompt", "completion"):
        require(pricing, k, f"{where}.pricing")
    rec = {
        "provider": require(e, "provider_name", where),
        "tag": require(e, "tag", where),
    }
    for k, field in PRICE_FIELDS.items():
        rec[field] = per_mtok(pricing.get(k))
    rec["other_pricing"] = other_pricing(pricing, where)
    rec["time_weighted"] = time_weighted(rec, where)
    rec["context_length"] = require(e, "context_length", where)
    rec["max_completion_tokens"] = e.get("max_completion_tokens")
    rec["quantization"] = e.get("quantization")
    rec["throughput_tok_s"] = p50(e.get("throughput_last_30m"))
    rec["latency_s"] = p50(e.get("latency_last_30m"))
    rec["uptime"] = e.get("uptime_last_1d")
    rec["status"] = e.get("status")
    rec["supported_parameters"] = e.get("supported_parameters") or []
    rec["raw"] = {
        "name": e.get("name"),
        "pricing": pricing,
        "service_tier": e.get("service_tier"),
        "max_prompt_tokens": e.get("max_prompt_tokens"),
        "supports_implicit_caching": e.get("supports_implicit_caching"),
        "throughput_last_30m": e.get("throughput_last_30m"),
        "latency_last_30m": e.get("latency_last_30m"),
        "uptime_last_5m": e.get("uptime_last_5m"),
        "uptime_last_30m": e.get("uptime_last_30m"),
        "uptime_last_1d": e.get("uptime_last_1d"),
    }
    for f in ("usd_per_mtok_cache_read", "usd_per_mtok_cache_write", "max_completion_tokens",
              "quantization", "throughput_tok_s", "latency_s", "uptime", "status"):
        if rec[f] is None:
            nulls[f] += 1
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "workingset" / "market" / "openrouter")
    ap.add_argument("--refresh", action="store_true", help="re-fetch instead of using the cache")
    ap.add_argument("--out", type=Path, default=REPO / "data" / "market")
    args = ap.parse_args()

    list_path = args.cache / "models.json"
    if args.refresh or not list_path.exists():
        fetched = dt.datetime.now(dt.timezone.utc)
    else:  # the snapshot is dated by when the cached list was fetched
        fetched = dt.datetime.fromtimestamp(list_path.stat().st_mtime, dt.timezone.utc)
    listing = fetch(f"{API}/models", args.cache, args.refresh)
    by_id = {m["id"]: m for m in require(listing, "data", "/models")}

    nulls: Counter = Counter()
    models, problems = [], []
    for key, (orid, hf) in MODELS.items():
        m = by_id.get(orid)
        if m is None:
            sys.exit(f"error: {orid} ({key}) is not in OpenRouter's model list")
        api_hf = m.get("hugging_face_id")
        if api_hf != hf:
            problems.append(f"{key}: API hugging_face_id {api_hf!r} != catalogue repo {hf!r}")
        detail = fetch(f"{API}/models/{orid}/endpoints", args.cache, args.refresh)
        eps = require(require(detail, "data", orid), "endpoints", orid)
        if not eps:
            sys.exit(f"error: {orid} lists no endpoints")
        endpoints = [endpoint_record(e, f"{orid} endpoint {i}", nulls) for i, e in enumerate(eps)]
        # an endpoint with a null price keeps its null and sorts last
        endpoints.sort(key=lambda r: tuple(
            (v is None, v if v is not None else 0.0)
            for v in (r["usd_per_mtok_output"], r["usd_per_mtok_input"])))
        variants = []
        for vid, vm in sorted(by_id.items()):
            if vid.split(":")[0] == orid and vid != orid:
                p = require(vm, "pricing", vid)
                variants.append({
                    "openrouter_id": vid,
                    **{f: per_mtok(p.get(k)) for k, f in PRICE_FIELDS.items()},
                    "other_pricing": other_pricing(p, vid),
                    "raw": {"pricing": p},
                })
        models.append({
            "ws_model": key,
            "openrouter_id": orid,
            "hugging_face_id": api_hf,
            "url": f"https://openrouter.ai/{orid}",
            "context_length": require(m, "context_length", orid),
            "endpoints": endpoints,
            "variants": variants,
        })

    if problems:
        sys.exit("error: catalogue mapping disagrees with the API:\n  " + "\n  ".join(problems))

    out = {
        "source": {
            "id": "openrouter",
            "url": f"{API}/models",
            "fetched": fetched.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "script": "scripts/market_openrouter.py",
            "terms": TERMS,
            "method": METHOD,
        },
        "models": models,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"openrouter-{fetched.date().isoformat()}.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")

    n = sum(len(m["endpoints"]) for m in models)
    print(f"wrote {path.relative_to(REPO) if path.is_relative_to(REPO) else path}: {len(models)} models, {n} endpoints")
    for m in models:
        provs = {e["provider"] for e in m["endpoints"]}
        print(f"  {m['ws_model']:7} {m['openrouter_id']:40} {len(m['endpoints']):3} endpoints, "
              f"{len(provs):2} providers, {len(m['variants'])} variants")
    print("null optional fields (of %d endpoints):" % n)
    for f, c in sorted(nulls.items()):
        print(f"  {f:26} {c}")
    print("this snapshot is for local use: do not commit it without the owner's decision "
          "(OpenRouter's terms, source.terms)")


if __name__ == "__main__":
    main()
