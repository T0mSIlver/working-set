"""Build the market page's data from a comparison JSON.

    uv run scripts/market_page.py [COMPARISON] [--sources provider_prices|all] [--artifact OUT.html]

Writes interactive/src/market-data.js (`export const MARKET = {...}`), the compact
subset interactive/market.html needs. With --artifact it also writes one
self-contained HTML file (data and script inlined) for publishing as an Artifact;
that file goes outside the repo.

The committed market-data.js carries only rows whose source is provider_prices
(the providers' own pages). `--sources all` adds the OpenRouter listing, whose
terms restrict reuse: use it for the private Artifact only.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_JS = ROOT / "interactive/src/market-data.js"
PAGE = ROOT / "interactive/market.html"
SCRIPT = ROOT / "interactive/src/market.js"
KINDS = ("code", "english", "french", "json_tool_calls")
LEVELS = ("low", "median", "high")


def sig(x, n=5):
    if x is None:
        return None
    if x == 0:
        return 0
    return round(x, n - 1 - int(math.floor(math.log10(abs(x)))))


def need(d: dict, key: str, where: str):
    if key not in d:
        raise KeyError(f"{where}: field '{key}' missing from the comparison JSON")
    return d[key]


def rent_block(rent: dict, TERMS) -> dict:
    out = {}
    for gpu, terms in rent.items():
        out[gpu] = {}
        for term in TERMS:
            r = need(terms, term, f"rent.{gpu}")
            if r["low"] is None:
                raise ValueError(f"rent.{gpu}.{term}: no priced seller; the band needs one")
            out[gpu][term] = {
                "low": r["low"], "median": sig(r["median"]), "high": r["high"],
                "n": r["n_sellers"], "agg": r.get("n_aggregators_excluded", 0),
                "tiers": r.get("tiers"), "out_tiers": r.get("other_tiers_excluded", []),
                "gb300": r.get("gb300_excluded", []), "sold_out": r.get("sold_out", []),
                "sellers": [[s["seller"], s["usd_per_gpu_h"], s.get("commit_months"), s["read"], s["url"],
                             s.get("tier"), s.get("availability")]
                            for s in r["sellers"]],
            }
    return out




def build(comp: dict, sources: str, snapshot: Path) -> dict:
    models = need(comp, "models", "top level")
    first = next(iter(models.values()))
    rent0 = json.dumps(first["rent"], sort_keys=True)
    TERMS = list(next(iter(first["rent"].values())).keys())
    wl0 = first["workload"]
    pp_rows = json.loads((snapshot / "provider_prices.json").read_text())["rows"]
    pp_index = {(r["ws_model"], r["provider"], r["tier"]): r for r in pp_rows}
    price_sources = {s["id"]: s for s in first["snapshot"]["price_sources"]}

    out_models = []
    pp_pages = {}
    SENS_WORDS = {"cache_as_input": "the price of cached input", "mfu": "prefill speed", "decode": "decode speed",
                  "limit": "the predicted session limit", "rent": "GPU rent",
                  "ttl": "how long the provider keeps a prompt cached"}
    for key, m in models.items():
        if json.dumps(m["rent"], sort_keys=True) != rent0:
            raise ValueError(f"{key}: rent differs from the first model's; the page assumes one rent table")
        if m["workload"]["mixed_tokens_per_session_hour"] != wl0["mixed_tokens_per_session_hour"]:
            raise ValueError(f"{key}: workload differs from the first model's; the page assumes one workload")
        cpt = (m.get("chars_per_token") or {}).get("chars_per_token")
        if cpt is None:
            raise KeyError(f"{key}: chars_per_token missing; run scripts/market_tokenizers.py")
        configs = []
        for c in m["configs"]:
            if not c["viable"]:
                continue
            e = c["error"]
            configs.append({"id": c["id"], "gpu": c["gpu"], "tp": c["tp"], "arm": c["arm"], "n": c["n_gpu"],
                            "limit": c["limit"], "bind": c["binding"],
                            "err": {"s": e["status"], "bs": e["binding_status"], "bq": e["binding_quantity"],
                                    "lo": e.get("limit_low"), "hi": e.get("limit_high")}})
        if not configs:
            print(f"warning: {key} has no viable configuration; its row has no band", file=sys.stderr)
        provs = []
        for p in m["providers"]:
            if sources == "provider_prices" and p["source"] != "provider_prices":
                continue
            if p["source"] == "provider_prices":
                row = pp_index.get((key, p["provider"], p["tag"]))
                if row is None:
                    raise KeyError(f"{key}: {p['provider']} / {p['tag']} is in the comparison but not in "
                                   f"{snapshot / 'provider_prices.json'}: the comparison is older than the price "
                                   "file; rerun scripts/market_compare.py")
                url, read = row["url"], row["read"]
                pp_pages[(p["provider"], url)] = read
            else:
                src = price_sources[p["source"]]
                url, read = src["url"], str(src["fetched"])[:10]
            us = {}
            for term in TERMS:
                t = p["u_star"].get(term) if p.get("u_star") else None
                us[term] = [sig(t[l]["u_star"], 4) if t and t.get(l) else None for l in LEVELS] if t else None
            f = p["fidelity"]
            provs.append({
                "src": p["source"], "p": p["provider"], "tag": p["tag"],
                "in": p["usd_per_mtok_input"], "out": p["usd_per_mtok_output"],
                "cr": p["usd_per_mtok_cache_read"], "cw": p["usd_per_mtok_cache_write"],
                "sh": sig(p["usd_per_session_hour"]), "mix": sig(p["usd_per_mtok_mixed"]),
                "q": f.get("quantization_declared"), "arm": p["arm_note"],
                "ctx": f.get("context_offered"), "ctxr": sig(f.get("context_ratio"), 3),
                "mo": f.get("max_output"), "fits": p["fits_context"],
                "over": sig(p.get("share_requests_over_context"), 3),
                "flags": p["flags"], "u": us, "url": url, "read": read,
            })
            lw = p.get("listed_window_prices")
            if lw:
                provs[-1]["lw"] = {"in": lw.get("usd_per_mtok_input"), "out": lw.get("usd_per_mtok_output"),
                                   "cr": lw.get("usd_per_mtok_cache_read"),
                                   "u": {t: [sig(v.get(l), 4) for l in LEVELS] for t, v in (p.get("u_star_listed_window") or {}).items()}}
        # every prompt fresh: the comparison's rough contrast, used only if it covers every row
        cn = m.get("contrast_no_cache")
        nc = None
        if cn:
            if cn["workload"]["mixed_tokens_per_session_hour"] != wl0["mixed_tokens_per_session_hour"]:
                raise ValueError(f"{key}: contrast_no_cache uses another token count per session-hour")
            ncp = {(q["provider"], q["tag"], q["source"]): q for q in cn["providers"]}
            for pr in provs:
                q = ncp.get((pr["p"], pr["tag"], pr["src"]))
                if q is None:
                    raise KeyError(f"{key}: {pr['p']} / {pr['tag']} missing from contrast_no_cache.providers")
                pr["nc"] = {"mix": sig(q["usd_per_mtok_mixed"]), "sh": sig(q["usd_per_session_hour"]),
                            "u": {t: [sig(v[l]["u_star"], 4) if v.get(l) else None for l in LEVELS]
                                  for t, v in q["u_star"].items() if isinstance(v, dict) and "low" in v}}
            nc = {"configs": [{"id": c["id"], "gpu": c["gpu"], "tp": c["tp"], "arm": c["arm"], "n": c["n_gpu"],
                               "limit": c["limit"], "bind": c["binding"]} for c in cn["configs"] if c["limit"] > 0],
                  "mix": {k: sig(v, 4) for k, v in cn["workload"]["mix_billed"].items()},
                  "note": cn.get("workload_note")}
        sens = sorted(m["sensitivity"]["assumptions"], key=lambda a: -(a["median_abs_dln"] or 0))
        top = sens[0] if sens else None
        out_models.append({"key": key, "name": m["model_name"], "cpt": cpt, "configs": configs, "prov": provs,
                           "note": m.get("model_note"), "nc": nc,
                           "sens": ({"words": SENS_WORDS.get(top["kind"], re.sub(r"\s*\([^)]*\)$", "", top["label"])),
                                     "label": top["label"]} if top else None),
                           "max_ctx": next((p["fidelity"]["model_max_context"] for p in m["providers"]), None)})

    mb = need(wl0, "mix_billed", "workload")
    mix = wl0["mix_at_miss_rate"]
    snap = first["snapshot"]
    return {
        "date": comp.get("date"), "sources": sources, "caveat": first.get("caveat"),
        "workload": {k: sig(wl0[k]) if isinstance(wl0[k], float) else wl0[k]
                     for k in ("prompt_mean", "output", "warm_turn", "think_time_s", "sub_ratio", "cap", "miss_rate",
                               "requests_per_session_hour", "mixed_tokens_per_session_hour")},
        "mix": {k: sig(v, 4) for k, v in mix.items()},
        "billing": {"ttl_s": mb["cache_ttl_s"], "miss": sig(mb["miss"], 4), "cached": sig(mb["cached"], 4)},
        "rent_tiers": first.get("rent_tiers"),
        "rent_other": ({"tiers": first["rent_other_tiers"]["tiers"], "rent": rent_block(first["rent_other_tiers"]["rent"], TERMS)}
                       if first.get("rent_other_tiers") else None),
        "terms": TERMS,
        "rent": rent_block(first["rent"], TERMS),
        "fx": snap.get("rental_fx"),
        "price_pages": [[p, u, r] for (p, u), r in sorted(pp_pages.items())],
        "openrouter": ({"url": price_sources["openrouter"]["url"], "read": str(price_sources["openrouter"]["fetched"])[:10]}
                       if sources == "all" and "openrouter" in price_sources else None),
        "models": out_models,
    }


def artifact(data_js_obj: str) -> str:
    page = PAGE.read_text(encoding="utf-8")
    m = re.search(r"<!-- ARTIFACT:BEGIN -->(.*)<!-- ARTIFACT:END -->", page, re.S)
    if not m:
        raise ValueError(f"{PAGE}: ARTIFACT:BEGIN / ARTIFACT:END markers missing")
    body = m.group(1)
    script = SCRIPT.read_text(encoding="utf-8")
    imp = "import { MARKET } from './market-data.js';"
    if imp not in script:
        raise ValueError(f"{SCRIPT}: expected the line {imp!r}")
    script = script.replace(imp, "const MARKET = " + data_js_obj + ";")
    head = ('<title>Rent against token prices</title>\n'
            '<meta name="claude:viewport-fit-sides" content="cover">\n')
    return head + body.strip() + '\n<script type="module">\n' + script.replace("</script", "<\\/script") + "\n</script>\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("comparison", type=Path, nargs="?",
                    default=ROOT / "data/market/comparison-2026-11-18.json")
    ap.add_argument("--sources", choices=("provider_prices", "all"), default="provider_prices")
    ap.add_argument("--snapshot", type=Path, default=ROOT / "data/market",
                    help="directory holding provider_prices.json (for each price page's URL and date)")
    ap.add_argument("--artifact", type=Path, help="also write one self-contained HTML file here (not in the repo)")
    ap.add_argument("--data-out", type=Path, default=DATA_JS)
    args = ap.parse_args()

    data = build(json.loads(args.comparison.read_text(encoding="utf-8")), args.sources, args.snapshot)
    obj = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    if args.artifact:
        if args.artifact.resolve().is_relative_to(ROOT):
            raise ValueError("--artifact must point outside the repo: it may carry OpenRouter's listing")
        args.artifact.parent.mkdir(parents=True, exist_ok=True)
        args.artifact.write_text(artifact(obj), encoding="utf-8")
        print(f"wrote {args.artifact} ({args.artifact.stat().st_size} B, sources={args.sources})")
    else:
        if args.sources != "provider_prices" and args.data_out.resolve() == DATA_JS.resolve():
            raise ValueError("the committed market-data.js carries provider_prices rows only; "
                             "use --artifact or --data-out for --sources all")
        args.data_out.write_text("// generated by scripts/market_page.py; do not edit\nexport const MARKET = "
                                 + obj + ";\n", encoding="utf-8")
        print(f"wrote {args.data_out} ({args.data_out.stat().st_size} B, sources={args.sources})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
