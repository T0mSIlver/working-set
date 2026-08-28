#!/usr/bin/env python3
"""vLLM /metrics scraper — feeds measure_mfu.py and decode_mbu.py.

Run this next to the vLLM backend while measure_mfu.py fires requests through
whatever proxy sits in front. One timestamped snapshot block per interval;
each block holds every vllm:* line of a full /metrics dump, so ANY pair of
blocks can be deltaed afterwards. Counter deltas around an isolated request
give the engine-side prefill time and FLOP count (client TTFT includes the
proxy tax and can only bound MFU from below).

  python scrape_metrics.py http://backend:8000/metrics --interval 0.5
  python scrape_metrics.py http://backend:8000/metrics --interval 2 --api-key KEY

Each block header carries the time the request was SENT plus the round trip
it took, because the counters are as-of some instant inside that interval:
decode_mbu.py divides wall time by step counts, so a window's endpoints are
only as sharp as the scrape that bounded them.

Ctrl-C to stop. Analysis gotcha (2026-08-27): vllm:estimated_flops_per_gpu_total
flushes MID-request — reconcile FLOPs over whole windows, not per scrape step.
"""
import argparse, datetime, time
import requests

ap = argparse.ArgumentParser()
ap.add_argument("url")
ap.add_argument("--interval", type=float, default=1.0)
ap.add_argument("--api-key")
# TLS: a corporate interception proxy presents its own certificate, so the
# system trust store is not enough. --ca-bundle keeps verification and is the
# right fix; --insecure turns it off for a scrape of an internal endpoint.
ap.add_argument("--ca-bundle", help="path to the CA bundle to verify against")
ap.add_argument("--insecure", action="store_true",
                help="skip TLS verification (internal endpoints only)")
args = ap.parse_args()

verify = args.ca_bundle or (False if args.insecure else True)
if args.insecure:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    print("TLS verification DISABLED")

hdr = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
out = f"mfu_scrape_{datetime.datetime.now(datetime.timezone.utc):%Y%m%dT%H%M%SZ}.log"
print(f"writing {out} - ctrl-C to stop")

with open(out, "a", buffering=1) as f:
    while True:
        now = time.time()
        human = datetime.datetime.fromtimestamp(now, datetime.timezone.utc)
        try:
            r = requests.get(args.url, headers=hdr, timeout=5, verify=verify)
            r.raise_for_status()
            lines = [l for l in r.text.splitlines() if l.startswith("vllm:")]
            f.write(f"===== {now:.3f} {human:%Y-%m-%d %H:%M:%S} UTC "
                    f"rtt={time.time()-now:.4f}\n")
            f.writelines(l + "\n" for l in lines)
        except Exception as e:
            f.write(f"===== {now:.3f} {human:%Y-%m-%d %H:%M:%S} UTC rtt=nan\n")
            f.write(f"# SCRAPE FAILED {now:.3f} : {e}\n")
        time.sleep(args.interval)
