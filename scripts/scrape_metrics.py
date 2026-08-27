#!/usr/bin/env python3
"""vLLM /metrics scraper — companion to measure_mfu.py (research/prefill.md #1).

Run this next to the vLLM backend while measure_mfu.py fires requests through
whatever proxy sits in front. One timestamped snapshot block per interval;
each block holds every vllm:* line of a full /metrics dump, so ANY pair of
blocks can be deltaed afterwards. Counter deltas around an isolated request
give the engine-side prefill time and FLOP count (client TTFT includes the
proxy tax and can only bound MFU from below).

  python scrape_metrics.py http://backend:8000/metrics
  python scrape_metrics.py http://backend:8000/metrics --interval 2 --api-key KEY

Ctrl-C to stop. Analysis gotcha (2026-08-27): vllm:estimated_flops_per_gpu_total
flushes MID-request — reconcile FLOPs over whole windows, not per scrape step.
"""
import argparse, datetime, time
import requests

ap = argparse.ArgumentParser()
ap.add_argument("url")
ap.add_argument("--interval", type=float, default=1.0)
ap.add_argument("--api-key")
args = ap.parse_args()

hdr = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
out = f"mfu_scrape_{datetime.datetime.now(datetime.timezone.utc):%Y%m%dT%H%M%SZ}.log"
print(f"writing {out} - ctrl-C to stop")

with open(out, "a", buffering=1) as f:
    while True:
        now = time.time()
        human = datetime.datetime.fromtimestamp(now, datetime.timezone.utc)
        f.write(f"===== {now:.3f} {human:%Y-%m-%d %H:%M:%S} UTC\n")
        try:
            text = requests.get(args.url, headers=hdr, timeout=5).text
            f.writelines(l + "\n" for l in text.splitlines() if l.startswith("vllm:"))
        except Exception as e:
            f.write(f"# SCRAPE FAILED {now:.3f} : {e}\n")
        time.sleep(args.interval)
