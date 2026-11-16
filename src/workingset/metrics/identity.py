"""Whether a /metrics URL answers from one server.

A route in front of several vLLM pods (a KServe or OpenShift route, a
gateway) can answer each scrape from a different pod. vLLM labels its series
with model_name and engine, not with the pod, so the text reads like one
server's while its counters jump between pods: a window over two scrapes
then subtracts one pod's counter from another's. Each server process exports
its own `process_start_time_seconds`, so several values over a few scrapes,
each on a fresh connection, say the URL is shared.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

# Two pods picked at random answer all 8 scrapes from one pod with p = 2**-7.
PROBES = 8

_START = re.compile(r"^process_start_time_seconds(?:\{[^}]*\})?\s+(\S+)\s*$",
                    re.M)

SHARED_FIX = ("Pass each pod's own /metrics with --metrics-url, once per pod, "
              "or run without /metrics (ws replay: --no-metrics with "
              "--kv-block-tokens)")


def start_time(text: str) -> float | None:
    """The server process's start time from a /metrics text, None if absent."""
    m = _START.search(text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def check_one_server(url: str, headers: dict[str, str], *, fix: str = SHARED_FIX,
                     probes: int = PROBES, timeout: float = 10.0,
                     verify: bool | str = True) -> str | None:
    """Scrape `url` `probes` times, each on its own connection. Raises
    ValueError, saying what to pass instead, when the scrapes came from more
    than one server process. Returns a note when it cannot tell (a scrape
    failed or no process_start_time_seconds), None when one process answered
    them all."""
    import httpx

    def one(_: int) -> float | None:
        r = httpx.get(url, headers=headers, timeout=timeout, verify=verify)
        r.raise_for_status()
        return start_time(r.text)

    try:
        with ThreadPoolExecutor(max_workers=probes) as ex:
            got = list(ex.map(one, range(probes)))
    except Exception as e:                          # noqa: BLE001
        return (f"{url}: a scrape failed while checking that one server "
                f"answers it ({type(e).__name__}: {e}); not checked")
    seen = sorted({t for t in got if t is not None})
    if len(seen) > 1:
        raise ValueError(
            f"{url} answers from {len(seen)} different servers over {probes} "
            f"scrapes (process_start_time_seconds "
            f"{', '.join(f'{t:.0f}' for t in seen)}): something in front of it "
            "picks a pod per request, so its counters jump between pods and "
            f"every window over them is wrong. {fix}")
    if not seen:
        return (f"{url} has no process_start_time_seconds: ws cannot tell "
                "whether one server answers it")
    return None
