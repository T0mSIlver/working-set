"""What `ws test` reads off the server before it sends load, and the check of
the config against it.

The config is a claim about the deployment (max_model_len, the KV layout, the
reserve); every prediction `ws test` scores is priced on it. A run against a
server that differs measures that server against the wrong prediction, so the
differences stop the run (`--allow-config-mismatch` to go on) rather than ride
along silently.
"""
from __future__ import annotations

import re

from . import model as M

POOL_TOLERANCE = 0.15       # the server's KV pool vs the config's, relative


def cache_info(text: str, engine: str | None = None) -> dict:
    """block_size, pool_tokens (num_gpu_blocks x block_size) and
    prefix_caching from vllm:cache_config_info; a key is None when the line or
    the label is absent. `engine` picks one engine of a multi-engine dump."""
    out: dict = {"block_size": None, "pool_tokens": None,
                 "prefix_caching": None}
    for line in text.splitlines():
        if not line.startswith("vllm:cache_config_info"):
            continue
        lab = dict(re.findall(r'(\w+)="([^"]*)"', line))
        if engine is not None and lab.get("engine", engine) != str(engine):
            continue
        try:
            out["block_size"] = int(lab["block_size"])
        except (KeyError, ValueError):
            pass
        if out["block_size"] and lab.get("num_gpu_blocks", "").isdigit():
            out["pool_tokens"] = int(lab["num_gpu_blocks"]) * out["block_size"]
        pc = lab.get("enable_prefix_caching")
        if pc is not None:
            out["prefix_caching"] = pc.lower() == "true"
        break
    return out


def has_spec_decode(text: str) -> bool:
    return bool(re.search(r"^vllm:spec_decode", text, re.M))


def read_server(base_url: str, api_key: str, metrics_url: str | None,
                headers: dict | None = None, engine: str | None = None) -> dict:
    """{"max_model_len", "model", "block_size", "pool_tokens",
    "prefix_caching", "spec_decode", "errors"}: each value None when the
    server does not give it, and `errors` says what could not be read."""
    from .bench import _endpoint_info, read_metrics
    s: dict = {"max_model_len": None, "model": "", "block_size": None,
               "pool_tokens": None, "prefix_caching": None,
               "spec_decode": None, "resolved": None, "errors": []}
    info = _endpoint_info(base_url.rstrip("/"), api_key)
    if "error" in info:
        s["errors"].append(f"GET {base_url}/models failed ({info['error']})")
    else:
        s["max_model_len"] = info.get("max_model_len")
        s["model"] = info.get("model", "")
    if metrics_url:
        try:
            text = read_metrics(metrics_url, headers or {})
        except ValueError as e:
            s["errors"].append(str(e))
        else:
            s.update(cache_info(text, engine))
            s["spec_decode"] = has_spec_decode(text)
            from .metrics import adapter_for
            s["resolved"] = sorted(adapter_for(text, engine).resolution()
                                   .resolved)
    return s


def blind_metrics(server: dict) -> str | None:
    """Why the /metrics gives the rails nothing to watch, None when it does."""
    res = server.get("resolved")
    if res is None or {"requests_running", "kv_cache_usage"} & set(res):
        return None
    return ("the metrics adapter resolved neither requests_running nor "
            "kv_cache_usage from the /metrics, so the queue and KV rails and "
            "the occupancy fit have nothing to read")


def check_config(cfg, server: dict, metrics_url: str | None
                 ) -> tuple[list[str], list[str]]:
    """(differences, notes). A difference is a fact about the server that
    disagrees with, or is missing for, the config; a note is a comparison that
    could not run because the run has no /metrics."""
    d = cfg.deployment
    diffs: list[str] = list(server["errors"])
    notes: list[str] = []
    mml = server["max_model_len"]
    if mml is None:
        if not server["errors"]:
            diffs.append("/v1/models gives no max_model_len, so "
                         "deployment.max_model_len is unchecked")
    elif int(mml) != d.max_model_len:
        diffs.append(f"the server's max_model_len is {int(mml):,}, the config's "
                     f"deployment.max_model_len is {d.max_model_len:,}")
    if not metrics_url:
        notes.append("the server's KV pool was not compared with the config's "
                     "(no --metrics-url)")
    elif server["pool_tokens"] is None:
        if not server["errors"]:
            diffs.append(f"{metrics_url} has no vllm:cache_config_info "
                         "num_gpu_blocks and block_size, so the KV pool is "
                         "unchecked")
    else:
        pred = M.kv_pool_tokens(cfg.to_model(), cfg.to_topology())
        got = server["pool_tokens"]
        if not pred or abs(got - pred) > POOL_TOLERANCE * pred:
            layout = ("replicate" if d.kv_sharding == "replicate"
                      else "dcp")
            diffs.append(
                f"the server's KV pool is {got:,} tokens, the config predicts "
                f"{pred:,.0f} (deployment.kv_sharding = {layout!r}; "
                f"calibration.reserve_gib "
                + ("unset" if cfg.calibration.reserve_gib is None
                   else f"{cfg.calibration.reserve_gib:g}") + ")")
    spec = cfg.to_model().spec_tokens
    if server["spec_decode"] is not None:
        if spec > 0 and not server["spec_decode"]:
            notes.append(f"the model has spec_tokens {spec} but /metrics has "
                         "no spec_decode counters: acceptance is not "
                         "measured here")
        elif spec == 0 and server["spec_decode"]:
            notes.append("/metrics has spec_decode counters but the model "
                         "has spec_tokens 0: the server drafts and the "
                         "prediction does not")
    return diffs, notes


MISMATCH_HELP = (
    "Fix deployment.max_model_len; for the pool set deployment.kv_sharding "
    "(plain TP is \"replicate\") and calibration.reserve_gib (or run `ws "
    "calibrate --vllm-log` on the server's log); or pass "
    "--allow-config-mismatch to run anyway, with the differences written into "
    "the record's not-established notes")
