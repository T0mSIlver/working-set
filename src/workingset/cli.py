"""`ws` — the workingset command line.

    ws predict CONFIG [--closed] [--json]      price a configuration
    ws init [--model KEY --gpu PART --tp N ...] write a starter config
    ws models                                  list model / GPU keys
    ws selfcheck                               run the model's self-checks
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace

from . import __version__
from . import model as M
from .config import RunConfig, load_config
from .predict import predict


def _fmt(x) -> str:
    if isinstance(x, float) and not math.isfinite(x):
        return "inf"
    if isinstance(x, int) and x >= 999999:
        return "never binds"
    if isinstance(x, float):
        return f"{x:,.2f}"
    if isinstance(x, int):
        return f"{x:,}"
    return str(x)


def cmd_predict(args) -> int:
    cfg = load_config(args.config)
    cfg = _apply_overrides(cfg, args)
    p = predict(cfg, closed=args.closed, n_iter=args.n_iter, seed=args.seed)
    if args.json:
        out = {"config": cfg.to_dict(), "predictions": p.to_dict(),
               "system": p.system(), "mode": "closed" if args.closed else "open",
               "workingset": __version__}
        print(json.dumps(out, indent=2, allow_nan=False))
        return 0
    d, w = cfg.deployment, cfg.workload
    print(f"{cfg.to_model().name} on {d.gpus} (TP{d.tensor_parallel} x DP{d.replicas}), "
          f"chunk {d.max_num_batched_tokens:,}, max_model_len {d.max_model_len:,}")
    print(f"operating point: {w.users} users/group, think {w.think_time_s} s, "
          f"miss {w.miss_rate:.0%}, {'closed' if args.closed else 'open'} loop")
    print()
    rows = [("cache (warm p5, users)", p.warm_capacity_p5),
            ("decode (users at floor)", p.decode_ceiling_users),
            ("latency (miss TTFT = budget)", p.latency_ceiling_users),
            ("saturation (prefill duty 100%)", p.saturation_ceiling_users)]
    for k, v in rows:
        mark = "  <- binds" if k.startswith(p.binding_constraint) else ""
        print(f"  {k:32} {_fmt(v):>14}{mark}")
    if p.replicas > 1:
        print(f"  (per replica group; x{p.replicas} under balanced routing)")
    print()
    print(f"  req/s main {p.req_rate_main}  prefill duty {p.prefill_duty:.1%}  "
          f"TTFT miss {_fmt(p.ttft_miss_s)} s  hit {_fmt(p.ttft_hit_s)} s  "
          f"B* {p.bstar_misses}")
    return 0


def _apply_overrides(cfg: RunConfig, args) -> RunConfig:
    dep = {k: v for k, v in (("model", args.model), ("gpu", args.gpu),
                             ("tensor_parallel", args.tp), ("replicas", args.dp),
                             ("weight_dtype", args.weight_dtype),
                             ("kv_dtype", args.kv_dtype),
                             ("max_num_batched_tokens", args.chunk),
                             ("max_model_len", args.max_model_len),
                             ("ram_gib", args.ram_gib)) if v is not None}
    wl = {k: v for k, v in (("users", args.users), ("miss_rate", args.miss_rate),
                            ("think_time_s", args.think)) if v is not None}
    if dep:
        cfg = replace(cfg, deployment=replace(cfg.deployment, **dep))
    if wl:
        cfg = replace(cfg, workload=replace(cfg.workload, **wl))
    return cfg


def cmd_init(args) -> int:
    cfg = _apply_overrides(RunConfig(), args)
    cfg.validate()
    text = cfg.dumps("json" if args.json else "toml")
    if args.output and args.output != "-":
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {args.output}")
    else:
        sys.stdout.write(text)
    return 0


def cmd_models(_args) -> int:
    print("models:")
    for k, m in M.MODELS.items():
        arms = "fp8" + (", nvfp4" if m.nvfp4_w else "")
        print(f"  {k:8} {m.name}  [{arms}]")
    print("gpus:")
    for k, g in M.GPUS.items():
        print(f"  {k:8} {g.vram / 1e9:.0f} GB, {g.hbm_bw / 1e12:.1f} TB/s"
              + (", nvfp4" if g.supports_nvfp4 else ""))
    return 0


def cmd_selfcheck(_args) -> int:
    M._selfcheck()
    return 0


def _add_deploy_flags(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--model", help="model key (see `ws models`)")
    ap.add_argument("--gpu", help="GPU part (H200, B300)")
    ap.add_argument("--tp", type=int, help="tensor-parallel group size")
    ap.add_argument("--dp", type=int, help="data-parallel replica groups")
    ap.add_argument("--weight-dtype", choices=M.WEIGHT_DTYPES)
    ap.add_argument("--kv-dtype", choices=M.KV_DTYPES)
    ap.add_argument("--chunk", type=int, help="max_num_batched_tokens")
    ap.add_argument("--max-model-len", type=int)
    ap.add_argument("--ram-gib", type=float, help="CPU KV offload per group, GiB")
    ap.add_argument("--users", type=int, help="operating point, users per group")
    ap.add_argument("--miss-rate", type=float)
    ap.add_argument("--think", type=float, help="think time, s")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="ws", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"workingset {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("predict", help="price a configuration")
    p.add_argument("config", help="workingset.toml / .json / a harness .py")
    p.add_argument("--closed", action="store_true",
                   help="closed-loop conversion (think_time_s is then the WAITING "
                        "time Z per request; the model supplies the service time)")
    p.add_argument("--json", action="store_true")
    p.add_argument("--n-iter", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    _add_deploy_flags(p)
    p.set_defaults(fn=cmd_predict)

    p = sub.add_parser("init", help="write a starter config")
    p.add_argument("-o", "--output", default="workingset.toml")
    p.add_argument("--json", action="store_true")
    _add_deploy_flags(p)
    p.set_defaults(fn=cmd_init)

    p = sub.add_parser("models", help="list model and GPU keys")
    p.set_defaults(fn=cmd_models)

    p = sub.add_parser("selfcheck", help="run the model's self-checks")
    p.set_defaults(fn=cmd_selfcheck)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except (ValueError, KeyError, FileNotFoundError) as e:
        # a config or model refusal is a user-facing message, not a traceback
        print(f"ws: {e.args[0] if e.args else e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
