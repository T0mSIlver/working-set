"""`ws` — the workingset command line.

    ws predict CONFIG [--closed] [--json]      price a configuration
    ws test CONFIG [H-key ...] [--exclusive]   test predictions on an endpoint
    ws report RUN.json                         re-print a run's report
    ws hypotheses                              list the H-* and what they need
    ws init [--model KEY --gpu PART --tp N ...] write a starter config
    ws models                                  list model / GPU keys
    ws selfcheck                               run the model's self-checks
    ws metrics probe|tail|window ...           sample a live /metrics endpoint
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
from .test_cmd import cmd_hypotheses, cmd_report, cmd_test


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

    p = sub.add_parser(
        "test", help="test the predictions against a live endpoint",
        description="Put a configuration's predictions to a live "
                    "OpenAI-compatible endpoint, one hypothesis at a time.",
        epilog="Examples:\n"
               "  ws test workingset.toml --dry-run\n"
               "  ws test workingset.toml --all --exclusive --out run.json\n"
               "  ws test workingset.toml H-ttft-miss H-itl-mean\n"
               "  ws test workingset.toml --exclusive --burst 8\n",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", help="workingset.toml / .json / a harness .py")
    p.add_argument("keys", nargs="*", metavar="H-key",
                   help="hypotheses to test (default: all; see `ws hypotheses`)")
    p.add_argument("--all", action="store_true",
                   help="every hypothesis (the default when no key is given)")
    p.add_argument("--exclusive", action="store_true",
                   help="this run OWNS the endpoint and may generate the "
                        "population it measures. Without it, hypotheses that "
                        "need to are listed as skipped, never silently run.")
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan, predictions, selected hypotheses and "
                        "the sampler self-check; send nothing")
    # endpoint overrides. `--model` stays the model KEY here, as it is in
    # `ws predict` and `ws init` — a downloaded harness names only the served
    # checkpoint, so `ws test harness.py --model 27B` is how you say which
    # model it is. The served id (what goes in the JSON body) is --model-id.
    p.add_argument("--base-url", help="override the config's endpoint base URL")
    p.add_argument("--model-id", "--served-model", dest="model_id",
                   help="override the served model id sent in the request body")
    p.add_argument("--api-key-env", help="env var NAME holding the API key")
    p.add_argument("--api", choices=("completions", "chat"),
                   help="which OpenAI-compatible route to drive "
                        "(default: completions, as the harness does)")
    p.add_argument("--metrics-url", help="vLLM /metrics, for server-side "
                                         "covariates and window deltas")
    # probe knobs
    p.add_argument("--rungs", help="ladder multipliers of predicted_limit_users")
    p.add_argument("--max-users", type=int, help="hard cap on any rung")
    p.add_argument("--ramp-s", type=float, help="per-rung ramp, s")
    p.add_argument("--measure-s", type=float, help="per-rung measure window, s")
    p.add_argument("--turns-per-user", type=int,
                   help="cap turns per session, 0 = unlimited in the window")
    p.add_argument("--burst", type=int, metavar="N",
                   help="fire N simultaneous forced misses (enables H-burst)")
    p.add_argument("--burst-users", type=int,
                   help="standing load for the burst (default: operating point)")
    p.add_argument("--sample-requests", type=int,
                   help="sessions the cheap probe opens (default 8)")
    p.add_argument("--chars-per-token", type=float,
                   help="synthetic-text calibration (default 4.0)")
    p.add_argument("--context-cap-tokens", type=int,
                   help="cap on sampled contexts (default: max_model_len)")
    p.add_argument("--request-timeout-s", type=float,
                   help="per-request read timeout; keep well above the TTFT "
                        "budget so breaches are measured, not dropped")
    p.add_argument("--freeze-threshold-ms", type=float, metavar="MS",
                   help="a gap at or above this counts as a FREEZE "
                        "(default 100). Keep it BELOW the smaller predicted "
                        "freeze; the freeze ladder is threshold-free.")
    # --- shared-endpoint safety rails ------------------------------------
    # These bind WITHOUT --exclusive, where the endpoint belongs to somebody
    # else. Each defaults to the conservative value in
    # `shared.ProbeBudget.conservative()`; --exclusive takes them all off.
    g = p.add_argument_group(
        "shared-endpoint safety rails",
        "Bind without --exclusive. Any rail that trips aborts the run, "
        "records the reason and exits non-zero.")
    g.add_argument("--max-extra-load", type=int, metavar="N",
                   help="never more than N of OUR requests in flight, canary "
                        "included (default 2; 0 = no cap)")
    g.add_argument("--abort-if-waiting", type=float, metavar="N",
                   help="abort when the server's requests_waiting gauge "
                        "exceeds N (default 0, i.e. abort on any queue). "
                        "Needs --metrics-url")
    g.add_argument("--abort-if-kv-above", type=float, metavar="F",
                   help="abort when KV occupancy exceeds this fraction "
                        "(default 0.90). Needs --metrics-url")
    g.add_argument("--max-probe-tokens", type=int, metavar="T",
                   help="total intended prompt tokens the run may send "
                        "(default 2,000,000; 0 = no cap)")
    g.add_argument("--no-canary", action="store_true",
                   help="drop the periodic 1-token canary (it is the only "
                        "contention signal when no --metrics-url is given)")
    g.add_argument("--canary-every-s", type=float, metavar="S")
    g.add_argument("--canary-baseline-s", type=float, metavar="S",
                   help="the run's first S seconds set the canary baseline "
                        "p50 (default 60)")
    g.add_argument("--canary-window-s", type=float, metavar="S",
                   help="trailing window the canary p50 is compared over "
                        "(default 60)")
    g.add_argument("--canary-drift", type=float, metavar="X",
                   help="abort when the trailing canary p50 exceeds X times "
                        "the baseline p50 (default 3.0)")
    g.add_argument("--canary-min-n", type=int, metavar="N",
                   help="samples each canary window needs before the drift "
                        "rule can fire (default 5)")
    # --- shared-endpoint covariate fit -----------------------------------
    g = p.add_argument_group(
        "shared-endpoint covariate fit",
        "Other people's traffic is a covariate, not noise: the probe stamps "
        "every request with the server's load and regresses it out.")
    g.add_argument("--shared-lengths", metavar="F,F,...",
                   help="prompt lengths to probe, as fractions of the context "
                        "cap (default 0.1,0.25,0.5,0.75,1.0)")
    g.add_argument("--shared-rounds", type=int, metavar="N",
                   help="passes over the prompt-length ladder (default 3)")
    g.add_argument("--shared-warm-turns", type=int, metavar="N",
                   help="warm prefix-hit turns per round (default 2)")
    g.add_argument("--shared-ladder", action="store_true",
                   help="run for --shared-duration-s cycling the lengths, and "
                        "report TTFT/ITL binned by the concurrency the server "
                        "happened to be carrying")
    g.add_argument("--shared-duration-s", type=float, metavar="S",
                   help="length of a --shared-ladder run (default 300)")
    g.add_argument("--max-extrapolation", type=float, metavar="SD",
                   help="a fitted verdict is refused when the operating point "
                        "lies more than SD observed standard deviations "
                        "outside the probed range of any regressor "
                        "(default 1.0)")
    p.add_argument("--seed", type=int, help="probe RNG seed")
    p.add_argument("--no-ignore-eos", action="store_true",
                   help="drop the vLLM ignore_eos extension (strict OpenAI "
                        "endpoints); decode tok/s then depends on natural EOS")
    p.add_argument("--n-iter", type=int, default=400,
                   help="model Monte-Carlo iterations for the predictions")
    p.add_argument("--predict-seed", type=int, default=0)
    p.add_argument("-o", "--out", help="write the run record here (JSON)")
    _add_deploy_flags(p)
    p.set_defaults(fn=cmd_test)

    p = sub.add_parser("report", help="re-print a run record's report")
    p.add_argument("record", help="run.json written by `ws test --out`")
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("hypotheses", help="list the H-* and what they need")
    p.set_defaults(fn=cmd_hypotheses)

    p = sub.add_parser("models", help="list model and GPU keys")
    p.set_defaults(fn=cmd_models)

    p = sub.add_parser("selfcheck", help="run the model's self-checks")
    p.set_defaults(fn=cmd_selfcheck)

    # `ws metrics` owns its own subtree, and builds it itself.
    from .metrics.cli import add_subparser as _add_metrics
    _add_metrics(sub)
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
