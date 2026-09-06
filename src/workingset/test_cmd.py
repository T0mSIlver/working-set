"""`ws test` and `ws report` — drive the probes, score the hypotheses.

Kept out of cli.py so the argument wiring stays readable: this module owns the
plan/dry-run/run/record flow, cli.py owns the parser.
"""
from __future__ import annotations

import asyncio
import sys
from dataclasses import replace

from . import __version__
from .hypotheses import (NOT_ESTABLISHED, REGISTRY, Measurement, RunContext,
                         Verdict, plan as make_plan)
from .probe.options import ProbeOptions
from .probe.population import eval_rung
from .probe.request import EndpointSpec, make_client
from .probe.session import sampler_selfcheck
from .record import RunRecord, not_established_notes
from .report import print_report
from .shared import BudgetAbort, ProbeBudget, SharedOptions


# ============================================================================
# wiring
# ============================================================================
def build_options(args, cfg) -> ProbeOptions:
    """Flags override the config; the context cap follows max_model_len unless
    the user says otherwise (the model's max_seq_len analogue)."""
    kw = {"context_cap_tokens": cfg.deployment.max_model_len}
    for flag, field in (("rungs", "rungs"), ("max_users", "max_users"),
                        ("ramp_s", "ramp_s"), ("measure_s", "measure_s"),
                        ("turns_per_user", "turns_per_user"),
                        ("burst", "burst"), ("burst_users", "burst_users"),
                        ("sample_requests", "sample_requests"),
                        ("chars_per_token", "chars_per_token"),
                        ("context_cap_tokens", "context_cap_tokens"),
                        ("request_timeout_s", "request_timeout_s"),
                        ("api", "api"),
                        ("freeze_threshold_ms", "freeze_threshold_ms"),
                        ("seed", "seed")):
        v = getattr(args, flag, None)
        if v is not None:
            kw[field] = v
    kw["ignore_eos"] = not getattr(args, "no_ignore_eos", False)
    return ProbeOptions(**kw)


def build_budget(args) -> ProbeBudget:
    """The safety rails. WITHOUT `--exclusive` the defaults are the
    conservative ones (`ProbeBudget.conservative`) — a shared endpoint gets
    the timid budget unless the operator says otherwise, and every flag given
    overrides just that one rail. `--exclusive` takes them off, because the
    queue the ladder builds IS the measurement there."""
    kw = {}
    for flag, field_ in (("max_extra_load", "max_extra_load"),
                         ("abort_if_waiting", "abort_if_waiting"),
                         ("abort_if_kv_above", "abort_if_kv_above"),
                         ("max_probe_tokens", "max_probe_tokens"),
                         ("canary_every_s", "canary_every_s"),
                         ("canary_baseline_s", "canary_baseline_s"),
                         ("canary_window_s", "canary_window_s"),
                         ("canary_drift", "canary_drift"),
                         ("canary_min_n", "canary_min_n")):
        v = getattr(args, flag, None)
        if v is not None:
            kw[field_] = v
    if getattr(args, "no_canary", False):
        kw["canary"] = False
    if getattr(args, "exclusive", False):
        return ProbeBudget.for_exclusive(**kw)
    return ProbeBudget.conservative(**kw)


def build_shared(args) -> SharedOptions:
    kw = {}
    for flag, field_ in (("shared_lengths", "lengths"),
                         ("shared_rounds", "rounds"),
                         ("shared_warm_turns", "warm_turns"),
                         ("shared_duration_s", "duration_s"),
                         ("max_extrapolation", "max_extrapolation"),
                         ("seed", "seed")):
        v = getattr(args, flag, None)
        if v is not None:
            kw[field_] = v
    kw["ladder"] = bool(getattr(args, "shared_ladder", False))
    return SharedOptions(**kw)


def build_endpoint(args, cfg) -> EndpointSpec:
    return EndpointSpec.from_config(
        cfg.endpoint, api=getattr(args, "api", None) or "completions",
        base_url=getattr(args, "base_url", None),
        model=getattr(args, "model_id", None),
        api_key_env=getattr(args, "api_key_env", None))


def open_metrics(url: str | None):
    """Duck-typed hand-off to `workingset.metrics.MetricsSampler`, built by a
    separate module. Only `at(t)` and `window(t0, t1)` are relied on; start /
    stop are called when they exist, and the constructor is tried positionally
    then by keyword so the two modules need not agree on more than the name."""
    if not url:
        return None
    try:
        from .metrics import MetricsSampler        # type: ignore
    except Exception as e:
        raise SystemExit(f"--metrics-url needs workingset.metrics: {e}")
    try:
        return MetricsSampler(url)
    except TypeError:
        return MetricsSampler(url=url)


async def _maybe(obj, *names):
    for n in names:
        fn = getattr(obj, n, None)
        if fn is None:
            continue
        r = fn()
        if asyncio.iscoroutine(r):
            await r
        return


# ============================================================================
# dry run
# ============================================================================
def dry_run(cfg, preds, opts, ep, pl, args, out=None) -> int:
    from .hypotheses.base import SHARED
    from .shared import plan_lines

    out = out or sys.stdout
    w = lambda s="": print(s, file=out)          # noqa: E731
    slo, wl = cfg.slo, cfg.workload
    mode = "exclusive" if args.exclusive else "shared"
    budget = build_budget(args)
    sopts = build_shared(args)

    w("DRY RUN — plan only, no requests sent\n")
    w(f"endpoint : {ep.base_url}  model={ep.model or '<unset>'}  "
      f"api={ep.api}  key={'set' if ep.api_key else 'unset'}")
    w(f"mode     : {mode}"
      + ("  (may generate the population it measures)" if args.exclusive
         else "  (shares the endpoint; exclusive hypotheses are skipped)"))
    w(f"metrics  : {args.metrics_url or 'none'}")
    w(f"SLOs     : p{slo.percentile} TTFT <= {slo.ttft_budget_s:g}s, "
      f"per-user p50 decode >= {slo.itl_floor_tok_s:g} tok/s")
    w(f"probes   : {', '.join(sorted(pl.probes)) or 'none'}")

    w("\nPROBE BUDGET — the rails this run may not cross"
      + ("" if args.exclusive else " (conservative by default; every one of "
                                   "these is a flag)"))
    for line in budget.describe(bool(args.metrics_url)):
        w(f"  {line}")
    if not args.exclusive:
        w("  any rail that trips aborts the run, writes a record carrying the "
          "reason, and exits non-zero")

    if SHARED in pl.probes:
        w("\nSHARED-MODE PLAN — other people's traffic is a covariate, not "
          "noise")
        for line in plan_lines(cfg, opts, sopts, budget,
                               bool(args.metrics_url)):
            w(f"  {line}")

    if pl.run_ladder:
        from .probe.ladder import build_ladder
        ladder = build_ladder(preds.predicted_limit_users, opts.rungs,
                              opts.max_users, preds.operating_point_users)
        w("\nLOAD LADDER (users; each rung: "
          f"ramp {opts.ramp_s:g}s + measure {opts.measure_s:g}s)")
        w(f"{'users':>8} {'+sub':>6} {'offered req/s':>14}")
        for pop in ladder:
            n_sub = round(pop * wl.subagent_ratio)
            w(f"{pop:>8} {n_sub:>6} "
              f"{(pop + n_sub) / wl.think_time_s:>14.2f}")
        est = len(ladder) * (opts.ramp_s + opts.measure_s)
        w(f"full-ladder wall clock (no early stop): ~{est / 60:.0f} min")
    if "sample" in pl.probes:
        n = opts.sample_requests * (opts.sample_warm_turns + 2)
        w(f"\nSAMPLE PROBE: {opts.sample_requests} sessions x "
          f"{opts.sample_warm_turns + 2} turns = ~{n} requests "
          "(1 establishing + warm turns + 1 forced miss each)")
    if "burst" in pl.probes:
        # the standing load the RUN will use, from the run's own rule
        pop = RunContext(cfg, preds, opts, ep, burst=opts.burst,
                         burst_users=opts.burst_users)._burst_pop()
        w(f"\nBURST PROBE: {opts.burst} simultaneous forced misses at a "
          f"{pop}-user standing load, after a {opts.ramp_s:g}s ramp")

    # sampler self-check: the sampled raw median and log-sd must reproduce the
    # configured (median, sigma) — this is the distribution warm capacity and
    # the prefill tail are priced on
    w("\nPROMPT-LENGTH SAMPLER (n=20,000 per class; tokens, clipped to "
      f"[prefix, {opts.context_cap_tokens:,}])")
    rows, ok = sampler_selfcheck(wl, opts)
    w(f"{'class':<10} {'median cfg':>10} {'median smp':>10} {'sigma cfg':>9} "
      f"{'sigma smp':>9} {'p5':>8} {'p50':>8} {'p95':>9} {'mean':>9}  check")
    for r in rows:
        w(f"{r['class']:<10} {r['median_cfg']:>10,.0f} {r['median_smp']:>10,.0f} "
          f"{r['sigma_cfg']:>9.2f} {r['sigma_smp']:>9.3f} {r['p5']:>8,.0f} "
          f"{r['p50']:>8,.0f} {r['p95']:>9,.0f} {r['mean']:>9,.0f}  "
          f"{'PASS' if r['ok'] else 'FAIL'}")
    w(f"(chars per token: {opts.chars_per_token:g}; a p50 user prompt is "
      f"~{rows[0]['p50'] * opts.chars_per_token / 1e3:,.0f}k chars on the wire)")

    w("\nPREDICTIONS UNDER TEST")
    for k in ("warm_capacity_p5", "decode_ceiling_users",
              "latency_ceiling_users", "saturation_ceiling_users",
              "binding_constraint", "predicted_limit_users",
              "operating_point_users", "ttft_miss_s", "bstar_misses",
              "steady_decode_seqs", "steady_decode_tok_s", "itl_normal_ms",
              "itl_worst_freeze_ms"):
        v = getattr(preds, k, None)
        if v is not None:
            w(f"  {k:<26} {v}")

    w(f"\nHYPOTHESES SELECTED ({len(pl.selected)})")
    for h in pl.selected:
        req = ",".join(sorted(h.requires)) or "-"
        probes = ",".join(sorted(h.probes | h.conditional_probes(pl.probes)))
        w(f"  {h.key:<14} needs [{req}]  reads [{probes or '-'}]")
        w(f"    {_statement(h, cfg, preds, pl.probes)}")
    if pl.skipped:
        w(f"\nSKIPPED ({len(pl.skipped)}) — listed, never silently run")
        for h, reason in pl.skipped:
            w(f"  {h.key:<14} {reason}")

    if not ok:
        print("\nSAMPLER SELF-CHECK FAILED", file=sys.stderr)
        return 1
    w("\nsampler self-check PASSED; run without --dry-run to measure")
    return 0


# ============================================================================
# the run
# ============================================================================
def _statement(h, cfg, preds, probes) -> str:
    """A hypothesis may phrase itself differently depending on what the run
    will actually do (H-ttft-miss only claims a ladder rung when there is
    one)."""
    fn = getattr(h, "statement_for", None)
    return fn(cfg, preds, probes) if fn else h.statement(cfg, preds)


def _row(h, ctx, pr, m, v) -> dict:
    return {"key": h.key, "title": h.title, "requires": sorted(h.requires),
            "probes": sorted(h.probes | h.conditional_probes(ctx.probes)),
            "statement": _statement(h, ctx.cfg, ctx.predictions, ctx.probes),
            "prediction": pr.to_dict(), "measurement": m.to_dict(),
            "verdict": v.to_dict()}


async def _score(ctx, pl, rows: list) -> list[dict]:
    """Score each hypothesis, appending to `rows` as it goes.

    `rows` is the CALLER's list. On Ctrl-C the verdicts already computed are
    the run's only output, and the harness printed exactly those — it built
    its PREDICTED vs MEASURED block over whatever had completed. Returning a
    fresh list from here meant an interrupt threw them away.
    """
    for h in pl.selected:
        pr = h.predict(ctx.cfg, ctx.predictions)
        m = await h.measure(ctx)
        rows.append(_row(h, ctx, pr, m, h.verdict(pr, m)))
    return rows


async def _score_from_cache(ctx, pl, done: set) -> list[dict]:
    """After an interrupt: score the hypotheses that never ran against the
    evidence already in the cache, WITHOUT starting a probe.

    `ctx.freeze()` makes every probe answer from the cache or not at all, so a
    hypothesis reads a completed rung if one exists and otherwise reports "not
    separable" — never a verdict over nothing, and never a fresh 64-user
    population fired off during a Ctrl-C.
    """
    ctx.freeze()
    rows = []
    for h in pl.selected:
        if h.key in done:
            continue
        pr = h.predict(ctx.cfg, ctx.predictions)
        try:
            m = await h.measure(ctx)
            v = h.verdict(pr, m)
        except (Exception, KeyboardInterrupt, asyncio.CancelledError) as e:
            # the hypothesis the interrupt landed inside will raise again on
            # the way out; that is a row, not a crash that loses every other
            # verdict the run had already produced
            m = Measurement(text="interrupted",
                            data={"reason": f"{type(e).__name__} while "
                                            "measuring"})
            v = Verdict(NOT_ESTABLISHED,
                        "the run was interrupted before this hypothesis "
                        "finished measuring")
        rows.append(_row(h, ctx, pr, m, v))
    return rows


def _progress(kind, payload) -> None:
    if kind == "rung":
        print(f"\n--- rung: {payload} users ---", flush=True)
    elif kind == "rung-done":
        r = payload
        v = "PASS" if r.passed else "FAIL (" + "; ".join(r.reasons) + ")"
        print(f"  TTFT hit p50 {r.ttft_hit_p50:.2f}s / miss p50 "
              f"{r.ttft_miss_p50:.2f}s | decode p50 {r.decode_p50:.1f} tok/s "
              f"| {r.achieved_rps:.2f} req/s | {v}", flush=True)
    elif kind == "blown":
        print("  SLOs clearly blown — stopping the ladder early (higher rungs "
              "would only stress the endpoint)", flush=True)
    elif kind == "sample":
        print(f"\n--- sample probe: {payload} sessions ---", flush=True)
    elif kind == "burst":
        print(f"\n--- burst probe: N={payload[0]} at {payload[1]} standing "
              "users ---", flush=True)
    elif kind == "shared":
        print("\n--- shared-endpoint probe (rails on) ---", flush=True)
    elif kind == "shared-round":
        print(f"  round {payload[0]}: {payload[1]} prompt lengths",
              flush=True)


async def run_test(cfg, preds, opts, ep, pl, metrics, exclusive: bool,
                   budget=None, shared_opts=None) -> tuple:
    """Returns (rows, probe cache, capacity bracket, interrupted, aborted)."""
    interrupted = False
    aborted: BudgetAbort | None = None
    rows: list[dict] = []
    await _maybe(metrics, "start", "open")
    try:
        async with make_client() as client:
            ctx = RunContext(cfg, preds, opts, ep, client=client,
                             metrics=metrics, exclusive=exclusive,
                             burst=opts.burst, burst_users=opts.burst_users,
                             probes=pl.probes, on_progress=_progress,
                             budget=budget, shared_opts=shared_opts)
            try:
                await _score(ctx, pl, rows)
            except (KeyboardInterrupt, asyncio.CancelledError):
                # the verdicts already computed ARE the run's output; the rest
                # are scored against the cache, starting nothing new
                interrupted = True
                rows += await _score_from_cache(ctx, pl,
                                                {r["key"] for r in rows})
            except BudgetAbort as e:
                # a safety rail on somebody else's endpoint. Everything the
                # probe HAD measured is kept (the partial result rides on the
                # exception), the rest is scored against the cache without
                # sending anything more, and the run exits non-zero.
                aborted = e
                if e.result is not None:
                    ctx.seed(shared=e.result)
                rows += await _score_from_cache(ctx, pl,
                                                {r["key"] for r in rows})
    finally:
        await _maybe(metrics, "stop", "aclose", "close")

    cached = ctx.cached()
    if interrupted and ctx.partial:
        # a partially-measured rung still carries information: shown for
        # orientation, excluded from every verdict
        pop, n_sub, traces, ms = ctx.partial
        if ms is not None and any(t.t_send >= ms for t in traces):
            r = eval_rung(pop, n_sub, traces, ms, cfg, opts)
            r.partial = True
            cached["rungs"].append(r.to_dict())
    bracket = _bracket(cached["rungs"])
    return rows, cached, bracket, interrupted, aborted


def _bracket(rung_dicts) -> list:
    full = [r for r in rung_dicts if not r.get("partial")]
    passes = [r["pop"] for r in full if r.get("passed")]
    fails = [r["pop"] for r in full if not r.get("passed")]
    return [max(passes) if passes else None, min(fails) if fails else None]


# ============================================================================
# commands
# ============================================================================
def cmd_test(args) -> int:
    from .cli import _apply_overrides       # deferred: cli imports this module
    from .config import load_config
    from .predict import predict

    cfg = _apply_overrides(load_config(args.config), args)
    # the flag wins; the config's own metrics_url is the fallback
    metrics_url = args.metrics_url or cfg.endpoint.metrics_url
    args.metrics_url = metrics_url
    if metrics_url != cfg.endpoint.metrics_url:
        cfg = replace(cfg, endpoint=replace(cfg.endpoint,
                                            metrics_url=metrics_url))
    preds = predict(cfg, n_iter=args.n_iter, seed=args.predict_seed)
    opts = build_options(args, cfg)
    ep = build_endpoint(args, cfg)

    selected = REGISTRY.select(None if (args.all or not args.keys) else args.keys)
    pl = make_plan(selected, exclusive=args.exclusive,
                   metrics=bool(args.metrics_url), burst=opts.burst)

    if args.dry_run:
        return dry_run(cfg, preds, opts, ep, pl, args)
    if not pl.selected:
        print("nothing to run: every selected hypothesis was skipped.",
              file=sys.stderr)
        for h, reason in pl.skipped:
            print(f"  {h.key:<14} {reason}", file=sys.stderr)
        return 2

    metrics = open_metrics(args.metrics_url)
    budget, sopts = build_budget(args), build_shared(args)
    try:
        rows, cached, bracket, interrupted, aborted = asyncio.run(
            run_test(cfg, preds, opts, ep, pl, metrics, args.exclusive,
                     budget=budget, shared_opts=sopts))
    except KeyboardInterrupt:
        print("\ninterrupted before any hypothesis was scored", file=sys.stderr)
        return 130

    opt_dict = {**opts.to_dict(), "probe_budget": budget.to_dict(),
                "shared": sopts.to_dict()}
    rec = RunRecord.new(
        __version__, mode="exclusive" if args.exclusive else "shared",
        interrupted=interrupted, config=cfg.to_dict(),
        predictions=preds.to_dict(), options=opt_dict,
        endpoint=ep.redacted(), plan=pl.to_dict(),
        rungs=cached["rungs"], sample=cached["sample"], burst=cached["burst"],
        shared=cached.get("shared"), hypotheses=rows,
        skipped=[{**h.describe(), "reason": r} for h, r in pl.skipped],
        not_established=not_established_notes(
            cfg, opts, pl, rungs=cached["rungs"], sample=cached["sample"],
            burst=cached["burst"], hypotheses=rows, exclusive=args.exclusive,
            metrics=bool(metrics_url), interrupted=interrupted,
            shared=cached.get("shared")),
        measured_capacity_bracket=bracket)

    print_report(rec)
    if args.out:
        rec.save(args.out)
        print(f"\nrun record written to {args.out}")
    if aborted is not None:
        # the record is written FIRST: an abort that loses its own evidence
        # tells the operator nothing about why the endpoint was left alone
        print(f"\nPROBE ABORTED: {aborted.reason}", file=sys.stderr)
        return 3
    return 130 if interrupted else 0


def cmd_report(args) -> int:
    print_report(RunRecord.load(args.record))
    return 0


def cmd_hypotheses(_args) -> int:
    print(f"{'key':<14} {'requires':<20} what it measures")
    for h in REGISTRY:
        req = ",".join(sorted(h.requires)) or "-"
        print(f"{h.key:<14} {req:<20} {h.title}")
    return 0
