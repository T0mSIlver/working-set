#!/usr/bin/env python3
"""Black-box prefill MFU probe via /chat/completions (no /metrics access).

Streams one-token chat completions of random text at several target prompt
lengths, reads the ACTUAL prompt token count from the response usage, takes
min-of-N TTFT per length, fits TTFT(L) = c0 + c1*L + c2*L^2 to strip constant
overhead (network, proxy, scheduling), then prices our analytic FLOPs against
the compute part (research/prefill.md convention: active GEMM params,
excl. embed/lm_head).

  python measure_mfu.py --url https://host/path/to/model-route --api-key $KEY

Requests go to {url}/chat/completions. Caveats: through a shared router the
result is a LOWER BOUND on backend MFU; run in a quiet window.
--peak-tflops/--gpus/--params-b require knowing the deployment.
"""
import argparse, random, re, statistics, sys, time
import requests

RATIO = 2.0  # tokens per generated word, recalibrated from usage after warmup
USE_STREAM_OPTIONS = True


def probe(api, hdr, model, target_tokens, text=None):
    """One streamed request. Returns (ttft_seconds, actual_prompt_tokens, text)."""
    global USE_STREAM_OPTIONS
    text = text or " ".join(
        "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=random.randint(3, 9)))
        for _ in range(max(8, int(target_tokens / RATIO))))
    payload = {"model": model, "messages": [{"role": "user", "content": text}],
               "max_tokens": 1, "temperature": 0, "stream": True}
    if USE_STREAM_OPTIONS:
        payload["stream_options"] = {"include_usage": True}
    t0 = time.monotonic()
    r = requests.post(f"{api}/chat/completions", headers=hdr, timeout=1800,
                      stream=True, json=payload)
    if r.status_code != 200:
        if USE_STREAM_OPTIONS and "stream_options" in r.text:
            USE_STREAM_OPTIONS = False
            return probe(api, hdr, model, target_tokens, text=text)
        sys.exit(f"POST /chat/completions -> HTTP {r.status_code}:\n{r.text[:300]}")
    first, body = None, b""
    for chunk in r.iter_content(chunk_size=None):
        if first is None:
            first = time.monotonic() - t0
        body += chunk
    if first is None:
        sys.exit("stream ended with no data")
    m = None
    for m in re.finditer(rb'"prompt_tokens"\s*:\s*(\d+)', body):
        pass
    ptoks = int(m.group(1)) if m else None
    return first, ptoks, text


def fit_quadratic(pts):  # least squares on [1, L, L^2], plain normal equations
    X = [[1.0, L, L * L] for L, _ in pts]
    y = [t for _, t in pts]
    A = [[sum(r[i] * r[j] for r in X) for j in range(3)] for i in range(3)]
    b = [sum(r[i] * t for r, t in zip(X, y)) for i in range(3)]
    for i in range(3):  # gaussian elimination
        p = max(range(i, 3), key=lambda k: abs(A[k][i]))
        A[i], A[p], b[i], b[p] = A[p], A[i], b[p], b[i]
        for k in range(i + 1, 3):
            f = A[k][i] / A[i][i]
            A[k] = [a - f * c for a, c in zip(A[k], A[i])]
            b[k] -= f * b[i]
    c = [0.0] * 3
    for i in (2, 1, 0):
        c[i] = (b[i] - sum(A[i][j] * c[j] for j in range(i + 1, 3))) / A[i][i]
    return c


def main():
    global RATIO
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True,
                    help="base such that {url}/chat/completions is the endpoint")
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--model", help="model name in the JSON body "
                                    "(default: last segment of --url)")
    ap.add_argument("--lengths", default="2048,4096,8192,16384,32768",
                    help="comma-separated target prompt lengths")
    ap.add_argument("--runs", type=int, default=3, help="runs per length (min is kept)")
    ap.add_argument("--pause", type=float, default=5.0,
                    help="idle seconds between runs, so backend counter scrapes "
                         "show flat plateaus between requests")
    ap.add_argument("--params-b", type=float,
                    help="active GEMM params in billions (excl. embed/lm_head)")
    ap.add_argument("--attn-layers", type=int, default=0)
    ap.add_argument("--attn-d", type=int, default=0, help="heads x head_dim")
    ap.add_argument("--peak-tflops", type=float,
                    help="dense per-GPU peak at the served precision")
    ap.add_argument("--gpus", type=int, default=1, help="GPUs serving one replica (TP)")
    args = ap.parse_args()

    hdr = {"Authorization": f"Bearer {args.api_key}"}
    api = args.url.rstrip("/")
    model = args.model or api.rsplit("/", 1)[-1]
    lengths = [int(x) for x in args.lengths.split(",")]
    print(f"model={model}  targets={lengths}  runs/length={args.runs}")

    if args.runs < 1 or args.gpus < 1 or args.pause < 0 \
            or (args.peak_tflops is not None and args.peak_tflops <= 0) \
            or (args.params_b is not None and args.params_b <= 0):
        sys.exit("invalid args: need runs>=1, gpus>=1, pause>=0, "
                 "positive --peak-tflops/--params-b when given")
    if args.params_b and not (args.attn_layers and args.attn_d):
        print("WARNING: --attn-layers/--attn-d not set — the quadratic "
              "attention term is omitted from every FLOP/MFU figure "
              "(~10%+ of a long prefill)", file=sys.stderr)

    if len(lengths) == 1 and args.runs == 1:  # single-shot: one request, no
        L = lengths[0]                        # warmup/probe/fit — pair with
        t_start = time.time()                 # before/after /metrics scrapes
        t, ptoks, _ = probe(api, hdr, model, L)
        est = "" if ptoks else "  (ESTIMATED: no usage in stream, token count is the TARGET)"
        print(f"window {t_start:.2f} -> {time.time():.2f} (unix)  "
              f"ttft {t:.3f}s  prompt_tokens={ptoks}{est}")
        n = ptoks or L
        print(f"client-side bound (incl. proxy overhead): {n / t:,.0f} tok/s{est}")
        if args.params_b and args.peak_tflops:
            flops = 2 * args.params_b * 1e9 * n + 2 * n * n * args.attn_d * args.attn_layers
            print(f"model FLOPs: {flops:.4e}  "
                  f"client-side MFU >= {flops / (t * args.peak_tflops * 1e12 * args.gpus):.1%}{est}  "
                  f"(true MFU = FLOPs / (delta_prefill_time_sum x peak x gpus))")
        return

    if len(lengths) > 1:  # sweep mode only: warmup + calibration + cache probe
        run_probes(api, hdr, model, lengths)
    run_sweep(api, hdr, model, lengths, args)


def run_probes(api, hdr, model, lengths):
    global RATIO
    # warmup (discarded) + tokens-per-word calibration from usage
    _, ptoks, text = probe(api, hdr, model, 1024)
    if ptoks:
        RATIO *= ptoks / max(1024, 1)
        print(f"warmup: 1024-target prompt counted {ptoks} tokens "
              f"-> ratio recalibrated to {RATIO:.2f} tok/word-unit")
    else:
        print("warmup: no usage in stream — token counts are TARGETS, not measured")

    # cache probe: identical prompt twice — a big drop means prefix caching is ON
    t1, _, text = probe(api, hdr, model, max(lengths))
    t2, _, _ = probe(api, hdr, model, max(lengths), text=text)
    print(f"cache probe @ {max(lengths)}: cold {t1:.3f}s, repeat {t2:.3f}s"
          + ("  << prefix caching ACTIVE; sweep stays valid (fresh random text)"
             if t2 < 0.5 * t1 else ""))


def run_sweep(api, hdr, model, lengths, args):
    best = []
    for L in lengths:
        runs = []
        for i in range(args.runs):
            time.sleep(args.pause)
            t_start = time.time()
            t, ptoks, _ = probe(api, hdr, model, L)
            runs.append((t, ptoks or L))
            print(f"    run target={L} #{i}: window {t_start:.2f} -> "
                  f"{time.time():.2f} (unix)  ttft {t:.3f}s  prompt_tokens={ptoks}")
        t, n = min(runs)
        best.append((n, t))
        print(f"  target={L:>6}: min {t:.3f}s @ {n} tok  "
              f"median {statistics.median(r[0] for r in runs):.3f}s")

    if len({n for n, _ in best}) >= 3:
        c0, _c1, _c2 = fit_quadratic(best)
        print(f"fit: TTFT = {c0 * 1e3:.0f} ms + {_c1 * 1e6:.2f} us/tok"
              f" + {_c2 * 1e12:.3f} ps/tok^2   (overhead c0 stripped below)")
    else:
        c0 = 0.0
        print("(<3 lengths: no overhead fit — figures below include proxy "
              "overhead, i.e. lower bounds; use engine-side deltas for truth)")

    if not (args.params_b and args.peak_tflops):
        print("no --params-b/--peak-tflops given: reporting throughput only")
        for n, t in best:
            print(f"  L={n:>6}: {n / max(t - c0, 1e-9):,.0f} tok/s prefill")
        return
    peak_group = args.peak_tflops * 1e12 * args.gpus
    for n, t in best:
        flops = 2 * args.params_b * 1e9 * n + 2 * n * n * args.attn_d * args.attn_layers
        mfu = flops / (max(t - c0, 1e-9) * peak_group)
        print(f"  L={n:>6}: {n / max(t - c0, 1e-9):>9,.0f} tok/s  MFU={mfu:.1%}")


if __name__ == "__main__":
    main()
