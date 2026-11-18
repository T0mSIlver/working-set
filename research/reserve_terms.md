# What vLLM holds back from the KV pool: three terms

workingset prices the KV pool as `tp x vram - weights - tp x reserve`, with
one per-GPU reserve: `ACT_RESERVE` (about 18.0 GiB), solved on the H200 from
the projected 1-GPU anchor `BASELINE_POOL_TOKENS_27B_1GPU`, plus
`GPU.reserve_extra` on the B300. This note splits that reserve into the three
terms vLLM's own start-up profile computes, says what vLLM 0.33's code does
with each, and what four of our own start-up logs show. `ws calibrate
--vllm-log` reads the terms (`docs/decode-sweep.md`); `[calibration]` can
carry them.

Verdict: keep the current defaults. The logs confirm vLLM's arithmetic to the
rounding of its log lines, and they point at one likely error (the B300
transfer leaves out about 10 GiB per GPU of utilisation holdback), but every
log is TP4, so they cannot say how the terms move with TP, and the single
reserve also absorbs two things the terms do not describe (below). The logs
that would settle it are listed at the end.

## The three terms

Per GPU, in vLLM 0.33 (`vllm/v1/worker/gpu_worker.py`,
`determine_available_memory`, and `vllm/utils/mem_utils.py`,
`memory_profiling`, at tag v0.33.0):

```
requested    = ceil(gpu_total x gpu_memory_utilization)
KV memory    = requested - consumed - activation
consumed     = free at start-up snapshot - free after the profile run
             = weights + non_torch
activation   = (torch peak - torch allocated after the profile run)
               + CUDA graph estimate
```

so the bytes held back from the pool on each GPU are

| term | value | scales with |
|---|---|---|
| holdback | `(1 - gpu_memory_utilization) x gpu_total` | the GPU's memory |
| activation | profile peak + CUDA graph estimate | model, `max_num_batched_tokens`, `max_num_seqs` (graph sizes) |
| non_torch | memory taken outside torch's allocator after the snapshot | backend buffers; see TP below |

`gpu_total` is what CUDA reports (`torch.cuda.mem_get_info`), not the
catalogue's `vram`: about 139.8 GiB on the H200 (the catalogue keeps the
vendor 141e9 B = 131.3 GiB) and about 267.7 GiB on the B300 (catalogue
268.6 GiB from nvidia-smi). Against the catalogue's `vram` the model's
per-GPU reserve is

```
reserve = vram - gpu_memory_utilization x gpu_total + activation + non_torch
```

which is `model.ReserveTerms.reserve()`.

What the code settles:

- **Every worker pays its own activation.** Each worker profiles itself and
  subtracts its own peak; vLLM then sizes every worker's cache to the
  smallest. llm-d-planner's planner charges activation once per replica
  (`activation x dp`), but its own table contradicts that: Qwen3-32B shows
  5.64 GiB per worker at both TP1 and TP2, and its TP2 KV memory per worker
  is `0.9 x 79.18 - 30.59 - 5.64 - 0.54 = 34.49` GiB, which is activation
  charged per GPU. `kv_pool_tokens` already charges the reserve per GPU.
- **NCCL buffers sit in the holdback, not in non_torch.** vLLM 0.33
  initialises the distributed environment and the cuBLAS handles *before*
  its start-up snapshot ("This ensures NCCL buffers are allocated before we
  measure available memory"), and `requested` is a share of the total, not
  of what is free. Memory used before the snapshot (CUDA context, NCCL,
  cuBLAS: 1.6 GiB per GPU in our H200 TP4 log on vLLM 0.29, 2.5 GiB in our B300 TP4 logs on 0.33)
  therefore comes out of the holdback and does not shrink the pool. llm-d's
  "about 2.1 GiB per GPU from TP2 up" was measured on vLLM 0.19 and does not
  carry over.
  BLIS also keeps NCCL and workspace memory outside the utilisation budget,
  and computes activation from `max_num_batched_tokens` and the hidden size;
  its own split misses vLLM by 22% on its one recorded case, so only its
  structure is borrowed here.
- **The CUDA graph estimate is part of activation** since vLLM 0.21
  (`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS`, on by default), and it
  dominates when `max_num_seqs` is large.
- **The breakdown is printed once, after CUDA graph capture,** at INFO:
  `Free memory on device (free/total GiB) on startup. Desired GPU memory
  utilization is (u, requested GiB). Actual usage is X GiB for consumed
  memory (weights + non-torch), Y GiB for peak activation, and Z GiB for
  CUDAGraph memory. ...` Together with `Model loading took X GiB memory`
  it gives all three terms. The `Memory profiling takes ... torch peak memory
  increase` line that llm-d reads is DEBUG-only in 0.33 and no longer
  prints non-torch memory. A server started with `--kv-cache-memory` skips
  the profile and carries no reserve; `ws calibrate` refuses its log.

## What our logs show

Four start-up logs from our own deployments, all TP4 with an FP8 cache.
Per GPU, rounded:

| model | GPU | vLLM | util | holdback | activation (graph estimate) | non_torch |
|---|---|---|---|---|---|---|
| Qwen3.8-27B | H200 | 0.25.1 | 0.92 | 11.2 | not printed | not printed |
| Qwen3.8-27B | H200 | 0.29 | 0.92 | 11.2 | 2.6 (1.3) | 1.2 |
| Qwen3.8-Flash-Next | B300 | 0.33 | 0.92 | 21.4 | 5.7 (4.3) | 1.0 |
| DeepSeek-V4.1-Flash | B300 | 0.33 | 0.80 | 53.5 | 37.2 (34.6) | 6.4 |

(GiB. The 0.25.1 log has no breakdown line; `ws calibrate` reads its pool
only.)

- In the three logs that print it, `requested - consumed - activation`
  equals vLLM's "Available KV cache memory" to 0.01 GiB, so the formula
  above is the one the servers ran.
- Activation is a few GiB until `max_num_seqs` is large: DeepSeek-V4.1-Flash
  at `max_num_seqs` 512 spends 35 GiB per GPU on the CUDA graph estimate
  (the captured graphs took about 21 GiB). llm-d's "0.8 to 3.9 GiB by
  architecture" holds only for small graph sets.
- Non-torch is about 1 GiB at TP4 for the two Qwen models, 6.4 GiB for
  DeepSeek-V4.1-Flash (its speculative drafter and expert-parallel buffers
  are candidates; not traced).
- The holdback is the largest term on the B300: 21.4 GiB at 0.92.

The single solved reserve also absorbs two things the terms do not:

1. **vLLM's token figure is not the pool divided by bytes per token.** In
   vLLM 0.33, `GPU KV cache size: N tokens` is `concurrency x
   max_model_len`, where concurrency is the pool's blocks over the blocks one
   `max_model_len` request needs, summed over every cache group
   (`get_max_concurrency_for_kv_cache_config` in
   `vllm/v1/core/kv_cache_utils.py`; each spec's `max_memory_usage_bytes` in
   `vllm/v1/kv_cache_interface.py`). A recurrent-state group charges its
   fixed blocks per request, a sliding-window group its ring at the
   admission cap, and padding layers ("Add 3 padding layers, may waste at
   most 6.25% KV cache memory") their pages. So N x `kv_bpt` is less than
   the pool by `concurrency x` those fixed bytes. In our logs vLLM charges
   each request about 1.3 GiB beyond its tokens for the 27B at
   `max_model_len` 262,144 (the model's own recurrent state is under 0.1 GiB
   of that), about 1.0 GiB for Flash-Next and almost nothing for
   DeepSeek-V4.1-Flash. A reserve solved from N (`reserve_gib`, today's
   `fit_reserve`, and the H200 anchor's projection) folds `concurrency x
   fixed / tp` into bytes per GPU, about 16 GiB per GPU for the 27B log,
   and that amount changes with `max_model_len`. The model charges the state
   per session instead (`docs/scenarios.md`, limitation 13). The terms are
   read from byte lines only, so they do not carry it; on a hybrid model
   the pool they give is larger, in tokens, than N. `ws calibrate` prints
   the per-request figure and how much of the fitted reserve it is.
   (Pointed out by a reading of BLIS, `inference-sim`, which models it;
   checked here against vLLM 0.33's code.)
2. **Weights as loaded.** vLLM reports 7 to 14% more weight memory per GPU
   than `hbm_weights / tp` for these three models, and `ws calibrate`
   prints both. Part of that may be tensors every rank holds whole (BLIS's
   model graphs mark some `tp: replicated`, for example in
   DeepSeek-V4.1-Flash); a share that does not shrink with TP sits in the
   per-GPU reserve and makes a reserve solved at one TP wrong at another.
   One log cannot separate it: `Model loading took X GiB` at two TP widths
   gives `X(tp) = sharded / tp + replicated`. The catalogue's weights are
   the place to fix it, not a reserve term.

This is why `ws calibrate --vllm-log --write` still stores the single
`reserve_gib` by default (it reproduces the pool vLLM reported), and stores
the terms only with `--reserve terms`.

## The B300 `reserve_extra` patch

The B300 reserve is the H200's 18.0 GiB plus `reserve_extra = 9.75e9` B
(9.1 GiB), the H200's hidden over-provision added back. The terms show what
that transfer leaves out: the part of the reserve that depends on the GPU is
`vram - u x gpu_total`, which is

- H200: 131.3 - 0.92 x 139.8 = 2.7 GiB
- B300: 268.6 - 0.92 x 267.7 = 22.3 GiB

a difference of 19.6 GiB per GPU at utilisation 0.92 (22.2 GiB at vLLM's
default 0.90), against the 9.1 GiB the patch adds. Activation and non_torch
transferring unchanged (the patch's own assumption), the B300 reserve is about
10.5 GiB per GPU short at 0.92, because the holdback is a share of a memory
twice the size. `research/gpu_b300.md` §3 item 1 assumes the reserve "scales
mostly with model and batch, not VRAM"; the holdback is the part that does
not. The B300 Flash-Next log points the same way: with `reserve_extra`
raised to about 21.1e9 B the model's pool comes close to the one vLLM
reported. That is one model on one TP, and the model has no
utilisation input (the 0.92 is our deployments' setting, not vLLM's
default), so it is not enough to change the default here.

## TP8

No log at TP8, nor at TP1 or TP2 on vLLM 0.33. From the code: the holdback
does not depend on TP; activation is paid per GPU and, in llm-d's data, does
not shrink with TP (Qwen3-32B, TP1 and TP2); non_torch past the snapshot is
about 1 GiB at TP4. If those hold at TP8, the three-term reserve of a TP8
group is `8 x (holdback + activation + non_torch)`, and the current model's
`8 x 18.0 GiB` on the H200 overstates it for non-hybrid models whose
activation is a few GiB. Whether non_torch or activation grows from TP4 to
TP8 (custom all-reduce buffers are registered during graph capture,
expert-parallel all-to-all buffers grow with the group) is exactly what a TP8
log would show.

## Recommendation

- Keep `ACT_RESERVE` and `reserve_extra` as they are until the logs below
  exist; the published numbers do not move.
- Use `ws calibrate --vllm-log` on every new deployment: it prints the three
  terms, the reserve they sum to and the default, the weights gap and the
  bytes per token, and `--reserve terms` stores the terms for that
  deployment.
- Once the logs exist, the likely change is: the reserve as the three terms
  with a stated utilisation (0.92), activation and non_torch per model from
  a log where one exists, vLLM's fixed per-request pages for hybrid and
  windowed models, and any replicated weights in the catalogue, replacing
  the solved anchor. A reserve fitted from the token figure (`ws calibrate
  --reserve pool`) should then be fitted from the KV memory in bytes. That is a modelling change
  (Python, then `interactive/src`, then the goldens).

## Logs that would settle it

All with vLLM 0.33, `--gpu-memory-utilization 0.92`, an FP8 cache, the same
`max_num_seqs` and `max_num_batched_tokens` within a set, and the start-up
log kept past CUDA graph capture (the "Free memory on device" line):

1. Qwen3.8-27B FP8 on H200 at TP1 and at TP8 (TP4 again on 0.33 is a
   bonus): the anchor model and GPU, the TP dependence of activation and
   non_torch, and an H200 log on the version the study targets.
2. Qwen3.8-27B FP8 on B300 at any TP: the same model on both GPUs, which
   tests the H200-to-B300 transfer above directly.
3. One catalogue model on B300 at TP8 (Kimi K3.1 Pro or GLM-5.3): TP8 on the
   B300.
4. Optional: one of the above at two `max_num_seqs` values (for example 128
   and 512), to size the CUDA graph part of activation.

## Sources

- vLLM v0.33.0, `vllm/v1/worker/gpu_worker.py` (`init_device`,
  `determine_available_memory`, `compile_or_warm_up_model`),
  `vllm/utils/mem_utils.py` (`memory_profiling`, `MemoryProfilingResult`),
  `vllm/v1/worker/utils.py` (`request_memory`),
  `vllm/v1/core/kv_cache_utils.py` (`update_kv_cache_capacity`),
  https://github.com/vllm-project/vllm/tree/v0.33.0
- llm-d-planner v0.3.3, `docs/empirical-vllm-memory-results.md`,
  `docs/accuracy/vllm-v0.19.0-accuracy-report.md`,
  `src/planner/capacity_planner.py`,
  `src/planner/data/configuration/vllm_memory_constants.json` (H100,
  vLLM 0.11 and 0.19): https://github.com/llm-d-incubation/llm-d-planner
