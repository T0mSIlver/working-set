# What vLLM holds back from the KV pool: three terms

workingset prices the KV pool as `tp x vram - weights - tp x reserve`, with
one per-GPU reserve: `ACT_RESERVE` (about 18.0 GiB), solved on the H200 from
the projected 1-GPU anchor `BASELINE_POOL_TOKENS_27B_1GPU`, plus
`GPU.reserve_extra` on the B300. This note splits that reserve into the three
terms vLLM's own start-up profile computes, says what vLLM 0.33's code does
with each, and what start-up logs show. `ws calibrate
--vllm-log` reads the terms (`docs/decode-sweep.md`); `[calibration]` can
carry them.

Verdict: the logs confirm vLLM's arithmetic to the rounding of its log
lines and show one error, the B300 transfer leaving out about 11e9 B per GPU
of utilisation holdback; the B300's `reserve_extra` is raised to 20.76e9 B
for it (Tom's pick on 2026-11-18). The rest of the default reserve stays:
how the terms move with TP is unmeasured, and the single reserve also
absorbs two things the terms do not describe (below). The logs that would
settle it are listed at the end.

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
catalogue's `vram`. On the H200 the two differ: the catalogue keeps the
vendor's 141e9 B, while the part delivers about 150.75e9 B (thundergolfer.com,
`research/gpu_b300.md`). On the B300 the catalogue's 288.4e9 B is the
nvidia-smi figure (Oracle OCI dump). CUDA's total sits a little under
nvidia-smi's. Against the catalogue's `vram` the model's
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
  cuBLAS: 1.5 to 2.5 GiB per GPU in the logs read) therefore comes out of
  the holdback and does not shrink the pool. llm-d's
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

## What start-up logs show

Start-up logs from H200 and B300 servers on vLLM 0.29 to 0.33, read with
`ws calibrate --vllm-log`. Rounded, per GPU:

- `requested - consumed - activation` equals vLLM's "Available KV cache
  memory" to 0.01 GiB in every log that prints the breakdown, so the formula
  above is the one the servers ran.
- Activation is a few GiB (2.5 to 6) until `max_num_seqs` is large; with
  `max_num_seqs` in the hundreds the CUDA graph estimate alone reached about
  35 GiB per GPU. llm-d's "0.8 to 3.9 GiB by architecture" holds only for
  small graph sets.
- Non-torch is about 1 GiB, and about 6 GiB on one large MoE with a
  speculative drafter (its drafter and expert-parallel buffers are
  candidates; not traced).
- At a utilisation near 0.9 the holdback is the largest term on the B300,
  over 20 GiB.

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
   the pool by `concurrency x` those fixed bytes. On a hybrid model at a
   long `max_model_len` vLLM charged about 1 GiB per request beyond its
   tokens, of which the model's own recurrent state is under 0.1 GiB; on a
   model without recurrent layers, almost nothing. A reserve solved from N
   (`reserve_gib`, today's `fit_reserve`, and the H200 anchor's projection)
   folds `concurrency x fixed / tp` into bytes per GPU, which reached about
   16 GiB per GPU, and that amount changes with `max_model_len`. The model charges the state
   per session instead (`docs/scenarios.md`, limitation 13). The terms are
   read from byte lines only, so they do not carry it; on a hybrid model
   the pool they give is larger, in tokens, than N. `ws calibrate` prints
   the per-request figure and how much of the fitted reserve it is.
   (Pointed out by a reading of BLIS, `inference-sim`, which models it;
   checked here against vLLM 0.33's code.)
2. **Weights as loaded.** vLLM reported 7 to 14% more weight memory per GPU
   than `hbm_weights / tp`, and `ws calibrate`
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

Until 2026-11-18 the B300 reserve was the H200's 18.0 GiB plus
`reserve_extra = 9.75e9` B (9.1 GiB), the H200's hidden over-provision added
back. The terms show what
that transfer leaves out: the part of the reserve that depends on the GPU is
`vram - u x gpu_total`. With the nvidia-smi totals above and u = 0.92 (an
assumption: the model has no utilisation input; vLLM's default is 0.90):

- H200: 141e9 - 0.92 x 150.75e9 = 2.31e9 B
- B300: 288.4e9 - 0.92 x 288.4e9 = 23.07e9 B

a difference of 20.76e9 B per GPU (23.51e9 at 0.90), against the 9.75e9 the
patch adds. Activation and non_torch transferring unchanged (the patch's own
assumption), the B300 reserve is about 11e9 B per GPU short, because the
holdback is a share of a memory twice the size. `research/gpu_b300.md` §3
item 1 assumes the reserve "scales mostly with model and batch, not VRAM";
the holdback is the part that does not. A B300 log points the same way.

`reserve_extra` is now that difference, 20.76e9 B per GPU: the old patch's
9.75e9 of H200 over-provision plus 11.01e9 of extra holdback,
`0.08 x (288.4e9 - 150.75e9)`. Every B300 pool loses 11.01e9 B per GPU
against the old patch. A deployment run at another utilisation sets its own
reserve with `ws calibrate --vllm-log --reserve terms`.

## TP8

How the terms move with TP is unmeasured on vLLM 0.33. From the code: the
holdback does not depend on TP; activation is paid per GPU and, in llm-d's
data, does not shrink with TP (Qwen3-32B, TP1 and TP2); non_torch past the
snapshot is about 1 GiB at the width the logs read ran. If those hold at TP8, the three-term reserve of a TP8
group is `8 x (holdback + activation + non_torch)`, and the current model's
`8 x 18.0 GiB` on the H200 overstates it for non-hybrid models whose
activation is a few GiB. Whether non_torch or activation grows with the
width up to TP8 (custom all-reduce buffers are registered during graph capture,
expert-parallel all-to-all buffers grow with the group) is exactly what a TP8
log would show.

## Recommendation

- Keep `ACT_RESERVE` as it is until the logs below exist; the B300's
  `reserve_extra` carries the holdback at 0.92 from 2026-11-18.
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

1. Qwen3.8-27B FP8 on H200 at TP1 and at TP8 (a middle width too is a
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
