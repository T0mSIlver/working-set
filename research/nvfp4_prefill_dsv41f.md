# Does NVFP4 speed up DeepSeek-V4.1-Flash prefill? — open question

**Status (2026-11-04):** unmeasured. No public source compares prefill of
`nvidia/DeepSeek-V4.1-Flash-NVFP4` with the native checkpoint, on B300 or any
other GPU. The study keeps pricing prefill at the FP8 rate for both
(`research/prefill.md`, "Weight dtype is NOT priced").

## 1. What is 4-bit in the native checkpoint

Only the routed experts: MXFP4 (E2M1 values, one E8M0 scale per 32 values),
288.8 GB of the 510.29 GB checkpoint. The Engram tables (202.8 GB) and every
other projection are FP8; embeddings, lm_head, gates and compressors are BF16.
Per decoded token, the routed experts are 4.51 GB of the 13.0 GB of weights
read, about 35%. The main KV cache is also FP4, and that one is the trained
format (QAT in post-training, tech report § 2.4.4).
Details: `research/model_dsv41flash.md` § 1 and § 4.

Whether the experts were trained in FP4 or quantized after training is not
documented here: say "ships with MXFP4 experts", not "trained in FP4".

## 2. What the NVFP4 checkpoint changes

Only the routed experts: E8M0 block-32 scales become E4M3 block-16 scales plus
two FP32 globals per projection, +5.9% per expert, +3.3% on the checkpoint
(`research/nvfp4_2026-10.md` § 3). Decode reads more bytes, so decode cannot
gain.

## 3. Why prefill might gain (inference, unmeasured)

- The native experts run in vLLM through DeepGEMM's fused MoE kernel
  (`deep_gemm_mega_moe`), which secondary sources describe as FP8 activations
  × FP4 weights. On Blackwell, as far as we know, an FP8×FP4 multiply runs at
  the FP8 rate (B300: 4.5 PFLOPS dense, `research/prefill.md`).
- NVIDIA's ModelOpt NVFP4 checkpoints usually quantize activations to FP4 as
  well (W4A4), which would put the expert multiplies on the dense-FP4 rate
  (B300: 13.5 PFLOPS). Ceiling: up to 3× on the **expert share** of prefill
  only.
- Against it: attention, shared experts and the Engram projections stay
  FP8/BF16; at long context attention dominates prefill. For the earlier
  V4-Flash, vLLM's recipe says NVFP4 experts cannot use the fused MegaMoE
  kernel and fall back to the default MoE backend, losing its compute/NVLink
  overlap.

To verify before quoting: DeepGEMM's own repo for the MegaMoE formats
(PR #304), and the per-format B300 tensor-core rates.

## 4. To settle it

Run both checkpoints on B300 with the same vLLM image, prompt lengths and
batch settings; report prefill tokens/s and TTFT at several context lengths.

## 5. Sources (read 2026-11-04)

vLLM's blog and recipes are the most useful source on this model's serving.

- [vLLM blog: DeepSeek-V4.1-Flash, 5× agentic throughput since day 0](https://vllm.ai/blog/2026-10-07-deepseek-v41-flash)
  (2026-10-07). Prefill gains that are **not** from NVFP4 weights: sparse MQA
  logits make prefill 1.43× faster at 512K and 2× at 1M on 4×GB300; bounded
  replay of the sliding-window cache cuts prefill compute 30–40% (GB200);
  MegaAttention reads an NVFP4 compressed **KV** format, 45% smaller than FP8.
- [vLLM recipe: DeepSeek-V4.1-Flash](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash):
  "Routed expert weights are MXFP4; everything else is MXFP8 block-quantized,
  UE8M0 scales throughout." Names `deep_gemm_mega_moe` for data + expert
  parallel; no NVFP4 variant.
- [vLLM recipe: DeepSeek-V4-Flash, NVFP4 variant](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash?variant=nvfp4):
  NVFP4 experts do not support `deep_gemm_mega_moe` (FP8-only).
- [nvidia/DeepSeek-V4.1-Flash-NVFP4](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-NVFP4):
  accuracy against the MXFP4 checkpoint only; no throughput or TTFT.
- [DataNorth on the NVIDIA release](https://datanorth.ai/news/nvidia-releases-deepseek-v4-1-flash-nvfp4):
  notes the missing speed figures.
- Community repacks, no speed numbers:
  [LibertAIDAI](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-NVFP4),
  [AxionML](https://huggingface.co/AxionML/DeepSeek-V4.1-Flash-NVFP4),
  [sekkit](https://huggingface.co/sekkit/DeepSeek-V4.1-Flash-NVFP4) (a
  2026-10-14 update stores Engram tables as MXFP4).
- [NVIDIA forum thread](https://forums.developer.nvidia.com/t/nvidia-just-released-nvfp4-quant-for-deepseek-v4-1-flash/383539):
  MXFP4 on DGX Spark only, no NVFP4 comparison.
- MegaMoE formats, secondary:
  [DeepGEMM guide](https://agentpedia.codes/blog/deepgemm-guide),
  [besthub on Mega MoE and the FP4 indexer](https://www.besthub.dev/articles/deepseek-introduces-mega-moe-and-fp4-indexer-inside-the-new-gpu-fusion-kernel-18ea60795f7b).
