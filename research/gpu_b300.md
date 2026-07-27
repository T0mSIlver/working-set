# NVIDIA B300 (Blackwell Ultra) — hardware constants note

**Purpose:** defensible `GPU` constants for adding the **B300** to
`scripts/scenario_model.py` and the explorer, alongside the study's calibrated
H200 baseline.

> **Egress note (2026-07-27):** first written with every NVIDIA domain
> blocked at the proxy, from search snippets + secondary coverage.
>
> **Re-verified same day after the block lifted, from primary sources:** the
> GB300 NVL72 page ("576 TB/s"/72 = 8.0 TB/s exactly; 1,080/72 = 15 PFLOPS
> dense FP4), the "Inside Blackwell Ultra" blog ("288 GB of HBM3e per GPU",
> "8 TB/s per GPU", eight 12-Hi stacks, NVLink-5 1.8 TB/s, up to 1,400 W),
> and — the load-bearing find — a **real B300 nvidia-smi dump** (Oracle OCI
> quickstart, shape BM.GPU.B300.8, driver 590.48.01): **275,040 MiB =
> 288.4e9 B usable per GPU**. One premise did not reproduce: both DGX and
> HGX B300 pages now state "2.1 TB" total (no "2.3 TB" decimal variant), so
> the old two-page reconciliation is retired — the per-GPU figure now rests
> on the direct measurement instead. Also confirmed: thundergolfer.com's
> H200 claim (vendor "141 GB" → **150.75e9 B actual**, +6.9%). Consequence:
> the § 3 reserve-transfer sensitivity became the measured central case —
> see below. Minor: the HGX B300 form is 108/8 = 13.5 PFLOPS dense FP4/GPU;
> 15 belongs to the GB300 form.

## 1. Constants

| Constant | Value | Confidence | Basis |
|---|---|---|---|
| HBM capacity | **288 GB HBM3e** / GPU (8 × 12-high stacks) | VERIFIED (primary) | NVIDIA "Inside Blackwell Ultra" blog, read directly 2026-07-27 |
| Usable bytes | **288.4e9 B** (275,040 MiB = 268.6 GiB) — nominal decimal bytes, NO Hopper-style ~7% over-provision | VERIFIED (measured) | Real nvidia-smi dump, Oracle OCI BM.GPU.B300.8 quickstart README |
| Memory bandwidth | **8.0 TB/s** (8.0e12 B/s) | HIGH | GB300 NVL72 "576 TB/s"/72 = HGX B300 "64 TB/s"/8 = 8.0 exactly |
| FP4 tensor cores | **Native NVFP4** (5th-gen tensor cores, ~15 PFLOPS dense/GPU GB300 form) | HIGH (support) / MEDIUM (PFLOPS) | NVIDIA Blackwell Ultra materials; 72 × 15 ≈ the published "1.1 EF dense NVFP4" for NVL72 |
| FP8 | Supported (≈ half the FP4 rate) | HIGH | Blackwell tensor-core path |
| NVLink | 5th gen, 1.8 TB/s/GPU; domain **8** (HGX/DGX B300) or **72** (GB300 NVL72) | HIGH | NVIDIA GB300 NVL72 page (snippet), CoreWeave, Introl |
| TDP | 1,400 W | HIGH | multiple secondary |
| vs B200 | +50% HBM (288 vs 192 GB), same 8 TB/s, 1.5× dense FP4 | HIGH | Tom's Hardware headline claim |

Model constants adopted:

```python
GPUS["B300"] = GPU("B300", vram=288.4e9, hbm_bw=8.0e12, supports_nvfp4=True,
                   reserve_extra=9.75e9)
# vram: measured usable bytes (Oracle nvidia-smi dump). reserve_extra: the
# H200's measured hidden HBM margin (150.75e9 actual vs the 141e9 the
# calibration uses) — the solved reserve silently absorbs it on the H200, so
# a part WITHOUT the over-provision must add it back (see § 3.2).
```

## 2. What the B300 changes for this study

**Capacity ×2.04, bandwidth ×1.67 vs H200** (288/141, 8.0/4.8). Relative to
B200 the B300 is a *capacity* upgrade at unchanged bandwidth — i.e. exactly
the axis this study's warm-session metric prices. It is also the **only part
in the study allowed to run NVFP4 weights** (native FP4 tensor cores; the
Hopper fallback path is excluded — see `research/nvfp4.md` § 2). NVLink
domains: TP ≤ 8 stays inside an HGX B300 node, and a GB300 NVL72 rack keeps
up to TP/EP 72 on-fabric — the study's "N > 2 TP is a projection" caveat
still applies to the *bandwidth haircut*, but the fabric no longer caps the
domain at 8 the way single-node H200 systems do.

## 3. Assumptions carried into the model (flagged)

1. **Reserve transfer.** The per-GPU activation/workspace reserve
   (≈ 18.0 GiB) is *solved* from the H200 anchor and applied to the B300, now
   **plus the measured `reserve_extra` correction** (item 2). Activation/
   workspace scale mostly with model and batch, not VRAM, so the transfer
   itself remains plausible; a vLLM startup log on a B300 (as was captured
   for 2×H200 TP2) would still be a welcome end-to-end check.
2. **Unit-convention mismatch across generations — RESOLVED BY MEASUREMENT
   (2026-07-27), promoted from sensitivity to central.** Both sides are now
   pinned: thundergolfer.com (read in full) documents the H200 delivering
   **150.75e9 B** usable against its 141e9 vendor figure (+6.9%), and the
   Oracle OCI dump shows the B300 delivering **288.4e9 B** — nominal, no
   over-provision. The calibration keeps the H200 at 141e9 (its numbers are
   exact by construction — the hidden 9.75e9 lives inside the solved
   reserve), and the B300 adds `reserve_extra = 9.75e9` per GPU so the
   phantom margin does not transfer. Effect vs the uncorrected model:
   −9.35e9 B of pool per B300 (−4% for the 35B-A3B on 1×B300, −12% for
   GLM-5.2 FP8 on 4×B300); `tables.py` prints central vs uncorrected.
3. **B300 FLOPS are not modelled.** The decode model is an HBM roofline;
   FP4 compute throughput only matters if compute becomes the binding
   constraint, which the roofline model cannot see (limitation carried from
   the baseline).

## Sources

- **Oracle OCI B300 quickstart README — the real BM.GPU.B300.8 nvidia-smi
  dump (275,040 MiB/GPU, driver 590.48.01, power cap 1100 W on the
  air-cooled SXM6 AC variant):**
  https://github.com/oracle-quickstart/oci-gpu-quickstarts/blob/main/nvidia/B300/README-B300.md
- NVIDIA (canonical; read directly 2026-07-27 after the block lifted): GB300 NVL72 —
  https://www.nvidia.com/en-us/data-center/gb300-nvl72/ · DGX B300 —
  https://www.nvidia.com/en-us/data-center/dgx-b300/ · "Inside NVIDIA
  Blackwell Ultra" — https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/
- Tom's Hardware, "Nvidia announces Blackwell Ultra B300 — 1.5× faster than
  B200 with 288GB HBM3e and 15 PFLOPS dense FP4":
  https://www.tomshardware.com/pc-components/gpus/nvidia-announces-blackwell-ultra-b300-1-5x-faster-than-b200-with-288gb-hbm3e-and-15-pflops-dense-fp4
- The Register on Blackwell Ultra: https://www.theregister.com/2025/03/18/nvidia_blackwell_ultra/
- Jonathon Belotti, "Why does an NVIDIA H100 80GB card offer 85.52 GB?"
  (the Hopper unit-convention analysis): https://thundergolfer.com/blog/nvidia-gpu-memory-capacity
- OEM datasheet mirrors: Lenovo Press GB300 NVL72 —
  https://lenovopress.lenovo.com/datasheet/ds0207-lenovo-nvidia-gb300-nvl72 ·
  Supermicro GB300 NVL72 — https://www.supermicro.com/datasheet/datasheet_SuperCluster_GB300_NVL72.pdf ·
  PNY DGX B300 — https://www.pny.com/en-eu/file%20library/professional/datasheet/dgx/dgx-scale-ai-infrastructure-datasheet-gtc25-dgx-b300-pny-webonly.pdf
- Deployment corroboration: CoreWeave HGX B300 —
  https://www.coreweave.com/blog/engineered-for-agentic-ai-nvidia-hgx-b300-on-coreweave-cloud ·
  AWS P6-B300 GA — https://aws.amazon.com/about-aws/whats-new/2025/11/amazon-ec2-p6-b300-instances-nvidia-blackwell-ultra-gpus-available
