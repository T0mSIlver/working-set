# NVIDIA B300 (Blackwell Ultra) — hardware constants note

**Purpose:** defensible `GPU` constants for adding the **B300** to
`scripts/scenario_model.py` and the explorer, alongside the study's calibrated
H200 baseline.

> **Egress note (2026-07-27):** every NVIDIA-owned domain 403s at this
> environment's proxy gateway, so no figure below was read from a primary
> NVIDIA datasheet. Values come from search-index snippets of NVIDIA pages
> plus reputable secondary coverage (Tom's Hardware, The Register, OEM
> datasheet mirrors), cross-checked for internal consistency (per-GPU ×
> node-count = published aggregates). Confidence is tiered per row.

## 1. Constants

| Constant | Value | Confidence | Basis |
|---|---|---|---|
| HBM capacity | **288 GB HBM3e** / GPU (8 × 12-high stacks) | HIGH | NVIDIA "Inside Blackwell Ultra" blog (snippet), Tom's Hardware, The Register |
| Usable bytes | **≈ 288e9 B** (= 268.2 GiB) | MEDIUM | Reconciles NVIDIA's own HGX B300 "2.3 TB" (decimal) and DGX B300 "2.1 TB" (binary TiB) 8-GPU aggregates to one per-GPU figure |
| Memory bandwidth | **8.0 TB/s** (8.0e12 B/s) | HIGH | GB300 NVL72 "576 TB/s"/72 = HGX B300 "64 TB/s"/8 = 8.0 exactly |
| FP4 tensor cores | **Native NVFP4** (5th-gen tensor cores, ~15 PFLOPS dense/GPU GB300 form) | HIGH (support) / MEDIUM (PFLOPS) | NVIDIA Blackwell Ultra materials; 72 × 15 ≈ the published "1.1 EF dense NVFP4" for NVL72 |
| FP8 | Supported (≈ half the FP4 rate) | HIGH | Blackwell tensor-core path |
| NVLink | 5th gen, 1.8 TB/s/GPU; domain **8** (HGX/DGX B300) or **72** (GB300 NVL72) | HIGH | NVIDIA GB300 NVL72 page (snippet), CoreWeave, Introl |
| TDP | 1,400 W | HIGH | multiple secondary |
| vs B200 | +50% HBM (288 vs 192 GB), same 8 TB/s, 1.5× dense FP4 | HIGH | Tom's Hardware headline claim |

Model constants adopted:

```python
GPUS["B300"] = GPU("B300", vram=288e9, hbm_bw=8.0e12, supports_nvfp4=True)
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
   (≈ 18.0 GiB) is *solved* from the H200 anchor and applied to the B300
   unchanged. Activation/workspace scale mostly with model and batch, not
   VRAM, so this is plausible — but no B300 measurement anchors it. One vLLM
   startup log on a B300 (as was captured for 2×H200 TP2) would pin it.
2. **Unit-convention mismatch across generations.** For Hopper, vendor "GB"
   *understates* usable bytes (~7%, documented for H100: 80 GB advertised,
   85.5e9 usable — thundergolfer.com; the same source claims the H200
   similarly delivers ~7% over 141e9). The study keeps the H200 at its
   141e9-vendor convention — the calibration absorbs any excess into the
   solved reserve, so H200 numbers are unaffected. But the *transferred*
   reserve then under-counts true per-GPU overhead by the same hidden ~10e9 B,
   which **overstates B300 pools by ~10 GB per GPU if the Hopper
   over-provision claim is right** — ~4% of the pool for the 35B-A3B on
   1×B300, up to ~12% for GLM-5.2 on 4×B300 (the fraction grows with the
   weight share of VRAM). Carried as a B300 sensitivity
   (`tables.py`), resolvable only by a real B300 memory dump — no genuine
   `nvidia-smi` reading from a B300 was locatable.
3. **B300 FLOPS are not modelled.** The decode model is an HBM roofline;
   FP4 compute throughput only matters if compute becomes the binding
   constraint, which the roofline model cannot see (limitation carried from
   the baseline).

## Sources

- NVIDIA (canonical, blocked at gateway — cited via snippets): GB300 NVL72 —
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
