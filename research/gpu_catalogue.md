# Nvidia datacenter GPU catalogue for the model (2026-11-04)

The model prices two parts today, H200 and B300 (`GPUS` in
`src/workingset/model.py`, mirrored in `interactive/src/config.js`). This note
collects every field a `GPU` row needs for the rest of Nvidia's current
inference lineup, then says what the model must change before rack-scale
NVLink domains and PCIe cards can be priced honestly.

Conventions:
- **P** primary (nvidia.com product page, datasheet or Nvidia technical blog),
  **S** secondary, **calc** our arithmetic. All read 2026-11-04 unless dated.
  The Nvidia pages were read through a fetch-and-summarise tool, not the raw
  datasheet PDFs: re-read the PDF before a figure becomes a model constant.
- **Dense** throughput. Nvidia heads its tables with 2:4-sparse figures, which
  no LLM GEMM reaches; Hopper pages footnote "with sparsity", Blackwell pages
  show FP4 as "sparse | dense" and every other precision sparse only. Dense =
  sparse / 2 wherever Nvidia gives sparse only.
- Per GPU unless stated. DGX/HGX memory totals are in binary units (DGX B300
  "2.1 TB" = 8 x 268.6 GiB, the measured 288.4e9 B per GPU of
  `research/gpu_b300.md`), so "1,440 GB" on DGX B200 is 8 x 180 GiB = 193e9 B.

## 1. Spec rows

### Hopper (FP8 tensor cores, no FP4)

| | H100 SXM | H100 NVL | H200 SXM (in model) | H200 NVL | H20 |
|---|---|---|---|---|---|
| Memory | 80 GB HBM3 | 94 GB HBM3 | 141 GB HBM3e | 141 GB HBM3e | 96 GB HBM3 (S) |
| Bandwidth | 3.35 TB/s | 3.9 TB/s | 4.8 TB/s | 4.8 TB/s | 4.0 TB/s (S) |
| FP8 dense | 1,979 TF | 1,671 TF | 1,979 TF | 1,671 TF | 296 TF (S) |
| BF16 dense | 989 TF | 835 TF | 989 TF | 835 TF | 148 TF (S) |
| FP4 | no | no | no | no | no |
| NVLink | gen 4, 900 GB/s, domain 8 | 600 GB/s bridge, domain 2 | gen 4, 900 GB/s, domain 8 | 900 GB/s bridge, domain 2 or 4 | conflict: 900 GB/s HGX boards reported (S) vs "PCIe only" (getdeploying) |
| Scale-out | ConnectX-7 400G per GPU (DGX) | server choice | ConnectX-7 400G | server choice | – |
| TDP | 700 W | 350-400 W | 700 W | 600 W | conflict: 350 / 400 / 500 W (S) |
| Form | SXM5 | PCIe dual-slot | SXM5 | PCIe dual-slot | SXM (S) |

Sources: [H100](https://www.nvidia.com/en-us/data-center/h100/) (P; "188 GB" is
the NVL pair), [H200](https://www.nvidia.com/en-us/data-center/h200/) (P).
H20 has no Nvidia datasheet: [getdeploying](https://getdeploying.com/gpus/nvidia-h20),
[flopper.io](https://flopper.io/gpu/nvidia-h20-96gb/spec-sheet) (S). No H20E
confirmed; the B30A China part has no published specs
([Tom's Hardware](https://www.tomshardware.com/tech-industry/artificial-intelligence/nvidia-responds-to-reports-that-its-h20-gpu-for-china-is-ending-production-next-gen-b30a-green-light-up-to-the-united-states-government-according-to-ceo-jensen-huang), S).

### Blackwell and Blackwell Ultra (FP8 and NVFP4 tensor cores)

| | HGX B200 | HGX B300 (in model) | GB200 NVL72 | GB300 NVL72 |
|---|---|---|---|---|
| Memory | 192 GB HBM3e; DGX 1,440 GB / 8 = 180 GiB | 288 GB HBM3e; measured 288.4e9 B | 186 GB per GPU (372 GB per superchip) | 288 GB HBM3e; rack 20 TB / 72 |
| Bandwidth | 8 TB/s | 8 TB/s | 8 TB/s (576 TB/s rack) | 8 TB/s (576 TB/s rack) |
| FP4 dense | 9 PF (HGX 72 PF / 8) | 13.5 PF (HGX 108 PF / 8) | 10 PF (rack 720 PF) | 15 PF (rack 1,080 PF) |
| FP8 dense | 4.5 PF (calc) | 4.5 PF (calc) | 5 PF (calc, rack 720 PF sparse) | 5 PF (calc) |
| BF16 dense | 2.25 PF (calc) | 2.25 PF (calc) | 2.5 PF (calc) | 2.5 PF (calc) |
| NVLink | gen 5, 1.8 TB/s, domain 8 | gen 5, 1.8 TB/s, domain 8 | gen 5, 1.8 TB/s, domain 72 (130 TB/s rack) | same, domain 72 |
| Scale-out | ConnectX-7 400G per GPU | ConnectX-8 800G per GPU | ConnectX-7 400G (S) | ConnectX-8 800G (S) |
| TDP | ~1,000 W (S) | up to 1,400 W (P); 1,100 W SXM6 AC air part (S) | ~1,200 W (P, blog) | up to 1,400 W (P, blog) |
| System power | DGX B200 ~14.3 kW (P) | DGX B300 ~14 kW (P) | rack ~120-132 kW (S) | rack 132 kW nominal, ~155 kW peak (HPE, S) |
| Cooling | air or liquid | air (AC) or liquid | liquid | liquid |

Sources: [HGX](https://www.nvidia.com/en-us/data-center/hgx/),
[DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/),
[DGX B300](https://www.nvidia.com/en-us/data-center/dgx-b300/),
[GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/),
[GB300 NVL72](https://www.nvidia.com/en-us/data-center/gb300-nvl72/),
[Inside Blackwell Ultra](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/) (all P);
rack power [HPE](https://www.hpe.com/jp/ja/collaterals/collateral.a50009244enw.html),
[iConnect007/TrendForce](https://iconnect007.com/article/144511/nvidia-gb300-to-feature-enhanced-specifications-full-rack-shipments-expected-to-gradually-scale-in-3q25/144508/design) (S).

Two traps, both already in `research/gpu_b300.md` for the B300:
- Blackwell Ultra raised **dense FP4 only** (10 to 15 PF per chip). FP8 and
  BF16 did not move, and sparse FP4 stays 20 PF, so the sparse column hides
  the change.
- The HGX parts run below the Grace-Blackwell parts' power: per-GPU FP4 is
  9 / 13.5 PF in HGX, 10 / 15 PF in NVL72; FP8 4.5 vs 5 PF.

### Rubin (shipping since August 2026)

Nvidia's FY27 Q2 call said production shipments began in August 2026, with
racks at CoreWeave, Google Cloud, Azure, OCI and Nebius
([Tom's Hardware](https://www.tomshardware.com/tech-industry/artificial-intelligence/nvidia-expects-to-sell-usd20-billion-worth-of-vera-rubin-hardware-this-quarter-would-account-for-20-percent-of-data-center-revenue-its-fastest-ramp-in-company-history), S;
the [product page](https://www.nvidia.com/en-us/data-center/vera-rubin-nvl72/)
says "Available Now", P). The 2025 roadmap's "NVL144" counted dies; the
product is **Vera Rubin NVL72** (72 GPU packages, 36 Vera CPUs).

| Rubin GPU, per GPU | Value | Source |
|---|---|---|
| Memory | 288 GB HBM4 | P |
| Bandwidth | **conflict**: 19.2 TB/s (product page, 1,400 TB/s rack / 72) vs "up to 22 TB/s" ([Rubin platform blog](https://developer.nvidia.com/blog/inside-the-nvidia-rubin-platform-six-new-chips-one-ai-supercomputer/)) | P, both |
| NVFP4 | 35 PF dense ("training"); the 50 PF "inference" figure is Transformer Engine compute, not plain dense | P |
| FP8 dense | 17.5 PF | P |
| BF16 | not published | – |
| NVLink | gen 6, domain 72; **conflict** 3.0 TB/s (216 TB/s rack, page) vs 3.6 TB/s (260 TB/s, blog) | P, both |
| Scale-out | ConnectX-9, 1.6 Tb/s per GPU | P (blog) |
| TDP, rack power | not published | – |

Rubin CPX (128 GB GDDR7, 30 PF NVFP4, announced September 2025) has no
confirmed shipping product as of today
([Futurum](https://futurumgroup.com/insights/nvidias-new-rubin-cpx-targets-future-of-large-scale-inference/), S).

### PCIe cards without NVLink

| | RTX PRO 6000 Blackwell Server Edition | L40S |
|---|---|---|
| Memory | 96 GB GDDR7 ECC | 48 GB GDDR6 ECC |
| Bandwidth | 1.597 TB/s | 0.864 TB/s |
| FP4 dense | ~2 PF (page lists 4 PF with no sparsity footnote; halved by Blackwell convention, unverified) | no |
| FP8 dense | ~1 PF (page 2 PF, same caveat) | 733 TF (page lists dense and sparse) |
| BF16 dense | ~0.5 PF (page 1 PF, same caveat) | 362 TF |
| Interconnect | PCIe Gen5, no NVLink | PCIe Gen4, no NVLink |
| TDP | up to 600 W | 350 W |

Sources: [RTX PRO 6000 Server](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/),
[L40S](https://www.nvidia.com/en-us/data-center/l40s/) (P). The RTX PRO 6000
is SM120 (GB202), not the SM100 of B200/B300: it has FP4 tensor cores but not
the datacenter Blackwell kernel set (see §3.3).

## 2. Rental prices, USD per GPU-hour, on demand

Band = low / median / high across providers. Aggregator figures are
getdeploying.com's pages (`/reference/cloud-gpu/nvidia-<part>`), updated
2026-11-04; first-party pages were read the same day.

| Part | Low | Median | High | Providers | Neocloud band |
|---|---|---|---|---|---|
| H100 SXM | 1.80 | 3.63 | 11.07 (Google) | 51 | 2.5-4.5 |
| H200 SXM | 2.09 (stock unverified) | 4.79 | 13.78 (Azure) | 42 | 3.6-6.3 |
| H20 | – | – | – | 0 tracked | – |
| B200 | 3.75 | 8.01 (6.47 one price per provider) | 14.24 (AWS) | 30 | 6.0-8.6 |
| B300 | 5.58 (8.70 lowest in stock) | 9.50 | 17.80 (AWS) | 16 | 7.0-10.0 |
| GB200 NVL72 | 10.50 (CoreWeave) | 13.25 | 27.04 (Azure) | 4 | 10.5 |
| GB300 NVL72 | 13.30 (out of stock) | – (2 providers) | 18.00 (OCI) | 2 | – |
| RTX PRO 6000 | 0.66 | 2.31 | 6.00 | 44 | 1.7-2.5 |
| L40S | 0.38 | 1.64 | 3.51 | 38 | 0.8-2.3 |
| Rubin | no public rate; CoreWeave early access since 2026-09-30 | | | | |

The highs are hyperscaler list prices (AWS, Azure, Google, OCI); the lows are
marketplace listings whose stock is often unverified. The market moved since
the model's prices were read (2026-09-03: H200 median $4.40, B300 $7.89): H200
+9%, B300 +20%.

First-party on-demand rates read 2026-11-04:

| Provider | H100 | H200 | B200 | B300 | GB200 | RTX PRO 6000 | L40S |
|---|---|---|---|---|---|---|---|
| [CoreWeave](https://www.coreweave.com/pricing) | 6.16 | 6.31 | 8.60 | sales | 10.50 | 2.50 | 2.25 |
| [Lambda](https://lambda.ai/pricing) | 3.99-4.29 | – | 6.69-6.99 | – | – | – | – |
| [Nebius](https://www.nebius.com/prices) | 4.50 | 5.40 | 8.50 | 9.50 | sales | 1.80 | 1.55 |
| [RunPod](https://www.runpod.io/pricing) secure | 3.99 | 5.29 | 7.99 | 8.99 | – | 2.49 | 1.09 |
| AWS (trackers: [Vantage](https://instances.vantage.sh/aws/ec2/p6-b300.48xlarge), GPUSmith) | 6.88 | 7.91 | 14.24 | 17.80 | – | – | – |

European providers (getdeploying, USD; no EUR price page could be read:
Scaleway returned 403, OVHcloud truncated): Scaleway H100 3.23, B300 8.45;
OVHcloud H100 2.99, H200 6.20; Seeweb H200 2.93, B300 5.62; Verda H200 6.38,
B300 11.57.

Silicon Data's neocloud index reads lower (H100 2.53, H200 3.10, B200 5.62 in
September; [silicondata.com](https://www.silicondata.com/products/silicon-index?gpu=h100),
snippets only).

**Purchase prices.** Nvidia publishes none. Analyst estimates (S, snippets,
not fetched): GB200 NVL72 rack ~$3.0-3.9M (SemiAnalysis, Wolfe); GB300 NVL72
$4.3M (Wolfe) vs $6-6.5M (Tom's Hardware), conflicting; Vera Rubin NVL72 up
to ~$8.8M ([Tom's Hardware](https://www.tomshardware.com/tech-industry/artificial-intelligence/price-of-nvidias-vera-rubin-nvl72-racks-skyrockets-to-as-much-as-usd8-8-million-apiece-but-server-makers-margins-will-be-tight-nvidia-is-moving-closer-to-shipping-entire-full-scale-systems));
HGX B300 8-GPU server $400-520K ([American Compute](https://www.amcompute.com/appraisals/b300)).

## 3. What the model needs to change

### 3.1 Rows it can take as they are

H100 SXM, B200 and GB200/GB300 per GPU fit today's `GPU` dataclass: memory,
bandwidth, dense FP8, `supports_nvfp4`, TDP and price. Two fields need a rule
rather than a datasheet number:

- **`vram` and `reserve_extra`.** The reserve is solved on the H200, whose
  "141 GB" delivers ~150.75e9 B. A part's `reserve_extra` is the H200's hidden
  9.75e9 minus the part's own over-provision, so each new part needs its
  nvidia-smi total. Blackwell delivers nominal decimal bytes (measured on
  B300; DGX B200's 180 GiB = 193e9 B says the same), so B200 takes
  `reserve_extra=9.75e9` like the B300. Hopper parts need their nvidia-smi
  MiB before they get a row.
- **Power.** Only Hopper fractions are measured (`research/power.md`).
  Transfer the same fractions of TDP (idle ~10.5%, decode 0.55, prefill 0.90)
  and take `host_w` from system power: DGX B200 (14.3 kW - 8 x 1,000 W) / 8 =
  790 W; NVL72 racks (132 kW / 72) - GPU TDP = ~430 W (GB300) and ~500 W
  (GB200). Every new power row is an extrapolation and says so.

### 3.2 Rack-scale NVLink (NVL72)

Setting `nvlink_domain=72` is correct but does not make NVL72 pay off in the
model. `tp_efficiency` charges 0.90 per doubling inside the domain, so TP64
keeps 0.53 of aggregate bandwidth, and the model has no other way to use 72
GPUs on one model: its only axes are DP x TP. Real NVL72 serving of large MoE
models runs **wide expert parallelism** (experts spread over 32-72 GPUs, DP
attention, all-to-all dispatch over NVLink), which TP cannot represent.

Proposal, in order:
1. Add the NVL72 rows with `nvlink_domain=72` and keep the cross-node 0.65
   for groups past 72. They price correctly for DP x TP<=8 deployments, which
   is what an NVL72 rack runs when used as nine 8-GPU nodes.
2. Add an `ep` axis to `Topology`: routed-expert weights stored once across
   `dp x tp` GPUs of an EP group, attention weights per TP group, and a
   per-step all-to-all term `2 x layers x batch x hidden x bytes / nvlink_bw`
   where `nvlink_bw` becomes a `GPU` field (900e9 Hopper, 1.8e12 Blackwell,
   3.0-3.6e12 Rubin). This is where NVL72 differs from eight HGX boxes, and
   it needs a measured anchor (vLLM or SGLang wide-EP on GB200) before the
   explorer shows it.
3. Replace the flat 0.90 per doubling with a time term from `nvlink_bw` (TP
   all-reduce bytes per step / link bandwidth + a fixed latency), so the
   haircut differs by generation instead of being one constant.

Step 1 is WS-7. Steps 2 and 3 are a new issue: they change decode, capacity
and golden for every model.

### 3.3 PCIe cards and the Blackwell marker

- **No NVLink.** Today `nvlink_domain=1` would send any TP>1 group through
  the 0.65 cross-node haircut (TP2 0.65, TP4 0.42), the same penalty as
  InfiniBand between nodes. PCIe P2P inside one server is faster than that
  but slower than NVLink. Proposal: a per-part `tp_eff_per_doubling` field
  (0.90 NVLink, 0.80 PCIe Gen5, 0.75 PCIe Gen4; all unmeasured and labelled
  so) with `nvlink_domain` = cards per server (8), so the cross-node 0.65
  keeps meaning "leaves the server". The 2-card NVL bridges of H100/H200 NVL
  do not fit this scheme cleanly; leave those two parts out.
- **`supports_nvfp4` doubles as "Blackwell".** `check_dtype_supported` uses
  it to allow GLM-5.3-Flash's FP8 KV and to pick GLM-5.3's Blackwell cache
  row. Both depend on SM100 kernels (FlashMLA, DeepGEMM), which the
  RTX PRO 6000 (SM120) and the L40S (SM89) do not have. Proposal: replace
  the boolean with `arch` (`"hopper"`, `"ada"`, `"blackwell"`,
  `"blackwell_rtx"`, `"rubin"`) and derive `supports_nvfp4` (blackwell,
  blackwell_rtx, rubin) and a `sm100_kernels` flag (blackwell, rubin) from
  it. Until someone verifies which MLA/sparse-attention models vLLM serves on
  SM120, refuse them on RTX PRO 6000 and L40S rather than price them.
- **Weights per card.** 48 GB (L40S) and 96 GB (RTX PRO 6000) hold the
  27B/35B-A3B models only; the 744B-class models need more cards than a
  server has. `min_tp_for` already reports that; the explorer must show "does
  not fit" instead of a zero pool.

### 3.4 Price band

`GPU.eur_gpu_h` is one median. Add `eur_gpu_h_band = (low, median, high)` and
keep `eur_gpu_h` = median as the default. Use the neocloud band for low/high,
not the raw extremes: the raw low is unverified marketplace stock and the raw
high is hyperscaler list, which no buyer of 8+ GPUs pays. Proposed values at
EUR/USD 1.16 (the rate the model already uses, read 2026-09-03; re-read
before landing), rounded to 0.10:

| Part | Low | Median | High |
|---|---|---|---|
| H100 SXM | 2.20 | 3.10 | 3.90 |
| H200 | 3.10 | 4.10 | 5.40 |
| B200 | 5.20 | 6.90 | 7.40 |
| B300 | 6.00 | 8.20 | 8.60 |
| GB200 NVL72 | 9.10 | 11.40 | 13.80 |
| RTX PRO 6000 | 1.50 | 2.00 | 2.20 |
| L40S | 0.70 | 1.40 | 2.00 |

GB200 has one neocloud rate, so its band runs from CoreWeave ($10.50) to OCI
($16.00). Every median sits inside its band. Changing the H200 and B300
medians moves every golden cost figure, so that lands in Python first with a
golden regeneration.

### 3.5 Which parts to add first

| Part | Add? | Why |
|---|---|---|
| B200 | yes | most deployed Blackwell (DGX B200 fleets), every field sourced |
| H100 SXM | yes, after its nvidia-smi total | the installed base; needs `reserve_extra` |
| GB300 NVL72, GB200 NVL72 | yes, as rows with domain 72 | priced as DP x TP<=8 until the EP axis lands |
| RTX PRO 6000 Server | after `arch` | cheap small-model serving; kernel support unverified |
| L40S | after `arch` | same, and FP8-only |
| Rubin NVL72 | not yet | bandwidth and NVLink conflict on Nvidia's own pages, no TDP, no rental price |
| H20, H100/H200 NVL | no | China-only with conflicting specs; 2-4 card bridges fit neither regime |

## 4. Link fields for a communication cost model

For the parallelism spec (`.scratch/specs/parallelism.md`). NVLink figures
are per GPU and **bidirectional** (Nvidia's convention); NCCL bus bandwidth
compares against half of it. Read 2026-11-04.

| Part | NVLink gen | NVLink per GPU | Domain | Host PCIe | Scale-out NIC per GPU |
|---|---|---|---|---|---|
| H100 SXM | 4 | 900 GB/s (18 links) | 8 | Gen5 x16, 128 GB/s | ConnectX-7 400G |
| H100 NVL | 4 | 600 GB/s bridge | 2 | Gen5 x16 | server choice |
| H200 SXM | 4 | 900 GB/s | 8 | Gen5 x16 | ConnectX-7 400G |
| H200 NVL | 4 | 900 GB/s bridge | 2 or 4 | Gen5 x16 | server choice |
| H20 | 4 (S) | 900 GB/s (resellers) | 8 | Gen5 x16 | not found |
| B200 | 5 | 1.8 TB/s (18 x 100 GB/s) | 8 | Gen5 (not stated by Nvidia) | ConnectX-7 400G |
| B300 | 5 | 1.8 TB/s | 8 | **Gen6** x16, 256 GB/s | ConnectX-8 800G |
| GB200 NVL72 | 5 | 1.8 TB/s | 72 (130 TB/s rack) | Grace Gen5; GPU-Grace C2C 900 GB/s (stated for the Ultra part only) | ConnectX-7 400G (S) |
| GB300 NVL72 | 5 | 1.8 TB/s | 72 | Gen6 x16 | ConnectX-8 800G |
| Vera Rubin NVL72 | 6 | 3.6 TB/s (Rubin blog, HGX Rubin NVL8 page); the NVLink page's 3.0 TB/s is marked preliminary | 72 | Gen6 x16; Vera C2C 1.8 TB/s | ConnectX-9 1.6 Tb/s (2 x 800G) |
| RTX PRO 6000 Server | none | – | 1 | Gen5 x16 | server choice |
| L40S | none | – | 1 | Gen4 x16, 64 GB/s | server choice |

Sources: [NVLink](https://www.nvidia.com/en-us/data-center/nvlink/),
[HGX](https://www.nvidia.com/en-us/data-center/hgx/), the product pages in §1,
[Inside Blackwell Ultra](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/)
(B300 Gen6, C2C), [Rubin platform blog](https://developer.nvidia.com/blog/inside-the-nvidia-rubin-platform-six-new-chips-one-ai-supercomputer/),
[RTX PRO 6000 datasheet](https://www.akamai.com/site/en/documents/corporate/nvidia-rtx-pro-6000-blackwell-server-edition-datasheet.pdf),
[NCCL 2.27 blog](https://developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27/) (P);
H20 from [Tom's Hardware](https://www.tomshardware.com/pc-components/gpus/the-tale-of-nvidias-hgx-h20-how-an-ai-gpu-became-a-political-lightning-rod) (S).

### Measured collectives

| System | Collective | Number | Source |
|---|---|---|---|
| 4 x GB200 (one tray) | small all-reduce | NCCL ring 11.0 µs; one-shot multicast kernel 2.37 µs; speed-of-light 1.40 µs | [arXiv 2607.16100](https://arxiv.org/pdf/2607.16100) (P) |
| GB200 NVL72, 2-32 GPUs | barrier | 0.85-1.7 µs | same |
| 8 x B200 / 8 x H200 | 8 B all-reduce, tuned nccl-tests | ~3 µs / ~2.3 µs | [Nvidia forum](https://forums.developer.nvidia.com/t/inter-gpu-latency-on-b200-higher-than-on-hopper/352473) (S) |
| 4 x H100 | all-reduce busbw, large | ~366 GB/s with NVLS (77% of 450 GB/s per direction) | [dev.to](https://dev.to/member_2e5ba30f/where-tensor-parallel-inference-hits-the-nvlink-wall-1l8p) (S) |
| GB200 NVL72, EP8 | MoE all-to-all, 2048 tok/rank | dispatch 753 GB/s, combine 728 GB/s | [TensorRT-LLM blog 18](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog18_Optimizing_MoE_Communication_with_One_Sided_AlltoAll_Over_NVLink.html) (P) |
| GB200 NVL72, EP8 | all-to-all, 1 tok/rank | dispatch 19 µs, combine 31 µs (latency-bound) | same |
| GB200 NVL72, EP64 | all-to-all, 2048 tok/rank | dispatch 588 GB/s, combine 631 GB/s | same |
| H800, EP8 intranode | DeepEP normal kernels | dispatch 153 GB/s, combine 158 GB/s | [DeepEP README @9eb2f84](https://github.com/deepseek-ai/DeepEP/blob/9eb2f84/README.md) (P) |
| H800, internode CX-7 400G | DeepEP normal kernels | 43-58 GB/s per NIC | same |
| H800, EP8 | DeepEP low-latency | dispatch 77 µs, combine 114 µs | same |
| H800, EP256 | DeepEP low-latency | dispatch ~194 µs, combine ~360 µs, 39-40 GB/s | same |
| RTX PRO 6000 Server | P2P same PCIe switch | 40 GB/s (45 GB/s via the NIC path) | [Nvidia forum](https://forums.developer.nvidia.com/t/pcie5-p2p-gpu-via-nics-faster-than-pcie-switch/357843) (S) |
| 8 x H100 NIXL KV write | 1 GiB | 39 GB/s same NUMA, 27.5 GB/s cross-NUMA (NIC unstated) | [dranet NIXL test](https://dranet.sigs.k8s.io/docs/user/nixl-kv-transfer/) (S) |

Not found: 8-GPU all-reduce busbw for H100/H200/B200 from a primary
source; untuned default small-message latency; 72-GPU all-reduce busbw;
8-card all-reduce on L40S or RTX PRO 6000; a measured ConnectX-7 or XDR
`ib_write_lat`; NIXL over a named 400G NIC. InfiniBand NDR switch latency is
~130 ns per hop (resellers only); a 400G NIC is 50 GB/s per direction at line
rate.

Two cautions for the cost model:
- nccl-tests applies the ring correction whatever algorithm ran, so NVLS
  bus bandwidth can read above the per-direction link rate
  ([PERFORMANCE.md](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)).
- PCIe P2P on the RTX PRO 6000 is fragile in practice: vLLM on 8 cards
  needed `--disable-custom-all-reduce` and `NCCL_P2P_DISABLE=1`
  ([vLLM forum](https://discuss.vllm.ai/t/vllm-hangs-during-worker-initialization-on-blackwell-pcie-gpus-unless-disable-custom-all-reduce-is-used/2540), S).
  A PCIe server can end up all-reducing through host memory, which belongs in
  the PCIe parts' link figure, not in the NVLink haircut.
