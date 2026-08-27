# Power & energy-cost constants — sources and what is *not* measured

**Purpose:** defensible power constants for the energy-cost layer (€/kWh × the
watts each configuration actually draws), alongside the study's H200/B300
capacity model. The rest of the study prices *time*; this note prices the
**power state the GPU is in during that time**: compute-bound prefill bursts
near the power cap, bandwidth-bound decode well below it, warm idle otherwise.

> **Status: mixed provenance, weaker than the capacity notes.** TDPs and
> system maxima are vendor spec (HIGH). Idle and the prefill/decode split
> rest on **measured Hopper studies** — none of them on this study's exact
> (H200 SXM, vLLM, large-batch) operating point, and *nothing* measured
> exists for B300 power states at all. Every B300 duty figure below is an
> extrapolation rule, labeled as such. The single measurement that would
> firm this up most: one `nvidia-smi --query-gpu=power.draw -lms 100` trace
> alongside a `vllm bench` run on the 2×H200 hardware that already exists —
> it would replace the two softest rows (idle, `p_decode`) with house data.

---

## 1. Constants for the model

| Constant | H200 SXM | B300 (HGX/SXM) | Provenance | Basis |
|---|---|---|---|---|
| `tdp_w` | **700** | **1,400** | SPEC (HIGH) | H200 page: "Up to 700W (configurable)" SXM (NVL is 600W — not this study's part). B300: "up to 1,400 W" per NVIDIA's "Inside Blackwell Ultra"; the **air-cooled SXM6 AC variant caps at 1,100 W** (Oracle BM.GPU.B300.8 nvidia-smi dump, `research/gpu_b300.md`) — see §3 rule. |
| `idle_w` | **80** | **150** | MEASURED (H100 proxy) / EXTRAPOLATED | H100 warm idle measured **72.5 W ± 0.1** per GPU at 0.1 s resolution (NLR facility-planning study, 4×H100 nodes). H200 carries 141 GB HBM3e vs 80 GB HBM3 → +10% allowance, → 80 W. B300: no measurement anywhere; rule = same **~10.5% of TDP** fraction → ~150 W. |
| `p_decode_w` | **385** (= 0.55 × TDP) | **770** (= 0.55 × TDP) | MEASURED band (H100) / EXTRAPOLATED (B300) | Splitwise (ISCA'24, DGX-H100): token phase power is flat in batch size and survives a **700→350 W power cap with "almost no latency impact"** — measured decode draw ≲ 0.5 TDP. Mixed vLLM serving on H200-class parts measured 229–477 W (0.33–0.68 TDP). 0.55 central, **0.45–0.75 band** (upper end covers large-batch decode drifting compute-ward). B300 transfers the *fraction*, not the watts — same memory-bound physics, zero Blackwell measurements. |
| `p_prefill_w` | **630** (= 0.90 × TDP) | **1,260** (= 0.90 × TDP) | MEASURED anchor (H100) / EXTRAPOLATED (B300) | Compute-bound phases are **power-cap-limited**: Splitwise finds prompt-phase latency "increases substantially" under any cap, and NVIDIA's max-perf MLPerf H200 submissions *raise* the cap to 1,000 W to go faster. Saturated vLLM inference measured **~0.85–0.89 of summed GPU TDP** node-wide (NLR: 2.8 kW sustained on 4×700 W H100s + host). Central 0.90, band 0.80–1.00. Holds across the study's MFU bracket ([30–60%] when written; tightened to [35–55%] 2026-08-27) — the cap binds before the FLOP peak does, so power is flat-ish in MFU while *time* is not. |
| `host_w_per_gpu` | **575** | **500** | DERIVED from SPEC | DGX H200: 10.2 kW system max − 8 × 700 W = 4.6 kW of CPUs/NVSwitch/NICs/fans/PSU loss → **575 W/GPU**. DGX B300: 14.5 kW "estimated system power" (19–19.7 kW transient peak) − 8 × 1,400 W = 3.3 kW → 412 W/GPU; if the GPUs inside are the 1,100 W-capped variant the same arithmetic gives 712 W/GPU. **500 W sits below the 562 W mid of that band, weighted toward the 1,400 W-TDP scenario (412 W/GPU).** Flat adder, not duty-cycled — see §4. |
| `pue` | **1.5 default** (colo) · presets 2.0 (server room) / 1.2 (hyperscale) | same | SURVEY (HIGH for colo, MEDIUM for room) | Uptime Institute 2024 survey: industry mean **1.56**, flat five years running; capacity-weighted **1.47** (2023). Small facilities sit systematically worse (Uptime size analysis; LBNL's small-DC program pegs closets/rooms near **2.0**). |
| `eur_per_kwh` | **0.19 default**, slider **0.05–0.60** | same | STATISTICAL (HIGH) | Eurostat non-household electricity, EU average **€0.1902/kWh** H1-2025 (€0.1899 H2-2024). Country spread H2-2024: €0.0767 (FI) – €0.2578 (CY); the 0.05–0.60 range also covers household tariffs (DE households ~€0.40) and cheap PPAs. |

Model constants proposed:

```python
POWER = {                      # watts
    "H200": Power(tdp_w=700.0,  idle_w=80.0,  p_decode_w=385.0,
                  p_prefill_w=630.0, host_w_per_gpu=575.0),
    "B300": Power(tdp_w=1400.0, idle_w=150.0, p_decode_w=770.0,
                  p_prefill_w=1260.0, host_w_per_gpu=500.0),
}
PUE_DEFAULT = 1.5              # colo; presets: 2.0 server room, 1.2 hyperscale
EUR_PER_KWH_DEFAULT = 0.19     # Eurostat non-household EU avg; slider 0.05–0.60
```

## 2. The measured evidence, and how far it stretches

Three independent measured sources agree on the shape *(all Hopper, none on
this exact stack — the proxy is labeled in every row above)*:

1. **Splitwise** (Patel et al., ISCA 2024; DGX-A100 + DGX-H100, Llama2-70B /
   BLOOM-176B): prompt-phase power rises with batch and uses the full GPU
   power budget; token-phase power is **flat in batch size** and tolerates a
   50% power cap (700→350 W) with almost no latency impact — their
   production design caps token machines to 50% per GPU. This is the
   load-bearing "decode ≪ TDP" measurement.
2. **NLR facility-planning study** (arXiv 2604.07345; 4×H100-SXM nodes,
   vLLM, Llama-3-70B, 0.1 s-resolution facility metering): warm idle
   **72.5 W/GPU**; saturated inference **2.8 kW sustained per node** (~0.85
   of the 2.8 kW summed GPU TDP once ~0.3 kW of host is netted out). This is
   the ceiling anchor: *mixed* continuous batching at full utilisation —
   chunked prefill interleaved with decode — runs near, not at, the cap.
3. **HPC serving comparison** (arXiv 2507.00418; vLLM): H200-class parts
   measured **228.8 W** (1.1B model) to **476.7 W** (90B model) averaged over
   serving — i.e. real mixed serving lands mid-band, bracketing the
   `p_decode` central value from both sides.

Cross-check that keeps the split honest: `0.55·TDP` decode + `0.90·TDP`
prefill, mixed at a decode-heavy duty, reproduces the NLR 0.85 saturation
figure without tuning.

**Measurement caveat carried forward:** nvidia-smi samples A100/H100 power
only ~25% of the runtime and its real error is ~±5% (±30 W at 700 W), per
"Part-time Power Measurements" (arXiv 2312.02741). Fine for constants with
±20% bands; worth knowing before any future house validation run.

## 3. B300 extrapolation rule (nothing measured exists)

No published idle/phase power for any Blackwell Ultra part was found —
MLPerf Blackwell submissions report performance, not power. Rule adopted:
**transfer Hopper's fractions of TDP** (idle ~10.5%, decode 0.55, prefill
0.90) onto the 1,400 W cap. Direction of error: unknown for idle (bigger HBM
array, newer power gating); decode fraction plausibly *lower* on B300 (8 TB/s
of HBM3e behind a 2× power cap serves the same memory-bound work), which
would make B300 energy-per-token look better than modeled — flag, don't
claim. **Deployment correction:** if the target is the air-cooled 1,100 W
SXM6 AC variant (the one real B300 nvidia-smi dump the study has), scale all
three B300 GPU wattages by 1,100/1,400 ≈ 0.79.

## 4. Recommended power model

Per GPU, from the duty fractions the model already computes (`prefill_duty`
`d_p` = min(1, prefill duty — clamped at saturation); decode-active fraction `d_d`; the remainder idles warm):

```
P_gpu   = d_p·p_prefill_w + d_d·p_decode_w + (1 − d_p − d_d)·idle_w
P_total = (P_gpu + host_w_per_gpu) × pue          # per GPU, at the meter
cost    = P_total × n_gpus × hours × eur_per_kwh / 1000
```

Error bars, stated honestly: the GPU term carries **±20–25%** (the
`p_decode` 0.45–0.75 band dominates; prefill duty is usually small); the
host adder is a **spec ceiling used flat**, over-charging a lightly-loaded
chassis by up to ~2× (measured lean HPC hosts run ~100–150 W/GPU; a DGX
with NVSwitch, 8 NICs and fans at load sits far higher — the flat 575 W is
the conservative choice, same direction as the study's other bookkeeping);
`pue` and `eur_per_kwh` are exact user-chosen multipliers, not model error.
B300 rows: add "entirely extrapolated" on top (§3). Double-counting trap
for the integrator: during mixed continuous batching a chunked-prefill pass
is *prefill*-priced — `d_p` and `d_d` must partition time, not overlap.

## 5. What this establishes / does not establish

**Establishes:** spec-grade TDP and system-max anchors for both parts; a
measured Hopper basis for "decode ≈ half TDP, prefill ≈ the cap, warm idle
≈ 10% of TDP"; survey-grade PUE and Eurostat price defaults; a duty-cycle
formula whose mixed prediction reproduces an independent facility
measurement.

**Does not establish:** any Blackwell-measured power state; H200-specific
(vs H100) idle or decode watts; host draw as a function of load; power
during PCIe restore / KV transfers (unpriced, same gap as the capacity
model); tail-power transients (the 19.7 kW DGX B300 *peak* vs 14.5 kW
sustained — sizing question, not an energy-cost one).

## Sources

- NVIDIA H200 page (SXM "Up to 700W (configurable)"):
  https://www.nvidia.com/en-us/data-center/h200/ · datasheet:
  https://resources.nvidia.com/en-us-hopper-architecture/hpc-datasheet-sc23
- NVIDIA DGX H200 datasheet (10.2 kW max, 6×3.3 kW PSU):
  https://resources.nvidia.com/en-us-dgx-systems/dgx-h200-datasheet
- NVIDIA DGX B300 user guide (14.5 kW, 12×3.2 kW PSU):
  https://docs.nvidia.com/dgx/dgxb300-user-guide/introduction-to-dgxb300.html ·
  Data Center Best Practices with DGX B300 (14.5–15 kW estimated system,
  19–19.7 kW estimated peak; 58 kW avg/76 kW peak per 4-system rack):
  https://docs.nvidia.com/dgx-pdf/data-center-best-practices-with-dgx-b300-v1.pdf
- B300 TDP: NVIDIA "Inside Blackwell Ultra" (up to 1,400 W) — see
  `research/gpu_b300.md`; air-cooled 1,100 W cap: Oracle OCI B300 quickstart
  (same note's primary dump).
- Splitwise — Patel et al., ISCA 2024 (phase power + power-cap experiments):
  https://arxiv.org/abs/2311.18677
- NLR, "Measurement of Generative AI Workload Power Profiles for
  Whole-Facility Data Center Infrastructure Planning" (72.5 W idle; 2.8 kW
  saturated 4×H100 node): https://arxiv.org/abs/2604.07345
- "Serving LLMs in HPC Clusters: Qualcomm Cloud AI 100 Ultra and NVIDIA
  Data Center GPUs" (H200-class measured 228.8–476.7 W under vLLM):
  https://arxiv.org/abs/2507.00418
- "Part-time Power Measurements: nvidia-smi's Lack of Attention" (sampling
  caveat): https://arxiv.org/abs/2312.02741
- MLPerf power-cap corroboration — Cisco UCS C885A M8 MLPerf white paper
  (H200 at 700 W default, 1,000 W max-perf config):
  https://www.cisco.com/c/en/us/products/collateral/servers-unified-computing/ucs-c-series-rack-servers/mlperf-ucs-c885a-h100-h200-wp.html
- Uptime Institute Global Data Center Survey 2024 (mean PUE 1.56):
  https://datacenter.uptimeinstitute.com/rs/711-RIA-145/images/2024.GlobalDataCenterSurvey.Report.pdf ·
  size analysis (capacity-weighted 1.47):
  https://journal.uptimeinstitute.com/large-data-centers-are-mostly-more-efficient-analysis-confirms/ ·
  LBNL small data centers program: https://datacenters.lbl.gov/
- Eurostat electricity price statistics (non-household EU avg €0.1902/kWh
  H1-2025; country spread):
  https://ec.europa.eu/eurostat/statistics-explained/index.php?title=Electricity_price_statistics ·
  https://ec.europa.eu/eurostat/web/products-eurostat-news/w/ddn-20260508-2
