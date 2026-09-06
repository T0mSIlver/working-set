# Golden vectors

`src/workingset/model.py` is the source of truth. `interactive/src/*.js` is a
hand-maintained JS mirror of it: the same constants, the same formulas, typed
twice. Until this fixture existed, nothing checked that the two still said the
same thing — a change to one could sit unmirrored in the other indefinitely,
and the explorer would publish numbers the package would not.

**`vectors.json` is what the Python model says.** Every deployment the explorer
can reach — 7 models x 2 GPU parts x 1-8 GPUs x every DP/TP split of those that
holds the weights x the servable weight/KV dtype arms, **594 of them, each
priced at least once** — plus a one-knob-at-a-time sweep on four anchors and
the all-defaults reference of each model/part. Each state is priced by
`workingset.model` and written out with its inputs, its outputs, a `cond` block
of conditioning diagnostics, and the tolerance class of each output.

The knobs are stratified over the deployments rather than crossed with them
(the full product is astronomical); the one-at-a-time sweep is what isolates
which control a disagreement follows. Every sweep value is inside its control's
real range in `interactive/index.html`, so no vector describes a state a user
cannot produce.

**`tests/js/golden.test.mjs` is the check.** It replays every state through the
JS modules and asserts agreement.

---

## Regenerating

```sh
uv run scripts/golden.py             # rewrite tests/golden/vectors.json
uv run scripts/golden.py --check     # fail if the committed file is stale
uv run scripts/golden.py --jobs 1    # same bytes, one process
```

Byte-reproducible: every draw is seeded, every float is rounded to 12
significant digits, the JSON separators are fixed, and the work is split across
processes only for speed — the output does not depend on `--jobs`. `--check`
regenerates in memory and diffs, the way `scripts/sync_harness.py --check`
does; CI runs it, so the fixture cannot drift behind the model.

Runtime: ~10.5 min of CPU, so ~2.5 min wall on four cores and ~2 min on eight
(821 states, two Monte-Carlo warm fills and a decode-ceiling bisection each,
plus a 24-state spread probe at three seeds and two sampling scales). The JS
side is ~80 s. That is the price of pricing every legal deployment rather than
a sample of them.

## Running the mirror test

```sh
node --import ./tests/js/register.mjs --test tests/js/golden.test.mjs
```

No npm install, no browser. It prints two tables — the worst relative
disagreement per quantity and per model — and asserts three things:

1. every key of a vector's `out` block has a JS counterpart (so deleting a line
   from `drive.mjs` cannot turn a comparison into a silent skip);
2. every allowlist entry matched something (so a fixed difference cannot leave
   a standing licence behind);
3. no disagreement exceeds its tolerance without an allowlist entry.

```sh
GOLDEN_NO_ALLOWLIST=1 node --import ./tests/js/register.mjs --test tests/js/golden.test.mjs
```

...ignores `known_disagreements.json`, which is how you see the raw set.

**Why Node and not a browser.** The explorer's math modules are DOM-free;
`main.js` (slider wiring, URL fragments, hover handlers) is not, and it calls
`computeAndRender()` at module scope. `tests/js/loader.mjs` swaps that one
module for a stub — see `tests/js/stub-main.mjs` for the whole of what it
replaces — and `register.mjs` adds a small DOM shim for the one module-scope
`addEventListener` in `harness.js`. Every module that computes a number loads
unmodified, and the test enters the mirror through the same doors the page
does: `activeModel()`, `currentTopo()`, `currentWL()`, `seedFor()`.

The same loader's `load` hook stamps `format: "module"` on the explorer's `.js`
sources. They are ESM but the repo has no `package.json` declaring
`{"type":"module"}` — deliberately, since the nearest one to `interactive/`
would ship to the Worker — so without the hook the parse would fall back to
Node's ESM syntax detection, which is only the default from 22.x. With it the
structural floor is `module.register` (Node 20.6). CI pins 24, which is the
only version verified end to end.

## Tolerance classes

`exact` — closed form on both sides, no sampling anywhere in the chain. The
KV pool, the bandwidth, the FLOP counts, the MFU ceiling. Relative 1e-6, which
is float noise. A failure here is a real difference: a constant typed twice and
edited once, a dropped term, a different rounding convention.

`mc` — Monte-Carlo on both sides, with *different samplers*. Python draws
200,000 context lengths from numpy's PCG64; the explorer draws 20,000 from
mulberry32 with Box-Muller normals. The two will never agree exactly, and the
band has to say how close is close enough.

The bands are measured, not guessed. `scripts/golden.py` runs a probe of
24 states at three seeds, at two sampling scales, and records the p50 / p90 /
max relative spread of every sampled quantity in the `mc_spread` block of
`vectors.json`:

- `mc_spread.python` — the scale the vectors are generated at.
- `mc_spread.mirror` — the same, with the context draw cut to the explorer's
  20,000. That is the one place the two sides sample materially differently
  (the explorer's warm fills and decode sweeps run at comparable counts), so it
  is what sets the comparison's noise floor.

The band is **3 x the p90 spread under `mc_spread.mirror`**, floored at 2% and
capped at 25%. 3x because the measured spread is the range of three seeds of
*one* sampler while the comparison runs two independent ones. p90 rather than
max because a couple of states put an estimator somewhere it is ill-conditioned
(the latency ceiling as a miss's own prefill approaches the SLA; the queue wait
as rho approaches 1) and their spread runs an order of magnitude above every
other state's — a band set on those would license real drift everywhere else.
Those states are named in the allowlist instead, by the condition that makes
them ill-conditioned rather than by name.

Quantities that are the *same statistic* take the widest of their group's
bands (`BAND_GROUPS` in `golden.py`): `power_draw`'s `d_p` is `prefill_duty`
clamped at 1, and the clamp shrinks its measured spread, so an independent
derivation would hand the identical figure a tighter band and it would trip on
noise its twin absorbs.

`flag` — booleans (`steady_saturated`, `max_users_decode_censored`). Equal or
not.

Two quantities have their bands derived from the probe states with
`warm_p5 >= 12` only (`max_users_cache`, `warm_p5_all`): below that a fill
holds a handful of sessions and one session either way is 30%, which the
allowlist already owns by condition. Letting that tail set the band would grant
the same slack to a configuration holding twenty thousand.

**On the `exact` residuals.** Every value in `vectors.json` is stored at 12
significant digits so the file regenerates byte-identically on any machine
(full precision would make `--check` depend on the last ulp of `np.log2` and
`**` across libm versions — a flaky CI gate in exchange for six digits nothing
reads). The `exact` assertion is at 1e-6, six orders coarser than that
quantisation. So a reported worst residual of ~5e-12 is the **storage floor**,
not a measured agreement to 5e-12: read it as "nothing above 1e-11".

## The allowlist

`known_disagreements.json` is the list of differences that are known, explained
and tolerated. Each entry names a quantity (an exact name, or a `*`-suffixed
prefix), a `where` clause, the relative error actually observed, a `max_rel`
ceiling, and a one-line hypothesis of the cause.

A `where` clause matches the vector's state fields, the derived
`n_gpu` / `replicas`, and — under a `_` prefix — the vector's `cond` block:
the conditioning diagnostics `golden.py` emits beside the outputs so an entry
can be gated on the **cause** rather than on a model name.

| `where` key | what it says |
|---|---|
| `_duty` | prefill duty; `1/(1-rho)` amplifies every queue figure as it approaches 1 |
| `_sla_headroom` | `1 - E[S\|miss]/SLA`; the latency ceiling's `k = 2(SLA - c)` vanishes at 0 and goes negative past it |
| `_sla10_headroom` | the same against the 10 s budget `spikeMetrics` hard-wires — use this one for the `*_sla10` quantities |
| `_sla_f_unreachable` | 1 where `sla_miss_rate` returned its `hi` clamp: the SLA survives an all-cold stream, so there is no root in `[0, 1]` to compare |
| `_warm_p5` | the warm count in sessions — one session either way is 33% of three |
| `_decode_ceiling` | same, for the decode bisection |
| `_ctx_cv2` | squared coefficient of variation of the context length; `E[S^2\|miss]` runs on `L^4`, so its sampling variance scales with this |
| `_steady_nmax_ratio` | demand / aggregate throughput at the **mirror's own** widest decode `n`. At `>= 1` the load runs off the end of the axis `steadyDecodePoint` inverts, which is when it stops resolving and starts reporting `saturated` |

```jsonc
{
  "quantity": "max_users_cache",
  "where": { "sub_ratio": [0.1, 0.4, 0.5, 1.0] },
  "observed_rel": 0.09,
  "max_rel": 0.2,
  "hypothesis": "the explorer approximates the user-class warm count as
                 p5(all) x (1 - p_sub); Python counts user-class sessions
                 inside each fill"
}
```

The ceiling matters: a known 9% difference does not license a new 90% one, and
a `max_rel` of 1.0 with no `where` licenses **everything** (`relErr` is bounded
by 1 for any finite pair), which is not an allowlist entry, it is a deletion of
the quantity. When a difference is fixed, delete its entry — the test
**asserts** that every entry matched something, so a stale allowlist fails the
build rather than sitting there silently protective.

An entry is a **debt, not a decision**. It records that the two implementations
disagree and why we believe they do; it does not record that the disagreement
is correct.

### The taxonomy of what it finds

Every `exact`-class quantity agrees to the storage floor: no constant is typed
twice and edited once, and no closed form has drifted. Everything on the list
is a sampled quantity, and every entry in the allowlist is one of these:

1. **A hard-wired constant on the mirror's side.** `prefill.js`'s
   `spikeMetrics()` prices `bstar`, `fsla` and `drain` against
   `SPIKE_SLA_S = 10` and `SPIKE_BURST = 32` rather than `state.sla` and
   `state.burst`. The vectors price Python at the same 10 and 32 so the two
   answer the same question; the hard-wire itself is recorded here, not
   silently absorbed.
2. **A hard-wired constant on Python's side.** `power_draw` uses
   `DECODE_FLOOR_TOKS` where `cost.js` reads the live floor, so the decode duty
   comes out **exactly 40/floor** apart — 8x at a 5 tok/s floor. That exact
   factor holds *before* `d_d`'s `min(1 - d_p, ...)` clip and only where the
   two sides' decode ceilings agree; where either fails it is approximate. It
   carries into the per-GPU watts, the kW and the electricity line (but not
   the monthly total, which the hardware line dominates). A Python bug.
3. **A genuine algorithmic difference.** `steady_decode_point` bisects integer
   `n` out to 4,096, redrawing at each probe; `steadyDecodePoint` inverts the
   interpolated aggregate of the sweep the page already drew, whose widest `n`
   is 1.15 x the warm p95. Where the load runs past that axis
   (`_steady_nmax_ratio >= 1`) the mirror reports `saturated` and Python
   resolves a point — two answers to different questions, not a numeric
   disagreement. Below the axis end the same difference is bounded at ~18%.
   `sla_miss_rate` is the second of these: Python bisects `f` over `[0, 1]` and
   clamps to it, the explorer solves the closed form and clamps only at zero,
   and their "no load meets this" tests differ (Python compares the whole TTFT
   at `f = 0`, the explorer the miss's own prefill alone).
4. **A standing approximation.** `max_users_cache`: the explorer scales the
   whole warm p5 by `(1 - p_sub)`, where Python counts user-class sessions
   inside each fill. Invisible at ordinary counts, visible at three sessions.
5. **A sign-stable sampler-structure bias.** `decode_p50_n1` reads 5-7% *low*
   on the mirror, consistently, and hardest under FP16 KV where the sampled KV
   term dominates the step. `decodeCurves` draws one context pool per iteration
   and reads it cumulatively (common random numbers across `n`), where
   `decode_curves` redraws per `n`; at `n = 1` that structure and the smaller
   draw count show. `decode_p50_n8` and `_n64` — the same estimator where the
   average over `n` contexts washes it out — stay inside their bands, which is
   the evidence for the reading.
6. **Ill-conditioned estimators**, which are not disagreements about the model
   at all: the latency ceiling where a miss already eats its whole TTFT budget
   (`k = 2(SLA - c)` is then a difference of two nearly equal sampled numbers),
   the Pollaczek-Khinchine wait at `rho -> 1`, `E[S^2|miss]` on a heavy-tailed
   context distribution, warm counts of two or three sessions, a decode ceiling
   of one. Each is allowlisted by the condition that makes it ill-conditioned,
   never by model name.

(2) and the two hard-wires in (1) are bugs. (3) is a modelling difference.
Fixing any of them is a separate change: **this fixture measures, it does not
repair.**

## The rule

**A modelling change lands in Python first. The JS test failing is the intended
signal, not the bug.**

So when this test goes red after a model change:

1. If you changed `src/workingset/model.py`, regenerate the vectors
   (`uv run scripts/golden.py`) and then port the change to
   `interactive/src/*.js` until the test passes again. Do not add an allowlist
   entry to paper over an unported change.
2. If you changed `interactive/src/*.js` and the test went red, the mirror has
   drifted from the model. Either the JS is wrong, or the change belongs in
   Python first and the JS second.
3. Only add an allowlist entry when the difference is *deliberate* — a
   documented approximation, a convention the explorer takes for UI reasons —
   and say which in the hypothesis.

## Coverage limitations

Four things the fixture does not pin, and one thing it pins only partly. None
is a silent omission — the `mapping` block names each at its row.

**Four quantities are duplicated in `drive.mjs`, not called.** The explorer
computes them somewhere the driver cannot reach, so for these the test pins the
*model*, not the explorer's own line of code: an edit to the render.js
expression would pass.

| quantity | where the explorer computes it |
|---|---|
| `ttft_miss_fcfs`, `ttft_hit_fcfs`, `ttft_hit_ps` | inline in `render.js`'s `Object.assign` onto the operating point |
| `max_users_cache` | `render.js` `warmUsersNow`, inline |
| `mean_passes` | `prefill.js` `meanPasses`, module-private |

Moving the three `render.js` expressions into `prefill.js` and exporting
`meanPasses` would close this. It is a **follow-up**, not part of this change:
it touches the render path, which `AGENTS.md` requires be verified
byte-identical in a headless browser.

**`itl_spike` / `spike_token_debt` / `op.tokensLost` are not compared at all.**
`itlSpikeRatio` is private to `render.js` *and* prices the spike differently on
purpose — `step = min(C, E[L])` with `prior = 0` when the chunk exceeds the
mean context, against Python's always-`C` at `prior = E[L]/2` — and its decode
leg reads an interpolated stress point rather than a `decode_curves` probe.
Comparable only once it is exported and the two pricings are reconciled, which
is a Python-first modelling change, not a test change.

**`state.burst` moves nothing that is compared.** `spikeMetrics()` hard-wires
`SPIKE_BURST = 32`; the slider reaches the tile only through the inline
`render.js` arithmetic above. So the vectors compare
`burst_drain_seconds_b32` — both sides at 32 — and sample `burst` anyway, to
record that it is inert in the compared set rather than to imply it is covered.

**`state_dt` (fp32 recurrent state) and `wover` (+15% deployed weights)** are
explorer-only controls with no counterpart in `workingset.model`, so they are
held at their defaults. A golden vector for them would encode a JS convention
as if it were the model's.

**`operatingPoint()`'s ceiling scaling** is a documented convention difference,
not a disagreement: the explorer reports all four ceilings system-wide
(per-group closed form x `replicas`), Python keeps all four per replica group.
The vectors compare the per-group closed forms both sides actually compute.

**Everything the page draws rather than computes** — SVG geometry, the frontier
table's ranking, tooltips. `AGENTS.md` already asks for a headless-browser
check on those.
