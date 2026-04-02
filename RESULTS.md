# Phase 2 Summary After Phase 2g.1

This note freezes the repository's current Phase 2 state after the fairness-hardened Phase 2g protocol, the local Phase 2g.1 boundary-check closure pass, and the downstream refreshes sourced from the refined Phase 2g.1 configs.

It is an internal research-engineering summary, not a paper-style claim set.

Phase 3 follow-on status:

- the repository has now completed a narrow Phase 3a step:
  - a small real-data MLP baseline on `sklearn.datasets.load_digits`
  - explicit `train / val / test` reporting
  - deterministic mini-batch ordering
  - reproducible artifacts under `outputs/digits_mlp/`
- this file still remains a Phase 2 summary
- it should not be read as evidence that a real-data predictive-coding baseline or real-data PC-vs-MLP comparison is already complete

Local artifact-retention note:

- the scientific conclusions in this file reflect the full Phase 2 history
- the locally retained `outputs/` tree may be lighter than the full Phase 2 history after cleanup and may keep only the current `digits_mlp` Phase 3 baseline run plus `.gitkeep`
- older generated outputs are treated as reproducible artifacts that can be regenerated when needed

## Project State

Stable baseline and infrastructure:

- Phase 0: the baseline predictive-coding math remains fixed by [spec_math.md](/e:/CodeSpace/PredictiveCoding/spec_math.md)
- Phase 1: structured experiment outputs, benchmark scripts, saved traces, and plots are in place
- Phase 1.5: seed semantics and output-layout semantics are explicit and stable

Experimental findings layers:

- Phase 2a: initial PC-vs-MLP comparison
- Phase 2b: narrow PC sensitivity analysis
- Phase 2c: small multi-seed aggregate comparison
- Phase 2d: diagnostics for the tuned-PC vs MLP gap
- Phase 2e: tuned-PC budget-performance tradeoff
- Phase 2f: split-aware protocol hardening plus joint PC tuning
- Phase 2g: matched small-scope PC/MLP tuning plus refreshed downstream multiseed and budget studies
- Phase 2g.1: local boundary-check closure pass for search-space truncation risk

The current strongest Phase 2 claims should now come from the best-known Phase 2 evidence chain:

- Phase 2g matched search
- Phase 2g.1 local boundary check
- Phase 2g.1-refreshed downstream multiseed and budget-tradeoff artifacts

## Phase 2a: Initial PC-vs-MLP Comparison

What it added:

- a minimal NumPy MLP baseline trained with ordinary backpropagation and SGD
- a direct PC-vs-MLP comparison runner

Question:

- how does the baseline predictive-coding implementation compare against a standard MLP on the current toy tasks?

Main finding:

- on the regression toy benchmarks, the initial PC baseline underperformed the MLP baseline

Interpretation:

- the initial gap was large enough to justify focused PC-side diagnostics before expanding model families or benchmark scope

## Phase 2b: Small PC Sensitivity Analysis

What it added:

- a narrow one-factor-at-a-time PC sensitivity layer
- controlled trials over `eta_x`, `eta_w`, `train_steps`, and `state_init`

Question:

- is the early regression gap mainly caused by obviously conservative PC hyperparameters?

Main finding:

- modest tuning helped materially on both regression toy benchmarks
- `eta_w_double` was the best one-factor result on both tasks
- tuned PC still trailed the fixed MLP reference under the then-current workflow

Interpretation:

- the initial gap was not just a trivial consequence of clearly bad default PC settings
- but the search was still too narrow to support a strong final comparison claim

## Phase 2c: Multi-Seed Stability Check

What it added:

- a small multi-seed study over:
  - default PC
  - a fixed tuned PC preset
  - MLP

Question:

- are the default-vs-tuned and tuned-vs-MLP conclusions stable across seeds?

Main finding:

- tuned PC beat default PC stably across seeds on both regression toy benchmarks
- the fixed MLP baseline also beat the fixed tuned PC preset stably across seeds

Interpretation:

- the tuned-PC improvement was real
- but the MLP comparison was still tied to a narrow PC preset and a weaker evaluation protocol

## Phase 2d: Diagnostics

What it added:

- learning-curve summaries
- a budget-diagnostic PC branch
- energy-vs-metric correlation summaries

Question:

- why did the tuned PC preset still trail the MLP reference?

Main finding:

- more PC inference budget helped the old tuned preset
- PC energy tracked task MSE closely on aggregated curves

Interpretation:

- the old gap looked more like an optimization-efficiency or budget issue than evidence that PC was optimizing a completely misaligned internal quantity

## Phase 2e: Budget-Performance Tradeoff

What it added:

- a small `1x / 2x / 4x` tuned-PC budget study with a fixed MLP reference

Question:

- if the tuned PC preset gets more inference budget, does that close the gap?

Main finding:

- more budget improved the old tuned preset on both regression tasks
- the improvement showed diminishing returns under that legacy workflow

Interpretation:

- this was useful evidence about the old tuned preset
- but it still did not resolve the fairness problem that MLP remained mostly fixed while PC had only narrow tuning

## Phase 2f: Split-Aware Evaluation and Joint PC Tuning

What it added:

- explicit split-aware benchmark protocol support
- deterministic joint PC tuning
- downstream refreshes sourced from the selected Phase 2f PC config

Question:

- what changes if the repository stops relying on train-heavy or ambiguous metric reporting and instead tunes PC on an explicit held-out split?

Main finding:

- earlier PC-vs-MLP gap estimates were materially confounded by under-tuning on the PC side
- after split-aware PC tuning, PC became clearly stronger than the old fixed tuned presets

Interpretation:

- Phase 2f was a major methodology improvement
- but it was still not the fairest final comparison because MLP had not yet received a matched small-scope tuning pass

## Phase 2g: Matched Search and Refreshed Downstream Comparison

What it added:

- a matched small-scope search for both PC and MLP
- validation-based config selection
- held-out test reporting
- refreshed downstream multiseed and budget-tradeoff studies using the selected Phase 2g PC and MLP configs

Question:

- under a fairer small-scope protocol, does PC beat, match, or trail MLP on held-out test?

Protocol:

- both PC and MLP are tuned under finite, explicit search spaces
- configs are selected by `val_metric`
- headline method comparisons are reported on `test_metric`
- downstream multiseed and budget summaries preserve that distinction explicitly

### `toy_regression`

Matched-search best configs:

- best PC: `cfg_021`
  - `eta_x = 0.05`
  - `eta_w = eta_b = 0.4`
  - `train_steps = eval_steps = 25`
  - `epochs = 240`
  - `state_init = "forward"`
  - val MSE: `3.568103814199095e-05`
  - test MSE: `3.5716213568204525e-05`
- best MLP: `cfg_012`
  - `eta_w = eta_b = 0.2`
  - `epochs = 320`
  - val MSE: `0.000166951111924279`
  - test MSE: `0.000167243513956922`

Refreshed multiseed result:

- selected PC mean test MSE: `3.428560223330302e-05`
- selected MLP mean test MSE: `0.00012926517952227103`
- selected PC beats selected MLP on `5/5` seeds

Refreshed budget result:

- tuned PC `1x / 2x / 4x` mean test MSE:
  - `3.428560223330302e-05 / 4.569058567057169e-05 / 5.874398960991758e-05`
- selected MLP mean test MSE: `0.00012926517952227103`
- extra PC inference budget no longer helps on held-out test
- the best current budget variant is already `1x`

Current interpretation for this benchmark:

- under the fairer Phase 2g protocol, PC beats MLP on held-out test
- the current seed check supports that conclusion as stable on this benchmark

### `toy_sine_regression`

Matched-search best configs:

- best PC: `cfg_108`
  - `eta_x = 0.15`
  - `eta_w = eta_b = 0.2`
  - `train_steps = eval_steps = 120`
  - `epochs = 320`
  - `state_init = "forward"`
  - val MSE: `0.02162854962546966`
  - test MSE: `0.021631965873664307`
- best MLP: `cfg_012`
  - `eta_w = eta_b = 0.2`
  - `epochs = 320`
  - val MSE: `0.01513598557530098`
  - test MSE: `0.015138264607443988`

Refreshed multiseed result:

- selected PC mean test MSE: `0.029057912018180847`
- selected MLP mean test MSE: `0.016455286068690497`
- selected MLP beats selected PC on `4/5` seeds
- selected PC beats selected MLP on `1/5` seed

Refreshed budget result:

- tuned PC `1x / 2x / 4x` mean test MSE:
  - `0.029057912018180847 / 0.029922700944118003 / 0.025246694934550024`
- selected MLP mean test MSE: `0.016455286068690497`
- extra PC inference budget helps partially on held-out test
- the help is non-monotonic:
  - `2x` is worse than `1x`
  - `4x` is better than `1x`
- even the best current `4x` budget still trails selected MLP on mean test MSE

Current interpretation for this benchmark:

- under the fairer Phase 2g protocol, MLP beats PC on held-out test
- the current seed check supports that conclusion more often than not, but not unanimously

## Phase 2g.1: Local Boundary-Check Closure Pass

What it added:

- a compact local extension around the current Phase 2g best configs
- explicit reporting of:
  - previous Phase 2g best config
  - boundary-check best config
  - whether the held-out test winner changed
  - whether the selected best config moved beyond the old search bounds

Question:

- are the current Phase 2g conclusions materially dependent on the original search-space truncation?

Main findings:

- `toy_regression`
  - the held-out test winner remained `PC`
  - the best PC config improved further by moving below the old `eta_x` lower edge to `0.025`
  - the best MLP config also improved by moving beyond the old `eta_w` and `epochs` upper edges
- `toy_sine_regression`
  - the held-out test winner remained `MLP`
  - the best PC config improved further by moving beyond all previously active upper edges:
    - `eta_x`
    - `eta_w`
    - `train_steps`
    - `epochs`
  - the best MLP config also improved further by moving beyond the old `epochs` upper edge

Interpretation:

- the benchmark-level winners survived the local boundary extension
- both benchmarks remain boundary-sensitive in the narrower sense that selected best configs moved beyond the old matched-search bounds
- Phase 2 is therefore stable enough to close as a methodology stage, but not saturated enough to justify “globally optimal” language

## Phase 2g.1 Downstream Refresh

What it added:

- refreshed multiseed studies sourced from the refined Phase 2g.1 best PC and MLP configs
- refreshed budget-tradeoff studies sourced from the refined Phase 2g.1 best PC config and the refined Phase 2g.1 MLP comparison config
- downstream summary artifacts that now point to the best-known Phase 2 config source rather than the earlier Phase 2g selected configs

Question:

- after the Phase 2g.1 boundary check improves both PC and MLP best configs, does the downstream story change materially?

Main finding:

- `toy_regression`
  - refined multiseed mean test MSE:
    - PC: `2.470167272781127e-05`
    - MLP: `9.392473608660969e-05`
  - refined PC still beats refined MLP on `5/5` seeds
  - refined budget study still prefers `1x`, and PC remains ahead of MLP at all tested budgets
- `toy_sine_regression`
  - refined multiseed mean test MSE:
    - PC: `0.014516499089550083`
    - MLP: `0.013660899062922338`
  - refined MLP still beats refined PC on `3/5` seeds, while PC is better on `2/5`
  - refined budget study now prefers `1x`; extra PC inference budget no longer helps on held-out test under the refined base config

Interpretation:

- the benchmark-level winners did not change
- the main scientific update is not a winner flip, but a cleaner best-known evidence chain and a narrower budget interpretation:
  - `toy_regression`: PC remains clearly ahead
  - `toy_sine_regression`: MLP remains ahead, and the earlier “more PC budget may help” story no longer survives under the refined base config

## Current Conclusions

- earlier train-only and train/eval-style Phase 2 conclusions were weaker than the current Phase 2g / 2g.1 conclusions
- Phase 2f showed that under-tuning on the PC side was a major confound
- Phase 2g then made the comparison fairer by tuning both PC and MLP under the same validation/test protocol
- Phase 2g.1 showed that the current benchmark-level winners are not immediately overturned by a small local boundary extension
- the Phase 2g.1 downstream refresh moved the best-known Phase 2 evidence chain onto the refined boundary-check configs
- the strongest current repository-level conclusion remains benchmark-dependent:
  - `toy_regression`: boundary-check-refined PC beats boundary-check-refined MLP on held-out test, does so stably across the current multi-seed check, remains ahead in the refined budget study, and keeps that winner after the Phase 2g.1 boundary check
  - `toy_sine_regression`: boundary-check-refined MLP beats boundary-check-refined PC on held-out test, does so on most seeds in the current multi-seed check, remains ahead in the refined budget study, and keeps that winner after the Phase 2g.1 boundary check
- extra PC inference budget is no longer a universal improvement:
  - it does not help on `toy_regression`
  - it does not help on `toy_sine_regression` under the refined base config either
- PC energy still tracks task MSE closely on aggregated curves, so the current evidence still does not support the claim that PC is optimizing a completely misaligned internal quantity
- Phase 2 is now methodologically stable enough to proceed to Phase 3, but not exhaustive enough to support claims of search saturation or global optimality

## Limitations

- the current benchmarks are still toy tasks
- the current search spaces are finite and intentionally modest, not exhaustive
- the matched searches are still single-seed model-selection procedures rather than nested multi-seed selection protocols
- the boundary-check closure pass and refined downstream refreshes improve confidence, but they still do not saturate the finite search spaces
- Phase 2e-style budget studies are about inference-step budget, not wall-clock or FLOP efficiency
- a real-data predictive-coding baseline is still pending
- a real-data matched PC-vs-MLP evaluation is still pending

## Recommended Next Step

Proceed from the completed Phase 3a `digits` MLP baseline to the first real-data predictive-coding baseline, while carrying forward the Phase 2g / 2g.1 protocol discipline:

- keep explicit train/val/test separation
- keep deterministic seed roles and explicit checkpoint selection by validation
- keep held-out test as the final headline metric

Why this is justified now:

- the repository has already hardened the toy-benchmark comparison protocol substantially
- the strongest current conclusion is no longer a simple "PC always trails MLP" story
- the Phase 2g.1 closure pass did not reverse the benchmark-level winners
- the Phase 2g.1 downstream refresh did not reverse those winners either
- the most informative unresolved questions are now:
  - does a real-data predictive-coding baseline train stably on `digits` under the same protocol?
  - after that, do the benchmark-dependent Phase 2 conclusions survive in a real-data PC-vs-MLP comparison?
