# Validation and Acceptance Criteria

This file defines what counts as success for each early phase.

The goal is to stop the project from drifting into "the code runs, therefore it must be correct."

---

## General validation philosophy

A valid implementation should demonstrate all of the following:

1. **shape correctness**
2. **numerical sanity**
3. **behavioral sanity**
4. **reproducibility**
5. **traceability to the math spec**

---

## Phase 0 acceptance checks

These are the minimum checks that must pass before Phase 0 is considered done.

### A. Shape tests

Verify at least:

- `A^l`, `MU^l`, `E^l` have expected shapes for a small network
- `W^l` and `b^l` updates preserve shape
- batch size 1 and batch size > 1 both work

### B. Deterministic seeding

Under a fixed seed:

- parameter initialization is reproducible
- a deterministic smoke inference run produces the same energy trace within tolerance

### C. Energy computation sanity

On a tiny hand-constructed example:

- energy matches a manually computed expected value
- zero local error produces zero contribution for that layer

### D. Inference sanity

On a small deterministic network and batch:

- running inference for a modest number of steps should reduce energy overall
- if energy is not strictly monotone each step, the test should at least verify a clear net decrease under stable hyperparameters

### E. Parameter update sanity

On a small example:

- applying one update step changes at least one parameter tensor in the expected nonzero direction
- no NaN or Inf values appear

### F. End-to-end smoke learning

On a toy dataset:

- training runs for several epochs
- final loss or energy is better than the initial loss or energy
- the model performs better than a trivial constant predictor baseline

---

## Recommended test inventory

The repository should aim to contain at least the following tests.

### `test_shapes.py`

- layer forward shape checks
- multi-layer shape propagation
- bias broadcasting checks

### `test_energy.py`

- manually checked energy on tiny tensors
- zero-error case
- multi-layer aggregation sanity

### `test_inference.py`

- clamped layers remain unchanged
- free hidden states change under nonzero error
- energy decreases on a stable small setup

### `test_weight_updates.py`

- parameter tensors change after update
- updates preserve shapes
- no NaN / Inf created

### `test_regression_smoke.py`

- end-to-end run on tiny synthetic data
- training improves a basic metric

---

## Phase 1 additional checks

After the minimal baseline works, add:

- regression tests for previously fixed bugs
- experiment log file creation
- output directory writing
- CLI / script usability checks where practical

---

## Phase 2 stronger checks

Target additions:

- finite-difference checks for small instances where feasible
- sensitivity sweeps over inference steps and step size
- comparison plots against a plain MLP baseline on toy data
- explicit documentation of failure regions

### Legacy vs refreshed Phase 2 conclusions

Phase 2 conclusions should now be read in four layers:

1. Legacy Phase 2 conclusions
   - These refer to the earlier comparison workflow built around:
     - train-only or ambiguous single-field metric reporting
     - one-factor-at-a-time PC sensitivity
     - fixed downstream tuned presets such as the old `eta_w_double`
   - Under that methodology, the stable conclusion was:
     - tuned PC beat default PC on the two regression toy benchmarks
     - tuned PC still trailed the fixed MLP baseline
     - larger inference budgets helped tuned PC, but with diminishing returns
   - Those conclusions remain useful as a historical record, but they are now treated as methodology-limited rather than final.
   - The corresponding local output directories may no longer be retained after output cleanup; the conclusions survive here as documented historical summaries and can be regenerated if needed.

2. Phase 2f conclusions
   - These refer to the protocol-hardening step built around:
     - explicit split-aware metric reporting
     - deterministic joint PC tuning
     - downstream refreshes sourced from the Phase 2f-selected PC config
   - Phase 2f showed that the earlier PC-vs-MLP gap was materially confounded by under-tuning on the PC side.
   - However, Phase 2f was still not the fairest final comparison because MLP remained closer to a fixed reference than a matched-tuned baseline.
   - As with the legacy Phase 2 outputs, the original Phase 2f local output directories may be pruned after cleanup and should be treated as reproducible rather than permanently retained artifacts.

3. Phase 2g conclusions
   - These refer to the current strongest Phase 2 workflow built around:
     - true train/val/test protocol support
     - matched small-scope tuning for both PC and MLP
     - `val_metric` used for configuration selection
     - held-out `test_metric` used for the final headline comparison
   - Phase 2g is the fairer comparison protocol and should now anchor the repository's top-level Phase 2 conclusions.

4. Phase 2g.1 conclusions
   - These refer to the local boundary-check closure pass built around:
     - the same Phase 2g train/val/test protocol
     - the same selection/reporting split discipline
     - a compact local extension beyond the old matched-search boundaries
   - Phase 2g.1 does not replace Phase 2g with a new large search.
   - Its purpose is narrower:
     - test whether the Phase 2g headline winners survive a principled local boundary extension
     - reveal whether the current best configs are still boundary-sensitive

### What changed after split-aware evaluation

The benchmark protocol no longer reports a single ambiguous evaluation number:

- regression benchmarks keep the small training samples
- they now use deterministic dense validation and test grids distinct from training
- blobs classification now uses deterministic held-out validation and test data
- experiment artifacts clearly distinguish:
  - `train_metric`
  - `val_metric`
  - `test_metric`
  - `metric_name`
  - `metric_higher_is_better`

This matters because the final reported result can now be separated from configuration selection:

- `val_metric` is the selection signal
- `test_metric` is the final held-out report
- test is no longer reused as the tuning target

### What changed after joint tuning

The earlier one-factor sensitivity studies were intentionally narrow and useful for diagnosis, but they underexplored interactions among:

- `eta_x`
- `eta_w` / `eta_b`
- inference step budget
- epoch count

Phase 2f showed that joint tuning on the PC side alone already changed the downstream picture substantially.
Phase 2g went further by giving both PC and MLP a matched small-scope search under the same validation/test protocol.

That makes Phase 2g the fairer basis for claims such as:

- whether PC beats, matches, or trails MLP
- whether that conclusion is stable across seeds
- whether more PC inference budget still helps on held-out test

### Current strongest Phase 2 evidence chain

Evidence used:

- `outputs/phase2g_matched_search/toy_regression/aggregate_summary.json`
- `outputs/phase2g_matched_search/toy_sine_regression/aggregate_summary.json`
- `outputs/phase2g1_boundary_check/toy_regression/aggregate_summary.json`
- `outputs/phase2g1_boundary_check/toy_sine_regression/aggregate_summary.json`
- `outputs/pc_multiseed_phase2g1_toy_regression/aggregate_summary.json`
- `outputs/pc_multiseed_phase2g1_toy_sine_regression/aggregate_summary.json`
- `outputs/pc_budget_tradeoff_phase2g1_toy_regression/aggregate_summary.json`
- `outputs/pc_budget_tradeoff_phase2g1_toy_sine_regression/aggregate_summary.json`

These are the current Phase 2 authority artifacts retained locally by default after output cleanup.

#### `toy_regression`

Matched-search result:

- best PC config: `cfg_021`
  - `eta_x = 0.05`
  - `eta_w = eta_b = 0.4`
  - `train_steps = eval_steps = 25`
  - `epochs = 240`
  - `state_init = "forward"`
  - val MSE: `3.568103814199095e-05`
  - test MSE: `3.5716213568204525e-05`
- best MLP config: `cfg_012`
  - `eta_w = eta_b = 0.2`
  - `epochs = 320`
  - val MSE: `0.000166951111924279`
  - test MSE: `0.000167243513956922`
- selection uses validation
- final headline comparison uses held-out test

Boundary-check result:

- Phase 2g.1 extended the matched-search neighborhood locally beyond the old edges
- the best PC config moved past the old lower edge on `eta_x`:
  - old best PC: `eta_x = 0.05`
  - boundary-check best PC: `eta_x = 0.025`
  - new best PC test MSE: `1.9999929568712413e-05`
- the best MLP config also moved past the old upper edges:
  - old best MLP: `eta_w = 0.2`, `epochs = 320`
  - boundary-check best MLP: `eta_w = 0.4`, `epochs = 480`
  - new best MLP test MSE: `0.00011400586610806358`
- headline held-out test winner remained `PC`
- conclusion stability:
  - benchmark-level winner survived the boundary extension
  - this benchmark is still boundary-sensitive because both selected best configs moved outside the original search bounds

Refined downstream result:

- selected refined PC mean test MSE: `2.470167272781127e-05`
- selected refined MLP mean test MSE: `9.392473608660969e-05`
- selected refined PC beats default PC on `5/5` seeds
- selected refined PC beats selected refined MLP on `5/5` seeds

Refined budget result:

- tuned PC `1x / 2x / 4x` mean test MSE:
  - `2.470167272781127e-05 / 3.425462497355812e-05 / 4.565706464243781e-05`
- selected refined MLP mean test MSE: `9.392473608660969e-05`
- extra PC inference budget does not help on held-out test
- the best current budget variant is already `1x`

Held-out test conclusion:

- PC beats MLP on this benchmark
- the current refined multi-seed check supports that conclusion as stable

#### `toy_sine_regression`

Matched-search result:

- best PC config: `cfg_108`
  - `eta_x = 0.15`
  - `eta_w = eta_b = 0.2`
  - `train_steps = eval_steps = 120`
  - `epochs = 320`
  - `state_init = "forward"`
  - val MSE: `0.02162854962546966`
  - test MSE: `0.021631965873664307`
- best MLP config: `cfg_012`
  - `eta_w = eta_b = 0.2`
  - `epochs = 320`
  - val MSE: `0.01513598557530098`
  - test MSE: `0.015138264607443988`
- selection uses validation
- final headline comparison uses held-out test

Boundary-check result:

- Phase 2g.1 extended the matched-search neighborhood locally beyond the old edges
- the best PC config moved past every previously active upper edge:
  - old best PC: `eta_x = 0.15`, `eta_w = 0.2`, `train_steps = 120`, `epochs = 320`
  - boundary-check best PC: `eta_x = 0.2`, `eta_w = 0.3`, `train_steps = 180`, `epochs = 480`
  - new best PC test MSE: `0.017236549943255866`
- the best MLP config also moved past the old epoch upper edge:
  - old best MLP: `eta_w = 0.2`, `epochs = 320`
  - boundary-check best MLP: `eta_w = 0.2`, `epochs = 480`
  - new best MLP test MSE: `0.010870819845007006`
- headline held-out test winner remained `MLP`
- conclusion stability:
  - benchmark-level winner survived the boundary extension
  - this benchmark is still boundary-sensitive because both selected best configs moved outside the original search bounds

Refined downstream result:

- selected refined PC mean test MSE: `0.014516499089550083`
- selected refined MLP mean test MSE: `0.013660899062922338`
- selected refined PC beats default PC on `5/5` seeds
- selected refined MLP beats selected refined PC on `3/5` seeds
- selected refined PC beats selected refined MLP on `2/5` seeds

Refined budget result:

- tuned PC `1x / 2x / 4x` mean test MSE:
  - `0.014516499089550083 / 0.014547698280740706 / 0.014975233433198332`
- selected refined MLP mean test MSE: `0.013660899062922338`
- extra PC inference budget does not help on held-out test under the refined base config
- the best current budget variant is already `1x`

Held-out test conclusion:

- MLP beats PC on this benchmark
- the current refined multi-seed check supports that conclusion more often than not, but not unanimously

### Current best Phase 2 interpretation

The strongest current Phase 2 conclusion is benchmark-dependent rather than global, and it now includes the Phase 2g.1 closure check plus the refined downstream refresh:

- earlier train-only and train/eval-style conclusions were weaker and more confounded
- Phase 2f showed that PC under-tuning was a major confound
- Phase 2g then made the comparison fairer by tuning both PC and MLP under the same validation/test protocol
- Phase 2g.1 checked whether those matched-search winners were just artifacts of truncated search bounds
- under that stronger reading:
  - `toy_regression`: boundary-check-refined PC beats boundary-check-refined MLP on held-out test, does so stably across the current multi-seed check, remains ahead in the refined budget study, and that winner survives the local boundary extension
  - `toy_sine_regression`: boundary-check-refined MLP beats boundary-check-refined PC on held-out test, does so on most seeds in the current multi-seed check, remains ahead in the refined budget study, and that winner also survives the local boundary extension

This keeps the earlier diagnostic interpretation narrow:

- the repository still has evidence that PC energy tracks task MSE closely on aggregated curves
- the current evidence still does not support the claim that PC is optimizing a completely misaligned internal quantity
- there is still no single blanket statement such as:
  - "PC trails MLP on the regression toy tasks"
  - or "PC beats MLP on the regression toy tasks"
- the current evidence supports a more cautious statement:
  - performance is benchmark-dependent under the fairer Phase 2g / 2g.1 protocol
  - the benchmark-level winners appear stable under a small local boundary extension
  - remaining differences are plausibly about optimization efficiency, budget sensitivity, or benchmark-specific inductive fit rather than a wholesale objective mismatch

### What Phase 2 now establishes

- the current strongest Phase 2 comparison protocol is:
  - matched small-scope tuning for PC and MLP
  - selection by `val_metric`
  - final reporting by held-out `test_metric`
  - local boundary checking around the matched-search winners
- under that protocol:
  - `toy_regression`: PC is the current benchmark-level winner
  - `toy_sine_regression`: MLP is the current benchmark-level winner
- both benchmark-level winners survived the local Phase 2g.1 boundary extension
- the refreshed downstream multiseed and budget-tradeoff studies remain aligned with those winners

### What Phase 2 still does not establish

- it does not show that the search spaces are saturated
- it does not show that the current best configs are globally optimal
- it does not show that the current matched selection is equivalent to nested multi-seed selection
- it does not show that the toy-benchmark conclusions automatically transfer to real datasets

### Recommended next engineering step

Phase 2 is now stable enough to proceed to Phase 3, but that statement has a precise meaning:

- stable enough means the current benchmark-level winners survived the Phase 2g.1 boundary extension
- it does not mean the toy-search spaces are exhaustive or that the best configs are fully saturated

The most justified next step is therefore to proceed to Phase 3 on a small real dataset, while carrying forward the Phase 2g / 2g.1 protocol discipline:

- keep explicit train/val/test separation
- keep deterministic matched small-scope tuning
- keep held-out test for final reporting
- avoid falling back to train-only or train/eval-style headline claims

Why this is the better next step:

- the repository now has a fairer toy-benchmark comparison protocol
- the strongest current conclusion is not a universal winner but a benchmark-dependent split result
- the most informative unresolved question is therefore external validity:
  - do these conclusions hold on a small real dataset?

Important caveats that should remain explicit:

- the current toy benchmarks are still simple
- the current search spaces are finite and not exhaustive
- both benchmarks still show boundary sensitivity in the sense that the local boundary-check best configs moved beyond the old search bounds
- the matched searches are still single-seed selection procedures, not nested multi-seed model-selection protocols
- Phase 3 real-data evaluation is still pending

---

## Logging expectations

Training and evaluation scripts should eventually record at least:

- epoch index
- training energy
- optional validation energy
- inference-step energy trace for selected batches
- parameter norms
- state norms if useful

---

## Tolerances and realism

Not every check must demand exact equality.

Use:

- exact equality only where algebraically guaranteed and numerically trivial
- `np.allclose`-style tolerances for floating-point comparisons
- behavioral checks for convergence trends rather than fragile exact trajectories

---

## Failure reporting

When a validation check fails, the report should ideally say:

1. what was expected
2. what was observed
3. which part of the math or code is most likely implicated
4. whether the issue is deterministic or seed-sensitive

---

## Minimal Phase 0 completion statement

Phase 0 is complete only when all of the following are true:

- the baseline implementation matches `spec_math.md`
- at least 5 meaningful tests pass
- a toy experiment runs end-to-end
- inference energy is observable and behaves sanely
- the codebase is ready for cleanup and expansion in Phase 1
