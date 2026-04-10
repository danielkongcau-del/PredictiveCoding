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

These are the Phase 2 authority artifacts used to establish the current conclusions. Depending on local cleanup policy, you may need to regenerate some of them before inspecting the corresponding directories again.

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

### Phase 3 done means what

Phase 3 is now complete in a deliberately narrow sense:

- a small real-data baseline exists on `sklearn.datasets.load_digits`
- the real-data protocol uses explicit `train / val / test` splits
- a deterministic mini-batch pipeline exists
- the digits MLP baseline writes a reproducible artifact set under `outputs/digits_mlp/`
- the first digits predictive-coding baseline writes a reproducible artifact set under `outputs/digits_pc/`

Concretely, the current Phase 3 baseline establishes:

- `src/pc/datasets.py` provides a deterministic, stratified `load_digits` split returning `SupervisedDataSplit`
- inputs are batch-first `(batch, 64)` `float64` arrays normalized by `16.0`
- targets are batch-first `(batch, 10)` `float64` one-hot arrays
- `src/pc/minibatch.py` provides deterministic mini-batch ordering under an explicit `batch_order_seed`
- `experiments/digits_mlp.py` runs a real-data MLP baseline without changing the existing MLP math
- `experiments/digits_pc.py` runs a real-data predictive-coding baseline without changing the existing PC math
- the canonical `digits_pc` baseline has been refreshed from a narrow stabilization sweep selected by validation accuracy
- `experiments/summarize_digits_baselines.py` provides a small standalone side-by-side digest of the current canonical baseline summaries
- the resulting `summary.json` explicitly separates:
  - `train_metric`
  - `val_metric`
  - `test_metric`
  - `selection_metric_source = "val_metric"`
  - `report_metric_source = "test_metric"`

### Phase 3 done does not mean what

Phase 3 completion does **not** mean:

- a real-data matched PC-vs-MLP comparison has already been completed
- MNIST is already part of the repository's active baseline workflow
- the real-data search space has been explored or tuned aggressively

### Phase 3a and 3b protocol alignment

Phase 3a and Phase 3b now share the same real-data protocol contract on `digits`:

- both use `src/pc/datasets.py::load_digits_split()`
- both use explicit `train / val / test` splits with deterministic `data_seed`
- both use `src/pc/minibatch.py::iter_minibatches()` for deterministic batch ordering
- both report:
  - `dataset_name = "digits"`
  - `primary_metric_name = "accuracy"`
  - `selection_metric_source = "val_metric"`
  - `report_metric_source = "test_metric"`
- both record the same explicit seed roles:
  - `run_seed`
  - `data_seed`
  - `model_init_seed`
  - `batch_order_seed`
- both use majority-class accuracy as the baseline metric definition
- both restore the best checkpoint selected by validation accuracy before writing the final `train / val / test` report

This alignment is intentional protocol hardening.
It does **not** yet mean the repository has completed a real-data PC-vs-MLP comparison workflow.

### What the next phase can now assume

The repository now has the minimum prerequisites for a cautious real-data comparison phase:

- a deterministic real-data dataset entry on `digits`
- a deterministic mini-batch helper
- a canonical standalone MLP baseline
- a canonical standalone PC baseline
- protocol-alignment checks across those two baselines
- a first-pass side-by-side digest under `outputs/digits_baselines/`
- a current canonical Phase 3 output set under:
  - `outputs/digits_mlp/`
  - `outputs/digits_pc/`
  - `outputs/digits_baselines/`

### Phase 4 preparation inference-baseline hardening

Before any FMPC work, stronger standalone predictive-coding inference baselines should now be read under these rules:

- `pc_euler` remains the authoritative default inference backend
- `pc_rk2` is allowed only as an explicit stronger baseline variant, not a silent replacement
- `fmpc` may exist only as a reserved backend label until its semantics are actually implemented
- the predictive-coding energy and parameter-update math stay unchanged
- validation should confirm:
  - explicit `backend="pc_euler"` is backward-compatible with the prior default path
  - `pc_rk2` preserves batch-first shapes and finite values
  - real-data artifact summaries record the chosen inference backend explicitly
  - standalone real-data PC summaries keep teacher-reference handling explicit:
    - by default, teacher-reference metrics are disabled in standalone predict-mode evaluation summaries
    - the disable reason should be written explicitly rather than inferred from trivial all-zero metrics
    - meaningful FMPC teacher targets should come from dedicated train-mode teacher export paths instead

Teacher-reference metric scope should remain explicit:

- the current teacher is the slow iterative PC path under the current parameters
- standalone predict-mode summaries should not pretend they provide a meaningful FMPC teacher comparison by default
- where teacher-reference metrics are enabled in future dedicated protocols, the current update-direction cosine is the cosine between flattened terminal displacements from `z0`
- it is not yet a per-step transport-path cosine and should not be described that way
- if either terminal displacement has zero norm, the cosine should stay `null` rather than being fabricated

### Phase 4 seal-off note

Phase 4 is now sealed as an FMPC-v0 preparation checkpoint:

- standalone predict-mode real-data summaries should keep `teacher_reference` disabled by default
- meaningful FMPC teacher targets should be read only from the dedicated teacher-only preparation/export protocol
- the next phase may assume this preparation scaffold exists and begin the offline FMPC-v0 student stage

### Phase 5 offline FMPC-v0 student v0

The initial Phase 5 slice should be read narrowly:

- the teacher remains the frozen iterative PC path
- teacher supervision must come from the dedicated Phase 4 teacher-only preparation/export protocol
- the student is an offline endpoint transporter on `digits`
- the student input is `concat([z0, target_onehot])`
- the student target is `delta_z = z_star - z0`
- the primary training loss is MSE on `delta_z`
- validation selects the best checkpoint by held-out `state_rms_gap`
- final test reporting happens once, after restoring the best validation-selected checkpoint

Phase 5 v0 acceptance now also requires:

- teacher artifacts are portable:
  - new manifests use relative paths
  - loaders stay backward-compatible with older absolute-path manifests where possible
- student evaluation loads an exact serialized teacher checkpoint by default
  - config-plus-seed teacher retraining is only an explicit legacy fallback
- summaries report an explicit identity / zero-delta baseline:
  - `z_hat_identity = z0`
  - the student must be compared against that baseline on the same endpoint metrics
- final validation is not based only on the old trivial 2-step smoke teacher
  - a more meaningful digits teacher recipe must be used for acceptance-oriented checks

This does not yet mean:

- a full MeanFlow identity objective exists
- trajectory supervision is active
- refinement is part of the default training path
- a formal real-data PC-vs-student comparison has been completed

What is still missing before any stronger real-data claim:

- matched tuning
- a formal real-data comparison runner
- multi-seed aggregation
- a second real dataset
- any claim that the current standalone baseline numbers are a fair winner/loser result

### Recommended current engineering step

The current engineering priority is to close Phase 5 v0 acceptance on `digits` before attempting a stronger FMPC variant:

- keep explicit train/val/test separation
- keep the teacher frozen and sourced from dedicated preparation/export artifacts
- require portable manifests plus exact teacher checkpoint loading
- require an explicit identity baseline in student summaries
- validate on a non-trivial canonical digits teacher before considering interval-conditioned or trajectory-aware extensions

### Phase 5A student-signal rescue

Before any FMPC-v1-style extension is considered, the repository should answer a narrower question:

- under the fixed endpoint contract
  - input: `concat([z0, target_onehot])`
  - target: `delta_z = z_star - z0`
- does any simple learned student beat the explicit identity / zero-delta baseline on the canonical non-trivial `digits` teacher-

The required Phase 5A comparison set is:

- `identity`
- `class_mean_delta`
- `ridge`
- `mlp_standardized`

Phase 5A validation should require:

- all families use the same batch-first `float64` endpoint contract
- train-stat normalization is estimated only from the train split
- final metrics are computed after inverse-transform in the original hidden-state space
- the comparison artifacts make it obvious whether a learned family beats identity on:
  - validation `state_rms_gap`
  - held-out test `state_rms_gap`

Phase 5A should be considered passed only if at least one learned family:

- `ridge` or `mlp_standardized`

beats the identity baseline on both validation and held-out test `state_rms_gap`.

If every learned family still loses to identity, the next allowed escalation remains narrow:

- endpoint-only feature augmentation

This still does not authorize:

- trajectory-aware supervision
- MeanFlow / JVP objectives
- refinement
- core iterative `fmpc` backend integration

### Phase 5B offline interval-conditioned transporter

Phase 5B should also be read narrowly:

- the teacher remains the frozen iterative PC path
- supervision must come from trajectory-enabled teacher artifacts produced by the dedicated preparation/export protocol
- the default interval student input is `concat([z_s, target_onehot, tau_s, tau_t])`
- the default interval target is `u_star = (z_t - z_s) / (tau_t - tau_s)`
- rollout transport remains explicit:
  - `z_hat_t = z_hat_s + (tau_t - tau_s) * u_hat`
- no MeanFlow identity objective, JVP objective, refinement, or online joint training is introduced in this phase

The required Phase 5B comparison set is:

- `identity`
- carried-forward Phase 5A endpoint ridge baseline
- `interval_ridge`
- `interval_mlp_standardized`

Phase 5B validation should require:

- trajectory artifacts make `K`, trajectory tensor shape, exact checkpoint reference, and `tau_k = k / K` explicit
- the default interval-pair training policy is not dominated by short intervals
- final rollout evaluation is performed on explicit teacher-step-aligned schedules:
  - `1-step`
  - `2-step`
  - `3-step`
- rollout is self-fed between knots rather than teacher-forced

Phase 5B should be considered passed only if at least one learned interval family:

- `interval_ridge` or `interval_mlp_standardized`

beats the carried-forward Phase 5A endpoint ridge baseline on both validation and held-out test `final_state_rms_gap` under an explicit rollout schedule.

If every learned interval family still loses to the carried-forward endpoint ridge baseline, the next allowed rescue remains below MeanFlow / JVP:

- endpoint-free interval feature augmentation only

This still does not authorize:

- trajectory-aware supervision
- MeanFlow identity objectives
- JVP objectives
- refinement
- core iterative `fmpc` backend integration

### Phase 5B rollout-aware rescue

The first rescue step for a failing Phase 5B run should remain conservative:

- keep the original interval target as the primary objective
- add only a small rollout-aware auxiliary term
- keep that auxiliary term teacher-supervised and explicit
- stay below MeanFlow / JVP / refinement

The intended rescue semantics are:

- train interval students primarily on:
  - `u_star = (z_t - z_s) / (tau_t - tau_s)`
- for the standardized MLP family only, optionally add auxiliary supervision on fixed self-fed rollout schedules:
  - `2-step`
  - `3-step`
- start from teacher `z_0`
- feed each next knot with the student-predicted state, not the teacher state
- penalize:
  - intermediate knot state gaps versus teacher knot states
  - final endpoint gap versus teacher `z_K`

Validation should keep the rescue explicit:

- primary interval loss and rollout-aware auxiliary losses must be reported separately
- the final Phase 5B decision must still use the unchanged gate:
  - a learned interval family
  - under a true multi-step rollout schedule
  - must beat the carried-forward Phase 5A endpoint ridge baseline on both validation and held-out test `final_state_rms_gap`
- passing the rescue does not by itself authorize MeanFlow / JVP; it only justifies opening that next exploration stage

### Phase 5B.2 gradient-augmented interval rescue

If the rollout-aware rescue still fails narrowly, the next allowed Phase 5B rescue remains below MeanFlow / JVP:

- add frozen-teacher current-state dynamical features at `z_s`
- keep the interval target teacher-supervised
- keep rollout evaluation on the same explicit `1-step / 2-step / 3-step` teacher-aligned schedules

The intended semantics are:

- compute features only from the current state `z_s` plus the frozen teacher and current sample input/target
- do not leak:
  - `z_t`
  - `z_star`
  - future teacher states
- the first narrow feature pack should be:
  - `g_s`
  - `e_out_s`
  - `F_s`
- where:
  - `g_s` is the frozen teacher's one-step hidden-state inference field at `z_s`, expressed in normalized-time units compatible with `u_star`
  - `e_out_s = target_onehot - y_hat_s`
  - `F_s` is a per-sample scalar teacher energy at `z_s`

The first learned rescue families should be:

- `interval_ridge_aug`
- `interval_ridge_residual`
- optional `interval_mlp_aug`

Residual-target semantics must remain explicit:

- direct target:
  - `u_star = (z_t - z_s) / (tau_t - tau_s)`
- residual target:
  - `u_res = u_star - g_s`
- reconstructed prediction:
  - `u_hat = g_s + u_res_hat`

Training-distribution rescue must also stay explicit:

- keep the original span-balanced interval sampler available
- add a knot-focused training option that upweights the exact spans used by the acceptance schedules:
  - `2-step`
  - `3-step`
- summaries must make it obvious whether a candidate used:
  - the baseline span-balanced distribution
  - or the knot-focused rescue mix

The Phase 5B pass/fail gate remains unchanged:

- portable exact-checkpoint-backed teacher artifacts
- a learned interval family under a true multi-step schedule
  - `2-step` or `3-step`
  must beat the carried-forward Phase 5A endpoint ridge baseline on both validation and held-out test `final_state_rms_gap`
- the winner's test `energy_gap_to_teacher` must remain within the existing tolerance relative to the carried-forward endpoint ridge baseline

Passing Phase 5B.2 still does not imply:

- MeanFlow identity training has been validated
- JVP objectives have been validated
- refinement is required
- online PC-weight training has been justified

Important caveats that should remain explicit:

- the current toy benchmarks are still simple
- the current search spaces are finite and not exhaustive
- both Phase 2 toy benchmarks still showed boundary sensitivity even though the benchmark-level winners survived the local boundary check
- the matched Phase 2 searches are still single-seed selection procedures, not nested multi-seed model-selection protocols
- the current Phase 3 result is still only a pair of standalone real-data baselines, a narrow PC stabilization pass, and a digest, not a real-data matched-comparison workflow

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

---

## Phase 5B seal-off note

Phase 5B is now considered passed and sealed.

The fresh final validation established all of the following at once:

- portable, relative-path trajectory artifacts
- exact checkpoint reload with numerically negligible teacher-state reproduction error
- a learned interval family under a true multi-step rollout schedule
- `interval_ridge_residual` under `3-step` rollout beating the carried-forward Phase 5A endpoint ridge baseline on both validation and held-out test `final_state_rms_gap`
- no disqualifying energy-gap regression relative to the carried-forward endpoint ridge baseline

This seal-off does not imply that MeanFlow identity or JVP objectives have already been validated; it only justifies opening that next stage.

## Phase 6A validation entry

The next stage should be read narrowly as:

- MeanFlow-style
- teacher-supervised
- average-velocity
- still offline at first

The Phase 6A contract should remain conservative:

- keep the frozen iterative PC teacher from Phase 5B.2
- keep portable relative-path teacher artifacts and exact checkpoint reload
- keep the current-state feature machinery from Phase 5B.2
- add a manual NumPy JVP for the explicit MLP family only
- the initial Phase 6A version treated teacher-derived context features as frozen side information in the JVP path
- the Phase 6A.1 rescue should instead make the teacher-derived current-state feature block feature-aware:
  - compute directional derivatives of current-state teacher features along `g_s`
  - inject those derivatives into the model input tangent
  - explicitly include `d g_s / d tau_s` in the residual MeanFlow identity
- if the feature-aware rescue still fails but the diagnostic linear residual family becomes
  the fresh true multi-step winner, the next allowed rescue remains inside Phase 6A and becomes
  an explicit two-branch residual decomposition:
  - `u_hat = u_local + u_corr`
  - `u_local` is a simple local-dynamics branch anchored to:
    - `g_s`
    - `e_out_s`
    - `F_s`
  - `u_corr` is a neural transport-correction branch using the richer augmented input
  - the MeanFlow identity must still apply to the full reconstructed `u_hat`
- if the two-branch neural family is still conceptually right but loses to the carried-forward
  feature-aware linear winner, the next allowed rescue remains inside Phase 6A and becomes
  warm-started two-branch residual training:
  - warm-start `u_local` from the carried-forward Phase 6A.1 linear residual winner on the same
    fresh exact-checkpoint-backed teacher artifact
  - keep `u_corr` zero-initialized or near-zero-initialized
  - train in explicit stages:
    - Stage A: correction-only warmup with the local branch frozen
    - Stage B: joint hybrid fine-tuning
  - keep the full-`u_hat` teacher anchor
  - keep the full-`u_hat` feature-aware MeanFlow identity
  - keep `d g_s / d tau_s` active inside the residual identity
- keep direct teacher average-velocity supervision as the anchor objective
- when stabilizing MeanFlow training inside Phase 6A, only allow:
  - a teacher-only warmup
  - a simple ramp to nonzero identity weight
  - a late fixed hybrid stage
  - and, if needed, a reduced identity scope on the exact `2-step` / `3-step` acceptance segments

The required comparison set should include:

- `identity`
- carried-forward Phase 5A endpoint ridge baseline
- carried-forward Phase 5B.2 winner:
  - `interval_ridge_residual`
  - `3-step`
- `teacher_only_mlp_aug`
- `meanflow_mlp_aug`
- `meanflow_mlp_residual`
- `meanflow_linear_residual`
- `meanflow_twobranch_residual`
- `meanflow_twobranch_residual_warmstart`

The required rollout schedules remain:

- `1-step`
- `2-step`
- `3-step`

Phase 6A should be considered passed only if all of the following hold under one fresh final validation run:

- teacher artifacts remain portable and exact-checkpoint-backed
- at least one MeanFlow neural family:
  - `meanflow_mlp_aug`
  - or `meanflow_mlp_residual`
  wins under a true multi-step rollout:
  - `2-step`
  - or `3-step`
- that winning candidate beats the sealed Phase 5B.2 winner on both:
  - validation `final_state_rms_gap`
  - held-out test `final_state_rms_gap`
- the winner's held-out test `energy_gap_to_teacher` is not materially worse than the sealed Phase 5B.2 baseline

Important boundary:

- `teacher_only_mlp_aug` is diagnostic only
- `meanflow_linear_residual` is also diagnostic only
- even if it wins, that does not count as a Phase 6A pass
- under Phase 6A.2, `meanflow_twobranch_residual` is the first neural rescue family that may
  count as a passing candidate if it clears the same held-out competitiveness gate
- under Phase 6A.3, `meanflow_twobranch_residual_warmstart` is the next allowed neural rescue
  family; passing still requires a neural winner rather than a carried-forward linear diagnostic
- a rescue inside Phase 6A may change the curriculum or identity scope, but it must remain:
  - teacher-supervised
  - MeanFlow/JVP-based
  - below any teacher-reduction or teacher-free stage

## Phase 6A seal-off note

Phase 6A is now sealed as **not passed**.

The final state of evidence is:

- Phase 6A.1 showed that feature-aware MeanFlow identity is directionally correct
- the carried-forward linear residual diagnostic became the fresh true multi-step winner
- this implies the core identity repair was meaningful
- but no neural MeanFlow family cleared the required competitiveness gate against the sealed
  Phase 5B.2 winner
- the warm-started Phase 6A.3 two-branch neural rescue remained unstable and failed the fresh
  multi-step acceptance run

Therefore:

- Phase 6A artifacts remain readable and portable
- Phase 6A findings remain diagnostically useful
- but Phase 6A does **not** count as a passed stage

## Phase TF1 validation entry

The next stage is now opened as:

- `Phase TF1 — Teacher-free FMPC v1`

This opening should be interpreted conservatively:

- it is a new project decision, not evidence that Phase 6A passed
- the sealed Phase 5B.2 winner remains the strongest passed transport baseline
- any Phase TF1 result should still be compared back to:
  - the carried-forward Phase 5A endpoint ridge baseline
  - the sealed Phase 5B.2 winner
  - and, where relevant, the best diagnostic families from Phase 6A

## Phase TF1 validation contract

This section defines the first teacher-free FMPC validation rule set.

Baseline positioning:

- teacher-based FMPC remains frozen baseline / diagnostic reference
- Phase TF1 is the new main line
- TF1 validation must not depend on teacher trajectories, teacher fixed points,
  or teacher-generated regression targets

Checkpoint selection:

- selection uses validation only
- the concrete selection field is:
  - `selection_metric = "val_transported_final_energy"`
- selector policy is now an explicit part of the TF1 experiment contract
- supported checkpoint selectors are:
  - `energy_only`
  - `val_accuracy_only`
  - `gate_constrained_accuracy_then_energy`
  - `gate_constrained_accuracy_then_val_accuracy`
- TF1 summaries should also expose:
  - `selection_metric_source = "val_metric"`
  - `report_metric_source = "test_metric"`
  - `checkpoint_selector`
  - `selected_epoch_passes_gate`
  - `gate_passing_epoch_count`
  - `selector_fallback_used`
  - `selected_epoch_selection_reason`
- test is report-only and must not participate in checkpoint selection or
  pass/fail gating

Apples-to-apples transport baselines:

- every TF1 validation run must report:
  - `identity/no-transport`
  - `local_field_only`
- these comparisons must use:
  - identical rollout knots
  - identical `transport_steps`
  - identical `theta` snapshot
  - identical split / batch
  - identical energy metric

First-pass TF1 gate:

- the run is fully teacher-free:
  - no teacher manifests
  - no teacher checkpoints
  - no teacher trajectories
  - no teacher-generated regression targets
- focused tests and regressions must pass
- the canonical digits TF1 run must produce complete artifacts with no NaN / Inf
- the selected TF1 checkpoint must satisfy on **validation**:
  - `val_transported_final_energy < val_identity_final_energy`
  - `val_transported_final_energy <= val_local_field_only_final_energy`
  - `val_accuracy > majority baseline`

Report-only expectations:

- final test metrics are still required in `summary.json`
- test results are interpretive and diagnostic only for the first TF1 stage
- canonical TF1 experiment labels should distinguish:
  - `mechanism_smoke`
  - `baseline_comparable`
  - `baseline_working_default`
- `baseline_working_default` is the current evidence-driven but still
  provisional working TF1 preset; it should not be read as a sealed stage pass

## Phase TF1 seal-off note

Phase TF1 is now sealed as the first completed teacher-free FMPC stage.

Sealed TF1 conclusion:

- `baseline_working_default` is the correct sealed TF1 working preset
- it clearly improves over `baseline_comparable`
- it remains the main TF1 preset after selector/default-adoption/external-comparison
  validation
- however, the gap to the canonical slow-PC digits baseline remains materially open
- the narrow TF1 accuracy-tuning pass did not materially narrow that gap

Therefore:

- TF1 is sealed as a successful teacher-free mainline establishment stage
- TF1 is **not** sealed as an accuracy-competitive replacement for the canonical
  slow-PC digits baseline

## Phase TF2 validation entry

The next active stage is now:

- `Phase TF2 - iFMPC bridge stage`

Active-state sync:

- the older teacher-based Phase 4 / 5A / 6A entries above remain sealed
  baseline / diagnostic history
- the active line on `main` remains the teacher-free:
  - `Phase TF2 - iFMPC bridge stage`

Interpretation:

- TF2 builds on the sealed TF1 working preset and selector contract
- TF2 remains fully teacher-free
- TF2 must not depend on JPC runtime availability
- TF2 must be judged against both:
  - the sealed TF1 working default
  - the canonical slow-PC digits baseline

## Phase TF2 validation contract

Bridge-stage scope:

- keep the current layered PC energy substrate
- keep the baseline local parameter-update rule
- add micro-step interleaving during training only
- keep slow-PC predict/eval unchanged
- treat -PC ideas as diagnostics/substrate desiderata only in TF2A

Selector semantics carried forward:

- validation remains the only source of checkpoint selection
- test remains report-only
- TF2 keeps the selector policy logic already established in TF1
- the canonical selector remains:
  - `gate_constrained_accuracy_then_val_accuracy`

Matched-budget requirement:

- TF2 must make `theta_update_budget` explicit in config and summary
- the canonical default is:
  - `theta_update_budget = "matched"`
- under matched-budget incremental updates:
  - normalize by the number of theta updates actually applied under the active
    cadence for that batch
  - `terminal_only` therefore keeps the terminal theta update at the base learning
    rate
  - `every_micro_step` divides by `micro_steps`
  - `every_2_micro_steps` divides by the number of due in-loop theta updates

Must-have acceptance:

- fully teacher-free
- no JPC runtime dependency
- forward-init diagnostics are present in TF2 artifacts
- immediate theta updates happen each micro-step when enabled
- mixed-policy supervision works when enabled
- validation-only selector semantics are preserved
- no NaN / Inf appear in the canonical run or narrow suite

Target acceptance:

- improve mean validation accuracy over the sealed TF1 working default
- improve mean test accuracy over the sealed TF1 working default
- reduce the gap to the canonical slow-PC digits baseline

Required TF2A suite grid:

- `incremental_weight_updates in {false, true}`
- `supervision_policy in {"local_only", "mixed"}`
- `micro_steps in {2, 4}`
- `seeds in {0, 1, 2}`

Keep fixed in the suite:

- `family_lineage = tf1_mlp_aug`
- `feature_aware_tangents = false`
- `identity_loss_weight = 0.2`
- `hybrid_ramp_epochs = 10`
- `bootstrap_substeps = 4`
- `checkpoint_selector = "gate_constrained_accuracy_then_val_accuracy"`
- `theta_update_budget = "matched"`

Current TF2 adoption interpretation:

- `tf2_canonical` remains the hypothesis-driven iFMPC candidate
- `tf2_corrective_transport_default` remains the historical plain corrective working reference
- `tf2_corrective_transport_terminal_angleclip_default` is the currently adopted TF2 experimental default on `main`
- JPC remains reference-only and must not be a runtime dependency for TF2
- the completed JPC probe currently supports prioritizing incremental scheduling
  over substrate scaling
- muPC-style scaling remains a future candidate mechanism, not the current TF2
  mainline
- current TF2 evidence supports corrective transport more strongly than full
  incremental iFMPC / interleaved parameter-learning
- with `use_teacher_free_features = true` and `feature_aware_tangents = false`, the
  current mainline uses an explicit truncated identity approximation that freezes the
  appended feature block inside the JVP path
- the completed matched identity-semantics suite found no material validation-selected
  gain from `feature_aware_tangents = true` in either:
  - `tf2_canonical`
  - `tf2_corrective_transport_default`
- the canonical TF2 default therefore remains:
  - `feature_aware_tangents = false`
- the completed corrective-transport attribution suite now indicates:
  - the current empirical advantage of `tf2_corrective_transport_default` is
    explained primarily by avoiding frequent in-loop theta updates under the
    matched budget
  - `local_only` supervision adds a smaller secondary gain once cadence is already
    `terminal_only`
  - `every_2_micro_steps` and `after_warmup` both partially rescue the canonical
    hypothesis preset, but neither beats the current corrective default
  - `micro_steps = 4` is preferred over `micro_steps = 2` in both the canonical and
    corrective families
  - no tested attribution config narrows the slow-PC test gap below the current
    corrective transport default
- the next narrow TF2 move should therefore remain inside the corrective family:
  - keep `terminal_only`
  - keep `local_only`
  - keep `theta_update_budget = "matched"`
  - probe a slightly larger micro-step count before any broader semantic change
- the completed corrective micro-step horizon suite now indicates:
  - there is no instability through `micro_steps = 10` under the tested
    corrective-family settings
  - under fixed outer training, larger `micro_steps` improve mean validation and
    test accuracy, but this comes with much larger runtime and inner-loop budget
  - under matched inner compute, `micro_steps = 4` remains the best tested setting
    and `6`, `8`, and `10` all degrade:
    - validation-selected test accuracy
    - gate-passing epoch coverage
    - transported validation energy
  - the current evidence therefore reads the `micro_steps > 4` gain as mainly a
    compute-budget effect rather than a genuine transport-horizon gain
  - `micro_steps = 4` should remain the corrective-transport default
  - the next single narrow move should stay inside the current corrective default
    and test transport quality rather than more micro-step compute
- the completed corrective curriculum suite now indicates:
  - no tested bootstrap↔identity curriculum materially improves the current
    fixed-4-step corrective default
  - the best non-default setting is:
    - `identity_loss_weight = 0.1`
    - `warmup_epochs = 5`
    - `hybrid_ramp_epochs = 10`
  - but its mean validation-selected test gain is only about `+0.0015`, with no
    mean validation gain
  - no stage-1 axis winner clears the materiality threshold for opening the
    staged combined-candidate pass
  - `tf2_corrective_transport_default` should therefore keep:
    - `identity_loss_weight = 0.2`
    - `warmup_epochs = 5`
    - `hybrid_ramp_epochs = 10`
  - the next single narrow move should now target bootstrap-target fidelity
    rather than more curriculum tuning
- the completed bootstrap-target fidelity suite now indicates:
  - under a shared local-field / shared-horizon offline probe, higher-fidelity
    `u_boot` candidates do reduce direct target error relative to the
    `rk2_s64` reference
  - however, the current default `rk2_s4` is already close enough that the
    pruned end-to-end candidates:
    - `rk2_s4`
    - `rk2_s8`
    - `rk2_s16`
    produce the same validation-selected accuracy and gate behavior in the
    current multiseed study
  - the non-default candidates only trade substantially higher runtime for
    microscopic validation-energy differences
  - `tf2_corrective_transport_default` should therefore keep:
    - `bootstrap_integrator = "rk2"`
    - `bootstrap_substeps = 4`
  - current evidence says:
    - bootstrap-target fidelity is not the current limiter for fixed-4-step
      corrective transport
  - the next single narrow move should therefore go beyond
    curriculum/bootstrap-fidelity tuning and target a different transport-quality
    bottleneck
- the completed bootstrap-target source-bias suite now indicates:
  - detached slow-PC endpoints can look slightly stronger than the current
    local-field endpoint in the offline diagnostic:
    - lower endpoint hidden energy
    - lower output MSE
    - slightly higher endpoint accuracy
  - the best offline detached challenger in the current tiny family is:
    - `diagnostic_slow_pc_k16`
  - however, the end-to-end comparison between:
    - current local-field source
    - detached slow-PC `K = 16`
    shows no validation-selected accuracy gain and no test-accuracy gain
  - the detached challenger only increases runtime while slightly reducing
    transported validation energy
  - current evidence therefore says:
    - bootstrap-target terminal-source bias is not the current limiter for the
      fixed-4-step corrective default
  - detached slow-PC sources remain:
    - diagnostic-only
    - baseline-only
    - not admissible as the TF2 mainline target source
- the originally logged next single narrow move at that point was:
- psi-side transport expressivity under the fixed teacher-free local-field
  source
- latest local diagnostic chain has since advanced beyond that point through:
  psi-expressivity -> downstream coupling -> lag1 target snapshot -> batch-frozen
  target/state cache -> open-vs-closed-loop trajectory coupling -> partial-open-loop
  handoff localization -> mirrored handoff asymmetry -> terminal-step
  supervision-bundle split -> terminal-step action-output stabilization -> terminal-step direction anchoring
- active narrow local diagnostic question has since moved to:
  terminal local-field direction trust-region in the true closed-loop regime
- active narrow local adoption question has now moved to:
  terminal local-field stabilizer adoption/selection in the true closed-loop regime
- package-level mainline confirmation has now concluded:
  - `tf2_corrective_transport_default` remains the historical corrective working reference
  - `tf2_corrective_transport_terminal_angleclip_default` is the currently adopted TF2 experimental default on `main`
  - closure path:
    terminal-step direction anchoring -> true closed-loop trust-region rescue ->
    adoption/selection -> mainline confirmation
- the completed external comparison / gap-closure pass now indicates:
  - `tf2_corrective_transport_terminal_angleclip_default` materially narrows the
    canonical slow-PC digits test gap relative to the best pre-adoption TF2
    comparator:
    - adopted gap: about `-0.0615`
    - historical corrective gap: about `-0.0763`
    - canonical gap: about `-0.0919`
  - relative to the historical corrective working reference, the adopted default
    gains about:
    - `+0.0059` mean val accuracy
    - `+0.0148` mean test accuracy
  - relative to `tf2_canonical`, the adopted default gains about:
    - `+0.0148` mean val accuracy
    - `+0.0304` mean test accuracy
  - the adopted default keeps:
    - `selected_epoch_passes_gate_rate = 1.0`
    - `selector_fallback_used_rate = 0.0`
  - however, the canonical slow-PC digits baseline still leads by about:
    - `+0.0437` mean val accuracy
    - `+0.0615` mean test accuracy
  - interpretation:
    - `tf2_corrective_transport_default` is now mainly historical
    - `tf2_canonical` is clearly subordinate in the current TF2 phase
    - the next stage should remain:
      - `continue TF2 bridge inside the adopted package`
- the completed adopted-package vs slow-PC gap-decomposition pass now indicates:
  - the external slow-PC gap remains materially open for the adopted package:
    - about `-0.0437` mean val accuracy
    - about `-0.0615` mean test accuracy
  - selector/checkpoint effects are not the current limiter:
    - `selected_epoch_passes_gate_rate = 1.0`
    - `selector_fallback_used_rate = 0.0`
  - diagnostic-only internal supervised decomposition inside the adopted package
    indicates that transport quality is not the main remaining limiter:
    - under the same trained model and target-clamped validation protocol, the
      transported endpoint slightly improves on that model's own slow-PC
      fixed-point output error by about:
      - `-0.0087` val supervised output MSE
      - `-0.0087` test supervised output MSE
    - under the same diagnostic, the transported endpoint also sits slightly
      below that model's own slow-PC fixed-point hidden energy by about:
      - `-0.0036` val supervised final energy
      - `-0.0037` test supervised final energy
  - however, the adopted model's own slow-PC supervised output/readout still
    trails the canonical slow-PC digits baseline by about:
    - `+0.0115` val supervised output MSE
    - `+0.0119` test supervised output MSE
    - `+0.0577` val supervised final energy
    - `+0.0594` test supervised final energy
  - the lightweight validation-knot breakdown for the adopted package still
    peaks at the terminal knot, but the residual internal transport deviation is
    small relative to the remaining external accuracy gap:
    - mean endpoint hidden-state RMS gap to the model's own slow-PC fixed point:
      about `0.0352`
    - mean endpoint output-state RMS gap to the model's own slow-PC outputs:
      about `0.0311`
  - diagnosis:
    - the remaining slow-PC gap is now mainly an output/readout mismatch inside
      the adopted package
    - it is not mainly a selector/checkpoint issue
    - it is not mainly a residual transport-quality issue under the current
      train-time package
  - the completed adopted-package readout-alignment confirmation pass now
    indicates:
    - a minimal output-side alignment aid based on transported readout
      weighting does not materially change the adopted package
    - both `final_micro_step_only` and `every_micro_step` variants are
      indistinguishable from the current adopted default on:
      - validation-selected accuracy
      - test accuracy
      - gate robustness
      - report output MSE
      - supervised transported output MSE
      - internal slow-PC output MSE gap
    - no readout-alignment candidate clears the promotion threshold
    - current evidence therefore does not support adopting a transported
      readout-alignment weight as the next TF2 package change
  - next single narrow move:
    - shift away from this specific readout-weighting aid and target a
      different remaining issue inside the adopted package
  - the completed adopted-package readout-refit / endpoint-separability pass
    now indicates:
    - transported-endpoint readout refit is not a rescue:
      - on the frozen transported basis, the refit head drives supervised
        output MSE down to about `0.0017`
      - but integrated prediction-mode behavior degrades sharply to about:
        - `0.4844` mean val accuracy
        - `0.4770` mean test accuracy
    - slow-PC-endpoint readout refit does materially improve integrated
      behavior over both the adopted control and the transported-endpoint
      refit:
      - about `+0.1133` mean val accuracy vs adopted control
      - about `+0.1244` mean test accuracy vs adopted control
      - about `+0.4644` mean val accuracy vs transported-endpoint refit
      - about `+0.4815` mean test accuracy vs transported-endpoint refit
    - diagnosis:
      - the remaining adopted-package readout mismatch is now best explained as
        an endpoint-basis / representation mismatch between transported
        endpoints and that model's own slow-PC endpoints
      - it is not mainly a simple transported-head fitting problem
  - next single narrow move:
    - run one adopted-package endpoint-basis / separability diagnostic at the
      hidden-to-output interface, without changing the TF2 transport family
  - the completed adopted-package endpoint-basis / separability suite now
    indicates:
    - the remaining mismatch is not a simple interface-separability deficit
    - under the current frozen head, transported endpoints are actually more
      separable than the same model's own slow-PC endpoints:
      - validation frozen-head accuracy:
        - transported about `0.9348`
        - slow-PC about `0.8407`
      - validation frozen-head output MSE:
        - transported about `0.0517`
        - slow-PC about `0.0605`
      - validation between-class centroid margin:
        - transported about `0.6334`
        - slow-PC about `0.4700`
      - validation Fisher separability ratio:
        - transported about `1.8067`
        - slow-PC about `1.5028`
    - within-class spread only increases slightly under transported endpoints:
      - validation mean within-class RMS delta:
        - about `+0.0226`
    - the dominant difference is instead a readout-relevant basis shift:
      - per-sample endpoint delta row-space RMS fraction:
        - about `0.5448`
      - class-centroid displacement row-space fraction:
        - about `0.5657`
    - validation knot-wise geometry does not show a separability collapse:
      - Fisher separability rises monotonically across knots
      - nearest-centroid accuracy rises monotonically across knots
      - total endpoint divergence from the model's own slow-PC endpoints grows
        monotonically and is largest at the terminal knot
  - diagnosis:
    - the remaining adopted-package mismatch is now best explained as:
      - distortion in the readout-relevant row-space
    - it is not mainly:
      - reduced between-class margin
      - inflated within-class spread
  - next single narrow move:
    - run one adopted-package readout-sensitive / output-sensitive terminal
      direction diagnostic inside the current package
  - the completed adopted-package output-sensitive terminal direction suite now
    indicates:
    - isolating the terminal intervention to the readout row-space does not
      improve the adopted package
    - both tested row-space-sensitive candidates underperform the current
      adopted default:
      - row-space-only angle clip:
        - about `0.7963` mean val accuracy
        - about `0.7896` mean test accuracy
        - vs adopted control:
          - `-0.0393` val accuracy
          - `-0.0444` test accuracy
      - row-space-only hard replace upper bound:
        - about `0.8104` mean val accuracy
        - about `0.8148` mean test accuracy
        - vs adopted control:
          - `-0.0252` val accuracy
          - `-0.0193` test accuracy
    - both candidates also worsen the targeted terminal row-space distortion:
      - adopted control validation row-space fraction:
        - about `0.5448`
      - row-space-only angle clip validation row-space fraction:
        - about `0.5949`
      - row-space-only hard replace validation row-space fraction:
        - about `0.5931`
      - both candidates increase validation row-space RMS by about:
        - `+0.0115` to `+0.0122`
    - gate robustness does not collapse:
      - `selected_epoch_passes_gate_rate = 1.0`
      - `selector_fallback_used_rate = 0.0`
      across the tested candidates
  - decision:
    - keep the current adopted TF2 experimental default unchanged:
      - `tf2_corrective_transport_terminal_angleclip_default`
  - next single narrow move:
    - run one adopted-package terminal row-space / orthogonal-component
      coupling diagnostic, because the gain of the current full-vector angle
      clip does not survive row-space-only isolation
  - the completed adopted-package terminal row-space / orthogonal-component
    coupling suite now indicates:
    - neither decomposed intervention matches the current adopted control
    - row-space-only angle clip underperforms by about:
      - `-0.0393` mean val accuracy
      - `-0.0444` mean test accuracy
    - orthogonal-only angle clip also underperforms by about:
      - `-0.0444` mean val accuracy
      - `-0.0289` mean test accuracy
    - gate robustness does not collapse for either decomposed candidate:
      - `selected_epoch_passes_gate_rate = 1.0`
      - `selector_fallback_used_rate = 0.0`
    - both decomposed variants also worsen the targeted row-space distortion:
      - adopted control validation row-space RMS:
        - about `0.1536`
      - row-space-only validation row-space RMS:
        - about `0.1651`
      - orthogonal-only validation row-space RMS:
        - about `0.1683`
      - adopted control validation row-space fraction:
        - about `0.5448`
      - row-space-only validation row-space fraction:
        - about `0.5949`
      - orthogonal-only validation row-space fraction:
        - about `0.6168`
    - both decomposed variants reduce total endpoint RMS slightly, but they do
      so while increasing the fraction of mismatch that lands in the
      readout-relevant row-space
  - diagnosis:
    - the gain of the current full-vector terminal angle clip is now best read
      as a coupled row-space / orthogonal control effect
    - neither subspace alone is an adequate replacement
  - decision:
    - keep the current adopted TF2 experimental default unchanged:
      - `tf2_corrective_transport_terminal_angleclip_default`
  - next single narrow move:
    - run one adopted-package split-threshold terminal coupling diagnostic that
      keeps both row-space and orthogonal components active but tests whether
      their clip strengths should differ
  - the completed adopted-package split-threshold terminal coupling suite now
    indicates:
    - no split-threshold candidate beats the current adopted control
    - stricter row-space clip than orthogonal clip underperforms by about:
      - `-0.0326` mean val accuracy
      - `-0.0311` mean test accuracy
    - stricter orthogonal clip than row-space clip also underperforms by about:
      - `-0.0407` mean val accuracy
      - `-0.0304` mean test accuracy
    - the balanced `30 / 30` split-threshold sanity check also underperforms by
      about:
      - `-0.0393` mean val accuracy
      - `-0.0444` mean test accuracy
    - gate robustness remains intact across the tested split-threshold
      variants:
      - `selected_epoch_passes_gate_rate = 1.0`
      - `selector_fallback_used_rate = 0.0`
    - but all tested split-threshold variants worsen the targeted row-space
      distortion metrics:
      - adopted control validation row-space RMS:
        - about `0.1536`
      - row-strict split-threshold validation row-space RMS:
        - about `0.1657`
      - orth-strict split-threshold validation row-space RMS:
        - about `0.1662`
      - balanced `30 / 30` split-threshold validation row-space RMS:
        - about `0.1651`
      - adopted control validation row-space fraction:
        - about `0.5448`
      - row-strict split-threshold validation row-space fraction:
        - about `0.5989`
      - orth-strict split-threshold validation row-space fraction:
        - about `0.6053`
      - balanced `30 / 30` split-threshold validation row-space fraction:
        - about `0.5949`
    - all tested split-threshold variants reduce total endpoint RMS slightly,
      but only by shifting a larger fraction of the mismatch into the
      readout-relevant row-space
  - diagnosis:
    - the gain of the current full-vector terminal angle clip does not seem to
      be recoverable by keeping both subspaces active with different independent
      clip strengths
    - the current evidence therefore favors the unified full-vector cone
      geometry over the tested split-threshold decompositions
  - decision:
    - keep the current adopted TF2 experimental default unchanged:
      - `tf2_corrective_transport_terminal_angleclip_default`
  - next single narrow move:
    - run one adopted-package unified-cone vs split-subspace cone geometry
      diagnostic to explain why the full-vector terminal clip dominates all
      tested decomposed cone variants
  - the completed adopted-package unified-cone vs split-subspace cone geometry
    suite now indicates:
    - no split-subspace cone matches the current adopted control
    - the best split candidate in this tiny family:
      - row-strict `20 / 45`
      still underperforms by about:
      - `-0.0326` mean val accuracy
      - `-0.0311` mean test accuracy
    - every tested split-subspace cone leaves the stabilized full-space terminal
      action far outside the adopted `30` degree cone:
      - adopted control mean terminal full-space angle to the local-field
        anchor:
        - about `30.0` degrees
      - best split candidate:
        - about `58.99` degrees
      - all tested split candidates have:
        - full-space angle-above-`30` degree rate `= 1.0`
      - adopted control has:
        - full-space angle-above-`30` degree rate `= 0.0`
    - the split-subspace cones often preserve subspace-local norm ratios more
      literally than the adopted control, but that does not rescue behavior:
      - adopted control validation row/orth norm-ratio absolute change:
        - about `1.3657`
      - all tested split-subspace cases:
        - about `0.0`
    - current evidence therefore says:
      - the useful property is not literal subspace ratio preservation
      - the useful property is the shared full-space angular constraint of the
        unified terminal cone
      - that constraint does not factorize into the tested row-space /
        orthogonal sub-cones
  - diagnosis:
    - the gain of the adopted full-vector terminal angle clip is now best read
      as:
      - `shared_full_space_geometry_not_literal_subspace_ratio_preservation`
  - decision:
    - keep the current adopted TF2 experimental default unchanged:
      - `tf2_corrective_transport_terminal_angleclip_default`
  - next single narrow move:
    - run one narrow geometry-preserving unified-cone-shape diagnostic inside
      the adopted full-vector family, rather than another split-subspace sweep
  - the completed adopted-package unified-cone-shape suite now indicates:
    - no refined unified full-space cone shape is yet strong enough to replace
      the current adopted control
    - the stricter interior-margin `20` degree unified cone is the only tested
      candidate that improves validation-selected behavior:
      - `+0.0059` mean val accuracy
      - `+0.0074` mean test accuracy
    - the same `20` degree interior-margin candidate also improves the targeted
      row-space distortion metrics:
      - validation row-space RMS:
        - about `0.1536 -> 0.1508`
      - validation row-space fraction:
        - about `0.5448 -> 0.5335`
      - validation full-space terminal angle to anchor:
        - about `30.0 -> 20.0` degrees
      - validation `> 30` degree violation rate remains:
        - `0.0`
    - however, the `20` degree interior-margin candidate loses gate robustness:
      - mean gate-passing epoch count:
        - about `16.8 -> 11.2`
      - `selected_epoch_passes_gate_rate`:
        - about `1.0 -> 0.8`
      - `selector_fallback_used_rate`:
        - about `0.0 -> 0.2`
    - the relaxed-boundary `40` degree unified cone does not help:
      - `-0.0089` mean val accuracy
      - `-0.0104` mean test accuracy
      - validation row-space fraction worsens to about:
        - `0.5575`
      - validation `> 30` degree violation rate becomes:
        - `1.0`
  - diagnosis:
    - the current `30` degree hard full-vector cone remains the best-supported
      adopted default after accuracy-versus-robustness tradeoff
    - the `20` degree interior-margin variant is promising on accuracy and
      row-space distortion, but not yet safe on gate robustness
  - decision:
    - keep the current adopted TF2 experimental default unchanged:
      - `tf2_corrective_transport_terminal_angleclip_default`
  - next single narrow move:
    - if TF2 work continues inside the adopted package, test at most one smooth
      unified-cone projection variant that tries to keep the interior-margin
      gain without paying the current gate-robustness cost
  - the completed adopted-package smooth unified-cone suite now indicates:
    - the smooth full-space cone remains non-adopted as a replacement default
    - relative to the current adopted hard `30` degree cone, the smooth
      candidate improves validation-selected behavior and row-space distortion:
      - `+0.0037` mean val accuracy
      - `+0.0074` mean test accuracy
      - validation row-space RMS:
        - about `0.1536 -> 0.1517`
      - validation row-space fraction:
        - about `0.5448 -> 0.5376`
      - validation full-space angle to anchor:
        - about `30.0 -> 24.1` degrees
      - validation `> 30` degree violation rate remains:
        - `0.0`
    - relative to the old hard `20` degree interior-margin reference, the
      smooth candidate partially recovers robustness:
      - mean gate-passing epoch count:
        - about `11.2 -> 12.4`
      - `selected_epoch_passes_gate_rate`:
        - about `0.8 -> 1.0`
      - `selector_fallback_used_rate`:
        - about `0.2 -> 0.0`
    - but the smooth candidate still does not match the current adopted hard
      `30` degree cone on gate robustness:
      - mean gate-passing epoch count:
        - about `16.8 -> 12.4`
  - diagnosis:
    - the smooth unified-cone projection improves accuracy and row-space
      distortion, but not enough to replace the current hard `30` degree
      adopted control
  - decision:
    - keep the current adopted TF2 experimental default unchanged:
      - `tf2_corrective_transport_terminal_angleclip_default`
  - next single narrow move:
    - if TF2 work continues inside the adopted package, treat this unified-cone
      shape family as locally saturated and avoid another cone-shape sweep
      unless a confirmation-level reason appears
  - the completed adopted-package unified-cone robustness suite now indicates:
    - this confirmation-level follow-up does not justify further work inside the
      unified-cone family
    - both the hard `20` degree interior-margin reference and the smooth
      unified-cone reference remain blocked mainly by:
      - `systematic_threshold_margin_collapse_mainly_on_energy_side`
    - this is not primarily an accuracy-threshold issue:
      - both non-adopted variants keep near-identical negative accuracy-margin
        epoch fractions relative to the adopted control:
        - about `0.0067`
    - it is mainly an energy-side gate-margin issue:
      - adopted control negative gate-energy-margin epoch fraction:
        - about `0.72`
      - hard `20` reference:
        - about `0.8133`
      - smooth unified-cone reference:
        - about `0.7933`
      - gate-failure attribution remains overwhelmingly energy-side for all
        three cases:
        - about `0.99` energy-only
        - about `0.01` both
        - about `0.0` accuracy-only
    - it is also not well explained by higher temporal volatility:
      - adopted control mean validation-accuracy volatility:
        - about `0.0293`
      - hard `20` reference:
        - about `0.0266`
      - smooth unified-cone reference:
        - about `0.0266`
      - adopted control mean validation-energy volatility:
        - about `0.00702`
      - hard `20` reference:
        - about `0.00694`
      - smooth unified-cone reference:
        - about `0.00693`
    - behavior-wise, both non-adopted variants still lose gate coverage:
      - adopted control mean gate-passing epoch count:
        - `16.8`
      - hard `20` reference:
        - `11.2`
      - smooth unified-cone reference:
        - `12.4`
      - hard `20` `seed_gate_positive_rate`:
        - `0.8`
      - smooth unified-cone `seed_gate_positive_rate`:
        - `1.0`
      - adopted control `seed_gate_positive_rate`:
        - `1.0`
  - diagnosis:
    - the unified-cone family is now best treated as locally saturated under
      the fixed gate and selector contract
  - decision:
    - do not run another unified-cone confirmation inside this family
    - keep the current adopted TF2 experimental default unchanged:
      - `tf2_corrective_transport_terminal_angleclip_default`
  - next single narrow move:
    - move to a different remaining package-internal issue rather than another
      unified-cone follow-up
  - the completed adopted-package late-rollout basis-drift source-localization
    suite now indicates:
    - the remaining readout-relevant basis drift is mainly injected by
      preterminal rollout accumulation rather than terminal jump injection
    - validation mean preterminal contribution share is about:
      - `0.7447`
    - test mean preterminal contribution share is about:
      - `0.7451`
  - diagnosis:
    - `preterminal_accumulation_dominates`
  - next single narrow move:
    - run one narrow adopted-package late-rollout drift-control diagnostic next
  - the completed adopted-package late-rollout drift-control suite now indicates:
    - earlier same-geometry full-vector `30` degree control does reduce terminal
      row-space drift and improves validation-selected behavior
    - but the penultimate-plus-terminal and last-two-preterminal-plus-terminal
      variants both collapse gate coverage to `0.0`
    - therefore this is not an adoption path under the current gate / selector
      contract
  - diagnosis:
    - `mixed_result_but_not_adoption_level`
  - next single narrow move:
    - run one narrow source-localization diagnostic on the preterminal update
      formulation itself rather than another cone-family sweep
  - the completed adopted-package preterminal-update source-localization suite
    now indicates:
    - the failed earlier-control gate collapse is best localized to the
      preterminal on-policy handoff state rather than to the preterminal
      direction source or the preterminal norm handling
    - swapping only the preterminal on-policy handoff back to the cached
      batch-start successor restores:
      - `seed_gate_positive_rate: 0.0 -> 1.0`
      - `selected_epoch_passes_gate_rate: 0.0 -> 1.0`
      - `selector_fallback_used_rate: 1.0 -> 0.0`
    - swapping only the preterminal direction source to on-policy live local
      field keeps:
      - mean gate-passing epoch count at `0.0`
    - swapping only the preterminal norm handling to anchor norm also keeps:
      - mean gate-passing epoch count at `0.0`
  - diagnosis:
    - `preterminal_handoff_state_is_primary_blocker`
  - next single narrow move:
    - run one narrow adopted-package confirmation on the smallest preterminal
      on-policy handoff reformulation that preserves the current selector/gate
      contract
  - the completed adopted-package preterminal handoff reformulation
    confirmation now says:
    - the minimal cached-handoff reformulation restores the fixed selector/gate
      contract to the adopted-control level
    - but it recovers only about `3.4%` of the failed earlier-control
      validation-accuracy gain and essentially none of the earlier terminal
      row-space RMS gain
    - the resulting confirmation-level diagnosis is:
      - `handoff_reformulation_recovers_partially_but_not_adoption_level`
  - next single narrow move:
    - run one narrower handoff-state source-localization step on the
      preterminal successor handoff itself rather than another cone-family
      sweep
  - the completed adopted-package successor-handoff source-localization suite
    now says:
    - swapping only preterminal `z_on_next` back to the cached batch-start
      successor fully restores the gate contract but collapses back to near the
      adopted-control behavior
    - swapping only preterminal `z_lf_next` back to the cached batch-start
      successor leaves the failed earlier-control reference unchanged
    - swapping both successor components back to cached matches the
      `z_on_next`-only result, so the remaining blocker is not a cross-source
      successor inconsistency
    - the resulting diagnosis is:
      - `stale_successor_value_is_primary_blocker`
  - next single narrow move:
    - run one confirmation-level reformulation on the preterminal on-policy
      successor-value component only, without reopening any cone-family sweep
  - the completed adopted-package preterminal successor-value confirmation now
    says:
    - the best narrow candidate is the fixed `25%` live / `75%` cached
      successor-value blend
    - it preserves the full selector/gate contract and improves validation
      accuracy plus terminal row-space metrics over the current adopted control
    - but it retains only about `31%` of the failed earlier-control
      validation-accuracy gain and about `24.5%` of the failed earlier-control
      terminal row-space RMS gain
    - the stronger fixed `50%` live / `50%` cached blend recovers more
      geometry but gives up too much gate robustness
    - the resulting diagnosis is:
      - `successor_value_reformulation_recovers_partially_but_not_adoption_level`
  - next single narrow move:
    - run one smaller confirmation-level follow-up on the same preterminal
      successor-value component only, using the best narrow reformulation
      candidate from this pass
  - the completed low-live successor-value follow-up now says:
    - the safer `20%` live / `80%` cached blend keeps the full selector/gate
      contract but is weaker than the `25%` live / `75%` cached anchor on both
      gain retention and terminal row-space recovery
    - the more aggressive `30%` live / `70%` cached blend retains more of the
      earlier-control gain than the `25/75` anchor
    - but `30/70` gives up full gate robustness:
      - `seed_gate_positive_rate: 0.8`
      - `selected_epoch_passes_gate_rate: 0.8`
      - `selector_fallback_used_rate: 0.2`
    - the resulting diagnosis is:
      - `local_successor_value_refinement_improves_but_not_to_adoption_level`
  - next single narrow move:
    - move to a deeper diagnostic on the live preterminal successor-value
      formulation itself rather than another low-live blend sweep
  - the completed successor-value carry-vs-increment source-localization now
    says:
    - `live carry + cached increment` restores the full selector/gate contract
      but collapses back to near-control accuracy and terminal row-space metrics
    - `cached carry + live increment` remains almost identical to the failed
      higher-gain reference and keeps the gate collapse
    - the resulting diagnosis is:
      - `live_successor_increment_is_primary_blocker`
  - next single narrow move:
    - run one confirmation-level reformulation on the preterminal successor
      increment only
  - the completed preterminal successor-increment confirmation now says:
    - the increment-only direction trust-region toward the cached increment
      keeps much more of the failed earlier-control gain than the safe
      cached-increment lower bound:
      - validation-accuracy retention vs failed anchor: about `75.9%`
      - terminal row-space RMS retention vs failed anchor: about `65.8%`
    - but it still reopens too much of the gate collapse:
      - `seed_gate_positive_rate: 0.4`
      - `selected_epoch_passes_gate_rate: 0.4`
      - `selector_fallback_used_rate: 0.6`
    - the resulting diagnosis is:
      - `live_successor_increment_blocker_persists`
  - next single narrow move:
    - run the next narrower increment-internal source-localization step
  - the completed increment-internal source-localization now says:
    - `exact cached direction + live magnitude` restores the full
      selector/gate contract but collapses back to near-control accuracy and
      terminal row-space metrics
    - `cached magnitude + live direction` remains almost identical to the
      failed higher-gain unstable reference and keeps the gate collapse
    - the resulting diagnosis is:
      - `live_successor_increment_direction_is_primary_blocker`
  - next single narrow move:
    - run one confirmation-level reformulation on increment direction only
  - the completed increment-direction confirmation now says:
    - the weaker `45` degree trust-region keeps more of the failed
      earlier-control gain than the current `30` degree partial-signal
      reference, but it fully reopens the gate collapse
    - the stronger `20` degree trust-region partially recovers gate
      robustness:
      - `seed_gate_positive_rate: 0.8`
      - `selected_epoch_passes_gate_rate: 0.8`
      - `selector_fallback_used_rate: 0.2`
      - but it still does not keep the full selector/gate contract intact
    - the resulting diagnosis is:
      - `live_successor_increment_direction_blocker_persists`
  - next single narrow move:
    - run one minimal direction-magnitude interaction diagnostic next
  - the completed direction-magnitude interaction diagnostic now says:
    - `30` degree direction trust-region + cached magnitude is effectively
      identical to the current `30` degree direction-only reference:
      - validation-accuracy retention vs failed anchor: about `75.9%`
      - gate-robustness recovery vs control: about `25%`
      - terminal row-space RMS retention vs failed anchor: about `65.9%`
    - `20` degree direction trust-region + cached magnitude is effectively
      identical to the current `20` degree direction-only reference:
      - validation-accuracy retention vs failed anchor: about `44.8%`
      - gate-robustness recovery vs control: about `54.8%`
      - terminal row-space RMS retention vs failed anchor: about `42.0%`
    - neither interaction candidate keeps the full selector/gate contract intact
  - the resulting diagnosis is:
    - `live_successor_increment_interaction_blocker_persists`
  - next single narrow move:
    - run one deeper diagnostic on the live successor increment formulation
      itself rather than another broader successor-value, interaction, or
      cone-family sweep
  - the completed deeper live successor-increment formulation diagnostic now
    says:
    - swapping only the learned residual term to the cached analogue while
      keeping the live detached local-field anchor term leaves behavior
      effectively identical to the failed live/live earlier-control reference:
      - `mean_val_accuracy: 0.8570`
      - `mean_gate_passing_epoch_count: 0.0`
      - `selector_fallback_used_rate: 1.0`
      - `mean_val_terminal_rowspace_rms: 0.1425`
    - swapping only the detached local-field anchor term to the cached
      analogue while keeping the live learned residual term also leaves
      behavior effectively identical to the failed live/live earlier-control
      reference:
      - `mean_val_accuracy: 0.8570`
      - `mean_gate_passing_epoch_count: 0.0`
      - `selector_fallback_used_rate: 1.0`
      - `mean_val_terminal_rowspace_rms: 0.1425`
    - neither single-term substitution restores any gate robustness, and both
      retain essentially `100%` of the failed-anchor accuracy / row-space gain
  - the resulting diagnosis is:
    - `bad_live_direction_source_not_yet_localized_but_formulation_blocker_strengthened`
  - next single narrow move:
    - run one very narrow remaining-issue triage pass before creating any
      different adopted-package diagnostic
  - the completed remaining-issue triage now says:
    - no different package-internal issue survives as a credible next
      adopted-package diagnostic under the current selector/gate contract
    - the adopted-package readout-alignment confirmation is exactly a no-op:
      - `readout_align_final_w050` and `readout_align_every_w050` remain
        identical to the adopted control on validation accuracy, test accuracy,
        gate counts, transported energy, and report-output MSE
    - the older bootstrap-source-bias, one-step target-lag, and
      bootstrap↔identity curriculum families do not provide a credible reopen
      path from the current adopted-package state:
      - detached slow-PC bootstrap source did not materially beat the
        local-field source end-to-end
      - one-step lagged target snapshots did not improve behavior
      - no tested bootstrap↔identity curriculum materially beat the fixed-4-step
        corrective default
    - combined with the strengthened live successor-increment formulation
      blocker, the adopted corrective TF2 package is now best treated as
      locally saturated under the current selector/gate contract
  - the resulting recommendation is:
    - stop package-internal TF2 digging from this state
    - only reopen TF2 mainline digging if a genuinely different issue family
      appears from new evidence or if the project explicitly decides to leave
      the current package / selector-gate contract
  - the post-triage decision memo is:
    - package-internal TF2 digging is closed from the current state because the
      adopted package has already exhausted the credible internal issue families
      surfaced by the evidence chain:
      - readout alignment is an exact no-op on the adopted package
      - detached bootstrap-source, one-step target-lag, and bootstrap-to-identity
        curriculum follow-ups do not materially improve the adopted package
      - the late-rollout successor-value and successor-increment line now ends
        in a strengthened formulation-level blocker rather than an
        adoption-viable local repair
    - the recommended next project move is:
      - keep the current adopted TF2 package frozen as the bridge result on `main`
      - move the next planning effort to a post-TF2 teacher-free FMPC / EF
        exploratory line that explicitly leaves the current corrective package
        or selector-gate contract
      - do not reframe that exploratory line as replacing the current active
        `Phase TF2 - iFMPC bridge stage` on `main` until it is explicitly
        chartered

Required reporting:

- per-run outputs must include:
  - `incremental_weight_updates`
  - `supervision_policy`
  - `micro_steps`
  - `theta_update_budget`
  - `theta_micro_lr`
  - `theta_micro_bias_lr`
  - `val_accuracy`
  - `test_accuracy`
  - `gate_passing_epoch_count`
  - `val_transported_final_energy`
  - `selected_epoch`
  - `selected_epoch_passes_gate`
  - `selector_fallback_used`
  - `forward_init_stability_metrics`
- aggregate summary must include:
  - mean/std validation accuracy by config
  - mean/std test accuracy by config
  - mean gate-passing epoch count by config
  - pairwise comparison against the sealed TF1 working default
  - pairwise gap to the canonical slow-PC digits baseline
  - whether mixed-policy supervision helps
  - whether incremental theta updates help
  - whether matched-budget TF2 narrows the slow-PC gap materially
  - recommended next stage
