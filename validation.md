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
- does any simple learned student beat the explicit identity / zero-delta baseline on the canonical non-trivial `digits` teacher?

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

Before Phase 6A is considered passed, it should outperform the sealed Phase 5B interval baseline under a fresh validation protocol rather than by reusing older endpoint-only gates.
