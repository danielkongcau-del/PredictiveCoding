# PLANS.md

This document tracks the staged execution plan for the repository.

The purpose is not to predict every future detail, but to prevent large unfocused implementation jumps.

---

## Phase 0 — Minimal mathematically faithful baseline

### Goal

Implement a small supervised predictive coding network that is demonstrably correct on toy problems.

### Scope

- fully connected network only
- batch-first NumPy implementation
- hidden-state iterative inference
- local weight updates from the current spec
- toy regression or toy classification
- basic logging and tests

### Deliverables

- core parameter container(s)
- activation functions + derivatives
- energy computation
- inference loop
- local parameter updates
- a `PCNetwork` or equivalent minimal model wrapper
- one toy training script
- one smoke test proving inference lowers energy on a simple case

### Exit criteria

- can train on a tiny synthetic dataset without runtime errors
- energy is tracked per inference step
- at least 5 tests pass
- code path matches `spec_math.md`

### Risks

- sign errors in state updates
- shape confusion between batch-major and feature-major math
- inference instability from poor step sizes

---

## Phase 1 — Stabilization and engineering cleanup

### Goal

Turn the minimal baseline into a reusable small research codebase.

### Scope

- cleaner module split
- config handling
- deterministic seeding utilities
- better logging
- more tests
- experiment output saving
- richer docstrings

### Deliverables

- cleaned `src/pc/` module boundaries
- reproducible experiment script layout
- saved plots / metrics to `outputs/`
- regression tests for previously fixed bugs

### Exit criteria

- repeated runs under fixed seed are reproducible within tolerance
- shape and energy tests are stable
- experiment scripts are easy to run from command line

### Risks

- excessive refactoring before math is stable
- accidental behavior changes during cleanup

---

## Phase 2 — Numerical validation and baseline comparisons

### Goal

Establish stronger confidence that the implementation behaves as intended.

### Scope

- finite-difference checks on the energy where feasible
- ablations on inference steps and step size
- simple baseline comparison against a standard MLP on toy data
- failure-case diagnostics

### Deliverables

- numerical check utilities
- plots showing energy trajectories
- comparison notebook or script
- documented expected hyperparameter ranges

### Exit criteria

- local update rules pass small numerical sanity checks where applicable
- failure modes are documented rather than mysterious
- baseline comparisons are reproducible

### Risks

- confusing exact backprop equivalence with approximate behavior
- over-interpreting noisy toy results

---

## Phase 2f — Generalization-aware evaluation and joint tuning

Status:

- Completed as a protocol-hardening phase
- Superseded for final comparative claims by the fairer Phase 2g matched-search workflow

### Goal

Reduce methodological confounds in the current Phase 2 conclusions by introducing a small, explicit evaluation protocol that separates tuning from final comparison.

### Scope

- keep the current NumPy-only PC baseline and current minimal MLP baseline
- keep the current toy benchmarks as the immediate evaluation domain
- add explicit train / validation / final-eval protocol support for benchmark comparisons
- support small fixed joint tuning for both PC and MLP under the same evaluation rules
- refresh downstream PC-vs-MLP comparison artifacts using the new protocol
- keep search spaces narrow and deterministic
- do not introduce a generic tuning framework or change the baseline math

### Deliverables

- split-aware benchmark protocol definitions for the current comparison tasks
- a small joint-tuning runner that can evaluate both PC and MLP under the same protocol
- refreshed comparison summaries produced from the new protocol rather than single fixed settings alone
- documentation of the exact tuning budget, validation criterion, and final evaluation rule
- regression tests for artifact generation, protocol correctness, and deterministic behavior

### Exit criteria

- the repository can run a reproducible comparison with explicit tuning and held-out final evaluation
- PC and MLP are both tuned under the same narrow, documented protocol
- refreshed comparison outputs clearly separate:
  - tuning criterion
  - selected configuration
  - final evaluation metric
- the resulting conclusions are traceable to saved artifacts rather than single-run defaults

### Risks

- expanding the scope into a generic experiment-management or search framework
- creating data leakage between tuning and final evaluation
- overfitting conclusions to tiny validation splits on the current toy benchmarks
- interpreting better protocol discipline as evidence of a new model capability

### Initial implementation plan

Files likely to touch:

- `PLANS.md`
- `README.md`
- `RESULTS.md`
- `src/pc/benchmark_specs.py`
- `src/pc/comparison.py`
- `src/pc/pc_multiseed.py`
- one new narrow protocol module such as `src/pc/evaluation_protocol.py`
- one new narrow runner such as `src/pc/joint_tuning.py`
- one new experiment entry script such as `experiments/joint_tuning_compare.py`
- focused tests for protocol and artifact generation

Tests that should verify success:

- deterministic split generation under fixed seeds
- artifact generation for the new protocol runner
- protocol correctness:
  - validation metric is used for model/config selection
  - final evaluation metric is written separately
- metric-direction correctness when selecting the best tuned configuration
- reproducibility across repeated runs with the same explicit seeds

Assumptions:

- Phase 0 predictive-coding math remains unchanged
- the immediate Phase 2f target is methodology, not a new model family
- the current toy benchmarks are still useful as a first-pass protocol testbed
- any search space must remain small, explicit, and comparable across PC and MLP

---

## Phase 2g — Matched validation-selected PC/MLP search

Status:

- Completed as the current end-of-Phase-2 comparison protocol
- Current strongest repository-level comparison claims should be grounded in the Phase 2g matched-search artifacts together with the Phase 2g.1 closure pass and downstream refreshes, rather than earlier train-only or train/eval-style conclusions

### Goal

Make the PC-vs-MLP comparison fairer by giving both baselines matched small-scope tuning, selecting configurations on validation performance, and reporting the headline result on held-out test performance.

### Scope

- keep the NumPy-only predictive-coding baseline and current minimal MLP baseline
- keep the existing toy benchmarks as the current experimental domain
- keep the Phase 2f train/val/test protocol
- add a deterministic small PC search space
- add a deterministic small MLP search space
- refresh downstream multiseed and budget-tradeoff studies so they consume the selected Phase 2g PC and MLP configs
- keep search spaces modest, explicit, and interpretable
- do not introduce a generic HPO framework or alter the baseline math

### Deliverables

- a matched-search runner that searches both PC and MLP on the same protocol
- per-benchmark best-config summaries for PC and MLP
- aggregate matched-search artifacts that explicitly separate:
  - selection metric source
  - final report metric source
  - selected PC config
  - selected MLP config
  - held-out test winner
- refreshed multiseed studies for the selected Phase 2g configs
- refreshed budget-tradeoff studies sourced from the selected Phase 2g PC and MLP configs
- documentation updates that distinguish:
  - legacy Phase 2 conclusions
  - Phase 2f conclusions
  - Phase 2g conclusions
  - Phase 2g.1 conclusions

### Exit criteria

- PC and MLP are both tuned under a small matched validation-selected search
- best-config selection is driven by `val_metric`, not `test_metric`
- final headline comparisons are reported on held-out `test_metric`
- downstream multiseed and budget studies use the selected Phase 2g configs
- artifacts make it immediately obvious:
  - how configs were selected
  - which split determined selection
  - which split determined the final headline result
  - which PC and MLP configs were selected
  - how they compare on held-out test

### Risks

- overclaiming from simple toy benchmarks
- finite search spaces being mistaken for exhaustive tuning
- single-seed search selection overstating robustness of the chosen configs
- mixing legacy, Phase 2f, and Phase 2g outputs when interpreting conclusions
- reading budget-tradeoff results as wall-clock efficiency evidence rather than inference-budget evidence

### Current takeaway

- `toy_regression`: the matched-tuned PC baseline beats the matched-tuned MLP baseline on held-out test and remains ahead across the current multi-seed check
- `toy_sine_regression`: the matched-tuned MLP baseline beats the matched-tuned PC baseline on held-out test and remains ahead on most seeds in the current multi-seed check
- extra PC inference budget no longer helps on `toy_regression`
- extra PC inference budget helps partially on `toy_sine_regression`, but the best current budget still trails the selected MLP baseline on held-out test
- local output-retention policy should treat the Phase 2g matched-search and refreshed downstream artifacts as the default retained evidence set; older generated outputs can be regenerated if needed

---

## Phase 2g.1 — Local boundary-check closure pass

Status:

- Completed as a small closure check on search-space truncation risk
- Phase 2 is now considered methodologically stable enough to proceed to Phase 3
- The best-known Phase 2 evidence chain now consists of:
  - Phase 2g matched PC/MLP selection
  - Phase 2g.1 local boundary extension
  - Phase 2g.1-refreshed downstream multiseed and budget-tradeoff studies
- Remaining caveat: Phase 2 is not search-exhaustive and should not be described as globally saturated

### Goal

Test whether the current Phase 2g headline conclusions are materially changed by a small local extension beyond the matched-search boundaries.

### Scope

- keep the full Phase 2g train/val/test protocol unchanged
- keep benchmark definitions unchanged
- keep the matched-search conclusion as the baseline reference point
- add only a compact local neighborhood around the current Phase 2g best configs
- probe boundary-hit dimensions without exploding into a new full Cartesian HPO stage

### Deliverables

- a small `phase2g1_boundary_check` runner
- per-benchmark artifacts that record:
  - previous Phase 2g best PC config and metrics
  - previous Phase 2g best MLP config and metrics
  - boundary-check best PC config and metrics
  - boundary-check best MLP config and metrics
  - whether the held-out test headline changed
  - whether the boundary-check best configs moved outside the old search bounds
- documentation updates stating whether the benchmark-level winners survived the closure pass

### Exit criteria

- local boundary extensions are evaluated under the same rules:
  - selection by validation
  - headline reporting by held-out test
- the artifacts make it obvious whether the prior benchmark-level conclusion changed
- the repository has an explicit answer to:
  - did the prior Phase 2g headline survive the boundary check-
  - which benchmark is still boundary-sensitive-
  - is Phase 2 stable enough to move on-

### Risks

- interpreting a local closure pass as exhaustive search coverage
- understating residual boundary sensitivity because the local neighborhood is still finite
- confusing “winner survived the check” with “best possible config has been found”

### Current takeaway

- `toy_regression`:
  - the held-out test winner remained `PC`
  - both PC and MLP improved when allowed a small extension beyond the old bounds
  - the headline conclusion survived
  - the refined downstream multiseed and budget-tradeoff outputs remained aligned with that winner
- `toy_sine_regression`:
  - the held-out test winner remained `MLP`
  - both PC and MLP improved under the local extension
  - the headline conclusion survived
  - the refined downstream multiseed and budget-tradeoff outputs remained aligned with that winner
- both benchmarks remain boundary-sensitive in the narrow sense that selected best configs moved beyond the old search edges
- nonetheless, the benchmark-level winners did not flip, so Phase 2 is stable enough to proceed to Phase 3 as long as the remaining caveats stay explicit

---

## Phase 3 — Standalone real-data digits baselines

Status:

- Completed as a narrow standalone-baseline slice
- This phase now means:
  - a clean real-data entry point using `sklearn.datasets.load_digits`
  - a deterministic mini-batch utility layer
  - a reproducible real-data MLP baseline with explicit train / val / test reporting
  - a reproducible real-data predictive-coding baseline on the same protocol
  - protocol-alignment checks across the standalone MLP and PC baselines
  - a standalone side-by-side summary digest for human inspection
- This phase does not mean:
  - a real-data matched PC-vs-MLP comparison already exists
  - matched tuning has already been completed
  - multi-seed real-data aggregation has already been completed
  - a second real dataset or MNIST has already been implemented

### Goal

Move from toy data to first-pass real-data baselines while keeping the codebase simple, the protocol explicit, and the artifacts easy to inspect.

### Scope

- Phase 3a:
  - add a deterministic `load_digits` data-loading entry point
  - add a deterministic mini-batch helper
  - add a real-data MLP baseline on `digits`
- Phase 3b:
  - add a real-data predictive-coding baseline on the same `digits` protocol
  - add protocol-alignment checks between the standalone MLP and PC baselines
  - add a first-pass side-by-side summary digest and a narrow PC stabilization sweep
- Across both slices:
  - keep explicit train / val / test separation
  - avoid changing the existing toy benchmark registry
- Not part of Phase 3:
  - real-data matched PC-vs-MLP comparison
  - real-data matched tuning
  - multi-seed real-data aggregation
  - any broader real-data tuning or multi-dataset comparison studies

### Deliverables

- `src/pc/datasets.py`
- deterministic `load_digits` split helper returning `SupervisedDataSplit`
- `src/pc/minibatch.py`
- `src/pc/real_mlp.py`
- `src/pc/real_pc.py`
- `experiments/digits_mlp.py`
- `experiments/digits_pc.py`
- `experiments/summarize_digits_baselines.py`
- metadata recording split fractions, seed, normalization, split sizes, and class counts
- deterministic mini-batch ordering with explicit `batch_order_seed`
- a reproducible artifact set:
  - `outputs/digits_mlp/`
  - `outputs/digits_pc/`
  - `outputs/digits_baselines/`
- tests covering reproducibility, shape contracts, one-hot targets, split integrity, minibatch determinism, MLP/PC smoke runs, and protocol alignment
- a narrow stabilization sweep that hardens the canonical `digits_pc` default without becoming formal HPO

### Exit criteria

- Phase 3 exit:
  - `load_digits` split loading is deterministic and well-tested
  - inputs are batch-first `(batch, 64)` float64 arrays normalized by `16.0`
  - targets are batch-first `(batch, 10)` float64 one-hot arrays
  - train / val / test metadata is explicit and self-consistent
  - deterministic mini-batch iteration is available and well-tested
  - the digits MLP baseline writes reproducible artifacts and clearly separates:
    - `train_metric`
    - `val_metric`
    - `test_metric`
    - `selection_metric_source = "val_metric"`
    - `report_metric_source = "test_metric"`
  - the digits PC baseline writes reproducible artifacts and clearly separates:
    - `train_metric`
    - `val_metric`
    - `test_metric`
    - `selection_metric_source = "val_metric"`
    - `report_metric_source = "test_metric"`
  - both standalone baselines perform clearly above the majority-class baseline
  - the protocol-alignment checks confirm no hidden mismatch in:
    - dataset entry
    - seed roles
    - baseline metric definition
    - validation-selected best-checkpoint rule
  - the repository has a first-pass side-by-side summary artifact for the canonical digits baselines

### Risks

- overstating standalone digits baselines as a real-data PC-vs-MLP conclusion
- letting the first real-data step sprawl into matched tuning or a larger framework refactor
- treating a narrow PC stabilization sweep as formal HPO
- confusing a side-by-side digest with a fair comparison protocol

### Current takeaway

- Phase 3 is complete only in the narrow sense of standalone real-data digits baselines
- the repository now has:
  - a deterministic real-data `digits` split
  - a deterministic mini-batch helper
  - a reproducible MLP baseline run under explicit train / val / test reporting
  - a reproducible PC baseline run under the same protocol
  - protocol-alignment checks between those baselines
  - a first-pass side-by-side summary artifact
- the next natural work item is a cautious Phase 4 start:
  - define a narrow real-data comparison protocol on `digits`
  - keep validation-selected checkpoints and held-out test reporting
  - keep the initial comparison scope small before considering matched tuning or broader search

---

## Phase 4 - FMPC-v0 preparation infrastructure

Status:

- Sealed as the current FMPC-v0 preparation checkpoint
- Standalone predict-mode `teacher_reference` metrics are disabled by default in real-data summaries
- Meaningful FMPC teacher targets must come from the dedicated teacher-only preparation/export protocol
- The next phase is the offline FMPC-v0 student stage

Repository note:

- the earlier controlled real-data comparison framing did not become the Phase 4 deliverable
- Phase 4 closed as infrastructure preparation for offline FMPC-v0 student work instead

### Goal

Prepare the real-data predictive-coding stack for offline FMPC-v0 student work without changing the baseline predictive-coding math or claiming a completed real-data comparison.

### Scope

- keep the existing standalone `digits_mlp` and `digits_pc` baselines as the starting point
- define one narrow real-data comparison contract:
  - which configs are canonical starting points
  - which split is used for selection
  - which split is used for final reporting
  - which artifacts count as authority artifacts
- keep the first comparison scope small:
  - one dataset
  - one protocol
  - no broad search framework
  - no multi-dataset claims
- delay matched tuning and broader multi-seed aggregation until the comparison contract is stable

### Deliverables

- a documented real-data comparison protocol for `digits`
- a narrow comparison artifact contract that stays distinct from the Phase 2 toy comparison pipeline
- documentation updates clarifying:
  - what counts as a comparison-ready baseline
  - what is still out of scope
- focused tests that confirm the comparison protocol does not drift from the current Phase 3 standalone-baseline rules

### Exit criteria

- the repository has a comparison protocol on `digits` that is explicit enough to implement without hidden assumptions
- the protocol preserves:
  - validation-selected checkpoints
  - held-out test reporting
  - deterministic seed roles
- the repository can begin a formal real-data comparison step without first re-litigating protocol semantics

### Risks

- letting a narrow real-data comparison step sprawl immediately into matched tuning or large search
- silently drifting away from the current standalone-baseline protocol
- overstating first-pass comparison outputs as a stronger claim than the protocol supports

### FMPC-v0 preparation

Status:

- Implemented as the sealed Phase 4 preparation slice
- This subsection does not authorize any predictive-coding math change or any flow module implementation
- Preparation checkpoints implemented so far include:
  - one harder standalone real-data benchmark option
  - explicit pluggable inference backend labels:
    - `pc_euler` as the default slow iterative backend
    - `pc_rk2` as a stronger numerical-baseline variant
    - reserved placeholder `fmpc` without implementation yet
  - hidden-state flatten / unflatten helpers
  - teacher-target extraction hooks from the current slow iterative teacher
  - FMPC-oriented evaluation helpers and logging for:
    - hidden-state gap to teacher
    - energy gap to teacher
    - terminal update-direction alignment
    - explicit inference backend / step-count metadata
    - lightweight wall-clock timing in the standalone real-data PC artifact

Goal:

- prepare the real-data predictive-coding stack for a future FMPC-v0 transporter without changing the current iterative predictive-coding baseline

Scope:

- add one harder real-data benchmark in addition to `digits`
- add stronger standalone inference-budget baselines for real-data PC
  - keep `euler` as the authoritative default
  - add an explicit higher-order integrator baseline such as `rk2`
- expose hidden-state flatten / unflatten helpers for batch-first `float64` state lists
- expose teacher-target extraction hooks from the current iterative PC path
- make the inference backend pluggable while keeping the current iterative backend as the default
- keep this as infrastructure preparation only:
  - no flow module
  - no matched tuning
  - no formal comparison pipeline

Deliverables:

- one additional deterministic real-data dataset entry in `src/pc/datasets.py`
  - current planned first choice: `fashion_mnist`
- a small backend-selection seam around the current iterative inference path in:
  - `src/pc/inference.py`
  - `src/pc/training.py`
  - `src/pc/real_pc.py`
  - `experiments/digits_pc.py`
- a narrow standalone real-data inference-baseline study that compares explicit integrator/step choices without becoming a formal comparison pipeline
- hidden-state flatten / unflatten helpers, likely in a new narrow helper module such as:
  - `src/pc/state_io.py`
  - or a similarly explicit helper module
- teacher-target extraction hooks that can materialize reproducible transport targets from the current iterative teacher path without changing the teacher itself
- stronger standalone real-data inference-budget artifacts that remain outside the Phase 2 comparison pipeline
- standalone real-data PC summaries that keep teacher-reference fields explicit but disable them by default until a dedicated, semantically meaningful FMPC comparison protocol exists
- a teacher-only FMPC-v0 preparation scaffold that can:
  - select `digits` or `fashion_mnist`
  - train a standard PC teacher
  - export `z0` / `z_star` supervision targets
  - reserve student transport/refinement settings as explicit placeholders
- documentation updates in:
  - `PLANS.md`
  - `README.md`
  - `validation.md`

Files likely to touch when this work starts:

- `src/pc/datasets.py`
- `src/pc/inference.py`
- `src/pc/training.py`
- `src/pc/real_pc.py`
- `src/pc/models.py`
- `experiments/digits_pc.py`
- one new narrow helper module for state packing / target extraction
- tests for digits, inference, and real-data protocol alignment

Tests to add or update:

- update `tests/test_digits_data.py` or add a new real-data dataset-loader test for the harder benchmark
- add a round-trip test for hidden-state flatten / unflatten helpers
- add a teacher-target extraction test that checks determinism, shape contracts, and batch-first ordering
- update `tests/test_inference.py` to confirm the default iterative backend remains the authoritative path
- update `tests/test_digits_pc_smoke.py` to confirm the default backend and artifact contract remain stable
- update `tests/test_real_data_protocol_alignment.py` so backend-related protocol fields stay explicit and aligned

Exit criteria:

- the repository can load `digits` plus one harder deterministic real-data benchmark under the same explicit split discipline
- the current iterative PC path can expose hidden-state tensors in a flatten / unflatten form without semantic drift
- teacher-target extraction hooks exist and are deterministic under fixed seeds
- the inference backend can be selected explicitly, with the iterative backend remaining the default and producing the current baseline behavior
- stronger standalone inference-budget baselines exist for real-data PC without being misread as a matched comparison
- a teacher-only preparation run can materialize explicit FMPC-ready target artifacts without claiming a transporter already exists

Risks:

- hidden-state packing helpers accidentally changing shape conventions or dtype
- backend abstraction silently changing the current iterative semantics
- teacher-target definitions becoming ambiguous:
  - final inferred states only
  - or trajectory-based supervision
- the harder real-data benchmark choice conflicting with the current documentation that Phase 3 only covers `digits`

Repository note:

- Phase 4 now closes on FMPC-v0 preparation infrastructure rather than a formal comparison protocol
- the next phase should start from the existing teacher/export/backend scaffold and add the offline FMPC-v0 student conservatively

---

## Phase 5 — Offline FMPC-v0 student on digits

Status:

- Active as the current post-Phase-4 implementation slice
- Restricted to an offline student transporter trained against Phase 4 teacher-target artifacts
- This phase does not authorize any change to baseline predictive-coding energy, state updates, or weight updates
- Current acceptance work is limited to a Phase 5 v0 seal-off repair:
  - portable teacher artifacts
  - exact teacher checkpoint loading
  - explicit identity / zero-delta baseline metrics
  - non-trivial canonical digits validation
- This repair slice does not authorize:
  - trajectory-aware supervision
  - MeanFlow / JVP losses
  - refinement-by-default
  - any `backend="fmpc"` injection into the core iterative PC path

### Goal

Train a first-pass offline FMPC-v0 student on `digits` that learns endpoint hidden-state transport from `z0` to `z_star` while leaving the iterative PC teacher frozen.

### Scope

- dataset:
  - `digits` only
- teacher:
  - existing iterative PC teacher
  - frozen
  - supervision must come from the dedicated Phase 4 teacher-only preparation/export protocol
- student input:
  - `concat([z0, target_onehot])`
- student target:
  - `delta_z = z_star - z0`
- student output:
  - `delta_z_hat`
- transporter output:
  - `z_hat = z0 + delta_z_hat`
- primary training objective:
  - `MSE(delta_z_hat, delta_z)`
- evaluation metrics:
  - endpoint state error
  - energy gap
  - update-direction alignment
  - optional timing / speedup summaries
- optional refinement:
  - placeholder only
  - disabled by default
- not part of Phase 5 v0:
  - full MeanFlow identity training
  - JVP-based objectives
  - online joint training with the teacher
  - formal PC-vs-student comparison claims

### Deliverables

- a narrow offline student data-loader module that reads Phase 4 teacher-target artifacts and validates:
  - `z0`
  - `z_star`
  - `target_onehot`
  - `delta_z`
  - `concat([z0, target_onehot])`
- a NumPy-first student transporter module for `digits`
- a narrow offline student training module that:
  - trains on `delta_z`
  - selects checkpoints on validation loss
  - reports held-out test metrics separately
- one experiment entry point for the `digits` offline student
- artifact outputs such as:
  - `config.json`
  - `epoch_metrics.csv`
  - `summary.json`
- tests covering:
  - teacher-target artifact loading
  - batch-first float64 contracts
  - deterministic initialization and batch order
  - offline student smoke training
  - artifact generation

### Files likely to touch in the first slice

- `PLANS.md`
- `src/pc/fmpc_student_data.py`
- `src/pc/fmpc_student.py`
- `experiments/fmpc_v0_student.py`
- focused tests under `tests/`

### Exit criteria

- the repository can read a Phase 4 teacher-only preparation artifact and validate a stable offline student data contract
- the offline student trains on `digits` without touching teacher weights or baseline PC math
- training and evaluation artifacts clearly separate:
  - training loss
  - validation selection metric
  - held-out test metric
- optional refinement remains explicitly off by default
- no code path silently routes `backend="fmpc"` into the baseline iterative inference stack

### Risks

- confusing teacher-export semantics with standalone predict-mode evaluation
- silently widening scope into MeanFlow / JVP / CFG before the endpoint transporter works
- coupling the student too tightly to the teacher runtime instead of the saved artifact contract
- introducing hidden assumptions about output-layer inclusion in `z`

### Current first patch target

- lock down the offline student data contract first
- do not start with a full trainer
- consume Phase 4 teacher-target artifacts exactly as written and fail loudly on contract drift

### Phase 5 v0 acceptance-repair slice

Goal:

- close the current Phase 5 v0 acceptance gaps without widening scope beyond offline endpoint transport on `digits`

Files likely to touch:

- `PLANS.md`
- `validation.md`
- `src/pc/fmpc_protocol.py`
- `src/pc/fmpc_student_data.py`
- `src/pc/fmpc_student.py`
- `experiments/fmpc_v0_prepare.py`
- `experiments/fmpc_v0_student.py`
- focused tests under `tests/`

Required repairs:

- teacher checkpoint serialization:
  - save an exact NumPy-readable teacher checkpoint during Phase 4 preparation
  - default student evaluation must load this checkpoint directly
  - config-plus-seed teacher retraining may remain only as an explicit fallback mode
- portable teacher artifact contract:
  - new `teacher_targets/manifest.json` files must use relative paths
  - loaders should remain backward-compatible with older absolute-path manifests when those files still exist locally
- explicit identity baseline:
  - student summaries must report the same endpoint / energy / direction / timing metrics for:
    - the trained student
    - the identity or zero-delta baseline `z_hat = z0`
- non-trivial canonical digits validation:
  - Phase 5 acceptance cannot rely only on the old 2-step smoke teacher
  - the repository must expose a clearer digits validation recipe that uses a meaningfully larger teacher inference budget

Acceptance checks:

- portable artifact check:
  - new teacher manifests are relocatable and do not depend on machine-specific absolute paths
- exact teacher recovery check:
  - loading the serialized teacher checkpoint reproduces exported `z_star` within a very small documented tolerance
- baseline sanity check:
  - student summary explicitly shows whether the student beats the identity baseline on validation and held-out test
- non-triviality check:
  - the canonical digits validation teacher produces visibly non-zero `delta_z` statistics rather than a near-identity transport task

What this slice explicitly does not change:

- baseline PC energy
- baseline PC state updates
- baseline PC local parameter updates
- the standalone `digits_pc` / `digits_mlp` baselines
- trajectory-aware or MeanFlow-style FMPC training

### Phase 5A / student-signal rescue

Goal:

- determine whether the existing endpoint-only student input contract
  - `concat([z0, target_onehot])`
  supports any simple learned student that can beat the explicit identity / zero-delta baseline on the canonical non-trivial `digits` teacher

Scope:

- keep the teacher frozen
- keep `digits` as the only dataset
- keep endpoint-only supervision:
  - input: `concat([z0, target_onehot])`
  - target: `delta_z = z_star - z0`
- keep all final metrics in the original hidden-state space after inverse-transform
- do not introduce:
  - trajectory-aware supervision
  - MeanFlow / JVP objectives
  - refinement
  - any `backend="fmpc"` injection into the core iterative PC path

Required families:

- `identity`
- `class_mean_delta`
- `ridge`
- `mlp_standardized`

Required implementation constraints:

- `class_mean_delta` may use only train-split statistics
- `ridge` must be deterministic closed-form multi-output ridge regression
- `mlp_standardized` must remain NumPy-first and use explicit train-stat normalization
- all train-stat normalization must be estimated on the train split only

Deliverables:

- a narrow endpoint baseline-suite module covering the above families
- explicit normalization helpers for:
  - `z0`
  - `delta_z`
- a small compare/search runner that:
  - evaluates the families under a tiny deterministic search space
  - selects the winning learned candidate by `val_state_rms_gap`
  - reports held-out test metrics once for the final validation-selected winner
- clear saved artifacts showing:
  - the identity baseline
  - the class-mean baseline
  - the best ridge candidate
  - the best standardized-MLP candidate
  - which family wins overall
  - whether the winning learned family beats the identity baseline on validation and test

Files likely to touch:

- `PLANS.md`
- `validation.md`
- `src/pc/fmpc_student_normalization.py`
- `src/pc/fmpc_student_baselines.py`
- `src/pc/fmpc_student_suite.py`
- `src/pc/fmpc_student.py`
- `experiments/fmpc_v0_student_suite.py`
- focused tests under `tests/`

Acceptance checks:

- contract check:
  - all families consume the same batch-first `float64` endpoint contract
- normalization check:
  - saved normalization statistics come only from the train split
  - inverse-transform restores predictions to the original hidden-state space before metric computation
- suite visibility check:
  - `identity`, `class_mean_delta`, `ridge`, and `mlp_standardized` all appear explicitly in the saved comparison artifacts
- Phase 5A pass condition:
  - at least one learned family
    - `ridge` or `mlp_standardized`
  beats the identity baseline on both:
    - validation `state_rms_gap`
    - held-out test `state_rms_gap`

Failure escalation rule:

- if all learned families still fail against identity, the next allowed step is only:
  - endpoint-only feature augmentation
- this still does not authorize:
  - trajectory-aware supervision
  - MeanFlow / JVP objectives
  - refinement
  - core iterative `fmpc` backend integration

### Phase 5B / offline interval-conditioned transporter

Goal:

- determine whether a teacher-supervised interval-conditioned student, trained only on frozen `digits` teacher trajectories, can beat the carried-forward Phase 5A endpoint ridge baseline under explicit 1-step / 2-step / 3-step rollout schedules

Scope:

- keep the teacher frozen
- keep `digits` as the only dataset
- require trajectory-enabled teacher artifacts from the dedicated preparation/export path
- keep the default student input:
  - `concat([z_s, target_onehot, tau_s, tau_t])`
- keep the default target:
  - `u_star = (z_t - z_s) / (tau_t - tau_s)`
- keep rollout transport explicit:
  - `z_hat_t = z_hat_s + (tau_t - tau_s) * u_hat`
- evaluate only explicit teacher-step-aligned rollout schedules:
  - `1-step`
  - `2-step`
  - `3-step`
- do not introduce:
  - trajectory-consistency losses
  - MeanFlow identity losses
  - JVP objectives
  - refinement
  - online joint training with PC weights
  - any `backend="fmpc"` injection into the core iterative PC path

Required implementation constraints:

- trajectory artifacts must remain portable and exact-teacher-backed
- the trajectory contract must make explicit:
  - teacher inference steps `K`
  - trajectory tensor shape
  - split metadata
  - normalized time rule `tau_k = k / K`
- interval training-pair generation must avoid naive short-interval dominance
- the default training policy should be interval-length-balanced, or an equally explicit alternative
- all arrays remain batch-first `float64`

Required learned families:

- `interval_ridge`
- `interval_mlp_standardized`

Required fixed baselines:

- `identity`
- carried-forward Phase 5A endpoint ridge baseline

Deliverables:

- an interval trajectory data-loader / sampler module
- interval normalization helpers for:
  - `z_s`
  - `u_star`
- interval-conditioned student families:
  - `interval_ridge`
  - `interval_mlp_standardized`
- a narrow compare/search runner that:
  - searches a small deterministic candidate set
  - evaluates explicit rollout schedules
  - selects a winner by validation `final_state_rms_gap`
  - reports held-out test once for the final validation-selected winner
- saved artifacts such as:
  - `config.json`
  - `candidates.csv`
  - `summary.json`

Files likely to touch:

- `PLANS.md`
- `validation.md`
- `src/pc/fmpc_protocol.py`
- `src/pc/fmpc_interval_data.py`
- `src/pc/fmpc_interval_normalization.py`
- `src/pc/fmpc_interval_student.py`
- `experiments/fmpc_interval_suite.py`
- `experiments/fmpc_v0_prepare.py`
- focused tests under `tests/`

Acceptance checks:

- trajectory contract check:
  - trajectory-enabled teacher artifacts load exactly from the serialized checkpoint path
  - `z_trajectory` shape and endpoint semantics are explicit and stable
- sampling check:
  - the default interval-pair training policy is not dominated by short spans
- rollout check:
  - `1-step`, `2-step`, and `3-step` schedules use explicit teacher-step-aligned knot indices
  - rollout is self-fed rather than teacher-forced between knots
- Phase 5B pass condition:
  - at least one learned interval family
    - `interval_ridge` or `interval_mlp_standardized`
  beats the carried-forward Phase 5A endpoint ridge baseline on both:
    - validation `final_state_rms_gap`
    - held-out test `final_state_rms_gap`
  under an explicit rollout schedule

Failure escalation rule:

- if every learned interval family still loses to the carried-forward Phase 5A endpoint ridge baseline, the next allowed rescue remains below MeanFlow / JVP:
  - endpoint-free interval feature augmentation only
- this still does not authorize:
  - trajectory-aware supervision
  - MeanFlow identity objectives
  - JVP objectives
  - refinement
  - core iterative `fmpc` backend integration

### Phase 5B rollout-aware rescue

Goal:

- keep the existing interval-conditioned student contract, but reduce rollout distribution shift in true multi-step evaluation by adding a small amount of self-fed, teacher-supervised auxiliary training below the MeanFlow / JVP boundary

Scope:

- still `digits` only
- still frozen iterative PC teacher only
- still teacher-supervised only
- still no:
  - MeanFlow identity objectives
  - JVP objectives
  - refinement
  - online joint PC-weight training
  - any `backend="fmpc"` injection into the core iterative inference path

Required rescue constraint:

- the original interval target remains the primary target:
  - `u_star = (z_t - z_s) / (tau_t - tau_s)`
- any rollout-aware term must remain auxiliary and explicit

Rescue design:

- apply rollout-aware auxiliary supervision only to the interval standardized-MLP family
- use fixed teacher-step-aligned rollout schedules:
  - `2-step`
  - `3-step`
- start rollouts from teacher `z_0`
- then self-feed the student's predicted states between knots
- build auxiliary corrective targets from those predicted states to the corresponding teacher knot states
- keep intermediate-knot and final-endpoint auxiliary losses explicit in code and saved summaries

Acceptance gate after the rescue:

- keep the original Phase 5B gate unchanged
- the rescue only counts as successful if a learned interval family with a true multi-step schedule:
  - `2-step` or `3-step`
  beats the carried-forward Phase 5A endpoint ridge baseline on both:
  - validation `final_state_rms_gap`
  - held-out test `final_state_rms_gap`
- the winning candidate's test `energy_gap_to_teacher` must remain within the previously stated tolerance relative to the carried-forward endpoint ridge baseline

What this rescue still does not authorize:

- trajectory-consistency losses beyond the fixed auxiliary rollout schedules
- MeanFlow identity training
- JVP-style objectives
- alpha curricula
- refinement

### Phase 5B.2 gradient-augmented interval rescue

Goal:

- keep Phase 5B fully teacher-supervised and endpoint-free, but reduce multi-step rollout drift by conditioning interval students on frozen-teacher local dynamical features computed only at the current state `z_s`

Scope:

- still `digits` only
- still frozen iterative PC teacher only
- still exact teacher checkpoint loading only
- still no:
  - MeanFlow identity objectives
  - JVP objectives
  - refinement
  - online joint PC-weight training
  - any `backend="fmpc"` injection into the core iterative inference path

Required current-state feature contract:

- features are computed only from:
  - current hidden state `z_s`
  - frozen teacher parameters
  - current sample input / target
- features must not leak:
  - `z_t`
  - `z_star`
  - future teacher states
- the first rescue feature pack should stay narrow and explicit:
  - `g_s`
  - `e_out_s`
  - `F_s`
- where:
  - `g_s` is the frozen teacher's one-step hidden-state inference field at `z_s`, expressed in the same normalized-time units as `u_star`
  - `e_out_s = target_onehot - y_hat_s`
  - `F_s` is a per-sample scalar teacher energy at `z_s`

Required learned families:

- `interval_ridge_aug`
- `interval_ridge_residual`
- optional `interval_mlp_aug`

Required target semantics:

- keep the original direct target available:
  - `u_star = (z_t - z_s) / (tau_t - tau_s)`
- add a residual target family:
  - `u_res = u_star - g_s`
- residual-family summaries must reconstruct and report:
  - `u_hat = g_s + u_res_hat`

Required training-distribution rescue:

- keep the existing span-balanced interval sampler available
- add an explicit knot-focused training option that upweights the exact interval spans used by the acceptance schedules:
  - `2-step`
  - `3-step`
- this rescue must remain explicit in:
  - config
  - candidate rows
  - summary

Deliverables:

- a narrow teacher-state feature module for interval students
- augmented interval input helpers that keep the original non-augmented contract intact
- at least the two new learned ridge families above
- a small compare/search extension that carries forward:
  - `identity`
  - carried-forward Phase 5A endpoint ridge
  - existing `interval_ridge`
  - the new augmented ridge families
- clear saved artifacts exposing:
  - which feature contract each candidate used
  - whether knot-focused sampling was used
  - whether the winner is a true multi-step learned interval family

Files likely to touch:

- `PLANS.md`
- `validation.md`
- `src/pc/fmpc_interval_features.py`
- `src/pc/fmpc_interval_data.py`
- `src/pc/fmpc_interval_normalization.py`
- `src/pc/fmpc_interval_student.py`
- `experiments/fmpc_interval_suite.py`
- focused tests under `tests/`

Acceptance gate:

- keep the original Phase 5B gate unchanged
- Phase 5B.2 only counts as successful if a learned interval family under a true multi-step schedule:
  - `2-step`
  - or `3-step`
  beats the carried-forward Phase 5A endpoint ridge baseline on both:
  - validation `final_state_rms_gap`
  - held-out test `final_state_rms_gap`
- the winner's test `energy_gap_to_teacher` must still remain within the existing tolerance relative to the carried-forward endpoint ridge baseline

What this rescue still does not authorize:

- MeanFlow identity training
- JVP-style objectives
- refinement
- alpha curricula

---

## Phase 6 — Benchmarking and paper-oriented experiments

### Goal

Approach paper-grade experimental discipline.

### Scope

- scripted experiments
- multiple seeds
- ablation tables
- result aggregation
- figure generation
- checkpointing and config snapshots

### Deliverables

- benchmark runner(s)
- result aggregation scripts
- figure export scripts
- experiment manifests

### Exit criteria

- experiments are reproducible from documented commands
- outputs are organized and comparable across runs
- claims are backed by saved artifacts

---

## Standing planning rules

For any new task, specify:

1. Current phase
2. Why the task belongs in this phase
3. Files to touch
4. Validation to run
5. What will explicitly not be changed

## Near-term recommended first tasks

1. Create the package skeleton in `src/pc/`
2. Implement activation functions and derivatives
3. Implement energy computation for the baseline spec
4. Implement hidden-state inference loop
5. Implement weight updates
6. Build a tiny `PCNetwork` wrapper
7. Add smoke tests
8. Add a toy experiment script

## Nice-to-have later

- config dataclasses
- richer CLI interface
- experiment registry
- profiling hooks
- optional Numba or Cython acceleration after the baseline is stable

---

## Phase 5B Seal-Off

- Phase 5B is now considered passed and sealed.
- Fresh final validation established:
  - portable exact-checkpoint-backed teacher trajectory artifacts
  - a true multi-step learned interval winner
  - `interval_ridge_residual` under `3-step` rollout beating the carried-forward Phase 5A endpoint ridge baseline on both validation and held-out test `final_state_rms_gap`
  - no energy-gate regression relative to the carried-forward endpoint ridge baseline

## Phase 6A — MeanFlow-Style Teacher-Supervised Average-Velocity

### Goal

Open the next FMPC stage conservatively by moving from interval-conditioned transport to MeanFlow-style, teacher-supervised average-velocity modeling.

### Scope

- keep `digits` as the first target
- keep the iterative PC teacher frozen
- stay offline and teacher-supervised
- keep the successful Phase 5B.2 augmented interval pipeline intact and side-by-side
- continue to avoid:
  - online joint PC-weight training
  - refinement
  - any `backend="fmpc"` injection into the core iterative inference path
- the intended new ingredients are:
  - MeanFlow-style average-velocity supervision
  - a manual NumPy forward-mode JVP for the explicit MLP family
- this phase still does not authorize:
  - teacher removal
  - online joint PC-weight training
  - refinement
  - any core iterative `fmpc` backend integration

### Entry condition

- Phase 5B is passed and sealed
- new work should start from the Phase 5B sealed checkpoint rather than re-opening endpoint or interval rescue scope unless a new regression is found

### Core contract

- keep the interval teacher parameterization:
  - `u_star = (z_t - z_s) / (tau_t - tau_s)`
- keep the current-state teacher local field from Phase 5B.2:
  - `g_s`
- define a current-state MeanFlow-style student:
  - `u_theta(z_s, tau_s, tau_t, context)`
- use the forward-time identity:
  - `u = g_s + dt * d/dtau_s u`
  - where `dt = tau_t - tau_s`
- expand the total derivative as:
  - `d/dtau_s u = J_z u * g_s + partial_tau_s u`
- the initial Phase 6A formulation treated:
  - `target_onehot`
  - teacher-derived context features
  as frozen side information in the JVP path
- the Phase 6A.1 redesign keeps `target_onehot` frozen but makes the teacher-derived
  current-state feature block feature-aware by injecting its directional derivatives
  along `g_s` into the input tangent

### Required families

- `teacher_only_mlp_aug`
  - direct regression to `u_star`
  - diagnostic baseline only
- `meanflow_mlp_aug`
  - direct prediction of `u_hat`
  - direct teacher supervision plus MeanFlow identity loss
- `meanflow_mlp_residual`
  - direct prediction of `r_hat`
  - reconstruct `u_hat = g_s + r_hat`
  - apply MeanFlow identity to the reconstructed `u_hat`

### Default training recipe

- reuse the Phase 5B.2 current-state feature pack:
  - `g_s`
  - `e_out_s`
  - `F_s`
- keep explicit train-stat normalization
- keep rollout schedules:
  - `1-step`
  - `2-step`
  - `3-step`
- start with an explicit hybrid curriculum:
  - early teacher-only warmup
  - middle ramp from zero identity weight to a small nonzero identity weight
  - late fixed hybrid stage
- keep direct teacher supervision as the anchor objective throughout
- allow a conservative reduced identity scope when needed:
  - `all_intervals`
  - or `acceptance_schedule_segments_only`
  - where the latter only applies the identity term on the exact `2-step` / `3-step` teacher-step-aligned spans used by acceptance rollouts
- any rollout-aware auxiliary term must remain:
  - small
  - explicit
  - clearly separated from the MeanFlow identity term

### Narrow rescue direction

If the first Phase 6A pass fails narrowly, the next allowed rescue inside Phase 6A is:

- keep the same MeanFlow-style neural family definitions
- keep the same frozen-teacher feature pack
- keep the same manual NumPy JVP
- stabilize the hybrid optimization by:
  - separating teacher and identity updates in code rather than collapsing them into one mixed target update
  - making the identity curriculum explicit in config and summary
  - optionally restricting identity application to acceptance-schedule segments only

This rescue still does not authorize:

- teacher-feature amortization / reduction
- teacher-free training
- online joint training
- refinement

### Phase 6A.1 feature-aware MeanFlow identity

If the curriculum-only rescue still fails, the next allowed rescue remains inside Phase 6A and becomes:

- keep the successful Phase 5B.2 feature contract intact:
  - `g_s`
  - `e_out_s`
  - `F_s`
- add directional derivatives of those teacher-derived current-state features along `g_s`
- compute those tangents conservatively by symmetric finite differences in state space
- keep direct teacher average-velocity supervision as the anchor objective
- keep the MeanFlow identity and the manual NumPy JVP explicit and separate in code
- add one diagnostic linear residual family to distinguish:
  - identity-formula errors
  - from MLP-family bottlenecks

The Phase 6A.1 feature-aware input tangent should follow:

- `dX_s = concat([g_s, 0_target, 1_tau_s, 0_tau_t, D_g teacher_features(z_s)])`

with:

- `D_g feat(z_s) ≈ [feat(z_s + eps * g_s) - feat(z_s - eps * g_s)] / (2 * eps)`

For the residual family:

- `u_hat = g_s + r_hat`
- the MeanFlow identity must explicitly include `d g_s / d tau_s`
- the identity therefore applies to the reconstructed `u_hat`, not only to the residual block

Phase 6A.1 should compare at least:

- `teacher_only_mlp_aug`
- `meanflow_mlp_aug`
- `meanflow_mlp_residual`
- `meanflow_linear_residual`

### Phase 6A.2 two-branch MeanFlow residual decomposition

If the feature-aware Phase 6A.1 rescue still fails, but the diagnostic linear residual
family becomes the fresh true multi-step winner, the next allowed rescue remains inside
Phase 6A and becomes:

- keep the successful Phase 5B.2 feature contract intact:
  - `g_s`
  - `e_out_s`
  - `F_s`
- keep the Phase 6A.1 feature-aware tangent contract intact:
  - current-state teacher feature directional derivatives remain active
  - residual MeanFlow identity must still include `d g_s / d tau_s`
- replace the monolithic neural residual family with an explicit two-branch decomposition:
  - `u_hat = u_local + u_corr`
- keep direct teacher supervision on the full reconstructed `u_hat`
- keep the MeanFlow identity on the full reconstructed `u_hat`
- avoid broad hyperparameter expansion; the rescue should be structural and diagnostic

The intended branch roles are:

- local branch:
  - dedicated to teacher-like local dynamics
  - narrow input set:
    - `g_s`
    - `e_out_s`
    - `F_s`
  - simple structured family:
    - linear / affine
- correction branch:
  - dedicated to transport correction beyond the local field
  - uses the richer augmented Phase 6A.1 input:
    - `concat([z_s, target_onehot, tau_s, tau_t, g_s, e_out_s, F_s])`
  - small neural family
  - output head should start near zero when feasible so the full model begins near the local solution

The preferred direct-supervision design is:

- anchor loss on the full prediction:
  - `L_teacher_full = MSE(u_hat, u_star)`
- optional small branch-separation helper:
  - `u_corr_star = u_star - stopgrad(u_local)`
  - `L_corr = MSE(u_corr, u_corr_star)`
- local-branch anchor:
  - `L_local = MSE(u_local, g_s)`

The MeanFlow identity must still apply to the full combined transport law:

- `u_hat = u_local + u_corr`
- `u_hat 鈮-g_s + dt * d/dtau_s u_hat`
- with:
  - `d/dtau_s u_hat = d/dtau_s u_local + d/dtau_s u_corr`

Phase 6A.2 should compare at least:

- `teacher_only_mlp_aug`
- `meanflow_mlp_aug`
- `meanflow_mlp_residual`
- `meanflow_linear_residual`
- `meanflow_twobranch_residual`

### Phase 6A.3 warm-started two-branch MeanFlow residual training

If the Phase 6A.2 two-branch neural family remains weaker than the carried-forward
Phase 6A.1 linear winner, the next allowed rescue remains inside Phase 6A and
becomes:

- keep the successful Phase 6A.2 decomposition intact:
  - `u_hat = u_local + u_corr`
- keep the Phase 6A.1 / 6A.2 feature-aware tangent machinery intact:
  - teacher-derived current-state feature tangents remain active
  - residual identity must still include `d g_s / d tau_s`
  - identity still applies to the full reconstructed `u_hat`
- fix the optimization path rather than redesigning the family again:
  - warm-start the local branch from the carried-forward Phase 6A.1 linear residual winner
  - keep the correction branch zero-initialized or near-zero-initialized
  - add explicit staged training:
    - Stage A: correction-only warmup with the local branch frozen
    - Stage B: joint hybrid fine-tuning

The preferred warm-start contract is:

- reconstruct the carried-forward Phase 6A.1 linear residual winner on the same fresh
  checkpoint-backed teacher artifact
- extract the teacher-feature linear block corresponding to:
  - `g_s`
  - `e_out_s`
  - `F_s`
- map that block into the two-branch local branch so the new family starts near the
  best known feature-aware linear solution without depending on machine-specific
  historical run directories

The preferred staged-training contract is:

- Stage A:
  - local branch frozen
  - correction branch trainable
  - direct teacher supervision and MeanFlow identity both remain active
  - correction targets are defined relative to the frozen `u_local`
- Stage B:
  - local branch unfrozen
  - both branches receive the same full-`u_hat` teacher anchor and full-`u_hat`
    MeanFlow identity signal

Phase 6A.3 should compare at least:

- carried-forward Phase 5A endpoint ridge baseline
- carried-forward Phase 5B.2 winner
- carried-forward Phase 6A.1 linear residual winner
- carried-forward Phase 6A.2 two-branch best candidate
- `meanflow_twobranch_residual`
- `meanflow_twobranch_residual_warmstart`

### Files likely to touch

- `PLANS.md`
- `validation.md`
- `src/pc/fmpc_interval_features.py`
- `src/pc/fmpc_meanflow_jvp.py`
- `src/pc/fmpc_meanflow_student.py`
- `experiments/fmpc_meanflow_suite.py`
- minimal export updates in `src/pc/__init__.py`
- focused tests under `tests/`

### Acceptance direction

- the comparison baseline to beat is the sealed Phase 5B.2 winner:
  - `interval_ridge_residual`
  - `3-step`
- Phase 6A only counts as passed if a MeanFlow neural family:
  - `meanflow_mlp_aug`
  - or `meanflow_mlp_residual`
  - or `meanflow_twobranch_residual`
  - or `meanflow_twobranch_residual_warmstart`
  wins under a true multi-step rollout:
  - `2-step`
  - or `3-step`
  and beats the sealed Phase 5B.2 baseline on both validation and held-out test `final_state_rms_gap`

## Phase 6A seal-off note

Phase 6A is now sealed as **incomplete / not passed**.

What the repo established during Phase 6A:

- the original frozen-side-information MeanFlow identity was structurally insufficient
- the Phase 6A.1 feature-aware MeanFlow identity fixed the main missing tangent terms:
  - teacher-derived current-state feature tangents became active
  - residual identity explicitly retained `d g_s / d tau_s`
- the diagnostic feature-aware linear residual family became the fresh true multi-step winner
  - this strongly suggests the corrected MeanFlow identity is directionally right

What remained unresolved at seal-off:

- no neural MeanFlow family became competitive enough with the sealed Phase 5B.2 winner
- the Phase 6A.2 two-branch neural family remained weaker than the carried-forward
  Phase 6A.1 linear winner
- the Phase 6A.3 warm-started two-branch neural family was unstable under fresh final validation
  and failed badly under true multi-step rollout

Interpretation:

- Phase 6A produced useful diagnostics and repaired part of the identity design
- but it did **not** produce a passing neural MeanFlow stage
- therefore Phase 6A should be treated as a sealed exploratory branch with known defects,
  not as a passed stage

## Phase TF1 — Teacher-free FMPC v1

The next stage is now opened by project decision as:

- `Phase TF1 — Teacher-free FMPC v1`

This stage starts from a repo state where:

- Phase 5B.2 remains the last clearly passed transport baseline
- Phase 6A is sealed but incomplete
- MeanFlow diagnostics from Phase 6A may still inform design choices,
  but they do not constitute a passing prerequisite

Phase TF1 should be treated as a new stage entry rather than a retroactive claim that
Phase 6A succeeded.

## Phase TF1 — Teacher-free FMPC v1

Objective:

- start the new main FMPC line without teacher artifacts
- keep the existing layered predictive-coding substrate intact
- replace only the **training-time** hidden-state inference path with a minimal
  teacher-free FMPC transporter
- preserve baseline parameter-update math after transport

Core contract:

- context remains supervised training context `c = (x, y)` with:
  - `x^0 = x` clamped
  - `x^L = y` clamped
- free latent state is:
  - `z = flatten(x^1, ..., x^(L-1))`
- local energy substrate is the existing baseline PC energy:
  - `E_theta(z; c) := F(states(z; x, y), theta)`
- exact local hidden-state flow is:
  - `g_theta(z; c) = -∇_z E_theta(z; c)`
- the teacher-free average-velocity model is:
  - `u_psi(z_t, r, t; c)`
- first-pass transport uses one-step or few-step coarse rollout only
- after transport, `theta` is updated with the same baseline local parameter
  updates already used by the slow PC implementation

Non-goals:

- no teacher trajectories
- no teacher fixed points
- no teacher-generated regression targets
- no modification of the slow predict/eval path
- no integration through `backend="fmpc"`
- no general PCG support in v1

Files to add / modify:

- add:
  - `src/pc/fmpc_tf1_flow.py`
  - `src/pc/fmpc_tf1_jvp.py`
  - `src/pc/fmpc_tf1.py`
  - `experiments/fmpc_tf1.py`
  - `tests/test_fmpc_tf1_flow.py`
  - `tests/test_fmpc_tf1_jvp.py`
  - `tests/test_fmpc_tf1_targets.py`
  - `tests/test_fmpc_tf1_smoke.py`
- modify:
  - `PLANS.md`
  - `AGENTS.md`
  - `validation.md`
  - `spec_math.md`
  - `src/pc/__init__.py`

Minimal implementation:

- implement `tf1_mlp_core` first
- input contract:
  - `concat([z_t, target_onehot, t, r])`
- fallback input tangent:
  - `concat([g_t, 0_target, 1_t, -1_r])`
- direct anchor target:
  - self-bootstrap average velocity from the current exact local flow
- MeanFlow identity:
  - `u_hat ≈ g_t + r * JVP_T(u_hat)`
- `psi` optimization uses a single weighted loss:
  - `L = L_boot + lambda_id * L_id`

Optional enhancement path after core is green:

- `tf1_mlp_aug`
- input contract:
  - `concat([z_t, target_onehot, t, r, g_t, e_out_t, F_t])`
- feature-aware tangents remain optional, not mandatory

Training schedule:

- Stage A: bootstrap warmup
  - canonical default: `warmup_epochs = 5`
  - `theta` does not freeze
  - `lambda_id = 0`
  - `psi` trains on `L_boot`
  - `theta` updates use `local_field_only` transport
- Stage B: hybrid
  - learned `u_psi` transport becomes active for `theta`
  - `theta` and `psi` are updated jointly
  - `lambda_id` ramps from `0` to a small nonzero value

Experiment entrypoint:

- `experiments/fmpc_tf1.py`
- named presets:
  - `mechanism_smoke` = small TF1 substrate `(64, 16, 10)`
  - `baseline_comparable` = baseline-sized digits substrate `(64, 64, 10)`
  - `baseline_working_default` = current evidence-driven but provisional
    working TF1 preset:
    - baseline-sized digits substrate `(64, 64, 10)`
    - `tf1_mlp_aug`
    - `transport_steps = 1`
    - `warmup_epochs = 5`
    - `feature_aware_tangents = false`
    - `identity_loss_weight = 0.2`
    - `checkpoint_selector = gate_constrained_accuracy_then_val_accuracy`

Checkpoint-selection contract:

- selector policy is now part of the explicit main TF1 experiment contract
- supported checkpoint selectors are:
  - `energy_only`
  - `val_accuracy_only`
  - `gate_constrained_accuracy_then_energy`
  - `gate_constrained_accuracy_then_val_accuracy`
- selector logic uses validation only
- test remains report-only
- old presets remain unchanged as historical/reference presets
- the working-default preset is evidence-driven but still provisional

Acceptance criteria for the first TF1 mechanism-validating pass:

- the run is fully teacher-free:
  - no teacher manifests
  - no teacher checkpoints
  - no teacher trajectories
  - no teacher-generated regression targets
- checkpoint selection uses:
  - `selection_metric = "val_transported_final_energy"`
- pass/fail gating uses **validation only**
- the same run must report strict apples-to-apples baselines against:
  - `identity/no-transport`
  - `local_field_only`
- these baselines must share:
  - identical rollout knots
  - identical `transport_steps`
  - identical `theta` snapshot
  - identical batch split
  - identical energy metric
- the best TF1 candidate must satisfy on validation:
  - `val_transported_final_energy < val_identity_final_energy`
  - `val_transported_final_energy <= val_local_field_only_final_energy`
  - `val_accuracy > majority baseline`
- test remains report-only

Risks / open questions:

- a self-bootstrap target may collapse toward `local_field_only` and provide only
  a weak learning signal
- few-step coarse transport may improve validation energy while leaving slow-PC
  accuracy gains modest
- feature-aware tangents may help later, but they should not be required for the
  first viable TF1 core
- the current `backend="fmpc"` placeholder remains intentionally unused in TF1 v1

## Phase TF1 seal-off note

Phase TF1 is now sealed as the first completed **teacher-free FMPC v1** stage.

What TF1 established:

- a fully teacher-free FMPC main path that does not depend on:
  - teacher manifests
  - teacher checkpoints
  - teacher trajectories
  - teacher-generated regression targets
- explicit selector-integrated checkpoint selection in the main TF1 path
- the evidence-driven working TF1 preset:
  - `baseline_working_default`
- small, reproducible supporting studies for:
  - selector alignment
  - gate coverage
  - multiseed confirmation
  - default adoption
  - external comparison
  - narrow accuracy tuning

What remained unresolved at seal-off:

- `baseline_working_default` clearly beats `baseline_comparable`, but the gap to the
  canonical slow-PC digits baseline remains material
- the narrow TF1 accuracy-improvement sweep did not materially reduce that gap
- TF1 therefore does **not** yet justify replacing the canonical slow iterative PC
  baseline as the strongest digits accuracy reference

Interpretation:

- TF1 succeeded as a teacher-free bridge-establishment stage
- TF1 did **not** yet close the accuracy gap to the canonical slow-PC baseline
- `baseline_working_default` should be treated as the sealed TF1 working default,
  not as a claim that TF1 is already competitive enough to stop further bridge work

## Phase TF2 - iFMPC bridge stage

Objective:

- test whether **incremental scheduling** is the missing mechanism between the sealed
  TF1 working default and a stronger teacher-free FMPC path
- keep the current layered predictive-coding energy substrate intact
- keep the baseline local parameter-update rule intact
- add only:
  - learned transport micro-steps for `z`
  - optional immediate local `theta` updates at each micro-step
  - mixed-policy teacher-free supervision for `psi`
  - explicit forward-init stability diagnostics

Why TF2 is a bridge stage rather than the final paradigm:

- it still uses the existing layered PC substrate and the same local energy
- it keeps the baseline local parameter-update rule rather than introducing a new
  parameter-learning rule
- it leaves slow iterative PC predict/eval untouched
- it borrows:
  - iPC-inspired scheduling ideas
  - -PC-inspired stability/conditioning concerns
  but does not yet adopt:
  - a new scaling mechanism
  - a new substrate class
  - a generalized iterative FMPC paradigm

Formal algorithm contract:

- hidden latent:
  - `z = flatten(x^1, ..., x^(L-1))`
- train-time context:
  - `c = (x, y)` with `x^0 = x` and `x^L = y` clamped
- energy substrate:
  - `F_theta(z; c)` is the current target-clamped layered-PC batch energy
- local field:
  - `g_theta(z; c) = --_z F_theta(z; c)`
- slow predict/eval remains the current canonical slow-PC path

Micro-step schedule:

- let `H = micro_steps`
- use uniform knots:
  - `t_k = k / H`
  - `-t = 1 / H`
  - `r_k = 1 - t_k`
- maintain two train-time hidden-state streams:
  - `z_on_k`: learned on-policy state
  - `z_lf_k`: detached local-field-only shadow state

Frozen-within-micro-step semantics:

- within one micro-step `k`, all supervision targets and state advances must be
  computed under one frozen parameter snapshot `(theta_k, psi_k)`
- this includes:
  - `u_boot`
  - `u_id`
  - learned transport outputs
  - `z_on_{k+1}`
  - `z_lf_{k+1}`
- only after these quantities are computed may parameter updates be applied

Required micro-step order:

1. compute supervision targets and learned transport under frozen `(theta_k, psi_k)`
2. advance `z_on` and `z_lf`
3. apply one immediate local `theta` update when enabled
4. apply one `psi` update

State updates:

- learned on-policy transport:
  - `z_on_{k+1} = z_on_k + -t * u_psi(z_on_k, r_k, t_k; c)`
- local-field shadow transport:
  - `z_lf_{k+1} = z_lf_k + -t * g_theta(z_lf_k; c)`

Mixed-policy supervision:

- `supervision_policy in {"local_only", "mixed"}`
- `local_only`:
  - supervise `psi` only on detached `z_lf_k`
- `mixed`:
  - supervise `psi` on the concatenation of:
    - detached `z_lf_k`
    - detached `z_on_k`
  - use simple batch concatenation for equal source weighting

TF2 supervision targets remain exactly TF1-style teacher-free targets:

- `u_boot` from `bootstrap_average_velocity_target(...)`
- `u_id = g_t + r_k * D_T u_psi(...)`
- loss remains:
  - `L = L_boot + lambda_id * L_id`
- when `use_teacher_free_features = true`:
  - `feature_aware_tangents = true` means the identity JVP includes chain-rule
    directional-derivative terms through the appended feature block
  - `feature_aware_tangents = false` remains allowed, but it must be interpreted as
    a truncated feature-frozen identity approximation rather than the full augmented
    total derivative

Matched theta-update budget:

- `theta_update_budget in {"matched", "unmatched"}`
- canonical default:
  - `theta_update_budget = "matched"`
- if `incremental_weight_updates = true` and budget is `matched`:
  - normalize by the number of theta updates that are actually applied under the
    active cadence for that batch
  - `terminal_only` therefore uses the base learning rate on the one terminal theta
    update
  - `every_micro_step` divides by `micro_steps`
  - `every_2_micro_steps` divides by the number of due in-loop theta updates
- if `incremental_weight_updates = true` and budget is `unmatched`:
  - `theta_micro_lr = base_theta_lr`
  - `theta_micro_bias_lr = base_theta_bias_lr`
- if `incremental_weight_updates = false`:
  - no theta updates happen inside the rollout
  - one terminal theta update is applied after the final micro-step using the
    existing base learning rates
  - `theta_micro_lr` and `theta_micro_bias_lr` are still recorded for transparency

What remains PC:

- the layered energy substrate
- the definition of the local hidden-state field `g_theta`
- the baseline local parameter-update rule
- target-clamped train-time semantics
- slow iterative predict/eval

What has moved beyond baseline PC:

- learned average-velocity transport `u_psi`
- micro-step interleaving of state transport and parameter updates
- mixed-policy teacher-free supervision for `psi`
- selector-governed checkpoint choice carried forward from TF1

Files to add / modify:

- modify:
  - `PLANS.md`
  - `validation.md`
  - `spec_math.md`
  - `src/pc/__init__.py`
- add:
  - `src/pc/fmpc_tf2.py`
  - `src/pc/fmpc_tf2_suite.py`
  - `experiments/fmpc_tf2.py`
  - `experiments/fmpc_tf2_suite.py`
  - `tests/test_fmpc_tf2_dynamics.py`
  - `tests/test_fmpc_tf2_targets.py`
  - `tests/test_fmpc_tf2_smoke.py`
  - `tests/test_fmpc_tf2_suite_smoke.py`

Reuse without changing TF1 behavior:

- `src/pc/fmpc_tf1_flow.py`
- `src/pc/fmpc_tf1_jvp.py`

Experiment entrypoints:

- `experiments/fmpc_tf2.py`
  - canonical single-run TF2 bridge experiment
- `experiments/fmpc_tf2_suite.py`
  - narrow multiseed bridge-validation suite

Canonical TF2A defaults:

- family lineage:
  - `tf1_mlp_aug`
- `use_teacher_free_features = true`
- `feature_aware_tangents = false`
- `micro_steps = 4`
- `incremental_weight_updates = true`
- `supervision_policy = "mixed"`
- `theta_update_budget = "matched"`
- `identity_loss_weight = 0.2`
- `hybrid_ramp_epochs = 10`
- `bootstrap_substeps = 4`
- `checkpoint_selector = "gate_constrained_accuracy_then_val_accuracy"`

Current preset interpretation:

- keep `tf2_canonical` as the hypothesis-driven iFMPC candidate
  - `micro_steps = 4`
  - `incremental_weight_updates = true`
  - `supervision_policy = "mixed"`
  - `theta_update_budget = "matched"`
  - `feature_aware_tangents = false`
- add `tf2_corrective_transport_default` as the empirical bridge winner
  - `micro_steps = 4`
  - `incremental_weight_updates = false`
  - `supervision_policy = "local_only"`
  - `theta_update_budget = "matched"`
- do not silently replace the hypothesis-driven preset with the empirical preset

JPC status after the completed probe:

- JPC remains reference-only in TF2
- TF2 must not depend on JPC runtime
- the completed JPC probe supports prioritizing incremental scheduling over
  substrate scaling in the current phase
- the probe does not provide strong evidence that muPC-style scaling should now
  replace incremental scheduling as the main TF2 focus
- muPC-style scaling remains a future candidate mechanism, not the present TF2
  mainline

TF2A suite grid:

- `incremental_weight_updates in {false, true}`
- `supervision_policy in {"local_only", "mixed"}`
- `micro_steps in {2, 4}`
- `seeds in {0, 1, 2}`

Keep fixed across the suite:

- `family_lineage = tf1_mlp_aug`
- `feature_aware_tangents = false`
- `identity_loss_weight = 0.2`
- `hybrid_ramp_epochs = 10`
- `bootstrap_substeps = 4`
- `checkpoint_selector = "gate_constrained_accuracy_then_val_accuracy"`
- `theta_update_budget = "matched"`

Forward-init stability diagnostics required in TF2:

- per-layer hidden forward-init RMS
- per-layer hidden forward-init mean L2 norm
- initial target-clamped energy
- initial hidden-gradient RMS
- initial hidden-gradient mean L2 norm

Acceptance criteria:

- must-have acceptance:
  - fully teacher-free
  - no JPC runtime dependency
  - forward-init diagnostics present
  - immediate theta updates happen each micro-step when enabled
  - mixed-policy supervision works when enabled
  - validation-only selector semantics are preserved
  - no NaN / Inf
- target acceptance:
  - improve mean validation accuracy over the sealed TF1 working default
  - improve mean test accuracy over the sealed TF1 working default
  - reduce the gap to the canonical slow-PC digits baseline

Risks / open questions:

- immediate theta drift may destabilize on-policy supervision
- mixed-policy supervision may improve gate coverage more than final slow-PC accuracy
- matched-budget micro-updates may still be too weak if the true bottleneck is substrate
  scaling rather than scheduling
- -PC-inspired conditioning concerns may matter, but TF2A treats them as diagnostics only
  and does not add a new scaling mechanism yet

Interpretation:

- current TF2 evidence supports corrective transport strongly
- current TF2 evidence does not yet support full incremental iFMPC /
  interleaved parameter-learning as the empirical winner
- if TF2 improves over sealed TF1 but still trails slow-PC materially, the next stage
  should remain inside `continue TF2 bridge`
- if TF2 improves weakly and the diagnostics continue to point to poor conditioning,
  the next stage should become `strengthen substrate scaling later`
- only a clearly stronger bridge result should justify `move toward generalized TF3 later`

### TF2 audit patch — EF alignment

Goal:

- audit the active TF2 bridge code against the current repository spec before making
  any further EF-style transport changes
- keep the patch minimal, explicit, and reversible

Audit targets:

1. verify the MeanFlow-style identity / JVP semantics when
   `use_teacher_free_features = true`
   - determine whether appended feature-dependent psi inputs require chain-rule
     directional-derivative terms in the identity target
   - if `feature_aware_tangents = false` remains allowed, fence that path explicitly
     as a truncated identity approximation rather than silently treating it as the
     full total derivative
2. verify the semantics of `theta_update_budget = "matched"` under:
   - `terminal_only`
   - `every_2_micro_steps`
   - `every_micro_step`
   so that matched budget is defined against the actual number of theta updates that
   are applied under the active cadence
3. strengthen the focused TF2 tests so the intended semantics are enforced

Files to touch:

- `PLANS.md`
- `spec_math.md`
- `validation.md`
- `src/pc/fmpc_tf1_jvp.py`
- `src/pc/fmpc_tf2.py`
- `tests/test_fmpc_tf2_dynamics.py`
- `tests/test_fmpc_tf2_targets.py`
- optionally `tests/test_fmpc_tf2_smoke.py` if new provenance/reporting fields are
  added

Planned minimal changes:

- make the TF2/spec wording explicit about the difference between:
  - feature-aware total-derivative identity semantics
  - truncated feature-frozen identity semantics
- correct matched-budget theta micro learning rates so they normalize by the number
  of theta updates implied by the active cadence rather than always by
  `micro_steps`
- add focused tests that would have failed before the patch

Validation to run:

- `tests/test_fmpc_tf2_dynamics.py`
- `tests/test_fmpc_tf2_targets.py`
- `tests/test_fmpc_tf2_smoke.py`

Assumptions:

- TF2 remains teacher-free and NumPy-only
- this patch does not introduce muPC-style scaling, JPC runtime dependence, or a new
  selector policy
- if the current default `feature_aware_tangents = false` is kept, it must be
  documented as an explicit approximation rather than an unqualified `D_T`

### TF2 identity semantics decision pass

Goal:

- decide whether the canonical TF2 identity semantics should remain:
  - `feature_aware_tangents = false`
  as the current truncated identity approximation
- or switch to:
  - `feature_aware_tangents = true`
  as the augmented-input total-derivative approximation

Scope:

- do not change TF2 core math
- do not change baseline PC math
- do not introduce muPC-style scaling
- do not start TF3
- only compare the two identity semantics under otherwise matched TF2 settings

Files to touch:

- `PLANS.md`
- `validation.md`
- optionally `spec_math.md` if the default decision needs extra contract wording
- `src/pc/__init__.py`
- `src/pc/fmpc_tf2_identity_semantics_suite.py`
- `experiments/fmpc_tf2_identity_semantics_suite.py`
- `tests/test_fmpc_tf2_identity_semantics_suite_smoke.py`
- `tests/test_fmpc_tf2_smoke.py`

Planned suite:

- compare `feature_aware_tangents in {false, true}`
- evaluate at least:
  - `tf2_canonical`
  - `tf2_corrective_transport_default`
- keep each preset's own:
  - selector
  - `micro_steps`
  - supervision policy
  - theta-update cadence
  - theta-update budget
- seeds:
  - `{0, 1, 2, 3, 4}`

Decision rule:

- promote feature-aware tangents to the canonical TF2 default only if the
  `tf2_canonical` runs show:
  - stable completion with no NaN/Inf failures
  - materially better validation-selected behavior
  - and no meaningful regression in gate-passing coverage
- otherwise keep the truncated identity approximation as the canonical default and
  document that it remains preferred empirically

Validation to run:

- `tests/test_fmpc_tf2_identity_semantics_suite_smoke.py`
- `tests/test_fmpc_tf2_targets.py`
- `tests/test_fmpc_tf2_smoke.py`

Deliverables:

- `outputs/fmpc_tf2_identity_semantics_suite/config.json`
- `outputs/fmpc_tf2_identity_semantics_suite/aggregate_runs.csv`
- `outputs/fmpc_tf2_identity_semantics_suite/aggregate_summary.json`

Decision outcome:

- the completed matched identity-semantics suite showed:
  - no validation accuracy gain
  - no test accuracy gain
  - no gate-coverage gain
  from `feature_aware_tangents = true` under either:
  - `tf2_canonical`
  - `tf2_corrective_transport_default`
- the canonical TF2 default therefore remains:
  - `feature_aware_tangents = false`
- feature-aware tangents remain available as a more complete augmented-input identity
  approximation, but not as the current default

### TF2 corrective-transport attribution pass

Goal:

- explain, with the smallest factor-isolating ablations, why
  `tf2_corrective_transport_default` currently beats or is preferred over
  `tf2_canonical`
- narrow the next post-identity-semantics research move to one concrete direction

Scope:

- do not change TF2 core math
- do not reopen the TF2 identity-semantics decision
- keep:
  - `feature_aware_tangents = false`
  - `theta_update_budget = "matched"`
  - the current selector policy
- compare only the existing TF2 bridge family on `digits`

Files to touch:

- `README.md`
- `PLANS.md`
- `src/pc/__init__.py`
- `src/pc/fmpc_tf2_attribution_suite.py`
- `experiments/fmpc_tf2_attribution_suite.py`
- `tests/test_fmpc_tf2_attribution_suite_smoke.py`

Planned suite:

- use a small hand-built attribution grid around:
  - `tf2_canonical`
  - `tf2_corrective_transport_default`
- isolate the current candidate factors:
  - theta-update cadence:
    - `terminal_only`
    - `every_2_micro_steps`
    - `every_micro_step`
  - supervision policy:
    - `local_only`
    - `mixed`
  - interleaving start:
    - `epoch_0`
    - `after_warmup`
  - micro-step count:
    - `2`
    - `4`
- keep:
  - family lineage
  - identity semantics
  - selector policy
  - dataset
  fixed

Validation to run:

- `tests/test_fmpc_tf2_attribution_suite_smoke.py`
- `tests/test_fmpc_tf2_smoke.py`

Expected deliverables:

- one attribution suite artifact set under:
  - `outputs/fmpc_tf2_attribution_suite/`
- an evidence-backed explanation for:
  - why the corrective transport default is currently preferred
  - which factor should remain default
  - what single next change is most promising to narrow the slow-PC gap

Attribution outcome:

- the completed attribution suite shows that the current empirical TF2 advantage is
  explained primarily by cadence:
  - moving away from `every_micro_step` toward `terminal_only` under matched budget
    produces the largest stable gain
- `local_only` supervision adds a smaller secondary gain once cadence is already
  `terminal_only`
- `every_2_micro_steps` and `after_warmup` each partially rescue the canonical
  hypothesis preset, but neither beats `tf2_corrective_transport_default`
- `micro_steps = 4` is preferred over `micro_steps = 2` in both the canonical and
  corrective families
- no tested attribution configuration narrows the slow-PC test gap below the current
  corrective default
- the narrow next move after attribution should therefore be:
  - keep `tf2_corrective_transport_default`
  - keep `feature_aware_tangents = false`
  - keep `theta_update_budget = "matched"`
  - test a slightly larger micro-step count before reopening broader TF2 semantics

### TF2 corrective micro-step horizon pass

Goal:

- determine whether increasing `micro_steps` beyond `4` inside the current
  corrective-transport default produces a genuine transport-horizon gain
  or merely increases inner-loop training compute

Scope:

- do not change TF2 core math
- do not reopen identity semantics, cadence semantics, or supervision semantics
- keep fixed:
  - `feature_aware_tangents = false`
  - `incremental_weight_updates = false`
  - `supervision_policy = "local_only"`
  - `theta_update_cadence = "terminal_only"`
  - `theta_update_budget = "matched"`
  - current selector policy
- vary only:
  - `micro_steps in {4, 6, 8, 10}`
  - comparison protocol:
    - fixed outer-training
    - matched inner-compute

Files to touch:

- `PLANS.md`
- `validation.md`
- `src/pc/__init__.py`
- `src/pc/fmpc_tf2_microstep_horizon_suite.py`
- `experiments/fmpc_tf2_microstep_horizon_suite.py`
- `tests/test_fmpc_tf2_microstep_horizon_suite_smoke.py`

Matching rule:

- use the current corrective default as the base:
  - `base_micro_steps = 4`
  - `base_epochs = 60`
- fixed outer-training protocol:
  - keep `epochs = 60`
- matched inner-compute protocol:
  - set `epochs = round(base_epochs * base_micro_steps / micro_steps)`
  - with the active grid this becomes:
    - `4 -> 60`
    - `6 -> 40`
    - `8 -> 30`
    - `10 -> 24`
- this keeps the coarse training budget
  - `epochs * micro_steps`
  exactly matched at `240`

Validation to run:

- `tests/test_fmpc_tf2_microstep_horizon_suite_smoke.py`
- `tests/test_fmpc_tf2_smoke.py`

Expected deliverables:

- one narrow suite artifact set under:
  - `outputs/fmpc_tf2_microstep_horizon_suite/`
- a decision on whether:
  - `micro_steps > 4` remains better under matched inner compute
  - or the current gain is mainly a compute-budget effect

Outcome:

- the completed micro-step horizon suite shows:
  - no instability through `micro_steps = 10` under either protocol
  - under fixed outer training, larger `micro_steps` continue to improve:
    - best tested mean test accuracy: `micro_steps = 10`
  - under matched inner compute, `micro_steps = 4` remains the clear winner and
    larger values degrade:
    - `6`, `8`, and `10` all lose validation-selected test accuracy
    - gate-passing coverage also falls sharply as `micro_steps` grows
- the current evidence therefore says:
  - the apparent gain from `micro_steps > 4` is mainly a compute-budget effect,
    not a genuine transport-horizon win
  - `micro_steps = 4` should remain the corrective-transport default
- the next narrow TF2 move should be:
  - keep `micro_steps = 4`
  - test one transport-quality change inside the current corrective default
    rather than adding more inner-loop micro-step compute

### TF2 bootstrap-identity curriculum pass

Goal:

- determine whether a better curriculum between the existing bootstrap target and
  the existing identity target improves the current fixed-4-step corrective
  default without adding extra micro-step compute

Scope:

- do not change TF2 core math
- do not reopen:
  - identity semantics
  - cadence semantics
  - supervision semantics
  - micro-step count
- keep fixed:
  - `preset = tf2_corrective_transport_default`
  - `use_teacher_free_features = true`
  - `feature_aware_tangents = false`
  - `micro_steps = 4`
  - `incremental_weight_updates = false`
  - `supervision_policy = "local_only"`
  - `theta_update_cadence = "terminal_only"`
  - `theta_update_budget = "matched"`
  - `bootstrap_integrator = "rk2"`
  - `bootstrap_substeps = 4`
  - current selector policy
- vary only:
  - `identity_loss_weight`
  - `warmup_epochs`
  - `hybrid_ramp_epochs`

Files to touch:

- `PLANS.md`
- `validation.md`
- `src/pc/__init__.py`
- `src/pc/fmpc_tf2_curriculum_suite.py`
- `experiments/fmpc_tf2_curriculum_suite.py`
- `tests/test_fmpc_tf2_curriculum_suite_smoke.py`

Staged / pruned search rule:

- Stage 1: run only the single-axis sweeps around the current default:
  - identity sweep:
    - `(0.1, 5, 10)`
    - `(0.2, 5, 10)`  current default
    - `(0.4, 5, 10)`
  - warmup sweep:
    - `(0.2, 0, 10)`
    - `(0.2, 5, 10)`  current default
    - `(0.2, 10, 10)`
  - ramp sweep:
    - `(0.2, 5, 0)`
    - `(0.2, 5, 10)`  current default
    - `(0.2, 5, 20)`
    - `(0.2, 5, 40)`
- Stage 2:
  - identify the single best non-default setting on each axis
  - only if one or more axis winners beats the current default by a material
    margin in mean validation-selected test accuracy, run exactly one combined
    candidate that merges those winning axis values
  - otherwise stop after Stage 1

Materiality threshold for opening Stage 2:

- mean test-accuracy delta vs the current default `>= 0.005`
- mean validation-accuracy delta vs the current default `>= 0.0`
- no increase in failure / NaN incidence

Validation to run:

- `tests/test_fmpc_tf2_curriculum_suite_smoke.py`
- `tests/test_fmpc_tf2_smoke.py`

Expected deliverables:

- one narrow suite artifact set under:
  - `outputs/fmpc_tf2_curriculum_suite/`
- a decision on whether:
  - better bootstrap↔identity curriculum improves fixed-4-step transport quality
  - the corrective default should change
  - or the next move should narrow to bootstrap-target fidelity rather than
    curriculum

Outcome:

- the completed curriculum suite shows:
  - no instability or NaN/Inf incidence across the tested settings
  - no stage-1 axis winner clears the materiality threshold for opening stage 2
  - the best non-default setting is:
    - `identity_loss_weight = 0.1`
    - `warmup_epochs = 5`
    - `hybrid_ramp_epochs = 10`
  - but its mean test gain over the current default is only about `+0.0015`,
    with no mean validation gain
- the current evidence therefore says:
  - no tested bootstrap↔identity curriculum materially improves the current
    fixed-4-step corrective default
  - `tf2_corrective_transport_default` should keep:
    - `identity_loss_weight = 0.2`
    - `warmup_epochs = 5`
    - `hybrid_ramp_epochs = 10`
- the next narrow TF2 move should now be:
  - keep the current curriculum
  - test bootstrap-target fidelity rather than more curriculum tuning

### TF2 bootstrap-target fidelity pass

Goal:

- determine whether the current fixed-4-step corrective default is bottlenecked by
  bootstrap-target fidelity rather than by transport horizon or curriculum

Scope:

- do not change TF2 core math
- do not reopen:
  - identity semantics
  - supervision semantics
  - cadence semantics
  - micro-step count
  - curriculum knobs
- keep fixed:
  - `preset = tf2_corrective_transport_default`
  - `use_teacher_free_features = true`
  - `feature_aware_tangents = false`
  - `micro_steps = 4`
  - `incremental_weight_updates = false`
  - `supervision_policy = "local_only"`
  - `theta_update_cadence = "terminal_only"`
  - `theta_update_budget = "matched"`
  - `identity_loss_weight = 0.2`
  - `warmup_epochs = 5`
  - `hybrid_ramp_epochs = 10`
  - current selector policy
- vary only the existing bootstrap target construction:
  - `bootstrap_integrator in {"euler", "rk2"}`
  - `bootstrap_substeps in {1, 2, 4, 8, 16}`

Files to touch:

- `PLANS.md`
- `validation.md`
- `src/pc/__init__.py`
- `src/pc/fmpc_tf2_bootstrap_fidelity_suite.py`
- `experiments/fmpc_tf2_bootstrap_fidelity_suite.py`
- `tests/test_fmpc_tf2_bootstrap_fidelity_suite_smoke.py`

Offline-first evaluation rule:

- do not begin with a training grid
- first build a direct bootstrap-target fidelity probe on shared sampled
  corrective-default states `(z_t, t_k, r_k)`
- sample states from the actual corrective-default training regime using only the
  local-only shadow stream `z_lf_k`, since that is where `u_boot` is supervised in
  the current default
- compare every candidate target against a fixed high-fidelity reference built from:
  - the same `hidden_local_flow(context, z)`
  - the same horizon `r_k`
  - `bootstrap_integrator = "rk2"`
  - `bootstrap_substeps = 64`

Offline metrics to report:

- MSE to the reference average velocity
- relative MSE
- cosine similarity to the reference
- endpoint displacement error over horizon `r`
- hidden energy after one bootstrap transport step
- wall-clock cost per target evaluation

Pruning rule:

- always keep the current default candidate:
  - `("rk2", 4)`
- rank non-default candidates by offline target fidelity
- carry forward at most 2 promising non-default candidates into end-to-end TF2
  runs

Validation to run:

- `tests/test_fmpc_tf2_bootstrap_fidelity_suite_smoke.py`
- `tests/test_fmpc_tf2_smoke.py`

Expected deliverables:

- one narrow artifact set under:
  - `outputs/fmpc_tf2_bootstrap_fidelity_suite/`
- a decision on whether:
  - higher-fidelity `u_boot` materially improves held-out corrective TF2 behavior
  - the corrective default should change
  - or bootstrap-target fidelity is not the current limiter

Outcome:

- the completed bootstrap-target fidelity suite now indicates:
  - offline target fidelity does improve monotonically as `bootstrap_substeps`
    increase under `rk2`
  - but the current default `rk2_s4` is already extremely close to the shared
    `rk2_s64` reference under the same local field and horizon semantics
  - the pruned end-to-end candidates:
    - `rk2_s4`
    - `rk2_s8`
    - `rk2_s16`
    produce identical validation-selected accuracy and gate-coverage behavior
    within the current multiseed study
  - the higher-fidelity candidates only reduce
    `val_transported_final_energy` at floating-point-noise scale while adding
    substantial runtime cost
  - `tf2_corrective_transport_default` should therefore keep:
    - `bootstrap_integrator = "rk2"`
    - `bootstrap_substeps = 4`
  - current evidence says:
    - `u_boot` fidelity is not the current limiter for the fixed-4-step
      corrective default
- the next single narrow TF2 move should now go beyond
  curriculum/bootstrap-fidelity tuning and target a different transport-quality
  bottleneck

### TF2 bootstrap-target source-bias pass

Goal:

- determine whether the current fixed-4-step corrective default is bottlenecked by
  bootstrap-target source bias rather than by bootstrap-target numerical fidelity

Scope:

- do not change TF2 core math
- do not reopen:
  - identity semantics
  - supervision semantics
  - cadence semantics
  - micro-step count
  - curriculum knobs
  - bootstrap integrator / substeps
- keep fixed:
  - `preset = tf2_corrective_transport_default`
  - `use_teacher_free_features = true`
  - `feature_aware_tangents = false`
  - `micro_steps = 4`
  - `incremental_weight_updates = false`
  - `supervision_policy = "local_only"`
  - `theta_update_cadence = "terminal_only"`
  - `theta_update_budget = "matched"`
  - `identity_loss_weight = 0.2`
  - `warmup_epochs = 5`
  - `hybrid_ramp_epochs = 10`
  - `bootstrap_integrator = "rk2"`
  - `bootstrap_substeps = 4`
  - current selector policy
- vary only the bootstrap terminal-source family:
  - current local-field endpoint `z_T^lf`
  - diagnostic-only detached slow-PC endpoints `z_T^pc[K]` for a tiny
    `K in {4, 8, 16}` family
  - optional single blended diagnostic source only if the offline probe clearly
    justifies it

Files to touch:

- `PLANS.md`
- `validation.md` only if the conclusion changes the next narrow move
- `src/pc/__init__.py`
- `src/pc/fmpc_tf2_bootstrap_source_bias_suite.py`
- `experiments/fmpc_tf2_bootstrap_source_bias_suite.py`
- `tests/test_fmpc_tf2_bootstrap_source_bias_suite_smoke.py`

Offline-first evaluation rule:

- do not begin with a training grid
- first build a source-bias diagnostic on shared sampled corrective-default
  states `(z_t, t_k, r_k)`
- compare:
  - `u_lf = (z_T^lf - z_t) / r_k`
  - `u_pc[K] = (z_T^pc[K] - z_t) / r_k`
- all `z_T^pc[K]` endpoints must be:
  - detached
  - `theta`-frozen
  - produced by baseline PC hidden-state inference only
  - labeled diagnostic-only / baseline-only

Offline metrics to report:

- MSE and cosine similarity between `u_lf` and `u_pc[K]`
- endpoint hidden energy for each source
- endpoint output-error quantity for each source
- endpoint classification accuracy for each source
- wall-clock cost per target evaluation

Pruning rule:

- always keep the current local-field source
- select at most one detached slow-PC challenger for end-to-end comparison

Validation to run:

- `tests/test_fmpc_tf2_bootstrap_source_bias_suite_smoke.py`
- `tests/test_fmpc_tf2_smoke.py`

Expected deliverables:

- one narrow artifact set under:
  - `outputs/fmpc_tf2_bootstrap_source_bias_suite/`
- a decision on whether:
  - the current `u_boot` is bottlenecked by terminal-source bias
  - a detached slow-PC source materially beats the current local-field source
  - or the next narrow bottleneck lies elsewhere

Outcome:

- the completed bootstrap-target source-bias suite now indicates:
  - detached slow-PC endpoints do look slightly stronger than the current
    local-field endpoint in the offline diagnostic:
    - lower endpoint hidden energy
    - lower output MSE
    - slightly higher endpoint accuracy
  - the best offline challenger in the current tiny family is:
    - `diagnostic_slow_pc_k16`
  - however, the end-to-end comparison between:
    - current local-field source
    - detached slow-PC `K = 16`
    shows no validation-selected accuracy gain and no test-accuracy gain
  - the detached slow-PC challenger only trades extra runtime for:
    - a small gate-count increase
    - a small reduction in transported validation energy
    - with no held-out accuracy benefit
  - current evidence therefore says:
    - the fixed-4-step corrective default is not bottlenecked by terminal-source
      bias
    - the mainline-safe result remains:
      - `tf2_corrective_transport_default`
      - local-field bootstrap source
  - detached slow-PC sources remain diagnostic-only / baseline-only and should
    not be promoted directly into the TF2 mainline
- the originally logged next single narrow TF2 move at that point was:
- psi-side transport expressivity under the fixed teacher-free local-field
  source
- latest local diagnostic chain has since advanced beyond that point through:
  psi-expressivity -> downstream coupling -> lag1 target snapshot -> batch-frozen
  target/state cache -> open-vs-closed-loop trajectory coupling -> partial-open-loop
  handoff localization -> mirrored handoff asymmetry -> terminal-step
  supervision-bundle split -> terminal-step action-output stabilization -> terminal-step direction anchoring
- active narrow local diagnostic question has since moved to:
  terminal local-field direction trust-region in the true closed-loop regime
