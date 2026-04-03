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
  - did the prior Phase 2g headline survive the boundary check?
  - which benchmark is still boundary-sensitive?
  - is Phase 2 stable enough to move on?

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

## Phase 5 — Variant extensions

### Goal

Add explicitly labeled predictive coding variants without breaking the baseline.

### Candidate variants

- separate recognition / initialization network
- bidirectional weights
- convolutional PC
- temporal / recurrent PC
- alternative energies or output likelihoods

### Rules for this phase

- each variant must live behind an explicit name
- baseline behavior must remain preserved
- new math must be documented in `spec_math.md` or a variant-specific spec

### Exit criteria

- at least one non-baseline variant is implemented and validated
- users can tell which formulation they are running

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
