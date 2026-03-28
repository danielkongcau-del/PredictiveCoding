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

## Phase 3 — Real dataset baseline

### Goal

Move from toy data to a standard small dataset while keeping the codebase simple.

### Scope

- MNIST or another small tabular / image baseline
- mini-batch training
- train / validation split
- saved learning curves

### Deliverables

- `experiments/mnist_mlp.py`
- dataset loading helper(s)
- validation metrics
- reproducible config for at least one baseline run

### Exit criteria

- stable training on a standard dataset
- clear logs and plots for training / validation behavior
- documented runtime and hyperparameters

### Risks

- blaming the implementation for dataset preprocessing issues
- performance chasing before baseline stability is established

---

## Phase 4 — Deeper networks and scaling groundwork

### Goal

Prepare the implementation for deeper architectures without losing interpretability.

### Scope

- deeper MLP support
- improved initialization choices
- more careful inference scheduling
- optional damping / clipping / normalization safeguards

### Deliverables

- experiments showing depth > initial baseline
- documented stabilization heuristics
- tests that cover multi-layer edge cases

### Exit criteria

- deeper networks run reliably on small benchmarks
- stabilization logic is documented and not hidden in ad hoc code

### Risks

- hiding fundamental math issues behind heuristics
- introducing too many tuning knobs at once

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
