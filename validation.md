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
