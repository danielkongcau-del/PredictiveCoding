# AGENTS.md

This file defines the standing instructions for Codex and human contributors working in this repository.

## Mission

Build a **reliable, extensible, near-paper-grade predictive coding implementation** from scratch, starting with a NumPy-only baseline.

The repository should favor:

- mathematical fidelity to `spec_math.md`
- explicit code paths
- testability
- incremental extension

## Mandatory read order

Before editing code, read:

1. `README.md`
2. `AGENTS.md`
3. `PLANS.md`
4. `spec_math.md`
5. `validation.md`

Do not assume the current code is authoritative if it conflicts with the spec.

## Hard constraints

1. **Do not introduce deep learning frameworks** in Phase 0–2.
   - Forbidden unless explicitly requested: PyTorch, JAX, TensorFlow.

2. **Do not silently change the mathematics.**
   - If a code change requires altering the energy, state updates, or weight updates, update `spec_math.md` first or raise the inconsistency.

3. **Keep APIs batch-first.**
   - External-facing tensors/arrays should use shape `(batch, features)` unless explicitly documented otherwise.

4. **Every nontrivial algorithmic change must include validation.**
   - Add or update tests.
   - Run the relevant test subset.

5. **Prefer readable NumPy to clever abstractions.**
   - Explicit loops over layers are acceptable.
   - Hidden magic, metaprogramming, and unnecessary class hierarchies are discouraged.

6. **Document shape contracts.**
   - Every public function must state expected shapes.

7. **Keep randomness controlled.**
   - Experiments and tests should use explicit seeds.

## When to make a plan first

Create or update a written plan before coding if the task involves any of the following:

- a new model variant
- changes spanning more than 2 source files
- modifications to inference dynamics
- changes to the objective / energy
- changes to training loops or validation criteria
- performance optimization affecting semantics

A plan can be short, but it must specify:

- what changes will be made
- what files will be touched
- what tests will verify success
- what assumptions are being made

## Coding standards

### General

- Python 3.11+
- Type hints for public functions where practical
- Small functions over monolithic scripts
- Minimize hidden state
- Avoid in-place mutation unless it is intentional and documented

### Numerical style

- Prefer `float64` in early phases unless otherwise justified
- Avoid unnecessary broadcasting tricks if they obscure the math
- When clipping or stabilizing, comment why the safeguard exists

### Logging / observability

Core experiment code should make it easy to inspect:

- energy per inference step
- parameter norms
- activation norms
- optional gradient or update norms

## Required module boundaries

The implementation should generally evolve toward these responsibilities:

- `activations.py`: activation functions and derivatives
- `energy.py`: energy computation and helper terms
- `inference.py`: latent/state update dynamics
- `layers.py`: parameter containers and initialization helpers
- `models.py`: network composition
- `training.py`: training loop / optimizer-like routines
- `utils.py`: seeding, metrics, serialization helpers

Do not merge everything into a single file unless the phase explicitly calls for a tiny prototype.

## Test expectations

At minimum, the repository should eventually contain tests for:

- shape consistency
- deterministic initialization under fixed seeds
- energy computation on a small known example
- inference reducing energy in a smoke test
- weight updates moving parameters in the expected direction
- end-to-end learning on a toy task

Where possible, tests should use very small networks and small synthetic datasets.

## Allowed simplifications in early phases

The following are acceptable in Phase 0–1:

- MSE objective on one-hot labels
- identity output activation
- simple SGD-style parameter updates
- fixed inference step count
- no trainable recognition network
- CPU-only execution

## Disallowed shortcuts

Avoid the following unless explicitly approved by the spec or plan:

- replacing iterative inference with backprop while still calling it PC
- adding autodiff-based gradient computation for the baseline path
- using hidden helper frameworks that obscure the local update rule
- changing the layer ordering convention without updating the spec
- writing tests that merely assert code runs without checking behavior

## Definition of a good change

A good contribution:

- matches `spec_math.md`
- is small enough to review
- includes tests or validation hooks
- makes future phases easier rather than harder
- leaves the repo more legible than before

## Commit / patch style for Codex

When producing a patch or summary, include:

1. What was changed
2. Why it was changed
3. What assumptions were made
4. What tests were run
5. Any unresolved questions

## If you detect ambiguity

If the math, shape conventions, or phase goals appear ambiguous:

- do not guess silently
- point to the ambiguous section
- propose a concrete resolution
- proceed only if the chosen interpretation is made explicit
