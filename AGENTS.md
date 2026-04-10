# AGENTS.md

This file defines the standing working rules for Codex and human contributors in this repository.

## Mission

Build a reliable, extensible, near-paper-grade predictive coding implementation from scratch, starting with a NumPy-only baseline.

The repository should favor:

- mathematical fidelity to [spec_math.md](/e:/CodeSpace/PredictiveCoding/spec_math.md)
- explicit code paths
- testability
- incremental extension

## Mandatory Read Order

Before editing code, read:

1. `README.md`
2. `AGENTS.md`
3. `CURRENT_STATE.md`
4. `PLANS.md`
5. `spec_math.md`
6. the applicable stage addendum under `specs/` if the task touches stage-specific FMPC logic
7. `validation.md`

If repository documents conflict, precedence remains:

- `spec_math.md > validation.md > AGENTS.md > CURRENT_STATE.md > PLANS.md > README.md`

For low-context prompt drafting or web GPT orientation:

- start with `PROMPT_CONTEXT.md`
- then read `LLM_BRIEF.md`, `AGENTS.md`, and `CURRENT_STATE.md`
- open `RESULTS.md`, `spec_math.md`, the applicable `specs/` addendum, `PLANS.md`, `validation.md`, or `archive/` only as needed

This does not replace the mandatory read order above when editing code.

## Non-Negotiable Constraints

1. Do not introduce deep learning frameworks in early repository phases unless explicitly requested.
   Forbidden by default:
   - PyTorch
   - JAX
   - TensorFlow
2. Do not silently change the mathematics.
   - If a change affects the energy, state updates, weight updates, or acceptance contract, update `spec_math.md` first or call out the inconsistency.
3. Keep external APIs batch-first.
   - Arrays should default to shape `(batch, features)` unless explicitly documented otherwise.
4. Every nontrivial algorithmic change must include validation.
   - Add or update tests.
   - Run the relevant test subset.
5. Keep randomness explicit.
   - Use fixed seeds for tests and experiments.

## Current FMPC Status

- Teacher-based FMPC remains frozen as baseline / diagnostic reference.
- The post-reference FMPC line begins at:
  - `stage_03_transport_core_v1/`
- Human-readable FMPC stage names are:
  - `stage_01_reference_prep/` -> `FMPC Stage 01 Reference Prep`
  - `stage_02_interval_velocity/` -> `FMPC Stage 02 Interval Velocity Exploration`
  - `stage_03_transport_core_v1/` -> `FMPC Stage 03 Transport Core v1`
  - `stage_04_incremental_bridge/` -> `FMPC Stage 04 Incremental Bridge`
  - `stage_05_ef_core_probe/` -> `FMPC Stage 05 EF Core Probe`
- FMPC Stage 03 is sealed as the first artifact-independent transport stage.
- The current active FMPC stage is:
  - `FMPC Stage 04 Incremental Bridge`
- Unless something is explicitly labeled baseline or diagnostic, new FMPC work must not depend on:
  - teacher trajectories
  - teacher fixed points
  - teacher-generated regression targets

## When A Written Plan Is Required

Create or update a written plan before coding if the task involves:

- a new model variant
- changes spanning more than 2 source files
- inference-dynamics changes
- objective / energy changes
- training-loop or validation-criteria changes
- semantic performance optimizations

A sufficient plan can be short, but it must state:

- what will change
- which files will be touched
- what tests or validation will be run
- what assumptions are being made

## Engineering Style

- Prefer readable NumPy over clever abstraction.
- Small explicit functions beat hidden magic.
- Public functions should document shape expectations.
- Prefer `float64` in early phases unless there is a clear reason not to.
- Avoid obscure broadcasting tricks when they reduce legibility.
- If you clip, regularize, or stabilize something, comment why the safeguard exists.

Core experiment code should make it easy to inspect:

- energy per inference step
- parameter norms
- activation norms
- optional gradient or update norms

## Module Boundary Expectations

The repository should continue to evolve toward these responsibilities:

- `activations.py`
- `energy.py`
- `inference.py`
- `layers.py`
- `models.py`
- `training.py`
- `utils.py`

Do not collapse everything into one file unless the phase explicitly calls for a tiny prototype.

## Testing Expectations

At minimum, the repository should continue to support tests for:

- shape consistency
- deterministic initialization under fixed seeds
- energy computation on a small known example
- inference reducing energy in a smoke test
- weight updates moving parameters in the expected direction
- end-to-end learning on a toy task

Where practical, keep tests small and deterministic.

## Allowed Early-Phase Simplifications

These remain acceptable in the early baseline line:

- MSE on one-hot labels
- identity output activation
- simple SGD-style parameter updates
- fixed inference step count
- no trainable recognition network
- CPU-only execution

## Disallowed Shortcuts

Do not do these unless explicitly approved by the spec or plan:

- replace iterative inference with backprop while still calling it predictive coding
- add autodiff-based gradient computation for the baseline path
- hide the local update rule behind opaque helper frameworks
- change the layer ordering convention without updating the spec
- write tests that only assert code runs without checking behavior

## What A Good Change Looks Like

A good change:

- matches `spec_math.md`
- is small enough to review
- includes tests or validation hooks
- makes future phases easier rather than harder
- leaves the repository more legible

## Patch Summary Expectations

When summarizing a change, include:

1. what changed
2. why it changed
3. what assumptions were made
4. what tests were run
5. any unresolved questions

## Ambiguity Protocol

If the math, shape conventions, or phase goals seem ambiguous:

- do not guess silently
- point to the ambiguous section
- propose a concrete resolution
- proceed only once the chosen interpretation is explicit

## History

This short AGENTS file keeps the current standing rules only.

Historical long-form wording now lives in:

- [archive/AGENTS_HISTORY.md](/e:/CodeSpace/PredictiveCoding/archive/AGENTS_HISTORY.md)
