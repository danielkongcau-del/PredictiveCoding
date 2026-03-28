# Predictive Coding from Scratch

A NumPy-first research codebase for building a **reliable, extensible, near-paper-grade** predictive coding (PC) implementation.

This repository is intended to be developed collaboratively with Codex. The design goal is to avoid the common failure mode of producing code that merely "looks like predictive coding" without being mathematically pinned down, testable, or extensible.

## Current scope

Phase 0 and Phase 1 focus on a **supervised, fully-connected, batch-first predictive coding network** implemented with:

- Python 3.11+
- NumPy
- SciPy (optional, for numerical checks / solvers in later phases)
- pytest
- matplotlib

Out of scope for the first implementation:

- PyTorch / JAX / TensorFlow
- CNNs
- RNNs / temporal PC
- advanced optimizers
- mixed precision / GPU kernels

Those may be added later, but only after the Phase 0–2 math and tests are stable.

## Read order for Codex and human contributors

Before making code changes, read these files in order:

1. `README.md`
2. `AGENTS.md`
3. `PLANS.md`
4. `spec_math.md`
5. `validation.md`

If any of those files conflict, precedence is:

`spec_math.md` > `validation.md` > `AGENTS.md` > `PLANS.md` > `README.md`

## Design principles

1. **Math first.** Code must implement the current mathematical spec, not a guessed variant.
2. **Small verified steps.** Every nontrivial change should preserve or improve tests.
3. **Local clarity over premature cleverness.** Prefer simple, inspectable NumPy implementations before optimization.
4. **Batch-first everywhere.** Public APIs should default to arrays shaped `(batch, features)`.
5. **Extensibility without abstraction theater.** Modules should be small and explicit, but not over-engineered.

## Proposed repository layout

```text
predictive-coding/
  README.md
  AGENTS.md
  PLANS.md
  spec_math.md
  validation.md
  pyproject.toml          # or requirements.txt + setup.cfg later
  src/
    pc/
      __init__.py
      activations.py
      energy.py
      inference.py
      layers.py
      models.py
      training.py
      utils.py
  experiments/
    toy_linear.py
    toy_mlp.py
    mnist_mlp.py          # later phase
  tests/
    test_shapes.py
    test_energy.py
    test_inference.py
    test_weight_updates.py
    test_regression_smoke.py
  outputs/
    .gitkeep
  references/
    notes.md
```

## Initial implementation target

The first usable milestone is a minimal supervised PC network with:

- 2–3 fully connected layers
- configurable hidden activation
- squared-error energy
- iterative hidden-state inference
- local weight updates
- per-step energy tracking
- toy regression and toy classification examples
- at least 5 meaningful tests

## Suggested development workflow with Codex

Use this loop:

1. Ask Codex to restate the relevant part of `spec_math.md` before coding.
2. Ask it to propose a narrow plan for the current phase.
3. Implement one small feature at a time.
4. Run tests after every algorithmic change.
5. If the mathematical spec changes, update `spec_math.md` first.

## Example prompt to give Codex

```text
Goal:
Implement Phase 0 of this predictive coding repository.

Context:
Read README.md, AGENTS.md, PLANS.md, spec_math.md, and validation.md first.
We are implementing a NumPy-only supervised predictive coding network.

Constraints:
- Do not introduce torch/jax/tensorflow.
- Do not change the math without updating spec_math.md.
- Keep all public arrays batch-first.
- Add or update tests for every algorithmic change.
- Prefer explicit, readable NumPy over abstraction-heavy code.

Done when:
- A minimal PCNetwork can run end-to-end on a toy problem.
- Energy decreases during inference on a deterministic smoke test.
- All Phase 0 validation checks pass.
```

## Practical notes

- For classification in early phases, use **MSE on one-hot targets** for simplicity.
- Output activations should default to identity in Phase 0.
- Softmax and cross-entropy are later additions, not initial requirements.
- The first implementation is expected to prioritize correctness and inspectability over speed.
- In prediction mode with `state_init="forward"`, inference reduces to the forward pass under the current Phase 0 formulation because the initialized local prediction errors are already zero.
- That prediction-time behavior is expected for this baseline and should not be treated as a bug unless the mathematical spec changes.

## Non-goals for the first version

The following are deliberately postponed:

- reproducing a specific paper's full benchmark table
- automatic hyperparameter search
- deep-network stabilization tricks beyond simple documented safeguards
- custom C/C++ kernels
- multi-GPU / distributed training

## Expected outputs for each experiment

Every experiment script should eventually save:

- a JSON or YAML config dump
- scalar logs per epoch
- optionally, per-inference-step energy traces
- at least one plot to `outputs/`

## A note on predictive coding variants

There are many valid PC formulations. This repository starts with a **single clearly specified baseline formulation** so that implementation and validation stay coherent. New variants should be added only after the baseline is stable and should be explicitly labeled rather than silently replacing the baseline.
