# Predictive Coding from Scratch

A NumPy-first research codebase for building a **reliable, extensible, near-paper-grade** predictive coding (PC) implementation.

This repository is intended to be developed collaboratively with Codex. The design goal is to avoid the common failure mode of producing code that merely "looks like predictive coding" without being mathematically pinned down, testable, or extensible.

## Project status

Current stable milestone:

- Phase 0 complete: predictive coding baseline math and tests are stable
- Phase 1 complete: structured experiment outputs and benchmark scripts added
- Phase 1.5 complete: reproducibility seed semantics clarified and output layout supports:
  - default: `single_dir`
  - optional archival mode: `run_id_subdir`

Current scope:

- NumPy-only predictive coding baseline
- structured outputs under `outputs/`
- toy regression, sine regression, and blobs classification benchmarks

Not included yet:

- baseline MLP comparison
- CNN/RNN/temporal PC
- MNIST-scale experiments

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

- 2-3 fully connected layers
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
- optional plots to `outputs/` when plot generation is enabled

Phase 1 now standardizes the on-disk run layout as:

```text
outputs/
  <experiment_name>/
    config.json
    epoch_metrics.csv
    summary.json
    energy_traces.npz
    energy_traces_manifest.json
    plots/                 # only when plot generation is enabled
```

Notes on the saved artifacts:

- each benchmark now writes into a single stable directory under `outputs/`; rerunning the same benchmark overwrites the previous artifacts in that directory.
- the current default output layout is therefore `single_dir`.
- an optional archival layout is already supported as `run_id_subdir`; that layout restores `outputs/<experiment_name>/<run_id>/...` without changing the rest of the runner logic.
- `run_id` still exists in `config.json` and `summary.json` for traceability; timestamps are only the default when no override is provided.
- `epoch_metrics.csv` is 1-based and records one row per completed epoch.
- `epoch=1` means the first parameter update has already finished.
- `train_steps` is recorded in every metrics row for traceability.
- `summary.json` records `phase="Phase 1"` and `math_version="phase0-baseline"` so runs stay tied to the current baseline math.
- `summary.json` also records `primary_metric_higher_is_better` so metric direction is explicit.
- raw inference traces are saved even when plotting is disabled.
- plot generation is optional and requires `matplotlib`.
- the toy benchmark scripts now enable plotting by default, so direct script runs produce `plots/*.png`.
- `config.json` records seed roles explicitly: `seed` remains the run seed, while `seeds.run_seed`, `seeds.data_seed`, and `seeds.model_init_seed` make the current reproducibility semantics explicit.
- the three seed roles may differ even when a benchmark is otherwise unchanged.
- for the current toy benchmarks, `data.data_seed` and `model.model_init_seed` are also written directly into the corresponding sections for local traceability.
- the current linear and sine toy datasets are closed-form deterministic generators, so their `data_seed` is recorded for reproducibility semantics even though those generators do not currently consume randomness.

## Benchmark commands

With the `pc` environment active, the Phase 1 toy benchmarks are:

```powershell
& 'D:\Anaconda\envs\pc\python.exe' experiments/toy_regression.py
& 'D:\Anaconda\envs\pc\python.exe' experiments/toy_sine_regression.py
& 'D:\Anaconda\envs\pc\python.exe' experiments/toy_blobs_classification.py
```

## Current freeze state

The repository is currently frozen at the end of Phase 1.5:

- the baseline predictive-coding math is frozen
- the current toy benchmarks and output schemas are frozen
- repository hygiene now assumes generated outputs, egg-info, caches, and temporary artifacts are not versioned
- the next planned work should start with a narrow Phase 2 comparison against a standard MLP baseline

## A note on predictive coding variants

There are many valid PC formulations. This repository starts with a **single clearly specified baseline formulation** so that implementation and validation stay coherent. New variants should be added only after the baseline is stable and should be explicitly labeled rather than silently replacing the baseline.
