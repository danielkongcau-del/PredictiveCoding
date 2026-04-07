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

Current status split:

- Stable infrastructure phases: Phase 0, Phase 1, and Phase 1.5
- Experimental findings phases: Phase 2a through Phase 2g.1
- Phase 2 is now considered sufficiently stable to close as a toy-benchmark methodology phase
- Phase 3 is complete as a **standalone real-data baseline phase** on `sklearn.datasets.load_digits`
- the current canonical `digits_pc` baseline reflects a narrow stabilization sweep chosen by `val_metric`
- This does not mean a real-data PC-vs-MLP comparison is already done
- Phase 4 is sealed as the FMPC-v0 preparation checkpoint
- teacher-based FMPC is now frozen as baseline / diagnostic reference
- the current active FMPC line is the teacher-free:
  - `Phase TF2 - iFMPC bridge stage`
- the historical corrective TF2 working reference remains:
  - `tf2_corrective_transport_default`
- the next TF2 experimental default on `main` is:
  - `tf2_corrective_transport_terminal_angleclip_default`
- current TF2 work should be read through `AGENTS.md`, `PLANS.md`, and `validation.md`;
  this README status block may stay lighter than the active bridge-stage contract
- The current Phase 2 freeze summary lives in `RESULTS.md`
- Earlier train-only and train/eval-style Phase 2 conclusions are now treated as methodology-limited historical results
- Current strongest Phase 2 conclusion comes from the best-known Phase 2 evidence chain:
  - Phase 2g matched PC/MLP search
  - Phase 2g.1 local boundary check
  - Phase 2g.1-refreshed downstream multiseed and budget-tradeoff studies
  - both PC and MLP receive matched small-scope tuning
  - configurations are selected by `val_metric`
  - final headline comparisons are reported on held-out `test_metric`
  - a small boundary-extension check was run beyond the original search edges
  - `toy_regression`: boundary-check-refined PC beats boundary-check-refined MLP on held-out test, stays ahead across the current multi-seed check, and remains ahead in the refreshed budget study
  - `toy_sine_regression`: boundary-check-refined MLP beats boundary-check-refined PC on held-out test, stays ahead on most seeds in the current multi-seed check, and remains ahead in the refreshed budget study
  - Phase 2 is stable enough to proceed to Phase 3, but that does not imply exhaustive search saturation or global optimality
- Local output-retention note:
  - the scientific Phase 2 conclusions are preserved in `RESULTS.md` and `validation.md`
  - local output cleanup is allowed; older Phase 2 generated outputs may need to be regenerated if you want to inspect them again
- the current canonical Phase 3 artifact set is:
  - `outputs/digits_mlp/`
  - `outputs/digits_pc/`
  - `outputs/digits_baselines/`
  - optional retained reference: `outputs/digits_pc_stabilization/`

Current scope:

- NumPy-only predictive coding baseline
- structured outputs under `outputs/`
- toy regression, sine regression, and blobs classification benchmarks
- Phase 2 comparison runs against a minimal standard MLP baseline
- narrow Phase 2b PC sensitivity studies on the regression toy benchmarks
- narrow Phase 2c multi-seed aggregate studies on the regression toy benchmarks
- narrow Phase 2d diagnostic studies on the regression toy benchmarks
- narrow Phase 2e tuned-PC budget tradeoff studies on the regression toy benchmarks
- Phase 2f train/val/test-aware protocol hardening and PC joint search
- Phase 2g matched small-scope PC/MLP search plus refreshed downstream multiseed and budget-tradeoff studies
- Phase 2g.1 local boundary-check study for search-space truncation risk plus refined downstream refreshes
- Phase 3 standalone real-data baselines on `digits`

Not included yet:

- CNN/RNN/temporal PC
- MNIST-scale experiments
- real-data matched PC-vs-MLP comparison
- generic hyperparameter tuning frameworks or large search grids

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
- local retention is intentionally lighter than the full historical experiment tree:
  - generated outputs are considered reproducible artifacts rather than permanent local fixtures
  - after cleanup, your local `outputs/` tree may keep only `.gitkeep` plus the current `digits_mlp` baseline artifacts

## Benchmark commands

With the `pc` environment active, the existing PC toy benchmarks are:

```powershell
& 'D:\Anaconda\envs\pc\python.exe' experiments/toy_regression.py
& 'D:\Anaconda\envs\pc\python.exe' experiments/toy_sine_regression.py
& 'D:\Anaconda\envs\pc\python.exe' experiments/toy_blobs_classification.py
```

Phase 2a adds a minimal standard-MLP comparison script:

```powershell
& 'D:\Anaconda\envs\pc\python.exe' experiments/compare_pc_vs_mlp.py
```

Phase 2b adds a narrow predictive-coding sensitivity script:

```powershell
& 'D:\Anaconda\envs\pc\python.exe' experiments/pc_sensitivity.py
& 'D:\Anaconda\envs\pc\python.exe' experiments/pc_sensitivity.py toy_regression
```

Phase 2c adds a narrow multi-seed aggregate script:

```powershell
& 'D:\Anaconda\envs\pc\python.exe' experiments/pc_multiseed.py
& 'D:\Anaconda\envs\pc\python.exe' experiments/pc_multiseed.py toy_regression
```

Phase 2d adds a narrow diagnostic script for understanding the tuned-PC vs MLP gap:

```powershell
& 'D:\Anaconda\envs\pc\python.exe' experiments/pc_diagnostics.py
& 'D:\Anaconda\envs\pc\python.exe' experiments/pc_diagnostics.py toy_regression
```

Phase 3 currently adds two **standalone digits baselines** with aligned protocol semantics:

```powershell
& 'D:\Anaconda\envs\pc\python.exe' experiments/digits_mlp.py
& 'D:\Anaconda\envs\pc\python.exe' experiments/digits_pc.py
& 'D:\Anaconda\envs\pc\python.exe' experiments/summarize_digits_baselines.py
```

Phase 4 preparation also adds one harder standalone real-data PC benchmark option:

```powershell
& 'D:\Anaconda\envs\pc\python.exe' experiments/fashion_mnist_pc.py
& 'D:\Anaconda\envs\pc\python.exe' experiments/digits_pc_inference_baselines.py
& 'D:\Anaconda\envs\pc\python.exe' experiments/fmpc_v0_prepare.py
```

Notes:

- this uses `sklearn.datasets.fetch_openml("Fashion-MNIST")` through the repository dataset loader
- it is a standalone benchmark option for infrastructure preparation, not a formal comparison pipeline
- `experiments/digits_pc_inference_baselines.py` is a standalone inference-baseline hardening study:
  - it compares explicit `euler` and `rk2` predictive-coding inference methods under small fixed step budgets
  - it does not change the predictive-coding learning rule
  - it is not a formal real-data PC-vs-MLP comparison
- `experiments/fmpc_v0_prepare.py` is a teacher-only FMPC-v0 preparation scaffold:
  - it trains a standard real-data PC teacher under the current baseline path
  - it exports `z0` / `z_star` teacher targets plus optional trajectories
  - it records future student transport/refinement settings only as placeholders
  - it does not implement a transporter and does not write any FMPC result
- standalone `digits_pc` summaries keep `teacher_reference` explicit but disable it by default:
  - predict-mode candidate-vs-teacher gaps are often trivial under forward initialization
  - meaningful FMPC teacher targets should instead come from `experiments/fmpc_v0_prepare.py`

These write:

```text
outputs/
  digits_mlp/
    config.json
    epoch_metrics.csv
    summary.json
    plots/                 # only when plot_curves=True and matplotlib is available
  digits_pc/
    config.json
    epoch_metrics.csv
    summary.json
    plots/                 # only when plot_curves=True and matplotlib is available
  digits_baselines/
    summary.json          # first-pass side-by-side summary of existing digits_mlp and digits_pc runs
```

The local tree may also retain:

- `outputs/digits_pc_stabilization/`
  This is a narrow baseline-hardening artifact from the standalone PC stabilization sweep, not a formal comparison artifact.

By default, rerunning either script writes back into the same stable directory:

- `outputs/digits_mlp/`
- `outputs/digits_pc/`
- `outputs/digits_baselines/`
- that means a new default run overwrites the previous `config.json`, `epoch_metrics.csv`, `summary.json`, and optional plots in the corresponding directory
- use a custom `run_id` or a copied output directory yourself if you want to preserve multiple local snapshots

Meaning of the current Phase 3 output:

- it is a small real-data baseline setup on `sklearn.datasets.load_digits`
- it uses explicit `train / val / test` splits
- it uses deterministic mini-batch ordering with explicit seed roles
- it selects the reported checkpoint by `val_metric`
- it reports the headline result on held-out `test_metric`
- the `digits_mlp` and `digits_pc` runs now share those protocol rules
- the canonical `digits_pc` baseline now reflects the best single-run candidate from a small stabilization sweep selected by `val_metric`
- `experiments/summarize_digits_baselines.py` can produce a small side-by-side digest from the two existing `summary.json` files
- the current Phase 3 checkpoint should therefore be read from:
  - `outputs/digits_mlp/`
  - `outputs/digits_pc/`
  - `outputs/digits_baselines/`

What it does not mean:

- it is not a real-data PC-vs-MLP comparison
- it is not an MNIST result
- it does not mean matched tuning or real-data comparison artifacts are already implemented
- the side-by-side summary is only a first-pass human-facing digest, not a formal comparison pipeline
- the current Phase 3 evidence chain is still a pair of standalone baselines plus a digest, not a real-data matched-comparison workflow

What Phase 3 has completed:

- a deterministic real-data `digits` entry point
- deterministic mini-batch ordering with explicit seed roles
- a standalone canonical `digits_mlp` baseline
- a standalone canonical `digits_pc` baseline
- protocol-alignment checks between those two baselines
- a first-pass side-by-side digest for human inspection

What Phase 3 has not completed:

- formal real-data comparison artifacts
- matched tuning
- multi-seed real-data aggregation
- a second real dataset
- MNIST

Phase 2e adds a narrow tuned-PC budget tradeoff script:

```powershell
& 'D:\Anaconda\envs\pc\python.exe' experiments/pc_budget_tradeoff.py
& 'D:\Anaconda\envs\pc\python.exe' experiments/pc_budget_tradeoff.py toy_regression
```

Phase 2f adds a train/val/test-aware joint PC search:

```powershell
& 'D:\Anaconda\envs\pc\python.exe' experiments/pc_joint_search.py
& 'D:\Anaconda\envs\pc\python.exe' experiments/pc_joint_search.py toy_regression
```

Phase 2g adds a matched PC/MLP search under the same validation/test protocol:

```powershell
& 'D:\Anaconda\envs\pc\python.exe' experiments/phase2g_matched_search.py
& 'D:\Anaconda\envs\pc\python.exe' experiments/phase2g_matched_search.py toy_regression
```

Phase 2g.1 adds a small local boundary-check study around the current Phase 2g best configs:

```powershell
& 'D:\Anaconda\envs\pc\python.exe' experiments/phase2g1_boundary_check.py
& 'D:\Anaconda\envs\pc\python.exe' experiments/phase2g1_boundary_check.py toy_regression
```

The first Phase 2b pass stays intentionally small:

- only `toy_regression` and `toy_sine_regression`
- only one-at-a-time PC trials around the current default
- only the current PC knobs:
  - `eta_x`
  - `eta_w` with `eta_b = eta_w`
  - `train_steps` with `eval_steps = train_steps`
  - `state_init`

The first MLP classification baseline stays intentionally narrow:

- it uses the same toy one-hot targets as the current PC benchmark
- it trains with mean squared error averaged over all output elements
- it is still evaluated with argmax accuracy

For Phase 2 comparison summaries, `primary_metric_difference_mlp_minus_pc` always means:

- `mlp_primary_metric_value - pc_primary_metric_value`
- for `accuracy`, a positive value means the MLP is better
- for `mse`, a negative value means the MLP is better because lower is better

Phase 2b sensitivity outputs follow the existing repository style:

```text
outputs/
  pc_sensitivity_<benchmark>/
    base_pc_config.json
    candidate_grid.json
    trial_table.csv
    aggregate_summary.json
    mlp_reference/
      config.json
      epoch_metrics.csv
      summary.json
    trials/
      default/
      eta_x_half/
      ...
    plots/                 # only when summary plotting is enabled
```

Important note on `state_init` in Phase 2b:

- `state_init` is not treated as a tiny numeric tweak
- it affects the full predictive-coding run path
- that includes hidden-state initialization during training inference and during prediction/eval inference

For `trial_table.csv`, the numeric delta columns are raw subtractions from the default PC trial:

- `primary_metric_delta_vs_default = trial_primary_metric_value - default_primary_metric_value`
- `final_pre_update_energy_delta_vs_default = trial_final_pre_update_energy - default_final_pre_update_energy`
- for `mse`, a negative `primary_metric_delta_vs_default` means the trial improved on the default

Phase 2c keeps the seed policy small and explicit:

- default benchmarks:
  - `toy_regression`: `[0, 1, 2, 3, 4]`
  - `toy_sine_regression`: `[3, 4, 5, 6, 7]`
- `data_seed` stays fixed at the benchmark default
- `run_seed` and `model_init_seed` vary together
- this phase therefore mainly measures initialization stability, not dataset sampling variability

Phase 2c outputs follow the existing repository style:

```text
outputs/
  pc_multiseed_<benchmark>/
    study_config.json
    seed_records.csv
    aggregate_summary.json
    seeds/
      seed_0000/
        default_pc/
        tuned_pc/
        mlp/
    plots/                 # only when summary plotting is enabled
```

For `seed_records.csv`, the pairwise comparison fields mean:

- `primary_metric_delta_tuned_pc_minus_default_pc = tuned_pc_metric - default_pc_metric`
- `primary_metric_delta_mlp_minus_tuned_pc = mlp_metric - tuned_pc_metric`
- `tuned_pc_beats_default_pc` and `tuned_pc_beats_mlp` respect the current metric direction
- for `mse`, `True` means the tuned PC metric is lower

Phase 2d diagnostics also keep the repository style small and explicit:

```text
outputs/
  pc_diagnostics_<benchmark>/
    study_config.json
    seed_records.csv
    epoch_records.csv
    epoch_summary.csv
    diagnostic_summary.json
    seeds/
      seed_0000/
        default_pc/
        tuned_pc/
        tuned_pc_budget2x/
        mlp/
    plots/                 # only when summary plotting is enabled
```

Important Phase 2d notes:

- `default_pc` and `tuned_pc` are the main PC variants
- `tuned_pc_budget2x` is only a budget diagnostic branch, not a new main comparison model
- `mlp` is the MLP reference
- Pearson correlations in `diagnostic_summary.json` are computed on the aggregated epoch-level mean curves for each benchmark and PC variant, not on pooled `seed x epoch` rows
- `final_minus_best_metric = final_metric - best_metric`
- for the current regression tasks with `mse`, `final_minus_best_metric` should be non-negative
- values closer to zero mean less late-epoch degradation from the best epoch
- the default Phase 2d seed sets still keep `data_seed` fixed while `run_seed` and `model_init_seed` vary together, so this phase mainly measures initialization stability rather than dataset sampling variability

Phase 2e keeps the repository style narrow and explicit:

```text
outputs/
  pc_budget_tradeoff_<benchmark>/
    study_config.json
    seed_budget_records.csv
    budget_summary.csv
    aggregate_summary.json
    seeds/
      seed_0000/
        tuned_pc_1x/
        tuned_pc_2x/
        tuned_pc_4x/
        mlp/
    plots/                 # only when summary plotting is enabled
```

Important Phase 2e notes:

- Phase 2e is not a wall-clock- or FLOP-matched efficiency comparison
- it is a tuned-PC inference-budget vs performance study with MLP as a fixed reference
- `tuned_pc_1x`, `tuned_pc_2x`, and `tuned_pc_4x` are the only tuned-PC budget variants
- `mlp` is a fixed reference line and is not part of the tuned-PC budget axis
- `budget = inference step count`, not runtime or hardware efficiency
- the default Phase 2e seed sets keep `data_seed` fixed while `run_seed` and `model_init_seed` vary together, so this phase still mainly measures initialization stability rather than dataset sampling variability

Important Phase 2f and Phase 2g notes:

- the hardened protocol separates:
  - `train_metric` for training-set behavior
  - `val_metric` for configuration selection
  - `test_metric` for final held-out reporting
- Phase 2f improves the protocol but still leaves MLP mostly as a fixed reference
- Phase 2g is the fairer downstream comparison because both PC and MLP receive matched small-scope tuning
- Phase 2g aggregate artifacts explicitly record:
  - how configs were selected
  - which split determined selection
  - which split determined the final headline result
  - the selected PC config
  - the selected MLP config
  - the held-out test comparison between them
- Phase 2g.1 keeps the same validation/test discipline and only extends the search locally around Phase 2g boundary-hit best configs
- Phase 2g.1 is a closure check on search-space truncation risk, not a new large search stage
- Phase 2g.1 did not reverse the benchmark-level winners from Phase 2g:
  - `toy_regression` remained PC > MLP
  - `toy_sine_regression` remained MLP > PC
- Phase 2g.1 also refreshed the downstream multiseed and budget-tradeoff studies to consume the refined best-known configs
- The resulting best-known Phase 2 evidence chain is:
  - select configs on validation
  - report headlines on held-out test
  - verify benchmark-level winner stability under a small local boundary extension

## Frozen reference point

The repository's frozen pre-Phase-2 baseline remains the `phase1_5-stable` tag:

- the baseline predictive-coding math is frozen
- the current toy benchmarks and output schemas are frozen
- repository hygiene now assumes generated outputs, egg-info, caches, and temporary artifacts are not versioned
- the current Phase 2 branch builds a minimal MLP comparison path on top of that frozen baseline

## A note on predictive coding variants

There are many valid PC formulations. This repository starts with a **single clearly specified baseline formulation** so that implementation and validation stay coherent. New variants should be added only after the baseline is stable and should be explicitly labeled rather than silently replacing the baseline.
