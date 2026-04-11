# RESULTS.md

This file keeps the current frozen results summary only.

- Historical long-form results now live in [archive/RESULTS_HISTORY.md](/e:/CodeSpace/PredictiveCoding/archive/RESULTS_HISTORY.md).
- Use [CURRENT_STATE.md](/e:/CodeSpace/PredictiveCoding/CURRENT_STATE.md), [PLANS.md](/e:/CodeSpace/PredictiveCoding/PLANS.md), and [validation.md](/e:/CodeSpace/PredictiveCoding/validation.md) for active FMPC decisions.

## Current Result Layers

- Phase 2 toy-benchmark freeze summary
- Phase 3 standalone real-data baseline snapshot
- FMPC Stage 04 frozen bridge snapshot
- FMPC Stage 05 exploratory probe snapshot

## Frozen Toy-Benchmark Summary

Strongest current Phase 2 evidence chain:

- Phase 2g matched PC/MLP search
- Phase 2g.1 local boundary check
- Phase 2g.1-refreshed downstream multiseed and budget studies

Current benchmark-level conclusions:

- `toy_regression`:
  - boundary-check-refined PC beats boundary-check-refined MLP on held-out test
  - PC remains ahead across the current multiseed refresh
  - extra PC inference budget is not needed to keep that lead
- `toy_sine_regression`:
  - boundary-check-refined MLP beats boundary-check-refined PC on held-out test
  - MLP remains ahead on most seeds in the current multiseed refresh
  - extra PC inference budget does not reverse that result under the refined base config

Interpretation:

- Phase 2 is closed as a toy-benchmark methodology phase
- Phase 2 conclusions are benchmark-dependent, not universal
- the current evidence does not support the claim that PC is optimizing a completely misaligned internal quantity
- the current searches are still finite and non-exhaustive

## Standalone Real-Data Baseline Snapshot

Current canonical baseline summaries on `sklearn.datasets.load_digits`:

- `digits_mlp`:
  - `best_epoch = 99`
  - `val_accuracy = 0.9111111111111111`
  - `test_accuracy = 0.9481481481481482`
- `digits_pc`:
  - `best_epoch = 55`
  - `val_accuracy = 0.8444444444444444`
  - `test_accuracy = 0.9185185185185185`

Current interpretation:

- this is a standalone baseline snapshot, not a completed real-data PC-vs-MLP comparison
- matched real-data tuning and multiseed aggregation are still not implemented

Relevant artifacts:

- `outputs/digits_mlp/`
- `outputs/digits_pc/`
- `outputs/digits_baselines/`
- optional retained reference: `outputs/digits_pc_stabilization/`

## FMPC Stage 04 Frozen Bridge Snapshot

Current adopted bridge default on `main`:

- `tf2_corrective_transport_terminal_angleclip_default`

Current status:

- FMPC Stage 04 Incremental Bridge is frozen as the bridge result on `main`
- the corrective bridge package is treated as locally saturated under the current selector-gate contract
- package-internal Stage 04 digging should stay closed unless genuinely new evidence appears or the project explicitly leaves the current package or contract

Relevant artifacts:

- `outputs/stage_04_incremental_bridge/`

## FMPC Stage 05 Exploratory Probe Snapshot

Current probe:

- `src/pc/stage_05_ef_core_probe/fmpc_ef_exploratory_probe.py`
- current core contract:
  - corrected residual MeanFlow v1
  - bootstrap residual supervision
  - corrected residual identity curriculum

Current mechanism-level result:

- one-step validation energy delta vs identity:
  - `-0.0001462306001205338`
- configured two-step validation energy delta vs identity:
  - `-0.0001503257951186998`
- configured two-step validation fixed-point residual delta vs identity:
  - `-8.007742858070393e-07`
- validation accuracy:
  - `0.28888888888888886`
- test accuracy:
  - `0.3`

Current interpretation:

- the current corrected residual MeanFlow Stage 05 probe has positive mechanism signal
- task accuracy is still report-only and remains well below the frozen Stage 04 bridge result
- the current next question is whether frozen-bridge vs corrected residual core comparison justifies a Stage 05 v2 charter

Relevant artifacts:

- `outputs/stage_05_ef_core_probe/fmpc_ef_exploratory_probe/`

## Where Detailed History Went

Use these files for longer historical context:

- historical Phase 2 and early real-data narrative:
  - [archive/RESULTS_HISTORY.md](/e:/CodeSpace/PredictiveCoding/archive/RESULTS_HISTORY.md)
- historical plan chain:
  - [archive/PLANS_HISTORY.md](/e:/CodeSpace/PredictiveCoding/archive/PLANS_HISTORY.md)
- historical validation chain:
  - [archive/validation_history.md](/e:/CodeSpace/PredictiveCoding/archive/validation_history.md)
