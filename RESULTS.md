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
- current implemented contracts:
  - corrected residual MeanFlow v1
  - corrected residual MeanFlow v2 two-branch residual core
  - bootstrap residual supervision
  - corrected residual identity curriculum

Current mechanism-level result:

- current Stage 05 v2 multiseed snapshot:
  - mean one-step validation energy delta vs identity:
    - `-0.00017276987415493217`
  - mean configured two-step validation energy delta vs identity:
    - `-0.0001763621381912032`
  - mean configured two-step validation fixed-point residual delta vs identity:
    - `-9.455861891589161e-07`
  - mean validation accuracy:
    - `0.27037037037037037`
  - mean test accuracy:
    - `0.27283950617283953`

Current interpretation:

- the current Stage 05 v2 two-branch corrected residual MeanFlow probe has positive mechanism signal
- it improves mechanism magnitude over the Stage 05 v1 baseline under the current multiseed rule
- task accuracy is still report-only and remains well below the frozen Stage 04 bridge result

Relevant artifacts:

- `outputs/stage_05_ef_core_probe/fmpc_ef_exploratory_probe/`
- `outputs/stage_05_ef_core_probe/corrected_residual_core_v1_vs_v2_comparison/`

## FMPC Stage 05 Frozen-Bridge Comparison Snapshot

Current formal comparison:

- `outputs/stage_05_ef_core_probe/frozen_bridge_vs_two_branch_corrected_core_comparison/`

Current comparison result:

- Stage 04 frozen bridge:
  - mean one-step validation energy delta vs identity:
    - `0.0003669573859489221`
  - mean configured-step validation energy delta vs identity:
    - `-0.004070538730777469`
  - mean configured-step validation fixed-point residual delta vs identity:
    - `-1.8072299104931235e-05`
  - mean validation accuracy:
    - `0.811111111111111`
  - mean test accuracy:
    - `0.8197530864197531`
- Stage 05 two-branch corrected residual core v2:
  - mean one-step validation energy delta vs identity:
    - `-0.00017276987415493217`
  - mean configured-step validation energy delta vs identity:
    - `-0.0001763621381912032`
  - mean configured-step validation fixed-point residual delta vs identity:
    - `-9.455861891589161e-07`
  - mean validation accuracy:
    - `0.27037037037037037`
  - mean test accuracy:
    - `0.27283950617283953`

Current interpretation:

- Stage 05 v2 clears the refreshed multiseed mechanism-first exploration rule
- Stage 05 v2 is stronger than the frozen bridge on one-step mechanism
- Stage 05 v2 remains weaker than the frozen bridge on configured-step mechanism magnitude
- Stage 05 v2 remains far below the frozen bridge on report-only accuracy
- Stage 05 v2 does not replace the frozen bridge result on `main`
- the refreshed comparison supports using Stage 05 v2 as the new exploratory reference

Relevant artifacts:

- `outputs/stage_05_ef_core_probe/frozen_bridge_vs_two_branch_corrected_core_comparison/`

## FMPC Stage 05 V1 vs V2 Comparison Snapshot

Current formal comparison:

- `outputs/stage_05_ef_core_probe/corrected_residual_core_v1_vs_v2_comparison/`

Current comparison result:

- Stage 05 corrected residual core v1:
  - mean one-step validation energy delta vs identity:
    - `-0.00015049783324024477`
  - mean configured-step validation energy delta vs identity:
    - `-0.00015441938077428072`
  - mean configured-step validation fixed-point residual delta vs identity:
    - `-8.351992466021287e-07`
  - mean validation accuracy:
    - `0.2740740740740741`
  - mean test accuracy:
    - `0.2716049382716049`
- Stage 05 corrected residual core v2:
  - mean one-step validation energy delta vs identity:
    - `-0.00017276987415493217`
  - mean configured-step validation energy delta vs identity:
    - `-0.0001763621381912032`
  - mean configured-step validation fixed-point residual delta vs identity:
    - `-9.455861891589161e-07`
  - mean validation accuracy:
    - `0.27037037037037037`
  - mean test accuracy:
    - `0.27283950617283953`

Current interpretation:

- Stage 05 v2 improves mean configured-step mechanism magnitude over v1
- Stage 05 v2 keeps all required multiseed mechanism-first checks negative vs identity
- task accuracy remains report-only in this comparison
- this comparison does not claim that Stage 05 replaces the frozen Stage 04 bridge result on `main`

Relevant artifacts:

- `outputs/stage_05_ef_core_probe/corrected_residual_core_v1_vs_v2_comparison/`

## Where Detailed History Went

Use these files for longer historical context:

- historical Phase 2 and early real-data narrative:
  - [archive/RESULTS_HISTORY.md](/e:/CodeSpace/PredictiveCoding/archive/RESULTS_HISTORY.md)
- historical plan chain:
  - [archive/PLANS_HISTORY.md](/e:/CodeSpace/PredictiveCoding/archive/PLANS_HISTORY.md)
- historical validation chain:
  - [archive/validation_history.md](/e:/CodeSpace/PredictiveCoding/archive/validation_history.md)
