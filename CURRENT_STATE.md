# CURRENT_STATE.md

This file is the short active-state summary for the repository.

- Use it for the current operational state.
- Use [PLANS.md](/e:/CodeSpace/PredictiveCoding/PLANS.md) as the only canonical long-horizon plan file.
- Do not create or treat `PLANS2.md` as an authoritative planning document.

## Active Branch And Line

- Active branch:
  - `main`
- Active algorithmic line:
  - `Phase TF2 - iFMPC bridge stage`

## Current Adopted Defaults

- Current adopted TF2 experimental default on `main`:
  - `tf2_corrective_transport_terminal_angleclip_default`
- Current canonical TF2 identity default:
  - `feature_aware_tangents = false`

## Historical Reference Presets

- Historical corrective working reference:
  - `tf2_corrective_transport_default`
- Hypothesis-driven subordinate preset:
  - `tf2_canonical`

## Operationally Relevant Sealed Conclusions

- Stay inside the corrective TF2 package on `main`; do not open AlphaFlow, `µPC`-style TF2 mainline scaling, or TF3 from the current state.
- The adopted angle-clip package materially improves over the historical corrective reference and remains the current TF2 experimental default on `main`.
- The remaining slow-PC gap is not mainly a selector/checkpoint issue and not mainly a simple head-fit problem.
- The dominant remaining mismatch inside the adopted package is best read as readout-relevant endpoint-basis distortion rather than simple separability collapse.
- Row-space-only, orthogonal-only, and split-threshold terminal interventions are all non-adopted; keep the current full-vector terminal angle clip unchanged.
- The unified-cone vs split-subspace geometry pass says the current gain is best explained by a shared full-space angular constraint, not by literal row/orth ratio preservation.
- The unified-cone-shape pass says the `20` degree interior-margin cone improves accuracy and row-space distortion, but not enough gate robustness to replace the current adopted `30` degree hard cone.

## Current Narrow Open Question

- Current next narrow move:
  - if TF2 work continues inside the adopted package, test at most one smooth unified-cone projection variant that tries to keep the interior-margin gain without paying the current gate-robustness cost
- Current active question:
  - whether a single smooth unified-cone projection can preserve the `20` degree interior-margin gain while recovering the current adopted control's gate robustness

## Relevant Suites And Artifacts

- Core preset and terminal-intervention code:
  - [src/pc/fmpc_tf2.py](/e:/CodeSpace/PredictiveCoding/src/pc/fmpc_tf2.py)
- Current source-of-truth validation log:
  - [validation.md](/e:/CodeSpace/PredictiveCoding/validation.md)
- Current long-horizon plan:
  - [PLANS.md](/e:/CodeSpace/PredictiveCoding/PLANS.md)
- Recent authority artifact sets:
  - [outputs/fmpc_tf2_external_comparison_suite](/e:/CodeSpace/PredictiveCoding/outputs/fmpc_tf2_external_comparison_suite)
  - [outputs/fmpc_tf2_gap_decomposition_suite](/e:/CodeSpace/PredictiveCoding/outputs/fmpc_tf2_gap_decomposition_suite)
  - [outputs/fmpc_tf2_readout_refit_suite](/e:/CodeSpace/PredictiveCoding/outputs/fmpc_tf2_readout_refit_suite)
  - [outputs/fmpc_tf2_endpoint_basis_suite](/e:/CodeSpace/PredictiveCoding/outputs/fmpc_tf2_endpoint_basis_suite)
  - [outputs/fmpc_tf2_output_sensitive_terminal_direction_suite](/e:/CodeSpace/PredictiveCoding/outputs/fmpc_tf2_output_sensitive_terminal_direction_suite)
  - [outputs/fmpc_tf2_terminal_coupling_suite](/e:/CodeSpace/PredictiveCoding/outputs/fmpc_tf2_terminal_coupling_suite)
  - [outputs/fmpc_tf2_split_threshold_coupling_suite](/e:/CodeSpace/PredictiveCoding/outputs/fmpc_tf2_split_threshold_coupling_suite)
  - [outputs/fmpc_tf2_unified_cone_geometry_suite](/e:/CodeSpace/PredictiveCoding/outputs/fmpc_tf2_unified_cone_geometry_suite)
  - [outputs/fmpc_tf2_unified_cone_shape_suite](/e:/CodeSpace/PredictiveCoding/outputs/fmpc_tf2_unified_cone_shape_suite)

## Read Order And Precedence

- Read in this order:
  1. `README.md`
  2. `AGENTS.md`
  3. `CURRENT_STATE.md`
  4. `PLANS.md`
  5. `spec_math.md`
  6. `validation.md`
- If there is a conflict, precedence is:
  - `spec_math.md > validation.md > AGENTS.md > CURRENT_STATE.md > PLANS.md > README.md`

## Commit Note

- This file intentionally tracks active state rather than a moving commit tip.
- Use `git rev-parse HEAD` when you need the exact current commit.
