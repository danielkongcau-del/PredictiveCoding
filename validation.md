# Validation and Acceptance Criteria

This file now keeps the active validation contract only.

- Historical validation detail has moved to [archive/validation_history.md](/e:/CodeSpace/PredictiveCoding/archive/validation_history.md).
- The goal of this file is to preserve current acceptance rules without forcing every new session to read the full historical evidence chain.

## General Validation Philosophy

A valid change should still demonstrate:

1. shape correctness
2. numerical sanity
3. behavioral sanity
4. reproducibility
5. traceability to the math layer rooted at `spec_math.md`

## Stage 04 Validation Status

The current Stage 04 corrective bridge package is frozen as the bridge result on `main`.

Current control:

- `tf2_corrective_transport_terminal_angleclip_default`

Current validation interpretation:

- the adopted corrective bridge package is locally saturated under the current selector-gate contract
- no different package-internal issue family is currently recommended for pursuit
- package-internal Stage 04 digging should stop from this state unless:
  - a genuinely different issue family appears from new evidence, or
  - the project explicitly decides to leave the current package or selector-gate contract

### Stage 04 Sealed Families

The following families are sealed from the current state:

- terminal / unified-cone follow-ups
- successor-value and successor-increment package-internal follow-ups
- readout alignment follow-ups
- bootstrap-source bias follow-ups
- target-lag follow-ups
- bootstrap-to-identity curriculum follow-ups

### Why Stage 04 Is Closed

Condensed evidence summary:

- the current hard full-vector `30` degree terminal angle clip remains the local winner
- selector / checkpoint is not the main limiter
- simple head-fit / separability is not the main limiter
- readout alignment is an exact no-op on the adopted package
- older bootstrap-target-side follow-ups do not materially improve the adopted package
- the late-rollout successor-value and successor-increment line ends in a strengthened formulation-level blocker rather than an adoption-viable local fix

## Stage 05 Exploratory Validation Contract

Stage 05 is mechanism-first.

Current Stage 05 core contract:

- corrected residual MeanFlow transport family
- bootstrap residual supervision
- corrected residual identity curriculum
- no teacher dependency in target construction

Current Stage 05 implementation status:

- the v1 single-branch corrected residual core remains the backward-compatible baseline
- the v2 two-branch corrected residual core now also exists
- the v2 branch structure is:
  - `m_psi = m_traj + m_state`
  - `m_traj_input = concat([z_t, target_onehot, t, r])`
  - `m_state_input = concat([g_t, e_out_t, F_t])`

Primary acceptance metrics:

- one-step energy decrease relative to identity or no-transport
- few-step fixed-point residual decrease relative to identity or no-transport
- deterministic artifact generation
- artifact-independent target construction

Secondary report-only metrics:

- validation accuracy
- test accuracy

What this means operationally:

- task accuracy is not the gate for the current exploratory stage
- a weak task metric does not invalidate a probe if mechanism metrics are positive
- a strong task metric alone does not justify promotion without a clear mechanism advantage

## Stage 05 Required Artifacts

Current canonical exploratory runs should produce:

- `config.json`
- `epoch_metrics.csv`
- `summary.json`

Summary fields should continue to make these items explicit:

- stage identity
- transport family
- residual identity mode
- whether target construction is artifact-independent
- local flow definition
- direct anchor source
- transport scope
- transport steps
- bootstrap target contract
- selection-metric source
- report-metric source
- mechanism metrics such as energy and residual deltas

## Current Stage 05 Known Result

The first canonical Stage 05 exploratory probe now shows:

- positive one-step validation energy delta vs identity
- positive configured two-step validation energy delta vs identity
- positive configured two-step fixed-point residual delta vs identity
- deterministic artifact generation
- artifact-independent target construction

At the same time:

- validation and test accuracy remain low
- the probe does not challenge the frozen Stage 04 bridge result yet

Current interpretation:

- the exploratory line has now cleared the narrow v1 to v2 mechanism-improvement check
- the two-branch Stage 05 v2 core is now the current exploratory candidate
- the refreshed frozen-bridge comparison also supports treating Stage 05 v2 as the new exploratory reference
- the dedicated Stage 05 v2 diagnostics now also say:
  - all comparison seeds select the final training epoch
  - configured-step mechanism metrics and validation accuracy still improve at the training boundary
  - the current low report-only accuracy is not primarily a selection-rule artifact
  - the current narrow diagnosis is `likely_undertrained`
- the dedicated Stage 05 v2 longer-training validation now also says:
  - a stronger same-family `24`-epoch budget materially improves configured-step mechanism magnitude
  - the same stronger budget materially improves report-only validation and test accuracy
  - the stronger budget still selects the final training epoch on every seed
- the next Stage 05 v2 budget-push validation now also says:
  - a stronger same-family `768`-epoch budget materially improves configured-step mechanism magnitude over the `384`-epoch reference
  - the same stronger budget materially improves report-only validation and test accuracy
  - the stronger budget still selects the final training epoch on every seed
  - the explicit stop-rule layer still says:
    - `budget_line_still_looks_boundary_limited = true`
    - `budget_line_should_continue = true`
    - `budget_line_should_stop_and_open_v3 = false`
  - the contextual accuracy note now places the stronger Stage 05 v2 budget:
    - above the frozen Stage 04 bridge accuracy level
    - mixed relative to the standalone `digits_pc` baseline
    - below the standalone `digits_mlp` baseline
- it still does not justify replacing the frozen bridge result on `main`

## Current Recommended Validation Move

The next validation move is:

- keep the current Stage 05 v2 transport family fixed and continue pushing budget on the same v2 family before inventing a new Stage 05 family
- the latest same-family budget reference is now the `384 -> 768 epochs` validation under:
  - `outputs/stage_05_ef_core_probe/stage05_v2_budget_push_validation_384_to_768/`

That next step should preserve:

- the frozen Stage 04 bridge result on `main`
- the mechanism-first validation contract for Stage 05

It should not be framed as:

- a Stage 04 package-internal reopening
- a new adoption sweep inside the saturated corrective bridge package

## Relevant Active Artifacts

- frozen Stage 04 authority artifact tree:
  - `outputs/stage_04_incremental_bridge/`
- Stage 05 exploratory probe:
  - `outputs/stage_05_ef_core_probe/fmpc_ef_exploratory_probe/`
- Stage 05 frozen-bridge comparison:
  - `outputs/stage_05_ef_core_probe/frozen_bridge_vs_corrected_core_comparison/`
- Stage 05 v1 vs v2 comparison:
  - `outputs/stage_05_ef_core_probe/corrected_residual_core_v1_vs_v2_comparison/`
- Stage 05 refreshed bridge vs v2 comparison:
  - `outputs/stage_05_ef_core_probe/frozen_bridge_vs_two_branch_corrected_core_comparison/`
- Stage 05 v2 diagnostics:
  - `outputs/stage_05_ef_core_probe/stage05_v2_diagnostics/`
- Stage 05 v2 longer-training validation:
  - `outputs/stage_05_ef_core_probe/stage05_v2_longer_training_validation/`
- Stage 05 v2 budget-push validation:
  - `outputs/stage_05_ef_core_probe/stage05_v2_budget_push_validation_384_to_768/`

## Document Layering

Use the current docs this way:

- low-context summary:
  - [LLM_BRIEF.md](/e:/CodeSpace/PredictiveCoding/LLM_BRIEF.md)
- math layer:
  - [spec_math.md](/e:/CodeSpace/PredictiveCoding/spec_math.md)
  - applicable stage addenda under [specs/](/e:/CodeSpace/PredictiveCoding/specs)
- operational state:
  - [CURRENT_STATE.md](/e:/CodeSpace/PredictiveCoding/CURRENT_STATE.md)
- active plan:
  - [PLANS.md](/e:/CodeSpace/PredictiveCoding/PLANS.md)
- historical validation detail:
  - [archive/validation_history.md](/e:/CodeSpace/PredictiveCoding/archive/validation_history.md)
