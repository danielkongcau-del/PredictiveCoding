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

- Stay inside the corrective TF2 package on `main`; do not open AlphaFlow, `muPC`-style TF2 mainline scaling, or TF3 from the current state.
- The adopted angle-clip package materially improves over the historical corrective reference and remains the current TF2 experimental default on `main`.
- The remaining slow-PC gap is not mainly a selector/checkpoint issue and not mainly a simple head-fit problem.
- The dominant remaining mismatch inside the adopted package is best read as readout-relevant endpoint-basis distortion rather than simple separability collapse.
- Row-space-only, orthogonal-only, and split-threshold terminal interventions are all non-adopted; keep the current full-vector terminal angle clip unchanged.
- The unified-cone vs split-subspace geometry pass says the current gain is best explained by a shared full-space angular constraint, not by literal row/orth ratio preservation.
- The unified-cone-shape pass says the `20` degree interior-margin cone improves accuracy and row-space distortion, but not enough gate robustness to replace the current adopted `30` degree hard cone.
- The smooth unified-cone projection pass also remains non-adopted:
  - it improves accuracy and row-space distortion relative to the current hard `30` degree control
  - it partially recovers the old `20` degree interior-margin robustness loss
  - but it still does not recover enough gate robustness to replace the current adopted hard `30` degree cone
- The unified-cone robustness-tradeoff pass now closes this family at confirmation level:
  - both the hard `20` degree interior-margin reference and the smooth unified-cone reference are blocked mainly by systematic energy-side gate-margin collapse
  - neither variant shows a volatility-led or rare-bad-epoch explanation strong enough to justify another family-internal confirmation
  - the current hard `30` degree full-vector cone remains the local winner under the fixed selector/gate contract
- The late-rollout basis-drift localization pass now says the remaining readout-relevant basis distortion is injected mainly by preterminal rollout accumulation rather than by the final terminal jump:
  - validation mean row-space gap share already present by the preterminal knot is about `0.7447`
  - test mean row-space gap share already present by the preterminal knot is about `0.7451`
  - the terminal jump contributes only about `0.255`
- The late-rollout drift-control pass says this preterminal drift is not adoption-recoverable by simply moving the same adopted full-vector `30` degree cone earlier:
  - penultimate-plus-terminal and last-two-preterminal-plus-terminal variants both reduce terminal row-space RMS and improve accuracy
  - but both collapse gate coverage to `0.0`, force selector fallback to `1.0`, and therefore remain non-adopted
  - the remaining blocker now points to the preterminal update formulation itself rather than another cone-family follow-up
- The preterminal-update source-localization pass now says the earlier-control gate collapse is primarily a preterminal on-policy handoff-state problem:
  - swapping only the preterminal on-policy handoff back to the cached batch-start successor restores gate coverage to the adopted-control level
  - swapping only the preterminal direction source to on-policy live local field does not recover any gate coverage
  - swapping only the preterminal norm handling to anchor norm does not recover any gate coverage
  - the current active diagnosis is:
    - `preterminal_handoff_state_is_primary_blocker`
- The smallest preterminal on-policy handoff reformulation confirmation now says the cached-handoff reformulation is not adoption-viable:
  - it restores gate coverage to the adopted-control level
  - but it recovers only about `3.4%` of the earlier-control validation-accuracy gain and essentially none of the earlier row-space RMS gain
  - the current confirmation-level diagnosis is:
    - `handoff_reformulation_recovers_partially_but_not_adoption_level`
- The successor-handoff source-localization pass now says the blocker is the learned/on-policy successor value itself rather than the local-field successor bundle:
  - swapping only preterminal `z_on_next` back to the cached batch-start successor fully restores the gate contract but collapses back to near-control accuracy and row-space metrics
  - swapping only preterminal `z_lf_next` back to the cached batch-start successor leaves the failed earlier-control reference unchanged
  - swapping both successor components back to cached matches the `z_on_next`-only result, so the remaining blocker is not a cross-source successor inconsistency
  - the current active diagnosis is:
    - `stale_successor_value_is_primary_blocker`
- The preterminal successor-value reformulation confirmation now says this component is still promising but not yet adoption-viable:
  - the best narrow candidate is the fixed `25%` live / `75%` cached successor-value blend
  - it preserves the full selector/gate contract and improves validation accuracy plus terminal row-space metrics over the current adopted control
  - but it retains only about `31%` of the failed earlier-control validation-accuracy gain and about `24.5%` of the terminal row-space RMS gain
  - the `50%` live / `50%` cached blend recovers more geometry but gives up too much gate robustness
  - the current confirmation-level diagnosis is:
    - `successor_value_reformulation_recovers_partially_but_not_adoption_level`
- The tiny low-live successor-value follow-up now says the immediate `25/75` neighborhood also does not reach adoption level:
  - the safer `20%` live / `80%` cached blend keeps the full selector/gate contract but is weaker than the current `25%` live / `75%` cached anchor on both gain retention and terminal row-space recovery
  - the more aggressive `30%` live / `70%` cached blend retains more of the earlier-control gain than the `25/75` anchor, but it gives up full gate robustness:
    - `seed_gate_positive_rate: 0.8`
    - `selected_epoch_passes_gate_rate: 0.8`
    - `selector_fallback_used_rate: 0.2`
  - the current local follow-up diagnosis is:
    - `local_successor_value_refinement_improves_but_not_to_adoption_level`
- The successor-value carry-vs-increment source-localization pass now says the remaining blocker sits in the live successor increment rather than in the predecessor carry state:
  - `live carry + cached increment` restores the full selector/gate contract:
    - `seed_gate_positive_rate: 1.0`
    - `selected_epoch_passes_gate_rate: 1.0`
    - `selector_fallback_used_rate: 0.0`
    - but it collapses back to near-control accuracy and terminal row-space metrics
  - `cached carry + live increment` stays almost identical to the failed higher-gain reference:
    - `mean_val_accuracy: 0.8563`
    - `mean_gate_passing_epoch_count: 0.0`
    - `selector_fallback_used_rate: 1.0`
  - the current source-localization diagnosis is:
    - `live_successor_increment_is_primary_blocker`

## Current Narrow Open Question

- Current next narrow move:
  - unified-cone work should continue to be treated as locally saturated under the current selector/gate contract
  - if TF2 work continues inside the adopted package, move to a confirmation-level reformulation on the preterminal successor increment only rather than another low-live blend sweep or terminal-cone follow-up
- Current active question:
  - can the live preterminal successor increment be minimally reformulated so that it keeps more of the earlier-control gain without reopening the gate collapse

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
  - [outputs/fmpc_tf2_smooth_unified_cone_suite](/e:/CodeSpace/PredictiveCoding/outputs/fmpc_tf2_smooth_unified_cone_suite)
  - [outputs/fmpc_tf2_unified_cone_robustness_suite](/e:/CodeSpace/PredictiveCoding/outputs/fmpc_tf2_unified_cone_robustness_suite)
  - [outputs/fmpc_tf2_basis_drift_localization_suite](/e:/CodeSpace/PredictiveCoding/outputs/fmpc_tf2_basis_drift_localization_suite)
  - [outputs/fmpc_tf2_late_rollout_drift_control_suite](/e:/CodeSpace/PredictiveCoding/outputs/fmpc_tf2_late_rollout_drift_control_suite)
  - [outputs/fmpc_tf2_preterminal_source_localization_suite](/e:/CodeSpace/PredictiveCoding/outputs/fmpc_tf2_preterminal_source_localization_suite)
  - [outputs/fmpc_tf2_preterminal_handoff_confirmation_suite](/e:/CodeSpace/PredictiveCoding/outputs/fmpc_tf2_preterminal_handoff_confirmation_suite)
  - [outputs/fmpc_tf2_successor_handoff_source_suite](/e:/CodeSpace/PredictiveCoding/outputs/fmpc_tf2_successor_handoff_source_suite)
  - [outputs/fmpc_tf2_successor_value_confirmation_suite](/e:/CodeSpace/PredictiveCoding/outputs/fmpc_tf2_successor_value_confirmation_suite)
  - [outputs/fmpc_tf2_successor_value_followup_suite](/e:/CodeSpace/PredictiveCoding/outputs/fmpc_tf2_successor_value_followup_suite)
  - [outputs/fmpc_tf2_successor_value_source_suite](/e:/CodeSpace/PredictiveCoding/outputs/fmpc_tf2_successor_value_source_suite)

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
